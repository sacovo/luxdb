import asyncio
import secrets
import socket
import threading

import pytest
import numpy as np

from luxdb.knn_store import IndexAlreadyExistsException, open_store
from luxdb.server import Server
from luxdb.sync_client import SyncClient, connect
from tests import generate_data


def server_thread(host, port, secret, path, barrier):

    with open_store(path) as store:
        server = Server(host, port=port, store=store, secret=secret)
        loop = asyncio.new_event_loop()
        loop.create_task(server.start(callback=barrier.wait))
        loop.run_forever()


@pytest.fixture
def start_server(tmpdir, unused_tcp_port):

    secret = secrets.token_hex()
    barrier = threading.Barrier(2, timeout=1)
    thread = threading.Thread(target=server_thread,
                              args=('127.0.0.1', unused_tcp_port, secret, tmpdir / 'store.db', barrier))

    thread.daemon = True
    thread.start()
    barrier.wait()  # Wait til the server is actually listening
    yield unused_tcp_port, secret


@pytest.fixture
def client(start_server):
    with connect('127.0.0.1', *start_server) as client:
        yield client


class TestClient:
    def test_connect(self, start_server):
        client = SyncClient('127.0.0.1', *start_server)
        assert client.socket is None

        client.connect()
        assert client.socket is not None and isinstance(client.socket, socket.socket)

    def test_db_empty(self, start_server):
        with connect('127.0.0.1', *start_server) as client:
            result = client.index_exists('this-should-not-exist')
            assert result == False

    def test_client(self, client):
        assert False == client.index_exists('this-should-not-exist')

    def test_get_indexes(self, client):
        client.create_index('first-index', 'l2', 12)
        client.create_index('second-index', 'l2', 12)

        indexes = client.get_indexes()
        assert len(indexes) == 2

    def test_add_too_much(self, client):
        name = 'too-much'

        client.create_index(name, 'l2', 12)
        client.init_index(name, 1000)

        data, ids = generate_data(2000, 12)

        with pytest.raises(RuntimeError):
            client.add_items(name, data, ids)
        client.resize_index(name, 2000)
        client.add_items(name, data, ids)
        assert client.count(name) == 2000

    def test_add_wrong_dimension(self, client):
        name = 'wrong-dimension'
        dim = 12
        client.create_index(name, 'l2', dim)
        client.init_index(name, 1000)

        data, ids = generate_data(500, dim + 4)
        with pytest.raises(RuntimeError):
            client.add_items(name, data, ids)

    def test_init_twice(self, client):
        name = 'init-twice'
        client.create_index(name, 'l2', 12)
        client.init_index(name, 1000)

        with pytest.raises(RuntimeError):
            client.init_index(name, 1000)

    def test_single_connection(self, client):
        name = 'test-create'
        max_elements = 100
        assert client.create_index(name, 'l2', 12)
        client.init_index(name, max_elements, ef_construction=140, M=12)

        assert client.index_exists(name)

        info = client.info(name)

        assert info['dim'] == 12
        assert info['max_elements'] == max_elements
        assert info['element_count'] == 0
        assert info['ef_construction'] == 140
        assert info['M'] == 12

        assert client.get_ef_construction(name) == 140

        client.set_ef(name, 160)
        assert client.get_ef(name) == 160

        client.resize_index(name, 200)
        assert client.max_elements(name) == 200

        data, ids = generate_data(20, 12)

        client.add_items(name, data, ids)

        assert client.count(name) == 20

        all_ids = client.get_ids(name)
        assert np.array_equal(np.sort(all_ids), ids)

        items = client.get_items(name, ids[:5])
        assert np.array_equal(data[:5], items)

        labels, distances = client.query_index(name, data[:5], k=1)

        assert ids[0] in labels
        assert len(labels) == 5
        assert len(distances) == 5
        assert sum(distances) == 0

        labels, distances = client.query_index(name, data, k=1)
        assert (ids == labels.reshape(ids.shape)).all()

        client.delete_item(name, ids[0])

        labels, distances = client.query_index(name, data, k=1)

        assert ids[0] not in labels

        with pytest.raises(IndexAlreadyExistsException):
            client.create_index(name, 'l2', 12)

        client.delete_index(name)
        assert client.index_exists(name) == False
