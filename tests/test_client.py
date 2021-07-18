# nosec
"""Test client"""
import asyncio
import threading

import pytest

import numpy as np

from luxdb.client import Client, connect
from luxdb.knn_store import IndexAlreadyExistsException, open_store
from luxdb.server import Server
from tests import generate_data


def server_thread(host, port, path, barrier):

    with open_store(path) as store:
        server = Server(host, port=port, store=store)
        loop = asyncio.new_event_loop()
        loop.create_task(server.start(callback=barrier.wait))
        loop.run_forever()


@pytest.fixture
def start_server(tmpdir, unused_tcp_port):
    barrier = threading.Barrier(2, timeout=1)
    thread = threading.Thread(target=server_thread, args=('127.0.0.1', unused_tcp_port, tmpdir / 'store.db', barrier))

    thread.daemon = True
    thread.start()
    barrier.wait()  # Wait til the server is actually listening
    yield unused_tcp_port


@pytest.fixture
async def client(start_server):
    async with connect('127.0.0.1', start_server) as client:
        yield client


class TestClient:
    @pytest.mark.asyncio
    async def test_connect(self, start_server):
        client = Client('127.0.0.1', start_server)
        assert client.reader is None
        assert client.writer is None

        await client.connect()
        assert client.writer is not None and isinstance(client.writer, asyncio.StreamWriter)
        assert client.reader is not None and isinstance(client.reader, asyncio.StreamReader)

    @pytest.mark.asyncio
    async def test_db_empty(self, start_server):
        async with connect('127.0.0.1', start_server) as client:
            result = await client.index_exists('this-should-not-exist')
            assert result == False

    @pytest.mark.asyncio
    async def test_client(self, client):
        assert False == await client.index_exists('this-should-not-exist')

    @pytest.mark.asyncio
    async def test_get_indexes(self, client):
        await client.create_index('first-index', 'l2', 12)
        await client.create_index('second-index', 'l2', 12)

        indexes = await client.get_indexes()
        assert len(indexes) == 2

    @pytest.mark.asyncio
    async def test_single_connection(self, client):
        name = 'test-create'
        max_elements = 100
        assert await client.create_index(name, 'l2', 12)
        await client.init_index(name, max_elements, ef_construction=140, M=12)

        assert await client.index_exists(name)

        info = await client.info(name)

        assert info['dim'] == 12
        assert info['max_elements'] == max_elements
        assert info['element_count'] == 0
        assert info['ef_construction'] == 140
        assert info['M'] == 12

        assert await client.get_ef_construction(name) == 140

        await client.set_ef(name, 160)
        assert await client.get_ef(name) == 160

        await client.resize_index(name, 200)
        assert await client.max_elements(name) == 200

        data, ids = generate_data(20, 12)

        await client.add_items(name, data, ids)

        assert await client.count(name) == 20

        all_ids = await client.get_ids(name)
        assert np.array_equal(np.sort(all_ids), ids)

        items = await client.get_items(name, ids[:5])
        assert np.array_equal(data[:5], items)

        labels, distances = await client.query_index(name, data[:5], k=1)

        assert ids[0] in labels
        assert len(labels) == 5
        assert len(distances) == 5
        assert sum(distances) == 0

        labels, distances = await client.query_index(name, data, k=1)
        assert (ids == labels.reshape(ids.shape)).all()

        await client.delete_item(name, ids[0])

        labels, distances = await client.query_index(name, data, k=1)

        assert ids[0] not in labels

        with pytest.raises(IndexAlreadyExistsException):
            await client.create_index(name, 'l2', 12)

        await client.delete_index(name)
        assert await client.index_exists(name) == False
