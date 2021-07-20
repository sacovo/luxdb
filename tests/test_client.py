# nosec
"""Test client"""
import asyncio
import os
import secrets
import threading

import pytest

import numpy as np

from luxdb.client import Client, connect
from luxdb.knn_store import IndexAlreadyExistsException, open_store
from luxdb.server import Server
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
async def client(start_server):
    async with connect('127.0.0.1', *start_server) as client:
        yield client


class TestClient:
    @pytest.mark.asyncio
    async def test_connect(self, start_server):
        client = Client('127.0.0.1', *start_server)
        assert client.reader is None
        assert client.writer is None

        await client.connect()
        assert client.writer is not None and isinstance(client.writer, asyncio.StreamWriter)
        assert client.reader is not None and isinstance(client.reader, asyncio.StreamReader)

    @pytest.mark.asyncio
    async def test_invalid_secret(self, start_server):
        port = start_server[0]
        client = Client('127.0.0.1', port, '')
        with pytest.raises(RuntimeError):
            await client.connect()

    @pytest.mark.asyncio
    async def test_db_empty(self, start_server):
        async with connect('127.0.0.1', *start_server) as client:
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
    async def test_add_too_much(self, client):
        name = 'too-much'

        await client.create_index(name, 'l2', 12)
        await client.init_index(name, 1000)

        data, ids = generate_data(2000, 12)

        with pytest.raises(RuntimeError):
            await client.add_items(name, data, ids)
        await client.resize_index(name, 2000)
        await client.add_items(name, data, ids)
        assert await client.count(name) == 2000

    @pytest.mark.asyncio
    async def test_add_wrong_dimension(self, client):
        name = 'wrong-dimension'
        dim = 12
        await client.create_index(name, 'l2', dim)
        await client.init_index(name, 1000)

        data, ids = generate_data(500, dim + 4)
        with pytest.raises(RuntimeError):
            await client.add_items(name, data, ids)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get('RUN_LARGE'), reason='not running large tests')
    async def test_adding_a_lot(self, client):
        name = 'big'
        dim = 100
        num_items = 100_000
        await client.create_index(name, 'l2', dim)
        await client.init_index(name, num_items)

        data, ids = generate_data(num_items, dim)

        await client.add_items(name, data, ids)
        assert await client.count(name) == num_items

    @pytest.mark.asyncio
    async def test_init_twice(self, client):
        name = 'init-twice'
        await client.create_index(name, 'l2', 12)
        await client.init_index(name, 1000)

        with pytest.raises(RuntimeError):
            await client.init_index(name, 1000)

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
