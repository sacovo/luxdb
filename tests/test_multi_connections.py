import asyncio
import logging
import secrets
from tests import generate_data
from luxdb.sync_client import SyncClient, connect
import threading
from luxdb.server import Server
from luxdb.knn_store import open_store
import pytest


def server_thread(host, port, secret, path, barrier):

    with open_store(path) as store:
        server = Server(host, port=port, store=store, secret=secret)
        loop = asyncio.new_event_loop()
        loop.set_debug(True)
        logging.basicConfig(level=logging.DEBUG)
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


DIMENSION = 50


def first_client(port, secret, barrier):
    with connect('127.0.0.1', port, secret) as client:

        # -- Block 00
        name = 'index-01'
        client.create_index(name, 'l2', DIMENSION)
        client.init_index(name, 15000)

        barrier.wait()  # Wait 01
        barrier.reset()
        # --------------------

        # -- Block 01
        client.resize_index(name, 25000)
        data, ids = generate_data(2500, DIMENSION)
        client.add_items(name, data, ids + 12000)

        #        assert client.count(name) == 12000 + 2500
        barrier.wait()
        barrier.reset()
        # --------------------

        # -- Block 02

        data, _ = generate_data(300, DIMENSION)

        labels, distances = client.query_index(name, data, k=1)

        assert len(labels) == len(data)
        assert len(distances) == len(data)
        # --------------------


def second_client(port, secret, barrier):
    with connect('127.0.0.1', port, secret) as client:

        # Block 00
        name = 'index-01'
        barrier.wait()  # Wait 01
        # --------------------

        # Block 01
        assert client.index_exists(name)

        data, ids = generate_data(12000, DIMENSION)
        client.add_items(name, data, ids)

        barrier.wait()
        # --------------------

        # Block 02
        assert client.max_elements(name) == 25000
        data, ids = generate_data(2500, DIMENSION)
        client.add_items(name, data, ids + 12000 + 2500)


#        assert client.count(name) == 12000 + 5000
# --------------------


def third_client(port, secret, barrier):
    name = 'index-03'
    with connect('127.0.0.1', port, secret) as client:
        # Block 00
        client.create_index(name, 'l2', DIMENSION)

        client.init_index(name, 20000)
        assert client.max_elements(name) == 20000
        data, ids = generate_data(19000, DIMENSION)
        client.add_items(name, data, ids)

        assert client.count(name) == len(data)
        client.delete_index(name)
        assert client.index_exists(name) == False


def test_multi_clients(start_server):

    client_actions = [first_client, second_client, third_client]
    barrier = threading.Barrier(2, timeout=10)
    client_threads = [threading.Thread(target=action, args=(*start_server, barrier)) for action in client_actions]

    for thread in client_threads:
        thread.start()

    for thread in client_threads:
        thread.join()

    with connect('127.0.0.1', *start_server) as client:
        name = 'index-01'
        assert client.index_exists(name)
        assert client.count(name) == 12000 + 5000

        assert client.index_exists('index-03') == False

        client.delete_index(name)
