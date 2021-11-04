"""Test store functionality"""
import os

import pytest

from luxdb.exceptions import IndexDoesNotExistException, IndexNotInitializedException
from luxdb.knn_store import (IndexAlreadyExistsException, KNNStore, UnknownSpaceException, open_store)
from tests import generate_data


class TestStore:
    """Test store functionality"""
    @pytest.mark.asyncio
    async def test_create_index(self):
        store = KNNStore()

        store.create_index('test', 'l2', 12)

        assert store.index_exists('test')

    def test_separate_stores(self):
        store1 = KNNStore()
        store2 = KNNStore()
        store1.create_index('test', 'l2', 12)

        assert store1.index_exists('test')
        assert not store2.index_exists('test')

        store2.create_index('test', 'cosine', 23)

        assert store2.index_exists('test')

    @pytest.mark.asyncio
    async def test_index_does_not_exist(self):
        store = KNNStore()

        with pytest.raises(IndexDoesNotExistException):
            await store.info('this-does-not-exist')

    def test_remove_index(self):
        store = KNNStore()
        name = 'test-remove'
        store.create_index(name, 'l2', 12)

        store.delete_index(name)

        assert store.index_exists(name) == False

    @pytest.mark.asyncio
    async def test_set_ef(self):
        store = KNNStore()
        name = 'test-set-ef'

        store.create_index(name, 'l2', 12)
        await store.init_index(name, 120)

        await store.set_ef(name, 210)

        assert await store.get_ef(name) == 210

    def test_unknown_space(self):
        store = KNNStore()

        space = 'l1'

        with pytest.raises(UnknownSpaceException):
            store.create_index('test-unknown', space, 10)

    @pytest.mark.asyncio
    async def test_init_index(self):
        store = KNNStore()

        max_elements = 1000
        dimension = 12
        name = 'test-1'
        store.create_index(name, 'l2', dimension)
        await store.init_index(name, max_elements)

        assert await store.max_elements(name) == max_elements
        assert await store.count(name) == 0

        max_elements = 2000
        ef_construction = 30
        M = 30
        name = 'test-2'

        store.create_index(name, 'l2', dimension)
        await store.init_index(name, max_elements, ef_construction, M)
        assert await store.max_elements(name) == max_elements
        assert await store.count(name) == 0
        assert await store.get_ef_construction(name) == ef_construction

    @pytest.mark.asyncio
    async def test_fetch_empty_store(self):
        store = KNNStore()

        store.create_index('test', 'l2', 120)
        await store.init_index('test', 100)

        assert await store.count('test') == 0
        assert await store.get_ids('test') == []

    @pytest.mark.asyncio
    async def test_fetch_non_init_store(self):
        store = KNNStore()

        store.create_index('test', 'l2', 120)

        with pytest.raises(IndexNotInitializedException):
            await store.count('test')
            await store.get_ids('test')

    @pytest.mark.asyncio
    async def test_resize_index(self):
        store = KNNStore()
        name = 'test-resize'
        dimension = 12
        max_elements = 1000
        store.create_index(name, 'l2', dimension)
        await store.init_index(name, max_elements)

        await store.resize_index(name, 500)

        assert await store.max_elements(name) == 500
        await store.resize_index(name, 1200)
        assert await store.max_elements(name) == 1200

    def test_create_twice_raises(self):
        store = KNNStore()

        dimension = 12
        name = 'test-twice'

        store.create_index(name, 'l2', dimension)

        assert store.index_exists(name)

        with pytest.raises(IndexAlreadyExistsException):
            store.create_index(name, 'l2', dimension)

    @pytest.mark.asyncio
    async def test_adding_elements(self):
        store = KNNStore()

        max_elements = 1000
        dimension = 12

        name = 'test-adding'

        store.create_index(name, 'l2', dimension)
        await store.init_index(name, max_elements)

        num_elements = 800

        data, ids = generate_data(num_elements, dimension)

        await store.add_items(name, data, ids)

        assert await store.count(name) == num_elements

    @pytest.mark.asyncio
    async def test_query(self):
        store = KNNStore()

        max_elements = 600
        dimension = 6
        name = 'test-query'

        store.create_index(name, 'cosine', dimension)
        await store.init_index(name, max_elements)

        num_elements = 500

        data, ids = generate_data(num_elements, dimension)
        await store.add_items(name, data, ids)

        labels, distances = await store.query_index(name, data, k=1)
        assert all(labels.reshape(ids.shape) == ids)

        assert len(labels) == num_elements
        assert len(distances) == num_elements

        labels, distances = await store.query_index(name, data, k=2)
        assert labels.shape == (num_elements, 2)
        assert distances.shape == (num_elements, 2)

        labels, distances = await store.query_index(name, data[:20], k=2)
        assert labels.shape == (20, 2)
        assert distances.shape == (20, 2)

    @pytest.mark.asyncio
    async def test_delete(self):
        store = KNNStore()

        max_elements = 12
        dimension = 5

        name = 'test-delete'
        store.create_index(name, 'l2', dimension)
        await store.init_index(name, max_elements)

        data, ids = generate_data(10, dimension)

        await store.add_items(name, data, ids)
        await store.delete_item(name, ids[0])

        labels, _ = await store.query_index(name, data, k=2)

        assert ids[0] not in labels

    @pytest.mark.asyncio
    async def test_store_and_load(self, tmpdir):
        path = tmpdir / 'test.db'

        store = KNNStore(path)
        store.load_database()

        name = 'test-store'
        dimension = 100
        max_elements = 1000
        store.create_index(name, 'l2', dimension)
        await store.init_index(name, max_elements)

        num_elements = 500

        data, ids = generate_data(num_elements, dimension)

        await store.add_items(name, data, ids)

        store.close()

        del store

        store = KNNStore(path)
        store.load_database()

        assert store.index_exists(name)
        assert await store.count(name) == num_elements

    @pytest.mark.asyncio
    async def test_contextmanager(self, tmpdir):
        path = tmpdir / 'test-context.db'
        dim = 10
        max_elements = 1000
        name = 'test-context'
        num_elements = 500

        assert not os.path.exists(path)

        with open_store(path) as store:
            store.create_index(name, 'l2', dim)
            await store.init_index(name, max_elements)
            data, ids = generate_data(num_elements, dim)
            await store.add_items(name, data, ids)

        assert os.path.exists(path)

        with open_store(path) as store:
            assert store.index_exists(name)
            assert await store.count(name) == num_elements
            assert await store.max_elements(name) == max_elements
