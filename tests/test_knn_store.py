"""Test store functionality"""
import os
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest

from luxdb.exceptions import IndexDoesNotExistException
from luxdb.knn_store import (IndexAlreadyExistsException, KNNStore, UnknownSpaceException, open_store)
from tests import generate_data


class TestStore:
    """Test store functionality"""
    def test_create_index(self):
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

    def test_index_does_not_exist(self):
        store = KNNStore()

        with pytest.raises(IndexDoesNotExistException):
            store.info('this-does-not-exist')

    def test_remove_index(self):
        store = KNNStore()
        name = 'test-remove'
        store.create_index(name, 'l2', 12)

        store.delete_index(name)

        assert store.index_exists(name) == False

    def test_set_ef(self):
        store = KNNStore()
        name = 'test-set-ef'

        store.create_index(name, 'l2', 12)
        store.init_index(name, 120)

        store.set_ef(name, 210)

        assert store.get_ef(name) == 210

    def test_unknown_space(self):
        store = KNNStore()

        space = 'l1'

        with pytest.raises(UnknownSpaceException):
            store.create_index('test-unknown', space, 10)

    def test_init_index(self):
        store = KNNStore()

        max_elements = 1000
        dimension = 12
        name = 'test-1'
        store.create_index(name, 'l2', dimension)
        store.init_index(name, max_elements)

        assert store.max_elements(name) == max_elements
        assert store.count(name) == 0

        max_elements = 2000
        ef_construction = 30
        M = 30
        name = 'test-2'

        store.create_index(name, 'l2', dimension)
        store.init_index(name, max_elements, ef_construction, M)
        assert store.max_elements(name) == max_elements
        assert store.count(name) == 0
        assert store.get_ef_construction(name) == ef_construction

    def test_resize_index(self):
        store = KNNStore()
        name = 'test-resize'
        dimension = 12
        max_elements = 1000
        store.create_index(name, 'l2', dimension)
        store.init_index(name, max_elements)

        store.resize_index(name, 500)

        assert store.max_elements(name) == 500
        store.resize_index(name, 1200)
        assert store.max_elements(name) == 1200

    def test_create_twice_raises(self):
        store = KNNStore()

        dimension = 12
        name = 'test-twice'

        store.create_index(name, 'l2', dimension)

        assert store.index_exists(name)

        with pytest.raises(IndexAlreadyExistsException):
            store.create_index(name, 'l2', dimension)

    def test_adding_elements(self):
        store = KNNStore()

        max_elements = 1000
        dimension = 12

        name = 'test-adding'

        store.create_index(name, 'l2', dimension)
        store.init_index(name, max_elements)

        num_elements = 800

        data, ids = generate_data(num_elements, dimension)

        store.add_items(name, data, ids)

        assert store.count(name) == num_elements

    def test_query(self):
        store = KNNStore()

        max_elements = 600
        dimension = 6
        name = 'test-query'

        store.create_index(name, 'cosine', dimension)
        store.init_index(name, max_elements)

        num_elements = 500

        data, ids = generate_data(num_elements, dimension)
        store.add_items(name, data, ids)

        labels, distances = store.query_index(name, data, k=1)
        assert all(labels.reshape(ids.shape) == ids)

        assert len(labels) == num_elements
        assert len(distances) == num_elements

        labels, distances = store.query_index(name, data, k=2)
        assert labels.shape == (num_elements, 2)
        assert distances.shape == (num_elements, 2)

        labels, distances = store.query_index(name, data[:20], k=2)
        assert labels.shape == (20, 2)
        assert distances.shape == (20, 2)

    def test_delete(self):
        store = KNNStore()

        max_elements = 12
        dimension = 5

        name = 'test-delete'
        store.create_index(name, 'l2', dimension)
        store.init_index(name, max_elements)

        data, ids = generate_data(10, dimension)

        store.add_items(name, data, ids)
        store.delete_item(name, ids[0])

        labels, _ = store.query_index(name, data, k=2)

        assert ids[0] not in labels

    def test_store_and_load(self, tmpdir):
        path = tmpdir / 'test.db'

        store = KNNStore(path)
        store.load_database()

        name = 'test-store'
        dimension = 100
        max_elements = 1000
        store.create_index(name, 'l2', dimension)
        store.init_index(name, max_elements)

        num_elements = 500

        data, ids = generate_data(num_elements, dimension)

        store.add_items(name, data, ids)

        store.close()

        del store

        store = KNNStore(path)
        store.load_database()

        assert store.index_exists(name)
        assert store.count(name) == num_elements

    def test_contextmanager(self, tmpdir):
        path = tmpdir / 'test-context.db'
        dim = 10
        max_elements = 1000
        name = 'test-context'
        num_elements = 500

        with open_store(path) as store:
            store.create_index(name, 'l2', dim)
            store.init_index(name, max_elements)
            data, ids = generate_data(num_elements, dim)
            store.add_items(name, data, ids)

        assert os.path.exists(path)

        with open_store(path) as store:
            assert store.index_exists(name)
            assert store.count(name) == num_elements
            assert store.max_elements(name) == max_elements
