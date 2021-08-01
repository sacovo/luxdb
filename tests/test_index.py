from pathlib import Path
from tests import generate_data
import hnswlib

import pytest

from luxdb.index import Index

SPACE = 'l2'
DIM = 12


@pytest.fixture
def index():
    space = SPACE
    dim = DIM
    hnsw_index = hnswlib.Index(space=space, dim=dim)
    return Index(hnsw_index)


class TestIndex:
    def test_create_index(self, index: Index):
        assert index.dim == DIM
        assert index.space == SPACE

    def test_init_index(self, index: Index):
        max_elements = 1000

        index.init_index(max_elements)

        assert index.get_max_elements() == max_elements
        assert index.get_current_count() == 0

    def test_set_ef(self, index: Index):
        ef = 120
        index.set_ef(ef)

        assert index.ef == ef

    def test_add_items(self, index: Index):
        index.init_index(200)
        data, ids = generate_data(100, DIM)
        index.add_items(data, ids)

        assert index.get_current_count() == len(data)

    def test_store_and_load(self, index: Index, tmpdir):
        index.init_index(200)
        data, ids = generate_data(200, DIM)
        index.add_items(data[:100], ids[:100])

        index.write_to_path(Path(tmpdir))

        index.add_items(data[100:], ids[100:])

        assert index.get_current_count() == len(data)

        del index._v_index

        index.read_from_path(Path(tmpdir))

        assert index.get_current_count() == 100
