"""This module implements a storage for multi-dimensional vectors and fast nearest neighbor search.

It wraps around https://github.com/nmslib/hnswlib and provides thread safe access to the operations.

The Store can also be saved and loaded onto the disk.
"""
import asyncio
from functools import partial
import concurrent.futures
import logging
from contextlib import contextmanager
from typing import Dict

import hnswlib
import numpy.typing as npt
import persistent
import transaction
import ZODB
import ZODB.FileStorage
from BTrees.OOBTree import \
    OOBTree  # pylint: disable=import-error,no-name-in-module

from luxdb.exceptions import (IndexAlreadyExistsException, IndexDoesNotExistException, UnknownSpaceException)
from luxdb.index import Index

LOG = logging.getLogger('store')


class KNNStore(persistent.Persistent):
    """
    Store multiple indexes with different attributes.
    """

    ALLOWED_SPACES = ['l2', 'ip', 'cosine']

    def __init__(self, path: str = None, storage=None):
        """Create the database under the specified path or with the given storage.

        If `None` is given the database will be created in memory only.
        """
        self.transaction = transaction.TransactionManager()
        if path is not None:
            LOG.debug('Connecting to storage at %s', path)
            self.storage = ZODB.FileStorage.FileStorage(path)
        else:
            self.storage = storage

        self.db = ZODB.DB(self.storage)

        if self.storage is None:
            LOG.debug('Creating database in memory.')
            self.load_database()
        else:
            self.root = None

        self.path = path

    def get_index(self, name: str) -> hnswlib.Index:
        """Access the index with the given name."""
        try:
            return self.root['indexes'][name]
        except KeyError as e:
            raise IndexDoesNotExistException(name) from e

    def load_database(self):
        """Initialize the database from the filesystem"""
        LOG.debug('Loading database.')
        connection = self.db.open(self.transaction)
        self.root = connection.root()

        if not 'indexes' in self.root:
            LOG.info('DB does not exist, creating empty database.')
            self.root['indexes'] = OOBTree()
            self.transaction.commit()

        LOG.debug('Loaded database.')

    def close(self):
        """Closes the connection to the database so the locks are released."""
        self.db.close()

    def index_exists(self, name: str) -> bool:
        """Returns true if the index already exists."""
        LOG.debug('Index %s exists: %s', name, name in self.root['indexes'])
        return name in self.root['indexes']

    def create_index(self, name: str, space: str, dim: int) -> bool:
        """Create a new index with the given name and parameters."""
        with self.transaction:
            if space not in KNNStore.ALLOWED_SPACES:
                raise UnknownSpaceException(space)

            LOG.debug('Creating new index: %s, space: %s, dim: %d', name, space, dim)
            if name in self.root['indexes']:
                LOG.error('Index with name %s already exists', name)
                raise IndexAlreadyExistsException(name)

            index = hnswlib.Index(space=space, dim=dim)

            self.root['indexes'][name] = Index(index)

            LOG.debug('New index created: %s', name)
            return True

    def init_index(self, name: str, max_elements: int, ef_construction: int = 200, M: int = 16) -> None:
        """Init the index with the given parameters.

        More information about the parameters is available here:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        LOG.debug('Initializing index %s with max_elements: %d, ef_construction: %d, M: %d', name, max_elements,
                  ef_construction, M)

        with self.transaction:
            index = self.get_index(name)
            index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
            )

        LOG.info('Initialized index %s', name)

    def delete_index(self, name: str) -> None:
        """Deletes the index from the store."""
        LOG.debug('Deleting index %s', name)
        with self.transaction:
            del self.root['indexes'][name]
            LOG.info('Deleted index %s', name)

    async def add_items(self, name: str, data: npt.ArrayLike, ids: npt.ArrayLike):
        """Add the items with the ids to the index."""
        LOG.debug('Adding items to %s', name)
        loop = asyncio.get_running_loop()
        index = self.get_index(name)
        with self.transaction, concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, partial(index.add_items, data, ids))

        LOG.debug('Added items to %s', name)

    def set_ef(self, name: str, new_ef: int):
        """Sets the ef value on the index."""
        with self.transaction:
            index = self.get_index(name)

            index.set_ef(new_ef)

    def get_ef(self, name: str) -> int:
        """Return the current value of ef"""
        return self.get_index(name).ef

    def get_ef_construction(self, name: str) -> int:
        """Return the value of construction_ef"""
        return self.get_index(name).ef_construction

    async def query_index(self, name: str, vector: npt.ArrayLike, k: int = 1) -> npt.NDArray:
        """Return the k nearest vectors to the given vector"""
        loop = asyncio.get_running_loop()
        index = self.get_index(name)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, partial(index.knn_query, vector, k))

    def delete_item(self, name: str, label: int) -> None:
        """This does just remove the item from search results, it's still in the index."""
        with self.transaction:
            index = self.get_index(name)

            index.mark_deleted(label)

    async def resize_index(self, name: str, new_size: int) -> None:
        """Resize the index to accommodate more or less items."""
        LOG.debug('Resizing the index %s to size %d', name, new_size)
        loop = asyncio.get_running_loop()
        index = self.get_index(name)
        with self.transaction, concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, partial(index.resize_index, new_size))

    def max_elements(self, name: str) -> int:
        """Return the maximum number of elements that can be stored in the index."""
        return self.get_index(name).get_max_elements()

    def count(self, name: str) -> int:
        """Return the current amount of elements in the index."""
        return self.get_index(name).get_current_count()

    def info(self, name: str) -> Dict:
        """Get information about the index."""
        index = self.get_index(name)
        return {
            'space': index.space,
            'dim': index.dim,
            'M': index.M,
            'ef_construction': index.ef_construction,
            'max_elements': index.get_max_elements(),
            'element_count': index.get_current_count(),
            'ef': index.ef,
        }

    def get_items(self, name: str, ids):
        """get vectors with given labels"""
        return self.get_index(name).get_items(ids)

    def get_ids(self, name: str):
        """get all ids in the index"""
        return self.get_index(name).get_ids()

    def get_indexes(self):
        """Returns all indexes in the database."""
        return list(self.root['indexes'].keys())


@contextmanager
def open_store(path):
    """Open a store, load it if possible and save it as soon as the context is over."""
    store = KNNStore(path)
    store.load_database()

    try:
        yield store
    finally:
        store.close()
