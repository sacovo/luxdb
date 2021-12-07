"""This module implements a storage for multi-dimensional vectors and fast nearest neighbor search.

It wraps around https://github.com/nmslib/hnswlib and provides thread safe access to the operations.

The Store can also be saved and loaded onto the disk.
"""
import asyncio
import concurrent.futures
import logging
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from pathlib import Path
from typing import Dict
import tempfile

import numpy.typing as npt

import hnswlib
import persistent
import transaction
import ZODB
import ZODB.FileStorage
from BTrees.OOBTree import \
    OOBTree  # pylint: disable=import-error,no-name-in-module
from luxdb.exceptions import (IndexAlreadyExistsException, IndexDoesNotExistException, IndexNotInitializedException,
                              UnknownSpaceException)
from luxdb.index import Index

LOG = logging.getLogger('store')


class KNNStore(persistent.Persistent):
    """
    Store multiple indexes with different attributes.
    """

    ALLOWED_SPACES = ['l2', 'ip', 'cosine']

    def __init__(self, path: str = None):
        """Create the database under the specified path or with the given storage.

        If `None` is given the database will be created in memory only.
        """
        self.transaction = transaction.TransactionManager()

        if path is not None:
            LOG.debug('Connecting to storage at %s', path)
            self.storage = ZODB.FileStorage.FileStorage(path)
            self.path = Path(path)
            self.root = None
            if not self.path.is_dir():
                self.path = self.path.parent
        else:
            LOG.debug('Creating database in memory.')
            self.storage = None
            self.path = Path(tempfile.mkdtemp())
        self.db = ZODB.DB(self.storage)
        self.path = self.path / 'indexes'
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)

        if self.storage is None:
            self.load_database()

    def get_index(self, name: str) -> hnswlib.Index:
        """Access the index with the given name."""
        try:
            index = self.root['indexes'][name]
            return index
        except KeyError as e:
            raise IndexDoesNotExistException(name) from e

    @asynccontextmanager
    async def _index_for_write(self, name: str):
        index: Index = self.get_index(name)
        await index.lock.acquire_write()
        index.read_from_path(self.path)
        if index.M == 0:
            raise IndexNotInitializedException(name)
        try:
            yield index
        finally:
            index.write_to_path(self.path)
            index.lock.release_write()

    @asynccontextmanager
    async def _index_for_init(self, name: str):
        index: Index = self.get_index(name)
        await index.lock.acquire_write()
        try:
            yield index
        finally:
            index.write_to_path(self.path)
            index.lock.release_write()

    @asynccontextmanager
    async def _index_for_read(self, name: str):
        index: Index = self.get_index(name)
        await index.lock.acquire_read()
        index.read_from_path(self.path)
        if index.M == 0:
            raise IndexNotInitializedException(name)
        try:
            yield index
        finally:
            await index.lock.release_read()

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
        if space not in KNNStore.ALLOWED_SPACES:
            raise UnknownSpaceException(space)

        LOG.debug('Creating new index: %s, space: %s, dim: %d', name, space, dim)
        if name in self.root['indexes']:
            LOG.error('Index with name %s already exists', name)
            raise IndexAlreadyExistsException(name)

        index = hnswlib.Index(space=space, dim=dim)

        self.root['indexes'][name] = Index(index)
        self.transaction.commit()

        LOG.debug('New index created: %s', name)
        return True

    async def init_index(self, name: str, max_elements: int, ef_construction: int = 200, M: int = 16) -> None:
        """Init the index with the given parameters.

        More information about the parameters is available here:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        LOG.debug('Initializing index %s with max_elements: %d, ef_construction: %d, M: %d', name, max_elements,
                  ef_construction, M)

        async with self._index_for_init(name) as index:
            index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
            )
            self.transaction.commit()

        LOG.info('Initialized index %s', name)

    def delete_index(self, name: str) -> None:
        """Deletes the index from the store."""
        LOG.debug('Deleting index %s', name)
        del self.root['indexes'][name]
        self.transaction.commit()
        LOG.info('Deleted index %s', name)

    def import_index(self, name: str, index: Index) -> None:
        """Import an index from outside into the store."""
        if self.index_exists(name):
            raise IndexAlreadyExistsException(name)
        index.write_to_path(self.path)
        self.root['indexes'][name] = index
        self.transaction.commit()
        LOG.info('Imported index: %s', name)

    async def add_items(self, name: str, data: npt.ArrayLike, ids: npt.ArrayLike):
        """Add the items with the ids to the index."""
        LOG.debug('Adding items to %s', name, stack_info=True)
        loop = asyncio.get_running_loop()
        async with self._index_for_write(name) as index:
            LOG.debug('Got write lock for index %s', name, stack_info=True)
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, partial(index.add_items, data, ids))
                LOG.debug('Added items, closing transaction', stack_info=True)
                self.transaction.commit()
            LOG.debug('Committed, releasing lock.')

        LOG.debug('Added items to %s', name)

    async def set_ef(self, name: str, new_ef: int):
        """Sets the ef value on the index."""
        async with self._index_for_write(name) as index:
            index.set_ef(new_ef)
            self.transaction.commit()

    async def get_ef(self, name: str) -> int:
        """Return the current value of ef"""
        async with self._index_for_read(name) as index:
            return index.ef

    async def get_ef_construction(self, name: str) -> int:
        """Return the value of construction_ef"""
        async with self._index_for_read(name) as index:
            return index.ef_construction

    async def query_index(self, name: str, vector: npt.ArrayLike, k: int = 1) -> npt.NDArray:
        """Return the k nearest vectors to the given vector"""
        loop = asyncio.get_running_loop()
        index = self.get_index(name)
        async with self._index_for_read(name) as index:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, partial(index.knn_query, vector, k))

    async def delete_item(self, name: str, label: int) -> None:
        """This does just remove the item from search results, it's still in the index."""
        async with self._index_for_write(name) as index:
            index.mark_deleted(label)
            self.transaction.commit()

    async def resize_index(self, name: str, new_size: int) -> None:
        """Resize the index to accommodate more or less items."""
        LOG.debug('Resizing the index %s to size %d', name, new_size)
        loop = asyncio.get_running_loop()
        async with self._index_for_write(name) as index:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, partial(index.resize_index, new_size))
                self.transaction.commit()

    async def max_elements(self, name: str) -> int:
        """Return the maximum number of elements that can be stored in the index."""
        async with self._index_for_read(name) as index:
            return index.get_max_elements()

    async def count(self, name: str) -> int:
        """Return the current amount of elements in the index."""
        async with self._index_for_read(name) as index:
            return index.get_current_count()

    async def info(self, name: str) -> Dict:
        """Get information about the index."""
        index = self.get_index(name)
        async with self._index_for_read(name) as index:
            return {
                'space': index.space,
                'dim': index.dim,
                'M': index.M,
                'ef_construction': index.ef_construction,
                'max_elements': index.get_max_elements(),
                'element_count': index.get_current_count(),
                'ef': index.ef,
            }

    async def get_items(self, name: str, ids):
        """get vectors with given labels"""
        async with self._index_for_read(name) as index:
            if index.get_current_count() == 0:
                return []
            return index.get_items(ids)

    async def get_ids(self, name: str):
        """get all ids in the index"""
        async with self._index_for_read(name) as index:
            if index.get_current_count() == 0:
                return []
            return index.get_ids()

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
