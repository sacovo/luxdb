"""This module implements a storage for multi-dimensional vectors and fast nearest neighbor search.

It wraps around https://github.com/nmslib/hnswlib and provides thread safe access to the operations.

The Store can also be saved and loaded onto the disk.
"""
import logging
import os
import pickle  # nosec
from contextlib import contextmanager
from threading import Condition, Lock, current_thread
from typing import Dict, List

import hnswlib
import numpy.typing as npt

from luxdb.exceptions import (IndexAlreadyExistsException, IndexDoesNotExistException, UnknownSpaceException)

LOG = logging.getLogger('store')


class ExclusiveLock:
    """
    This implements a lock specific for the library we are using.

    If one thread is in A, no threads may be in B or in C, but other threads may enter A.
    As soon as all threads have left A, other threads can enter B or C. As soon as one thread enters
    B, no other threads can enter A or C.

    """
    def __init__(self, states: List[str]):
        self.state_lock = Lock()
        self.condition = Condition(self.state_lock)

        self.locks = {state: Lock() for state in states}
        self.counters = {state: 0 for state in states}

    def acquire(self, requested_state: str):
        """
        If the path is free this will make sure that all other locks are locked.
        Otherwise wait until the path becomes free.
        """
        LOG.debug("Acquiring state lock for %s (Thread: %s)", requested_state, current_thread().ident)
        with self.state_lock:
            LOG.debug("Got state_lock for locking for %s (Thread: %s)", requested_state, current_thread().ident)

            self.condition.wait_for(lambda: self.locks[requested_state].acquire(blocking=False))

            LOG.debug("Got state_lock for %s (Thread: %s)", requested_state, current_thread().ident)

            self.counters[requested_state] += 1
            LOG.debug("State counter for %s is %d (Thread: %s)", requested_state, self.counters[requested_state],
                      current_thread().ident)

            for key, lock in self.locks.items():
                LOG.debug("Locking other locks for state %s (Thread: %s)", requested_state, current_thread().ident)
                if key != requested_state and not lock.locked():
                    LOG.debug("Acquiring lock for %s (Thread: %s)", key, current_thread().ident)
                    lock.acquire()

            LOG.debug("State locks are setup, releasing lock for %s (Thread: %s)", requested_state,
                      current_thread().ident)
            self.locks[requested_state].release()
            LOG.debug("Lock acquired for %s (Thread: %s)", requested_state, current_thread().ident)

    def release(self, requested_state: str):
        """Releases all other locks, as soon as all threads from the same state have released the lock."""
        LOG.debug("Releasing lock for state %s (Thread: %s)", requested_state, current_thread().ident)

        with self.state_lock:
            LOG.debug("Got state_lock for releasing of %s (Thread: %s)", requested_state, current_thread().ident)
            self.condition.wait_for(lambda: self.locks[requested_state].acquire(blocking=False))

            self.counters[requested_state] -= 1
            LOG.debug("State counter for %s is %d (Thread: %s)", requested_state, self.counters[requested_state],
                      current_thread().ident)

            if self.counters[requested_state] == 0:
                LOG.debug("Releasing locks for the other states. (Thread: %s)", current_thread().ident)

                for key, lock in self.locks.items():
                    LOG.debug("Trying to release lock for %s (Thread: %s)", key, current_thread().ident)
                    if key != requested_state and lock.locked():
                        LOG.debug("Unlocking lock %s (Thread: %s)", requested_state, current_thread().ident)
                        lock.release()
                        LOG.debug("Lock released for %s (Thread: %s)", key, current_thread().ident)
                self.condition.notify_all()

            self.locks[requested_state].release()
            LOG.debug("Released lock for state %s", requested_state)


class ReadWriteLock:
    """Simple read write lock"""
    def __init__(self):
        self.read_condition = Condition(Lock())
        self.readers = 0

    @contextmanager
    def read(self):
        """Context for reading:

        with lock.read():
            # Do reading
        """
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self):
        """Context for writing lock

        with lock.write():
            # Write stuff
        """
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    def acquire_read(self):
        """Get the lock, if no writer has it this immediately returns, otherwise we wait."""
        self.read_condition.acquire()
        try:
            self.readers += 1
        finally:
            self.read_condition.release()

    def release_read(self):
        """Decrement the number of readers and if zero notify all waiting writers."""
        self.read_condition.acquire()
        try:
            self.readers -= 1
            if self.readers == 0:
                self.read_condition.notifyAll()
        finally:
            self.read_condition.release()

    def acquire_write(self):
        """Acquire the lock so that no other writer or reader can acquire it."""
        self.read_condition.acquire()
        self.read_condition.wait_for(lambda: self.readers == 0)

    def release_write(self):
        """Release the write lock so other writers or readers can take it."""
        self.read_condition.notifyAll()
        self.read_condition.release()


ALLOWED_SPACES = ['l2', 'ip', 'cosine']


class KNNStore:
    """
    Store multiple indexes with different attributes.
    """

    EXCLUSIVE_STATES = ['add', 'resize', 'query']

    def __init__(self, path: str):
        LOG.debug("Creating database with path: %s", path)
        self.indexes = {}
        self.path = path
        self.locks: Dict[str, ExclusiveLock] = {}
        self.save_lock = ReadWriteLock()

    def get_index(self, name: str) -> hnswlib.Index:
        """Access the index with the given name."""
        try:
            return self.indexes[name]
        except KeyError as e:
            raise IndexDoesNotExistException(name) from e

    def get_lock(self, name: str) -> ExclusiveLock:
        """Access to the lock for the index with the given name."""
        try:
            return self.locks[name]
        except KeyError as e:
            raise IndexDoesNotExistException(name) from e

    def save_database(self):
        """Stores all index to the filesystem."""
        with self.save_lock.write():

            LOG.debug("Saving database to: %s", self.path)
            with open(self.path, 'wb') as f:
                pickle.dump(self.indexes, f)
                LOG.debug("Saved database to: %s", self.path)

    def load_database(self):
        """Initialize the database from the filesystem"""
        LOG.debug("Loading database from %s", self.path)

        if not os.path.exists(self.path):
            LOG.warning("Database path does not exist: %s", self.path)
            return

        with open(self.path, 'rb') as f:
            self.indexes = pickle.load(f)  # nosec

        LOG.debug("Loaded database from %s", self.path)

    def index_exists(self, name: str) -> bool:
        """Returns true if the index already exists."""
        LOG.debug("Index %s exists: %s", name, name in self.indexes)
        return name in self.indexes

    def create_index(self, name: str, space: str, dim: int) -> bool:
        """Create a new index with the given name and parameters."""
        with self.save_lock.read():
            if space not in ALLOWED_SPACES:
                raise UnknownSpaceException(space)

            LOG.debug("Creating new index: %s, space: %s, dim: %d", name, space, dim)
            if name in self.indexes:
                LOG.error("Index with name %s already exists", name)
                raise IndexAlreadyExistsException(name)

            index = hnswlib.Index(space=space, dim=dim)

            self.indexes[name] = index
            self.locks[name] = ExclusiveLock(KNNStore.EXCLUSIVE_STATES)
            LOG.debug("New index created: %s", name)
            return True

    def init_index(self, name: str, max_elements: int, ef_construction: int = 200, M: int = 16) -> None:
        """Init the index with the given parameters.

        More information about the parameters is available here:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        LOG.debug("Initializing index %s with max_elements: %d, ef_construction: %d, M: %d", name, max_elements,
                  ef_construction, M)

        with self.save_lock.read():
            index = self.get_index(name)
            index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
            )

        LOG.info("Initialized index %s", name)

    def delete_index(self, name: str) -> None:
        """Deletes the index from the store."""
        LOG.debug("Deleting index %s", name)
        with self.save_lock.read():
            del self.indexes[name]
            del self.locks[name]
            LOG.info("Deleted index %s", name)

    def add_items(self, name: str, data: npt.ArrayLike, ids: npt.ArrayLike):
        """Add the items with the ids to the index."""
        LOG.debug("Adding items to %s", name)
        with self.save_lock.read():

            self.get_lock(name).acquire('add')

            index = self.get_index(name)
            index.add_items(data, ids)

            self.get_lock(name).release('add')
        LOG.debug("Added items to %s", name)

    def set_ef(self, name: str, new_ef: int):
        """Sets the ef value on the index."""
        with self.save_lock.read():
            index = self.get_index(name)

            index.set_ef(new_ef)

    def get_ef(self, name: str) -> int:
        """Return the current value of ef"""
        return self.get_index(name).ef

    def get_ef_construction(self, name: str) -> int:
        """Return the value of construction_ef"""
        return self.get_index(name).ef_construction

    def query_index(self, name: str, vector: npt.ArrayLike, k: int = 1) -> npt.NDArray:
        """Return the k nearest vectors to the given vector"""
        with self.save_lock.read():
            try:
                self.get_lock(name).acquire('query')
                index = self.get_index(name)
                return index.knn_query(vector, k)
            finally:
                self.get_lock(name).release('query')

    def delete_item(self, name: str, label: int) -> None:
        """This does just remove the item from search results, it's still in the index."""
        with self.save_lock.read():
            index = self.get_index(name)

            index.mark_deleted(label)

    def resize_index(self, name: str, new_size: int) -> None:
        """Resize the index to accommodate more or less items."""
        with self.save_lock.read():
            self.get_lock(name).acquire('resize')
            index = self.get_index(name)
            index.resize_index(new_size)
            self.get_lock(name).release('resize')

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
            'max_elements': index.max_elements,
            'element_count': index.element_count,
            'ef': index.ef,
        }


@contextmanager
def open_store(path):
    """Open a store, load it if possible and save it as soon as the context is over."""
    store = KNNStore(path)

    if os.path.exists(path):
        store.load_database()
    try:
        yield store
    finally:
        store.save_database()
