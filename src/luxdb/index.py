"""Implementation of persistence."""
import logging
from asyncio import Condition, Lock

import hnswlib
import persistent

LOG = logging.getLogger('store')


class ReadWriteLock:
    """Simple read write lock"""
    def __init__(self):
        self.read_condition = Condition(Lock())
        self.readers = 0

    async def acquire_read(self):
        """Get the lock, if no writer has it this immediately returns, otherwise we wait."""
        await self.read_condition.acquire()
        try:
            self.readers += 1
        finally:
            self.read_condition.release()

    async def release_read(self):
        """Decrement the number of readers and if zero notify all waiting writers."""
        await self.read_condition.acquire()
        try:
            self.readers -= 1
            if self.readers == 0:
                self.read_condition.notify_all()
        finally:
            self.read_condition.release()

    async def acquire_write(self):
        """Acquire the lock so that no other writer or reader can acquire it."""
        await self.read_condition.acquire()
        await self.read_condition.wait_for(lambda: self.readers == 0)

    def release_write(self):
        """Release the write lock so other writers or readers can take it."""
        self.read_condition.notify_all()
        self.read_condition.release()


class Index(persistent.Persistent):
    """Wrapper around hnswlib.Index to record accesses and store changes."""

    EXCLUSIVE_STATES = ['add', 'resize', 'query']

    def __init__(self, index: hnswlib.Index):
        self.index = index
        self._v_lock = ReadWriteLock()  # ExclusiveLock(self.EXCLUSIVE_STATES)

    @property
    def lock(self):
        """Return the lock that is prefixed with _v_ so it's not persisted."""
        if not hasattr(self, '_v_lock'):  # Not initialized if read from db
            self._v_lock = ReadWriteLock()  # ExclusiveLock(self.EXCLUSIVE_STATES)
        return self._v_lock

    def _mark_change(self):
        """Mark the index as changed since the latest commit."""
        self._p_changed = True

    def init_index(self, max_elements: int, ef_construction: int = 200, M: int = 16):
        """Inits the index and marks the index as changed."""
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
        )
        self._mark_change()

    def add_items(self, data, ids):
        """Add items and mark index as changed."""
        LOG.debug('Adding to index.', stack_info=True)
        self.index.add_items(data, ids)
        LOG.debug('Added to index.')
        self._mark_change()
        LOG.debug('Returning.', stack_info=True)

    def set_ef(self, new_ef):
        """Set the the ef."""
        self.index.set_ef(new_ef)
        self._mark_change()

    @property
    def ef(self):
        """Access the ef of the index."""
        return self.index.ef

    @property
    def ef_construction(self):
        """Access the ef_construction of the index"""
        return self.index.ef_construction

    @property
    def space(self):
        """Access the space of the index."""
        return self.index.space

    @property
    def dim(self):
        """Access the dim of the index."""
        return self.index.dim

    @property
    def M(self):
        """Access the M of the index."""
        return self.index.M

    def knn_query(self, vector, k):
        """Query the index."""
        return self.index.knn_query(vector, k)

    def resize_index(self, new_size):
        """Resize the index and mark the index."""
        self.index.resize_index(new_size)
        self._mark_change()

    def mark_deleted(self, label):
        """Mark an item as deleted."""
        self.index.mark_deleted(label)
        self._mark_change()

    def get_max_elements(self):
        """Get the maximum number of elements for the index."""
        return self.index.get_max_elements()

    def get_current_count(self):
        """Get the current number of elements in the index."""
        return self.index.get_current_count()

    def get_items(self, ids):
        """Return the array of vectors (N*dim) with the given ids"""
        return self.index.get_items(ids)

    def get_ids(self):
        """Return the ids in the index"""
        return self.index.get_ids_list()
