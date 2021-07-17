"""Implementation of persistence."""
import logging
from threading import Lock

import hnswlib
import persistent

LOG = logging.getLogger('store')


class Index(persistent.Persistent):
    """Wrapper around hnswlib.Index to record accesses and store changes."""

    EXCLUSIVE_STATES = ['add', 'resize', 'query']

    def __init__(self, index: hnswlib.Index):
        self.index = index
        self._v_lock = Lock()  # ExclusiveLock(self.EXCLUSIVE_STATES)

    @property
    def lock(self):
        """Return the lock that is prefixed with _v_ so it's not persisted."""
        if not hasattr(self, '_v_lock'):  # Not initialized if read from db
            self._v_lock = Lock()  # ExclusiveLock(self.EXCLUSIVE_STATES)
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
        with self.lock:
            self.index.add_items(data, ids)
            self._mark_change()

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
        with self.lock:
            return self.index.knn_query(vector, k)

    def resize_index(self, new_size):
        """Resize the index and mark the index."""
        with self.lock:
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
