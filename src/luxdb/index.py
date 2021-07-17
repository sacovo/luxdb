"""Implementation of persistence."""
import logging
from contextlib import contextmanager
from threading import Condition, Lock, current_thread
from typing import List

import hnswlib
import persistent

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

    @contextmanager
    def __call__(self, tag):
        self.acquire(tag)
        try:
            yield self
        finally:
            self.release(tag)

    def acquire(self, requested_state: str):
        """
        If the path is free this will make sure that all other locks are locked.
        Otherwise wait until the path becomes free.
        """
        LOG.debug('Acquiring state lock for %s (Thread: %s)', requested_state, current_thread().ident)
        with self.state_lock:
            LOG.debug('Got state_lock for locking for %s (Thread: %s)', requested_state, current_thread().ident)

            self.condition.wait_for(lambda: self.locks[requested_state].acquire(blocking=False))

            LOG.debug('Got state_lock for %s (Thread: %s)', requested_state, current_thread().ident)

            self.counters[requested_state] += 1
            LOG.debug('State counter for %s is %d (Thread: %s)', requested_state, self.counters[requested_state],
                      current_thread().ident)

            for key, lock in self.locks.items():
                LOG.debug('Locking other locks for state %s (Thread: %s)', requested_state, current_thread().ident)
                if key != requested_state and not lock.locked():
                    LOG.debug('Acquiring lock for %s (Thread: %s)', key, current_thread().ident)
                    lock.acquire()

            LOG.debug('State locks are setup, releasing lock for %s (Thread: %s)', requested_state,
                      current_thread().ident)
            self.locks[requested_state].release()
            LOG.debug('Lock acquired for %s (Thread: %s)', requested_state, current_thread().ident)

    def release(self, requested_state: str):
        """Releases all other locks, as soon as all threads from the same state have released the lock."""
        LOG.debug('Releasing lock for state %s (Thread: %s)', requested_state, current_thread().ident)

        with self.state_lock:
            LOG.debug('Got state_lock for releasing of %s (Thread: %s)', requested_state, current_thread().ident)
            self.condition.wait_for(lambda: self.locks[requested_state].acquire(blocking=False))

            self.counters[requested_state] -= 1
            LOG.debug('State counter for %s is %d (Thread: %s)', requested_state, self.counters[requested_state],
                      current_thread().ident)

            if self.counters[requested_state] == 0:
                LOG.debug('Releasing locks for the other states. (Thread: %s)', current_thread().ident)

                for key, lock in self.locks.items():
                    LOG.debug('Trying to release lock for %s (Thread: %s)', key, current_thread().ident)
                    if key != requested_state and lock.locked():
                        LOG.debug('Unlocking lock %s (Thread: %s)', requested_state, current_thread().ident)
                        lock.release()
                        LOG.debug('Lock released for %s (Thread: %s)', key, current_thread().ident)
                self.condition.notify_all()

            self.locks[requested_state].release()
            LOG.debug('Released lock for state %s', requested_state)


class Index(persistent.Persistent):
    """Wrapper around hnswlib.Index to record accesses and store changes."""

    EXCLUSIVE_STATES = ['add', 'resize', 'query']

    def __init__(self, index: hnswlib.Index):
        self.index = index
        self._v_lock = ExclusiveLock(self.EXCLUSIVE_STATES)

    @property
    def lock(self):
        """Return the lock that is prefixed with _v_ so it's not persisted."""
        if not hasattr(self, '_v_lock'):  # Not initialized if read from db
            self._v_lock = ExclusiveLock(self.EXCLUSIVE_STATES)
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
        with self.lock('add'):
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
        with self.lock('query'):
            return self.index.knn_query(vector, k)

    def resize_index(self, new_size):
        """Resize the index and mark the index."""
        with self.lock('resize'):
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
