"""Exceptions in the LuxDB"""


class KNNBaseException(Exception):
    """Base class for exceptions in this module"""


class IndexAlreadyExistsException(KNNBaseException):
    """Thrown if an index already exists after creation was requested."""
    def __init__(self, name: str):
        """Create a new Exception for the given name."""
        self.name = name
        super().__init__(name)


class IndexDoesNotExistException(KNNBaseException):
    """The requested index does not exist in the database."""
    def __init__(self, name: str):
        self.name = name
        super().__init__(name)


class UnknownSpaceException(KNNBaseException):
    """Space is not known"""
    def __init__(self, space):
        self.space = space
        super().__init__(space)


class NotACommandException(KNNBaseException):
    """Tried to execute something that is not a command."""
    def __init__(self, obj):
        self.obj = obj
        super().__init__(obj)


class IndexNotInitializedException(KNNBaseException):
    """Tried do access an index that was not initialized."""
    def __init__(self, obj):
        self.obj = obj
        super().__init__(obj)
