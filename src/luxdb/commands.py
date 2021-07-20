"""This module provides command objects that are sent from the client to the server."""
import logging
import secrets
from enum import Enum, auto

from luxdb.exceptions import KNNBaseException, NotACommandException

LOG = logging.getLogger('commands')


class CommandState(Enum):
    """States that a command can be in."""
    CREATED = auto()
    SENT = auto()
    RECEIVED = auto()
    EXECUTED = auto()
    FAILED = auto()
    SUCCEEDED = auto()


class Result:
    """Wrapper around a result, so that we can also transport state and implement other stuff later on."""
    def __init__(self, data, state):
        self.data = data
        self.state = state

    def raise_error(self):
        """Raise an exception if the result is failed."""
        if self.state == CommandState.FAILED:
            raise self.data

    def get_value(self) -> any:
        """Checks if the result is successfull or not and returns the value if so."""
        self.raise_error()
        return self.data


class Command:
    """Base class for executing remote commands."""
    state = CommandState.CREATED
    result = None

    def __init__(self, **kwargs):
        self.command_args = kwargs

    async def execute(self, store) -> Result:
        """Executes the command and stores the result or error."""
        self.state = CommandState.EXECUTED
        try:
            self.result = await self.execute_command(store, **self.command_args)
            self.state = CommandState.SUCCEEDED
        except KNNBaseException as e:
            self.result = e
            self.state = CommandState.FAILED
        except Exception as e:  # pylint: disable=W0703
            logging.getLogger().exception('Unhandled exception %r', e)
            self.result = e
            self.state = CommandState.FAILED

        return Result(self.result, self.state)

    async def execute_command(self, store, **kwargs):
        """Commands need to implement this to return the command."""
        raise NotImplementedError('Commands need to provide this method')


async def execute_command(command: Command, store):
    """Checks if the command is an instance of command and if so execute it."""
    if isinstance(command, Command):
        return await command.execute(store)
    raise NotACommandException('Command is not instance of command!')


class ConnectCommand(Command):
    """Command to test if connection is established."""
    def __init__(self, **kwargs):
        self.payload = secrets.token_bytes()
        super().__init__()

    async def execute_command(self, store, **kwargs):
        return self.payload


class CreateIndexCommand(Command):
    """Create a new index on the receiving storage

    For the parameters refer to the corresponding method in KNNStore.
    """
    async def execute_command(self, store, **kwargs):
        return store.create_index(**kwargs)


class InitIndexCommand(Command):
    """Init the index."""
    async def execute_command(self, store, **kwargs):
        return await store.init_index(**kwargs)


class IndexExistsCommand(Command):
    """Check if the index exists."""
    async def execute_command(self, store, **kwargs):
        return store.index_exists(**kwargs)


class DeleteIndexCommand(Command):
    """Delete an index."""
    async def execute_command(self, store, **kwargs):
        return store.delete_index(**kwargs)


class AddItemsCommand(Command):
    """Add items to the specified index."""
    async def execute_command(self, store, **kwargs):
        return await store.add_items(**kwargs)


class SetEFCommand(Command):
    """Set the ef."""
    async def execute_command(self, store, **kwargs):
        return await store.set_ef(**kwargs)


class GetEFCommand(Command):
    """Get the ef."""
    async def execute_command(self, store, **kwargs):
        return await store.get_ef(**kwargs)


class GetEFConstructionCommand(Command):
    """get construction ef"""
    async def execute_command(self, store, **kwargs):
        return await store.get_ef_construction(**kwargs)


class QueryIndexCommand(Command):
    """Query the index."""
    async def execute_command(self, store, **kwargs):
        return await store.query_index(**kwargs)


class DeleteItemCommand(Command):
    """Delete an item from the index."""
    async def execute_command(self, store, **kwargs):
        return await store.delete_item(**kwargs)


class ResizeIndexCommand(Command):
    """Resize the index to a new size"""
    async def execute_command(self, store, **kwargs):
        return await store.resize_index(**kwargs)


class MaxElementsCommand(Command):
    """Get the current limit on the index."""
    async def execute_command(self, store, **kwargs):
        return await store.max_elements(**kwargs)


class CountCommand(Command):
    """Get the current amount of items."""
    async def execute_command(self, store, **kwargs):
        return await store.count(**kwargs)


class InfoCommand(Command):
    """Get information about the index."""
    async def execute_command(self, store, **kwargs):
        return await store.info(**kwargs)


class GetIndexesCommand(Command):
    """Get list of indexes."""
    async def execute_command(self, store, **kwargs):
        return store.get_indexes(**kwargs)


class GetItemsCommand(Command):
    """Get items with specified ids"""
    async def execute_command(self, store, **kwargs):
        return await store.get_items(**kwargs)


class GetIdsCommand(Command):
    """Get ids in index"""
    async def execute_command(self, store, **kwargs):
        return await store.get_ids(**kwargs)
