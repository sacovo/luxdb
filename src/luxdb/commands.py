"""This module provides command objects that are sent from the client to the server."""
import asyncio
import concurrent.futures
from enum import Enum, auto
from functools import partial
import logging

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

    is_cpu_heavy = False

    def __init__(self, **kwargs):
        self.command_args = kwargs

    async def execute(self, store) -> Result:
        """Executes the command and stores the result or error."""
        self.state = CommandState.EXECUTED
        try:
            if self.is_cpu_heavy:
                LOG.debug('Executing command %r in thread pool.', self)
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    self.result = await loop.run_in_executor(pool,
                                                             partial(self.execute_command, store, **self.command_args))
            else:
                LOG.debug('Executing command %r in event loop.', self)
                self.result = self.execute_command(store, **self.command_args)
            self.state = CommandState.SUCCEEDED
        except KNNBaseException as e:
            self.result = e
            self.state = CommandState.FAILED
        except Exception as e:  # pylint: disable=W0703
            logging.getLogger().exception('Unhandled exception %r', e)
            self.result = e
            self.state = CommandState.FAILED

        return Result(self.result, self.state)

    def execute_command(self, store, **kwargs):
        """Commands need to implement this to return the command."""
        raise NotImplementedError('Commands need to provide this method')


def execute_command(command: Command, store):
    """Checks if the command is an instance of command and if so execute it."""
    if isinstance(command, Command):
        return command.execute(store)
    raise NotACommandException('Command is not instance of command!')


class CreateIndexCommand(Command):
    """Create a new index on the receiving storage

    For the parameters refer to the corresponding method in KNNStore.
    """
    def execute_command(self, store, **kwargs):
        return store.create_index(**kwargs)


class InitIndexCommand(Command):
    """Init the index."""
    def execute_command(self, store, **kwargs):
        return store.init_index(**kwargs)


class IndexExistsCommand(Command):
    """Check if the index exists."""
    def execute_command(self, store, **kwargs):
        return store.index_exists(**kwargs)


class DeleteIndexCommand(Command):
    """Delete an index."""
    def execute_command(self, store, **kwargs):
        return store.delete_index(**kwargs)


class AddItemsCommand(Command):
    """Add items to the specified index."""
    def execute_command(self, store, **kwargs):
        return store.add_items(**kwargs)


class SetEFCommand(Command):
    """Set the ef."""
    def execute_command(self, store, **kwargs):
        return store.set_ef(**kwargs)


class GetEFCommand(Command):
    """Get the ef."""
    def execute_command(self, store, **kwargs):
        return store.get_ef(**kwargs)


class GetEFConstructionCommand(Command):
    """get construction ef"""
    def execute_command(self, store, **kwargs):
        return store.get_ef_construction(**kwargs)


class QueryIndexCommand(Command):
    """Query the index."""
    is_cpu_heavy = True

    def execute_command(self, store, **kwargs):
        return store.query_index(**kwargs)


class DeleteItemCommand(Command):
    """Delete an item from the index."""
    def execute_command(self, store, **kwargs):
        return store.delete_item(**kwargs)


class ResizeIndexCommand(Command):
    """Resize the index to a new size"""
    is_cpu_heavy = True

    def execute_command(self, store, **kwargs):
        return store.resize_index(**kwargs)


class MaxElementsCommand(Command):
    """Get the current limit on the index."""
    def execute_command(self, store, **kwargs):
        return store.max_elements(**kwargs)


class CountCommand(Command):
    """Get the current amount of items."""
    def execute_command(self, store, **kwargs):
        return store.count(**kwargs)


class InfoCommand(Command):
    """Get information about the index."""
    def execute_command(self, store, **kwargs):
        return store.info(**kwargs)
