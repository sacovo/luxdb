import pickle

import numpy as np
import pytest

from luxdb.commands import (AddItemsCommand, Command, CommandState, CreateIndexCommand, InitIndexCommand,
                            execute_command)
from luxdb.exceptions import KNNBaseException, NotACommandException
from luxdb.knn_store import KNNStore
from tests import generate_data


class TestCommands:
    @pytest.mark.asyncio
    async def test_base_command(self):
        class TestCommand(Command):
            def execute_command(self, store: KNNStore, **kwargs):
                return (store, kwargs)

        store = KNNStore()

        kwargs = {'name': 'test_base', 'a': 1, 'b': 3}
        test_command = TestCommand(**kwargs)

        assert test_command.state == CommandState.CREATED
        assert test_command.result == None
        assert test_command.command_args == kwargs

        await test_command.execute(store)
        assert test_command.state == CommandState.SUCCEEDED
        assert test_command.result[0] == store
        assert test_command.result[1] == kwargs

        with pytest.raises(NotImplementedError):
            Command().execute_command(None)

    @pytest.mark.asyncio
    async def test_execute_non_command(self):
        store = KNNStore()
        with pytest.raises(NotACommandException):
            execute_command(object(), store)

    @pytest.mark.asyncio
    async def test_base_failing(self):
        class TestFailingCommand(Command):
            def execute_command(self, store: KNNStore, **kwargs):
                raise KNNBaseException()

        kwargs = {'name': 'test_base', 'a': 1, 'b': 3}

        store = KNNStore()
        command = TestFailingCommand(**kwargs)
        await command.execute(store)

        assert command.state == CommandState.FAILED
        assert isinstance(command.result, KNNBaseException)

        class TestFailingUnknownCommand(Command):
            def execute_command(self, store: KNNStore, **kwargs):
                raise Exception()

        command = TestFailingUnknownCommand(**kwargs)
        await command.execute(store)
        assert command.state == CommandState.FAILED
        assert isinstance(command.result, Exception)

    @pytest.mark.asyncio
    async def test_create_index_command(self):
        store = KNNStore()
        name = 'test-create'
        command = CreateIndexCommand(name=name, space='l2', dim=12)
        await command.execute(store)
        assert store.index_exists(name)
        assert command.state == CommandState.SUCCEEDED

        command = CreateIndexCommand(name=name, space='l2', dim=12)
        await command.execute(store)
        assert command.state == CommandState.FAILED

    @pytest.mark.asyncio
    async def test_add_items_command(self):
        store = KNNStore()
        name = 'test-add-items'
        dim = 12
        num_elements = 50

        await execute_command(CreateIndexCommand(name=name, space='l2', dim=dim), store)
        await execute_command(InitIndexCommand(name=name, max_elements=100), store)

        data, ids = generate_data(num_elements, dim)
        command = AddItemsCommand(name=name, data=data, ids=ids)
        await execute_command(command, store)

        assert command.state == CommandState.SUCCEEDED
        assert store.count(name) == num_elements

    @pytest.mark.asyncio
    def test_serialization(self):
        name = 'test'

        cmd = CreateIndexCommand(name=name, space='l2', dim=12)
        serialized = pickle.dumps(cmd)
        deserialized_cmd = pickle.loads(serialized)

        assert isinstance(deserialized_cmd, CreateIndexCommand)
        assert cmd.command_args['name'] == name

        data, ids = generate_data(100, 12)
        cmd = AddItemsCommand(name=name, data=data, ids=ids)

        deserialized_cmd = pickle.loads(pickle.dumps(cmd))
        assert isinstance(deserialized_cmd, AddItemsCommand)
        assert np.array_equal(data, deserialized_cmd.command_args['data'])
        assert np.array_equal(ids, deserialized_cmd.command_args['ids'])
