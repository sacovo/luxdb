import pickle
import secrets

import numpy as np
import pytest

from luxdb.commands import (AddItemsCommand, Command, CommandState, CreateIndexCommand, GetIdsCommand, GetItemsCommand,
                            InitIndexCommand, execute_command)
from luxdb.exceptions import KNNBaseException, NotACommandException
from luxdb.knn_store import KNNStore
from tests import generate_data


@pytest.fixture
def store():
    return KNNStore()


@pytest.fixture
def name():
    return secrets.token_hex()


class TestCommands:
    @pytest.mark.asyncio
    async def test_base_command(self):
        class TestCommand(Command):
            async def execute_command(self, store: KNNStore, **kwargs):
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
            await Command().execute_command(None)

    @pytest.mark.asyncio
    async def test_execute_non_command(self):
        store = KNNStore()
        with pytest.raises(NotACommandException):
            await execute_command(object(), store)

    @pytest.mark.asyncio
    async def test_base_failing(self):
        class TestFailingCommand(Command):
            async def execute_command(self, store: KNNStore, **kwargs):
                raise KNNBaseException()

        kwargs = {'name': 'test_base', 'a': 1, 'b': 3}

        store = KNNStore()
        command = TestFailingCommand(**kwargs)
        await command.execute(store)

        assert command.state == CommandState.FAILED
        assert isinstance(command.result, KNNBaseException)

        class TestFailingUnknownCommand(Command):
            async def execute_command(self, store: KNNStore, **kwargs):
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
        assert await store.count(name) == num_elements

    @pytest.mark.asyncio
    async def test_add_wrong_dimension(self, store, name):
        dim = 12
        num_elements = 50
        await execute_command(CreateIndexCommand(name=name, space='l2', dim=dim), store)
        await execute_command(InitIndexCommand(name=name, max_elements=100), store)

        data, ids = generate_data(num_elements, dim + 4)
        command = AddItemsCommand(name=name, data=data, ids=ids)

        with pytest.raises(RuntimeError):
            result = await execute_command(command, store)
            result.raise_error()

    @pytest.mark.asyncio
    async def test_init_twice(self, store, name):
        store.create_index(name, 'l2', 12)
        command = InitIndexCommand(name=name, max_elements=1000)
        await execute_command(command, store)

        with pytest.raises(RuntimeError):
            result = await execute_command(command, store)
            result.get_value()

    @pytest.mark.asyncio
    async def test_get_items_command(self, store: KNNStore, name):
        store.create_index(name, 'l2', 12)
        await store.init_index(name, 10000)
        data, ids = generate_data(500, 12)
        await execute_command(AddItemsCommand(name=name, data=data, ids=ids), store)

        get_items_command = GetItemsCommand(name=name, ids=ids)

        result = await execute_command(get_items_command, store)

        assert get_items_command.state == CommandState.SUCCEEDED
        assert np.array_equal(result.data, data)

        get_items_command = GetItemsCommand(name=name, ids=ids[:20])

        result = await execute_command(get_items_command, store)
        assert np.array_equal(result.data, data[:20])

        result = await execute_command(GetIdsCommand(name=name), store)
        assert np.array_equal(np.sort(result.data), ids)

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
