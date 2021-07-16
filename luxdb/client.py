"""Client to connect to a database server."""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict

import numpy.typing as npt

from luxdb.commands import (AddItemsCommand, CountCommand, CreateIndexCommand, DeleteIndexCommand, DeleteItemCommand,
                            GetEFCommand, GetEFConstructionCommand, IndexExistsCommand, InfoCommand, InitIndexCommand,
                            MaxElementsCommand, QueryIndexCommand, ResizeIndexCommand, Result, SetEFCommand)
from luxdb.connection import receive_result, send_command, write_close


class Client:
    """Client to connect to a database."""
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.reader = None
        self.writer = None

    async def connect(self):
        """Connect to the server"""
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def send_command(self, command) -> Result:
        """Send the command to the server and return the result the server sends back."""
        await send_command(self.writer, command)
        result = await receive_result(self.reader)

        return result.get_value()

    async def index_exists(self, name: str) -> bool:
        """Check if the index already exists."""
        command = IndexExistsCommand(name=name)
        return await self.send_command(command)

    async def create_index(self, name: str, space: str, dim: int) -> bool:
        """Create a new index with the given space (l2, ip, cosine) and dimension.

        More information about the parameters is available here:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        command = CreateIndexCommand(name=name, space=space, dim=dim)
        return await self.send_command(command)

    async def init_index(self, name: str, max_elements: int, ef_construction: int = 200, M: int = 16) -> None:
        """Initialize the index with the max_elements, ef_construction and M.

        More information about the parameters is available here:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        command = InitIndexCommand(name=name, max_elements=max_elements, ef_construction=ef_construction, M=M)
        return await self.send_command(command)

    async def delete_index(self, name: str) -> None:
        """Delete the index with the given name."""
        command = DeleteIndexCommand(name=name)
        return await self.send_command(command)

    async def add_items(self, name: str, data: npt.ArrayLike, ids: npt.ArrayLike) -> None:
        """Add the given items to the index."""
        command = AddItemsCommand(name=name, data=data, ids=ids)
        return await self.send_command(command)

    async def set_ef(self, name: str, new_ef: int):
        """Set the ef to a new value."""
        command = SetEFCommand(name=name, new_ef=new_ef)
        return await self.send_command(command)

    async def query_index(self, name: str, vector: npt.ArrayLike, k: int) -> npt.ArrayLike:
        """Find the k nearest neighbors of every element in vector.
        Returns a tuple with the labels of the found neighbors and the distances.
        """
        command = QueryIndexCommand(name=name, vector=vector, k=k)
        return await self.send_command(command)

    async def get_ef(self, name: str) -> int:
        """Get the ef value."""
        command = GetEFCommand(name=name)
        return await self.send_command(command)

    async def get_ef_construction(self, name: str) -> int:
        """Get the ef construction value"""
        command = GetEFConstructionCommand(name=name)
        return await self.send_command(command)

    async def delete_item(self, name: str, label: int) -> None:
        """Mark an item as deleted, this will exclude it from search results."""
        command = DeleteItemCommand(name=name, label=label)
        return await self.send_command(command)

    async def resize_index(self, name: str, new_size: int) -> None:
        """Resize the index to fit more ore less items"""
        command = ResizeIndexCommand(name=name, new_size=new_size)
        return await self.send_command(command)

    async def count(self, name: str) -> int:
        """Return the current amount of items in the index."""
        command = CountCommand(name=name)
        return await self.send_command(command)

    async def max_elements(self, name: str) -> int:
        """Return the maximal amount of items"""
        command = MaxElementsCommand(name=name)
        return await self.send_command(command)

    async def info(self, name: str) -> Dict:
        """Get collection of information about the index.

        Returns a dict with space, dim, M, ef_construction, ef, max_elements, element_count
        """
        command = InfoCommand(name=name)
        return await self.send_command(command)

    async def quit(self) -> None:
        """Quit the connection and inform the server about it."""
        await write_close(self.writer)


@asynccontextmanager
async def connect(host, port) -> Client:
    """Provides a context manager for a client connection to host and port"""
    client = Client(host, port)
    await client.connect()
    try:
        yield client
    finally:
        await client.quit()
