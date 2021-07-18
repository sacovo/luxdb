"""Client to connect to a database server."""

from contextlib import contextmanager
import socket
from typing import Dict, List

from luxdb.commands import (AddItemsCommand, CountCommand, CreateIndexCommand, DeleteIndexCommand, DeleteItemCommand,
                            GetEFCommand, GetEFConstructionCommand, GetIdsCommand, GetIndexesCommand, GetItemsCommand,
                            IndexExistsCommand, InfoCommand, InitIndexCommand, MaxElementsCommand, QueryIndexCommand,
                            ResizeIndexCommand, Result, SetEFCommand)
from luxdb.connection import receive_result_sync, send_close_sync, send_command_sync


class SyncClient:
    """Client to connect to a database. This is the synchronous version of the client."""
    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.socket = None

    def connect(self):
        """Connect to the server"""
        self.socket = socket.create_connection((self.host, self.port))

    def send_command(self, command) -> Result:
        """Send the command to the server and return the result the server sends back."""
        send_command_sync(self.socket, command)
        result = receive_result_sync(self.socket)

        return result.get_value()

    def index_exists(self, name: str) -> bool:
        """Check if the index already exists."""
        command = IndexExistsCommand(name=name)
        return self.send_command(command)

    def create_index(self, name: str, space: str, dim: int) -> bool:
        """Create a new index with the given space (l2, ip, cosine) and dimension.

        More information about the parameters is available here:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        command = CreateIndexCommand(name=name, space=space, dim=dim)
        return self.send_command(command)

    def init_index(self, name: str, max_elements: int, ef_construction: int = 200, M: int = 16) -> None:
        """Initialize the index with the max_elements, ef_construction and M.

        More information about the parameters is available here:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        """
        command = InitIndexCommand(name=name, max_elements=max_elements, ef_construction=ef_construction, M=M)
        return self.send_command(command)

    def delete_index(self, name: str) -> None:
        """Delete the index with the given name."""
        command = DeleteIndexCommand(name=name)
        return self.send_command(command)

    def add_items(self, name: str, data, ids) -> None:
        """Add the given items to the index."""
        command = AddItemsCommand(name=name, data=data, ids=ids)
        return self.send_command(command)

    def set_ef(self, name: str, new_ef: int):
        """Set the ef to a new value."""
        command = SetEFCommand(name=name, new_ef=new_ef)
        return self.send_command(command)

    def query_index(self, name: str, vector, k: int):
        """Find the k nearest neighbors of every element in vector.
        Returns a tuple with the labels of the found neighbors and the distances.
        """
        command = QueryIndexCommand(name=name, vector=vector, k=k)
        return self.send_command(command)

    def get_ef(self, name: str) -> int:
        """Get the ef value."""
        command = GetEFCommand(name=name)
        return self.send_command(command)

    def get_ef_construction(self, name: str) -> int:
        """Get the ef construction value"""
        command = GetEFConstructionCommand(name=name)
        return self.send_command(command)

    def delete_item(self, name: str, label: int) -> None:
        """Mark an item as deleted, this will exclude it from search results."""
        command = DeleteItemCommand(name=name, label=label)
        return self.send_command(command)

    def resize_index(self, name: str, new_size: int) -> None:
        """Resize the index to fit more ore less items"""
        command = ResizeIndexCommand(name=name, new_size=new_size)
        return self.send_command(command)

    def count(self, name: str) -> int:
        """Return the current amount of items in the index."""
        command = CountCommand(name=name)
        return self.send_command(command)

    def max_elements(self, name: str) -> int:
        """Return the maximal amount of items"""
        command = MaxElementsCommand(name=name)
        return self.send_command(command)

    def info(self, name: str) -> Dict:
        """Get collection of information about the index.

        Returns a dict with space, dim, M, ef_construction, ef, max_elements, element_count
        """
        command = InfoCommand(name=name)
        return self.send_command(command)

    def get_indexes(self) -> None:
        """Return a list of all indexes in the db"""
        command = GetIndexesCommand()
        return self.send_command(command)

    def quit(self) -> None:
        """Quit the connection and inform the server about it."""
        send_close_sync(self.socket)

    def get_items(self, name: str, ids: List[int]):
        """Return array with the items with the id"""
        command = GetItemsCommand(name=name, ids=ids)
        return self.send_command(command)

    def get_ids(self, name: str):
        """Return all ids in the index."""
        command = GetIdsCommand(name=name)
        return self.send_command(command)


@contextmanager
def connect(host, port) -> SyncClient:
    """Provides a context manager for a client connection to host and port"""
    client = SyncClient(host, port)
    client.connect()
    try:
        yield client
    finally:
        client.quit()
