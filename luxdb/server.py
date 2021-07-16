"""This module provides a simple tcp server to connect to the database."""
import argparse
import asyncio
import logging
import socket

from luxdb.connection import receive_command, write_result
from luxdb.knn_store import KNNStore, open_store

LOG = logging.getLogger("server")


class Server:
    """TCP Server that allows clients to connect to the database.

    Receives commands, executes them on the database and sends the result back.

    """
    def __init__(self, host: str, port: int, store: KNNStore):
        """Define the host and port where the server will listen, and the store that should be used."""
        self.host = host
        self.port = port
        self.store = store
        self.server = None

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handles a single incoming connection asynchronously"""

        command = await receive_command(reader)

        while command is not None:
            result = await command.execute(self.store)

            await write_result(writer, result)

            command = await receive_command(reader)

        writer.close()
        await writer.wait_closed()

    async def create_server(self):
        """Creates the server socket but doesn't start listening yet."""
        self.server = await asyncio.start_server(self.connection, self.host, self.port)

        for server_socket in self.server.sockets:
            host, port = socket.getnameinfo(server_socket.getsockname(), socket.NI_NUMERICSERV | socket.NI_NUMERICHOST)
            LOG.info('Serving on %s:%s', host, port)
            self.host = host
            self.port = int(port)

    async def start(self, callback=None):
        """Start the server and process incoming connections.

        The callback is called shortly before the server starts listening, you can use it to notify
        clients that the server is listening, so they can connect without an error.
        """
        if self.server is None:
            await self.create_server()

        async with self.server:
            if callback is not None:
                asyncio.get_running_loop().call_soon(callback)
            await self.server.serve_forever()


async def serve(args: dict):
    """Main method for the server thread."""

    with open_store(args.path) as store:
        server = Server(host=args.host, port=args.port, store=store)
        try:
            await server.start()
        except KeyboardInterrupt:
            print("Shutting down server..")


def main():
    """Main method for the server."""
    try:
        parser = argparse.ArgumentParser(description="Multidimensional vector database (server).")

        parser.add_argument("--host", type=str, default='127.0.0.1', help="Host where the server should listen.")
        parser.add_argument("--port", type=int, default=None, help="Port to listen on.")
        parser.add_argument(
            '-log',
            '--loglevel',
            default='warning',
            help='Provide logging level. Example --loglevel debug, default=warning',
        )
        parser.add_argument("path", type=str, help="Path where the database is stored or should be stored.")
        args = parser.parse_args()
        logging.basicConfig(level=args.loglevel.upper())

        asyncio.run(serve(args), debug=args.loglevel.upper() == 'DEBUG')
    except KeyboardInterrupt:
        print("Shutting down server...")
