"""This module provides a simple tcp server to connect to the database."""
import argparse
import asyncio
import logging
import os
import signal
import socket
from functools import partial

from cryptography.fernet import InvalidToken
from luxdb.connection import gen_key, receive_command, send_close, send_result
from luxdb.knn_store import KNNStore, open_store

LOG = logging.getLogger('server')


def shutdown(server, signum):
    """Gracefully shut the server down."""
    LOG.info('%s, shutting down', signal.strsignal(signum))
    server.shutdown()


class Server:
    """TCP Server that allows clients to connect to the database.

    Receives commands, executes them on the database and sends the result back.

    """
    def __init__(self, host: str, port: int, store: KNNStore, secret):
        """Define the host and port where the server will listen, and the store that should be used."""
        self.host = host
        self.port = port
        self.store = store
        self.server = None
        self.secret = gen_key(secret)

    async def _next_command(self, reader, writer):
        try:
            return await receive_command(reader, self.secret)
        except InvalidToken:
            LOG.error('Invalid token received, closing connection.')
            await send_close(writer)
            writer.close()
            return None

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handles a single incoming connection asynchronously"""
        LOG.debug('New connection established.')

        LOG.debug('Waiting for command')
        command = await self._next_command(reader, writer)

        LOG.debug('Command received: %s', command)

        while command is not None:
            result = await command.execute(self.store)
            LOG.debug('Sending result to')

            await send_result(writer, result, self.secret)
            LOG.debug('Sent result to client.')

            LOG.debug('Waiting for command')
            command = await self._next_command(reader, writer)

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

    def shutdown(self):
        """Gracefully terminate the database server."""
        self.server.close()

    async def start(self, callback=None):
        """Start the server and process incoming connections.

        The callback is called shortly before the server starts listening, you can use it to notify
        clients that the server is listening, so they can connect without an error.
        """
        if self.server is None:
            await self.create_server()

        async with self.server:
            loop = asyncio.get_running_loop()
            if callback is not None:
                loop.call_soon(callback)

            await self.server.start_serving()
            await self.server.wait_closed()


async def serve(args: dict):
    """Main method for the server thread."""

    with open_store(args.path) as store:
        loop = asyncio.get_running_loop()
        server = Server(host=args.host, port=args.port, store=store, secret=args.secret)

        for signum in [signal.SIGTERM, signal.SIGINT]:
            loop.add_signal_handler(signum, partial(shutdown, server, signum))

        await server.start()


def main():
    """Main method for the server."""
    parser = argparse.ArgumentParser(description='Multidimensional vector database (server).')

    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host where the server should listen.')
    parser.add_argument('--port', type=int, default=None, help='Port to listen on.')
    parser.add_argument(
        '-log',
        '--loglevel',
        default='warning',
        help='Provide logging level. Example --loglevel debug, default=warning',
    )
    parser.add_argument('--secret',
                        type=str,
                        default=os.environ.get('LUXDB_SECRET', ''),
                        help='Pass the secret the server should use, if not specified it is taken from $LUXDB_SECRET')
    parser.add_argument('path', type=str, help='Path where the database is stored or should be stored.')
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel.upper())

    return asyncio.run(serve(args), debug=args.loglevel.upper() == 'DEBUG')
