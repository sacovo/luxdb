"""Helper functions to send and receive objects over socket streams."""
import asyncio
import socket
import logging
import pickle  # nosec
from typing import Tuple

from luxdb.commands import Command, Result

LOG = logging.getLogger('connections')


def pack_obj(obj) -> Tuple[int, bytes]:
    """Packs an object for sending and returns the size and the data"""
    data = pickle.dumps(obj)
    size = len(data)
    return size, data


async def send_obj(writer: asyncio.StreamWriter, obj: any) -> None:
    """Send object over the writer. first the size and after that the data."""
    size, data = pack_obj(obj)

    LOG.debug('Sending %d bytes of data.', size)

    writer.write(size.to_bytes(8, 'big'))
    writer.write(data)

    await writer.drain()


def send_obj_sync(writer: socket.socket, obj: any) -> None:
    """Sends an object to another socket synchronously."""
    size, data = pack_obj(obj)

    writer.sendall(size.to_bytes(8, 'big'))
    writer.sendall(data)


async def receive_obj(reader: asyncio.StreamReader) -> any:
    """Receive an object over the reader."""
    size = int.from_bytes(await reader.readexactly(8), 'big')

    if size == 0:
        LOG.info('Command with size 0 received, return None')
        return None

    LOG.debug('Expecting %d bytes of data', size)
    data = await reader.readexactly(size)
    LOG.debug('Received %d bytes of data', size)

    obj = pickle.loads(data)  # nosec
    #obj = json.loads(data)

    return obj


def receive_obj_sync(reader: socket.socket) -> any:
    """Receive an object over the socket synchronously."""
    size = int.from_bytes(reader.recv(8, socket.MSG_WAITALL), 'big')

    if size == 0:
        LOG.info('Command with size 0 received, return None')
        return None

    data = reader.recv(size, socket.MSG_WAITALL)
    obj = pickle.loads(data)  # nosec

    return obj


async def receive_command(reader: asyncio.StreamReader) -> Command:
    """Waits for the sender to send a command object."""
    cmd = await receive_obj(reader)

    if isinstance(cmd, Command):
        return cmd

    if cmd is None:
        return None

    raise Exception(f'Received unknown object: {cmd}')


async def receive_result(reader: asyncio.StreamReader) -> Result:
    """Receive a result over the reader."""
    result = await receive_obj(reader)

    if isinstance(result, Result):
        return result

    raise Exception(f'Received unknown object: {result}')


def receive_result_sync(reader: socket.socket) -> Result:
    """Receive a result over the reader synchronously."""
    result = receive_obj_sync(reader)

    if isinstance(result, Result):
        return result

    raise Exception(f'Received unknown object: {result}')


async def send_command(writer: asyncio.StreamWriter, cmd: Command) -> None:
    """Send the command over the writer."""
    await send_obj(writer, cmd)


def send_command_sync(writer: socket.socket, cmd: Command) -> None:
    """Send the command over the writer synchronously."""
    send_obj_sync(writer, cmd)


async def send_result(writer: asyncio.StreamWriter, result: Result) -> None:
    """Send the result over the writer."""
    await send_obj(writer, result)


def send_result_sync(writer: socket.socket, result: Result) -> None:
    'Send the result over the writer synchronously.'
    send_obj_sync(writer, result)


async def send_close(writer: asyncio.StreamWriter) -> None:
    """Sends an empty object to signal closing of the connection and then close the writer."""
    writer.write((0).to_bytes(8, 'big'))
    await writer.drain()
    writer.close()

    await writer.wait_closed()


def send_close_sync(writer: socket.socket) -> None:
    """Sends an empty object to signal closing of the connection and then close the writer synchronously."""
    writer.sendall((0).to_bytes(8, 'big'))
    writer.close()
