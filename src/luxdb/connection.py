"""Helper functions to send and receive objects over socket streams."""
import asyncio
import base64
import logging
import os
import pickle  # nosec
import socket
from typing import Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from luxdb.commands import Command, CommandState, Result

LOG = logging.getLogger('connections')

INT_LENGTH = 8

SALT = os.environ.get('LUXDB_SALT', 'wYfJIy4Nx1hPcxiljwg').encode()
ITERATIONS = int(os.environ.get('KDF_ITERATIONS', '0')) or 1 << 18
FERNET_TTL = int(os.environ.get('FERNET_TTL', 0)) or 60  # Seconds


def gen_key(secret: str):
    """Return a key derived from the secert."""
    kdf = PBKDF2HMAC(
        hashes.SHA256(),
        32,
        salt=SALT,
        iterations=ITERATIONS,
    )
    return Fernet(base64.urlsafe_b64encode(kdf.derive(secret.encode())))


def pack_obj(obj, secret: Fernet) -> Tuple[int, bytes]:
    """Packs an object for sending and returns the size and the data"""
    data = secret.encrypt(pickle.dumps(obj))

    size = len(data)
    return size, data


async def send_obj(writer: asyncio.StreamWriter, obj: any, secret: Fernet) -> None:
    """Send object over the writer. first the size and after that the data."""
    size, data = pack_obj(obj, secret)

    LOG.debug('Sending %d bytes of data.', size)

    writer.write(size.to_bytes(INT_LENGTH, 'big'))
    writer.write(data)

    await writer.drain()


def send_obj_sync(writer: socket.socket, obj: any, secret: Fernet) -> None:
    """Sends an object to another socket synchronously."""
    size, data, = pack_obj(obj, secret)

    writer.sendall(size.to_bytes(INT_LENGTH, 'big'))
    writer.sendall(data)


async def receive_obj(reader: asyncio.StreamReader, secret: Fernet) -> any:
    """Receive an object over the reader."""
    size = int.from_bytes(await reader.readexactly(INT_LENGTH), 'big')

    if size == 0:
        LOG.info('Command with size 0 received, return None')
        return None

    LOG.debug('Expecting %d bytes of data', size)
    data = secret.decrypt(await reader.readexactly(size), FERNET_TTL)
    LOG.debug('Received %d bytes of data', size)

    obj = pickle.loads(data)  # nosec

    return obj


def receive_obj_sync(reader: socket.socket, secret: Fernet) -> any:
    """Receive an object over the socket synchronously."""
    size = int.from_bytes(reader.recv(INT_LENGTH, socket.MSG_WAITALL), 'big')

    if size == 0:
        LOG.info('Command with size 0 received, return None')
        return None

    data = secret.decrypt(reader.recv(size, socket.MSG_WAITALL), FERNET_TTL)

    obj = pickle.loads(data)  # nosec

    return obj


async def receive_command(reader: asyncio.StreamReader, secret: Fernet) -> Command:
    """Waits for the sender to send a command object."""
    cmd = await receive_obj(reader, secret)

    if isinstance(cmd, Command):
        return cmd

    if cmd is None:
        return None

    raise Exception(f'Received unknown object: {cmd}')


async def receive_result(reader: asyncio.StreamReader, secret: Fernet) -> Result:
    """Receive a result over the reader."""
    result = await receive_obj(reader, secret)

    if result is None:
        return Result(None, CommandState.FAILED)

    if isinstance(result, Result):
        return result

    raise Exception(f'Received unknown object: {result}')


def receive_result_sync(reader: socket.socket, secret: Fernet) -> Result:
    """Receive a result over the reader synchronously."""
    result = receive_obj_sync(reader, secret)

    if result is None:
        return Result(TypeError('Result was None'), CommandState.FAILED)

    if isinstance(result, Result):
        return result

    raise Exception(f'Received unknown object: {result}')


async def send_command(writer: asyncio.StreamWriter, cmd: Command, secret: Fernet) -> None:
    """Send the command over the writer."""
    await send_obj(writer, cmd, secret)


def send_command_sync(writer: socket.socket, cmd: Command, secret: Fernet) -> None:
    """Send the command over the writer synchronously."""
    send_obj_sync(writer, cmd, secret)


async def send_result(writer: asyncio.StreamWriter, result: Result, secret: Fernet) -> None:
    """Send the result over the writer."""
    await send_obj(writer, result, secret)


def send_result_sync(writer: socket.socket, result: Result, secret: Fernet) -> None:
    'Send the result over the writer synchronously.'
    send_obj_sync(writer, result, secret)


async def send_close(writer: asyncio.StreamWriter) -> None:
    """Sends an empty object to signal closing of the connection and then close the writer."""
    writer.write((0).to_bytes(INT_LENGTH, 'big'))
    await writer.drain()
    writer.close()

    await writer.wait_closed()


def send_close_sync(writer: socket.socket) -> None:
    """Sends an empty object to signal closing of the connection and then close the writer synchronously."""
    writer.sendall((0).to_bytes(INT_LENGTH, 'big'))
    writer.close()
