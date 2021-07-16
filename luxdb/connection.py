"""Helper functions to send and receive objects over socket streams."""
import asyncio
import logging
import pickle  # nosec

from luxdb.commands import Command, Result

LOG = logging.getLogger("connections")


async def send_obj(writer: asyncio.StreamWriter, obj: any) -> None:
    """Send object over the writer. first the size and after that the data."""
    data = pickle.dumps(obj)
    #data = json.dumps(obj)

    size = len(data)

    LOG.debug("Sending %d bytes of data.", size)

    writer.write(size.to_bytes(8, 'big'))
    writer.write(data)

    await writer.drain()


async def receive_obj(reader: asyncio.StreamReader) -> any:
    """Receive an object over the reader."""
    size = int.from_bytes(await reader.readexactly(8), 'big')

    if size == 0:
        LOG.info("Command with size 0 received, return None")
        return None

    LOG.debug("Expecting %d bytes of data", size)
    data = await reader.readexactly(size)
    LOG.debug("Received %d bytes of data", size)

    obj = pickle.loads(data)  # nosec
    #obj = json.loads(data)

    return obj


async def receive_command(reader: asyncio.StreamReader) -> Command:
    """Waits for the sender to send a command object."""
    cmd = await receive_obj(reader)

    if isinstance(cmd, Command):
        return cmd

    if cmd is None:
        return None

    raise Exception(f"Received unknown object: {cmd}")


async def receive_result(reader: asyncio.StreamReader) -> Result:
    """Receive a result over the reader."""
    result = await receive_obj(reader)

    if isinstance(result, Result):
        return result

    raise Exception(f"Received unknown object: {result}")


async def send_command(writer: asyncio.StreamWriter, cmd: Command) -> None:
    """Send the command over the writer."""
    await send_obj(writer, cmd)


async def write_result(writer: asyncio.StreamWriter, result: Result) -> None:
    """Send the result over the writer."""
    await send_obj(writer, result)


async def write_close(writer: asyncio.StreamWriter) -> None:
    """Sends an empty object to signal closing of the connection and then close the writer."""
    writer.write((0).to_bytes(8, 'big'))
    await writer.drain()
    writer.close()

    await writer.wait_closed()
