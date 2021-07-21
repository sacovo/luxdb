# LuxDB

This is a simple database for multidimensional vectors. It basically provides persistance and connectivity with asyncio to [Hnswlib](https://github.com/nmslib/hnswlib). The project contains the server and also a simple client.

Still under development, there will be breaking changes and you will loose data if you only store it in this database. So don't use it for anything that you want to keep.

### TODO
- Language agnostic transport layer =>
- Performance?
- Rollbacks, transactions, ...

## Features

In LuxDB you can store your vectors and query them for their nearest neighbors. Vectors are stored in **indexes** that have a name, a dimension and a metric. After you created the index you
need to initialize it with a maximum size. (See the code below for some usage examples.) And luxdb/client.py or luxdb/sync_client.py for a list of all available operations.

Changes are stored on the disk and can be used after the database is shutdown. This is achieved with [ZODB](https://zodb.org), each index is stored seperatly in a `OOBTree`. The store can be created with a path, in that case a `FileStorage` will be created there. You can also provide a Storage in the constructor of the storage. For testing you can omit `path` and `storage`, in that case the data will be stored in memory only.

Authentication and encryption happens with [Fernet](https://cryptography.io/en/latest/fernet/#), server and client need a shared secret in order to communicate. From this secret a key is derived, and this key is used to encrypt commands that are sent to the server. This guarantees, that only clients with the secret can execute commands on the server. The client needs the secret in the constructor, the server can either take it from the command line (--secret) or from an enviroment variable (LUXDB_SECRET).

## Usage

Start the server, either with docker:

```bash
docker run -p 8484:8484 registry.gitlab.com/sacovo/luxdb
docker run -p 8484:8484 registry.gitlab.com/sacovo/luxdb --loglevel=info --port 8484 --host 0.0.0.0
```

Or directly (after installing the dependencies in requirements.txt)
```
./luxdb-server --port 8484 --loglevel debug path/to/storage.db
```

The docker container stores the database in `/data/` so you can mount something there in order to store data.

Look into the [snippets](https://gitlab.com/sacovo/luxdb/-/snippets) to see some example configurations and code snippets that show how to use the client.

You can then use the client to connect to the server and add or retrieve data.

```python
from luxdb.client import connect
# Connect to the server
async with connect(host, port, secret) as client:
	name = 'my-index'
	# Create an index for 12 dimensional vectors
	await client.create_index(name, 'l2', 12)
	# Initialize the index
	await client.init_index(name, max_elements)
	# Add some data
	data = np.float32(np.random.random((1000, 12)))
	labels = np.arange(1000)
	await client.add_items(name, data, labels)
	# Search the nearest neighbors of data[0]
	found, distances = await client.query_index(name, data[0], k=5)
	# Or the nearest neighbors of all elements
	found, distances = await client.query_index(name, data, k=2)
	# You can resize the index after it was create
	await client.resize_index(name, new_size)
	# Get all ids that are stored
	ids = client.get_ids(name)
	# Get the element with specific ids
	elements = client.get_elements(name, [1, 3, 5])
```
For more usage examples you can check the tests in `tests/test_client.py`

## Environment variables

| Name             | Descripton                                                                          | Default                 |
|------------------|-------------------------------------------------------------------------------------|-------------------------|
| `LUXDB_SECRET`   | Secret used to encrypt and authenticate communication with clients.                 | `''`                    |
| `LUXBD_SALT`     | Salt used in key derivation from secret. Needs to be the same on client and server. | `'wYfJIy4Nx1hPcxiljwg'` |
| `KDF_ITERATIONS` | Iterations to use in key derivation                                                 | `1 << 18`               |
| `FERNET_TTL`     | Time in seconds that messages are valid after they are encrypted.                   | `60`                    |
|                  |                                                                                     |                         |

## Project structure
The project consists of a wrapper around a collection of `hnswlib.Index` objects, a server that performs modifications and lookups and a client. Communication between the server and the client happen through Command objects.

## Development

### Tests

The iteration count of the key derivation function can be changed through the environment variable `KDF_ITERATIONS`. To speed up tests you can set it to a low value. Don't set it to a low value in any other context!
