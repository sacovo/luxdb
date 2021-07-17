# LuxDB

This is a simple database for multidimensional vectors. It basically provides persistance and connectivity with asyncio to [Hnswlib](https://github.com/nmslib/hnswlib). The project contains the server and also a simple client.

Still under development, there will be breaking changes and you will loose data if you only store it in this database. So don't use it for anything that you want to keep.

### TODO
- ~~Sane storage backend (not pickle)~~ (Might still need some polishing)
- Language agnostic transport layer
- Performance?
- Rollbacks, transactions, ...
- Authentication

## (Lack of) Features

Persistence is achieved with [ZOBD](https://zodb.org), each index is stored seperatly in a `OOBTree`. The store can be created with a path, in that case a `FileStorage` will be created there. You can also provide a Storage in the constructor of the storage. For testing you can omit `path` and `storage`, in that case the data will be stored in memory only.

There is no authentication, you need to provide that through a proxy or make sure you are only allowing access to the database to trusted clients.

So there is just creation of indexes, adding items and searching for near neighbors in the indexes as well as storing them on the file system.

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
async with connect(host, port) as client:
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
```
For more usage examples you can check the tests in `tests/test_client.py`

## Project structure
The project consists of a wrapper around a collection of `hnswlib.Index`objects, a server that performs modifications and lookups and a client. Communication between the server and the client happen through Command objects.
