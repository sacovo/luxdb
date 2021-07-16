# LuxDB

This is a simple database for multidimensional vectors. It basically provides persistance and connectivity with asyncio to [Hnswlib](https://github.com/nmslib/hnswlib). The project contains the server and also a simple client. 

Still under heavy development, there will be breaking changes and you will loose data if you only store it in this database. So don't use it for anything that you want to keep.

### TODO
- Sane storage backend (not pickle)
- Language agnostic transport layer
- Performance?
- Rollbacks, transactions, ...
- Authentication

## (Lack of) Features
At the moment there is only simple persistence through python pickles, and data is only written to the disk at shutdown. This is only for the moment, and the project will probably use something like [ZODB](https://zodb.org/en/latest/index.html) to provide better persistence.

There is also no authentication, you need to provide that through a proxy or make sure you are only allowing access to the database to trusted clients.

So there is just creation of indexes, adding items and searching for near neighbors in the indexes.

```Python
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
