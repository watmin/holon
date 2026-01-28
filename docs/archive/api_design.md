# Holon API Design

## Abstract Store Interface

Holon's core API revolves around an abstract `Store` interface that provides insertion and querying capabilities, abstracting the underlying backend (CPU, GPU, or Remote). Users interact with a `Store` instance without needing to know the implementation details.

### Base Store Class

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class Store(ABC):
    @abstractmethod
    def insert(self, data: str, data_type: str = 'json') -> str:
        """
        Insert a data blob (JSON or EDN string) into the store.
        Returns a unique ID for the inserted data.

        :param data: The data blob as a string.
        :param data_type: 'json' or 'edn'.
        :return: Unique identifier for the data.
        """
        pass

    @abstractmethod
    def query(self, probe: str, data_type: str = 'json', top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Query the store with a probe data blob.
        Returns a list of (id, similarity_score, original_data) tuples for top matches.

        :param probe: The query probe as a string.
        :param data_type: 'json' or 'edn'.
        :param top_k: Number of top results to return.
        :param threshold: Minimum similarity score to include in results.
        :return: List of tuples (data_id, score, data_dict).
        """
        pass

    @abstractmethod
    def get(self, data_id: str) -> Dict[str, Any]:
        """
        Retrieve original data by ID.

        :param data_id: Unique identifier.
        :return: Original data as a dictionary.
        """
        pass

    @abstractmethod
    def delete(self, data_id: str) -> bool:
        """
        Delete data by ID.

        :param data_id: Unique identifier.
        :return: True if deleted, False otherwise.
        """
        pass
```

### Backend Implementations

- **CPUStore**: Local in-memory store using CPU for all operations.
- **GPUStore**: Local in-memory store with GPU acceleration.
- **RemoteStore**: Client for remote service, handling HTTP communication to a server using MongoDB and Qdrant.

### Usage Example

```python
from holon import CPUStore

# Initialize store
store = CPUStore(dimensions=16000)  # 16k dimensions

# Insert data
data_id = store.insert('{"name": "Alice", "age": 30}', 'json')

# Query similar data
results = store.query(probe='{"name": "Bob", "age": 25}', data_type='json', top_k=5, threshold=0.5)
for data_id, score, data in results:
    print(f"ID: {data_id}, Score: {score}, Data: {data}")

# Retrieve specific data
data = store.get(data_id)
```

## Configuration

Stores can be configured with parameters like:
- `dimensions`: Vector dimensionality (default 16000).
- `similarity_metric`: 'cosine', 'dot_product', etc. (default 'cosine').
- For RemoteStore: `host`, `port`, `api_key`, etc.

## Error Handling

- `ValueError`: For invalid data_type or malformed input.
- `KeyError`: For non-existent data IDs.
- `ConnectionError`: For RemoteStore connectivity issues.

## Future Extensions

- Batch insert/query methods for performance.
- Streaming interfaces for large datasets.
- Custom atomization rules or vector initialization schemes.
