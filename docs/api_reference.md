# Holon API Reference

## HTTP API

Holon provides a RESTful HTTP API for remote memory operations.

### Endpoints

#### POST /insert
Insert a data blob.

**Request**:
```json
{
  "data": "{\"user\": \"alice\", \"action\": \"login\"}",
  "data_type": "json"
}
```

**Response**:
```json
{
  "id": "uuid-string"
}
```

#### POST /batch_insert
Insert multiple data items efficiently with optimized indexing.

**Request**:
```json
{
  "items": [
    "{\"user\": \"alice\", \"action\": \"login\"}",
    "{\"user\": \"bob\", \"action\": \"logout\"}"
  ],
  "data_type": "json"
}
```

**Response**:
```json
{
  "ids": ["uuid1", "uuid2"]
}
```

#### POST /query
Query the memory.

**Request**:
```json
{
  "probe": "{\"user\": \"alice\"}",
  "data_type": "json",
  "top_k": 10,
  "threshold": 0.0,
  "guard": "{\"status\": \"success\"}",
  "negations": {"action": {"$not": "logout"}},
  "any_marker": "$any"
}
```

**Response**:
```json
{
  "results": [
    {
      "id": "uuid",
      "score": 0.85,
      "data": {"user": "alice", "action": "login"}
    }
  ]
}
```

#### GET /health
Health check.

**Response**:
```json
{
  "status": "healthy",
  "backend": "cpu",
  "items_count": 42
}
```

## Python API

### CPUStore Class

```python
from holon import CPUStore

store = CPUStore(dimensions=16000, backend='cpu')

# Insert
id = store.insert('{"key": "value"}', data_type='json')

# Query
results = store.query(
    '{"key": "value"}',
    top_k=10,
    threshold=0.0,
    guard={"status": "active"},
    negations={"error": {"$not": "true"}},
    any_marker="$any"
)

# Returns: List[Tuple[id, score, data]]

# Get by ID
data = store.get(id)

# Delete
success = store.delete(id)
```

### Parameters

- **probe**: JSON/EDN string for fuzzy search.
- **data_type**: "json" or "edn".
- **top_k**: Max results.
- **threshold**: Min similarity score.
- **guard**: JSON string for exact pattern matching.
- **negations**: Dict for value exclusions.
- **any_marker**: Custom wildcard marker (default "$any").

## Error Handling

- Invalid JSON: 400 Bad Request
- Not found: 404
- Server errors: 500