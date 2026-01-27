# Holon API Reference

See the main [README](../README.md) for overview and quick start.

## Unified Client Interface

**Recommended**: Use `HolonClient` for a consistent interface whether working locally or remotely:

```python
from holon import CPUStore, HolonClient

# Local usage
store = CPUStore()
client = HolonClient(local_store=store)

# Remote usage
client = HolonClient(remote_url="http://localhost:8000")

# Same interface either way!
client.insert_json({"type": "task", "title": "Review code"})
results = client.search_json({"type": "task"})
```

The client abstracts all vector operations - you work with data, not vectors.

## HTTP API (v1)

Holon provides a RESTful HTTP API for VSA/HDC neural memory operations with advanced querying capabilities.

### Core Philosophy

Holon follows a "kernel + userland" philosophy like Clojure:
- **Kernel**: Minimal, composable primitives for VSA/HDC operations
- **Userland**: Domain-specific tools built on top (task managers, quote finders, etc.)

## V1 API (Recommended)

### Health & Status

#### GET /api/v1/health
Health check with backend and item count information.

**Response**:
```json
{
  "status": "healthy",
  "backend": "cpu|gpu|auto",
  "items_count": 42
}
```

### Item Management (Generic Data Storage)

#### POST /api/v1/items
Create a single item.

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
  "id": "uuid-string",
  "created": true
}
```

#### POST /api/v1/items/batch
Create multiple items efficiently.

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
  "ids": ["uuid1", "uuid2"],
  "created": 2
}
```

#### GET /api/v1/items/{id}
Retrieve a specific item by ID.

**Response**:
```json
{
  "id": "uuid-string",
  "data": {"user": "alice", "action": "login"}
}
```

### Search & Similarity

#### POST /api/v1/search
Advanced neural similarity search with guards, negations, and compound conditions.

**Request**:
```json
{
  "probe": "{\"user\": \"alice\"}",
  "data_type": "json",
  "top_k": 10,
  "threshold": 0.0,
  "guard": {
    "$or": [
      {"status": "active", "role": "admin"},
      {"priority": "high"}
    ]
  },
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
  ],
  "count": 1
}
```

### Vector Operations (VSA/HDC Primitives)

#### POST /api/v1/vectors/encode
Encode structured data into a vector for similarity operations.

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
  "vector": [0.1, -0.2, 0.8, ...],
  "encoding_type": "structural_json"
}
```

#### POST /api/v1/vectors/encode/mathematical
Encode mathematical primitives (fundamental VSA/HDC operations).

**Request**:
```json
{
  "primitive": "addition",
  "value": 5
}
```

**Response**:
```json
{
  "vector": [0.1, -0.2, 0.8, ...],
  "encoding_type": "mathematical_addition"
}
```

#### POST /api/v1/vectors/compose
Compose vectors using mathematical operations (bind/bundle).

**Request**:
```json
{
  "operation": "bind",
  "vectors": [[0.1, -0.2, 0.8, ...], [0.3, 0.5, -0.1, ...]]
}
```

**Response**:
```json
{
  "vector": [0.4, 0.3, 0.7, ...],
  "encoding_type": "compose_bind_2_vectors"
}
```


## Python API

### HolonClient Class

```python
from holon import CPUStore, HolonClient

# Initialize store and client
store = CPUStore(dimensions=16000, backend='auto')  # 'cpu', 'gpu', or 'auto'
client = HolonClient(local_store=store)

# Insert operations
single_id = client.insert_json({"user": "alice", "action": "login"})
batch_ids = client.insert_batch_json([
    {"user": "bob", "action": "logout"},
    {"user": "charlie", "action": "edit"}
])

# Advanced queries with complex guards
results = client.search_json(
    {"user": "alice"},  # Similarity probe
    top_k=10,
    threshold=0.0,
    guard={
        "$or": [  # Complex compound conditions
            {"status": "active", "role": "admin"},
            {"priority": "high", "department": "engineering"}
        ]
    },
    negations={"action": {"$not": "logout"}}
)

# Returns: List[Dict[id, score, data]]

# Vector encoding (bootstrapping)
vector = client.encode_vectors_json({"user": "alice"})

# Data retrieval
data = client.get(single_id)
success = client.delete(single_id)

# Bulk operations
client.insert_batch_json(large_dataset)  # Efficient batch insertion
```

### CPUStore Class (Advanced Usage)

For advanced users, testing, and direct vector operations, you can use CPUStore directly. This provides lower-level access to VSA/HDC primitives:

### Advanced Query Features

#### Complex Guard Syntax
Guards support sophisticated compound conditions with `$or` logic:

```python
# Simple guards
client.search_json({"role": "developer"}, guard={"status": "active"})

# Compound OR conditions
client.search_json({}, guard={
    "$or": [
        {"priority": "high", "status": "todo"},
        {"project": "urgent", "category": "side"}
    ]
})

# Nested OR logic
client.search_json({"project": "work"}, guard={
    "status": "active",
    "tags": {
        "$or": [
            {"$any": True},  # Any tagged items
            ["urgent", "critical"]  # OR specific tag arrays
        ]
    }
})

# Combined with negations
client.search_json(
    {"project": "side"},
    guard={
        "$or": [
            {"priority": "high"},
            {"priority": "medium", "status": "todo"}
        ]
    },
    negations={"status": {"$not": "waiting"}}
)
```

#### Negation Patterns
```python
# Simple exclusions
negations={"status": {"$not": "failed"}}

# Multiple exclusions
negations={"status": {"$not": ["failed", "error"]}}

# Nested negations
negations={"user.preferences": {"$not": {"theme": "dark"}}}
```

#### Wildcard Patterns
```python
# Wildcard in probe (doesn't match anything)
client.search_json({"role": {"$any": true}})  # Matches any role value

# Wildcard in guard
client.search_json({"tags": ["urgent"]}, guard={"role": {"$any": True}})
```

### Backend Options

- **'auto'**: Automatically selects GPU (if CuPy available) or CPU
- **'cpu'**: NumPy-based CPU operations (always available)
- **'gpu'**: CuPy-based GPU acceleration (requires CUDA)

### Performance Features

- **ANN Indexing**: Automatic FAISS integration for >1000 items (260x speedup)
- **Bulk Operations**: Optimized batch inserts with deferred index rebuilds
- **Parallel Processing**: Multi-core encoding for large datasets
- **Memory Efficient**: 70KB per item with intelligent vector management

For detailed performance benchmarks and optimization tips, see the [Performance Guide](performance.md).

### Data Types & Formats

#### JSON Support
Standard JSON with nested objects, arrays, and primitive types.

#### EDN Support
Extended Data Notation with richer semantics:
- **Keywords**: `:user`, `:admin` (self-evaluating identifiers)
- **Sets**: `#{:clojure :python :ml}` (unique collections)
- **Symbols**: `login`, `edit` (unquoted identifiers)
- **Rich Types**: Dates, UUIDs, and custom tagged elements

```python
# EDN examples
store.insert('{:user "alice" :role :admin}', data_type='edn')
store.insert('{:skills #{"clojure" "python"}}', data_type='edn')
```

### ANN Indexing & Scaling

- **Automatic**: Switches to FAISS ANN indexing when >1000 items
- **Performance**: 260x speedup vs brute-force similarity search
- **Accuracy**: 100% consistency with brute-force results
- **Memory**: Efficient for 10k-100k+ item datasets

### Vector Bootstrapping

Use the `/api/v1/vectors/encode` endpoint or `encoder.encode_data()` to create vectors for custom similarity operations:

```python
# HTTP API
POST /api/v1/vectors/encode
{
  "data": "{\"type\": \"login\"}",
  "data_type": "json"
}

# Python API
vector1 = client.encode_vectors_json({"type": "login"})
vector2 = client.encode_vectors_json({"type": "authentication"})

# Use for custom similarity calculations (advanced)
# Note: Client interface abstracts vector operations for typical use cases
```

## Error Handling

### HTTP Status Codes
- **400 Bad Request**: Invalid JSON, malformed queries, validation errors
- **404 Not Found**: Item not found by ID
- **500 Internal Server Error**: Server-side processing errors

### Common Error Patterns
- **Query probe encoding failures**: Check data structure matches expected format
- **Guard validation errors**: Ensure guard conditions are well-formed
- **Top-k limits exceeded**: System maximum is 100 results per query
- **ANN index rebuilding**: May cause temporary query slowdowns during bulk inserts

### Validation Rules
- `top_k`: Must be ≥ 1 and ≤ 100
- `threshold`: Must be between 0.0 and 1.0
- `data_type`: Must be "json" or "edn"
- `guard`: Must be valid dict structure (can contain `$or`)
- `negations`: Must use `{"$not": value}` pattern
