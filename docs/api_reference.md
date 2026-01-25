# Holon API Reference

See the main [README](../README.md) for overview and quick start.

## HTTP API

Holon provides a RESTful HTTP API for VSA/HDC neural memory operations with advanced querying capabilities.

### Endpoints

#### GET /health
Health check with backend and item count information.

**Response**:
```json
{
  "status": "healthy",
  "backend": "cpu|gpu|auto",
  "items_count": 42
}
```

#### POST /encode
Encode data into a vector without storing it. Useful for vector bootstrapping and analysis.

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
  "vector": [0.1, -0.2, 0.8, ...]
}
```

#### POST /insert
Insert a single data item.

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
Insert multiple data items efficiently with optimized bulk indexing. Defers ANN index rebuilds for better performance.

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
  ]
}
```

## Python API

### CPUStore Class

```python
from holon import CPUStore

# Initialize with auto GPU/CPU selection
store = CPUStore(dimensions=16000, backend='auto')  # 'cpu', 'gpu', or 'auto'

# Insert operations
single_id = store.insert('{"user": "alice", "action": "login"}', data_type='json')
batch_ids = store.batch_insert([
    '{"user": "bob", "action": "logout"}',
    '{"user": "charlie", "action": "edit"}'
], data_type='json')

# Advanced queries with complex guards
results = store.query(
    '{"user": "alice"}',  # Similarity probe
    top_k=10,
    threshold=0.0,
    guard={
        "$or": [  # Complex compound conditions
            {"status": "active", "role": "admin"},
            {"priority": "high", "department": "engineering"}
        ]
    },
    negations={"action": {"$not": "logout"}},
    any_marker="$any"
)

# Returns: List[Tuple[id, score, data_dict]]

# Vector encoding (bootstrapping)
vector = store.encoder.encode_data({"user": "alice"})

# Data retrieval
data = store.get(single_id)
success = store.delete(single_id)

# Bulk operations
store.start_bulk_insert()  # Defer ANN rebuilds
for item in large_dataset:
    store.insert(item)
store.end_bulk_insert()  # Rebuild ANN index once
```

### Advanced Query Features

#### Complex Guard Syntax
Guards support sophisticated compound conditions with `$or` logic:

```python
# Simple guards
store.query('{"role": "developer"}', guard={"status": "active"})

# Compound OR conditions
store.query('{}', guard={
    "$or": [
        {"priority": "high", "status": "todo"},
        {"project": "urgent", "category": "side"}
    ]
})

# Nested OR logic
store.query('{"project": "work"}', guard={
    "status": "active",
    "tags": {
        "$or": [
            {"$any": True},  # Any tagged items
            ["urgent", "critical"]  # OR specific tag arrays
        ]
    }
})

# Combined with negations
store.query(
    '{"project": "side"}',
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
store.query('{"role": {"$any": true}}')  # Matches any role value

# Wildcard in guard
store.query('{"tags": ["urgent"]}', guard={"role": {"$any": True}})
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

Use the `/encode` endpoint or `encoder.encode_data()` to create vectors for custom similarity operations:

```python
# Encode custom vectors for similarity
vector1 = store.encoder.encode_data({"type": "login"})
vector2 = store.encoder.encode_data({"type": "authentication"})

# Use for custom similarity calculations
similarity = store.vector_manager.similarity(vector1, vector2)
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
