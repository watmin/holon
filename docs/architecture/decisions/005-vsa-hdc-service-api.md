# ADR 005: VSA/HDC-as-a-Service API Design

## Status
Proposed

## Context

We want to offer Holon as a hosted service where:
- **Storage**: MongoDB-like for records, Qdrant for vectors
- **Compute**: GPU-accelerated vector operations
- **API**: Generic primitives that apps can build against

The API should be:
1. **Complete** - All VSA/HDC primitives available
2. **Composable** - Operations can be combined
3. **Efficient** - Minimize round trips for common patterns
4. **Stateless** - Each request is self-contained (storage is external)

## Current API Inventory

### What We Have

```
Storage:
  POST   /api/v1/items              - Insert single item
  POST   /api/v1/items/batch        - Batch insert
  GET    /api/v1/items/{id}         - Get by ID
  DELETE /api/v1/items/{id}         - Delete by ID
  POST   /api/v1/store/clear        - Clear all

Query:
  POST   /api/v1/search             - Similarity search with guards/negations
  POST   /api/v1/search/by-vector   - Search using raw vector

Vector Operations:
  POST   /api/v1/vectors/encode     - Encode data → vector
  POST   /api/v1/vectors/compose    - Bind/bundle operations
  POST   /api/v1/vectors/similarity - Compute similarity

Diagnostics:
  GET    /api/v1/health             - Health check
  GET    /api/v1/diagnostics        - Performance stats
```

### What's Missing

1. **Collections/Namespaces** - Separate vector spaces per use case
2. **Unbind** - Reverse of bind (essential for VSA)
3. **Permute** - Sequence position encoding
4. **Batch vector operations** - Encode many, bind many
5. **Vector arithmetic** - Add, subtract, scale
6. **Resonance/Cleanup** - VSA memory cleanup
7. **Projection** - Return only specific fields
8. **Pagination** - Cursor for large result sets

## Proposed API Design

### Core Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOLON API PHILOSOPHY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Layer 1: PRIMITIVES (kernel)                                    │
│    encode, bind, bundle, unbind, permute, similarity             │
│    These are the atoms. Never change.                            │
│                                                                   │
│  Layer 2: STORAGE (persistence)                                  │
│    insert, get, delete, search                                   │
│    Backed by Qdrant + MongoDB                                    │
│                                                                   │
│  Layer 3: COMPOSITION (convenience)                              │
│    batch operations, pipelines, transactions                     │
│    Built from primitives                                         │
│                                                                   │
│  Layer 4: USERLAND (apps)                                        │
│    Sudoku solver, quote finder, semantic search...               │
│    Built by users, not us                                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Proposed Endpoints

#### 1. Collections (Namespaces)

```http
POST   /api/v1/collections
  Create a new collection with its own vector space
  Request:  { "name": "sudoku_patterns", "dimensions": 16384 }
  Response: { "id": "col_xxx", "name": "sudoku_patterns", "dimensions": 16384 }

GET    /api/v1/collections
  List all collections

GET    /api/v1/collections/{name}
  Get collection metadata

DELETE /api/v1/collections/{name}
  Delete collection and all its items
```

#### 2. Items (Storage)

```http
POST   /api/v1/collections/{name}/items
  Insert item into collection
  Request:  { "data": {...}, "vector": [...] (optional) }
  Response: { "id": "item_xxx", "vector_id": "vec_xxx" }

POST   /api/v1/collections/{name}/items/batch
  Batch insert
  Request:  { "items": [{...}, {...}] }
  Response: { "ids": [...], "count": N }

GET    /api/v1/collections/{name}/items/{id}
  Get item by ID
  Query params: ?include_vector=true

DELETE /api/v1/collections/{name}/items/{id}
  Delete item
```

#### 3. Search (Query)

```http
POST   /api/v1/collections/{name}/search
  Similarity search
  Request: {
    "probe": {...} | [...],      // Data or vector
    "limit": 10,
    "threshold": 0.0,
    "guard": {...},              // Must match
    "exclude": {...},            // Must not match (renamed from negations)
    "include_vectors": false,    // Return vectors with results
    "projection": ["field1"]     // Only return these fields
  }
  Response: {
    "results": [{ "id": "...", "score": 0.87, "data": {...} }],
    "count": N,
    "cursor": "..." (if more results)
  }

POST   /api/v1/collections/{name}/search/cursor/{cursor_id}
  Continue paginated search
```

#### 4. Vector Primitives (The Kernel)

```http
POST   /api/v1/vectors/encode
  Encode structured data to vector
  Request:  { "data": {...}, "config": {...} }
  Response: { "vector": [...], "dimensions": 16384 }

POST   /api/v1/vectors/encode/batch
  Batch encode
  Request:  { "items": [{...}, {...}] }
  Response: { "vectors": [[...], [...]] }

POST   /api/v1/vectors/bind
  Bind vectors (role-filler composition)
  Request:  { "vectors": [[...], [...]] }
  Response: { "vector": [...] }

POST   /api/v1/vectors/unbind
  Unbind vectors (reverse of bind)
  Request:  { "bound": [...], "key": [...] }
  Response: { "vector": [...] }

POST   /api/v1/vectors/bundle
  Bundle vectors (superposition)
  Request:  { "vectors": [[...], [...]], "weights": [1.0, 0.5] (optional) }
  Response: { "vector": [...] }

POST   /api/v1/vectors/permute
  Permute vector (sequence position encoding)
  Request:  { "vector": [...], "positions": 3 }
  Response: { "vector": [...] }

POST   /api/v1/vectors/similarity
  Compute similarity between vectors
  Request:  { "a": [...], "b": [...], "method": "cosine" }
  Response: { "similarity": 0.87 }

POST   /api/v1/vectors/similarity/batch
  Batch similarity (one query vs many)
  Request:  { "query": [...], "candidates": [[...], [...]] }
  Response: { "similarities": [0.87, 0.65, ...] }
```

#### 5. Vector Arithmetic

```http
POST   /api/v1/vectors/add
  Add vectors
  Request:  { "vectors": [[...], [...]] }
  Response: { "vector": [...] }

POST   /api/v1/vectors/subtract
  Subtract vectors (a - b)
  Request:  { "a": [...], "b": [...] }
  Response: { "vector": [...] }

POST   /api/v1/vectors/scale
  Scale vector by factor
  Request:  { "vector": [...], "factor": 0.5 }
  Response: { "vector": [...] }

POST   /api/v1/vectors/normalize
  Normalize vector
  Request:  { "vector": [...] }
  Response: { "vector": [...] }

POST   /api/v1/vectors/threshold
  Apply bipolar thresholding
  Request:  { "vector": [...] }
  Response: { "vector": [...] }
```

#### 6. Pipelines (Composition)

```http
POST   /api/v1/pipeline
  Execute multiple operations in sequence
  Request: {
    "operations": [
      { "op": "encode", "data": {...}, "as": "vec1" },
      { "op": "encode", "data": {...}, "as": "vec2" },
      { "op": "bind", "vectors": ["$vec1", "$vec2"], "as": "bound" },
      { "op": "search", "collection": "patterns", "probe": "$bound" }
    ]
  }
  Response: {
    "results": {
      "vec1": [...],
      "vec2": [...],
      "bound": [...],
      "search": [...]
    }
  }
```

#### 7. Cleanup/Resonance (VSA Memory)

```http
POST   /api/v1/vectors/cleanup
  Clean up noisy vector against known patterns
  Request: {
    "vector": [...],
    "memory": [[...], [...]] | "collection_name",
    "iterations": 3
  }
  Response: { "vector": [...], "best_match_score": 0.95 }
```

## API Examples

### Example 1: Semantic Search (Simple)

```python
# Insert documents
client.post("/collections/docs/items", {
    "data": {"title": "VSA Tutorial", "content": "..."}
})

# Search
results = client.post("/collections/docs/search", {
    "probe": {"content": "hyperdimensional computing"},
    "limit": 10
})
```

### Example 2: Constraint Encoding (Advanced)

```python
# Encode constraint pattern
cell_vec = client.post("/vectors/encode", {"data": {"row": 0, "col": 0}})
digit_vec = client.post("/vectors/encode", {"data": {"digit": 5}})

# Bind position to digit
placement = client.post("/vectors/bind", {
    "vectors": [cell_vec["vector"], digit_vec["vector"]]
})

# Store
client.post("/collections/sudoku/items", {
    "data": {"row": 0, "col": 0, "digit": 5},
    "vector": placement["vector"]
})
```

### Example 3: Pipeline (Efficient)

```python
# Single request for multiple operations
result = client.post("/pipeline", {
    "operations": [
        {"op": "encode", "data": {"type": "query", "text": "find cats"}, "as": "query"},
        {"op": "search", "collection": "images", "probe": "$query", "limit": 5}
    ]
})
# Returns both the encoded query and search results in one round trip
```

## Implementation Phases

### Phase 1: Core Primitives (Current)
- [x] encode, bind, bundle
- [x] similarity
- [x] insert, get, delete, search
- [ ] unbind, permute
- [ ] batch operations

### Phase 2: Collections
- [ ] Collection CRUD
- [ ] Per-collection dimensions
- [ ] Collection-scoped search

### Phase 3: Advanced Query
- [ ] Projection
- [ ] Pagination/cursors
- [ ] Compound guards ($and, $or, $not)

### Phase 4: Pipelines
- [ ] Multi-operation pipelines
- [ ] Variable references
- [ ] Conditional operations

### Phase 5: Persistence
- [ ] MongoDB integration for records
- [ ] Qdrant integration for vectors
- [ ] Synchronization

## Consequences

### Positive
- **Composable**: Apps can build anything from primitives
- **Efficient**: Pipelines reduce round trips
- **Scalable**: Collections enable multi-tenancy
- **Complete**: All VSA/HDC operations available

### Negative
- **Complexity**: More endpoints to maintain
- **Learning curve**: Users need to understand VSA/HDC
- **Latency**: HTTP overhead for fine-grained operations

### Mitigations
- Good documentation with examples
- Client libraries that abstract common patterns
- Pipeline API for batch operations
- WebSocket option for low-latency use cases

## References
- [Current API Reference](../../api_reference.md)
- [VSA/HDC Architecture](001-vsa-hdc-architecture.md)
- [Sudoku Solver Findings](../../sudoku_geometric_solution_findings.md)
