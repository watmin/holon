# Legacy API Deprecation Notice

## Background

In early 2026, we redesigned Holon's HTTP API to follow a "kernel + userland" philosophy inspired by Clojure. The original API was functional but had several issues:

1. **Overloaded `/encode` endpoint**: Three different encoding operations crammed together
2. **Missing basic CRUD operations**: No GET by ID, no proper resource hierarchy
3. **Everything was POST**: No proper HTTP method semantics
4. **Mixed concerns**: Data operations and vector operations weren't separated

## What Changed

### Removed Endpoints (Legacy API)

- `GET /health` → Use `GET /api/v1/health`
- `POST /insert` → Use `POST /api/v1/items`
- `POST /batch_insert` → Use `POST /api/v1/items/batch`
- `POST /query` → Use `POST /api/v1/search`
- `POST /encode` → Split into:
  - `POST /api/v1/vectors/encode` (structural data)
  - `POST /api/v1/vectors/encode/mathematical` (math primitives)
  - `POST /api/v1/vectors/compose` (vector composition)

### New Design Principles

1. **Kernel First**: Minimal, composable primitives for VSA/HDC operations
2. **Resource-Based**: Proper RESTful endpoints with HTTP methods
3. **Separation of Concerns**: Data operations vs vector operations
4. **Unified Client**: Same interface for local and remote usage

## Migration Guide

### Before (Legacy)
```python
# Direct HTTP calls
requests.post("/insert", json={"data": "..."})
requests.post("/query", json={"probe": "..."})
requests.post("/encode", json={"data": "..."})
```

### After (New API)
```python
# Using unified client (recommended)
from holon import HolonClient

client = HolonClient("http://localhost:8000")
client.insert_json({"type": "task", "title": "Review"})
results = client.search_json({"type": "task"})
vector = client.encode_vectors_json({"action": "login"})
```

### Direct HTTP (if needed)
```python
# Health
GET /api/v1/health

# CRUD operations
POST /api/v1/items
GET /api/v1/items/{id}
POST /api/v1/items/batch

# Search
POST /api/v1/search

# Vector operations
POST /api/v1/vectors/encode
POST /api/v1/vectors/encode/mathematical
POST /api/v1/vectors/compose
```

## Timeline

- **Removed**: Legacy endpoints removed immediately (no backward compatibility)
- **Current**: Only v1 API available
- **Future**: May add v2+ APIs with proper versioning and deprecation periods

## Why No Backward Compatibility

Since Holon currently has no external consumers (only internal development), we prioritized clean design over compatibility. This allows us to:

1. **Iterate quickly** on the API design
2. **Establish good patterns** from the start
3. **Avoid technical debt** from legacy endpoints
4. **Focus on userland innovation** rather than maintenance

## Impact on Existing Code

All existing challenge solutions and examples needed updates to use the new API. This was intentional - we want to ensure all our code demonstrates the clean, kernel-first approach.

## Lessons Learned

1. **Start with the kernel**: Design the minimal primitives first, build userland on top
2. **Unified interfaces matter**: Users shouldn't care about local vs remote
3. **Vector abstraction is key**: Users work with data, vectors stay internal
4. **Early breaking changes are OK** when you have no external users
