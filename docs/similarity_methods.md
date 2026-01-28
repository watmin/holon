# Holon Similarity Methods

## Overview

Holon supports Qdrant-native distance metrics for vector similarity search. Methods are selected via the `similarity` parameter in search operations.

## Available Methods

### 1. Cosine Similarity (Default)
```python
# Default cosine similarity
results = client.search(probe={"text": "query"})

# Or explicit cosine
results = client.search(probe={"text": "query"}, similarity="cosine")
```

**What it does**: Standard cosine similarity between vectors.

**When to use**: General-purpose similarity, baseline comparisons.

---

### 2. Euclidean Distance Similarity
```python
results = client.search(probe={"text": "query"}, similarity="euclidean")
```

**What it does**: Converts Euclidean distance to similarity (closer = more similar).

**When to use**: When absolute spatial distance matters more than directional similarity.

---

### 3. Manhattan Distance Similarity
```python
results = client.search(probe={"text": "query"}, similarity="manhattan")
```

**What it does**: Converts Manhattan (taxicab) distance to similarity.

**When to use**: When you want sum-of-absolute-differences style similarity.

---

### 4. Dot Product Similarity
```python
results = client.search(probe={"text": "query"}, similarity="dot_product")
```

**What it does**: Raw dot product similarity (magnitude-weighted).

**When to use**: When vector magnitudes should influence similarity scores.

---


## Choosing the Right Method

### For Simple Tasks
```python
# Use default cosine similarity
results = client.search(probe={"text": "query"})
```

### For Different Distance Measures
```python
# Use specific distance metrics
results = client.search(probe={"text": "query"}, similarity="euclidean")
results = client.search(probe={"text": "query"}, similarity="manhattan")
results = client.search(probe={"text": "query"}, similarity="dot_product")
```


## Performance Considerations

| Method | Speed | Accuracy Potential | Configuration Needed |
|--------|-------|-------------------|---------------------|
| Cosine | Fastest | Baseline | None |
| Euclidean | Fast | Good | None |
| Manhattan | Fast | Good | None |
| Dot Product | Fast | Good | None |

## Qdrant Compatibility

All similarity methods are **natively supported by Qdrant**:

- ✅ `cosine` - Qdrant's default and recommended metric
- ✅ `euclidean` - Qdrant's native euclidean distance
- ✅ `manhattan` - Qdrant's native manhattan distance
- ✅ `dot_product` - Qdrant's native dot product

**Implementation:** Direct mapping to Qdrant's distance metrics for optimal performance and native vector database operations.

## Usage Examples

All similarity methods work the same way - just specify the distance metric:

```python
# Different distance metrics for different use cases
results = client.search(probe={"text": "query"}, similarity="cosine")
results = client.search(probe={"text": "query"}, similarity="euclidean")
results = client.search(probe={"text": "query"}, similarity="manhattan")
results = client.search(probe={"text": "query"}, similarity="dot_product")
```

## Future Extensions

The similarity framework is designed to be extensible. New methods can be added by:
1. Implementing the method in `AdvancedSimilarityEngine`
2. Adding configuration handling in the client
3. Updating this documentation

This provides users with explicit control over similarity methods while maintaining clean configuration.
