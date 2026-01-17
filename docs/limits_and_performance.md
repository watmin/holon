# Holon Limits and Performance

## Overview

This document records performance characteristics, scaling limits, and probe effectiveness findings from extensive testing of the Holon VSA/HDC implementation.

## Implementation Details

### Vector Encoding
- **Dimensions:** 16,000 (configurable)
- **Dtype:** int8 (1 byte per dimension = 16KB per vector)
- **Encoding:** Recursive structural binding/bundling
  - Maps: `key * value` bindings bundled
  - Sequences: Items bundled (no position binding)
  - Sets: Items bundled with set indicator
  - Scalars: Atomic vectors for strings, numbers, keywords, etc.

### Memory Usage
- **Per vector:** 16KB (int8 × 16000)
- **Per item:** ~70KB (varies with unique atoms)
- **Scaling:** Linear with dataset size and atom diversity

## Performance Benchmarks

### Dataset: 10,000 complex JSON items
- **Insertion:** 7.22s (~1384 items/sec)
- **Memory:** 506MB total (0.051MB/item)
- **Unique atoms:** 19,113
- **Query time:** 0.23-0.48s per probe

### Scaling Projections
- **10k items:** 0.35s queries, 500MB memory
- **100k items:** ~3.5s queries, 5GB memory (estimated)
- **1M items:** ~35s queries, 50GB memory (impractical without indexing)

## Probe Effectiveness

### Score Ranges (0-1 normalized dot product)
- **Exact match:** 0.72 (strong, but <1.0 due to bundling)
- **Partial structure:** 0.13 (detectable substructure match)
- **Single atom match:** 0.25 (moderate, finds related items)
- **Rare atom:** 0.24 (surprisingly effective)
- **Unrelated:** 0.02 (weak, but returns results = false positives)
- **Empty:** 0.00 (perfect, no matches)

### Ineffective Probe Types

#### 1. Unrelated Probes
- **Example:** `{"completely": "different", "data": "here"}`
- **Score:** 0.019 (low but detectable)
- **Issue:** Returns 10 results (false positives)
- **Reason:** Atom collisions ("completely" as string) or vector noise creates spurious matches
- **Mitigation:** Threshold filtering (>0.05), better atom uniqueness

#### 2. Empty Probes
- **Example:** `{}`
- **Score:** 0.000 (perfect)
- **Issue:** No meaningful matches
- **Reason:** No atoms to correlate against dataset
- **Mitigation:** Validation/prevent empty queries

#### 3. Malformed Probes
- **Example:** `{"missing": "closing"`
- **Score:** N/A (parsing failure)
- **Issue:** Hard failure before similarity computation
- **Reason:** Invalid JSON structure prevents encoding
- **Mitigation:** Input validation and error handling

## Scaling Limits

### Memory
- **Atom explosion:** High diversity → many unique vectors
- **Limit:** ~50k items before 1GB+ memory
- **Mitigation:** Vector compression, atom deduplication

### Query Speed
- **Algorithm:** O(N) brute force similarity search
- **Limit:** >1s queries at 100k items
- **Mitigation:** Approximate indexing (HNSW, LSH), GPU acceleration

### Accuracy
- **False positives:** Unrelated probes return low-score matches
- **Partial matching:** Scores <0.5 even for good structural matches
- **Limit:** Reliable detection up to 50k items
- **Mitigation:** Score thresholding, hierarchical encoding

## Recommendations

### For Production Use (<50k items)
- Current implementation is suitable
- Use score thresholds (>0.1 for matches)
- Monitor atom count for memory

### For Large Scale (>100k items)
- Implement approximate nearest neighbor search
- Add vector quantization/compression
- Consider hierarchical data representations

### For Better Accuracy
- Experiment with different similarity metrics
- Tune vector dimensions (4k-32k)
- Implement query expansion for partial matches

## Test Environment
- **System:** 54GB RAM, 14 cores
- **Dataset:** 10k varied JSON objects
- **Framework:** Pure NumPy, no GPU acceleration

## Future Improvements
1. **Indexing:** HNSW for sub-linear queries
2. **Compression:** Vector quantization to reduce memory
3. **Hybrid:** Combine with traditional DB for metadata filtering
4. **GPU:** CUDA acceleration for large N
5. **Hierarchical:** Multi-resolution encoding for better partial matching