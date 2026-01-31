# Dimension Selection Guide

## Quick Reference

| Use Case | Recommended | Records/GB | Why |
|----------|-------------|------------|-----|
| Simple documents (<20 fields) | 1024 | ~817K | Fast, memory efficient |
| Complex documents (20-100 fields) | 4096 | ~233K | Best accuracy/memory balance |
| Very complex (100+ fields, time encoding) | 4096-8192 | ~119K-233K | Handles high field counts |
| Maximum headroom (unknown complexity) | 16384 | ~60K | Rarely needed |

## The Trade-offs

Higher dimensions provide:
- More orthogonal random vectors (less interference)
- Better discrimination for complex structures
- Higher bundle capacity (combining many items)

Lower dimensions provide:
- Faster queries (2x speed at 1024 vs 4096)
- Lower memory usage (linear scaling)
- Sufficient accuracy for most use cases

## Empirical Findings

### Where Dimensions Matter

| Test | 512d | 1024d | 4096d | 16384d |
|------|------|-------|-------|--------|
| 100-field doc precision | **40%** ❌ | 100% | 100% | 100% |
| Category discrimination | 100% | 100% | 100% | 100% |
| Bundle capacity (1000) | 100% | 100% | 100% | 100% |
| Near-duplicate score gap | 0.217 | 0.194 | 0.186 | 0.187 |
| Time encoding score | 0.808 | 0.783 | **0.858** | 0.825 |

**Key insight**: 512 dimensions fail at 100+ field documents. 1024+ handles everything we tested.

### Query Speed

| Dimension | Queries/sec | ms/query |
|-----------|-------------|----------|
| 512 | 647 | 1.55 |
| 1024 | 605 | 1.65 |
| 2048 | 458 | 2.19 |
| 4096 | 252 | 3.97 |
| 8192 | 260 | 3.84 |
| 16384 | 161 | 6.21 |

## Capacity Planning

### Memory Requirements

| Records | 1024d | 4096d | 16384d |
|---------|-------|-------|--------|
| 1M | 1.2 GB | 4.3 GB | 16.6 GB |
| 10M | 12 GB | 43 GB | 166 GB |
| 100M | 122 GB | 430 GB | 1.7 TB |
| 1B | 1.2 TB | 4.3 TB | 16.6 TB |

For billion-record scale, use Qdrant or similar vector database for persistence.

## Recommendations by Scenario

### Scenario 1: DynamoDB-style Document Store
**Use 4096 dimensions**
- Documents typically have 10-50 fields
- Time encoding for temporal queries
- Best balance of accuracy and capacity

### Scenario 2: Log/Event Storage
**Use 1024 dimensions**
- High volume, simpler structures
- Speed matters more than complexity handling
- 3.7x more records per GB than 4096

### Scenario 3: Complex Nested Data (GraphQL, FHIR, etc.)
**Use 4096-8192 dimensions**
- Deep nesting with many fields
- Need reliable discrimination at all levels

### Scenario 4: Prototyping/Development
**Use 1024 dimensions**
- Fast iteration
- Can always increase later
- Sufficient for testing

## VSA/HDC Theory

The theoretical capacity of hyperdimensional vectors scales approximately with √d:
- Bundling capacity: ~√d items can be superposed cleanly
- Binding capacity: ~d/log(d) key-value pairs

In practice, Holon's encoding is more robust due to:
- Bipolar thresholding after operations
- Role-based binding (keys bound to values)
- Recursive structure encoding

## Changing Dimensions

Dimensions are set at store creation and cannot be changed for existing vectors:

```python
from holon import CPUStore, HolonClient

# Create with specific dimensions
store = CPUStore(dimensions=4096)
client = HolonClient(local_store=store)
```

To migrate to different dimensions:
1. Create new store with desired dimensions
2. Re-insert all documents (re-encoding required)
3. Vectors from different dimensions are incompatible

## Benchmark Scripts

Run the benchmarks yourself:

```bash
# Accuracy vs dimension
./scripts/run_with_venv.sh python scripts/benchmarks/dimension_accuracy_benchmark.py

# Stress test to find limits
./scripts/run_with_venv.sh python scripts/benchmarks/dimension_stress_benchmark.py
```
