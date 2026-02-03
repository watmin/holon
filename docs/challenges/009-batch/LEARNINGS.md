# Challenge 009 Learnings: Deterministic Training via Symbolic Program Synthesis

## Overview

This challenge explored **learning encoding programs** for VSA/HDC instead of hand-tuning them. The core idea: with structured data, we can use symbolic search over primitive compositions rather than gradient descent.

## Key Findings

### 1. Weight Learning Works

We successfully learned field weights that improve classification accuracy:

| Metric | Uniform Weights | Learned Weights | Improvement |
|--------|-----------------|-----------------|-------------|
| Clean data | 89% | 100% | +11% |
| 10% noisy labels | 77% | 100% | +23% |
| 30% noisy labels | 83% | 100% | +17% |

**Conclusion**: Learned weights consistently outperform uniform weights, even with significant label noise.

### 2. VSA Encoding is Inherently Robust

Even with 30% of training labels mislabeled, we achieved 100% accuracy on clean test data. This robustness comes from:

- **High-dimensional bundling**: Noise averages out in high dimensions
- **Prototype voting**: Incorrect examples get outvoted by correct ones
- **Similarity structure**: The geometric relationships survive noise

This suggests VSA/HDC may be uniquely suited for noisy real-world data.

### 3. Cross-Validation Prevents Overfitting

Without cross-validation, the optimizer found spurious correlations in noise fields. With CV:

| Technique | Noise Field Weight | Signal Field Weight | Quality |
|-----------|-------------------|---------------------|---------|
| No CV, no regularization | 2.37 | 0.71 | Overfit |
| CV + regularization | 1.00 | 1.37 | Better |

Cross-validation forces the optimizer to find weights that generalize.

### 4. Training is Fast

| Method | Time | Accuracy |
|--------|------|----------|
| Coordinate descent | 46s | 99.6% |
| Hill climb | 5-7s | 100% |
| Random search | 6-7s | 100% |

For 500-1000 training examples, weight learning takes seconds, not hours.

### 5. Regularization Keeps Weights Interpretable

Without regularization, weights can become extreme (0.0 to 3.0+). With L2 penalty:

```python
weight_penalty = sum((w - 1.0) ** 2 for w in weights.values())
accuracy = accuracy - regularization * weight_penalty
```

Weights stay near 1.0, making them more interpretable and less prone to overfitting.

---

## What Makes This "Deterministic Training"?

Traditional neural training:
```
data → neural network → gradient descent (stochastic) → weights → predictions
```

Our approach:
```
data → VSA primitives → symbolic search (deterministic) → composition program → predictions
```

Key differences:

| Aspect | Neural Training | Symbolic Synthesis |
|--------|-----------------|-------------------|
| Search method | Gradient descent | Grid/random/coordinate |
| Randomness | Random init, SGD, dropout | Deterministic (given seed) |
| Interpretability | Black box weights | Named field weights |
| Reproducibility | Approximate | Exact |
| Hardware | GPU required | CPU sufficient |

---

## Limitations Discovered

### 1. Synthetic Data is Too Easy

Our synthetic ticket data has clean separation between classes. Real data has:
- Overlapping categories
- Missing fields
- Ambiguous cases

We need to test on messier real-world data.

### 2. Signal/Noise Discrimination is Imperfect

The optimizer found that noise fields could have high weights (1.44) similar to signal fields (1.37). This suggests:
- Random search finds local optima
- More sophisticated search (Bayesian optimization?) might help
- Feature selection as preprocessing could help

### 3. Scalability Unknown

We tested with 500-1000 examples and 7 fields. Open questions:
- Does this scale to 100+ fields?
- What about 100k+ training examples?
- How does encoding time scale?

---

## Next Steps

### Phase 2: Interaction Discovery ✓ COMPLETE

Discovered how to find field pairs that should be bound together:
- Key insight: zero individual field weights when testing interactions
- Successfully solves XOR (54% → 100%)
- Automatically finds (priority, urgency) in interaction data

### Phase 3: Full Program Synthesis ✓ COMPLETE

Genetic algorithm evolves encoding programs that compose primitives:

```python
# Actually synthesized program (from 007-program-synthesis.py):
def encode(item, encoder):
    vectors = []
    # Discovered: bind(type, tier) is discriminative
    if 'type' in item and 'tier' in item:
        a = encoder.encode_data({'type': item['type']})
        b = encoder.encode_data({'tier': item['tier']})
        vectors.append(2.0 * encoder.bind(a, b))
    # Additional components...
    bundled = np.sum(vectors, axis=0)
    return threshold_bipolar(bundled)
```

Results:
- Baseline: 83% → Synthesized: 84%
- Correctly discovered `bind(type, tier)` as discriminative
- Generates executable Python code from evolved programs

### Primitives Inventory

The inventory (006) revealed we're only using 3 of 11+ available primitives:

| Category | Primitives | Used in 009? |
|----------|------------|--------------|
| Composition | bind, bundle | ✓ Used |
| Learning | prototype | ✓ Used |
| Signal | negate, amplify, resonance | ✗ Underutilized |
| Comparison | difference, blend | ✗ Underutilized |
| Sequence | permute, cleanup | ✗ Underutilized |
| Encoders | TorchHD, Enhanced, Semantic | ✗ Underutilized |

### Phase 4: Distance Metrics Extension ✓ COMPLETE

Added comprehensive distance metrics to Holon:

| Metric | Type | Qdrant Native? | Best For |
|--------|------|---------------|----------|
| Cosine | Similarity | ✓ | Semantic similarity |
| Dot Product | Similarity | ✓ | Normalized vectors |
| Euclidean | Distance | ✓ | Geometric relationships |
| Manhattan | Distance | ✓ | Grid-like spaces |
| **Hamming** | Distance | ✗ | **Bipolar VSA vectors** |
| **Overlap** | Similarity | ✗ | **Shared features** |
| **Agreement** | Similarity | ✗ | **Balanced bipolar view** |
| Chebyshev | Distance | ✗ | Outlier sensitivity |
| Weighted Cosine | Similarity | ✗ | **Learned importance** |

Key findings:
- **Cosine ↔ Agreement: 100% correlated** for bipolar vectors (identical information)
- For well-separated data, metric choice doesn't matter much
- Metric choice matters most for edge cases and noisy data
- Weighted metrics provide additional optimization lever for training

### Phase 5: Real-World Testing (Future)

Apply to existing challenges:
- API Pattern Analyzer (batch 008)
- Event Correlation (batch 008)
- Compare learned programs vs hand-tuned

### Better Baselines (Future)

Compare against:
- TF-IDF weighting
- Information gain weighting
- Neural embeddings (BERT, etc.)

---

### 6. Discriminative Field Discovery Works

In the "hard categories" test where only ONE field (`type`) distinguished classes:

| Field | Expected | Learned Weight |
|-------|----------|----------------|
| type | HIGH (discriminative) | 1.50 ✓ |
| shared_keyword | LOW (non-discriminative) | 1.00 ✓ |
| noise_1, noise_2 | LOW | 1.00 ✓ |
| status, priority | LOW | 1.00 ✓ |

The optimizer correctly discovered that `type` was the only field that mattered.

---

## Phase 2: Interaction Discovery

### 7. Field Interactions Can Be Automatically Discovered

When categories depend on field COMBINATIONS (not individual values), the algorithm finds the right bindings:

| Experiment | Baseline | With Interaction | Improvement |
|------------|----------|------------------|-------------|
| Interaction data | 89.2% | 100.0% | +10.8% |
| XOR data | 54.2% | 100.0% | +45.8% |

The XOR result is especially significant: individual fields provide ZERO information (each is 50/50 in each class), only `bind(feature_a, feature_b)` can discriminate.

### 8. Key Insight: Zero Individual Weights When Testing Interactions

The critical fix for interaction discovery:

```python
# Wrong: Test interaction WITH individual fields
vec = bundle([encode(A), encode(B), bind(A, B)])  # Diluted!

# Right: Zero individual weights when interaction matters
self.weights[A] = 0.0
self.weights[B] = 0.0
vec = bundle([bind(A, B)])  # Clear signal!
```

When individual fields provide no information (like XOR), including them actually **hurts** by adding noise that drowns out the interaction signal.

### 9. Interaction Discovery Algorithm

```
1. Start with learned field weights (Phase 1)
2. For each candidate field pair:
   a. Zero out individual weights for (field_a, field_b)
   b. Add bind(field_a, field_b) with weight 2.0
   c. Evaluate accuracy
3. Keep interactions that improve accuracy > threshold
4. Repeat until no improvement found
```

### 10. Real-World Implications

This has practical implications for feature engineering:
- **Don't just use individual features** - interactions matter
- **Interactions can dominate** - sometimes individual features add noise
- **Automatic discovery works** - no need to manually specify interactions

---

## Code Summary

| File | Purpose |
|------|---------|
| `001-weighted-encoder.py` | WeightedEncoder class with fit() method |
| `002-noisy-training.py` | Noise robustness experiments |
| `003-hard-categories.py` | Overlapping category discrimination |
| `004-interaction-discovery.py` | InteractionEncoder with automatic binding discovery |
| `005-xor-debug.py` | Debug script analyzing why interactions work |
| `006-holon-primitives-inventory.py` | Catalog of ALL Holon primitives |
| `007-program-synthesis.py` | Genetic algorithm for program evolution |
| `008-distance-metrics-demo.py` | Test new distance metrics on VSA vectors |
| `009-improved-with-metrics.py` | Compare encoders with different metrics |
| `010-weighted-metric-synthesis.py` | Learn field + dimension weights together |
| `common.py` | Data generation, evaluation utilities |

### New Holon Module

| File | Purpose |
|------|---------|
| `holon/distance.py` | Comprehensive distance metrics for VSA vectors |

### Core Classes

```python
class WeightedEncoder:
    def encode(self, item: dict) -> np.ndarray:
        """Encode with field weights applied."""

    def fit(self, X_train, y_train, ...) -> TrainingResult:
        """Learn optimal weights via symbolic search."""

    def predict(self, X, prototypes) -> List[str]:
        """Classify using learned prototypes."""
```

### Training Methods

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| `random_search` | Sample random weight configs | Fast | Good |
| `hill_climb` | Perturb best weights | Fast | Good |
| `coordinate_descent` | Optimize one field at a time | Slow | Best |

---

## Phase 5: Large-Scale Stress Test ✓ COMPLETE

### The Test

We pushed the system to real-world scale:

| Parameter | Value |
|-----------|-------|
| Total samples | 500,000 |
| Categories | 100 |
| Unique values per field | 1,000 |
| Fields per record | 10 |
| Vector dimensions | 8,192 |
| Training samples | 400,000 |
| Test samples | 100,000 |

### Results

| Metric | Single-Thread | 10 Workers | Speedup |
|--------|---------------|------------|---------|
| Encode time | 103s | 35.5s | **2.9x** |
| Encode rate | 3,885/sec | 11,263/sec | **2.9x** |
| Total time | 142s | 58.7s | **2.4x** |
| Accuracy | 94.5% | 94.5% | ✓ |
| Peak memory | 7.5 GB | 7.5 GB | - |

### Key Technical Insights

**1. Parallel encoding requires shared codebook**

Each worker process creates its own random vectors for symbols. Without coordination, worker A's vector for "billing" differs from worker B's. The fix:

```python
# Main process: pre-populate codebook
for symbol in all_symbols:
    encoder.vector_manager.get_vector(symbol)

# Workers: copy the pre-populated vectors
def _init_worker(dimensions, atom_vectors, position_vectors):
    store = CPUStore(dimensions=dimensions)
    store.encoder.vector_manager.atom_vectors = dict(atom_vectors)
```

**2. Store.similarity() now abstracts distance metrics**

Users no longer need to import DistanceEngine directly:

```python
# Before (awkward)
from holon import DistanceEngine, DistanceMetric
engine = DistanceEngine()
sim = engine.similarity(vec1, vec2, DistanceMetric.HAMMING)

# After (clean)
sim = store.similarity(vec1, vec2, metric="hamming")
```

**3. Batch matrix multiply for classification**

Instead of N × K similarity computations, use matrix multiplication:

```python
# Slow: O(N × K) individual similarity calls
for item in X_test:
    for label, proto in prototypes.items():
        sim = store.similarity(encode(item), proto)

# Fast: O(1) matrix operation
similarities = np.dot(test_vectors, proto_matrix.T)
predictions = np.argmax(similarities, axis=1)
```

### What This Proves

1. **Holon scales to 500k+ records** - encoding at 11k/sec with 10 cores
2. **Simple prototype averaging works** - 94.5% accuracy with no fancy learning
3. **Memory is manageable** - 7.5 GB for 500k × 8192-dim vectors
4. **Parallelization is possible** - but requires careful codebook sharing
5. **The "deterministic training" vision works at scale**

---

## Conclusions

1. **Deterministic training for VSA is feasible** - We can learn encoding parameters without gradient descent

2. **The approach is robust to noise** - 30% label noise still yields 100% test accuracy

3. **Simple search methods work** - Random search and hill climbing find good weights quickly

4. **Cross-validation is essential** - Prevents overfitting to noise fields

5. **It scales** - 500k samples, 100 categories, 1000 cardinality → 94.5% accuracy in 59 seconds

6. **Parallelization works** - 2.9x speedup with 10 workers (shared codebook required)

The radical vision of "symbolic program synthesis for vector encoding" is validated from weight learning through large-scale deployment.

---

## Phase 6: Qdrant Scale Test (1000 Categories) ✓ COMPLETE

### The Test

Pushed to 1000 categories with actual Qdrant persistence:

| Parameter | Value |
|-----------|-------|
| Categories | 1000 |
| Samples per category | 100 |
| Total samples | 100,000 |
| Train/test split | 80,000 / 20,000 |
| Dimensions | 4,096 |
| Workers | 10 |

### Results

| Metric | 100 Categories | 1000 Categories |
|--------|----------------|-----------------|
| **Accuracy** | 93.8% | **81.6%** |
| Encode rate | 5,941/sec | 19,731/sec |
| Qdrant insert rate | 58/sec | 54/sec |
| Query latency (avg) | 26.9ms | 35.5ms |
| Peak memory | 291 MB | 3,024 MB |

### Key Insights

**1. Accuracy degrades gracefully with more categories**

Going from 100 to 1000 categories (10x), accuracy dropped from 93.8% to 81.6% (12.2 percentage points). This is actually quite good - with 1000 categories, random guessing would be 0.1%. We're at 816x random baseline.

**2. Qdrant insert is the bottleneck**

- Encoding: 19,731/sec (fast)
- Qdrant insert: 54/sec (slow)
- Total insert time: ~25 minutes for 80k vectors

The bottleneck is HTTP/gRPC network overhead for 4096-dim vectors. At 54/sec, inserting 1M vectors would take ~5 hours.

**3. Query latency is acceptable**

35.5ms average for ANN search across 80k vectors is reasonable for most use cases.

**4. Memory scales linearly**

- 5k samples (100 cat): 291 MB
- 100k samples (1000 cat): 3,024 MB
- ~30 KB per sample at 4096D

### Accuracy Analysis

Why did accuracy drop to 81.6%?

1. **More categories = more confusion**: With 1000 prototypes, similar categories can overlap
2. **Fewer samples per prototype**: Only 80 training samples per category (80k / 1000)
3. **Signal dilution**: Each field value appears in more categories

To improve accuracy at 1000 categories:
- Increase samples per category (more training data)
- Use higher dimensions (8192 instead of 4096)
- Apply Phase 1 weighted encoding to emphasize discriminative fields
- Use Phase 2 interaction discovery for field combinations

### What This Proves

1. **Qdrant integration works at scale** - 80k vectors inserted and queried
2. **1000 categories is viable** - 81.6% accuracy (816x random)
3. **Query latency is production-ready** - 35.5ms average
4. **Memory is manageable** - ~3 GB for 100k samples

---

## Brutal Honesty

### What We Proved
- Encoding is fast (11k-20k/sec parallel)
- Prototype classification works (94.5% at 100 cat, 81.6% at 1000 cat)
- Parallelization is possible
- Memory scales reasonably (3 GB for 100k × 4096)
- Qdrant integration works (80k vectors, 35ms queries)
- **1000 categories is viable** (81.6% accuracy)

### What We Didn't Prove
- **Real-world data**: All tests use synthetic data with planted signal
- **vs. Baselines**: Never compared to TF-IDF, neural embeddings, or traditional ML
- **Insert speed at scale**: 54/sec is slow for millions of vectors
- **Edge cases**: Unknown behavior on adversarial/pathological data

### Remaining Open Questions
- Does 81.6% hold on messier real data?
- Can we improve 1000-category accuracy with more training data?
- What's the accuracy/dimension tradeoff at 1000 categories?
- Can we speed up Qdrant inserts? (gRPC batching, grpc streaming?)

---

*Completed: February 2026*
