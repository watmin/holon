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

## Conclusions

1. **Deterministic training for VSA is feasible** - We can learn encoding parameters without gradient descent

2. **The approach is robust to noise** - 30% label noise still yields 100% test accuracy

3. **Simple search methods work** - Random search and hill climbing find good weights quickly

4. **Cross-validation is essential** - Prevents overfitting to noise fields

5. **More work needed** - Phase 2 (interactions) and Phase 3 (full synthesis) are future work

The radical vision of "symbolic program synthesis for vector encoding" is validated at the simplest level (weight learning). Whether it scales to full program synthesis remains to be seen.

---

*Completed: February 2026*
