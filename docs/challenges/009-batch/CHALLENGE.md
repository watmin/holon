# Challenge Batch 009: Deterministic Training via Symbolic Program Synthesis

## Overview

This challenge explores whether we can **learn encoding programs** for VSA/HDC instead of hand-tuning them. The core insight: with structured data, the search space is constrained by schema, making symbolic synthesis tractable.

**The Radical Claim**: We can achieve "training" without gradient descent by synthesizing compositions of VSA primitives.

---

## The Problem

Currently, Holon requires manual decisions:
- Which fields matter most?
- What weights to use in bundle()?
- Which field interactions to bind()?
- What to negate() to improve discrimination?

**Goal**: Learn these decisions from labeled examples.

---

## Approach: Three Phases

### Phase 1: Weighted Field Composition (This Challenge)

Learn optimal weights for each field in a structured record.

```python
# Instead of:
vec = encode_data({"priority": "high", "type": "billing", "keywords": ["refund"]})

# Learn:
vec = weighted_encode(item, weights={
    "priority": 0.5,   # Learned: less important
    "type": 2.0,       # Learned: very important
    "keywords": 1.5,   # Learned: moderately important
})
```

**Training**: Grid search or gradient-free optimization to find weights that maximize classification accuracy.

### Phase 2: Interaction Discovery (Future)

Learn which field pairs should be bound together:

```python
# Discover that (priority, type) interaction matters
interaction = bind(encode("priority"), encode("type"))
base = bundle([base, interaction])
```

### Phase 3: Full Program Synthesis (Future)

Learn complete encoding programs using genetic programming or similar.

---

## Phase 1 Specification

### Input

1. **Training data**: List of labeled structured records
   ```python
   training_data = [
       ({"priority": "high", "type": "billing", "text": "refund needed"}, "billing_team"),
       ({"priority": "low", "type": "technical", "text": "login failed"}, "tech_team"),
       ...
   ]
   ```

2. **Schema hint** (optional): Field names and types
   ```python
   schema = {
       "priority": "categorical",
       "type": "categorical",
       "text": "text",
       "created_at": "timestamp"
   }
   ```

### Output

1. **Learned weights**: Dictionary mapping field names to importance weights
   ```python
   weights = {"priority": 0.5, "type": 2.1, "text": 1.3, "created_at": 0.2}
   ```

2. **Encoding function**: Callable that applies learned weights
   ```python
   def learned_encode(item) -> np.ndarray:
       ...
   ```

### Training Algorithm

```
1. Initialize weights = {field: 1.0 for field in schema}
2. For each iteration:
   a. Encode all training items with current weights
   b. Build prototypes for each label
   c. Classify training set using k-NN or prototype matching
   d. Compute accuracy
   e. Perturb weights (grid search / random search / Bayesian optimization)
   f. Keep weights if accuracy improves
3. Return best weights
```

### Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Accuracy improvement | >5% over uniform weights | On held-out test set |
| Training time | <60 seconds | For 1000 training examples |
| Interpretability | Weights are meaningful | High-weight fields should be semantically relevant |
| Generalization | Accuracy on new data | Should not overfit to training set |

---

## Test Scenarios

### Scenario 1: Ticket Routing (Clean Data)

- 1000 synthetic tickets
- 4 teams: billing, technical, shipping, account
- Clean labels (no noise)
- Expected: Discover that "type" and "keywords" matter most

### Scenario 2: Ticket Routing (Noisy Data)

- Same as above, but 10% mislabeled
- Expected: Robust learning despite noise

### Scenario 3: API Pattern Detection

- Reuse data from 008-batch/002-api-pattern-analyzer
- Compare learned weights to hand-tuned approach
- Expected: Match or exceed 95.9% precision

### Scenario 4: High Cardinality

- 100+ unique values per field
- 10+ categories
- Expected: Weights still discoverable

---

## Implementation Plan

### File Structure

```
scripts/challenges/009-batch/
├── 001-weighted-encoder.py      # Core WeightedEncoder class
├── 002-ticket-routing-clean.py  # Scenario 1
├── 003-ticket-routing-noisy.py  # Scenario 2
├── 004-api-patterns.py          # Scenario 3
├── 005-high-cardinality.py      # Scenario 4
└── common.py                    # Shared utilities
```

### Core Classes

```python
class WeightedEncoder:
    """Encoder that applies learned field weights."""

    def __init__(self, base_encoder: Encoder, weights: Dict[str, float] = None):
        self.encoder = base_encoder
        self.weights = weights or {}

    def encode(self, item: dict) -> np.ndarray:
        """Encode item with field weights applied."""
        ...

    def fit(self, X: List[dict], y: List[str],
            method: str = "grid", max_iter: int = 100) -> Dict[str, float]:
        """Learn optimal weights from labeled data."""
        ...


class ProgramSynthesizer:
    """Future: Learn complete encoding programs."""
    pass
```

---

## Key Questions to Answer

1. **Does weight learning improve accuracy?**
   - Compare uniform weights vs learned weights
   - Measure on held-out test set

2. **Are learned weights interpretable?**
   - Do high-weight fields match human intuition?
   - Can we explain why certain fields matter?

3. **How much training data is needed?**
   - Test with 100, 500, 1000, 5000 examples
   - Find minimum viable training set size

4. **How robust to noise?**
   - Test with 5%, 10%, 20% label noise
   - Measure accuracy degradation

5. **Does it generalize?**
   - Train on one distribution, test on shifted distribution
   - Simulate real-world drift

---

## Comparison Baselines

| Approach | Description |
|----------|-------------|
| Uniform weights | All fields weighted equally (current default) |
| Hand-tuned | Manually set weights based on domain knowledge |
| TF-IDF inspired | Weight by inverse field frequency |
| Neural | Train a small MLP on encoded vectors |

---

## What Makes This Different

Traditional ML:
```
data → neural network → gradient descent → weights → predictions
```

Our approach:
```
data → VSA primitives → symbolic search → composition program → predictions
```

Key differences:
- **Deterministic**: No random initialization, no stochastic gradients
- **Interpretable**: Weights are per-field, not hidden layer activations
- **Composable**: Can stack phases (weights → interactions → full program)
- **Few-shot friendly**: Should work with small training sets

---

## References

- [ADR 001: VSA/HDC Architecture](../../architecture/decisions/001-vsa-hdc-architecture.md)
- [Batch 002 LEARNINGS: Primitive composition](../002-batch/LEARNINGS.md)
- [Batch 008 CHALLENGES: Real-world applications](../008-batch/CHALLENGES.md)

---

*Created: February 2026*
