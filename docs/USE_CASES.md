# Holon Use Cases: When Structure Similarity Matters

## What Holon Does

Holon turns structured data (JSON, logs, configs) into high-dimensional vectors where **similar structures → similar vectors**.

| Capability | Performance |
|------------|-------------|
| Encoding speed | 25-30k/sec (CPU, 12 cores) |
| Classification accuracy | 84-94% (1000 categories) |
| Scale tested | 5 million records in 7.5 minutes |
| Memory | ~4GB per 1M records at 4096D |

No ML models, no training loop, no GPU required.

---

## Applications

### 1. Log/Event Anomaly Detection

Encode log entries → build "normal" prototypes → flag distant entries.

```python
# No labeled anomalies needed - just structure similarity
normal_proto = average([encode(log) for log in normal_logs])

for log in stream:
    if similarity(encode(log), normal_proto) < threshold:
        alert("Anomaly detected")
```

**Why Holon wins**: No training required, works on any JSON structure, handles schema drift gracefully.

### 2. Configuration Drift Detection

Encode configs → `difference()` against baseline → get delta as vector.

```python
baseline = store.encoder.encode_data(production_config)
current = store.encoder.encode_data(new_config)
drift = store.difference(current, baseline)

# The drift vector can be compared against known "bad drifts"
# or flagged if it exceeds a magnitude threshold
```

**Why Holon wins**: The `difference()` operation is mathematically meaningful - you get the actual structural delta as a vector.

### 3. Fraud/Attack Pattern Matching

Encode transactions → compare to known fraud prototypes.

| Pattern | Structure |
|---------|-----------|
| Account takeover | `{login: X, change_email: Y, transfer: Z}` |
| Card testing | `{small_purchase: [1,2,3,4,5...]}` |
| Credential stuffing | `{failed_login: [...], success: 1}` |

**Why Holon wins**: Matches *structure* not values. A new fraud variant with the same pattern shape still matches the prototype.

### 4. Schema Matching for Data Integration

Encode table schemas → find similar tables across databases.

```python
# Different names, same structure → high similarity
schema_a = {"user_id": "int", "email": "str", "created_at": "timestamp"}
schema_b = {"customer_id": "int", "mail": "str", "signup_date": "timestamp"}

vec_a = encode(schema_a)
vec_b = encode(schema_b)
# → High structural similarity despite different field names
```

**Why Holon wins**: No NLP on column names needed, pure structural comparison.

### 5. API Request Fingerprinting

Encode API requests → detect unusual structures.

```python
# Normal requests cluster together
normal_requests = [encode(req) for req in training_sample]
normal_proto = average(normal_requests)

# Unusual structure = potential attack
def is_suspicious(request):
    return similarity(encode(request), normal_proto) < threshold
```

**Why Holon wins**: Catches structure-based attacks (extra fields, unusual nesting) without training on attack examples.

### 6. IoT Device Fingerprinting

Encode telemetry patterns → cluster by device type.

```python
# Devices of the same type produce similar telemetry structures
device_vectors = {device_id: encode(telemetry) for device_id, telemetry in data}

# Cluster without labels
clusters = cluster(device_vectors)
# → Devices naturally group by type based on structure
```

**Why Holon wins**: Works without labeled device data, just structural similarity.

### 7. Event Sequence Matching

Use chained mode to encode ordered events → find similar user journeys.

```python
journey_a = {"$mode": "chained", "sequence": ["login", "browse", "cart", "purchase"]}
journey_b = {"$mode": "chained", "sequence": ["login", "browse", "cart", "abandon"]}

# Similar prefix, different ending → moderate similarity
```

**Why Holon wins**: Order matters in the encoding, captures behavioral patterns.

---

## When to Use Holon

| Situation | Use Holon | Use Something Else |
|-----------|:---------:|:------------------:|
| Structured data (JSON, logs, configs) | ✓ | |
| No labeled training data available | ✓ | |
| Need to explain "why similar" | ✓ | |
| Need composable operations (diff, negate) | ✓ | |
| Semantic similarity ("means the same") | | Use embeddings |
| Free-form text search | | Use LLMs/embeddings |
| Image/audio similarity | | Use neural nets |
| Need 99%+ accuracy | | Train a classifier |

---

## Honest Assessment

### What We Proved

- 1000-way classification at 84% accuracy
- 5 million records processed in 7.5 minutes
- Linear scaling with no accuracy degradation
- Works on CPU with no ML infrastructure

### What We Didn't Prove

- That 84% beats simpler alternatives (TF-IDF, decision trees)
- Performance on real-world messy data
- Accuracy on adversarial/pathological inputs

### The Real Value Proposition

Holon trades peak accuracy for **simplicity and speed**:

| Approach | Training Time | Accuracy | Complexity |
|----------|---------------|----------|------------|
| Holon | 0 (just encode) | ~84% | Low |
| Gradient boosting | Minutes-hours | 90-95% | Medium |
| Neural networks | Hours-days | 95-99% | High |

**Best fit**: Prototype quickly, no labeled data, "good enough" accuracy is acceptable.

---

## Getting Started

See [basic_usage.py](../examples/basic_usage.py) for a quick start, or [Challenge 009 Learnings](challenges/009-batch/LEARNINGS.md) for the full scale testing details.
