# Holon Encoding Guide: When to Use Which Configuration

## Overview

Holon provides flexible N-gram encoding configurations through the `_encode_config` parameter. While basic bigrams (`n_sizes: [2]`) perform best for general substring matching, different configurations excel in specific scenarios.

## Quick Reference

| Scenario | Recommended Config | Why | Performance |
|----------|-------------------|-----|-------------|
| **General Substring Matching** | `{"n_sizes": [2]}` | Optimal F1 (75%) | ⭐⭐⭐⭐⭐ |
| **Single Word Queries** | `{"n_sizes": [1]}` | Individual term matching | ⭐⭐⭐ |
| **Phrase/Sentence Queries** | `{"n_sizes": [2, 3]}` | Context preservation | ⭐⭐⭐⭐ |
| **Mixed Query Types** | `{"n_sizes": [1, 2, 3]}` | Universal coverage | ⭐⭐⭐ |
| **Speed-Critical** | `{"n_sizes": [2]}` | Fastest encoding/decoding | ⭐⭐⭐⭐⭐ |
| **Memory-Constrained** | `{"n_sizes": [1]}` | Minimal vector size | ⭐⭐⭐⭐⭐ |

## Detailed Use Cases

### 1. Content Search & Analysis

#### Document Substring Location (Your PDF Use Case)
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [2]
  }
}
```
**Best for**: Finding phrases within larger documents
**Performance**: 75% F1, fast encoding
**Example**: PDF paragraph search, code function location

#### Full-Text Book Analysis (Enhanced Kernel)
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [1, 2, 3],        // Multi-resolution (VSA standard)
    "weights": [0.2, 0.6, 0.4],  // Progressive trigram weighting
    "length_penalty": true,       // Normalize query length differences
    "term_weighting": true,       // Weight important terms higher
    "positional_weighting": true, // Earlier patterns more important
    "discrimination_boost": true  // Enhance unique components
  }
}
```
**Best for**: Complex text analysis requiring multiple n-gram sizes
**Performance**: 63.9% F1 (improvement over basic n-grams)
**Example**: Academic research, detailed text analysis

### 2. Query Characteristics

#### Short/Single Word Queries
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [1],
    "term_weighting": true
  }
}
```
**Best for**: Keyword search, entity recognition
**Performance**: Good for individual terms
**Example**: "find all mentions of 'calculus'"

#### Long Phrase/Sentence Queries
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [2, 3],
    "positional_weighting": true
  }
}
```
**Best for**: Exact phrase matching, context preservation
**Performance**: Excellent for multi-word phrases
**Example**: "to be or not to be" quotations

#### Conceptual/Semantic Queries
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [1, 2],
    "weights": [0.4, 0.6],
    "discrimination_boost": true
  }
}
```
**Best for**: Fuzzy/conceptual matching
**Performance**: Better semantic similarity
**Example**: Finding related concepts, not exact phrases

### 3. Domain-Specific Applications

#### DNA/Genetic Sequence Analysis
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [3, 6, 9],        // Codons, genes, segments
    "weights": [0.5, 0.3, 0.2]  // Biological hierarchy
  }
}
```
**Best for**: Genetic pattern recognition
**Performance**: Domain-specific optimization
**Example**: Finding gene sequences, mutation patterns

#### Code Analysis
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [1, 2],           // Tokens + pairs
    "weights": [0.3, 0.7],       // Favor relationships
    "positional_weighting": true // Code structure matters
  }
}
```
**Best for**: Function/variable relationships
**Performance**: Syntax-aware matching
**Example**: Finding similar code patterns

#### Time Series / Sensor Data
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [3, 5, 7],        // Local + medium + long patterns
    "weights": [0.4, 0.4, 0.2]  // Favor local patterns
  }
}
```
**Best for**: Pattern recognition in sequential data
**Performance**: Temporal relationship capture
**Example**: Anomaly detection, trend analysis

### 4. Performance Optimization

#### Maximum Speed (Latency Critical)
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [2]  // Minimal computation
  }
}
```
**Best for**: Real-time applications, high-throughput
**Performance**: Fastest encoding/decoding
**Trade-off**: Less flexible matching

#### Maximum Accuracy (Quality Critical)
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [1, 2, 3],
    "weights": [0.2, 0.6, 0.4],
    "length_penalty": true,
    "term_weighting": true,
    "positional_weighting": true,
    "discrimination_boost": true
  }
}
```
**Best for**: Research, precision applications
**Performance**: Highest quality matching
**Trade-off**: Slower, more complex

#### Memory Efficiency
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [1]  // Smallest vectors
  }
}
```
**Best for**: Large datasets, embedded systems
**Performance**: Minimal memory footprint
**Trade-off**: Less sophisticated matching

### 5. Data Characteristics

#### Clean, Structured Data
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [2, 3]  // Precise patterns
  }
}
```
**Best for**: Well-formed content, exact matching
**Performance**: High precision on clean data

#### Noisy, Variable Data
```json
{
  "_encode_mode": "ngram",
  "_encode_config": {
    "n_sizes": [1, 2],
    "weights": [0.6, 0.4],  // Favor flexibility
    "discrimination_boost": true  // Robust to noise
  }
}
```
**Best for**: User-generated content, variable formats
**Performance**: Better handling of inconsistencies

## Configuration Parameters Reference

### Core Parameters
- **`n_sizes`**: Array of n-gram sizes to generate
- **`weights`**: Relative importance of each n-gram size

### Enhancement Parameters
- **`length_penalty`**: Normalize for query length differences
- **`term_weighting`**: Weight terms by importance/density
- **`positional_weighting`**: Favor earlier n-grams in sequences
- **`discrimination_boost`**: Enhance distinctive vector components
- **`idf_weighting`**: Use corpus statistics for weighting (future)

## Choosing the Right Configuration

### Decision Flowchart

```
Start: What type of data?
├── Text/Documents → Go to Text Branch
├── Sequences (DNA/Time) → Go to Sequence Branch
└── Other → Custom configuration

Text Branch: What queries?
├── Single words → n_sizes: [1]
├── Phrases → n_sizes: [2]
├── Mixed → n_sizes: [1, 2]
└── Complex analysis → n_sizes: [1, 2, 3]

Sequence Branch: What patterns?
├── Local patterns → n_sizes: [3]
├── Medium patterns → n_sizes: [3, 5]
└── Hierarchical → n_sizes: [3, 6, 9]

Performance Branch: What matters most?
├── Speed → n_sizes: [2]
├── Accuracy → n_sizes: [1, 2, 3] + enhancements
└── Memory → n_sizes: [1]
```

### Testing Recommendations

1. **Start Simple**: Begin with `{"n_sizes": [2]}` - proven performance
2. **Add Complexity Gradually**: Test impact of each enhancement
3. **Measure Performance**: Track F1, precision, recall, speed
4. **Domain-Specific Tuning**: Adjust weights based on your data patterns

## Summary

**Customization empowers users** to optimize holon for their specific needs:

- **Performance**: Choose speed vs accuracy trade-offs
- **Domain**: Adapt to text, DNA, code, time-series, etc.
- **Query Types**: Optimize for words, phrases, or concepts
- **Data Characteristics**: Handle clean vs noisy data
- **Resource Constraints**: Balance memory vs quality

Basic bigrams (`n_sizes: [2]`) work best for general substring matching, but the full configuration space enables optimization for specific use cases.

## Continuous Value Encoding

For continuous scalar values (not text), Holon provides specialized encoding methods.

### Inline Markers (Recommended)

Use `$log` and `$linear` markers directly in your data structures for the most natural API:

```python
# Log encoding: equal ratios → equal similarity
data = {
    "event": "traffic",
    "src_ip": "10.0.0.1",
    "rate_pps": {"$log": 1000},      # Magnitude-aware
    "bytes": {"$log": 1500000}       # Magnitude-aware
}
client.insert_json(data)

# Linear encoding: equal differences → equal similarity
data = {
    "sensor": "temp-1",
    "temperature": {"$linear": 72.5},  # Distance-aware
    "latency_ms": {"$linear": 15}      # Distance-aware
}
client.insert_json(data)

# Custom decay rate with $scale
data = {"rate": {"$log": 1000, "$scale": 500}}  # Faster similarity decay
```

**Why use inline markers?**
- Numbers without markers encode as strings (no magnitude relationship)
- With markers, similar magnitudes produce similar vectors
- Works naturally in nested structures

### Comparison: Default vs Markers

```python
# WITHOUT markers (default string encoding)
{"rate": 100}   # Random vector for string "100"
{"rate": 200}   # Unrelated random vector for string "200"
# similarity ≈ 0.0 (orthogonal)

# WITH $log marker
{"rate": {"$log": 100}}
{"rate": {"$log": 200}}
# similarity ≈ 0.98 (2x ratio = very similar)

# WITH $linear marker
{"latency": {"$linear": 100}}
{"latency": {"$linear": 110}}
# similarity depends on scale (absolute +10 difference)
```

### Method-Based Encoding (Alternative)

For direct vector manipulation, use the method-based API:

#### Log-Scale Encoding

For multiplicative quantities where ratios matter (rates, sizes, counts):

```python
# Equal ratios produce equal similarity
rate_100 = store.encode_scalar_log(100)
rate_1000 = store.encode_scalar_log(1000)
rate_10000 = store.encode_scalar_log(10000)

store.similarity(rate_100, rate_1000)   # ~0.94 (10x ratio)
store.similarity(rate_1000, rate_10000) # ~0.92 (10x ratio)
store.similarity(rate_100, rate_10000)  # ~0.86 (100x ratio)
```

**Use cases**: Network traffic rates, file sizes, request counts, prices

### Linear Encoding

For additive quantities where absolute differences matter:

```python
temp_72 = store.encode_scalar(72.0, mode="linear")
temp_75 = store.encode_scalar(75.0, mode="linear")
temp_100 = store.encode_scalar(100.0, mode="linear")

# 72°F similar to 75°F, less similar to 100°F
```

**Use cases**: Temperatures, positions, coordinates

### Circular Encoding

For cyclic/periodic values that wrap around:

```python
# Hours on a 24-hour clock
hour_23 = store.encode_scalar(23.0, mode="circular", period=24.0)
hour_1 = store.encode_scalar(1.0, mode="circular", period=24.0)

# 23:00 and 01:00 are only 2 hours apart!
store.similarity(hour_23, hour_1)  # High similarity

# Compass bearings (0° = 360°)
north = store.encode_scalar(5.0, mode="circular", period=360.0)
almost_north = store.encode_scalar(355.0, mode="circular", period=360.0)
store.similarity(north, almost_north)  # High similarity
```

**Use cases**: Time of day, day of week, compass bearings, angles, phase

### Choosing the Right Mode

| Data Type | Inline Marker | Method API | Period | Example |
|-----------|---------------|------------|--------|---------|
| Network rates | `{"$log": 1000}` | `encode_scalar_log(1000)` | - | 100 pps vs 100,000 pps |
| File sizes | `{"$log": 1048576}` | `encode_scalar_log(size)` | - | 1 KB vs 1 GB |
| Prices | `{"$log": 99.99}` | `encode_scalar_log(price)` | - | $10 vs $10,000 |
| Latency | `{"$linear": 50}` | `encode_scalar(50, "linear")` | - | 10ms vs 100ms |
| Temperatures | `{"$linear": 72.5}` | `encode_scalar(72.5, "linear")` | - | 72°F vs 100°F |
| X/Y coordinates | `{"$linear": 10}` | `encode_scalar(10, "linear")` | - | Position (10, 20) |
| Hour of day | - | `encode_scalar(23.5, "circular", period=24)` | 24.0 | 23:30 near 00:30 |
| Day of week | - | `encode_scalar(6, "circular", period=7)` | 7.0 | Sunday near Monday |
| Compass bearing | - | `encode_scalar(5, "circular", period=360)` | 360.0 | 5° near 355° |
| Month of year | - | `encode_scalar(11, "circular", period=12)` | 12.0 | December near January |
| User IDs | (none - default) | - | - | Exact match only |
| Port numbers | (none - default) | - | - | Exact match only |
| Status codes | (none - default) | - | - | Exact match only |

**Note**: Circular encoding is currently only available via the method API.
Use `$log` and `$linear` for inline encoding in data structures.

### The $scale Parameter

The `$scale` parameter controls how quickly similarity decays with distance:

```python
# Default scale (1000) - moderate decay
{"rate": {"$log": 100}}
{"rate": {"$log": 1000}}   # similarity ~0.94

# Smaller scale (100) - faster decay
{"rate": {"$log": 100, "$scale": 100}}
{"rate": {"$log": 1000, "$scale": 100}}  # similarity ~0.91

# Larger scale (5000) - slower decay
{"rate": {"$log": 100, "$scale": 5000}}
{"rate": {"$log": 1000, "$scale": 5000}}  # similarity ~0.95
```

**Rule of thumb**:
- Smaller scale → values need to be closer to match
- Larger scale → more tolerant of distance
