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
