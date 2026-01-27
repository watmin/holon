# ADR 003: Advanced Query System with Guards, Negations, and $or

## Status
Accepted

## Context
Basic similarity queries insufficient for production use. Users need:
- **Exact Filtering**: Find records with specific attribute values
- **Exclusion Logic**: Find records WITHOUT certain properties
- **Compound Conditions**: AND/OR combinations of constraints
- **Performance**: Efficient filtering before expensive similarity computation

Traditional vector similarity approaches lack these controls.

## Decision
Implement multi-stage query system:
1. **Similarity Search**: Vector-based fuzzy matching (primary operation)
2. **Guard Filtering**: Exact attribute matching (post-similarity filter)
3. **Negation Filtering**: Exclusion-based filtering
4. **$or Support**: Disjunctive queries with structured conditions

## Implementation Details

### Query Pipeline
```
Probe Vector → Similarity Search → Guard Filter → Negation Filter → Results
```

### Guard System
- **Simple Guards**: `{"role": "developer"}` - exact match required
- **Structured Guards**: `{"$or": [{"team": "backend"}, {"team": "frontend"}]}` - disjunctive conditions

### Negation System
- **Simple Negations**: `{"name": {"$not": "Alice"}}` - exclude specific values
- **Complex Negations**: Multiple exclusion criteria

### $or Implementation
- **Query Level**: `{"$or": [{"role": "developer"}, {"role": "designer"}]}`
- **Guard Level**: Guards can contain $or for complex filtering

## Performance Characteristics

### ANN Optimization
- Similarity search uses FAISS ANN indexing (>1000 items)
- Guards/negations applied to top-k results only
- Maintains sub-millisecond query times

### Memory Efficiency
- Guards prevent unnecessary vector operations
- Negations filter results without additional similarity computations

## Consequences

### Positive
- **Expressiveness**: Rich query language for complex conditions
- **Performance**: Guards reduce similarity search scope
- **Familiarity**: SQL-like semantics for developers
- **Composability**: Multiple filters combine predictably

### Negative
- **Complexity**: Multi-stage pipeline harder to debug
- **Order Dependency**: Filter order affects performance
- **Semantic Complexity**: Guards vs similarity can be confusing

### Mitigations
- Clear documentation with examples
- Consistent error messages
- Performance profiling tools
- Query validation and optimization hints

## Usage Examples

```python
# Basic similarity with guards
results = client.search_json(
    {"role": "developer"},  # similarity probe
    guard={"team": "backend"}  # exact filter
)

# Negations
results = client.search_json(
    {"role": "developer"},
    negations={"name": {"$not": "Alice"}}
)

# Complex $or conditions
results = client.search_json({}, guard={
    "$or": [
        {"priority": "high", "status": "todo"},
        {"project": "urgent", "category": "side"}
    ]
})
```

## References
- [Query Examples](../../examples/advanced_queries.py)
- [Guard System Tests](../../tests/test_guards.py)
- [Performance Benchmarks](../performance.md)
