# ADR 002: Dual JSON/EDN Data Format Support

## Status
Accepted

## Context
Users need to store structured data with different requirements:
- **JSON**: Universal compatibility, simple structures, numeric data
- **EDN**: Richer semantics, keywords, sets, complex nesting, symbolic data

The system must:
- Support both formats seamlessly
- Preserve semantic meaning in vector encoding
- Enable cross-format queries where meaningful
- Maintain performance parity

## Decision
Implement dual format support with:
- **JSON Parser**: Standard json.loads() with custom atomization
- **EDN Parser**: Custom recursive parser handling keywords, sets, symbols
- **Unified Encoding**: All data normalized to internal representation before vectorization
- **Format-Aware Queries**: Data type specified at query time

## Implementation Details

### EDN Features Supported
- Keywords: `:user`, `:role`
- Sets: `#{:clojure :python}`
- Symbols: `some-symbol`
- Complex nesting: `{:users [{:name "Alice"} {:name "Bob"}]}`

### Encoding Strategy
- **Scalars**: Direct atomization (strings, numbers, keywords, symbols)
- **Collections**: Recursive binding/bundling preserving structure
- **Sets**: Special encoding as unordered collections
- **Keywords**: Treated as distinct from strings for semantic precision

## Consequences

### Positive
- **Rich Semantics**: EDN enables more expressive data modeling
- **Backward Compatibility**: JSON users unaffected
- **Type Safety**: Format specification prevents confusion
- **Extensibility**: Easy to add new data formats

### Negative
- **Complexity**: Dual parsing/encoding paths to maintain
- **Query Complexity**: Must specify data_type for each operation
- **Learning Curve**: EDN syntax unfamiliar to JSON-only users

### Mitigations
- Comprehensive examples for both formats
- Clear error messages for format mismatches
- Optional data_type inference from content
- Shared internal representation minimizes duplication

## Usage Examples

```python
# JSON
client.insert_json({"name": "Alice", "role": "developer"})

# EDN
client.insert('{:name "Alice" :role :developer :skills #{:clojure :python}}', data_type="edn")

# Query (format must match)
results = client.search(probe='{:role :developer}', data_type="edn")
```

## References
- [EDN Specification](https://github.com/edn-format/edn)
- [JSON Specification](https://www.json.org/json-en.html)
- [EDN Usage Examples](../../examples/edn_usage.py)
