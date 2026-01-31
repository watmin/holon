# Challenge Batch 007: Complex Data Encoding & Scale Limits

*Ideas for demonstrating Holon's advanced capabilities and finding its breaking points.*

---

## Part A: Complex Challenges (Suited for Holon)

### 1. Multi-Modal Code Understanding

**Goal**: Unify multiple code metadata sources into searchable vectors.

**Data Sources**:
- AST structure (function signatures, class hierarchies)
- Docstrings and comments
- Git history (author, recency, churn)
- Test coverage metrics
- Dependency relationships

**Example Queries**:
```python
# Find functions similar to this one with high test coverage
client.search_json(
    probe={"ast": {"type": "FunctionDef", "args": ["self", "data"]}},
    guard={"coverage": {"$gte": 80}, "last_modified": {"$gte": "2024-01-01"}}
)

# Find all error handlers similar to this pattern
client.search_json(
    probe={"$or": [
        {"ast": {"type": "ExceptHandler", "exception": "ValueError"}},
        {"ast": {"type": "ExceptHandler", "exception": "TypeError"}}
    ]}
)
```

**What This Tests**:
- Binding multiple metadata sources into coherent vectors
- Prototype learning across code patterns
- Complex $or + guard composition

---

### 2. Hierarchical Document Retrieval

**Goal**: Navigate deeply nested legal/technical documents with cross-references.

**Data Structure**:
```json
{
  "document": "Contract-2024-001",
  "section": "5.2.1",
  "parent_sections": ["5", "5.2"],
  "clause_type": "indemnification",
  "references": ["Section 3.1", "Appendix A"],
  "text": "The party shall indemnify...",
  "amendments": [{"date": "2024-03", "change": "Added liability cap"}]
}
```

**Example Queries**:
```python
# Find clauses similar to indemnification but NOT in Section 5
client.search_json(
    probe={"clause_type": "indemnification"},
    guard={"section": {"$not": {"$contains": "5."}}},
    negations={"parent_sections": {"$not": "5"}}
)

# Find all clauses that reference Appendix A
client.search_json(
    probe={},
    guard={"references": {"$contains": "Appendix A"}}
)
```

**What This Tests**:
- Deep nesting representation
- Cross-document reference handling
- Negation with hierarchical data

---

### 3. Event Sequence Matching (Anomaly Detection)

**Goal**: Find similar event patterns in logs/transactions with temporal awareness.

**Data Structure**:
```json
{
  "session_id": "sess_12345",
  "events": {
    "_encode_mode": "chained",
    "sequence": ["login", "view_account", "transfer", "transfer", "logout"]
  },
  "user_id": "user_001",
  "total_amount": 50000,
  "duration_seconds": 120,
  "ip_changes": 3
}
```

**Example Queries**:
```python
# Find sessions similar to known fraud pattern
fraud_pattern = {
    "events": {"_encode_mode": "chained", "sequence": ["login", "transfer", "transfer"]},
    "duration_seconds": {"$lt": 60}
}
client.search_json(probe=fraud_pattern, threshold=0.3)

# Learn fraud prototype from examples
fraud_vectors = [client.encode_vectors(case) for case in known_fraud_cases]
fraud_prototype = client.prototype(fraud_vectors)

# Find similar sessions
client.search_json(probe={"_vector": fraud_prototype}, threshold=0.4)
```

**What This Tests**:
- Chained encoding for temporal sequences
- Prototype learning from anomalies
- Combining sequence similarity with numeric guards

---

### 4. Knowledge Graph Fragment Matching

**Goal**: Match partial subgraph patterns against a knowledge base.

**Data Structure**:
```json
{
  "entity": "Python",
  "type": "ProgrammingLanguage",
  "relations": {
    "created_by": "Guido van Rossum",
    "influenced_by": ["ABC", "C", "Lisp"],
    "used_for": ["web", "ml", "scripting"]
  },
  "neighbors": {
    "similar_to": ["Ruby", "JavaScript"],
    "competes_with": ["Java", "Go"]
  }
}
```

**Example Queries**:
```python
# Find entities with similar relational structure
client.search_json(
    probe={
        "type": "ProgrammingLanguage",
        "relations": {"used_for": {"$any": True}},  # Has used_for relation
        "neighbors": {"similar_to": ["Ruby"]}
    }
)

# Find entities influenced by Lisp
client.search_json(
    probe={"relations": {"influenced_by": "Lisp"}}
)
```

**What This Tests**:
- Graph topology encoding
- Partial pattern matching with $any
- Relation-based similarity

---

### 5. Multi-Format Semantic Alignment

**Goal**: Query across JSON, EDN, and other formats with unified similarity.

**Concept**: Same entity encoded in multiple formats should be retrievable by any format.

```python
# Insert same data in different formats
client.insert_json({"name": "Alice", "skills": ["python", "ml"]})
client.insert('{:name "Alice" :skills #{"python" "ml"}}', data_type="edn")

# Query should find both
results = client.search_json({"name": "Alice"})  # Finds JSON version
results = client.search('{:name "Alice"}', data_type="edn")  # Finds EDN version

# Cross-format query (advanced)
# Can we find the EDN record with a JSON probe?
```

**What This Tests**:
- Format-agnostic encoding
- Set vs list handling (EDN sets vs JSON arrays)
- Cross-format retrieval quality

---

## Part B: Scale & Noise Limit Experiments

### Experiment 1: Category Saturation

**Question**: At how many categories do prototypes start overlapping too much?

**Method**:
1. Create N categories with M examples each
2. Build prototype for each category
3. Test classification accuracy on held-out examples
4. Increase N until accuracy < 90%

**Expected**: Based on VSA theory, should handle 1000+ categories in 16k dimensions.

---

### Experiment 2: Similar Item Density

**Question**: With N nearly-identical items, can we still find a specific target?

**Method**:
1. Insert target item with unique marker
2. Insert N items that are 95% similar to target
3. Query for target
4. Measure: Does target appear in top-10? Top-1?
5. Increase N until target is lost

**Expected**: Should handle 1000+ similar items due to high dimensionality.

---

### Experiment 3: Binding Depth

**Question**: How deep can nesting go before signal is lost?

**Method**:
1. Create data with N levels of nesting
2. Query for deepest-level field
3. Measure retrieval success
4. Increase N until retrieval fails

**Expected**: Unknown - this is a key limit to discover.

---

### Experiment 4: Field Count Dilution

**Question**: With N fields per record, do important fields get drowned?

**Method**:
1. Create records with N fields, one "important" field
2. Query using only the important field
3. Measure precision
4. Increase N until precision drops below 80%

**Expected**: Should handle 50+ fields, but unknown where limit is.

---

### Experiment 5: Vocabulary Size

**Question**: How large can the effective vocabulary be?

**Method**:
1. Use N unique field names/values
2. Test if similar values still cluster
3. Increase N until similarity breaks down

**Expected**: VSA should handle very large vocabularies (100k+).

---

## Part C: Benchmark Suite Design

### Metrics to Track

| Metric | Description |
|--------|-------------|
| **Precision@K** | Target in top-K results |
| **Recall** | All relevant items found |
| **MRR** | Mean Reciprocal Rank |
| **Latency** | Query time in ms |
| **Memory** | RAM usage per 1k items |
| **Encoding time** | Insert throughput |

### Baseline Comparisons

- **Elasticsearch**: Keyword search + filters
- **PostgreSQL JSONB**: Exact queries + GIN indexes
- **Pinecone/Qdrant**: Embedding similarity
- **Raw numpy**: Brute-force cosine similarity

### Success Criteria

Holon should excel when:
- Fuzzy matching is needed (not just keywords)
- Structure matters (not just text embedding)
- Multiple query types combine (similarity + guards + negation)
- Data is heterogeneous (mixed schemas)

---

## Implementation Priority

1. **Experiment 2 (Similar Item Density)** - Quick to implement, unknown limit
2. **Experiment 3 (Binding Depth)** - Important for hierarchical data use case
3. **Challenge 1 (Code Understanding)** - Builds on existing code_structure_search.py
4. **Challenge 3 (Event Sequences)** - Novel use case, tests chained encoding

---

*Generated from Claude session, Jan 2026*
