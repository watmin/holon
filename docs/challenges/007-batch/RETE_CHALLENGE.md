# Challenge: Holon-Powered Rete / Rule Engine

*Exploring how VSA/HDC can complement or enhance rule-based systems like Clara.*

---

## Background

### What is Rete?

Rete is a pattern-matching algorithm for implementing production rule systems. It efficiently matches facts against rules by:
1. Building a discrimination network from rule conditions
2. Sharing common condition tests across rules
3. Maintaining working memory of matched facts
4. Propagating changes incrementally

### What is Clara?

[Clara](https://github.com/cerner/clara-rules) is a Clojure implementation of forward-chaining rules with:
- **Facts**: Data in working memory (records/maps)
- **Rules**: Conditions → Actions
- **Sessions**: Stateful rule execution contexts
- **Queries**: Retrieve matched facts
- **Truth Maintenance**: Automatic retraction on fact changes

Example Clara rule:
```clojure
(defrule high-priority-alert
  [Order (= status :pending) (> total 10000)]
  [Customer (= id ?customer-id) (= tier :platinum)]
  =>
  (insert! (->Alert ?customer-id :high-priority)))
```

---

## The Challenge

### Goal

Build a Holon-enhanced rule engine that combines:
1. **Exact matching** (traditional Rete) for precise conditions
2. **Fuzzy matching** (Holon) for similarity-based conditions
3. **Prototype rules** that fire on "similar enough" facts

### Why This Could Work

| Rete Strength | Holon Strength | Combined |
|---------------|----------------|----------|
| Exact pattern match | Fuzzy similarity | Match "close enough" patterns |
| Discrete conditions | Continuous similarity | Gradual rule activation |
| Binary true/false | Similarity scores | Ranked rule matches |
| Explicit joins | Implicit similarity | Find related facts without explicit keys |

### Use Cases

1. **Fraud Detection**: "Flag transactions similar to known fraud patterns"
2. **Recommendation**: "Fire rules for products similar to user preferences"
3. **Anomaly Response**: "Apply rules when system state deviates from normal"
4. **Fuzzy Classification**: "Route tickets similar to past escalations"

---

## Design Proposal

### Fact Representation

```python
# Facts are JSON structures with type marker
fact1 = {
    "_type": "Order",
    "id": "order-123",
    "customer_id": "cust-456",
    "status": "pending",
    "total": 15000,
    "items": ["laptop", "monitor", "keyboard"]
}

fact2 = {
    "_type": "Customer",
    "id": "cust-456",
    "name": "Acme Corp",
    "tier": "platinum",
    "history": {"orders": 50, "returns": 2}
}
```

### Rule Representation

```python
# Traditional Rete-style rule
rule_exact = {
    "name": "high-value-platinum",
    "conditions": [
        {"_type": "Order", "status": "pending", "total": {"$gt": 10000}},
        {"_type": "Customer", "tier": "platinum", "id": {"$bind": "customer_id"}}
    ],
    "join": {"Order.customer_id": "Customer.id"},
    "action": "create_alert"
}

# Holon-enhanced fuzzy rule
rule_fuzzy = {
    "name": "similar-to-fraud",
    "conditions": [
        {
            "_type": "Transaction",
            "_similar_to": fraud_prototype,  # Holon prototype
            "_threshold": 0.7
        }
    ],
    "action": "flag_for_review"
}

# Hybrid rule: exact + fuzzy
rule_hybrid = {
    "name": "suspicious-platinum",
    "conditions": [
        {"_type": "Customer", "tier": "platinum"},  # Exact
        {
            "_type": "Transaction",
            "_similar_to": {"pattern": "rapid-transfers"},  # Fuzzy
            "_threshold": 0.5
        }
    ],
    "join": {"Transaction.customer_id": "Customer.id"},
    "action": "escalate"
}
```

### Session API

```python
from holon.rete import ReteSession

# Create session with rules
session = ReteSession(rules=[rule_exact, rule_fuzzy, rule_hybrid])

# Insert facts
session.insert(order_fact)
session.insert(customer_fact)

# Fire rules (returns activations)
activations = session.fire_rules()
# [{"rule": "high-value-platinum", "facts": [...], "score": 1.0},
#  {"rule": "similar-to-fraud", "facts": [...], "score": 0.82}]

# Query matched facts
results = session.query({"_type": "Alert"})

# Retract fact (with truth maintenance)
session.retract(order_fact)
```

---

## Implementation Phases

### Phase 1: Holon-Native Pattern Matching

Use Holon's existing capabilities as a pattern matcher:

```python
class HolonMatcher:
    def __init__(self):
        self.store = CPUStore()
        self.client = HolonClient(local_store=self.store)
        self.fact_index = {}  # id -> fact

    def insert_fact(self, fact):
        fact_id = self.client.insert_json(fact)
        self.fact_index[fact_id] = fact
        return fact_id

    def match_pattern(self, pattern, threshold=0.0):
        """Find facts matching pattern (exact + fuzzy)."""
        # Split pattern into probe (fuzzy) and guard (exact)
        probe = {k: v for k, v in pattern.items()
                 if not k.startswith("$") and not isinstance(v, dict)}
        guard = {k: v for k, v in pattern.items()
                 if isinstance(v, dict) and any(op.startswith("$") for op in v)}

        results = self.client.search_json(
            probe=probe,
            guard=guard,
            threshold=threshold,
            limit=100
        )
        return [(r["data"], r["score"]) for r in results]

    def match_prototype(self, prototype_vec, threshold=0.5):
        """Find facts similar to a learned prototype."""
        # Use direct vector similarity
        results = []
        for fact_id, fact in self.fact_index.items():
            fact_vec = self.client.encode_vectors(fact)
            similarity = normalized_dot_similarity(prototype_vec, fact_vec)
            if similarity >= threshold:
                results.append((fact, similarity))
        return sorted(results, key=lambda x: x[1], reverse=True)
```

### Phase 2: Rule Compilation

Compile rules into Holon queries:

```python
class Rule:
    def __init__(self, name, conditions, action, join=None):
        self.name = name
        self.conditions = conditions
        self.action = action
        self.join = join or {}

    def compile_condition(self, condition):
        """Convert condition to Holon probe + guard."""
        probe = {}
        guard = {}
        similarity_check = None

        for key, value in condition.items():
            if key == "_similar_to":
                similarity_check = value
            elif key == "_threshold":
                continue  # Used with _similar_to
            elif key.startswith("$"):
                continue  # Skip special markers
            elif isinstance(value, dict) and any(k.startswith("$") for k in value):
                guard[key] = value
            else:
                probe[key] = value

        return {
            "probe": probe,
            "guard": guard,
            "similarity": similarity_check,
            "threshold": condition.get("_threshold", 0.0)
        }
```

### Phase 3: Join Network

Handle multi-condition rules with variable bindings:

```python
class JoinNode:
    def __init__(self, left_condition, right_condition, join_spec):
        self.left = left_condition
        self.right = right_condition
        self.join_spec = join_spec  # {"Order.customer_id": "Customer.id"}

    def match(self, left_facts, right_facts):
        """Join facts based on join specification."""
        results = []
        for left in left_facts:
            for right in right_facts:
                if self.check_join(left, right):
                    results.append({"left": left, "right": right})
        return results

    def check_join(self, left, right):
        for left_path, right_path in self.join_spec.items():
            left_val = self.get_nested(left, left_path)
            right_val = self.get_nested(right, right_path)
            if left_val != right_val:
                return False
        return True
```

### Phase 4: Truth Maintenance

Track fact dependencies for automatic retraction:

```python
class TruthMaintenanceSystem:
    def __init__(self):
        self.derived_facts = {}  # fact_id -> set of source_fact_ids
        self.dependents = {}     # source_fact_id -> set of derived_fact_ids

    def record_derivation(self, derived_id, source_ids):
        self.derived_facts[derived_id] = set(source_ids)
        for source_id in source_ids:
            if source_id not in self.dependents:
                self.dependents[source_id] = set()
            self.dependents[source_id].add(derived_id)

    def retract(self, fact_id):
        """Retract fact and all derived facts."""
        to_retract = {fact_id}

        # Cascade to dependents
        if fact_id in self.dependents:
            for dependent_id in self.dependents[fact_id]:
                to_retract.update(self.retract(dependent_id))

        return to_retract
```

---

## Key Innovations

### 1. Fuzzy Condition Matching

Traditional Rete: `(= status :pending)` - binary match

Holon-enhanced: `(_similar_to pending_prototype)` - similarity score

```python
# Instead of exact enum matching
condition = {"status": "pending"}

# Allow fuzzy status matching
condition = {
    "status": {"_similar_to": "pending", "_threshold": 0.8}
}
# Matches: "pending", "awaiting", "in-progress" (if similar enough)
```

### 2. Prototype-Based Rules

Learn rule conditions from examples:

```python
# Traditional: Manually specify fraud indicators
fraud_rule = {
    "conditions": [
        {"amount": {"$gt": 10000}},
        {"country": {"$in": ["NG", "RU", "CN"]}},
        {"time": {"$lt": "06:00"}}
    ]
}

# Holon: Learn from examples
fraud_examples = [known_fraud_1, known_fraud_2, ...]
fraud_prototype = session.learn_prototype(fraud_examples)

# Rule uses learned prototype
fraud_rule = {
    "conditions": [
        {"_similar_to": fraud_prototype, "_threshold": 0.7}
    ]
}
```

### 3. Similarity-Weighted Activations

Rules return confidence scores:

```python
activations = session.fire_rules()
# [
#   {"rule": "exact-match-rule", "confidence": 1.0, "facts": [...]},
#   {"rule": "fuzzy-fraud-rule", "confidence": 0.85, "facts": [...]},
#   {"rule": "prototype-rule", "confidence": 0.72, "facts": [...]}
# ]

# Can threshold or rank by confidence
high_confidence = [a for a in activations if a["confidence"] > 0.8]
```

### 4. Incremental Prototype Updates

Update prototypes as new examples arrive:

```python
# Initial prototype from 10 examples
fraud_proto = session.learn_prototype(initial_examples)

# New confirmed fraud case
session.update_prototype("fraud", new_fraud_case, weight=0.1)
# Prototype evolves: proto = 0.9 * proto + 0.1 * new_case
```

---

## Comparison with Clara

| Feature | Clara | Holon-Rete |
|---------|-------|------------|
| **Conditions** | Exact predicates | Exact + fuzzy similarity |
| **Matching** | Binary | Continuous scores |
| **Learning** | Manual rules | Prototype from examples |
| **Updates** | Rewrite rules | Evolve prototypes |
| **Joins** | Explicit variables | Explicit + implicit similarity |
| **Performance** | Rete network | Rete + ANN indexing |

### What Clara Does Better

- Mature, battle-tested implementation
- Rich DSL for rule definition
- Logical operations (accumulate, not, exists)
- Explanation/tracing

### What Holon-Rete Could Add

- Fuzzy matching without manual thresholds
- Learn rules from examples
- Handle noisy/incomplete data
- Similarity-based joins ("find related facts")

---

## Success Metrics

### Functional

1. **Exact matching works**: Traditional Rete conditions fire correctly
2. **Fuzzy matching works**: Similarity conditions fire with appropriate thresholds
3. **Joins work**: Multi-condition rules join correctly
4. **Truth maintenance works**: Retractions cascade properly
5. **Prototypes work**: Learned prototypes match expected facts

### Performance

1. **Insert latency**: < 10ms per fact
2. **Fire latency**: < 100ms for 1000 facts, 100 rules
3. **Memory**: < 1MB per 1000 facts
4. **Incremental**: Adding facts doesn't re-evaluate all rules

### Quality

1. **Precision**: Fuzzy rules don't over-fire
2. **Recall**: Fuzzy rules catch intended cases
3. **Ranking**: Higher similarity = higher confidence

---

## Open Questions

1. **How to express fuzzy conditions in a DSL?** Clara's syntax is Clojure-native. What's the Python equivalent?

2. **How to handle negation?** "NOT similar to X" - what's the threshold?

3. **How to do fuzzy joins?** "Find Customer similar to Order's customer pattern" without explicit ID match.

4. **How to explain activations?** Why did this rule fire with 0.72 confidence?

5. **How to integrate with existing Rete?** Can Holon be a plugin to Clara rather than replacement?

---

## Implementation Roadmap

### MVP (1 week)
- [ ] Basic pattern matching with Holon
- [ ] Single-condition rules
- [ ] Exact + fuzzy conditions

### V1 (2 weeks)
- [ ] Multi-condition rules with joins
- [ ] Prototype learning
- [ ] Confidence scores

### V2 (1 month)
- [ ] Truth maintenance
- [ ] Incremental updates
- [ ] Rule DSL

### V3 (future)
- [ ] Clara integration
- [ ] Explanation system
- [ ] Persistent sessions

---

## Existing Work

There's already a basic Rete demo in the codebase:

```
examples/rete_demo.py
```

This implements:
- Fact insertion with metadata
- Rules as query patterns
- Forward chaining with iterations
- Derived fact generation

Use this as a starting point for the enhanced implementation.

---

## References

- [Rete Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Rete_algorithm)
- [Clara Rules](https://github.com/cerner/clara-rules)
- [Clara Documentation](http://www.clara-rules.org/)
- [Production Systems and Rete](https://www.cs.cmu.edu/~dst/LispBook/book-final.pdf) (Chapter 7)
- [Holon rete_demo.py](../../examples/rete_demo.py) - Existing basic implementation

---

*Challenge designed for Holon VSA/HDC exploration, Jan 2026*
