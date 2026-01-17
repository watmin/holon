#!/usr/bin/env python3

from holon import CPUStore
import json

# Test vector tricks: $any and vector negation
store = CPUStore()

# Insert test data
data = [
    {"user": "alice", "action": "login", "status": "success"},
    {"user": "alice", "action": "logout", "status": "success"},
    {"user": "bob", "action": "login", "status": "failed"},
    {"user": "alice", "action": "view", "status": "success"},
]

for item in data:
    store.insert(json.dumps(item))

print("Inserted 4 items")

# Test $any: match alice with any action
probe_any = json.dumps({"user": "alice", "action": "$any"})
results_any = store.query(probe_any, top_k=10)
print(f"\nProbe with $any: {len(results_any)} results")
for res in results_any:
    print(f"  {res[2]['user']} - {res[2]['action']} - {res[2]['status']}")

# Should match all alice, regardless of action

# Test vector negation: alice with any action, but exclude status success
# But since vector subtraction, it should reduce similarity for success items
probe_any_neg = json.dumps({"user": "alice", "action": "$any"})
results_neg = store.query(probe_any_neg, top_k=10, negations={"status": "success"})
print(f"\nProbe with $any and negation (exclude success): {len(results_neg)} results")
for res in results_neg:
    print(f"  {res[2]['user']} - {res[2]['action']} - {res[2]['status']}")

# With vector subtraction, success items should have lower similarity, but since we have data fallback, it will exclude them.

print("\nVector tricks: $any for wildcards, subtraction for negation patterns")