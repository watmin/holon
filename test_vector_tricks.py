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

# Test nested negation: exclude {"status": "success"}
probe_any_neg = json.dumps({"user": "alice", "action": "$any"})
results_neg = store.query(probe_any_neg, top_k=10, negations={"status": "success"})
print(f"\nProbe with $any and flat negation (exclude success): {len(results_neg)} results")
for res in results_neg:
    print(f"  {res[2]['user']} - {res[2]['action']} - {res[2]['status']}")

# Test deep negation: suppose data had {"meta": {"details": {"status": "success"}}}, exclude that
# But our data is flat. For demo, add nested data.

nested_data = {"user": "charlie", "meta": {"details": {"status": "success"}}, "action": "login"}
store.insert(json.dumps(nested_data))

probe_deep = json.dumps({"user": "charlie"})
results_deep = store.query(probe_deep, top_k=10, negations={"meta": {"details": {"status": "success"}}})
print(f"\nProbe with deep negation (exclude nested success): {len(results_deep)} results")
for res in results_deep:
    print(f"  {res[2]}")

print("\nVector tricks: $any for wildcards, subtraction for negation patterns at any depth")