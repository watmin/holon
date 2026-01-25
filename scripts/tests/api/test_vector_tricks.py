#!/usr/bin/env python3

import json

from holon import CPUStore

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

# Test user-specified $any: match alice with any action
probe_any = json.dumps({"user": "alice", "action": {"$any": True}})
results_any = store.query(probe_any, top_k=10)
print(f"\nProbe with $any: {len(results_any)} results")
for res in results_any:
    print(f"  {res[2]['user']} - {res[2]['action']} - {res[2]['status']}")

# Should match all alice, regardless of action

# Test user-specified negation: exclude {"status": {"$not": "success"}}
probe_any_neg = json.dumps({"user": "alice", "action": {"$any": True}})
results_neg = store.query(
    probe_any_neg, top_k=10, negations={"status": {"$not": "success"}}
)
print(
    f"\nProbe with $any and user-specified negation (exclude success): {len(results_neg)} results"
)
for res in results_neg:
    print(f"  {res[2]['user']} - {res[2]['action']} - {res[2]['status']}")

# Test deep negation: suppose data had {"meta": {"details": {"status": "success"}}}, exclude that
# But our data is flat. For demo, add nested data.

nested_data = {
    "user": "charlie",
    "meta": {"details": {"status": "success"}},
    "action": "login",
}
store.insert(json.dumps(nested_data))

probe_deep = json.dumps({"user": "charlie"})
results_deep = store.query(
    probe_deep, top_k=10, negations={"meta": {"details": {"status": "success"}}}
)
print(
    f"\nProbe with deep negation (exclude nested success): {len(results_deep)} results"
)
for res in results_deep:
    print(f"  {res[2]}")

# Test $or: match alice OR success
probe_or = json.dumps({"$or": [{"user": "alice"}, {"status": "success"}]})
results_or = store.query(probe_or, top_k=10)
print(f"\nProbe with $or (alice OR success): {len(results_or)} results")
for res in results_or:
    status = res[2].get("status", "N/A")
    print(f"  {res[2]['user']} - {res[2]['action']} - {status}")

print(
    "\nVector tricks: $any for wildcards, $or for disjunctions, "
    "subtraction for negation patterns at any depth"
)
