#!/usr/bin/env python3

import json

from holon import CPUStore, HolonClient

# Test vector tricks: $any and vector negation
store = CPUStore()
client = HolonClient(local_store=store)

# Insert test data
data = [
    {"user": "alice", "action": "login", "status": "success"},
    {"user": "alice", "action": "logout", "status": "success"},
    {"user": "bob", "action": "login", "status": "failed"},
    {"user": "alice", "action": "view", "status": "success"},
]

for item in data:
    client.insert_json(item)

print("Inserted 4 items")

# Test user-specified $any: match alice with any action
probe_any = {"user": "alice", "action": {"$any": True}}
results_any = client.search_json(probe_any, limit=10)
print(f"\nProbe with $any: {len(results_any)} results")
for res in results_any:
    print(f"  {res['data']['user']} - {res['data']['action']} - {res['data']['status']}")

# Should match all alice, regardless of action

# Test user-specified negation: exclude {"status": {"$not": "success"}}
probe_any_neg = {"user": "alice", "action": {"$any": True}}
results_neg = client.search_json(
    probe_any_neg, limit=10, negations={"status": {"$not": "success"}}
)
print(
    f"\nProbe with $any and user-specified negation (exclude success): {len(results_neg)} results"
)
for res in results_neg:
    print(f"  {res['data']['user']} - {res['data']['action']} - {res['data']['status']}")

# Test deep negation: suppose data had {"meta": {"details": {"status": "success"}}}, exclude that
# But our data is flat. For demo, add nested data.

nested_data = {
    "user": "charlie",
    "meta": {"details": {"status": "success"}},
    "action": "login",
}
client.insert_json(nested_data)

probe_deep = {"user": "charlie"}
results_deep = client.search_json(
    probe_deep, limit=10, negations={"meta": {"details": {"status": "success"}}}
)
print(
    f"\nProbe with deep negation (exclude nested success): {len(results_deep)} results"
)
for res in results_deep:
    print(f"  {res['data']}")

# Test $or: match alice OR success
probe_or = {"$or": [{"user": "alice"}, {"status": "success"}]}
results_or = client.search_json(probe_or, limit=10)
print(f"\nProbe with $or (alice OR success): {len(results_or)} results")
for res in results_or:
    status = res['data'].get("status", "N/A")
    print(f"  {res['data']['user']} - {res['data']['action']} - {status}")

print(
    "\nVector tricks: $any for wildcards, $or for disjunctions, "
    "subtraction for negation patterns at any depth"
)
