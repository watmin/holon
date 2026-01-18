#!/usr/bin/env python3

from holon import CPUStore
import json

# Test negation in probes
store = CPUStore()

# Insert test data
data1 = {"user": "alice", "action": "login", "status": "success"}
data2 = {"user": "alice", "action": "login", "status": "failed"}
data3 = {"user": "bob", "action": "login", "status": "success"}

for data in [data1, data2, data3]:
    store.insert(json.dumps(data))

print("Inserted 3 items")

# Normal probe: should return all alice logins
probe_normal = json.dumps({"user": "alice", "action": "login"})
results_normal = store.query(probe_normal, top_k=10)
print(f"\nNormal probe results: {len(results_normal)}")
for res in results_normal:
    print(f"  Status: {res[2]['status']}")

# Probe with negation: exclude failed
probe_neg = json.dumps({"user": "alice", "action": "login"})
results_neg = store.query(probe_neg, top_k=10, negations={"status": {"$not": "failed"}})
print(f"\nNegated probe results: {len(results_neg)}")
for res in results_neg:
    print(f"  Status: {res[2]['status']}")

# Debug
print(f"Normal results: {len(results_normal)}")
print(f"Neg results: {len(results_neg)}")
failed_in_normal = sum(1 for r in results_normal if r[2].get('status') == 'failed')
failed_in_neg = sum(1 for r in results_neg if r[2].get('status') == 'failed')
print(f"Failed in normal: {failed_in_normal}, in neg: {failed_in_neg}")
if failed_in_neg == 0 and failed_in_normal > 0:
    print("✅ Negation working: excluded failed status")
else:
    print("❌ Negation not working")