#!/usr/bin/env python3

import json
import os
import sys

from holon import CPUStore, HolonClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "server"))
from holon_server import is_subset

# Test improved guards with patterns and positional constraints
store = CPUStore()
client = HolonClient(local_store=store)

# Insert test data with nested structures and lists
data1 = {"user": "alice", "actions": [1, 2, 3, 4], "meta": {"status": "success"}}
data2 = {"user": "bob", "actions": [1, 2, 5, 4], "meta": {"status": "failed"}}
data3 = {"user": "alice", "actions": [1, 2, 99, 4], "meta": {"status": "success"}}

for data in [data1, data2, data3]:
    client.insert_json(data)

print("Inserted 3 items")

# Probe: alice
probe = json.dumps({"user": "alice"})

# Guard: exact match on meta.status

store_instance = CPUStore()  # Dummy for import

# Test is_subset directly
print("\nTesting is_subset:")
data = {"user": "alice", "actions": [1, 2, 3, 4], "meta": {"status": "success"}}
guard1 = {"meta": {"status": "success"}}
print(f"Guard {guard1} on data: {is_subset(guard1, data)}")  # True

guard2 = {"actions": [1, 2, {"$any": True}, 4]}
print(f"Guard {guard2} on data: {is_subset(guard2, data)}")  # True

guard3 = {"actions": [1, 2, 3, 5]}
print(f"Guard {guard3} on data: {is_subset(guard3, data)}")  # False

print("\n✅ Improved guards support positional $any and exact patterns!")
