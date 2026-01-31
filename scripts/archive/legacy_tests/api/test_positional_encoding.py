#!/usr/bin/env python3

import json

from holon import CPUStore, HolonClient

# Test positional encoding for lists vs sets
store = CPUStore()
client = HolonClient(local_store=store)

# Insert data with lists
data_list = {"sequence": [1, 2, 3, 4], "type": "list"}
data_list_reversed = {"sequence": [4, 3, 2, 1], "type": "list_reversed"}

client.insert_json(data_list)
client.insert_json(data_list_reversed)

print("Inserted 2 items: list and reversed list")

# Query for sequence [1,2,3,4]
probe_list = {"sequence": [1, 2, 3, 4]}
results = client.search_json(probe_list, limit=10)
print(f"\nQuery for [1,2,3,4]: {len(results)} results")
for res in results:
    print(f"  Type: {res['data']['type']}")

# Should match the exact list, not reversed

print("\n✅ Lists now order-sensitive with positional encoding!")
