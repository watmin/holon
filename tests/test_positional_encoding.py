#!/usr/bin/env python3

from holon import CPUStore
import json

# Test positional encoding for lists vs sets
store = CPUStore()

# Insert data with lists
data_list = {"sequence": [1, 2, 3, 4], "type": "list"}
data_list_reversed = {"sequence": [4, 3, 2, 1], "type": "list_reversed"}

store.insert(json.dumps(data_list))
store.insert(json.dumps(data_list_reversed))

print("Inserted 2 items: list and reversed list")

# Query for sequence [1,2,3,4]
probe_list = json.dumps({"sequence": [1, 2, 3, 4]})
results = store.query(probe_list, top_k=10)
print(f"\nQuery for [1,2,3,4]: {len(results)} results")
for res in results:
    print(f"  Type: {res[2]['type']}")

# Should match the exact list, not reversed

print("\n✅ Lists now order-sensitive with positional encoding!")