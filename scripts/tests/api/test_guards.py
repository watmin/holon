#!/usr/bin/env python3

import json

from holon import CPUStore, HolonClient

# Test guard functionality
if __name__ == "__main__":
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert test data
    data1 = {"user": "alice", "action": "login", "status": "success"}
    data2 = {"user": "bob", "action": "login", "status": "failed"}
    data3 = {"user": "alice", "action": "logout", "status": "success"}

    client.insert_json(data1)
    client.insert_json(data2)
    client.insert_json(data3)

    print("Inserted 3 items")

    # Query without guard
    probe = {"user": "alice"}
    results = client.search_json(probe, top_k=10)
    print(f"\nQuery without guard: {len(results)} results")
    for res in results:
        print(
            f"  User: {res['data']['user']}, Action: {res['data']['action']}, Status: {res['data']['status']}"
        )

    # Query with guard: only success status
    success_guard = {"status": "success"}

    results_guarded = client.search_json(probe, top_k=10, guard=success_guard)
    print(f"\nQuery with guard (status=='success'): {len(results_guarded)} results")
    for res in results_guarded:
        print(
            f"  User: {res['data']['user']}, Action: {res['data']['action']}, Status: {res['data']['status']}"
        )

    # Query with guard: only login actions
    login_guard = {"action": "login"}

    results_login = client.search_json(probe, top_k=10, guard=login_guard)
    print(f"\nQuery with guard (action=='login'): {len(results_login)} results")
    for res in results_login:
        print(
            f"  User: {res['data']['user']}, Action: {res['data']['action']}, Status: {res['data']['status']}"
        )
