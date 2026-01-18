#!/usr/bin/env python3

from holon import CPUStore
import json

# Test guard functionality
if __name__ == "__main__":
    store = CPUStore()

    # Insert test data
    data1 = json.dumps({"user": "alice", "action": "login", "status": "success"})
    data2 = json.dumps({"user": "bob", "action": "login", "status": "failed"})
    data3 = json.dumps({"user": "alice", "action": "logout", "status": "success"})

    store.insert(data1)
    store.insert(data2)
    store.insert(data3)

    print("Inserted 3 items")

    # Query without guard
    probe = json.dumps({"user": "alice"})
    results = store.query(probe, top_k=10)
    print(f"\nQuery without guard: {len(results)} results")
    for res in results:
        print(f"  User: {res[2]['user']}, Action: {res[2]['action']}, Status: {res[2]['status']}")

    # Query with guard: only success status
    success_guard = {"status": "success"}

    results_guarded = store.query(probe, top_k=10, guard=success_guard)
    print(f"\nQuery with guard (status=='success'): {len(results_guarded)} results")
    for res in results_guarded:
        print(f"  User: {res[2]['user']}, Action: {res[2]['action']}, Status: {res[2]['status']}")

    # Query with guard: only login actions
    login_guard = {"action": "login"}

    results_login = store.query(probe, top_k=10, guard=login_guard)
    print(f"\nQuery with guard (action=='login'): {len(results_login)} results")
    for res in results_login:
        print(f"  User: {res[2]['user']}, Action: {res[2]['action']}, Status: {res[2]['status']}")