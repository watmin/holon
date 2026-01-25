#!/usr/bin/env python3

import json

from holon import CPUStore

# Test guards filtering out violating items
store = CPUStore()

# Insert similar data with variations
data = [
    {"user": "alice", "action": "login", "status": "success", "ip": "192.168.1.1"},
    {
        "user": "alice",
        "action": "login",
        "status": "failed",
        "ip": "192.168.1.2",
    },  # Similar but failed
    {
        "user": "alice",
        "action": "logout",
        "status": "success",
        "ip": "192.168.1.1",
    },  # Different action
    {
        "user": "bob",
        "action": "login",
        "status": "success",
        "ip": "10.0.0.1",
    },  # Different user
    {
        "user": "alice",
        "action": "login",
        "status": "success",
        "ip": "192.168.1.3",
        "extra": "data",
    },  # Extra field
]

for item in data:
    store.insert(json.dumps(item))

print(f"Inserted {len(data)} items")

# Probe: something that matches most (alice login)
probe = json.dumps({"user": "alice", "action": "login"})

# Query without guard - should return alice logins (success and failed)
results_no_guard = store.query(probe, top_k=10)
print(f"\nQuery without guard: {len(results_no_guard)} results")
for res in results_no_guard:
    print(
        f"  User: {res[2]['user']}, Action: {res[2]['action']}, "
        f"Status: {res[2]['status']}, IP: {res[2]['ip']}"
    )

# Query with guard: only success status
guard_success = {"status": "success"}
results_guarded = store.query(probe, top_k=10, guard=guard_success)
print(f"\nQuery with guard (status=='success'): {len(results_guarded)} results")
for res in results_guarded:
    print(
        f"  User: {res[2]['user']}, Action: {res[2]['action']}, "
        f"Status: {res[2]['status']}, IP: {res[2]['ip']}"
    )

# Verify filtering: should exclude the failed login
failed_logins = [r for r in results_no_guard if r[2]["status"] == "failed"]
print(f"\nFailed logins in unguarded results: {len(failed_logins)}")
print(
    f"Failed logins in guarded results: "
    f"{len([r for r in results_guarded if r[2]['status'] == 'failed'])}"
)

if len([r for r in results_guarded if r[2]["status"] == "failed"]) == 0:
    print("✅ Guard successfully filtered out failed logins!")
else:
    print("❌ Guard failed to filter.")

# Test nested guard: require 'ip' key present
guard_ip = {"ip": None}
results_ip_guard = store.query(probe, top_k=10, guard=guard_ip)
print(f"\nQuery with guard ('ip' present): {len(results_ip_guard)} results")
# Should be same as no guard since all have ip

# Test guard that excludes some: require 'extra' field
guard_extra = {"extra": None}
results_extra_guard = store.query(probe, top_k=10, guard=guard_extra)
print(f"\nQuery with guard ('extra' present): {len(results_extra_guard)} results")
print("  Should be 1 result (the one with extra field)")
