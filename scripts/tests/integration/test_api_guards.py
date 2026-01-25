#!/usr/bin/env python3

import json
import time

import requests

# Test API with guards
if __name__ == "__main__":
    BASE_URL = "http://localhost:8000"

    # Start server in background
    import os
    import subprocess

    print("Starting Holon server...")
    server = subprocess.Popen(
        ["python", "scripts/server/holon_server.py"], cwd=os.getcwd()
    )

    # Wait for server
    time.sleep(2)

    try:
        # Insert test data
        data1 = {"user": "alice", "action": "login", "status": "success"}
        data2 = {"user": "bob", "action": "login", "status": "failed"}
        data3 = {"user": "alice", "action": "logout", "status": "success"}

        for data in [data1, data2, data3]:
            response = requests.post(
                f"{BASE_URL}/insert", json={"data": json.dumps(data)}
            )
            print(f"Insert: {response.status_code}")

        # Query without guard
        probe = {"user": "alice"}
        response = requests.post(
            f"{BASE_URL}/query", json={"probe": json.dumps(probe), "top_k": 10}
        )
        results = response.json()["results"]
        print(f"\nQuery without guard: {len(results)} results")
        for res in results:
            print(
                f"  {res['data']['user']} - {res['data']['action']} - {res['data']['status']}"
            )

        # Query with guard: status == success (presence of "status" key, but since we check presence, and value is dict)
        # Guard: {"status": null} or just {"status": {}}
        # But user said "null" is better, but for subset, since we ignore value, {"status": null} would work.

        guard = {"status": None}  # But None is null in JSON
        response = requests.post(
            f"{BASE_URL}/query",
            json={"probe": json.dumps(probe), "top_k": 10, "guard": json.dumps(guard)},
        )
        results = response.json()["results"]
        print(f"\nQuery with guard (status present): {len(results)} results")
        for res in results:
            print(
                f"  {res['data']['user']} - {res['data']['action']} - {res['data']['status']}"
            )

        # Guard for nested: {"complex": {"nested": {"foo": null}}}
        # But our data doesn't have that. For test, use existing.

        guard_nested = {"user": None}  # Should match all since all have user
        response = requests.post(
            f"{BASE_URL}/query",
            json={
                "probe": json.dumps(probe),
                "top_k": 10,
                "guard": json.dumps(guard_nested),
            },
        )
        results = response.json()["results"]
        print(f"\nQuery with guard (user present): {len(results)} results")

    finally:
        server.terminate()
        server.wait()
