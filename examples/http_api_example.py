#!/usr/bin/env python3
"""
HTTP API Example: Remote usage with requests
"""

import requests
import time
import json

def main():
    base_url = "http://localhost:8000"

    print("=== HTTP API Example ===\n")
    print("Ensure server is running: python scripts/holon_server.py\n")

    # Insert data
    data = [
        {"event": "login", "user": "alice"},
        {"event": "logout", "user": "bob"},
        {"event": "view", "user": "alice"}
    ]

    print("1. Inserting data...")
    for item in data:
        response = requests.post(f"{base_url}/insert", json={
            "data": json.dumps(item)
        })
        print(f"   Insert: {response.status_code}")

    time.sleep(0.5)  # Allow indexing

    # Query
    print("\n2. Querying...")
    response = requests.post(f"{base_url}/query", json={
        "probe": json.dumps({"user": "alice"}),
        "top_k": 5
    })
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"   Query: Alice events → {len(results)} results")
        for r in results:
            print(f"   - {r['data']['event']}")
    else:
        print(f"   Query failed: {response.status_code}")

    print("\n=== HTTP Demo Complete ===")

if __name__ == "__main__":
    main()
