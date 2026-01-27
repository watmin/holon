#!/usr/bin/env python3
"""
HTTP API Example: Remote usage with requests

⚠️  DEPRECATED: Use examples/unified_client_example.py instead

This example shows direct HTTP API usage. For new code, use HolonClient
which provides the same interface for both local and remote usage.
"""

import json
import time

import requests


def main():
    base_url = "http://localhost:8000"

    print("=== DEPRECATED HTTP API Example ===\n")
    print("⚠️  This shows direct HTTP calls to the new v1 API.")
    print("   For new code, use examples/unified_client_example.py")
    print("   It provides the same interface for local and remote usage.\n")
    print("Ensure server is running: python scripts/server/holon_server.py\n")

    # Insert data
    data = [
        {"event": "login", "user": "alice"},
        {"event": "logout", "user": "bob"},
        {"event": "view", "user": "alice"},
    ]

    print("1. Inserting data...")
    for item in data:
        response = requests.post(
            f"{base_url}/api/v1/items",
            json={"data": json.dumps(item), "data_type": "json"},
        )
        print(f"   Insert: {response.status_code}")

    time.sleep(0.5)  # Allow indexing

    # Query
    print("\n2. Querying...")
    response = requests.post(
        f"{base_url}/api/v1/search",
        json={"probe": json.dumps({"user": "alice"}), "data_type": "json", "top_k": 5},
    )
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
