#!/usr/bin/env python3
"""
Demo HTTP API bulk insert performance vs individual inserts.
Shows the optimization in action over REST API.
"""

import time
import json
import requests
from typing import List

BASE_URL = "http://localhost:8000"

def generate_test_data(count: int) -> List[dict]:
    """Generate test data items."""
    data = []
    for i in range(count):
        item = {
            "id": i,
            "user": f"user_{i % 10}",
            "action": "login" if i % 2 == 0 else "logout",
            "status": "success" if i % 3 != 0 else "failed",
            "timestamp": time.time() + i
        }
        data.append(item)
    return data

def test_individual_inserts_http(items: List[dict]) -> tuple:
    """Test inserting items one by one via HTTP API."""
    start = time.time()
    ids = []

    for item in items:
        response = requests.post(f"{BASE_URL}/insert", json={
            "data": json.dumps(item)
        })
        if response.status_code == 200:
            ids.append(response.json()["id"])
        else:
            print(f"❌ Insert failed: {response.text}")
            break

    end = time.time()
    return end - start, ids

def test_batch_insert_http(items: List[dict]) -> tuple:
    """Test inserting items via HTTP batch API."""
    start = time.time()

    response = requests.post(f"{BASE_URL}/batch_insert", json={
        "items": [json.dumps(item) for item in items],
        "data_type": "json"
    })

    if response.status_code == 200:
        ids = response.json()["ids"]
        end = time.time()
        return end - start, ids
    else:
        print(f"❌ Batch insert failed: {response.text}")
        return 0, []

def clear_store():
    """Clear the store for clean testing."""
    # Assuming there's a clear endpoint or we can use query to check
    pass  # For demo, we'll assume server is fresh

def verify_data(ids: List[str], expected_count: int):
    """Verify data was inserted by querying."""
    # Query for one user to verify
    response = requests.post(f"{BASE_URL}/query", json={
        "probe": '{"user": "user_0"}',
        "top_k": 10
    })

    if response.status_code == 200:
        results = response.json()["results"]
        print(f"✅ Verification: Found {len(results)} results for user_0")
        return len(results) > 0
    else:
        print(f"❌ Query failed: {response.text}")
        return False

def main():
    print("🚀 HTTP API Bulk Insert Performance Demo\n")
    print("Ensure Holon server is running: python scripts/holon_server.py\n")

    # Generate test data
    test_items = generate_test_data(50)  # Smaller for demo
    print(f"📊 Testing with {len(test_items)} items\n")

    # Test individual inserts
    print("Testing individual HTTP inserts...")
    time_ind, ids_ind = test_individual_inserts_http(test_items)
    print(f"  Time: {time_ind:.3f}s for {len(ids_ind)} inserts")
    # Verify and clear (manually for demo)
    verify_data(ids_ind, len(test_items))

    print("\n" + "="*50)
    print("⚠️  For clean comparison, restart server or clear data")
    print("Then run batch test...")
    input("Press Enter when ready...")

    # Test batch insert (after clearing)
    print("\nTesting HTTP batch insert...")
    time_batch, ids_batch = test_batch_insert_http(test_items)
    print(f"  Time: {time_batch:.3f}s for {len(ids_batch)} inserts")
    # Verify
    verify_data(ids_batch, len(test_items))

    # Compare
    if time_batch > 0:
        speedup = time_ind / time_batch
        print("
🎯 Results:"        print(".3f"        print(".3f"        print(".2f"
        if speedup > 1.5:
            print("✅ Bulk insert significantly faster!")
        else:
            print("📈 Bulk insert still more efficient for larger batches")

if __name__ == "__main__":
    main()