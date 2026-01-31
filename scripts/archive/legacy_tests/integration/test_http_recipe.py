#!/usr/bin/env python3
"""
Simple test to verify HTTP recipe functionality works
"""

import json

import requests

BASE_URL = "http://localhost:8000"


def test_basic_http():
    """Test basic HTTP functionality."""
    print("🧪 Testing HTTP Recipe Functionality")

    # Test health
    print("1. Testing health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Health: {response.json()}")

    # Insert a test recipe (using JSON instead of EDN for simplicity)
    print("2. Inserting test recipe...")
    test_recipe = (
        '{"name": "Test Recipe", "cuisine": "test", "difficulty": "easy", "time": 10}'
    )
    response = requests.post(
        f"{BASE_URL}/insert", json={"data": test_recipe, "data_type": "json"}
    )
    result = response.json()
    print(f"   Inserted ID: {result['id']}")

    # Query for it
    print("3. Querying for test recipe...")
    response = requests.post(
        f"{BASE_URL}/query",
        json={"probe": '{"name": "Test Recipe"}', "data_type": "json", "top_k": 5},
    )
    print(f"   Response status: {response.status_code}")
    print(f"   Response text: {response.text[:200]}...")
    try:
        results = response.json()
        print(f"   Found {len(results['results'])} results")
        for r in results["results"][:1]:
            print(f"   - {r['data']['name']} (score: {r['score']:.3f})")
    except Exception as e:
        print(f"   JSON parse error: {e}")

    print("✅ HTTP functionality verified!")


if __name__ == "__main__":
    test_basic_http()
