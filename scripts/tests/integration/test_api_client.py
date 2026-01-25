#!/usr/bin/env python3

"""
Simple client to test the Holon HTTP API.
"""

import json

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_insert():
    """Test single insert."""
    print("Testing /insert...")
    data = {
        "data": '{"name": "Alice", "skills": ["python", "ai"]}',
        "data_type": "json",
    }
    response = requests.post(f"{BASE_URL}/insert", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Inserted ID: {result['id']}")
    return result["id"]


def test_batch_insert():
    """Test batch insert."""
    print("Testing /batch_insert...")
    data = {
        "items": [
            '{"name": "Bob", "skills": ["clojure", "ml"]}',
            '{"name": "Charlie", "skills": ["python", "data"]}',
        ],
        "data_type": "json",
    }
    response = requests.post(f"{BASE_URL}/batch_insert", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Inserted IDs: {result['ids']}")
    print()


def test_query():
    """Test query."""
    print("Testing /query...")
    data = {
        "probe": '{"skills": ["python"]}',
        "data_type": "json",
        "top_k": 5,
        "threshold": 0.0,
    }
    response = requests.post(f"{BASE_URL}/query", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Found {len(result['results'])} results:")
    for r in result["results"][:3]:  # Show first 3
        print(".3f")
    print()


def main():
    print("Holon API Test Client")
    print("=" * 40)

    # Test health
    test_health()

    # Test insert
    inserted_id = test_insert()

    # Test batch insert
    test_batch_insert()

    # Test query
    test_query()

    print("API testing complete!")


if __name__ == "__main__":
    main()
