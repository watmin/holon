#!/usr/bin/env python3
"""
Test script for the new v1 API endpoints.
"""

import json
import requests
import subprocess
import time
import sys
from contextlib import contextmanager


@contextmanager
def holon_server():
    """Start Holon server for testing."""
    print("🚀 Starting Holon server...")

    # Start server on a different port to avoid conflicts
    server = subprocess.Popen([
        sys.executable, "scripts/server/holon_server.py"
    ], cwd="/home/watmin/work/holon",
       env={**dict(os.environ), "PORT": "8001"})

    time.sleep(2)  # Wait for startup

    try:
        # Test health check
        response = requests.get("http://localhost:8001/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server started successfully")
            yield "http://localhost:8001"
        else:
            print("❌ Server health check failed")
            yield None
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        yield None
    finally:
        server.terminate()
        server.wait()
        print("🛑 Server stopped")


def test_new_api(base_url):
    """Test the new v1 API endpoints."""
    if not base_url:
        return False

    print(f"\n🧪 Testing new v1 API at {base_url}")

    try:
        # 1. Health check
        print("1. Testing /api/v1/health...")
        response = requests.get(f"{base_url}/api/v1/health")
        assert response.status_code == 200
        health = response.json()
        print(f"   ✅ Health: {health['status']}, Items: {health['items_count']}")

        # 2. Create single item
        print("2. Testing POST /api/v1/items...")
        item_data = {"event": "login", "user": "alice"}
        response = requests.post(
            f"{base_url}/api/v1/items",
            json={"data": json.dumps(item_data), "data_type": "json"}
        )
        assert response.status_code == 200
        result = response.json()
        item_id = result["id"]
        print(f"   ✅ Created item with ID: {item_id}")

        # 3. Get item by ID
        print("3. Testing GET /api/v1/items/{id}...")
        response = requests.get(f"{base_url}/api/v1/items/{item_id}")
        assert response.status_code == 200
        retrieved = response.json()
        assert retrieved["id"] == item_id
        assert retrieved["data"]["user"] == "alice"
        print(f"   ✅ Retrieved item: {retrieved['data']}")

        # 4. Vector encoding
        print("4. Testing POST /api/v1/vectors/encode...")
        response = requests.post(
            f"{base_url}/api/v1/vectors/encode",
            json={"data": json.dumps({"type": "test"}), "data_type": "json"}
        )
        assert response.status_code == 200
        vector_result = response.json()
        assert "vector" in vector_result
        assert "encoding_type" in vector_result
        print(f"   ✅ Encoded vector: {len(vector_result['vector'])}D, type: {vector_result['encoding_type']}")

        # 5. Search
        print("5. Testing POST /api/v1/search...")
        response = requests.post(
            f"{base_url}/api/v1/search",
            json={
                "probe": json.dumps({"user": "alice"}),
                "data_type": "json",
                "top_k": 5
            }
        )
        assert response.status_code == 200
        search_result = response.json()
        assert "results" in search_result
        assert "count" in search_result
        print(f"   ✅ Search returned {search_result['count']} results")

        print("\n🎉 All new API tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import os
    with holon_server() as base_url:
        success = test_new_api(base_url)
        if success:
            print("\n✅ New v1 API is working correctly!")
        else:
            print("\n❌ New v1 API tests failed!")
            sys.exit(1)
