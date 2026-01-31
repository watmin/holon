#!/usr/bin/env python3
"""
Test unified client with HTTP API.
"""

import os
import subprocess
import sys
import time

from holon import HolonClient


def test_http_client():
    """Test HolonClient with HTTP API."""
    print("🌐 Testing Unified Client with HTTP API")
    print("=" * 45)

    # Start server
    print("🚀 Starting Holon server...")
    server = subprocess.Popen([
        sys.executable, "scripts/server/holon_server.py"
    ], cwd="/home/watmin/work/holon",
       env={**dict(os.environ), "PORT": "8002"})

    time.sleep(3)  # Wait for startup

    try:
        # Test HTTP client
        client = HolonClient(remote_url="http://localhost:8002")

        # Health check
        print("1. Health check...")
        health = client.health()
        print(f"   ✅ Status: {health['status']}")

        # Insert data
        print("2. Inserting data...")
        task_id = client.insert_json({"type": "task", "title": "Test HTTP", "priority": "high"})
        print(f"   ✅ Inserted task with ID: {task_id}")

        # Search
        print("3. Searching...")
        results = client.search_json({"type": "task"}, top_k=5)
        print(f"   ✅ Found {len(results)} results")

        # Vector encoding
        print("4. Vector encoding...")
        vector = client.encode_vectors_json({"action": "test"})
        print(f"   ✅ Encoded {len(vector)}D vector")

        print("\n🎉 HTTP client test successful!")
        return True

    except Exception as e:
        print(f"❌ HTTP client test failed: {e}")
        return False
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    success = test_http_client()
    sys.exit(0 if success else 1)
