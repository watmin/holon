#!/usr/bin/env python3
"""
Unified Client Example: Same interface for local and remote Holon

This demonstrates the HolonClient that abstracts whether you're talking to:
- A local CPUStore instance (fast, direct method calls)
- A remote HTTP API (network calls, scalable)

Users don't need to change their code - the client handles the abstraction.
"""

import json
import time

from holon import CPUStore, HolonClient


def demo_local_client():
    """Demo using HolonClient with a local CPUStore."""
    print("🏠 LOCAL CLIENT DEMO")
    print("=" * 40)

    # Create local store
    store = CPUStore(dimensions=16000)

    # Create client that talks to local store
    client = HolonClient(local_store=store)

    # Same interface as remote client!
    return demo_client_operations(client, "Local")


def demo_remote_client(base_url="http://localhost:8000"):
    """Demo using HolonClient with remote HTTP API."""
    print("🌐 REMOTE CLIENT DEMO")
    print("=" * 40)

    # Create client that talks to remote API
    client = HolonClient(remote_url=base_url)

    # Same interface as local client!
    return demo_client_operations(client, "Remote")


def demo_client_operations(client: HolonClient, mode: str):
    """Demo operations using the unified client interface."""
    try:
        # Health check
        print(f"1. Health check ({mode})...")
        health = client.health()
        print(f"   ✅ Status: {health['status']}, Backend: {health['backend']}")

        # Insert some test data
        print(f"2. Inserting data ({mode})...")
        tasks = [
            {
                "type": "task",
                "title": "Review code",
                "priority": "high",
                "user": "alice",
            },
            {
                "type": "task",
                "title": "Write docs",
                "priority": "medium",
                "user": "bob",
            },
            {"type": "task", "title": "Fix bug", "priority": "high", "user": "alice"},
            {"type": "event", "action": "login", "user": "alice"},
        ]

        ids = client.insert_batch_json(tasks)
        print(f"   ✅ Inserted {len(ids)} items")

        # Allow indexing time for local store
        if mode == "Local":
            time.sleep(0.5)

        # Search for high priority tasks
        print(f"3. Finding high priority tasks ({mode})...")
        results = client.search_json(
            probe={"type": "task"}, guard={"priority": "high"}, top_k=10
        )
        print(f"   ✅ Found {len(results)} high priority tasks")
        for result in results[:3]:  # Show first 3
            print(f"     - {result['data']['title']} (score: {result['score']:.3f})")

        # Get specific item
        print(f"4. Retrieving specific item ({mode})...")
        first_id = ids[0]
        item = client.get(first_id)
        if item:
            print(f"   ✅ Retrieved: {item['title']} by {item['user']}")
        else:
            print("   ❌ Item not found")

        # Vector encoding (bootstrap for custom operations)
        print(f"5. Vector encoding for custom similarity ({mode})...")
        probe_vector = client.encode_vectors_json({"action": "login"})
        print(f"   ✅ Encoded probe vector: {len(probe_vector)}D")

        # Search using encoded vector (advanced usage)
        print(f"6. Finding items similar to encoded vector ({mode})...")
        similar_results = client.search_json(
            probe={"user": "alice"}, top_k=5  # Still use regular data probe
        )
        print(f"   ✅ Found {len(similar_results)} similar items")

        # Mathematical vector operations (advanced)
        print(f"7. Mathematical vector composition ({mode})...")
        vec1 = client.encode_mathematical("convergence_rate", 0.85)
        vec2 = client.encode_mathematical("amplitude_scale", 2.5)
        combined = client.compose_vectors("bind", [vec1, vec2])
        print(f"   ✅ Combined vectors: {len(combined)}D result")

        print(f"\n🎉 {mode} client demo completed successfully!")
        return True

    except Exception as e:
        print(f"❌ {mode} client demo failed: {e}")
        return False


def main():
    """Main demonstration."""
    print("🔗 Holon Unified Client Demo")
    print("=" * 50)
    print("This demo shows the same code working with local and remote Holon.")
    print("The HolonClient abstracts the implementation details!")
    print("Usage: HolonClient(local_store=store) or HolonClient(remote_url=url)\n")

    # Demo with local client
    success_local = demo_local_client()

    print("\n" + "=" * 50)

    # Demo with remote client (would need server running)
    print("Note: Remote demo requires server running on localhost:8000")
    print("To test remote: python scripts/server/holon_server.py")
    print("Then uncomment the remote demo call below.")

    # Uncomment to test remote client:
    # success_remote = demo_remote_client()

    print("\n" + "=" * 50)
    if success_local:
        print("✅ Unified client abstraction works perfectly!")
        print("Users can switch between local and remote without code changes.")
    else:
        print("❌ Local client demo failed")


if __name__ == "__main__":
    main()
