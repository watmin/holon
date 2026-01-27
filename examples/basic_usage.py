#!/usr/bin/env python3

"""
Basic usage example for Holon using the unified client.

This example works the same whether using a local store or remote API.
"""

from holon import CPUStore, HolonClient


def main():
    # Initialize the store (you could also use a remote URL with HolonClient(remote_url="..."))
    store = CPUStore(dimensions=16000)

    # Create unified client - same interface for local or remote
    client = HolonClient(local_store=store)

    # Insert some data (convenience method for JSON)
    data1 = {"name": "Alice", "age": 30, "city": "New York"}
    data2 = {"name": "Bob", "age": 25, "city": "San Francisco"}
    data3 = {"name": "Charlie", "age": 35, "city": "New York"}

    id1 = client.insert_json(data1)
    id2 = client.insert_json(data2)
    id3 = client.insert_json(data3)

    print(f"Inserted data with IDs: {id1}, {id2}, {id3}")

    # Query for similar data (convenience method for JSON)
    probe = {"name": "Alice", "city": "New York"}
    results = client.search_json(probe, top_k=5, threshold=0.0)

    print("Query results:")
    for result in results:
        print(
            f"ID: {result['id']}, Score: {result['score']:.4f}, Data: {result['data']}"
        )

    # Retrieve specific data
    retrieved = client.get(id1)
    print(f"Retrieved data for {id1}: {retrieved}")


if __name__ == "__main__":
    main()
