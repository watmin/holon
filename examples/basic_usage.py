#!/usr/bin/env python3

"""
Basic usage example for Holon CPUStore.
"""

from holon import CPUStore


def main():
    # Initialize the store
    store = CPUStore(dimensions=16000)

    # Insert some data
    data1 = '{"name": "Alice", "age": 30, "city": "New York"}'
    data2 = '{"name": "Bob", "age": 25, "city": "San Francisco"}'
    data3 = '{"name": "Charlie", "age": 35, "city": "New York"}'

    id1 = store.insert(data1, "json")
    id2 = store.insert(data2, "json")
    id3 = store.insert(data3, "json")

    print(f"Inserted data with IDs: {id1}, {id2}, {id3}")

    # Query for similar data
    probe = '{"name": "Alice", "city": "New York"}'
    results = store.query(probe, "json", top_k=5, threshold=0.0)

    print("Query results:")
    for data_id, score, data in results:
        print(f"ID: {data_id}, Score: {score:.4f}, Data: {data}")

    # Retrieve specific data
    retrieved = store.get(id1)
    print(f"Retrieved data for {id1}: {retrieved}")


if __name__ == "__main__":
    main()
