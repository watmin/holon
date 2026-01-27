#!/usr/bin/env python3

"""
EDN usage example for Holon CPUStore.
Demonstrates handling of richer EDN data structures.
"""

from holon import CPUStore, HolonClient
from holon.atomizer import atomize, parse_data


def main():
    # Initialize the store and client
    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Sample EDN data with rich structures
    data1 = (
        '{:name "Alice", :age 30, :skills #{"clojure" "python" "ml"}, :status :active}'
    )
    data2 = '{:name "Bob", :age 25, :skills #{"python" "data"}, :status :inactive}'
    data3 = '{:name "Charlie", :age 35, :skills #{"clojure" "ai"}, :status :active}'

    # Parse and show atomization
    parsed1 = parse_data(data1, "edn")
    atoms1 = atomize(parsed1)
    print(f"Data 1 atoms: {atoms1}")

    # Insert EDN data
    id1 = client.insert(data1, data_type="edn")
    id2 = client.insert(data2, data_type="edn")
    id3 = client.insert(data3, data_type="edn")

    print(f"Inserted EDN data with IDs: {id1}, {id2}, {id3}")

    # Query with EDN probe - partial match
    probe = '{:skills #{"clojure"}, :status :active}'
    results = client.search(probe, data_type="edn", top_k=5, threshold=0.0)

    print("EDN Query results (partial match on skills and status):")
    for result in results:
        print(
            f"ID: {result['id']}, Score: {result['score']:.4f}, Data: {result['data']}"
        )

    # Query with exact name match
    probe2 = '{:name "Alice"}'
    results2 = client.search(probe2, data_type="edn", top_k=5, threshold=0.0)

    print("EDN Query results (exact name match):")
    for result in results2:
        print(
            f"ID: {result['id']}, Score: {result['score']:.4f}, Data: {result['data']}"
        )

    # Retrieve specific data
    retrieved = client.get(id1)
    print(f"Retrieved EDN data for {id1}: {retrieved}")


if __name__ == "__main__":
    main()
