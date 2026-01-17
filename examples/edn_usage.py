#!/usr/bin/env python3

"""
EDN usage example for Holon CPUStore.
Demonstrates handling of richer EDN data structures.
"""

from holon import CPUStore
from holon.atomizer import atomize, parse_data

def main():
    # Initialize the store
    store = CPUStore(dimensions=16000)

    # Sample EDN data with rich structures
    data1 = '{:name "Alice", :age 30, :skills #{"clojure" "python" "ml"}, :status :active}'
    data2 = '{:name "Bob", :age 25, :skills #{"python" "data"}, :status :inactive}'
    data3 = '{:name "Charlie", :age 35, :skills #{"clojure" "ai"}, :status :active}'

    # Parse and show atomization
    parsed1 = parse_data(data1, 'edn')
    atoms1 = atomize(parsed1)
    print(f"Data 1 atoms: {atoms1}")

    # Insert EDN data
    id1 = store.insert(data1, 'edn')
    id2 = store.insert(data2, 'edn')
    id3 = store.insert(data3, 'edn')

    print(f"Inserted EDN data with IDs: {id1}, {id2}, {id3}")

    # Query with EDN probe - partial match
    probe = '{:skills #{"clojure"}, :status :active}'
    results = store.query(probe, 'edn', top_k=5, threshold=0.0)

    print("EDN Query results (partial match on skills and status):")
    for data_id, score, data in results:
        print(f"ID: {data_id}, Score: {score:.4f}, Data: {data}")

    # Query with exact name match
    probe2 = '{:name "Alice"}'
    results2 = store.query(probe2, 'edn', top_k=5, threshold=0.0)

    print("EDN Query results (exact name match):")
    for data_id, score, data in results2:
        print(f"ID: {data_id}, Score: {score:.4f}, Data: {data}")

    # Retrieve specific data
    retrieved = store.get(id1)
    print(f"Retrieved EDN data for {id1}: {retrieved}")

if __name__ == "__main__":
    main()