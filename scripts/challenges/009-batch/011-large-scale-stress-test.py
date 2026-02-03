#!/usr/bin/env python3
"""Large-scale stress test - GO NUTS edition."""

import sys
import time
import random
import tracemalloc
import numpy as np

# Force unbuffered
print("Starting LARGE SCALE stress test...", flush=True)

sys.path.insert(0, str(__file__).rsplit('/', 4)[0])

from holon import CPUStore

print("Imports done", flush=True)


def generate_large_data(
    n_samples: int = 500_000,
    n_categories: int = 100,
    cardinality: int = 1000,
    n_fields: int = 10,
):
    """Generate large-scale test data with high cardinality."""
    print(f"Generating {n_samples:,} samples...", flush=True)
    print(f"  - {n_categories} categories", flush=True)
    print(f"  - {cardinality} unique values per field", flush=True)
    print(f"  - {n_fields} fields", flush=True)

    random.seed(42)

    items = []
    labels = []

    categories = [f"cat_{i:03d}" for i in range(n_categories)]

    # Pre-generate value pools
    value_pools = {
        f"field_{i}": [f"f{i}_v{j}" for j in range(cardinality)]
        for i in range(n_fields)
    }

    # Each category has preferences for the first 3 fields (signal)
    cat_prefs = {}
    for cat in categories:
        cat_prefs[cat] = {
            "field_0": random.choice(value_pools["field_0"]),
            "field_1": random.choice(value_pools["field_1"]),
            "field_2": random.choice(value_pools["field_2"]),
        }

    start = time.time()
    for i in range(n_samples):
        if i > 0 and i % 100000 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            print(f"  Generated {i:,} ({rate:.0f}/sec)", flush=True)

        cat = random.choice(categories)
        prefs = cat_prefs[cat]

        item = {}

        # Signal fields (correlated with category)
        for f in ["field_0", "field_1", "field_2"]:
            if random.random() < 0.7:
                item[f] = prefs[f]
            else:
                item[f] = random.choice(value_pools[f])

        # Noise fields (not correlated)
        for i_f in range(3, n_fields):
            item[f"field_{i_f}"] = random.choice(value_pools[f"field_{i_f}"])

        items.append(item)
        labels.append(cat)

    elapsed = time.time() - start
    print(f"  Generated {n_samples:,} in {elapsed:.1f}s", flush=True)

    return items, labels


def main():
    # Start memory tracking
    tracemalloc.start()

    print("="*70, flush=True)
    print("LARGE SCALE Stress Test - GO NUTS Edition", flush=True)
    print("="*70, flush=True)

    # Generate data - GO BIG
    items, labels = generate_large_data(
        n_samples=500_000,
        n_categories=100,
        cardinality=1000,
        n_fields=10,
    )
    print(f"Generated {len(items):,} items", flush=True)

    # Memory checkpoint
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory after data gen: {current/1024/1024:.0f} MB (peak: {peak/1024/1024:.0f} MB)", flush=True)

    # Split
    split_idx = int(0.8 * len(items))
    X_train, y_train = items[:split_idx], labels[:split_idx]
    X_test, y_test = items[split_idx:], labels[split_idx:]
    print(f"Split: {len(X_train):,} train, {len(X_test):,} test", flush=True)

    # Create store
    store = CPUStore(dimensions=8192)  # Larger dimensions for high cardinality
    print(f"Store created ({store.dimensions}D)", flush=True)

    # Encode training data
    print("\nEncoding training data...", flush=True)
    start = time.time()
    train_vectors = np.zeros((len(X_train), store.dimensions), dtype=np.int8)
    report_interval = max(10000, len(X_train) // 10)
    for i, item in enumerate(X_train):
        train_vectors[i] = store.encoder.encode_data(item)
        if (i + 1) % report_interval == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(X_train) - i - 1) / rate
            current, peak = tracemalloc.get_traced_memory()
            print(f"  Encoded {i+1:,}/{len(X_train):,} ({rate:.0f}/sec, {remaining:.0f}s left, {peak/1024/1024:.0f}MB)", flush=True)
    encode_time = time.time() - start
    print(f"Encoding took {encode_time:.1f}s ({len(X_train)/encode_time:.0f}/sec)", flush=True)

    # Build prototypes using efficient numpy
    print("\nBuilding prototypes...", flush=True)
    proto_start = time.time()

    # Get unique labels and their indices
    unique_labels = sorted(set(y_train))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Build prototypes as matrix multiplication
    prototypes = {}
    for label in unique_labels:
        # Get indices for this label
        indices = [i for i, l in enumerate(y_train) if l == label]
        if indices:
            subset = train_vectors[indices]
            mean = np.mean(subset, axis=0)
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
            prototypes[label] = proto

    proto_matrix = np.stack([prototypes[l] for l in unique_labels])
    proto_time = time.time() - proto_start
    print(f"Built {len(prototypes)} prototypes in {proto_time:.1f}s", flush=True)

    # Classify test data using matrix ops
    print("\nClassifying test data (batch matrix multiply)...", flush=True)
    classify_start = time.time()

    # Encode test data
    test_vectors = np.zeros((len(X_test), store.dimensions), dtype=np.int8)
    for i, item in enumerate(X_test):
        test_vectors[i] = store.encoder.encode_data(item)

    # Matrix multiply for all similarities at once
    similarities = np.dot(test_vectors.astype(np.float32), proto_matrix.T.astype(np.float32))
    pred_indices = np.argmax(similarities, axis=1)
    y_pred = [unique_labels[i] for i in pred_indices]

    classify_time = time.time() - classify_start

    # Accuracy
    correct = sum(1 for p, t in zip(y_pred, y_test) if p == t)
    accuracy = correct / len(y_test)

    print(f"Classification took {classify_time:.1f}s", flush=True)
    print(f"Accuracy: {accuracy:.1%}", flush=True)

    # Final memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "="*70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*70, flush=True)
    print(f"""
Samples:          {len(items):,}
Categories:       {len(unique_labels)}
Dimensions:       {store.dimensions}
Cardinality:      1000 unique values per field

Performance:
  Encode time:    {encode_time:.1f}s ({len(X_train)/encode_time:,.0f}/sec)
  Prototype time: {proto_time:.1f}s
  Classify time:  {classify_time:.1f}s
  TOTAL:          {encode_time + proto_time + classify_time:.1f}s

Accuracy:         {accuracy:.1%}
Peak Memory:      {peak/1024/1024:,.0f} MB

STATUS:           {"PASSED" if accuracy > 0.3 else "NEEDS WORK"} ✓
""", flush=True)


if __name__ == "__main__":
    main()
