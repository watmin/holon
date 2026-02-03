#!/usr/bin/env python3
"""Large-scale stress test - GO NUTS edition with parallel encoding."""

import sys
import time
import random
import tracemalloc
from multiprocessing import Pool, cpu_count, shared_memory
import numpy as np

# Force unbuffered
print("Starting LARGE SCALE stress test...", flush=True)

sys.path.insert(0, str(__file__).rsplit('/', 4)[0])

from holon import CPUStore

print("Imports done", flush=True)

# Global encoder for worker processes (initialized once per worker)
_worker_encoder = None
_worker_dimensions = None


def _init_worker(dimensions, atom_vectors, position_vectors):
    """Initialize encoder in worker process with SHARED codebook."""
    global _worker_encoder, _worker_dimensions
    _worker_dimensions = dimensions
    store = CPUStore(dimensions=dimensions)
    _worker_encoder = store.encoder
    # CRITICAL: Copy the pre-populated vectors so all workers use same symbols
    _worker_encoder.vector_manager.atom_vectors = dict(atom_vectors)
    _worker_encoder.vector_manager.position_vectors = dict(position_vectors)


def _encode_batch(items_batch):
    """Encode a batch of items in worker process."""
    global _worker_encoder, _worker_dimensions
    vectors = np.zeros((len(items_batch), _worker_dimensions), dtype=np.int8)
    for i, item in enumerate(items_batch):
        vectors[i] = _worker_encoder.encode_data(item)
    return vectors


def _extract_all_symbols(items):
    """Extract all unique symbols from items to pre-populate codebook."""
    symbols = set()
    for item in items:
        for key, value in item.items():
            symbols.add(key)
            if isinstance(value, str):
                symbols.add(value)
            elif isinstance(value, (int, float)):
                symbols.add(str(value))
            elif isinstance(value, dict):
                for k, v in value.items():
                    symbols.add(k)
                    if isinstance(v, str):
                        symbols.add(v)
    return symbols


def parallel_encode(items, dimensions, n_workers=10, batch_size=1000, encoder=None):
    """Encode items in parallel using multiple processes with shared codebook."""
    print(f"  Using {n_workers} workers, batch_size={batch_size}", flush=True)

    # Step 1: Pre-populate codebook in main process
    if encoder is None:
        store = CPUStore(dimensions=dimensions)
        encoder = store.encoder

    print("  Pre-populating codebook...", flush=True)
    symbols = _extract_all_symbols(items)
    for sym in symbols:
        encoder.vector_manager.get_vector(sym)
    # Also need position vectors for list encoding
    for pos in range(100):  # Reasonable max position
        encoder.vector_manager.get_position_vector(pos)

    print(f"  Codebook has {len(encoder.vector_manager.atom_vectors)} symbols", flush=True)

    # Extract the codebooks to share with workers
    atom_vectors = encoder.vector_manager.atom_vectors
    position_vectors = encoder.vector_manager.position_vectors

    # Split into batches
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])

    print(f"  Created {len(batches)} batches", flush=True)

    # Process in parallel
    start = time.time()
    results = []

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(dimensions, atom_vectors, position_vectors),
    ) as pool:
        for i, batch_result in enumerate(pool.imap(_encode_batch, batches)):
            results.append(batch_result)
            if (i + 1) % 50 == 0:
                done = sum(len(r) for r in results)
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (len(items) - done) / rate
                print(
                    f"  Encoded {done:,}/{len(items):,} ({rate:.0f}/sec, {remaining:.0f}s left)",
                    flush=True,
                )

    # Concatenate results
    all_vectors = np.vstack(results)
    return all_vectors, encoder


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000, help="Total samples")
    parser.add_argument("--categories", type=int, default=100, help="Number of categories")
    parser.add_argument("--dimensions", type=int, default=4096, help="Vector dimensions")
    parser.add_argument("--workers", type=int, default=12, help="Parallel workers")
    parser.add_argument("--cardinality", type=int, default=1000, help="Values per field")
    args = parser.parse_args()

    # Start memory tracking
    tracemalloc.start()

    print("="*70, flush=True)
    print(f"LARGE SCALE Stress Test - {args.samples:,} samples, {args.categories} categories", flush=True)
    print("="*70, flush=True)

    # Estimate memory
    est_vectors = args.samples * args.dimensions / 1e9  # GB for int8
    est_float = args.samples * args.dimensions * 4 / 1e9  # GB for float32
    print(f"Estimated memory: {est_vectors:.1f}GB vectors + {est_float:.1f}GB float32 = {est_vectors + est_float:.1f}GB", flush=True)

    # Generate data - GO BIG
    items, labels = generate_large_data(
        n_samples=args.samples,
        n_categories=args.categories,
        cardinality=args.cardinality,
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

    # Configuration
    dimensions = args.dimensions
    n_workers = args.workers

    # Create store (for prototype building and classification)
    store = CPUStore(dimensions=dimensions)
    print(f"Store created ({store.dimensions}D)", flush=True)

    # Stop tracemalloc before heavy operations (causes 8x slowdown)
    _, peak_before_encode = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Encode training data IN PARALLEL
    print(f"\nEncoding training data (PARALLEL with {n_workers} workers)...", flush=True)
    start = time.time()
    train_vectors, shared_encoder = parallel_encode(
        X_train, dimensions, n_workers=n_workers, batch_size=1000, encoder=store.encoder
    )
    encode_time = time.time() - start
    print(f"Encoding took {encode_time:.1f}s ({len(X_train)/encode_time:,.0f}/sec)", flush=True)

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
    print(f"\nClassifying test data (parallel encode + batch matrix multiply)...", flush=True)
    classify_start = time.time()

    # Encode test data IN PARALLEL (reuse shared encoder with same codebook)
    test_vectors, _ = parallel_encode(
        X_test, dimensions, n_workers=n_workers, batch_size=1000, encoder=shared_encoder
    )

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

    # Estimate actual memory usage from numpy arrays
    train_mem = train_vectors.nbytes / 1024 / 1024
    test_mem = test_vectors.nbytes / 1024 / 1024
    proto_mem = proto_matrix.nbytes / 1024 / 1024
    total_mem = train_mem + test_mem + proto_mem

    print("\n" + "="*70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*70, flush=True)
    print(f"""
Samples:          {len(items):,}
Categories:       {len(unique_labels)}
Dimensions:       {store.dimensions}
Cardinality:      {args.cardinality} unique values per field
Workers:          {n_workers} (parallel encoding)

Performance:
  Encode time:    {encode_time:.1f}s ({len(X_train)/encode_time:,.0f}/sec)
  Prototype time: {proto_time:.1f}s
  Classify time:  {classify_time:.1f}s
  TOTAL:          {encode_time + proto_time + classify_time:.1f}s

Accuracy:         {accuracy:.1%}

Memory (vectors only):
  Train vectors:  {train_mem:,.0f} MB
  Test vectors:   {test_mem:,.0f} MB
  Prototypes:     {proto_mem:,.1f} MB
  Total:          {total_mem:,.0f} MB

STATUS:           {"PASSED" if accuracy > 0.3 else "NEEDS WORK"}
""", flush=True)


if __name__ == "__main__":
    main()
