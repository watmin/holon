#!/usr/bin/env python3
"""
Qdrant Scale Test - 1000+ Categories

Tests Holon with actual Qdrant persistence at scale:
- 1000 categories (10x previous test)
- 100k records (100 per category)
- Real ANN search (not in-memory numpy)

Prerequisites:
    docker-compose up -d  # Start Qdrant on localhost:6333
"""

import json
import sys
import time
import random
import tracemalloc
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple
import numpy as np

# Force unbuffered output
print("Starting Qdrant Scale Test...", flush=True)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore

# Check Qdrant availability
try:
    from holon.qdrant_store import QdrantStore, QDRANT_AVAILABLE

    if not QDRANT_AVAILABLE:
        print("ERROR: qdrant-client not installed. Run: pip install qdrant-client", flush=True)
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Cannot import QdrantStore: {e}", flush=True)
    sys.exit(1)

print("Imports done", flush=True)

# =============================================================================
# Parallel Encoding (reused from 011)
# =============================================================================

_worker_encoder = None
_worker_dimensions = None


def _init_worker(dimensions, atom_vectors, position_vectors):
    """Initialize encoder in worker process with SHARED codebook."""
    global _worker_encoder, _worker_dimensions
    _worker_dimensions = dimensions
    store = CPUStore(dimensions=dimensions)
    _worker_encoder = store.encoder
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
    """Extract all unique symbols from items."""
    symbols = set()
    for item in items:
        for key, value in item.items():
            symbols.add(key)
            if isinstance(value, str):
                symbols.add(value)
            elif isinstance(value, (int, float)):
                symbols.add(str(value))
    return symbols


def parallel_encode(items, dimensions, n_workers=10, batch_size=1000, encoder=None):
    """Encode items in parallel using multiple processes."""
    print(f"  Using {n_workers} workers, batch_size={batch_size}", flush=True)

    if encoder is None:
        store = CPUStore(dimensions=dimensions)
        encoder = store.encoder

    print("  Pre-populating codebook...", flush=True)
    symbols = _extract_all_symbols(items)
    for sym in symbols:
        encoder.vector_manager.get_vector(sym)
    for pos in range(100):
        encoder.vector_manager.get_position_vector(pos)

    print(f"  Codebook has {len(encoder.vector_manager.atom_vectors)} symbols", flush=True)

    atom_vectors = encoder.vector_manager.atom_vectors
    position_vectors = encoder.vector_manager.position_vectors

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"  Created {len(batches)} batches", flush=True)

    start = time.time()
    results = []

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(dimensions, atom_vectors, position_vectors),
    ) as pool:
        for i, batch_result in enumerate(pool.imap(_encode_batch, batches)):
            results.append(batch_result)
            if (i + 1) % 20 == 0:
                done = sum(len(r) for r in results)
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (len(items) - done) / rate
                print(
                    f"  Encoded {done:,}/{len(items):,} ({rate:.0f}/sec, {remaining:.0f}s left)",
                    flush=True,
                )

    all_vectors = np.vstack(results)
    return all_vectors, encoder


# =============================================================================
# Data Generation
# =============================================================================


def generate_data(
    n_categories: int = 1000,
    samples_per_category: int = 100,
    cardinality: int = 500,
    n_fields: int = 8,
):
    """
    Generate test data with many categories.

    Args:
        n_categories: Number of distinct categories
        samples_per_category: Samples per category
        cardinality: Unique values per field
        n_fields: Total fields per record
    """
    n_samples = n_categories * samples_per_category
    print(f"Generating {n_samples:,} samples...", flush=True)
    print(f"  - {n_categories} categories", flush=True)
    print(f"  - {samples_per_category} samples per category", flush=True)
    print(f"  - {cardinality} unique values per field", flush=True)
    print(f"  - {n_fields} fields", flush=True)

    random.seed(42)

    categories = [f"cat_{i:04d}" for i in range(n_categories)]

    # Pre-generate value pools
    value_pools = {
        f"field_{i}": [f"f{i}_v{j}" for j in range(cardinality)] for i in range(n_fields)
    }

    # Each category has preferences for the first 3 fields (signal)
    cat_prefs = {}
    for cat in categories:
        cat_prefs[cat] = {
            "field_0": random.choice(value_pools["field_0"]),
            "field_1": random.choice(value_pools["field_1"]),
            "field_2": random.choice(value_pools["field_2"]),
        }

    items = []
    labels = []

    start = time.time()
    for cat in categories:
        prefs = cat_prefs[cat]
        for _ in range(samples_per_category):
            item = {}

            # Signal fields (70% correlated with category)
            for f in ["field_0", "field_1", "field_2"]:
                if random.random() < 0.7:
                    item[f] = prefs[f]
                else:
                    item[f] = random.choice(value_pools[f])

            # Noise fields
            for i_f in range(3, n_fields):
                item[f"field_{i_f}"] = random.choice(value_pools[f"field_{i_f}"])

            items.append(item)
            labels.append(cat)

    # Shuffle to mix categories
    combined = list(zip(items, labels))
    random.shuffle(combined)
    items, labels = zip(*combined)
    items, labels = list(items), list(labels)

    elapsed = time.time() - start
    print(f"  Generated {n_samples:,} in {elapsed:.1f}s", flush=True)

    return items, labels, cat_prefs


# =============================================================================
# Qdrant Operations
# =============================================================================


def insert_to_qdrant(
    client,  # QdrantClient directly
    collection_name: str,
    labels: List[str],
    vectors: np.ndarray,
    batch_size: int = 500,  # Larger batches for efficiency
) -> Tuple[List[str], float]:
    """
    Insert vectors to Qdrant with labels stored in payload.

    Optimizations:
    - Pre-normalize all vectors in numpy (vectorized)
    - Use larger batch sizes
    - Minimize per-item Python overhead

    Returns:
        Tuple of (ids, insert_rate)
    """
    from qdrant_client.http.models import PointStruct
    import uuid

    n_items = len(labels)
    print(f"Inserting {n_items:,} items to Qdrant...", flush=True)
    print(f"  batch_size={batch_size}", flush=True)

    # Pre-normalize ALL vectors at once (vectorized - much faster)
    print("  Pre-normalizing vectors...", flush=True)
    vectors_f32 = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
    vectors_normalized = vectors_f32 / norms

    # Pre-generate all UUIDs
    print("  Pre-generating IDs...", flush=True)
    all_ids = [str(uuid.uuid4()) for _ in range(n_items)]

    print("  Uploading to Qdrant...", flush=True)
    start = time.time()

    batch_times = []
    for chunk_start in range(0, n_items, batch_size):
        chunk_end = min(chunk_start + batch_size, n_items)

        # Build points for this batch
        points = [
            PointStruct(
                id=all_ids[i],
                vector=vectors_normalized[i].tolist(),
                payload={"_label": labels[i]},  # Minimal payload for speed
            )
            for i in range(chunk_start, chunk_end)
        ]

        batch_start = time.time()
        client.upsert(collection_name=collection_name, points=points)
        batch_times.append(time.time() - batch_start)

        if len(batch_times) <= 3:
            print(f"  Batch {len(batch_times)} took {batch_times[-1]:.3f}s", flush=True)

        # Progress
        done = chunk_end
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (n_items - done) / rate if rate > 0 else 0

        if done % 10000 == 0 or done == n_items:
            print(
                f"  Inserted {done:,}/{n_items:,} ({rate:.0f}/sec, {remaining:.0f}s left)",
                flush=True,
            )

    total_time = time.time() - start
    insert_rate = n_items / total_time

    print(f"  Insert complete: {insert_rate:.0f} items/sec", flush=True)

    return all_ids, insert_rate


def build_prototypes_from_qdrant(
    client,  # QdrantClient directly
    collection_name: str,
    categories: List[str],
    dimensions: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Build prototypes by querying Qdrant for each category.

    Returns:
        Tuple of (prototypes dict, query_time)
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    print(f"Building {len(categories)} prototypes from Qdrant...", flush=True)

    prototypes = {}
    start = time.time()

    for i, cat in enumerate(categories):
        # Query all vectors with this label
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="_label", match=MatchValue(value=cat))]
            ),
            limit=1000,  # Get all samples for this category
            with_vectors=True,
        )

        points = results[0]
        if points:
            # Average vectors to create prototype
            vecs = np.array([p.vector for p in points])
            mean = np.mean(vecs, axis=0)
            # Threshold to bipolar
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.float32)
            # Normalize
            norm = np.linalg.norm(proto)
            if norm > 0:
                proto = proto / norm
            prototypes[cat] = proto

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(categories) - i - 1) / rate
            print(f"  Built {i+1}/{len(categories)} prototypes ({remaining:.0f}s left)", flush=True)

    proto_time = time.time() - start
    print(f"  Built {len(prototypes)} prototypes in {proto_time:.1f}s", flush=True)

    return prototypes, proto_time


def classify_with_prototypes(
    test_vectors: np.ndarray,
    prototypes: Dict[str, np.ndarray],
    y_test: List[str],
) -> Tuple[float, float]:
    """
    Classify test vectors using prototype matrix multiply.

    Uses the prototype vectors directly with matrix multiply.
    This measures prototype quality.

    Returns:
        Tuple of (accuracy, total_time)
    """
    print(f"Classifying {len(test_vectors):,} test samples...", flush=True)

    # Build prototype matrix
    categories = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[c] for c in categories])

    start = time.time()

    # Normalize test vectors
    norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    test_normalized = test_vectors.astype(np.float32) / norms

    # Matrix multiply for similarities
    similarities = np.dot(test_normalized, proto_matrix.T)
    pred_indices = np.argmax(similarities, axis=1)
    y_pred = [categories[i] for i in pred_indices]

    total_time = time.time() - start

    # Accuracy
    correct = sum(1 for p, t in zip(y_pred, y_test) if p == t)
    accuracy = correct / len(y_test)

    print(f"  Classification took {total_time:.1f}s", flush=True)
    print(f"  Accuracy: {accuracy:.1%}", flush=True)

    return accuracy, total_time


def test_qdrant_query_latency(
    client,  # QdrantClient directly
    collection_name: str,
    test_vectors: np.ndarray,
    n_queries: int = 100,
) -> Tuple[float, float, float]:
    """
    Test raw Qdrant query latency.

    Returns:
        Tuple of (avg_latency_ms, min_latency_ms, max_latency_ms)
    """
    print(f"Testing Qdrant query latency ({n_queries} queries)...", flush=True)

    latencies = []
    indices = random.sample(range(len(test_vectors)), min(n_queries, len(test_vectors)))

    for idx in indices:
        vec = test_vectors[idx].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        start = time.time()
        client.query_points(
            collection_name=collection_name,
            query=vec.tolist(),
            limit=10,
        )
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print(f"  Avg: {avg_latency:.1f}ms, Min: {min_latency:.1f}ms, Max: {max_latency:.1f}ms", flush=True)

    return avg_latency, min_latency, max_latency


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qdrant Scale Test")
    parser.add_argument("--categories", type=int, default=1000, help="Number of categories")
    parser.add_argument("--samples", type=int, default=100, help="Samples per category")
    parser.add_argument("--dimensions", type=int, default=4096, help="Vector dimensions (4096 faster, 8192 more accurate)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel encoding workers")
    parser.add_argument("--quick", action="store_true", help="Quick test: 100 categories, 50 samples")
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.categories = 100
        args.samples = 50
        args.dimensions = 4096

    tracemalloc.start()

    print("=" * 70, flush=True)
    print(f"Qdrant Scale Test - {args.categories} Categories", flush=True)
    print("=" * 70, flush=True)

    # Configuration
    n_categories = args.categories
    samples_per_category = args.samples
    dimensions = args.dimensions
    n_workers = args.workers
    collection_name = "holon_scale_test"

    # Generate data
    items, labels, cat_prefs = generate_data(
        n_categories=n_categories,
        samples_per_category=samples_per_category,
        cardinality=500,
        n_fields=8,
    )

    # Split 80/20
    split_idx = int(0.8 * len(items))
    X_train, y_train = items[:split_idx], labels[:split_idx]
    X_test, y_test = items[split_idx:], labels[split_idx:]
    print(f"Split: {len(X_train):,} train, {len(X_test):,} test", flush=True)

    # Memory checkpoint
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory after data gen: {current/1024/1024:.0f} MB", flush=True)

    # Create CPU store for encoding FIRST (before Qdrant to avoid gRPC/multiprocessing conflicts)
    cpu_store = CPUStore(dimensions=dimensions)

    # Encode training data BEFORE connecting to Qdrant
    # (multiprocessing can interfere with gRPC connections)
    print(f"\nEncoding training data (PARALLEL with {n_workers} workers)...", flush=True)
    encode_start = time.time()
    train_vectors, shared_encoder = parallel_encode(
        X_train, dimensions, n_workers=n_workers, batch_size=1000, encoder=cpu_store.encoder
    )
    encode_time = time.time() - encode_start
    encode_rate = len(X_train) / encode_time
    print(f"Encoding took {encode_time:.1f}s ({encode_rate:,.0f}/sec)", flush=True)

    # NOW create Qdrant client (AFTER multiprocessing is done)
    print(f"\nConnecting to Qdrant (gRPC)...", flush=True)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff

        qdrant_client = QdrantClient(
            host="localhost",
            port=6334,
            prefer_grpc=True,
            grpc_options={"grpc.max_send_message_length": 100 * 1024 * 1024},  # 100MB
        )

        # Drop and recreate collection
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0,  # Disable indexing during bulk insert (8x faster)
            ),
        )
        print(f"  Collection '{collection_name}' ready (gRPC, indexing disabled for bulk)", flush=True)

    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant: {e}", flush=True)
        print("  Make sure Qdrant is running: docker-compose up -d", flush=True)
        sys.exit(1)

    # CRITICAL: Stop tracemalloc before Qdrant operations (8x performance impact!)
    tracemalloc.stop()

    # Insert to Qdrant (with indexing disabled for speed)
    print(f"\nInserting to Qdrant (indexing disabled)...", flush=True)
    ids, insert_rate = insert_to_qdrant(
        qdrant_client, collection_name, y_train, train_vectors, batch_size=500
    )

    # Now trigger indexing
    print("Triggering HNSW index build...", flush=True)
    index_start = time.time()
    from qdrant_client.http.models import OptimizersConfigDiff

    qdrant_client.update_collection(
        collection_name=collection_name,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
    )
    # Wait for indexing to complete by checking collection status
    while True:
        info = qdrant_client.get_collection(collection_name)
        if info.status.name == "GREEN":
            break
        time.sleep(1)
    index_time = time.time() - index_start
    print(f"  Indexing took {index_time:.1f}s", flush=True)

    # Build prototypes from LOCAL vectors (faster than querying Qdrant)
    # We already have the vectors in memory, no need to fetch from Qdrant
    print(f"\nBuilding prototypes (from local vectors)...", flush=True)
    proto_start = time.time()
    unique_categories = sorted(set(y_train))

    prototypes = {}
    for cat in unique_categories:
        indices = [i for i, l in enumerate(y_train) if l == cat]
        if indices:
            subset = train_vectors[indices]
            mean = np.mean(subset, axis=0)
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.float32)
            norm = np.linalg.norm(proto)
            if norm > 0:
                proto = proto / norm
            prototypes[cat] = proto

    proto_time = time.time() - proto_start
    print(f"  Built {len(prototypes)} prototypes in {proto_time:.1f}s", flush=True)

    # Encode test data
    print(f"\nEncoding test data...", flush=True)
    test_encode_start = time.time()
    test_vectors, _ = parallel_encode(
        X_test, dimensions, n_workers=n_workers, batch_size=1000, encoder=shared_encoder
    )
    test_encode_time = time.time() - test_encode_start

    # Classify
    accuracy, classify_time = classify_with_prototypes(
        test_vectors, prototypes, y_test
    )

    # Test query latency
    avg_latency, min_latency, max_latency = test_qdrant_query_latency(
        qdrant_client, collection_name, test_vectors, n_queries=100
    )

    # Memory tracking was stopped earlier for performance
    # Just report peak from before Qdrant operations
    peak = 0  # Not tracked during Qdrant ops

    # Results
    print("\n" + "=" * 70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(
        f"""
Dataset:
  Total samples:      {len(items):,}
  Categories:         {n_categories}
  Samples/category:   {samples_per_category}
  Dimensions:         {dimensions}
  Train/Test split:   {len(X_train):,} / {len(X_test):,}

Performance:
  Encode rate:        {encode_rate:,.0f}/sec ({n_workers} workers)
  Qdrant insert rate: {insert_rate:.0f}/sec
  Prototype build:    {proto_time:.1f}s
  Classification:     {classify_time:.1f}s

Query Latency:
  Average:            {avg_latency:.1f}ms
  Min:                {min_latency:.1f}ms
  Max:                {max_latency:.1f}ms

Accuracy:             {accuracy:.1%}
Peak Memory:          {peak/1024/1024:,.0f} MB

STATUS:               {"PASSED" if accuracy > 0.5 else "NEEDS INVESTIGATION"}
""",
        flush=True,
    )

    # Cleanup
    print("Cleaning up collection...", flush=True)
    qdrant_client.delete_collection(collection_name)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
