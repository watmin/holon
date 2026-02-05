#!/usr/bin/env python3
"""
Challenge 010-001: Realistic Data Stress Test

Combines:
1. RealisticDataGenerator - messy, production-like data
2. DeterministicVectorManager - order-independent consensus
3. Holon's full primitive set - push expressiveness

Goals:
- Test with heterogeneous schemas (not uniform synthetic data)
- Stress atomizer with high cardinality (50k+ unique atoms)
- Verify consensus across simulated distributed nodes
- Use more Holon primitives than previous challenges

This is the "proof of concept" for deterministic AI on complex data.
"""

import sys
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

# Local imports from this batch
from holon import DeterministicVectorManager
from realistic_data_generator import RealisticDataGenerator

from holon import CPUStore
from holon.encoder import Encoder, ListEncodeMode


class DeterministicEncoder(Encoder):
    """Encoder using DeterministicVectorManager for consensus."""

    def __init__(self, dimensions: int = 4096, global_seed: int = 42):
        vm = DeterministicVectorManager(dimensions=dimensions, global_seed=global_seed)
        super().__init__(vector_manager=vm)


def encode_record(record: dict, encoder: Encoder) -> np.ndarray:
    """Encode a single record using Holon's structural encoding."""
    return encoder.encode_data(record)


def build_prototypes(
    vectors: np.ndarray,
    labels: List[str],
    encoder: Encoder,
) -> Dict[str, np.ndarray]:
    """Build prototype vectors for each category using Holon's prototype() primitive."""
    unique_labels = sorted(set(labels))
    prototypes = {}

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        if indices:
            subset = [vectors[i] for i in indices]
            # Use Holon's prototype primitive (majority voting)
            prototypes[label] = encoder.prototype(subset, threshold=0.5)

    return prototypes


def classify_batch(
    vectors: np.ndarray,
    prototypes: Dict[str, np.ndarray],
) -> List[str]:
    """Classify vectors using cosine similarity to prototypes."""
    labels = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[l] for l in labels])

    # Batch matrix multiply for efficiency
    similarities = np.dot(vectors.astype(np.float32), proto_matrix.T.astype(np.float32))
    pred_indices = np.argmax(similarities, axis=1)

    return [labels[i] for i in pred_indices]


def demo_consensus(encoder1: Encoder, encoder2: Encoder, records: List[dict]) -> bool:
    """Verify that two encoders produce identical vectors for same records."""
    print("\n--- Distributed Consensus Test ---")

    sample = records[:100]
    matches = 0

    for record in sample:
        vec1 = encode_record(record, encoder1)
        vec2 = encode_record(record, encoder2)
        if np.array_equal(vec1, vec2):
            matches += 1

    print(f"Records tested: {len(sample)}")
    print(f"Exact matches: {matches}")
    print(f"Consensus: {'ACHIEVED' if matches == len(sample) else 'FAILED'}")

    return matches == len(sample)


def demo_primitives(encoder: Encoder, records: List[dict], labels: List[str]):
    """Demonstrate underutilized Holon primitives."""
    print("\n--- Holon Primitives Demo ---")

    # Encode some records
    sample_records = records[:100]
    sample_labels = labels[:100]
    vectors = [encode_record(r, encoder) for r in sample_records]

    # 1. PROTOTYPE - find common pattern across category
    print("\n1. prototype() - Extract common pattern")
    billing_vecs = [v for v, l in zip(vectors, sample_labels) if "api_request" in l]
    if billing_vecs:
        proto = encoder.prototype(billing_vecs[:20], threshold=0.5)
        print(f"   api_request prototype from {len(billing_vecs[:20])} examples")
        print(f"   Non-zero elements: {np.sum(proto != 0)} / {len(proto)}")

    # 2. DIFFERENCE - what changed between records
    print("\n2. difference() - Detect changes")
    if len(vectors) >= 2:
        delta = encoder.difference(vectors[0], vectors[1])
        print(f"   Change vector between records 0 and 1")
        print(f"   Positive (added): {np.sum(delta > 0)}")
        print(f"   Negative (removed): {np.sum(delta < 0)}")

    # 3. NEGATE - remove component
    print("\n3. negate() - Remove component from superposition")
    if len(vectors) >= 3:
        bundle_abc = encoder.bundle(vectors[:3])
        negated = encoder.negate(bundle_abc, vectors[0])

        # Check similarity before and after
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        sim_before = cosine(bundle_abc, vectors[0])
        sim_after = cosine(negated, vectors[0])
        print(f"   Similarity to removed component before: {sim_before:.4f}")
        print(f"   Similarity to removed component after:  {sim_after:.4f}")
        print(f"   Reduction: {(sim_before - sim_after) / abs(sim_before):.1%}")

    # 4. AMPLIFY - strengthen component
    print("\n4. amplify() - Strengthen component")
    if len(vectors) >= 3:
        bundle_abc = encoder.bundle(vectors[:3])
        amplified = encoder.amplify(bundle_abc, vectors[0], strength=2.0)

        sim_before = cosine(bundle_abc, vectors[0])
        sim_after = cosine(amplified, vectors[0])
        print(f"   Similarity before amplify: {sim_before:.4f}")
        print(f"   Similarity after amplify:  {sim_after:.4f}")
        print(f"   Boost: {(sim_after - sim_before) / abs(sim_before):.1%}")

    # 5. RESONANCE - extract matching parts
    print("\n5. resonance() - Extract matching components")
    if len(vectors) >= 2:
        resonant = encoder.resonance(vectors[0], vectors[1])
        non_zero = np.sum(resonant != 0)
        print(f"   Resonant dimensions: {non_zero} / {len(resonant)}")
        print(f"   These are the dimensions where both records agree")

    # 6. BLEND - interpolate between records
    print("\n6. blend() - Interpolate between records")
    if len(vectors) >= 2:
        blended = encoder.blend(vectors[0], vectors[1], alpha=0.5)
        sim_to_first = cosine(blended, vectors[0])
        sim_to_second = cosine(blended, vectors[1])
        print(f"   50/50 blend similarity to record 0: {sim_to_first:.4f}")
        print(f"   50/50 blend similarity to record 1: {sim_to_second:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000, help="Total samples")
    parser.add_argument("--dimensions", type=int, default=4096, help="Vector dimensions")
    parser.add_argument("--cardinality", type=int, default=50000, help="Value cardinality")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Challenge 010: Realistic Data Stress Test")
    print("=" * 70)
    print(f"Samples: {args.samples:,}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Cardinality target: {args.cardinality:,}")
    print(f"Seed: {args.seed}")

    # Generate realistic data
    print("\n--- Generating Realistic Data ---")
    gen = RealisticDataGenerator(
        seed=args.seed,
        cardinality=args.cardinality,
        missing_field_rate=0.20,
        extra_field_rate=0.10,
        type_coercion_rate=0.05,
    )

    start = time.time()
    records, schemas, categories = gen.generate_dataset(args.samples)
    gen_time = time.time() - start

    stats = gen.get_stats()
    print(f"Generated {len(records):,} records in {gen_time:.1f}s")
    print(f"Unique atoms observed (in value pools): {stats['unique_atoms']:,}")
    print(f"Schemas: {len(stats['schemas'])}")
    print(f"Unique categories: {len(set(categories))}")

    # Schema distribution
    schema_counts = Counter(schemas)
    print("\nSchema distribution:")
    for s, c in schema_counts.most_common():
        print(f"  {s}: {c:,} ({100*c/len(records):.1f}%)")

    # Split train/test
    split_idx = int(0.8 * len(records))
    train_records, train_labels = records[:split_idx], categories[:split_idx]
    test_records, test_labels = records[split_idx:], categories[split_idx:]
    print(f"\nTrain: {len(train_records):,}, Test: {len(test_records):,}")

    # Create deterministic encoders
    print("\n--- Creating Deterministic Encoders ---")
    encoder1 = DeterministicEncoder(dimensions=args.dimensions, global_seed=args.seed)
    encoder2 = DeterministicEncoder(dimensions=args.dimensions, global_seed=args.seed)

    # Verify consensus
    consensus = demo_consensus(encoder1, encoder2, records[:100])

    # Demo primitives
    demo_primitives(encoder1, records[:200], categories[:200])

    # Encode training data
    print("\n--- Encoding Training Data ---")
    start = time.time()
    train_vectors = np.array([encode_record(r, encoder1) for r in train_records])
    encode_time = time.time() - start
    print(f"Encoded {len(train_vectors):,} vectors in {encode_time:.1f}s ({len(train_vectors)/encode_time:,.0f}/sec)")

    # Check atomizer cardinality
    vm_stats = encoder1.vector_manager.get_stats()
    print(f"Atoms in codebook: {vm_stats['atoms_cached']:,}")

    # Build prototypes
    print("\n--- Building Prototypes ---")
    start = time.time()
    prototypes = build_prototypes(train_vectors, train_labels, encoder1)
    proto_time = time.time() - start
    print(f"Built {len(prototypes)} prototypes in {proto_time:.2f}s")

    # Encode and classify test data
    print("\n--- Classifying Test Data ---")
    start = time.time()
    test_vectors = np.array([encode_record(r, encoder1) for r in test_records])
    predictions = classify_batch(test_vectors, prototypes)
    classify_time = time.time() - start

    # Accuracy
    correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
    accuracy = correct / len(test_labels)

    print(f"Classification time: {classify_time:.1f}s")
    print(f"Accuracy: {accuracy:.1%} ({correct:,}/{len(test_labels):,})")

    # Per-schema accuracy
    print("\nPer-schema accuracy:")
    for schema in set(schemas):
        schema_mask = [s == schema for s in schemas[split_idx:]]
        schema_preds = [p for p, m in zip(predictions, schema_mask) if m]
        schema_true = [t for t, m in zip(test_labels, schema_mask) if m]
        if schema_preds:
            schema_acc = sum(1 for p, t in zip(schema_preds, schema_true) if p == t) / len(schema_preds)
            print(f"  {schema}: {schema_acc:.1%} ({len(schema_preds):,} samples)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Records:          {len(records):,}
Schemas:          {len(stats['schemas'])}
Categories:       {len(set(categories))}
Atoms:            {vm_stats['atoms_cached']:,}
Dimensions:       {args.dimensions}

Performance:
  Data gen:       {gen_time:.1f}s
  Encoding:       {encode_time:.1f}s ({len(train_vectors)/encode_time:,.0f}/sec)
  Prototypes:     {proto_time:.2f}s
  Classification: {classify_time:.1f}s

Accuracy:         {accuracy:.1%}
Consensus:        {'ACHIEVED' if consensus else 'FAILED'}

STATUS:           {'SUCCESS' if accuracy > 0.5 and consensus else 'NEEDS WORK'}
""")


if __name__ == "__main__":
    main()
