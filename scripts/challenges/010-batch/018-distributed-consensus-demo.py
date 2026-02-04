#!/usr/bin/env python3
"""
Challenge 010-018: Distributed Consensus Demo

Demonstrates that multiple nodes can process the same data independently
and arrive at identical results - no synchronization required.

Scenario:
- Node A: Trains on shard 1 of historical data
- Node B: Trains on shard 2 of historical data
- Node C: Trains on shard 3 of historical data
- All nodes: Process the same stream, make same detection decisions

This proves the "deterministic AI" vision:
- No random initialization (vectors are hash-derived)
- No gradient descent (accumulator is pure addition)
- No synchronization (same inputs → same outputs)
"""

import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager
from holon.encoder import Encoder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


def generate_benign_data(n: int, seed: int) -> List[dict]:
    """Generate benign API requests."""
    import random
    random.seed(seed)

    templates = [
        {"method": "GET", "path": "/api/users", "headers": ["Content-Type"]},
        {"method": "GET", "path": "/api/users/{id}", "headers": ["Content-Type", "Authorization"]},
        {"method": "POST", "path": "/api/users", "headers": ["Content-Type"], "body": "user_create"},
        {"method": "GET", "path": "/api/orders", "headers": ["Content-Type"], "query": "page"},
        {"method": "POST", "path": "/api/orders", "headers": ["Content-Type"], "body": "order_create"},
    ]

    return [random.choice(templates).copy() for _ in range(n)]


def generate_malicious_data(n: int, seed: int) -> List[dict]:
    """Generate malicious requests."""
    import random
    random.seed(seed)

    templates = [
        {"method": "GET", "path": "/../../../etc/passwd", "headers": []},
        {"method": "GET", "path": "/api/users/' OR 1=1--", "headers": []},
        {"method": "TRACE", "path": "/api/debug", "headers": []},
        {"method": "GET", "path": "/.git/config", "headers": []},
        {"method": "POST", "path": "/api/admin", "headers": [], "body": "shell_exec"},
    ]

    return [random.choice(templates).copy() for _ in range(n)]


class DistributedNode:
    """
    A node that can train and detect independently.

    Key property: Two nodes with same global_seed produce
    identical vectors for the same atoms.
    """

    def __init__(self, node_id: str, global_seed: int = 42, dimensions: int = 4096):
        self.node_id = node_id
        self.vm = DeterministicVectorManager(dimensions=dimensions, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)
        self.accumulator = None
        self.observation_count = 0

    def train(self, records: List[dict]):
        """Train on a shard of data."""
        self.accumulator = self.encoder.create_accumulator()

        for record in records:
            vec = self.encoder.encode_data(record)
            self.accumulator = self.encoder.accumulate(self.accumulator, vec)
            self.observation_count += 1

    def get_proto(self) -> np.ndarray:
        """Get normalized prototype for detection."""
        return self.encoder.normalize_accumulator(self.accumulator)

    def detect(self, record: dict, threshold: float = 0.4) -> Tuple[float, bool]:
        """Detect if record is anomalous."""
        vec = self.encoder.encode_data(record)
        proto = self.get_proto()
        sim = cosine_similarity(vec, proto)
        flagged = sim < threshold
        return sim, flagged


def main():
    print("=" * 80)
    print("Challenge 010-018: Distributed Consensus Demo")
    print("=" * 80)
    print("""
Scenario: 3 nodes train on SHARDED data, then process SAME stream.

Key question: Do they agree on detections despite training on different shards?

Distributed consensus property:
- Same global_seed = same atom vectors
- Same atoms = same encoded vectors
- Same operations = same results
- NO synchronization required (except sharing global_seed)
""")

    # Configuration
    GLOBAL_SEED = 42
    DIMENSIONS = 4096
    SHARD_SIZE = 3000
    TEST_SIZE = 100

    # Generate training data shards
    print("\n--- Generating Data Shards ---")

    shard_1 = generate_benign_data(SHARD_SIZE, seed=100)
    shard_2 = generate_benign_data(SHARD_SIZE, seed=200)
    shard_3 = generate_benign_data(SHARD_SIZE, seed=300)

    print(f"Shard 1: {len(shard_1)} records (seed=100)")
    print(f"Shard 2: {len(shard_2)} records (seed=200)")
    print(f"Shard 3: {len(shard_3)} records (seed=300)")

    # Generate test stream (same for all nodes)
    test_benign = generate_benign_data(TEST_SIZE, seed=999)
    test_malicious = generate_malicious_data(TEST_SIZE, seed=888)
    test_stream = test_benign + test_malicious
    test_labels = [False] * len(test_benign) + [True] * len(test_malicious)

    print(f"\nTest stream: {len(test_stream)} records ({len(test_benign)} benign, {len(test_malicious)} malicious)")

    # Create nodes with SAME global seed
    print("\n--- Creating Distributed Nodes ---")

    node_a = DistributedNode("Node-A", global_seed=GLOBAL_SEED, dimensions=DIMENSIONS)
    node_b = DistributedNode("Node-B", global_seed=GLOBAL_SEED, dimensions=DIMENSIONS)
    node_c = DistributedNode("Node-C", global_seed=GLOBAL_SEED, dimensions=DIMENSIONS)

    print(f"Created 3 nodes with global_seed={GLOBAL_SEED}")

    # Verify atom consensus
    print("\n--- Verifying Atom Consensus ---")

    test_atoms = ["GET", "/api/users", "Content-Type", "method", "path"]
    all_match = True

    for atom in test_atoms:
        va = node_a.vm.get_vector(atom)
        vb = node_b.vm.get_vector(atom)
        vc = node_c.vm.get_vector(atom)

        match_ab = np.array_equal(va, vb)
        match_bc = np.array_equal(vb, vc)

        if not (match_ab and match_bc):
            all_match = False
            print(f"  ✗ MISMATCH: {atom}")

    if all_match:
        print(f"  ✓ All {len(test_atoms)} atoms identical across nodes")
    else:
        print("  ✗ CONSENSUS FAILED")
        return

    # Train each node on its shard
    print("\n--- Training Nodes on Different Shards ---")

    start = time.time()
    node_a.train(shard_1)
    time_a = time.time() - start
    print(f"Node-A trained on shard_1: {node_a.observation_count} records in {time_a:.2f}s")

    start = time.time()
    node_b.train(shard_2)
    time_b = time.time() - start
    print(f"Node-B trained on shard_2: {node_b.observation_count} records in {time_b:.2f}s")

    start = time.time()
    node_c.train(shard_3)
    time_c = time.time() - start
    print(f"Node-C trained on shard_3: {node_c.observation_count} records in {time_c:.2f}s")

    # Compare accumulators
    print("\n--- Comparing Accumulators ---")

    proto_a = node_a.get_proto()
    proto_b = node_b.get_proto()
    proto_c = node_c.get_proto()

    sim_ab = cosine_similarity(proto_a, proto_b)
    sim_bc = cosine_similarity(proto_b, proto_c)
    sim_ac = cosine_similarity(proto_a, proto_c)

    print(f"Prototype similarity:")
    print(f"  Node-A vs Node-B: {sim_ab:.4f}")
    print(f"  Node-B vs Node-C: {sim_bc:.4f}")
    print(f"  Node-A vs Node-C: {sim_ac:.4f}")

    if sim_ab > 0.9 and sim_bc > 0.9 and sim_ac > 0.9:
        print("\n  ✓ Prototypes are highly similar despite different training shards!")
        print("    (Same benign patterns → same learned representation)")

    # Process test stream on all nodes
    print("\n--- Processing Test Stream ---")

    threshold = 0.45
    print(f"Detection threshold: {threshold}")

    results_a = []
    results_b = []
    results_c = []

    for record in test_stream:
        sim_a, flag_a = node_a.detect(record, threshold)
        sim_b, flag_b = node_b.detect(record, threshold)
        sim_c, flag_c = node_c.detect(record, threshold)

        results_a.append((sim_a, flag_a))
        results_b.append((sim_b, flag_b))
        results_c.append((sim_c, flag_c))

    # Check decision agreement
    print("\n--- Detection Agreement ---")

    agree_all = 0
    agree_ab = 0
    agree_bc = 0
    agree_ac = 0

    for i in range(len(test_stream)):
        fa = results_a[i][1]
        fb = results_b[i][1]
        fc = results_c[i][1]

        if fa == fb == fc:
            agree_all += 1
        if fa == fb:
            agree_ab += 1
        if fb == fc:
            agree_bc += 1
        if fa == fc:
            agree_ac += 1

    print(f"All 3 nodes agree:  {agree_all}/{len(test_stream)} ({100*agree_all/len(test_stream):.1f}%)")
    print(f"Node-A == Node-B:   {agree_ab}/{len(test_stream)} ({100*agree_ab/len(test_stream):.1f}%)")
    print(f"Node-B == Node-C:   {agree_bc}/{len(test_stream)} ({100*agree_bc/len(test_stream):.1f}%)")
    print(f"Node-A == Node-C:   {agree_ac}/{len(test_stream)} ({100*agree_ac/len(test_stream):.1f}%)")

    # Similarity correlation
    sims_a = [r[0] for r in results_a]
    sims_b = [r[0] for r in results_b]
    sims_c = [r[0] for r in results_c]

    corr_ab = np.corrcoef(sims_a, sims_b)[0, 1]
    corr_bc = np.corrcoef(sims_b, sims_c)[0, 1]
    corr_ac = np.corrcoef(sims_a, sims_c)[0, 1]

    print(f"\nSimilarity correlation:")
    print(f"  Node-A vs Node-B: {corr_ab:.4f}")
    print(f"  Node-B vs Node-C: {corr_bc:.4f}")
    print(f"  Node-A vs Node-C: {corr_ac:.4f}")

    # Detection performance per node
    print("\n--- Detection Performance Per Node ---")

    for node_name, results in [("Node-A", results_a), ("Node-B", results_b), ("Node-C", results_c)]:
        tp = sum(1 for i, (_, flag) in enumerate(results) if test_labels[i] and flag)
        fp = sum(1 for i, (_, flag) in enumerate(results) if not test_labels[i] and flag)
        fn = sum(1 for i, (_, flag) in enumerate(results) if test_labels[i] and not flag)
        tn = sum(1 for i, (_, flag) in enumerate(results) if not test_labels[i] and not flag)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        print(f"{node_name}: TP={tp}, FP={fp}, FN={fn}, TN={tn} | P={precision:.1%} R={recall:.1%} F1={f1:.3f}")

    # Now test with MERGED accumulators
    print("\n--- Merging Accumulators (Simulated Reduce) ---")

    # In a real system, nodes would send their accumulators to a reducer
    # The reducer adds them together
    merged_accum = node_a.accumulator + node_b.accumulator + node_c.accumulator
    merged_proto = node_a.encoder.normalize_accumulator(merged_accum)

    print(f"Merged accumulator from {node_a.observation_count + node_b.observation_count + node_c.observation_count} total observations")

    # Test merged prototype
    merged_results = []
    for record in test_stream:
        vec = node_a.encoder.encode_data(record)
        sim = cosine_similarity(vec, merged_proto)
        flag = sim < threshold
        merged_results.append((sim, flag))

    tp = sum(1 for i, (_, flag) in enumerate(merged_results) if test_labels[i] and flag)
    fp = sum(1 for i, (_, flag) in enumerate(merged_results) if not test_labels[i] and flag)
    fn = sum(1 for i, (_, flag) in enumerate(merged_results) if test_labels[i] and not flag)
    tn = sum(1 for i, (_, flag) in enumerate(merged_results) if not test_labels[i] and not flag)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"Merged:  TP={tp}, FP={fp}, FN={fn}, TN={tn} | P={precision:.1%} R={recall:.1%} F1={f1:.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("DISTRIBUTED CONSENSUS SUMMARY")
    print("=" * 80)
    print(f"""
Setup:
- 3 nodes with same global_seed ({GLOBAL_SEED})
- Each trained on DIFFERENT shard ({SHARD_SIZE} records each)
- All processed SAME test stream ({len(test_stream)} records)

Consensus Results:
- Atom vectors:      IDENTICAL across all nodes
- Prototype similarity: >{min(sim_ab, sim_bc, sim_ac):.1%} (trained on different data!)
- Decision agreement: {100*agree_all/len(test_stream):.1f}%
- Similarity correlation: >{min(corr_ab, corr_bc, corr_ac):.4f}

Key Properties Demonstrated:
1. DETERMINISTIC: Same atom → same vector (no random init)
2. COMPOSABLE: Accumulators can be merged (map-reduce friendly)
3. CONVERGENT: Different shards → similar prototypes
4. CONSENSUS: Nodes agree on detections without coordination

This is "deterministic AI" - fully reproducible, distributable,
no gradients, no magic weights, no synchronization needed.
""")


if __name__ == "__main__":
    main()
