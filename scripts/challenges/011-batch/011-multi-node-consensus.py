#!/usr/bin/env python3
"""
Challenge 011-011: Multi-Node Consensus

Deploy detection across multiple sensors sharing the same priors.

Key Properties:
1. IDENTICAL PRIORS: All nodes use same global_seed → same base vectors
2. INDEPENDENT RECENT: Each node maintains local recent knowledge
3. MERGEABLE ACCUMULATORS: Nodes can share/merge their accumulators
4. CONSISTENT DETECTION: Same packet → same classification (modulo recent)

Architecture:

  [Prior Knowledge] ← shared, frozen
       ↓
  ┌────┴────┬────┴────┐
  │         │         │
Node A    Node B    Node C   ← each with own recent knowledge
  │         │         │
  └────┬────┴────┬────┘
       ↓
  [Merged View] ← optional aggregation
"""

import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from scapy.all import IP, TCP, UDP, ICMP, Raw, Packet

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager

# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42  # CRITICAL: All nodes must use same seed!
DIMENSIONS = 4096


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# SENSOR NODE
# =============================================================================

@dataclass
class DetectionResult:
    """Detection result from a node."""
    is_anomalous: bool
    anomaly_score: float
    perspective_scores: Dict[str, float]


class SensorNode:
    """
    A single sensor node in the distributed detection system.

    All nodes share the same VectorManager (via same global_seed)
    but maintain independent recent knowledge.
    """

    def __init__(self, node_id: str, global_seed: int = GLOBAL_SEED):
        self.node_id = node_id
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=global_seed)

        # Shared prior (frozen)
        self.prior_vectors: Dict[str, np.ndarray] = {}

        # Local recent knowledge (adaptive)
        self.recent_vectors: Dict[str, np.ndarray] = {}
        self.recent_counts: Dict[str, int] = {}
        self.decay_factor = 0.995

        # Stats
        self.packets_processed = 0
        self.anomalies_detected = 0

    def load_prior(self, prior_vectors: Dict[str, np.ndarray]):
        """Load shared prior knowledge."""
        # IMPORTANT: We copy the vectors but they should be IDENTICAL
        # across nodes because they're generated from the same VectorManager
        self.prior_vectors = {k: v.copy() for k, v in prior_vectors.items()}

    def process_packet(self, pkt: Packet, threshold: float = 0.5) -> DetectionResult:
        """Process a packet and detect anomalies."""
        self.packets_processed += 1

        # Parse and encode
        vectors = self._encode_packet(pkt)

        # Compare to prior and recent
        scores = {}
        for perspective, vec in vectors.items():
            prior_vec = self.prior_vectors.get(perspective, np.zeros(DIMENSIONS))
            prior_sim = cosine_similarity(vec, prior_vec)

            recent_vec = self.recent_vectors.get(perspective, np.zeros(DIMENSIONS))
            recent_sim = cosine_similarity(vec, recent_vec)

            # Weighted blend
            scores[perspective] = 0.6 * prior_sim + 0.4 * recent_sim

        # Compute anomaly score
        avg_score = np.mean(list(scores.values())) if scores else 0.0
        anomaly_score = 1.0 - avg_score
        is_anomalous = anomaly_score > threshold

        if is_anomalous:
            self.anomalies_detected += 1

        # Update recent (with lower weight if anomalous)
        weight = 0.1 if is_anomalous else 1.0
        for perspective, vec in vectors.items():
            self._update_recent(perspective, vec, weight)

        return DetectionResult(
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            perspective_scores=scores,
        )

    def _encode_packet(self, pkt: Packet) -> Dict[str, np.ndarray]:
        """Encode packet into perspective vectors."""
        vectors = {}

        # L3 perspective
        l3_atoms = []
        if IP in pkt:
            l3_atoms.append(f"src_prefix:{pkt[IP].src.rsplit('.', 1)[0]}")
            l3_atoms.append(f"dst_prefix:{pkt[IP].dst.rsplit('.', 1)[0]}")
        l3_vec = np.zeros(DIMENSIONS, dtype=np.float64)
        for atom in l3_atoms:
            l3_vec += self.vm.get_vector(atom)
        vectors["l3"] = l3_vec

        # L4 perspective
        l4_atoms = []
        if TCP in pkt:
            l4_atoms.append("proto:tcp")
            l4_atoms.append(f"dst_port:{pkt[TCP].dport}")
        elif UDP in pkt:
            l4_atoms.append("proto:udp")
            l4_atoms.append(f"dst_port:{pkt[UDP].dport}")
        elif ICMP in pkt:
            l4_atoms.append("proto:icmp")
        l4_vec = np.zeros(DIMENSIONS, dtype=np.float64)
        for atom in l4_atoms:
            l4_vec += self.vm.get_vector(atom)
        vectors["l4"] = l4_vec

        # Payload perspective
        payload_atoms = []
        if Raw in pkt:
            payload = bytes(pkt[Raw].load)
            for i, b in enumerate(payload[:8]):
                payload_atoms.append(f"byte_{i}:{hex(b)}")
        payload_vec = np.zeros(DIMENSIONS, dtype=np.float64)
        for atom in payload_atoms:
            payload_vec += self.vm.get_vector(atom)
        vectors["payload"] = payload_vec

        return vectors

    def _update_recent(self, perspective: str, vec: np.ndarray, weight: float):
        """Update recent knowledge with decay."""
        if perspective not in self.recent_vectors:
            self.recent_vectors[perspective] = np.zeros(DIMENSIONS, dtype=np.float64)
            self.recent_counts[perspective] = 0

        # Decay existing
        self.recent_vectors[perspective] *= self.decay_factor
        # Add new
        self.recent_vectors[perspective] += vec * weight
        self.recent_counts[perspective] += 1

    def get_recent_accumulator(self) -> Dict[str, np.ndarray]:
        """Get current recent accumulators for merging."""
        return {k: v.copy() for k, v in self.recent_vectors.items()}

    def merge_accumulators(self, other_accumulators: List[Dict[str, np.ndarray]]):
        """Merge accumulators from other nodes."""
        for other in other_accumulators:
            for perspective, vec in other.items():
                if perspective in self.recent_vectors:
                    self.recent_vectors[perspective] += vec
                else:
                    self.recent_vectors[perspective] = vec.copy()


# =============================================================================
# DISTRIBUTED SYSTEM
# =============================================================================

class DistributedDetector:
    """
    Coordinator for distributed detection across multiple nodes.
    """

    def __init__(self, num_nodes: int = 3, global_seed: int = GLOBAL_SEED):
        self.global_seed = global_seed
        self.nodes = [
            SensorNode(node_id=f"node_{i}", global_seed=global_seed)
            for i in range(num_nodes)
        ]

        # Shared prior (generated once, distributed to all)
        self.shared_prior = self._generate_prior()

        # Distribute prior to all nodes
        for node in self.nodes:
            node.load_prior(self.shared_prior)

    def _generate_prior(self) -> Dict[str, np.ndarray]:
        """Generate shared prior knowledge."""
        vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=self.global_seed)
        prior = {}

        # Normal traffic patterns
        normal_atoms = {
            "l3": ["src_prefix:192.168.1", "dst_prefix:93.184.216"],
            "l4": ["proto:tcp", "dst_port:80", "dst_port:443"],
            "payload": ["byte_0:0x47", "byte_1:0x45", "byte_2:0x54"],  # "GET"
        }

        for perspective, atoms in normal_atoms.items():
            vec = np.zeros(DIMENSIONS, dtype=np.float64)
            for atom in atoms:
                vec += vm.get_vector(atom)
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            prior[perspective] = vec

        return prior

    def verify_consensus(self) -> Dict[str, float]:
        """Verify that all nodes produce identical vectors for same atoms."""
        test_atoms = ["test_atom_1", "test_atom_2", "consensus_check"]

        results = {}
        for atom in test_atoms:
            vectors = [node.vm.get_vector(atom) for node in self.nodes]

            # Check all pairs
            agreements = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    agreements.append(np.array_equal(vectors[i], vectors[j]))

            results[atom] = sum(agreements) / len(agreements) if agreements else 1.0

        return results

    def process_distributed(
        self,
        packets: List[Packet],
        distribution: str = "round_robin"
    ) -> Dict[str, List[DetectionResult]]:
        """
        Distribute packets across nodes and collect results.

        distribution: "round_robin" | "random" | "hash"
        """
        results = {node.node_id: [] for node in self.nodes}

        for i, pkt in enumerate(packets):
            if distribution == "round_robin":
                node_idx = i % len(self.nodes)
            elif distribution == "random":
                node_idx = random.randint(0, len(self.nodes) - 1)
            elif distribution == "hash":
                # Hash by source IP
                if IP in pkt:
                    node_idx = hash(pkt[IP].src) % len(self.nodes)
                else:
                    node_idx = 0
            else:
                node_idx = 0

            node = self.nodes[node_idx]
            result = node.process_packet(pkt)
            results[node.node_id].append(result)

        return results

    def merge_node_knowledge(self):
        """Merge recent knowledge across all nodes."""
        # Collect all accumulators
        all_accumulators = [node.get_recent_accumulator() for node in self.nodes]

        # Merge into each node
        for i, node in enumerate(self.nodes):
            # Merge all except own
            others = [acc for j, acc in enumerate(all_accumulators) if j != i]
            node.merge_accumulators(others)

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics from all nodes."""
        return {
            node.node_id: {
                "packets": node.packets_processed,
                "anomalies": node.anomalies_detected,
                "anomaly_rate": node.anomalies_detected / max(1, node.packets_processed),
            }
            for node in self.nodes
        }


# =============================================================================
# PACKET GENERATORS
# =============================================================================

def generate_normal_traffic(n: int) -> List[Packet]:
    """Generate normal traffic."""
    packets = []
    for _ in range(n):
        pkt = IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
            sport=random.randint(49152, 65535),
            dport=random.choice([80, 443]),
            flags="PA"
        ) / Raw(load=b"GET / HTTP/1.1\r\n")
        packets.append(pkt)
    return packets


def generate_attack_traffic(n: int) -> List[Packet]:
    """Generate attack traffic."""
    packets = []
    for i in range(n):
        src_ip = f"10.{(i // 256) % 256}.{i % 256}.1"
        pkt = IP(src=src_ip, dst="192.168.1.100") / TCP(
            sport=40000 + (i % 1000),
            dport=80,
            flags="S"
        )
        packets.append(pkt)
    return packets


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 80)
    print("CHALLENGE 011-011: MULTI-NODE CONSENSUS")
    print("=" * 80)

    # Create distributed detector with 3 nodes
    detector = DistributedDetector(num_nodes=3, global_seed=GLOBAL_SEED)

    # Verify consensus
    print("\n" + "=" * 60)
    print("CONSENSUS VERIFICATION")
    print("=" * 60)

    consensus = detector.verify_consensus()
    all_agree = all(v == 1.0 for v in consensus.values())

    print(f"\nVector agreement across nodes:")
    for atom, agreement in consensus.items():
        print(f"  {atom}: {agreement:.0%} agreement")

    print(f"\n{'✓' if all_agree else '✗'} All nodes produce IDENTICAL vectors: {all_agree}")

    # Test with normal traffic
    print("\n" + "=" * 60)
    print("TEST 1: NORMAL TRAFFIC (distributed)")
    print("=" * 60)

    normal_packets = generate_normal_traffic(300)
    results = detector.process_distributed(normal_packets, distribution="round_robin")

    stats = detector.get_stats()
    print("\nPer-node statistics:")
    for node_id, node_stats in stats.items():
        print(f"  {node_id}: {node_stats['packets']} packets, "
              f"{node_stats['anomalies']} anomalies ({node_stats['anomaly_rate']:.1%})")

    # Test with attack traffic
    print("\n" + "=" * 60)
    print("TEST 2: ATTACK TRAFFIC (distributed)")
    print("=" * 60)

    attack_packets = generate_attack_traffic(150)
    results = detector.process_distributed(attack_packets, distribution="round_robin")

    stats = detector.get_stats()
    print("\nPer-node statistics (including attack):")
    for node_id, node_stats in stats.items():
        print(f"  {node_id}: {node_stats['packets']} packets, "
              f"{node_stats['anomalies']} anomalies ({node_stats['anomaly_rate']:.1%})")

    # Test hash-based distribution
    print("\n" + "=" * 60)
    print("TEST 3: HASH-BASED DISTRIBUTION")
    print("=" * 60)

    # Reset nodes
    detector = DistributedDetector(num_nodes=3, global_seed=GLOBAL_SEED)

    mixed_packets = generate_normal_traffic(150) + generate_attack_traffic(150)
    random.shuffle(mixed_packets)

    results = detector.process_distributed(mixed_packets, distribution="hash")

    stats = detector.get_stats()
    print("\nPer-node statistics (hash distribution):")
    for node_id, node_stats in stats.items():
        print(f"  {node_id}: {node_stats['packets']} packets, "
              f"{node_stats['anomalies']} anomalies ({node_stats['anomaly_rate']:.1%})")

    # Test accumulator merging
    print("\n" + "=" * 60)
    print("TEST 4: ACCUMULATOR MERGING")
    print("=" * 60)

    # Before merge - check recent vectors similarity
    print("\nRecent vector similarity BEFORE merge:")
    for perspective in ["l3", "l4", "payload"]:
        vecs = [node.recent_vectors.get(perspective, np.zeros(DIMENSIONS))
                for node in detector.nodes]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = cosine_similarity(vecs[i], vecs[j])
                print(f"  {perspective} node_{i} ↔ node_{j}: {sim:.3f}")

    # Merge
    detector.merge_node_knowledge()

    print("\nRecent vector similarity AFTER merge:")
    for perspective in ["l3", "l4", "payload"]:
        vecs = [node.recent_vectors.get(perspective, np.zeros(DIMENSIONS))
                for node in detector.nodes]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = cosine_similarity(vecs[i], vecs[j])
                print(f"  {perspective} node_{i} ↔ node_{j}: {sim:.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print("""
1. CONSENSUS ACHIEVED: All nodes with same global_seed produce IDENTICAL vectors
   - No coordination needed for atom → vector mapping
   - Priors can be distributed without vector synchronization

2. DETECTION CONSISTENCY:
   - Same packet → same prior similarity (since vectors are identical)
   - Recent knowledge diverges (each node sees different traffic)
   - After merge, nodes converge to shared view

3. DISTRIBUTION STRATEGIES:
   - Round-robin: Even load, but related packets may hit different nodes
   - Hash by src_ip: Same source always hits same node (good for cardinality)
   - Random: Maximum load balancing, minimum locality

4. ACCUMULATOR MERGING:
   - Nodes can share recent knowledge periodically
   - Merged accumulators = union of all observed patterns
   - Post-merge similarity increases (shared knowledge)

5. ARCHITECTURE IMPLICATIONS:
   - Prior knowledge: Generate once, distribute everywhere
   - Recent knowledge: Local or periodically merged
   - Detection: Can run independently, results are comparable
""")


if __name__ == "__main__":
    main()
