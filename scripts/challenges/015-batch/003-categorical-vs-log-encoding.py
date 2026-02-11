#!/usr/bin/env python3
"""
BATCH 15 - CHALLENGE 003: Categorical vs Log Encoding Comparison

HYPOTHESIS: Using $log encoding for numeric fields (payload_size, bytes_out)
will improve detection accuracy compared to categorical buckets because
magnitude relationships are preserved.

COMPARISON:
===========
Old approach (categorical):
  "size_class": "tiny" if size < 100 else "small" if < 500 else "medium" if < 2000 else "large"
  - 100 bytes and 499 bytes both → "small" (no distinction)
  - 501 bytes and 1999 bytes both → "medium" (4x difference lost)

New approach ($log):
  "payload_size": {"$log": size}
  - 100 bytes similar to 150 bytes (1.5x)
  - 100 bytes less similar to 1000 bytes (10x)
  - Magnitude relationships preserved

METRICS:
========
1. Detection accuracy (TP, FP, FN rates)
2. Attack type separation (do different attacks cluster distinctly?)
3. Magnitude correlation (do similar-sized payloads cluster?)
4. Attribution quality (invert() decomposition accuracy)

Run: ./scripts/run_with_venv.sh python scripts/challenges/015-batch/003-categorical-vs-log-encoding.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import complexity, invert


# =============================================================================
# PACKET GENERATION (From Batch 014)
# =============================================================================


@dataclass
class Packet:
    src_port: int
    dst_port: int
    protocol: str
    flags: str
    payload_size: int
    label: str  # Ground truth


def generate_normal_traffic(count: int, seed: int = 42) -> List[Packet]:
    """Generate normal traffic mix with varied payload sizes."""
    rng = np.random.default_rng(seed)
    packets = []

    for _ in range(count):
        r = rng.random()
        if r < 0.4:  # 40% HTTPS
            packets.append(Packet(
                src_port=rng.integers(49152, 65535),
                dst_port=443,
                protocol="TCP",
                flags="A" if rng.random() > 0.1 else "PA",
                payload_size=int(rng.exponential(500)),  # Varies: 0-2000+
                label="normal"
            ))
        elif r < 0.7:  # 30% HTTP
            packets.append(Packet(
                src_port=rng.integers(49152, 65535),
                dst_port=80,
                protocol="TCP",
                flags="A" if rng.random() > 0.1 else "PA",
                payload_size=int(rng.exponential(800)),
                label="normal"
            ))
        elif r < 0.85:  # 15% DNS
            packets.append(Packet(
                src_port=rng.integers(49152, 65535),
                dst_port=53,
                protocol="UDP",
                flags="",
                payload_size=int(rng.exponential(100)),  # Small DNS queries
                label="normal"
            ))
        else:  # 15% Other
            packets.append(Packet(
                src_port=rng.integers(49152, 65535),
                dst_port=rng.integers(1024, 49151),
                protocol="TCP" if rng.random() > 0.3 else "UDP",
                flags="A",
                payload_size=int(rng.exponential(300)),
                label="normal"
            ))

    return packets


def generate_dns_reflection(count: int, seed: int = 123) -> List[Packet]:
    """DNS reflection attack: spoofed src_port=53, LARGE responses."""
    rng = np.random.default_rng(seed)
    return [
        Packet(
            src_port=53,
            dst_port=rng.integers(49152, 65535),
            protocol="UDP",
            flags="",
            payload_size=int(rng.exponential(4000)),  # Large amplified responses
            label="dns_reflection"
        )
        for _ in range(count)
    ]


def generate_syn_flood(count: int, seed: int = 456) -> List[Packet]:
    """SYN flood: many SYN packets with ZERO payload."""
    rng = np.random.default_rng(seed)
    return [
        Packet(
            src_port=rng.integers(1024, 65535),
            dst_port=rng.choice([80, 443, 8080, 22, 3389]),
            protocol="TCP",
            flags="S",
            payload_size=0,  # SYN packets have no payload
            label="syn_flood"
        )
        for _ in range(count)
    ]


def generate_exfiltration(count: int, seed: int = 789) -> List[Packet]:
    """Data exfiltration: VERY LARGE payloads leaving the network."""
    rng = np.random.default_rng(seed)
    return [
        Packet(
            src_port=rng.integers(49152, 65535),
            dst_port=443,
            protocol="TCP",
            flags="PA",
            payload_size=int(rng.exponential(100000)),  # Huge payloads
            label="exfiltration"
        )
        for _ in range(count)
    ]


def generate_port_scan(count: int, seed: int = 101) -> List[Packet]:
    """Port scan: tiny probes to many ports."""
    rng = np.random.default_rng(seed)
    return [
        Packet(
            src_port=rng.integers(49152, 65535),
            dst_port=rng.integers(1, 1024),  # Scanning well-known ports
            protocol="TCP",
            flags="S",
            payload_size=0,  # SYN probes
            label="port_scan"
        )
        for _ in range(count)
    ]


# =============================================================================
# ENCODING APPROACHES
# =============================================================================


def encode_categorical(client: HolonClient, pkt: Packet) -> np.ndarray:
    """Original categorical bucket encoding (from Batch 014)."""
    src_port_band = (
        "dns" if pkt.src_port == 53 else
        "ntp" if pkt.src_port == 123 else
        "wellknown" if pkt.src_port < 1024 else
        "ephemeral"
    )
    dst_port_band = (
        "http" if pkt.dst_port in [80, 8080] else
        "https" if pkt.dst_port == 443 else
        "dns" if pkt.dst_port == 53 else
        "wellknown" if pkt.dst_port < 1024 else
        "ephemeral"
    )

    # Categorical size encoding
    if pkt.payload_size == 0:
        size_class = "zero"
    elif pkt.payload_size < 100:
        size_class = "tiny"
    elif pkt.payload_size < 500:
        size_class = "small"
    elif pkt.payload_size < 2000:
        size_class = "medium"
    elif pkt.payload_size < 10000:
        size_class = "large"
    else:
        size_class = "huge"

    return client.encode({
        "src_port_band": src_port_band,
        "dst_port_band": dst_port_band,
        "protocol": pkt.protocol,
        "flags": pkt.flags if pkt.flags else "none",
        "size_class": size_class,
        "direction": "amplified" if pkt.src_port < 1024 and pkt.dst_port >= 1024 else "normal",
    })


def encode_log(client: HolonClient, pkt: Packet) -> np.ndarray:
    """New $log encoding for payload size."""
    src_port_band = (
        "dns" if pkt.src_port == 53 else
        "ntp" if pkt.src_port == 123 else
        "wellknown" if pkt.src_port < 1024 else
        "ephemeral"
    )
    dst_port_band = (
        "http" if pkt.dst_port in [80, 8080] else
        "https" if pkt.dst_port == 443 else
        "dns" if pkt.dst_port == 53 else
        "wellknown" if pkt.dst_port < 1024 else
        "ephemeral"
    )

    # Log encoding for size - preserves magnitude relationships
    # Handle zero specially since log(0) is undefined
    if pkt.payload_size == 0:
        size_encoding = "zero"
    else:
        size_encoding = {"$log": pkt.payload_size}

    return client.encode({
        "src_port_band": src_port_band,
        "dst_port_band": dst_port_band,
        "protocol": pkt.protocol,
        "flags": pkt.flags if pkt.flags else "none",
        "payload_size": size_encoding,
        "direction": "amplified" if pkt.src_port < 1024 and pkt.dst_port >= 1024 else "normal",
    })


def encode_hybrid(client: HolonClient, pkt: Packet) -> np.ndarray:
    """Hybrid: both categorical AND log encoding for maximum signal."""
    src_port_band = (
        "dns" if pkt.src_port == 53 else
        "ntp" if pkt.src_port == 123 else
        "wellknown" if pkt.src_port < 1024 else
        "ephemeral"
    )
    dst_port_band = (
        "http" if pkt.dst_port in [80, 8080] else
        "https" if pkt.dst_port == 443 else
        "dns" if pkt.dst_port == 53 else
        "wellknown" if pkt.dst_port < 1024 else
        "ephemeral"
    )

    # Categorical for coarse grouping
    if pkt.payload_size == 0:
        size_class = "zero"
    elif pkt.payload_size < 100:
        size_class = "tiny"
    elif pkt.payload_size < 500:
        size_class = "small"
    elif pkt.payload_size < 2000:
        size_class = "medium"
    elif pkt.payload_size < 10000:
        size_class = "large"
    else:
        size_class = "huge"

    # Log for fine-grained magnitude
    if pkt.payload_size == 0:
        size_log = "zero"
    else:
        size_log = {"$log": pkt.payload_size}

    return client.encode({
        "src_port_band": src_port_band,
        "dst_port_band": dst_port_band,
        "protocol": pkt.protocol,
        "flags": pkt.flags if pkt.flags else "none",
        "size_class": size_class,      # Categorical
        "payload_size": size_log,       # Log magnitude
        "direction": "amplified" if pkt.src_port < 1024 and pkt.dst_port >= 1024 else "normal",
    })


# =============================================================================
# EXPERIMENT FRAMEWORK
# =============================================================================


class DetectionExperiment:
    """Run detection experiment with different encoding approaches."""

    def __init__(self, client: HolonClient, encode_fn, name: str):
        self.client = client
        self.encode_fn = encode_fn
        self.name = name
        self.baseline_proto = None
        self.attack_codebook = []

    def learn_baseline(self, normal_packets: List[Packet]):
        """Learn baseline from normal traffic."""
        vecs = [self.encode_fn(self.client, p) for p in normal_packets]
        self.baseline_proto = self.client.prototype(vecs)
        self.baseline_complexity = np.mean([complexity(v) for v in vecs])

    def learn_attack_signatures(
        self,
        dns_packets: List[Packet],
        syn_packets: List[Packet],
        exfil_packets: List[Packet],
        scan_packets: List[Packet]
    ):
        """Learn attack family prototypes."""
        self.attack_codebook = [
            ("normal", self.baseline_proto),
            ("dns_reflection", self.client.prototype(
                [self.encode_fn(self.client, p) for p in dns_packets]
            )),
            ("syn_flood", self.client.prototype(
                [self.encode_fn(self.client, p) for p in syn_packets]
            )),
            ("exfiltration", self.client.prototype(
                [self.encode_fn(self.client, p) for p in exfil_packets]
            )),
            ("port_scan", self.client.prototype(
                [self.encode_fn(self.client, p) for p in scan_packets]
            )),
        ]

    def detect(self, packets: List[Packet], threshold: float = 0.7) -> Dict:
        """Detect anomalies and classify."""
        results = {
            "predictions": [],
            "confidences": [],
            "attributions": [],
        }

        for pkt in packets:
            vec = self.encode_fn(self.client, pkt)
            sim = cosine_similarity(vec, self.baseline_proto)

            is_anomaly = sim < threshold

            # Attribution via invert
            components = invert(vec, self.attack_codebook, top_k=3, threshold=0.3)
            top_match = components[0][0] if components else "unknown"

            results["predictions"].append("anomaly" if is_anomaly else "normal")
            results["confidences"].append(1 - sim if is_anomaly else sim)
            results["attributions"].append(top_match)

        return results

    def evaluate(
        self,
        test_packets: List[Packet],
        threshold: float = 0.7
    ) -> Dict:
        """Evaluate detection performance."""
        results = self.detect(test_packets, threshold)

        # Calculate metrics
        tp, fp, tn, fn = 0, 0, 0, 0
        correct_attribution = 0

        for i, pkt in enumerate(test_packets):
            is_actually_anomaly = pkt.label != "normal"
            predicted_anomaly = results["predictions"][i] == "anomaly"
            attributed = results["attributions"][i]

            if is_actually_anomaly and predicted_anomaly:
                tp += 1
                if attributed == pkt.label:
                    correct_attribution += 1
            elif is_actually_anomaly and not predicted_anomaly:
                fn += 1
            elif not is_actually_anomaly and predicted_anomaly:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        attribution_accuracy = correct_attribution / tp if tp > 0 else 0

        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "attribution_accuracy": attribution_accuracy,
        }


def measure_cluster_separation(
    client: HolonClient,
    encode_fn,
    packets_by_label: Dict[str, List[Packet]]
) -> Dict[str, float]:
    """Measure how well different attack types are separated."""
    # Encode all packets
    vecs_by_label = {}
    for label, packets in packets_by_label.items():
        vecs_by_label[label] = [encode_fn(client, p) for p in packets]

    # Calculate within-cluster and between-cluster similarities
    labels = list(vecs_by_label.keys())
    within_sims = {}
    between_sims = {}

    for label in labels:
        vecs = vecs_by_label[label]
        # Within-cluster: average pairwise similarity
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(cosine_similarity(vecs[i], vecs[j]))
        within_sims[label] = np.mean(sims) if sims else 0

    # Between-cluster: average similarity between different labels
    for i, label1 in enumerate(labels):
        for label2 in labels[i + 1:]:
            key = f"{label1}_vs_{label2}"
            sims = []
            for v1 in vecs_by_label[label1][:50]:  # Sample for speed
                for v2 in vecs_by_label[label2][:50]:
                    sims.append(cosine_similarity(v1, v2))
            between_sims[key] = np.mean(sims) if sims else 0

    avg_within = np.mean(list(within_sims.values()))
    avg_between = np.mean(list(between_sims.values()))
    separation = avg_within - avg_between

    return {
        "avg_within_cluster": avg_within,
        "avg_between_cluster": avg_between,
        "separation_score": separation,  # Higher = better separated
    }


def measure_magnitude_correlation(
    client: HolonClient,
    encode_fn,
    packets: List[Packet]
) -> float:
    """Measure if similar payload sizes produce similar vectors."""
    # Sort by payload size
    sorted_pkts = sorted([p for p in packets if p.payload_size > 0],
                         key=lambda p: p.payload_size)

    if len(sorted_pkts) < 10:
        return 0.0

    # Encode
    vecs = [encode_fn(client, p) for p in sorted_pkts]
    sizes = [p.payload_size for p in sorted_pkts]

    # Calculate correlation between size ratio and similarity
    # For adjacent pairs, size ratio should correlate with similarity
    correlations = []
    for i in range(len(vecs) - 1):
        size_ratio = sizes[i + 1] / sizes[i] if sizes[i] > 0 else 1
        sim = cosine_similarity(vecs[i], vecs[i + 1])
        # Closer ratio to 1 should mean higher similarity
        expected_sim = 1.0 / (1.0 + np.log10(max(size_ratio, 1.01)))
        correlations.append((expected_sim, sim))

    # Return correlation coefficient
    if len(correlations) > 2:
        expected = [c[0] for c in correlations]
        actual = [c[1] for c in correlations]
        corr = np.corrcoef(expected, actual)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    return 0.0


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print("=" * 70)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       BATCH 15 - CHALLENGE 003: CATEGORICAL VS LOG ENCODING           ║
║                                                                        ║
║  Comparing payload_size encoding approaches for detection accuracy     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize client
    client = HolonClient(dimensions=4096)

    # ==========================================================================
    # GENERATE DATA
    # ==========================================================================

    print_section("DATA GENERATION")

    # Training data
    train_normal = generate_normal_traffic(500, seed=42)
    train_dns = generate_dns_reflection(100, seed=100)
    train_syn = generate_syn_flood(100, seed=200)
    train_exfil = generate_exfiltration(100, seed=300)
    train_scan = generate_port_scan(100, seed=400)

    # Test data (different seeds)
    test_normal = generate_normal_traffic(200, seed=999)
    test_dns = generate_dns_reflection(50, seed=998)
    test_syn = generate_syn_flood(50, seed=997)
    test_exfil = generate_exfiltration(50, seed=996)
    test_scan = generate_port_scan(50, seed=995)

    test_all = test_normal + test_dns + test_syn + test_exfil + test_scan

    print(f"Training: {len(train_normal)} normal, {len(train_dns)} DNS, "
          f"{len(train_syn)} SYN, {len(train_exfil)} exfil, {len(train_scan)} scan")
    print(f"Testing: {len(test_all)} total packets")

    # Payload size distribution
    all_sizes = [p.payload_size for p in train_normal + train_dns + train_exfil]
    print(f"\nPayload size distribution:")
    print(f"  Min: {min(all_sizes)}, Max: {max(all_sizes)}, Median: {np.median(all_sizes):.0f}")

    # ==========================================================================
    # RUN EXPERIMENTS
    # ==========================================================================

    experiments = [
        ("Categorical", encode_categorical),
        ("Log ($log)", encode_log),
        ("Hybrid (both)", encode_hybrid),
    ]

    results = {}

    for name, encode_fn in experiments:
        print_section(f"EXPERIMENT: {name}")

        exp = DetectionExperiment(client, encode_fn, name)

        # Learn baseline and signatures
        exp.learn_baseline(train_normal)
        exp.learn_attack_signatures(train_dns, train_syn, train_exfil, train_scan)

        # Evaluate detection
        eval_result = exp.evaluate(test_all, threshold=0.7)

        # Measure cluster separation
        packets_by_label = {
            "normal": test_normal,
            "dns_reflection": test_dns,
            "syn_flood": test_syn,
            "exfiltration": test_exfil,
            "port_scan": test_scan,
        }
        separation = measure_cluster_separation(client, encode_fn, packets_by_label)

        # Measure magnitude correlation
        magnitude_corr = measure_magnitude_correlation(client, encode_fn, train_normal + train_exfil)

        results[name] = {
            **eval_result,
            **separation,
            "magnitude_correlation": magnitude_corr,
        }

        print(f"\nDetection Performance:")
        print(f"  Precision:    {eval_result['precision']:.3f}")
        print(f"  Recall:       {eval_result['recall']:.3f}")
        print(f"  F1 Score:     {eval_result['f1']:.3f}")
        print(f"  Attribution:  {eval_result['attribution_accuracy']:.3f}")

        print(f"\nCluster Separation:")
        print(f"  Within-cluster avg:  {separation['avg_within_cluster']:.3f}")
        print(f"  Between-cluster avg: {separation['avg_between_cluster']:.3f}")
        print(f"  Separation score:    {separation['separation_score']:.3f}")

        print(f"\nMagnitude Preservation:")
        print(f"  Size-similarity correlation: {magnitude_corr:.3f}")

    # ==========================================================================
    # COMPARISON SUMMARY
    # ==========================================================================

    print_section("COMPARISON SUMMARY")

    print(f"\n{'Approach':<20} {'F1':>8} {'Attrib':>8} {'Sep':>8} {'MagCorr':>8}")
    print("-" * 56)

    for name in results:
        r = results[name]
        print(f"{name:<20} {r['f1']:>8.3f} {r['attribution_accuracy']:>8.3f} "
              f"{r['separation_score']:>8.3f} {r['magnitude_correlation']:>8.3f}")

    # Determine winner
    print("\n" + "-" * 56)

    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    best_attr = max(results.items(), key=lambda x: x[1]['attribution_accuracy'])
    best_sep = max(results.items(), key=lambda x: x[1]['separation_score'])
    best_mag = max(results.items(), key=lambda x: x[1]['magnitude_correlation'])

    print(f"\nBest F1 Score:       {best_f1[0]} ({best_f1[1]['f1']:.3f})")
    print(f"Best Attribution:    {best_attr[0]} ({best_attr[1]['attribution_accuracy']:.3f})")
    print(f"Best Separation:     {best_sep[0]} ({best_sep[1]['separation_score']:.3f})")
    print(f"Best Mag Correlation: {best_mag[0]} ({best_mag[1]['magnitude_correlation']:.3f})")

    # ==========================================================================
    # DETAILED ANALYSIS: WHY LOG HELPS
    # ==========================================================================

    print_section("DETAILED ANALYSIS: MAGNITUDE SENSITIVITY")

    # Compare specific payload size pairs
    print("\nHow encoding handles payload size differences:")
    print("-" * 60)

    test_sizes = [
        (100, 150, "1.5x ratio"),
        (100, 500, "5x ratio (crosses bucket boundary)"),
        (100, 1000, "10x ratio"),
        (1000, 10000, "10x ratio (large scale)"),
        (100, 100000, "1000x ratio"),
    ]

    for size1, size2, desc in test_sizes:
        pkt1 = Packet(49152, 443, "TCP", "PA", size1, "test")
        pkt2 = Packet(49152, 443, "TCP", "PA", size2, "test")

        vec1_cat = encode_categorical(client, pkt1)
        vec2_cat = encode_categorical(client, pkt2)
        sim_cat = cosine_similarity(vec1_cat, vec2_cat)

        vec1_log = encode_log(client, pkt1)
        vec2_log = encode_log(client, pkt2)
        sim_log = cosine_similarity(vec1_log, vec2_log)

        print(f"\n{size1:>6} vs {size2:>6} bytes ({desc}):")
        print(f"  Categorical: {sim_cat:.3f}")
        print(f"  Log:         {sim_log:.3f}")
        print(f"  Difference:  {sim_log - sim_cat:+.3f}")

    # ==========================================================================
    # CONCLUSIONS
    # ==========================================================================

    print_section("CONCLUSIONS")

    cat_f1 = results["Categorical"]["f1"]
    log_f1 = results["Log ($log)"]["f1"]
    hybrid_f1 = results["Hybrid (both)"]["f1"]

    cat_mag = results["Categorical"]["magnitude_correlation"]
    log_mag = results["Log ($log)"]["magnitude_correlation"]

    print(f"""
FINDINGS:

1. DETECTION (F1 Score):
   - Categorical: {cat_f1:.3f}
   - Log:         {log_f1:.3f}
   - Hybrid:      {hybrid_f1:.3f}
   - {'Log wins' if log_f1 > cat_f1 else 'Categorical wins' if cat_f1 > log_f1 else 'Tie'} by {abs(log_f1 - cat_f1):.3f}

2. MAGNITUDE PRESERVATION:
   - Categorical: {cat_mag:.3f} (loses within-bucket differences)
   - Log:         {log_mag:.3f} (preserves ratios)
   - {'Log wins' if log_mag > cat_mag else 'Categorical wins'} by {abs(log_mag - cat_mag):.3f}

3. KEY INSIGHT:
   Categorical encoding creates "bucket boundaries" where 499 bytes
   and 501 bytes are maximally different (different buckets), but
   100 bytes and 499 bytes are identical (same bucket).

   Log encoding makes similarity proportional to magnitude ratio:
   - 100 vs 150 (1.5x) → high similarity
   - 100 vs 1000 (10x) → moderate similarity
   - 100 vs 100000 (1000x) → low similarity

4. RECOMMENDATION:
   {'Use $log for payload_size and bytes_out fields' if log_f1 >= cat_f1 else 'Categorical still works well'}
   {'Consider hybrid for maximum signal richness' if hybrid_f1 > max(cat_f1, log_f1) else ''}
""")


if __name__ == "__main__":
    main()
