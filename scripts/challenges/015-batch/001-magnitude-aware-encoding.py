#!/usr/bin/env python3
"""
BATCH 15 - CHALLENGE 001: Magnitude-Aware Numeric Encoding

HYPOTHESIS: Log encoding clusters similar magnitudes while string encoding
treats all values as unrelated. This enables "find similar intensity"
queries without hard-coded thresholds.

This demo shows:
1. String encoding: numbers are quasi-orthogonal (no magnitude relationship)
2. Log encoding: similar magnitudes have high similarity
3. Practical application: clustering traffic by rate magnitude

Run: ./scripts/run_with_venv.sh python scripts/challenges/015-batch/001-magnitude-aware-encoding.py
"""

import numpy as np

from holon import Encoder, VectorManager
from holon.distance import cosine_similarity


def print_header(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_similarity_matrix(labels: list, vectors: list, title: str):
    """Print a similarity matrix for a set of vectors."""
    print(f"\n{title}")
    print("-" * 60)

    # Header row
    header = "        " + "".join(f"{l:>10}" for l in labels)
    print(header)

    # Matrix rows
    for i, (label, vec) in enumerate(zip(labels, vectors)):
        row = f"{label:>8}"
        for j, vec2 in enumerate(vectors):
            sim = cosine_similarity(vec, vec2)
            row += f"{sim:>10.3f}"
        print(row)


def demo_string_vs_log_encoding():
    """Demonstrate the difference between string and log encoding."""
    print_header("PART 1: String vs Log Encoding")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Test values spanning multiple orders of magnitude
    values = [100, 200, 1000, 10000, 100000]
    labels = [str(v) for v in values]

    # String encoding (default behavior)
    print("\nString Encoding (default): Numbers → random vectors, no magnitude relationship")
    string_vectors = [enc.encode_data(v) for v in values]
    print_similarity_matrix(labels, string_vectors, "String Encoding Similarity Matrix")

    # Log encoding
    print("\n\nLog Encoding ($log): Similar magnitudes → similar vectors")
    log_vectors = [enc.encode_data({"$log": v}) for v in values]
    print_similarity_matrix(labels, log_vectors, "Log Encoding Similarity Matrix")

    # Analysis
    print("\n" + "-" * 60)
    print("ANALYSIS:")
    print("-" * 60)

    # String encoding similarities
    sim_100_200_str = cosine_similarity(string_vectors[0], string_vectors[1])
    sim_100_1000_str = cosine_similarity(string_vectors[0], string_vectors[2])

    # Log encoding similarities
    sim_100_200_log = cosine_similarity(log_vectors[0], log_vectors[1])
    sim_100_1000_log = cosine_similarity(log_vectors[0], log_vectors[2])
    sim_1000_10000_log = cosine_similarity(log_vectors[2], log_vectors[3])

    print(f"\nString: 100 vs 200 = {sim_100_200_str:.3f} (essentially random)")
    print(f"String: 100 vs 1000 = {sim_100_1000_str:.3f} (also random)")
    print(f"\nLog: 100 vs 200 (2x) = {sim_100_200_log:.3f} (high - same ballpark)")
    print(f"Log: 100 vs 1000 (10x) = {sim_100_1000_log:.3f} (moderate - one order apart)")
    print(f"Log: 1000 vs 10000 (10x) = {sim_1000_10000_log:.3f} (similar - also 10x ratio)")

    print("\nKEY INSIGHT: Log encoding makes 10x ratios produce consistent similarity drops,")
    print("regardless of the absolute values involved.")


def demo_traffic_magnitude_clustering():
    """Demonstrate clustering traffic samples by rate magnitude."""
    print_header("PART 2: Traffic Magnitude Clustering")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Traffic samples at different magnitude tiers
    traffic_samples = [
        # Tier 1: Low-rate scanners (10-100 pps)
        {"type": "scanner", "src_ip": "10.0.0.1", "rate_pps": {"$log": 15}},
        {"type": "scanner", "src_ip": "10.0.0.2", "rate_pps": {"$log": 45}},
        {"type": "scanner", "src_ip": "10.0.0.3", "rate_pps": {"$log": 80}},
        # Tier 2: Medium traffic (1k-10k pps)
        {"type": "normal", "src_ip": "10.0.1.1", "rate_pps": {"$log": 2000}},
        {"type": "normal", "src_ip": "10.0.1.2", "rate_pps": {"$log": 5000}},
        {"type": "normal", "src_ip": "10.0.1.3", "rate_pps": {"$log": 8000}},
        # Tier 3: High-volume attacks (100k+ pps)
        {"type": "attack", "src_ip": "10.0.2.1", "rate_pps": {"$log": 150000}},
        {"type": "attack", "src_ip": "10.0.2.2", "rate_pps": {"$log": 500000}},
        {"type": "attack", "src_ip": "10.0.2.3", "rate_pps": {"$log": 800000}},
    ]

    labels = [
        "scan-15", "scan-45", "scan-80",
        "norm-2k", "norm-5k", "norm-8k",
        "atk-150k", "atk-500k", "atk-800k"
    ]

    # Encode all samples
    vectors = [enc.encode_data(sample) for sample in traffic_samples]

    print("\nTraffic samples with $log-encoded rates:")
    for label, sample in zip(labels, traffic_samples):
        rate = sample["rate_pps"]["$log"]
        print(f"  {label}: {rate:,} pps")

    print_similarity_matrix(labels, vectors, "\nTraffic Sample Similarity Matrix")

    # Analyze clustering
    print("\n" + "-" * 60)
    print("CLUSTER ANALYSIS:")
    print("-" * 60)

    # Within-tier similarities
    def avg_similarity(indices):
        sims = []
        for i in indices:
            for j in indices:
                if i < j:
                    sims.append(cosine_similarity(vectors[i], vectors[j]))
        return np.mean(sims) if sims else 0

    # Cross-tier similarities
    def cross_similarity(tier1_indices, tier2_indices):
        sims = []
        for i in tier1_indices:
            for j in tier2_indices:
                sims.append(cosine_similarity(vectors[i], vectors[j]))
        return np.mean(sims)

    tier1 = [0, 1, 2]  # scanners
    tier2 = [3, 4, 5]  # normal
    tier3 = [6, 7, 8]  # attacks

    print(f"\nWithin-tier (should be HIGH):")
    print(f"  Scanners (10-100 pps): {avg_similarity(tier1):.3f}")
    print(f"  Normal (1k-10k pps):   {avg_similarity(tier2):.3f}")
    print(f"  Attacks (100k+ pps):   {avg_similarity(tier3):.3f}")

    print(f"\nCross-tier (should be LOWER):")
    print(f"  Scanners ↔ Normal:  {cross_similarity(tier1, tier2):.3f}")
    print(f"  Normal ↔ Attacks:   {cross_similarity(tier2, tier3):.3f}")
    print(f"  Scanners ↔ Attacks: {cross_similarity(tier1, tier3):.3f}")

    print("\nKEY INSIGHT: Same-magnitude traffic clusters together naturally,")
    print("without any hard-coded threshold configuration.")


def demo_find_similar_intensity():
    """Demonstrate finding attacks of similar intensity."""
    print_header("PART 3: Find Similar Intensity Attacks")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Reference attack we're investigating
    reference_attack = {
        "event_type": "ddos",
        "src_ip": "192.168.1.100",
        "rate_pps": {"$log": 50000},
        "bytes_per_sec": {"$log": 75000000},  # 75 MB/s
    }

    # Historical attacks with varying intensities
    historical_attacks = [
        # Similar intensity (should rank high)
        {
            "event_type": "ddos",
            "src_ip": "10.0.0.1",
            "rate_pps": {"$log": 45000},
            "bytes_per_sec": {"$log": 68000000},
            "label": "similar-1"
        },
        {
            "event_type": "ddos",
            "src_ip": "10.0.0.2",
            "rate_pps": {"$log": 60000},
            "bytes_per_sec": {"$log": 90000000},
            "label": "similar-2"
        },
        # Much smaller (should rank lower)
        {
            "event_type": "ddos",
            "src_ip": "10.0.0.3",
            "rate_pps": {"$log": 1000},
            "bytes_per_sec": {"$log": 1500000},
            "label": "small-attack"
        },
        # Much larger (should rank lower)
        {
            "event_type": "ddos",
            "src_ip": "10.0.0.4",
            "rate_pps": {"$log": 5000000},
            "bytes_per_sec": {"$log": 7500000000},
            "label": "massive-attack"
        },
        # Different event type but similar magnitude
        {
            "event_type": "exfiltration",
            "src_ip": "10.0.0.5",
            "rate_pps": {"$log": 55000},
            "bytes_per_sec": {"$log": 80000000},
            "label": "exfil-similar-mag"
        },
    ]

    # Encode reference
    ref_vec = enc.encode_data({k: v for k, v in reference_attack.items() if k != "label"})

    print("\nReference Attack:")
    print(f"  Type: {reference_attack['event_type']}")
    print(f"  Rate: {reference_attack['rate_pps']['$log']:,} pps")
    print(f"  Bandwidth: {reference_attack['bytes_per_sec']['$log']:,} bytes/sec")

    print("\nHistorical Attacks (searching for similar intensity):")
    print("-" * 60)

    # Calculate similarities and rank
    results = []
    for attack in historical_attacks:
        # Remove label for encoding
        attack_data = {k: v for k, v in attack.items() if k != "label"}
        vec = enc.encode_data(attack_data)
        sim = cosine_similarity(ref_vec, vec)
        results.append((attack["label"], attack, sim))

    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)

    for label, attack, sim in results:
        rate = attack["rate_pps"]["$log"]
        bw = attack["bytes_per_sec"]["$log"]
        print(f"  {sim:.3f}  {label:20s}  {rate:>10,} pps  {bw:>15,} B/s")

    print("\n" + "-" * 60)
    print("ANALYSIS:")
    print("-" * 60)
    print("\n'Similar intensity' attacks rank highest (similar-1, similar-2)")
    print("despite having different exact values. The exfiltration event also")
    print("ranks high because its MAGNITUDE is similar, even though the type differs.")
    print("\nSmall and massive attacks rank lower due to magnitude difference,")
    print("not because of hard-coded thresholds.")


def demo_linear_vs_log_encoding():
    """Demonstrate when to use linear vs log encoding."""
    print_header("PART 4: Linear vs Log Encoding Use Cases")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    print("\n--- Use Case: Response Time Monitoring ---\n")

    # Scenario: Latency spikes
    # Linear: +10ms is +10ms whether baseline is 10ms or 100ms
    # Log: 10ms→20ms (2x) is "worse" than 100ms→110ms (1.1x)

    latencies = [10, 20, 100, 110, 200]

    print("Latency values: 10ms, 20ms, 100ms, 110ms, 200ms")
    print("\nQuestion: Is 10→20ms more similar to 100→110ms or to 10→100ms?")

    # Linear encoding
    lin_vecs = [enc.encode_data({"$linear": v}) for v in latencies]

    # Log encoding
    log_vecs = [enc.encode_data({"$log": v}) for v in latencies]

    print("\nLINEAR ENCODING (equal differences = equal similarity):")
    print(f"  10ms ↔ 20ms (diff=10):   {cosine_similarity(lin_vecs[0], lin_vecs[1]):.3f}")
    print(f"  100ms ↔ 110ms (diff=10): {cosine_similarity(lin_vecs[2], lin_vecs[3]):.3f}")
    print(f"  10ms ↔ 100ms (diff=90):  {cosine_similarity(lin_vecs[0], lin_vecs[2]):.3f}")

    print("\nLOG ENCODING (equal ratios = equal similarity):")
    print(f"  10ms ↔ 20ms (2x):        {cosine_similarity(log_vecs[0], log_vecs[1]):.3f}")
    print(f"  100ms ↔ 110ms (1.1x):    {cosine_similarity(log_vecs[2], log_vecs[3]):.3f}")
    print(f"  100ms ↔ 200ms (2x):      {cosine_similarity(log_vecs[2], log_vecs[4]):.3f}")

    print("\n" + "-" * 60)
    print("WHEN TO USE EACH:")
    print("-" * 60)
    print("""
LINEAR ($linear):
  - Latency/response times (added delay matters)
  - Temperature (5°C change is 5°C change)
  - Offsets and positions
  - When absolute difference matters

LOG ($log):
  - Packet rates (10x is 10x regardless of baseline)
  - Byte counts (orders of magnitude)
  - Prices spanning wide ranges
  - Resource usage percentages
  - When proportional change matters
""")


def demo_practical_traffic_analysis():
    """Demonstrate practical traffic analysis with magnitude encoding."""
    print_header("PART 5: Practical Traffic Analysis Demo")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Simulated traffic log with magnitude-aware encoding
    traffic_log = [
        # Normal baseline traffic
        {"ts": 1, "src": "10.0.0.1", "dst": "web", "pps": {"$log": 500}, "label": "normal"},
        {"ts": 2, "src": "10.0.0.2", "dst": "web", "pps": {"$log": 800}, "label": "normal"},
        {"ts": 3, "src": "10.0.0.3", "dst": "web", "pps": {"$log": 1200}, "label": "normal"},
        # Reconnaissance (low rate, but notable)
        {"ts": 4, "src": "evil.1", "dst": "web", "pps": {"$log": 50}, "label": "recon"},
        {"ts": 5, "src": "evil.2", "dst": "web", "pps": {"$log": 30}, "label": "recon"},
        # Attack ramp-up
        {"ts": 6, "src": "botnet.1", "dst": "web", "pps": {"$log": 5000}, "label": "ramp"},
        {"ts": 7, "src": "botnet.2", "dst": "web", "pps": {"$log": 15000}, "label": "ramp"},
        # Full attack
        {"ts": 8, "src": "botnet.3", "dst": "web", "pps": {"$log": 100000}, "label": "attack"},
        {"ts": 9, "src": "botnet.4", "dst": "web", "pps": {"$log": 250000}, "label": "attack"},
        {"ts": 10, "src": "botnet.5", "dst": "web", "pps": {"$log": 180000}, "label": "attack"},
    ]

    print("\nTraffic log with $log-encoded rates:")
    print("-" * 60)
    for entry in traffic_log:
        pps = entry["pps"]["$log"]
        print(f"  ts={entry['ts']:2d}  {entry['label']:8s}  {entry['src']:12s} → {entry['dst']}  {pps:>10,} pps")

    # Build baseline from normal traffic
    normal_entries = [e for e in traffic_log if e["label"] == "normal"]
    normal_vecs = [enc.encode_data({k: v for k, v in e.items() if k not in ["label", "ts"]})
                   for e in normal_entries]
    baseline = enc.prototype(normal_vecs)

    print("\n\nBaseline built from normal traffic (ts 1-3)")
    print("Anomaly score = 1 - similarity to baseline")
    print("-" * 60)

    for entry in traffic_log:
        entry_data = {k: v for k, v in entry.items() if k not in ["label", "ts"]}
        vec = enc.encode_data(entry_data)
        sim = cosine_similarity(baseline, vec)
        anomaly_score = 1 - sim

        bar_len = int(anomaly_score * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)

        pps = entry["pps"]["$log"]
        print(f"  ts={entry['ts']:2d}  {entry['label']:8s}  {pps:>10,} pps  "
              f"anomaly={anomaly_score:.3f}  [{bar}]")

    print("\n" + "-" * 60)
    print("OBSERVATIONS:")
    print("-" * 60)
    print("""
1. Normal traffic (500-1200 pps) has LOW anomaly scores
2. Recon (30-50 pps) has MODERATE scores - different magnitude from normal
3. Attack traffic (100k+ pps) has HIGH scores - orders of magnitude different

The magnitude-aware encoding naturally separates traffic tiers without
explicit threshold configuration. The vector similarity captures the
"unusual magnitude" signal directly.
""")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" BATCH 15: MAGNITUDE-AWARE NUMERIC ENCODING")
    print(" Demonstrating $log and $linear markers for VSA/HDC")
    print("=" * 70)

    demo_string_vs_log_encoding()
    demo_traffic_magnitude_clustering()
    demo_find_similar_intensity()
    demo_linear_vs_log_encoding()
    demo_practical_traffic_analysis()

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print("""
NEW MARKERS:
  {"$log": value}           - Log10 encoding (equal ratios = equal similarity)
  {"$linear": value}        - Positional encoding (equal differences = equal similarity)
  {"$log": value, "$scale": n}  - Custom decay rate

KEY BENEFITS:
  1. Cluster by magnitude without thresholds
  2. "Find similar intensity" queries work naturally
  3. Proportional changes captured in similarity
  4. Choose encoding mode based on domain semantics

DEFAULT BEHAVIOR UNCHANGED:
  Bare numbers still encode as strings (exact matching).
  Use $log/$linear markers to opt-in to magnitude awareness.
""")


if __name__ == "__main__":
    main()
