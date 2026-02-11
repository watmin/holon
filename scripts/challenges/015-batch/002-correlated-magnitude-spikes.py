#!/usr/bin/env python3
"""
BATCH 15 - CHALLENGE 002: Correlated Magnitude Spikes

HYPOTHESIS: Log encoding enables detection of correlated magnitude changes
across heterogeneous metrics. When CPU, memory, and network all spike
proportionally, the pattern is detectable without normalizing scales.

This demo shows:
1. Detecting correlated spikes across different metric scales
2. Finding "similar behavior" patterns in time series
3. Magnitude-aware anomaly correlation

Run: ./scripts/run_with_venv.sh python scripts/challenges/015-batch/002-correlated-magnitude-spikes.py
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from holon import Encoder, VectorManager
from holon.distance import cosine_similarity


@dataclass
class SystemMetrics:
    """System metrics at a point in time."""
    timestamp: int
    cpu_percent: float
    memory_mb: float
    network_pps: float
    disk_iops: float
    label: str = "normal"


def print_header(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def encode_metrics_log(enc: Encoder, metrics: SystemMetrics) -> np.ndarray:
    """Encode system metrics using $log for magnitude awareness."""
    return enc.encode_data({
        "cpu_percent": {"$log": max(metrics.cpu_percent, 0.1)},
        "memory_mb": {"$log": max(metrics.memory_mb, 1)},
        "network_pps": {"$log": max(metrics.network_pps, 1)},
        "disk_iops": {"$log": max(metrics.disk_iops, 1)},
    })


def encode_metrics_string(enc: Encoder, metrics: SystemMetrics) -> np.ndarray:
    """Encode system metrics using default string encoding."""
    return enc.encode_data({
        "cpu_percent": metrics.cpu_percent,
        "memory_mb": metrics.memory_mb,
        "network_pps": metrics.network_pps,
        "disk_iops": metrics.disk_iops,
    })


def demo_correlated_spike_detection():
    """Demonstrate detecting correlated spikes across metrics."""
    print_header("PART 1: Correlated Spike Detection")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Time series with different patterns
    time_series: List[SystemMetrics] = [
        # Normal baseline (t=0-4)
        SystemMetrics(0, cpu_percent=10, memory_mb=2000, network_pps=500, disk_iops=100, label="normal"),
        SystemMetrics(1, cpu_percent=12, memory_mb=2100, network_pps=600, disk_iops=120, label="normal"),
        SystemMetrics(2, cpu_percent=8, memory_mb=1900, network_pps=450, disk_iops=90, label="normal"),
        SystemMetrics(3, cpu_percent=15, memory_mb=2200, network_pps=700, disk_iops=110, label="normal"),
        SystemMetrics(4, cpu_percent=11, memory_mb=2050, network_pps=550, disk_iops=105, label="normal"),

        # Correlated spike - everything goes up proportionally (t=5-7)
        SystemMetrics(5, cpu_percent=50, memory_mb=6000, network_pps=5000, disk_iops=500, label="corr-spike"),
        SystemMetrics(6, cpu_percent=70, memory_mb=8000, network_pps=8000, disk_iops=700, label="corr-spike"),
        SystemMetrics(7, cpu_percent=60, memory_mb=7000, network_pps=6000, disk_iops=600, label="corr-spike"),

        # CPU-only spike (t=8-9)
        SystemMetrics(8, cpu_percent=90, memory_mb=2100, network_pps=600, disk_iops=100, label="cpu-spike"),
        SystemMetrics(9, cpu_percent=85, memory_mb=2000, network_pps=550, disk_iops=110, label="cpu-spike"),

        # Network-only spike (t=10-11)
        SystemMetrics(10, cpu_percent=12, memory_mb=2100, network_pps=50000, disk_iops=100, label="net-spike"),
        SystemMetrics(11, cpu_percent=14, memory_mb=2200, network_pps=80000, disk_iops=120, label="net-spike"),

        # Recovery to normal (t=12-14)
        SystemMetrics(12, cpu_percent=13, memory_mb=2100, network_pps=600, disk_iops=100, label="normal"),
        SystemMetrics(13, cpu_percent=10, memory_mb=2000, network_pps=500, disk_iops=95, label="normal"),
        SystemMetrics(14, cpu_percent=9, memory_mb=1950, network_pps=480, disk_iops=90, label="normal"),
    ]

    print("\nTime series with different spike patterns:")
    print("-" * 80)
    print(f"{'t':>3}  {'Label':12s}  {'CPU%':>6}  {'Mem MB':>8}  {'Net PPS':>10}  {'Disk IOPS':>10}")
    print("-" * 80)
    for m in time_series:
        print(f"{m.timestamp:>3}  {m.label:12s}  {m.cpu_percent:>6.0f}  {m.memory_mb:>8.0f}  "
              f"{m.network_pps:>10.0f}  {m.disk_iops:>10.0f}")

    # Build baseline from normal samples
    normal_samples = [m for m in time_series if m.label == "normal"]
    normal_vecs = [encode_metrics_log(enc, m) for m in normal_samples]
    baseline = enc.prototype(normal_vecs)

    print("\n\nAnomaly Detection (with $log encoding):")
    print("-" * 80)
    print(f"{'t':>3}  {'Label':12s}  {'Anomaly':>8}  {'Pattern'}")
    print("-" * 80)

    for m in time_series:
        vec = encode_metrics_log(enc, m)
        sim = cosine_similarity(baseline, vec)
        anomaly = 1 - sim

        bar_len = int(anomaly * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"{m.timestamp:>3}  {m.label:12s}  {anomaly:>8.3f}  [{bar}]")

    # Now find similar patterns
    print("\n\n--- Pattern Similarity Analysis ---")
    print("\nComparing spike patterns to each other:")

    # Reference: first correlated spike
    ref_corr = encode_metrics_log(enc, time_series[5])  # t=5, corr-spike
    ref_cpu = encode_metrics_log(enc, time_series[8])   # t=8, cpu-spike
    ref_net = encode_metrics_log(enc, time_series[10])  # t=10, net-spike

    print(f"\nReference patterns:")
    print(f"  Correlated spike (t=5): CPU=50%, Mem=6000MB, Net=5000pps, Disk=500")
    print(f"  CPU-only spike (t=8):   CPU=90%, Mem=2100MB, Net=600pps, Disk=100")
    print(f"  Net-only spike (t=10):  CPU=12%, Mem=2100MB, Net=50000pps, Disk=100")

    print(f"\nPattern similarities:")
    print(f"  Corr-spike(t=5) ↔ Corr-spike(t=6): {cosine_similarity(ref_corr, encode_metrics_log(enc, time_series[6])):.3f}")
    print(f"  Corr-spike(t=5) ↔ CPU-spike(t=8):  {cosine_similarity(ref_corr, ref_cpu):.3f}")
    print(f"  Corr-spike(t=5) ↔ Net-spike(t=10): {cosine_similarity(ref_corr, ref_net):.3f}")
    print(f"  CPU-spike(t=8) ↔ CPU-spike(t=9):   {cosine_similarity(ref_cpu, encode_metrics_log(enc, time_series[9])):.3f}")
    print(f"  Net-spike(t=10) ↔ Net-spike(t=11): {cosine_similarity(ref_net, encode_metrics_log(enc, time_series[11])):.3f}")

    print("\n" + "-" * 60)
    print("KEY INSIGHT: Same-type spikes have higher similarity than cross-type,")
    print("even though all have elevated anomaly scores.")


def demo_magnitude_ratio_preservation():
    """Demonstrate that log encoding preserves magnitude ratios."""
    print_header("PART 2: Magnitude Ratio Preservation")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Create metrics with same RATIOS but different absolute scales
    # Pattern A: 10x spike across all metrics
    baseline_a = SystemMetrics(0, cpu_percent=5, memory_mb=1000, network_pps=100, disk_iops=50, label="base-A")
    spike_a = SystemMetrics(1, cpu_percent=50, memory_mb=10000, network_pps=1000, disk_iops=500, label="spike-A")

    # Pattern B: Same 10x ratio but at different scale
    baseline_b = SystemMetrics(2, cpu_percent=20, memory_mb=4000, network_pps=500, disk_iops=200, label="base-B")
    spike_b = SystemMetrics(3, cpu_percent=200, memory_mb=40000, network_pps=5000, disk_iops=2000, label="spike-B")

    # Pattern C: 2x spike (different ratio)
    baseline_c = SystemMetrics(4, cpu_percent=10, memory_mb=2000, network_pps=200, disk_iops=100, label="base-C")
    spike_c = SystemMetrics(5, cpu_percent=20, memory_mb=4000, network_pps=400, disk_iops=200, label="spike-C")

    print("\nPatterns with different scales but same/different ratios:")
    print("-" * 70)
    print("Pattern A: 5% → 50% CPU (10x), all metrics 10x")
    print("Pattern B: 20% → 200% CPU (10x), all metrics 10x (different absolute scale)")
    print("Pattern C: 10% → 20% CPU (2x), all metrics 2x")

    # Compute deltas (spike - baseline) using difference primitive
    vec_base_a = encode_metrics_log(enc, baseline_a)
    vec_spike_a = encode_metrics_log(enc, spike_a)
    delta_a = enc.difference(vec_base_a, vec_spike_a)

    vec_base_b = encode_metrics_log(enc, baseline_b)
    vec_spike_b = encode_metrics_log(enc, spike_b)
    delta_b = enc.difference(vec_base_b, vec_spike_b)

    vec_base_c = encode_metrics_log(enc, baseline_c)
    vec_spike_c = encode_metrics_log(enc, spike_c)
    delta_c = enc.difference(vec_base_c, vec_spike_c)

    print("\n\nDelta vector similarities (comparing the CHANGE patterns):")
    print("-" * 60)
    print(f"  Delta-A ↔ Delta-B (both 10x): {cosine_similarity(delta_a, delta_b):.3f}")
    print(f"  Delta-A ↔ Delta-C (10x vs 2x): {cosine_similarity(delta_a, delta_c):.3f}")
    print(f"  Delta-B ↔ Delta-C (10x vs 2x): {cosine_similarity(delta_b, delta_c):.3f}")

    print("\n" + "-" * 60)
    print("KEY INSIGHT: Patterns with same proportional change (10x) are more")
    print("similar to each other than to patterns with different ratios (2x),")
    print("regardless of absolute scale.")


def demo_incident_clustering():
    """Demonstrate clustering incidents by behavior similarity."""
    print_header("PART 3: Incident Clustering by Behavior")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Historical incidents with different characteristics
    incidents = [
        # DDoS attacks (high network, moderate CPU)
        {"id": "INC-001", "type": "ddos", "cpu": 40, "mem": 3000, "net": 500000, "disk": 200},
        {"id": "INC-002", "type": "ddos", "cpu": 55, "mem": 4000, "net": 800000, "disk": 250},
        {"id": "INC-003", "type": "ddos", "cpu": 35, "mem": 2800, "net": 350000, "disk": 180},

        # Crypto mining (high CPU, moderate everything else)
        {"id": "INC-004", "type": "mining", "cpu": 95, "mem": 4000, "net": 1000, "disk": 300},
        {"id": "INC-005", "type": "mining", "cpu": 98, "mem": 5000, "net": 1500, "disk": 400},
        {"id": "INC-006", "type": "mining", "cpu": 92, "mem": 3500, "net": 800, "disk": 250},

        # Data exfiltration (high disk + network, low CPU)
        {"id": "INC-007", "type": "exfil", "cpu": 15, "mem": 2000, "net": 100000, "disk": 5000},
        {"id": "INC-008", "type": "exfil", "cpu": 20, "mem": 2500, "net": 150000, "disk": 8000},
        {"id": "INC-009", "type": "exfil", "cpu": 12, "mem": 1800, "net": 80000, "disk": 4000},

        # Memory leak (growing memory, stable everything else)
        {"id": "INC-010", "type": "memleak", "cpu": 25, "mem": 15000, "net": 500, "disk": 100},
        {"id": "INC-011", "type": "memleak", "cpu": 30, "mem": 20000, "net": 600, "disk": 120},
        {"id": "INC-012", "type": "memleak", "cpu": 22, "mem": 12000, "net": 450, "disk": 90},
    ]

    def encode_incident(inc):
        return enc.encode_data({
            "cpu": {"$log": max(inc["cpu"], 1)},
            "mem": {"$log": max(inc["mem"], 1)},
            "net": {"$log": max(inc["net"], 1)},
            "disk": {"$log": max(inc["disk"], 1)},
        })

    # Encode all incidents
    incident_vecs = [(inc, encode_incident(inc)) for inc in incidents]

    print("\nHistorical incidents:")
    print("-" * 70)
    print(f"{'ID':10s}  {'Type':10s}  {'CPU%':>6}  {'Mem MB':>8}  {'Net PPS':>10}  {'Disk IOPS':>10}")
    print("-" * 70)
    for inc in incidents:
        print(f"{inc['id']:10s}  {inc['type']:10s}  {inc['cpu']:>6}  {inc['mem']:>8}  "
              f"{inc['net']:>10}  {inc['disk']:>10}")

    # New incident to classify
    new_incident = {"id": "INC-NEW", "type": "unknown", "cpu": 45, "mem": 3500, "net": 450000, "disk": 220}
    new_vec = encode_incident(new_incident)

    print(f"\n\nNew incident to classify:")
    print(f"  {new_incident['id']}: CPU={new_incident['cpu']}%, Mem={new_incident['mem']}MB, "
          f"Net={new_incident['net']}pps, Disk={new_incident['disk']}iops")

    print("\nSimilarity to historical incidents:")
    print("-" * 60)

    # Calculate similarities and sort
    results = []
    for inc, vec in incident_vecs:
        sim = cosine_similarity(new_vec, vec)
        results.append((inc, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    for inc, sim in results:
        print(f"  {sim:.3f}  {inc['id']:10s}  ({inc['type']})")

    # Group by type
    print("\n\nAverage similarity by incident type:")
    print("-" * 40)
    type_sims = {}
    for inc, sim in results:
        t = inc["type"]
        if t not in type_sims:
            type_sims[t] = []
        type_sims[t].append(sim)

    for t, sims in sorted(type_sims.items(), key=lambda x: -np.mean(x[1])):
        avg = np.mean(sims)
        print(f"  {t:10s}: {avg:.3f}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION: New incident most similar to DDoS attacks based on")
    print("magnitude profile (high network, moderate CPU), not exact values.")


def demo_string_vs_log_incident_matching():
    """Compare incident matching with string vs log encoding."""
    print_header("PART 4: String vs Log Encoding for Incident Matching")

    vm = VectorManager(dimensions=4096)
    enc = Encoder(vm)

    # Reference incident
    ref = {"cpu": 50, "mem": 5000, "net": 100000, "disk": 500}

    # Similar magnitude but different exact values
    similar = {"cpu": 45, "mem": 4500, "net": 90000, "disk": 450}

    # Same exact values (edge case)
    exact = {"cpu": 50, "mem": 5000, "net": 100000, "disk": 500}

    # Very different magnitude
    different = {"cpu": 5, "mem": 500, "net": 1000, "disk": 50}

    print("\nReference incident: CPU=50%, Mem=5000MB, Net=100000pps, Disk=500iops")
    print("\nComparison incidents:")
    print("  Similar:   CPU=45%, Mem=4500MB, Net=90000pps, Disk=450iops (10% lower)")
    print("  Exact:     CPU=50%, Mem=5000MB, Net=100000pps, Disk=500iops (identical)")
    print("  Different: CPU=5%, Mem=500MB, Net=1000pps, Disk=50iops (10x lower)")

    # String encoding
    def encode_string(inc):
        return enc.encode_data(inc)

    # Log encoding
    def encode_log(inc):
        return enc.encode_data({
            "cpu": {"$log": max(inc["cpu"], 1)},
            "mem": {"$log": max(inc["mem"], 1)},
            "net": {"$log": max(inc["net"], 1)},
            "disk": {"$log": max(inc["disk"], 1)},
        })

    ref_str = encode_string(ref)
    ref_log = encode_log(ref)

    print("\n\nString Encoding Similarities:")
    print("-" * 40)
    print(f"  Ref ↔ Similar:   {cosine_similarity(ref_str, encode_string(similar)):.3f}")
    print(f"  Ref ↔ Exact:     {cosine_similarity(ref_str, encode_string(exact)):.3f}")
    print(f"  Ref ↔ Different: {cosine_similarity(ref_str, encode_string(different)):.3f}")

    print("\n\nLog Encoding Similarities:")
    print("-" * 40)
    print(f"  Ref ↔ Similar:   {cosine_similarity(ref_log, encode_log(similar)):.3f}")
    print(f"  Ref ↔ Exact:     {cosine_similarity(ref_log, encode_log(exact)):.3f}")
    print(f"  Ref ↔ Different: {cosine_similarity(ref_log, encode_log(different)):.3f}")

    print("\n" + "-" * 60)
    print("OBSERVATION:")
    print("- String encoding only matches EXACT values (1.0 for identical)")
    print("- Log encoding captures SIMILAR magnitudes (0.95+ for 10% diff)")
    print("- Log encoding clearly separates different magnitudes (lower sim for 10x)")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" BATCH 15 - CHALLENGE 002: Correlated Magnitude Spikes")
    print(" Multi-field magnitude correlation with $log encoding")
    print("=" * 70)

    demo_correlated_spike_detection()
    demo_magnitude_ratio_preservation()
    demo_incident_clustering()
    demo_string_vs_log_incident_matching()

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print("""
KEY FINDINGS:

1. CORRELATED SPIKE DETECTION
   Log encoding enables detecting when multiple metrics spike together
   proportionally, even at different absolute scales.

2. RATIO PRESERVATION
   The difference between baseline and spike encodes the RATIO of change,
   not the absolute change. 10x spikes cluster together regardless of scale.

3. BEHAVIOR-BASED CLUSTERING
   Incidents cluster by behavior profile (high net = DDoS, high CPU = mining)
   without requiring exact value matching.

4. STRING VS LOG TRADEOFFS
   - String: Use for exact matching (IDs, codes, status)
   - Log: Use for magnitude-aware similarity (rates, counts, sizes)

PRACTICAL APPLICATIONS:
   - Anomaly correlation across heterogeneous metrics
   - Incident classification by behavior profile
   - "Find similar" queries across order-of-magnitude ranges
   - Detecting proportional spikes in distributed systems
""")


if __name__ == "__main__":
    main()
