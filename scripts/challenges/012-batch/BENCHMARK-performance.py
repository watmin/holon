#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 BENCHMARK: Single-Core Performance
=============================================================================

Measures packets-per-second throughput for the zero-hardcode detector.

We're measuring:
1. Packet encoding time (encode_data + encode_scalar_log)
2. Similarity computation time
3. Detection logic overhead
4. Total end-to-end throughput

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/BENCHMARK-performance.py
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, Any
from collections import deque
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


DIMENSIONS = 4096
WARMUP_PACKETS = 200
DECAY = 0.98


class FrozenBaseline:
    """Frozen z-score baseline."""

    def __init__(self):
        self.samples = []
        self.frozen = False
        self.mean = 0.0
        self.std = 1.0

    def observe(self, value: float):
        if not self.frozen:
            self.samples.append(value)

    def freeze(self):
        self.frozen = True
        if self.samples:
            self.mean = np.mean(self.samples)
            self.std = max(
                np.std(self.samples) if len(self.samples) > 1 else 0.05, 0.02
            )

    def z_score(self, value: float) -> float:
        return (value - self.mean) / self.std


class BenchmarkDetector:
    """Minimal detector for benchmarking."""

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        self.rate_accum = self.encoder.create_accumulator()
        self._rate_norm = None
        self.rate_baseline = FrozenBaseline()

        self.pattern_accum = self.encoder.create_accumulator()
        self._pattern_norm = None
        self.pattern_baseline = FrozenBaseline()

        self.recent_pattern = self.encoder.create_accumulator()

        self.in_anomaly_state = False
        self.anomaly_window = deque(maxlen=15)
        self.consecutive_normal = 0

    def process(self, packet: Dict[str, Any], pps: float) -> bool:
        """Process packet and return anomaly status."""
        self.packet_count += 1

        # Encode
        packet_vec = self.encoder.encode_data(packet)
        rate_vec = self.store.encode_scalar_log(pps)

        if self.packet_count <= self.warmup_packets:
            # Warmup phase
            self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
            self.pattern_accum = self.encoder.accumulate(
                self.pattern_accum, packet_vec
            )

            if self.packet_count > 100:
                temp_rate = self.encoder.normalize_accumulator(self.rate_accum)
                temp_pattern = self.encoder.normalize_accumulator(self.pattern_accum)
                rate_sim = self.store.similarity(rate_vec, temp_rate, metric="cosine")
                pattern_sim = self.store.similarity(
                    packet_vec, temp_pattern, metric="cosine"
                )
                self.rate_baseline.observe(rate_sim)
                self.pattern_baseline.observe(pattern_sim)

            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                self._rate_norm = self.encoder.normalize_accumulator(self.rate_accum)
                self._pattern_norm = self.encoder.normalize_accumulator(
                    self.pattern_accum
                )
                self.recent_pattern = self.pattern_accum.copy()
                self.rate_baseline.freeze()
                self.pattern_baseline.freeze()

            return False

        # Detection phase
        self.recent_pattern = DECAY * self.recent_pattern + packet_vec.astype(
            np.float64
        )

        rate_sim = self.store.similarity(rate_vec, self._rate_norm, metric="cosine")
        pattern_sim = self.store.similarity(
            packet_vec, self._pattern_norm, metric="cosine"
        )

        rate_z = self.rate_baseline.z_score(rate_sim)
        pattern_z = self.pattern_baseline.z_score(pattern_sim)

        rate_anomalous = rate_z < -2.5
        pattern_anomalous = pattern_z < -2.0
        rate_confirms = rate_z < -0.5

        raw_anomaly = rate_anomalous or (pattern_anomalous and rate_confirms)

        if raw_anomaly:
            self.consecutive_normal = 0
        else:
            self.consecutive_normal += 1

        if self.consecutive_normal >= 5:
            for _ in range(3):
                self.anomaly_window.append(0)
        else:
            self.anomaly_window.append(1 if raw_anomaly else 0)

        fraction = (
            sum(self.anomaly_window) / len(self.anomaly_window)
            if self.anomaly_window
            else 0
        )

        if not self.in_anomaly_state:
            if fraction > 0.5:
                self.in_anomaly_state = True
        else:
            if fraction < 0.2:
                self.in_anomaly_state = False

        return self.in_anomaly_state


def gen_packet(rng: random.Random) -> dict:
    """Generate a random packet."""
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]
    if proto == "TCP":
        return {
            "protocol": "TCP",
            "src_port": rng.randint(49152, 65535),
            "dst_port": rng.choice([80, 443, 8080, 22]),
            "flags": rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[
                0
            ],
            "payload_size": rng.randint(0, 1500),
        }
    elif proto == "UDP":
        return {
            "protocol": "UDP",
            "src_port": rng.randint(49152, 65535),
            "dst_port": rng.choice([53, 443, 123]),
            "payload_size": rng.randint(20, 512),
        }
    else:
        return {
            "protocol": "ICMP",
            "icmp_type": rng.choice([0, 8]),
            "payload_size": 64,
        }


def benchmark_component(name: str, func, iterations: int = 10000):
    """Benchmark a single component."""
    # Warmup
    for _ in range(100):
        func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    us_per_op = (elapsed / iterations) * 1_000_000

    return ops_per_sec, us_per_op


def main():
    print("=" * 75)
    print("BATCH 012 BENCHMARK: Single-Core Performance")
    print("=" * 75)
    print(f"\n  Dimensions: {DIMENSIONS}")
    print(f"  Warmup packets: {WARMUP_PACKETS}")

    rng = random.Random(42)
    store = CPUStore(dimensions=DIMENSIONS)
    encoder = store.encoder

    # Pre-generate packets for consistent benchmarking
    packets = [gen_packet(rng) for _ in range(10000)]
    packet_idx = [0]

    def get_packet():
        p = packets[packet_idx[0] % len(packets)]
        packet_idx[0] += 1
        return p

    # Component benchmarks
    print("\n" + "-" * 75)
    print("COMPONENT BENCHMARKS")
    print("-" * 75)

    results = {}

    # 1. encode_data
    def bench_encode_data():
        return encoder.encode_data(get_packet())

    ops, us = benchmark_component("encode_data", bench_encode_data)
    results["encode_data"] = (ops, us)
    print(f"\n  encode_data():        {ops:>10,.0f} ops/sec  ({us:>6.1f} µs/op)")

    # 2. encode_scalar_log
    rate = 100.0

    def bench_encode_scalar():
        return store.encode_scalar_log(rate)

    ops, us = benchmark_component("encode_scalar_log", bench_encode_scalar)
    results["encode_scalar_log"] = (ops, us)
    print(f"  encode_scalar_log():  {ops:>10,.0f} ops/sec  ({us:>6.1f} µs/op)")

    # 3. similarity
    vec1 = encoder.encode_data(packets[0])
    vec2 = encoder.encode_data(packets[1])

    def bench_similarity():
        return store.similarity(vec1, vec2, metric="cosine")

    ops, us = benchmark_component("similarity", bench_similarity)
    results["similarity"] = (ops, us)
    print(f"  similarity():         {ops:>10,.0f} ops/sec  ({us:>6.1f} µs/op)")

    # 4. accumulate
    accum = encoder.create_accumulator()

    def bench_accumulate():
        nonlocal accum
        accum = encoder.accumulate(accum, vec1)
        return accum

    ops, us = benchmark_component("accumulate", bench_accumulate)
    results["accumulate"] = (ops, us)
    print(f"  accumulate():         {ops:>10,.0f} ops/sec  ({us:>6.1f} µs/op)")

    # 5. normalize_accumulator
    def bench_normalize():
        return encoder.normalize_accumulator(accum)

    ops, us = benchmark_component("normalize_accumulator", bench_normalize)
    results["normalize"] = (ops, us)
    print(f"  normalize_accum():    {ops:>10,.0f} ops/sec  ({us:>6.1f} µs/op)")

    # End-to-end detector benchmark
    print("\n" + "-" * 75)
    print("END-TO-END DETECTOR BENCHMARK")
    print("-" * 75)

    # Test with different packet counts
    for n_packets in [1000, 5000, 10000, 50000]:
        detector = BenchmarkDetector(warmup_packets=min(200, n_packets // 5))

        # Pre-generate all packets
        test_packets = [gen_packet(rng) for _ in range(n_packets)]

        # Time the full run
        start = time.perf_counter()
        for pkt in test_packets:
            detector.process(pkt, 100.0)
        elapsed = time.perf_counter() - start

        pps = n_packets / elapsed
        us_per_pkt = (elapsed / n_packets) * 1_000_000

        print(f"\n  {n_packets:>6,} packets: {pps:>10,.0f} pkt/sec  ({us_per_pkt:>6.1f} µs/pkt)")

    # Warmup vs detection phase comparison
    print("\n" + "-" * 75)
    print("WARMUP vs DETECTION PHASE")
    print("-" * 75)

    detector = BenchmarkDetector(warmup_packets=1000)
    test_packets = [gen_packet(rng) for _ in range(5000)]

    # Warmup phase timing
    start = time.perf_counter()
    for i, pkt in enumerate(test_packets[:1000]):
        detector.process(pkt, 100.0)
    warmup_elapsed = time.perf_counter() - start
    warmup_pps = 1000 / warmup_elapsed

    # Detection phase timing
    start = time.perf_counter()
    for pkt in test_packets[1000:]:
        detector.process(pkt, 100.0)
    detect_elapsed = time.perf_counter() - start
    detect_pps = 4000 / detect_elapsed

    print(f"\n  Warmup phase:    {warmup_pps:>10,.0f} pkt/sec")
    print(f"  Detection phase: {detect_pps:>10,.0f} pkt/sec")

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    # Calculate theoretical max based on components
    encode_us = results["encode_data"][1] + results["encode_scalar_log"][1]
    sim_us = results["similarity"][1] * 2  # Two similarity calls
    total_component_us = encode_us + sim_us

    print(f"""
  COMPONENT BREAKDOWN (per packet):
    Encoding (data + rate):  {encode_us:>6.1f} µs
    Similarity (2x):         {sim_us:>6.1f} µs
    Other overhead:          ~{us_per_pkt - total_component_us:>5.1f} µs
    ─────────────────────────────────
    Total:                   {us_per_pkt:>6.1f} µs/packet

  THROUGHPUT (single core):
    Detection phase:         {detect_pps:>10,.0f} packets/sec
    Warmup phase:            {warmup_pps:>10,.0f} packets/sec

  SAMPLING RATES:
    At 1 Gbps (~1.5M pps):   1:{1_500_000/detect_pps:.0f} sample rate
    At 10 Gbps (~15M pps):   1:{15_000_000/detect_pps:.0f} sample rate
    At 100 Gbps (~150M pps): 1:{150_000_000/detect_pps:.0f} sample rate

  MULTI-CORE PROJECTION (linear scaling assumed):
    4 cores:  ~{detect_pps * 4:>10,.0f} packets/sec
    8 cores:  ~{detect_pps * 8:>10,.0f} packets/sec
    16 cores: ~{detect_pps * 16:>10,.0f} packets/sec
    """)


if __name__ == "__main__":
    main()
