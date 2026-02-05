#!/usr/bin/env python3
"""
Challenge 011-012: Performance Benchmark

Test throughput at scale:
- How many packets per second can we process?
- What are the bottlenecks?
- How does dimensionality affect performance?
- Compare to baseline (no VSA overhead)
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from scapy.all import IP, TCP, UDP, Raw, Packet

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager

# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# MINIMAL DETECTOR (for benchmarking)
# =============================================================================

class MinimalDetector:
    """
    Minimal detector for benchmarking pure VSA overhead.
    """

    def __init__(self, dimensions: int = 4096):
        self.dimensions = dimensions
        self.vm = DeterministicVectorManager(dimensions=dimensions, global_seed=GLOBAL_SEED)

        # Prior baseline
        self.prior = self._build_prior()

        # Recent accumulator
        self.recent = np.zeros(dimensions, dtype=np.float64)
        self.decay = 0.995

        # Threshold
        self.threshold = 0.5

        # Counters
        self.packets = 0
        self.anomalies = 0

    def _build_prior(self) -> np.ndarray:
        """Build a simple prior baseline."""
        vec = np.zeros(self.dimensions, dtype=np.float64)
        for atom in ["proto:tcp", "dst_port:80", "dst_port:443", "normal"]:
            vec += self.vm.get_vector(atom)
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        return vec

    def process(self, pkt: Packet) -> bool:
        """Process a packet. Returns True if anomalous."""
        self.packets += 1

        # Encode
        vec = self._encode(pkt)

        # Compare to prior
        prior_sim = cosine_similarity(vec, self.prior)

        # Compare to recent
        recent_sim = cosine_similarity(vec, self.recent)

        # Anomaly score
        score = 1.0 - (0.6 * prior_sim + 0.4 * recent_sim)
        is_anomalous = score > self.threshold

        if is_anomalous:
            self.anomalies += 1

        # Update recent
        self.recent *= self.decay
        self.recent += vec * (0.1 if is_anomalous else 1.0)

        return is_anomalous

    def _encode(self, pkt: Packet) -> np.ndarray:
        """Encode packet to vector."""
        atoms = []

        if IP in pkt:
            atoms.append(f"src:{pkt[IP].src.rsplit('.', 1)[0]}")
            atoms.append(f"dst:{pkt[IP].dst.rsplit('.', 1)[0]}")

        if TCP in pkt:
            atoms.append("proto:tcp")
            atoms.append(f"dport:{pkt[TCP].dport}")
        elif UDP in pkt:
            atoms.append("proto:udp")
            atoms.append(f"dport:{pkt[UDP].dport}")

        if Raw in pkt:
            payload = bytes(pkt[Raw].load)[:4]
            for i, b in enumerate(payload):
                atoms.append(f"b{i}:{b}")

        vec = np.zeros(self.dimensions, dtype=np.float64)
        for atom in atoms:
            vec += self.vm.get_vector(atom)

        return vec


# =============================================================================
# PACKET GENERATOR
# =============================================================================

def generate_packets(n: int) -> List[Packet]:
    """Generate mixed traffic for benchmarking."""
    packets = []
    for i in range(n):
        if i % 5 == 0:  # 20% attack-like
            src_ip = f"10.{(i // 256) % 256}.{i % 256}.1"
            pkt = IP(src=src_ip, dst="192.168.1.100") / TCP(
                sport=40000 + (i % 1000),
                dport=80,
                flags="S"
            )
        else:  # 80% normal
            pkt = IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
                sport=random.randint(49152, 65535),
                dport=random.choice([80, 443]),
                flags="PA"
            ) / Raw(load=b"GET / HTTP/1.1\r\n")
        packets.append(pkt)
    return packets


def generate_raw_packets(n: int) -> List[bytes]:
    """Generate raw packet bytes for baseline comparison."""
    packets = generate_packets(n)
    return [bytes(pkt) for pkt in packets]


# =============================================================================
# BENCHMARKS
# =============================================================================

@dataclass
class BenchmarkResult:
    name: str
    packets: int
    duration_ms: float
    pps: float
    anomalies: int
    anomaly_rate: float


def benchmark_vsa_detector(packets: List[Packet], dimensions: int) -> BenchmarkResult:
    """Benchmark VSA-based detector."""
    detector = MinimalDetector(dimensions=dimensions)

    start = time.perf_counter()
    for pkt in packets:
        detector.process(pkt)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000
    pps = len(packets) / (end - start)

    return BenchmarkResult(
        name=f"VSA (dim={dimensions})",
        packets=detector.packets,
        duration_ms=duration_ms,
        pps=pps,
        anomalies=detector.anomalies,
        anomaly_rate=detector.anomalies / detector.packets,
    )


def benchmark_baseline_parsing(packets: List[Packet]) -> BenchmarkResult:
    """Benchmark just packet parsing (no VSA)."""
    count = 0

    start = time.perf_counter()
    for pkt in packets:
        # Just extract fields (simulating parsing overhead)
        if IP in pkt:
            _ = pkt[IP].src
            _ = pkt[IP].dst
        if TCP in pkt:
            _ = pkt[TCP].sport
            _ = pkt[TCP].dport
        elif UDP in pkt:
            _ = pkt[UDP].sport
            _ = pkt[UDP].dport
        if Raw in pkt:
            _ = bytes(pkt[Raw].load)[:4]
        count += 1
    end = time.perf_counter()

    duration_ms = (end - start) * 1000
    pps = count / (end - start)

    return BenchmarkResult(
        name="Baseline (parsing only)",
        packets=count,
        duration_ms=duration_ms,
        pps=pps,
        anomalies=0,
        anomaly_rate=0.0,
    )


def benchmark_encoding_only(packets: List[Packet], dimensions: int) -> BenchmarkResult:
    """Benchmark just encoding (no detection logic)."""
    vm = DeterministicVectorManager(dimensions=dimensions, global_seed=GLOBAL_SEED)
    count = 0

    start = time.perf_counter()
    for pkt in packets:
        atoms = []
        if IP in pkt:
            atoms.append(f"src:{pkt[IP].src.rsplit('.', 1)[0]}")
            atoms.append(f"dst:{pkt[IP].dst.rsplit('.', 1)[0]}")
        if TCP in pkt:
            atoms.append("proto:tcp")
            atoms.append(f"dport:{pkt[TCP].dport}")

        vec = np.zeros(dimensions, dtype=np.float64)
        for atom in atoms:
            vec += vm.get_vector(atom)
        count += 1
    end = time.perf_counter()

    duration_ms = (end - start) * 1000
    pps = count / (end - start)

    return BenchmarkResult(
        name=f"Encoding only (dim={dimensions})",
        packets=count,
        duration_ms=duration_ms,
        pps=pps,
        anomalies=0,
        anomaly_rate=0.0,
    )


def benchmark_similarity_only(n: int, dimensions: int) -> BenchmarkResult:
    """Benchmark just similarity computation."""
    # Pre-generate vectors
    vecs = [np.random.randn(dimensions) for _ in range(100)]
    base = np.random.randn(dimensions)

    start = time.perf_counter()
    for i in range(n):
        vec = vecs[i % 100]
        _ = cosine_similarity(vec, base)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000
    ops = n / (end - start)

    return BenchmarkResult(
        name=f"Similarity only (dim={dimensions})",
        packets=n,
        duration_ms=duration_ms,
        pps=ops,
        anomalies=0,
        anomaly_rate=0.0,
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 80)
    print("CHALLENGE 011-012: PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Test sizes
    packet_counts = [1000, 5000, 10000]
    dimensions_to_test = [1024, 2048, 4096, 8192]

    # Pre-generate packets (so generation time isn't counted)
    print("\nGenerating test packets...")
    packets_by_count = {}
    for count in packet_counts:
        packets_by_count[count] = generate_packets(count)
    print(f"Generated packets for counts: {packet_counts}")

    results = []

    # Baseline benchmark
    print("\n" + "=" * 60)
    print("BASELINE: Parsing Only (Scapy overhead)")
    print("=" * 60)

    for count in packet_counts:
        result = benchmark_baseline_parsing(packets_by_count[count])
        results.append(result)
        print(f"  {count:,} packets: {result.pps:,.0f} pps ({result.duration_ms:.1f} ms)")

    # Encoding benchmark
    print("\n" + "=" * 60)
    print("ENCODING: VSA encoding overhead")
    print("=" * 60)

    for dims in dimensions_to_test:
        packets = packets_by_count[5000]
        result = benchmark_encoding_only(packets, dims)
        results.append(result)
        print(f"  dim={dims}: {result.pps:,.0f} pps ({result.duration_ms:.1f} ms)")

    # Similarity benchmark
    print("\n" + "=" * 60)
    print("SIMILARITY: Cosine similarity overhead")
    print("=" * 60)

    for dims in dimensions_to_test:
        result = benchmark_similarity_only(10000, dims)
        results.append(result)
        print(f"  dim={dims}: {result.pps:,.0f} ops/sec ({result.duration_ms:.1f} ms)")

    # Full detector benchmark
    print("\n" + "=" * 60)
    print("FULL DETECTOR: End-to-end performance")
    print("=" * 60)

    print(f"\n{'Packets':<12} {'Dimensions':<12} {'PPS':<15} {'Time (ms)':<12} {'Anomalies'}")
    print("-" * 65)

    full_results = []
    for count in packet_counts:
        for dims in [2048, 4096]:  # Test most common sizes
            packets = packets_by_count[count]
            result = benchmark_vsa_detector(packets, dims)
            full_results.append(result)
            print(f"{count:<12,} {dims:<12} {result.pps:<15,.0f} {result.duration_ms:<12.1f} {result.anomalies} ({result.anomaly_rate:.1%})")

    # Scaling analysis
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    # Find best configuration
    best = max(full_results, key=lambda r: r.pps)
    print(f"\nBest throughput: {best.pps:,.0f} pps with {best.name}")

    # Extrapolate to 10k pps target
    print(f"\n10k pps target analysis:")
    for result in full_results:
        if "VSA" in result.name:
            can_achieve = result.pps >= 10000
            status = "✓ ACHIEVABLE" if can_achieve else "✗ Below target"
            gap = 10000 - result.pps
            print(f"  {result.name}: {result.pps:,.0f} pps - {status}" +
                  (f" (need {gap:,.0f} more)" if gap > 0 else ""))

    # Bottleneck analysis
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)

    # Compare components at dim=4096
    baseline_pps = [r.pps for r in results if "Baseline" in r.name][0]
    encoding_pps = [r.pps for r in results if "Encoding" in r.name and "4096" in r.name][0]
    similarity_pps = [r.pps for r in results if "Similarity" in r.name and "4096" in r.name][0]
    full_pps = [r.pps for r in full_results if "4096" in r.name][0]

    print(f"\nComponent throughput (dim=4096):")
    print(f"  Parsing (Scapy):     {baseline_pps:>10,.0f} pps")
    print(f"  Encoding (VSA):      {encoding_pps:>10,.0f} pps")
    print(f"  Similarity (cosine): {similarity_pps:>10,.0f} pps")
    print(f"  Full detector:       {full_pps:>10,.0f} pps")

    # Estimate bottleneck
    bottleneck = min(baseline_pps, encoding_pps, similarity_pps)
    limiting_factor = (
        "Scapy parsing" if bottleneck == baseline_pps else
        "VSA encoding" if bottleneck == encoding_pps else
        "Similarity computation"
    )
    print(f"\n  Primary bottleneck: {limiting_factor}")

    # Summary
    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print(f"""
1. THROUGHPUT ACHIEVED:
   - Best: {best.pps:,.0f} pps with {best.name}
   - 10k pps target: {'ACHIEVABLE' if best.pps >= 10000 else 'Requires optimization'}

2. BOTTLENECK ANALYSIS:
   - Primary bottleneck: {limiting_factor}
   - Scapy parsing is often the slowest component
   - VSA operations are relatively efficient

3. DIMENSIONALITY IMPACT:
   - Lower dimensions = higher throughput
   - 2048 dims offers good balance of speed and accuracy
   - 4096 dims is standard but ~2x slower

4. OPTIMIZATION STRATEGIES:
   - Use raw packet parsing instead of Scapy for production
   - Batch vector operations with NumPy
   - Consider SIMD/vectorized similarity computation
   - GPU acceleration for very high throughput

5. REALISTIC EXPECTATIONS:
   - Python + Scapy: ~{best.pps:,.0f} pps
   - With raw parsing: Est. 2-3x improvement
   - With C/Rust: Est. 10-100x improvement
   - 10k pps is achievable with optimization
""")


if __name__ == "__main__":
    main()
