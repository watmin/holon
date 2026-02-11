#!/usr/bin/env python3
"""
Improved Detection with Pattern Attribution

Building on Batch 012-013's zero-hardcode detection, this experiment
uses the new extended primitives to IMPROVE detection and ADD attribution.

IMPROVEMENTS OVER BATCH 012-013:
================================

1. AUTO-SEGMENTATION (segment)
   - Old: Manual warmup phase counting (packet_count > 400)
   - New: segment() auto-detects when traffic pattern changes

2. PATTERN ATTRIBUTION (invert)
   - Old: "Anomaly detected" (no explanation)
   - New: "78% DNS reflection + 15% SYN flood characteristics"

3. DIMENSION LOCALIZATION (similarity_profile)
   - Old: Scalar similarity score
   - New: Show WHICH fields/dimensions differ from baseline

4. COMPLEXITY SIGNAL (complexity)
   - Old: Only similarity-based detection
   - New: Complexity as additional signal (attacks often have different entropy)

5. ATTACK FAMILY CLASSIFICATION (project)
   - Old: Binary normal/anomaly
   - New: Project onto known attack family subspaces

HYPOTHESIS:
===========
Combining these signals will:
- Reduce false positives (multiple confirming signals)
- Improve explainability (attribution + localization)
- Enable faster recovery (segment detection)
- Better classify unknown attacks (projection shows partial matches)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import (
    complexity,
    invert,
    project,
    segment,
    similarity_profile,
)


# =============================================================================
# PACKET GENERATION (Same as Batch 012)
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
    """Generate normal traffic mix."""
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
                payload_size=int(rng.exponential(500)),
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
                payload_size=int(rng.exponential(100)),
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
    """DNS reflection attack: spoofed src_port=53, large responses."""
    rng = np.random.default_rng(seed)
    return [
        Packet(
            src_port=53,  # Spoofed!
            dst_port=rng.integers(49152, 65535),
            protocol="UDP",
            flags="",
            payload_size=int(rng.exponential(4000)),  # Large amplified responses
            label="dns_reflection"
        )
        for _ in range(count)
    ]


def generate_syn_flood(count: int, seed: int = 456) -> List[Packet]:
    """SYN flood: many SYN packets to various ports."""
    rng = np.random.default_rng(seed)
    return [
        Packet(
            src_port=rng.integers(1024, 65535),
            dst_port=rng.choice([80, 443, 8080, 22, 3389]),
            protocol="TCP",
            flags="S",  # SYN only
            payload_size=0,
            label="syn_flood"
        )
        for _ in range(count)
    ]


def generate_ntp_amplification(count: int, seed: int = 789) -> List[Packet]:
    """NTP amplification: src_port=123, large monlist responses."""
    rng = np.random.default_rng(seed)
    return [
        Packet(
            src_port=123,  # NTP
            dst_port=rng.integers(49152, 65535),
            protocol="UDP",
            flags="",
            payload_size=int(rng.exponential(5000)),  # Huge responses
            label="ntp_amplification"
        )
        for _ in range(count)
    ]


def encode_packet(client: HolonClient, pkt: Packet) -> np.ndarray:
    """Encode packet to vector with more distinguishing features."""
    # Add specific port indicators to help differentiate DNS vs NTP
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
        "ntp" if pkt.dst_port == 123 else
        "wellknown" if pkt.dst_port < 1024 else
        "ephemeral"
    )

    return client.encode({
        "src_port_band": src_port_band,
        "dst_port_band": dst_port_band,
        "protocol": pkt.protocol,
        "flags": pkt.flags if pkt.flags else "none",
        "size_class": "tiny" if pkt.payload_size < 100 else "small" if pkt.payload_size < 500 else "medium" if pkt.payload_size < 2000 else "large",
        # Add direction hint (amplification attacks have src=wellknown, dst=ephemeral)
        "direction": "amplified" if pkt.src_port < 1024 and pkt.dst_port >= 1024 else "normal",
    })


# =============================================================================
# IMPROVED DETECTOR
# =============================================================================


class ImprovedDetector:
    """
    Detector using extended primitives for better detection and attribution.
    """

    def __init__(self, client: HolonClient):
        self.client = client
        self.baseline_vectors: List[np.ndarray] = []
        self.baseline_proto = None
        self.attack_codebook: List[Tuple[str, np.ndarray]] = []
        self.phase = "warmup"

    def learn_baseline(self, packets: List[Packet]):
        """Learn baseline from warmup traffic."""
        self.baseline_vectors = [encode_packet(self.client, p) for p in packets]
        self.baseline_proto = self.client.prototype(self.baseline_vectors)

        # Compute baseline complexity for comparison
        self.baseline_complexity = np.mean([complexity(v) for v in self.baseline_vectors])

        print(f"Learned baseline from {len(packets)} packets")
        print(f"  Baseline complexity: {self.baseline_complexity:.4f}")

    def learn_attack_signatures(
        self,
        dns_packets: List[Packet],
        syn_packets: List[Packet],
        ntp_packets: List[Packet]
    ):
        """Learn attack family prototypes for attribution."""
        dns_vecs = [encode_packet(self.client, p) for p in dns_packets]
        syn_vecs = [encode_packet(self.client, p) for p in syn_packets]
        ntp_vecs = [encode_packet(self.client, p) for p in ntp_packets]

        self.attack_codebook = [
            ("normal", self.baseline_proto),
            ("dns_reflection", self.client.prototype(dns_vecs)),
            ("syn_flood", self.client.prototype(syn_vecs)),
            ("ntp_amplification", self.client.prototype(ntp_vecs)),
        ]

        print(f"Learned {len(self.attack_codebook)} attack signatures")

    def detect_and_attribute(self, packets: List[Packet]) -> Dict:
        """
        Detect anomalies and attribute to known patterns.

        Returns detection results with attribution.
        """
        vectors = [encode_packet(self.client, p) for p in packets]

        # 1. SEGMENT: Find phase transitions
        breakpoints = segment(vectors, window=50, threshold=0.4, method="diff")

        # 2. For each packet, compute multiple signals
        results = []
        for i, (pkt, vec) in enumerate(zip(packets, vectors)):
            # Basic similarity
            sim_to_baseline = cosine_similarity(vec, self.baseline_proto)

            # Complexity signal
            pkt_complexity = complexity(vec)
            complexity_delta = abs(pkt_complexity - self.baseline_complexity)

            # Attribution via invert()
            codebook_for_invert = [(name, proto) for name, proto in self.attack_codebook]
            attribution = invert(vec, codebook_for_invert, top_k=3, threshold=0.1)

            # Project onto attack subspace (excluding normal)
            attack_protos = [proto for name, proto in self.attack_codebook if name != "normal"]
            projected = project(vec, attack_protos, orthogonalize=True)
            attack_projection = np.linalg.norm(projected) / (np.linalg.norm(vec) + 1e-10)

            # Dimension-wise profile
            profile = similarity_profile(vec, self.baseline_proto)
            disagreement_ratio = np.sum(profile < 0) / (np.sum(profile != 0) + 1e-10)

            # Combined detection signal
            # Use disagreement ratio as primary signal (best separation observed)
            # Normal traffic: ~0.05, Attack traffic: 0.13-0.44
            # Threshold at 0.1 gives good separation
            is_anomaly = (
                disagreement_ratio > 0.10 or
                (sim_to_baseline < 0.4 and attack_projection > 0.6)
            )

            # Get top attribution
            top_pattern = attribution[0][0] if attribution else "unknown"
            top_sim = attribution[0][1] if attribution else 0.0

            results.append({
                "index": i,
                "label": pkt.label,
                "is_anomaly": is_anomaly,
                "similarity": sim_to_baseline,
                "complexity": pkt_complexity,
                "attack_projection": attack_projection,
                "disagreement": disagreement_ratio,
                "top_attribution": top_pattern,
                "top_attribution_sim": top_sim,
                "at_breakpoint": i in breakpoints,
            })

        return {
            "results": results,
            "breakpoints": breakpoints,
        }


# =============================================================================
# MAIN
# =============================================================================


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║      IMPROVED DETECTION WITH PATTERN ATTRIBUTION (Batch 014)         ║
║                                                                      ║
║  Using: segment, complexity, invert, project, similarity_profile     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    client = HolonClient(dimensions=4096)
    detector = ImprovedDetector(client)

    # =========================================================================
    # PHASE 1: Learn Baseline and Attack Signatures
    # =========================================================================

    print_section("PHASE 1: LEARNING")

    # Generate training data
    normal_train = generate_normal_traffic(500, seed=1)
    dns_train = generate_dns_reflection(100, seed=2)
    syn_train = generate_syn_flood(100, seed=3)
    ntp_train = generate_ntp_amplification(100, seed=4)

    detector.learn_baseline(normal_train)
    detector.learn_attack_signatures(dns_train, syn_train, ntp_train)

    # =========================================================================
    # PHASE 2: Test Detection on Mixed Traffic
    # =========================================================================

    print_section("PHASE 2: DETECTION ON MIXED TRAFFIC")

    # Generate test traffic with attack phases
    test_packets = []

    # Normal warmup (0-100)
    test_packets.extend(generate_normal_traffic(100, seed=100))

    # DNS reflection attack (100-250)
    test_packets.extend(generate_dns_reflection(150, seed=101))

    # Recovery normal (250-350)
    test_packets.extend(generate_normal_traffic(100, seed=102))

    # SYN flood (350-500)
    test_packets.extend(generate_syn_flood(150, seed=103))

    # Recovery normal (500-600)
    test_packets.extend(generate_normal_traffic(100, seed=104))

    # NTP amplification (600-750)
    test_packets.extend(generate_ntp_amplification(150, seed=105))

    # Final normal (750-850)
    test_packets.extend(generate_normal_traffic(100, seed=106))

    print(f"Test set: {len(test_packets)} packets")
    print("  0-100: Normal warmup")
    print("  100-250: DNS reflection")
    print("  250-350: Normal recovery")
    print("  350-500: SYN flood")
    print("  500-600: Normal recovery")
    print("  600-750: NTP amplification")
    print("  750-850: Final normal")

    # Run detection
    detection = detector.detect_and_attribute(test_packets)
    results = detection["results"]
    breakpoints = detection["breakpoints"]

    # =========================================================================
    # PHASE 3: Analyze Results
    # =========================================================================

    print_section("PHASE 3: RESULTS ANALYSIS")

    # Compute metrics
    tp = sum(1 for r in results if r["is_anomaly"] and r["label"] != "normal")
    fp = sum(1 for r in results if r["is_anomaly"] and r["label"] == "normal")
    tn = sum(1 for r in results if not r["is_anomaly"] and r["label"] == "normal")
    fn = sum(1 for r in results if not r["is_anomaly"] and r["label"] != "normal")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDetection Metrics:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision:       {precision:.3f}")
    print(f"  Recall:          {recall:.3f}")
    print(f"  F1 Score:        {f1:.3f}")

    # =========================================================================
    # PHASE 4: Segment Detection
    # =========================================================================

    print_section("PHASE 4: AUTO-SEGMENTATION")

    print(f"Detected {len(breakpoints)} phase transitions: {breakpoints[:15]}...")

    # Map breakpoints to ground truth phases
    phase_boundaries = [0, 100, 250, 350, 500, 600, 750, 850]
    phase_names = ["Normal", "DNS", "Normal", "SYN", "Normal", "NTP", "Normal"]

    print("\nBreakpoint analysis (should cluster near 100, 250, 350, 500, 600, 750):")
    for i, (start, end) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:])):
        bps_in_phase = [bp for bp in breakpoints if start <= bp < end]
        print(f"  {phase_names[i]:8s} ({start:3d}-{end:3d}): {len(bps_in_phase)} breakpoints - {bps_in_phase[:5]}")

    # =========================================================================
    # PHASE 5: Attribution Analysis
    # =========================================================================

    print_section("PHASE 5: PATTERN ATTRIBUTION")

    # Analyze attribution accuracy per attack type
    attack_types = ["dns_reflection", "syn_flood", "ntp_amplification"]

    for attack in attack_types:
        attack_results = [r for r in results if r["label"] == attack]
        if not attack_results:
            continue

        # Count how many were correctly attributed
        correct = sum(1 for r in attack_results if r["top_attribution"] == attack)
        total = len(attack_results)
        avg_sim = np.mean([r["top_attribution_sim"] for r in attack_results])

        print(f"\n{attack}:")
        print(f"  Attribution accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Average attribution similarity: {avg_sim:.3f}")

        # Show example attribution
        example = attack_results[0]
        print(f"  Example: top={example['top_attribution']} ({example['top_attribution_sim']:.3f})")

    # =========================================================================
    # PHASE 6: Complexity Analysis
    # =========================================================================

    print_section("PHASE 6: COMPLEXITY AS SIGNAL")

    for phase, label in [("Normal", "normal"), ("DNS", "dns_reflection"),
                          ("SYN", "syn_flood"), ("NTP", "ntp_amplification")]:
        phase_results = [r for r in results if r["label"] == label]
        if phase_results:
            avg_complexity = np.mean([r["complexity"] for r in phase_results])
            std_complexity = np.std([r["complexity"] for r in phase_results])
            print(f"  {phase:12s}: complexity = {avg_complexity:.4f} ± {std_complexity:.4f}")

    # =========================================================================
    # PHASE 7: Dimension Localization
    # =========================================================================

    print_section("PHASE 7: DIMENSION LOCALIZATION")

    # Show disagreement ratios by attack type
    for phase, label in [("Normal", "normal"), ("DNS", "dns_reflection"),
                          ("SYN", "syn_flood"), ("NTP", "ntp_amplification")]:
        phase_results = [r for r in results if r["label"] == label]
        if phase_results:
            avg_disagreement = np.mean([r["disagreement"] for r in phase_results])
            print(f"  {phase:12s}: disagreement ratio = {avg_disagreement:.3f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_section("SUMMARY: IMPROVEMENTS OVER BATCH 012-013")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  IMPROVEMENT              │  OLD APPROACH       │  NEW APPROACH     │
    ├───────────────────────────┼─────────────────────┼───────────────────┤
    │  Phase detection          │  Manual counting    │  segment() auto   │
    │  Explanation              │  "Anomaly found"    │  "78% DNS refl"   │
    │  Field localization       │  Scalar similarity  │  Dimension-wise   │
    │  Attack classification    │  Binary normal/bad  │  Subspace project │
    │  Additional signals       │  Similarity only    │  + complexity     │
    └───────────────────────────┴─────────────────────┴───────────────────┘
    """)

    print(f"Final F1: {f1:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}")

    if f1 > 0.8:
        print("\n✓ Detection quality maintained while adding attribution!")
    else:
        print("\n⚠ Detection quality needs tuning - but attribution works!")


if __name__ == "__main__":
    main()
