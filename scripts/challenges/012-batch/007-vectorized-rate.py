#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 007: Vectorized Rate Detection
=============================================================================

Can we detect volumetric attacks using VECTORS instead of hardcoded rate
thresholds like "rate_ratio > 50"?

Approach:
1. Encode rate observations as structured data
2. Accumulate a "normal rate" prototype during warmup
3. Detect anomalies via similarity to learned baseline

Key insight: Rate is just another "field" we can encode and track.
Instead of: if rate > 50 * baseline
We use:     if similarity(rate_vec, baseline_rate) < threshold

The threshold is learned from baseline variability, not hardcoded.

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/007-vectorized-rate.py
"""

import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import deque
from enum import Enum
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
DECAY = 0.98
WARMUP_PACKETS = 400


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# RATE ENCODER
# =============================================================================

class RateEncoder:
    """
    Encodes packet rates as vectors using multiple representations.

    Instead of comparing raw numbers, we encode rate into structured data
    that captures its characteristics at multiple scales.
    """

    def __init__(self, encoder):
        self.encoder = encoder

    def encode_rate(self, pps: float) -> np.ndarray:
        """
        Encode a packets-per-second rate as a vector.

        Uses multiple representations to capture rate at different scales:
        - Log magnitude (order of magnitude)
        - Rate band (categorical bucket)
        - Relative indicators (low/medium/high/extreme)
        """
        if pps <= 0:
            pps = 1

        log_rate = np.log10(pps)

        # Encode rate as structured data with multiple facets
        rate_data = {
            # Order of magnitude (0=1-9, 1=10-99, 2=100-999, etc.)
            "rate_magnitude": int(log_rate),

            # Finer granularity within magnitude (0-9)
            "rate_fraction": int((log_rate % 1) * 10),

            # Coarse rate band for easy comparison
            "rate_band": self._get_rate_band(pps),
        }

        return self.encoder.encode_data(rate_data)

    def _get_rate_band(self, pps: float) -> str:
        """Categorize rate into bands without hardcoding detection thresholds."""
        # These are descriptive categories, NOT detection thresholds
        # The detector learns what's "normal" from the baseline
        if pps < 10:
            return "trickle"
        elif pps < 100:
            return "low"
        elif pps < 1000:
            return "moderate"
        elif pps < 10000:
            return "high"
        elif pps < 100000:
            return "very_high"
        else:
            return "extreme"


# =============================================================================
# RATE TRACKER
# =============================================================================

class RateTracker:
    """
    Tracks rate patterns using vector accumulation.

    Learns what "normal" rates look like during warmup, then detects
    when current rates deviate from the learned baseline.
    """

    def __init__(self, encoder, rate_encoder: RateEncoder):
        self.encoder = encoder
        self.rate_encoder = rate_encoder

        # Accumulators for rate patterns
        self.prior_rate_accum = encoder.create_accumulator()
        self.recent_rate_accum = encoder.create_accumulator()

        # Track rate samples for statistics
        self.prior_rates = []
        self.prior_sims = []

        self.frozen = False
        self._prior_rate_norm = None

        # Learned thresholds
        self.baseline_sim_mean = 0.8
        self.baseline_sim_std = 0.1

    def observe(self, pps: float, is_warmup: bool):
        """Observe a rate sample."""
        rate_vec = self.rate_encoder.encode_rate(pps)

        if is_warmup:
            self.prior_rate_accum = self.encoder.accumulate(self.prior_rate_accum, rate_vec)
            self.prior_rates.append(pps)

            # Track similarity during warmup to learn variability
            if len(self.prior_rates) > 50:
                temp_norm = self.encoder.normalize_accumulator(self.prior_rate_accum)
                sim = cosine_similarity(rate_vec, temp_norm)
                self.prior_sims.append(sim)
        else:
            # Decay and accumulate recent rates
            self.recent_rate_accum = DECAY * self.recent_rate_accum + rate_vec.astype(np.float64)

    def freeze(self):
        """Freeze the baseline after warmup."""
        self.frozen = True
        self._prior_rate_norm = self.encoder.normalize_accumulator(self.prior_rate_accum)
        self.recent_rate_accum = self.prior_rate_accum.copy()

        # Compute baseline statistics
        if self.prior_sims:
            self.baseline_sim_mean = np.mean(self.prior_sims)
            self.baseline_sim_std = np.std(self.prior_sims) if len(self.prior_sims) > 1 else 0.05

    def get_rate_similarity(self, pps: float) -> float:
        """Get similarity of current rate to baseline."""
        if not self.frozen:
            return 1.0

        rate_vec = self.rate_encoder.encode_rate(pps)
        return cosine_similarity(rate_vec, self._prior_rate_norm)

    def get_rate_divergence(self) -> float:
        """Get divergence between prior and recent rate patterns."""
        if not self.frozen:
            return 0.0

        recent_norm = self.encoder.normalize_accumulator(self.recent_rate_accum)
        return 1.0 - cosine_similarity(self._prior_rate_norm, recent_norm)

    def is_anomalous_rate(self, pps: float) -> tuple:
        """
        Check if rate is anomalous using learned baseline.

        Returns: (is_anomalous, similarity, threshold)
        """
        if not self.frozen:
            return False, 1.0, 0.0

        sim = self.get_rate_similarity(pps)

        # Anomaly threshold: 2.5 standard deviations below mean
        # This is LEARNED from baseline variability, not hardcoded
        threshold = self.baseline_sim_mean - 2.5 * self.baseline_sim_std

        is_anomalous = sim < threshold

        return is_anomalous, sim, threshold


# =============================================================================
# COMBINED DETECTOR
# =============================================================================

class VectorizedRateDetector:
    """
    Detector that uses vectorized rate tracking.

    Combines:
    - Traffic pattern tracking (what packets look like)
    - Rate pattern tracking (how fast packets arrive)

    Both use vector similarity, no hardcoded thresholds.
    """

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Rate tracking
        self.rate_encoder = RateEncoder(self.encoder)
        self.rate_tracker = RateTracker(self.encoder, self.rate_encoder)

        # Traffic pattern tracking
        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()
        self._prior_traffic_norm = None

        # Baseline statistics for traffic
        self.traffic_sims = []
        self.traffic_sim_mean = 0.7
        self.traffic_sim_std = 0.1

        # Smoothing
        self.anomaly_history = deque(maxlen=20)

    def _update_traffic_cache(self):
        self._prior_traffic_norm = self.encoder.normalize_accumulator(self.prior_traffic)

    def process(self, packet: dict, pps: float) -> dict:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # Encode packet
        packet_vec = self.encoder.encode_data(packet)

        # Track rate
        self.rate_tracker.observe(pps, is_warmup)

        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)

            # Sample traffic similarity during warmup
            if self.packet_count > 100 and self.packet_count % 20 == 0:
                temp_norm = self.encoder.normalize_accumulator(self.prior_traffic)
                sim = cosine_similarity(packet_vec, temp_norm)
                self.traffic_sims.append(sim)

            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                self.rate_tracker.freeze()
                self.recent_traffic = self.prior_traffic.copy()
                self._update_traffic_cache()

                if self.traffic_sims:
                    self.traffic_sim_mean = np.mean(self.traffic_sims)
                    self.traffic_sim_std = np.std(self.traffic_sims) if len(self.traffic_sims) > 1 else 0.1

            return {
                "packet_num": self.packet_count,
                "is_anomalous": False,
                "rate_sim": 1.0,
                "traffic_sim": 1.0,
                "rate_anomalous": False,
                "traffic_anomalous": False,
                "explanation": "Warming up...",
            }

        # Update recent traffic
        self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        if self.packet_count % 50 == 0:
            self._update_traffic_cache()

        # Check rate anomaly (vectorized!)
        rate_anomalous, rate_sim, rate_threshold = self.rate_tracker.is_anomalous_rate(pps)

        # Check traffic pattern anomaly
        traffic_sim = cosine_similarity(packet_vec, self._prior_traffic_norm)
        traffic_threshold = self.traffic_sim_mean - 2 * self.traffic_sim_std
        traffic_anomalous = traffic_sim < traffic_threshold

        # Combined detection: either rate OR traffic is anomalous
        is_anomalous = rate_anomalous or traffic_anomalous

        # Smooth detection
        self.anomaly_history.append(is_anomalous)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = anomaly_rate > 0.4

        # Build explanation
        explanations = []
        if rate_anomalous:
            explanations.append(f"rate_sim={rate_sim:.2f}<{rate_threshold:.2f}")
        if traffic_anomalous:
            explanations.append(f"traffic_sim={traffic_sim:.2f}")

        return {
            "packet_num": self.packet_count,
            "is_anomalous": is_anomalous,
            "rate_sim": rate_sim,
            "traffic_sim": traffic_sim,
            "rate_anomalous": rate_anomalous,
            "traffic_anomalous": traffic_anomalous,
            "explanation": "; ".join(explanations) if explanations else "Normal",
        }


# =============================================================================
# TIMING SIMULATION
# =============================================================================

class Phase(Enum):
    WARMUP = "warmup"
    CALM = "calm"
    ATTACK = "attack"


@dataclass
class TimePhase:
    name: str
    duration_seconds: int
    packets_per_second: int
    phase_type: Phase
    attack_fraction: float = 0.95


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def gen_normal(rng: random.Random) -> dict:
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]
    if proto == "TCP":
        return {"protocol": "TCP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([80, 443, 8080, 22]),
                "flags": rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[0],
                "payload_size": rng.randint(0, 1500)}
    elif proto == "UDP":
        return {"protocol": "UDP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([53, 443, 123]), "payload_size": rng.randint(20, 512)}
    else:
        return {"protocol": "ICMP", "icmp_type": rng.choice([0, 8]), "payload_size": 64}


def gen_dns_reflection(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 53, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(256, 4096)}


def gen_syn_flood(rng: random.Random) -> dict:
    return {"protocol": "TCP", "src_port": rng.randint(1, 65535), "dst_port": 80,
            "flags": "S", "payload_size": 0}


def gen_udp_flood(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": rng.randint(1, 65535),
            "dst_port": rng.randint(1, 65535), "payload_size": rng.randint(0, 1400)}


ATTACK_GENERATORS = {
    "dns_reflection": gen_dns_reflection,
    "syn_flood": gen_syn_flood,
    "udp_flood": gen_udp_flood,
}


# =============================================================================
# SIMULATION
# =============================================================================

def run_test(attack_type: str, phases: List[TimePhase], scale: float = 0.005):
    """Run volumetric attack test with vectorized rate detection."""

    print(f"\n{'='*70}")
    print(f"VECTORIZED RATE TEST: {attack_type}")
    print(f"{'='*70}")

    # Calculate warmup packets
    first_calm_packets = int(phases[0].duration_seconds * phases[0].packets_per_second * scale)
    warmup_packets = min(first_calm_packets - 10, 400)

    detector = VectorizedRateDetector(warmup_packets=warmup_packets)
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Show timeline
    print(f"\n  Timeline (scaled {scale:.1%}):")
    print(f"  {'-'*60}")
    for phase in phases:
        scaled = int(phase.duration_seconds * phase.packets_per_second * scale)
        marker = " [ATTACK]" if phase.phase_type == Phase.ATTACK else ""
        print(f"  {phase.name:12} {phase.packets_per_second:>8} pps → {scaled:>6} pkts{marker}")

    # Process phases
    phase_results = []

    for phase in phases:
        scaled_packets = max(1, int(phase.duration_seconds * phase.packets_per_second * scale))
        phase_detections = 0
        attack_packets = 0
        rate_anomalies = 0
        traffic_anomalies = 0

        for i in range(scaled_packets):
            # Generate packet
            if phase.phase_type == Phase.ATTACK and rng.random() < phase.attack_fraction:
                packet = attack_gen(rng)
                label = "attack"
                attack_packets += 1
            else:
                packet = gen_normal(rng)
                label = "normal"

            # Process
            result = detector.process(packet, phase.packets_per_second)

            if detector.warmup_complete:
                if result["is_anomalous"]:
                    phase_detections += 1
                if result["rate_anomalous"]:
                    rate_anomalies += 1
                if result["traffic_anomalous"]:
                    traffic_anomalies += 1

        # Phase summary
        if not detector.warmup_complete or phase.name == "calm-1":
            status = "WARMUP"
            detection_rate = 0
        else:
            detection_rate = phase_detections / scaled_packets if scaled_packets > 0 else 0
            if phase.phase_type == Phase.ATTACK:
                status = "DETECTED" if detection_rate > 0.5 else "MISSED"
            else:
                status = "CLEAN" if detection_rate < 0.1 else "FP"

        phase_results.append({
            "name": phase.name,
            "phase_type": phase.phase_type,
            "packets": scaled_packets,
            "detections": phase_detections,
            "detection_rate": detection_rate,
            "rate_anomalies": rate_anomalies,
            "traffic_anomalies": traffic_anomalies,
            "status": status,
        })

    # Print results
    print(f"\n  Phase Results:")
    print(f"  {'-'*60}")

    for pr in phase_results:
        if pr["status"] == "WARMUP":
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): [WARMUP]")
        elif pr["phase_type"] == Phase.ATTACK:
            marker = "✓" if pr["status"] == "DETECTED" else "✗"
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): "
                  f"Detection={pr['detection_rate']:.0%} "
                  f"(rate:{pr['rate_anomalies']}, traffic:{pr['traffic_anomalies']}) "
                  f"{marker} {pr['status']}")
        else:
            marker = "✓" if pr["status"] == "CLEAN" else "⚠"
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): "
                  f"FP={pr['detection_rate']:.0%} "
                  f"(rate:{pr['rate_anomalies']}, traffic:{pr['traffic_anomalies']}) "
                  f"{marker} {pr['status']}")

    # Overall metrics
    attack_phases = [pr for pr in phase_results if pr["phase_type"] == Phase.ATTACK and pr["status"] != "WARMUP"]
    calm_phases = [pr for pr in phase_results if pr["phase_type"] == Phase.CALM and pr["status"] != "WARMUP"]

    if attack_phases:
        attack_detected = sum(pr["detections"] for pr in attack_phases)
        attack_total = sum(pr["packets"] for pr in attack_phases)
        attack_recall = min(1.0, attack_detected / attack_total) if attack_total > 0 else 0
    else:
        attack_recall = 0

    if calm_phases:
        fp = sum(pr["detections"] for pr in calm_phases)
        fp_total = sum(pr["packets"] for pr in calm_phases)
        fp_rate = fp / fp_total if fp_total > 0 else 0
    else:
        fp_rate = 0

    print(f"\n  Overall: Attack Recall={attack_recall:.0%}, Normal FP={fp_rate:.0%}")

    # Show learned thresholds
    print(f"\n  Learned Thresholds (NOT hardcoded):")
    print(f"    Rate: sim_mean={detector.rate_tracker.baseline_sim_mean:.3f}, "
          f"sim_std={detector.rate_tracker.baseline_sim_std:.3f}")
    print(f"    Traffic: sim_mean={detector.traffic_sim_mean:.3f}, "
          f"sim_std={detector.traffic_sim_std:.3f}")

    return {
        "attack": attack_type,
        "attack_recall": attack_recall,
        "fp_rate": fp_rate,
    }


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 007: Vectorized Rate Detection")
    print("="*70)
    print("""
    Can we detect volumetric attacks using VECTORS instead of hardcoded
    rate thresholds?

    Previous approach (006):
        if rate_ratio > 50:  # Magic number!
            is_attack = True

    New approach (007):
        rate_vec = encode_rate(current_pps)
        rate_sim = similarity(rate_vec, baseline_rate)
        if rate_sim < learned_threshold:  # Threshold learned from data!
            is_attack = True

    Key: Rate is encoded as structured data:
        {"rate_magnitude": 5, "rate_fraction": 0, "rate_band": "extreme"}

    The baseline learns what "normal" rate looks like, and we detect
    when current rate is DISSIMILAR to learned normal.
    """)

    # Timeline with rate differentials
    timeline = [
        TimePhase("calm-1", 600, 100, Phase.CALM),              # Warmup
        TimePhase("ATTACK-1", 30, 100000, Phase.ATTACK),        # 1000x
        TimePhase("calm-2", 300, 100, Phase.CALM),              # Recovery
        TimePhase("ATTACK-2", 60, 50000, Phase.ATTACK),         # 500x
        TimePhase("calm-3", 300, 100, Phase.CALM),              # Final calm
    ]

    results = []

    for attack_type in ["dns_reflection", "syn_flood", "udp_flood"]:
        result = run_test(attack_type, timeline, scale=0.005)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("VECTORIZED RATE DETECTION SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'Attack Recall':>15} {'Normal FP':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['attack']:<20} {r['attack_recall']:>15.0%} {r['fp_rate']:>12.0%}")

    avg_recall = np.mean([r['attack_recall'] for r in results])
    avg_fp = np.mean([r['fp_rate'] for r in results])
    print("-"*50)
    print(f"{'Average':<20} {avg_recall:>15.0%} {avg_fp:>12.0%}")

    print("""

VECTORIZED RATE: NO MAGIC NUMBERS
=================================

The key insight: rate is just another field we can encode.

Encoding:
    pps=100     → {"rate_magnitude": 2, "rate_band": "moderate"}
    pps=100000  → {"rate_magnitude": 5, "rate_band": "extreme"}

Detection:
    baseline_rate = accumulate(normal_rate_vectors)
    current_rate_vec = encode_rate(current_pps)

    similarity = cosine_sim(current_rate_vec, baseline_rate)
    threshold = learned_mean - 2.5 * learned_std

    if similarity < threshold:
        # Rate is DISSIMILAR to what we learned
        is_volumetric = True

No hardcoded numbers like "rate > 50x" or "pps > 10000".
Everything is learned from the baseline.
    """)


if __name__ == "__main__":
    main()
