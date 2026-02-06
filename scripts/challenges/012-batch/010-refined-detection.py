#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 010: Refined Detection
=============================================================================

Building on 009's improvements, this experiment:

1. ANALYZES which signals are most reliable
2. EXPLORES signal combinations that reduce FP
3. PROPOSES Holon extensions that could help

Key observations from 009:
- Rate signal is 100% accurate for volumetric attacks
- Multiscale triggers spuriously during calm periods (recovery artifact)
- Confidence signal has good separation

New techniques:
1. GATED DETECTION: Only trust pattern signals if rate confirms
2. HYSTERESIS: Different thresholds for entering/exiting anomaly state
3. SUSTAINED CONFIRMATION: Require N consecutive anomalous packets

Potential Holon extensions explored:
1. Adaptive decay based on stability
2. Z-score normalization primitive
3. Mahalanobis distance for multivariate detection

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/010-refined-detection.py
"""

import sys
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import deque
from enum import Enum
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


DIMENSIONS = 4096
WARMUP_PACKETS = 400


# =============================================================================
# Z-SCORE TRACKER (Potential Holon Extension)
# =============================================================================

class ZScoreTracker:
    """
    Tracks z-scores of similarity values for standardized anomaly detection.

    Z-score = (value - mean) / std

    Benefits:
    - Standardized threshold (e.g., z < -2) works across different distributions
    - Handles different baseline variabilities automatically

    PROPOSAL: Add to Holon as a primitive for statistical tracking.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.running_sum = 0.0
        self.running_sq_sum = 0.0

    def update(self, value: float):
        """Add value and update running statistics."""
        if len(self.values) == self.window_size:
            # Remove oldest value from running sums
            old = self.values[0]
            self.running_sum -= old
            self.running_sq_sum -= old * old

        self.values.append(value)
        self.running_sum += value
        self.running_sq_sum += value * value

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self.running_sum / len(self.values)

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 1.0
        n = len(self.values)
        variance = (self.running_sq_sum / n) - (self.mean ** 2)
        return max(0.01, np.sqrt(max(0, variance)))

    def z_score(self, value: float) -> float:
        """Get z-score of a value relative to tracked distribution."""
        return (value - self.mean) / self.std


# =============================================================================
# HYSTERESIS STATE MACHINE
# =============================================================================

class HysteresisDetector:
    """
    Uses different thresholds for entering vs exiting anomaly state.

    - ENTER threshold: More stringent (must be clearly anomalous)
    - EXIT threshold: More lenient (must be clearly normal)

    This prevents oscillation between states.

    PROPOSAL: Could be a Holon primitive for state-based detection.
    """

    def __init__(self, enter_z: float = -2.5, exit_z: float = -1.0):
        self.enter_z = enter_z  # Must be this anomalous to enter
        self.exit_z = exit_z    # Must be this normal to exit

        self.in_anomaly_state = False

    def update(self, z_score: float) -> bool:
        """Update state and return whether currently in anomaly state."""
        if not self.in_anomaly_state:
            # Currently normal - check if should enter anomaly
            if z_score < self.enter_z:
                self.in_anomaly_state = True
        else:
            # Currently in anomaly - check if should exit
            if z_score > self.exit_z:
                self.in_anomaly_state = False

        return self.in_anomaly_state


# =============================================================================
# CONFIRMATION COUNTER
# =============================================================================

class ConfirmationCounter:
    """
    Requires N consecutive anomalous packets before confirming.

    Reduces spurious single-packet false positives.
    """

    def __init__(self, required_count: int = 3):
        self.required_count = required_count
        self.consecutive_count = 0
        self.confirmed = False

    def update(self, is_anomalous: bool) -> bool:
        """Update and return whether anomaly is confirmed."""
        if is_anomalous:
            self.consecutive_count += 1
            if self.consecutive_count >= self.required_count:
                self.confirmed = True
        else:
            self.consecutive_count = 0
            # Don't immediately unconfirm - let hysteresis handle that

        return self.confirmed

    def reset(self):
        self.consecutive_count = 0
        self.confirmed = False


# =============================================================================
# GATED DETECTOR
# =============================================================================

class GatedDetector:
    """
    Pattern anomalies are only trusted if rate also confirms.

    Logic:
    - If RATE is anomalous → definitely anomaly (volumetric attack)
    - If PATTERN is anomalous AND RATE is elevated → likely anomaly
    - If PATTERN is anomalous BUT RATE is normal → could be recovery artifact

    This gates pattern-based detection with rate confirmation.
    """

    def __init__(
        self,
        store: CPUStore,
        warmup_packets: int = WARMUP_PACKETS,
        rate_z_confirm: float = -1.0,  # Rate must be at least this anomalous
    ):
        self.store = store
        self.encoder = store.encoder
        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Rate tracking with z-scores
        self.rate_encoder = ContinuousRateEncoder(DIMENSIONS)
        self.rate_accum = self.encoder.create_accumulator()
        self._rate_norm = None
        self.rate_z_tracker = ZScoreTracker(window_size=50)

        # Pattern tracking with z-scores
        self.pattern_accum = self.encoder.create_accumulator()
        self._pattern_norm = None
        self.pattern_z_tracker = ZScoreTracker(window_size=50)

        # Hysteresis state machines
        self.rate_hysteresis = HysteresisDetector(enter_z=-2.5, exit_z=-0.5)
        self.pattern_hysteresis = HysteresisDetector(enter_z=-2.0, exit_z=-0.5)

        # Confirmation
        self.confirmation = ConfirmationCounter(required_count=5)

        # Gating threshold
        self.rate_z_confirm = rate_z_confirm

        # Output smoothing
        self.anomaly_history = deque(maxlen=10)

    def process(self, packet: dict, pps: float) -> dict:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # Encode
        packet_vec = self.encoder.encode_data(packet)
        rate_vec = self.rate_encoder.encode_rate(pps)

        if is_warmup:
            self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
            self.pattern_accum = self.encoder.accumulate(self.pattern_accum, packet_vec)

            # Sample similarities for z-score calibration
            if self.packet_count > 50:
                temp_rate = self.encoder.normalize_accumulator(self.rate_accum)
                temp_pattern = self.encoder.normalize_accumulator(self.pattern_accum)

                rate_sim = self.store.similarity(rate_vec, temp_rate, metric="cosine")
                pattern_sim = self.store.similarity(packet_vec, temp_pattern, metric="cosine")

                self.rate_z_tracker.update(rate_sim)
                self.pattern_z_tracker.update(pattern_sim)

            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                self._rate_norm = self.encoder.normalize_accumulator(self.rate_accum)
                self._pattern_norm = self.encoder.normalize_accumulator(self.pattern_accum)

            return {"packet_num": self.packet_count, "is_anomalous": False, "explanation": "Warming up..."}

        # Compute similarities
        rate_sim = self.store.similarity(rate_vec, self._rate_norm, metric="cosine")
        pattern_sim = self.store.similarity(packet_vec, self._pattern_norm, metric="cosine")

        # Update z-score trackers
        self.rate_z_tracker.update(rate_sim)
        self.pattern_z_tracker.update(pattern_sim)

        # Get z-scores
        rate_z = self.rate_z_tracker.z_score(rate_sim)
        pattern_z = self.pattern_z_tracker.z_score(pattern_sim)

        # Update hysteresis states
        rate_anomalous = self.rate_hysteresis.update(rate_z)
        pattern_anomalous = self.pattern_hysteresis.update(pattern_z)

        # GATED LOGIC
        # Rate anomaly is always trusted (volumetric)
        # Pattern anomaly only trusted if rate is at least somewhat elevated
        rate_confirms = rate_z < self.rate_z_confirm

        raw_anomaly = rate_anomalous or (pattern_anomalous and rate_confirms)

        # Confirmation
        confirmed = self.confirmation.update(raw_anomaly)

        # Check if we should exit confirmed state
        if confirmed and not raw_anomaly:
            # Been normal for a while, reset confirmation
            self.confirmation.reset()
            confirmed = False

        # Smooth output
        self.anomaly_history.append(confirmed)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = anomaly_rate > 0.5

        return {
            "packet_num": self.packet_count,
            "is_anomalous": is_anomalous,
            "rate_z": rate_z,
            "pattern_z": pattern_z,
            "rate_anomalous": rate_anomalous,
            "pattern_anomalous": pattern_anomalous,
            "rate_confirms": rate_confirms,
            "raw_anomaly": raw_anomaly,
            "confirmed": confirmed,
            "explanation": self._build_explanation(rate_z, pattern_z, rate_anomalous, pattern_anomalous),
        }

    def _build_explanation(self, rate_z, pattern_z, rate_anom, pattern_anom) -> str:
        parts = []
        if rate_anom:
            parts.append(f"rate_z={rate_z:.2f}")
        if pattern_anom:
            parts.append(f"pattern_z={pattern_z:.2f}")
        return "; ".join(parts) if parts else "Normal"


# =============================================================================
# CONTINUOUS RATE ENCODER
# =============================================================================

class ContinuousRateEncoder:
    def __init__(self, dimensions: int, scale: float = 1000.0):
        self.dimensions = dimensions
        self.scale = scale

    def encode_rate(self, pps: float) -> np.ndarray:
        if pps <= 0:
            pps = 1
        log_rate = np.log10(pps)
        return self._positional_encode(log_rate)

    def _positional_encode(self, value: float) -> np.ndarray:
        indices = np.arange(self.dimensions)
        freqs = 1 / (self.scale ** (indices / self.dimensions))
        values = np.where(indices % 2 == 0, np.sin(value * freqs), np.cos(value * freqs))
        return np.sign(values).astype(np.int8)


# =============================================================================
# SIMULATION
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


def run_test(attack_type: str, phases: List[TimePhase], scale: float = 0.005):
    print(f"\n{'='*70}")
    print(f"GATED DETECTOR TEST: {attack_type}")
    print(f"{'='*70}")

    first_calm_packets = int(phases[0].duration_seconds * phases[0].packets_per_second * scale)
    warmup_packets = min(first_calm_packets - 10, 400)

    store = CPUStore(dimensions=DIMENSIONS)
    detector = GatedDetector(store, warmup_packets=warmup_packets)
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    phase_results = []

    for phase in phases:
        scaled_packets = max(1, int(phase.duration_seconds * phase.packets_per_second * scale))
        phase_detections = 0
        rate_triggers = 0
        pattern_triggers = 0

        for i in range(scaled_packets):
            if phase.phase_type == Phase.ATTACK and rng.random() < phase.attack_fraction:
                packet = attack_gen(rng)
            else:
                packet = gen_normal(rng)

            result = detector.process(packet, phase.packets_per_second)

            if detector.warmup_complete:
                if result["is_anomalous"]:
                    phase_detections += 1
                if result.get("rate_anomalous", False):
                    rate_triggers += 1
                if result.get("pattern_anomalous", False):
                    pattern_triggers += 1

        if not detector.warmup_complete or phase.name == "calm-1":
            status = "WARMUP"
            detection_rate = 0
        else:
            detection_rate = phase_detections / scaled_packets if scaled_packets > 0 else 0
            if phase.phase_type == Phase.ATTACK:
                status = "DETECTED" if detection_rate > 0.5 else "MISSED"
            else:
                status = "CLEAN" if detection_rate < 0.03 else "FP"

        phase_results.append({
            "name": phase.name,
            "phase_type": phase.phase_type,
            "packets": scaled_packets,
            "detections": phase_detections,
            "detection_rate": detection_rate,
            "rate_triggers": rate_triggers,
            "pattern_triggers": pattern_triggers,
            "status": status,
        })

    # Print results
    print(f"\n  Phase Results:")
    print(f"  {'-'*70}")

    for pr in phase_results:
        if pr["status"] == "WARMUP":
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): [WARMUP]")
        else:
            marker = "✓" if pr["status"] in ["DETECTED", "CLEAN"] else "✗"
            phase_type = "Attack" if pr["phase_type"] == Phase.ATTACK else "Normal"
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): {phase_type} "
                  f"Det={pr['detection_rate']:.0%} "
                  f"(rate:{pr['rate_triggers']}, pattern:{pr['pattern_triggers']}) "
                  f"{marker} {pr['status']}")

    # Metrics
    attack_phases = [pr for pr in phase_results if pr["phase_type"] == Phase.ATTACK and pr["status"] != "WARMUP"]
    calm_phases = [pr for pr in phase_results if pr["phase_type"] == Phase.CALM and pr["status"] != "WARMUP"]

    attack_detected = sum(pr["detections"] for pr in attack_phases)
    attack_total = sum(pr["packets"] for pr in attack_phases)
    attack_recall = min(1.0, attack_detected / attack_total) if attack_total > 0 else 0

    fp = sum(pr["detections"] for pr in calm_phases)
    fp_total = sum(pr["packets"] for pr in calm_phases)
    fp_rate = fp / fp_total if fp_total > 0 else 0

    print(f"\n  Overall: Attack Recall={attack_recall:.0%}, Normal FP={fp_rate:.0%}")

    return {
        "attack": attack_type,
        "attack_recall": attack_recall,
        "fp_rate": fp_rate,
    }


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 010: Refined Detection")
    print("="*70)
    print("""
    Refined techniques:

    1. Z-SCORE NORMALIZATION
       - Standardized anomaly thresholds (z < -2.5)
       - Works across different baseline variabilities

    2. HYSTERESIS STATE MACHINE
       - Enter threshold: z < -2.5 (must be clearly anomalous)
       - Exit threshold: z > -0.5 (must be clearly normal)
       - Prevents oscillation

    3. GATED DETECTION
       - Pattern anomalies gated by rate confirmation
       - Prevents FP from recovery artifacts

    4. CONFIRMATION COUNTER
       - Require 5 consecutive anomalous packets
       - Reduces single-packet FPs

    POTENTIAL HOLON EXTENSIONS:
    - ZScoreAccumulator: Track mean/std with efficient updates
    - HysteresisThreshold: State-based detection primitive
    - GatedBundle: Conditional superposition based on signal
    """)

    timeline = [
        TimePhase("calm-1", 600, 100, Phase.CALM),
        TimePhase("ATTACK-1", 30, 100000, Phase.ATTACK),
        TimePhase("calm-2", 300, 100, Phase.CALM),
        TimePhase("ATTACK-2", 60, 50000, Phase.ATTACK),
        TimePhase("calm-3", 300, 100, Phase.CALM),
    ]

    results = []
    for attack_type in ["dns_reflection", "syn_flood", "udp_flood"]:
        result = run_test(attack_type, timeline, scale=0.005)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("REFINED DETECTION SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'Attack Recall':>15} {'Normal FP':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['attack']:<20} {r['attack_recall']:>15.0%} {r['fp_rate']:>12.0%}")

    avg_recall = np.mean([r['attack_recall'] for r in results])
    avg_fp = np.mean([r['fp_rate'] for r in results])
    print("-"*50)
    print(f"{'Average':<20} {avg_recall:>15.0%} {avg_fp:>12.0%}")

    print(f"""

COMPARISON
==========
  008: 100% recall, 8% FP
  009: 100% recall, 5% FP
  010: {avg_recall:.0%} recall, {avg_fp:.0%} FP

PROPOSED HOLON EXTENSIONS
=========================

1. ZScoreAccumulator
   - Efficiently track running mean/std
   - Provide z_score(value) primitive
   - Useful for any statistical anomaly detection

2. HysteresisThreshold
   - State machine with enter/exit thresholds
   - Prevents oscillation in noisy signals
   - Generic primitive for any threshold-based detection

3. AdaptiveDecay
   - Adjust decay rate based on signal stability
   - Stable → faster decay (more responsive)
   - Noisy → slower decay (more stable)

4. MultiScaleTower
   - Track same signal at multiple timescales
   - Detect divergence between fast/slow
   - Useful for trend detection

These would be statistical/signal-processing primitives that
complement Holon's existing vector operations.
    """)


if __name__ == "__main__":
    main()
