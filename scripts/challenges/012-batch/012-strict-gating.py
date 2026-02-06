#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 012: Strict Gating for Low FP
=============================================================================

Problem: Pattern anomalies during calm periods (recovery artifacts) cause FPs.

Solution: Strict gating logic:
1. Rate anomaly alone is ALWAYS trusted (volumetric)
2. Pattern anomaly alone is NEVER trusted (could be recovery)
3. Pattern anomaly + rate slightly elevated = trusted

Also explores:
- EWMA (Exponential Weighted Moving Average) for smoothing
- Minimum confirmation duration
- Recovery detection (detect when returning to normal)

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/012-strict-gating.py
"""

import sys
import random
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
from enum import Enum
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


DIMENSIONS = 4096
WARMUP_PACKETS = 400


# =============================================================================
# FROZEN BASELINE (from 011)
# =============================================================================

class FrozenBaseline:
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
            self.std = max(np.std(self.samples) if len(self.samples) > 1 else 0.05, 0.02)

    def z_score(self, value: float) -> float:
        return (value - self.mean) / self.std


# =============================================================================
# EWMA SMOOTHER
# =============================================================================

class EWMASmoother:
    """
    Exponential Weighted Moving Average for signal smoothing.

    EWMA = alpha * current + (1 - alpha) * EWMA

    Lower alpha = smoother but slower to respond.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# =============================================================================
# STRICT GATED DETECTOR
# =============================================================================

class StrictGatedDetector:
    """
    Strict gating: pattern anomalies require rate confirmation.

    Logic:
    - Rate z < -3 → anomaly (volumetric)
    - Pattern z < -2.5 AND rate z < -1 → anomaly (pattern + rate confirms)
    - Pattern z < -2.5 AND rate z >= -1 → NOT anomaly (just recovery)
    """

    def __init__(
        self,
        warmup_packets: int = WARMUP_PACKETS,
        rate_threshold: float = -3.0,       # Strong rate anomaly
        pattern_threshold: float = -2.5,    # Pattern anomaly
        rate_confirm: float = -1.0,         # Rate must be at least this for pattern
        min_confirmation: int = 5,          # Consecutive packets
        ewma_alpha: float = 0.15,           # Smoothing factor
    ):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Thresholds
        self.rate_threshold = rate_threshold
        self.pattern_threshold = pattern_threshold
        self.rate_confirm = rate_confirm
        self.min_confirmation = min_confirmation

        # Rate
        self.rate_encoder = ContinuousRateEncoder(DIMENSIONS)
        self.rate_accum = self.encoder.create_accumulator()
        self._rate_norm = None
        self.rate_baseline = FrozenBaseline()

        # Pattern
        self.pattern_accum = self.encoder.create_accumulator()
        self._pattern_norm = None
        self.pattern_baseline = FrozenBaseline()

        # Smoothing
        self.ewma = EWMASmoother(alpha=ewma_alpha)

        # Confirmation
        self.consecutive = 0

        # State tracking
        self.in_anomaly_state = False
        self.anomaly_start = None

    def process(self, packet: dict, pps: float) -> dict:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        packet_vec = self.encoder.encode_data(packet)
        rate_vec = self.rate_encoder.encode_rate(pps)

        if is_warmup:
            self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
            self.pattern_accum = self.encoder.accumulate(self.pattern_accum, packet_vec)

            if self.packet_count > 100:
                temp_rate = self.encoder.normalize_accumulator(self.rate_accum)
                temp_pattern = self.encoder.normalize_accumulator(self.pattern_accum)

                rate_sim = self.store.similarity(rate_vec, temp_rate, metric="cosine")
                pattern_sim = self.store.similarity(packet_vec, temp_pattern, metric="cosine")

                self.rate_baseline.observe(rate_sim)
                self.pattern_baseline.observe(pattern_sim)

            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                self._rate_norm = self.encoder.normalize_accumulator(self.rate_accum)
                self._pattern_norm = self.encoder.normalize_accumulator(self.pattern_accum)
                self.rate_baseline.freeze()
                self.pattern_baseline.freeze()

            return {"packet_num": self.packet_count, "is_anomalous": False}

        # Compute z-scores
        rate_sim = self.store.similarity(rate_vec, self._rate_norm, metric="cosine")
        pattern_sim = self.store.similarity(packet_vec, self._pattern_norm, metric="cosine")

        rate_z = self.rate_baseline.z_score(rate_sim)
        pattern_z = self.pattern_baseline.z_score(pattern_sim)

        # STRICT GATING LOGIC
        # 1. Strong rate anomaly → always anomaly
        rate_anomaly = rate_z < self.rate_threshold

        # 2. Pattern anomaly ONLY if rate is at least somewhat elevated
        pattern_anomaly = (pattern_z < self.pattern_threshold) and (rate_z < self.rate_confirm)

        raw_anomaly = rate_anomaly or pattern_anomaly

        # Confirmation
        if raw_anomaly:
            self.consecutive += 1
        else:
            self.consecutive = max(0, self.consecutive - 2)  # Faster reset

        confirmed = self.consecutive >= self.min_confirmation

        # EWMA smoothing of confirmed signal
        smoothed = self.ewma.update(1.0 if confirmed else 0.0)
        is_anomalous = smoothed > 0.3

        # State tracking
        if is_anomalous and not self.in_anomaly_state:
            self.in_anomaly_state = True
            self.anomaly_start = self.packet_count
        elif not is_anomalous and self.in_anomaly_state:
            self.in_anomaly_state = False

        return {
            "packet_num": self.packet_count,
            "is_anomalous": is_anomalous,
            "rate_z": rate_z,
            "pattern_z": pattern_z,
            "rate_anomaly": rate_anomaly,
            "pattern_anomaly": pattern_anomaly,
            "raw": raw_anomaly,
            "confirmed": confirmed,
            "smoothed": smoothed,
        }


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
    print(f"STRICT GATING TEST: {attack_type}")
    print(f"{'='*70}")

    first_calm_packets = int(phases[0].duration_seconds * phases[0].packets_per_second * scale)
    warmup_packets = min(first_calm_packets - 10, 400)

    detector = StrictGatedDetector(warmup_packets=warmup_packets)
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
                if result.get("rate_anomaly", False):
                    rate_triggers += 1
                if result.get("pattern_anomaly", False):
                    pattern_triggers += 1

        if not detector.warmup_complete or phase.name == "calm-1":
            status = "WARMUP"
            detection_rate = 0
        else:
            detection_rate = phase_detections / scaled_packets if scaled_packets > 0 else 0
            if phase.phase_type == Phase.ATTACK:
                status = "DETECTED" if detection_rate > 0.5 else "MISSED"
            else:
                # Even stricter FP threshold
                status = "CLEAN" if detection_rate < 0.02 else "FP"

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
    print("BATCH 012 - CHALLENGE 012: Strict Gating for Low FP")
    print("="*70)
    print("""
    Strict gating rules:

    1. Rate z < -3.0        → ANOMALY (volumetric attack)
    2. Pattern z < -2.5 AND → ANOMALY (pattern + rate confirms)
       Rate z < -1.0
    3. Pattern z < -2.5 AND → NORMAL (just recovery artifact)
       Rate z >= -1.0

    Additional:
    - EWMA smoothing (alpha=0.15) for output stability
    - Require 5 consecutive raw anomalies before confirming
    - Faster reset (decrement by 2) when not anomalous
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
    print("STRICT GATING SUMMARY")
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
  008: 100% recall,  8% FP  (simple)
  009: 100% recall,  5% FP  (voting)
  011: 100% recall,  5% FP  (frozen z-scores)
  012: {avg_recall:.0%} recall, {avg_fp:.0%} FP  (strict gating)
    """)


if __name__ == "__main__":
    main()
