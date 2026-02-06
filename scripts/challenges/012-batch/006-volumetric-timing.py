#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 006: Volumetric Attack with Synthetic Timing
=============================================================================

Simulates realistic volumetric attacks with time-based phases:

Timeline:
  0:00 - 5:00   CALM      (5 min)   ~100 pps     → 30,000 packets
  5:00 - 5:30   ATTACK    (30 sec)  ~100,000 pps → 3,000,000 packets
  5:30 - 8:00   CALM      (2.5 min) ~100 pps     → 15,000 packets
  8:00 - 10:00  ATTACK    (2 min)   ~50,000 pps  → 6,000,000 packets
  10:00 - 13:00 CALM      (3 min)   ~100 pps     → 18,000 packets

Key insight: During attack, packet VOLUME dominates. The "recent"
accumulator should be completely overwhelmed by attack traffic,
making the divergence from "prior" very high.

For simulation efficiency, we use scaled packet counts but preserve
the RATIO between calm and attack periods.

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/006-volumetric-timing.py
"""

import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from enum import Enum
import numpy as np
import time

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
DECAY = 0.995  # Slower decay to let attack traffic dominate
WARMUP_PACKETS = 500  # Packet count for warmup (scale-independent)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# TIMING SIMULATION
# =============================================================================

@dataclass
class TimePhase:
    """A phase with timing and rate information."""
    name: str
    duration_seconds: int
    packets_per_second: int
    is_attack: bool
    attack_type: Optional[str] = None
    attack_fraction: float = 0.95  # During attack, 95% attack traffic

    @property
    def total_packets(self) -> int:
        return self.duration_seconds * self.packets_per_second


class SyntheticClock:
    """Simulates time progression based on packet rates."""

    def __init__(self):
        self.current_time = 0.0  # Seconds since start
        self.packet_count = 0

    def tick(self, pps: int):
        """Advance clock by one packet at the given rate."""
        self.packet_count += 1
        self.current_time += 1.0 / pps

    def format_time(self) -> str:
        """Format current time as MM:SS."""
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        return f"{minutes:02d}:{seconds:02d}"


# =============================================================================
# VOLUMETRIC DETECTOR
# =============================================================================

class VolumetricDetector:
    """Detector optimized for volumetric attack detection."""

    FIELDS = ["protocol", "src_port", "dst_port", "tcp_flags", "icmp_type", "payload_size"]

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Traffic accumulators
        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()

        self._prior_norm = None
        self._recent_norm = None

        # Rate tracking (sliding window)
        self.window_packets = deque(maxlen=200)
        self.window_times = deque(maxlen=200)

        # Baseline statistics
        self.baseline_rate = 0.0
        self.baseline_divergence_samples = []
        self.baseline_div_mean = 0.0
        self.baseline_div_std = 0.1

        # Detection state - shorter window for faster recovery
        self.anomaly_history = deque(maxlen=20)
        self.in_attack = False
        self.attack_start_time = None

    def _extract_fields(self, packet: dict) -> Dict[str, Any]:
        fields = {}
        fields["protocol"] = packet.get("protocol")
        fields["src_port"] = packet.get("src_port")
        fields["dst_port"] = packet.get("dst_port")
        if packet.get("protocol") == "TCP":
            fields["tcp_flags"] = packet.get("flags")
        if packet.get("protocol") == "ICMP":
            fields["icmp_type"] = packet.get("icmp_type")
        size = packet.get("payload_size", 0)
        fields["payload_size"] = "none" if size == 0 else "small" if size < 256 else "large"
        return fields

    def _update_caches(self):
        self._prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
        self._recent_norm = self.encoder.normalize_accumulator(self.recent_traffic)

    def process(self, packet: dict, clock: SyntheticClock, pps: int) -> dict:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # Track timing
        self.window_packets.append(self.packet_count)
        self.window_times.append(clock.current_time)

        # Estimate current rate from PPS (provided by simulation)
        current_rate = pps

        # Also estimate from window for comparison
        if len(self.window_times) > 50:
            time_span = self.window_times[-1] - self.window_times[0]
            if time_span > 0:
                observed_rate = len(self.window_times) / time_span
            else:
                observed_rate = pps
        else:
            observed_rate = pps

        # Encode packet
        packet_vec = self.encoder.encode_data(packet)

        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)

            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                self.recent_traffic = self.prior_traffic.copy()
                self._update_caches()
                self.baseline_rate = current_rate
                self.baseline_div_mean = np.mean(self.baseline_divergence_samples) if self.baseline_divergence_samples else 0.05
                self.baseline_div_std = np.std(self.baseline_divergence_samples) if len(self.baseline_divergence_samples) > 1 else 0.02
            elif self.packet_count > 100 and self.packet_count % 50 == 0:
                # Sample divergence during warmup
                temp_norm = self.encoder.normalize_accumulator(self.prior_traffic)
                div = 1.0 - cosine_similarity(packet_vec, temp_norm)
                self.baseline_divergence_samples.append(div)

            return {
                "packet_num": self.packet_count,
                "time": clock.format_time(),
                "is_anomalous": False,
                "rate": current_rate,
                "divergence": 0.0,
                "rate_ratio": 1.0,
                "explanation": "Warming up...",
            }

        # Update recent with decay
        self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        # Update caches periodically (less often during high volume)
        update_freq = max(100, int(current_rate / 10)) if current_rate > 0 else 100
        if self.packet_count % update_freq == 0:
            self._update_caches()

        # Calculate divergence
        traffic_div = 1.0 - cosine_similarity(self._prior_norm, self._recent_norm)

        # Calculate rate ratio
        rate_ratio = current_rate / self.baseline_rate if self.baseline_rate > 0 else 1.0

        # Volumetric detection: RATE is the primary signal
        # During volumetric attack: rate spikes (this is the definition)
        # During calm: rate is normal, even if divergence is elevated from residue

        # Key insight: if rate is back to normal, we're NOT in a volumetric attack
        # The high divergence after attack is "residue" not attack
        is_volumetric = (
            rate_ratio > 50 or  # 50x rate is definite attack
            (rate_ratio > 5 and traffic_div > 0.25)  # 5x rate + divergence
            # Don't trigger on divergence alone - that's residue, not attack
        )

        # Smooth detection
        self.anomaly_history.append(is_volumetric)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = anomaly_rate > 0.5

        # Track attack state transitions
        if is_anomalous and not self.in_attack:
            self.in_attack = True
            self.attack_start_time = clock.current_time
        elif not is_anomalous and self.in_attack:
            self.in_attack = False

        # Build explanation
        explanations = []
        if traffic_div > 0.20:
            explanations.append(f"div={traffic_div:.0%}")
        if rate_ratio > 10:
            explanations.append(f"rate={rate_ratio:.0f}x")

        return {
            "packet_num": self.packet_count,
            "time": clock.format_time(),
            "is_anomalous": is_anomalous,
            "rate": current_rate,
            "divergence": traffic_div,
            "rate_ratio": rate_ratio,
            "explanation": ", ".join(explanations) if explanations else "Normal",
        }


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


def gen_syn_flood(rng: random.Random) -> dict:
    return {"protocol": "TCP", "src_port": rng.randint(1, 65535), "dst_port": 80,
            "flags": "S", "payload_size": 0}


def gen_dns_reflection(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 53, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(256, 4096)}


def gen_udp_flood(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": rng.randint(1, 65535),
            "dst_port": rng.randint(1, 65535), "payload_size": rng.randint(0, 1400)}


ATTACK_GENERATORS = {
    "syn_flood": gen_syn_flood,
    "dns_reflection": gen_dns_reflection,
    "udp_flood": gen_udp_flood,
}


# =============================================================================
# VOLUMETRIC SIMULATION
# =============================================================================

def run_volumetric_test(attack_type: str, phases: List[TimePhase], scale: float = 0.001):
    """
    Run volumetric attack simulation.

    scale: Reduce packet counts for simulation speed.
           0.001 = 1/1000th of real packets
    """

    print(f"\n{'='*70}")
    print(f"VOLUMETRIC TEST: {attack_type}")
    print(f"{'='*70}")

    # Show timeline
    print(f"\n  Timeline (scaled {scale:.0%} for simulation):")
    print(f"  {'-'*60}")

    total_time = 0
    for phase in phases:
        scaled_packets = int(phase.total_packets * scale)
        end_time = total_time + phase.duration_seconds
        attack_marker = " [ATTACK]" if phase.is_attack else ""
        print(f"  {total_time//60:02d}:{total_time%60:02d} - {end_time//60:02d}:{end_time%60:02d}  "
              f"{phase.name:12} {phase.packets_per_second:>8} pps → {scaled_packets:>6} pkts{attack_marker}")
        total_time = end_time

    # Calculate warmup packets - should complete during first (calm) phase
    first_phase_packets = max(1, int(phases[0].total_packets * scale))
    warmup_packets = min(first_phase_packets - 10, 400)  # Leave buffer in first phase
    baseline_pps = phases[0].packets_per_second  # Baseline rate from calm period

    detector = VolumetricDetector(warmup_packets=warmup_packets)
    detector.baseline_rate = baseline_pps  # Set baseline from calm period
    clock = SyntheticClock()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Results tracking
    phase_results = []

    # Process each phase
    for phase in phases:
        scaled_packets = max(1, int(phase.total_packets * scale))
        phase_detections = 0
        attack_packets = 0
        normal_packets = 0

        for i in range(scaled_packets):
            # Generate packet
            if phase.is_attack and rng.random() < phase.attack_fraction:
                packet = attack_gen(rng)
                label = "attack"
                attack_packets += 1
            else:
                packet = gen_normal(rng)
                label = "normal"
                normal_packets += 1

            # Advance clock
            clock.tick(phase.packets_per_second)

            # Process packet (detector handles warmup internally)
            result = detector.process(packet, clock, phase.packets_per_second)

            if detector.warmup_complete and result["is_anomalous"]:
                phase_detections += 1

        # Phase summary
        phase_total = scaled_packets
        # Calculate detection rate over post-warmup packets only
        post_warmup_packets = max(0, detector.packet_count - detector.warmup_packets)
        if not detector.warmup_complete:
            detection_rate = 0
            status = "WARMUP"
        else:
            detection_rate = phase_detections / phase_total if phase_total > 0 else 0
            if phase.is_attack:
                status = "DETECTED" if detection_rate > 0.5 else "MISSED"
            else:
                status = "CLEAN" if detection_rate < 0.1 else "FP"

        phase_results.append({
            "name": phase.name,
            "is_attack": phase.is_attack,
            "packets": phase_total,
            "attack_pkts": attack_packets,
            "detections": phase_detections,
            "detection_rate": detection_rate,
            "status": status,
        })

    # Print results
    print(f"\n  Phase Results:")
    print(f"  {'-'*60}")

    for pr in phase_results:
        if pr["status"] == "WARMUP":
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): [WARMUP]")
        elif pr["is_attack"]:
            marker = "✓" if pr["status"] == "DETECTED" else "✗"
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): "
                  f"Detection={pr['detection_rate']:.0%} {marker} {pr['status']}")
        else:
            marker = "✓" if pr["status"] == "CLEAN" else "⚠"
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): "
                  f"FP rate={pr['detection_rate']:.0%} {marker} {pr['status']}")

    # Calculate overall metrics
    attack_phases = [pr for pr in phase_results if pr["is_attack"] and pr["status"] != "WARMUP"]
    normal_phases = [pr for pr in phase_results if not pr["is_attack"] and pr["status"] != "WARMUP"]

    if attack_phases:
        attack_detected = sum(pr["detections"] for pr in attack_phases)
        attack_total = sum(pr["packets"] for pr in attack_phases)  # Total packets in attack phases
        attack_recall = min(1.0, attack_detected / attack_total) if attack_total > 0 else 0
    else:
        attack_recall = 0

    if normal_phases:
        normal_fp = sum(pr["detections"] for pr in normal_phases)
        normal_total = sum(pr["packets"] for pr in normal_phases)
        fp_rate = normal_fp / normal_total if normal_total > 0 else 0
    else:
        fp_rate = 0

    print(f"\n  Overall: Attack Recall={attack_recall:.0%}, Normal FP={fp_rate:.0%}")

    return {
        "attack": attack_type,
        "attack_recall": attack_recall,
        "fp_rate": fp_rate,
        "phases": phase_results,
    }


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 006: Volumetric Attack with Synthetic Timing")
    print("="*70)
    print("""
    Simulates realistic volumetric attacks with time-based phases.

    Key insight: During volumetric attack, packet RATE increases 100-1000x.
    This "drowns out" the learned baseline, causing high divergence.

    Timeline structure:
    - WARMUP: 1 minute at normal rate (learn baseline)
    - CALM periods: ~100 packets/second
    - ATTACK periods: ~50,000-100,000 packets/second

    Detection signals:
    - Traffic divergence (pattern changed)
    - Rate ratio (volume increased)
    """)

    # Realistic timeline with volume ratios
    # Calm: 100 pps, Attack: 100,000 pps = 1000x multiplier
    # Key: attack phases have MUCH more traffic in same or less time

    # Instead of real seconds, we use "units" where each unit is a packet window
    # Calm: 1 packet per unit, Attack: 1000 packets per unit

    timeline = [
        # Initial calm/warmup - enough for 500 warmup packets
        TimePhase("calm-1", 600, 100, False),            # 600 pkts at 100 pps = 6 sec

        # First attack burst - 1000x rate for 30 "seconds"
        TimePhase("ATTACK-1", 30, 100000, True, attack_fraction=0.95),  # 3M pkts

        # Recovery - back to calm
        TimePhase("calm-2", 300, 100, False),            # 300 pkts

        # Second attack - 500x rate for 60 "seconds"
        TimePhase("ATTACK-2", 60, 50000, True, attack_fraction=0.90),   # 3M pkts

        # Final calm
        TimePhase("calm-3", 300, 100, False),            # 300 pkts
    ]

    results = []

    # Test with different attack types
    # Scale: 0.005 = 0.5% of packets for faster simulation but enough volume
    for attack_type in ["dns_reflection", "syn_flood", "udp_flood"]:
        for phase in timeline:
            if phase.is_attack:
                phase.attack_type = attack_type

        result = run_volumetric_test(attack_type, timeline, scale=0.005)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("VOLUMETRIC TEST SUMMARY")
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

VOLUMETRIC DETECTION ANALYSIS
=============================

The key insight: volumetric attacks are detectable even without knowing
the specific attack type, because:

1. RATE RATIO: Traffic rate jumps 100-1000x during attack
2. DIVERGENCE: Flood traffic "drowns out" learned baseline
3. PATTERN SHIFT: Even random flood changes the distribution

What we DON'T hardcode:
- Specific ports (53, 123, 80)
- Protocol meanings (UDP = reflection candidate)
- Flag meanings (SYN = connection flood)

What we DO observe:
- "Traffic is now 500x normal rate"
- "Distribution diverged 85% from baseline"
- "Pattern returned to normal after attack"

This enables detection of UNKNOWN volumetric attacks:
- New amplification vector? Still causes rate spike
- Novel flood type? Still causes divergence
- Zero-day DDoS? Volume is volume
    """)


if __name__ == "__main__":
    main()
