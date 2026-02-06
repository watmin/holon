#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 009: Accuracy Improvements
=============================================================================

Exploring techniques to reduce false positives without hardcoded values:

1. RESONANCE FILTERING
   - Extract the "normal" part of a packet using resonance()
   - Measure how much is LEFT OVER (the anomalous residual)
   - High residual = anomaly

2. MULTI-SCALE ANALYSIS
   - Fast accumulator (short memory, quick to adapt)
   - Slow accumulator (long memory, stable baseline)
   - Anomaly when fast diverges FROM slow (not from frozen baseline)

3. TREND DETECTION
   - Track the DERIVATIVE of similarity
   - Sudden drop = more concerning than gradually low value

4. CONFIDENCE-WEIGHTED THRESHOLDS
   - During warmup, measure baseline STABILITY
   - Unstable baseline → require stronger signal
   - Stable baseline → can use tighter threshold

5. CODEBOOK CLEANUP
   - Build codebook of "normal" prototypes
   - Use cleanup() to snap to nearest normal
   - Large cleanup distance = anomaly

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/009-accuracy-improvements.py
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


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
WARMUP_PACKETS = 400


# =============================================================================
# TECHNIQUE 1: RESONANCE-BASED RESIDUAL DETECTION
# =============================================================================

class ResidualDetector:
    """
    Uses resonance() to extract the "normal" part, then measures residual.

    Key insight: resonance(packet, baseline) keeps only the parts that
    AGREE with baseline. What's LEFT is the anomalous part.

    residual_ratio = |packet - resonance(packet, baseline)| / |packet|
    """

    def __init__(self, store: CPUStore):
        self.store = store
        self.encoder = store.encoder
        self.baseline_accum = self.encoder.create_accumulator()
        self._baseline_norm = None
        self.frozen = False

        # Track residual ratios during warmup
        self.warmup_residuals = []
        self.residual_mean = 0.5
        self.residual_std = 0.1

    def observe(self, packet_vec: np.ndarray, is_warmup: bool):
        if is_warmup:
            self.baseline_accum = self.encoder.accumulate(self.baseline_accum, packet_vec)

            # Sample residuals after initial accumulation
            if len(self.warmup_residuals) > 20:
                temp_norm = self.encoder.normalize_accumulator(self.baseline_accum)
                ratio = self._compute_residual_ratio(packet_vec, temp_norm)
                self.warmup_residuals.append(ratio)

    def freeze(self):
        self.frozen = True
        self._baseline_norm = self.encoder.normalize_accumulator(self.baseline_accum)

        if self.warmup_residuals:
            self.residual_mean = np.mean(self.warmup_residuals)
            self.residual_std = np.std(self.warmup_residuals) if len(self.warmup_residuals) > 1 else 0.1

    def _compute_residual_ratio(self, packet_vec: np.ndarray, baseline: np.ndarray) -> float:
        """Compute how much of the packet is NOT explained by baseline."""
        # Get the resonating part (what agrees with baseline)
        resonating = self.store.resonance(packet_vec, baseline)

        # Residual is what's left
        residual = packet_vec.astype(float) - resonating.astype(float)

        # Ratio: what fraction of the packet is unexplained?
        packet_norm = np.linalg.norm(packet_vec)
        residual_norm = np.linalg.norm(residual)

        if packet_norm < 1e-10:
            return 0.0

        return residual_norm / packet_norm

    def get_residual_anomaly(self, packet_vec: np.ndarray) -> Tuple[bool, float, float]:
        """Check if packet has high unexplained residual."""
        if not self.frozen:
            return False, 0.0, 0.0

        ratio = self._compute_residual_ratio(packet_vec, self._baseline_norm)

        # Anomaly if residual is > 2 std above mean
        threshold = self.residual_mean + 2 * self.residual_std

        return ratio > threshold, ratio, threshold


# =============================================================================
# TECHNIQUE 2: MULTI-SCALE ANALYSIS
# =============================================================================

class MultiScaleTracker:
    """
    Tracks at multiple timescales:
    - Fast (decay=0.95): Adapts quickly, captures recent behavior
    - Slow (decay=0.995): Stable baseline, resistant to transients

    Anomaly when fast DIVERGES from slow (not from frozen baseline).
    This allows the detector to adapt to gradual drift while catching sudden changes.
    """

    def __init__(self, store: CPUStore, fast_decay: float = 0.95, slow_decay: float = 0.995):
        self.store = store
        self.encoder = store.encoder

        self.fast_accum = self.encoder.create_accumulator()
        self.slow_accum = self.encoder.create_accumulator()

        self.fast_decay = fast_decay
        self.slow_decay = slow_decay

        # Track fast-slow divergence during warmup
        self.warmup_divergences = []
        self.divergence_mean = 0.1
        self.divergence_std = 0.05

    def observe(self, packet_vec: np.ndarray, is_warmup: bool):
        # Update both accumulators
        vec_float = packet_vec.astype(np.float64)
        self.fast_accum = self.fast_decay * self.fast_accum + vec_float
        self.slow_accum = self.slow_decay * self.slow_accum + vec_float

        if is_warmup and np.linalg.norm(self.slow_accum) > 100:
            div = self._get_divergence()
            self.warmup_divergences.append(div)

    def freeze(self):
        if self.warmup_divergences:
            self.divergence_mean = np.mean(self.warmup_divergences)
            self.divergence_std = np.std(self.warmup_divergences) if len(self.warmup_divergences) > 1 else 0.05

    def _get_divergence(self) -> float:
        """Get divergence between fast and slow accumulators."""
        fast_norm = self.encoder.normalize_accumulator(self.fast_accum)
        slow_norm = self.encoder.normalize_accumulator(self.slow_accum)

        sim = self.store.similarity(fast_norm, slow_norm, metric="cosine")
        return 1.0 - sim

    def get_multiscale_anomaly(self) -> Tuple[bool, float, float]:
        """Check if fast has diverged from slow."""
        divergence = self._get_divergence()

        # Anomaly if divergence > 3 std above mean
        threshold = self.divergence_mean + 3 * self.divergence_std

        return divergence > threshold, divergence, threshold


# =============================================================================
# TECHNIQUE 3: TREND DETECTION
# =============================================================================

class TrendDetector:
    """
    Tracks the DERIVATIVE of similarity, not just the value.

    A sudden DROP in similarity is more concerning than a gradually low value.

    Detects:
    - Rapid decrease in similarity (attack onset)
    - Sustained low similarity (ongoing attack)
    """

    def __init__(self, store: CPUStore, window_size: int = 20):
        self.store = store
        self.encoder = store.encoder

        self.similarity_history = deque(maxlen=window_size)
        self.derivative_history = deque(maxlen=window_size)

        # Learned statistics
        self.derivative_mean = 0.0
        self.derivative_std = 0.02

        self.baseline_accum = self.encoder.create_accumulator()
        self._baseline_norm = None
        self.frozen = False

        # Warmup tracking
        self.warmup_derivatives = []

    def observe(self, packet_vec: np.ndarray, is_warmup: bool):
        if is_warmup:
            self.baseline_accum = self.encoder.accumulate(self.baseline_accum, packet_vec)

            # Track similarity during warmup
            if len(self.similarity_history) > 5:
                temp_norm = self.encoder.normalize_accumulator(self.baseline_accum)
                sim = self.store.similarity(packet_vec, temp_norm, metric="cosine")

                if self.similarity_history:
                    derivative = sim - self.similarity_history[-1]
                    self.derivative_history.append(derivative)
                    self.warmup_derivatives.append(derivative)

                self.similarity_history.append(sim)

    def freeze(self):
        self.frozen = True
        self._baseline_norm = self.encoder.normalize_accumulator(self.baseline_accum)

        if self.warmup_derivatives:
            self.derivative_mean = np.mean(self.warmup_derivatives)
            self.derivative_std = np.std(self.warmup_derivatives) if len(self.warmup_derivatives) > 1 else 0.02

    def update(self, packet_vec: np.ndarray):
        """Update with new packet (post-warmup)."""
        if not self.frozen:
            return

        sim = self.store.similarity(packet_vec, self._baseline_norm, metric="cosine")

        if self.similarity_history:
            derivative = sim - self.similarity_history[-1]
            self.derivative_history.append(derivative)

        self.similarity_history.append(sim)

    def get_trend_anomaly(self) -> Tuple[bool, float, float, str]:
        """Check for anomalous trend."""
        if len(self.derivative_history) < 3:
            return False, 0.0, 0.0, ""

        # Recent derivative (smoothed)
        recent_derivative = np.mean(list(self.derivative_history)[-5:])

        # Check for rapid DROP (negative derivative)
        drop_threshold = self.derivative_mean - 3 * self.derivative_std

        if recent_derivative < drop_threshold:
            return True, recent_derivative, drop_threshold, "rapid_drop"

        return False, recent_derivative, drop_threshold, "stable"


# =============================================================================
# TECHNIQUE 4: CONFIDENCE-WEIGHTED THRESHOLDS
# =============================================================================

class ConfidenceWeightedDetector:
    """
    Weights detection threshold by baseline STABILITY.

    - High stability (low variance) → tight threshold (sensitive)
    - Low stability (high variance) → loose threshold (conservative)

    This prevents FPs when baseline is noisy.
    """

    def __init__(self, store: CPUStore, base_std_multiplier: float = 2.0):
        self.store = store
        self.encoder = store.encoder

        self.baseline_accum = self.encoder.create_accumulator()
        self._baseline_norm = None
        self.frozen = False

        self.base_multiplier = base_std_multiplier

        # Warmup tracking
        self.warmup_sims = []
        self.sim_mean = 0.5
        self.sim_std = 0.1
        self.stability_score = 1.0  # 0=unstable, 1=stable

    def observe(self, packet_vec: np.ndarray, is_warmup: bool):
        if is_warmup:
            self.baseline_accum = self.encoder.accumulate(self.baseline_accum, packet_vec)

            if len(self.warmup_sims) > 10:
                temp_norm = self.encoder.normalize_accumulator(self.baseline_accum)
                sim = self.store.similarity(packet_vec, temp_norm, metric="cosine")
                self.warmup_sims.append(sim)

    def freeze(self):
        self.frozen = True
        self._baseline_norm = self.encoder.normalize_accumulator(self.baseline_accum)

        if self.warmup_sims:
            self.sim_mean = np.mean(self.warmup_sims)
            self.sim_std = np.std(self.warmup_sims) if len(self.warmup_sims) > 1 else 0.1

            # Stability score: low std = stable = 1.0, high std = unstable = 0.0
            # Typical std is 0.05-0.15
            self.stability_score = max(0.0, min(1.0, 1.0 - (self.sim_std - 0.05) / 0.10))

    def get_adaptive_threshold(self) -> float:
        """
        Get threshold adapted to baseline stability.

        Stable baseline: use 2.0 * std (sensitive)
        Unstable baseline: use 4.0 * std (conservative)
        """
        # Adaptive multiplier: stable → 2.0, unstable → 4.0
        adaptive_mult = self.base_multiplier + (1.0 - self.stability_score) * 2.0

        return self.sim_mean - adaptive_mult * self.sim_std

    def check(self, packet_vec: np.ndarray) -> Tuple[bool, float, float]:
        if not self.frozen:
            return False, 0.0, 0.0

        sim = self.store.similarity(packet_vec, self._baseline_norm, metric="cosine")
        threshold = self.get_adaptive_threshold()

        return sim < threshold, sim, threshold


# =============================================================================
# TECHNIQUE 5: CODEBOOK CLEANUP
# =============================================================================

class CodebookDetector:
    """
    Build a codebook of "normal" patterns during warmup.
    Use cleanup() to snap observations to nearest normal.
    Large cleanup distance = anomaly.
    """

    def __init__(self, store: CPUStore, codebook_size: int = 10):
        self.store = store
        self.encoder = store.encoder

        self.codebook_size = codebook_size
        self.codebook = []
        self.warmup_vectors = []

        # Track cleanup distances during warmup
        self.warmup_distances = []
        self.distance_mean = 0.3
        self.distance_std = 0.1

    def observe(self, packet_vec: np.ndarray, is_warmup: bool):
        if is_warmup:
            self.warmup_vectors.append(packet_vec.copy())

    def freeze(self):
        """Build codebook from warmup vectors."""
        if len(self.warmup_vectors) < self.codebook_size:
            self.codebook = self.warmup_vectors
            return

        # Sample representative vectors
        # Simple approach: evenly spaced samples
        step = len(self.warmup_vectors) // self.codebook_size
        self.codebook = [self.warmup_vectors[i * step] for i in range(self.codebook_size)]

        # Optionally: use prototype() to get archetypal patterns
        # For now, just compute cleanup distances on warmup data
        for vec in self.warmup_vectors[-100:]:
            dist = self._cleanup_distance(vec)
            self.warmup_distances.append(dist)

        if self.warmup_distances:
            self.distance_mean = np.mean(self.warmup_distances)
            self.distance_std = np.std(self.warmup_distances) if len(self.warmup_distances) > 1 else 0.1

    def _cleanup_distance(self, vec: np.ndarray) -> float:
        """Get distance from vec to nearest codebook entry."""
        if not self.codebook:
            return 0.0

        best_sim = -1.0
        for cb_vec in self.codebook:
            sim = self.store.similarity(vec, cb_vec, metric="cosine")
            best_sim = max(best_sim, sim)

        # Convert similarity to distance
        return 1.0 - best_sim

    def check(self, packet_vec: np.ndarray) -> Tuple[bool, float, float]:
        if not self.codebook:
            return False, 0.0, 0.0

        distance = self._cleanup_distance(packet_vec)

        # Anomaly if distance > 2.5 std above mean
        threshold = self.distance_mean + 2.5 * self.distance_std

        return distance > threshold, distance, threshold


# =============================================================================
# COMBINED DETECTOR
# =============================================================================

class ImprovedDetector:
    """
    Combines multiple techniques with voting.

    Anomaly requires agreement from multiple signals.
    """

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Technique detectors
        self.residual = ResidualDetector(self.store)
        self.multiscale = MultiScaleTracker(self.store)
        self.trend = TrendDetector(self.store)
        self.confidence = ConfidenceWeightedDetector(self.store)
        self.codebook = CodebookDetector(self.store)

        # Rate tracking (from 008)
        self.rate_encoder = ContinuousRateEncoder(DIMENSIONS)
        self.rate_accum = self.encoder.create_accumulator()
        self._rate_norm = None
        self.rate_warmup_sims = []
        self.rate_mean = 1.0
        self.rate_std = 0.0

        # Smoothing
        self.anomaly_history = deque(maxlen=15)

    def process(self, packet: dict, pps: float) -> dict:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # Encode packet
        packet_vec = self.encoder.encode_data(packet)
        rate_vec = self.rate_encoder.encode_rate(pps)

        # Update all detectors
        self.residual.observe(packet_vec, is_warmup)
        self.multiscale.observe(packet_vec, is_warmup)
        self.trend.observe(packet_vec, is_warmup)
        self.confidence.observe(packet_vec, is_warmup)
        self.codebook.observe(packet_vec, is_warmup)

        # Rate tracking
        if is_warmup:
            self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
            if len(self.rate_warmup_sims) > 10:
                temp_norm = self.encoder.normalize_accumulator(self.rate_accum)
                sim = self.store.similarity(rate_vec, temp_norm, metric="cosine")
                self.rate_warmup_sims.append(sim)

        if is_warmup:
            if self.packet_count == self.warmup_packets:
                self._freeze_all()

            return {
                "packet_num": self.packet_count,
                "is_anomalous": False,
                "signals": {},
                "explanation": "Warming up...",
            }

        # Post-warmup: check all detectors
        self.trend.update(packet_vec)

        signals = {}

        # 1. Residual check
        res_anom, res_val, res_thresh = self.residual.get_residual_anomaly(packet_vec)
        signals["residual"] = {"anomalous": res_anom, "value": res_val, "threshold": res_thresh}

        # 2. Multi-scale check
        ms_anom, ms_val, ms_thresh = self.multiscale.get_multiscale_anomaly()
        signals["multiscale"] = {"anomalous": ms_anom, "value": ms_val, "threshold": ms_thresh}

        # 3. Trend check
        tr_anom, tr_val, tr_thresh, tr_type = self.trend.get_trend_anomaly()
        signals["trend"] = {"anomalous": tr_anom, "value": tr_val, "threshold": tr_thresh, "type": tr_type}

        # 4. Confidence-weighted check
        cw_anom, cw_val, cw_thresh = self.confidence.check(packet_vec)
        signals["confidence"] = {"anomalous": cw_anom, "value": cw_val, "threshold": cw_thresh}

        # 5. Codebook check
        cb_anom, cb_val, cb_thresh = self.codebook.check(packet_vec)
        signals["codebook"] = {"anomalous": cb_anom, "value": cb_val, "threshold": cb_thresh}

        # 6. Rate check
        rate_anom, rate_val, rate_thresh = self._check_rate(rate_vec)
        signals["rate"] = {"anomalous": rate_anom, "value": rate_val, "threshold": rate_thresh}

        # Voting: require 2+ signals OR rate anomaly
        anomaly_votes = sum([
            signals["residual"]["anomalous"],
            signals["multiscale"]["anomalous"],
            signals["trend"]["anomalous"],
            signals["confidence"]["anomalous"],
            signals["codebook"]["anomalous"],
        ])

        is_anomalous = rate_anom or (anomaly_votes >= 2)

        # Smooth detection
        self.anomaly_history.append(is_anomalous)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = anomaly_rate > 0.5

        # Build explanation
        triggered = [name for name, sig in signals.items() if sig.get("anomalous", False)]

        return {
            "packet_num": self.packet_count,
            "is_anomalous": is_anomalous,
            "signals": signals,
            "votes": anomaly_votes,
            "triggered": triggered,
            "explanation": f"Votes: {anomaly_votes}/5, Triggered: {triggered}" if triggered else "Normal",
        }

    def _freeze_all(self):
        self.warmup_complete = True
        self.residual.freeze()
        self.multiscale.freeze()
        self.trend.freeze()
        self.confidence.freeze()
        self.codebook.freeze()

        self._rate_norm = self.encoder.normalize_accumulator(self.rate_accum)
        if self.rate_warmup_sims:
            self.rate_mean = np.mean(self.rate_warmup_sims)
            self.rate_std = np.std(self.rate_warmup_sims) if len(self.rate_warmup_sims) > 1 else 0.05

    def _check_rate(self, rate_vec: np.ndarray) -> Tuple[bool, float, float]:
        if self._rate_norm is None:
            return False, 1.0, 0.0

        sim = self.store.similarity(rate_vec, self._rate_norm, metric="cosine")
        threshold = self.rate_mean - 2.5 * self.rate_std

        return sim < threshold, sim, threshold


# =============================================================================
# CONTINUOUS RATE ENCODER (from 008)
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
    print(f"IMPROVED DETECTOR TEST: {attack_type}")
    print(f"{'='*70}")

    first_calm_packets = int(phases[0].duration_seconds * phases[0].packets_per_second * scale)
    warmup_packets = min(first_calm_packets - 10, 400)

    detector = ImprovedDetector(warmup_packets=warmup_packets)
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Process phases
    phase_results = []

    for phase in phases:
        scaled_packets = max(1, int(phase.duration_seconds * phase.packets_per_second * scale))
        phase_detections = 0
        signal_counts = {
            "residual": 0, "multiscale": 0, "trend": 0,
            "confidence": 0, "codebook": 0, "rate": 0
        }

        for i in range(scaled_packets):
            if phase.phase_type == Phase.ATTACK and rng.random() < phase.attack_fraction:
                packet = attack_gen(rng)
            else:
                packet = gen_normal(rng)

            result = detector.process(packet, phase.packets_per_second)

            if detector.warmup_complete:
                if result["is_anomalous"]:
                    phase_detections += 1
                for sig_name in signal_counts:
                    if result.get("signals", {}).get(sig_name, {}).get("anomalous", False):
                        signal_counts[sig_name] += 1

        if not detector.warmup_complete or phase.name == "calm-1":
            status = "WARMUP"
            detection_rate = 0
        else:
            detection_rate = phase_detections / scaled_packets if scaled_packets > 0 else 0
            if phase.phase_type == Phase.ATTACK:
                status = "DETECTED" if detection_rate > 0.5 else "MISSED"
            else:
                status = "CLEAN" if detection_rate < 0.05 else "FP"

        phase_results.append({
            "name": phase.name,
            "phase_type": phase.phase_type,
            "packets": scaled_packets,
            "detections": phase_detections,
            "detection_rate": detection_rate,
            "signal_counts": signal_counts,
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
            sig_summary = ", ".join(f"{k[:3]}:{v}" for k, v in pr["signal_counts"].items() if v > 0)
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): "
                  f"{'Attack' if pr['phase_type'] == Phase.ATTACK else 'Normal'} "
                  f"Det={pr['detection_rate']:.0%} {marker} {pr['status']}")
            if sig_summary:
                print(f"               Signals: {sig_summary}")

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

    # Show learned parameters
    print(f"\n  Learned Parameters:")
    print(f"    Residual: mean={detector.residual.residual_mean:.3f}, std={detector.residual.residual_std:.3f}")
    print(f"    MultiScale: mean={detector.multiscale.divergence_mean:.3f}, std={detector.multiscale.divergence_std:.3f}")
    print(f"    Confidence: stability={detector.confidence.stability_score:.2f}, threshold={detector.confidence.get_adaptive_threshold():.3f}")
    print(f"    Codebook: mean_dist={detector.codebook.distance_mean:.3f}, std={detector.codebook.distance_std:.3f}")

    return {
        "attack": attack_type,
        "attack_recall": attack_recall,
        "fp_rate": fp_rate,
    }


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 009: Accuracy Improvements")
    print("="*70)
    print("""
    Testing multiple techniques to reduce false positives:

    1. RESIDUAL: resonance() extracts normal part, measure what's left
    2. MULTISCALE: Fast vs slow accumulator divergence
    3. TREND: Derivative of similarity (sudden drops)
    4. CONFIDENCE: Adaptive threshold based on baseline stability
    5. CODEBOOK: Distance to nearest "normal" prototype
    6. RATE: Positional encoding of log(rate)

    Voting: Require 2+ signals OR rate anomaly
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
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'Attack Recall':>15} {'Normal FP':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['attack']:<20} {r['attack_recall']:>15.0%} {r['fp_rate']:>12.0%}")

    avg_recall = np.mean([r['attack_recall'] for r in results])
    avg_fp = np.mean([r['fp_rate'] for r in results])
    print("-"*50)
    print(f"{'Average':<20} {avg_recall:>15.0%} {avg_fp:>12.0%}")

    print(f"\n  Comparison with 008:")
    print(f"    008: 100% recall, 8% FP")
    print(f"    009: {avg_recall:.0%} recall, {avg_fp:.0%} FP")


if __name__ == "__main__":
    main()
