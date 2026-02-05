#!/usr/bin/env python3
"""
Challenge 011-004: Adaptive Decay Mechanics

Explore different decay strategies for the accumulator:

1. Fixed decay: decay=0.9995 (~2000 observation window)
2. Rate-adaptive: adjust decay based on traffic rate
3. Time-based: decay based on elapsed time, not observations
4. Multi-horizon: maintain fast + slow accumulators

Problem: At 100k pkt/sec DDoS, decay=0.9995 gives 20ms window.
Want: Configurable "influential period" in time units.

Key insight: decay^n = 0.5 when n = ln(0.5) / ln(decay)
So decay=0.9995 → half-life of 1386 observations.
"""

import sys
import time
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# DECAY CALCULATORS
# =============================================================================

def decay_for_half_life(half_life_observations: int) -> float:
    """
    Calculate decay factor for a given half-life in observations.

    decay^n = 0.5 → decay = 0.5^(1/n)
    """
    return 0.5 ** (1.0 / half_life_observations)


def half_life_for_decay(decay: float) -> float:
    """Calculate half-life in observations for a given decay factor."""
    if decay >= 1.0:
        return float('inf')
    return math.log(0.5) / math.log(decay)


# =============================================================================
# ACCUMULATOR VARIANTS
# =============================================================================

class FixedDecayAccumulator:
    """Standard fixed decay accumulator."""

    def __init__(self, dimensions: int, decay: float):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0):
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)

    def get_half_life(self) -> float:
        return half_life_for_decay(self.decay)


class TimeBasedDecayAccumulator:
    """
    Time-based decay: decay is computed from elapsed time, not observation count.

    This ensures consistent "memory" regardless of traffic rate.
    """

    def __init__(self, dimensions: int, half_life_seconds: float):
        self.dimensions = dimensions
        self.half_life_seconds = half_life_seconds
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.last_update_time = None
        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0, current_time: float = None):
        if current_time is None:
            current_time = time.time()

        # Apply time-based decay to existing accumulator
        if self.last_update_time is not None:
            elapsed = current_time - self.last_update_time
            # Continuous decay: decay = exp(-ln(2) * elapsed / half_life)
            decay_factor = math.exp(-math.log(2) * elapsed / self.half_life_seconds)
            self.accumulator = decay_factor * self.accumulator

        # Add new observation
        self.accumulator = self.accumulator + weight * vector.astype(np.float64)
        self.last_update_time = current_time
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


class RateAdaptiveDecayAccumulator:
    """
    Rate-adaptive decay: adjusts decay to maintain consistent time window.

    Target: effective_window_seconds worth of observations.
    """

    def __init__(self, dimensions: int, target_window_seconds: float):
        self.dimensions = dimensions
        self.target_window_seconds = target_window_seconds
        self.accumulator = np.zeros(dimensions, dtype=np.float64)

        # Rate estimation
        self.recent_times = deque(maxlen=100)
        self.current_rate = 100.0  # Default: 100 obs/sec
        self.decay = 0.999  # Initial decay

        self.count = 0

    def _estimate_rate(self, current_time: float) -> float:
        """Estimate current observation rate."""
        self.recent_times.append(current_time)

        if len(self.recent_times) < 10:
            return self.current_rate

        time_span = self.recent_times[-1] - self.recent_times[0]
        if time_span < 0.001:
            return self.current_rate

        return len(self.recent_times) / time_span

    def _compute_decay(self, rate: float) -> float:
        """
        Compute decay to achieve target window at current rate.

        half_life_obs = rate * target_window_seconds / 2
        decay = 0.5^(1/half_life_obs)
        """
        half_life_obs = rate * self.target_window_seconds / 2
        half_life_obs = max(10, half_life_obs)  # Minimum half-life
        return 0.5 ** (1.0 / half_life_obs)

    def update(self, vector: np.ndarray, weight: float = 1.0, current_time: float = None):
        if current_time is None:
            current_time = time.time()

        # Update rate estimate
        self.current_rate = self._estimate_rate(current_time)
        self.decay = self._compute_decay(self.current_rate)

        # Apply decay and update
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)

    def get_current_decay(self) -> float:
        return self.decay

    def get_current_half_life(self) -> float:
        return half_life_for_decay(self.decay)


class MultiHorizonAccumulator:
    """
    Multi-horizon: maintain fast (short-term) and slow (long-term) accumulators.

    Fast: captures recent trends
    Slow: captures baseline/normal patterns

    Divergence between them signals change (attack transition).
    """

    def __init__(
        self,
        dimensions: int,
        fast_half_life: int = 100,
        slow_half_life: int = 5000,
    ):
        self.dimensions = dimensions

        fast_decay = decay_for_half_life(fast_half_life)
        slow_decay = decay_for_half_life(slow_half_life)

        self.fast = FixedDecayAccumulator(dimensions, fast_decay)
        self.slow = FixedDecayAccumulator(dimensions, slow_decay)

        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0):
        self.fast.update(vector, weight)
        self.slow.update(vector, weight)
        self.count += 1

    def get_fast_normalized(self) -> np.ndarray:
        return self.fast.get_normalized()

    def get_slow_normalized(self) -> np.ndarray:
        return self.slow.get_normalized()

    def get_divergence(self) -> float:
        """
        Measure divergence between fast and slow accumulators.

        High divergence = traffic pattern is changing.
        """
        fast_norm = self.get_fast_normalized()
        slow_norm = self.get_slow_normalized()
        return 1.0 - cosine_similarity(fast_norm, slow_norm)


# =============================================================================
# TRAFFIC SIMULATOR
# =============================================================================

class TrafficSimulator:
    """
    Simulate traffic at different rates with pattern changes.
    """

    def __init__(self, seed: int = 42):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.encoder = Encoder(vector_manager=self.vm)
        self.rng = random.Random(seed)

    def _normal_packet(self) -> dict:
        return {
            "protocol": self.rng.choice(["TCP", "UDP"]),
            "dst_port": self.rng.choice([443, 80, 8080]),
            "flags": self.rng.choice([0x10, 0x12, 0x18]),
        }

    def _attack_packet(self) -> dict:
        return {
            "protocol": "TCP",
            "dst_port": 443,
            "flags": 0x02,  # SYN only
        }

    def generate_variable_rate_stream(
        self,
        phases: List[Tuple[str, float, float]],  # (type, duration_sec, rate_per_sec)
    ) -> List[Tuple[dict, float, str]]:
        """
        Generate stream with variable rates.

        Returns: [(packet, timestamp, phase_type), ...]
        """
        stream = []
        current_time = 0.0

        for phase_type, duration, rate in phases:
            phase_end = current_time + duration
            interval = 1.0 / rate

            while current_time < phase_end:
                if phase_type == "normal":
                    packet = self._normal_packet()
                else:
                    packet = self._attack_packet()

                stream.append((packet, current_time, phase_type))
                current_time += interval * (0.8 + 0.4 * self.rng.random())  # Add jitter

        return stream


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_fixed_decay():
    """Compare different fixed decay values."""
    print("\n" + "=" * 70)
    print("Experiment 1: Fixed Decay Values")
    print("=" * 70)

    decay_values = [0.99, 0.995, 0.999, 0.9995, 0.9999]

    print(f"\n{'Decay':<12} {'Half-life':<15} {'Window (obs)':<15}")
    print("-" * 45)

    for decay in decay_values:
        hl = half_life_for_decay(decay)
        window = hl * 2  # Roughly 95% of influence within 3 half-lives
        print(f"{decay:<12.4f} {hl:<15.1f} {window:<15.1f}")

    # Simulate impact on DDoS detection
    print("\n--- Impact on DDoS Detection (100k pkt/sec) ---")
    rate = 100000

    for decay in decay_values:
        hl = half_life_for_decay(decay)
        window_time = hl / rate * 1000  # ms
        print(f"  decay={decay}: half-life = {window_time:.1f} ms")


def experiment_time_based():
    """Compare time-based vs observation-based decay."""
    print("\n" + "=" * 70)
    print("Experiment 2: Time-Based vs Observation-Based")
    print("=" * 70)

    sim = TrafficSimulator(seed=42)

    # Variable rate stream: slow → fast → slow
    phases = [
        ("normal", 1.0, 100),    # 1 sec at 100/sec (slow)
        ("normal", 0.1, 10000),  # 0.1 sec at 10k/sec (burst)
        ("normal", 1.0, 100),    # 1 sec at 100/sec (slow again)
    ]

    stream = sim.generate_variable_rate_stream(phases)
    print(f"\nStream: {len(stream)} packets over ~2.1 seconds")
    print(f"  Slow phase: 100 pkt/sec")
    print(f"  Burst phase: 10,000 pkt/sec")

    # Track similarity evolution for each accumulator type
    fixed_accum = FixedDecayAccumulator(DIMENSIONS, decay=0.999)
    time_accum = TimeBasedDecayAccumulator(DIMENSIONS, half_life_seconds=0.5)

    fixed_sims = []
    time_sims = []

    for packet, timestamp, phase in stream:
        vec = sim.encoder.encode_data(packet)

        # Get similarities before update
        fixed_sim = cosine_similarity(vec, fixed_accum.get_normalized())
        time_sim = cosine_similarity(vec, time_accum.get_normalized())

        fixed_sims.append((timestamp, fixed_sim))
        time_sims.append((timestamp, time_sim))

        # Update
        fixed_accum.update(vec)
        time_accum.update(vec, current_time=timestamp)

    # Sample at phase transitions
    print("\n--- Similarity at Phase Transitions ---")
    print(f"{'Time':<10} {'Fixed':<12} {'Time-based':<12} {'Phase':<12}")
    print("-" * 50)

    checkpoints = [0.0, 0.99, 1.0, 1.05, 1.1, 2.0]
    for cp in checkpoints:
        # Find closest sample
        fixed_idx = min(range(len(fixed_sims)), key=lambda i: abs(fixed_sims[i][0] - cp))
        time_idx = min(range(len(time_sims)), key=lambda i: abs(time_sims[i][0] - cp))

        phase = "slow" if cp < 1.0 or cp > 1.1 else "burst"
        print(f"{cp:<10.2f} {fixed_sims[fixed_idx][1]:<12.3f} {time_sims[time_idx][1]:<12.3f} {phase:<12}")


def experiment_rate_adaptive():
    """Test rate-adaptive decay."""
    print("\n" + "=" * 70)
    print("Experiment 3: Rate-Adaptive Decay")
    print("=" * 70)

    sim = TrafficSimulator(seed=42)

    # Variable rate with attack
    phases = [
        ("normal", 1.0, 100),     # Normal at 100/sec
        ("attack", 0.5, 10000),   # Attack burst at 10k/sec
        ("normal", 1.0, 100),     # Back to normal
    ]

    stream = sim.generate_variable_rate_stream(phases)
    print(f"\nStream: {len(stream)} packets")
    print(f"  Phase 1: normal at 100/sec (1 sec)")
    print(f"  Phase 2: attack at 10k/sec (0.5 sec)")
    print(f"  Phase 3: normal at 100/sec (1 sec)")

    # Rate-adaptive accumulator targeting 5-second window
    rate_accum = RateAdaptiveDecayAccumulator(DIMENSIONS, target_window_seconds=5.0)

    decay_history = []
    for packet, timestamp, phase in stream:
        vec = sim.encoder.encode_data(packet)
        rate_accum.update(vec, current_time=timestamp)
        decay_history.append((timestamp, rate_accum.get_current_decay(), phase))

    # Show decay adaptation
    print("\n--- Decay Adaptation ---")
    print(f"{'Time':<10} {'Decay':<12} {'Half-life':<12} {'Phase':<10}")
    print("-" * 45)

    for cp in [0.5, 1.0, 1.1, 1.25, 1.5, 2.0]:
        idx = min(range(len(decay_history)), key=lambda i: abs(decay_history[i][0] - cp))
        ts, decay, phase = decay_history[idx]
        hl = half_life_for_decay(decay)
        print(f"{ts:<10.2f} {decay:<12.6f} {hl:<12.1f} {phase:<10}")


def experiment_multi_horizon():
    """Test multi-horizon accumulator for change detection."""
    print("\n" + "=" * 70)
    print("Experiment 4: Multi-Horizon Change Detection")
    print("=" * 70)

    sim = TrafficSimulator(seed=42)

    # Normal → Attack → Normal
    phases = [
        ("normal", 1.0, 1000),    # Normal
        ("attack", 0.5, 1000),    # Attack
        ("normal", 1.0, 1000),    # Recovery
    ]

    stream = sim.generate_variable_rate_stream(phases)
    print(f"\nStream: {len(stream)} packets")

    # Multi-horizon: fast (100 obs half-life) + slow (2000 obs half-life)
    mh_accum = MultiHorizonAccumulator(DIMENSIONS, fast_half_life=100, slow_half_life=2000)

    divergence_history = []

    for packet, timestamp, phase in stream:
        vec = sim.encoder.encode_data(packet)
        mh_accum.update(vec)

        if mh_accum.count >= 50:  # Need some warmup
            div = mh_accum.get_divergence()
            divergence_history.append((timestamp, div, phase))

    # Show divergence at transitions
    print("\n--- Fast/Slow Divergence ---")
    print(f"{'Time':<10} {'Divergence':<12} {'Phase':<12}")
    print("-" * 35)

    for cp in [0.5, 0.95, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]:
        if divergence_history:
            idx = min(range(len(divergence_history)), key=lambda i: abs(divergence_history[i][0] - cp))
            ts, div, phase = divergence_history[idx]
            marker = "<<<" if div > 0.1 else ""
            print(f"{ts:<10.2f} {div:<12.4f} {phase:<12} {marker}")

    print("""
Key insight: High divergence indicates the fast accumulator
has diverged from the slow baseline → pattern change detected.

This enables:
- Attack start detection: fast shows attack, slow shows normal
- Attack end detection: fast shows normal, slow still shows attack
""")


def main():
    print("=" * 80)
    print("Challenge 011-004: Adaptive Decay Mechanics")
    print("=" * 80)
    print("""
Problem: Fixed decay doesn't adapt to traffic rate.
  - decay=0.9995 → half-life of ~1386 observations
  - At 100 pkt/sec → 13.86 seconds of "memory"
  - At 100k pkt/sec → 13.86 ms of "memory"

Solutions explored:
1. Fixed decay: tune for expected rate
2. Time-based: decay based on elapsed time
3. Rate-adaptive: adjust decay to maintain time window
4. Multi-horizon: fast + slow for change detection
""")

    experiment_fixed_decay()
    experiment_time_based()
    experiment_rate_adaptive()
    experiment_multi_horizon()

    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print("""
Key findings:

1. FIXED DECAY: Simple but rate-dependent
   - Good for stable-rate environments
   - Fails when rate varies (DDoS bursts)

2. TIME-BASED: Consistent memory regardless of rate
   - half_life_seconds directly controls "influence period"
   - decay = exp(-ln(2) * elapsed / half_life)
   - Best for environments with variable rates

3. RATE-ADAPTIVE: Automatic adjustment
   - Estimates current rate from recent observations
   - Computes decay to achieve target window
   - Good for unknown/changing environments

4. MULTI-HORIZON: Best for change detection
   - Fast accumulator: recent 100 obs
   - Slow accumulator: recent 2000 obs
   - Divergence = pattern change signal
   - Enables attack start/end detection

Recommendations:
- Stable rate: Fixed decay (simple, fast)
- Variable rate: Time-based or rate-adaptive
- Change detection: Multi-horizon
- DDoS: Multi-horizon + checkpoints (experiment 005)
""")


if __name__ == "__main__":
    main()
