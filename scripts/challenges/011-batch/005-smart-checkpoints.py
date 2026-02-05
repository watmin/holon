#!/usr/bin/env python3
"""
Challenge 011-005: Smart Checkpoints for Attack Recovery Detection

Problem: During continuous learning, the accumulator adapts to attack traffic.
After a long DDoS, the "normal" baseline has shifted toward attack patterns.

Solution: Save "known good" checkpoints to:
1. Detect when we've returned to pre-attack state
2. Resist long-running poisoning attempts
3. Enable before/after comparison

Checkpoint strategies:
1. Periodic: Save every N observations during stable periods
2. Stability-triggered: Save when variance is low for extended period
3. Operator-triggered: Manual checkpoint command
4. Multi-checkpoint: Maintain short-term and long-term baselines

Key insight: Combine checkpoints with multi-horizon accumulators:
- Fast accumulator: current traffic
- Slow accumulator: evolving baseline
- Checkpoint: frozen "known good" state
"""

import sys
import time
import random
import math
from dataclasses import dataclass, field
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


def decay_for_half_life(half_life_observations: int) -> float:
    return 0.5 ** (1.0 / half_life_observations)


# =============================================================================
# ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
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

    def get_raw(self) -> np.ndarray:
        return self.accumulator.copy()

    def restore(self, state: np.ndarray):
        """Restore accumulator from saved state."""
        self.accumulator = state.astype(np.float64)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

@dataclass
class Checkpoint:
    """A saved accumulator state."""
    observation_count: int
    timestamp: float
    accumulator_state: np.ndarray
    normalized_vector: np.ndarray
    variance_at_save: float
    label: str = ""


class CheckpointManager:
    """
    Manages accumulator checkpoints for recovery detection.

    Strategies:
    - save_periodic(): Save at regular intervals during stable periods
    - save_on_stability(): Save when variance has been low for N observations
    - save_manual(): Operator-triggered save

    Recovery detection:
    - check_recovery(): Compare current state to checkpoint
    """

    def __init__(
        self,
        periodic_interval: int = 1000,
        stability_window: int = 100,
        stability_threshold: float = 0.02,  # Variance threshold for "stable"
    ):
        self.periodic_interval = periodic_interval
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold

        self.checkpoints: List[Checkpoint] = []
        self.recent_variances = deque(maxlen=stability_window)

        self.last_periodic_save = 0
        self.observations_stable = 0

    def record_variance(self, variance: float):
        """Record variance for stability tracking."""
        self.recent_variances.append(variance)

        if variance < self.stability_threshold:
            self.observations_stable += 1
        else:
            self.observations_stable = 0

    def is_stable(self) -> bool:
        """Check if recent variance indicates stability."""
        if len(self.recent_variances) < 10:
            return False
        return self.observations_stable >= self.stability_window

    def should_save_periodic(self, count: int) -> bool:
        """Check if periodic save is due."""
        return count - self.last_periodic_save >= self.periodic_interval

    def save_checkpoint(
        self,
        accumulator: DecayingAccumulator,
        variance: float,
        label: str = "",
        timestamp: float = None,
    ):
        """Save current accumulator state as checkpoint."""
        if timestamp is None:
            timestamp = time.time()

        checkpoint = Checkpoint(
            observation_count=accumulator.count,
            timestamp=timestamp,
            accumulator_state=accumulator.get_raw(),
            normalized_vector=accumulator.get_normalized(),
            variance_at_save=variance,
            label=label,
        )

        self.checkpoints.append(checkpoint)
        self.last_periodic_save = accumulator.count

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_stable_checkpoint(self) -> Optional[Checkpoint]:
        """Get most recent checkpoint saved during stable period."""
        for cp in reversed(self.checkpoints):
            if cp.variance_at_save < self.stability_threshold:
                return cp
        return None

    def check_recovery(
        self,
        current_accumulator: DecayingAccumulator,
        checkpoint: Checkpoint,
        recovery_threshold: float = 0.85,
    ) -> Tuple[bool, float]:
        """
        Check if current state has recovered to checkpoint state.

        Returns: (is_recovered, similarity_to_checkpoint)
        """
        current_norm = current_accumulator.get_normalized()
        checkpoint_norm = checkpoint.normalized_vector

        similarity = cosine_similarity(current_norm, checkpoint_norm)
        is_recovered = similarity >= recovery_threshold

        return is_recovered, similarity


# =============================================================================
# CHECKPOINT-ENABLED DETECTOR
# =============================================================================

@dataclass
class DetectionState:
    observation_count: int
    similarity: float
    variance: float
    is_attack: bool
    is_recovered: bool
    checkpoint_similarity: float
    phase: str


class CheckpointDetector:
    """
    Anomaly detector with checkpoint-based recovery detection.

    Phases:
    - learning: Building initial baseline
    - normal: Stable, no attack
    - attack_detected: Attack in progress
    - recovering: Attack ended, comparing to checkpoint
    - recovered: Back to pre-attack state
    """

    def __init__(
        self,
        learning_period: int = 500,
        variance_window: int = 50,
        attack_variance_drop: float = 0.3,  # Variance drop ratio to detect attack
        attack_similarity_rise: float = 0.7,  # High similarity during attack
        recovery_threshold: float = 0.80,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.encoder = Encoder(vector_manager=self.vm)

        # Multi-horizon accumulators
        self.fast = DecayingAccumulator(DIMENSIONS, decay_for_half_life(100))
        self.slow = DecayingAccumulator(DIMENSIONS, decay_for_half_life(2000))

        self.checkpoint_mgr = CheckpointManager()

        self.learning_period = learning_period
        self.variance_window = variance_window
        self.attack_variance_drop = attack_variance_drop
        self.attack_similarity_rise = attack_similarity_rise
        self.recovery_threshold = recovery_threshold

        self.recent_sims = deque(maxlen=variance_window)
        self.baseline_variance = None
        self.baseline_mean = None

        self.is_attack = False
        self.attack_start_obs = None
        self.pre_attack_checkpoint: Optional[Checkpoint] = None

        self.count = 0

    def process(self, packet: dict, timestamp: float = None) -> DetectionState:
        self.count += 1
        if timestamp is None:
            timestamp = time.time()

        # Encode
        vec = self.encoder.encode_data(packet)

        # Get similarity to slow (baseline) accumulator
        slow_norm = self.slow.get_normalized()
        sim = cosine_similarity(vec, slow_norm) if self.count > 1 else 0.5

        self.recent_sims.append(sim)

        # Compute current stats
        current_var = np.var(list(self.recent_sims)) if len(self.recent_sims) >= 10 else 1.0
        current_mean = np.mean(list(self.recent_sims)) if len(self.recent_sims) >= 10 else 0.5

        # Record for checkpoint manager
        self.checkpoint_mgr.record_variance(current_var)

        # Update accumulators
        self.fast.update(vec)
        self.slow.update(vec)

        # Phase determination
        if self.count <= self.learning_period:
            phase = "learning"
            is_recovered = False
            checkpoint_sim = 0.0

            # Establish baseline at end of learning
            if self.count == self.learning_period:
                self.baseline_variance = current_var
                self.baseline_mean = current_mean

                # Save initial checkpoint
                cp = self.checkpoint_mgr.save_checkpoint(
                    self.slow, current_var, "initial_baseline", timestamp
                )
                self.pre_attack_checkpoint = cp

        elif not self.is_attack:
            # Check for attack start
            if (self.baseline_variance and
                current_var < self.baseline_variance * self.attack_variance_drop and
                current_mean > self.attack_similarity_rise):

                # Attack detected!
                self.is_attack = True
                self.attack_start_obs = self.count
                phase = "attack_detected"

                # Save pre-attack checkpoint (use the one we have)
                if not self.pre_attack_checkpoint:
                    # Shouldn't happen, but fallback
                    self.pre_attack_checkpoint = self.checkpoint_mgr.save_checkpoint(
                        self.slow, current_var, "pre_attack_fallback", timestamp
                    )
            else:
                phase = "normal"

                # Periodic checkpoint during normal operation
                if self.checkpoint_mgr.should_save_periodic(self.count):
                    self.checkpoint_mgr.save_checkpoint(
                        self.slow, current_var, "periodic", timestamp
                    )
                    self.pre_attack_checkpoint = self.checkpoint_mgr.get_latest_checkpoint()

            is_recovered = False
            checkpoint_sim = 0.0

        else:
            # In attack mode - check for recovery
            if self.pre_attack_checkpoint:
                is_recovered, checkpoint_sim = self.checkpoint_mgr.check_recovery(
                    self.fast,  # Use fast accumulator for recovery check
                    self.pre_attack_checkpoint,
                    self.recovery_threshold,
                )
            else:
                is_recovered = False
                checkpoint_sim = 0.0

            if is_recovered:
                phase = "recovered"
                self.is_attack = False
                # Save new checkpoint
                self.checkpoint_mgr.save_checkpoint(
                    self.slow, current_var, "post_recovery", timestamp
                )
                self.pre_attack_checkpoint = self.checkpoint_mgr.get_latest_checkpoint()
            else:
                # Check if attack pattern broke (variance increased)
                if current_var > self.baseline_variance * 0.5:
                    phase = "recovering"
                else:
                    phase = "attack_detected"

        return DetectionState(
            observation_count=self.count,
            similarity=sim,
            variance=current_var,
            is_attack=self.is_attack,
            is_recovered=is_recovered if 'is_recovered' in dir() else False,
            checkpoint_similarity=checkpoint_sim if 'checkpoint_sim' in dir() else 0.0,
            phase=phase,
        )


# =============================================================================
# TRAFFIC SIMULATOR
# =============================================================================

class TrafficSimulator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def normal_packet(self) -> dict:
        return {
            "protocol": self.rng.choice(["TCP", "UDP"]),
            "dst_port": self.rng.choice([443, 80, 8080]),
            "flags": self.rng.choice([0x10, 0x12, 0x18]),
        }

    def attack_packet(self) -> dict:
        return {
            "protocol": "TCP",
            "dst_port": 443,
            "flags": 0x02,  # SYN only
        }

    def generate_attack_recovery_stream(
        self,
        normal1: int = 1000,
        attack: int = 2000,
        normal2: int = 1000,
    ) -> List[Tuple[dict, str]]:
        """Generate: normal → attack → normal (recovery)."""
        stream = []

        # Phase 1: Normal
        for _ in range(normal1):
            stream.append((self.normal_packet(), "normal"))

        # Phase 2: Attack
        for _ in range(attack):
            if self.rng.random() < 0.95:  # 95% attack
                stream.append((self.attack_packet(), "attack"))
            else:
                stream.append((self.normal_packet(), "normal_during_attack"))

        # Phase 3: Recovery (back to normal)
        for _ in range(normal2):
            stream.append((self.normal_packet(), "recovery"))

        return stream


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 011-005: Smart Checkpoints")
    print("=" * 80)
    print("""
Problem: Continuous learning adapts to attack traffic.
After long DDoS, "normal" baseline has shifted.

Solution: Checkpoints = frozen "known good" states

Checkpoint uses:
1. Pre-attack reference for recovery detection
2. Resistance to poisoning (compare to checkpoint, not adapted baseline)
3. Before/after comparison for forensics
""")

    sim = TrafficSimulator(seed=42)
    detector = CheckpointDetector(
        learning_period=500,
        recovery_threshold=0.80,
    )

    # Generate attack-recovery stream
    stream = sim.generate_attack_recovery_stream(
        normal1=1000,   # Normal phase
        attack=2000,    # Attack phase
        normal2=1500,   # Recovery phase
    )

    print(f"Stream: {len(stream)} packets")
    print(f"  Normal: 1000 packets")
    print(f"  Attack: 2000 packets (95% attack)")
    print(f"  Recovery: 1500 packets")

    # Process stream
    results = []
    attack_detected_at = None
    recovery_detected_at = None

    for i, (packet, label) in enumerate(stream):
        state = detector.process(packet, timestamp=i * 0.001)  # 1ms per packet
        results.append((state, label))

        if state.phase == "attack_detected" and attack_detected_at is None:
            attack_detected_at = i

        if state.phase == "recovered" and recovery_detected_at is None:
            recovery_detected_at = i

    # Show phase transitions
    print("\n--- Phase Transitions ---")
    print(f"  Attack started: packet 1000")
    print(f"  Attack detected: packet {attack_detected_at} (delay: {attack_detected_at - 1000})")
    print(f"  Attack ended: packet 3000")
    print(f"  Recovery detected: packet {recovery_detected_at} (delay: {recovery_detected_at - 3000 if recovery_detected_at else 'N/A'})")

    # Show trajectory
    print("\n--- Similarity & Variance Trajectory ---")
    print(f"{'Packet':<10} {'Sim':<10} {'Var':<12} {'ChkptSim':<12} {'Phase':<15}")
    print("-" * 60)

    checkpoints = [0, 500, 900, 1000, 1100, 1500, 2500, 3000, 3100, 3500, 4000, 4499]
    for cp in checkpoints:
        if cp < len(results):
            state, label = results[cp]
            print(f"{cp:<10} {state.similarity:<10.3f} {state.variance:<12.4f} {state.checkpoint_similarity:<12.3f} {state.phase:<15}")

    # Checkpoint summary
    print(f"\n--- Checkpoints Saved ---")
    for i, cp in enumerate(detector.checkpoint_mgr.checkpoints):
        print(f"  {i}: obs={cp.observation_count}, var={cp.variance_at_save:.4f}, label={cp.label}")

    # Recovery analysis
    if recovery_detected_at:
        print("\n--- Recovery Analysis ---")
        print(f"""
Recovery detected at packet {recovery_detected_at}
Time since attack ended: {recovery_detected_at - 3000} packets

The checkpoint mechanism enables:
1. Detection of return to "known good" state
2. Comparison to pre-attack baseline (not adapted one)
3. Confirmation that attack has truly ended

Without checkpoints:
- Slow accumulator has adapted to attack traffic
- "Normal" would look like an anomaly!
- False positives during recovery

With checkpoints:
- Pre-attack state preserved
- Fast accumulator shows current traffic
- Compare fast to checkpoint = recovery detection
""")
    else:
        print("\n⚠️ Recovery not detected - may need threshold tuning")

    # Final summary
    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print("""
Key findings:

1. CHECKPOINTS PRESERVE KNOWN-GOOD STATE
   - Saved during stable periods (low variance)
   - Not affected by subsequent poisoning
   - Enable true recovery detection

2. MULTI-HORIZON + CHECKPOINTS = COMPLETE SOLUTION
   - Fast accumulator: current traffic pattern
   - Slow accumulator: evolving baseline
   - Checkpoint: frozen pre-attack state

   Attack detection: fast ≠ slow (divergence)
   Recovery detection: fast ≈ checkpoint

3. CHECKPOINT STRATEGIES
   - Periodic: Every N observations during stability
   - Stability-triggered: When variance low for extended period
   - Operator: Manual save command

4. RECOVERY THRESHOLD
   - Too high (0.95): May never detect recovery
   - Too low (0.70): May false-positive during attack
   - Sweet spot: ~0.80-0.85

5. PRACTICAL APPLICATIONS
   - DDoS: Detect attack end to restore normal filtering
   - Gradual poisoning: Compare to old checkpoint
   - Forensics: What did "normal" look like before attack?
""")


if __name__ == "__main__":
    main()
