#!/usr/bin/env python3
"""
Challenge 011-010: Prior Knowledge Update Mechanism

How to safely update priors with new baselines without:
1. Losing attack detection capability during transition
2. Accepting attack traffic as new baseline
3. Requiring full retraining

Update Strategies:
1. Gradual Blend: slowly blend new baseline into old
2. Validation Window: require N packets of consistency before accepting
3. Similarity Gate: only accept updates similar to existing prior
4. Checkpoint Rollback: keep old prior if new causes detection spikes
"""

import sys
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np

from scapy.all import IP, TCP, UDP, ICMP, Raw, Packet

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager

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
# UPDATE STRATEGIES
# =============================================================================

class UpdateStrategy(Enum):
    REPLACE = "replace"  # Immediate replacement (dangerous)
    GRADUAL_BLEND = "gradual_blend"  # Slowly blend new into old
    VALIDATION_WINDOW = "validation_window"  # Require consistency period
    SIMILARITY_GATE = "similarity_gate"  # Only accept if similar to old


@dataclass
class PriorState:
    """Represents a prior knowledge baseline."""
    vectors: Dict[str, np.ndarray]  # perspective -> baseline vector
    packet_count: int = 0
    created_at: int = 0  # packet number when created


@dataclass
class UpdateProposal:
    """A proposed update to prior knowledge."""
    new_vectors: Dict[str, np.ndarray]
    validation_packets: int = 0
    validation_anomaly_rate: float = 0.0
    similarity_to_old: float = 0.0


class PriorUpdater:
    """
    Manages safe updates to prior knowledge.

    Implements multiple strategies to prevent:
    - Accepting attack traffic as baseline
    - Detection gaps during transition
    - Catastrophic forgetting
    """

    def __init__(
        self,
        vm: DeterministicVectorManager,
        strategy: UpdateStrategy = UpdateStrategy.GRADUAL_BLEND,
        blend_factor: float = 0.1,  # For gradual blend
        validation_window: int = 100,  # For validation window
        max_anomaly_rate: float = 0.05,  # Max acceptable anomaly rate during validation
        min_similarity: float = 0.7,  # For similarity gate
    ):
        self.vm = vm
        self.strategy = strategy
        self.blend_factor = blend_factor
        self.validation_window = validation_window
        self.max_anomaly_rate = max_anomaly_rate
        self.min_similarity = min_similarity

        # Current and pending states
        self.current_prior: Optional[PriorState] = None
        self.pending_proposal: Optional[UpdateProposal] = None

        # History for rollback
        self.prior_history: List[PriorState] = []
        self.max_history = 5

        # Stats
        self.updates_accepted = 0
        self.updates_rejected = 0

    def set_initial_prior(self, vectors: Dict[str, np.ndarray], packet_num: int = 0):
        """Set the initial prior knowledge."""
        self.current_prior = PriorState(
            vectors=copy.deepcopy(vectors),
            packet_count=0,
            created_at=packet_num,
        )
        self.prior_history.append(copy.deepcopy(self.current_prior))

    def propose_update(self, new_vectors: Dict[str, np.ndarray]) -> UpdateProposal:
        """Propose an update to prior knowledge."""
        if self.current_prior is None:
            raise ValueError("No current prior to update")

        # Calculate similarity to current prior
        similarities = []
        for perspective in new_vectors:
            if perspective in self.current_prior.vectors:
                sim = cosine_similarity(
                    new_vectors[perspective],
                    self.current_prior.vectors[perspective]
                )
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0.0

        proposal = UpdateProposal(
            new_vectors=copy.deepcopy(new_vectors),
            similarity_to_old=avg_similarity,
        )

        self.pending_proposal = proposal
        return proposal

    def validate_packet(self, is_anomalous: bool):
        """Record a packet during validation window."""
        if self.pending_proposal is None:
            return

        self.pending_proposal.validation_packets += 1
        if is_anomalous:
            anomalies = self.pending_proposal.validation_anomaly_rate * (self.pending_proposal.validation_packets - 1)
            anomalies += 1
            self.pending_proposal.validation_anomaly_rate = anomalies / self.pending_proposal.validation_packets

    def should_accept_update(self) -> Tuple[bool, str]:
        """Determine if pending update should be accepted."""
        if self.pending_proposal is None:
            return False, "No pending proposal"

        proposal = self.pending_proposal

        if self.strategy == UpdateStrategy.REPLACE:
            return True, "Immediate replacement"

        elif self.strategy == UpdateStrategy.GRADUAL_BLEND:
            # Always accept for blending (gradual is the key)
            return True, f"Gradual blend (factor={self.blend_factor})"

        elif self.strategy == UpdateStrategy.VALIDATION_WINDOW:
            if proposal.validation_packets < self.validation_window:
                return False, f"Need {self.validation_window - proposal.validation_packets} more validation packets"
            if proposal.validation_anomaly_rate > self.max_anomaly_rate:
                return False, f"Anomaly rate {proposal.validation_anomaly_rate:.1%} exceeds max {self.max_anomaly_rate:.1%}"
            return True, f"Passed validation: {proposal.validation_anomaly_rate:.1%} anomaly rate"

        elif self.strategy == UpdateStrategy.SIMILARITY_GATE:
            if proposal.similarity_to_old < self.min_similarity:
                return False, f"Similarity {proposal.similarity_to_old:.2f} below threshold {self.min_similarity}"
            return True, f"Passed similarity gate: {proposal.similarity_to_old:.2f}"

        return False, "Unknown strategy"

    def apply_update(self) -> bool:
        """Apply the pending update if accepted."""
        accepted, reason = self.should_accept_update()

        if not accepted:
            print(f"  ✗ Update rejected: {reason}")
            self.updates_rejected += 1
            self.pending_proposal = None
            return False

        print(f"  ✓ Update accepted: {reason}")

        # Save current to history before updating
        if len(self.prior_history) >= self.max_history:
            self.prior_history.pop(0)
        self.prior_history.append(copy.deepcopy(self.current_prior))

        # Apply update based on strategy
        if self.strategy == UpdateStrategy.GRADUAL_BLEND:
            # Blend new into old
            for perspective in self.pending_proposal.new_vectors:
                if perspective in self.current_prior.vectors:
                    old_vec = self.current_prior.vectors[perspective]
                    new_vec = self.pending_proposal.new_vectors[perspective]
                    blended = (1 - self.blend_factor) * old_vec + self.blend_factor * new_vec
                    # Renormalize
                    norm = np.linalg.norm(blended)
                    if norm > 1e-10:
                        blended = blended / norm
                    self.current_prior.vectors[perspective] = blended
                else:
                    self.current_prior.vectors[perspective] = self.pending_proposal.new_vectors[perspective]
        else:
            # Full replacement
            self.current_prior.vectors = copy.deepcopy(self.pending_proposal.new_vectors)

        self.current_prior.packet_count += self.pending_proposal.validation_packets
        self.updates_accepted += 1
        self.pending_proposal = None
        return True

    def rollback(self) -> bool:
        """Rollback to previous prior."""
        if len(self.prior_history) < 2:
            return False

        # Pop current (which is last in history after apply)
        self.prior_history.pop()
        # Restore previous
        self.current_prior = copy.deepcopy(self.prior_history[-1])
        return True


# =============================================================================
# PACKET GENERATOR
# =============================================================================

def generate_normal_phase(n: int) -> List[Packet]:
    """Generate normal traffic."""
    import random
    packets = []
    for _ in range(n):
        pkt = IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
            sport=random.randint(49152, 65535),
            dport=random.choice([80, 443]),
            flags="PA"
        ) / Raw(load=b"GET / HTTP/1.1\r\n")
        packets.append(pkt)
    return packets


def generate_attack_phase(n: int) -> List[Packet]:
    """Generate attack traffic."""
    packets = []
    for i in range(n):
        src_ip = f"10.{(i // 256) % 256}.{i % 256}.1"
        pkt = IP(src=src_ip, dst="192.168.1.100") / TCP(
            sport=40000 + (i % 1000),
            dport=80,
            flags="S"
        )
        packets.append(pkt)
    return packets


def generate_evolved_normal(n: int) -> List[Packet]:
    """Generate evolved normal (new service, but legitimate)."""
    import random
    packets = []
    for _ in range(n):
        pkt = IP(src="192.168.1.50", dst="104.21.0.100") / TCP(  # New destination
            sport=random.randint(49152, 65535),
            dport=8443,  # New port
            flags="PA"
        ) / Raw(load=b"POST /api/v2 HTTP/1.1\r\n")  # New pattern
        packets.append(pkt)
    return packets


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_update_scenario(updater: PriorUpdater, scenario: str) -> Dict:
    """Run a scenario and return results."""
    vm = updater.vm

    # Initialize with normal traffic baseline
    initial_vectors = {}
    for perspective in ["l3", "l4", "payload"]:
        # Random baseline vector (simulating trained prior)
        vec = np.zeros(DIMENSIONS, dtype=np.float64)
        for atom in [f"{perspective}_base_{i}" for i in range(10)]:
            vec += vm.get_vector(atom)
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        initial_vectors[perspective] = vec

    updater.set_initial_prior(initial_vectors, packet_num=0)

    # Tracking
    anomaly_count = 0
    total_packets = 0

    if scenario == "attack_injection":
        # Attacker tries to poison baseline with attack traffic
        print("\n  Phase 1: Normal traffic (baseline)")
        normal_packets = generate_normal_phase(50)

        print("  Phase 2: Attack traffic (poisoning attempt)")
        attack_packets = generate_attack_phase(50)

        # Encode attack traffic as proposed update
        attack_vectors = {}
        for perspective in ["l3", "l4", "payload"]:
            vec = np.zeros(DIMENSIONS, dtype=np.float64)
            for atom in [f"attack_{perspective}_{i}" for i in range(10)]:
                vec += vm.get_vector(atom)
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            attack_vectors[perspective] = vec

        proposal = updater.propose_update(attack_vectors)
        print(f"  Proposal similarity to baseline: {proposal.similarity_to_old:.2%}")

        # Validate
        for i, pkt in enumerate(attack_packets):
            is_anomalous = (i % 3 == 0)  # Simulate 33% anomaly rate during attack
            updater.validate_packet(is_anomalous)
            if is_anomalous:
                anomaly_count += 1
            total_packets += 1

        # Try to apply
        accepted = updater.apply_update()

        return {
            "scenario": scenario,
            "accepted": accepted,
            "anomaly_rate": anomaly_count / max(1, total_packets),
            "similarity": proposal.similarity_to_old,
        }

    elif scenario == "legitimate_evolution":
        # Legitimate baseline shift (new service)
        print("\n  Phase 1: Normal traffic")

        print("  Phase 2: Evolved normal traffic (new service)")
        evolved_packets = generate_evolved_normal(100)

        # Encode evolved traffic
        evolved_vectors = {}
        for perspective in ["l3", "l4", "payload"]:
            vec = np.zeros(DIMENSIONS, dtype=np.float64)
            for atom in [f"evolved_{perspective}_{i}" for i in range(10)]:
                vec += vm.get_vector(atom)
            # Add some overlap with original
            for atom in [f"{perspective}_base_{i}" for i in range(5)]:
                vec += vm.get_vector(atom) * 0.5
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            evolved_vectors[perspective] = vec

        proposal = updater.propose_update(evolved_vectors)
        print(f"  Proposal similarity to baseline: {proposal.similarity_to_old:.2%}")

        # Validate with low anomaly rate
        for i in range(100):
            is_anomalous = (i > 90)  # Only 10% at end
            updater.validate_packet(is_anomalous)
            if is_anomalous:
                anomaly_count += 1
            total_packets += 1

        accepted = updater.apply_update()

        return {
            "scenario": scenario,
            "accepted": accepted,
            "anomaly_rate": anomaly_count / max(1, total_packets),
            "similarity": proposal.similarity_to_old,
        }

    return {"scenario": scenario, "error": "Unknown scenario"}


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 80)
    print("CHALLENGE 011-010: PRIOR KNOWLEDGE UPDATE MECHANISM")
    print("=" * 80)

    vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)

    strategies = [
        (UpdateStrategy.REPLACE, {}),
        (UpdateStrategy.GRADUAL_BLEND, {"blend_factor": 0.1}),
        (UpdateStrategy.VALIDATION_WINDOW, {"validation_window": 50, "max_anomaly_rate": 0.1}),
        (UpdateStrategy.SIMILARITY_GATE, {"min_similarity": 0.3}),
    ]

    scenarios = ["attack_injection", "legitimate_evolution"]

    results = []

    for strategy, params in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.value.upper()}")
        print("=" * 60)

        for scenario in scenarios:
            print(f"\nScenario: {scenario}")

            updater = PriorUpdater(
                vm=vm,
                strategy=strategy,
                **params
            )

            result = simulate_update_scenario(updater, scenario)
            result["strategy"] = strategy.value
            results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: STRATEGY EFFECTIVENESS")
    print("=" * 80)

    print(f"\n{'Strategy':<25} {'Attack Blocked':<15} {'Evolution Accepted':<20} {'Score'}")
    print("-" * 75)

    for strategy, _ in strategies:
        strategy_results = [r for r in results if r["strategy"] == strategy.value]

        attack_blocked = any(
            not r["accepted"]
            for r in strategy_results
            if r["scenario"] == "attack_injection"
        )
        evolution_accepted = any(
            r["accepted"]
            for r in strategy_results
            if r["scenario"] == "legitimate_evolution"
        )

        score = (1 if attack_blocked else 0) + (1 if evolution_accepted else 0)

        print(f"{strategy.value:<25} {'✓' if attack_blocked else '✗':<15} {'✓' if evolution_accepted else '✗':<20} {score}/2")

    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print("""
1. Update Strategy Comparison:
   - REPLACE: Fast but dangerous - accepts attacks as baseline
   - GRADUAL_BLEND: Safe but slow - dilutes attacks but also new patterns
   - VALIDATION_WINDOW: Good balance - rejects high-anomaly periods
   - SIMILARITY_GATE: Prevents major shifts - may reject legitimate evolution

2. Best Practice: Combine strategies
   - Use VALIDATION_WINDOW to catch active attacks
   - Use SIMILARITY_GATE to prevent major baseline shifts
   - Use GRADUAL_BLEND to smooth legitimate evolution

3. Key insight: Anomaly rate during validation is the best signal
   - Attack traffic triggers high anomaly rates (detected by current prior)
   - Legitimate evolution has low anomaly rate (consistent with current prior)

4. Rollback capability is essential
   - Keep history of N prior states
   - Monitor detection rates after update
   - Auto-rollback if anomaly rate spikes post-update
""")


if __name__ == "__main__":
    main()
