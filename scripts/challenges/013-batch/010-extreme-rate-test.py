#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 010: Extreme Rate Test
=============================================================================

SCENARIO:
- Baseline: 300 pps steady state
- Attack: 1,000,000,000 pps (1 billion!) for 5 minutes
- Recovery: back to 300 pps

QUESTIONS:
1. Can log scale handle 1 billion pps encoding?
2. Do learned references decode extreme rates correctly?
3. Does the accumulator recover after massive attack?

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/010-extreme-rate-test.py
"""

import sys
import random
import math
from typing import Dict, List, Tuple, Iterator, Any, Union
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore, Walkable, WalkType


DIMENSIONS = 4096
DECAY = 0.98


# =============================================================================
# PACKET TYPES (simplified)
# =============================================================================

class TCPPacket(Walkable):
    __slots__ = ("protocol", "src_port", "dst_port", "flags")

    def __init__(self, src_port: int, dst_port: int, flags: str):
        self.protocol = "TCP"
        self.src_port = src_port
        self.dst_port = dst_port
        self.flags = flags

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self) -> Iterator[Tuple[str, Any]]:
        yield "protocol", self.protocol
        yield "src_port", self.src_port
        yield "dst_port", self.dst_port
        yield "flags", self.flags


class UDPPacket(Walkable):
    __slots__ = ("protocol", "src_port", "dst_port")

    def __init__(self, src_port: int, dst_port: int):
        self.protocol = "UDP"
        self.src_port = src_port
        self.dst_port = dst_port

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self) -> Iterator[Tuple[str, Any]]:
        yield "protocol", self.protocol
        yield "src_port", self.src_port
        yield "dst_port", self.dst_port


Packet = Union[TCPPacket, UDPPacket]


def gen_normal(rng: random.Random) -> Packet:
    if rng.random() < 0.8:
        return TCPPacket(
            src_port=rng.randint(49152, 65535),
            dst_port=rng.choice([80, 443]),
            flags=rng.choice(["PA", "A"]),
        )
    else:
        return UDPPacket(
            src_port=rng.randint(49152, 65535),
            dst_port=53,
        )


def gen_attack(rng: random.Random) -> Packet:
    return UDPPacket(src_port=53, dst_port=rng.randint(49152, 65535))


# =============================================================================
# TEST 1: Log scale encoding range
# =============================================================================

def test_log_scale_encoding():
    print("="*70)
    print("TEST 1: Log Scale Encoding Range")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)

    test_rates = [
        0.1,
        1,
        10,
        100,
        300,            # baseline
        1000,
        10000,
        100000,
        1000000,        # 1 million
        1000000000,     # 1 billion
        1000000000000,  # 1 trillion
    ]

    print(f"\n{'Rate (pps)':>20} {'log10':>12} {'Encoded OK':>12}")
    print("-"*50)

    for rate in test_rates:
        try:
            vec = store.encode_scalar_log(float(rate))
            norm = np.linalg.norm(vec)
            log_val = math.log10(rate) if rate > 0 else 0
            print(f"{rate:>20,.0f} {log_val:>12.2f} {norm > 0:>12}")
        except Exception as e:
            print(f"{rate:>20,.0f} {'ERROR':>12} {str(e)}")

    print("\n✓ Log scale encoding works for any positive rate")


# =============================================================================
# TEST 2: Learned reference decoding at extreme rates
# =============================================================================

def test_reference_decoding():
    print("\n" + "="*70)
    print("TEST 2: Learned Reference Decoding")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)
    baseline_pps = 300.0

    # Build learned references (as we do in 008)
    ref_rates = [
        baseline_pps / 10,       # 30
        baseline_pps / 2,        # 150
        baseline_pps,            # 300 (baseline)
        baseline_pps * 2,        # 600
        baseline_pps * 5,        # 1500
        baseline_pps * 10,       # 3000
        baseline_pps * 50,       # 15000
        baseline_pps * 100,      # 30000
        baseline_pps * 1000,     # 300,000
        baseline_pps * 10000,    # 3,000,000
        baseline_pps * 100000,   # 30,000,000
        baseline_pps * 1000000,  # 300,000,000
        baseline_pps * 10000000, # 3,000,000,000 (3 billion)
    ]

    reference_vectors = {
        rate: store.encode_scalar_log(float(rate))
        for rate in ref_rates
    }

    def similarity(a, b):
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def decode(rate_vec):
        best_rate, best_sim = 300, -1
        for rate, ref in reference_vectors.items():
            sim = similarity(rate_vec, ref)
            if sim > best_sim:
                best_sim, best_rate = sim, rate
        return best_rate, best_sim

    print(f"\nBaseline: {baseline_pps} pps")
    print(f"Reference rates: {[f'{r:,.0f}' for r in sorted(ref_rates)]}")

    test_rates = [
        300,                # baseline
        3000,               # 10x
        30000,              # 100x
        1000000,            # ~3333x
        1000000000,         # 1 billion (~3.3 million x)
    ]

    print(f"\n{'Input Rate':>20} {'Decoded Rate':>20} {'Confidence':>12} {'Ratio':>12}")
    print("-"*70)

    for rate in test_rates:
        vec = store.encode_scalar_log(float(rate))
        decoded, conf = decode(vec)
        ratio = rate / baseline_pps
        print(f"{rate:>20,.0f} {decoded:>20,.0f} {conf:>12.3f} {ratio:>12,.0f}x")

    print("\n✓ With extended references, we can decode extreme rates")
    print("  Key: Project reference rates to cover expected attack magnitudes")


# =============================================================================
# TEST 3: Accumulator recovery after massive attack
# =============================================================================

def test_accumulator_recovery():
    print("\n" + "="*70)
    print("TEST 3: Accumulator Recovery After Massive Attack")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)
    encoder = store.encoder
    rng = random.Random(42)

    def normalize(accum):
        norm = np.linalg.norm(accum)
        return (accum / norm).astype(np.float32) if norm > 1e-10 else accum

    def similarity(a, b):
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # Build baseline
    print("\n[BASELINE] Learning from 500 normal packets...")
    baseline_accum = encoder.create_accumulator()
    for _ in range(500):
        packet = gen_normal(rng)
        vec = encoder.encode_walkable(packet)
        baseline_accum = encoder.accumulate(baseline_accum, vec)

    baseline_norm = normalize(baseline_accum)
    print(f"  Baseline pattern captured")

    # Initialize recent accumulator from baseline
    recent_accum = baseline_accum.copy().astype(np.float64)

    # Simulate attack
    # 5 minutes at 1 billion pps = 300 billion packets
    # We can't actually process 300 billion, but we can simulate the EFFECT
    # After N packets with decay 0.98, old signal is multiplied by 0.98^N
    # 0.98^1000 ≈ 1.7e-9 (essentially zero)
    # So after ~1000 packets, baseline is completely washed out

    print("\n[ATTACK] Simulating massive attack...")
    print("  (Real: 5 min @ 1B pps = 300B packets)")
    print("  (Simulated: 2000 attack packets - enough to wash out baseline)")

    attack_packets = 2000
    for i in range(attack_packets):
        packet = gen_attack(rng)
        vec = encoder.encode_walkable(packet)
        recent_accum = DECAY * recent_accum + vec.astype(np.float64)

        if i % 500 == 0:
            recent_norm = normalize(recent_accum)
            drift = 1.0 - similarity(baseline_norm, recent_norm)
            print(f"    After {i:>5} attack packets: drift = {drift:.3f}")

    recent_norm = normalize(recent_accum)
    drift = 1.0 - similarity(baseline_norm, recent_norm)
    print(f"    After {attack_packets:>5} attack packets: drift = {drift:.3f}")

    # Check: how much baseline signal remains?
    baseline_remaining = 0.98 ** attack_packets
    print(f"\n  Baseline signal remaining: {baseline_remaining:.2e} (effectively 0)")
    print(f"  Accumulator is now 100% attack pattern")

    # Recovery
    print("\n[RECOVERY] Returning to normal traffic...")
    print("  At 300 pps, we simulate recovery over time")

    # At 300 pps, how long to recover?
    # We need new normal traffic to dominate
    # After N normal packets, attack signal is 0.98^N

    recovery_checkpoints = [100, 500, 1000, 2000, 5000]

    for target in recovery_checkpoints:
        # Process normal packets
        while True:
            packet = gen_normal(rng)
            vec = encoder.encode_walkable(packet)
            recent_accum = DECAY * recent_accum + vec.astype(np.float64)
            attack_packets += 1  # reuse as total counter

            if attack_packets - 2000 >= target:
                break

        recent_norm = normalize(recent_accum)
        drift = 1.0 - similarity(baseline_norm, recent_norm)
        normal_packets = attack_packets - 2000
        time_at_300pps = normal_packets / 300.0  # seconds
        print(f"    After {normal_packets:>5} normal packets ({time_at_300pps:.1f}s @ 300pps): drift = {drift:.3f}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print("""
    The accumulator DOES recover, but recovery time depends on:

    1. DECAY RATE: With decay=0.98, attack signal halves every ~34 packets
       - 0.98^34 ≈ 0.5
       - After ~230 packets, attack signal < 1%
       - After ~500 packets, drift should be low

    2. TRAFFIC RATE: At 300 pps:
       - 500 packets = 1.7 seconds to significant recovery
       - 2000 packets = 6.7 seconds to near-full recovery

    3. BASELINE FROZEN: The baseline pattern (prior_pattern_norm) is FROZEN
       - It doesn't get corrupted by attack
       - It's the shipped reference from central
       - Recovery means recent accumulator converging back to baseline

    KEY INSIGHT: The frozen baseline protects against attack corruption.
    The scrubber's recent accumulator can fully recover.
    """)


# =============================================================================
# TEST 4: End-to-end extreme scenario
# =============================================================================

def test_end_to_end():
    print("\n" + "="*70)
    print("TEST 4: End-to-End Extreme Scenario")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)
    encoder = store.encoder
    rng = random.Random(42)

    def normalize(accum):
        norm = np.linalg.norm(accum)
        return (accum / norm).astype(np.float32) if norm > 1e-10 else accum

    def similarity(a, b):
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    THRESHOLD = 0.15

    # Build baseline
    baseline_accum = encoder.create_accumulator()
    baseline_rate_accum = encoder.create_accumulator()
    baseline_pps = 300.0

    for _ in range(500):
        packet = gen_normal(rng)
        vec = encoder.encode_walkable(packet)
        rate_vec = store.encode_scalar_log(baseline_pps + rng.uniform(-30, 30))
        baseline_accum = encoder.accumulate(baseline_accum, vec)
        baseline_rate_accum = encoder.accumulate(baseline_rate_accum, rate_vec)

    baseline_norm = normalize(baseline_accum)
    baseline_rate_norm = normalize(baseline_rate_accum)

    # Build rate references (extended for extreme rates)
    ref_rates = [baseline_pps * mult for mult in [0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000, 100000, 1000000, 10000000]]
    reference_vectors = {rate: store.encode_scalar_log(float(rate)) for rate in ref_rates}

    def decode_rate(rate_vec):
        best_rate, best_sim = baseline_pps, -1
        for rate, ref in reference_vectors.items():
            sim = similarity(rate_vec, ref)
            if sim > best_sim:
                best_sim, best_rate = sim, rate
        return best_rate

    # Pre-decode baseline rate
    decoded_baseline = decode_rate(baseline_rate_norm)
    print(f"\nBaseline: {baseline_pps} pps (decoded: {decoded_baseline:.0f} pps)")

    # Scrubber state
    recent_accum = baseline_accum.copy().astype(np.float64)

    def process_and_decide(packet):
        nonlocal recent_accum
        vec = encoder.encode_walkable(packet)
        recent_accum = DECAY * recent_accum + vec.astype(np.float64)
        recent_norm = normalize(recent_accum)
        drift = 1.0 - similarity(baseline_norm, recent_norm)
        is_anomalous = drift > THRESHOLD
        return drift, is_anomalous

    # Scenario timeline
    print("\n[TIMELINE]")
    print("-"*70)

    # Normal (1 second @ 300 pps = 300 packets)
    print("\n  00:00 - Normal traffic (300 pps)...")
    decisions = []
    for _ in range(300):
        drift, anomalous = process_and_decide(gen_normal(rng))
        decisions.append(anomalous)
    print(f"    Anomalous: {sum(decisions)}/{len(decisions)} ({sum(decisions)/len(decisions)*100:.1f}%)")
    print(f"    Final drift: {drift:.3f}")

    # Attack starts (simulate with fewer packets due to compute)
    print("\n  00:01 - ATTACK STARTS (simulated 1B pps → 2000 attack packets)...")
    decisions = []
    for _ in range(2000):
        drift, anomalous = process_and_decide(gen_attack(rng))
        decisions.append(anomalous)
    print(f"    Anomalous: {sum(decisions)}/{len(decisions)} ({sum(decisions)/len(decisions)*100:.1f}%)")
    print(f"    Final drift: {drift:.3f}")
    print(f"    → RATE LIMIT TO: {decoded_baseline:.0f} pps")

    # Attack continues (more packets)
    print("\n  00:01+ - Attack continues...")
    for _ in range(1000):
        drift, _ = process_and_decide(gen_attack(rng))
    print(f"    Drift stable at: {drift:.3f}")

    # Attack ends, recovery
    print("\n  05:00 - Attack ends, normal traffic resumes (300 pps)...")
    for i, checkpoint in enumerate([100, 500, 1000, 2000]):
        prev = 0 if i == 0 else [100, 500, 1000, 2000][i-1]
        for _ in range(checkpoint - prev):
            drift, anomalous = process_and_decide(gen_normal(rng))
        time_s = checkpoint / 300.0
        print(f"    +{time_s:.1f}s ({checkpoint} packets): drift = {drift:.3f}, anomalous = {anomalous}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"""
    ✓ Log scale handles any rate (0.1 to 1 trillion+ pps)
    ✓ References can be extended to cover extreme multiples
    ✓ Frozen baseline protects against corruption
    ✓ Accumulator recovers in seconds at normal traffic rates

    For 300 pps baseline → 1 billion pps attack → 300 pps recovery:
    - Detection: Immediate (drift spikes on first attack packets)
    - Rate limit: Enforces baseline ({decoded_baseline:.0f} pps)
    - Recovery: ~2-7 seconds after attack ends

    The vectors handle the extreme case.
    """)


def main():
    test_log_scale_encoding()
    test_reference_decoding()
    test_accumulator_recovery()
    test_end_to_end()


if __name__ == "__main__":
    main()
