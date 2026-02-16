#!/usr/bin/env python3
"""
Dimension Scaling: How Many Payload Bytes Can One Vector Handle?

QUESTION:
=========
VSA theory says ~√d items can be superposed in one vector.
At d=4096, that's ~64. But the accumulator stores float sums,
not bipolar — does this extend the effective capacity?

EXPERIMENT:
===========
For each (dimensions, payload_size) combination:
  1. Train on 200 normal payloads (6 familiar values per position)
  2. Attack: first 4 and last 4 bytes are unfamiliar, middle is familiar
  3. Measure drill-down accuracy: can we correctly classify each position?

Optimizations:
  - Fewer training packets (200 vs 500) — capacity question, not precision
  - Skip drill-down on ALL positions — sample 20 familiar + 8 unfamiliar
  - Flush output after each row for incremental progress
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity


def run_trial(dimensions, num_bytes, seed=42):
    """Run one trial with given dimensions and payload size."""
    client = HolonClient(dimensions=dimensions)
    rng = np.random.default_rng(seed)

    familiar_pool = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30]
    unfamiliar_pool = [0xFF, 0xAC, 0xFB, 0xCA, 0xDE, 0xAD, 0xBE, 0xEF]

    # Anomalous positions: first 4 and last 4
    anomalous_set = set(list(range(4)) + list(range(num_bytes - 4, num_bytes)))

    def make_normal():
        return [int(rng.choice(familiar_pool)) for _ in range(num_bytes)]

    def make_attack():
        return [
            int(rng.choice(unfamiliar_pool if i in anomalous_set else familiar_pool))
            for i in range(num_bytes)
        ]

    def to_dict(payload):
        return {f"p{i}": f"0x{b:02x}" for i, b in enumerate(payload)}

    # Learn baseline
    accum = client.create_accumulator()
    for _ in range(200):
        vec = client.encode(to_dict(make_normal()))
        accum = client.accumulate(accum, vec)
    baseline = client.normalize_accumulator(accum)

    # Overall similarity
    legit_sims = [
        cosine_similarity(client.encode(to_dict(make_normal())), baseline)
        for _ in range(20)
    ]
    attack_sims = [
        cosine_similarity(client.encode(to_dict(make_attack())), baseline)
        for _ in range(20)
    ]

    legit_mean = np.mean(legit_sims)
    attack_mean = np.mean(attack_sims)

    # Drill-down: sample positions instead of checking all
    attack_payload = make_attack()

    # Sample up to 20 familiar + all 8 unfamiliar positions
    familiar_positions = sorted(set(range(num_bytes)) - anomalous_set)
    sample_familiar = list(rng.choice(
        familiar_positions,
        size=min(20, len(familiar_positions)),
        replace=False,
    ))
    sample_unfamiliar = sorted(anomalous_set)

    def score_position(pos):
        field = f"p{pos}"
        value = f"0x{attack_payload[pos]:02x}"
        role_vec = client.get_vector(field)
        val_vec = client.get_vector(value)
        bound = role_vec * val_vec
        return cosine_similarity(bound, baseline)

    fam_sims = [score_position(p) for p in sample_familiar]
    unfam_sims = [score_position(p) for p in sample_unfamiliar]

    fam_mean = np.mean(fam_sims)
    unfam_mean = np.mean(unfam_sims)
    drill_sep = fam_mean - unfam_mean

    # Classification accuracy on sampled positions
    threshold = (fam_mean + unfam_mean) / 2
    correct = 0
    total = len(sample_familiar) + len(sample_unfamiliar)
    for p in sample_familiar:
        if score_position(p) > threshold:
            correct += 1
    for p in sample_unfamiliar:
        if score_position(p) <= threshold:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "dim": dimensions,
        "nb": num_bytes,
        "sqrt_d": int(dimensions ** 0.5),
        "ratio": num_bytes / (dimensions ** 0.5),
        "legit": legit_mean,
        "attack": attack_mean,
        "sep": legit_mean / attack_mean if attack_mean > 0.001 else 999,
        "fam": fam_mean,
        "unfam": unfam_mean,
        "gap": drill_sep,
        "acc": accuracy,
    }


def main():
    print("=" * 95)
    print("DIMENSION SCALING: How Many Bytes Can One Vector Handle?")
    print("=" * 95)
    print()
    print("  Attack: first 4 + last 4 bytes unfamiliar, middle familiar")
    print("  Training: 200 packets, 6 familiar byte values per position")
    print("  Drill-down: sampled 20 familiar + 8 unfamiliar positions")
    print()

    dims = [4096, 8192, 16384, 32768]
    byte_counts = [10, 32, 64, 128, 256, 512, 1024, 1500]

    hdr = (f"  {'dim':>6} {'√d':>4} {'bytes':>6} {'b/√d':>5} "
           f"{'legit':>7} {'attack':>7} {'sep':>5} "
           f"{'fam_s':>7} {'unf_s':>7} {'gap':>7} {'acc%':>6} {'verdict':>8}")
    print(hdr)
    print(f"  {'-' * 90}")

    for dim in dims:
        for nb in byte_counts:
            r = run_trial(dim, nb)

            if r["acc"] >= 0.95:
                verdict = "WORKS"
            elif r["acc"] >= 0.80:
                verdict = "WEAK"
            elif r["acc"] >= 0.60:
                verdict = "POOR"
            else:
                verdict = "FAILS"

            print(
                f"  {r['dim']:>6} {r['sqrt_d']:>4} {r['nb']:>6} {r['ratio']:>5.1f} "
                f"{r['legit']:>7.4f} {r['attack']:>7.4f} {r['sep']:>4.1f}x "
                f"{r['fam']:>7.4f} {r['unfam']:>7.4f} {r['gap']:>7.4f} "
                f"{r['acc']*100:>5.1f}% {verdict:>8}",
                flush=True,
            )

        print(flush=True)

    # Summary
    print("=" * 95)
    print("KEY FINDINGS")
    print("=" * 95)
    print()
    print("  Theory predicts √d capacity. Actual results:")
    print()
    print(f"  {'Dimensions':>12} {'√d':>6} {'Theory Max':>12} {'Actual Max (>90% acc)':>25}")
    print(f"  {'-' * 60}")

    for dim in dims:
        sqrt_d = int(dim ** 0.5)
        # Find max bytes where accuracy >= 90%
        best = 0
        for nb in byte_counts:
            r = run_trial(dim, nb)
            if r["acc"] >= 0.90:
                best = nb
        print(f"  {dim:>12} {sqrt_d:>6} {sqrt_d:>12} {best if best > 0 else 'NONE':>25}")

    print()


if __name__ == "__main__":
    main()
