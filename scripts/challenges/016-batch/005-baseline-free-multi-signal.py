#!/usr/bin/env python3
"""
Baseline-Free Multi-Signal Detection

HYPOTHESIS:
===========
Three independent baseline-free measures should detect attacks during
cold start (no baseline available):
  - coherence():   pairwise window homogeneity
  - complexity():  information-theoretic entropy of dimension distribution
  - purity():      quantum-inspired accumulator concentration

Each captures a different mathematical property of the same phenomenon
(attack traffic is homogeneous). Combined, they should be robust.

PRIMITIVES DEMONSTRATED:
========================
1. coherence()     - Window homogeneity (pairwise similarity distribution)
2. complexity()    - Pattern mixedness (entropy measure)
3. purity()        - Accumulator concentration (quantum-inspired)
4. significance()  - Principled thresholds for all measures

SCENARIO:
=========
Cold start: no baseline exists. Can we detect attacks from the first window?
Also: boiling frog — baseline corruption where similarity-to-baseline fails.

VECTOR PROPERTIES EXPLOITED:
============================
- Pairwise similarity distribution (coherence)
- Dimension entropy (complexity)
- Accumulator spectral concentration (purity)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.accumulator import accumulate, create_accumulator, normalize_accumulator, purity
from holon.distance import significance
from holon.primitives import bundle, coherence, complexity


def make_packet(client, src_ip, dst_ip, proto, src_port, dst_port, pkt_len):
    return client.encode(
        {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "proto": proto,
            "src_port": src_port,
            "dst_port": dst_port,
            "pkt_len": pkt_len,
        }
    )


def generate_window(client, rng, attack_fraction=0.0, attack_type="dns_amp", count=50):
    """Generate a window of traffic with controllable attack fraction."""
    vecs = []
    for _ in range(count):
        if rng.random() < attack_fraction:
            if attack_type == "dns_amp":
                vecs.append(
                    make_packet(
                        client,
                        src_ip=rng.choice(["8.8.8.8", "1.1.1.1"]),
                        dst_ip="192.168.1.100",
                        proto="UDP",
                        src_port=53,
                        dst_port=int(rng.integers(1024, 65535)),
                        pkt_len=1400,
                    )
                )
            elif attack_type == "syn_flood":
                vecs.append(
                    make_packet(
                        client,
                        src_ip=f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
                        dst_ip="192.168.1.100",
                        proto="TCP",
                        src_port=int(rng.integers(1024, 65535)),
                        dst_port=80,
                        pkt_len=60,
                    )
                )
        else:
            vecs.append(
                make_packet(
                    client,
                    src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                    dst_ip=f"192.168.1.{rng.integers(1, 10)}",
                    proto=rng.choice(["TCP", "UDP"]),
                    src_port=int(rng.integers(1024, 65535)),
                    dst_port=int(rng.choice([80, 443, 22, 8080])),
                    pkt_len=int(rng.integers(64, 1500)),
                )
            )
    return vecs


def compute_signals(vecs):
    """Compute all three baseline-free signals."""
    c = coherence(vecs)
    bundled = bundle(vecs)
    x = complexity(bundled)

    acc = create_accumulator(len(vecs[0]))
    for v in vecs:
        acc = accumulate(acc, v)
    p = purity(acc)

    return c, x, p


def main():
    client = HolonClient(dimensions=4096)
    d = 4096

    print("=" * 70)
    print("EXPERIMENT 1: Cold start detection — no baseline")
    print("=" * 70)

    scenarios = [
        ("Normal traffic", 0.0, "dns_amp"),
        ("5% DNS amp", 0.05, "dns_amp"),
        ("10% DNS amp", 0.10, "dns_amp"),
        ("20% DNS amp", 0.20, "dns_amp"),
        ("50% DNS amp", 0.50, "dns_amp"),
        ("100% DNS amp", 1.00, "dns_amp"),
        ("50% SYN flood", 0.50, "syn_flood"),
        ("100% SYN flood", 1.00, "syn_flood"),
    ]

    print(f"\n{'Scenario':<20} {'Coherence':>10} {'Complexity':>11} {'Purity':>8} {'Combined':>10} {'Verdict':>10}")
    print("-" * 75)

    for label, frac, atype in scenarios:
        vecs = generate_window(client, np.random.default_rng(42), frac, atype, count=60)
        coh, comp, pur = compute_signals(vecs)

        # Combined score: high coherence, low complexity, high purity → attack
        combined = coh * (1.0 - comp) * pur
        # Normalize to be more interpretable (scale by expected ranges)
        is_attack = combined > 0.01 or coh > 0.15 or pur > 0.5

        verdict = "ATTACK" if is_attack and frac > 0.0 else "NORMAL" if not is_attack else "FALSE+"
        print(f"{label:<20} {coh:>10.4f} {comp:>11.4f} {pur:>8.4f} {combined:>10.6f} {verdict:>10}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Each signal independently — which triggers first?")
    print("=" * 70)

    thresholds = {"coherence": 0.10, "complexity": 0.80, "purity": 0.20}

    print(f"\n{'Attack %':>10} {'Coh':>8} {'Coh Det':>8} {'Comp':>8} {'Comp Det':>9} {'Pur':>8} {'Pur Det':>8}")
    print("-" * 65)

    for pct in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]:
        vecs = generate_window(
            client, np.random.default_rng(int(pct * 10000) + 1), pct, "dns_amp", count=80
        )
        coh, comp, pur = compute_signals(vecs)

        coh_det = "YES" if coh > thresholds["coherence"] else "no"
        comp_det = "YES" if comp < thresholds["complexity"] else "no"
        pur_det = "YES" if pur > thresholds["purity"] else "no"

        print(f"{pct * 100:>9.0f}% {coh:>8.4f} {coh_det:>8} {comp:>8.4f} {comp_det:>9} {pur:>8.4f} {pur_det:>8}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Boiling frog — baseline corruption attack")
    print("=" * 70)

    print("\n  Attacker slowly increases attack traffic over 20 windows.")
    print("  Baseline-similarity detection adapts (misses the slow ramp).")
    print("  Baseline-free signals should still detect.\n")

    # Build initial baseline from normal traffic
    baseline_vecs = generate_window(client, np.random.default_rng(42), 0.0, "dns_amp", count=200)
    baseline_vec = bundle(baseline_vecs)

    # Simulate adaptive baseline that slowly corrupts
    adaptive_baseline = baseline_vec.copy()

    print(f"{'Window':>8} {'Atk %':>7} {'Sim→Base':>10} {'Sim Det':>8} {'Coherence':>10} {'Coh Det':>8}")
    print("-" * 55)

    for window_idx in range(20):
        attack_frac = min(1.0, window_idx * 0.05)  # 0%, 5%, 10%, ..., 95%
        vecs = generate_window(
            client, np.random.default_rng(window_idx + 100), attack_frac, "dns_amp", count=60
        )

        current = bundle(vecs)

        # Similarity to (possibly corrupted) baseline
        sim = cosine_similarity(current, adaptive_baseline)
        sim_detected = sim < 0.5

        # Baseline-free: coherence of current window
        coh = coherence(vecs)
        coh_detected = coh > 0.10

        # Slowly corrupt the baseline (simulate adaptive system)
        from holon.primitives import blend

        adaptive_baseline = blend(adaptive_baseline, current, 0.1)

        print(
            f"{window_idx:>8} {attack_frac * 100:>6.0f}% {sim:>10.4f} "
            f"{'YES' if sim_detected else 'no':>8} {coh:>10.4f} {'YES' if coh_detected else 'no':>8}"
        )

    print("\n  Key insight: similarity-to-baseline misses the slow ramp")
    print("  because the baseline itself adapts. Coherence doesn't care")
    print("  about baseline — it measures the window's internal structure.")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Signal correlation — are they truly independent?")
    print("=" * 70)

    cohs, comps, purs = [], [], []
    for trial in range(50):
        pct = trial / 49.0
        vecs = generate_window(
            client, np.random.default_rng(trial + 200), pct, "dns_amp", count=60
        )
        c, x, p = compute_signals(vecs)
        cohs.append(c)
        comps.append(x)
        purs.append(p)

    # Compute correlations
    from numpy import corrcoef

    r_coh_comp = corrcoef(cohs, comps)[0, 1]
    r_coh_pur = corrcoef(cohs, purs)[0, 1]
    r_comp_pur = corrcoef(comps, purs)[0, 1]

    print(f"\n  Correlation matrix (across 50 attack levels):")
    print(f"    coherence ↔ complexity: {r_coh_comp:+.4f}")
    print(f"    coherence ↔ purity:     {r_coh_pur:+.4f}")
    print(f"    complexity ↔ purity:    {r_comp_pur:+.4f}")
    print(f"\n  High correlation = redundant signals, low = independent")

    print()


if __name__ == "__main__":
    main()
