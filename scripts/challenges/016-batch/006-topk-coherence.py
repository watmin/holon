#!/usr/bin/env python3
"""
Top-K Coherence: Fixing the 50% Attack Fraction Problem

PROBLEM:
========
Mean coherence (001) needs 50%+ attack traffic to trigger. This is because
mean pairwise similarity is dominated by the O(n²) normal-normal pairs.
At 10% attack (6 of 60 packets), only 15 attack-attack pairs exist out of
1770 total pairs — the signal is drowned.

HYPOTHESIS:
===========
Look at the TAIL of the pairwise similarity distribution, not the mean.
A cluster of 5 identical attack packets produces ~10 pairs with similarity
near 1.0. Normal traffic pairs cluster near 0.0. The 95th or 99th percentile
of pairwise similarities should spike even at low attack fractions.

APPROACH:
=========
1. Compute full pairwise similarity matrix
2. Extract upper triangle (unique pairs)
3. Compare: mean, median, 90th/95th/99th percentile, max
4. Test across attack fractions 0% to 100%

VECTOR PROPERTIES EXPLOITED:
============================
- Tail of pairwise similarity distribution (not central tendency)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import coherence


def make_packet(client, src_ip, dst_ip, proto, src_port, dst_port, pkt_len, ttl):
    return client.encode(
        {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "proto": proto,
            "src_port": src_port,
            "dst_port": dst_port,
            "pkt_len": pkt_len,
            "ttl": ttl,
        }
    )


def generate_mixed_window(client, rng, attack_fraction, attack_type="dns_amp", count=60):
    """Generate a window with controllable attack fraction."""
    vecs = []
    labels = []
    protos = ["TCP", "UDP", "TCP", "TCP"]
    dst_ports = [80, 443, 22, 8080, 3306]

    for _ in range(count):
        if rng.random() < attack_fraction:
            if attack_type == "dns_amp":
                vecs.append(
                    make_packet(
                        client,
                        src_ip=rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"]),
                        dst_ip="192.168.1.100",
                        proto="UDP",
                        src_port=53,
                        dst_port=int(rng.integers(1024, 65535)),
                        pkt_len=int(rng.integers(512, 4096)),
                        ttl=int(rng.choice([240, 245, 250])),
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
                        ttl=int(rng.choice([64, 128, 255])),
                    )
                )
            labels.append("attack")
        else:
            vecs.append(
                make_packet(
                    client,
                    src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                    dst_ip=f"192.168.1.{rng.integers(1, 10)}",
                    proto=rng.choice(protos),
                    src_port=int(rng.integers(1024, 65535)),
                    dst_port=int(rng.choice(dst_ports)),
                    pkt_len=int(rng.integers(64, 1500)),
                    ttl=int(rng.choice([64, 128, 255])),
                )
            )
            labels.append("normal")
    return vecs, labels


def extract_pairwise_similarities(vecs):
    """Compute all unique pairwise similarities."""
    n = len(vecs)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(vecs[i], vecs[j]))
    return np.array(sims)


def compute_stats(sims):
    """Compute distribution statistics from pairwise similarities."""
    return {
        "mean": np.mean(sims),
        "median": np.median(sims),
        "p90": np.percentile(sims, 90),
        "p95": np.percentile(sims, 95),
        "p99": np.percentile(sims, 99),
        "max": np.max(sims),
        "std": np.std(sims),
        "top10_mean": np.mean(np.sort(sims)[-10:]),
    }


def main():
    client = HolonClient(dimensions=4096)

    print("=" * 80)
    print("EXPERIMENT 1: Pairwise similarity distribution — mean vs tail")
    print("=" * 80)
    print()
    print("For each attack fraction, compute statistics on ALL pairwise similarities.")
    print("Goal: find a statistic that triggers at LOW attack fractions.")

    header = f"{'Atk%':>5} {'Mean':>7} {'Med':>7} {'P90':>7} {'P95':>7} {'P99':>7} {'Max':>7} {'Top10':>7} {'Std':>7}"
    print(f"\n  DNS Amplification:")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for pct in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
        vecs, _ = generate_mixed_window(
            client, np.random.default_rng(int(pct * 10000) + 42), pct, "dns_amp", count=60
        )
        sims = extract_pairwise_similarities(vecs)
        s = compute_stats(sims)
        print(
            f"  {pct * 100:>4.0f}% {s['mean']:>7.4f} {s['median']:>7.4f} "
            f"{s['p90']:>7.4f} {s['p95']:>7.4f} {s['p99']:>7.4f} "
            f"{s['max']:>7.4f} {s['top10_mean']:>7.4f} {s['std']:>7.4f}"
        )

    print(f"\n  SYN Flood:")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for pct in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
        vecs, _ = generate_mixed_window(
            client, np.random.default_rng(int(pct * 10000) + 42), pct, "syn_flood", count=60
        )
        sims = extract_pairwise_similarities(vecs)
        s = compute_stats(sims)
        print(
            f"  {pct * 100:>4.0f}% {s['mean']:>7.4f} {s['median']:>7.4f} "
            f"{s['p90']:>7.4f} {s['p95']:>7.4f} {s['p99']:>7.4f} "
            f"{s['max']:>7.4f} {s['top10_mean']:>7.4f} {s['std']:>7.4f}"
        )

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Detection threshold comparison")
    print("=" * 80)
    print()
    print("Which statistic gives earliest detection with no false positives?")

    # Establish baselines from pure normal traffic (multiple trials)
    normal_stats = []
    for trial in range(20):
        vecs, _ = generate_mixed_window(
            client, np.random.default_rng(trial + 1000), 0.0, "dns_amp", count=60
        )
        sims = extract_pairwise_similarities(vecs)
        normal_stats.append(compute_stats(sims))

    # Compute thresholds: max observed normal + margin
    thresholds = {}
    for key in ["mean", "p90", "p95", "p99", "max", "top10_mean"]:
        vals = [s[key] for s in normal_stats]
        thresholds[key] = np.max(vals) * 1.1  # 10% margin above worst normal

    print(f"\n  Thresholds (max normal × 1.1):")
    for key, val in thresholds.items():
        print(f"    {key:>12}: {val:.4f}")

    print(f"\n  DNS Amp — earliest detection:")
    print(f"  {'Atk%':>5} {'Mean':>7} {'P90':>7} {'P95':>7} {'P99':>7} {'Max':>7} {'Top10':>7}")
    print(f"  {'-' * 48}")

    for pct in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        detected = {}
        vecs, _ = generate_mixed_window(
            client, np.random.default_rng(int(pct * 10000) + 42), pct, "dns_amp", count=60
        )
        sims = extract_pairwise_similarities(vecs)
        s = compute_stats(sims)

        markers = {}
        for key in ["mean", "p90", "p95", "p99", "max", "top10_mean"]:
            markers[key] = " ✓" if s[key] > thresholds[key] else "  "

        print(
            f"  {pct * 100:>4.0f}% "
            f"{s['mean']:>5.4f}{markers['mean']} "
            f"{s['p90']:>5.4f}{markers['p90']} "
            f"{s['p95']:>5.4f}{markers['p95']} "
            f"{s['p99']:>5.4f}{markers['p99']} "
            f"{s['max']:>5.4f}{markers['max']} "
            f"{s['top10_mean']:>5.4f}{markers['top10_mean']}"
        )

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Window size vs detection sensitivity")
    print("=" * 80)

    print(f"\n  Fixed 10% attack fraction, varying window size:")
    print(f"  {'WinSize':>8} {'Mean':>7} {'P95':>7} {'P99':>7} {'Top10':>7} {'#Pairs':>7}")
    print(f"  {'-' * 42}")

    for win_size in [20, 40, 60, 80, 100, 150, 200]:
        vecs, _ = generate_mixed_window(
            client, np.random.default_rng(42), 0.10, "dns_amp", count=win_size
        )
        sims = extract_pairwise_similarities(vecs)
        s = compute_stats(sims)
        n_pairs = len(sims)
        print(
            f"  {win_size:>8} {s['mean']:>7.4f} {s['p95']:>7.4f} "
            f"{s['p99']:>7.4f} {s['top10_mean']:>7.4f} {n_pairs:>7}"
        )

    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Stability — multiple trials at same attack fraction")
    print("=" * 80)

    print(f"\n  20 trials at 10% DNS amp attack, 60-packet windows:")
    print(f"  {'Trial':>6} {'Mean':>7} {'P95':>7} {'P99':>7} {'Top10':>7}")
    print(f"  {'-' * 34}")

    means, p95s, p99s, top10s = [], [], [], []
    for trial in range(20):
        vecs, _ = generate_mixed_window(
            client, np.random.default_rng(trial + 2000), 0.10, "dns_amp", count=60
        )
        sims = extract_pairwise_similarities(vecs)
        s = compute_stats(sims)
        means.append(s["mean"])
        p95s.append(s["p95"])
        p99s.append(s["p99"])
        top10s.append(s["top10_mean"])
        if trial < 10:
            print(f"  {trial:>6} {s['mean']:>7.4f} {s['p95']:>7.4f} {s['p99']:>7.4f} {s['top10_mean']:>7.4f}")

    print(f"  {'...':>6}")
    print(f"  {'μ':>6} {np.mean(means):>7.4f} {np.mean(p95s):>7.4f} {np.mean(p99s):>7.4f} {np.mean(top10s):>7.4f}")
    print(f"  {'σ':>6} {np.std(means):>7.4f} {np.std(p95s):>7.4f} {np.std(p99s):>7.4f} {np.std(top10s):>7.4f}")
    print(f"  {'cv':>6} {np.std(means)/np.mean(means):>7.2%} {np.std(p95s)/np.mean(p95s):>7.2%} {np.std(p99s)/np.mean(p99s):>7.2%} {np.std(top10s)/np.mean(top10s):>7.2%}")

    print()


if __name__ == "__main__":
    main()
