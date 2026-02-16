#!/usr/bin/env python3
"""
Bundle with Confidence for Trust-Aware Drift Detection

HYPOTHESIS:
===========
bundle_with_confidence() returns per-dimension agreement margins. Dimensions
with high consensus in the baseline are more trustworthy for drift detection.
Using margins as weights in weighted_cosine_similarity should produce a
BETTER drift signal than unweighted cosine.

PRIMITIVES DEMONSTRATED:
========================
1. bundle_with_confidence() - Bundle + per-dimension margins
2. weighted_cosine (via DistanceEngine) - Dimension-weighted similarity
3. cosine_similarity() - Standard unweighted (for comparison)
4. significance() - Z-score for both metrics

VECTOR PROPERTIES EXPLOITED:
============================
- Per-dimension agreement strength (discarded by regular bundle)
- Trust-weighted comparison (Fisher information analog)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.distance import DistanceEngine, DistanceMetric, significance
from holon.primitives import bundle, bundle_with_confidence


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


def generate_baseline(client, rng, count=200):
    """Normal traffic with strong patterns on some fields, noise on others."""
    vecs = []
    for _ in range(count):
        vecs.append(
            make_packet(
                client,
                src_ip=f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
                dst_ip="192.168.1.1",  # ALWAYS same dst_ip (high confidence)
                proto="TCP",  # ALWAYS TCP (high confidence)
                src_port=int(rng.integers(1024, 65535)),  # Random (low confidence)
                dst_port=443,  # ALWAYS 443 (high confidence)
                pkt_len=int(rng.integers(64, 1500)),  # Random (low confidence)
            )
        )
    return vecs


def generate_subtle_attack(client, rng, count=50):
    """Attack that only changes LOW-confidence fields (src_port, pkt_len).
    Standard cosine should catch this weakly. Weighted cosine should NOT
    flag this since only noisy dimensions changed."""
    vecs = []
    for _ in range(count):
        vecs.append(
            make_packet(
                client,
                src_ip=f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
                dst_ip="192.168.1.1",  # Same as baseline
                proto="TCP",  # Same as baseline
                src_port=53,  # Changed but low-confidence field
                dst_port=443,  # Same as baseline
                pkt_len=4096,  # Changed but low-confidence field
            )
        )
    return vecs


def generate_real_attack(client, rng, count=50):
    """Attack that changes HIGH-confidence fields (dst_ip, proto, dst_port).
    Both metrics should catch this, but weighted should be MORE sensitive."""
    vecs = []
    for _ in range(count):
        vecs.append(
            make_packet(
                client,
                src_ip=f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
                dst_ip="10.10.10.10",  # CHANGED (was 192.168.1.1)
                proto="UDP",  # CHANGED (was TCP)
                src_port=int(rng.integers(1024, 65535)),
                dst_port=53,  # CHANGED (was 443)
                pkt_len=int(rng.integers(64, 1500)),
            )
        )
    return vecs


def generate_mixed_attack(client, rng, count=50, attack_frac=0.1):
    """Mostly normal with a small fraction of attack packets."""
    vecs = []
    for _ in range(count):
        if rng.random() < attack_frac:
            vecs.append(
                make_packet(
                    client,
                    src_ip="45.33.32.156",
                    dst_ip="10.10.10.10",
                    proto="UDP",
                    src_port=53,
                    dst_port=int(rng.integers(1024, 65535)),
                    pkt_len=1400,
                )
            )
        else:
            vecs.append(
                make_packet(
                    client,
                    src_ip=f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
                    dst_ip="192.168.1.1",
                    proto="TCP",
                    src_port=int(rng.integers(1024, 65535)),
                    dst_port=443,
                    pkt_len=int(rng.integers(64, 1500)),
                )
            )
    return vecs


def main():
    client = HolonClient(dimensions=4096)
    d = 4096
    engine = DistanceEngine()

    print("=" * 70)
    print("EXPERIMENT 1: Confidence margins from baseline")
    print("=" * 70)

    baseline_vecs = generate_baseline(client, np.random.default_rng(42))
    baseline_bundled, margins = bundle_with_confidence(baseline_vecs)

    # Analyze margin distribution
    high_conf = np.sum(margins > 0.8)
    med_conf = np.sum((margins > 0.3) & (margins <= 0.8))
    low_conf = np.sum(margins <= 0.3)

    print(f"\n  Baseline built from {len(baseline_vecs)} packets")
    print(f"  Margin distribution:")
    print(f"    High confidence (>0.8): {high_conf:>5} dims ({high_conf / d * 100:.1f}%)")
    print(f"    Medium (0.3-0.8):       {med_conf:>5} dims ({med_conf / d * 100:.1f}%)")
    print(f"    Low confidence (≤0.3):  {low_conf:>5} dims ({low_conf / d * 100:.1f}%)")
    print(f"  Mean margin: {np.mean(margins):.4f}")
    print(f"  Median margin: {np.median(margins):.4f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Weighted vs unweighted drift detection")
    print("=" * 70)

    scenarios = [
        ("Normal traffic", generate_baseline, 77),
        ("Subtle attack (noisy dims)", generate_subtle_attack, 77),
        ("Real attack (stable dims)", generate_real_attack, 77),
        ("Mixed 10% attack", lambda c, r, count=50: generate_mixed_attack(c, r, count, 0.10), 77),
        ("Mixed 20% attack", lambda c, r, count=50: generate_mixed_attack(c, r, count, 0.20), 77),
    ]

    weights = margins  # numpy array, used as per-dimension weights

    print(f"\n{'Scenario':<30} {'Cosine':>8} {'W.Cosine':>8} {'C Z':>6} {'W Z':>6} {'Weighted Better?':>18}")
    print("-" * 80)

    for label, gen_fn, seed in scenarios:
        test_vecs = gen_fn(client, np.random.default_rng(seed))
        test_bundled = bundle(test_vecs)

        sim_cos = cosine_similarity(test_bundled, baseline_bundled)
        sim_wcos = engine.similarity(
            test_bundled, baseline_bundled, DistanceMetric.WEIGHTED_COSINE, weights=weights
        )

        z_cos = significance(sim_cos, d)
        z_wcos = significance(sim_wcos, d)

        # "Better" means: higher separation for attacks, lower false alarm for normal
        is_attack = "attack" in label.lower() or "mixed" in label.lower()
        if is_attack:
            better = "YES ✓" if abs(1.0 - sim_wcos) > abs(1.0 - sim_cos) else "no"
        else:
            better = "YES ✓" if abs(1.0 - sim_wcos) < abs(1.0 - sim_cos) else "no"

        print(f"{label:<30} {sim_cos:>8.4f} {sim_wcos:>8.4f} {z_cos:>6.1f} {z_wcos:>6.1f} {better:>18}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Detection sensitivity curve")
    print("=" * 70)

    print(f"\n{'Attack %':>10} {'Cos Drift':>10} {'W.Cos Drift':>12} {'Improvement':>12}")
    print("-" * 48)

    for pct in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        test_vecs = generate_mixed_attack(
            client, np.random.default_rng(int(pct * 10000) + 1), count=100, attack_frac=pct
        )
        test_bundled = bundle(test_vecs)

        cos_drift = 1.0 - cosine_similarity(test_bundled, baseline_bundled)
        wcos_drift = 1.0 - engine.similarity(
            test_bundled, baseline_bundled, DistanceMetric.WEIGHTED_COSINE, weights=weights
        )

        improvement = (wcos_drift - cos_drift) / max(cos_drift, 0.001) * 100 if cos_drift > 0.001 else 0
        print(f"{pct * 100:>9.0f}% {cos_drift:>10.4f} {wcos_drift:>12.4f} {improvement:>+11.1f}%")

    print()


if __name__ == "__main__":
    main()
