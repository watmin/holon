#!/usr/bin/env python3
"""
CCIPCA Convergence on Structured Encodings

HYPOTHESIS:
===========
CCIPCA converges to a meaningful subspace from holon-encoded structured data.
The eigenvalue spectrum should reveal low intrinsic dimensionality: structured
encodings (7-10 fields, each with limited vocabulary) can't span all 4096
dimensions. The subspace should stabilize within a few hundred vectors.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.update()     - Incremental CCIPCA learning
2. OnlineSubspace.eigenvalues  - Approximate eigenvalue spectrum
3. OnlineSubspace.residual()   - Reconstruction error (read-only)
4. OnlineSubspace.project()    - Low-dimensional embedding

SCENARIO:
=========
Generate 2000 normal packet-like encodings with realistic diversity
(varied IPs, ports, protocols, TTLs). Feed into OnlineSubspace one-by-one.
Track convergence metrics: residual stabilization, eigenvalue concentration,
and explained variance.

VECTOR PROPERTIES EXPLOITED:
============================
- Low intrinsic dimensionality of structured encodings
- Eigenvalue spectrum shape (knee point = true rank)
- Convergence rate of online PCA in high dimensions
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient
from holon.subspace import OnlineSubspace


def make_packet(client, rng):
    """Generate a diverse normal packet encoding."""
    protos = ["TCP", "UDP", "TCP", "TCP", "ICMP"]
    dst_ports = [80, 443, 22, 8080, 53, 3306, 8443]
    methods = ["GET", "POST", "GET", "GET", "PUT"]
    return client.encode(
        {
            "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
            "dst_ip": f"192.168.1.{rng.integers(1, 20)}",
            "proto": str(rng.choice(protos)),
            "src_port": str(rng.integers(1024, 65535)),
            "dst_port": str(rng.choice(dst_ports)),
            "pkt_len": str(rng.integers(64, 1500)),
            "ttl": str(rng.choice([64, 128, 255])),
            "method": str(rng.choice(methods)),
        }
    )


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 1: CCIPCA Convergence on Structured Encodings")
    print("=" * 70)

    # --- Test multiple k values ---
    k_values = [16, 32, 64, 128]
    n_vectors = 2000

    # Pre-generate all vectors (same data for all k values)
    print(f"\nEncoding {n_vectors} normal packet vectors...")
    vectors = [make_packet(client, rng) for _ in range(n_vectors)]
    print("Done.\n")

    print("-" * 70)
    print("PART A: Convergence speed across k values")
    print("-" * 70)

    for k in k_values:
        sub = OnlineSubspace(dim=4096, k=k, amnesia=2.0)

        residuals = []
        for v in vectors:
            r = sub.update(v)
            residuals.append(r)

        residuals = np.array(residuals)

        # Convergence: CV of last 100 residuals
        tail = residuals[-100:]
        cv = np.std(tail) / np.mean(tail) if np.mean(tail) > 0 else 0

        # Find stabilization point: where rolling CV drops below 15%
        window = 50
        stabilized_at = n_vectors
        for i in range(window, n_vectors):
            chunk = residuals[i - window : i]
            chunk_cv = np.std(chunk) / np.mean(chunk) if np.mean(chunk) > 0 else 0
            if chunk_cv < 0.15:
                stabilized_at = i
                break

        eigs = sub.eigenvalues
        top_5_pct = np.sum(eigs[:5]) / np.sum(eigs) * 100 if np.sum(eigs) > 0 else 0

        print(f"\n  k={k:3d}:")
        print(f"    Final residual (mean last 100):  {np.mean(tail):.4f}")
        print(f"    Final residual CV (last 100):    {cv:.4f}")
        print(f"    Stabilized at vector:            {stabilized_at}")
        print(f"    Top-5 eigenvalue share:          {top_5_pct:.1f}%")
        print(f"    Threshold:                       {sub.threshold:.4f}")
        print(f"    {sub}")

    print("\n" + "-" * 70)
    print("PART B: Eigenvalue spectrum (k=64)")
    print("-" * 70)

    sub = OnlineSubspace(dim=4096, k=64, amnesia=2.0)
    for v in vectors:
        sub.update(v)

    eigs = sub.eigenvalues
    sorted_eigs = np.sort(eigs)[::-1]

    print(f"\n  {'Rank':<6} {'Eigenvalue':>12} {'Cumulative %':>14}")
    print("  " + "-" * 34)

    cumsum = 0.0
    total = np.sum(sorted_eigs)
    for i in range(min(20, len(sorted_eigs))):
        cumsum += sorted_eigs[i]
        pct = cumsum / total * 100 if total > 0 else 0
        bar = "█" * int(sorted_eigs[i] / sorted_eigs[0] * 30) if sorted_eigs[0] > 0 else ""
        print(f"  {i + 1:<6} {sorted_eigs[i]:>12.4f} {pct:>13.1f}% {bar}")

    # Find knee point
    if total > 0:
        cumsum_pct = np.cumsum(sorted_eigs) / total
        knee_90 = int(np.searchsorted(cumsum_pct, 0.90)) + 1
        knee_95 = int(np.searchsorted(cumsum_pct, 0.95)) + 1
        knee_99 = int(np.searchsorted(cumsum_pct, 0.99)) + 1
        print(f"\n  Knee points: 90% at k={knee_90}, 95% at k={knee_95}, 99% at k={knee_99}")
        print(f"  Intrinsic dimensionality (90%): {knee_90} out of 4096")

    print("\n" + "-" * 70)
    print("PART C: Residual trajectory over time (k=64)")
    print("-" * 70)

    sub2 = OnlineSubspace(dim=4096, k=64, amnesia=2.0)
    residuals = []
    for v in vectors:
        residuals.append(sub2.update(v))
    residuals = np.array(residuals)

    checkpoints = [10, 50, 100, 200, 500, 1000, 2000]
    print(f"\n  {'Vectors seen':<14} {'Mean residual':>14} {'Std':>10} {'CV':>8}")
    print("  " + "-" * 48)
    for cp in checkpoints:
        if cp > len(residuals):
            break
        chunk = residuals[max(0, cp - 50) : cp]
        mean_r = np.mean(chunk)
        std_r = np.std(chunk)
        cv = std_r / mean_r if mean_r > 0 else 0
        print(f"  {cp:<14} {mean_r:>14.4f} {std_r:>10.4f} {cv:>8.4f}")

    print("\n" + "-" * 70)
    print("PART D: Projection sanity check")
    print("-" * 70)

    # Project a few vectors and check they're low-dimensional
    coords = np.array([sub.project(v) for v in vectors[:100]])
    print(f"\n  Projection shape: {coords.shape} (100 vectors × 64 components)")
    print(f"  Coordinate magnitude (mean): {np.mean(np.abs(coords)):.4f}")
    print(f"  Coordinate magnitude (std):  {np.std(coords):.4f}")
    print(f"  Non-trivial components (std > 0.1): {np.sum(np.std(coords, axis=0) > 0.1)}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    tail_cv = np.std(residuals[-100:]) / np.mean(residuals[-100:])
    eigs = sub.eigenvalues
    sorted_eigs = np.sort(eigs)[::-1]
    top5_share = np.sum(sorted_eigs[:5]) / np.sum(sorted_eigs) * 100

    checks = [
        ("Residual stabilizes (CV < 15% last 100)", tail_cv < 0.15, f"CV={tail_cv:.4f}"),
        ("Eigenvalue spectrum has knee (top-5 > 30%)", top5_share > 30, f"{top5_share:.1f}%"),
        ("Converges within 500 vectors", stabilized_at <= 500, f"at {stabilized_at}"),
    ]

    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc} ({detail})")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
