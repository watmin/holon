#!/usr/bin/env python3
"""
Streaming Drift Adaptation

HYPOTHESIS:
===========
A frozen subspace trained on one traffic pattern flags all new patterns as
anomalous. An adaptive subspace (with amnesia) tracks drift and accepts
the new pattern, while still detecting truly anomalous traffic.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace (frozen)    - Trained once, never updated → fails on drift
2. OnlineSubspace (adaptive)  - Continuous updates with amnesia → tracks drift
3. OnlineSubspace.eigenvalues - Spectrum shifts reveal subspace adaptation

SCENARIO:
=========
Phase 1 (0-500):    Train on web browsing traffic
Phase 2 (500-1000): Traffic shifts to API patterns (concept drift)
Phase 3 (1000-1100): Dedicated SSH brute force attack burst

Two detectors run in parallel:
  Frozen:   Trained on phase 1, then only scores (never updates)
  Adaptive: Trained on phase 1, then updates on everything through phase 2

Both should detect the SSH attack in phase 3. Only the adaptive detector
should accept API traffic in phase 2 as normal.

VECTOR PROPERTIES EXPLOITED:
============================
- Amnesia controls effective window size for subspace learning
- Eigenvalue spectrum shift during drift (new variance directions)
- Frozen vs adaptive comparison isolates the adaptation effect
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient
from holon.subspace import OnlineSubspace


def encode_web_traffic(client, rng):
    """Phase 1: Web browsing patterns."""
    pages = ["index", "about", "docs", "blog", "contact", "gallery", "faq"]
    return client.encode(
        {
            "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
            "dst_ip": "192.168.1.10",
            "proto": "TCP",
            "dst_port": "80",
            "path": str(rng.choice(pages)),
            "method": "GET",
            "status": str(rng.choice(["200", "200", "301", "404"])),
            "content": "html",
        }
    )


def encode_api_traffic(client, rng):
    """Phase 2: API-heavy patterns."""
    endpoints = ["users", "orders", "inventory", "billing", "analytics", "events"]
    return client.encode(
        {
            "src_ip": f"10.0.{rng.integers(1, 5)}.{rng.integers(1, 20)}",
            "dst_ip": "192.168.1.5",
            "proto": "TCP",
            "dst_port": "443",
            "path": str(rng.choice(endpoints)),
            "method": str(rng.choice(["GET", "POST", "PUT", "DELETE"])),
            "status": str(rng.choice(["200", "201", "204"])),
            "content": "json",
        }
    )


def encode_ssh_attack(client, rng):
    """Phase 3: SSH brute force burst."""
    return client.encode(
        {
            "src_ip": f"45.33.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
            "dst_ip": "192.168.1.5",
            "proto": "TCP",
            "dst_port": "22",
            "path": "auth",
            "method": "CONNECT",
            "status": "401",
            "content": "binary",
        }
    )


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)
    rng_api = np.random.default_rng(100)
    rng_attack = np.random.default_rng(200)

    print("=" * 70)
    print("EXPERIMENT 4: Streaming Drift Adaptation")
    print("=" * 70)

    # --- Generate all data ---
    n_web = 500
    n_api = 500
    n_attack = 100

    print(f"\n  Phase 1: {n_web} web traffic vectors (training)")
    print(f"  Phase 2: {n_api} API traffic vectors (drift)")
    print(f"  Phase 3: {n_attack} SSH attack vectors (anomaly)")

    web_vecs = [encode_web_traffic(client, rng) for _ in range(n_web)]
    api_vecs = [encode_api_traffic(client, rng_api) for _ in range(n_api)]
    attack_vecs = [encode_ssh_attack(client, rng_attack) for _ in range(n_attack)]

    # --- Train both detectors on phase 1 ---
    print("\n" + "-" * 70)
    print("PART A: Training both detectors on web traffic (Phase 1)")
    print("-" * 70)

    frozen = OnlineSubspace(dim=4096, k=64, amnesia=2.0, sigma_mult=3.5)
    adaptive = OnlineSubspace(dim=4096, k=64, amnesia=2.0, sigma_mult=3.5)

    for v in web_vecs:
        frozen.update(v)
        adaptive.update(v)

    print(f"\n  Frozen:   {frozen}")
    print(f"  Adaptive: {adaptive}")
    print(f"  Thresholds: frozen={frozen.threshold:.4f}, adaptive={adaptive.threshold:.4f}")

    # --- Phase 2: Score API traffic ---
    print("\n" + "-" * 70)
    print("PART B: Phase 2 — API traffic (drift)")
    print("-" * 70)

    frozen_api_residuals = []
    adaptive_api_residuals = []

    for v in api_vecs:
        frozen_api_residuals.append(frozen.residual(v))
        adaptive_api_residuals.append(adaptive.update(v))

    frozen_api = np.array(frozen_api_residuals)
    adaptive_api = np.array(adaptive_api_residuals)

    frozen_thresh = frozen.threshold
    adaptive_thresh = adaptive.threshold

    # Track how adaptive residuals decrease over time
    windows = [(0, 50), (50, 100), (100, 200), (200, 300), (400, 500)]

    print(f"\n  {'Window':<15} {'Frozen (mean)':>14} {'Adaptive (mean)':>16} {'Frozen FP%':>11} {'Adapt FP%':>10}")
    print("  " + "-" * 68)

    for start, end in windows:
        f_mean = np.mean(frozen_api[start:end])
        a_mean = np.mean(adaptive_api[start:end])
        f_fp = np.sum(frozen_api[start:end] > frozen_thresh) / (end - start) * 100
        a_fp = np.sum(adaptive_api[start:end] > adaptive_thresh) / (end - start) * 100
        print(f"  {f'{start}-{end}':<15} {f_mean:>14.4f} {a_mean:>16.4f} {f_fp:>10.1f}% {a_fp:>9.1f}%")

    # --- Phase 3: SSH attack burst ---
    print("\n" + "-" * 70)
    print("PART C: Phase 3 — SSH attack burst")
    print("-" * 70)

    frozen_attack_residuals = [frozen.residual(v) for v in attack_vecs]
    adaptive_attack_residuals = [adaptive.residual(v) for v in attack_vecs]

    frozen_attack = np.array(frozen_attack_residuals)
    adaptive_attack = np.array(adaptive_attack_residuals)

    frozen_attack_tp = np.sum(frozen_attack > frozen_thresh) / len(frozen_attack) * 100
    adaptive_attack_tp = np.sum(adaptive_attack > adaptive_thresh) / len(adaptive_attack) * 100

    print(f"\n  {'Detector':<15} {'Mean Residual':>14} {'Threshold':>12} {'TP Rate':>10}")
    print("  " + "-" * 53)
    print(f"  {'Frozen':<15} {np.mean(frozen_attack):>14.4f} {frozen_thresh:>12.4f} {frozen_attack_tp:>9.1f}%")
    print(f"  {'Adaptive':<15} {np.mean(adaptive_attack):>14.4f} {adaptive_thresh:>12.4f} {adaptive_attack_tp:>9.1f}%")

    # --- Eigenvalue comparison ---
    print("\n" + "-" * 70)
    print("PART D: Eigenvalue spectrum — frozen vs adaptive after drift")
    print("-" * 70)

    frozen_eigs = np.sort(frozen.eigenvalues)[::-1]
    adaptive_eigs = np.sort(adaptive.eigenvalues)[::-1]

    print(f"\n  {'Component':<12} {'Frozen':>10} {'Adaptive':>10} {'Change':>10}")
    print("  " + "-" * 44)
    for i in range(min(10, len(frozen_eigs))):
        change = (adaptive_eigs[i] - frozen_eigs[i]) / (frozen_eigs[i] + 1e-10) * 100
        print(f"  {i + 1:<12} {frozen_eigs[i]:>10.2f} {adaptive_eigs[i]:>10.2f} {change:>+9.1f}%")

    # --- Summary ---
    print("\n" + "-" * 70)
    print("PART E: Adaptation summary")
    print("-" * 70)

    frozen_api_fp = np.sum(frozen_api > frozen_thresh) / len(frozen_api) * 100
    adaptive_api_fp = np.sum(adaptive_api > adaptive_thresh) / len(adaptive_api) * 100

    adaptive_late_fp = np.sum(adaptive_api[300:] > adaptive_thresh) / len(adaptive_api[300:]) * 100

    print(f"\n  Frozen detector on API traffic:    FP = {frozen_api_fp:.1f}%")
    print(f"  Adaptive detector on API traffic:  FP = {adaptive_api_fp:.1f}%")
    print(f"  Adaptive (last 200 API vectors):   FP = {adaptive_late_fp:.1f}%")
    print(f"\n  Frozen on SSH attack:              TP = {frozen_attack_tp:.1f}%")
    print(f"  Adaptive on SSH attack:            TP = {adaptive_attack_tp:.1f}%")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Frozen flags API traffic as anomalous (FP > 50%)",
            frozen_api_fp > 50,
            f"FP={frozen_api_fp:.1f}%",
        ),
        (
            "Adaptive reduces API false positives vs frozen",
            adaptive_api_fp < frozen_api_fp,
            f"adaptive={adaptive_api_fp:.1f}% vs frozen={frozen_api_fp:.1f}%",
        ),
        (
            "Adaptive detects SSH attack (TP > 80%)",
            adaptive_attack_tp > 80,
            f"TP={adaptive_attack_tp:.1f}%",
        ),
        (
            "Frozen also detects SSH attack (TP > 80%)",
            frozen_attack_tp > 80,
            f"TP={frozen_attack_tp:.1f}%",
        ),
        (
            "Adaptive late-phase FP < 20% (subspace adapted)",
            adaptive_late_fp < 20,
            f"FP={adaptive_late_fp:.1f}%",
        ),
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
