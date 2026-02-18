#!/usr/bin/env python3
"""
Anomaly Separation — Residual as Detector

HYPOTHESIS:
===========
Subspace residuals cleanly separate in-distribution (normal) vectors from
out-of-distribution (anomalous) vectors. The adaptive threshold should
achieve high true-positive rate with zero false positives on holdout data.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.update()       - Train on normal traffic
2. OnlineSubspace.residual()     - Score without updating
3. OnlineSubspace.threshold      - Adaptive cutoff
4. OnlineSubspace.anomalous_component() - Extract what doesn't belong

SCENARIO:
=========
Train on 1000 normal web traffic vectors, then score:
  - 200 holdout normal vectors (should be below threshold)
  - 200 DNS amplification attack vectors (should be above)
  - 200 credential stuffing vectors (should be above)
  - 200 data exfiltration vectors (should be above — this is the hard one,
    since exfil shares some structure with normal traffic)

VECTOR PROPERTIES EXPLOITED:
============================
- Subspace membership as a tighter boundary than centroid distance
- Residual magnitude as a calibrated anomaly score
- Adaptive threshold from streaming residual statistics
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import bundle, prototype
from holon.subspace import OnlineSubspace


def encode_normal(client, rng):
    """Normal diverse web/API traffic."""
    protos = ["TCP", "UDP", "TCP", "TCP"]
    dst_ports = ["80", "443", "22", "8080", "3306"]
    paths = ["api", "static", "health", "metrics", "users"]
    agents = ["browser", "script", "browser", "browser"]
    return client.encode(
        {
            "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
            "dst_ip": f"192.168.1.{rng.integers(1, 10)}",
            "proto": str(rng.choice(protos)),
            "src_port": str(rng.integers(1024, 65535)),
            "dst_port": str(rng.choice(dst_ports)),
            "path": str(rng.choice(paths)),
            "status": str(rng.choice(["200", "200", "200", "301", "404"])),
            "agent": str(rng.choice(agents)),
        }
    )


def encode_dns_amp(client, rng):
    """DNS amplification: few reflectors, one target, large UDP/53."""
    reflectors = ["8.8.8.8", "1.1.1.1", "9.9.9.9"]
    return client.encode(
        {
            "src_ip": str(rng.choice(reflectors)),
            "dst_ip": "192.168.1.100",
            "proto": "UDP",
            "src_port": "53",
            "dst_port": str(rng.integers(1024, 65535)),
            "path": "dns",
            "status": "response",
            "agent": "resolver",
        }
    )


def encode_credential_stuffing(client, rng):
    """Credential stuffing: many IPs, same endpoint, POST, 401."""
    return client.encode(
        {
            "src_ip": f"45.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
            "dst_ip": "192.168.1.5",
            "proto": "TCP",
            "src_port": str(rng.integers(1024, 65535)),
            "dst_port": "443",
            "path": "auth",
            "status": "401",
            "agent": "script",
        }
    )


def encode_exfiltration(client, rng):
    """Data exfiltration: internal IP, large POST, export endpoint."""
    return client.encode(
        {
            "src_ip": "10.0.1.50",
            "dst_ip": f"203.0.113.{rng.integers(1, 10)}",
            "proto": "TCP",
            "src_port": str(rng.integers(1024, 65535)),
            "dst_port": "443",
            "path": str(rng.choice(["export", "backup", "dump"])),
            "status": "200",
            "agent": "script",
        }
    )


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 2: Anomaly Separation — Residual as Detector")
    print("=" * 70)

    # --- Generate data ---
    n_train = 1000
    n_test = 200

    print(f"\nGenerating {n_train} training vectors (normal)...")
    train_vecs = [encode_normal(client, rng) for _ in range(n_train)]

    print(f"Generating {n_test} holdout normal vectors...")
    rng_holdout = np.random.default_rng(99)
    normal_test = [encode_normal(client, rng_holdout) for _ in range(n_test)]

    print(f"Generating {n_test} DNS amplification vectors...")
    rng_dns = np.random.default_rng(100)
    dns_test = [encode_dns_amp(client, rng_dns) for _ in range(n_test)]

    print(f"Generating {n_test} credential stuffing vectors...")
    rng_cred = np.random.default_rng(101)
    cred_test = [encode_credential_stuffing(client, rng_cred) for _ in range(n_test)]

    print(f"Generating {n_test} exfiltration vectors...")
    rng_exfil = np.random.default_rng(102)
    exfil_test = [encode_exfiltration(client, rng_exfil) for _ in range(n_test)]

    # --- Train subspace ---
    print("\n" + "-" * 70)
    print("PART A: Training subspace (k=64)")
    print("-" * 70)

    sub = OnlineSubspace(dim=4096, k=64, amnesia=2.0, sigma_mult=3.5)
    train_residuals = []
    for v in train_vecs:
        train_residuals.append(sub.update(v))

    print(f"\n  Trained on {n_train} vectors")
    print(f"  {sub}")
    print(f"  Threshold: {sub.threshold:.4f}")
    print(f"  Mean training residual (last 100): {np.mean(train_residuals[-100:]):.4f}")

    # --- Score all test sets ---
    print("\n" + "-" * 70)
    print("PART B: Residual distributions")
    print("-" * 70)

    test_sets = [
        ("Normal (holdout)", normal_test),
        ("DNS amplification", dns_test),
        ("Credential stuffing", cred_test),
        ("Exfiltration", exfil_test),
    ]

    threshold = sub.threshold
    results = {}

    print(f"\n  Threshold: {threshold:.4f}")
    print(f"\n  {'Traffic Type':<25} {'Mean Res':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'FP/TP Rate':>12}")
    print("  " + "-" * 75)

    for label, vecs in test_sets:
        residuals = np.array([sub.residual(v) for v in vecs])
        above = np.sum(residuals > threshold)
        rate = above / len(residuals) * 100

        is_normal = label.startswith("Normal")
        rate_label = f"FP={rate:.1f}%" if is_normal else f"TP={rate:.1f}%"

        results[label] = {
            "residuals": residuals,
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "above_threshold": above,
            "rate": rate,
        }

        print(
            f"  {label:<25} {np.mean(residuals):>10.4f} {np.std(residuals):>8.4f} "
            f"{np.min(residuals):>8.4f} {np.max(residuals):>8.4f} {rate_label:>12}"
        )

    # --- Separation ratios ---
    print("\n" + "-" * 70)
    print("PART C: Separation ratios (anomaly mean / normal mean)")
    print("-" * 70)

    normal_mean = results["Normal (holdout)"]["mean"]
    print()
    for label in ["DNS amplification", "Credential stuffing", "Exfiltration"]:
        ratio = results[label]["mean"] / normal_mean if normal_mean > 0 else 0
        print(f"  {label:<25} {ratio:.2f}×")

    # --- Comparison: subspace vs centroid ---
    print("\n" + "-" * 70)
    print("PART D: Subspace residual vs cosine-to-centroid")
    print("-" * 70)

    centroid = prototype(train_vecs)

    print(f"\n  {'Traffic Type':<25} {'Subspace Res':>14} {'Cosine to Ctr':>15} {'1-Cosine':>10}")
    print("  " + "-" * 66)

    for label, vecs in test_sets:
        sub_res = np.mean([sub.residual(v) for v in vecs[:50]])
        cos_sims = np.array([cosine_similarity(v, centroid) for v in vecs[:50]])
        cos_dist = 1.0 - np.mean(cos_sims)
        print(f"  {label:<25} {sub_res:>14.4f} {np.mean(cos_sims):>15.4f} {cos_dist:>10.4f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    normal_fp = results["Normal (holdout)"]["above_threshold"]
    dns_tp = results["DNS amplification"]["rate"]
    cred_tp = results["Credential stuffing"]["rate"]
    exfil_tp = results["Exfiltration"]["rate"]
    dns_ratio = results["DNS amplification"]["mean"] / normal_mean if normal_mean > 0 else 0
    cred_ratio = results["Credential stuffing"]["mean"] / normal_mean if normal_mean > 0 else 0
    exfil_ratio = results["Exfiltration"]["mean"] / normal_mean if normal_mean > 0 else 0

    checks = [
        ("Zero false positives on holdout", normal_fp == 0, f"FP={normal_fp}"),
        ("DNS amplification TP > 95%", dns_tp > 95, f"TP={dns_tp:.1f}%"),
        ("Credential stuffing TP > 95%", cred_tp > 95, f"TP={cred_tp:.1f}%"),
        ("Exfiltration TP > 50%", exfil_tp > 50, f"TP={exfil_tp:.1f}%"),
        ("DNS separation > 1.2×", dns_ratio > 1.2, f"{dns_ratio:.2f}×"),
        ("Cred separation > 1.2×", cred_ratio > 1.2, f"{cred_ratio:.2f}×"),
        ("Exfil separation > 1.1×", exfil_ratio > 1.1, f"{exfil_ratio:.2f}×"),
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
