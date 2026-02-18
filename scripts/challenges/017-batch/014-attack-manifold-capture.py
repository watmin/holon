#!/usr/bin/env python3
"""
Attack Manifold Capture

HYPOTHESIS:
===========
We can train a SEPARATE subspace on attack-only traffic and it will
converge to a meaningful manifold that:
  1. Has low residual for same-type attack vectors (they fit the manifold)
  2. Has high residual for normal traffic (it doesn't fit)
  3. Has high residual for OTHER attack types (manifolds are distinct)

This is the foundational experiment for the Engram concept: if attack
subspaces don't capture attack structure, nothing downstream works.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace           - Train on attack traffic (not normal)
2. OnlineSubspace.residual  - Discriminate between matching and non-matching traffic
3. OnlineSubspace.eigenvalues - Verify convergence via spectrum stability
4. Engram / EngramLibrary   - Mint engrams from attack subspaces

SCENARIO:
=========
Train 4 separate attack subspaces (DNS amp, SYN flood, credential stuffing,
exfiltration), then cross-test each subspace against all traffic types.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient
from holon.subspace import OnlineSubspace

FIELDS = ["src_ip", "dst_ip", "proto", "dst_port", "path", "status", "ttl"]


def make_normal(rng):
    return {
        "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
        "dst_ip": f"192.168.1.{rng.integers(1, 10)}",
        "proto": str(rng.choice(["TCP", "UDP", "TCP", "TCP"])),
        "dst_port": str(rng.choice(["80", "443", "8080"])),
        "path": str(rng.choice(["api", "static", "health", "metrics"])),
        "status": str(rng.choice(["200", "200", "200", "301", "404"])),
        "ttl": str(rng.choice(["64", "128"])),
    }


def make_dns_amp(rng):
    return {
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "245",
    }


def make_syn_flood(rng):
    return {
        "src_ip": f"172.16.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.1",
        "proto": "TCP",
        "dst_port": "80",
        "path": "syn",
        "status": "000",
        "ttl": str(rng.integers(30, 60)),
    }


def make_cred_stuff(rng):
    return {
        "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.5",
        "proto": "TCP",
        "dst_port": "443",
        "path": "auth",
        "status": "401",
        "ttl": "64",
    }


def make_exfil(rng):
    return {
        "src_ip": "192.168.1.50",
        "dst_ip": f"203.0.113.{rng.integers(1, 50)}",
        "proto": "TCP",
        "dst_port": str(rng.choice(["8443", "4443", "9443"])),
        "path": "upload",
        "status": "200",
        "ttl": "128",
    }


ATTACK_TYPES = {
    "dns_amp": make_dns_amp,
    "syn_flood": make_syn_flood,
    "cred_stuff": make_cred_stuff,
    "exfil": make_exfil,
}


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 14: Attack Manifold Capture")
    print("=" * 70)

    n_train = 500
    n_test = 100

    # --- Generate test data ---
    normal_test = [client.encode(make_normal(np.random.default_rng(i + 5000)))
                   for i in range(n_test)]

    attack_train_vecs = {}
    attack_test_vecs = {}
    for name, factory in ATTACK_TYPES.items():
        attack_train_vecs[name] = [
            client.encode(factory(np.random.default_rng(i + 1000)))
            for i in range(n_train)
        ]
        attack_test_vecs[name] = [
            client.encode(factory(np.random.default_rng(i + 3000)))
            for i in range(n_test)
        ]

    # --- Part A: Train attack subspaces ---
    print(f"\nTraining {len(ATTACK_TYPES)} attack subspaces ({n_train} vectors each)...")
    attack_subs = {}
    for name in ATTACK_TYPES:
        sub = OnlineSubspace(dim=4096, k=32, amnesia=2.0, sigma_mult=3.5)
        for vec in attack_train_vecs[name]:
            sub.update(vec)
        attack_subs[name] = sub
        eigs = sub.eigenvalues
        active = int(np.sum(eigs > 1e-6))
        print(f"  {name:15s}: active_PCs={active:2d}, "
              f"threshold={sub.threshold:.2f}, "
              f"top_eig={eigs[0]:.1f}")

    # --- Part B: Cross-test residuals ---
    print("\n" + "-" * 70)
    print("PART B: Cross-test residuals (rows=subspace, cols=traffic type)")
    print("-" * 70)

    all_types = list(ATTACK_TYPES.keys()) + ["normal"]
    all_test = {**attack_test_vecs, "normal": normal_test}

    header = f"{'Subspace':>15s}"
    for t in all_types:
        header += f" {t:>12s}"
    print(f"\n{header}")
    print(f"  {'-' * (15 + 13 * len(all_types))}")

    residual_matrix = {}
    for sub_name, sub in attack_subs.items():
        row = {}
        line = f"  {sub_name:>13s}"
        for traffic_name in all_types:
            residuals = [sub.residual(v) for v in all_test[traffic_name]]
            mean_res = np.mean(residuals)
            row[traffic_name] = mean_res
            marker = " *" if traffic_name == sub_name else "  "
            line += f" {mean_res:>10.2f}{marker}"
        residual_matrix[sub_name] = row
        print(line)
    print(f"\n  (* = same attack type — should have lowest residual)")

    # --- Part C: Eigenvalue spectrum comparison ---
    print("\n" + "-" * 70)
    print("PART C: Eigenvalue spectrum comparison (cosine similarity)")
    print("-" * 70)

    eig_vecs = {}
    for name, sub in attack_subs.items():
        eig = sub.eigenvalues
        norm = np.linalg.norm(eig)
        eig_vecs[name] = eig / norm if norm > 1e-10 else eig

    header = f"{'':>15s}"
    for t in ATTACK_TYPES:
        header += f" {t:>12s}"
    print(f"\n{header}")
    print(f"  {'-' * (15 + 13 * len(ATTACK_TYPES))}")

    for name_a in ATTACK_TYPES:
        line = f"  {name_a:>13s}"
        for name_b in ATTACK_TYPES:
            cos = float(np.dot(eig_vecs[name_a], eig_vecs[name_b]))
            line += f" {cos:>12.4f}"
        print(line)

    # --- Part D: Convergence check ---
    print("\n" + "-" * 70)
    print("PART D: Convergence — eigenvalue stability over training")
    print("-" * 70)

    sub_conv = OnlineSubspace(dim=4096, k=32, amnesia=2.0)
    checkpoints = [50, 100, 200, 300, 500]
    eig_snapshots = {}
    idx = 0
    for vec in attack_train_vecs["dns_amp"]:
        sub_conv.update(vec)
        idx += 1
        if idx in checkpoints:
            eig_snapshots[idx] = sub_conv.eigenvalues.copy()

    print(f"\n  DNS amp eigenvalue drift (cosine vs final):")
    final_eig = eig_snapshots[500]
    final_norm = np.linalg.norm(final_eig)
    for cp in checkpoints[:-1]:
        cp_eig = eig_snapshots[cp]
        cp_norm = np.linalg.norm(cp_eig)
        if cp_norm > 1e-10 and final_norm > 1e-10:
            cos = float(np.dot(cp_eig, final_eig) / (cp_norm * final_norm))
        else:
            cos = 0.0
        print(f"    n={cp:3d} vs n=500: cosine={cos:.6f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = []

    for sub_name in ATTACK_TYPES:
        self_res = residual_matrix[sub_name][sub_name]
        others = [residual_matrix[sub_name][t] for t in all_types if t != sub_name]
        min_other = min(others)
        checks.append((
            f"{sub_name} self-residual < all others",
            self_res < min_other,
            f"self={self_res:.2f}, min_other={min_other:.2f}",
        ))

    for sub_name in ATTACK_TYPES:
        normal_res = residual_matrix[sub_name]["normal"]
        self_res = residual_matrix[sub_name][sub_name]
        checks.append((
            f"{sub_name} rejects normal traffic (ratio > 1.5x)",
            normal_res / max(self_res, 1e-10) > 1.5,
            f"normal/self={normal_res / max(self_res, 1e-10):.2f}x",
        ))

    conv_cos = float(np.dot(eig_snapshots[200], final_eig)
                     / (np.linalg.norm(eig_snapshots[200]) * final_norm))
    checks.append((
        "Eigenvalue spectrum converges (cos@200 vs @500 > 0.95)",
        conv_cos > 0.95,
        f"cos={conv_cos:.6f}",
    ))

    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc} ({detail})")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
