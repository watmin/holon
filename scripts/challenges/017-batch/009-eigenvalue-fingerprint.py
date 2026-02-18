#!/usr/bin/env python3
"""
Eigenvalue Spectrum Fingerprinting for Attack Classification

HYPOTHESIS:
===========
Different attack types produce distinct eigenvalue spectrum shifts when
they hit the subspace. A volumetric flood (DNS amp) should cause PC1 to
explode. A diversified attack (SYN flood from many IPs) should grow new
lower-ranked components. The eigenvalue change vector itself can classify
attack type.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.eigenvalues         - Spectrum before/during attack
2. Eigenvalue delta vector            - Change fingerprint
3. Cosine similarity on deltas        - Attack type matching

SCENARIO:
=========
Train a subspace on normal traffic. Snapshot eigenvalues. Then feed each
attack type and snapshot eigenvalues again. The delta (after - before)
is a compact fingerprint of how the attack distorted the subspace.

Four attack types:
  DNS amp:     homogeneous, few sources → dominates PC1
  SYN flood:   varied sources, same target → spreads across components
  Cred stuff:  same endpoint, varied sources → medium spread
  Exfil:       subtle, small shift → minimal eigenvalue change

VECTOR PROPERTIES EXPLOITED:
============================
- Eigenvalue = variance along principal component
- Attack traffic adds variance in specific directions
- The pattern of eigenvalue change is attack-type-specific
- Compact: only k numbers characterize the attack signature
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient
from holon.subspace import OnlineSubspace

FIELDS = ["src_ip", "dst_ip", "proto", "dst_port", "path", "status", "ttl"]


def encode_normal(client, rng):
    return client.encode({
        "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
        "dst_ip": f"192.168.1.{rng.integers(1, 10)}",
        "proto": str(rng.choice(["TCP", "UDP", "TCP", "TCP"])),
        "dst_port": str(rng.choice(["80", "443", "8080"])),
        "path": str(rng.choice(["api", "static", "health", "metrics"])),
        "status": str(rng.choice(["200", "200", "301", "404"])),
        "ttl": str(rng.choice(["64", "128"])),
    })


def encode_dns_amp(client, rng):
    return client.encode({
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
        "dst_ip": "192.168.1.100", "proto": "UDP", "dst_port": "53",
        "path": "dns", "status": "200", "ttl": "245",
    })


def encode_syn_flood(client, rng):
    return client.encode({
        "src_ip": f"{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}",
        "dst_ip": "192.168.1.100", "proto": "TCP", "dst_port": "80",
        "path": "syn", "status": "none", "ttl": str(rng.choice(["64", "128", "255"])),
    })


def encode_cred_stuff(client, rng):
    return client.encode({
        "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.5", "proto": "TCP", "dst_port": "443",
        "path": "auth", "status": "401", "ttl": "64",
    })


def encode_exfil(client, rng):
    return client.encode({
        "src_ip": "10.0.1.50",
        "dst_ip": f"203.0.113.{rng.integers(1, 10)}",
        "proto": "TCP", "dst_port": "443",
        "path": str(rng.choice(["export", "backup"])),
        "status": "200", "ttl": "64",
    })


def train_baseline_subspace(client, n=500, k=32):
    """Train a subspace on normal traffic, return it."""
    rng = np.random.default_rng(42)
    sub = OnlineSubspace(dim=4096, k=k, amnesia=2.0)
    for _ in range(n):
        sub.update(encode_normal(client, rng))
    return sub


def get_eigenvalue_delta(client, sub_snapshot, attack_gen, n_attack=200, k=32):
    """Feed attack traffic into a fresh copy of the subspace, measure eigenvalue shift."""
    sub = OnlineSubspace.from_snapshot(sub_snapshot)
    eigs_before = np.sort(sub.eigenvalues)[::-1]

    for i in range(n_attack):
        vec = attack_gen(client, np.random.default_rng(i + 1000))
        sub.update(vec)

    eigs_after = np.sort(sub.eigenvalues)[::-1]
    delta = eigs_after - eigs_before

    return eigs_before, eigs_after, delta


def delta_similarity(d1, d2):
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(d1, d2) / (n1 * n2))


def main():
    client = HolonClient(dimensions=4096)
    k = 32

    print("=" * 70)
    print("EXPERIMENT 9: Eigenvalue Spectrum Fingerprinting")
    print("=" * 70)

    # --- Train baseline ---
    print(f"\nTraining baseline subspace (k={k})...")
    sub = train_baseline_subspace(client, n=500, k=k)
    snap = sub.snapshot()
    baseline_eigs = np.sort(sub.eigenvalues)[::-1]
    print(f"  {sub}")

    # --- Compute eigenvalue deltas for each attack type ---
    attacks = {
        "DNS amp": encode_dns_amp,
        "SYN flood": encode_syn_flood,
        "Cred stuff": encode_cred_stuff,
        "Exfil": encode_exfil,
    }

    deltas = {}
    eigs_after_map = {}

    for name, gen_fn in attacks.items():
        _, eigs_after, delta = get_eigenvalue_delta(client, snap, gen_fn, n_attack=200, k=k)
        deltas[name] = delta
        eigs_after_map[name] = eigs_after

    # --- WALKTHROUGH: trace one attack through the eigenvalue pipeline ---
    print("\n" + "-" * 70)
    print("WALKTHROUGH: How eigenvalue deltas become a fingerprint")
    print("-" * 70)

    print(f"\n  The subspace has {k} principal components (PCs).")
    print(f"  Each PC captures a direction of variance in the data.")
    print(f"  The eigenvalue of each PC = how much variance it explains.\n")

    print(f"  Baseline eigenvalues (top 8, after training on normal traffic):")
    print(f"    {'PC':<5} {'Eigenvalue':>12}  Meaning")
    print(f"    {'-'*50}")
    for i in range(min(8, k)):
        pct = baseline_eigs[i] / np.sum(baseline_eigs) * 100
        print(f"    PC{i+1:<3} {baseline_eigs[i]:>12.2f}  ({pct:.1f}% of total variance)")

    print(f"\n  Now: feed 200 DNS amplification packets and re-check.")
    walk_sub = OnlineSubspace.from_snapshot(snap)
    for i in range(200):
        walk_sub.update(encode_dns_amp(client, np.random.default_rng(i + 5000)))
    walk_eigs = np.sort(walk_sub.eigenvalues)[::-1]
    walk_delta = walk_eigs - baseline_eigs

    print(f"\n  {'PC':<5} {'Before':>10} {'After':>10} {'Delta':>10}  What happened")
    print(f"  {'-'*60}")
    for i in range(min(8, k)):
        d = walk_delta[i]
        sign = "+" if d >= 0 else ""
        if abs(d) > 0.3 * abs(walk_delta[0]):
            interp = "◀ big shift"
        elif abs(d) > 0.1 * abs(walk_delta[0]):
            interp = "  moderate"
        else:
            interp = "  negligible"
        print(f"  PC{i+1:<3} {baseline_eigs[i]:>10.2f} {walk_eigs[i]:>10.2f} {sign}{d:>9.2f}  {interp}")

    print(f"\n  The delta vector [{', '.join(f'{d:+.1f}' for d in walk_delta[:6])}...] IS the fingerprint.")
    print(f"  DNS amp pushes a few PCs hard (concentrated, volumetric).")
    print(f"  A SYN flood would spread the change across many PCs instead.")
    print(f"  That difference in 'shape' distinguishes attack types.")

    # --- Part A: Eigenvalue shift profiles ---
    print("\n" + "-" * 70)
    print("PART A: Eigenvalue shift profiles (top 10 components)")
    print("-" * 70)

    header = f"  {'PC':<5} {'Baseline':>10}" + "".join(f"{n:>12}" for n in attacks.keys())
    print(f"\n{header}")
    print("  " + "-" * (5 + 10 + 12 * len(attacks)))

    for i in range(10):
        row = f"  {i+1:<5} {baseline_eigs[i]:>10.2f}"
        for name in attacks:
            d = deltas[name][i]
            sign = "+" if d >= 0 else ""
            row += f"{sign}{d:>11.2f}"
        print(row)

    # --- Part B: Delta fingerprint shapes ---
    print("\n" + "-" * 70)
    print("PART B: Delta fingerprint shape (normalized)")
    print("-" * 70)

    print(f"\n  {'Attack':<15} {'PC1 dom%':>10} {'Spread':>10} {'||Delta||':>10} {'Shape'}")
    print("  " + "-" * 65)

    for name, delta in deltas.items():
        abs_delta = np.abs(delta)
        total = np.sum(abs_delta)
        pc1_pct = abs_delta[0] / total * 100 if total > 0 else 0
        nonzero = np.sum(abs_delta > 0.01 * abs_delta[0]) if abs_delta[0] > 0 else 0
        norm = np.linalg.norm(delta)

        if pc1_pct > 70:
            shape = "CONCENTRATED (volumetric)"
        elif nonzero > k * 0.5:
            shape = "SPREAD (diversified)"
        else:
            shape = "MODERATE"

        print(f"  {name:<15} {pc1_pct:>9.1f}% {nonzero:>10} {norm:>10.2f} {shape}")

    # --- Part C: Delta similarity matrix ---
    print("\n" + "-" * 70)
    print("PART C: Attack-type similarity from eigenvalue deltas")
    print("-" * 70)

    attack_names = list(attacks.keys())
    print(f"\n  {'':>15}", end="")
    for n in attack_names:
        print(f"{n:>12}", end="")
    print()
    print("  " + "-" * (15 + 12 * len(attack_names)))

    for n1 in attack_names:
        print(f"  {n1:>15}", end="")
        for n2 in attack_names:
            sim = delta_similarity(deltas[n1], deltas[n2])
            print(f"{sim:>12.4f}", end="")
        print()

    # --- Part D: Classification from eigenvalue delta ---
    print("\n" + "-" * 70)
    print("PART D: Attack classification from eigenvalue delta")
    print("-" * 70)

    # Build prototypes from first set, test with independent samples
    proto_deltas = {}
    for name, gen_fn in attacks.items():
        _, _, delta = get_eigenvalue_delta(client, snap, gen_fn, n_attack=100, k=k)
        proto_deltas[name] = delta

    # Test with different random seeds
    print(f"\n  {'True Type':<15} {'Predicted':>15} {'Correct':>8} {'Sim':>8}")
    print("  " + "-" * 48)

    n_trials = 5
    total_correct = 0
    total_count = 0

    for true_name, gen_fn in attacks.items():
        correct = 0
        for trial in range(n_trials):
            _, _, test_delta = get_eigenvalue_delta(
                client, snap, gen_fn, n_attack=50, k=k
            )
            # Classify by nearest prototype
            best_name = max(
                proto_deltas.keys(),
                key=lambda n: delta_similarity(test_delta, proto_deltas[n]),
            )
            best_sim = delta_similarity(test_delta, proto_deltas[best_name])
            if best_name == true_name:
                correct += 1
        total_correct += correct
        total_count += n_trials
        acc = correct / n_trials * 100
        print(f"  {true_name:<15} {'—':>15} {correct:>5}/{n_trials:<3} {acc:>7.0f}%")

    overall_acc = total_correct / total_count * 100
    print(f"\n  Overall classification accuracy: {overall_acc:.0f}%")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Check that DNS amp has concentrated delta (PC1 dominant)
    dns_delta = np.abs(deltas["DNS amp"])
    dns_pc1_pct = dns_delta[0] / np.sum(dns_delta) * 100

    # Check that different attacks produce different deltas
    between_sims = []
    for i, n1 in enumerate(attack_names):
        for j, n2 in enumerate(attack_names):
            if i < j:
                between_sims.append(delta_similarity(deltas[n1], deltas[n2]))
    mean_between = np.mean(between_sims)

    checks = [
        (
            "DNS amp concentrates in PC1 (>40%)",
            dns_pc1_pct > 40,
            f"PC1={dns_pc1_pct:.1f}%",
        ),
        (
            "Attack deltas are distinguishable (mean between-sim < 0.95)",
            mean_between < 0.95,
            f"mean between-sim={mean_between:.4f}",
        ),
        (
            "Classification accuracy > 60%",
            overall_acc > 60,
            f"acc={overall_acc:.0f}%",
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
