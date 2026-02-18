#!/usr/bin/env python3
"""
Real Traffic Intrinsic Dimensionality (19-Field Sidecar Encodings)

HYPOTHESIS:
===========
The DDoS sidecar encodes packets with up to 19 fields (vs 7 in previous
experiments). More fields = higher intrinsic dimensionality, meaning
k=32 might be insufficient. We need to measure the intrinsic dimensionality
of realistic multi-field encodings and find the optimal k.

Fields (from veth-lab sidecar/src/main.rs):
  Base (16): src_ip, dst_ip, src_port, dst_port, protocol, src_port_band,
             dst_port_band, direction, size_class, ttl, df_bit, ip_id,
             ip_len, dscp, ecn, mf_bit, frag_offset
  TCP (2):   tcp_flags, tcp_window

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace with varying k   - Find the elbow
2. eigenvalues                      - Spectrum shape
3. residual                         - How k affects detection quality

VECTOR PROPERTIES EXPLOITED:
============================
- Intrinsic dimensionality depends on field count AND value cardinality
- More fields means more binding cross-terms → potentially higher manifold dim
- Some fields are correlated (src_port_band derived from src_port) → may not
  add independent dimensions
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient
from holon.subspace import OnlineSubspace

SIDECAR_FIELDS_BASE = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "src_port_band", "dst_port_band", "direction", "size_class",
    "ttl", "df_bit", "ip_id", "ip_len", "dscp", "ecn",
    "mf_bit", "frag_offset",
]
SIDECAR_FIELDS_TCP = ["tcp_flags", "tcp_window"]
ALL_FIELDS = SIDECAR_FIELDS_BASE + SIDECAR_FIELDS_TCP


def make_tcp_packet(rng):
    """Simulate realistic TCP packet encoding matching sidecar fields."""
    src_port = int(rng.integers(1024, 65535))
    dst_port = int(rng.choice([80, 443, 8080, 8443, 22, 3306, 5432, 6379]))
    ip_len = int(rng.choice([40, 52, 60, 100, 200, 500, 1500]))
    ttl = int(rng.choice([64, 128, 255]))

    return {
        "src_ip": f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": f"192.168.{rng.integers(1, 10)}.{rng.integers(1, 255)}",
        "src_port": str(src_port),
        "dst_port": str(dst_port),
        "protocol": "6",
        "src_port_band": str(src_port // 1024),
        "dst_port_band": str(dst_port // 1024),
        "direction": str(rng.choice(["inbound", "outbound"])),
        "size_class": "small" if ip_len < 100 else ("medium" if ip_len < 500 else "large"),
        "ttl": str(ttl),
        "df_bit": str(rng.choice([0, 1, 1, 1])),
        "ip_id": str(int(rng.integers(0, 65535))),
        "ip_len": str(ip_len),
        "dscp": str(rng.choice([0, 0, 0, 8, 46])),
        "ecn": str(rng.choice([0, 0, 0, 1, 2])),
        "mf_bit": "0",
        "frag_offset": "0",
        "tcp_flags": str(rng.choice([2, 16, 18, 24, 25])),
        "tcp_window": str(int(rng.choice([8192, 16384, 29200, 65535]))),
    }


def make_udp_packet(rng):
    """Simulate realistic UDP packet (no TCP fields)."""
    src_port = int(rng.integers(1024, 65535))
    dst_port = int(rng.choice([53, 123, 161, 443, 500, 5353]))
    ip_len = int(rng.choice([28, 40, 60, 100, 512]))
    ttl = int(rng.choice([64, 128, 255]))

    d = {
        "src_ip": f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": f"192.168.{rng.integers(1, 10)}.{rng.integers(1, 255)}",
        "src_port": str(src_port),
        "dst_port": str(dst_port),
        "protocol": "17",
        "src_port_band": str(src_port // 1024),
        "dst_port_band": str(dst_port // 1024),
        "direction": str(rng.choice(["inbound", "outbound"])),
        "size_class": "small" if ip_len < 100 else ("medium" if ip_len < 500 else "large"),
        "ttl": str(ttl),
        "df_bit": str(rng.choice([0, 1])),
        "ip_id": str(int(rng.integers(0, 65535))),
        "ip_len": str(ip_len),
        "dscp": "0",
        "ecn": "0",
        "mf_bit": "0",
        "frag_offset": "0",
    }
    return d


def make_mixed_packet(rng):
    """70% TCP, 30% UDP (typical enterprise traffic)."""
    if rng.random() < 0.7:
        return make_tcp_packet(rng)
    else:
        return make_udp_packet(rng)


def make_dns_amp_sidecar(rng):
    """DNS amplification in sidecar field format."""
    return {
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9", "208.67.222.222"])),
        "dst_ip": "192.168.1.100",
        "src_port": "53",
        "dst_port": str(int(rng.integers(1024, 65535))),
        "protocol": "17",
        "src_port_band": "0",
        "dst_port_band": str(int(rng.integers(1024, 65535)) // 1024),
        "direction": "inbound",
        "size_class": "large",
        "ttl": "245",
        "df_bit": "0",
        "ip_id": str(int(rng.integers(0, 65535))),
        "ip_len": "1500",
        "dscp": "0",
        "ecn": "0",
        "mf_bit": "0",
        "frag_offset": "0",
    }


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 11: Intrinsic Dimensionality of 19-Field Sidecar Encodings")
    print("=" * 70)

    # --- Generate training data ---
    n_train = 2000
    print(f"\nGenerating {n_train} mixed TCP/UDP packets (19 fields max)...")
    train_dicts = [make_mixed_packet(np.random.default_rng(i)) for i in range(n_train)]
    train_vecs = [client.encode(d) for d in train_dicts]
    n_tcp = sum(1 for d in train_dicts if "tcp_flags" in d)
    n_udp = n_train - n_tcp
    print(f"  TCP: {n_tcp}, UDP: {n_udp}")

    # --- Part A: Sweep k values ---
    print("\n" + "-" * 70)
    print("PART A: Residual convergence for varying k")
    print("-" * 70)

    k_values = [8, 16, 32, 64, 96, 128, 192, 256]
    results = {}

    print(f"\n  {'k':>5} {'Final Resid':>12} {'CV%':>6} {'Active':>8} {'90% var at':>10} {'99% var at':>10}")
    print(f"  {'-'*55}")

    for k in k_values:
        sub = OnlineSubspace(dim=4096, k=k, amnesia=2.0)
        residuals = []
        for v in train_vecs:
            r = sub.update(v)
            residuals.append(r)

        last_100 = residuals[-100:]
        mean_r = np.mean(last_100)
        cv = np.std(last_100) / mean_r * 100

        eigs = np.sort(sub.eigenvalues)[::-1]
        cumvar = np.cumsum(eigs) / np.sum(eigs)
        k90 = int(np.searchsorted(cumvar, 0.90)) + 1
        k99 = int(np.searchsorted(cumvar, 0.99)) + 1
        active = int(np.sum(eigs > 1e-6))

        results[k] = {
            "mean_r": mean_r, "cv": cv, "active": active,
            "k90": k90, "k99": k99, "sub": sub, "eigs": eigs,
        }

        print(f"  {k:>5} {mean_r:>12.2f} {cv:>5.1f}% {active:>8} {k90:>10} {k99:>10}")

    # --- Part B: Eigenvalue spectrum for k=256 ---
    print("\n" + "-" * 70)
    print("PART B: Eigenvalue spectrum (k=256)")
    print("-" * 70)

    eigs = results[256]["eigs"]
    cumvar = np.cumsum(eigs) / np.sum(eigs)

    print(f"\n  {'PC':>5} {'Eigenvalue':>12} {'% Var':>8} {'Cumul %':>8}  {'Bar'}")
    print(f"  {'-'*55}")

    for i in range(min(40, len(eigs))):
        pct = eigs[i] / np.sum(eigs) * 100
        bar = "█" * int(pct * 2) if pct > 0.5 else "▏"
        print(f"  PC{i+1:>3} {eigs[i]:>12.2f} {pct:>7.1f}% {cumvar[i]*100:>7.1f}%  {bar}")
        if cumvar[i] > 0.99:
            print(f"  ... (remaining {len(eigs) - i - 1} components < 1% total)")
            break

    # Identify knee points
    k90 = results[256]["k90"]
    k95 = int(np.searchsorted(cumvar, 0.95)) + 1
    k99 = results[256]["k99"]
    print(f"\n  Knee points: 90% at k={k90}, 95% at k={k95}, 99% at k={k99}")
    print(f"  Intrinsic dimensionality ≈ {k90} (90% threshold)")

    # --- Part C: Compare 7-field vs 19-field ---
    print("\n" + "-" * 70)
    print("PART C: 7-field (experiments 001-009) vs 19-field (sidecar) encoding")
    print("-" * 70)

    sub_7 = OnlineSubspace(dim=4096, k=128, amnesia=2.0)
    for i in range(n_train):
        d = {
            "src_ip": f"10.0.{np.random.default_rng(i).integers(1, 50)}.{np.random.default_rng(i).integers(1, 255)}",
            "dst_ip": f"192.168.1.{np.random.default_rng(i).integers(1, 10)}",
            "proto": str(np.random.default_rng(i).choice(["TCP", "UDP", "TCP", "TCP"])),
            "dst_port": str(np.random.default_rng(i).choice(["80", "443", "8080"])),
            "path": str(np.random.default_rng(i).choice(["api", "static", "health", "metrics"])),
            "status": str(np.random.default_rng(i).choice(["200", "200", "301", "404"])),
            "ttl": str(np.random.default_rng(i).choice(["64", "128"])),
        }
        sub_7.update(client.encode(d))

    eigs_7 = np.sort(sub_7.eigenvalues)[::-1]
    cumvar_7 = np.cumsum(eigs_7) / np.sum(eigs_7)
    k90_7 = int(np.searchsorted(cumvar_7, 0.90)) + 1
    k99_7 = int(np.searchsorted(cumvar_7, 0.99)) + 1

    print(f"\n  {'Encoding':>15} {'Fields':>8} {'k@90%':>8} {'k@99%':>8} {'k=32 Resid':>12}")
    print(f"  {'-'*55}")
    print(f"  {'7-field':>15} {'7':>8} {k90_7:>8} {k99_7:>8} {results.get(32, {}).get('mean_r', 0):>12.2f}")
    print(f"  {'19-field':>15} {'19':>8} {k90:>8} {k99:>8} {results[32]['mean_r']:>12.2f}")

    # --- Part D: Detection quality with different k values ---
    print("\n" + "-" * 70)
    print("PART D: Does k choice affect detection quality?")
    print("-" * 70)

    n_attack = 100
    attack_dicts = [make_dns_amp_sidecar(np.random.default_rng(i + 500)) for i in range(n_attack)]
    attack_vecs = [client.encode(d) for d in attack_dicts]

    normal_holdout = [make_mixed_packet(np.random.default_rng(i + 5000)) for i in range(200)]
    normal_holdout_vecs = [client.encode(d) for d in normal_holdout]

    print(f"\n  {'k':>5} {'TP%':>8} {'FP%':>8} {'Separation':>12} {'Threshold':>10}")
    print(f"  {'-'*45}")

    for k in [16, 32, 64, 128, 256]:
        if k not in results:
            continue
        sub = results[k]["sub"]
        atk_r = [sub.residual(v) for v in attack_vecs]
        nrm_r = [sub.residual(v) for v in normal_holdout_vecs]
        tp = sum(1 for r in atk_r if r > sub.threshold) / n_attack * 100
        fp = sum(1 for r in nrm_r if r > sub.threshold) / len(normal_holdout) * 100
        sep = np.mean(atk_r) / np.mean(nrm_r) if np.mean(nrm_r) > 0 else 0

        print(f"  {k:>5} {tp:>7.0f}% {fp:>7.0f}% {sep:>11.2f}× {sub.threshold:>10.2f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            f"19-field intrinsic dim > 7-field (more fields = wider manifold)",
            k90 > k90_7,
            f"19-field: k@90%={k90}, 7-field: k@90%={k90_7}",
        ),
        (
            "k=32 still achieves >90% TP on sidecar encodings",
            sum(1 for r in [results[32]["sub"].residual(v) for v in attack_vecs] if r > results[32]["sub"].threshold) / n_attack > 0.90,
            f"TP with k=32",
        ),
        (
            "Diminishing returns: k=128 residual within 5% of k=256",
            abs(results[128]["mean_r"] - results[256]["mean_r"]) / results[256]["mean_r"] < 0.05,
            f"k128={results[128]['mean_r']:.2f} vs k256={results[256]['mean_r']:.2f}",
        ),
        (
            "Spectrum has clear knee (90% variance in < 50% of components)",
            k90 < 128,
            f"k@90%={k90} out of 256",
        ),
    ]

    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc} ({detail})")

    print(f"\n  RECOMMENDATION: Use k={max(k90, 32)} for sidecar deployment")
    print(f"  (90% variance threshold = {k90}, minimum practical = 32)")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
