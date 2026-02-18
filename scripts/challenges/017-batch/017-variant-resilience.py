#!/usr/bin/env python3
"""
Variant Resilience — Matching Attack Variants to Stored Engrams

HYPOTHESIS:
===========
When an attack returns with different parameters (different source IPs,
different target ports, different TTLs), the stored engram should still
match — because the subspace captures the attack's STRUCTURE rather than
specific values.

This is critical for real-world utility: attackers don't replay attacks
identically. If engrams only match exact replays, they're useless.

PRIMITIVES DEMONSTRATED:
========================
1. EngramLibrary.match()      - Match variants to original engrams
2. OnlineSubspace.residual()  - Structural similarity despite value changes
3. Engram.residual()          - Per-engram variant checking

SCENARIO:
=========
Train engrams from "original" attacks, then generate 5 variants of each:
  Variant 1: Different source IPs
  Variant 2: Different target IP
  Variant 3: Different TTLs
  Variant 4: Different source IPs + different TTLs (multi-field change)
  Variant 5: Significantly different (stress test)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient
from holon.engram import EngramLibrary
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


# --- Original attack factories ---

def make_dns_amp_original(rng):
    return {
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "245",
    }


def make_syn_flood_original(rng):
    return {
        "src_ip": f"172.16.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.1",
        "proto": "TCP",
        "dst_port": "80",
        "path": "syn",
        "status": "000",
        "ttl": str(rng.integers(30, 60)),
    }


# --- Variant factories ---

def make_dns_amp_v1(rng):
    """Different source IPs (different resolvers)."""
    return {
        "src_ip": str(rng.choice(["208.67.222.222", "208.67.220.220", "4.2.2.1"])),
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "245",
    }


def make_dns_amp_v2(rng):
    """Different target IP."""
    return {
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
        "dst_ip": "192.168.1.200",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "245",
    }


def make_dns_amp_v3(rng):
    """Different TTL."""
    return {
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "230",
    }


def make_dns_amp_v4(rng):
    """Different source IPs AND different TTL."""
    return {
        "src_ip": str(rng.choice(["208.67.222.222", "4.2.2.1", "77.88.8.8"])),
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "220",
    }


def make_dns_amp_v5(rng):
    """Stress test: different IPs, target, TTL."""
    return {
        "src_ip": str(rng.choice(["77.88.8.8", "156.154.70.1", "198.101.242.72"])),
        "dst_ip": "192.168.1.250",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "200",
    }


def make_syn_flood_v1(rng):
    """Different source subnet."""
    return {
        "src_ip": f"10.99.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.1",
        "proto": "TCP",
        "dst_port": "80",
        "path": "syn",
        "status": "000",
        "ttl": str(rng.integers(30, 60)),
    }


def make_syn_flood_v2(rng):
    """Different target IP."""
    return {
        "src_ip": f"172.16.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.50",
        "proto": "TCP",
        "dst_port": "80",
        "path": "syn",
        "status": "000",
        "ttl": str(rng.integers(30, 60)),
    }


def make_syn_flood_v3(rng):
    """Different target port."""
    return {
        "src_ip": f"172.16.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.1",
        "proto": "TCP",
        "dst_port": "443",
        "path": "syn",
        "status": "000",
        "ttl": str(rng.integers(30, 60)),
    }


def make_syn_flood_v4(rng):
    """Different subnet + different port."""
    return {
        "src_ip": f"10.99.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.1",
        "proto": "TCP",
        "dst_port": "8080",
        "path": "syn",
        "status": "000",
        "ttl": str(rng.integers(30, 60)),
    }


def make_syn_flood_v5(rng):
    """Stress: different subnet, target, port, higher TTL."""
    return {
        "src_ip": f"10.99.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.99",
        "proto": "TCP",
        "dst_port": "8443",
        "path": "syn",
        "status": "000",
        "ttl": str(rng.integers(80, 120)),
    }


ORIGINALS = {
    "dns_amp": make_dns_amp_original,
    "syn_flood": make_syn_flood_original,
}

VARIANTS = {
    "dns_amp": {
        "v1_diff_src": make_dns_amp_v1,
        "v2_diff_dst": make_dns_amp_v2,
        "v3_diff_ttl": make_dns_amp_v3,
        "v4_multi_field": make_dns_amp_v4,
        "v5_stress": make_dns_amp_v5,
    },
    "syn_flood": {
        "v1_diff_src": make_syn_flood_v1,
        "v2_diff_dst": make_syn_flood_v2,
        "v3_diff_port": make_syn_flood_v3,
        "v4_multi_field": make_syn_flood_v4,
        "v5_stress": make_syn_flood_v5,
    },
}


def main():
    client = HolonClient(dimensions=4096)

    print("=" * 70)
    print("EXPERIMENT 17: Variant Resilience")
    print("=" * 70)

    n_train = 500
    n_test = 50

    # --- Build library from originals ---
    print(f"\nBuilding engram library from original attacks ({n_train} vectors each)...")
    library = client.create_engram_library()

    for name, factory in ORIGINALS.items():
        sub = OnlineSubspace(dim=4096, k=32, amnesia=2.0, sigma_mult=3.5)
        for i in range(n_train):
            sub.update(client.encode(factory(np.random.default_rng(i + 1000))))
        library.add(name, sub)
        print(f"  Minted: {name} (threshold={sub.threshold:.2f})")

    # --- Part A: Baseline — original attacks match perfectly ---
    print("\n" + "-" * 70)
    print("PART A: Baseline — original attacks")
    print("-" * 70)

    baseline_acc = {}
    for name, factory in ORIGINALS.items():
        correct = 0
        residuals = []
        for i in range(n_test):
            vec = client.encode(factory(np.random.default_rng(i + 3000)))
            matches = library.match(vec, top_k=2)
            top_name, top_res = matches[0]
            residuals.append(top_res)
            if top_name == name:
                correct += 1
        acc = correct / n_test * 100
        baseline_acc[name] = acc
        print(f"  {name:15s}: {correct}/{n_test} ({acc:.0f}%), "
              f"mean_res={np.mean(residuals):.2f}")

    # --- Part B: Variant matching ---
    print("\n" + "-" * 70)
    print("PART B: Variant matching — does the engram still recognize variants?")
    print("-" * 70)

    variant_results = {}
    for attack_name in ORIGINALS:
        print(f"\n  {attack_name} variants:")
        variant_results[attack_name] = {}
        for var_name, var_factory in VARIANTS[attack_name].items():
            correct = 0
            residuals = []
            for i in range(n_test):
                vec = client.encode(var_factory(np.random.default_rng(i + 5000)))
                matches = library.match(vec, top_k=2)
                top_name, top_res = matches[0]
                residuals.append(top_res)
                if top_name == attack_name:
                    correct += 1
            acc = correct / n_test * 100
            mean_res = np.mean(residuals)
            variant_results[attack_name][var_name] = {
                "acc": acc, "mean_res": mean_res, "correct": correct,
            }
            print(f"    {var_name:20s}: {correct}/{n_test} ({acc:>5.0f}%), "
                  f"mean_res={mean_res:.2f}")

    # --- Part C: Residual distribution comparison ---
    print("\n" + "-" * 70)
    print("PART C: Residual comparison — original vs variants vs normal")
    print("-" * 70)

    for attack_name in ORIGINALS:
        engram = library.get(attack_name)

        orig_res = [engram.residual(client.encode(
            ORIGINALS[attack_name](np.random.default_rng(i + 3000))))
            for i in range(n_test)]

        normal_res = [engram.residual(client.encode(
            make_normal(np.random.default_rng(i + 7000))))
            for i in range(n_test)]

        print(f"\n  {attack_name}:")
        print(f"    Original:  mean={np.mean(orig_res):>8.2f}, "
              f"std={np.std(orig_res):>6.2f}")

        for var_name, var_factory in VARIANTS[attack_name].items():
            var_res = [engram.residual(client.encode(
                var_factory(np.random.default_rng(i + 6000))))
                for i in range(n_test)]
            print(f"    {var_name:20s}: mean={np.mean(var_res):>8.2f}, "
                  f"std={np.std(var_res):>6.2f}")

        print(f"    Normal traffic:   mean={np.mean(normal_res):>8.2f}, "
              f"std={np.std(normal_res):>6.2f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = []

    for attack_name in ORIGINALS:
        checks.append((
            f"{attack_name} baseline matches 100%",
            baseline_acc[attack_name] == 100,
            f"acc={baseline_acc[attack_name]:.0f}%",
        ))

    for attack_name in ORIGINALS:
        for var_name, var_data in variant_results[attack_name].items():
            checks.append((
                f"{attack_name}/{var_name} matches correctly (>80%)",
                var_data["acc"] > 80,
                f"acc={var_data['acc']:.0f}%",
            ))

    # All variants should have lower residual than normal traffic
    for attack_name in ORIGINALS:
        engram = library.get(attack_name)
        normal_mean = np.mean([engram.residual(client.encode(
            make_normal(np.random.default_rng(i + 7000))))
            for i in range(20)])
        worst_var = max(
            variant_results[attack_name].values(),
            key=lambda v: v["mean_res"]
        )
        checks.append((
            f"{attack_name} worst variant closer than normal",
            worst_var["mean_res"] < normal_mean,
            f"worst_var={worst_var['mean_res']:.2f} < normal={normal_mean:.2f}",
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
