#!/usr/bin/env python3
"""
Engram Library Matching

HYPOTHESIS:
===========
An EngramLibrary storing 4 attack types can correctly match new instances
of each attack type to the right engram. The two-tier matching strategy
(eigenvalue pre-filter + full residual) produces the same correct result
as brute-force residual checking, but cheaper.

PRIMITIVES DEMONSTRATED:
========================
1. EngramLibrary.add()           - Mint engrams from trained subspaces
2. EngramLibrary.match()         - Two-tier matching
3. EngramLibrary.match_spectrum() - Eigenvalue-only pre-filter
4. EngramLibrary.save() / load()  - JSON persistence round-trip
5. Engram.residual()             - Per-engram residual check

SCENARIO:
=========
Train 4 attack subspaces, add to library. Generate fresh attack traffic
of each type, match against the library. Verify:
  - Correct engram is top match for each attack type
  - Normal traffic does not match any engram well
  - Persistence round-trip preserves matching behavior
"""

import sys
import tempfile
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

    print("=" * 70)
    print("EXPERIMENT 15: Engram Library Matching")
    print("=" * 70)

    n_train = 500
    n_test = 50

    # --- Part A: Build the engram library ---
    print(f"\nPart A: Training {len(ATTACK_TYPES)} attack subspaces and minting engrams...")

    library = client.create_engram_library()

    for name, factory in ATTACK_TYPES.items():
        sub = OnlineSubspace(dim=4096, k=32, amnesia=2.0, sigma_mult=3.5)
        for i in range(n_train):
            vec = client.encode(factory(np.random.default_rng(i + 1000)))
            sub.update(vec)
        engram = library.add(name, sub, rule=f"(({name}) => (drop))")
        print(f"  Minted: {engram}")

    print(f"\n  Library: {library}")

    # --- Part B: Match fresh attack traffic ---
    print("\n" + "-" * 70)
    print("PART B: Match fresh attack traffic against library")
    print("-" * 70)

    correct = 0
    total = 0
    per_type_results = {}

    for attack_name, factory in ATTACK_TYPES.items():
        type_correct = 0
        type_total = 0
        match_details = []

        for i in range(n_test):
            vec = client.encode(factory(np.random.default_rng(i + 3000)))
            matches = library.match(vec, top_k=3)
            top_name = matches[0][0] if matches else "none"
            top_score = matches[0][1] if matches else float("inf")

            if top_name == attack_name:
                type_correct += 1
                correct += 1
            type_total += 1
            total += 1
            match_details.append((top_name, top_score))

        per_type_results[attack_name] = (type_correct, type_total, match_details)
        acc = type_correct / type_total * 100
        mean_score = np.mean([d[1] for d in match_details])
        print(f"  {attack_name:15s}: {type_correct}/{type_total} correct ({acc:.0f}%), "
              f"mean_residual={mean_score:.2f}")

    overall_acc = correct / total * 100
    print(f"\n  Overall accuracy: {correct}/{total} ({overall_acc:.0f}%)")

    # --- Part C: Normal traffic rejection ---
    print("\n" + "-" * 70)
    print("PART C: Normal traffic should NOT match any engram well")
    print("-" * 70)

    normal_residuals = {}
    for i in range(n_test):
        vec = client.encode(make_normal(np.random.default_rng(i + 5000)))
        matches = library.match(vec, top_k=4)
        for name, res in matches:
            normal_residuals.setdefault(name, []).append(res)

    print(f"\n  {'Engram':>15s} {'Mean normal res':>16s} {'Mean attack res':>16s} {'Ratio':>8s}")
    print(f"  {'-' * 58}")
    for name in ATTACK_TYPES:
        normal_mean = np.mean(normal_residuals.get(name, [0]))
        attack_mean = np.mean([d[1] for d in per_type_results[name][2]])
        ratio = normal_mean / max(attack_mean, 1e-10)
        print(f"  {name:>15s} {normal_mean:>16.2f} {attack_mean:>16.2f} {ratio:>7.1f}x")

    # --- Part D: Spectrum pre-filter ---
    print("\n" + "-" * 70)
    print("PART D: Eigenvalue spectrum pre-filter accuracy")
    print("-" * 70)

    spectrum_correct = 0
    spectrum_total = 0

    for attack_name, factory in ATTACK_TYPES.items():
        sub_test = OnlineSubspace(dim=4096, k=32, amnesia=2.0)
        for i in range(100):
            vec = client.encode(factory(np.random.default_rng(i + 4000)))
            sub_test.update(vec)

        matches = library.match_spectrum(sub_test.eigenvalues, top_k=1)
        top = matches[0][0] if matches else "none"
        cos = matches[0][1] if matches else 0.0
        hit = top == attack_name
        if hit:
            spectrum_correct += 1
        spectrum_total += 1
        print(f"  {attack_name:15s} -> top match: {top:15s} (cos={cos:.4f}) {'HIT' if hit else 'MISS'}")

    spectrum_acc = spectrum_correct / spectrum_total * 100
    print(f"\n  Spectrum pre-filter accuracy: {spectrum_correct}/{spectrum_total} ({spectrum_acc:.0f}%)")

    # --- Part E: Persistence round-trip ---
    print("\n" + "-" * 70)
    print("PART E: Save/load persistence round-trip")
    print("-" * 70)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        save_path = f.name

    library.save(save_path)
    file_size = Path(save_path).stat().st_size
    print(f"  Saved to {save_path} ({file_size:,} bytes)")

    loaded = EngramLibrary.load(save_path)
    print(f"  Loaded: {loaded}")

    persistence_ok = True
    for attack_name, factory in ATTACK_TYPES.items():
        vec = client.encode(factory(np.random.default_rng(9999)))
        orig_matches = library.match(vec, top_k=1)
        load_matches = loaded.match(vec, top_k=1)

        orig_name, orig_score = orig_matches[0]
        load_name, load_score = load_matches[0]

        match = orig_name == load_name and abs(orig_score - load_score) < 1e-6
        if not match:
            persistence_ok = False
        print(f"  {attack_name}: orig=({orig_name}, {orig_score:.4f}) "
              f"loaded=({load_name}, {load_score:.4f}) "
              f"{'OK' if match else 'MISMATCH'}")

    Path(save_path).unlink()

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Overall matching accuracy > 90%",
            overall_acc > 90,
            f"acc={overall_acc:.0f}%",
        ),
        (
            "Each attack type correctly matched > 80%",
            all(r[0] / r[1] > 0.8 for r in per_type_results.values()),
            ", ".join(f"{n}={r[0]}/{r[1]}" for n, r in per_type_results.items()),
        ),
        (
            "Normal traffic has higher residual than attack traffic for all engrams",
            all(
                np.mean(normal_residuals.get(n, [0])) >
                np.mean([d[1] for d in per_type_results[n][2]])
                for n in ATTACK_TYPES
            ),
            "normal residuals consistently higher",
        ),
        (
            "Spectrum pre-filter accuracy >= 75%",
            spectrum_acc >= 75,
            f"acc={spectrum_acc:.0f}%",
        ),
        (
            "Save/load round-trip preserves matching",
            persistence_ok,
            "all scores match",
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
