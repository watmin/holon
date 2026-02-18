#!/usr/bin/env python3
"""
Eager Activation — Few-Shot Engram Matching

HYPOTHESIS:
===========
When an attack starts, we can confidently match it to a stored engram
using very few anomalous packets (target: 5-10). This enables "eager
activation" — deploying a stored mitigation rule almost instantly rather
than waiting to build a full attack subspace from scratch.

The key insight: each individual attack packet should have low residual
against the correct engram's subspace, so we can aggregate confidence
from just a handful of packets via majority vote or mean residual.

PRIMITIVES DEMONSTRATED:
========================
1. EngramLibrary.match()   - Per-packet matching
2. Majority vote           - Aggregate N packet-level matches
3. Mean residual           - Confidence measure over a window

SCENARIO:
=========
Build library from 4 attack types. Simulate an attack starting, feed
packets one at a time, measure how many packets are needed before we
can confidently identify the correct engram. Sweep N=1,2,3,5,10,20,50.
"""

import sys
from collections import Counter
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


def majority_vote(matches_history):
    """Return the most frequently top-matched engram name."""
    votes = [m[0][0] for m in matches_history if m]
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def mean_residual_by_engram(matches_history):
    """Compute mean residual per engram across all packets in the window."""
    totals = {}
    counts = {}
    for matches in matches_history:
        for name, res in matches:
            totals[name] = totals.get(name, 0.0) + res
            counts[name] = counts.get(name, 0) + 1
    return {n: totals[n] / counts[n] for n in totals}


def main():
    client = HolonClient(dimensions=4096)

    print("=" * 70)
    print("EXPERIMENT 16: Eager Activation — Few-Shot Engram Matching")
    print("=" * 70)

    n_train = 500
    window_sizes = [1, 2, 3, 5, 10, 20, 50]
    n_trials = 20

    # --- Build library ---
    print(f"\nBuilding engram library ({len(ATTACK_TYPES)} types, {n_train} training vectors each)...")
    library = client.create_engram_library()

    for name, factory in ATTACK_TYPES.items():
        sub = OnlineSubspace(dim=4096, k=32, amnesia=2.0, sigma_mult=3.5)
        for i in range(n_train):
            sub.update(client.encode(factory(np.random.default_rng(i + 1000))))
        library.add(name, sub)
    print(f"  {library}")

    # --- Part A: Sweep window sizes ---
    print("\n" + "-" * 70)
    print("PART A: Accuracy vs. window size (majority vote)")
    print("-" * 70)

    results = {}  # {attack_name: {window_size: accuracy}}
    overall_by_window = {}

    for attack_name, factory in ATTACK_TYPES.items():
        results[attack_name] = {}
        for ws in window_sizes:
            correct = 0
            for trial in range(n_trials):
                # Generate ws fresh attack packets
                matches_history = []
                for j in range(ws):
                    seed = trial * 1000 + j + 7000
                    vec = client.encode(factory(np.random.default_rng(seed)))
                    matches = library.match(vec, top_k=4)
                    matches_history.append(matches)

                vote = majority_vote(matches_history)
                if vote == attack_name:
                    correct += 1

            acc = correct / n_trials * 100
            results[attack_name][ws] = acc
            overall_by_window.setdefault(ws, []).append(acc)

    # Summary table
    header = f"  {'Window':>8s}"
    for name in ATTACK_TYPES:
        header += f" {name:>12s}"
    header += f" {'Overall':>10s}"
    print(f"\n{header}")
    print(f"  {'-' * (8 + 13 * len(ATTACK_TYPES) + 11)}")

    for ws in window_sizes:
        line = f"  {ws:>8d}"
        for name in ATTACK_TYPES:
            line += f" {results[name][ws]:>11.0f}%"
        overall = np.mean(overall_by_window[ws])
        line += f" {overall:>9.0f}%"
        print(line)

    # --- Part B: Confidence measure (mean residual gap) ---
    print("\n" + "-" * 70)
    print("PART B: Confidence gap (best vs second-best mean residual)")
    print("-" * 70)

    for attack_name in ATTACK_TYPES:
        factory = ATTACK_TYPES[attack_name]
        print(f"\n  {attack_name}:")
        for ws in [1, 5, 10]:
            gaps = []
            for trial in range(n_trials):
                history = []
                for j in range(ws):
                    seed = trial * 2000 + j + 8000
                    vec = client.encode(factory(np.random.default_rng(seed)))
                    history.append(library.match(vec, top_k=4))

                mean_res = mean_residual_by_engram(history)
                sorted_res = sorted(mean_res.items(), key=lambda x: x[1])
                if len(sorted_res) >= 2:
                    best_name, best_res = sorted_res[0]
                    second_res = sorted_res[1][1]
                    gap = second_res - best_res
                    gaps.append(gap)

            mean_gap = np.mean(gaps) if gaps else 0
            std_gap = np.std(gaps) if gaps else 0
            print(f"    N={ws:2d}: mean_gap={mean_gap:>7.2f} +/- {std_gap:.2f}")

    # --- Part C: False activation on normal traffic ---
    print("\n" + "-" * 70)
    print("PART C: False activation — normal traffic window matching")
    print("-" * 70)

    false_activations = {ws: 0 for ws in window_sizes}
    for ws in window_sizes:
        for trial in range(n_trials):
            history = []
            for j in range(ws):
                seed = trial * 3000 + j + 9000
                vec = client.encode(make_normal(np.random.default_rng(seed)))
                history.append(library.match(vec, top_k=4))

            mean_res = mean_residual_by_engram(history)
            if mean_res:
                best_name, best_res = min(mean_res.items(), key=lambda x: x[1])
                # Check if best residual is low enough to be a "match"
                engram = library.get(best_name)
                if best_res < engram.subspace.threshold:
                    false_activations[ws] += 1

    print(f"\n  {'Window':>8s} {'False activations':>20s}")
    print(f"  {'-' * 30}")
    for ws in window_sizes:
        print(f"  {ws:>8d} {false_activations[ws]:>16d}/{n_trials}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    single_acc = np.mean(overall_by_window[1])
    five_acc = np.mean(overall_by_window[5])
    ten_acc = np.mean(overall_by_window[10])

    checks = [
        (
            "Single-packet matching > 80%",
            single_acc > 80,
            f"acc={single_acc:.0f}%",
        ),
        (
            "5-packet window matching > 90%",
            five_acc > 90,
            f"acc={five_acc:.0f}%",
        ),
        (
            "10-packet window matching > 95%",
            ten_acc > 95,
            f"acc={ten_acc:.0f}%",
        ),
        (
            "Accuracy improves with window size",
            five_acc >= single_acc,
            f"N=1: {single_acc:.0f}%, N=5: {five_acc:.0f}%",
        ),
        (
            "No false activations on normal traffic (N=10)",
            false_activations[10] == 0,
            f"false_acts={false_activations[10]}/{n_trials}",
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
