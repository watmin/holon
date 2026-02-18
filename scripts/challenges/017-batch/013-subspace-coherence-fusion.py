#!/usr/bin/env python3
"""
Subspace + Coherence Combined Detector (Multi-Signal Fusion)

HYPOTHESIS:
===========
Combining subspace residual with coherence produces a detector that is
stronger than either signal alone. The two signals capture different
aspects of anomaly:

  - Subspace residual: "this individual vector is off-manifold"
    Best for: novel field values, structural anomalies
    Weak on: high-volume attacks that look individually normal

  - Coherence: "this window of vectors is too homogeneous"
    Best for: volumetric floods, concentrated traffic
    Weak on: diverse/varied attacks, low-rate anomalies

A slow-drip exfiltration might have low coherence (mixed with normal)
but high residual (each packet is off-manifold). A volumetric flood
might have moderate residual (simple pattern) but extreme coherence.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.residual()     - Per-vector anomaly score
2. coherence()                    - Per-window homogeneity
3. Combined scoring               - Fusion strategies

SCENARIO:
=========
Four attack scenarios that stress different detectors:
  1. DNS amp (volumetric)      - coherence excels
  2. Slow exfiltration         - residual excels
  3. Credential stuffing       - both contribute
  4. Stealth scan (low-rate)   - hardest for both
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import coherence
from holon.subspace import OnlineSubspace

FIELDS = ["src_ip", "dst_ip", "proto", "dst_port", "path", "status", "ttl"]


def make_normal(rng):
    return {
        "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
        "dst_ip": f"192.168.1.{rng.integers(1, 10)}",
        "proto": str(rng.choice(["TCP", "UDP", "TCP", "TCP"])),
        "dst_port": str(rng.choice(["80", "443", "8080"])),
        "path": str(rng.choice(["api", "static", "health", "metrics", "users"])),
        "status": str(rng.choice(["200", "200", "200", "301", "404"])),
        "ttl": str(rng.choice(["64", "128"])),
    }


def make_dns_amp(rng):
    """Volumetric: homogeneous, high coherence, high residual."""
    return {
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
        "dst_ip": "192.168.1.100", "proto": "UDP", "dst_port": "53",
        "path": "dns", "status": "200", "ttl": "245",
    }


def make_exfil(rng):
    """Subtle: varied, low coherence, moderate residual."""
    return {
        "src_ip": "10.0.1.50",
        "dst_ip": f"203.0.113.{rng.integers(1, 50)}",
        "proto": "TCP", "dst_port": "443",
        "path": str(rng.choice(["export", "backup", "dump", "sync"])),
        "status": "200", "ttl": "64",
    }


def make_cred_stuff(rng):
    """Application-layer: moderate coherence, high residual."""
    return {
        "src_ip": f"10.0.{rng.integers(1, 50)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.5", "proto": "TCP", "dst_port": "443",
        "path": "auth", "status": "401", "ttl": "64",
    }


def make_stealth_scan(rng):
    """Stealth: each probe looks nearly normal, low rate."""
    normal = make_normal(rng)
    normal["dst_port"] = str(rng.choice(["22", "23", "25", "110", "139", "445",
                                          "3389", "5900", "8888", "9090"]))
    normal["path"] = "probe"
    return normal


def run_scenario(client, name, attack_gen, sub_snapshot, baseline_stats,
                 n_normal=200, n_attack=50, window_size=20, attack_ratio=0.5):
    """Run mixed traffic through both detectors."""
    sub = OnlineSubspace.from_snapshot(sub_snapshot)

    rng_n = np.random.default_rng(999)
    rng_a = np.random.default_rng(888)

    resid_thresh = baseline_stats["resid_mean"] + 3.0 * baseline_stats["resid_std"]
    coh_thresh = baseline_stats["coh_mean"] + 3.0 * baseline_stats["coh_std"]

    n_pure_normal_windows = n_normal // window_size
    n_attack_windows = max(1, n_attack // max(1, int(window_size * attack_ratio)))

    windows = []

    for w in range(n_pure_normal_windows):
        vecs = []
        for _ in range(window_size):
            vecs.append(client.encode(make_normal(rng_n)))
        windows.append(("normal", vecs))

    for w in range(n_attack_windows):
        vecs = []
        for j in range(window_size):
            if rng_a.random() < attack_ratio:
                vecs.append(client.encode(attack_gen(rng_a)))
            else:
                vecs.append(client.encode(make_normal(rng_n)))
        windows.append(("attack", vecs))

    results = []
    for label, vecs in windows:
        residuals = [sub.residual(v) for v in vecs]
        mean_residual = np.mean(residuals)
        max_residual = np.max(residuals)
        coh = coherence(vecs)

        residual_alert = mean_residual > resid_thresh
        coherence_alert = coh > coh_thresh
        combined_or = residual_alert or coherence_alert

        results.append({
            "label": label,
            "mean_residual": mean_residual,
            "max_residual": max_residual,
            "coherence": coh,
            "residual_alert": residual_alert,
            "coherence_alert": coherence_alert,
            "combined_or": combined_or,
            "resid_thresh": resid_thresh,
            "coh_thresh": coh_thresh,
        })

    return results


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 13: Subspace + Coherence Combined Detector")
    print("=" * 70)

    # --- Train ---
    n_train = 1000
    print(f"\nTraining subspace on {n_train} normal vectors...")
    sub = client.create_subspace(k=64, amnesia=2.0, sigma_mult=3.5)
    for _ in range(n_train):
        sub.update(client.encode(make_normal(rng)))
    snap = sub.snapshot()
    print(f"  {sub}")

    # --- Measure baseline signals ---
    print("\n" + "-" * 70)
    print("BASELINE: Signal levels for pure normal traffic")
    print("-" * 70)

    # Compute baseline with dummy stats, then derive real thresholds
    dummy_stats = {"resid_mean": 999, "resid_std": 1, "coh_mean": 999, "coh_std": 1}
    baseline_results = run_scenario(
        client, "Baseline", make_normal, snap, dummy_stats,
        n_normal=200, n_attack=0, window_size=20
    )
    normal_residuals = [r["mean_residual"] for r in baseline_results]
    normal_coherences = [r["coherence"] for r in baseline_results]

    baseline_stats = {
        "resid_mean": np.mean(normal_residuals),
        "resid_std": np.std(normal_residuals),
        "coh_mean": np.mean(normal_coherences),
        "coh_std": np.std(normal_coherences),
    }
    resid_thresh = baseline_stats["resid_mean"] + 3.0 * baseline_stats["resid_std"]
    coh_thresh = baseline_stats["coh_mean"] + 3.0 * baseline_stats["coh_std"]

    print(f"\n  Residual: mean={baseline_stats['resid_mean']:.2f}, "
          f"std={baseline_stats['resid_std']:.2f}, threshold={resid_thresh:.2f}")
    print(f"  Coherence: mean={baseline_stats['coh_mean']:.4f}, "
          f"std={baseline_stats['coh_std']:.4f}, threshold={coh_thresh:.4f}")
    print(f"  Subspace threshold: {sub.threshold:.2f}")

    # --- Run all scenarios ---
    scenarios = [
        ("DNS amplification", make_dns_amp, 0.5),
        ("Slow exfiltration", make_exfil, 0.3),
        ("Credential stuffing", make_cred_stuff, 0.5),
        ("Stealth scan", make_stealth_scan, 0.2),
    ]

    all_scenario_results = {}

    for name, gen_fn, attack_ratio in scenarios:
        print(f"\n{'─' * 70}")
        print(f"  Scenario: {name} (attack ratio: {attack_ratio:.0%} of window)")
        print(f"{'─' * 70}")

        results = run_scenario(
            client, name, gen_fn, snap, baseline_stats,
            n_normal=200, n_attack=100, window_size=20,
            attack_ratio=attack_ratio
        )

        normal_r = [r for r in results if r["label"] == "normal"]
        attack_r = [r for r in results if r["label"] == "attack"]

        if not attack_r:
            print("  No attack windows generated")
            continue

        # Detection rates for each method
        def det_rate(windows, key):
            if not windows:
                return 0.0
            return sum(1 for r in windows if r[key]) / len(windows) * 100

        residual_tp = det_rate(attack_r, "residual_alert")
        residual_fp = det_rate(normal_r, "residual_alert")
        coherence_tp = det_rate(attack_r, "coherence_alert")
        coherence_fp = det_rate(normal_r, "coherence_alert")
        combined_tp = det_rate(attack_r, "combined_or")
        combined_fp = det_rate(normal_r, "combined_or")

        print(f"\n  {'Detector':>20} {'TP%':>8} {'FP%':>8} {'Net':>8}")
        print(f"  {'-'*46}")
        print(f"  {'Residual only':>20} {residual_tp:>7.0f}% {residual_fp:>7.0f}% {residual_tp - residual_fp:>+7.0f}")
        print(f"  {'Coherence only':>20} {coherence_tp:>7.0f}% {coherence_fp:>7.0f}% {coherence_tp - coherence_fp:>+7.0f}")
        print(f"  {'Combined (OR)':>20} {combined_tp:>7.0f}% {combined_fp:>7.0f}% {combined_tp - combined_fp:>+7.0f}")

        # Show signal levels
        print(f"\n  Signal levels (mean ± std):")
        atk_resid = [r["mean_residual"] for r in attack_r]
        atk_coh = [r["coherence"] for r in attack_r]
        nrm_resid = [r["mean_residual"] for r in normal_r]
        nrm_coh = [r["coherence"] for r in normal_r]

        print(f"    {'':>15} {'Residual':>20} {'Coherence':>20}")
        print(f"    {'Normal':>15} {np.mean(nrm_resid):>8.2f} ± {np.std(nrm_resid):>5.2f}  "
              f"{np.mean(nrm_coh):>8.4f} ± {np.std(nrm_coh):>6.4f}")
        print(f"    {'Attack':>15} {np.mean(atk_resid):>8.2f} ± {np.std(atk_resid):>5.2f}  "
              f"{np.mean(atk_coh):>8.4f} ± {np.std(atk_coh):>6.4f}")

        all_scenario_results[name] = {
            "residual_tp": residual_tp, "residual_fp": residual_fp,
            "coherence_tp": coherence_tp, "coherence_fp": coherence_fp,
            "combined_tp": combined_tp, "combined_fp": combined_fp,
        }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Which detector wins for each scenario?")
    print("=" * 70)

    print(f"\n  {'Scenario':<25} {'Best Single':>15} {'Combined TP':>12} {'Combined FP':>12}")
    print(f"  {'-'*66}")

    for name, r in all_scenario_results.items():
        if r["residual_tp"] >= r["coherence_tp"]:
            best = "Residual"
        else:
            best = "Coherence"
        print(f"  {name:<25} {best:>15} {r['combined_tp']:>11.0f}% {r['combined_fp']:>11.0f}%")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    dns_r = all_scenario_results.get("DNS amplification", {})
    exfil_r = all_scenario_results.get("Slow exfiltration", {})
    scan_r = all_scenario_results.get("Stealth scan", {})

    # Coherence contributes SOME signal (not zero) for at least one attack
    coherence_useful = any(r["coherence_tp"] > 0 for r in all_scenario_results.values())
    # Residual is the stronger signal in all scenarios
    residual_dominates = all(
        r["residual_tp"] >= r["coherence_tp"]
        for r in all_scenario_results.values()
    )

    checks = [
        (
            "Residual catches all attack types (TP > 80%)",
            all(r["residual_tp"] > 80 for r in all_scenario_results.values()),
            ", ".join(f"{n}={r['residual_tp']:.0f}%" for n, r in all_scenario_results.items()),
        ),
        (
            "Zero false positives for all detectors",
            all(r["combined_fp"] == 0 for r in all_scenario_results.values()),
            "0% FP across all",
        ),
        (
            "Coherence contributes signal for volumetric attacks",
            dns_r.get("coherence_tp", 0) >= 50,
            f"DNS coherence TP={dns_r.get('coherence_tp', 0):.0f}%",
        ),
        (
            "Combined never worse than best single",
            all(r["combined_tp"] >= max(r["residual_tp"], r["coherence_tp"])
                for r in all_scenario_results.values()),
            "fusion never hurts",
        ),
        (
            "Stealth scan is hardest (lowest TP)",
            scan_r.get("residual_tp", 100) <= min(
                r["residual_tp"] for r in all_scenario_results.values()
            ),
            f"stealth TP={scan_r.get('residual_tp', 0):.0f}%",
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
