#!/usr/bin/env python3
"""
Subspace vs Centroid — Non-Radial Anomaly Detection

HYPOTHESIS:
===========
A subspace detector catches anomalies that cosine-to-centroid misses.
When normal traffic is multi-modal (e.g., API calls AND web browsing),
the centroid sits between the modes. An anomaly equidistant from the
centroid but off-manifold is invisible to cosine drift, but the subspace
sees it clearly because it's NOT in any direction the normal data spans.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.residual()   - Detects off-manifold vectors
2. OnlineSubspace.project()    - Shows where vectors land in subspace
3. prototype() / bundle()      - Centroid-based detection for comparison

SCENARIO:
=========
Two clusters of normal traffic:
  Cluster A: Internal API calls (10.x IPs, TCP/443, /api/*, 200)
  Cluster B: External web traffic (various IPs, TCP/80, /static/*, 200/301)

Anomaly: SSH brute force (external IP, TCP/22, auth path, 401)
The SSH anomaly may have similar overall distance from the centroid
as normal data, but it lies in a direction neither cluster ever used.

VECTOR PROPERTIES EXPLOITED:
============================
- Multi-modal distributions have centroids in "no man's land"
- Subspace captures the SPAN of modes, not just the center
- Off-manifold residual is independent of centroid distance
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import prototype
from holon.subspace import OnlineSubspace


def encode_api_traffic(client, rng, count=200):
    """Cluster A: Internal API calls."""
    vecs = []
    endpoints = ["users", "orders", "products", "inventory", "billing"]
    for _ in range(count):
        vecs.append(
            client.encode(
                {
                    "src_ip": f"10.0.{rng.integers(1, 10)}.{rng.integers(1, 255)}",
                    "dst_ip": "192.168.1.5",
                    "proto": "TCP",
                    "dst_port": "443",
                    "path": str(rng.choice(endpoints)),
                    "method": str(rng.choice(["GET", "POST", "GET"])),
                    "status": "200",
                    "agent": "script",
                }
            )
        )
    return vecs


def encode_web_traffic(client, rng, count=200):
    """Cluster B: External web browsing."""
    vecs = []
    pages = ["index", "about", "docs", "blog", "faq"]
    for _ in range(count):
        vecs.append(
            client.encode(
                {
                    "src_ip": f"203.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                    "dst_ip": "192.168.1.10",
                    "proto": "TCP",
                    "dst_port": "80",
                    "path": str(rng.choice(pages)),
                    "method": "GET",
                    "status": str(rng.choice(["200", "200", "301"])),
                    "agent": "browser",
                }
            )
        )
    return vecs


def encode_ssh_brute_force(client, rng, count=100):
    """Anomaly: SSH brute force attempts."""
    vecs = []
    for _ in range(count):
        vecs.append(
            client.encode(
                {
                    "src_ip": f"45.33.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                    "dst_ip": "192.168.1.5",
                    "proto": "TCP",
                    "dst_port": "22",
                    "path": "auth",
                    "method": "CONNECT",
                    "status": "401",
                    "agent": "script",
                }
            )
        )
    return vecs


def encode_chimera(client, rng, count=100):
    """Craft chimera vectors: mix API and Web field values in unnatural combos.

    These vectors use familiar field values from both clusters but combine
    them in ways that never appear in normal traffic. They should be close
    to the centroid (which sits between the two clusters) but not on the
    manifold that either cluster spans.
    """
    vecs = []
    for _ in range(count):
        # Mix API cluster IPs with Web cluster ports/paths and vice versa
        vecs.append(
            client.encode(
                {
                    # API-like internal IP...
                    "src_ip": f"10.0.{rng.integers(1, 10)}.{rng.integers(1, 255)}",
                    # ...but Web-like destination
                    "dst_ip": "192.168.1.10",
                    "proto": "TCP",
                    # Web port...
                    "dst_port": "80",
                    # ...but API-like endpoint
                    "path": str(rng.choice(["users", "orders", "inventory"])),
                    # API-like method...
                    "method": str(rng.choice(["POST", "PUT", "DELETE"])),
                    # ...but Web-like status
                    "status": str(rng.choice(["301", "301", "404"])),
                    # API agent on Web port
                    "agent": "script",
                }
            )
        )
    return vecs


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 3: Subspace vs Centroid — Non-Radial Anomalies")
    print("=" * 70)

    # --- Generate multi-modal normal data ---
    print("\nGenerating multi-modal normal traffic...")
    api_vecs = encode_api_traffic(client, rng, count=400)
    web_vecs = encode_web_traffic(client, rng, count=400)
    all_normal = api_vecs + web_vecs

    # Shuffle to simulate interleaved arrival
    rng_shuffle = np.random.default_rng(42)
    indices = rng_shuffle.permutation(len(all_normal))
    all_normal = [all_normal[i] for i in indices]

    print(f"  Cluster A (API):  {len(api_vecs)} vectors")
    print(f"  Cluster B (Web):  {len(web_vecs)} vectors")
    print(f"  Total normal:     {len(all_normal)} vectors")

    # --- Build centroid ---
    centroid = prototype(all_normal)

    # --- Generate anomalies ---
    print("\nGenerating anomalies...")
    rng_anom = np.random.default_rng(200)
    ssh_vecs = encode_ssh_brute_force(client, rng_anom, count=100)
    chimera_vecs = encode_chimera(client, rng_anom, count=100)

    # --- Train subspace ---
    print("\n" + "-" * 70)
    print("PART A: Training subspace on multi-modal normal traffic")
    print("-" * 70)

    sub = OnlineSubspace(dim=4096, k=64, amnesia=2.0, sigma_mult=3.5)
    for v in all_normal:
        sub.update(v)

    print(f"\n  {sub}")
    print(f"  Threshold: {sub.threshold:.4f}")

    # --- Score all traffic types ---
    print("\n" + "-" * 70)
    print("PART B: Centroid distance vs subspace residual")
    print("-" * 70)

    test_sets = [
        ("API (normal)", api_vecs[:100]),
        ("Web (normal)", web_vecs[:100]),
        ("SSH brute force", ssh_vecs[:100]),
        ("Chimera (mixed)", chimera_vecs[:100]),
    ]

    threshold = sub.threshold

    print(f"\n  {'Traffic Type':<22} {'Cos→Centroid':>13} {'1-Cos':>8} {'Subspace Res':>13} {'Above Thr':>10}")
    print("  " + "-" * 68)

    cos_scores = {}
    sub_scores = {}

    for label, vecs in test_sets:
        cos_sims = np.array([cosine_similarity(v, centroid) for v in vecs])
        residuals = np.array([sub.residual(v) for v in vecs])
        above = np.sum(residuals > threshold)

        cos_scores[label] = cos_sims
        sub_scores[label] = residuals

        print(
            f"  {label:<22} {np.mean(cos_sims):>13.4f} {1 - np.mean(cos_sims):>8.4f} "
            f"{np.mean(residuals):>13.4f} {above:>7}/{len(vecs)}"
        )

    # --- Key comparison: overlap analysis ---
    print("\n" + "-" * 70)
    print("PART C: Detection overlap analysis")
    print("-" * 70)

    # For centroid: use 1-cosine distance, threshold at P95 of normal
    normal_cos_dists = np.concatenate([1 - cos_scores["API (normal)"], 1 - cos_scores["Web (normal)"]])
    cos_threshold = np.percentile(normal_cos_dists, 95)

    normal_sub_residuals = np.concatenate([sub_scores["API (normal)"], sub_scores["Web (normal)"]])

    print(f"\n  Centroid threshold (P95 of normal 1-cosine): {cos_threshold:.4f}")
    print(f"  Subspace threshold (adaptive):               {threshold:.4f}")

    print(f"\n  {'Traffic Type':<22} {'Centroid Det':>13} {'Subspace Det':>13} {'Centroid Misses':>16}")
    print("  " + "-" * 66)

    for label in ["SSH brute force", "Chimera (mixed)"]:
        cos_detected = np.sum(1 - cos_scores[label] > cos_threshold)
        sub_detected = np.sum(sub_scores[label] > threshold)

        # Cases where subspace catches but centroid misses
        cos_misses = 1 - cos_scores[label] <= cos_threshold
        sub_catches = sub_scores[label] > threshold
        only_subspace = np.sum(cos_misses & sub_catches)

        n = len(cos_scores[label])
        print(
            f"  {label:<22} {cos_detected:>6}/{n:<6} {sub_detected:>6}/{n:<6} "
            f"{only_subspace:>8} caught only by subspace"
        )

    # --- Projection visualization ---
    print("\n" + "-" * 70)
    print("PART D: Low-dimensional projection (first 3 components)")
    print("-" * 70)

    print(f"\n  {'Traffic Type':<22} {'PC1':>8} {'PC2':>8} {'PC3':>8} {'|Proj|':>8}")
    print("  " + "-" * 58)

    for label, vecs in test_sets:
        coords = np.array([sub.project(v) for v in vecs[:20]])
        mean_coords = np.mean(coords, axis=0)
        proj_norm = np.mean(np.linalg.norm(coords, axis=1))
        print(
            f"  {label:<22} {mean_coords[0]:>8.3f} {mean_coords[1]:>8.3f} "
            f"{mean_coords[2]:>8.3f} {proj_norm:>8.3f}"
        )

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    ssh_sub_tp = np.sum(sub_scores["SSH brute force"] > threshold)
    chimera_sub_tp = np.sum(sub_scores["Chimera (mixed)"] > threshold)
    chimera_cos_misses = np.sum(1 - cos_scores["Chimera (mixed)"] <= cos_threshold)

    normal_sub_fp = np.sum(normal_sub_residuals > threshold)

    checks = [
        (
            "Subspace detects SSH brute force (>80% TP)",
            ssh_sub_tp > 80,
            f"TP={ssh_sub_tp}/100",
        ),
        (
            "Subspace detects chimera anomalies (>50% TP)",
            chimera_sub_tp > 50,
            f"TP={chimera_sub_tp}/100",
        ),
        (
            "Low false positives on normal (<5%)",
            normal_sub_fp < 10,
            f"FP={normal_sub_fp}/200",
        ),
        (
            "Chimera closer to centroid than SSH",
            np.mean(1 - cos_scores["Chimera (mixed)"]) < np.mean(1 - cos_scores["SSH brute force"]),
            f"chimera dist={np.mean(1 - cos_scores['Chimera (mixed)']):.4f}, ssh dist={np.mean(1 - cos_scores['SSH brute force']):.4f}",
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
