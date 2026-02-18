#!/usr/bin/env python3
"""
Multi-Attack Separation via Subspace Peeling

HYPOTHESIS:
===========
When two attack types occur simultaneously, the anomalous component
is a superposition of both attacks' signatures. We can "peel" them
apart by:
  1. Cluster the anomalous components (or their fingerprints)
  2. Build a prototype for the dominant cluster
  3. Subtract (negate) that prototype from each anomalous component
  4. The residual reveals the second attack's signature

This is the VSA analog of iterative source separation: each "peel"
removes one attack type from the superposition.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.anomalous_component()  - Extract combined anomaly
2. surprise_fingerprint()                 - Fingerprint for clustering
3. prototype()                            - Dominant attack signature
4. negate()                               - Remove dominant pattern
5. unbind()                               - Identify fields after peeling

SCENARIO:
=========
Normal traffic, then a mixed attack: DNS amplification AND credential
stuffing occurring simultaneously. Can we separate the two?
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import prototype, unbind
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


def compute_fingerprint(client, anomaly_vec):
    """Per-field anomaly magnitude from an anomaly vector."""
    scores = {}
    for field in FIELDS:
        role_vec = client.get_vector(field)
        field_anomaly = unbind(anomaly_vec, role_vec)
        scores[field] = float(np.linalg.norm(field_anomaly))
    return scores


def fingerprint_vec(fp):
    """Convert fingerprint dict to numpy array for clustering."""
    return np.array([fp[f] for f in FIELDS])


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 12: Multi-Attack Separation via Subspace Peeling")
    print("=" * 70)

    # --- Train on normal ---
    n_train = 1000
    print(f"\nTraining subspace on {n_train} normal vectors...")
    sub = client.create_subspace(k=64, amnesia=2.0, sigma_mult=3.5)
    for _ in range(n_train):
        sub.update(client.encode(make_normal(rng)))
    print(f"  {sub}")

    # --- Generate mixed attack traffic ---
    n_each = 50
    print(f"\nGenerating mixed attack: {n_each} DNS amp + {n_each} credential stuffing")

    dns_dicts = [make_dns_amp(np.random.default_rng(i + 200)) for i in range(n_each)]
    cred_dicts = [make_cred_stuff(np.random.default_rng(i + 300)) for i in range(n_each)]
    dns_vecs = [client.encode(d) for d in dns_dicts]
    cred_vecs = [client.encode(d) for d in cred_dicts]

    # Mix them (interleaved, as they would arrive in real traffic)
    mixed_dicts = []
    mixed_vecs = []
    mixed_labels = []
    for i in range(n_each):
        mixed_dicts.append(dns_dicts[i])
        mixed_vecs.append(dns_vecs[i])
        mixed_labels.append("dns")
        mixed_dicts.append(cred_dicts[i])
        mixed_vecs.append(cred_vecs[i])
        mixed_labels.append("cred")

    # --- Part A: Detect all attacks ---
    print("\n" + "-" * 70)
    print("PART A: Detection (all mixed attack traffic)")
    print("-" * 70)

    residuals = [sub.residual(v) for v in mixed_vecs]
    detected_idx = [i for i, r in enumerate(residuals) if r > sub.threshold]
    detected_dns = sum(1 for i in detected_idx if mixed_labels[i] == "dns")
    detected_cred = sum(1 for i in detected_idx if mixed_labels[i] == "cred")

    print(f"  Total detected: {len(detected_idx)}/{len(mixed_vecs)}")
    print(f"  DNS amp:        {detected_dns}/{n_each}")
    print(f"  Cred stuffing:  {detected_cred}/{n_each}")

    # --- Part B: Extract anomalous components ---
    print("\n" + "-" * 70)
    print("PART B: Anomalous components and initial fingerprints")
    print("-" * 70)

    anomaly_vecs = [sub.anomalous_component(mixed_vecs[i]) for i in detected_idx]
    anomaly_labels = [mixed_labels[i] for i in detected_idx]
    anomaly_fps = [compute_fingerprint(client, av) for av in anomaly_vecs]
    anomaly_fp_vecs = np.array([fingerprint_vec(fp) for fp in anomaly_fps])

    # Show mean fingerprint across all detected (mixed)
    mean_fp = np.mean(anomaly_fp_vecs, axis=0)
    print(f"\n  Mean fingerprint (all detected, mixed):")
    print(f"  {'Field':<12} {'Score':>8}")
    print(f"  {'-'*22}")
    for i, field in enumerate(FIELDS):
        print(f"  {field:<12} {mean_fp[i]:>8.2f}")

    # --- Part C: Cluster the fingerprints ---
    print("\n" + "-" * 70)
    print("PART C: Separate attacks by fingerprint clustering")
    print("-" * 70)

    # Simple 2-means clustering on fingerprint vectors
    # Initialize with first DNS and first cred (we know there are two types)
    dns_indices = [i for i, l in enumerate(anomaly_labels) if l == "dns"]
    cred_indices = [i for i, l in enumerate(anomaly_labels) if l == "cred"]

    # K-means with 2 clusters
    from collections import defaultdict
    centers = [anomaly_fp_vecs[0].copy(), anomaly_fp_vecs[1].copy()]
    for _ in range(20):
        clusters = defaultdict(list)
        for i, fp in enumerate(anomaly_fp_vecs):
            d0 = np.linalg.norm(fp - centers[0])
            d1 = np.linalg.norm(fp - centers[1])
            clusters[0 if d0 < d1 else 1].append(i)
        for c in [0, 1]:
            if clusters[c]:
                centers[c] = np.mean(anomaly_fp_vecs[list(clusters[c])], axis=0)

    # Determine which cluster is DNS and which is cred
    cluster_dns_count = [
        sum(1 for i in clusters[c] if anomaly_labels[i] == "dns")
        for c in [0, 1]
    ]
    dns_cluster = 0 if cluster_dns_count[0] > cluster_dns_count[1] else 1
    cred_cluster = 1 - dns_cluster

    dns_correct = sum(1 for i in clusters[dns_cluster] if anomaly_labels[i] == "dns")
    cred_correct = sum(1 for i in clusters[cred_cluster] if anomaly_labels[i] == "cred")
    total_correct = dns_correct + cred_correct
    total = len(anomaly_labels)
    cluster_acc = total_correct / total * 100

    print(f"\n  Cluster {dns_cluster}: {len(clusters[dns_cluster])} vectors "
          f"(DNS={dns_correct}, Cred={len(clusters[dns_cluster]) - dns_correct})")
    print(f"  Cluster {cred_cluster}: {len(clusters[cred_cluster])} vectors "
          f"(DNS={len(clusters[cred_cluster]) - cred_correct}, Cred={cred_correct})")
    print(f"  Clustering accuracy: {cluster_acc:.1f}%")

    # --- Part D: Per-cluster fingerprints ---
    print("\n" + "-" * 70)
    print("PART D: Per-cluster surprise fingerprints (after separation)")
    print("-" * 70)

    cluster_fps = {}
    for c, label in [(dns_cluster, "DNS amp"), (cred_cluster, "Cred stuff")]:
        c_fps = anomaly_fp_vecs[list(clusters[c])]
        mean = np.mean(c_fps, axis=0)
        cluster_fps[label] = mean

    print(f"\n  {'Field':<12} {'DNS cluster':>12} {'Cred cluster':>12} {'Diff':>8}")
    print(f"  {'-'*46}")
    for i, field in enumerate(FIELDS):
        dns_v = cluster_fps["DNS amp"][i]
        cred_v = cluster_fps["Cred stuff"][i]
        diff = dns_v - cred_v
        marker = "  ◀" if abs(diff) > 1.0 else ""
        print(f"  {field:<12} {dns_v:>12.2f} {cred_v:>12.2f} {diff:>+7.1f}{marker}")

    # --- Part E: Subspace peeling ---
    print("\n" + "-" * 70)
    print("PART E: Subspace peeling — remove dominant attack, reveal the other")
    print("-" * 70)

    # Build a prototype of the dominant cluster's anomalous components
    dominant_anomalies = [anomaly_vecs[i] for i in clusters[dns_cluster]]
    dominant_proto = prototype(dominant_anomalies)

    print(f"\n  Built prototype from {len(dominant_anomalies)} DNS-cluster anomalous components")

    # Peel: for each cred-cluster vector, subtract the DNS prototype
    peeled_fps = []
    original_fps = []
    for i in clusters[cred_cluster]:
        original_fp = anomaly_fps[i]
        original_fps.append(original_fp)

        # Peel: remove DNS prototype from this anomalous component
        peeled = anomaly_vecs[i].astype(np.float64) - 0.5 * dominant_proto.astype(np.float64)
        peeled_fp = compute_fingerprint(client, peeled)
        peeled_fps.append(peeled_fp)

    # Compare before and after peeling for the cred cluster
    mean_orig = {f: np.mean([fp[f] for fp in original_fps]) for f in FIELDS}
    mean_peeled = {f: np.mean([fp[f] for fp in peeled_fps]) for f in FIELDS}

    print(f"\n  Cred-cluster anomalies before and after DNS peeling:")
    print(f"  {'Field':<12} {'Before':>10} {'After peel':>12} {'Change':>8}")
    print(f"  {'-'*44}")
    for field in FIELDS:
        change = mean_peeled[field] - mean_orig[field]
        print(f"  {field:<12} {mean_orig[field]:>10.2f} {mean_peeled[field]:>12.2f} {change:>+7.1f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Both attack types detected (>90% each)",
            detected_dns / n_each > 0.9 and detected_cred / n_each > 0.9,
            f"DNS={detected_dns}/{n_each}, Cred={detected_cred}/{n_each}",
        ),
        (
            "Fingerprint clustering separates types (>80% accuracy)",
            cluster_acc > 80,
            f"acc={cluster_acc:.1f}%",
        ),
        (
            "Clusters have distinct fingerprint shapes",
            np.linalg.norm(cluster_fps["DNS amp"] - cluster_fps["Cred stuff"]) > 1.0,
            f"L2 distance={np.linalg.norm(cluster_fps['DNS amp'] - cluster_fps['Cred stuff']):.2f}",
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
