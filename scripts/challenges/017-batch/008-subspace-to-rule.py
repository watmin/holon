#!/usr/bin/env python3
"""
End-to-End: Subspace → Surprise Fingerprint → Mitigation Rule

HYPOTHESIS:
===========
The subspace anomaly detection pipeline can produce actionable mitigation
rules by: (1) detecting anomalous traffic, (2) extracting the anomalous
component, (3) identifying which fields are surprising, (4) finding the
consensus values in those fields, and (5) generating an EDN-style rule
predicate targeting only the surprising fields.

This is the "material fingerprint to mitigate with" — a rule derived
entirely from the vector algebra, no signatures, no threat intel.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.update/residual()      - Detection
2. OnlineSubspace.anomalous_component()  - Isolation
3. unbind()                               - Per-field decomposition
4. concentration analysis                 - Consensus value extraction
5. Rule generation                        - EDN predicate output

SCENARIO:
=========
Normal web traffic → DNS amplification attack → generate mitigation rule.
The rule should target the fields that are actually surprising (proto=UDP,
dst_port=53, ttl=245) and NOT the fields that are familiar (status=200).

Then test the generated rule: does it block all attack traffic and pass
all normal traffic?

VECTOR PROPERTIES EXPLOITED:
============================
- Subspace boundary for detection (no thresholds)
- Anomalous component for surgical field isolation
- Field-value unbinding for consensus extraction
- Surprise magnitude ranking for rule minimality
"""

import sys
from collections import Counter
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
        "path": str(rng.choice(["api", "static", "health", "metrics", "users"])),
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
        "src_ip": f"{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.100",
        "proto": "TCP",
        "dst_port": "80",
        "path": "syn",
        "status": "none",
        "ttl": str(rng.choice(["64", "128", "255"])),
    }


def compute_field_surprise(client, vec, subspace):
    """Per-field anomaly magnitude from subspace anomalous component."""
    anomaly = subspace.anomalous_component(vec)
    scores = {}
    for field in FIELDS:
        role_vec = client.get_vector(field)
        field_anomaly = unbind(anomaly, role_vec)
        scores[field] = float(np.linalg.norm(field_anomaly))
    return scores


def generate_rule(client, attack_dicts, attack_vecs, subspace, top_k=3, min_consensus=0.5):
    """Generate mitigation rule from subspace analysis.

    Returns:
        rule_predicates: list of (field, operator, value) tuples
        rule_edn: EDN-formatted rule string
        stats: dict of analysis statistics
    """
    # Step 1: Compute mean surprise fingerprint
    all_scores = {f: [] for f in FIELDS}
    for v in attack_vecs:
        scores = compute_field_surprise(client, v, subspace)
        for f in FIELDS:
            all_scores[f].append(scores[f])

    mean_scores = {f: np.mean(s) for f, s in all_scores.items()}

    # Step 2: Rank fields by surprise, take top-k
    ranked = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    top_fields = [f for f, _ in ranked[:top_k]]

    # Step 3: For each top field, find consensus value
    predicates = []
    for field in top_fields:
        values = [d[field] for d in attack_dicts]
        counter = Counter(values)
        top_value, top_count = counter.most_common(1)[0]
        consensus = top_count / len(values)

        if consensus >= min_consensus:
            predicates.append((field, "=", top_value, consensus, mean_scores[field]))

    # Step 4: Format as EDN rule
    conditions = " ".join(f"(= {f} {v})" for f, _, v, _, _ in predicates)
    edn = f"((and {conditions}) => (drop))" if predicates else "(no-rule)"

    return predicates, edn, {
        "mean_scores": mean_scores,
        "ranked": ranked,
        "top_fields": top_fields,
    }


def test_rule(predicates, test_dicts, is_attack):
    """Test a rule against traffic. Returns (matches, total)."""
    matches = 0
    for d in test_dicts:
        match = all(d.get(f) == v for f, _, v, _, _ in predicates)
        if match:
            matches += 1
    return matches, len(test_dicts)


def run_scenario(client, name, attack_gen, n_attack=100):
    """Run full pipeline for one attack type."""
    rng_normal = np.random.default_rng(42)
    rng_attack = np.random.default_rng(100)

    # Train subspace on normal traffic
    n_train = 500
    normal_dicts = [make_normal(rng_normal) for _ in range(n_train)]
    normal_vecs = [client.encode(d) for d in normal_dicts]

    sub = OnlineSubspace(dim=4096, k=64, amnesia=2.0, sigma_mult=3.5)
    for v in normal_vecs:
        sub.update(v)

    # Generate attack traffic
    attack_dicts = [attack_gen(np.random.default_rng(i + 200)) for i in range(n_attack)]
    attack_vecs = [client.encode(d) for d in attack_dicts]

    # Detect: which are above threshold?
    detected_dicts = []
    detected_vecs = []
    for d, v in zip(attack_dicts, attack_vecs):
        if sub.residual(v) > sub.threshold:
            detected_dicts.append(d)
            detected_vecs.append(v)

    detection_rate = len(detected_dicts) / len(attack_dicts) * 100

    # Generate rule from detected traffic
    if detected_dicts:
        predicates, edn, stats = generate_rule(
            client, detected_dicts, detected_vecs, sub, top_k=3
        )
    else:
        predicates, edn, stats = [], "(no-rule)", {}

    # Test rule
    holdout_normal = [make_normal(np.random.default_rng(i + 500)) for i in range(200)]
    holdout_attack = [attack_gen(np.random.default_rng(i + 700)) for i in range(200)]

    tp, total_attack = test_rule(predicates, holdout_attack, True)
    fp, total_normal = test_rule(predicates, holdout_normal, False)

    return {
        "name": name,
        "detection_rate": detection_rate,
        "n_detected": len(detected_dicts),
        "predicates": predicates,
        "edn": edn,
        "stats": stats,
        "tp": tp,
        "tp_total": total_attack,
        "fp": fp,
        "fp_total": total_normal,
    }


def main():
    client = HolonClient(dimensions=4096)

    print("=" * 70)
    print("EXPERIMENT 8: Subspace → Surprise Fingerprint → Mitigation Rule")
    print("=" * 70)

    scenarios = [
        ("DNS Amplification", make_dns_amp),
        ("SYN Flood", make_syn_flood),
    ]

    results = []
    for name, gen_fn in scenarios:
        print(f"\n{'─' * 70}")
        print(f"  Scenario: {name}")
        print(f"{'─' * 70}")

        r = run_scenario(client, name, gen_fn)
        results.append(r)

        # Detection
        print(f"\n  Detection: {r['detection_rate']:.0f}% ({r['n_detected']} packets flagged)")

        # Surprise fingerprint
        if r["stats"]:
            print(f"\n  Surprise fingerprint (ranked):")
            for field, score in r["stats"]["ranked"]:
                bar = "█" * int(score / r["stats"]["ranked"][0][1] * 20) if r["stats"]["ranked"][0][1] > 0 else ""
                print(f"    {field:<12} {score:>8.2f} {bar}")

        # Rule
        print(f"\n  Generated rule:")
        print(f"    {r['edn']}")

        if r["predicates"]:
            print(f"\n  Rule predicates:")
            for field, op, value, consensus, surprise in r["predicates"]:
                print(f"    ({op} {field} {value})  consensus={consensus:.0%}  surprise={surprise:.1f}")

        # Test results
        tp_rate = r["tp"] / r["tp_total"] * 100 if r["tp_total"] > 0 else 0
        fp_rate = r["fp"] / r["fp_total"] * 100 if r["fp_total"] > 0 else 0
        print(f"\n  Rule effectiveness:")
        print(f"    True positives:  {r['tp']}/{r['tp_total']} ({tp_rate:.1f}%)")
        print(f"    False positives: {r['fp']}/{r['fp_total']} ({fp_rate:.1f}%)")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = []
    for r in results:
        tp_rate = r["tp"] / r["tp_total"] * 100 if r["tp_total"] > 0 else 0
        fp_rate = r["fp"] / r["fp_total"] * 100 if r["fp_total"] > 0 else 0

        checks.append((
            f"{r['name']}: detection > 90%",
            r["detection_rate"] > 90,
            f"{r['detection_rate']:.0f}%",
        ))
        checks.append((
            f"{r['name']}: rule TP > 50%",
            tp_rate > 50,
            f"TP={tp_rate:.1f}%",
        ))
        checks.append((
            f"{r['name']}: rule FP = 0%",
            fp_rate == 0,
            f"FP={fp_rate:.1f}%",
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
