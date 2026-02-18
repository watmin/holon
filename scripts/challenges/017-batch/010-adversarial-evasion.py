#!/usr/bin/env python3
"""
Adversarial Evasion — Can an Attacker Stay In-Subspace?

HYPOTHESIS:
===========
An attacker who knows the subspace (or can probe it) could craft vectors
that project cleanly onto the manifold while still carrying malicious field
values. We test three evasion strategies:

  1. Naive attack: normal attack traffic (no evasion)
  2. Field-minimized: change only 1-2 fields from normal template
  3. Projection-aware: project the attack vector onto the subspace
     and send the reconstructed version (zero residual by construction)

Strategy 3 is the worst case — it produces a vector with zero residual.
But the key question is: does the projected vector still carry the
malicious field values? If projection destroys the attack semantics,
subspace evasion is self-defeating.

PRIMITIVES DEMONSTRATED:
========================
1. OnlineSubspace.reconstruct()       - Attacker's evasion tool
2. OnlineSubspace.residual()          - Defender's detection
3. unbind() + cosine_similarity()     - Check if attack semantics survive
4. surprise_fingerprint()             - Attribution on evasive vectors

VECTOR PROPERTIES EXPLOITED:
============================
- The subspace spans "normal" directions — projecting onto it removes
  anomalous components, which may be exactly the attack-bearing components
- VSA binding preserves field associations — we can check if specific
  field=value pairs survive projection
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import unbind
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


def make_naive_dns_amp(rng):
    """Full DNS amp — all fields are attack-specific."""
    return {
        "src_ip": str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "dst_port": "53",
        "path": "dns",
        "status": "200",
        "ttl": "245",
    }


def make_minimal_attack(rng):
    """Minimal evasion: change only 2 fields (dst_port, path) from a normal template."""
    d = make_normal(rng)
    d["dst_port"] = "53"
    d["path"] = "dns"
    return d


def check_field_recovery(client, vec, field, expected_value):
    """Check if unbinding recovers the expected value for a field."""
    role_vec = client.get_vector(field)
    recovered = unbind(vec, role_vec)
    expected_vec = client.get_vector(str(expected_value))
    return float(cosine_similarity(recovered, expected_vec))


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("EXPERIMENT 10: Adversarial Evasion — Can Attackers Dodge the Subspace?")
    print("=" * 70)

    # --- Train subspace ---
    n_train = 1000
    print(f"\nTraining subspace on {n_train} normal vectors...")
    sub = client.create_subspace(k=64, amnesia=2.0, sigma_mult=3.5)
    for _ in range(n_train):
        sub.update(client.encode(make_normal(rng)))
    print(f"  {sub}")

    # --- Strategy 1: Naive attack ---
    print("\n" + "-" * 70)
    print("STRATEGY 1: Naive attack (no evasion)")
    print("-" * 70)

    n_test = 100
    naive_dicts = [make_naive_dns_amp(np.random.default_rng(i + 200)) for i in range(n_test)]
    naive_vecs = [client.encode(d) for d in naive_dicts]
    naive_residuals = [sub.residual(v) for v in naive_vecs]
    naive_detected = sum(1 for r in naive_residuals if r > sub.threshold)

    print(f"  Detection rate: {naive_detected}/{n_test} ({naive_detected/n_test*100:.0f}%)")
    print(f"  Mean residual: {np.mean(naive_residuals):.2f} (threshold: {sub.threshold:.2f})")

    # --- Strategy 2: Minimal field changes ---
    print("\n" + "-" * 70)
    print("STRATEGY 2: Minimal evasion (change only 2 fields)")
    print("-" * 70)

    minimal_dicts = [make_minimal_attack(np.random.default_rng(i + 300)) for i in range(n_test)]
    minimal_vecs = [client.encode(d) for d in minimal_dicts]
    minimal_residuals = [sub.residual(v) for v in minimal_vecs]
    minimal_detected = sum(1 for r in minimal_residuals if r > sub.threshold)

    print(f"  Detection rate: {minimal_detected}/{n_test} ({minimal_detected/n_test*100:.0f}%)")
    print(f"  Mean residual: {np.mean(minimal_residuals):.2f} (threshold: {sub.threshold:.2f})")

    # Check which fields are still surprising
    if minimal_detected > 0:
        sample_idx = next(i for i, r in enumerate(minimal_residuals) if r > sub.threshold)
        fp = client.surprise_fingerprint(minimal_vecs[sample_idx], sub, fields=FIELDS)
        print(f"  Top surprising fields: {list(fp.items())[:3]}")

    # --- Strategy 3: Projection-aware evasion ---
    print("\n" + "-" * 70)
    print("STRATEGY 3: Projection-aware (reconstruct attack onto subspace)")
    print("-" * 70)
    print("  The attacker encodes their packet, then sends reconstruct(vec)")
    print("  instead of vec. By construction, residual ≈ 0.\n")

    projected_vecs = [sub.reconstruct(v) for v in naive_vecs]
    projected_residuals = [sub.residual(v) for v in projected_vecs]
    projected_detected = sum(1 for r in projected_residuals if r > sub.threshold)

    print(f"  Detection rate: {projected_detected}/{n_test} ({projected_detected/n_test*100:.0f}%)")
    print(f"  Mean residual: {np.mean(projected_residuals):.4f} (threshold: {sub.threshold:.2f})")

    # --- The critical question: do attack semantics survive projection? ---
    print("\n" + "-" * 70)
    print("KEY QUESTION: Do attack field=value pairs survive projection?")
    print("-" * 70)
    print("  If projection destroys 'dst_port=53' and 'ttl=245', the attacker's")
    print("  evasive packet doesn't actually carry the attack payload.\n")

    # Attack-exclusive: values that NEVER appear in normal traffic
    attack_exclusive = {"dst_port": "53", "path": "dns", "ttl": "245"}
    # Shared: values that also appear in normal traffic
    shared_fields = {"proto": "UDP", "status": "200"}
    attack_fields = {**attack_exclusive, **shared_fields}
    normal_fields = {}

    print(f"  {'Field=Value':<20} {'Original':>10} {'Projected':>10} {'Survived?':>12}")
    print(f"  {'-'*55}")

    all_fields = {**attack_fields, **normal_fields}
    survival_results = {}

    for field, value in all_fields.items():
        orig_sims = []
        proj_sims = []
        for orig_v, proj_v in zip(naive_vecs[:20], projected_vecs[:20]):
            orig_sims.append(check_field_recovery(client, orig_v, field, value))
            proj_sims.append(check_field_recovery(client, proj_v, field, value))

        orig_mean = np.mean(orig_sims)
        proj_mean = np.mean(proj_sims)
        survived = proj_mean > 0.05
        survival_results[field] = (orig_mean, proj_mean, survived)

        label = f"{field}={value}"
        tag = "YES" if survived else "DESTROYED"
        is_attack = field in attack_fields
        marker = " ◀ attack" if is_attack else ""
        print(f"  {label:<20} {orig_mean:>10.4f} {proj_mean:>10.4f} {tag:>12}{marker}")

    # --- Strategy 3b: What if the attacker blends projected + original? ---
    print("\n" + "-" * 70)
    print("STRATEGY 3b: Blend attack with projection (alpha * original + (1-alpha) * projected)")
    print("-" * 70)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"\n  {'Alpha':>7} {'Det Rate':>10} {'Mean Resid':>12} {'dst_port=53':>14} {'ttl=245':>10}")
    print(f"  {'-'*55}")

    for alpha in alphas:
        blended_vecs = [
            alpha * orig.astype(np.float64) + (1 - alpha) * proj
            for orig, proj in zip(naive_vecs, projected_vecs)
        ]
        blend_residuals = [sub.residual(v) for v in blended_vecs]
        blend_detected = sum(1 for r in blend_residuals if r > sub.threshold)

        # Check field survival
        port_sims = [check_field_recovery(client, v, "dst_port", "53") for v in blended_vecs[:20]]
        ttl_sims = [check_field_recovery(client, v, "ttl", "245") for v in blended_vecs[:20]]

        det_pct = blend_detected / n_test * 100
        print(f"  {alpha:>7.1f} {det_pct:>9.0f}% {np.mean(blend_residuals):>12.2f} "
              f"{np.mean(port_sims):>14.4f} {np.mean(ttl_sims):>10.4f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    exclusive_destroyed = all(
        not survived for field, (_, _, survived) in survival_results.items()
        if field in attack_exclusive
    )
    shared_survived = any(
        survived for field, (_, _, survived) in survival_results.items()
        if field in shared_fields
    )

    checks = [
        (
            "Naive attack detected (baseline works)",
            naive_detected / n_test > 0.95,
            f"{naive_detected/n_test*100:.0f}%",
        ),
        (
            "Minimal evasion still detected (2 changed fields is enough)",
            minimal_detected / n_test > 0.10,
            f"{minimal_detected/n_test*100:.0f}%",
        ),
        (
            "Projection evasion defeats residual detector",
            projected_detected / n_test < 0.05,
            f"{projected_detected/n_test*100:.0f}%",
        ),
        (
            "Attack-exclusive fields (53, dns, 245) destroyed by projection",
            exclusive_destroyed,
            ", ".join(f"{f}={survival_results[f][1]:.3f}" for f in attack_exclusive),
        ),
        (
            "Shared fields (UDP, 200) survive (they're in normal traffic too)",
            shared_survived,
            ", ".join(f"{f}={survival_results[f][1]:.3f}" for f in shared_fields),
        ),
    ]

    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc} ({detail})")

    print("\n  CONCLUSION: Projection evasion is self-defeating for novel attacks.")
    print("  The attacker can dodge the residual detector, but projection")
    print("  destroys exactly the field values that make the attack unique")
    print("  (dst_port=53, path=dns, ttl=245). Shared values like proto=UDP")
    print("  survive because the subspace already spans those directions —")
    print("  but they're not useful for attack identification.")
    print("  Mitigation: combine subspace with field-value concentration monitoring.")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
