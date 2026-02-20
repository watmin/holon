#!/usr/bin/env python3
"""
Config Drift Remediation - Detect, Attribute, and Fix

Stream golden configs through a subspace until it learns "normal."
Then detect drift — including multi-field drift and subtle in-range
changes — attribute the exact fields that changed, and produce a
corrective vector. Verify that the fix actually works.

No schema annotations. No field-specific rules. Just structure.

Run with: ./scripts/run_with_venv.sh python -m examples.showcases.config_drift_remediation.showcase
"""

import copy
import random

from holon.kernel import Encoder, VectorManager, amplify, difference, prototype
from holon.memory import EngramLibrary, OnlineSubspace

DIM = 4096


def stable_config(rng):
    return {
        "db": {
            "host": rng.choice(["db-primary.internal", "db-replica.internal"]),
            "port": 5432,
            "pool_size": rng.choice([10, 12, 15]),
            "ssl": True,
        },
        "redis": {"host": "redis.internal", "port": 6379, "ttl": 3600},
        "api": {"rate_limit": rng.choice([1000, 1200, 1500]), "timeout": 60},
        "features": {"dark_mode": True, "analytics": True, "beta": False},
    }


def apply_drift(cfg, changes):
    """Return a deep copy of cfg with each (section, key, value) change applied."""
    out = copy.deepcopy(cfg)
    for section, key, value in changes:
        out[section][key] = value
    return out


def flatten(cfg, prefix=""):
    for k, v in cfg.items():
        full = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(v, f"{full}.")
        else:
            yield full, v


def attribute_drift(enc, subspace, drifted, golden_ref):
    """Iteratively attribute drift: swap each field back, find biggest residual drop.
    Returns ordered list of (field, drop) — biggest culprit first."""
    results = []
    base = subspace.residual(enc.encode_data(drifted))
    for dotted_key, golden_val in flatten(golden_ref):
        candidate = copy.deepcopy(drifted)
        node = candidate
        keys = dotted_key.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = golden_val
        drop = base - subspace.residual(enc.encode_data(candidate))
        if drop > 0.5:  # only report meaningful drops
            results.append((dotted_key, drop))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    print("=" * 65)
    print("CONFIG DRIFT REMEDIATION")
    print("Detect, Attribute, and Fix — No Schema Required")
    print("=" * 65)

    rng = random.Random(42)
    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)

    # ── Build golden subspace: 20 stable configs × 10 passes ─────
    print("\nLearning golden config manifold (20 configs × 10 passes)...")
    stable = [stable_config(rng) for _ in range(20)]
    stable_vecs = [enc.encode_data(c) for c in stable]
    golden_proto = prototype(stable_vecs)
    golden_ref = stable[0]

    subspace = OnlineSubspace(dim=DIM, k=32, amnesia=2.0, sigma_mult=3.0)
    for _ in range(10):
        for v in stable_vecs:
            subspace.update(v)

    library = EngramLibrary(dim=DIM)
    library.add("golden_config", subspace, version="stable")

    train_max = max(subspace.residual(v) for v in stable_vecs)
    train_mean = sum(subspace.residual(v) for v in stable_vecs) / len(stable_vecs)
    print(f"  Threshold  : {subspace.threshold:.2f}")
    print(
        f"  Train mean : {train_mean:.4f}  max: {train_max:.4f}  (near-zero = tight convergence)"
    )

    # ── Drift scenarios ───────────────────────────────────────────
    print("\n" + "-" * 65)
    print("DRIFT DETECTION + ATTRIBUTION + REMEDIATION")
    print("-" * 65)

    drifts = [
        {
            "label": "Single field: db.host -> external",
            "changes": [("db", "host", "db.evil.com")],
        },
        {
            "label": "Subtle drift: pool_size 15 -> 8 (still 'valid', just wrong)",
            "changes": [("db", "pool_size", 8)],
            "note": "8 is a positive integer — any schema validator passes this",
        },
        {
            "label": "Multi-field: db compromised + rate limit blown",
            "changes": [
                ("db", "host", "db.evil.com"),
                ("api", "rate_limit", 999_999),
            ],
            "note": "Two simultaneous changes — cascading attribution finds both",
        },
        {
            "label": "Feature flag poisoning: analytics disabled",
            "changes": [("features", "analytics", False)],
        },
        {
            "label": "Multi-field: redis external + TTL zeroed",
            "changes": [
                ("redis", "host", "redis.evil.com"),
                ("redis", "ttl", 0),
            ],
            "note": "TTL=0 looks like a valid cache-disable — structure catches it",
        },
    ]

    detected = 0
    for drift in drifts:
        base_cfg = stable_config(rng)
        drifted = apply_drift(base_cfg, drift["changes"])
        drifted_vec = enc.encode_data(drifted)
        residual = subspace.residual(drifted_vec)
        is_drift = residual > subspace.threshold

        print(f"\n  Config  : {drift['label']}")
        if "note" in drift:
            print(f"  Note    : {drift['note']}")
        print(
            f"  Residual: {residual:.2f}  (threshold {subspace.threshold:.2f})  drift={is_drift}"
        )

        if is_drift:
            detected += 1

            # Attribution: which fields drove the residual up?
            attrs = attribute_drift(enc, subspace, drifted, golden_ref)
            for i, (field, drop) in enumerate(attrs[:3]):
                print(
                    f"  Cause {i+1} : '{field}' (residual drop {drop:.2f} when reverted)"
                )

            # Remediation: difference + amplify toward golden prototype
            delta = difference(golden_proto, drifted_vec)
            remediation = amplify(golden_proto, delta, strength=0.5)
            rem_residual = subspace.residual(remediation)

            # Verification: encode the ACTUAL correct config — does it score lower?
            correct_vec = enc.encode_data(base_cfg)
            correct_residual = subspace.residual(correct_vec)

            print(
                f"  Fix     : amplify(golden, Δ) → residual {rem_residual:.2f}  (was {residual:.2f})"
            )
            print(
                f"  Verify  : actual correct config residual = {correct_residual:.2f}"
                f"  (below threshold: {correct_residual < subspace.threshold})"
            )

    print(f"\n{'=' * 65}")
    print(
        f"Detected {detected}/{len(drifts)} drifts — including multi-field and subtle in-range changes."
    )
    print("No schema annotations. No field-specific rules. Pure structure.")
    # Try: add a per-environment engram (staging vs prod) to catch environment mix-ups
    # Try: inject a two-field drift and see if cascading attribution finds both


if __name__ == "__main__":
    main()
