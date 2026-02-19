#!/usr/bin/env python3
"""
Config Drift Remediation - Detect, Attribute, and Fix

Stream 15 stable config versions through a subspace until it converges,
detect drift in 5 injected variants, attribute the changed key, and
generate a remediation vector via difference + amplify.

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
        },
        "redis": {"host": "redis.internal", "port": 6379},
        "api": {"rate_limit": rng.choice([1000, 1200, 1500]), "timeout": 60},
        "features": {"dark_mode": True, "analytics": True},
    }


def apply_drift(cfg, section, key, value):
    """Return a deep copy of cfg with cfg[section][key] set to value."""
    out = copy.deepcopy(cfg)
    out[section][key] = value
    return out


DRIFTS = [
    ("db.host→external", "db", "host", "db.evil.com"),
    ("db.port→mysql", "db", "port", 3306),
    ("api.rate_limit→∞", "api", "rate_limit", 999_999),
    ("features.analytics→off", "features", "analytics", False),
    ("redis.host→external", "redis", "host", "redis.evil.com"),
]


def flatten(cfg, prefix=""):
    for k, v in cfg.items():
        full = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(v, f"{full}.")
        else:
            yield full, v


def attribute_drift(enc, subspace, drifted, golden_ref):
    """Swap each leaf field back to golden; biggest residual drop = culprit."""
    base = subspace.residual(enc.encode_data(drifted))
    best_field, best_drop = "unknown", 0.0
    for dotted_key, golden_val in flatten(golden_ref):
        candidate = copy.deepcopy(drifted)
        node = candidate
        keys = dotted_key.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = golden_val
        drop = base - subspace.residual(enc.encode_data(candidate))
        if drop > best_drop:
            best_drop, best_field = drop, dotted_key
    return best_field, best_drop


def main():
    print("=" * 65)
    print("CONFIG DRIFT REMEDIATION")
    print("=" * 65)

    rng = random.Random(42)
    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)

    # ── Golden subspace: stream 15 stable configs × 10 passes ────
    print("\nStreaming 15 stable configs until subspace converges...")
    stable = [stable_config(rng) for _ in range(15)]
    stable_vecs = [enc.encode_data(c) for c in stable]
    golden_proto = prototype(stable_vecs)

    subspace = OnlineSubspace(dim=DIM, k=32, amnesia=2.0, sigma_mult=3.0)
    for _ in range(10):  # 10 passes → tight convergence
        for v in stable_vecs:
            subspace.update(v)

    library = EngramLibrary(dim=DIM)
    library.add("golden_config", subspace, version="stable")
    golden_ref = stable[0]
    train_max = max(subspace.residual(v) for v in stable_vecs)
    print(f"  Threshold      : {subspace.threshold:.2f}")
    print(f"  Max train resid: {train_max:.4f}  (near-zero = converged)")

    # ── Detect and remediate each drift ──────────────────────────
    print("\n" + "-" * 65)
    print("DRIFT DETECTION + REMEDIATION")
    print("-" * 65)

    detected = 0
    for drift_name, section, key, bad_value in DRIFTS:
        drifted = apply_drift(stable_config(rng), section, key, bad_value)
        drifted_vec = enc.encode_data(drifted)
        residual = subspace.residual(drifted_vec)
        is_drift = residual > subspace.threshold

        print(f"\n  Config  : {drift_name}")
        print(
            f"  Residual: {residual:.2f}  (threshold {subspace.threshold:.2f})  drift={is_drift}"
        )

        if is_drift:
            detected += 1
            field, drop = attribute_drift(enc, subspace, drifted, golden_ref)
            delta = difference(golden_proto, drifted_vec)
            remediation = amplify(golden_proto, delta, strength=0.5)
            rem_residual = subspace.residual(remediation)
            print(f"  Cause   : '{field}' (residual drop {drop:.2f} when reverted)")
            print(
                f"  Fix     : amplify(golden, Δ) → new residual {rem_residual:.2f}  "
                f"(was {residual:.2f})"
            )

    print(f"\n{'=' * 65}")
    print(f"Detected {detected}/{len(DRIFTS)} drifts.")
    # Try: add per-environment engrams (staging vs prod) to catch environment mix-ups.


if __name__ == "__main__":
    main()
