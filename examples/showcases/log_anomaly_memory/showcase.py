#!/usr/bin/env python3
"""
Log Anomaly Memory - Catching What Rules Miss

Every field is valid. The combination isn't.

Normal logs have correlated structure: certain users prefer certain actions
at certain hours with predictable response codes. A simple allowlist never
catches the insider threat who uses a valid action they've never used before,
or the account compromise where all fields look fine except in combination.

Holon learns the joint manifold of normal behavior — no schema, no rules,
no labels — and flags anything that doesn't fit the learned structure,
then pinpoints which field broke the pattern.

Run with: ./scripts/run_with_venv.sh python -m examples.showcases.log_anomaly_memory.showcase
"""

import random

from holon.kernel import Encoder, VectorManager, invert, prototype
from holon.kernel.walkable import LogScale, TimeScale
from holon.memory import EngramLibrary, OnlineSubspace

DIM = 4096
BASE_TS = 1_700_000_000  # 2023-11-14 00:00 UTC

# User profiles: tightly correlated behavioral fingerprints.
# Each user has a narrow action set, a specific active window, and
# consistent status codes. Per-field allowlists see these as fine;
# the joint manifold knows the difference.
USER_PROFILES = {
    "u_001": {
        "actions": ["view", "export"],
        "hours": [10, 11, 12, 13, 14, 15],
        "status": ["200"],
    },
    "u_002": {
        "actions": ["view", "export"],
        "hours": [10, 11, 12, 13, 14, 15],
        "status": ["200"],
    },
    "u_003": {
        "actions": ["login", "view"],
        "hours": [9, 10, 11, 14, 15, 16],
        "status": ["200", "301"],
    },
    "u_004": {
        "actions": ["login", "view"],
        "hours": [9, 10, 11, 14, 15, 16],
        "status": ["200", "301"],
    },
    "u_005": {"actions": ["edit"], "hours": [11, 12, 13, 14], "status": ["200"]},
    "u_006": {"actions": ["edit"], "hours": [11, 12, 13, 14], "status": ["200"]},
    "u_007": {"actions": ["export"], "hours": [10, 11, 12, 13, 14], "status": ["200"]},
    "u_008": {"actions": ["export"], "hours": [10, 11, 12, 13, 14], "status": ["200"]},
    "u_009": {
        "actions": ["login", "edit"],
        "hours": [8, 9, 10, 15, 16, 17],
        "status": ["200"],
    },
    "u_010": {
        "actions": ["login", "edit"],
        "hours": [8, 9, 10, 15, 16, 17],
        "status": ["200"],
    },
}

USERS = list(USER_PROFILES.keys())


def normal_log(rng, offset_s):
    user = rng.choice(USERS)
    profile = USER_PROFILES[user]
    hour = rng.choice(profile["hours"])
    ts = BASE_TS + offset_s + hour * 3600
    return {
        "timestamp": TimeScale(ts, resolution="hour"),
        "user": user,
        "action": rng.choice(profile["actions"]),
        "status": rng.choice(profile["status"]),
        "duration_ms": LogScale(rng.uniform(50, 300)),
    }


def attribute_field(enc, subspace, record, normal_ref):
    """Swap each field back to a representative normal value.
    The swap that most reduces residual names the culprit."""
    base = subspace.residual(enc.encode_data(record))
    best_field, best_drop = "unknown", 0.0
    for field in ["user", "action", "status"]:
        candidate = {**record, field: normal_ref[field]}
        drop = base - subspace.residual(enc.encode_data(candidate))
        if drop > best_drop:
            best_drop, best_field = drop, field
    return best_field, best_drop


def main():
    print("=" * 65)
    print("LOG ANOMALY MEMORY")
    print("Catching What Rules Can't: Combinatorial Anomalies")
    print("=" * 65)

    rng = random.Random(7)
    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)

    # Train on 600 normal logs — tight correlated behavior per user
    print("\nLearning normal behavior (600 logs, correlated user profiles)...")
    train = [normal_log(rng, i * 60) for i in range(600)]
    subspace = OnlineSubspace(dim=DIM, k=32, amnesia=2.0, sigma_mult=2.5)
    for r in train:
        subspace.update(enc.encode_data(r))

    library = EngramLibrary(dim=DIM)
    library.add("normal_ops", subspace, trained_on=600)

    # Normal prototype for invert() structural decomposition
    normal_proto = prototype([enc.encode_data(r) for r in train[:100]])
    normal_ref = train[10]  # a stable reference record for attribution swaps

    print(f"  Threshold  : {subspace.threshold:.2f}")
    print(f"  Explained  : {subspace.explained_ratio:.1%}")
    print("  (adaptive threshold: EMA of training residuals + 2.5σ — zero hand-tuning)")

    # Anomalies: every field is individually valid.
    # A per-field allowlist would pass all of these silently.
    anomalies = [
        {
            "label": "view-only user does 'edit'",
            "note": "u_001 only ever views/exports — 'edit' is a valid system action, never from this user",
            "record": {
                "timestamp": TimeScale(BASE_TS + 700 * 60 + 13 * 3600),
                "user": "u_001",
                "action": "edit",  # valid system action; u_001 has never done this
                "status": "200",
                "duration_ms": LogScale(180),
            },
        },
        {
            "label": "export-only user does 'login'",
            "note": "u_007 only exports — 'login' is a valid system action, u_007 never uses it",
            "record": {
                "timestamp": TimeScale(BASE_TS + 701 * 60 + 11 * 3600),
                "user": "u_007",
                "action": "login",  # valid system action; never from u_007
                "status": "200",
                "duration_ms": LogScale(75),
            },
        },
        {
            "label": "edit action returns unexpected 301 redirect",
            "note": "u_005 edits always return 200 — 301 is a valid HTTP code, wrong for this pattern",
            "record": {
                "timestamp": TimeScale(BASE_TS + 702 * 60 + 12 * 3600),
                "user": "u_005",
                "action": "edit",  # their normal action
                "status": "301",  # valid HTTP status; never seen paired with edit
                "duration_ms": LogScale(120),
            },
        },
        {
            "label": "login-only user suddenly does 'export'",
            "note": "u_009 does login/edit — 'export' is in the system, just not this user's pattern",
            "record": {
                "timestamp": TimeScale(BASE_TS + 703 * 60 + 9 * 3600),
                "user": "u_009",
                "action": "export",  # valid system action; u_009 never exports
                "status": "200",
                "duration_ms": LogScale(250),
            },
        },
    ]

    print("\n" + "-" * 65)
    print("ANOMALY DETECTION  (all fields individually valid)")
    print("-" * 65)

    codebook = [("normal_ops", normal_proto)]
    detected = 0

    for entry in anomalies:
        record = entry["record"]
        vec = enc.encode_data(record)
        residual = subspace.residual(vec)
        is_anomaly = residual > subspace.threshold
        if is_anomaly:
            detected += 1

        field, drop = attribute_field(enc, subspace, record, normal_ref)
        components = invert(vec, codebook, threshold=0.0)
        engram, efit = library.match(vec, top_k=1)[0]
        sim_to_normal = components[0][1] if components else 0.0

        print(f"\n  [{'ANOMALY' if is_anomaly else 'miss   '}] {entry['label']}")
        print(f"      Note    : {entry['note']}")
        print(f"      Residual: {residual:.2f}  (threshold {subspace.threshold:.2f})")
        print(
            f"      Culprit : '{field}' (residual drops {drop:.2f} when field is normalised)"
        )
        print(
            f"      Overlap : {sim_to_normal:.1%} structural similarity to learned normal"
        )
        print(f"      Engram  : '{engram}'  (nearest memory, fit {efit:.2f})")

    # Control: a genuinely normal log should not trigger
    print("\n" + "-" * 65)
    print("CONTROL: genuinely normal log (should NOT trigger)")
    print("-" * 65)
    for seed_offset in [800, 850, 900]:
        normal_test = normal_log(rng, seed_offset * 60)
        vec = enc.encode_data(normal_test)
        residual = subspace.residual(vec)
        status = "ANOMALY" if residual > subspace.threshold else "normal "
        print(
            f"\n  [{status}] user={normal_test['user']}  action={normal_test['action']}"
            f"  status={normal_test['status']}"
        )
        print(f"      Residual: {residual:.2f}  (threshold {subspace.threshold:.2f})")

    print(f"\n{'=' * 65}")
    print("Per-field allowlists see 0 violations in the anomaly set.")
    print(
        f"Holon catches {detected}/{len(anomalies)} combinatorial anomalies — zero rules written."
    )
    # Try: lower sigma_mult=1.5 for more sensitivity (may increase false positive rate)
    # Try: add a second engram for a "known_attack" profile and see match() rank it higher on anomalies


if __name__ == "__main__":
    main()
