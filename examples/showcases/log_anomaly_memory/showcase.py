#!/usr/bin/env python3
"""
Log Anomaly Memory - Detect and Remember

Generate 500 synthetic log lines, learn normal behavior as a subspace,
detect anomalous lines with field-level attribution, recall matching engram.

Run with: ./scripts/run_with_venv.sh python -m examples.showcases.log_anomaly_memory.showcase
"""

import random

from holon.kernel import Encoder, VectorManager
from holon.kernel.walkable import LogScale, TimeScale
from holon.memory import EngramLibrary, OnlineSubspace

DIM = 4096
USERS = [f"u_{i:03d}" for i in range(1, 21)]
ACTIONS = ["login", "view", "edit", "export"]
STATUSES = ["200", "200", "200", "301", "404"]
BASE_TS = 1_700_000_000  # 2023-11-14, synthetic epoch


def normal_log(rng, offset_s):
    return {
        "timestamp": TimeScale(BASE_TS + offset_s),
        "user": rng.choice(USERS),
        "action": rng.choice(ACTIONS),
        "status": rng.choice(STATUSES),
        "duration_ms": LogScale(rng.uniform(10, 500)),
    }


def attribute_field(enc, subspace, record, normal_record):
    """Swap each field back to a normal value; the swap that most reduces
    residual reveals the culprit field."""
    base = subspace.residual(enc.encode_data(record))
    best_field, best_drop = "unknown", 0.0
    for field in record:
        candidate = {**record, field: normal_record[field]}
        drop = base - subspace.residual(enc.encode_data(candidate))
        if drop > best_drop:
            best_drop, best_field = drop, field
    return best_field, best_drop


def main():
    print("=" * 65)
    print("LOG ANOMALY MEMORY")
    print("=" * 65)

    rng = random.Random(42)
    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)

    # ── Training: 400 normal logs ─────────────────────────────────
    print("\nGenerating 500 log lines, training on first 400...")
    train = [normal_log(rng, i * 30) for i in range(400)]
    subspace = OnlineSubspace(dim=DIM, k=32, amnesia=2.0, sigma_mult=2.5)
    for r in train:
        subspace.update(enc.encode_data(r))

    library = EngramLibrary(dim=DIM)
    library.add("normal_ops", subspace, trained_on=400)
    print(f"  Threshold : {subspace.threshold:.2f}")
    print(f"  Explained : {subspace.explained_ratio:.1%}")

    # Representative normal record used for attribution swaps
    normal_ref = normal_log(rng, 0)

    # ── Test set: 90 normal + 10 injected anomalies (one of each type) ──
    anomaly_pool = [
        {
            "timestamp": TimeScale(BASE_TS + 99999),
            "user": "u_005",
            "action": "exfiltrate",
            "status": "200",
            "duration_ms": LogScale(120),
        },
        {
            "timestamp": TimeScale(BASE_TS + 99999),
            "user": "root",
            "action": "login",
            "status": "200",
            "duration_ms": LogScale(80),
        },
        {
            "timestamp": TimeScale(BASE_TS + 99999),
            "user": "u_012",
            "action": "view",
            "status": "503",
            "duration_ms": LogScale(60),
        },
        {
            "timestamp": TimeScale(BASE_TS + 99999),
            "user": "u_007",
            "action": "edit",
            "status": "500",
            "duration_ms": LogScale(60),
        },
        {
            "timestamp": TimeScale(BASE_TS + 99999),
            "user": "u_001",
            "action": "login",
            "status": "401",
            "duration_ms": LogScale(50),
        },
    ]
    test = [normal_log(rng, 400 * 30 + i * 30) for i in range(90)]
    # Inject exactly 2 of each anomaly type for variety
    for anomaly in anomaly_pool:
        test.append(anomaly)
        test.append(anomaly)
    rng.shuffle(test)

    print("\n" + "-" * 65)
    print("ANOMALY DETECTION  (last 100 logs)")
    print("-" * 65)

    seen = set()
    detected = 0
    for record in test:
        vec = enc.encode_data(record)
        residual = subspace.residual(vec)
        if residual <= subspace.threshold:
            continue
        detected += 1
        field, drop = attribute_field(enc, subspace, record, normal_ref)
        engram, efit = library.match(vec, top_k=1)[0]
        key = (record.get("action"), record.get("status"))
        if key not in seen:
            seen.add(key)
            print(
                f"\n  Anomaly type detected: action={record['action']!r}  status={record['status']!r}"
            )
            print(f"    cause   : '{field}' (residual drop {drop:.2f} when normalised)")
            print(f"    residual: {residual:.2f}  (threshold {subspace.threshold:.2f})")
            print(f"    engram  : '{engram}'  (fit {efit:.2f})")

    print(f"\n{'=' * 65}")
    print(
        f"Detected {detected} anomalies in 100 test logs ({len(seen)} distinct types)."
    )
    # Try: lower sigma_mult for more sensitivity, or add a second engram for known attack patterns.


if __name__ == "__main__":
    main()
