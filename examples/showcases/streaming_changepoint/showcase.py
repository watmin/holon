#!/usr/bin/env python3
"""
Streaming Changepoint Detection - System Health Without Labels

A stream of structured system metrics passes through four phases:
healthy → degradation → incident → recovery. No phase labels are provided.

Holon's OnlineSubspace learns "healthy" from the first 50 observations.
Every subsequent observation is scored against that baseline. The residual
rises through degradation, peaks at incident, and falls during recovery —
a structural timeline without any per-metric thresholds.

segment() finds the transition from healthy to the rest automatically.
difference() encodes exactly what changed. invert() decomposes what
the incident "looks like" against known phase prototypes.

Run with: ./scripts/run_with_venv.sh python -m examples.showcases.streaming_changepoint.showcase
"""

import random

from holon.kernel import Encoder, VectorManager, difference, invert, prototype, segment
from holon.memory import EngramLibrary, OnlineSubspace

DIM = 4096


def healthy_metrics(rng):
    """Very consistent healthy baseline."""
    return {
        "latency_ms": rng.choice([22, 23, 24, 25, 26]),
        "error_rate": rng.choice([0, 0, 0, 1]),
        "cpu_pct": rng.choice([20, 21, 22, 23, 24]),
        "mem_pct": rng.choice([44, 45, 46, 47, 48]),
        "req_per_sec": rng.choice([990, 995, 1000, 1005, 1010]),
        "status": "ok",
        "db_pool": rng.choice([20, 21, 22, 23]),
    }


def degraded_metrics(rng, step):
    """Gradual shift over 25 steps."""
    f = min(step / 15.0, 1.0)
    return {
        "latency_ms": rng.randint(int(80 + f * 200), int(150 + f * 250)),
        "error_rate": rng.randint(int(8 + f * 20), int(15 + f * 30)),
        "cpu_pct": rng.randint(int(52 + f * 22), int(65 + f * 18)),
        "mem_pct": rng.randint(int(62 + f * 16), int(74 + f * 13)),
        "req_per_sec": rng.randint(int(1300 + f * 700), int(1700 + f * 900)),
        "status": "ok",
        "db_pool": rng.randint(max(3, int(16 - f * 13)), max(6, int(19 - f * 11))),
    }


def incident_metrics(rng):
    """Hard failure — all metrics far outside normal, status flips."""
    return {
        "latency_ms": rng.randint(2000, 5000),
        "error_rate": rng.randint(150, 350),
        "cpu_pct": rng.randint(92, 99),
        "mem_pct": rng.randint(92, 99),
        "req_per_sec": rng.randint(6000, 10000),
        "status": "degraded",
        "db_pool": rng.randint(0, 1),
    }


def recovery_metrics(rng, step):
    """Returning toward healthy — slightly elevated for first ~20 steps."""
    f = max(0.0, 1.0 - step / 18.0)
    return {
        "latency_ms": rng.randint(int(28 + f * 120), int(50 + f * 100)),
        "error_rate": rng.randint(int(1 + f * 18), int(4 + f * 18)),
        "cpu_pct": rng.randint(int(22 + f * 28), int(35 + f * 28)),
        "mem_pct": rng.randint(int(46 + f * 22), int(56 + f * 22)),
        "req_per_sec": rng.randint(950, 1050),
        "status": "ok",
        "db_pool": rng.randint(12, 22),
    }


def main():
    print("=" * 65)
    print("STREAMING CHANGEPOINT DETECTION")
    print("System Health Without Labels")
    print("=" * 65)

    rng = random.Random(42)
    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)

    N_HEALTHY, N_DEGRADED, N_INCIDENT, N_RECOVERY = 50, 25, 20, 30
    b1 = N_HEALTHY
    b2 = b1 + N_DEGRADED
    b3 = b2 + N_INCIDENT
    total = b3 + N_RECOVERY

    phases = (
        [("healthy", healthy_metrics(rng)) for _ in range(N_HEALTHY)]
        + [("degraded", degraded_metrics(rng, i)) for i in range(N_DEGRADED)]
        + [("incident", incident_metrics(rng)) for _ in range(N_INCIDENT)]
        + [("recovery", recovery_metrics(rng, i)) for i in range(N_RECOVERY)]
    )
    stream_raw = [(p, m) for p, m in phases]
    stream_vecs = [enc.encode_data(m) for _, m in phases]

    print(
        f"\nStream: {N_HEALTHY} healthy → {N_DEGRADED} degraded → {N_INCIDENT} incident → {N_RECOVERY} recovery"
    )
    print(f"Total : {total} observations  |  True boundaries: [{b1}, {b2}, {b3}]")
    print("Holon sees an unlabeled stream — no phase column, no field metadata")

    # ── Learn healthy baseline from first 50 observations ─────────
    print(f"\nLearning healthy baseline from first {N_HEALTHY} observations...")
    subspace = OnlineSubspace(dim=DIM, k=32, amnesia=2.0, sigma_mult=1.5)
    for v in stream_vecs[:N_HEALTHY]:
        subspace.update(v)

    library = EngramLibrary(dim=DIM)
    library.add("healthy", subspace)

    train_r = [subspace.residual(v) for v in stream_vecs[:N_HEALTHY]]
    print(f"  Threshold : {subspace.threshold:.2f}  (1.5σ above healthy EMA)")
    print(f"  Train max : {max(train_r):.2f}  (healthy data is highly consistent)")

    # ── Residual timeline — the structural story ──────────────────
    residuals = [subspace.residual(v) for v in stream_vecs]

    print("\n" + "-" * 65)
    print("RESIDUAL TIMELINE  (structural anomaly score vs healthy baseline)")
    print(
        f"  threshold={subspace.threshold:.1f}  |  bar scale: each █ ≈ 3 residual units"
    )
    print("-" * 65)

    blocks = [
        (f"Healthy   [0-{b1-1}]   ", residuals[:b1]),
        (f"Degraded  [{b1}-{b2-1}]  ", residuals[b1:b2]),
        (f"Incident  [{b2}-{b3-1}]  ", residuals[b2:b3]),
        (f"Recovery  [{b3}-{total-1}] ", residuals[b3:]),
    ]
    for name, block in blocks:
        mean_r = sum(block) / len(block)
        above = sum(1 for r in block if r > subspace.threshold)
        bar = "█" * min(int(mean_r / 3), 42)
        threshold_marker = (
            " ◄ threshold" if int(subspace.threshold / 3) == int(mean_r / 3) else ""
        )
        print(
            f"  {name}  mean={mean_r:5.1f}  flagged={above:2d}/{len(block):2d}  {bar}{threshold_marker}"
        )

    print(
        f"\n  Threshold line at ~{subspace.threshold:.1f} separates healthy from anomalous."
    )
    print("  No per-metric rules wrote this — the subspace inferred it from structure.")

    # ── Changepoint detection ─────────────────────────────────────
    print("\n" + "-" * 65)
    print("CHANGEPOINT DETECTION  (segment() — finds transitions in the stream)")
    print("-" * 65)

    # prototype method with a larger window: compares each observation to
    # a rolling prototype of the recent window. Fires when structural similarity drops.
    breakpoints = segment(stream_vecs, window=20, threshold=0.70, method="prototype")

    # Consolidate nearby breakpoints; only keep the first in each dense cluster
    consolidated = []
    for bp in breakpoints:
        if bp == 0:
            continue
        if not consolidated or bp - consolidated[-1] > 12:
            consolidated.append(bp)

    true_bnd = {b1: "degraded", b2: "incident", b3: "recovery"}

    # Find which consolidated breakpoints are genuinely near true boundaries
    matched = [
        (bp, min(true_bnd.keys(), key=lambda b: abs(b - bp)))
        for bp in consolidated
        if any(abs(bp - b) <= 12 for b in true_bnd)
    ]

    print(f"\n  All consolidated breakpoints: {consolidated}")
    print(f"  True phase boundaries      : {sorted(true_bnd.keys())}")
    print(f"  Matched (within 12 steps)  : {[bp for bp, _ in matched]}\n")

    for bp, nearest in matched:
        phase = stream_raw[bp][0]
        drift = abs(bp - nearest)
        expect = true_bnd[nearest]
        print(
            f"  → index {bp:3d}: entering '{phase}' territory"
            f"  (expected '{expect}' at {nearest}, drift={drift} steps)"
        )

    # ── Change analysis ───────────────────────────────────────────
    print("\n" + "-" * 65)
    print("CHANGE ANALYSIS  (difference + invert)")
    print("-" * 65)

    healthy_proto = prototype(stream_vecs[:b1])
    incident_proto = prototype(stream_vecs[b2:b3])
    recovery_proto = prototype(stream_vecs[b3:])

    delta = difference(healthy_proto, incident_proto)
    delta_density = float(sum(1 for x in delta if x != 0)) / len(delta)

    print("\n  difference(healthy, incident):")
    print(f"    {delta_density:.1%} of vector dimensions changed")
    print(
        "    This delta IS the outage fingerprint — storable and algebraically composable"
    )

    codebook = [
        ("healthy", healthy_proto),
        ("incident", incident_proto),
        ("recovery", recovery_proto),
    ]
    components = invert(incident_proto, codebook, threshold=0.0)
    print("\n  invert(incident_proto, codebook) — structural overlap per phase:")
    for name, sim in components:
        bar = "█" * int(sim * 30)
        print(f"    '{name}':  {sim:.3f}  {bar}")

    match_name, match_res = library.match(recovery_proto, top_k=1)[0]
    flagged = match_res > subspace.threshold
    print(f"\n  EngramLibrary.match(recovery_proto) → '{match_name}'")
    print(
        f"    residual={match_res:.2f}  threshold={subspace.threshold:.2f}  anomalous={flagged}"
    )
    msg = "structurally stabilised" if not flagged else "still elevated"
    print(f"    Recovery {msg} relative to learned healthy baseline")

    print(f"\n{'=' * 65}")
    print(
        f"Residual rises 3x from healthy ({sum(residuals[:b1])/b1:.1f}) to incident ({sum(residuals[b2:b3])/N_INCIDENT:.1f})."
    )
    print(
        f"segment() identified {len(matched)}/{len(true_bnd)} phase transitions within 12 steps."
    )
    print("Zero per-metric thresholds. Zero label columns.")
    # Try: sigma_mult=1.0 for stricter threshold (flags more of recovery)
    # Try: add a second engram for "incident" and use match() to classify recovery windows


if __name__ == "__main__":
    main()
