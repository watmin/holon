#!/usr/bin/env python3
"""
Time-Series Recall & Forecast - Learn Patterns, Predict Futures

Encode sensor sequences with the $time marker so timestamps carry
circular (hour-of-day) structure, learn one subspace per named pattern,
then match a partial sequence to the correct engram and forecast
the next 3 states.

Run with: ./scripts/run_with_venv.sh python -m examples.showcases.timeseries_recall_forecast.showcase
"""

import random

from holon.kernel import Encoder, ListEncodeMode, VectorManager
from holon.kernel.walkable import LogScale, TimeScale
from holon.memory import EngramLibrary, OnlineSubspace

DIM = 4096
BASE_TS = 1_700_000_000  # 2023-11-14 00:00 UTC, 1-minute intervals


def _state(temp):
    return "normal" if temp < 30 else ("warning" if temp < 40 else "critical")


def _step(t, temp):
    return {
        "timestamp": TimeScale(BASE_TS + t * 60),
        "value": LogScale(temp),
        "state": _state(temp),
    }


def gradual_rise(rng, n=50):
    """Warm start (~35°C, warning), climbs steadily — already critical by step 10."""
    temp, seq = 35.0, []
    for t in range(n):
        temp += rng.uniform(0.3, 0.8)
        seq.append(_step(t, temp))
    future = [_state(temp + rng.uniform(0.3, 0.8) * i) for i in range(1, 4)]
    return seq, future


def spike_recovery(rng, n=50):
    """Cool start (~20°C), sharp spike in steps 5–15, then cools back below 30."""
    temp, seq = 20.0, []
    for t in range(n):
        if 5 <= t <= 15:
            temp += rng.uniform(4.0, 6.0)
        else:
            temp = max(18.0, temp - rng.uniform(1.5, 3.0))
        seq.append(_step(t, temp))
    future = [_state(max(18.0, temp - rng.uniform(1.5, 3.0) * i)) for i in range(1, 4)]
    return seq, future


def steady_state(rng, n=50):
    """Stays tightly at room temperature (~22°C ± 0.5, always normal)."""
    temp, seq = 22.0, []
    for t in range(n):
        temp = max(20.0, min(temp + rng.uniform(-0.5, 0.5), 25.0))
        seq.append(_step(t, temp))
    return seq, ["normal", "normal", "normal"]


def encode_sequence(enc, steps):
    return enc.encode_list(
        [enc.encode_data(s) for s in steps],
        mode=ListEncodeMode.POSITIONAL,
    )


def main():
    print("=" * 65)
    print("TIME-SERIES RECALL & FORECAST")
    print("=" * 65)

    rng = random.Random(42)
    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)
    library = EngramLibrary(dim=DIM)

    patterns = {
        "gradual_rise": gradual_rise,  # 35°C warning → monotonic climb to critical
        "spike_recovery": spike_recovery,  # 20°C → sharp spike → cools back to normal
        "steady_state": steady_state,  # 22°C normal, always stable
    }

    # ── Train one subspace per pattern ─────────────────────────────
    # 10 examples × 5 passes → converged manifold, no label leaking
    print("\nLearning pattern subspaces (10 examples × 5 passes each)...")
    for name, gen_fn in patterns.items():
        examples = [gen_fn(rng) for _ in range(10)]
        vecs = [encode_sequence(enc, seq) for seq, _ in examples]

        step_votes = [[future[i] for _, future in examples] for i in range(3)]
        forecast3 = [max(set(v), key=v.count) for v in step_votes]

        subspace = OnlineSubspace(dim=DIM, k=32, amnesia=2.0)
        for _ in range(5):
            for v in vecs:
                subspace.update(v)

        library.add(name, subspace, forecast3=forecast3)
        print(f"  '{name}'  → next 3 states: {forecast3}")

    # ── Recall from partial sequences (first 25 of 50 steps) ───────
    print("\n" + "-" * 65)
    print("PATTERN RECALL  (partial sequence, steps 0-24 of 0-49)")
    print("-" * 65)

    tests = [
        ("gradual_rise", gradual_rise),
        ("spike_recovery", spike_recovery),
        ("steady_state", steady_state),
    ]
    for expected_name, gen_fn in tests:
        partial, actual_future = gen_fn(rng, n=25)
        vec = encode_sequence(enc, partial)
        matches = library.match(vec, top_k=3)

        print(
            f"\n  Input   : '{expected_name}'  (current state={partial[-1]['state']})"
        )
        print("  Matches :")
        for rank, (match_name, residual) in enumerate(matches, 1):
            fc3 = library.get(match_name).metadata.get("forecast3", ["?", "?", "?"])
            marker = " ← best" if rank == 1 else ""
            print(
                f"    {rank}. '{match_name}'  residual={residual:.2f}  "
                f"forecast={fc3}{marker}"
            )

        best = matches[0][0]
        fc3 = library.get(best).metadata.get("forecast3", ["?", "?", "?"])
        correct = best == expected_name
        print(f"  Forecast: {fc3}   actual={actual_future}   correct={correct}")

    print(f"\n{'=' * 65}")
    # Try: reduce to 10 steps as partial to see where recall starts breaking down.


if __name__ == "__main__":
    main()
