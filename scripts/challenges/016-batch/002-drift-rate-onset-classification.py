#!/usr/bin/env python3
"""
Drift Rate for Attack Onset Classification

HYPOTHESIS:
===========
drift_rate() gives the temporal derivative of similarity. Different attack
types produce different drift shapes:
  - Flash flood: massive negative spike
  - Ramp-up: accelerating negative drift
  - Organic growth: slow, steady drift
  - Pulsed attack: oscillating drift

Can we classify ATTACK TYPE purely from the drift rate shape?

PRIMITIVES DEMONSTRATED:
========================
1. drift_rate()     - Temporal derivative of similarity
2. significance()   - Principled threshold for drift magnitude
3. coherence()      - Corroboration signal

VECTOR PROPERTIES EXPLOITED:
============================
- Temporal dynamics (rate of change, not just state)
- Derivative classification (shape of change curve)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.distance import significance
from holon.primitives import coherence, drift_rate


def make_packet(client, src_ip, dst_ip, proto, src_port, dst_port, pkt_len):
    return client.encode(
        {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "proto": proto,
            "src_port": src_port,
            "dst_port": dst_port,
            "pkt_len": pkt_len,
        }
    )


def generate_stream_normal(client, rng, windows=20, per_window=30):
    """Steady normal traffic across multiple windows."""
    stream = []
    protos = ["TCP", "UDP"]
    ports = [80, 443, 22, 8080]
    for _ in range(windows):
        vecs = []
        for _ in range(per_window):
            vecs.append(
                make_packet(
                    client,
                    src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                    dst_ip=f"192.168.1.{rng.integers(1, 5)}",
                    proto=rng.choice(protos),
                    src_port=int(rng.integers(1024, 65535)),
                    dst_port=int(rng.choice(ports)),
                    pkt_len=int(rng.integers(64, 1500)),
                )
            )
        from holon.primitives import bundle

        stream.append(bundle(vecs))
    return stream


def generate_stream_flash_flood(client, rng, windows=20, per_window=30):
    """Normal traffic then sudden attack at window 10."""
    stream = []
    for w in range(windows):
        vecs = []
        for _ in range(per_window):
            if w >= 10:
                # Sudden DNS amplification
                vecs.append(
                    make_packet(
                        client,
                        src_ip="8.8.8.8",
                        dst_ip="192.168.1.100",
                        proto="UDP",
                        src_port=53,
                        dst_port=int(rng.integers(1024, 65535)),
                        pkt_len=1400,
                    )
                )
            else:
                vecs.append(
                    make_packet(
                        client,
                        src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                        dst_ip=f"192.168.1.{rng.integers(1, 5)}",
                        proto=rng.choice(["TCP", "UDP"]),
                        src_port=int(rng.integers(1024, 65535)),
                        dst_port=int(rng.choice([80, 443])),
                        pkt_len=int(rng.integers(64, 1500)),
                    )
                )
        from holon.primitives import bundle

        stream.append(bundle(vecs))
    return stream


def generate_stream_ramp_up(client, rng, windows=20, per_window=30):
    """Gradually increasing attack fraction from 0% to 100%."""
    stream = []
    for w in range(windows):
        attack_frac = max(0.0, (w - 5) / 15.0)  # ramps from window 5 to 20
        vecs = []
        for _ in range(per_window):
            if rng.random() < attack_frac:
                vecs.append(
                    make_packet(
                        client,
                        src_ip="8.8.8.8",
                        dst_ip="192.168.1.100",
                        proto="UDP",
                        src_port=53,
                        dst_port=int(rng.integers(1024, 65535)),
                        pkt_len=1400,
                    )
                )
            else:
                vecs.append(
                    make_packet(
                        client,
                        src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                        dst_ip=f"192.168.1.{rng.integers(1, 5)}",
                        proto=rng.choice(["TCP", "UDP"]),
                        src_port=int(rng.integers(1024, 65535)),
                        dst_port=int(rng.choice([80, 443])),
                        pkt_len=int(rng.integers(64, 1500)),
                    )
                )
        from holon.primitives import bundle

        stream.append(bundle(vecs))
    return stream


def generate_stream_pulsed(client, rng, windows=20, per_window=30):
    """Attack on/off every 2 windows."""
    stream = []
    for w in range(windows):
        is_attack = (w // 2) % 2 == 1 and w >= 4  # pulses start at window 4
        vecs = []
        for _ in range(per_window):
            if is_attack:
                vecs.append(
                    make_packet(
                        client,
                        src_ip=f"45.33.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                        dst_ip="192.168.1.100",
                        proto="TCP",
                        src_port=int(rng.integers(1024, 65535)),
                        dst_port=80,
                        pkt_len=60,
                    )
                )
            else:
                vecs.append(
                    make_packet(
                        client,
                        src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                        dst_ip=f"192.168.1.{rng.integers(1, 5)}",
                        proto=rng.choice(["TCP", "UDP"]),
                        src_port=int(rng.integers(1024, 65535)),
                        dst_port=int(rng.choice([80, 443])),
                        pkt_len=int(rng.integers(64, 1500)),
                    )
                )
        from holon.primitives import bundle

        stream.append(bundle(vecs))
    return stream


def classify_drift(rates):
    """Classify attack type from drift rate shape."""
    if not rates:
        return "INSUFFICIENT_DATA"

    min_rate = min(rates)
    max_abs = max(abs(r) for r in rates)

    # Count sign changes for oscillation
    sign_changes = sum(
        1 for i in range(1, len(rates)) if rates[i] * rates[i - 1] < 0
    )

    # Check for acceleration (consecutive negative values getting worse)
    neg_streak = 0
    max_neg_streak = 0
    for r in rates:
        if r < -0.05:
            neg_streak += 1
            max_neg_streak = max(max_neg_streak, neg_streak)
        else:
            neg_streak = 0

    if max_abs < 0.05:
        return "STABLE"
    elif min_rate < -0.5:
        return "FLASH_FLOOD"
    elif max_neg_streak >= 4:
        return "RAMP_UP"
    elif sign_changes >= len(rates) * 0.4:
        return "PULSED"
    elif min_rate < -0.1:
        return "GRADUAL_SHIFT"
    else:
        return "STABLE"


def print_drift_chart(rates, label, width=50):
    """ASCII visualization of drift rate over time."""
    if not rates:
        return
    print(f"\n  {label}:")
    max_abs = max(abs(r) for r in rates) if rates else 1.0
    max_abs = max(max_abs, 0.01)
    for i, r in enumerate(rates):
        bar_len = int(abs(r) / max_abs * (width // 2))
        if r >= 0:
            bar = " " * (width // 2) + "│" + "█" * bar_len
        else:
            bar = " " * (width // 2 - bar_len) + "█" * bar_len + "│"
        print(f"    w{i:02d} {bar} {r:+.4f}")


def main():
    client = HolonClient(dimensions=4096)

    print("=" * 70)
    print("EXPERIMENT 1: Drift rate shapes across attack types")
    print("=" * 70)

    scenarios = [
        ("Normal (stable)", generate_stream_normal, "STABLE"),
        ("Flash Flood", generate_stream_flash_flood, "FLASH_FLOOD"),
        ("Ramp-Up", generate_stream_ramp_up, "RAMP_UP"),
        ("Pulsed Attack", generate_stream_pulsed, "PULSED"),
    ]

    results = []
    for label, gen_fn, expected in scenarios:
        stream = gen_fn(client, np.random.default_rng(42), windows=20, per_window=30)
        rates = drift_rate(stream, window=1)
        classification = classify_drift(rates)
        correct = classification == expected
        results.append((label, expected, classification, correct, rates))

        print_drift_chart(rates, label)
        print(f"    Classification: {classification} (expected: {expected}) {'✓' if correct else '✗'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Scenario':<25} {'Expected':<15} {'Got':<15} {'Result':>8}")
    print("-" * 65)
    for label, expected, got, correct, _ in results:
        print(f"{label:<25} {expected:<15} {got:<15} {'PASS' if correct else 'FAIL':>8}")

    accuracy = sum(1 for _, _, _, c, _ in results if c) / len(results)
    print(f"\nClassification accuracy: {accuracy:.0%}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Windowed smoothing effect")
    print("=" * 70)

    stream = generate_stream_flash_flood(client, np.random.default_rng(42), windows=20, per_window=30)
    for window in [1, 3, 5]:
        rates = drift_rate(stream, window=window)
        print(f"\n  Window={window}: min_rate={min(rates):.4f}, max_abs={max(abs(r) for r in rates):.4f}")
        classification = classify_drift(rates)
        print(f"  Classification: {classification}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Drift rate + coherence combined signal")
    print("=" * 70)

    from holon.primitives import bundle

    for label, gen_fn, _ in scenarios:
        stream = gen_fn(client, np.random.default_rng(42), windows=20, per_window=30)
        rates = drift_rate(stream, window=1)

        # Also measure coherence at key windows
        mid_stream = gen_fn(client, np.random.default_rng(42), windows=20, per_window=30)
        # We only have bundled vectors, so coherence would need raw vecs
        # Use max absolute drift rate as the dynamic signal
        max_drift = max(abs(r) for r in rates) if rates else 0
        mean_drift = np.mean(rates) if rates else 0

        print(f"  {label:<25}: max|drift|={max_drift:.4f}, mean_drift={mean_drift:+.4f}")

    print()


if __name__ == "__main__":
    main()
