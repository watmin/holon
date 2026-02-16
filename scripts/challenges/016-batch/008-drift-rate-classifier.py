#!/usr/bin/env python3
"""
Improved Drift Rate Classification

PROBLEM:
========
Experiment 002 showed drift rate SHAPES are visually distinctive for different
attack onset types, but the naive threshold-based classifier achieved only 25%
accuracy. We need better feature extraction from drift rate time series.

HYPOTHESIS:
===========
Instead of simple threshold rules, extract statistical features from the drift
rate time series and use proper decision boundaries:
  - Peak count (pulsed attacks have multiple peaks)
  - Max absolute drift (flash floods have huge spikes)
  - Monotonicity / trend (ramp-ups show increasing drift)
  - Variance (stable traffic has low variance)
  - Zero-crossing rate (pulsed attacks oscillate)

APPROACH:
=========
1. Generate drift rate time series for each attack onset type
2. Extract statistical features from each series
3. Build a simple decision tree (by hand) using those features
4. Evaluate accuracy over many trials

CONSTRAINT:
===========
All stateless, per-window. No flow tracking.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import drift_rate


def make_packet(client, rng, src_ip=None, dst_ip=None, proto=None,
                src_port=None, dst_port=None, pkt_len=None, ttl=None):
    return client.encode({
        "src_ip": src_ip or f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
        "dst_ip": dst_ip or f"192.168.1.{rng.integers(1, 10)}",
        "proto": proto or str(rng.choice(["TCP", "UDP"])),
        "src_port": str(src_port if src_port is not None else int(rng.integers(1024, 65535))),
        "dst_port": str(dst_port if dst_port is not None else int(rng.choice([80, 443, 22, 8080]))),
        "pkt_len": str(pkt_len if pkt_len is not None else int(rng.integers(64, 1500))),
        "ttl": str(ttl if ttl is not None else int(rng.choice([64, 128, 255]))),
    })


def normal_packet(client, rng):
    return make_packet(client, rng)


def attack_packet(client, rng):
    return make_packet(client, rng, src_ip=rng.choice(["8.8.8.8", "1.1.1.1"]),
                       proto="UDP", src_port=53,
                       pkt_len=int(rng.integers(512, 4096)),
                       ttl=int(rng.choice([240, 245, 250])))


def generate_onset_stream(client, rng, onset_type, windows=20, win_size=30):
    """Generate a stream of windows with a specific onset pattern."""
    windows_out = []

    if onset_type == "stable":
        for _ in range(windows):
            vecs = [normal_packet(client, rng) for _ in range(win_size)]
            windows_out.append(vecs)

    elif onset_type == "flash_flood":
        for w in range(windows):
            if w < 8:
                vecs = [normal_packet(client, rng) for _ in range(win_size)]
            else:
                vecs = [attack_packet(client, rng) for _ in range(win_size)]
            windows_out.append(vecs)

    elif onset_type == "ramp_up":
        for w in range(windows):
            frac = max(0.0, (w - 5) / 15.0)
            frac = min(frac, 1.0)
            vecs = []
            for _ in range(win_size):
                if rng.random() < frac:
                    vecs.append(attack_packet(client, rng))
                else:
                    vecs.append(normal_packet(client, rng))
            windows_out.append(vecs)

    elif onset_type == "pulsed":
        for w in range(windows):
            if w >= 6 and (w - 6) % 4 < 2:
                vecs = [attack_packet(client, rng) for _ in range(win_size)]
            else:
                vecs = [normal_packet(client, rng) for _ in range(win_size)]
            windows_out.append(vecs)

    return windows_out


def compute_drift_series(windows):
    """Compute centroid per window, then drift rates."""
    centroids = []
    for win in windows:
        centroid = np.mean(win, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids.append(centroid)
    return drift_rate(centroids)


def extract_features(drifts):
    """Extract statistical features from a drift rate time series."""
    drifts = np.array(drifts)
    abs_drifts = np.abs(drifts)

    # Basic stats
    max_abs = np.max(abs_drifts)
    mean_abs = np.mean(abs_drifts)
    std_drift = np.std(drifts)
    variance = np.var(drifts)

    # Peak detection (values > 2σ above mean)
    peak_threshold = mean_abs + 2 * std_drift
    peaks = abs_drifts > peak_threshold
    peak_count = int(np.sum(peaks))

    # Zero-crossing rate
    signs = np.sign(drifts)
    crossings = np.sum(signs[1:] != signs[:-1])
    crossing_rate = crossings / max(len(drifts) - 1, 1)

    # Monotonicity: correlation with linear trend
    x = np.arange(len(drifts))
    if np.std(drifts) > 1e-10:
        trend_corr = np.corrcoef(x, drifts)[0, 1]
    else:
        trend_corr = 0.0

    # Late half vs early half energy ratio
    mid = len(drifts) // 2
    early_energy = np.mean(abs_drifts[:mid]) + 1e-12
    late_energy = np.mean(abs_drifts[mid:]) + 1e-12
    energy_ratio = late_energy / early_energy

    # Spike sharpness: max / mean ratio
    spike_ratio = max_abs / (mean_abs + 1e-12)

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "std": std_drift,
        "variance": variance,
        "peak_count": peak_count,
        "crossing_rate": crossing_rate,
        "trend_corr": trend_corr,
        "energy_ratio": energy_ratio,
        "spike_ratio": spike_ratio,
    }


def classify_onset(features):
    """Decision tree classifier for attack onset type.

    Based on observed feature distributions:
      - pulsed:      mean_abs ~0.41 (>>), crossing_rate ~0.91 (high)
      - flash_flood: max_abs ~0.58, spike_ratio ~7.9 (very high), mean_abs ~0.07
      - stable:      max_abs ~0.07, mean_abs ~0.026
      - ramp_up:     max_abs ~0.08, mean_abs ~0.027, trend_corr ~0.16
    """
    max_abs = features["max_abs"]
    mean_abs = features["mean_abs"]
    spike_ratio = features["spike_ratio"]
    crossing_rate = features["crossing_rate"]
    trend_corr = features["trend_corr"]

    # Pulsed: sustained high drift throughout the series (mean_abs >> others)
    if mean_abs > 0.15:
        return "pulsed"

    # Flash flood: one huge spike relative to background (spike_ratio >> 4)
    if spike_ratio > 5.0 and max_abs > 0.3:
        return "flash_flood"

    # Between stable and ramp_up: both have low max_abs and mean_abs
    # Ramp-up has positive trend correlation
    if trend_corr > 0.10:
        return "ramp_up"

    return "stable"


def main():
    client = HolonClient(dimensions=4096)
    onset_types = ["stable", "flash_flood", "ramp_up", "pulsed"]

    print("=" * 80)
    print("EXPERIMENT 1: Feature extraction from drift rate series")
    print("=" * 80)
    print()

    for onset in onset_types:
        windows = generate_onset_stream(
            client, np.random.default_rng(42), onset, windows=20, win_size=30
        )
        drifts = compute_drift_series(windows)
        features = extract_features(drifts)

        print(f"  {onset}:")
        for k, v in features.items():
            print(f"    {k:>16}: {v:.4f}")
        print(f"    {'drift_series':>16}: [{', '.join(f'{d:.3f}' for d in drifts[:10])}...]")
        print()

    print("=" * 80)
    print("EXPERIMENT 2: Classification accuracy over many trials")
    print("=" * 80)
    print()

    n_trials = 50
    results = {onset: {"correct": 0, "total": 0, "predictions": {}} for onset in onset_types}

    for trial in range(n_trials):
        for onset in onset_types:
            windows = generate_onset_stream(
                client, np.random.default_rng(trial * 100 + hash(onset) % 1000),
                onset, windows=20, win_size=30
            )
            drifts = compute_drift_series(windows)
            features = extract_features(drifts)
            predicted = classify_onset(features)

            results[onset]["total"] += 1
            if predicted == onset:
                results[onset]["correct"] += 1

            results[onset]["predictions"].setdefault(predicted, 0)
            results[onset]["predictions"][predicted] += 1

    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':>12} " + " ".join(f"{o:>12}" for o in onset_types))
    print(f"  {'-' * (14 + 13 * len(onset_types))}")

    total_correct = 0
    total_all = 0
    for actual in onset_types:
        row = f"  {actual:>12} "
        for predicted in onset_types:
            count = results[actual]["predictions"].get(predicted, 0)
            row += f"{count:>12}"
        accuracy = results[actual]["correct"] / results[actual]["total"] * 100
        row += f"  ({accuracy:.0f}%)"
        print(row)
        total_correct += results[actual]["correct"]
        total_all += results[actual]["total"]

    print(f"\n  Overall accuracy: {total_correct}/{total_all} ({total_correct/total_all*100:.1f}%)")

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Feature distributions — separation analysis")
    print("=" * 80)
    print()
    print("  For each feature, show mean±std across 50 trials per onset type:")
    print()

    feature_names = ["max_abs", "mean_abs", "std", "peak_count", "crossing_rate",
                     "trend_corr", "energy_ratio", "spike_ratio"]

    all_features = {onset: {f: [] for f in feature_names} for onset in onset_types}

    for trial in range(50):
        for onset in onset_types:
            windows = generate_onset_stream(
                client, np.random.default_rng(trial * 100 + hash(onset) % 1000),
                onset, windows=20, win_size=30
            )
            drifts = compute_drift_series(windows)
            features = extract_features(drifts)
            for f in feature_names:
                all_features[onset][f].append(features[f])

    for feat in feature_names:
        print(f"  {feat}:")
        for onset in onset_types:
            vals = all_features[onset][feat]
            print(f"    {onset:>12}: {np.mean(vals):>8.4f} ± {np.std(vals):>7.4f}  "
                  f"[{np.min(vals):>7.4f}, {np.max(vals):>7.4f}]")
        print()

    print()


if __name__ == "__main__":
    main()
