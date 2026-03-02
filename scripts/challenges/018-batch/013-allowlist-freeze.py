#!/usr/bin/env python3
"""
Allow-List Freeze Under Poisoning

HYPOTHESIS:
===========
If attack traffic is allowed to update the normal subspace, the manifold
shifts to accommodate attack patterns — the allow list gets poisoned.
Post-attack, the poisoned sub can no longer detect attacks (they project
onto it cleanly) and may also reject legitimate traffic whose manifold
was displaced by attack components.

Gating subspace updates — only learning from requests whose residual
is below a detection threshold, and freezing completely when the anomaly
rate is sustained — preserves the original manifold and maintains both
attack detection and normal acceptance.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable      - Rich structured HTTP/TLS encoding
2. OnlineSubspace.update        - Manifold learning (streaming)
3. OnlineSubspace.residual      - Per-vector anomaly score
4. OnlineSubspace.snapshot      - State export for cloning
5. OnlineSubspace.from_snapshot - State restore (independent clone)

SCENARIO:
=========
Train a normal subspace on 1000 clean browser requests. Clone it twice.
Path A (ungated) continues updating with ALL requests during a 500-request
mixed attack (50% normal, 50% GET flood). Path B (gated) only updates
with requests whose residual is below the detection threshold, and freezes
completely when the anomaly rate in a 50-request window exceeds 30%.
After the attack, score 500 clean normal AND 500 attack requests against
both versions.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon.kernel.encoder import Encoder
from holon.kernel.vector_manager import VectorManager
from holon.kernel.walkable import LinearScale
from holon.memory.subspace import OnlineSubspace

DIM = 4096


def make_normal(rng):
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]))
    path = str(rng.choice(["/", "/api/users", "/api/search", "/products", "/about", "/health"]))
    return {
        "method": str(rng.choice(["GET", "GET", "GET", "POST"])),
        "path": path, "path_parts": path.split("/"), "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Cookie", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", "text/html,application/xhtml+xml,*/*;q=0.8"],
            ["Accept-Language", "en-US,en;q=0.9"],
            ["Accept-Encoding", "gzip, deflate, br"],
            ["Cookie", f"session={rng.integers(1000, 9999)}"],
            ["Connection", "keep-alive"],
        ],
        "header_count": LinearScale(7), "has_cookie": "true",
        "tls": {
            "version": "TLS1.3",
            "ciphers": {"TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                        "TLS_CHACHA20_POLY1305_SHA256"},
            "cipher_order": ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                             "TLS_CHACHA20_POLY1305_SHA256"],
            "ext_types": {"server_name", "supported_versions", "key_share",
                          "signature_algorithms", "supported_groups",
                          "psk_key_exchange_modes"},
            "groups": {"x25519", "secp256r1", "secp384r1"},
            "alpn": ["h2", "http/1.1"],
        },
    }


def make_get_flood(rng):
    ua = str(rng.choice(["curl/8.0.1", "python-requests/2.31.0",
                          "Go-http-client/1.1", "libwww-perl/6.72"]))
    return {
        "method": "GET", "path": "/api/search",
        "path_parts": ["", "api", "search"], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept"],
        "headers": [["Host", "example.com"], ["User-Agent", ua], ["Accept", "*/*"]],
        "header_count": LinearScale(3), "has_cookie": "false",
        "tls": {
            "version": "TLS1.2",
            "ciphers": {"ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"},
            "cipher_order": ["ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"],
            "ext_types": {"server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms"},
            "groups": {"secp256r1"}, "alpn": ["http/1.1"],
        },
    }


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 013: Allow-List Freeze Under Poisoning")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Train baseline normal subspace
    # ===================================================================
    print("\nPHASE 1: Train baseline normal subspace (1000 browser requests)")

    baseline = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    training_residuals = []
    for i in range(1000):
        vec = encoder.encode_walkable(make_normal(np.random.default_rng(i)))
        res = baseline.update(vec)
        if i >= 100:
            training_residuals.append(res)

    training_residuals = np.array(training_residuals)
    baseline_threshold = baseline.threshold
    detect_threshold = np.percentile(training_residuals, 95)

    print(f"  {baseline}")
    print(f"  Adaptive threshold (3.5σ): {baseline_threshold:.4f}")
    print(f"  Detection threshold (p95): {detect_threshold:.4f}")

    # ===================================================================
    # PHASE 2: Clone into two independent paths
    # ===================================================================
    print("\nPHASE 2: Clone baseline into Path A (ungated) and Path B (gated)")

    snap = baseline.snapshot()
    ungated_sub = OnlineSubspace.from_snapshot(snap)
    gated_sub = OnlineSubspace.from_snapshot(snap)

    print(f"  Path A (ungated): n={ungated_sub.n}")
    print(f"  Path B (gated):   n={gated_sub.n}")

    # ===================================================================
    # PHASE 3: Attack phase — 500 mixed requests (50/50 normal + flood)
    # ===================================================================
    print("\nPHASE 3: Attack phase — 500 mixed requests (50% normal, 50% GET flood)")

    WINDOW = 50
    FREEZE_THRESHOLD = 0.3
    window_anomalies = []
    frozen_at = None
    gated_updates = 0
    gated_rejections = 0

    for i in range(500):
        rng = np.random.default_rng(30000 + i)
        is_attack = i % 2 == 1

        if is_attack:
            req = make_get_flood(rng)
        else:
            req = make_normal(rng)

        vec = encoder.encode_walkable(req)

        ungated_sub.update(vec)

        res_b = gated_sub.residual(vec)
        is_anomaly = res_b > detect_threshold
        window_anomalies.append(1 if is_anomaly else 0)
        if len(window_anomalies) > WINDOW:
            window_anomalies.pop(0)

        if frozen_at is None:
            anomaly_rate = sum(window_anomalies) / len(window_anomalies)
            if len(window_anomalies) >= WINDOW and anomaly_rate > FREEZE_THRESHOLD:
                frozen_at = i
                print(f"  *** FREEZE triggered at request {i} "
                      f"(anomaly rate {anomaly_rate:.1%} in last {WINDOW})")
            elif not is_anomaly:
                gated_sub.update(vec)
                gated_updates += 1
            else:
                gated_rejections += 1

    print(f"\n  Gated sub: {gated_updates} updates accepted, "
          f"{gated_rejections} rejected, freeze at {frozen_at}")
    print(f"  Path A post-attack: threshold={ungated_sub.threshold:.4f}, "
          f"n={ungated_sub.n}")
    print(f"  Path B post-attack: threshold={gated_sub.threshold:.4f}, "
          f"n={gated_sub.n}")

    # ===================================================================
    # PHASE 4: Post-attack scoring
    # ===================================================================
    print("\nPHASE 4: Post-attack — score 500 normal + 500 attack requests")

    post_normal_a, post_normal_b = [], []
    post_attack_a, post_attack_b = [], []

    for i in range(500):
        vec = encoder.encode_walkable(
            make_normal(np.random.default_rng(90000 + i)))
        post_normal_a.append(ungated_sub.residual(vec))
        post_normal_b.append(gated_sub.residual(vec))

    for i in range(500):
        vec = encoder.encode_walkable(
            make_get_flood(np.random.default_rng(80000 + i)))
        post_attack_a.append(ungated_sub.residual(vec))
        post_attack_b.append(gated_sub.residual(vec))

    post_normal_a = np.array(post_normal_a)
    post_normal_b = np.array(post_normal_b)
    post_attack_a = np.array(post_attack_a)
    post_attack_b = np.array(post_attack_b)

    fpr_a = float(np.mean(post_normal_a > detect_threshold))
    fpr_b = float(np.mean(post_normal_b > detect_threshold))
    attack_det_a = float(np.mean(post_attack_a > detect_threshold))
    attack_det_b = float(np.mean(post_attack_b > detect_threshold))

    print(f"\n  {'Metric':<40} {'Path A (ungated)':>16} {'Path B (gated)':>16}")
    print("  " + "-" * 74)
    print(f"  {'Normal: mean residual':<40} {np.mean(post_normal_a):>16.4f} "
          f"{np.mean(post_normal_b):>16.4f}")
    print(f"  {'Normal: std residual':<40} {np.std(post_normal_a):>16.4f} "
          f"{np.std(post_normal_b):>16.4f}")
    print(f"  {'Normal: FPR (above detect threshold)':<40} {fpr_a:>15.1%} "
          f"{fpr_b:>15.1%}")
    print(f"  {'Attack: mean residual':<40} {np.mean(post_attack_a):>16.4f} "
          f"{np.mean(post_attack_b):>16.4f}")
    print(f"  {'Attack: detection rate':<40} {attack_det_a:>15.1%} "
          f"{attack_det_b:>15.1%}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  Baseline threshold:         {baseline_threshold:.4f}")
    print(f"  Detection threshold (p95):  {detect_threshold:.4f}")
    print(f"  Freeze triggered at:        request {frozen_at}")
    print(f"  Path A (ungated) FPR:       {fpr_a:.1%}")
    print(f"  Path B (gated) FPR:         {fpr_b:.1%}")
    print(f"  Path A attack detection:    {attack_det_a:.1%}")
    print(f"  Path B attack detection:    {attack_det_b:.1%}")

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    frozen_fpr_ok = fpr_b < 0.05
    frozen_beats_ungated = fpr_b <= fpr_a
    gated_detects_attacks = attack_det_b > 0.50
    ungated_lost_detection = attack_det_a < attack_det_b
    freeze_early = frozen_at is not None and frozen_at < 100

    checks = [
        (
            "Gated FPR < 5% on post-attack normal traffic",
            frozen_fpr_ok,
            f"FPR={fpr_b:.1%}",
        ),
        (
            "Gated FPR <= ungated FPR (gating helps or neutral)",
            frozen_beats_ungated,
            f"gated={fpr_b:.1%}, ungated={fpr_a:.1%}",
        ),
        (
            "Gated sub still detects attacks post-attack (rate > 50%)",
            gated_detects_attacks,
            f"rate={attack_det_b:.1%}",
        ),
        (
            "Ungated lost attack detection vs gated (poisoning effect)",
            ungated_lost_detection,
            f"ungated={attack_det_a:.1%}, gated={attack_det_b:.1%}",
        ),
        (
            "Freeze triggers within first 100 attack requests",
            freeze_early,
            f"frozen_at={frozen_at}",
        ),
    ]

    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc} ({detail})")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
