#!/usr/bin/env python3
"""
Temporal Evolution of Eigenvalue Fingerprint

HYPOTHESIS:
===========
The eigenvalue spectrum evolves distinctly across attack lifecycle phases —
onset, peak, and subsidence. Each phase produces a different spectral shape,
and the evolution trajectory itself is a temporal fingerprint.

Normal → ramp-up → full attack → subsidence → recovery should trace a path
through spectrum space. Cosine similarity to reference engrams (normal vs
attack) should track the phase transitions with measurable hysteresis on
the decay side.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable     - Rich structured HTTP request encoding
2. OnlineSubspace.eigenvalues  - Per-window eigenvalue extraction
3. Spectral shape comparison   - Reference matching at each time step
4. Temporal trajectory          - Phase-resolved spectral evolution

SCENARIO:
=========
WAF context. Simulate a complete 800-request attack lifecycle across 5
phases: normal → ramp-up → full attack → subsidence → recovery. Build
reference engrams for normal and attack traffic. Slide a 50-request window
every 25 steps and track cosine similarity to both references.
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
    path = str(rng.choice(["/", "/api/users", "/api/search", "/products",
                            "/about", "/health"]))
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
        "headers": [["Host", "example.com"], ["User-Agent", ua],
                    ["Accept", "*/*"]],
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


def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def spectral_shape(eigs):
    """Rank-weighted log-compression for distributional comparison.

    Raw eigenvalue cosine similarity is dominated by the top few
    components (which are similar for all traffic types) and
    compresses into [0.8, 1.0].  Two transforms fix this:
    1. log1p compresses the dynamic range
    2. rank-weighting (component_index) elevates the tail where
       normal traffic (64 active) and attack (11 active) truly differ
    """
    e = np.maximum(np.asarray(eigs, dtype=float), 0.0)
    weights = np.arange(1, len(e) + 1, dtype=float)
    return np.log1p(e) * weights


PHASES = [
    (0, 200, "normal_pre",  0.0, 0.0),
    (200, 300, "ramp_up",   0.2, 0.8),
    (300, 500, "full_attack", 1.0, 1.0),
    (500, 600, "subsidence", 0.8, 0.0),
    (600, 800, "normal_post", 0.0, 0.0),
]


def attack_fraction(req_idx):
    for start, end, _, frac_start, frac_end in PHASES:
        if start <= req_idx < end:
            t = (req_idx - start) / (end - start)
            return frac_start + (frac_end - frac_start) * t
    return 0.0


def phase_label(req_idx):
    for start, end, label, _, _ in PHASES:
        if start <= req_idx < end:
            return label
    return "unknown"


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 006: Temporal Evolution of Eigenvalue Fingerprint")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Build reference engrams
    # ===================================================================
    print("\nPHASE 1: Build reference engrams (normal + attack)")

    normal_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(500):
        vec = encoder.encode_walkable(make_normal(np.random.default_rng(i)))
        normal_sub.update(vec)
    normal_eigenvalues = spectral_shape(normal_sub.eigenvalues)
    print(f"  Normal engram: {normal_sub}")

    attack_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(300):
        vec = encoder.encode_walkable(
            make_get_flood(np.random.default_rng(i + 50000)))
        attack_sub.update(vec)
    attack_eigenvalues = spectral_shape(attack_sub.eigenvalues)
    print(f"  Attack engram: {attack_sub}")

    # ===================================================================
    # PHASE 2: Stream 800 requests across 5 lifecycle phases
    # ===================================================================
    print("\nPHASE 2: Stream 800 requests across attack lifecycle")
    print(f"  Window: 50 requests, sampled every 25 steps")
    print(f"  Phases: normal_pre → ramp_up → full_attack → subsidence → normal_post\n")

    total_requests = 800
    window_size = 50
    step_size = 25

    all_vectors = []
    for i in range(total_requests):
        rng = np.random.default_rng(i + 100000)
        af = attack_fraction(i)
        if rng.random() < af:
            d = make_get_flood(rng)
        else:
            d = make_normal(rng)
        all_vectors.append(encoder.encode_walkable(d))

    windows = []
    print(f"  {'Window':>6} {'Pos':>5} {'Phase':<14} {'AttackSim':>10} {'NormalSim':>10}")
    print("  " + "-" * 50)

    for w_idx, w_start in enumerate(range(0, total_requests - window_size + 1, step_size)):
        w_end = w_start + window_size
        mid = (w_start + w_end) // 2

        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for vec in all_vectors[w_start:w_end]:
            sub.update(vec)

        eigs = spectral_shape(sub.eigenvalues)
        attack_sim = cosine_similarity(eigs, attack_eigenvalues)
        normal_sim = cosine_similarity(eigs, normal_eigenvalues)
        label = phase_label(mid)

        windows.append({
            "idx": w_idx, "start": w_start, "end": w_end,
            "mid": mid, "phase": label,
            "attack_sim": attack_sim, "normal_sim": normal_sim,
        })

        if w_idx % 4 == 0 or w_idx == 0:
            print(f"  {w_idx:>6} {mid:>5} {label:<14} "
                  f"{attack_sim:>10.4f} {normal_sim:>10.4f}")

    # ===================================================================
    # PHASE 3: Analyze by lifecycle phase
    # ===================================================================
    print("\n" + "-" * 70)
    print("RESULTS: Per-phase spectrum similarity")
    print("-" * 70)

    phase_names = ["normal_pre", "ramp_up", "full_attack", "subsidence", "normal_post"]
    phase_attack_sims = {p: [] for p in phase_names}
    phase_normal_sims = {p: [] for p in phase_names}

    for w in windows:
        phase_attack_sims[w["phase"]].append(w["attack_sim"])
        phase_normal_sims[w["phase"]].append(w["normal_sim"])

    print(f"\n  {'Phase':<14} {'AttackSim':>10} {'NormalSim':>10} {'Windows':>8}")
    print("  " + "-" * 45)

    phase_means_attack = {}
    phase_means_normal = {}
    for p in phase_names:
        a_sims = phase_attack_sims[p]
        n_sims = phase_normal_sims[p]
        if a_sims:
            ma = np.mean(a_sims)
            mn = np.mean(n_sims)
            phase_means_attack[p] = ma
            phase_means_normal[p] = mn
            print(f"  {p:<14} {ma:>10.4f} {mn:>10.4f} {len(a_sims):>8}")

    # ===================================================================
    # PHASE 4: Transition detection — slope changes
    # ===================================================================
    print("\n" + "-" * 70)
    print("ANALYSIS: Transition detection")
    print("-" * 70)

    attack_sims_series = [w["attack_sim"] for w in windows]
    slopes = []
    for i in range(1, len(attack_sims_series)):
        slopes.append(attack_sims_series[i] - attack_sims_series[i - 1])

    sign_changes = []
    for i in range(1, len(slopes)):
        if slopes[i - 1] * slopes[i] < 0:
            sign_changes.append(windows[i + 1]["mid"])

    print(f"  Attack similarity slope sign changes at positions: "
          f"{sign_changes[:10]}")

    phase_boundaries = [200, 300, 500, 600]
    detected_near_boundary = []
    for boundary in phase_boundaries:
        near = [sc for sc in sign_changes if abs(sc - boundary) < 75]
        if near:
            detected_near_boundary.append(boundary)
    print(f"  Phase boundaries detected (within 75 req): "
          f"{detected_near_boundary} / {phase_boundaries}")

    # ===================================================================
    # PHASE 5: Hysteresis check
    # ===================================================================
    ramp_windows = [w for w in windows if w["phase"] == "ramp_up"]
    sub_windows = [w for w in windows if w["phase"] == "subsidence"]

    ramp_start_attack = ramp_windows[0]["attack_sim"] if ramp_windows else 0.0
    sub_end_attack = sub_windows[-1]["attack_sim"] if sub_windows else 0.0

    print(f"\n  Hysteresis:")
    print(f"    Attack sim at ramp-up start:   {ramp_start_attack:.4f}")
    print(f"    Attack sim at subsidence end:   {sub_end_attack:.4f}")
    print(f"    Subsidence end > ramp start:    {sub_end_attack > ramp_start_attack}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    peak_attack_sim = phase_means_attack.get("full_attack", 0.0)
    pre_attack_sim = phase_means_attack.get("normal_pre", 1.0)
    post_normal_sim = phase_means_normal.get("normal_post", 0.0)
    pre_normal_sim = phase_means_normal.get("normal_pre", 0.0)

    checks = [
        (
            "Attack similarity peaks during full_attack phase",
            peak_attack_sim == max(phase_means_attack.values()),
            f"full_attack={peak_attack_sim:.4f}, "
            f"max={max(phase_means_attack.values()):.4f}",
        ),
        (
            "Normal similarity highest in normal phases (pre or post)",
            max(pre_normal_sim, post_normal_sim)
            >= max(phase_means_normal.get("ramp_up", 1.0),
                   phase_means_normal.get("full_attack", 1.0),
                   phase_means_normal.get("subsidence", 1.0)),
            f"pre={pre_normal_sim:.4f}, post={post_normal_sim:.4f}",
        ),
        (
            "Phase transitions detectable via slope sign changes",
            len(detected_near_boundary) >= 2,
            f"detected {len(detected_near_boundary)}/4 boundaries",
        ),
        (
            "Attack similarity in normal_pre < 0.5 (pre-attack baseline)",
            pre_attack_sim < 0.5,
            f"normal_pre attack_sim={pre_attack_sim:.4f}",
        ),
        (
            "Attack similarity in full_attack > 0.7",
            peak_attack_sim > 0.7,
            f"full_attack attack_sim={peak_attack_sim:.4f}",
        ),
        (
            "Hysteresis: subsidence end > ramp-up start",
            sub_end_attack > ramp_start_attack,
            f"sub_end={sub_end_attack:.4f} vs ramp_start={ramp_start_attack:.4f}",
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
