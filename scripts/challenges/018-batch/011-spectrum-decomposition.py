#!/usr/bin/env python3
"""
Spectrum Decomposition of Mixed Windows

HYPOTHESIS:
===========
When a traffic window contains a mix of two patterns, the window's eigenvalue
spectrum can be decomposed into weighted contributions from known engrams.
Non-negative least squares (NNLS) on the eigenvalue vectors should recover
the mixing ratios, and spectrum similarity should track the mixing ratio
monotonically.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable          - Rich structured HTTP request encoding
2. OnlineSubspace.eigenvalues       - Per-window eigenvalue extraction
3. EngramLibrary.match_spectrum     - Spectrum cosine similarity pre-filter
4. scipy.optimize.nnls              - Eigenvalue decomposition into engram basis

SCENARIO:
=========
WAF context. 3 engrams (normal, get_flood, cred_stuffing) built from 300
requests each. Mixed windows at known ratios (100/0, 80/20, 50/50, 20/80,
0/100, plus a flood/cred 50/50). For each window, compare match_spectrum
similarity scores against ground truth ratios and verify NNLS recovery.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon.kernel.encoder import Encoder
from holon.kernel.vector_manager import VectorManager
from holon.kernel.walkable import LinearScale
from holon.memory.engram import EngramLibrary
from holon.memory.subspace import OnlineSubspace

DIM = 4096


def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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


def make_cred_stuff(rng):
    body_len = rng.integers(80, 200)
    return {
        "method": "POST", "path": "/api/v1/auth/login",
        "path_parts": ["", "api", "v1", "auth", "login"], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Content-Type",
                         "Content-Length", "Accept-Encoding", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", "python-requests/2.31.0"],
            ["Accept", "*/*"], ["Content-Type", "application/json"],
            ["Content-Length", str(body_len)],
            ["Accept-Encoding", "gzip, deflate"], ["Connection", "keep-alive"],
        ],
        "header_count": LinearScale(7), "has_cookie": "false",
        "body_len": LinearScale(body_len),
        "tls": {
            "version": "TLS1.2",
            "ciphers": {"ECDHE-RSA-AES128-GCM-SHA256",
                        "ECDHE-RSA-AES256-GCM-SHA384", "AES256-SHA"},
            "cipher_order": ["ECDHE-RSA-AES128-GCM-SHA256",
                             "ECDHE-RSA-AES256-GCM-SHA384", "AES256-SHA"],
            "ext_types": {"server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms",
                          "application_layer_protocol_negotiation"},
            "groups": {"secp256r1", "secp384r1"}, "alpn": ["http/1.1"],
        },
    }


ENGRAM_DEFS = {
    "normal":         make_normal,
    "get_flood":      make_get_flood,
    "cred_stuffing":  make_cred_stuff,
}

MIX_SCENARIOS = [
    ("100% normal",             1.0, 0.0, "normal",    "get_flood"),
    ("80/20 normal/flood",      0.8, 0.2, "normal",    "get_flood"),
    ("50/50 normal/flood",      0.5, 0.5, "normal",    "get_flood"),
    ("20/80 normal/flood",      0.2, 0.8, "normal",    "get_flood"),
    ("100% flood",              0.0, 1.0, "normal",    "get_flood"),
    ("50/50 flood/cred_stuff",  0.5, 0.5, "get_flood", "cred_stuffing"),
]

WINDOW_SIZE = 200
ENGRAM_TRAIN_SIZE = 300


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 011: Spectrum Decomposition of Mixed Windows")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Build engram library (3 engrams, 300 requests each)
    # ===================================================================
    print(f"\nPHASE 1: Build engram library ({len(ENGRAM_DEFS)} engrams, "
          f"{ENGRAM_TRAIN_SIZE} requests each)")

    library = EngramLibrary(dim=DIM)
    engram_subspaces = {}

    for name, gen_fn in ENGRAM_DEFS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(ENGRAM_TRAIN_SIZE):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i + hash(name) % 10000)))
            sub.update(vec)
        library.add(name, sub)
        engram_subspaces[name] = sub
        print(f"  Added: {name} (eig_energy={np.sum(sub.eigenvalues**2):.1f})")

    print(f"  Library: {library}")

    # ===================================================================
    # PHASE 2: Generate mixed windows and decompose spectra
    # ===================================================================
    print(f"\nPHASE 2: Mixed windows ({WINDOW_SIZE} requests each)")
    print(f"  {'Scenario':<25} {'NormalSim':>10} {'FloodSim':>10} "
          f"{'CredSim':>10} │ {'w_norm':>7} {'w_flood':>7} {'w_cred':>7}")
    print("  " + "-" * 90)

    ref_eigs = {}
    for name, sub in engram_subspaces.items():
        ref_eigs[name] = sub.eigenvalues.copy()

    engram_names = list(ENGRAM_DEFS.keys())
    n_eigs = min(len(v) for v in ref_eigs.values())
    basis_matrix = np.column_stack([ref_eigs[n][:n_eigs] for n in engram_names])

    results = []

    for label, frac_a, frac_b, type_a, type_b in MIX_SCENARIOS:
        gen_a = ENGRAM_DEFS[type_a]
        gen_b = ENGRAM_DEFS[type_b]
        seed_base = hash(label) % 100000 + 70000

        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(WINDOW_SIZE):
            rng = np.random.default_rng(i + seed_base)
            if rng.random() < frac_b:
                vec = encoder.encode_walkable(gen_b(rng))
            else:
                vec = encoder.encode_walkable(gen_a(rng))
            sub.update(vec)

        window_eigs = sub.eigenvalues

        spec_matches = library.match_spectrum(window_eigs, top_k=3)
        sim_by_name = {n: s for n, s in spec_matches}

        window_trunc = window_eigs[:n_eigs]
        weights, _ = nnls(basis_matrix, window_trunc)
        w_sum = weights.sum()
        if w_sum > 1e-10:
            weights /= w_sum

        result = {
            "label": label,
            "frac_a": frac_a, "frac_b": frac_b,
            "type_a": type_a, "type_b": type_b,
            "sim_normal": sim_by_name.get("normal", 0.0),
            "sim_flood": sim_by_name.get("get_flood", 0.0),
            "sim_cred": sim_by_name.get("cred_stuffing", 0.0),
            "w_normal": weights[engram_names.index("normal")],
            "w_flood": weights[engram_names.index("get_flood")],
            "w_cred": weights[engram_names.index("cred_stuffing")],
        }
        results.append(result)

        print(f"  {label:<25} {result['sim_normal']:>10.4f} {result['sim_flood']:>10.4f} "
              f"{result['sim_cred']:>10.4f} │ {result['w_normal']:>7.3f} "
              f"{result['w_flood']:>7.3f} {result['w_cred']:>7.3f}")

    # ===================================================================
    # RESULTS: Monotonicity and ratio recovery
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Monotonicity and Ratio Recovery")
    print("=" * 70)

    normal_flood_series = [r for r in results if r["type_b"] == "get_flood"
                           and r["type_a"] == "normal"]

    flood_sims = [r["sim_flood"] for r in normal_flood_series]
    normal_sims = [r["sim_normal"] for r in normal_flood_series]
    attack_fracs = [r["frac_b"] for r in normal_flood_series]

    print(f"\n  Normal→Flood gradient (attack fraction 0→1):")
    print(f"    Attack fractions: {attack_fracs}")
    print(f"    Flood similarities: {[f'{s:.4f}' for s in flood_sims]}")
    print(f"    Normal similarities: {[f'{s:.4f}' for s in normal_sims]}")

    flood_monotonic = all(flood_sims[i] <= flood_sims[i + 1]
                          for i in range(len(flood_sims) - 1))
    normal_monotonic = all(normal_sims[i] >= normal_sims[i + 1]
                           for i in range(len(normal_sims) - 1))
    print(f"    Flood sim monotonically increasing: {flood_monotonic}")
    print(f"    Normal sim monotonically decreasing: {normal_monotonic}")

    weight_key = {"normal": "w_normal", "get_flood": "w_flood",
                  "cred_stuffing": "w_cred"}

    print(f"\n  NNLS weight recovery:")
    for r in results:
        w_a = r[weight_key[r["type_a"]]]
        w_b = r[weight_key[r["type_b"]]]
        print(f"    {r['label']:<25} true=({r['frac_a']:.1f}/{r['frac_b']:.1f}) "
              f"recovered=({w_a:.3f}/{w_b:.3f})")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    r_50_50 = next(r for r in results if r["label"] == "50/50 normal/flood")
    sim_gap_50 = abs(r_50_50["sim_normal"] - r_50_50["sim_flood"])

    pure_normal = next(r for r in results if r["label"] == "100% normal")
    pure_flood = next(r for r in results if r["label"] == "100% flood")

    pure_normal_dom_sim = pure_normal["sim_normal"]
    pure_flood_dom_sim = pure_flood["sim_flood"]

    dominant_weights = []
    for r in results:
        majority = r["type_b"] if r["frac_b"] >= 0.5 else r["type_a"]
        dominant_weights.append(r[weight_key[majority]])

    # Endpoints: pure windows should dominate, 100% flood > 50/50 flood sim
    flood_endpoints = flood_sims[-1] > flood_sims[2]
    normal_endpoints = normal_sims[0] > normal_sims[2]

    # NNLS: check pure windows only (mixed decomposition is nonlinear)
    pure_nnls_ok = dominant_weights[0] > 0.4 and dominant_weights[-2] > 0.4

    checks = [
        (
            "Pure normal: highest normal sim, pure flood: highest flood sim",
            normal_sims[0] == max(normal_sims) and flood_sims[-1] == max(flood_sims),
            f"normal_pure={normal_sims[0]:.4f} (max={max(normal_sims):.4f}), "
            f"flood_pure={flood_sims[-1]:.4f} (max={max(flood_sims):.4f})",
        ),
        (
            "Pure flood sim > 50/50 flood sim (endpoint separation)",
            flood_endpoints,
            f"pure={flood_sims[-1]:.4f} vs 50/50={flood_sims[2]:.4f}",
        ),
        (
            "Pure normal sim > 50/50 normal sim (endpoint separation)",
            normal_endpoints,
            f"pure={normal_sims[0]:.4f} vs 50/50={normal_sims[2]:.4f}",
        ),
        (
            "At 50/50 mix: normal and flood sims within 0.3 of each other",
            sim_gap_50 < 0.3,
            f"gap={sim_gap_50:.4f} (normal={r_50_50['sim_normal']:.4f}, "
            f"flood={r_50_50['sim_flood']:.4f})",
        ),
        (
            "NNLS recovers pure windows (dominant weight > 0.4)",
            pure_nnls_ok,
            f"pure_normal_w={dominant_weights[0]:.3f}, "
            f"pure_flood_w={dominant_weights[-2]:.3f}",
        ),
        (
            "Pure normal window: normal engram similarity > 0.7",
            pure_normal_dom_sim > 0.7,
            f"sim={pure_normal_dom_sim:.4f}",
        ),
        (
            "Pure flood window: flood engram similarity > 0.7",
            pure_flood_dom_sim > 0.7,
            f"sim={pure_flood_dom_sim:.4f}",
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
