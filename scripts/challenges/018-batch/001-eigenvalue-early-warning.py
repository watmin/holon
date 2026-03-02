#!/usr/bin/env python3
"""
Eigenvalue Shift as Early Warning

HYPOTHESIS:
===========
Eigenvalue spectrum divergence from baseline occurs earlier in an attack
than per-request residual hits. The divergence signal leads; the residual
signal confirms.

The signal is DIFFERENTIAL: track how much the live window's eigenvalue
spectrum diverges from the stored baseline. As attack traffic mixes in,
the variance structure shifts — that shift is the early warning.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable     - Rich structured HTTP request encoding
2. OnlineSubspace.eigenvalues  - Live window eigenvalue extraction
3. Differential spectrum signal - Divergence from baseline as detector
4. OnlineSubspace.residual     - Per-request anomaly scoring

SCENARIO:
=========
WAF context. Encode full HTTP request structure (headers in wire order,
path segments, TLS context with cipher sets/ordering). Train baseline on
500 normal browser requests. Simulate a slow-ramp GET flood with tool UAs,
TLS 1.2, and minimal headers. Track divergence vs per-request hit rate.
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
    """Normal browser traffic — full request structure.

    Chrome/Firefox, TLS 1.3, varied paths, cookies, standard header
    ordering. Mirrors what the http-lab proxy's RequestSample encodes.
    """
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    ]))
    path = str(rng.choice([
        "/", "/api/users", "/api/search", "/products",
        "/about", "/health", "/api/v1/feed", "/settings",
    ]))
    path_parts = path.split("/")
    method = str(rng.choice(["GET", "GET", "GET", "POST"]))
    accept = str(rng.choice([
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "application/json",
    ]))

    return {
        "method": method,
        "path": path,
        "path_parts": path_parts,
        "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Cookie", "Connection"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", ua],
            ["Accept", accept],
            ["Accept-Language", "en-US,en;q=0.9"],
            ["Accept-Encoding", "gzip, deflate, br"],
            ["Cookie", f"session={rng.integers(1000, 9999)}"],
            ["Connection", "keep-alive"],
        ],
        "header_count": LinearScale(7),
        "has_cookie": "true",
        "tls": {
            "version": "TLS1.3",
            "ciphers": {"TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                        "TLS_CHACHA20_POLY1305_SHA256"},
            "cipher_order": ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                             "TLS_CHACHA20_POLY1305_SHA256"],
            "ext_types": {"server_name", "supported_versions", "key_share",
                          "signature_algorithms", "supported_groups",
                          "psk_key_exchange_modes", "ec_point_formats",
                          "application_layer_protocol_negotiation",
                          "status_request", "signed_certificate_timestamp"},
            "ext_order": ["server_name", "supported_versions", "key_share",
                          "signature_algorithms", "supported_groups",
                          "psk_key_exchange_modes"],
            "groups": {"x25519", "secp256r1", "secp384r1"},
            "sig_algs": {"ecdsa_secp256r1_sha256", "rsa_pss_rsae_sha256",
                         "rsa_pkcs1_sha256"},
            "alpn": ["h2", "http/1.1"],
        },
    }


def make_get_flood(rng):
    """GET flood — curl/bot UA, TLS 1.2, minimal headers, hammering one path."""
    ua = str(rng.choice([
        "curl/8.0.1",
        "python-requests/2.31.0",
        "Go-http-client/1.1",
        "libwww-perl/6.72",
    ]))

    return {
        "method": "GET",
        "path": "/api/search",
        "path_parts": ["", "api", "search"],
        "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", ua],
            ["Accept", "*/*"],
        ],
        "header_count": LinearScale(3),
        "has_cookie": "false",
        "tls": {
            "version": "TLS1.2",
            "ciphers": {"ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"},
            "cipher_order": ["ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"],
            "ext_types": {"server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms"},
            "ext_order": ["server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms"],
            "groups": {"secp256r1"},
            "sig_algs": {"rsa_pkcs1_sha256"},
            "alpn": ["http/1.1"],
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


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 001: Eigenvalue Shift as Early Warning")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Train baseline on normal browser traffic
    # ===================================================================
    print("\nPHASE 1: Train baseline subspace (500 normal browser requests)")

    baseline_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0, sigma_mult=3.5)
    for i in range(500):
        vec = encoder.encode_walkable(make_normal(np.random.default_rng(i)))
        baseline_sub.update(vec)

    baseline_eigenvalues = baseline_sub.eigenvalues.copy()
    baseline_snap = baseline_sub.snapshot()

    print(f"  Baseline: {baseline_sub}")
    print(f"  Threshold: {baseline_sub.threshold:.4f}")

    # ===================================================================
    # PHASE 2: Slow-ramp GET flood with dual signal tracking
    # ===================================================================
    print("\nPHASE 2: Slow-ramp GET flood (400 requests, 90/10 → 100% attack)")
    print("  Signal A: spectrum divergence from baseline (1 - cosine_sim)")
    print("  Signal B: per-request residual hit rate\n")

    n_ramp = 400
    window_size = 50

    window_sub = OnlineSubspace.from_snapshot(baseline_snap)

    divergences = []
    hit_rates = []
    recent_hits = []

    div_threshold = 0.10
    hit_rate_threshold = 0.50

    div_threshold_req = None
    hit_threshold_req = None

    for i in range(n_ramp):
        if i < 200:
            attack_frac = 0.1 + 0.4 * (i / 200)
        elif i < 300:
            attack_frac = 0.5 + 0.5 * ((i - 200) / 100)
        else:
            attack_frac = 1.0

        req_rng = np.random.default_rng(i + 10000)
        if req_rng.random() < attack_frac:
            d = make_get_flood(req_rng)
        else:
            d = make_normal(req_rng)

        vec = encoder.encode_walkable(d)

        res = baseline_sub.residual(vec)
        is_hit = res > baseline_sub.threshold
        recent_hits.append(1.0 if is_hit else 0.0)

        window_sub.update(vec)

        hit_rate = (np.mean(recent_hits[-window_size:])
                    if len(recent_hits) >= window_size
                    else np.mean(recent_hits))

        divergence = 1.0 - cosine_similarity(window_sub.eigenvalues,
                                              baseline_eigenvalues)

        divergences.append(divergence)
        hit_rates.append(hit_rate)

        if div_threshold_req is None and divergence > div_threshold:
            div_threshold_req = i
        if hit_threshold_req is None and hit_rate > hit_rate_threshold:
            hit_threshold_req = i

        if i % 50 == 0 or i == n_ramp - 1:
            print(f"  Request {i+1:>4}: attack_frac={attack_frac:.2f}  "
                  f"divergence={divergence:.4f}  hit_rate={hit_rate:.4f}")

    # ===================================================================
    # PHASE 3: Control — pure normal traffic
    # ===================================================================
    print("\nPHASE 3: Control — pure normal traffic (200 requests)")

    normal_window_sub = OnlineSubspace.from_snapshot(baseline_snap)
    normal_divergences = []
    for i in range(200):
        vec = encoder.encode_walkable(make_normal(np.random.default_rng(i + 20000)))
        normal_window_sub.update(vec)
        div = 1.0 - cosine_similarity(normal_window_sub.eigenvalues,
                                       baseline_eigenvalues)
        normal_divergences.append(div)

    max_normal_div = max(normal_divergences)
    mean_normal_div = np.mean(normal_divergences)
    print(f"  Normal traffic divergence: mean={mean_normal_div:.4f}, "
          f"max={max_normal_div:.4f}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    lead_requests = None
    if div_threshold_req is not None and hit_threshold_req is not None:
        lead_requests = hit_threshold_req - div_threshold_req
        print(f"\n  Divergence crossed {div_threshold} at request: "
              f"{div_threshold_req + 1}")
        print(f"  Hit rate crossed {hit_rate_threshold} at request:    "
              f"{hit_threshold_req + 1}")
        print(f"  Eigenvalue signal LEADS by: {lead_requests} requests")
    elif div_threshold_req is not None:
        print(f"\n  Divergence crossed {div_threshold} at request: "
              f"{div_threshold_req + 1}")
        print(f"  Hit rate NEVER crossed {hit_rate_threshold}")
        lead_requests = n_ramp
    elif hit_threshold_req is not None:
        print(f"\n  Divergence NEVER crossed {div_threshold}")
        print(f"  Hit rate crossed {hit_rate_threshold} at request: "
              f"{hit_threshold_req + 1}")
        lead_requests = -1
    else:
        print(f"\n  Neither signal crossed its threshold")

    both_elevated = False
    if len(divergences) >= 100 and len(hit_rates) >= 100:
        div_at_100 = divergences[99]
        hit_at_100 = hit_rates[99]
        both_elevated = div_at_100 > 0.01 and hit_at_100 > 0.1
        print(f"\n  At request 100: divergence={div_at_100:.4f}, "
              f"hit_rate={hit_at_100:.4f}")
        print(f"  Both signals trending by request 100: {both_elevated}")

    final_div = divergences[-1]
    final_hit = hit_rates[-1]
    print(f"\n  Final state (request {n_ramp}): divergence={final_div:.4f}, "
          f"hit_rate={final_hit:.4f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Eigenvalue divergence leads hit rate by >= 20 requests",
            lead_requests is not None and lead_requests >= 20,
            f"lead={lead_requests} requests"
            if lead_requests is not None else "N/A",
        ),
        (
            "Both signals trending by request 100",
            both_elevated,
            f"div={divergences[99]:.4f}, hit={hit_rates[99]:.4f}"
            if len(divergences) >= 100 else "N/A",
        ),
        (
            f"Normal traffic: divergence stays below {div_threshold}",
            max_normal_div < div_threshold,
            f"max_normal_div={max_normal_div:.4f}",
        ),
        (
            f"Final attack state: divergence > {div_threshold}",
            final_div > div_threshold,
            f"final_div={final_div:.4f}",
        ),
        (
            "Final attack state: hit rate > 0.8",
            final_hit > 0.8,
            f"final_hit={final_hit:.4f}",
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
