#!/usr/bin/env python3
"""
Normal Manifold Membership as Allow List

HYPOTHESIS:
===========
A subspace trained on normal browser traffic captures the "normal manifold."
Residual scoring against this subspace acts as a pass/fail gate: requests
that project well onto the manifold (low residual) are allowed; requests
with high residual are rejected. This is Layer 0 of the concept firewall —
"allow what looks normal."

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable     - Rich structured HTTP request encoding
2. OnlineSubspace.residual     - Per-vector anomaly score
3. OnlineSubspace.threshold    - Adaptive cutoff from training distribution
4. Threshold sweep             - FPR/FNR trade-off characterization

SCENARIO:
=========
WAF context. Train a normal subspace on 1000 typical browser requests.
Score 500 new normal requests (should pass), 500 attack requests (mix of
GET flood, credential stuffing, scraper, TLS shuffle — should fail), and
200 unusual-but-legitimate requests (rare paths, uncommon UAs, but
structurally valid browser behavior). Sweep threshold to find the operating
point that minimizes FPR while keeping FNR < 10%.
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
            "cipher_order": ["TLS_AES_128_GCM_SHA256",
                             "TLS_AES_256_GCM_SHA384",
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


def make_cred_stuff(rng):
    body_len = rng.integers(80, 200)
    return {
        "method": "POST", "path": "/api/v1/auth/login",
        "path_parts": ["", "api", "v1", "auth", "login"],
        "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Content-Type",
                         "Content-Length", "Accept-Encoding", "Connection"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", "python-requests/2.31.0"],
            ["Accept", "*/*"], ["Content-Type", "application/json"],
            ["Content-Length", str(body_len)],
            ["Accept-Encoding", "gzip, deflate"],
            ["Connection", "keep-alive"],
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


def make_scraper(rng):
    pid = str(rng.integers(1, 99999))
    return {
        "method": "GET", "path": f"/products/{pid}",
        "path_parts": ["", "products", pid], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Encoding"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", "Scrapy/2.11.0 (+https://scrapy.org)"],
            ["Accept", "text/html"],
            ["Accept-Encoding", "gzip, deflate"],
        ],
        "header_count": LinearScale(4), "has_cookie": "false",
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


def make_tls_shuffle(rng):
    all_ciphers = ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                   "TLS_CHACHA20_POLY1305_SHA256",
                   "ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"]
    n_c = rng.integers(2, len(all_ciphers) + 1)
    ciphers = list(rng.choice(all_ciphers, size=n_c, replace=False))
    all_exts = ["server_name", "supported_versions", "key_share",
                "signature_algorithms", "supported_groups",
                "psk_key_exchange_modes", "ec_point_formats",
                "application_layer_protocol_negotiation",
                "status_request", "signed_certificate_timestamp"]
    n_e = rng.integers(4, len(all_exts) + 1)
    exts = list(rng.choice(all_exts, size=n_e, replace=False))
    ua = str(rng.choice(["Mozilla/5.0 Bot/1.0", "CustomClient/3.2"]))
    path = str(rng.choice(["/", "/api/search", "/api/users"]))
    return {
        "method": "GET", "path": path, "path_parts": path.split("/"),
        "version": str(rng.choice(["HTTP/1.1", "HTTP/2"])),
        "header_order": ["Host", "User-Agent", "Accept"],
        "headers": [["Host", "example.com"], ["User-Agent", ua],
                    ["Accept", "*/*"]],
        "header_count": LinearScale(3), "has_cookie": "false",
        "tls": {
            "version": str(rng.choice(["TLS1.2", "TLS1.3"])),
            "ciphers": set(ciphers), "cipher_order": ciphers,
            "ext_types": set(exts), "groups": {"x25519", "secp256r1"},
            "alpn": ["h2", "http/1.1"],
        },
    }


def make_unusual_legit(rng):
    """Unusual but legitimate browser traffic — edge cases that should still pass."""
    ua = str(rng.choice([
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0.0.0",
    ]))
    path = str(rng.choice([
        "/api/v2/graphql", "/internal/metrics", "/admin/dashboard",
        "/api/search?q=unusual+query&page=999",
        f"/products/{rng.integers(100000, 999999)}",
    ]))
    return {
        "method": str(rng.choice(["GET", "POST", "PUT"])),
        "path": path, "path_parts": path.split("/"), "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Cookie", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", str(rng.choice(["application/json", "text/html,*/*;q=0.8"]))],
            ["Accept-Language", str(rng.choice(["en-US,en;q=0.9", "fr-FR,fr;q=0.9", "de-DE,de;q=0.9"]))],
            ["Accept-Encoding", "gzip, deflate, br"],
            ["Cookie", f"session={rng.integers(1000, 9999)}"],
            ["Connection", "keep-alive"],
        ],
        "header_count": LinearScale(7), "has_cookie": "true",
        "tls": {
            "version": "TLS1.3",
            "ciphers": {"TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                        "TLS_CHACHA20_POLY1305_SHA256"},
            "cipher_order": ["TLS_AES_128_GCM_SHA256",
                             "TLS_AES_256_GCM_SHA384",
                             "TLS_CHACHA20_POLY1305_SHA256"],
            "ext_types": {"server_name", "supported_versions", "key_share",
                          "signature_algorithms", "supported_groups",
                          "psk_key_exchange_modes"},
            "groups": {"x25519", "secp256r1", "secp384r1"},
            "alpn": ["h2", "http/1.1"],
        },
    }


ATTACK_GENERATORS = [make_get_flood, make_cred_stuff, make_scraper,
                     make_tls_shuffle]


def score_requests(encoder, normal_sub, gen_fn, n, seed_base):
    residuals = []
    for i in range(n):
        vec = encoder.encode_walkable(
            gen_fn(np.random.default_rng(seed_base + i)))
        residuals.append(normal_sub.residual(vec))
    return np.array(residuals)


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 008: Normal Manifold Membership as Allow List")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Train normal subspace on 1000 requests
    # ===================================================================
    print("\nPHASE 1: Train normal subspace (1000 browser requests)")

    normal_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(1000):
        vec = encoder.encode_walkable(
            make_normal(np.random.default_rng(i)))
        normal_sub.update(vec)

    print(f"  Subspace: {normal_sub}")
    print(f"  Adaptive threshold: {normal_sub.threshold:.4f}")

    # ===================================================================
    # PHASE 2: Score test populations
    # ===================================================================
    print("\nPHASE 2: Score test populations")

    normal_residuals = score_requests(
        encoder, normal_sub, make_normal, 500, seed_base=50000)

    attack_residuals_by_type = {}
    for gen_fn in ATTACK_GENERATORS:
        name = gen_fn.__name__.replace("make_", "")
        attack_residuals_by_type[name] = score_requests(
            encoder, normal_sub, gen_fn, 125, seed_base=60000 + hash(name) % 10000)
    attack_residuals = np.concatenate(list(attack_residuals_by_type.values()))

    unusual_residuals = score_requests(
        encoder, normal_sub, make_unusual_legit, 200, seed_base=70000)

    print(f"\n  {'Population':<25} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10}")
    print("  " + "-" * 68)
    print(f"  {'Normal (500)':<25} {np.mean(normal_residuals):>10.4f} "
          f"{np.std(normal_residuals):>10.4f} "
          f"{np.min(normal_residuals):>10.4f} {np.max(normal_residuals):>10.4f}")
    for name, res in attack_residuals_by_type.items():
        print(f"  {'Attack/' + name:<25} {np.mean(res):>10.4f} "
              f"{np.std(res):>10.4f} "
              f"{np.min(res):>10.4f} {np.max(res):>10.4f}")
    print(f"  {'Attack (all 500)':<25} {np.mean(attack_residuals):>10.4f} "
          f"{np.std(attack_residuals):>10.4f} "
          f"{np.min(attack_residuals):>10.4f} {np.max(attack_residuals):>10.4f}")
    print(f"  {'Unusual-legit (200)':<25} {np.mean(unusual_residuals):>10.4f} "
          f"{np.std(unusual_residuals):>10.4f} "
          f"{np.min(unusual_residuals):>10.4f} {np.max(unusual_residuals):>10.4f}")

    # ===================================================================
    # PHASE 3: Threshold sweep — FPR vs FNR
    # ===================================================================
    print("\nPHASE 3: Threshold sweep (FPR vs FNR)")

    percentiles = [50, 75, 90, 95, 99]
    candidates = [(f"p{p}", np.percentile(normal_residuals, p))
                  for p in percentiles]
    candidates.append(("adaptive", normal_sub.threshold))

    gap_points = np.geomspace(
        normal_sub.threshold, np.min(attack_residuals), num=12)[1:-1]
    for i, t in enumerate(gap_points):
        candidates.append((f"gap_{i+1}", t))

    print(f"\n  {'Threshold':<12} {'Value':>10} {'FPR':>8} {'FNR':>8} "
          f"{'Unusual rej%':>13}")
    print("  " + "-" * 55)

    best_threshold = None
    best_label = None
    best_fpr = 1.0

    for label, thresh in candidates:
        fpr = np.mean(normal_residuals > thresh)
        fnr = np.mean(attack_residuals <= thresh)
        unusual_rej = np.mean(unusual_residuals > thresh)
        print(f"  {label:<12} {thresh:>10.4f} {fpr:>7.1%} {fnr:>7.1%} "
              f"{unusual_rej:>12.1%}")

        if fnr < 0.10 and (fpr < best_fpr or
                           (fpr == best_fpr and
                            (best_threshold is None or thresh > best_threshold))):
            best_fpr = fpr
            best_threshold = thresh
            best_label = label

    if best_threshold is None:
        best_idx = np.argmin([np.mean(attack_residuals <= t)
                              for _, t in candidates])
        best_threshold = candidates[best_idx][1]
        best_label = candidates[best_idx][0]
        best_fpr = np.mean(normal_residuals > best_threshold)

    best_fnr = np.mean(attack_residuals <= best_threshold)
    best_unusual_rej = np.mean(unusual_residuals > best_threshold)
    best_attack_rej = np.mean(attack_residuals > best_threshold)

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Optimal Operating Point")
    print("=" * 70)
    print(f"\n  Optimal threshold: {best_label} = {best_threshold:.4f}")
    print(f"  FPR (normal rejected):       {best_fpr:.1%}")
    print(f"  FNR (attacks missed):        {best_fnr:.1%}")
    print(f"  Attack rejection rate:       {best_attack_rej:.1%}")
    print(f"  Unusual-legit rejection:     {best_unusual_rej:.1%}")

    mean_normal = np.mean(normal_residuals)
    mean_attack = np.mean(attack_residuals)
    separation = mean_attack / max(mean_normal, 1e-12)

    print(f"\n  Distribution separation:")
    print(f"    Mean normal residual:  {mean_normal:.4f}")
    print(f"    Mean attack residual:  {mean_attack:.4f}")
    print(f"    Ratio (attack/normal): {separation:.2f}x")

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "FPR < 5% at optimal threshold",
            best_fpr < 0.05,
            f"FPR={best_fpr:.1%}",
        ),
        (
            "FNR < 10% at optimal threshold",
            best_fnr < 0.10,
            f"FNR={best_fnr:.1%}",
        ),
        (
            "Attack rejection > 90% at optimal threshold",
            best_attack_rej > 0.90,
            f"rejection={best_attack_rej:.1%}",
        ),
        (
            "Unusual-legit rejection < 30% at optimal threshold",
            best_unusual_rej < 0.30,
            f"rejection={best_unusual_rej:.1%}",
        ),
        (
            "Clear separation: mean_attack > 2 * mean_normal",
            mean_attack > 2 * mean_normal,
            f"ratio={separation:.2f}x",
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
