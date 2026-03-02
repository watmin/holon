#!/usr/bin/env python3
"""
Multi-Modal Normal Library

HYPOTHESIS:
===========
A single "normal" engram is too narrow for real traffic — legitimate users
arrive via browsers, API clients, and mobile apps, each with distinct HTTP
and TLS fingerprints. An EngramLibrary holding one engram per mode can
identify which mode a request belongs to (lowest residual) while still
rejecting attack traffic that doesn't match ANY mode. The "best-of-library"
score — min(residual across all engrams) — acts as an allow gate that
accepts structurally diverse legitimate traffic without false positives.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable    - Rich structured HTTP/TLS encoding
2. OnlineSubspace             - Per-mode manifold learning
3. EngramLibrary.add          - Multi-mode engram storage
4. Engram.residual            - Per-vector anomaly score
5. Best-of-library scoring    - min(residual) across modes

SCENARIO:
=========
Train 3 normal engrams (500 requests each) for browser-web, api-client,
and mobile-app traffic. Score 200 test requests per mode — the correct
engram should have the lowest residual. Score 200 attack requests (GET
flood + scraper) — all engrams should reject. The best-of-library min
residual should accept legitimate cross-mode traffic and reject attacks.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon.kernel.encoder import Encoder
from holon.kernel.vector_manager import VectorManager
from holon.kernel.walkable import LinearScale
from holon.memory.engram import EngramLibrary
from holon.memory.subspace import OnlineSubspace

DIM = 4096


def make_browser_web(rng):
    """Normal browser-web traffic: Chrome/Firefox, TLS 1.3, HTML pages, cookies."""
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]))
    path = str(rng.choice(["/", "/about", "/products", "/contact", "/blog",
                            "/settings", "/dashboard"]))
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


def make_api_client(rng):
    """API client traffic: JSON, Authorization headers, /api/v2 paths."""
    path = str(rng.choice(["/api/v2/users", "/api/v2/search", "/api/v2/orders",
                            "/api/v2/products", "/api/v2/recommendations"]))
    method = str(rng.choice(["GET", "GET", "POST", "PUT", "DELETE"]))
    return {
        "method": method,
        "path": path, "path_parts": path.split("/"), "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Authorization", "Accept",
                         "Content-Type", "Accept-Encoding"],
        "headers": [
            ["Host", "api.example.com"],
            ["User-Agent", "MyApp/3.1.0 (Linux; API-Client)"],
            ["Authorization", f"Bearer eyJ{rng.integers(1000000, 9999999)}"],
            ["Accept", "application/json"],
            ["Content-Type", "application/json"],
            ["Accept-Encoding", "gzip"],
        ],
        "header_count": LinearScale(6), "has_cookie": "false",
        "tls": {
            "version": "TLS1.3",
            "ciphers": {"TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384"},
            "cipher_order": ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384"],
            "ext_types": {"server_name", "supported_versions", "key_share",
                          "signature_algorithms", "supported_groups"},
            "groups": {"x25519", "secp256r1"},
            "alpn": ["h2"],
        },
    }


def make_mobile_app(rng):
    """Mobile app traffic: compact headers, /api/mobile paths."""
    path = str(rng.choice(["/api/mobile/feed", "/api/mobile/profile",
                            "/api/mobile/notifications", "/api/mobile/sync"]))
    return {
        "method": str(rng.choice(["GET", "POST"])),
        "path": path, "path_parts": path.split("/"), "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Authorization",
                         "X-App-Version", "Accept-Encoding"],
        "headers": [
            ["Host", "m.example.com"],
            ["User-Agent", str(rng.choice([
                "ExampleApp/4.2.0 (iPhone; iOS 17.2)",
                "ExampleApp/4.2.0 (Android 14; Pixel 8)",
            ]))],
            ["Accept", "application/json"],
            ["Authorization", f"Bearer mob_{rng.integers(100000, 999999)}"],
            ["X-App-Version", "4.2.0"],
            ["Accept-Encoding", "gzip"],
        ],
        "header_count": LinearScale(6), "has_cookie": "false",
        "tls": {
            "version": "TLS1.3",
            "ciphers": {"TLS_AES_128_GCM_SHA256", "TLS_CHACHA20_POLY1305_SHA256"},
            "cipher_order": ["TLS_CHACHA20_POLY1305_SHA256", "TLS_AES_128_GCM_SHA256"],
            "ext_types": {"server_name", "supported_versions", "key_share",
                          "signature_algorithms", "supported_groups"},
            "groups": {"x25519"},
            "alpn": ["h2"],
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


MODE_GENERATORS = {
    "browser_web": make_browser_web,
    "api_client": make_api_client,
    "mobile_app": make_mobile_app,
}


def encode_batch(encoder, gen_fn, n, seed_base):
    vecs = []
    for i in range(n):
        vec = encoder.encode_walkable(gen_fn(np.random.default_rng(seed_base + i)))
        vecs.append(vec)
    return vecs


def score_against_engram(engram, vecs):
    return np.array([engram.residual(v) for v in vecs])


def best_of_library_residuals(library, vecs):
    """For each vec, return min residual across all engrams."""
    results = np.empty(len(vecs))
    for i, v in enumerate(vecs):
        results[i] = min(eng.residual(v) for eng in
                         [library.get(n) for n in library.names()])
    return results


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 009: Multi-Modal Normal Library")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Train 3 normal engrams
    # ===================================================================
    print("\nPHASE 1: Train normal engrams (500 requests each)")

    library = EngramLibrary(dim=DIM)
    subspaces = {}

    for mode_name, gen_fn in MODE_GENERATORS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(500):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i)))
            sub.update(vec)
        subspaces[mode_name] = sub
        library.add(mode_name, sub, mode=mode_name)
        print(f"  {mode_name}: threshold={sub.threshold:.4f}, "
              f"n={sub.n}")

    print(f"\n  Library: {library}")

    # ===================================================================
    # PHASE 2: Mode identification — score test traffic against all engrams
    # ===================================================================
    print("\nPHASE 2: Mode identification (200 test requests per mode)")

    test_vecs = {}
    for mode_name, gen_fn in MODE_GENERATORS.items():
        test_vecs[mode_name] = encode_batch(
            encoder, gen_fn, 200, seed_base=50000 + hash(mode_name) % 10000)

    mode_names = list(MODE_GENERATORS.keys())
    correct_counts = {m: 0 for m in mode_names}
    cross_mode_ratios = []

    print(f"\n  {'Test Mode':<15}", end="")
    for m in mode_names:
        print(f"  {m:>14}", end="")
    print(f"  {'Correct?':>10}")
    print("  " + "-" * (15 + 14 * len(mode_names) + 12))

    per_mode_residuals = {}
    for test_mode in mode_names:
        residuals_by_engram = {}
        for engram_name in mode_names:
            engram = library.get(engram_name)
            residuals_by_engram[engram_name] = score_against_engram(
                engram, test_vecs[test_mode])
        per_mode_residuals[test_mode] = residuals_by_engram

        means = {m: np.mean(residuals_by_engram[m]) for m in mode_names}
        best_mode = min(means, key=means.get)
        correct = best_mode == test_mode
        if correct:
            correct_counts[test_mode] = 1

        same_mode_mean = means[test_mode]
        other_means = [means[m] for m in mode_names if m != test_mode]
        if same_mode_mean > 1e-12:
            ratio = np.mean(other_means) / same_mode_mean
            cross_mode_ratios.append(ratio)

        print(f"  {test_mode:<15}", end="")
        for m in mode_names:
            marker = " *" if m == best_mode else ""
            print(f"  {means[m]:>12.4f}{marker}", end="")
        status = "YES" if correct else "NO"
        print(f"  {status:>8}")

    per_request_correct = 0
    per_request_total = 0
    for test_mode in mode_names:
        residuals_by_engram = per_mode_residuals[test_mode]
        for i in range(len(test_vecs[test_mode])):
            per_request_total += 1
            res_per_engram = {m: residuals_by_engram[m][i] for m in mode_names}
            if min(res_per_engram, key=res_per_engram.get) == test_mode:
                per_request_correct += 1

    per_request_accuracy = per_request_correct / per_request_total
    mode_id_rate = sum(correct_counts.values()) / len(mode_names)
    avg_cross_ratio = np.mean(cross_mode_ratios) if cross_mode_ratios else 0.0

    print(f"\n  Mode identification (by mean): {mode_id_rate:.0%} "
          f"({sum(correct_counts.values())}/{len(mode_names)})")
    print(f"  Per-request identification:    {per_request_accuracy:.1%} "
          f"({per_request_correct}/{per_request_total})")
    print(f"  Cross-mode / same-mode ratio:  {avg_cross_ratio:.2f}x")

    # ===================================================================
    # PHASE 3: Attack rejection — all engrams should reject
    # ===================================================================
    print("\nPHASE 3: Attack rejection (100 get_flood + 100 scraper)")

    attack_vecs = (
        encode_batch(encoder, make_get_flood, 100, seed_base=60000) +
        encode_batch(encoder, make_scraper, 100, seed_base=70000)
    )

    attack_residuals_by_engram = {}
    for engram_name in mode_names:
        engram = library.get(engram_name)
        attack_residuals_by_engram[engram_name] = score_against_engram(
            engram, attack_vecs)

    print(f"\n  {'Engram':<15} {'Mean res':>10} {'Min res':>10} "
          f"{'Max res':>10} {'> thresh':>10}")
    print("  " + "-" * 58)

    for engram_name in mode_names:
        res = attack_residuals_by_engram[engram_name]
        thresh = subspaces[engram_name].threshold
        rej_rate = np.mean(res > thresh)
        print(f"  {engram_name:<15} {np.mean(res):>10.4f} "
              f"{np.min(res):>10.4f} {np.max(res):>10.4f} "
              f"{rej_rate:>9.1%}")

    # ===================================================================
    # PHASE 4: Best-of-library scoring
    # ===================================================================
    print("\nPHASE 4: Best-of-library scoring (min residual across all engrams)")

    bol_normal = {}
    for mode_name in mode_names:
        bol_normal[mode_name] = best_of_library_residuals(
            library, test_vecs[mode_name])

    bol_attack = best_of_library_residuals(library, attack_vecs)

    thresholds = [subspaces[m].threshold for m in mode_names]
    bol_threshold = max(thresholds)

    print(f"\n  Best-of-library threshold (max of per-mode): {bol_threshold:.4f}")

    print(f"\n  {'Population':<25} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10}")
    print("  " + "-" * 68)
    for mode_name in mode_names:
        res = bol_normal[mode_name]
        print(f"  {'Normal/' + mode_name:<25} {np.mean(res):>10.4f} "
              f"{np.std(res):>10.4f} "
              f"{np.min(res):>10.4f} {np.max(res):>10.4f}")
    print(f"  {'Attack (200)':<25} {np.mean(bol_attack):>10.4f} "
          f"{np.std(bol_attack):>10.4f} "
          f"{np.min(bol_attack):>10.4f} {np.max(bol_attack):>10.4f}")

    all_normal_bol = np.concatenate(list(bol_normal.values()))
    bol_fpr = np.mean(all_normal_bol > bol_threshold)
    bol_attack_rej = np.mean(bol_attack > bol_threshold)

    single_engram_fprs = []
    for mode_name in mode_names:
        other_modes = [m for m in mode_names if m != mode_name]
        for other in other_modes:
            res = per_mode_residuals[other][mode_name]
            thresh = subspaces[mode_name].threshold
            single_engram_fprs.append(np.mean(res > thresh))
    avg_single_fpr = np.mean(single_engram_fprs)

    print(f"\n  Best-of-library FPR (normal rejected): {bol_fpr:.1%}")
    print(f"  Best-of-library attack rejection:      {bol_attack_rej:.1%}")
    print(f"  Single-engram avg cross-mode FPR:      {avg_single_fpr:.1%}")
    print(f"  FPR improvement (single → library):    "
          f"{avg_single_fpr:.1%} → {bol_fpr:.1%}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Mode identification accuracy:     {per_request_accuracy:.1%}")
    print(f"  Cross/same-mode residual ratio:   {avg_cross_ratio:.2f}x")
    print(f"  Best-of-library FPR:              {bol_fpr:.1%}")
    print(f"  Best-of-library attack rejection: {bol_attack_rej:.1%}")

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Correct mode identified > 80% of requests",
            per_request_accuracy > 0.80,
            f"accuracy={per_request_accuracy:.1%}",
        ),
        (
            "Cross-mode residual ratio > 1.5x",
            avg_cross_ratio > 1.5,
            f"ratio={avg_cross_ratio:.2f}x",
        ),
        (
            "Attack best-of-library rejection > 95%",
            bol_attack_rej > 0.95,
            f"rejection={bol_attack_rej:.1%}",
        ),
        (
            "Best-of-library FPR < single-engram cross-mode FPR",
            bol_fpr < avg_single_fpr,
            f"library={bol_fpr:.1%} vs single={avg_single_fpr:.1%}",
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
