#!/usr/bin/env python3
"""
Attack Type Classification from Window Shape Alone

HYPOTHESIS:
===========
Different HTTP attack types produce distinct eigenvalue spectrum shapes
when observed over a traffic window. Classification can be performed using
only match_spectrum — no per-request residual computation needed.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable          - Rich structured HTTP request encoding
2. OnlineSubspace per attack type   - Engram creation
3. EngramLibrary.match_spectrum     - Shape-only classification

SCENARIO:
=========
WAF context. Full request structure encoding (headers in wire order, path
segments, TLS context with cipher sets/ordering). Build an engram library
from 4 HTTP attack types. Generate fresh windows per type, classify using
match_spectrum alone.
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


def make_normal(rng):
    """Normal browser traffic — full request structure."""
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    ]))
    path = str(rng.choice(["/", "/api/users", "/api/search", "/products",
                            "/about", "/health", "/api/v1/feed", "/settings"]))
    return {
        "method": str(rng.choice(["GET", "GET", "GET", "POST"])),
        "path": path,
        "path_parts": path.split("/"),
        "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Cookie", "Connection"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", ua],
            ["Accept", "text/html,application/xhtml+xml,*/*;q=0.8"],
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
                          "psk_key_exchange_modes"},
            "groups": {"x25519", "secp256r1", "secp384r1"},
            "alpn": ["h2", "http/1.1"],
        },
    }


def make_get_flood(rng):
    """GET flood — curl/bot UA, TLS 1.2, minimal headers."""
    ua = str(rng.choice(["curl/8.0.1", "python-requests/2.31.0",
                          "Go-http-client/1.1", "libwww-perl/6.72"]))
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
            "groups": {"secp256r1"},
            "alpn": ["http/1.1"],
        },
    }


def make_cred_stuff(rng):
    """Credential stuffing — POST /login, python-requests, varied bodies."""
    body_len = rng.integers(80, 200)
    return {
        "method": "POST",
        "path": "/api/v1/auth/login",
        "path_parts": ["", "api", "v1", "auth", "login"],
        "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Content-Type",
                         "Content-Length", "Accept-Encoding", "Connection"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", "python-requests/2.31.0"],
            ["Accept", "*/*"],
            ["Content-Type", "application/json"],
            ["Content-Length", str(body_len)],
            ["Accept-Encoding", "gzip, deflate"],
            ["Connection", "keep-alive"],
        ],
        "header_count": LinearScale(7),
        "has_cookie": "false",
        "body_len": LinearScale(body_len),
        "tls": {
            "version": "TLS1.2",
            "ciphers": {"ECDHE-RSA-AES128-GCM-SHA256",
                        "ECDHE-RSA-AES256-GCM-SHA384",
                        "AES256-SHA"},
            "cipher_order": ["ECDHE-RSA-AES128-GCM-SHA256",
                             "ECDHE-RSA-AES256-GCM-SHA384",
                             "AES256-SHA"],
            "ext_types": {"server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms",
                          "application_layer_protocol_negotiation"},
            "groups": {"secp256r1", "secp384r1"},
            "alpn": ["http/1.1"],
        },
    }


def make_scraper(rng):
    """Scraper — GET /products/{random}, Scrapy UA, high path cardinality."""
    product_id = str(rng.integers(1, 99999))
    return {
        "method": "GET",
        "path": f"/products/{product_id}",
        "path_parts": ["", "products", product_id],
        "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Encoding"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", "Scrapy/2.11.0 (+https://scrapy.org)"],
            ["Accept", "text/html"],
            ["Accept-Encoding", "gzip, deflate"],
        ],
        "header_count": LinearScale(4),
        "has_cookie": "false",
        "tls": {
            "version": "TLS1.2",
            "ciphers": {"ECDHE-RSA-AES128-GCM-SHA256",
                        "ECDHE-RSA-AES256-GCM-SHA384", "AES256-SHA"},
            "cipher_order": ["ECDHE-RSA-AES128-GCM-SHA256",
                             "ECDHE-RSA-AES256-GCM-SHA384", "AES256-SHA"],
            "ext_types": {"server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms",
                          "application_layer_protocol_negotiation"},
            "groups": {"secp256r1", "secp384r1"},
            "alpn": ["http/1.1"],
        },
    }


def make_tls_shuffle(rng):
    """TLS-randomized flood — shuffled cipher/extension ordering, bot UA."""
    all_ciphers = ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                   "TLS_CHACHA20_POLY1305_SHA256",
                   "ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"]
    n_ciphers = rng.integers(2, len(all_ciphers) + 1)
    chosen_ciphers = list(rng.choice(all_ciphers, size=n_ciphers, replace=False))

    all_exts = ["server_name", "supported_versions", "key_share",
                "signature_algorithms", "supported_groups",
                "psk_key_exchange_modes", "ec_point_formats",
                "application_layer_protocol_negotiation",
                "status_request", "signed_certificate_timestamp"]
    n_exts = rng.integers(4, len(all_exts) + 1)
    chosen_exts = list(rng.choice(all_exts, size=n_exts, replace=False))

    ua = str(rng.choice(["Mozilla/5.0 Bot/1.0", "CustomClient/3.2",
                          "Mozilla/5.0 (compatible; Bot/2.0)"]))
    path = str(rng.choice(["/", "/api/search", "/api/users", "/products"]))
    return {
        "method": "GET",
        "path": path,
        "path_parts": path.split("/"),
        "version": str(rng.choice(["HTTP/1.1", "HTTP/2"])),
        "header_order": ["Host", "User-Agent", "Accept"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", ua],
            ["Accept", "*/*"],
        ],
        "header_count": LinearScale(3),
        "has_cookie": "false",
        "tls": {
            "version": str(rng.choice(["TLS1.2", "TLS1.3"])),
            "ciphers": set(chosen_ciphers),
            "cipher_order": chosen_ciphers,
            "ext_types": set(chosen_exts),
            "groups": {"x25519", "secp256r1"},
            "alpn": ["h2", "http/1.1"],
        },
    }


ATTACK_GENERATORS = {
    "get_flood": make_get_flood,
    "cred_stuffing": make_cred_stuff,
    "scraper": make_scraper,
    "tls_shuffle": make_tls_shuffle,
}


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 002: Attack Type Classification from Window Shape")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Build engram library from 4 HTTP attack types
    # ===================================================================
    print("\nPHASE 1: Build engram library (4 HTTP attack types, 300 requests each)")

    library = EngramLibrary(dim=DIM)
    for name, gen_fn in ATTACK_GENERATORS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(300):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i + hash(name) % 10000)))
            sub.update(vec)
        library.add(name, sub, attack_type=name)
        print(f"  Added: {name} (eig energy={np.sum(sub.eigenvalues**2):.1f})")

    print(f"  Library: {library}")

    # ===================================================================
    # PHASE 2: Classify fresh windows (10 per attack type)
    # ===================================================================
    print("\nPHASE 2: Classify 10 fresh windows per attack type (200 requests)")

    n_trials = 10
    n_window = 200
    per_type_correct = {}

    for true_type, gen_fn in ATTACK_GENERATORS.items():
        correct = 0
        for trial in range(n_trials):
            seed = trial * 1000 + hash(true_type) % 10000 + 50000
            sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
            for i in range(n_window):
                vec = encoder.encode_walkable(
                    gen_fn(np.random.default_rng(i + seed)))
                sub.update(vec)

            matches = library.match_spectrum(sub.eigenvalues, top_k=4)
            predicted = matches[0][0] if matches else "none"
            if predicted == true_type:
                correct += 1

        per_type_correct[true_type] = correct
        print(f"  {true_type:<15}: {correct}/{n_trials} "
              f"({correct/n_trials*100:.0f}%)")

    total_correct = sum(per_type_correct.values())
    total_count = len(ATTACK_GENERATORS) * n_trials
    overall_acc = total_correct / total_count * 100
    print(f"\n  Overall accuracy: {total_correct}/{total_count} "
          f"({overall_acc:.0f}%)")

    # ===================================================================
    # PHASE 3: Normal traffic windows
    # ===================================================================
    print("\nPHASE 3: Normal browser windows (10 windows)")

    normal_max_sims = []
    for trial in range(10):
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(n_window):
            vec = encoder.encode_walkable(
                make_normal(np.random.default_rng(i + trial * 1000 + 90000)))
            sub.update(vec)
        matches = library.match_spectrum(sub.eigenvalues, top_k=4)
        max_sim = matches[0][1] if matches else 0.0
        top_name = matches[0][0] if matches else "none"
        normal_max_sims.append(max_sim)
        print(f"  Normal window {trial+1:>2}: max_sim={max_sim:.4f} "
              f"(top: {top_name})")

    max_normal_sim = max(normal_max_sims)
    print(f"\n  Normal windows: max_sim={max_normal_sim:.4f}")

    # ===================================================================
    # PHASE 4: Cross-type discrimination matrix
    # ===================================================================
    print("\nPHASE 4: Cross-type eigenvalue similarity matrix")

    attack_names = list(ATTACK_GENERATORS.keys())
    all_engrams = [library.get(n) for n in attack_names]

    print(f"\n  {'':>15}", end="")
    for n in attack_names:
        print(f"  {n:>12}", end="")
    print()
    print("  " + "-" * (15 + 14 * len(attack_names)))

    cross_sims = {}
    for n1 in attack_names:
        e1 = library.get(n1)
        print(f"  {n1:>15}", end="")
        for n2 in attack_names:
            e2 = library.get(n2)
            eig1 = e1.eigenvalue_signature
            eig2 = e2.eigenvalue_signature
            min_len = min(len(eig1), len(eig2))
            sim = float(np.dot(eig1[:min_len], eig2[:min_len]))
            cross_sims[(n1, n2)] = sim
            print(f"  {sim:>12.4f}", end="")
        print()

    off_diagonal = [v for (n1, n2), v in cross_sims.items() if n1 != n2]
    mean_cross = np.mean(off_diagonal) if off_diagonal else 0.0
    max_cross = max(off_diagonal) if off_diagonal else 0.0
    print(f"\n  Mean cross-type similarity: {mean_cross:.4f}")
    print(f"  Max cross-type similarity:  {max_cross:.4f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    min_type_correct = min(per_type_correct.values())
    worst = min(per_type_correct, key=per_type_correct.get)

    # Note: eigenvalue spectra capture variance SHAPE, not direction.
    # match_spectrum discriminates BETWEEN attack types (classification)
    # but cannot reliably distinguish attack-from-normal (detection).
    # Detection is Layer 0/1's job (residual scoring on vector directions).
    # Classification is Layer 2's job (spectrum matching on variance shape).
    checks = [
        (
            "Overall classification accuracy > 75%",
            overall_acc > 75,
            f"acc={overall_acc:.0f}%",
        ),
        (
            "Each attack type correct > 60%",
            min_type_correct / n_trials * 100 > 60,
            f"worst={min_type_correct/n_trials*100:.0f}% ({worst})",
        ),
        (
            "Cross-type similarity < 0.95 (types are distinguishable)",
            max_cross < 0.95,
            f"max_cross={max_cross:.4f}",
        ),
    ]

    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc} ({detail})")

    print(f"\n  FINDING: Normal traffic matches some attack eigenvalue "
          f"spectra (max_sim={max_normal_sim:.4f}).")
    print(f"  This is expected — eigenvalue spectra capture variance shape,")
    print(f"  not direction. match_spectrum is for CLASSIFICATION (Layer 2),")
    print(f"  not DETECTION (Layers 0-1). Detection uses residual scoring.")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
