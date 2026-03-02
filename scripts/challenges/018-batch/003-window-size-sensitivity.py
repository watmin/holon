#!/usr/bin/env python3
"""
Window Size Sensitivity

HYPOTHESIS:
===========
There is a minimum window size at which eigenvalue matching becomes reliable,
and accuracy is monotonically non-decreasing with window size. The accuracy
curve should show a clear knee.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable          - Rich structured HTTP request encoding
2. OnlineSubspace at varying counts - Window size effect on spectrum
3. EngramLibrary.match_spectrum     - Classification at each size

SCENARIO:
=========
WAF context. Use a 4-engram library of HTTP attack types with full request
structure encoding. Sweep window sizes from 10 to 500 requests, classify
10 independent windows per type at each size.
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
    ua = str(rng.choice(["Mozilla/5.0 Bot/1.0", "CustomClient/3.2",
                          "Mozilla/5.0 (compatible; Bot/2.0)"]))
    path = str(rng.choice(["/", "/api/search", "/api/users", "/products"]))
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


ATTACK_GENERATORS = {
    "get_flood": make_get_flood,
    "cred_stuffing": make_cred_stuff,
    "scraper": make_scraper,
    "tls_shuffle": make_tls_shuffle,
}

WINDOW_SIZES = [10, 25, 50, 100, 200, 500]
N_TRIALS = 10


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 003: Window Size Sensitivity")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Build engram library (300 requests per attack type)
    # ===================================================================
    print("\nPHASE 1: Build engram library (4 HTTP attack types)")

    library = EngramLibrary(dim=DIM)
    for name, gen_fn in ATTACK_GENERATORS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(300):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i + hash(name) % 10000)))
            sub.update(vec)
        library.add(name, sub, attack_type=name)
    print(f"  Library: {library}")

    # ===================================================================
    # PHASE 2: Sweep window sizes
    # ===================================================================
    print(f"\nPHASE 2: Classification accuracy vs window size")
    print(f"  Sizes: {WINDOW_SIZES}, Trials: {N_TRIALS} per type per size")

    accuracy_by_size = {}

    for ws in WINDOW_SIZES:
        total_correct = 0
        total_count = 0
        per_type = {}

        for true_type, gen_fn in ATTACK_GENERATORS.items():
            correct = 0
            for trial in range(N_TRIALS):
                seed = trial * 1000 + ws * 100 + hash(true_type) % 10000 + 60000
                sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
                for i in range(ws):
                    vec = encoder.encode_walkable(
                        gen_fn(np.random.default_rng(i + seed)))
                    sub.update(vec)

                matches = library.match_spectrum(sub.eigenvalues, top_k=1)
                if matches and matches[0][0] == true_type:
                    correct += 1

            per_type[true_type] = correct / N_TRIALS
            total_correct += correct
            total_count += N_TRIALS

        accuracy_by_size[ws] = {"overall": total_correct / total_count,
                                "per_type": per_type}
        print(f"  ws={ws:>4}: {total_correct}/{total_count} "
              f"({total_correct/total_count*100:.0f}%)")

    # ===================================================================
    # PHASE 3: Results table
    # ===================================================================
    print("\n" + "-" * 70)
    print("RESULTS: Accuracy vs Window Size")
    print("-" * 70)

    header = f"  {'Window':>8}"
    for name in ATTACK_GENERATORS:
        header += f"  {name:>12}"
    header += f"  {'OVERALL':>10}"
    print(header)
    print("  " + "-" * (8 + 14 * len(ATTACK_GENERATORS) + 12))

    for ws in WINDOW_SIZES:
        row = f"  {ws:>8}"
        for name in ATTACK_GENERATORS:
            acc = accuracy_by_size[ws]["per_type"][name]
            row += f"  {acc:>11.0%}"
        row += f"  {accuracy_by_size[ws]['overall']:>9.0%}"
        print(row)

    # ===================================================================
    # PHASE 4: Analysis
    # ===================================================================
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    overall_accs = [accuracy_by_size[ws]["overall"] for ws in WINDOW_SIZES]

    monotonic = all(overall_accs[i] >= overall_accs[i-1] - 0.05
                    for i in range(1, len(overall_accs)))

    knee_size = next((ws for ws in WINDOW_SIZES
                      if accuracy_by_size[ws]["overall"] >= 0.7), None)

    saturation_size = next(
        (WINDOW_SIZES[i] for i in range(1, len(WINDOW_SIZES))
         if overall_accs[i] - overall_accs[i-1] < 0.05 and overall_accs[i] > 0.7),
        None)

    print(f"  Monotonically non-decreasing: {monotonic}")
    print(f"  First size with >70% accuracy: {knee_size}")
    print(f"  Saturation point (gain < 5%):  {saturation_size}")
    print(f"  Accuracy curve: {' → '.join(f'{a:.0%}' for a in overall_accs)}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Reliable matching (>70%) achievable in <= 100 requests",
            knee_size is not None and knee_size <= 100,
            f"knee_size={knee_size}",
        ),
        (
            "Accuracy curve is approximately monotonic",
            monotonic,
            f"accs={[f'{a:.0%}' for a in overall_accs]}",
        ),
        (
            "Largest window achieves >80% accuracy",
            overall_accs[-1] > 0.8,
            f"acc_at_{WINDOW_SIZES[-1]}={overall_accs[-1]:.0%}",
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
