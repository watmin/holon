#!/usr/bin/env python3
"""
Eigenvalue Matching as Pre-Filter for Per-Request Scoring

HYPOTHESIS:
===========
A dual-signal pre-filter (spectrum + alignment) can narrow the candidate
set before expensive per-request residual scoring. Spectrum (magnitude)
narrows to the right attack TYPE. Alignment (direction) disambiguates
VARIANTS within a type. Together they should match brute-force accuracy
while cutting compute by 75%.

This builds directly on experiment 004's finding: spectrum alone cannot
reliably separate subspaces because it only measures variance shape, not
orientation. The pre-filter needs both signals.

Three pipelines compared:
  - Spectrum-only:  match_spectrum → top-2 → residual
  - Dual-signal:    spectrum × alignment → top-2 → residual
  - Brute-force:    residual against all 8 engrams

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable          - Rich structured HTTP request encoding
2. EngramLibrary.match_spectrum     - Magnitude signal (tier 1a)
3. EngramLibrary.match_alignment    - Directional signal (tier 1b)
4. Engram.residual                  - Per-request residual scoring (tier 2)
5. Dual-signal vs spectrum-only vs brute-force comparison

SCENARIO:
=========
WAF context. 8 engrams: 4 attack types × 2 parameter variants each.
For 200 probe windows, compare two-stage (spectrum → top-2 → residual)
against brute-force (residual against all 8).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon.kernel.encoder import Encoder
from holon.kernel.vector_manager import VectorManager
from holon.kernel.walkable import LinearScale
from holon.memory.engram import EngramLibrary
from holon.memory.subspace import OnlineSubspace

DIM = 4096


# ---------------------------------------------------------------------------
# Traffic generators — each attack TYPE has a different count of varying
# fields to create distinct eigenvalue spectrum shapes.
# ---------------------------------------------------------------------------

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


# --- GET FLOOD: ~1 varying field (UA only) → peaked eigenvalue spectrum ---

def make_get_flood_fast(rng):
    ua = str(rng.choice(["curl/8.0.1", "curl/7.88.1",
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


def make_get_flood_slow(rng):
    ua = str(rng.choice(["python-requests/2.31.0", "python-requests/2.28.0",
                          "python-urllib3/1.26.18"]))
    return {
        "method": "GET", "path": "/api/users",
        "path_parts": ["", "api", "users"], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Encoding",
                         "Connection"],
        "headers": [["Host", "example.com"], ["User-Agent", ua],
                    ["Accept", "*/*"],
                    ["Accept-Encoding", "gzip, deflate"],
                    ["Connection", "keep-alive"]],
        "header_count": LinearScale(5), "has_cookie": "false",
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


# --- CREDENTIAL STUFFING: ~5 varying fields → moderate eigenvalue spread ---

def make_cred_stuff_json(rng):
    body_len = rng.integers(80, 200)
    ua = str(rng.choice(["python-requests/2.31.0", "python-requests/2.28.0"]))
    accept = str(rng.choice(["*/*", "application/json"]))
    encoding = str(rng.choice(["gzip, deflate", "gzip, deflate, br", "gzip"]))
    conn = str(rng.choice(["keep-alive", "close"]))
    return {
        "method": "POST", "path": "/api/v1/auth/login",
        "path_parts": ["", "api", "v1", "auth", "login"],
        "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Content-Type",
                         "Content-Length", "Accept-Encoding", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", accept], ["Content-Type", "application/json"],
            ["Content-Length", str(body_len)],
            ["Accept-Encoding", encoding], ["Connection", conn],
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


def make_cred_stuff_form(rng):
    body_len = rng.integers(40, 120)
    ua = str(rng.choice(["Mozilla/5.0 (compatible; LoginBot/1.0)",
                          "Mozilla/5.0 (compatible; LoginBot/2.0)"]))
    accept = str(rng.choice(["text/html", "text/html, */*"]))
    encoding = str(rng.choice(["gzip, deflate", "gzip"]))
    conn = str(rng.choice(["keep-alive", "close"]))
    return {
        "method": "POST", "path": "/api/v1/auth/login",
        "path_parts": ["", "api", "v1", "auth", "login"],
        "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Content-Type",
                         "Content-Length", "Accept", "Accept-Encoding",
                         "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Content-Type", "application/x-www-form-urlencoded"],
            ["Content-Length", str(body_len)],
            ["Accept", accept], ["Accept-Encoding", encoding],
            ["Connection", conn],
        ],
        "header_count": LinearScale(7), "has_cookie": "false",
        "body_len": LinearScale(body_len),
        "tls": {
            "version": "TLS1.2",
            "ciphers": {"ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"},
            "cipher_order": ["ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA"],
            "ext_types": {"server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms"},
            "groups": {"secp256r1"}, "alpn": ["http/1.1"],
        },
    }


# --- SCRAPER: ~8 varying fields → wide eigenvalue spread ---

def make_scraper_scrapy(rng):
    category = str(rng.choice(["electronics", "books", "clothing", "toys"]))
    pid = str(rng.integers(1, 999))
    depth = str(rng.integers(1, 5))
    ref_cat = str(rng.choice(["electronics", "books", "clothing", "toys"]))
    accept = str(rng.choice(["text/html", "text/html, application/xhtml+xml",
                              "text/html, */*"]))
    lang = str(rng.choice(["en-US", "en-GB", "de-DE", "fr-FR"]))
    encoding = str(rng.choice(["gzip, deflate", "gzip, deflate, br", "gzip"]))
    conn = str(rng.choice(["keep-alive", "close"]))
    return {
        "method": "GET", "path": f"/products/{category}/{pid}",
        "path_parts": ["", "products", category, pid], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Referer", "X-Crawl-Depth",
                         "Connection"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", "Scrapy/2.11.0 (+https://scrapy.org)"],
            ["Accept", accept], ["Accept-Language", lang],
            ["Accept-Encoding", encoding],
            ["Referer", f"https://example.com/products/{ref_cat}"],
            ["X-Crawl-Depth", depth], ["Connection", conn],
        ],
        "header_count": LinearScale(8), "has_cookie": "false",
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


def make_scraper_wget(rng):
    section = str(rng.choice(["blog", "wiki", "docs", "help"]))
    page = str(rng.integers(1, 500))
    ref_section = str(rng.choice(["blog", "wiki", "docs", "help"]))
    accept = str(rng.choice(["*/*", "text/html", "text/html, */*"]))
    lang = str(rng.choice(["en-US", "en-GB", "ja-JP", "zh-CN"]))
    encoding = str(rng.choice(["gzip, deflate", "gzip", "identity"]))
    conn = str(rng.choice(["keep-alive", "close"]))
    return {
        "method": "GET", "path": f"/pages/{section}/{page}",
        "path_parts": ["", "pages", section, page], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Referer", "Connection"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", "Wget/1.21.4 (linux-gnu)"],
            ["Accept", accept], ["Accept-Language", lang],
            ["Accept-Encoding", encoding],
            ["Referer", f"https://example.com/pages/{ref_section}"],
            ["Connection", conn],
        ],
        "header_count": LinearScale(7), "has_cookie": "false",
        "tls": {
            "version": "TLS1.2",
            "ciphers": {"ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA",
                        "AES128-SHA"},
            "cipher_order": ["ECDHE-RSA-AES128-GCM-SHA256", "AES256-SHA",
                             "AES128-SHA"],
            "ext_types": {"server_name", "supported_groups",
                          "ec_point_formats", "signature_algorithms"},
            "groups": {"secp256r1"}, "alpn": ["http/1.1"],
        },
    }


# --- TLS SHUFFLE: ~15 varying fields via random set membership → very wide ---

def make_tls_shuffle_v1(rng):
    all_ciphers = ["ECDHE-RSA-AES128-GCM-SHA256", "ECDHE-RSA-AES256-GCM-SHA384",
                   "AES256-SHA", "AES128-SHA", "DES-CBC3-SHA"]
    n_c = rng.integers(2, len(all_ciphers) + 1)
    ciphers = list(rng.choice(all_ciphers, size=n_c, replace=False))
    all_exts = ["server_name", "supported_groups", "ec_point_formats",
                "signature_algorithms", "status_request",
                "signed_certificate_timestamp", "session_ticket"]
    n_e = rng.integers(3, len(all_exts) + 1)
    exts = list(rng.choice(all_exts, size=n_e, replace=False))
    all_groups = ["secp256r1", "secp384r1", "secp521r1"]
    n_g = rng.integers(1, len(all_groups) + 1)
    groups = list(rng.choice(all_groups, size=n_g, replace=False))
    ua = str(rng.choice(["Mozilla/5.0 Bot/1.0", "CustomClient/3.2",
                          "Mozilla/5.0 (compatible; Scanner/1.0)"]))
    path = str(rng.choice(["/", "/api/search", "/api/users", "/admin",
                            "/wp-login.php", "/.env"]))
    return {
        "method": "GET", "path": path, "path_parts": path.split("/"),
        "version": str(rng.choice(["HTTP/1.1", "HTTP/1.0"])),
        "header_order": ["Host", "User-Agent", "Accept"],
        "headers": [["Host", "example.com"], ["User-Agent", ua],
                    ["Accept", "*/*"]],
        "header_count": LinearScale(3), "has_cookie": "false",
        "tls": {
            "version": "TLS1.2",
            "ciphers": set(ciphers), "cipher_order": ciphers,
            "ext_types": set(exts), "groups": set(groups),
            "alpn": ["http/1.1"],
        },
    }


def make_tls_shuffle_v2(rng):
    all_ciphers = ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                   "TLS_CHACHA20_POLY1305_SHA256"]
    n_c = rng.integers(1, len(all_ciphers) + 1)
    ciphers = list(rng.choice(all_ciphers, size=n_c, replace=False))
    all_exts = ["server_name", "supported_versions", "key_share",
                "signature_algorithms", "supported_groups",
                "psk_key_exchange_modes", "application_layer_protocol_negotiation",
                "status_request", "signed_certificate_timestamp"]
    n_e = rng.integers(3, len(all_exts) + 1)
    exts = list(rng.choice(all_exts, size=n_e, replace=False))
    all_groups = ["x25519", "secp256r1", "secp384r1"]
    n_g = rng.integers(1, len(all_groups) + 1)
    groups = list(rng.choice(all_groups, size=n_g, replace=False))
    ua = str(rng.choice(["HeadlessChrome/120.0", "PhantomJS/2.1.1",
                          "Mozilla/5.0 (compatible; Bot/2.0)"]))
    path = str(rng.choice(["/", "/api/search", "/products", "/login",
                            "/admin/config", "/robots.txt"]))
    return {
        "method": "GET", "path": path, "path_parts": path.split("/"),
        "version": str(rng.choice(["HTTP/2", "HTTP/1.1"])),
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Encoding"],
        "headers": [["Host", "example.com"], ["User-Agent", ua],
                    ["Accept", "*/*"],
                    ["Accept-Encoding", "gzip, deflate, br"]],
        "header_count": LinearScale(4), "has_cookie": "false",
        "tls": {
            "version": "TLS1.3",
            "ciphers": set(ciphers), "cipher_order": ciphers,
            "ext_types": set(exts), "groups": set(groups),
            "alpn": ["h2", "http/1.1"],
        },
    }


ENGRAM_GENERATORS = {
    "get_flood_fast":   make_get_flood_fast,
    "get_flood_slow":   make_get_flood_slow,
    "cred_stuff_json":  make_cred_stuff_json,
    "cred_stuff_form":  make_cred_stuff_form,
    "scraper_scrapy":   make_scraper_scrapy,
    "scraper_wget":     make_scraper_wget,
    "tls_shuffle_v1":   make_tls_shuffle_v1,
    "tls_shuffle_v2":   make_tls_shuffle_v2,
}

ATTACK_TYPE_MAP = {
    "get_flood_fast":   "get_flood",
    "get_flood_slow":   "get_flood",
    "cred_stuff_json":  "cred_stuffing",
    "cred_stuff_form":  "cred_stuffing",
    "scraper_scrapy":   "scraper",
    "scraper_wget":     "scraper",
    "tls_shuffle_v1":   "tls_shuffle",
    "tls_shuffle_v2":   "tls_shuffle",
}

PROBES_PER_TYPE = 50
PROBE_WINDOW_SIZE = 50
REQUESTS_TO_SCORE = 10


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 005: Eigenvalue Pre-Filter for Per-Request Scoring")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Build engram library (8 engrams, 300 requests each)
    # ===================================================================
    print("\nPHASE 1: Build engram library (4 attack types × 2 variants = 8)")

    library = EngramLibrary(dim=DIM)
    for name, gen_fn in ENGRAM_GENERATORS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(300):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i + hash(name) % 10000)))
            sub.update(vec)
        library.add(name, sub, attack_type=ATTACK_TYPE_MAP[name])
        print(f"  Added: {name} (type={ATTACK_TYPE_MAP[name]}, "
              f"eig_energy={np.sum(sub.eigenvalues**2):.1f})")

    print(f"  Library: {library}")

    # ===================================================================
    # PHASE 2: Three pipelines compared
    # ===================================================================
    print(f"\nPHASE 2: Compare three pipelines "
          f"({PROBES_PER_TYPE} probes × 4 attack types)")
    print(f"  Spectrum-only: match_spectrum → top-2 → residual")
    print(f"  Dual-signal:   spectrum × alignment → top-2 → residual")
    print(f"  Brute-force:   residual against all 8 engrams")

    attack_types = ["get_flood", "cred_stuffing", "scraper", "tls_shuffle"]
    type_to_generators = {}
    for name, atype in ATTACK_TYPE_MAP.items():
        type_to_generators.setdefault(atype, []).append(name)

    spec_only_correct = 0
    dual_correct = 0
    brute_force_correct = 0
    total_probes = 0
    spec_agreement = 0
    dual_agreement = 0
    dual_improvement_cases = []

    t_spec_total = 0.0
    t_dual_total = 0.0
    t_brute_total = 0.0

    per_type_results = {at: {"spec": 0, "dual": 0, "brute": 0, "total": 0}
                        for at in attack_types}

    for attack_type in attack_types:
        gen_names = type_to_generators[attack_type]

        for probe_idx in range(PROBES_PER_TYPE):
            true_name = gen_names[probe_idx % len(gen_names)]
            gen_fn = ENGRAM_GENERATORS[true_name]
            seed_base = probe_idx * 1000 + hash(true_name) % 10000 + 60000

            probe_sub = OnlineSubspace(dim=DIM, k=64, amnesia=1.0)
            for i in range(PROBE_WINDOW_SIZE):
                vec = encoder.encode_walkable(
                    gen_fn(np.random.default_rng(i + seed_base)))
                probe_sub.update(vec)

            probe_vecs = []
            for i in range(REQUESTS_TO_SCORE):
                vec = encoder.encode_walkable(
                    gen_fn(np.random.default_rng(i + seed_base + 100000)))
                probe_vecs.append(vec)

            # --- Pipeline A: Spectrum-only pre-filter → top-2 → residual ---
            t0 = time.perf_counter()
            n_reliable = min(probe_sub.n // 2, 64)
            spec_matches = library.match_spectrum(
                probe_sub.eigenvalues[:n_reliable], top_k=2)
            spec_candidates = [m[0] for m in spec_matches]

            spec_scores = {}
            for cname in spec_candidates:
                engram = library.get(cname)
                avg_res = np.mean([engram.residual(v) for v in probe_vecs])
                spec_scores[cname] = avg_res
            spec_best = min(spec_scores, key=spec_scores.get)
            t_spec_total += time.perf_counter() - t0

            # --- Pipeline B: Dual-signal pre-filter → top-2 → residual ---
            t0 = time.perf_counter()
            spec_all = library.match_spectrum(
                probe_sub.eigenvalues[:n_reliable], top_k=8)
            align_all = library.match_alignment(probe_sub, top_k=8)

            spec_dict = {n: s for n, s in spec_all}
            align_dict = {n: s for n, s in align_all}
            combined_scores = {
                n: spec_dict.get(n, 0) * align_dict.get(n, 0)
                for n in library.names()
            }
            dual_candidates = sorted(
                combined_scores, key=combined_scores.get, reverse=True)[:2]

            dual_scores = {}
            for cname in dual_candidates:
                engram = library.get(cname)
                avg_res = np.mean([engram.residual(v) for v in probe_vecs])
                dual_scores[cname] = avg_res
            dual_best = min(dual_scores, key=dual_scores.get)
            t_dual_total += time.perf_counter() - t0

            # --- Pipeline C: Brute-force residual against all 8 ---
            t0 = time.perf_counter()
            brute_scores = {}
            for ename in library.names():
                engram = library.get(ename)
                avg_res = np.mean([engram.residual(v) for v in probe_vecs])
                brute_scores[ename] = avg_res
            brute_best = min(brute_scores, key=brute_scores.get)
            t_brute_total += time.perf_counter() - t0

            total_probes += 1
            per_type_results[attack_type]["total"] += 1

            if spec_best == true_name:
                spec_only_correct += 1
                per_type_results[attack_type]["spec"] += 1
            if dual_best == true_name:
                dual_correct += 1
                per_type_results[attack_type]["dual"] += 1
            if brute_best == true_name:
                brute_force_correct += 1
                per_type_results[attack_type]["brute"] += 1

            if spec_best == brute_best:
                spec_agreement += 1
            if dual_best == brute_best:
                dual_agreement += 1

            if dual_best == brute_best and spec_best != brute_best:
                dual_improvement_cases.append({
                    "true": true_name,
                    "spec_pick": spec_best,
                    "dual_pick": dual_best,
                    "brute_pick": brute_best,
                })

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Spectrum-Only vs Dual-Signal vs Brute-Force")
    print("=" * 70)

    spec_acc = spec_only_correct / total_probes * 100
    dual_acc = dual_correct / total_probes * 100
    brute_acc = brute_force_correct / total_probes * 100
    spec_agree_pct = spec_agreement / total_probes * 100
    dual_agree_pct = dual_agreement / total_probes * 100

    spec_ratio = spec_acc / brute_acc if brute_acc > 0 else 0.0
    dual_ratio = dual_acc / brute_acc if brute_acc > 0 else 0.0

    print(f"\n  Total probes: {total_probes}")
    print(f"\n  {'Pipeline':<22} {'Accuracy':>10} {'vs Brute':>10} "
          f"{'Agreement':>10}")
    print("  " + "-" * 55)
    print(f"  {'Spectrum-only':<22} {spec_acc:>9.1f}% {spec_ratio:>9.2f}x "
          f"{spec_agree_pct:>9.1f}%")
    print(f"  {'Dual (spec × align)':<22} {dual_acc:>9.1f}% {dual_ratio:>9.2f}x "
          f"{dual_agree_pct:>9.1f}%")
    print(f"  {'Brute-force':<22} {brute_acc:>9.1f}%      1.00x    100.0%")

    print(f"\n  Per-type breakdown:")
    print(f"    {'Attack type':<18} {'Spec-only':>10} {'Dual':>10} "
          f"{'Brute':>10}")
    print("    " + "-" * 50)
    for at in attack_types:
        r = per_type_results[at]
        s_pct = r["spec"] / r["total"] * 100 if r["total"] else 0
        d_pct = r["dual"] / r["total"] * 100 if r["total"] else 0
        b_pct = r["brute"] / r["total"] * 100 if r["total"] else 0
        print(f"    {at:<18} {s_pct:>9.1f}% {d_pct:>9.1f}% {b_pct:>9.1f}%")

    compute_saving = (1.0 - 2.0 / len(ENGRAM_GENERATORS)) * 100
    print(f"\n  Compute savings: {compute_saving:.0f}% "
          f"(2/{len(ENGRAM_GENERATORS)} engrams scored per request)")
    print(f"  Wall-clock: spec={t_spec_total:.3f}s, dual={t_dual_total:.3f}s, "
          f"brute={t_brute_total:.3f}s")

    if dual_improvement_cases:
        print(f"\n  Cases where dual-signal beat spectrum-only "
              f"({len(dual_improvement_cases)}):")
        for dc in dual_improvement_cases[:5]:
            print(f"    true={dc['true']:<20} spec→{dc['spec_pick']:<20} "
                  f"dual→{dc['dual_pick']}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Dual-signal accuracy >= 95% of brute-force",
            dual_ratio >= 0.95,
            f"ratio={dual_ratio:.4f} "
            f"(dual={dual_acc:.1f}%, brute={brute_acc:.1f}%)",
        ),
        (
            "Dual-signal agreement with brute-force > 90%",
            dual_agree_pct > 90,
            f"agreement={dual_agree_pct:.1f}%",
        ),
        (
            "Dual-signal >= spectrum-only accuracy",
            dual_acc >= spec_acc,
            f"dual={dual_acc:.1f}% vs spec={spec_acc:.1f}%",
        ),
        (
            "Compute savings = 75% (2/8 engrams scored)",
            compute_saving == 75.0,
            f"saving={compute_saving:.0f}%",
        ),
        (
            "Brute-force accuracy > 50% (baseline sanity)",
            brute_acc > 50,
            f"acc={brute_acc:.1f}%",
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
