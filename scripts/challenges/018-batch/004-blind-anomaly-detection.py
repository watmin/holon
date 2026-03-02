#!/usr/bin/env python3
"""
Blind Anomaly Detection — Unknown Attack Shape

HYPOTHESIS:
===========
Eigenvalue spectrum similarity (match_spectrum) captures variance SHAPE
but not DIRECTION. A novel attack can have similar variance distribution
but different principal component orientations. By adding a directional
signal — subspace alignment via principal angles — we can separate known
from unknown attacks.

Two complementary signals:
  - match_spectrum:   eigenvalue shape similarity (magnitude)
  - match_alignment:  principal angle alignment (direction)

Known attacks should score high on BOTH. Unknown attacks may score high
on spectrum (similar variance shape) but LOW on alignment (different
directions in 4096-space).

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable          - Rich structured HTTP request encoding
2. EngramLibrary.match_spectrum     - Magnitude signal (variance shape)
3. EngramLibrary.match_alignment    - Directional signal (subspace orientation)
4. Combined gating for novelty      - Both signals required for "known"

SCENARIO:
=========
WAF context. Build library from 3 known HTTP attack types (GET flood,
credential stuffing, scraper). Generate windows with a 4th unknown type
(TLS-randomized flood) that is NOT in the library. Also generate normal,
known-attack, and mixed windows. The unknown should have low similarity
to everything — distinct from both known attacks (high match) and normal
(no match needed, that's Layer 0/1's job).
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
    """The UNKNOWN attack — not in the library."""
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


KNOWN_ATTACKS = {
    "get_flood": make_get_flood,
    "cred_stuffing": make_cred_stuff,
    "scraper": make_scraper,
}


def build_window(encoder, gen_fn, n=200, seed_offset=0):
    sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(n):
        vec = encoder.encode_walkable(
            gen_fn(np.random.default_rng(i + seed_offset)))
        sub.update(vec)
    return sub


def build_mixed_window(encoder, gen_a, gen_b, ratio_a, n=200, seed_offset=0):
    sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(n):
        r = np.random.default_rng(i + seed_offset)
        if r.random() < ratio_a:
            d = gen_a(r)
        else:
            d = gen_b(r)
        vec = encoder.encode_walkable(d)
        sub.update(vec)
    return sub


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 004: Blind Anomaly Detection — Unknown Attack Shape")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Build library from 3 KNOWN attack types only
    # ===================================================================
    print("\nPHASE 1: Build library (3 known attack types — TLS shuffle excluded)")

    library = EngramLibrary(dim=DIM)
    for name, gen_fn in KNOWN_ATTACKS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(300):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i + hash(name) % 10000)))
            sub.update(vec)
        library.add(name, sub, attack_type=name)
        print(f"  Added: {name}")

    print(f"  Library: {library}")
    print(f"  NOT in library: tls_shuffle (the unknown)")

    # ===================================================================
    # PHASE 2: Score windows — spectrum (magnitude) vs alignment (direction)
    # ===================================================================
    print("\nPHASE 2: Score windows — dual signal (spectrum + alignment)")
    print(f"  {'Category':<25} {'Spectrum':>10} {'Alignment':>10}  "
          f"{'Combined':>10}")
    print("  " + "-" * 60)

    n_trials = 10

    known_spectrums = {}
    known_alignments = {}
    known_combined = {}

    for attack_name, gen_fn in KNOWN_ATTACKS.items():
        specs, aligns, combs = [], [], []
        for t in range(n_trials):
            w = build_window(encoder, gen_fn, seed_offset=t*1000 + 70000)
            spec_matches = library.match_spectrum(w.eigenvalues, top_k=3)
            align_matches = library.match_alignment(w, top_k=3)
            max_spec = spec_matches[0][1] if spec_matches else 0.0
            max_align = align_matches[0][1] if align_matches else 0.0
            specs.append(max_spec)
            aligns.append(max_align)
            combs.append(max_spec * max_align)
        known_spectrums[attack_name] = specs
        known_alignments[attack_name] = aligns
        known_combined[attack_name] = combs
        print(f"  Known {attack_name:<18} {np.mean(specs):>10.4f} "
              f"{np.mean(aligns):>10.4f}  {np.mean(combs):>10.4f}")

    # Unknown attack windows (tls_shuffle — NOT in library)
    unknown_specs, unknown_aligns, unknown_combs = [], [], []
    for t in range(n_trials):
        w = build_window(encoder, make_tls_shuffle,
                        seed_offset=t*1000 + 80000)
        spec_matches = library.match_spectrum(w.eigenvalues, top_k=3)
        align_matches = library.match_alignment(w, top_k=3)
        max_spec = spec_matches[0][1] if spec_matches else 0.0
        max_align = align_matches[0][1] if align_matches else 0.0
        unknown_specs.append(max_spec)
        unknown_aligns.append(max_align)
        unknown_combs.append(max_spec * max_align)
    print(f"  Unknown tls_shuffle    {np.mean(unknown_specs):>10.4f} "
          f"{np.mean(unknown_aligns):>10.4f}  {np.mean(unknown_combs):>10.4f}")

    # Mixed (50/50 normal + unknown)
    mixed_specs, mixed_aligns, mixed_combs = [], [], []
    for t in range(n_trials):
        w = build_mixed_window(encoder, make_normal, make_tls_shuffle,
                               ratio_a=0.5, seed_offset=t*1000 + 85000)
        spec_matches = library.match_spectrum(w.eigenvalues, top_k=3)
        align_matches = library.match_alignment(w, top_k=3)
        max_spec = spec_matches[0][1] if spec_matches else 0.0
        max_align = align_matches[0][1] if align_matches else 0.0
        mixed_specs.append(max_spec)
        mixed_aligns.append(max_align)
        mixed_combs.append(max_spec * max_align)
    print(f"  Mixed normal+unknown   {np.mean(mixed_specs):>10.4f} "
          f"{np.mean(mixed_aligns):>10.4f}  {np.mean(mixed_combs):>10.4f}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Spectrum-Only vs Alignment vs Combined")
    print("=" * 70)

    known_min_spec = min(np.mean(known_spectrums[n]) for n in KNOWN_ATTACKS)
    known_min_align = min(np.mean(known_alignments[n]) for n in KNOWN_ATTACKS)
    known_min_combined = min(np.mean(known_combined[n]) for n in KNOWN_ATTACKS)

    unknown_max_spec = max(unknown_specs)
    unknown_max_align = max(unknown_aligns)
    unknown_max_combined = max(unknown_combs)

    spec_gap = known_min_spec - unknown_max_spec
    align_gap = known_min_align - unknown_max_align
    combined_gap = known_min_combined - unknown_max_combined

    print(f"\n  {'Signal':<20} {'Known min':>10} {'Unknown max':>12} {'Gap':>10}")
    print("  " + "-" * 55)
    print(f"  {'Spectrum only':<20} {known_min_spec:>10.4f} "
          f"{unknown_max_spec:>12.4f} {spec_gap:>10.4f}")
    print(f"  {'Alignment only':<20} {known_min_align:>10.4f} "
          f"{unknown_max_align:>12.4f} {align_gap:>10.4f}")
    print(f"  {'Combined (S*A)':<20} {known_min_combined:>10.4f} "
          f"{unknown_max_combined:>12.4f} {combined_gap:>10.4f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Spectrum alone: negative or near-zero gap (motivation)",
            spec_gap < 0.02,
            f"gap={spec_gap:.4f}",
        ),
        (
            "Alignment provides separation where spectrum doesn't",
            align_gap > spec_gap,
            f"align_gap={align_gap:.4f} vs spec_gap={spec_gap:.4f}",
        ),
        (
            "Alignment gap > 3x spectrum gap (meaningful improvement)",
            align_gap > 3 * max(spec_gap, 0.001),
            f"align_gap={align_gap:.4f} vs 3*spec={3*max(spec_gap, 0.001):.4f}",
        ),
        (
            "Combined: known min > unknown max (correct ordering)",
            known_min_combined > unknown_max_combined,
            f"known_min={known_min_combined:.4f} vs "
            f"unknown_max={unknown_max_combined:.4f}",
        ),
        (
            "Combined gap is positive and actionable (> 0.02)",
            combined_gap > 0.02,
            f"gap={combined_gap:.4f}",
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
