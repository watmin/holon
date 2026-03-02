#!/usr/bin/env python3
"""
Engram Promotion (CI/CD Training Pipeline)

HYPOTHESIS:
===========
Engrams learned from integration test traffic in pre-production can be deployed
as artifacts alongside application code. On Day 1 in production, the firewall
already knows what "normal" looks like for new features — no cold-start, no
blind window.

A preprod sidecar observes integration test traffic for a new endpoint, detects
that it doesn't match any existing engram (library miss), and mints a new
engram. That engram generalizes from structured-but-limited test traffic to
the more varied production traffic for the same endpoint. Attacks against the
new endpoint are still rejected, and existing v1 patterns suffer no regression.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable    - Rich structured HTTP/TLS encoding
2. OnlineSubspace             - Per-endpoint manifold learning
3. EngramLibrary.add          - Engram minting and storage
4. EngramLibrary.save / load  - Library serialization round-trip
5. Engram.residual            - Per-vector anomaly scoring
6. Library miss detection     - Residual > threshold across all engrams

SCENARIO:
=========
Phase 1: Train v1 library with 2 engrams (/api/users, /api/search).
Phase 2: Preprod sidecar detects library miss on new /api/v2/recommendations
         test traffic, mints a new engram, saves v2 library.
Phase 3: Load v2 in production. Score 500 real reco requests (should match)
         and 200 attacks (should reject).
Phase 4: Regression check — v1 traffic still matches in v2 library.
Phase 5: Engram diff — exactly 1 new engram between v1 and v2.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon.kernel.encoder import Encoder
from holon.kernel.vector_manager import VectorManager
from holon.kernel.walkable import LinearScale
from holon.memory.engram import EngramLibrary
from holon.memory.subspace import OnlineSubspace

DIM = 4096


def make_users_traffic(rng):
    """Existing /api/users traffic — v1 baseline."""
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]))
    return {
        "method": str(rng.choice(["GET", "GET", "POST"])),
        "path": "/api/users", "path_parts": ["", "api", "users"], "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Cookie", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", "application/json"],
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


def make_search_traffic(rng):
    """Existing /api/search traffic — v1 baseline."""
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]))
    query = str(rng.choice(["shoes", "laptop", "phone case", "headphones", "book"]))
    return {
        "method": "GET",
        "path": f"/api/search?q={query}", "path_parts": ["", "api", f"search?q={query}"],
        "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Cookie", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", "application/json"],
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


def make_reco_test(rng):
    """Integration test traffic for /api/v2/recommendations — LLM agent driven.

    Structurally matches production: real browser UAs, cookies, full headers.
    Limited variety in user IDs and paths (controlled test scenarios), but
    the HTTP structure is production-realistic because the LLM agents use
    real browsers.
    """
    user_id = str(rng.integers(1, 50))
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]))
    return {
        "method": "GET",
        "path": f"/api/v2/recommendations?user={user_id}",
        "path_parts": ["", "api", "v2", f"recommendations?user={user_id}"],
        "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Authorization",
                         "Accept-Language", "Accept-Encoding", "Cookie"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", "application/json"],
            ["Authorization", f"Bearer test_token_{rng.integers(1000, 9999)}"],
            ["Accept-Language", "en-US,en;q=0.9"],
            ["Accept-Encoding", "gzip, deflate, br"],
            ["Cookie", f"session={rng.integers(1000, 9999)}"],
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


def make_reco_prod(rng):
    """Production traffic for /api/v2/recommendations — more varied than tests."""
    user_id = str(rng.integers(1, 100000))
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "ExampleApp/4.2.0 (iPhone; iOS 17.2)",
    ]))
    return {
        "method": "GET",
        "path": f"/api/v2/recommendations?user={user_id}",
        "path_parts": ["", "api", "v2", f"recommendations?user={user_id}"],
        "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Authorization",
                         "Accept-Language", "Accept-Encoding", "Cookie"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", "application/json"],
            ["Authorization", f"Bearer prod_{rng.integers(100000, 999999)}"],
            ["Accept-Language", str(rng.choice(["en-US,en;q=0.9", "fr-FR,fr;q=0.9"]))],
            ["Accept-Encoding", "gzip, deflate, br"],
            ["Cookie", f"session={rng.integers(10000, 99999)}"],
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


def make_reco_attack(rng):
    """Attack traffic targeting the new endpoint."""
    user_id = str(rng.integers(1, 100000))
    return {
        "method": "GET",
        "path": f"/api/v2/recommendations?user={user_id}",
        "path_parts": ["", "api", "v2", f"recommendations?user={user_id}"],
        "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", str(rng.choice(["python-requests/2.31.0", "curl/8.0.1"]))],
            ["Accept", "*/*"],
        ],
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


def encode_batch(encoder, gen_fn, n, seed_base):
    vecs = []
    for i in range(n):
        vec = encoder.encode_walkable(gen_fn(np.random.default_rng(seed_base + i)))
        vecs.append(vec)
    return vecs


def library_miss(library, vec):
    """True if vec doesn't match any engram (residual > threshold for all)."""
    for name in library.names():
        engram = library.get(name)
        if engram.residual(vec) <= engram.subspace.threshold:
            return False
    return True


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 016: Engram Promotion (CI/CD Training Pipeline)")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Train v1 library with existing endpoints
    # ===================================================================
    print("\nPHASE 1: Train v1 library (/api/users + /api/search)")

    v1_library = EngramLibrary(dim=DIM)
    v1_subspaces = {}

    for name, gen_fn, seed_base in [
        ("api_users", make_users_traffic, 0),
        ("api_search", make_search_traffic, 10000),
    ]:
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(500):
            vec = encoder.encode_walkable(gen_fn(np.random.default_rng(seed_base + i)))
            sub.update(vec)
        v1_subspaces[name] = sub
        v1_library.add(name, sub, endpoint=name)
        print(f"  {name}: threshold={sub.threshold:.4f}, n={sub.n}")

    print(f"  v1 library: {v1_library}")

    # ===================================================================
    # PHASE 2: Preprod integration tests — detect miss, mint new engram
    # ===================================================================
    print("\nPHASE 2: Preprod sidecar — 200 integration test requests for "
          "/api/v2/recommendations")

    reco_test_vecs = encode_batch(encoder, make_reco_test, 200, seed_base=20000)

    miss_count = sum(1 for v in reco_test_vecs if library_miss(v1_library, v))
    miss_rate = miss_count / len(reco_test_vecs)
    print(f"  Library misses: {miss_count}/{len(reco_test_vecs)} ({miss_rate:.0%})")

    reco_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for vec in reco_test_vecs:
        reco_sub.update(vec)

    print(f"  New engram trained: threshold={reco_sub.threshold:.4f}, n={reco_sub.n}")

    v2_library = EngramLibrary(dim=DIM)
    for name in v1_library.names():
        engram = v1_library.get(name)
        v2_library.add(name, engram.subspace, endpoint=name)
    v2_library.add("api_v2_recommendations", reco_sub, endpoint="api_v2_recommendations")

    print(f"  v2 library: {v2_library}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        v2_path = f.name
    v2_library.save(v2_path)
    print(f"  Saved v2 library to {v2_path}")

    # ===================================================================
    # PHASE 3: Production deployment — load v2, score real traffic + attacks
    # ===================================================================
    print("\nPHASE 3: Production deployment — load v2, score reco traffic + attacks")

    prod_library = EngramLibrary.load(v2_path)
    print(f"  Loaded: {prod_library}")

    reco_engram = prod_library.get("api_v2_recommendations")
    reco_threshold = reco_engram.subspace.threshold

    prod_vecs = encode_batch(encoder, make_reco_prod, 500, seed_base=30000)
    prod_residuals = np.array([reco_engram.residual(v) for v in prod_vecs])
    prod_accept_rate = float(np.mean(prod_residuals <= reco_threshold))

    print(f"\n  Production reco traffic (500 requests):")
    print(f"    Mean residual:  {np.mean(prod_residuals):.4f}")
    print(f"    Std residual:   {np.std(prod_residuals):.4f}")
    print(f"    Threshold:      {reco_threshold:.4f}")
    print(f"    Accept rate:    {prod_accept_rate:.1%}")

    attack_vecs = encode_batch(encoder, make_reco_attack, 200, seed_base=40000)
    attack_residuals = np.array([reco_engram.residual(v) for v in attack_vecs])
    attack_reject_rate = float(np.mean(attack_residuals > reco_threshold))

    print(f"\n  Attack traffic targeting reco endpoint (200 requests):")
    print(f"    Mean residual:  {np.mean(attack_residuals):.4f}")
    print(f"    Std residual:   {np.std(attack_residuals):.4f}")
    print(f"    Reject rate:    {attack_reject_rate:.1%}")

    # ===================================================================
    # PHASE 4: Regression check — v1 traffic still matches in v2 library
    # ===================================================================
    print("\nPHASE 4: Regression check — v1 traffic against v2 library")

    regression_results = {}
    for name, gen_fn, seed_base in [
        ("api_users", make_users_traffic, 50000),
        ("api_search", make_search_traffic, 60000),
    ]:
        vecs = encode_batch(encoder, gen_fn, 200, seed_base=seed_base)
        engram = prod_library.get(name)
        thresh = engram.subspace.threshold
        residuals = np.array([engram.residual(v) for v in vecs])
        accept_rate = float(np.mean(residuals <= thresh))
        regression_results[name] = accept_rate
        print(f"  {name}: accept_rate={accept_rate:.1%}, "
              f"mean_residual={np.mean(residuals):.4f}, threshold={thresh:.4f}")

    min_regression_accept = min(regression_results.values())

    # ===================================================================
    # PHASE 5: Engram diff — v1 vs v2
    # ===================================================================
    print("\nPHASE 5: Engram diff (v1 vs v2)")

    v1_names = set(v1_library.names())
    v2_names = set(prod_library.names())
    new_engrams = v2_names - v1_names
    removed_engrams = v1_names - v2_names
    shared_engrams = v1_names & v2_names

    print(f"  v1 engrams: {sorted(v1_names)}")
    print(f"  v2 engrams: {sorted(v2_names)}")
    print(f"  New:         {sorted(new_engrams)}")
    print(f"  Removed:     {sorted(removed_engrams)}")
    print(f"  Shared:      {sorted(shared_engrams)}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  Preprod library miss rate:      {miss_rate:.0%}")
    print(f"  Production reco accept rate:    {prod_accept_rate:.1%}")
    print(f"  Attack reject rate:             {attack_reject_rate:.1%}")
    print(f"  v1 regression (min accept):     {min_regression_accept:.1%}")
    print(f"  New engrams in v2:              {len(new_engrams)}")

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Preprod detects new patterns (miss rate > 80%)",
            miss_rate > 0.80,
            f"miss_rate={miss_rate:.0%}",
        ),
        (
            "Production coverage > 80% (preprod engram generalizes)",
            prod_accept_rate > 0.80,
            f"accept_rate={prod_accept_rate:.1%}",
        ),
        (
            "Attack rejection > 90% on new endpoint",
            attack_reject_rate > 0.90,
            f"reject_rate={attack_reject_rate:.1%}",
        ),
        (
            "No regression: v1 traffic still accepted in v2 library",
            min_regression_accept > 0.90,
            f"min_accept={min_regression_accept:.1%}",
        ),
        (
            "Engram diff: exactly 1 new engram in v2 vs v1",
            len(new_engrams) == 1 and len(removed_engrams) == 0,
            f"new={len(new_engrams)}, removed={len(removed_engrams)}",
        ),
        (
            "End-to-end: library miss → mint → deploy → coverage",
            miss_rate > 0.80 and prod_accept_rate > 0.80 and attack_reject_rate > 0.90,
            f"miss={miss_rate:.0%}, coverage={prod_accept_rate:.1%}, "
            f"rejection={attack_reject_rate:.1%}",
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
