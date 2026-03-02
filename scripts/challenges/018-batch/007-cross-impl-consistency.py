#!/usr/bin/env python3
"""
Cross-Implementation Eigenvalue Consistency

HYPOTHESIS:
===========
Eigenvalue spectra from OnlineSubspace (CCIPCA) are reproducible across
independent training runs on the same data. Since holon-rs is not available
in this environment, we simulate the cross-implementation scenario by
comparing spectra from:
  - Two runs on identical data in identical order (deterministic baseline)
  - Two runs on identical data in different order (order sensitivity)
  - Two runs with different k (shared-component stability)

This establishes the consistency bounds needed for cross-implementation
matching: if two independent Python runs disagree by X, a Rust port must
also stay within that bound to be considered correct.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable     - Rich structured HTTP request encoding
2. OnlineSubspace.eigenvalues  - Eigenvalue spectrum extraction
3. Cosine similarity of spectra - Cross-run consistency metric

SCENARIO:
=========
WAF context. Generate 500 normal HTTP requests with a fixed seed.
Train four subspaces under different conditions and compare their
eigenvalue spectra pairwise.
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
    print("EXPERIMENT 007: Cross-Implementation Eigenvalue Consistency")
    print("=" * 70)

    n_samples = 500

    # ===================================================================
    # PHASE 1: Generate vectors once (fixed seed)
    # ===================================================================
    print(f"\nPHASE 1: Generate {n_samples} encoded vectors (fixed seed)")

    vectors = []
    for i in range(n_samples):
        vec = encoder.encode_walkable(make_normal(np.random.default_rng(i)))
        vectors.append(vec)

    print(f"  Encoded {len(vectors)} vectors, dim={DIM}")

    # ===================================================================
    # PHASE 2: Train four subspaces under different conditions
    # ===================================================================
    print("\nPHASE 2: Train subspaces")

    print("  A: k=64, original order")
    sub_a = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for vec in vectors:
        sub_a.update(vec)

    print("  B: k=64, same order (independent run)")
    sub_b = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for vec in vectors:
        sub_b.update(vec)

    shuffled_indices = np.random.default_rng(42).permutation(n_samples)
    print("  C: k=64, shuffled order (seed=42)")
    sub_c = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for idx in shuffled_indices:
        sub_c.update(vectors[idx])

    print("  D: k=32, original order")
    sub_d = OnlineSubspace(dim=DIM, k=32, amnesia=2.0)
    for vec in vectors:
        sub_d.update(vec)

    # ===================================================================
    # PHASE 3: Compare eigenvalue spectra
    # ===================================================================
    print("\nPHASE 3: Pairwise eigenvalue spectrum comparison")

    eig_a = sub_a.eigenvalues
    eig_b = sub_b.eigenvalues
    eig_c = sub_c.eigenvalues
    eig_d = sub_d.eigenvalues

    sim_ab = cosine_similarity(eig_a, eig_b)
    sim_ac = cosine_similarity(eig_a, eig_c)
    sim_bc = cosine_similarity(eig_b, eig_c)
    sim_ad = cosine_similarity(eig_a[:32], eig_d)
    sim_cd = cosine_similarity(eig_c[:32], eig_d)

    print(f"\n  {'Pair':<35} {'Cosine Sim':>12}")
    print("  " + "-" * 50)
    print(f"  {'A vs B (same order, same k)' :<35} {sim_ab:>12.6f}")
    print(f"  {'A vs C (shuffled order, same k)':<35} {sim_ac:>12.6f}")
    print(f"  {'B vs C (shuffled order, same k)':<35} {sim_bc:>12.6f}")
    print(f"  {'A vs D (same order, k=64 vs 32)':<35} {sim_ad:>12.6f}")
    print(f"  {'C vs D (shuffled vs k=32)':<35} {sim_cd:>12.6f}")

    # ===================================================================
    # PHASE 4: Eigenvalue magnitude profiles
    # ===================================================================
    print("\nPHASE 4: Top eigenvalue magnitudes (first 10)")
    print(f"  {'Idx':>5}  {'A (k=64)':>12}  {'B (k=64)':>12}  "
          f"{'C (shuf)':>12}  {'D (k=32)':>12}")
    print("  " + "-" * 60)
    for i in range(min(10, len(eig_a))):
        d_val = f"{eig_d[i]:12.4f}" if i < len(eig_d) else f"{'—':>12}"
        print(f"  {i:>5}  {eig_a[i]:12.4f}  {eig_b[i]:12.4f}  "
              f"{eig_c[i]:12.4f}  {d_val}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Consistency Bounds")
    print("=" * 70)

    print(f"\n  Same data, same order:     cosine = {sim_ab:.6f}")
    print(f"  Same data, shuffled order: cosine = {sim_ac:.6f}")
    print(f"  Different k (first 32):    cosine = {sim_ad:.6f}")
    print(f"  Order sensitivity gap:     {sim_ab - sim_ac:.6f}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Same order, same data: cosine > 0.99 (deterministic)",
            sim_ab > 0.99,
            f"cosine={sim_ab:.6f}",
        ),
        (
            "Shuffled order: cosine > 0.8 (CCIPCA converges)",
            sim_ac > 0.8,
            f"cosine={sim_ac:.6f}",
        ),
        (
            "Different k (first 32 eigenvalues): cosine > 0.9",
            sim_ad > 0.9,
            f"cosine={sim_ad:.6f}",
        ),
        (
            "Shuffled similarity < same-order similarity (order matters)",
            sim_ac < sim_ab,
            f"shuffled={sim_ac:.6f} vs same={sim_ab:.6f}",
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
