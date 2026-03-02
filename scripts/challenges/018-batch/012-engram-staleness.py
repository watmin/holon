#!/usr/bin/env python3
"""
Engram Staleness Detection

HYPOTHESIS:
===========
An engram trained on "epoch 0" traffic degrades as the real world drifts:
browsers update, new API endpoints appear, TLS stacks evolve. The staleness
metric — mean_residual * (1 - spectrum_similarity) — rises gradually under
operational drift but spikes suddenly under attack traffic. This asymmetry
lets us distinguish "time to retrain" from "under attack" using a single
scalar that combines manifold fit (residual) with structural shape change
(eigenvalue spectrum drift).

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable  - Rich structured HTTP/TLS encoding
2. OnlineSubspace           - Manifold learning (epoch 0 baseline)
3. EngramLibrary            - Engram storage and spectrum matching
4. Engram.residual          - Per-vector anomaly score
5. match_spectrum           - Eigenvalue cosine similarity

SCENARIO:
=========
Train one engram on epoch-0 browser traffic (500 requests). Evolve through
5 epochs, each shifting one aspect of the distribution: new paths, updated
user agents, changed TLS ciphers, new content types, then all combined.
At each epoch, score 200 requests against the original engram. Track
mean residual, spectrum similarity, and a composite staleness metric.
Compare drift staleness against sudden attack staleness.
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


def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_epoch0(rng):
    """Epoch 0 baseline — the traffic the engram was trained on."""
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


def _apply_new_paths(req, rng):
    """Drift: new API paths appear alongside existing ones."""
    path = str(rng.choice(["/", "/api/users", "/api/search", "/products", "/about", "/health",
                            "/api/v2/graphql", "/api/v2/stream"]))
    req["path"] = path
    req["path_parts"] = path.split("/")


def _apply_ua_shift(req, rng):
    """Drift: browser versions advance."""
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    ]))
    for i, pair in enumerate(req["headers"]):
        if pair[0] == "User-Agent":
            req["headers"][i] = ["User-Agent", ua]
            break


def _apply_tls_cipher(req, _rng):
    """Drift: new cipher suite added to TLS stack."""
    req["tls"]["ciphers"] = {"TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                              "TLS_CHACHA20_POLY1305_SHA256", "TLS_AES_128_CCM_SHA256"}
    req["tls"]["cipher_order"] = ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384",
                                   "TLS_CHACHA20_POLY1305_SHA256", "TLS_AES_128_CCM_SHA256"]


def _apply_accept_type(req, rng):
    """Drift: new content types in Accept header."""
    accept = str(rng.choice([
        "text/html,application/xhtml+xml,*/*;q=0.8",
        "text/html,application/xhtml+xml,application/graphql,*/*;q=0.8",
        "application/graphql,text/html,*/*;q=0.8",
    ]))
    for i, pair in enumerate(req["headers"]):
        if pair[0] == "Accept":
            req["headers"][i] = ["Accept", accept]
            break


EPOCH_MUTATIONS = {
    1: [_apply_new_paths],
    2: [_apply_new_paths, _apply_ua_shift],
    3: [_apply_new_paths, _apply_ua_shift, _apply_tls_cipher],
    4: [_apply_new_paths, _apply_ua_shift, _apply_tls_cipher, _apply_accept_type],
    5: [_apply_new_paths, _apply_ua_shift, _apply_tls_cipher, _apply_accept_type],
}


def make_epoch(epoch):
    """Return a generator for the given epoch (cumulative drift)."""
    def gen(rng):
        req = make_epoch0(rng)
        for mutation in EPOCH_MUTATIONS.get(epoch, []):
            mutation(req, rng)
        return req
    return gen


def make_attack(rng):
    """GET flood attack — structurally alien to the trained engram."""
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


EPOCH_GENERATORS = {e: make_epoch(e) if e > 0 else make_epoch0 for e in range(6)}

EPOCH_DESCRIPTIONS = {
    0: "baseline (training distribution)",
    1: "+ new paths (/api/v2/graphql, /api/v2/stream)",
    2: "+ user-agent shift (Chrome/122, Firefox/123)",
    3: "+ TLS cipher (TLS_AES_128_CCM_SHA256)",
    4: "+ Accept types (application/graphql)",
    5: "all shifts combined (epochs 1-4)",
}


def encode_batch(encoder, gen_fn, n, seed_base):
    vecs = []
    for i in range(n):
        vec = encoder.encode_walkable(gen_fn(np.random.default_rng(seed_base + i)))
        vecs.append(vec)
    return vecs


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 012: Engram Staleness Detection")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Train engram on epoch 0
    # ===================================================================
    print("\nPHASE 1: Train baseline engram on epoch-0 traffic (500 requests)")

    sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(500):
        vec = encoder.encode_walkable(make_epoch0(np.random.default_rng(i)))
        sub.update(vec)

    library = EngramLibrary(dim=DIM)
    library.add("epoch0_normal", sub, epoch=0)
    engram = library.get("epoch0_normal")
    baseline_eigenvalues = sub.eigenvalues.copy()

    print(f"  threshold={sub.threshold:.4f}, n={sub.n}")
    print(f"  eigenvalue energy: {np.sum(baseline_eigenvalues**2):.4f}")

    # ===================================================================
    # PHASE 2: Score each epoch against original engram
    # ===================================================================
    print("\nPHASE 2: Score 200 requests per epoch against baseline engram")

    epoch_results = {}

    print(f"\n  {'Epoch':<8} {'Description':<45} {'Mean res':>10} "
          f"{'Frac>thr':>10} {'Spec sim':>10} {'Staleness':>10}")
    print("  " + "-" * 98)

    for epoch in range(6):
        gen_fn = EPOCH_GENERATORS[epoch]
        vecs = encode_batch(encoder, gen_fn, 200, seed_base=20000 + epoch * 1000)
        residuals = np.array([engram.residual(v) for v in vecs])

        epoch_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for v in vecs:
            epoch_sub.update(v)
        epoch_eigenvalues = epoch_sub.eigenvalues

        spectrum_sim = cosine_similarity(baseline_eigenvalues, epoch_eigenvalues)
        mean_res = float(np.mean(residuals))
        frac_above = float(np.mean(residuals > sub.threshold))
        staleness = mean_res * (1.0 - spectrum_sim)

        epoch_results[epoch] = {
            "mean_res": mean_res,
            "frac_above": frac_above,
            "spectrum_sim": spectrum_sim,
            "staleness": staleness,
            "residuals": residuals,
        }

        desc = EPOCH_DESCRIPTIONS[epoch]
        print(f"  {epoch:<8} {desc:<45} {mean_res:>10.4f} "
              f"{frac_above:>9.1%} {spectrum_sim:>10.4f} {staleness:>10.4f}")

    # ===================================================================
    # PHASE 3: Attack comparison
    # ===================================================================
    print("\nPHASE 3: Attack traffic (200 GET flood requests)")

    attack_vecs = encode_batch(encoder, make_attack, 200, seed_base=80000)
    attack_residuals = np.array([engram.residual(v) for v in attack_vecs])

    attack_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for v in attack_vecs:
        attack_sub.update(v)
    attack_eigenvalues = attack_sub.eigenvalues

    attack_spectrum_sim = cosine_similarity(baseline_eigenvalues, attack_eigenvalues)
    attack_mean_res = float(np.mean(attack_residuals))
    attack_frac_above = float(np.mean(attack_residuals > sub.threshold))
    attack_staleness = attack_mean_res * (1.0 - attack_spectrum_sim)

    print(f"  Mean residual:       {attack_mean_res:.4f}")
    print(f"  Fraction > thresh:   {attack_frac_above:.1%}")
    print(f"  Spectrum similarity: {attack_spectrum_sim:.4f}")
    print(f"  Staleness:           {attack_staleness:.4f}")

    # ===================================================================
    # PHASE 4: Drift vs attack analysis
    # ===================================================================
    print("\nPHASE 4: Drift vs attack analysis")

    staleness_values = [epoch_results[e]["staleness"] for e in range(6)]
    drift_deltas = [staleness_values[i+1] - staleness_values[i] for i in range(5)]
    max_drift_delta = max(abs(d) for d in drift_deltas)
    attack_delta = attack_staleness - staleness_values[0]

    print(f"\n  Staleness trajectory (epochs 0-5):")
    for e in range(6):
        bar = "#" * int(staleness_values[e] * 100)
        print(f"    Epoch {e}: {staleness_values[e]:.4f}  {bar}")

    print(f"\n  Attack staleness:    {attack_staleness:.4f}")
    print(f"  Max epoch-to-epoch Δ: {max_drift_delta:.4f}")
    print(f"  Attack Δ from base:   {attack_delta:.4f}")

    if max_drift_delta > 1e-10:
        spike_ratio = attack_delta / max_drift_delta
    else:
        spike_ratio = float("inf")
    print(f"  Attack spike ratio:   {spike_ratio:.1f}x vs max drift step")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  Epoch 0 staleness (baseline):  {staleness_values[0]:.4f}")
    print(f"  Epoch 5 staleness (all drift): {staleness_values[5]:.4f}")
    print(f"  Attack staleness:              {attack_staleness:.4f}")
    print(f"  Drift ratio (epoch5/epoch0):   ", end="")
    if staleness_values[0] > 1e-10:
        print(f"{staleness_values[5] / staleness_values[0]:.2f}x")
    else:
        print(f"inf (epoch0 ≈ 0)")
    print(f"  Attack spike ratio:            {spike_ratio:.1f}x")

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    mean_residuals = [epoch_results[e]["mean_res"] for e in range(6)]
    monotonic_rises = sum(1 for i in range(5) if mean_residuals[i+1] >= mean_residuals[i] - 1e-6)
    approx_monotonic = monotonic_rises >= 3

    spectrum_sims = [epoch_results[e]["spectrum_sim"] for e in range(6)]
    spectrum_decreases = sum(1 for i in range(5) if spectrum_sims[i+1] <= spectrum_sims[i] + 1e-6)
    approx_decreasing = spectrum_decreases >= 3

    threshold_crossed = any(
        staleness_values[e] > staleness_values[0] * 2.0
        for e in [3, 4, 5]
    )

    drift_staleness_epoch5 = staleness_values[5]
    drift_lower_than_attack = drift_staleness_epoch5 < attack_staleness

    attack_spike = spike_ratio > 3.0

    checks = [
        (
            "Mean residual approximately monotonic across epochs (≥3/5 rises)",
            approx_monotonic,
            f"rises={monotonic_rises}/5",
        ),
        (
            "Spectrum similarity approximately decreasing across epochs (≥3/5)",
            approx_decreasing,
            f"decreases={spectrum_decreases}/5",
        ),
        (
            "Staleness crosses detectable threshold by epoch 3-4 (>2x baseline)",
            threshold_crossed,
            f"e3={staleness_values[3]:.4f}, e4={staleness_values[4]:.4f}, "
            f"baseline={staleness_values[0]:.4f}",
        ),
        (
            "Drift staleness (epoch 5) < attack staleness",
            drift_lower_than_attack,
            f"drift={drift_staleness_epoch5:.4f}, attack={attack_staleness:.4f}",
        ),
        (
            "Attack spike ratio > 3x vs max epoch-to-epoch drift",
            attack_spike,
            f"ratio={spike_ratio:.1f}x",
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
