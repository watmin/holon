#!/usr/bin/env python3
"""
Engram Taxonomy from Spectrum Clustering

HYPOTHESIS:
===========
Given a library of both normal and attack engrams, clustering by DUAL
SIGNAL similarity (spectrum × alignment) produces a natural taxonomy.
Normal traffic types cluster together, attack types cluster together,
and subtypes within each cluster show higher affinity than cross-cluster
pairs. The dual signal captures both variance shape (spectrum) and
variance direction (alignment), so structurally similar traffic — even
across distinct behavioral categories — groups naturally.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable          - Rich structured HTTP request encoding
2. EngramLibrary.match_spectrum     - Eigenvalue shape similarity (magnitude)
3. OnlineSubspace.subspace_alignment - Principal angle alignment (direction)
4. Combined dual signal              - spectrum × alignment for taxonomy
5. Pairwise similarity matrix        - Full 8×8 engram comparison

SCENARIO:
=========
WAF context. Build 8 engrams from 3 normal traffic types (browser_web,
api_client, mobile_app) and 5 attack types (get_flood, cred_stuffing,
scraper, tls_shuffle, post_flood). Compute pairwise dual-signal
similarity and verify that a natural normal/attack taxonomy emerges.
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

NORMAL_NAMES = ["browser_web", "api_client", "mobile_app"]
ATTACK_NAMES = ["get_flood", "cred_stuffing", "scraper", "tls_shuffle",
                "post_flood"]


def make_browser_web(rng):
    ua = str(rng.choice([
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]))
    path = str(rng.choice(["/", "/about", "/products", "/contact", "/blog"]))
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
    path = str(rng.choice(["/api/v2/users", "/api/v2/search", "/api/v2/orders",
                            "/api/v2/products", "/api/v2/recommendations"]))
    method = str(rng.choice(["GET", "GET", "POST", "PUT", "DELETE"]))
    return {
        "method": method, "path": path, "path_parts": path.split("/"), "version": "HTTP/2",
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
    path = str(rng.choice(["/api/mobile/feed", "/api/mobile/profile",
                            "/api/mobile/notifications", "/api/mobile/sync"]))
    return {
        "method": str(rng.choice(["GET", "POST"])),
        "path": path, "path_parts": path.split("/"), "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Authorization",
                         "X-App-Version", "Accept-Encoding"],
        "headers": [
            ["Host", "m.example.com"],
            ["User-Agent", str(rng.choice(["ExampleApp/4.2.0 (iPhone; iOS 17.2)",
                                            "ExampleApp/4.2.0 (Android 14; Pixel 8)"]))],
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
        "headers": [["Host", "example.com"], ["User-Agent", ua], ["Accept", "*/*"]],
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
        "path_parts": ["", "api", "v1", "auth", "login"], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Accept", "Content-Type",
                         "Content-Length", "Accept-Encoding", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", "python-requests/2.31.0"],
            ["Accept", "*/*"], ["Content-Type", "application/json"],
            ["Content-Length", str(body_len)],
            ["Accept-Encoding", "gzip, deflate"], ["Connection", "keep-alive"],
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
            ["Accept", "text/html"], ["Accept-Encoding", "gzip, deflate"],
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
        "headers": [["Host", "example.com"], ["User-Agent", ua], ["Accept", "*/*"]],
        "header_count": LinearScale(3), "has_cookie": "false",
        "tls": {
            "version": str(rng.choice(["TLS1.2", "TLS1.3"])),
            "ciphers": set(ciphers), "cipher_order": ciphers,
            "ext_types": set(exts), "groups": {"x25519", "secp256r1"},
            "alpn": ["h2", "http/1.1"],
        },
    }


def make_post_flood(rng):
    body_len = rng.integers(100, 500)
    return {
        "method": "POST", "path": "/api/submit",
        "path_parts": ["", "api", "submit"], "version": "HTTP/1.1",
        "header_order": ["Host", "User-Agent", "Content-Type", "Content-Length", "Accept"],
        "headers": [
            ["Host", "example.com"],
            ["User-Agent", str(rng.choice(["curl/8.0.1", "Go-http-client/1.1"]))],
            ["Content-Type", "application/x-www-form-urlencoded"],
            ["Content-Length", str(body_len)],
            ["Accept", "*/*"],
        ],
        "header_count": LinearScale(5), "has_cookie": "false",
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


ENGRAM_GENERATORS = {
    "browser_web": make_browser_web,
    "api_client": make_api_client,
    "mobile_app": make_mobile_app,
    "get_flood": make_get_flood,
    "cred_stuffing": make_cred_stuff,
    "scraper": make_scraper,
    "tls_shuffle": make_tls_shuffle,
    "post_flood": make_post_flood,
}


def build_library(encoder, seed_offset=0):
    library = EngramLibrary(dim=DIM)
    subspaces = {}
    for name, gen_fn in ENGRAM_GENERATORS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(300):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i + seed_offset + hash(name) % 10000)))
            sub.update(vec)
        category = "normal" if name in NORMAL_NAMES else "attack"
        library.add(name, sub, category=category)
        subspaces[name] = sub
    return library, subspaces


def compute_pairwise_similarity(library, subspaces, names):
    """Compute 8×8 dual-signal similarity matrix: spectrum × alignment."""
    n = len(names)
    spectrum_matrix = np.zeros((n, n))
    alignment_matrix = np.zeros((n, n))
    combined_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        engram_i = library.get(name_i)
        sub_i = subspaces[name_i]

        spectrum_all = library.match_spectrum(sub_i.eigenvalues, top_k=n)
        spectrum_dict = {nm: sc for nm, sc in spectrum_all}

        for j, name_j in enumerate(names):
            spectrum_matrix[i, j] = spectrum_dict.get(name_j, 0.0)
            alignment_matrix[i, j] = sub_i.subspace_alignment(subspaces[name_j])
            combined_matrix[i, j] = spectrum_matrix[i, j] * alignment_matrix[i, j]

    return spectrum_matrix, alignment_matrix, combined_matrix


def find_clusters(combined_matrix, names):
    """Single-linkage agglomerative clustering on dual-signal similarity."""
    n = len(names)
    clusters = [[i] for i in range(n)]

    while len(clusters) > 1:
        best_sim = -1.0
        best_pair = (0, 1)
        for ci in range(len(clusters)):
            for cj in range(ci + 1, len(clusters)):
                sims = []
                for i in clusters[ci]:
                    for j in clusters[cj]:
                        sims.append(combined_matrix[i, j])
                avg_sim = np.mean(sims)
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_pair = (ci, cj)

        if best_sim < 0.05:
            break

        ci, cj = best_pair
        merged = clusters[ci] + clusters[cj]
        new_clusters = [c for idx, c in enumerate(clusters)
                        if idx != ci and idx != cj]
        new_clusters.append(merged)
        clusters = new_clusters

    return clusters


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 010: Engram Taxonomy from Spectrum Clustering")
    print("=" * 70)

    all_names = NORMAL_NAMES + ATTACK_NAMES

    # ===================================================================
    # PHASE 1: Build 8 engrams (300 requests each)
    # ===================================================================
    print("\nPHASE 1: Build engram library (3 normal + 5 attack = 8 engrams)")

    library, subspaces = build_library(encoder, seed_offset=0)

    for name in all_names:
        category = "normal" if name in NORMAL_NAMES else "attack"
        sub = subspaces[name]
        eig_energy = np.sum(sub.eigenvalues**2)
        print(f"  Added: {name:<15} ({category}, n={sub.n}, "
              f"eig_energy={eig_energy:.1f})")
    print(f"  Library: {library}")

    # ===================================================================
    # PHASE 2: Pairwise similarity matrix (dual signal)
    # ===================================================================
    print("\nPHASE 2: Pairwise similarity (spectrum × alignment)")

    spectrum_matrix, alignment_matrix, combined_matrix = \
        compute_pairwise_similarity(library, subspaces, all_names)

    max_label = max(len(n) for n in all_names)
    header = " " * (max_label + 2) + "  ".join(f"{n[:6]:>6}" for n in all_names)
    print(f"\n  Combined (spectrum × alignment):")
    print(f"  {header}")
    for i, name in enumerate(all_names):
        row = "  ".join(f"{combined_matrix[i, j]:>6.3f}" for j in range(len(all_names)))
        print(f"  {name:<{max_label}}  {row}")

    # ===================================================================
    # PHASE 3: Cluster analysis
    # ===================================================================
    print("\nPHASE 3: Cluster analysis")

    normal_indices = [all_names.index(n) for n in NORMAL_NAMES]
    attack_indices = [all_names.index(n) for n in ATTACK_NAMES]

    normal_intra = []
    for i in normal_indices:
        for j in normal_indices:
            if i != j:
                normal_intra.append(combined_matrix[i, j])
    normal_intra_mean = np.mean(normal_intra)

    attack_intra = []
    for i in attack_indices:
        for j in attack_indices:
            if i != j:
                attack_intra.append(combined_matrix[i, j])
    attack_intra_mean = np.mean(attack_intra)

    inter_cluster = []
    for i in normal_indices:
        for j in attack_indices:
            inter_cluster.append(combined_matrix[i, j])
            inter_cluster.append(combined_matrix[j, i])
    inter_cluster_mean = np.mean(inter_cluster)

    print(f"\n  Normal intra-cluster mean:       {normal_intra_mean:.4f}")
    print(f"  Attack intra-cluster mean:       {attack_intra_mean:.4f}")
    print(f"  Normal↔Attack inter-cluster mean: {inter_cluster_mean:.4f}")

    clusters = find_clusters(combined_matrix, all_names)
    print(f"\n  Discovered clusters ({len(clusters)}):")
    for ci, cluster in enumerate(clusters):
        members = [all_names[i] for i in cluster]
        categories = ["N" if m in NORMAL_NAMES else "A" for m in members]
        label = "/".join(categories)
        print(f"    Cluster {ci+1}: [{label}] {', '.join(members)}")

    normal_most_similar = {}
    for i in normal_indices:
        sims = [(all_names[j], combined_matrix[i, j])
                for j in range(len(all_names)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        normal_most_similar[all_names[i]] = sims

    print(f"\n  Normal engrams — nearest neighbors:")
    for name in NORMAL_NAMES:
        top3 = normal_most_similar[name][:3]
        top3_str = ", ".join(f"{n}={s:.3f}" for n, s in top3)
        print(f"    {name:<15} → {top3_str}")

    normals_cluster_together = True
    for i in normal_indices:
        mean_to_normals = np.mean([combined_matrix[i, j]
                                   for j in normal_indices if j != i])
        mean_to_attacks = np.mean([combined_matrix[i, j]
                                   for j in attack_indices])
        if mean_to_normals <= mean_to_attacks:
            normals_cluster_together = False
            break

    # ===================================================================
    # PHASE 4: Stability check — independent rebuild
    # ===================================================================
    print("\nPHASE 4: Stability check (independent rebuild)")

    library2, subspaces2 = build_library(encoder, seed_offset=500000)
    _, _, combined_matrix2 = compute_pairwise_similarity(
        library2, subspaces2, all_names)

    normal_intra2 = []
    for i in normal_indices:
        for j in normal_indices:
            if i != j:
                normal_intra2.append(combined_matrix2[i, j])

    attack_intra2 = []
    for i in attack_indices:
        for j in attack_indices:
            if i != j:
                attack_intra2.append(combined_matrix2[i, j])

    inter2 = []
    for i in normal_indices:
        for j in attack_indices:
            inter2.append(combined_matrix2[i, j])
            inter2.append(combined_matrix2[j, i])

    normal_intra_mean2 = np.mean(normal_intra2)
    attack_intra_mean2 = np.mean(attack_intra2)
    inter_mean2 = np.mean(inter2)

    print(f"  Build 2 — normal intra: {normal_intra_mean2:.4f}, "
          f"attack intra: {attack_intra_mean2:.4f}, "
          f"inter: {inter_mean2:.4f}")

    normals_cluster2 = True
    for i in normal_indices:
        mean_to_normals = np.mean([combined_matrix2[i, j]
                                   for j in normal_indices if j != i])
        mean_to_attacks = np.mean([combined_matrix2[i, j]
                                   for j in attack_indices])
        if mean_to_normals <= mean_to_attacks:
            normals_cluster2 = False
            break

    stable = (inter_mean2 < 0.3 and normal_intra_mean2 > 0.3
              and attack_intra_mean2 > 0.1)

    print(f"  Stable clustering: {stable}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Engram Taxonomy")
    print("=" * 70)

    print(f"\n  {'Metric':<40} {'Build 1':>10} {'Build 2':>10}")
    print("  " + "-" * 62)
    print(f"  {'Normal intra-cluster mean':<40} "
          f"{normal_intra_mean:>10.4f} {normal_intra_mean2:>10.4f}")
    print(f"  {'Attack intra-cluster mean':<40} "
          f"{attack_intra_mean:>10.4f} {attack_intra_mean2:>10.4f}")
    print(f"  {'Normal↔Attack inter-cluster mean':<40} "
          f"{inter_cluster_mean:>10.4f} {inter_mean2:>10.4f}")
    print(f"  {'Normals form group':<40} "
          f"{'yes' if normals_cluster_together else 'no':>10} "
          f"{'yes' if normals_cluster2 else 'no':>10}")

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Normal vs attack inter-cluster mean < 0.3",
            inter_cluster_mean < 0.3,
            f"inter={inter_cluster_mean:.4f}",
        ),
        (
            "Normal intra-cluster mean > 0.3",
            normal_intra_mean > 0.3,
            f"normal_intra={normal_intra_mean:.4f}",
        ),
        (
            "Attack intra-cluster mean > 0.1",
            attack_intra_mean > 0.1,
            f"attack_intra={attack_intra_mean:.4f}",
        ),
        (
            "Normal engrams form recognizable group (all 3 most similar)",
            normals_cluster_together,
            f"cluster={'yes' if normals_cluster_together else 'no'}",
        ),
        (
            "Clustering stable across 2 independent builds",
            stable,
            f"build2: inter={inter_mean2:.4f}, "
            f"normal_intra={normal_intra_mean2:.4f}, "
            f"attack_intra={attack_intra_mean2:.4f}",
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
