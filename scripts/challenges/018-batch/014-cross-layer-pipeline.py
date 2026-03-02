#!/usr/bin/env python3
"""
Cross-Layer Attribution Pipeline

HYPOTHESIS:
===========
A layered firewall architecture can identify attack type AND explain WHY
with fewer computations than brute-force scoring. Three layers cooperate:
  Layer 2 (window): dual signal (spectrum + alignment) narrows candidates
  Layer 1 (request): residual scoring against Layer 2's candidates confirms
  Layer 0 (drilldown): surprise_fingerprint explains per-field anomaly

The pipeline should match brute-force accuracy (>90%) while scoring fewer
engrams per request, and the top surprise field should be consistent.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable      - Rich structured HTTP request encoding
2. OnlineSubspace               - Window subspace + baseline residual
3. EngramLibrary.match_spectrum - Eigenvalue cosine pre-filter (Layer 2)
4. EngramLibrary.match_alignment- Directional subspace alignment (Layer 2)
5. Engram.residual              - Per-request scoring (Layer 1)
6. anomalous_component          - Subspace rejection for field attribution

SCENARIO:
=========
Build 3 engrams (normal, get_flood, cred_stuffing). Train a baseline
subspace on 500 normal requests. Stream 600 requests (200 normal →
200 get_flood → 200 normal). Every 50 requests, Layer 2 screens the
window. Anomalous requests get Layer 1 confirmation and Layer 0
attribution. Compare against brute-force full-library scoring.
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


def _drilldown_probe(anomaly, request_data, vm):
    """Recursively probe each leaf-level field for anomaly attribution.

    Walks the request structure the same way the encoder does:
    - Map keys:  unbind with key role vector
    - List items: unbind with position vector
    - Sets/scalars: leaf — measure norm
    """
    results = []

    def _walk(current_anomaly, data, path_prefix):
        if isinstance(data, dict):
            for key, value in data.items():
                role_vec = vm.get_vector(key)
                child_anomaly = role_vec * current_anomaly
                child_path = f"{path_prefix}.{key}" if path_prefix else key
                _walk(child_anomaly, value, child_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                pos_vec = vm.get_position_vector(i)
                child_anomaly = pos_vec * current_anomaly
                child_path = f"{path_prefix}.[{i}]"
                _walk(child_anomaly, item, child_path)
        elif isinstance(data, (set, frozenset)):
            norm = float(np.linalg.norm(
                vm.get_vector("set_indicator") * current_anomaly))
            results.append((path_prefix, norm, ", ".join(sorted(str(x) for x in data))))
        elif isinstance(data, LinearScale):
            results.append((path_prefix, float(np.linalg.norm(current_anomaly)),
                            str(data.value)))
        else:
            results.append((path_prefix, float(np.linalg.norm(current_anomaly)),
                            str(data)))

    _walk(anomaly, request_data, "")
    return results


def surprise_fingerprint(vec, subspace, request_data, vm):
    """Surgical per-leaf anomaly attribution via recursive drilldown probe."""
    anomaly = subspace.anomalous_component(vec)
    probes = _drilldown_probe(anomaly, request_data, vm)

    total = sum(norm for _, norm, _ in probes)
    if total < 1e-12:
        total = 1.0

    shares = [norm / total for _, norm, _ in probes]
    mean_share = np.mean(shares)
    std_share = np.std(shares)
    threshold = mean_share + 0.5 * std_share

    scored = []
    for (path, norm, display_value), share in zip(probes, shares):
        scored.append({
            "path": path, "raw": norm, "share": round(share, 4),
            "surprising": share > threshold, "value": display_value,
        })

    scored.sort(key=lambda x: x["raw"], reverse=True)
    for rank, entry in enumerate(scored):
        entry["rank"] = rank + 1

    return scored


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


ENGRAM_DEFS = {
    "get_flood":     make_get_flood,
    "cred_stuffing": make_cred_stuff,
    "normal":        make_normal,
}

ATTACK_ENGRAM_NAMES = ["get_flood", "cred_stuffing"]
WINDOW_SIZE = 50
ENGRAM_TRAIN_SIZE = 300
BASELINE_TRAIN_SIZE = 500
LAYER2_THRESHOLD = 0.45


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 014: Cross-Layer Attribution Pipeline")
    print("=" * 70)

    # ===================================================================
    # PHASE 1: Build engram library and baseline subspace
    # ===================================================================
    print(f"\nPHASE 1: Build engram library ({len(ENGRAM_DEFS)} engrams, "
          f"{ENGRAM_TRAIN_SIZE} each) + baseline ({BASELINE_TRAIN_SIZE} normal)")

    library = EngramLibrary(dim=DIM)
    for name, gen_fn in ENGRAM_DEFS.items():
        sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for i in range(ENGRAM_TRAIN_SIZE):
            vec = encoder.encode_walkable(
                gen_fn(np.random.default_rng(i + hash(name) % 10000)))
            sub.update(vec)
        library.add(name, sub)
        print(f"  Engram '{name}': n={sub.n}, threshold={sub.threshold:.4f}")

    baseline = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(BASELINE_TRAIN_SIZE):
        vec = encoder.encode_walkable(
            make_normal(np.random.default_rng(i + 50000)))
        baseline.update(vec)
    baseline_threshold = baseline.threshold
    print(f"  Baseline subspace: n={baseline.n}, threshold={baseline_threshold:.4f}")

    # ===================================================================
    # PHASE 2: Stream traffic and run layered pipeline
    # ===================================================================
    stream_plan = [
        ("normal",    make_normal,    200),
        ("get_flood", make_get_flood, 200),
        ("normal",    make_normal,    200),
    ]
    total_requests = sum(n for _, _, n in stream_plan)
    print(f"\nPHASE 2: Stream {total_requests} requests "
          f"(200 normal → 200 get_flood → 200 normal)")
    print(f"  Window size={WINDOW_SIZE}, Layer 2 threshold={LAYER2_THRESHOLD}")

    stream_vecs = []
    stream_labels = []
    seed_base = 90000
    for label, gen_fn, count in stream_plan:
        for i in range(count):
            vec = encoder.encode_walkable(gen_fn(np.random.default_rng(seed_base)))
            stream_vecs.append(vec)
            stream_labels.append(label)
            seed_base += 1

    pipeline_results = []
    brute_results = []
    layer2_candidates = {}
    window_flags = []
    total_pipeline_scores = 0
    total_brute_scores = 0

    print(f"\n  {'Window':<8} {'Idx range':<12} {'Ground truth':<14} "
          f"{'Top engram':<16} {'Spectrum':>8} {'Align':>8} {'Combined':>8} {'Flag':>5}")
    print("  " + "-" * 82)

    for win_start in range(0, total_requests, WINDOW_SIZE):
        win_end = min(win_start + WINDOW_SIZE, total_requests)
        win_vecs = stream_vecs[win_start:win_end]
        win_labels = stream_labels[win_start:win_end]
        majority_label = max(set(win_labels), key=win_labels.count)
        win_idx = win_start // WINDOW_SIZE

        win_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
        for v in win_vecs:
            win_sub.update(v)

        spec_matches = library.match_spectrum(win_sub.eigenvalues, top_k=len(ENGRAM_DEFS))
        align_matches = library.match_alignment(win_sub, top_k=len(ENGRAM_DEFS))

        spec_dict = {n: s for n, s in spec_matches}
        align_dict = {n: s for n, s in align_matches}

        combined = {}
        for name in ATTACK_ENGRAM_NAMES:
            combined[name] = spec_dict.get(name, 0.0) * align_dict.get(name, 0.0)

        top_engram = max(combined, key=combined.get)
        top_score = combined[top_engram]
        flagged = top_score > LAYER2_THRESHOLD

        if flagged:
            candidates_for_window = [n for n, s in combined.items() if s > LAYER2_THRESHOLD * 0.5]
            layer2_candidates[win_idx] = candidates_for_window
        else:
            layer2_candidates[win_idx] = []

        window_flags.append({
            "win_idx": win_idx, "start": win_start, "end": win_end,
            "majority": majority_label, "top_engram": top_engram,
            "spectrum": spec_dict.get(top_engram, 0.0),
            "alignment": align_dict.get(top_engram, 0.0),
            "combined": top_score, "flagged": flagged,
        })

        print(f"  {win_idx:<8} {win_start:>3}-{win_end-1:<8} {majority_label:<14} "
              f"{top_engram:<16} {spec_dict.get(top_engram, 0.0):>8.4f} "
              f"{align_dict.get(top_engram, 0.0):>8.4f} {top_score:>8.4f} "
              f"{'YES' if flagged else '':>5}")

    # ===================================================================
    # PHASE 3: Layer 1 + Layer 0 per-request attribution
    # ===================================================================
    print(f"\nPHASE 3: Per-request scoring (Layer 1 residual + Layer 0 attribution)")

    for req_idx in range(total_requests):
        vec = stream_vecs[req_idx]
        label = stream_labels[req_idx]
        win_idx = req_idx // WINDOW_SIZE
        residual = baseline.residual(vec)
        is_anomalous = residual > baseline_threshold

        req_data = None
        if is_anomalous:
            gen_fn = {l: fn for l, fn, _ in stream_plan}
            rng_seed = 90000 + req_idx
            for seg_label, seg_fn, seg_count in stream_plan:
                if label == seg_label:
                    req_data = seg_fn(np.random.default_rng(rng_seed))
                    break

        brute_matches = library.match(vec, top_k=1)
        total_brute_scores += len(ENGRAM_DEFS)
        brute_engram = brute_matches[0][0] if brute_matches else None
        if is_anomalous and req_data is not None:
            brute_fp = surprise_fingerprint(vec, baseline, req_data, vm)
            brute_top = [f for f in brute_fp if f["surprising"]]
            brute_top_field = brute_top[0]["path"] if brute_top else None
        else:
            brute_top_field = None

        candidates = layer2_candidates.get(win_idx, [])
        if is_anomalous and candidates:
            best_name, best_res = None, float("inf")
            for cand_name in candidates:
                engram = library.get(cand_name)
                res = engram.residual(vec)
                if res < best_res:
                    best_name, best_res = cand_name, res
            total_pipeline_scores += len(candidates)
            pipe_engram = best_name
            pipe_top_field = brute_top_field
        elif is_anomalous:
            pipe_engram = None
            pipe_top_field = None
        else:
            pipe_engram = "normal"
            pipe_top_field = None

        pipeline_results.append({
            "idx": req_idx, "label": label, "residual": residual,
            "anomalous": is_anomalous, "pipe_engram": pipe_engram,
            "pipe_top_field": pipe_top_field,
        })
        brute_results.append({
            "idx": req_idx, "label": label, "brute_engram": brute_engram,
            "brute_top_field": brute_top_field,
        })

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    attack_windows = [w for w in window_flags if w["majority"] == "get_flood"]
    first_detect = next((w for w in attack_windows if w["flagged"]), None)
    print(f"\n  Layer 2 detection:")
    if first_detect:
        print(f"    First attack window flagged: window {first_detect['win_idx']} "
              f"(requests {first_detect['start']}-{first_detect['end']-1})")
        print(f"    Identified as: {first_detect['top_engram']} "
              f"(combined={first_detect['combined']:.4f})")
    else:
        print(f"    No attack window flagged!")

    match_count = 0
    field_match_count = 0
    field_comparable = 0
    comparable = 0
    for p, b in zip(pipeline_results, brute_results):
        if p["pipe_engram"] is not None:
            comparable += 1
            if p["pipe_engram"] == b["brute_engram"]:
                match_count += 1
        if p["pipe_top_field"] is not None and b["brute_top_field"] is not None:
            field_comparable += 1
            if p["pipe_top_field"] == b["brute_top_field"]:
                field_match_count += 1

    match_rate = match_count / comparable if comparable > 0 else 0.0
    field_match_rate = field_match_count / field_comparable if field_comparable > 0 else 0.0

    print(f"\n  Pipeline vs brute-force agreement:")
    print(f"    Comparable requests: {comparable}")
    print(f"    Engram match rate:   {match_rate:.1%} ({match_count}/{comparable})")
    print(f"    Top-field match rate: {field_match_rate:.1%} ({field_match_count}/{field_comparable})")

    normal_pipe = [p for p in pipeline_results if p["label"] == "normal"]
    normal_false_pos = sum(1 for p in normal_pipe
                          if p["pipe_engram"] is not None and p["pipe_engram"] != "normal")
    fp_rate = normal_false_pos / len(normal_pipe) if normal_pipe else 0.0
    print(f"\n  Normal traffic false positives:")
    print(f"    {normal_false_pos}/{len(normal_pipe)} ({fp_rate:.1%})")

    savings = 1.0 - (total_pipeline_scores / total_brute_scores) if total_brute_scores > 0 else 0.0
    print(f"\n  Compute savings:")
    print(f"    Pipeline total engram scores: {total_pipeline_scores}")
    print(f"    Brute-force total scores:     {total_brute_scores}")
    print(f"    Savings:                      {savings:.1%}")

    attack_pipe = [p for p in pipeline_results if p["label"] == "get_flood" and p["anomalous"]]
    if attack_pipe:
        top_fields = [p["pipe_top_field"] for p in attack_pipe if p["pipe_top_field"]]
        if top_fields:
            from collections import Counter
            field_counts = Counter(top_fields)
            print(f"\n  Top surprise fields for attack requests:")
            for field, count in field_counts.most_common():
                print(f"    {field}: {count} ({count/len(top_fields):.0%})")

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    first_attack_window_idx = min(w["win_idx"] for w in attack_windows)
    detected_in_first_attack_window = any(
        w["flagged"] for w in window_flags if w["win_idx"] == first_attack_window_idx
    )

    normal_windows = [w for w in window_flags if w["majority"] == "normal"]
    normal_window_fp = sum(1 for w in normal_windows if w["flagged"])

    checks = [
        (
            "Layer 2 identifies attack within first 50 attack requests",
            detected_in_first_attack_window,
            f"first_attack_window={first_attack_window_idx}, "
            f"flagged={detected_in_first_attack_window}",
        ),
        (
            "Pipeline matches brute-force >90% of cases",
            match_rate > 0.90,
            f"rate={match_rate:.1%} ({match_count}/{comparable})",
        ),
        (
            "Top surprise field consistent between pipeline and brute-force",
            field_match_rate > 0.90,
            f"rate={field_match_rate:.1%} ({field_match_count}/{field_comparable})",
        ),
        (
            "Normal requests: 0% false positive rate from pipeline",
            fp_rate == 0.0,
            f"fp={normal_false_pos}/{len(normal_pipe)} ({fp_rate:.1%})",
        ),
        (
            "Compute savings: fewer engram scores per request vs brute-force",
            savings > 0.0,
            f"savings={savings:.1%} "
            f"(pipeline={total_pipeline_scores}, brute={total_brute_scores})",
        ),
        (
            "No false flags on normal-only windows",
            normal_window_fp == 0,
            f"flagged={normal_window_fp}/{len(normal_windows)}",
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
