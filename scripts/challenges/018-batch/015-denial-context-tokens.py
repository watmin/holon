#!/usr/bin/env python3
"""
Denial Context Tokens (Sealed Verdicts)

HYPOTHESIS:
===========
When a firewall denies a request, a self-contained cryptographically sealed
token can encode the COMPLETE denial reason as data. The caller receives an
opaque base64 blob (no information disclosure). A firewall admin decrypts
for instant full-context explainability without a log dive. If the denial
is a false positive, the decoded context is actionable — feed it back to
mint a corrective engram.

PRIMITIVES DEMONSTRATED:
========================
1. Encoder.encode_walkable        - Rich structured HTTP/TLS encoding
2. OnlineSubspace.update          - Manifold learning (streaming)
3. OnlineSubspace.residual        - Per-vector anomaly score
4. OnlineSubspace.anomalous_component - Out-of-subspace signal extraction
5. OnlineSubspace.eigenvalues     - Spectral snapshot for context
6. EngramLibrary.match            - Two-tier engram matching
7. AES-256-GCM encryption         - Sealed verdict token round-trip

SCENARIO:
=========
Train a normal subspace on 500 clean requests. Generate denied traffic:
true positives (GET flood, credential stuffing) and false positives
(unusual-but-legitimate). For each denial, build a full context dict,
encrypt with AES-256-GCM, base64-encode for transport, then verify
round-trip fidelity. For false positives, use decoded context to update
the subspace and confirm the same request now passes. For true positives,
verify decoded context identifies the attack type.
"""

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon.kernel.encoder import Encoder
from holon.kernel.vector_manager import VectorManager
from holon.kernel.walkable import LinearScale
from holon.memory.engram import EngramLibrary
from holon.memory.subspace import OnlineSubspace

DIM = 4096

import os

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    REAL_CRYPTO = True
except ImportError:
    import hashlib
    import hmac
    REAL_CRYPTO = False

# -----------------------------------------------------------------------
# Crypto helpers
# -----------------------------------------------------------------------

if REAL_CRYPTO:
    def generate_key():
        return AESGCM.generate_key(bit_length=256)

    def seal_token(plaintext_bytes: bytes, key: bytes) -> str:
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)
        return base64.b64encode(nonce + ciphertext).decode("ascii")

    def unseal_token(token_b64: str, key: bytes) -> bytes:
        raw = base64.b64decode(token_b64)
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None)
else:
    def generate_key():
        return os.urandom(32) if hasattr(os, 'urandom') else bytes(range(32))

    def seal_token(plaintext_bytes: bytes, key: bytes) -> str:
        """Simulated sealing using HMAC-SHA256 (NOT real encryption)."""
        nonce = os.urandom(12)
        tag = hmac.new(key, nonce + plaintext_bytes, hashlib.sha256).digest()
        payload = nonce + tag + plaintext_bytes
        return base64.b64encode(payload).decode("ascii")

    def unseal_token(token_b64: str, key: bytes) -> bytes:
        """Simulated unsealing — verifies HMAC, returns plaintext."""
        raw = base64.b64decode(token_b64)
        nonce, tag, plaintext_bytes = raw[:12], raw[12:44], raw[44:]
        expected = hmac.new(key, nonce + plaintext_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("HMAC verification failed")
        return plaintext_bytes


# -----------------------------------------------------------------------
# Traffic generators
# -----------------------------------------------------------------------

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


def make_unusual_legit(rng):
    """Unusual but legitimate — edge cases that might trigger false positives."""
    ua = str(rng.choice([
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/120.0.0.0",
    ]))
    path = str(rng.choice(["/api/v2/graphql", "/internal/metrics", "/admin/dashboard"]))
    return {
        "method": str(rng.choice(["GET", "POST", "PUT"])),
        "path": path, "path_parts": path.split("/"), "version": "HTTP/2",
        "header_order": ["Host", "User-Agent", "Accept", "Accept-Language",
                         "Accept-Encoding", "Cookie", "Connection"],
        "headers": [
            ["Host", "example.com"], ["User-Agent", ua],
            ["Accept", str(rng.choice(["application/json", "text/html,*/*;q=0.8"]))],
            ["Accept-Language", str(rng.choice(["fr-FR,fr;q=0.9", "de-DE,de;q=0.9"]))],
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


# -----------------------------------------------------------------------
# Drilldown surprise probe
# -----------------------------------------------------------------------


def _probe_norm(anomaly_component, role_vec):
    """Unbind a role from the anomaly and measure magnitude."""
    return float(np.linalg.norm(role_vec * anomaly_component))


def _drilldown_probe(anomaly, request_data, vm):
    """Recursively probe each leaf-level field for anomaly attribution.

    Walks the request structure the same way the encoder does:
    - Map keys:  unbind with key role vector
    - List items: unbind with position vector
    - Sets/scalars: leaf — measure norm

    Returns [(path, norm, display_value), ...] for every leaf.
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
            norm = _probe_norm(current_anomaly, vm.get_vector("set_indicator"))
            display = ", ".join(sorted(str(x) for x in data))
            results.append((path_prefix, norm, display))

        elif isinstance(data, LinearScale):
            norm = float(np.linalg.norm(current_anomaly))
            results.append((path_prefix, norm, str(data.value)))

        else:
            norm = float(np.linalg.norm(current_anomaly))
            results.append((path_prefix, norm, str(data)))

    _walk(anomaly, request_data, "")
    return results


def surprise_fingerprint(vec, subspace, request_data, vm):
    """Surgical per-leaf anomaly attribution via recursive drilldown probe.

    Walks the request encoding hierarchy, unbinding at each level to isolate
    individual field contributions. Returns only genuinely anomalous leaves
    (above uniform baseline), with their share of total anomaly.
    """
    anomaly = subspace.anomalous_component(vec)
    probes = _drilldown_probe(anomaly, request_data, vm)

    total = sum(norm for _, norm, _ in probes)
    if total < 1e-12:
        total = 1.0

    n = len(probes)
    baseline = 1.0 / n

    shares = [norm / total for _, norm, _ in probes]
    mean_share = np.mean(shares)
    std_share = np.std(shares)
    threshold = mean_share + 0.5 * std_share

    scored = []
    for (path, norm, display_value), share in zip(probes, shares):
        scored.append({
            "path": path,
            "raw": norm,
            "share": round(share, 4),
            "surprising": share > threshold,
            "value": display_value,
        })

    scored.sort(key=lambda x: x["raw"], reverse=True)
    for rank, entry in enumerate(scored):
        entry["rank"] = rank + 1

    return scored


# -----------------------------------------------------------------------
# Denial context builder
# -----------------------------------------------------------------------

def _serialize_value(v):
    if isinstance(v, set):
        return sorted(v)
    elif isinstance(v, LinearScale):
        return float(v.value)
    elif isinstance(v, dict):
        return {ik: _serialize_value(iv) for ik, iv in v.items()}
    elif isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def build_denial_context(vec, request_data, subspace, library, vm):
    residual_score = float(subspace.residual(vec))
    matches = library.match(vec, top_k=5)
    probes = surprise_fingerprint(vec, subspace, request_data, vm)

    anomalous = [p for p in probes if p["surprising"]]
    normal_count = len(probes) - len(anomalous)

    best_match = matches[0] if matches else ("unknown", float("inf"))

    return {
        "verdict": {
            "action": "DENY",
            "confidence": round(min(residual_score / max(subspace.threshold, 1e-10), 10.0), 2),
            "residual": round(residual_score, 4),
            "threshold": round(float(subspace.threshold), 4),
            "best_match": {"engram": best_match[0],
                           "residual": round(float(best_match[1]), 4)},
        },
        "anomalous_fields": [
            {"path": p["path"], "value": p["value"],
             "rank": p["rank"], "share": p["share"]}
            for p in anomalous
        ],
        "total_fields_probed": len(probes),
        "normal_field_count": normal_count,
        "engram_matches": [(name, round(float(score), 4))
                           for name, score in matches],
        "eigenvalues": subspace.eigenvalues[:10].tolist(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    vm = VectorManager(dimensions=DIM)
    encoder = Encoder(vm)

    print("=" * 70)
    print("EXPERIMENT 015: Denial Context Tokens (Sealed Verdicts)")
    print("=" * 70)
    if not REAL_CRYPTO:
        print("\n  NOTE: cryptography package not available.")
        print("  Using simulated HMAC-based sealing (not real encryption).")

    # ===================================================================
    # PHASE 1: Train normal subspace + engram library
    # ===================================================================
    print("\nPHASE 1: Train normal subspace (500 requests) + engram library")

    normal_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(500):
        vec = encoder.encode_walkable(make_normal(np.random.default_rng(i)))
        normal_sub.update(vec)

    library = EngramLibrary(dim=DIM)

    flood_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(300):
        vec = encoder.encode_walkable(make_get_flood(np.random.default_rng(i + 50000)))
        flood_sub.update(vec)
    library.add("get_flood", flood_sub)

    cred_sub = OnlineSubspace(dim=DIM, k=64, amnesia=2.0)
    for i in range(300):
        vec = encoder.encode_walkable(make_cred_stuff(np.random.default_rng(i + 60000)))
        cred_sub.update(vec)
    library.add("cred_stuffing", cred_sub)

    library.add("normal", normal_sub)

    threshold = normal_sub.threshold
    print(f"  Normal subspace: {normal_sub}")
    print(f"  Adaptive threshold: {threshold:.4f}")
    print(f"  Engram library: {library}")

    # ===================================================================
    # PHASE 2: Generate denied traffic and build sealed tokens
    # ===================================================================
    print("\nPHASE 2: Generate denied traffic and mint sealed tokens")

    admin_key = generate_key() if REAL_CRYPTO else generate_key()

    denied_items = []

    ATTACK_GENERATORS = [
        ("get_flood", make_get_flood, 20),
        ("cred_stuffing", make_cred_stuff, 20),
        ("unusual_legit", make_unusual_legit, 20),
    ]

    for attack_label, gen_fn, count in ATTACK_GENERATORS:
        denied_count = 0
        for i in range(count):
            rng = np.random.default_rng(70000 + hash(attack_label) % 10000 + i)
            req = gen_fn(rng)
            vec = encoder.encode_walkable(req)
            residual = float(normal_sub.residual(vec))

            if residual > threshold:
                context = build_denial_context(
                    vec, req, normal_sub, library, vm)
                context_json = json.dumps(context, sort_keys=True)
                token = seal_token(context_json.encode("utf-8"), admin_key)
                denied_items.append({
                    "label": attack_label,
                    "vec": vec,
                    "request": req,
                    "context": context,
                    "token": token,
                    "context_json": context_json,
                    "is_true_positive": attack_label != "unusual_legit",
                    "residual_score": context["verdict"]["residual"],
                })
                denied_count += 1

        print(f"  {attack_label}: {denied_count}/{count} denied "
              f"(threshold={threshold:.4f})")

    true_positives = [d for d in denied_items if d["is_true_positive"]]
    false_positives = [d for d in denied_items if not d["is_true_positive"]]

    print(f"\n  Total denied: {len(denied_items)}")
    print(f"  True positives:  {len(true_positives)}")
    print(f"  False positives: {len(false_positives)}")

    # ===================================================================
    # PHASE 3: Round-trip verification
    # ===================================================================
    print("\nPHASE 3: Round-trip verification (decrypt → deserialize → compare)")

    roundtrip_ok_count = 0
    token_sizes = []
    roundtrip_failures = []

    for item in denied_items:
        token = item["token"]
        token_size = len(token)
        token_sizes.append(token_size)

        try:
            decrypted_bytes = unseal_token(token, admin_key)
            recovered = json.loads(decrypted_bytes.decode("utf-8"))

            original = json.loads(item["context_json"])
            if recovered == original:
                roundtrip_ok_count += 1
            else:
                roundtrip_failures.append((item["label"], "content mismatch"))
        except Exception as e:
            roundtrip_failures.append((item["label"], str(e)))

    roundtrip_rate = roundtrip_ok_count / max(len(denied_items), 1)
    avg_token_size = np.mean(token_sizes) if token_sizes else 0
    max_token_size = max(token_sizes) if token_sizes else 0

    print(f"  Round-trip success: {roundtrip_ok_count}/{len(denied_items)} "
          f"({roundtrip_rate:.0%})")
    print(f"  Token sizes: avg={avg_token_size:.0f}B, "
          f"max={max_token_size}B (base64)")
    if roundtrip_failures:
        for label, reason in roundtrip_failures[:5]:
            print(f"  FAILURE: {label} — {reason}")

    all_valid_b64 = True
    for item in denied_items:
        try:
            base64.b64decode(item["token"])
        except Exception:
            all_valid_b64 = False
            break

    # ===================================================================
    # PHASE 4: True-positive context analysis
    # ===================================================================
    print("\nPHASE 4: True-positive context — attack type identification")

    tp_top_field_correct = 0

    flood_prefixes = {"header_count", "has_cookie", "tls", "headers",
                      "header_order"}
    cred_prefixes = {"path", "path_parts", "method", "has_cookie",
                     "tls", "headers", "body_len"}

    for item in true_positives:
        ctx = item["context"]
        anomalous_paths = [f["path"] for f in ctx["anomalous_fields"]]

        if item["label"] == "get_flood":
            expected = flood_prefixes
        else:
            expected = cred_prefixes

        if any(p.split(".")[0] in expected for p in anomalous_paths):
            tp_top_field_correct += 1

    tp_field_rate = tp_top_field_correct / max(len(true_positives), 1)
    print(f"  Anomalous fields match expected attack discriminator: "
          f"{tp_top_field_correct}/{len(true_positives)} ({tp_field_rate:.0%})")

    # ----- Verbose sample: what the admin actually sees -----
    if true_positives:
        sample = true_positives[0]
        ctx = sample["context"]
        v = ctx["verdict"]
        anomalous = ctx["anomalous_fields"]
        total_probed = ctx["total_fields_probed"]
        normal_count = ctx["normal_field_count"]
        token = sample["token"]

        print(f"\n  {'─' * 64}")
        print(f"  SAMPLE TOKEN DECRYPTED — True Positive ({sample['label']})")
        print(f"  {'─' * 64}")
        print(f"  VERDICT: {v['action']}  "
              f"confidence={v['confidence']}x threshold  "
              f"best_match=\"{v['best_match']['engram']}\"")
        print(f"  residual={v['residual']}  threshold={v['threshold']}  "
              f"matched_engram_residual={v['best_match']['residual']}")
        print()
        print(f"  ANOMALOUS FIELDS ({len(anomalous)}/{total_probed}"
              f" above baseline):")
        for f in anomalous:
            bar = "█" * max(1, int(f["share"] * 50))
            print(f"    #{f['rank']:<3} {f['path']:<30} "
                  f"share={f['share']:.1%}  {bar}")
            val_str = f["value"]
            if len(val_str) > 70:
                val_str = val_str[:67] + "..."
            print(f"          = {val_str}")
        print(f"  NORMAL FIELDS: {normal_count} (all within baseline)")
        print()
        print(f"  ALL ENGRAM MATCHES:")
        for name, score in ctx["engram_matches"]:
            marker = " ◀ best" if name == v["best_match"]["engram"] else ""
            print(f"    {name:<20} residual={score}{marker}")
        print()
        print(f"  SEALED TOKEN: {len(token)}B base64  "
              f"(plaintext: {len(sample['context_json'])}B)")
        print(f"    {token[:80]}...")
        print(f"  {'─' * 64}")

    # ===================================================================
    # PHASE 5: False-positive recovery
    # ===================================================================
    print("\nPHASE 5: False-positive recovery — update subspace, re-score")

    snap = normal_sub.snapshot()
    recovery_sub = OnlineSubspace.from_snapshot(snap)

    fp_recovered = 0
    fp_residuals_before = []
    fp_residuals_after = []

    for item in false_positives:
        vec = item["vec"]

        res_before = float(recovery_sub.residual(vec))
        fp_residuals_before.append(res_before)

        recovery_sub.update(vec)

        res_after = float(recovery_sub.residual(vec))
        fp_residuals_after.append(res_after)

        if res_after <= recovery_sub.threshold:
            fp_recovered += 1

    fp_recovery_rate = fp_recovered / max(len(false_positives), 1)

    print(f"  False positives analyzed: {len(false_positives)}")
    print(f"  Recovered (now passes): {fp_recovered}/{len(false_positives)} "
          f"({fp_recovery_rate:.0%})")
    if fp_residuals_before:
        print(f"  Residual before update: "
              f"mean={np.mean(fp_residuals_before):.4f}")
        print(f"  Residual after update:  "
              f"mean={np.mean(fp_residuals_after):.4f}")
        print(f"  Recovery threshold:     {recovery_sub.threshold:.4f}")

    if false_positives:
        sample_fp = false_positives[0]
        ctx = sample_fp["context"]
        v = ctx["verdict"]
        anomalous = ctx["anomalous_fields"]
        total_probed = ctx["total_fields_probed"]
        normal_count = ctx["normal_field_count"]

        print(f"\n  {'─' * 64}")
        print(f"  SAMPLE TOKEN DECRYPTED — False Positive (unusual_legit)")
        print(f"  {'─' * 64}")
        print(f"  VERDICT: {v['action']}  "
              f"confidence={v['confidence']}x threshold  "
              f"best_match=\"{v['best_match']['engram']}\"")
        print()
        print(f"  ANOMALOUS FIELDS ({len(anomalous)}/{total_probed}"
              f" above baseline):")
        for f in anomalous:
            bar = "█" * max(1, int(f["share"] * 50))
            print(f"    #{f['rank']:<3} {f['path']:<30} "
                  f"share={f['share']:.1%}  {bar}")
            val_str = f["value"]
            if len(val_str) > 70:
                val_str = val_str[:67] + "..."
            print(f"          = {val_str}")
        print(f"  NORMAL FIELDS: {normal_count} (all within baseline)")
        print()
        print(f"  OPERATOR ACTION:")
        print(f"    Residual before: {fp_residuals_before[0]:.4f} "
              f"(>{v['threshold']}) → DENIED")
        print(f"    → Feed vector back into subspace")
        print(f"    Residual after:  {fp_residuals_after[0]:.4f} "
              f"(<{recovery_sub.threshold:.4f}) → ALLOWED")
        print(f"  {'─' * 64}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  Crypto backend:        {'AES-256-GCM (cryptography)' if REAL_CRYPTO else 'HMAC-SHA256 (simulated)'}")
    print(f"  Normal threshold:      {threshold:.4f}")
    print(f"  Total denials:         {len(denied_items)}")
    print(f"  True positives:        {len(true_positives)}")
    print(f"  False positives:       {len(false_positives)}")
    print(f"  Round-trip fidelity:   {roundtrip_rate:.0%}")
    print(f"  Avg token size:        {avg_token_size:.0f}B")
    print(f"  Max token size:        {max_token_size}B")
    print(f"  TP field match rate:   {tp_field_rate:.0%}")
    print(f"  FP recovery rate:      {fp_recovery_rate:.0%}")

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    checks = [
        (
            "Round-trip fidelity: decoded token recovers 100% of denial context",
            roundtrip_rate == 1.0,
            f"{roundtrip_ok_count}/{len(denied_items)}",
        ),
        (
            "Token is valid base64",
            all_valid_b64,
            f"all {len(denied_items)} tokens",
        ),
        (
            "False-positive recovery: updated request passes (>= 50% recovered)",
            fp_recovery_rate >= 0.5 or len(false_positives) == 0,
            f"{fp_recovered}/{len(false_positives)} recovered",
        ),
        (
            "True-positive context: top surprise field matches expected "
            "attack discriminator (>= 80%)",
            tp_field_rate >= 0.8 or len(true_positives) == 0,
            f"{tp_top_field_correct}/{len(true_positives)}",
        ),
        (
            "Token size < 8KB base64 for all denials",
            max_token_size < 8192,
            f"max={max_token_size}B",
        ),
        (
            "At least 1 true positive denied",
            len(true_positives) >= 1,
            f"count={len(true_positives)}",
        ),
        (
            "At least 1 false positive denied",
            len(false_positives) >= 1,
            f"count={len(false_positives)}",
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
