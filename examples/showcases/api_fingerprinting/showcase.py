#!/usr/bin/env python3
"""
API Fingerprinting - Variant-Resilient Endpoint Recognition

Mint one engram per logical endpoint, then show that requests with extra
headers, swapped user-agents, or added body fields still match correctly,
while a genuinely unknown endpoint gets a uniformly high residual.

Run with: ./scripts/run_with_venv.sh python -m examples.showcases.api_fingerprinting.showcase
"""

import random

from holon.kernel import Encoder, VectorManager
from holon.memory import EngramLibrary, OnlineSubspace

DIM = 4096

AGENTS = ["curl/7.68", "python-requests/2.28", "PostmanRuntime/7.32"]

ENDPOINTS = {
    "list_users": ("GET", "/api/v1/users", 200, ["users", "total", "page"]),
    "create_user": ("POST", "/api/v1/users", 201, ["user_id", "created_at"]),
    "get_user": ("GET", "/api/v1/users/:id", 200, ["id", "name", "email"]),
    "list_orders": ("GET", "/api/v1/orders", 200, ["orders", "total", "page"]),
    "health": ("GET", "/health", 200, ["status", "uptime"]),
}


def make_request(rng, method, path, status, body_keys, extra_headers=None):
    req = {
        "method": method,
        "path": path,
        "status": status,
        "body_keys": body_keys,
        "headers": {
            "content-type": "application/json",
            "user-agent": rng.choice(AGENTS),
            "x-request-id": f"req-{rng.randint(1000, 9999)}",
        },
    }
    if rng.random() > 0.3:
        req["headers"]["authorization"] = "Bearer <token>"
    if extra_headers:
        req["headers"].update(extra_headers)
    return req


def main():
    print("=" * 65)
    print("API FINGERPRINTING")
    print("=" * 65)

    rng = random.Random(42)
    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)
    library = EngramLibrary(dim=DIM)

    # ── Train: 10 example requests per endpoint ───────────────────
    print("\nMinting endpoint fingerprints...")
    for name, (method, path, status, body_keys) in ENDPOINTS.items():
        subspace = OnlineSubspace(dim=DIM, k=32, amnesia=2.0)
        for _ in range(10):
            subspace.update(
                enc.encode_data(make_request(rng, method, path, status, body_keys))
            )
        library.add(name, subspace, method=method, path=path)
        print(f"  ✓  '{name}'  ({method} {path})")

    # ── Variant resilience: same endpoint, different surface ──────
    print("\n" + "-" * 65)
    print("VARIANT RESILIENCE")
    print("-" * 65)

    variants = [
        (
            "list_users",
            "GET",
            "/api/v1/users",
            200,
            ["users", "total", "page"],
            {"x-forwarded-for": "203.0.113.1", "x-custom": "extra"},
        ),
        (
            "create_user",
            "POST",
            "/api/v1/users",
            201,
            ["user_id", "created_at"],
            {"user-agent": "MyApp/3.0"},
        ),
        (
            "get_user",
            "GET",
            "/api/v1/users/:id",
            200,
            ["id", "name", "email", "last_login"],
            {},
        ),
    ]
    for expected, method, path, status, body_keys, extra in variants:
        req = make_request(rng, method, path, status, body_keys, extra)
        best, residual = library.match(enc.encode_data(req), top_k=1)[0]
        correct = best == expected
        print(f"\n  variant  : {method} {path}  (+{len(extra)} extra fields)")
        print(f"  matched  : '{best}'  residual={residual:.2f}  correct={correct}")

    # ── Anomaly: structurally unknown endpoint ────────────────────
    print("\n" + "-" * 65)
    print("ANOMALY DETECTION")
    print("-" * 65)

    unknown = {
        "method": "DELETE",
        "path": "/admin/purge",
        "status": 204,
        "body_keys": ["deleted_count"],
        "headers": {"content-type": "application/json"},
    }
    matches = library.match(enc.encode_data(unknown), top_k=3)
    print("\n  DELETE /admin/purge  (never seen before)")
    for rank, (name, res) in enumerate(matches, 1):
        print(f"    {rank}. '{name}'  residual={res:.2f}")
    avg = sum(r for _, r in matches) / len(matches)
    print(f"  avg residual={avg:.2f}  → uniformly high: no match")

    print(f"\n{'=' * 65}")
    # Try: swap method only (GET→POST on same path) to see how much it shifts residual.


if __name__ == "__main__":
    main()
