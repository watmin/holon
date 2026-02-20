#!/usr/bin/env python3
"""
Compositional Recall - Algebraic Queries Over Incident Memory

Vector databases give you similarity search. Holon gives you an algebra.

The same incident library can be queried with: basic recall, negation
("like this but NOT database-related"), amplification ("boost production
incidents"), and analogy ("what would this frontend issue look like if
it happened in the backend?"). Each is a single algebraic operation on
vectors — no pipeline, no re-indexing, no query rewrite.

Run with: ./scripts/run_with_venv.sh python -m examples.showcases.compositional_recall.showcase
"""

from holon.kernel import (
    Encoder,
    VectorManager,
    amplify,
    analogy,
    attend,
    negate,
    prototype,
    topk_similar,
)
from holon.memory import EngramLibrary, OnlineSubspace

DIM = 4096

# Incident library: service outages, bugs, security events across 4 service layers
INCIDENTS = [
    # Frontend
    {
        "id": "INC-001",
        "service": "frontend",
        "layer": "web",
        "severity": "high",
        "type": "latency",
        "env": "prod",
        "team": "ui",
        "resolved": True,
    },
    {
        "id": "INC-002",
        "service": "frontend",
        "layer": "web",
        "severity": "medium",
        "type": "crash",
        "env": "prod",
        "team": "ui",
        "resolved": True,
    },
    {
        "id": "INC-003",
        "service": "frontend",
        "layer": "web",
        "severity": "low",
        "type": "latency",
        "env": "staging",
        "team": "ui",
        "resolved": True,
    },
    {
        "id": "INC-004",
        "service": "frontend",
        "layer": "web",
        "severity": "critical",
        "type": "outage",
        "env": "prod",
        "team": "ui",
        "resolved": False,
    },
    # Backend API
    {
        "id": "INC-005",
        "service": "api",
        "layer": "backend",
        "severity": "high",
        "type": "latency",
        "env": "prod",
        "team": "platform",
        "resolved": True,
    },
    {
        "id": "INC-006",
        "service": "api",
        "layer": "backend",
        "severity": "medium",
        "type": "crash",
        "env": "prod",
        "team": "platform",
        "resolved": True,
    },
    {
        "id": "INC-007",
        "service": "api",
        "layer": "backend",
        "severity": "critical",
        "type": "outage",
        "env": "prod",
        "team": "platform",
        "resolved": False,
    },
    {
        "id": "INC-008",
        "service": "api",
        "layer": "backend",
        "severity": "low",
        "type": "latency",
        "env": "staging",
        "team": "platform",
        "resolved": True,
    },
    # Database
    {
        "id": "INC-009",
        "service": "postgres",
        "layer": "database",
        "severity": "critical",
        "type": "outage",
        "env": "prod",
        "team": "infra",
        "resolved": False,
    },
    {
        "id": "INC-010",
        "service": "postgres",
        "layer": "database",
        "severity": "high",
        "type": "latency",
        "env": "prod",
        "team": "infra",
        "resolved": True,
    },
    {
        "id": "INC-011",
        "service": "redis",
        "layer": "database",
        "severity": "medium",
        "type": "crash",
        "env": "prod",
        "team": "infra",
        "resolved": True,
    },
    {
        "id": "INC-012",
        "service": "postgres",
        "layer": "database",
        "severity": "low",
        "type": "latency",
        "env": "staging",
        "team": "infra",
        "resolved": True,
    },
    # Security
    {
        "id": "INC-013",
        "service": "auth",
        "layer": "security",
        "severity": "critical",
        "type": "intrusion",
        "env": "prod",
        "team": "security",
        "resolved": False,
    },
    {
        "id": "INC-014",
        "service": "auth",
        "layer": "security",
        "severity": "high",
        "type": "intrusion",
        "env": "prod",
        "team": "security",
        "resolved": True,
    },
    {
        "id": "INC-015",
        "service": "api",
        "layer": "security",
        "severity": "critical",
        "type": "intrusion",
        "env": "prod",
        "team": "security",
        "resolved": False,
    },
    {
        "id": "INC-016",
        "service": "auth",
        "layer": "security",
        "severity": "medium",
        "type": "crash",
        "env": "staging",
        "team": "security",
        "resolved": True,
    },
]


def show_results(label, results, incident_vecs, incidents, top_k=4):
    print(f"\n  Query   : {label}")
    ids = [inc["id"] for inc in incidents]
    sims = topk_similar(results, incident_vecs, k=top_k)
    for rank, (idx, sim) in enumerate(sims, 1):
        inc = incidents[idx]
        print(
            f"    {rank}. {inc['id']}  {inc['service']:12s}  {inc['layer']:10s}  "
            f"{inc['severity']:8s}  {inc['type']:10s}  env={inc['env']:7s}  sim={sim:.3f}"
        )


def main():
    print("=" * 65)
    print("COMPOSITIONAL RECALL")
    print("Algebraic Queries Over Incident Memory")
    print("=" * 65)

    vm = VectorManager(dimensions=DIM)
    enc = Encoder(vm)

    # Encode all incidents
    print(f"\nEncoding {len(INCIDENTS)} incidents across 4 service layers...")
    incident_vecs = [enc.encode_data(inc) for inc in INCIDENTS]

    # Build layer prototypes for algebraic operations
    db_vecs = [
        v for v, inc in zip(incident_vecs, INCIDENTS) if inc["layer"] == "database"
    ]
    frontend_vecs = [
        v for v, inc in zip(incident_vecs, INCIDENTS) if inc["layer"] == "web"
    ]
    backend_vecs = [
        v for v, inc in zip(incident_vecs, INCIDENTS) if inc["layer"] == "backend"
    ]
    security_vecs = [
        v for v, inc in zip(incident_vecs, INCIDENTS) if inc["layer"] == "security"
    ]
    prod_vecs = [v for v, inc in zip(incident_vecs, INCIDENTS) if inc["env"] == "prod"]

    db_proto = prototype(db_vecs)
    frontend_proto = prototype(frontend_vecs)
    backend_proto = prototype(backend_vecs)
    security_proto = prototype(security_vecs)
    prod_proto = prototype(prod_vecs)

    # Build OnlineSubspace per layer for EngramLibrary
    library = EngramLibrary(dim=DIM)
    for layer_name, vecs in [
        ("database", db_vecs),
        ("frontend", frontend_vecs),
        ("backend", backend_vecs),
        ("security", security_vecs),
    ]:
        ss = OnlineSubspace(dim=DIM, k=16, amnesia=2.0)
        for v in vecs * 3:
            ss.update(v)
        library.add(layer_name, ss)

    print(f"  Layer subspaces: {', '.join(library.names())}")

    # ── QUERY ─────────────────────────────────────────────────────
    # We have a new incident: a high-severity latency issue in prod
    probe_raw = {
        "service": "api",
        "layer": "backend",
        "severity": "high",
        "type": "latency",
        "env": "prod",
        "team": "platform",
        "resolved": False,
    }
    probe = enc.encode_data(probe_raw)

    print("\n" + "-" * 65)
    print("PROBE: high-severity latency in backend/prod (unresolved)")
    print("-" * 65)

    # 1. Basic recall — any vector DB can do this
    show_results("Basic recall (cosine similarity)", probe, incident_vecs, INCIDENTS)

    # 2. Negation — "similar to this, but NOT database-layer"
    # Subtracts the database structural signal. No re-indexing, no filter clause.
    probe_no_db = negate(probe, db_proto, method="project")
    show_results(
        "negate(probe, db_proto)  — exclude database incidents",
        probe_no_db,
        incident_vecs,
        INCIDENTS,
    )

    # 3. Amplification — "boost incidents with prod impact"
    # Amplifies the production-environment signal in the query vector.
    probe_prod = amplify(probe, prod_proto, strength=1.5)
    show_results(
        "amplify(probe, prod_proto)  — prioritise production impact",
        probe_prod,
        incident_vecs,
        INCIDENTS,
    )

    # 4. Analogy — "what would this backend latency look like if it happened in auth/security?"
    # analogy(a, b, c) computes: c + difference(a, b)
    # = "take this backend incident, apply the backend→security relationship"
    probe_as_security = analogy(backend_proto, security_proto, probe)
    show_results(
        "analogy(backend→security, probe)  — transfer pattern to security layer",
        probe_as_security,
        incident_vecs,
        INCIDENTS,
    )

    # 5. Attend — "what part of this incident overlaps with known security patterns?"
    # Returns the probe with security-resonant dimensions emphasized.
    probe_security_view = attend(probe, security_proto, strength=2.0, mode="soft")
    show_results(
        "attend(probe, security_proto)  — surface security-resonant signal",
        probe_security_view,
        incident_vecs,
        INCIDENTS,
    )

    # 6. EngramLibrary match — "which layer does this most resemble?"
    print("\n  Query   : EngramLibrary.match — which layer manifold fits best?")
    matches = library.match(probe, top_k=4)
    for rank, (name, residual) in enumerate(matches, 1):
        print(
            f"    {rank}. layer='{name}'  residual={residual:.2f}  (lower = better fit)"
        )

    print(f"\n{'=' * 65}")
    print("Each query is one algebraic operation on the same encoded library.")
    print("No re-indexing. No query rewrite. No pipeline.")
    # Try: compose operations — negate(amplify(probe, prod_proto), db_proto)
    # Try: add a "resolved=True" prototype and use negate() to find only open incidents


if __name__ == "__main__":
    main()
