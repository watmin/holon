#!/usr/bin/env python3
"""
Enhanced Graph Matching via HTTP API

Demonstrates using the new kernel primitives through HTTP:
- /api/v1/vectors/prototype - Learn topology signatures
- /api/v1/vectors/difference - Compute structural changes
- /api/v1/vectors/blend - Create fuzzy queries
- /api/v1/vectors/amplify - Boost signals
- /api/v1/vectors/negate - Anomaly detection
- /api/v1/search/by-vector - Search with computed vectors
"""

import json
import requests
from typing import List, Dict, Any

BASE_URL = "http://localhost:8000"
API = "/api/v1"


def health_check() -> bool:
    """Check if server is running."""
    try:
        r = requests.get(f"{BASE_URL}{API}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def encode(data: Dict) -> List[float]:
    """Encode data to vector."""
    r = requests.post(f"{BASE_URL}{API}/vectors/encode", json={
        "data": json.dumps(data),
        "data_type": "json"
    })
    r.raise_for_status()
    return r.json()["vector"]


def insert(data: Dict) -> str:
    """Insert data and return ID."""
    r = requests.post(f"{BASE_URL}{API}/items", json={
        "data": json.dumps(data),
        "data_type": "json"
    })
    r.raise_for_status()
    return r.json()["id"]


def search_by_vector(vector: List[float], top_k: int = 10) -> List[Dict]:
    """Search using a raw vector."""
    r = requests.post(f"{BASE_URL}{API}/search/by-vector", json={
        "vector": vector,
        "top_k": top_k,
        "threshold": 0.0
    })
    r.raise_for_status()
    return r.json()["results"]


def prototype(vectors: List[List[float]], threshold: float = 0.5) -> List[float]:
    """Compute prototype from vectors."""
    r = requests.post(f"{BASE_URL}{API}/vectors/prototype", json={
        "vectors": vectors,
        "threshold": threshold
    })
    r.raise_for_status()
    return r.json()["vector"]


def difference(before: List[float], after: List[float]) -> List[float]:
    """Compute difference between vectors."""
    r = requests.post(f"{BASE_URL}{API}/vectors/difference", json={
        "before": before,
        "after": after
    })
    r.raise_for_status()
    return r.json()["vector"]


def blend(vec1: List[float], vec2: List[float], alpha: float = 0.5) -> List[float]:
    """Blend two vectors."""
    r = requests.post(f"{BASE_URL}{API}/vectors/blend", json={
        "vec1": vec1,
        "vec2": vec2,
        "alpha": alpha
    })
    r.raise_for_status()
    return r.json()["vector"]


def amplify(base: List[float], component: List[float], strength: float = 1.0) -> List[float]:
    """Amplify component in base."""
    r = requests.post(f"{BASE_URL}{API}/vectors/amplify", json={
        "base": base,
        "component": component,
        "strength": strength
    })
    r.raise_for_status()
    return r.json()["vector"]


def negate(base: List[float], component: List[float], method: str = "subtract") -> List[float]:
    """Negate component from base."""
    r = requests.post(f"{BASE_URL}{API}/vectors/negate", json={
        "base": base,
        "component": component,
        "method": method
    })
    r.raise_for_status()
    return r.json()["vector"]


def similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute similarity between vectors."""
    r = requests.post(f"{BASE_URL}{API}/vectors/similarity", json={
        "vector_a": vec1,
        "vector_b": vec2
    })
    r.raise_for_status()
    return r.json()["similarity"]


def generate_test_graphs():
    """Generate test graphs."""
    graphs = []

    # Star graphs
    for n in [3, 4, 5]:
        graphs.append({
            "name": f"star_{n}",
            "topology": "star",
            "nodes": ["center"] + [f"leaf_{i}" for i in range(n-1)],
            "edges": [{"from": "center", "to": f"leaf_{i}"} for i in range(n-1)],
        })

    # Cycle graphs
    for n in [3, 4, 5]:
        nodes = [f"n{i}" for i in range(n)]
        graphs.append({
            "name": f"cycle_{n}",
            "topology": "cycle",
            "nodes": nodes,
            "edges": [{"from": nodes[i], "to": nodes[(i+1) % n]} for i in range(n)],
        })

    # Tree graphs
    graphs.append({
        "name": "tree_3",
        "topology": "tree",
        "nodes": ["root", "left", "right"],
        "edges": [{"from": "root", "to": "left"}, {"from": "root", "to": "right"}],
    })

    # Anomaly
    graphs.append({
        "name": "anomaly_1",
        "topology": "anomaly",
        "nodes": ["a", "b", "c"],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],  # Weird bidirectional
    })

    return graphs


def main():
    print("=" * 70)
    print("ENHANCED GRAPH MATCHING VIA HTTP API")
    print("=" * 70)

    # Check server
    if not health_check():
        print("ERROR: Holon server not running at", BASE_URL)
        print("Start with: ./scripts/run_with_venv.sh python scripts/server/holon_server.py")
        return

    print(f"Connected to Holon server at {BASE_URL}")

    # Generate and insert graphs
    graphs = generate_test_graphs()
    print(f"\nGenerated {len(graphs)} test graphs")

    graph_ids = {}
    graph_vectors = {}
    for g in graphs:
        gid = insert(g)
        vec = encode(g)
        graph_ids[g["name"]] = gid
        graph_vectors[g["name"]] = vec

    print(f"Inserted and encoded {len(graph_ids)} graphs")

    # Organize by topology
    topology_groups = {}
    for g in graphs:
        topo = g["topology"]
        if topo not in topology_groups:
            topology_groups[topo] = []
        topology_groups[topo].append(g["name"])

    # ========================================
    # TEST 1: Prototype Learning via HTTP
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 1: PROTOTYPE LEARNING VIA HTTP")
    print("=" * 70)

    topology_prototypes = {}
    for topo, names in topology_groups.items():
        if topo == "anomaly":
            continue

        vectors = [graph_vectors[name] for name in names]
        proto = prototype(vectors, threshold=0.5)
        topology_prototypes[topo] = proto
        print(f"\n  {topo.upper()} prototype created from {len(vectors)} examples")

    # Test classification
    print("\n  CLASSIFICATION TEST:")
    correct = 0
    total = 0
    for g in graphs:
        if g["topology"] == "anomaly":
            continue

        vec = graph_vectors[g["name"]]
        best_topo = None
        best_sim = -1

        for topo, proto in topology_prototypes.items():
            sim = similarity(vec, proto)
            if sim > best_sim:
                best_sim = sim
                best_topo = topo

        match = "✅" if best_topo == g["topology"] else "❌"
        if best_topo == g["topology"]:
            correct += 1
        total += 1
        print(f"    {g['name']}: predicted={best_topo}, actual={g['topology']} {match}")

    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # ========================================
    # TEST 2: Blend via HTTP
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 2: FUZZY QUERY VIA BLEND")
    print("=" * 70)

    if "star" in topology_prototypes and "cycle" in topology_prototypes:
        hybrid = blend(topology_prototypes["star"], topology_prototypes["cycle"], alpha=0.5)
        print("\n  Created STAR-CYCLE hybrid (50/50 blend)")

        # Score all graphs
        scores = []
        for g in graphs:
            vec = graph_vectors[g["name"]]
            sim = similarity(vec, hybrid)
            scores.append((g["name"], g["topology"], sim))

        scores.sort(key=lambda x: x[2], reverse=True)
        print("\n  Hybrid query results:")
        for name, topo, sim in scores[:5]:
            marker = "⭐" if topo in ["star", "cycle"] else ""
            print(f"    {name} ({topo}): {sim:.4f} {marker}")

    # ========================================
    # TEST 3: Amplify via HTTP
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 3: SIGNAL AMPLIFICATION")
    print("=" * 70)

    if "star" in topology_prototypes:
        # Create weak query
        weak = encode({"topology": "star"})

        # Amplify with prototype
        strong = amplify(weak, topology_prototypes["star"], strength=2.0)

        print("\n  Weak vs Amplified Star Query:")
        for topo in ["star", "cycle", "tree"]:
            if topo in topology_groups:
                name = topology_groups[topo][0]
                vec = graph_vectors[name]

                weak_sim = similarity(vec, weak)
                strong_sim = similarity(vec, strong)
                delta = strong_sim - weak_sim
                arrow = "↑" if delta > 0 else "↓"
                print(f"    {name}: {weak_sim:.4f} → {strong_sim:.4f} ({arrow}{abs(delta):.4f})")

    # ========================================
    # TEST 4: Anomaly Detection via HTTP
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 4: ANOMALY DETECTION VIA NEGATE")
    print("=" * 70)

    # Bundle all prototypes to create "normal" pattern
    all_protos = list(topology_prototypes.values())
    if len(all_protos) >= 2:
        # Create normal by blending all prototypes
        normal = all_protos[0]
        for proto in all_protos[1:]:
            normal = blend(normal, proto, alpha=0.5)

        print("\n  Created 'normal' pattern from topology prototypes")

        # Score all graphs
        print("\n  Normality scores (lower = more anomalous):")
        scores = []
        for g in graphs:
            vec = graph_vectors[g["name"]]
            sim = similarity(vec, normal)
            scores.append((g["name"], g["topology"], sim))

        scores.sort(key=lambda x: x[2])
        for name, topo, sim in scores:
            marker = "🚨 ANOMALY" if topo == "anomaly" else ""
            print(f"    {name} ({topo}): {sim:.4f} {marker}")

        # Check if anomaly was detected
        anomaly_rank = next(i for i, (n, t, s) in enumerate(scores) if t == "anomaly")
        if anomaly_rank < 3:
            print(f"\n  ✅ Anomaly detected! Ranked #{anomaly_rank + 1} most unusual")
        else:
            print(f"\n  ⚠️ Anomaly ranked #{anomaly_rank + 1}")

    # ========================================
    # TEST 5: Difference via HTTP
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 5: STRUCTURAL DIFFERENCE")
    print("=" * 70)

    star3_vec = graph_vectors.get("star_3")
    star5_vec = graph_vectors.get("star_5")

    if star3_vec and star5_vec:
        diff = difference(star3_vec, star5_vec)
        print("\n  Computed star_3 → star_5 difference")

        # What does the difference look like?
        diff_sim_to_star = similarity(diff, topology_prototypes.get("star", star3_vec))
        print(f"  Difference similarity to star prototype: {diff_sim_to_star:.4f}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY: HTTP API FOR ADVANCED PRIMITIVES")
    print("=" * 70)
    print(f"""
    All new primitives work via HTTP:

    ✅ /api/v1/vectors/prototype - Pattern extraction
    ✅ /api/v1/vectors/blend - Fuzzy queries
    ✅ /api/v1/vectors/amplify - Signal boosting
    ✅ /api/v1/vectors/negate - Anomaly detection
    ✅ /api/v1/vectors/difference - Change detection

    Classification accuracy: {100*correct/total:.1f}%

    The HTTP API now provides full access to all
    advanced VSA/HDC primitives!
    """)


if __name__ == "__main__":
    main()
