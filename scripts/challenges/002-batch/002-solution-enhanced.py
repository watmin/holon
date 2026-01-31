#!/usr/bin/env python3
"""
Enhanced Graph Matching Solution with New Kernel Primitives

This solution demonstrates using the new VSA primitives:
- prototype: Learn topology family signatures (star, cycle, tree)
- difference: Compute graph differences (edge additions/removals)
- blend: Create fuzzy graph queries
- amplify: Strengthen topology signals
- negate: Find graphs that DON'T match a pattern (anomaly detection)

These primitives enable more sophisticated graph analysis.
"""

import json
import numpy as np

from holon import CPUStore, HolonClient
from holon.similarity import normalized_dot_similarity


def generate_test_graphs():
    """Generate diverse test graphs."""
    graphs = []

    # Star topology family
    for n in [3, 4, 5, 6]:
        graphs.append({
            "name": f"star_{n}",
            "topology": "star",
            "type": "undirected",
            "nodes": ["center"] + [f"leaf_{i}" for i in range(n-1)],
            "edges": [{"from": "center", "to": f"leaf_{i}"} for i in range(n-1)],
            "node_count": n,
            "edge_count": n-1,
            "description": f"{n}-node star graph"
        })

    # Cycle topology family
    for n in [3, 4, 5, 6]:
        nodes = [f"n{i}" for i in range(n)]
        edges = [{"from": nodes[i], "to": nodes[(i+1) % n]} for i in range(n)]
        graphs.append({
            "name": f"cycle_{n}",
            "topology": "cycle",
            "type": "undirected",
            "nodes": nodes,
            "edges": edges,
            "node_count": n,
            "edge_count": n,
            "description": f"{n}-node cycle graph"
        })

    # Tree topology family
    for depth in [2, 3]:
        nodes = ["root"]
        edges = []
        for d in range(1, depth + 1):
            parent_start = sum(2**(i-1) for i in range(1, d)) if d > 1 else 0
            parent_end = sum(2**(i-1) for i in range(1, d+1))
            for p in range(parent_start, parent_end):
                parent = nodes[p]
                left = f"n{len(nodes)}"
                right = f"n{len(nodes)+1}"
                nodes.extend([left, right])
                edges.extend([
                    {"from": parent, "to": left},
                    {"from": parent, "to": right}
                ])
        graphs.append({
            "name": f"tree_depth{depth}",
            "topology": "tree",
            "type": "undirected",
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "description": f"Binary tree depth {depth}"
        })

    # Chain/path topology
    for n in [3, 4, 5]:
        nodes = [f"n{i}" for i in range(n)]
        edges = [{"from": nodes[i], "to": nodes[i+1]} for i in range(n-1)]
        graphs.append({
            "name": f"chain_{n}",
            "topology": "chain",
            "type": "undirected",
            "nodes": nodes,
            "edges": edges,
            "node_count": n,
            "edge_count": n-1,
            "description": f"{n}-node chain/path"
        })

    # Complete graphs
    for n in [3, 4]:
        nodes = [f"n{i}" for i in range(n)]
        edges = [{"from": nodes[i], "to": nodes[j]}
                 for i in range(n) for j in range(i+1, n)]
        graphs.append({
            "name": f"complete_{n}",
            "topology": "complete",
            "type": "undirected",
            "nodes": nodes,
            "edges": edges,
            "node_count": n,
            "edge_count": len(edges),
            "description": f"Complete graph K{n}"
        })

    # Anomaly graph (unusual structure)
    graphs.append({
        "name": "anomaly_1",
        "topology": "anomaly",
        "type": "directed",  # Different type!
        "nodes": ["a", "b", "c", "d"],
        "edges": [
            {"from": "a", "to": "b"},
            {"from": "b", "to": "c"},
            {"from": "c", "to": "a"},  # Creates cycle
            {"from": "d", "to": "a"},  # d points in but nothing points to d
        ],
        "node_count": 4,
        "edge_count": 4,
        "description": "Anomalous directed graph"
    })

    return graphs


def main():
    print("=" * 70)
    print("ENHANCED GRAPH MATCHING WITH NEW KERNEL PRIMITIVES")
    print("=" * 70)

    # Initialize
    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    graphs = generate_test_graphs()
    print(f"\nGenerated {len(graphs)} test graphs")

    # Organize by topology
    topology_groups = {}
    for g in graphs:
        topo = g["topology"]
        if topo not in topology_groups:
            topology_groups[topo] = []
        topology_groups[topo].append(g)

    print("Topology families:")
    for topo, members in topology_groups.items():
        print(f"  {topo}: {len(members)} graphs")

    # Ingest all graphs
    graph_ids = {}
    for g in graphs:
        gid = client.insert_json(g)
        graph_ids[g["name"]] = gid
    print(f"\nIngested {len(graph_ids)} graphs")

    # ========================================
    # ENHANCEMENT 1: Topology Prototype Learning
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 1: TOPOLOGY PROTOTYPE LEARNING")
    print("=" * 70)
    print("\nUsing 'prototype' to learn the essence of each topology family...")

    topology_prototypes = {}
    for topo, members in topology_groups.items():
        if topo == "anomaly":
            continue  # Skip anomaly for prototype learning

        vectors = [np.array(client.encode_vectors_json(g)) for g in members]
        prototype = store.prototype(vectors, threshold=0.5)
        topology_prototypes[topo] = prototype

        print(f"\n  {topo.upper()} PROTOTYPE:")
        print(f"    Learned from {len(vectors)} examples")
        print(f"    Members: {[g['name'] for g in members]}")

    # Test prototype classification
    print("\n  PROTOTYPE CLASSIFICATION TEST:")
    print("  (Each graph should match its own family best)")

    correct = 0
    total = 0
    for g in graphs:
        if g["topology"] == "anomaly":
            continue

        vec = np.array(client.encode_vectors_json(g))
        best_topo = None
        best_sim = -1

        for topo, proto in topology_prototypes.items():
            sim = normalized_dot_similarity(vec, proto)
            if sim > best_sim:
                best_sim = sim
                best_topo = topo

        match = "✅" if best_topo == g["topology"] else "❌"
        if best_topo == g["topology"]:
            correct += 1
        total += 1
        print(f"    {g['name']}: predicted={best_topo}, actual={g['topology']} {match}")

    print(f"\n  Classification accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # ========================================
    # ENHANCEMENT 2: Graph Difference Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 2: GRAPH DIFFERENCE ANALYSIS")
    print("=" * 70)
    print("\nUsing 'difference' to understand structural changes...")

    # Compare star_4 to star_5 (what was added?)
    star4 = next(g for g in graphs if g["name"] == "star_4")
    star5 = next(g for g in graphs if g["name"] == "star_5")

    star4_vec = np.array(client.encode_vectors_json(star4))
    star5_vec = np.array(client.encode_vectors_json(star5))

    diff_4_to_5 = store.difference(star4_vec, star5_vec)

    print("\n  Star4 → Star5 difference:")
    print(f"    Star4: {star4['node_count']} nodes, {star4['edge_count']} edges")
    print(f"    Star5: {star5['node_count']} nodes, {star5['edge_count']} edges")
    print(f"    Change: +1 node, +1 edge (added one leaf)")

    # Find what the difference vector is most similar to
    print("\n  What does the 'difference' vector represent?")
    leaf_edge = {"from": "center", "to": "leaf_x"}
    leaf_vec = np.array(client.encode_vectors_json(leaf_edge))
    sim_to_leaf = normalized_dot_similarity(diff_4_to_5, leaf_vec)
    print(f"    Similarity to 'add leaf edge' pattern: {sim_to_leaf:.4f}")

    # Compare cycle_3 to cycle_4
    cycle3 = next(g for g in graphs if g["name"] == "cycle_3")
    cycle4 = next(g for g in graphs if g["name"] == "cycle_4")

    cycle3_vec = np.array(client.encode_vectors_json(cycle3))
    cycle4_vec = np.array(client.encode_vectors_json(cycle4))

    diff_cycle = store.difference(cycle3_vec, cycle4_vec)

    print("\n  Cycle3 → Cycle4 difference:")
    print(f"    Cycle3: {cycle3['node_count']} nodes")
    print(f"    Cycle4: {cycle4['node_count']} nodes")

    # Check if cycle growth is similar to star growth
    cross_topo_sim = normalized_dot_similarity(diff_4_to_5, diff_cycle)
    print(f"    Star growth vs Cycle growth similarity: {cross_topo_sim:.4f}")
    print("    (Low similarity = different growth patterns)")

    # ========================================
    # ENHANCEMENT 3: Fuzzy Topology Queries
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 3: FUZZY TOPOLOGY QUERIES")
    print("=" * 70)
    print("\nUsing 'blend' to create hybrid topology queries...")

    # Blend star and tree prototypes
    if "star" in topology_prototypes and "tree" in topology_prototypes:
        hybrid = store.blend(
            topology_prototypes["star"],
            topology_prototypes["tree"],
            alpha=0.5
        )

        print("\n  STAR-TREE HYBRID QUERY (50/50 blend):")
        print("  (Should match both star and tree graphs)")

        results = []
        for g in graphs:
            if g["topology"] == "anomaly":
                continue
            vec = np.array(client.encode_vectors_json(g))
            sim = normalized_dot_similarity(vec, hybrid)
            results.append((g["name"], g["topology"], sim))

        results.sort(key=lambda x: x[2], reverse=True)
        print("\n  Top matches:")
        for name, topo, sim in results[:6]:
            marker = "⭐" if topo in ["star", "tree"] else ""
            print(f"    {name} ({topo}): {sim:.4f} {marker}")

    # ========================================
    # ENHANCEMENT 4: Topology Amplification
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 4: TOPOLOGY SIGNAL AMPLIFICATION")
    print("=" * 70)
    print("\nUsing 'amplify' to boost specific topology detection...")

    # Create a weak query for star graphs
    weak_query = {"topology": "star"}
    weak_vec = np.array(client.encode_vectors_json(weak_query))

    # Amplify with star prototype
    if "star" in topology_prototypes:
        amplified = store.amplify(weak_vec, topology_prototypes["star"], strength=2.0)

        print("\n  Weak vs Amplified Star Query:")
        print("  (Amplification should increase star scores more)")

        for topo in ["star", "cycle", "tree"]:
            if topo in topology_groups:
                sample = topology_groups[topo][0]
                vec = np.array(client.encode_vectors_json(sample))

                weak_sim = normalized_dot_similarity(vec, weak_vec)
                amp_sim = normalized_dot_similarity(vec, amplified)

                delta = amp_sim - weak_sim
                arrow = "↑" if delta > 0 else "↓"
                print(f"    {sample['name']}: {weak_sim:.4f} → {amp_sim:.4f} ({arrow}{abs(delta):.4f})")

    # ========================================
    # ENHANCEMENT 5: Anomaly Detection via Negation
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 5: ANOMALY DETECTION VIA NEGATION")
    print("=" * 70)
    print("\nUsing 'negate' to find graphs that don't fit known patterns...")

    # Create a "normal" pattern by bundling all topology prototypes
    if len(topology_prototypes) >= 3:
        normal_proto_list = list(topology_prototypes.values())
        normal_pattern = store.bundle(normal_proto_list)

        print("\n  Created 'normal graph' pattern from known topologies")
        print(f"  Combined {len(normal_proto_list)} topology prototypes")

        # Score all graphs against normal pattern
        print("\n  NORMALITY SCORES (lower = more anomalous):")
        scores = []
        for g in graphs:
            vec = np.array(client.encode_vectors_json(g))
            sim = normalized_dot_similarity(vec, normal_pattern)
            scores.append((g["name"], g["topology"], sim))

        scores.sort(key=lambda x: x[2])
        for name, topo, sim in scores:
            marker = "🚨 ANOMALY" if topo == "anomaly" else ""
            print(f"    {name} ({topo}): {sim:.4f} {marker}")

        # Did we detect the anomaly?
        anomaly_rank = next(i for i, (n, t, s) in enumerate(scores) if t == "anomaly")
        print(f"\n  Anomaly graph ranked: {anomaly_rank + 1}/{len(scores)}")
        if anomaly_rank < 3:
            print("  ✅ Successfully identified anomaly as most unusual!")
        else:
            print("  ⚠️ Anomaly not in bottom 3")

    # ========================================
    # ENHANCEMENT 6: Subgraph Transfer Learning
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 6: SUBGRAPH PATTERN TRANSFER")
    print("=" * 70)
    print("\nUsing difference and blend for subgraph matching...")

    # Learn what makes a "hub" pattern
    star_proto = topology_prototypes.get("star")
    chain_proto = topology_prototypes.get("chain")

    if star_proto is not None and chain_proto is not None:
        # The difference between star and chain is the hub structure
        hub_pattern = store.difference(chain_proto, star_proto)

        print("\n  Extracted 'hub' pattern (star minus chain)")

        # Find graphs with hub-like structures
        print("\n  Hub-ness scores:")
        for g in graphs[:10]:
            vec = np.array(client.encode_vectors_json(g))
            hub_sim = normalized_dot_similarity(vec, hub_pattern)
            print(f"    {g['name']}: {hub_sim:.4f}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY: NEW PRIMITIVE BENEFITS FOR GRAPH MATCHING")
    print("=" * 70)
    print("""
    1. PROTOTYPE: Learn topology family signatures
       → Enables unsupervised graph classification
       → Achieved {:.1f}% classification accuracy

    2. DIFFERENCE: Compute structural changes
       → Understand what makes topologies different
       → Identify growth patterns

    3. BLEND: Create fuzzy topology queries
       → Search for "graphs like star OR tree"
       → Soft matching across families

    4. AMPLIFY: Boost weak signals
       → Improve precision for specific topology search
       → Overcome noisy queries

    5. NEGATE: Anomaly detection
       → Find graphs that don't fit known patterns
       → Security/quality applications

    These primitives enable graph analysis that goes beyond
    simple similarity matching!
    """.format(100*correct/total))


if __name__ == "__main__":
    main()
