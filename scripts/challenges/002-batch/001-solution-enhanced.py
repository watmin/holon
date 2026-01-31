#!/usr/bin/env python3
"""
Enhanced RPM Solution with New Kernel Primitives

This solution demonstrates using the new VSA primitives:
- prototype: Extract common pattern from rule types
- difference: Compute row-to-row transformations
- blend: Create hybrid rule patterns
- amplify: Strengthen rule signals
- negate: Remove known patterns to isolate unknowns

These primitives enable more sophisticated reasoning about RPM matrices.
"""

import json
import random
import numpy as np

from holon import CPUStore, HolonClient


def generate_rpm_matrices():
    """Generate synthetic RPM matrices for testing."""
    matrices = []

    # Progression rule matrices
    for i in range(1, 5):
        matrices.append({
            "matrix-id": f"progression-{i}",
            "rule": "progression",
            "attributes": ["shape", "count", "color"],
            "panels": {
                "row1-col1": {"shapes": ["circle"], "count": 1, "color": "black"},
                "row1-col2": {"shapes": ["circle", "square"], "count": 2, "color": "white"},
                "row1-col3": {"shapes": ["circle", "square", "triangle"], "count": 3, "color": "red"},
                "row2-col1": {"shapes": ["star"], "count": 1, "color": "blue"},
                "row2-col2": {"shapes": ["star", "diamond"], "count": 2, "color": "green"},
                "row2-col3": {"shapes": ["star", "diamond", "pentagon"], "count": 3, "color": "yellow"},
                "row3-col1": {"shapes": ["hexagon"], "count": 1, "color": "purple"},
                "row3-col2": {"shapes": ["hexagon", "octagon"], "count": 2, "color": "orange"},
                "row3-col3": {"shapes": ["hexagon", "octagon", "cross"], "count": 3, "color": "pink"},
            },
            "complete": True
        })

    # XOR rule matrices
    for i in range(1, 4):
        matrices.append({
            "matrix-id": f"xor-{i}",
            "rule": "xor",
            "attributes": ["shape", "presence"],
            "panels": {
                "row1-col1": {"shapes": ["circle"], "count": 1, "color": "black"},
                "row1-col2": {"shapes": ["square"], "count": 1, "color": "black"},
                "row1-col3": {"shapes": ["circle", "square"], "count": 2, "color": "black"},
                "row2-col1": {"shapes": ["triangle"], "count": 1, "color": "black"},
                "row2-col2": {"shapes": ["diamond"], "count": 1, "color": "black"},
                "row2-col3": {"shapes": ["triangle", "diamond"], "count": 2, "color": "black"},
                "row3-col1": {"shapes": ["circle", "triangle"], "count": 2, "color": "black"},
                "row3-col2": {"shapes": ["square", "diamond"], "count": 2, "color": "black"},
                "row3-col3": {"shapes": [], "count": 0, "color": "black"},  # XOR result
            },
            "complete": True
        })

    # Union rule matrices
    for i in range(1, 4):
        matrices.append({
            "matrix-id": f"union-{i}",
            "rule": "union",
            "attributes": ["shape", "accumulation"],
            "panels": {
                "row1-col1": {"shapes": ["circle"], "count": 1, "color": "red"},
                "row1-col2": {"shapes": ["square"], "count": 1, "color": "blue"},
                "row1-col3": {"shapes": ["circle", "square"], "count": 2, "color": "purple"},
                "row2-col1": {"shapes": ["triangle"], "count": 1, "color": "green"},
                "row2-col2": {"shapes": ["diamond"], "count": 1, "color": "yellow"},
                "row2-col3": {"shapes": ["triangle", "diamond"], "count": 2, "color": "orange"},
                "row3-col1": {"shapes": ["star"], "count": 1, "color": "white"},
                "row3-col2": {"shapes": ["pentagon"], "count": 1, "color": "black"},
                "row3-col3": {"shapes": ["star", "pentagon"], "count": 2, "color": "gray"},
            },
            "complete": True
        })

    return matrices


def main():
    print("=" * 70)
    print("ENHANCED RPM SOLUTION WITH NEW KERNEL PRIMITIVES")
    print("=" * 70)

    # Initialize
    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    matrices = generate_rpm_matrices()
    print(f"\nGenerated {len(matrices)} RPM matrices")

    # Ingest matrices
    matrix_ids = {}
    for matrix in matrices:
        mid = client.insert_json(matrix)
        matrix_ids[matrix["matrix-id"]] = mid
    print(f"Ingested {len(matrix_ids)} matrices")

    # ========================================
    # ENHANCEMENT 1: Rule Prototype Learning
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 1: RULE PROTOTYPE LEARNING")
    print("=" * 70)
    print("\nUsing 'prototype' to extract the essence of each rule type...")

    # Collect vectors for each rule type
    rule_vectors = {"progression": [], "xor": [], "union": []}

    for matrix in matrices:
        rule = matrix["rule"]
        vec = client.encode_vectors_json(matrix)
        rule_vectors[rule].append(np.array(vec))

    # Create prototypes for each rule
    rule_prototypes = {}
    for rule, vectors in rule_vectors.items():
        if vectors:
            prototype = store.prototype(vectors, threshold=0.5)
            rule_prototypes[rule] = prototype
            print(f"\n  {rule.upper()} PROTOTYPE:")
            print(f"    Created from {len(vectors)} examples")

    # Test prototype matching
    print("\n  PROTOTYPE MATCHING TEST:")
    test_matrix = matrices[0]  # progression-1
    test_vec = np.array(client.encode_vectors_json(test_matrix))

    from holon.similarity import normalized_dot_similarity
    for rule, proto in rule_prototypes.items():
        sim = normalized_dot_similarity(test_vec, proto)
        marker = "✅" if rule == test_matrix["rule"] else ""
        print(f"    {test_matrix['matrix-id']} vs {rule} prototype: {sim:.4f} {marker}")

    # ========================================
    # ENHANCEMENT 2: Transformation Difference
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 2: ROW TRANSFORMATION DIFFERENCE")
    print("=" * 70)
    print("\nUsing 'difference' to extract what changes between rows...")

    # Take a progression matrix and compute row differences
    prog_matrix = matrices[0]

    row1_data = {
        "row1-col1": prog_matrix["panels"]["row1-col1"],
        "row1-col2": prog_matrix["panels"]["row1-col2"],
        "row1-col3": prog_matrix["panels"]["row1-col3"],
    }
    row2_data = {
        "row2-col1": prog_matrix["panels"]["row2-col1"],
        "row2-col2": prog_matrix["panels"]["row2-col2"],
        "row2-col3": prog_matrix["panels"]["row2-col3"],
    }
    row3_data = {
        "row3-col1": prog_matrix["panels"]["row3-col1"],
        "row3-col2": prog_matrix["panels"]["row3-col2"],
        "row3-col3": prog_matrix["panels"]["row3-col3"],
    }

    row1_vec = np.array(client.encode_vectors_json(row1_data))
    row2_vec = np.array(client.encode_vectors_json(row2_data))
    row3_vec = np.array(client.encode_vectors_json(row3_data))

    # Compute transformations
    transform_1_to_2 = store.difference(row1_vec, row2_vec)
    transform_2_to_3 = store.difference(row2_vec, row3_vec)

    # Check if transformations are similar (consistent rule)
    transform_sim = normalized_dot_similarity(transform_1_to_2, transform_2_to_3)
    print(f"\n  Transformation consistency (row1→2 vs row2→3): {transform_sim:.4f}")

    if transform_sim > 0.3:
        print("  ✅ High consistency indicates a systematic rule is being applied")
    else:
        print("  ⚠️ Low consistency - rule may vary between rows")

    # ========================================
    # ENHANCEMENT 3: Rule Blending
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 3: HYBRID RULE BLENDING")
    print("=" * 70)
    print("\nUsing 'blend' to create hybrid rule prototypes...")

    # Blend progression and union rules
    if "progression" in rule_prototypes and "union" in rule_prototypes:
        hybrid_prog_union = store.blend(
            rule_prototypes["progression"],
            rule_prototypes["union"],
            alpha=0.5
        )
        print("\n  Created PROGRESSION-UNION hybrid (50/50 blend)")

        # Test which matrices match the hybrid
        print("\n  HYBRID MATCHING:")
        for matrix in matrices[:6]:  # Test first 6
            vec = np.array(client.encode_vectors_json(matrix))
            sim = normalized_dot_similarity(vec, hybrid_prog_union)
            print(f"    {matrix['matrix-id']}: {sim:.4f}")

    # ========================================
    # ENHANCEMENT 4: Pattern Amplification
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 4: RULE SIGNAL AMPLIFICATION")
    print("=" * 70)
    print("\nUsing 'amplify' to strengthen rule-specific features...")

    # Amplify the progression signal in a mixed query
    if "progression" in rule_prototypes:
        # Create a weak query
        weak_query = {"rule": "progression"}
        weak_vec = np.array(client.encode_vectors_json(weak_query))

        # Amplify with the progression prototype
        amplified_vec = store.amplify(weak_vec, rule_prototypes["progression"], strength=2.0)

        print("\n  AMPLIFICATION COMPARISON:")
        print("  (Higher scores for progression matrices = better)")

        for matrix in matrices[:6]:
            vec = np.array(client.encode_vectors_json(matrix))
            weak_sim = normalized_dot_similarity(vec, weak_vec)
            amp_sim = normalized_dot_similarity(vec, amplified_vec)
            improvement = "↑" if amp_sim > weak_sim else "↓"
            print(f"    {matrix['matrix-id']}: weak={weak_sim:.4f} → amplified={amp_sim:.4f} {improvement}")

    # ========================================
    # ENHANCEMENT 5: Pattern Negation
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 5: PATTERN NEGATION")
    print("=" * 70)
    print("\nUsing 'negate' to find matrices that DON'T match a pattern...")

    if "xor" in rule_prototypes:
        # Start with a general matrix query
        general_query = {"attributes": ["shape"]}
        general_vec = np.array(client.encode_vectors_json(general_query))

        # Negate the XOR pattern
        non_xor_vec = store.negate(general_vec, rule_prototypes["xor"], method="subtract")

        print("\n  NEGATION RESULTS (lower XOR scores expected):")
        xor_sims = []
        non_xor_sims = []

        for matrix in matrices:
            vec = np.array(client.encode_vectors_json(matrix))
            sim = normalized_dot_similarity(vec, non_xor_vec)
            if matrix["rule"] == "xor":
                xor_sims.append(sim)
            else:
                non_xor_sims.append(sim)

        avg_xor = sum(xor_sims) / len(xor_sims) if xor_sims else 0
        avg_non_xor = sum(non_xor_sims) / len(non_xor_sims) if non_xor_sims else 0

        print(f"    Average similarity for XOR matrices: {avg_xor:.4f}")
        print(f"    Average similarity for non-XOR matrices: {avg_non_xor:.4f}")

        if avg_non_xor > avg_xor:
            print("    ✅ Negation successfully reduces XOR pattern similarity!")
        else:
            print("    ⚠️ Negation effect was subtle")

    # ========================================
    # ENHANCEMENT 6: Intelligent Missing Panel Prediction
    # ========================================
    print("\n" + "=" * 70)
    print("ENHANCEMENT 6: INTELLIGENT MISSING PANEL PREDICTION")
    print("=" * 70)
    print("\nCombining primitives for sophisticated panel prediction...")

    # Create an incomplete matrix
    incomplete = {
        "matrix-id": "test-incomplete",
        "rule": "progression",
        "panels": {
            "row1-col1": {"shapes": ["A"], "count": 1},
            "row1-col2": {"shapes": ["A", "B"], "count": 2},
            "row1-col3": {"shapes": ["A", "B", "C"], "count": 3},
            "row2-col1": {"shapes": ["D"], "count": 1},
            "row2-col2": {"shapes": ["D", "E"], "count": 2},
            # Missing: row2-col3
        }
    }

    # Step 1: Identify the rule using prototypes
    incomplete_vec = np.array(client.encode_vectors_json(incomplete))

    print("\n  Step 1: Rule Identification via Prototypes")
    best_rule = None
    best_sim = -1
    for rule, proto in rule_prototypes.items():
        sim = normalized_dot_similarity(incomplete_vec, proto)
        print(f"    vs {rule}: {sim:.4f}")
        if sim > best_sim:
            best_sim = sim
            best_rule = rule

    print(f"    → Detected rule: {best_rule}")

    # Step 2: Compute row transformation
    print("\n  Step 2: Row Transformation Analysis")
    row1 = {k: v for k, v in incomplete["panels"].items() if k.startswith("row1")}
    row2 = {k: v for k, v in incomplete["panels"].items() if k.startswith("row2")}

    row1_v = np.array(client.encode_vectors_json(row1))
    row2_v = np.array(client.encode_vectors_json(row2))

    # The partial row2 should follow similar pattern to row1
    col1_to_col2_row1 = store.difference(
        np.array(client.encode_vectors_json({"shapes": ["A"], "count": 1})),
        np.array(client.encode_vectors_json({"shapes": ["A", "B"], "count": 2}))
    )
    print("    Computed col1→col2 transformation from row1")

    # Step 3: Predict missing panel
    print("\n  Step 3: Prediction")
    col2_row2_vec = np.array(client.encode_vectors_json({"shapes": ["D", "E"], "count": 2}))

    # Apply the learned transformation
    predicted_vec = col2_row2_vec + col1_to_col2_row1  # Simple additive model

    # Find best match
    candidates = [
        {"shapes": ["D", "E", "F"], "count": 3},  # Correct for progression
        {"shapes": ["D", "E"], "count": 2},
        {"shapes": ["F"], "count": 1},
        {"shapes": [], "count": 0},
    ]

    print("    Candidate ranking:")
    for i, cand in enumerate(candidates):
        cand_vec = np.array(client.encode_vectors_json(cand))
        sim = normalized_dot_similarity(predicted_vec, cand_vec)
        marker = "← Expected" if cand["count"] == 3 else ""
        print(f"      {i+1}. {cand}: {sim:.4f} {marker}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY: NEW PRIMITIVE BENEFITS FOR RPM")
    print("=" * 70)
    print("""
    1. PROTOTYPE: Learn rule archetypes from examples
       → Enables rule classification without explicit labels

    2. DIFFERENCE: Extract transformation patterns
       → Captures "what changes" between rows/columns

    3. BLEND: Create hybrid rule patterns
       → Useful for finding matrices with mixed characteristics

    4. AMPLIFY: Strengthen weak signals
       → Improves precision when searching for specific rules

    5. NEGATE: Remove unwanted patterns
       → Find "everything except X" queries

    These primitives enable more sophisticated reasoning about
    visual analogy problems in high-dimensional space!
    """)


if __name__ == "__main__":
    main()
