#!/usr/bin/env python3
"""
Debug RPM Vector Encoding

Let's understand why our geometric rule learning is failing.
We'll examine the vector encodings and see if they preserve geometric relationships.
"""

import json

from holon import CPUStore


def debug_matrix_encoding():
    """Debug how matrices are encoded into vectors."""
    print("🔍 DEBUGGING RPM VECTOR ENCODING")
    print("=" * 50)

    store = CPUStore(dimensions=16000)

    # Create simple test matrices
    test_matrices = [
        {
            "matrix-id": "simple-progression",
            "rule": "progression",
            "attributes": {"shape", "count", "color"},
            "panels": {
                "row1-col1": {"shapes": ["circle"], "count": 1, "color": "black"},
                "row1-col2": {
                    "shapes": ["circle", "square"],
                    "count": 2,
                    "color": "white",
                },
                "row1-col3": {
                    "shapes": ["circle", "square", "triangle"],
                    "count": 3,
                    "color": "red",
                },
                "row2-col1": {
                    "shapes": ["circle", "square"],
                    "count": 2,
                    "color": "black",
                },
                "row2-col2": {
                    "shapes": ["circle", "square", "triangle"],
                    "count": 3,
                    "color": "white",
                },
                "row2-col3": {
                    "shapes": ["circle", "square", "triangle", "diamond"],
                    "count": 4,
                    "color": "red",
                },
                "row3-col1": {
                    "shapes": ["circle", "square", "triangle"],
                    "count": 3,
                    "color": "black",
                },
                "row3-col2": {
                    "shapes": ["circle", "square", "triangle", "diamond"],
                    "count": 4,
                    "color": "white",
                },
                "row3-col3": {
                    "shapes": ["circle", "square", "triangle", "diamond", "star"],
                    "count": 5,
                    "color": "red",
                },
            },
        },
        {
            "matrix-id": "simple-xor",
            "rule": "xor",
            "attributes": {"shape", "count", "color"},
            "panels": {
                "row1-col1": {"shapes": [], "count": 0, "color": "black"},
                "row1-col2": {
                    "shapes": ["square", "circle"],
                    "count": 2,
                    "color": "black",
                },
                "row1-col3": {"shapes": ["square"], "count": 1, "color": "black"},
                "row2-col1": {
                    "shapes": ["square", "circle"],
                    "count": 2,
                    "color": "black",
                },
                "row2-col2": {"shapes": [], "count": 0, "color": "black"},
                "row2-col3": {"shapes": ["circle"], "count": 1, "color": "black"},
                "row3-col1": {"shapes": ["square"], "count": 1, "color": "black"},
                "row3-col2": {"shapes": ["circle"], "count": 1, "color": "black"},
                "row3-col3": {"shapes": [], "count": 0, "color": "black"},
            },
        },
    ]

    # Insert matrices and examine their encodings
    for matrix in test_matrices:
        print(f"\n📊 Matrix: {matrix['matrix-id']} ({matrix['rule']} rule)")

        # Show the matrix structure
        print("  Structure:")
        for pos in [
            "row1-col1",
            "row1-col2",
            "row1-col3",
            "row2-col1",
            "row2-col2",
            "row2-col3",
            "row3-col1",
            "row3-col2",
            "row3-col3",
        ]:
            if pos in matrix["panels"]:
                panel = matrix["panels"][pos]
                shapes = panel.get("shapes", [])
                print(
                    f"    {pos}: {shapes} (count: {len(shapes)}, color: {panel.get('color', 'unknown')})"
                )

        # Insert and get ID
        matrix_json = json.dumps(matrix, default=list)
        matrix_id = store.insert(matrix_json)
        print(f"  Inserted with ID: {matrix_id}")

    # Test similarity between complete matrices
    print("\n🔗 Testing similarity between complete matrices:")
    results = store.query('{"rule": "progression"}', top_k=5)
    if results:
        for i, (id, score, data) in enumerate(results):
            print(".3f")
    results = store.query('{"rule": "xor"}', top_k=5)
    if results:
        for i, (id, score, data) in enumerate(results):
            print(".3f")
    # Test cross-rule similarity (should be low)
    print("\n🔄 Testing cross-rule similarity (should be low):")
    results = store.query('{"matrix-id": "simple-progression"}', top_k=5)
    progression_results = [r for r in results if r[2].get("rule") == "progression"]
    xor_results = [r for r in results if r[2].get("rule") == "xor"]

    if progression_results:
        print(".3f")
    if xor_results:
        print(".3f")
    # Now test missing panel completion
    print("\n🧩 Testing missing panel completion:")  # Create matrix with missing panel
    incomplete_matrix = {
        "matrix-id": "test-missing",
        "rule": "xor",
        "attributes": {"shape", "count", "color"},
        "missing-position": "row3-col3",
        "panels": {
            "row1-col1": {"shapes": [], "count": 0, "color": "black"},
            "row1-col2": {"shapes": ["square", "circle"], "count": 2, "color": "black"},
            "row1-col3": {"shapes": ["square"], "count": 1, "color": "black"},
            "row2-col1": {"shapes": ["square", "circle"], "count": 2, "color": "black"},
            "row2-col2": {"shapes": [], "count": 0, "color": "black"},
            "row2-col3": {"shapes": ["circle"], "count": 1, "color": "black"},
            "row3-col1": {"shapes": ["square"], "count": 1, "color": "black"},
            "row3-col2": {"shapes": ["circle"], "count": 1, "color": "black"}
            # row3-col3 is missing - should be [] (empty)
        },
    }

    print("  Incomplete XOR matrix (missing row3-col3):")
    for pos in [
        "row1-col1",
        "row1-col2",
        "row1-col3",
        "row2-col1",
        "row2-col2",
        "row2-col3",
        "row3-col1",
        "row3-col2",
    ]:
        panel = incomplete_matrix["panels"][pos]
        shapes = panel.get("shapes", [])
        print(f"    {pos}: {shapes} (count: {len(shapes)})")

    print("  Expected missing panel: [] (count: 0, color: black)")

    # Insert incomplete matrix
    matrix_json = json.dumps(incomplete_matrix, default=list)
    store.insert(matrix_json)

    # Query for completion
    probe = {"panels": incomplete_matrix["panels"], "rule": "xor"}

    print("\n🔮 Searching for geometrically similar complete matrices...")
    results = store.query(
        json.dumps(probe, default=list),
        negations={"missing-position": {"$any": True}},
        top_k=5,
    )

    print(f"  Found {len(results)} results:")
    found_correct = False
    for i, (id, score, data) in enumerate(results):
        matrix_name = data.get("matrix-id", "unknown")
        rule = data.get("rule", "unknown")

        # Check what the complete matrix has in row3-col3
        complete_panel = data.get("panels", {}).get("row3-col3", {})
        actual_shapes = complete_panel.get("shapes", [])
        actual_count = complete_panel.get("count", 0)
        actual_color = complete_panel.get("color", "unknown")

        is_correct = (
            actual_shapes == [] and actual_count == 0 and actual_color == "black"
        )
        status = "✅ CORRECT!" if is_correct else "❌ wrong"
        if is_correct:
            found_correct = True

        print(".3f")
    if found_correct:
        print("  🎯 SUCCESS: Found geometrically correct completion!")
    else:
        print("  ❌ FAILURE: No correct completion found")

        # Let's see what the actual complete XOR matrix looks like
        complete_results = store.query(
            '{"rule": "xor"}', negations={"missing-position": {"$any": True}}
        )
        if complete_results:
            data = complete_results[0][2]
            complete_panel = data.get("panels", {}).get("row3-col3", {})
            print(
                f"  Expected from complete matrix: {complete_panel.get('shapes', [])} "
                f"(count: {complete_panel.get('count', 0)}, color: {complete_panel.get('color', 'unknown')})"
            )


if __name__ == "__main__":
    debug_matrix_encoding()
