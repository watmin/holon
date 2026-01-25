#!/usr/bin/env python3
"""
Compare Comprehensive Test vs Debug Test

Let's see what's different between our working debug test and failing comprehensive test.
"""

import json

from holon import CPUStore


# Copy functions from our working debug
def generate_rpm_matrix(matrix_id, rule_type, attributes=None, missing_position=None):
    """Generate a synthetic RPM matrix with specified rule and attributes."""
    if attributes is None:
        attributes = {"shape", "count", "color"}

    shapes = ["circle", "square", "triangle", "diamond", "star"]
    colors = ["black", "white", "red", "blue", "green"]
    counts = [1, 2, 3, 4, 5]

    panels = {}

    if rule_type == "progression":
        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                panel_shapes = set()
                shape_count = row + col - 1
                for i in range(min(shape_count, len(shapes))):
                    panel_shapes.add(shapes[i])

                panel = {
                    "shapes": panel_shapes,
                    "count": len(panel_shapes),
                    "color": colors[(col - 1) % len(colors)],
                    "progression": "add-one",
                    "attributes": attributes,
                }
                panels[position] = panel

    elif rule_type == "xor":
        base_shapes = {"circle", "square", "triangle"}

        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                panel_shapes = set()
                for i, shape in enumerate(["circle", "square", "triangle"]):
                    if (row ^ col) & (1 << i):
                        panel_shapes.add(shape)

                panel = {
                    "shapes": panel_shapes,
                    "count": len(panel_shapes),
                    "color": "black",
                    "rule": "xor",
                    "attributes": attributes,
                }
                panels[position] = panel

    edn_data = {
        "matrix-id": matrix_id,
        "panels": panels,
        "rule": rule_type,
        "attributes": attributes,
    }

    if missing_position:
        edn_data["missing-position"] = missing_position

    return edn_data


def edn_to_json(edn_data):
    """Convert EDN-like Python dict to JSON-compatible format."""
    import json

    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(item) for item in obj]
        else:
            return obj

    return json.dumps(convert_sets(edn_data))


def compute_expected_missing_panel(matrix_data, missing_position):
    """Compute what the missing panel should be based on the matrix rule."""
    panels = matrix_data.get("panels", {})
    rule = matrix_data.get("rule", "")

    parts = missing_position.split("-")
    row = int(parts[0][3:])
    col = int(parts[1][3:])

    if rule == "progression":
        shape_count = row + col - 1
        shapes = set()
        shape_options = ["circle", "square", "triangle", "diamond", "star"]
        for i in range(min(shape_count, len(shape_options))):
            shapes.add(shape_options[i])

        colors = ["black", "white", "red", "blue", "green"]
        color = colors[(col - 1) % len(colors)]

        return {
            "shapes": shapes,
            "count": len(shapes),
            "color": color,
            "progression": "add-one",
            "attributes": matrix_data.get("attributes", set()),
        }

    elif rule == "xor":
        shapes = set()
        for i, shape in enumerate(["circle", "square", "triangle"]):
            if (row ^ col) & (1 << i):
                shapes.add(shape)

        return {
            "shapes": shapes,
            "count": len(shapes),
            "color": "black",
            "rule": "xor",
            "attributes": matrix_data.get("attributes", set()),
        }

    return {"shapes": set(), "count": 0, "color": "unknown", "rule": "unknown"}


def test_debug_approach():
    """Test the debug approach that works."""
    print("🔍 TESTING DEBUG APPROACH (that works)")
    print("=" * 50)

    store = CPUStore(dimensions=16000)

    # Create simple complete matrices (like debug script)
    complete_matrices = [
        generate_rpm_matrix(
            "debug-progression", "progression", {"shape", "count", "color"}
        ),
        generate_rpm_matrix("debug-xor", "xor", {"shape", "count", "color"}),
    ]

    for matrix in complete_matrices:
        matrix_json = edn_to_json(matrix)
        store.insert(matrix_json)

    # Test missing panel completion
    incomplete_matrix = generate_rpm_matrix(
        "debug-incomplete", "xor", {"shape", "count", "color"}, "row3-col3"
    )
    matrix_json = edn_to_json(incomplete_matrix)
    store.insert(matrix_json)

    expected_panel = compute_expected_missing_panel(incomplete_matrix, "row3-col3")
    print(
        f"Expected missing panel: {list(expected_panel['shapes'])} (count: {expected_panel['count']})"
    )

    # Search for completion
    probe_structure = {"panels": incomplete_matrix["panels"], "rule": "xor"}

    results = store.query(
        edn_to_json(probe_structure),
        negations={"missing-position": {"$any": True}},
        top_k=5,
    )

    print(f"Found {len(results)} completion candidates:")
    found_correct = False
    for result in results:
        data = result[2]
        actual_missing = data.get("panels", {}).get("row3-col3", {})
        actual_shapes = set(actual_missing.get("shapes", []))
        actual_count = actual_missing.get("count", 0)

        is_correct = (
            actual_shapes == expected_panel["shapes"]
            and actual_count == expected_panel["count"]
        )
        status = "✅ CORRECT!" if is_correct else "❌ wrong"
        if is_correct:
            found_correct = True
        print(f"  {status} {list(actual_shapes)} (count: {actual_count})")

    return found_correct


def test_comprehensive_approach():
    """Test the comprehensive approach (that fails)."""
    print("\n🔍 TESTING COMPREHENSIVE APPROACH (that fails)")
    print("=" * 50)

    store = CPUStore(dimensions=16000)

    # Create multiple matrices like comprehensive test
    for i in range(5):
        # Add complete matrices
        complete = generate_rpm_matrix(
            f"comp-complete-{i}", "xor", {"shape", "count", "color"}
        )
        matrix_json = edn_to_json(complete)
        store.insert(matrix_json)

        # Add incomplete matrix
        incomplete = generate_rpm_matrix(
            f"comp-incomplete-{i}", "xor", {"shape", "count", "color"}, "row3-col3"
        )
        matrix_json = edn_to_json(incomplete)
        store.insert(matrix_json)

    # Test one completion
    test_matrix = generate_rpm_matrix(
        "comp-test", "xor", {"shape", "count", "color"}, "row3-col3"
    )
    expected_panel = compute_expected_missing_panel(test_matrix, "row3-col3")
    print(
        f"Expected missing panel: {list(expected_panel['shapes'])} (count: {expected_panel['count']})"
    )

    probe_structure = {"panels": test_matrix["panels"], "rule": "xor"}

    results = store.query(
        edn_to_json(probe_structure),
        negations={"missing-position": {"$any": True}},
        top_k=5,
    )

    print(f"Found {len(results)} completion candidates:")
    found_correct = False
    for result in results:
        data = result[2]
        actual_missing = data.get("panels", {}).get("row3-col3", {})
        actual_shapes = set(actual_missing.get("shapes", []))
        actual_count = actual_missing.get("count", 0)

        is_correct = (
            actual_shapes == expected_panel["shapes"]
            and actual_count == expected_panel["count"]
        )
        status = "✅ CORRECT!" if is_correct else "❌ wrong"
        if is_correct:
            found_correct = True
        print(f"  {status} {list(actual_shapes)} (count: {actual_count})")

    return found_correct


def main():
    """Compare the two approaches."""
    print("🔬 COMPARING DEBUG vs COMPREHENSIVE APPROACHES")
    print("=" * 60)

    debug_works = test_debug_approach()
    comprehensive_works = test_comprehensive_approach()

    print("\n" + "=" * 60)
    print("🎯 COMPARISON RESULTS")
    print("=" * 60)

    print(
        f"Debug approach (simple matrices): {'✅ WORKS' if debug_works else '❌ FAILS'}"
    )
    print(
        f"Comprehensive approach (many matrices): {'✅ WORKS' if comprehensive_works else '❌ FAILS'}"
    )

    if debug_works and not comprehensive_works:
        print("\n🔍 DIAGNOSIS:")
        print("✅ Basic geometric reasoning works with clean, simple matrices")
        print("❌ Complex scenarios with many matrices cause interference")
        print(
            "💡 Issue: Vector similarity overwhelmed by noise from multiple similar matrices"
        )

        print("\n🛠️  POSSIBLE SOLUTIONS:")
        print("1. Better matrix differentiation in encoding")
        print("2. Improved similarity thresholds")
        print("3. More structured geometric representations")
        print("4. Rule-specific search strategies")


if __name__ == "__main__":
    main()
