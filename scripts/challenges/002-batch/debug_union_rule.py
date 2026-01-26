#!/usr/bin/env python3
"""
Debug script to understand why union rule fails in Challenge 2 RPM
"""

import json
from holon import CPUStore

# Import the matrix generation functions from the original
def generate_rpm_matrix(matrix_id, rule_type, attributes=None, missing_position=None):
    """Copy of the matrix generation from Challenge 2"""
    if attributes is None:
        attributes = {"shape", "count", "color"}

    shapes = ["circle", "square", "triangle", "diamond", "star"]
    colors = ["black", "white", "red", "blue", "green"]

    panels = {}

    if rule_type == "union":
        # Union of shapes from row and column headers
        row_shapes = [{"circle"}, {"square"}, {"triangle"}]
        col_shapes = [{"diamond"}, {"star"}, {"circle"}]  # Some overlap

        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                # Union of row and column shape sets
                panel_shapes = row_shapes[row - 1] | col_shapes[col - 1]

                panel = {
                    "shapes": panel_shapes,
                    "count": len(panel_shapes),
                    "color": colors[(row + col - 2) % len(colors)],
                    "rule": "union",
                    "attributes": attributes,
                }
                panels[position] = panel

    elif rule_type == "intersection":
        # Intersection for comparison
        row_shapes = [
            {"circle", "square"},
            {"square", "triangle"},
            {"triangle", "diamond"},
        ]
        col_shapes = [
            {"circle", "triangle"},
            {"square", "diamond"},
            {"circle", "square"},
        ]

        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                # Intersection of row and column shape sets
                panel_shapes = row_shapes[row - 1] & col_shapes[col - 1]

                panel = {
                    "shapes": panel_shapes,
                    "count": len(panel_shapes),
                    "color": colors[(row + col - 2) % len(colors)],
                    "rule": "intersection",
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
    """Convert EDN to JSON for Holon"""
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

def compute_expected_union_panel(matrix_data, missing_position):
    """Compute expected union panel using the FIXED logic"""
    panels = matrix_data.get("panels", {})
    parts = missing_position.split("-")
    row = int(parts[0][3:])
    col = int(parts[1][3:])

    # Collect all panels by row and column to analyze patterns
    row_panels = {}
    col_panels = {}

    for pos, panel in panels.items():
        parts = pos.split("-")
        p_row = int(parts[0][3:])
        p_col = int(parts[1][3:])

        if p_row not in row_panels:
            row_panels[p_row] = []
        if p_col not in col_panels:
            col_panels[p_col] = []

        row_panels[p_row].append(panel.get("shapes", set()))
        col_panels[p_col].append(panel.get("shapes", set()))

    # For the target row, find shapes that appear in all its panels (indicating row-specific shapes)
    target_row_panels = row_panels.get(row, [])
    if target_row_panels:
        # Start with intersection of all panels in the target row
        row_defining = set(target_row_panels[0])
        for panel_shapes in target_row_panels[1:]:
            row_defining &= set(panel_shapes)
    else:
        row_defining = set()

    # For the target column, find shapes that appear in all its panels
    target_col_panels = col_panels.get(col, [])
    if target_col_panels:
        col_defining = set(target_col_panels[0])
        for panel_shapes in target_col_panels[1:]:
            col_defining &= set(panel_shapes)
    else:
        col_defining = set()

    # The expected panel should be the union of the row-defining and column-defining shapes
    shapes = row_defining | col_defining

    # Color pattern based on position
    colors = ["black", "white", "red", "blue", "green"]
    color = colors[(row + col - 2) % len(colors)]

    return {
        "shapes": shapes,
        "count": len(shapes),
        "color": color,
    }

def debug_union_rule():
    """Debug the union rule issue"""
    print("🔍 Debugging Union Rule Failure")
    print("=" * 40)

    # Generate a union matrix with missing position
    matrix = generate_rpm_matrix("debug-union", "union", missing_position="row3-col3")

    print("📋 Generated Union Matrix (missing row3-col3):")
    panels = matrix["panels"]
    for pos in ["row1-col1", "row1-col2", "row1-col3", "row2-col1", "row2-col2", "row2-col3", "row3-col1", "row3-col2"]:
        if pos in panels:
            panel = panels[pos]
            print(f"  {pos}: {list(panel['shapes'])} (count: {panel['count']}, color: {panel['color']})")

    # Compute expected missing panel
    expected = compute_expected_union_panel(matrix, "row3-col3")
    print("\n🎯 Expected missing panel (row3-col3):")
    print(f"  Shapes: {list(expected['shapes'])} (count: {expected['count']})")
    print(f"  Color: {expected['color']}")

    # Now let's see what the original generation would produce for the complete matrix
    print("\n🔍 What the complete union matrix should have at row3-col3:")
    complete_matrix = generate_rpm_matrix("debug-union-complete", "union")
    complete_panel = complete_matrix["panels"]["row3-col3"]
    print(f"  Actual: {list(complete_panel['shapes'])} (count: {complete_panel['count']}, color: {complete_panel['color']})")

    print("\n📊 Analysis:")
    print(f"  Expected from pattern: {list(expected['shapes'])}")
    print(f"  Complete matrix has: {list(complete_panel['shapes'])}")
    print(f"  Match: {set(expected['shapes']) == set(complete_panel['shapes'])}")

if __name__ == "__main__":
    debug_union_rule()
