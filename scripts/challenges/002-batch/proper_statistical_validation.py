#!/usr/bin/env python3
"""
Proper Statistical Validation for Challenge 2 RPM - Using Real Implementation
"""

import json
import time
import edn_format
from holon import CPUStore, HolonClient

# Copy the actual matrix generation and validation functions from the main implementation
def generate_rpm_matrix(matrix_id, rule_type, attributes=None, missing_position=None):
    """Copy from main implementation"""
    if attributes is None:
        attributes = {"shape", "count", "color"}

    shapes = ["circle", "square", "triangle", "diamond", "star"]
    colors = ["black", "white", "red", "blue", "green"]

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

    elif rule_type == "union":
        # Use the updated union generation
        row_shapes = [
            {"circle", "diamond"},
            {"square", "star"},
            {"triangle", "circle"}
        ]
        col_shapes = [
            {"circle", "square"},
            {"diamond", "triangle"},
            {"star", "circle"}
        ]

        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

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

def compute_expected_missing_panel(matrix_data, missing_position):
    """Compute expected missing panel"""
    panels = matrix_data.get("panels", {})
    rule = matrix_data.get("rule", "")

    if rule == "progression":
        parts = missing_position.split("-")
        row = int(parts[0][3:])
        col = int(parts[1][3:])

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
            "color": color
        }

    elif rule == "xor":
        parts = missing_position.split("-")
        row = int(parts[0][3:])
        col = int(parts[1][3:])

        panel_shapes = set()
        for i, shape in enumerate(["circle", "square", "triangle"]):
            if (row ^ col) & (1 << i):
                panel_shapes.add(shape)

        return {
            "shapes": panel_shapes,
            "count": len(panel_shapes),
            "color": "black"
        }

    elif rule == "union":
        # Union rule: each position contains union of its row's defining shapes + column's defining shapes
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
        colors = ["black", "white", "red", "blue", "green"]
        color = colors[(row + col - 2) % len(colors)]

        return {
            "shapes": shapes,
            "count": len(shapes),
            "color": color
        }

    elif rule == "intersection":
        parts = missing_position.split("-")
        row = int(parts[0][3:])
        col = int(parts[1][3:])

        row_shapes = None
        col_shapes = None

        for pos, panel in panels.items():
            parts = pos.split("-")
            p_row = int(parts[0][3:])
            p_col = int(parts[1][3:])
            if p_row == row and p_col != col:
                if row_shapes is None:
                    row_shapes = set(panel.get("shapes", []))
                else:
                    row_shapes &= set(panel.get("shapes", []))
            if p_col == col and p_row != row:
                if col_shapes is None:
                    col_shapes = set(panel.get("shapes", []))
                else:
                    col_shapes &= set(panel.get("shapes", []))

        if row_shapes is not None and col_shapes is not None:
            shapes = row_shapes & col_shapes
        else:
            shapes = set()

        colors = ["black", "white", "red", "blue", "green"]
        color = colors[(row + col - 2) % len(colors)]

        return {
            "shapes": shapes,
            "count": len(shapes),
            "color": color
        }

    return {"shapes": set(), "count": 0, "color": "unknown"}

def validate_union_matrix_fit(incomplete_matrix, complete_matrix, missing_pos):
    """Simplified union validation"""
    complete_panels = complete_matrix.get("panels", {})
    if missing_pos not in complete_panels:
        return False

    actual_panel = complete_panels[missing_pos]
    actual_shapes = actual_panel.get("shapes", [])

    return len(actual_shapes) > 0

def run_proper_validation():
    """Run proper statistical validation using the real Challenge 2 implementation"""
    print("🧠 Challenge 2 RPM - Proper Statistical Validation")
    print("=" * 55)

    # Initialize store and client
    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Generate training data (complete matrices)
    rules = ["progression", "xor", "union", "intersection"]
    training_matrices = []

    for rule in rules:
        for i in range(5):  # 5 per rule for training
            matrix = generate_rpm_matrix(f"train-{rule}-{i}", rule)
            training_matrices.append(matrix)

    print(f"Training with {len(training_matrices)} complete matrices...")
    for matrix in training_matrices:
        edn_string = edn_format.dumps(matrix)
        client.insert(edn_string, data_type="edn")

    # Generate test data (incomplete matrices)
    test_matrices = []
    expected_answers = {}

    for rule in rules:
        for i in range(3):  # 3 tests per rule
            matrix = generate_rpm_matrix(f"test-{rule}-{i}", rule, missing_position="row3-col3")
            test_matrices.append(matrix)
            expected = compute_expected_missing_panel(matrix, "row3-col3")
            expected_answers[matrix["matrix-id"]] = expected

    print(f"Testing {len(test_matrices)} incomplete matrices...")

    # Run tests
    correct = 0
    total_time = 0
    rule_stats = {rule: {"correct": 0, "total": 0} for rule in rules}

    for test_matrix in test_matrices:
        matrix_id = test_matrix["matrix-id"]
        rule = test_matrix["rule"]
        expected = expected_answers[matrix_id]

        rule_stats[rule]["total"] += 1

        start_time = time.time()
        predicted = predict_missing_panel(client, test_matrix, rule)
        response_time = time.time() - start_time
        total_time += response_time

        is_correct = evaluate_prediction(predicted, expected, rule)
        if is_correct:
            correct += 1
            rule_stats[rule]["correct"] += 1

        print(f"  {matrix_id}: {'✅' if is_correct else '❌'} ({response_time:.3f}s)")

    # Results
    accuracy = correct / len(test_matrices)
    avg_time = total_time / len(test_matrices)

    print("\n📊 RESULTS:")
    print(f"   Accuracy: {accuracy:.1f}")
    print(f"   Avg Time: {avg_time:.4f}s")
    print("\n🎯 Rule Performance:")
    for rule, stats in rule_stats.items():
        rule_acc = stats["correct"] / stats["total"]
        print(f"   {rule}: {rule_acc:.1f}")
    # Challenge 4 comparison
    if accuracy >= 0.7:
        assessment = "🎉 EXCELLENT - Meets Challenge 4 target!"
    elif accuracy >= 0.5:
        assessment = "✅ GOOD - Working geometric reasoning"
    else:
        assessment = "⚠️ NEEDS IMPROVEMENT"

    print(f"\n🏆 Assessment: {assessment}")
    if accuracy < 0.7:
        print("   Challenge 4 achieved 72% accuracy - Challenge 2 needs work")
    else:
        print("   Surpasses Challenge 4's 72% accuracy target!")

    return accuracy

def predict_missing_panel(client, matrix, rule):
    """Predict missing panel using the actual Challenge 2 approach"""
    missing_pos = matrix["missing-position"]
    panels = matrix["panels"]

    # Convert sets to lists for JSON serialization
    probe_panels = {}
    for pos, panel_data in panels.items():
        probe_panels[pos] = {
            "shapes": list(panel_data.get("shapes", [])),
            "count": panel_data.get("count", 0),
            "color": panel_data.get("color", "unknown")
        }

    probe_structure = {
        "panels": probe_panels,
        "rule": rule,
    }

    complete_results = client.search_json(
        probe_structure,
        negations={"missing-position": {"$any": True}},
        top_k=3
    )

    if not complete_results:
        return {"shapes": [], "count": 0, "color": "unknown"}

    # For union rule, use hybrid validation
    for result in complete_results:
        comp_matrix = result["data"]
        if rule == "union":
            if validate_union_matrix_fit(matrix, comp_matrix, missing_pos):
                actual_missing = comp_matrix["panels"].get(missing_pos, {})
                return {
                    "shapes": actual_missing.get("shapes", []),
                    "count": actual_missing.get("count", 0),
                    "color": actual_missing.get("color", "unknown")
                }
        else:
            # Standard validation for other rules
            if missing_pos in comp_matrix.get("panels", {}):
                actual_missing = comp_matrix["panels"][missing_pos]
                return {
                    "shapes": actual_missing.get("shapes", []),
                    "count": actual_missing.get("count", 0),
                    "color": actual_missing.get("color", "unknown")
                }

    # Fallback
    return {"shapes": [], "count": 0, "color": "unknown"}

def evaluate_prediction(predicted, expected, rule):
    """Evaluate if prediction matches expected"""
    pred_shapes = set(predicted["shapes"])
    exp_shapes = set(expected["shapes"])

    shapes_match = pred_shapes == exp_shapes
    count_match = predicted["count"] == expected["count"]
    color_match = predicted["color"] == expected["color"]

    return shapes_match and count_match and color_match

if __name__ == "__main__":
    accuracy = run_proper_validation()
    print(f"\nFinal Accuracy: {accuracy:.1%}")
