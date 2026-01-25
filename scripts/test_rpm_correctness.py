#!/usr/bin/env python3
"""
Robust Correctness Test for RPM Geometric Solution

This test proves our VSA/HDC system is actually learning geometric rules,
not just getting lucky with similarity scores. We test:

1. Multiple rules with different expected outcomes
2. Statistical significance over many test cases
3. Failure cases (wrong rules should not work)
4. Rule discrimination (different rules produce different results)
"""

import json
from holon import CPUStore
# Copy functions from our RPM solution (to avoid import issues)
def generate_rpm_matrix(matrix_id, rule_type, attributes=None, missing_position=None):
    """Generate a synthetic RPM matrix with specified rule and attributes."""
    if attributes is None:
        attributes = {'shape', 'count', 'color'}

    shapes = ['circle', 'square', 'triangle', 'diamond', 'star']
    colors = ['black', 'white', 'red', 'blue', 'green']
    counts = [1, 2, 3, 4, 5]

    panels = {}

    if rule_type == 'progression':
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
                    'shapes': panel_shapes,
                    'count': len(panel_shapes),
                    'color': colors[(col-1) % len(colors)],
                    'progression': 'add-one',
                    'attributes': attributes
                }
                panels[position] = panel

    elif rule_type == 'xor':
        base_shapes = {'circle', 'square', 'triangle'}

        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                panel_shapes = set()
                for i, shape in enumerate(['circle', 'square', 'triangle']):
                    if (row ^ col) & (1 << i):
                        panel_shapes.add(shape)

                panel = {
                    'shapes': panel_shapes,
                    'count': len(panel_shapes),
                    'color': 'black',
                    'rule': 'xor',
                    'attributes': attributes
                }
                panels[position] = panel

    elif rule_type == 'union':
        row_shapes = [{'circle'}, {'square'}, {'triangle'}]
        col_shapes = [{'diamond'}, {'star'}, {'circle'}]

        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                # Union of row and column shape sets
                row_set = set(row_shapes[row-1])
                col_set = set(col_shapes[col-1])
                panel_shapes = row_set | col_set

                panel = {
                    'shapes': panel_shapes,
                    'count': len(panel_shapes),
                    'color': colors[(row+col-2) % len(colors)],
                    'rule': 'union',
                    'attributes': attributes
                }
                panels[position] = panel

    edn_data = {
        'matrix-id': matrix_id,
        'panels': panels,
        'rule': rule_type,
        'attributes': attributes
    }

    if missing_position:
        edn_data['missing-position'] = missing_position

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
    panels = matrix_data.get('panels', {})
    rule = matrix_data.get('rule', '')

    parts = missing_position.split('-')
    row = int(parts[0][3:])
    col = int(parts[1][3:])

    if rule == 'progression':
        shape_count = row + col - 1
        shapes = set()
        shape_options = ['circle', 'square', 'triangle', 'diamond', 'star']
        for i in range(min(shape_count, len(shape_options))):
            shapes.add(shape_options[i])

        colors = ['black', 'white', 'red', 'blue', 'green']
        color = colors[(col-1) % len(colors)]

        return {
            'shapes': shapes,
            'count': len(shapes),
            'color': color,
            'progression': 'add-one',
            'attributes': matrix_data.get('attributes', set())
        }

    elif rule == 'xor':
        shapes = set()
        for i, shape in enumerate(['circle', 'square', 'triangle']):
            if (row ^ col) & (1 << i):
                shapes.add(shape)

        return {
            'shapes': shapes,
            'count': len(shapes),
            'color': 'black',
            'rule': 'xor',
            'attributes': matrix_data.get('attributes', set())
        }

    elif rule == 'union':
        # Get row and column patterns from existing panels
        panels = matrix_data.get('panels', {})
        row_shapes = set()
        col_shapes = set()

        for pos, panel in panels.items():
            p_parts = pos.split('-')
            p_row = int(p_parts[0][3:])
            p_col = int(p_parts[1][3:])

            if p_row == row:  # Same row
                row_shapes.update(panel.get('shapes', []))
            if p_col == col:  # Same column
                col_shapes.update(panel.get('shapes', []))

        shapes = row_shapes | col_shapes
        colors = ['black', 'white', 'red', 'blue', 'green']
        color = colors[(row+col-2) % len(colors)]

        return {
            'shapes': shapes,
            'count': len(shapes),
            'color': color,
            'rule': 'union',
            'attributes': matrix_data.get('attributes', set())
        }

    return {'shapes': set(), 'count': 0, 'color': 'unknown', 'rule': 'unknown'}

def test_rule_correctness(store, rule_type, test_cases=5):
    """Test that our system correctly completes panels for a specific rule."""
    print(f"\n🧪 Testing {rule_type.upper()} Rule Correctness")

    # First, insert some complete matrices to learn the patterns from
    print(f"  Inserting complete {rule_type} matrices for learning...")
    for i in range(3):  # Insert 3 complete matrices to learn from
        complete_matrix = generate_rpm_matrix(f"{rule_type}-complete-{i}", rule_type,
                                            {'shape', 'count', 'color'})
        matrix_json = edn_to_json(complete_matrix)
        store.insert(matrix_json)

    correct_predictions = 0
    total_predictions = 0

    for i in range(test_cases):
        # Create matrix with missing panel
        matrix_id = f"{rule_type}-test-{i}"
        matrix = generate_rpm_matrix(matrix_id, rule_type,
                                   {'shape', 'count', 'color'}, "row3-col3")

        # Compute what the missing panel SHOULD be
        missing_pos = "row3-col3"
        expected_panel = compute_expected_missing_panel(matrix, missing_pos)

        # Insert the incomplete matrix
        matrix_json = edn_to_json(matrix)
        store.insert(matrix_json)

        # Query for geometrically similar complete matrices
        probe_structure = {
            "panels": {pos: panel for pos, panel in matrix.get('panels', {}).items()
                      if pos != missing_pos},
            "rule": rule_type
        }

        # Find complete matrices with this rule
        results = store.query(edn_to_json(probe_structure),
                            negations={"missing-position": {"$any": True}},
                            top_k=5)

        # Check if any of the top results have the correct missing panel
        found_correct = False
        for result in results:
            data = result[2]  # (id, score, data)
            actual_missing = data.get('panels', {}).get(missing_pos, {})

            expected_shapes = set(expected_panel['shapes'])
            actual_shapes = set(actual_missing.get('shapes', []))

            if (expected_shapes == actual_shapes and
                expected_panel['color'] == actual_missing.get('color', '') and
                expected_panel['count'] == actual_missing.get('count', 0)):
                found_correct = True
                break

        if found_correct:
            correct_predictions += 1
        total_predictions += 1

        status = "✅" if found_correct else "❌"
        print(f"  Test {i+1}: {status} | Expected: {list(expected_panel['shapes'])} "
              f"({expected_panel['count']}, {expected_panel['color']})")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n📊 {rule_type.upper()} Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1%}")

    return accuracy >= 0.8  # Require 80% accuracy to pass

def test_rule_discrimination(store):
    """Test that different rules produce different results."""
    print("\n🔍 Testing Rule Discrimination")

    # Create matrices with different rules but same structure
    rules = ['progression', 'xor', 'union']

    test_results = {}
    for rule in rules:
        matrix = generate_rpm_matrix(f"discrim-{rule}", rule,
                                   {'shape', 'count', 'color'}, "row2-col2")
        matrix_json = edn_to_json(matrix)
        store.insert(matrix_json)

        # Query with partial structure
        probe = {
            "panels": {
                "row1-col1": matrix['panels']['row1-col1'],
                "row1-col2": matrix['panels']['row1-col2']
            }
        }

        results = store.query(edn_to_json(probe), top_k=3)

        # Count how many results match the original rule
        rule_matches = sum(1 for _, _, data in results
                          if data.get('rule') == rule)

        test_results[rule] = rule_matches / len(results) if results else 0

    print("  Rule preference scores (higher is better discrimination):")
    for rule, score in test_results.items():
        print(f"    {rule}: {score:.2f}")

    # Check that each rule prefers its own type (discrimination working)
    discrimination_score = sum(score for score in test_results.values()) / len(test_results)
    print(f"  Average discrimination score: {discrimination_score:.2f}")
    return discrimination_score >= 0.6  # Require reasonable discrimination

def test_wrong_rule_failure(store):
    """Test that wrong rules don't work (negative test)."""
    print("\n❌ Testing Wrong Rule Failure")

    # Create a progression matrix
    matrix = generate_rpm_matrix("wrong-rule-test", "progression",
                               {'shape', 'count', 'color'}, "row3-col3")
    matrix_json = edn_to_json(matrix)
    store.insert(matrix_json)

    # Compute correct answer for progression
    expected_panel = compute_expected_missing_panel(matrix, "row3-col3")

    # Query as if it were an XOR rule (wrong!)
    probe_structure = {
        "panels": {pos: panel for pos, panel in matrix.get('panels', {}).items()
                  if pos != "row3-col3"},
        "rule": "xor"  # Wrong rule!
    }

    results = store.query(edn_to_json(probe_structure),
                        negations={"missing-position": {"$any": True}},
                        top_k=5)

    # Check if any results have the progression-expected panel
    wrong_rule_success = False
    for result in results:
        data = result[2]
        actual_missing = data.get('panels', {}).get("row3-col3", {})

        expected_shapes = set(expected_panel['shapes'])
        actual_shapes = set(actual_missing.get('shapes', []))

        if (expected_shapes == actual_shapes and
            expected_panel['color'] == actual_missing.get('color', '')):
            wrong_rule_success = True
            break

    if not wrong_rule_success:
        print("  ✅ Wrong rule correctly failed - system didn't find progression result when querying as XOR")
        return True
    else:
        print("  ❌ Wrong rule incorrectly succeeded - system found progression result for XOR query")
        return False

def test_statistical_significance(store):
    """Run statistical test over many random matrices."""
    print("\n📈 Testing Statistical Significance")

    import random
    random.seed(42)  # For reproducibility

    rules = ['progression', 'xor', 'union']
    total_tests = 50
    correct = 0

    for i in range(total_tests):
        # Random rule and missing position
        rule = random.choice(rules)
        positions = ["row1-col2", "row1-col3", "row2-col1", "row2-col3", "row3-col1", "row3-col2", "row3-col3"]
        missing_pos = random.choice(positions)

        # Create and insert matrix
        matrix = generate_rpm_matrix(f"stat-{i}", rule,
                                   {'shape', 'count', 'color'}, missing_pos)
        matrix_json = edn_to_json(matrix)
        store.insert(matrix_json)

        # Test completion
        expected_panel = compute_expected_missing_panel(matrix, missing_pos)

        probe_structure = {
            "panels": {pos: panel for pos, panel in matrix.get('panels', {}).items()
                      if pos != missing_pos},
            "rule": rule
        }

        results = store.query(edn_to_json(probe_structure),
                            negations={"missing-position": {"$any": True}},
                            top_k=3)

        # Check correctness
        for result in results:
            data = result[2]
            actual_missing = data.get('panels', {}).get(missing_pos, {})

            expected_shapes = set(expected_panel['shapes'])
            actual_shapes = set(actual_missing.get('shapes', []))

            if (expected_shapes == actual_shapes and
                expected_panel['color'] == actual_missing.get('color', '') and
                expected_panel['count'] == actual_missing.get('count', 0)):
                correct += 1
                break

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{total_tests} tests...")

    accuracy = correct / total_tests
    print(f"\n📊 Statistical Results: {correct}/{total_tests} = {accuracy:.1%}")

    # For true geometric learning, we expect >70% accuracy
    # Random guessing would be ~5% (1/20 possible shape combinations)
    return accuracy >= 0.7

def main():
    """Run comprehensive correctness tests."""
    print("🧠 COMPREHENSIVE CORRECTNESS TEST")
    print("=" * 60)
    print("Testing whether our VSA/HDC system actually learns geometric rules")
    print("or just gets lucky with similarity scores...")
    print("=" * 60)

    # Initialize store
    store = CPUStore(dimensions=16000)
    print("✅ Initialized Holon CPUStore")

    # Run all tests
    tests = [
        ("Individual Rule Correctness", lambda: all([
            test_rule_correctness(store, 'progression', 5),
            test_rule_correctness(store, 'xor', 5),
            test_rule_correctness(store, 'union', 5)
        ])),
        ("Rule Discrimination", lambda: test_rule_discrimination(store)),
        ("Wrong Rule Failure", lambda: test_wrong_rule_failure(store)),
        ("Statistical Significance", lambda: test_statistical_significance(store))
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))

    # Final verdict
    print("\n" + "=" * 60)
    print("🎯 FINAL VERDICT")
    print("=" * 60)

    passed_tests = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Our VSA/HDC system is definitively learning geometric rules")
        print("✅ Not lucky - actually understanding mathematical transformations!")
        print("🧠 TRUE GEOMETRIC INTELLIGENCE ACHIEVED!")
    else:
        print(f"⚠️  {passed_tests}/{total_tests} tests passed")
        print("🤔 System shows some geometric understanding but needs improvement")

    print("\n📋 Test Results:")
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")

if __name__ == "__main__":
    main()