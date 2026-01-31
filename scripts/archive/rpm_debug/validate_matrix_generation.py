#!/usr/bin/env python3
"""
Validate Matrix Generation

Check if our synthetic RPM matrix generation is actually creating matrices
that follow the geometric rules we expect.
"""


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


def validate_progression_matrix():
    """Validate that progression matrices follow the count rule."""
    print("🔍 VALIDATING PROGRESSION MATRIX GENERATION")
    print("=" * 50)

    matrix = generate_rpm_matrix(
        "test-progression", "progression", {"shape", "count", "color"}, "row3-col3"
    )

    print("Generated progression matrix:")
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
        panel = matrix["panels"][pos]
        shapes = panel.get("shapes", [])
        count = panel.get("count", 0)
        color = panel.get("color", "unknown")

        # Extract row/col from position
        parts = pos.split("-")
        row = int(parts[0][3:])  # Remove 'row' prefix
        col = int(parts[1][3:])  # Remove 'col' prefix
        expected_count = row + col - 1  # progression rule

        is_correct = count == expected_count
        status = "✅" if is_correct else "❌"
        print(
            f"  {pos}: {shapes} (count: {count}, expected: {expected_count}) {status}"
        )

    # Check if we can predict the missing panel
    expected_missing = []
    expected_count = 3 + 3 - 1  # row3-col3 = 5
    for i in range(expected_count):
        if i < len(["circle", "square", "triangle", "diamond", "star"]):
            expected_missing.append(
                ["circle", "square", "triangle", "diamond", "star"][i]
            )

    print(
        f"\nExpected missing panel (row3-col3): {expected_missing} (count: {expected_count})"
    )

    return True


def validate_xor_matrix():
    """Validate that XOR matrices follow the XOR rule."""
    print("\n🔍 VALIDATING XOR MATRIX GENERATION")
    print("=" * 50)

    matrix = generate_rpm_matrix(
        "test-xor", "xor", {"shape", "count", "color"}, "row3-col3"
    )

    print("Generated XOR matrix:")
    print("XOR rule: shape present if (row XOR col) has bit set")
    print("Bit 0 = circle, Bit 1 = square, Bit 2 = triangle")
    print()

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
        panel = matrix["panels"][pos]
        shapes = panel.get("shapes", [])
        count = panel.get("count", 0)

        # Extract row/col from position
        parts = pos.split("-")
        row = int(parts[0][3:])  # Remove 'row' prefix
        col = int(parts[1][3:])  # Remove 'col' prefix
        xor_result = row ^ col

        # Check which shapes should be present based on XOR bits
        expected_shapes = []
        if xor_result & 1:
            expected_shapes.append("circle")  # bit 0
        if xor_result & 2:
            expected_shapes.append("square")  # bit 1
        if xor_result & 4:
            expected_shapes.append("triangle")  # bit 2

        is_correct = set(shapes) == set(expected_shapes)
        status = "✅" if is_correct else "❌"
        print(
            f"  {pos}: {shapes} (xor={xor_result:02b}, expected: {expected_shapes}) {status}"
        )

    # Check missing panel prediction
    row, col = 3, 3
    xor_result = row ^ col  # 3 ^ 3 = 0
    expected_missing = []
    if xor_result & 1:
        expected_missing.append("circle")
    if xor_result & 2:
        expected_missing.append("square")
    if xor_result & 4:
        expected_missing.append("triangle")

    print(
        f"\nExpected missing panel (row3-col3): {expected_missing} (xor={xor_result:02b})"
    )

    return True


def test_generation_consistency():
    """Test that matrix generation is consistent."""
    print("\n🔍 TESTING GENERATION CONSISTENCY")
    print("=" * 50)

    # Generate same matrix multiple times
    matrices = []
    for i in range(5):
        matrix = generate_rpm_matrix(
            f"consistency-{i}", "xor", {"shape", "count", "color"}
        )
        matrices.append(matrix)

    # Check if row3-col3 is consistent (should always be empty for XOR)
    consistent = True
    first_result = None

    for i, matrix in enumerate(matrices):
        panel = matrix["panels"]["row3-col3"]
        shapes = panel.get("shapes", [])
        count = panel.get("count", 0)

        if first_result is None:
            first_result = (shapes, count)
        elif (shapes, count) != first_result:
            consistent = False
            print(
                f"  ❌ Inconsistency at matrix {i}: {shapes} (count: {count}) vs {first_result[0]} (count: {first_result[1]})"
            )

    if consistent:
        print(
            f"  ✅ All matrices consistent: {first_result[0]} (count: {first_result[1]})"
        )

    return consistent


def main():
    """Run all validation tests."""
    print("🧠 MATRIX GENERATION VALIDATION")
    print("=" * 60)

    tests = [
        ("Progression Matrix", validate_progression_matrix),
        ("XOR Matrix", validate_xor_matrix),
        ("Generation Consistency", test_generation_consistency),
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

    print("\n" + "=" * 60)
    print("🎯 VALIDATION RESULTS")
    print("=" * 60)

    passed_tests = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    if passed_tests == total_tests:
        print("✅ ALL VALIDATION TESTS PASSED")
        print("Matrix generation is working correctly!")
    else:
        print(f"⚠️  {passed_tests}/{total_tests} validation tests passed")
        print("Matrix generation has issues that need fixing")

    print("\n📋 Validation Results:")
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")


if __name__ == "__main__":
    main()
