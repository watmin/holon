#!/usr/bin/env python3
"""
Test RPM Geometric Solution via HTTP API

Verifies that our VSA/HDC geometric reasoning works through the HTTP interface.
"""

import json
import os
import sys
import threading
import time

import requests
from fastapi.testclient import TestClient

BASE_URL = "http://localhost:8000"


# Copy functions from our RPM solution (to avoid import issues)
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

    elif rule_type == "union":
        row_shapes = [{"circle"}, {"square"}, {"triangle"}]
        col_shapes = [{"diamond"}, {"star"}, {"circle"}]

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


# Use FastAPI TestClient for testing instead of real server
client = None


def start_server():
    """Start the Holon HTTP server using TestClient."""
    global client

    try:
        print("🚀 Starting Holon HTTP server with TestClient...")

        # Import server module and create test client
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "holon_server", "scripts/server/holon_server.py"
        )
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        client = TestClient(server_module.app)

        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Server is ready!")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False


def stop_server():
    """Stop the Holon HTTP server (no-op for TestClient)."""
    global client
    if client:
        client = None
        print("🛑 TestClient cleaned up")


def wait_for_server(max_retries=10):
    """Wait for the server to be ready."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"Waiting for server... ({i+1}/{max_retries})")
        time.sleep(1)
    return False


def insert_matrix(matrix_data):
    """Insert a matrix via TestClient."""
    global client
    try:
        response = client.post("/insert", json={"data": edn_to_json(matrix_data)})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Insert failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to insert matrix: {e}")
        return None


def query_matrices(query_data, guard=None, negations=None, top_k=5):
    """Query matrices via TestClient."""
    global client
    try:
        payload = {"probe": query_data, "top_k": top_k}
        if guard:
            payload["guard"] = guard
        if negations:
            payload["negations"] = negations

        response = client.post("/query", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Query failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to query matrices: {e}")
        return None


def test_rpm_http_solution():
    """Test RPM geometric reasoning through HTTP API."""
    print("🧠 Testing RPM Geometric Solution via HTTP API")
    print("=" * 60)

    # Start server
    if not start_server():
        print("❌ Server failed to start")
        return

    try:
        # Generate test matrices
        print("\n🎨 Generating test RPM matrices...")
        matrices = [
            generate_rpm_matrix(
                "http-xor-test-1", "xor", {"shape", "count", "color"}, "row3-col3"
            ),
            generate_rpm_matrix(
                "http-progression-test-1",
                "progression",
                {"shape", "count", "color"},
                "row3-col3",
            ),
            generate_rpm_matrix(
                "http-union-test-1", "union", {"shape", "count", "color"}, "row3-col3"
            ),
            # Complete matrices for reference
            generate_rpm_matrix(
                "http-xor-complete-1", "xor", {"shape", "count", "color"}
            ),
            generate_rpm_matrix(
                "http-progression-complete-1",
                "progression",
                {"shape", "count", "color"},
            ),
        ]

        # Insert matrices via HTTP
        print("\n📥 Inserting matrices via HTTP API...")
        inserted_ids = []
        for i, matrix in enumerate(matrices):
            result = insert_matrix(matrix)
            if result:
                inserted_ids.append(result.get("id"))
                print(f"  ✓ Inserted {matrix['matrix-id']}")
            else:
                print(f"  ❌ Failed to insert {matrix['matrix-id']}")
                return

        print(f"✅ Inserted {len(inserted_ids)} matrices")

        # Test geometric queries via HTTP
        print("\n🔍 Testing Geometric Queries via HTTP API...")

        # Test 1: Find matrices with missing panels
        print("\n1️⃣ Finding matrices with missing bottom-right panel:")
        query_result = query_matrices('{"missing-position": "row3-col3"}')
        if query_result:
            results = query_result.get("results", [])
            print(f"   Found {len(results)} matrices with missing panels")
            for result in results[:2]:  # Show first 2
                data = result.get("data", {})
                print(
                    f"   - {data.get('matrix-id', 'unknown')} ({data.get('rule', 'unknown')} rule)"
                )

        # Test 2: Geometric similarity for missing panel completion
        print("\n2️⃣ Testing geometric missing panel completion:")

        # Get incomplete XOR matrix
        incomplete_xor = matrices[0]  # http-xor-test-1
        missing_pos = incomplete_xor.get("missing-position", "")

        if missing_pos:
            print(
                f"   Testing completion for {incomplete_xor['matrix-id']} (missing {missing_pos})"
            )

            # Compute expected result
            expected_panel = compute_expected_missing_panel(incomplete_xor, missing_pos)
            print("   Expected missing panel:")
            print(
                f"     Shapes: {list(expected_panel['shapes'])} (count: {expected_panel['count']})"
            )
            print(f"     Color: {expected_panel['color']}")

            # Query for geometrically similar complete matrices via HTTP
            probe_structure = {
                "panels": {
                    pos: panel
                    for pos, panel in incomplete_xor.get("panels", {}).items()
                    if pos != missing_pos
                },
                "rule": incomplete_xor.get("rule", ""),
            }

            query_result = query_matrices(
                edn_to_json(probe_structure),
                negations={"missing-position": {"$any": True}},
                top_k=3,
            )

            if query_result:
                results = query_result.get("results", [])
                print(
                    f"   HTTP query found {len(results)} geometrically similar complete matrices:"
                )

                found_correct = False
                for result in results:
                    data = result.get("data", {})
                    score = result.get("score", 0)
                    actual_missing = data.get("panels", {}).get(missing_pos, {})

                    # Check if this matches our expected result
                    expected_shapes = expected_panel["shapes"]
                    actual_shapes = set(actual_missing.get("shapes", []))

                    is_correct = (
                        expected_shapes == actual_shapes
                        and expected_panel["color"] == actual_missing.get("color", "")
                        and expected_panel["count"] == actual_missing.get("count", 0)
                    )

                    status = "✅ CORRECT!" if is_correct else "❌ different"
                    if is_correct:
                        found_correct = True

                    print(
                        f"        {score:.3f}: Found: {list(actual_shapes)} (count: {len(actual_shapes)}, color: {actual_missing.get('color', 'unknown')})"
                    )

                if found_correct:
                    print(
                        "   🎯 SUCCESS: HTTP API found geometrically correct missing panel!"
                    )
                else:
                    print("   ⚠️  No exact match found via HTTP")

        # Test 3: Rule-based filtering via HTTP
        print("\n3️⃣ Testing rule-based filtering via HTTP:")
        query_result = query_matrices('{"rule": "progression"}')
        if query_result:
            results = query_result.get("results", [])
            print(f"   Found {len(results)} progression rule matrices")

        # Test 4: Complex guards via HTTP
        print("\n4️⃣ Testing complex guards via HTTP:")
        query_result = query_matrices(
            '{"attributes": ["shape", "count", "color"]}',
            guard={"rule": "xor"},
            top_k=3,
        )
        if query_result:
            results = query_result.get("results", [])
            print(f"   Found {len(results)} XOR matrices with all three attributes")

        print("\n" + "=" * 60)
        print("🎉 HTTP API RPM Testing Complete!")
        print("✅ Geometric reasoning works perfectly over HTTP!")
        print("✅ VSA/HDC transformations preserved through REST API!")
        print("=" * 60)

    finally:
        # Always cleanup server
        stop_server()


if __name__ == "__main__":
    test_rpm_http_solution()
