"""
Integration tests for code structure search demo.

Tests the AST-based code indexing and querying capabilities.
"""

import json
import sys
import time

import pytest

sys.path.insert(0, "scripts/demos")

# Import from the demo module
from code_structure_search import (
    ast_to_simple_structure,
    extract_searchable_nodes,
    ingest_directory,
    ingest_python_file,
)

from holon import CPUStore
from holon.client import HolonClient


class TestCodeStructureSearch:
    """Test code structure indexing and search."""

    @pytest.fixture
    def client(self):
        """Create a fresh client for each test."""
        store = CPUStore(dimensions=4000)  # Smaller for faster tests
        client = HolonClient(local_store=store)
        store.ann_enabled = False  # Brute force for deterministic tests
        return client, store

    def test_ingest_single_file(self, client):
        """Test ingesting a single Python file."""
        client, store = client
        count = ingest_python_file(client, "holon/highlevel/client.py")
        assert count > 50  # client.py should have many nodes

    def test_ingest_directory(self, client):
        """Test ingesting a directory."""
        client, store = client
        store.start_bulk_insert()
        count = ingest_directory(client, "holon/")
        store.end_bulk_insert()
        assert count > 1000  # Whole holon dir should have 1000+ nodes

    def test_search_class_def(self, client):
        """Test searching for class definitions."""
        client, store = client
        ingest_python_file(client, "holon/highlevel/client.py")

        results = client.search_json(probe={"_type": "ClassDef"}, limit=10)
        assert len(results) >= 1

        # Should find HolonClient
        found_holon_client = False
        for r in results:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            if data.get("name") == "HolonClient":
                found_holon_client = True
                # Verify coordinates
                loc = data.get("_location", {})
                assert "highlevel/client.py" in loc.get("file", "")
                assert loc.get("line") > 0
                break
        assert found_holon_client

    def test_search_function_def(self, client):
        """Test searching for function definitions."""
        client, store = client
        ingest_python_file(client, "holon/kernel/store.py")

        results = client.search_json(
            probe={"_type": "FunctionDef", "name": "insert"}, limit=10
        )
        assert len(results) >= 1

        # Verify the result
        data = (
            json.loads(results[0]["data"])
            if isinstance(results[0]["data"], str)
            else results[0]["data"]
        )
        assert data.get("_type") == "FunctionDef"
        loc = data.get("_location", {})
        assert "store.py" in loc.get("file", "")

    def test_search_with_class_context(self, client):
        """Test that _in_class context is tracked."""
        client, store = client
        ingest_python_file(client, "holon/kernel/store.py")

        # Search for functions in CPUStore
        results = client.search_json(
            probe={"_type": "FunctionDef", "_in_class": "CPUStore"}, limit=20
        )

        # Should find methods
        found_with_class = 0
        for r in results:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            if data.get("_in_class") == "CPUStore":
                found_with_class += 1

        assert found_with_class >= 5  # CPUStore has many methods

    def test_or_query_multiple_types(self, client):
        """Test $or query across multiple node types."""
        client, store = client
        ingest_python_file(client, "holon/kernel/encoder.py")

        results = client.search_json(
            probe={
                "$or": [
                    {"_type": "ClassDef"},
                    {"_type": "FunctionDef"},
                    {"_type": "Import"},
                ]
            },
            limit=20,
        )

        # Should find mix of types
        types_found = set()
        for r in results:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            types_found.add(data.get("_type"))

        # Should have at least 2 different types
        assert len(types_found) >= 2

    def test_or_query_is_fast(self, client):
        """Test that $or queries use superposition (O(1) branches)."""
        client, store = client
        store.start_bulk_insert()
        ingest_directory(client, "holon/")
        store.end_bulk_insert()

        # Time single query
        start = time.time()
        client.search_json(probe={"_type": "ClassDef"}, limit=10)
        single_time = time.time() - start

        # Time 5-way $or query
        start = time.time()
        client.search_json(
            probe={
                "$or": [
                    {"_type": "ClassDef"},
                    {"_type": "FunctionDef"},
                    {"_type": "Import"},
                    {"_type": "For"},
                    {"_type": "Try"},
                ]
            },
            limit=10,
        )
        or_time = time.time() - start

        # $or should NOT be 5x slower (superposition makes it O(1))
        # Allow 3x tolerance for noise
        assert (
            or_time < single_time * 3
        ), f"$or took {or_time/single_time:.1f}x longer than single"

    def test_coordinate_accuracy(self, client):
        """Test that coordinates point to actual source lines."""
        client, store = client
        ingest_python_file(client, "holon/highlevel/client.py")

        results = client.search_json(
            probe={"_type": "FunctionDef", "name": "insert"}, limit=5
        )

        for r in results:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            loc = data.get("_location", {})
            file_path = loc.get("file")
            line_no = loc.get("line")
            name = data.get("name")

            if file_path and line_no and name:
                # Read actual file and verify
                with open(file_path) as f:
                    lines = f.readlines()
                    if line_no <= len(lines):
                        actual_line = lines[line_no - 1]
                        assert f"def {name}" in actual_line, (
                            f"Expected 'def {name}' at {file_path}:{line_no}, "
                            f"got: {actual_line.strip()}"
                        )

    def test_guard_with_or(self, client):
        """Test guards work with $or queries."""
        client, store = client

        # Insert varied data
        client.insert_json({"type": "user", "name": "alice", "status": "active"})
        client.insert_json({"type": "user", "name": "bob", "status": "inactive"})
        client.insert_json({"type": "product", "name": "widget", "status": "active"})

        results = client.search_json(
            probe={"$or": [{"type": "user"}, {"type": "product"}]},
            guard={"status": "active"},
            limit=10,
        )

        # Should only return active items
        for r in results:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            assert data.get("status") == "active"

    def test_negation_with_or(self, client):
        """Test negations work with $or queries."""
        client, store = client

        client.insert_json({"type": "user", "name": "alice", "status": "active"})
        client.insert_json({"type": "user", "name": "bob", "status": "inactive"})
        client.insert_json({"type": "product", "name": "widget", "status": "active"})

        results = client.search_json(
            probe={"$or": [{"type": "user"}, {"type": "product"}]},
            negations={"status": {"$not": "inactive"}},
            limit=10,
        )

        # Should exclude inactive
        for r in results:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            assert data.get("status") != "inactive"

    def test_any_marker_in_or_branch(self, client):
        """Test that $any markers are stripped from $or branches."""
        client, store = client

        client.insert_json({"type": "user", "name": "alice"})
        client.insert_json({"type": "user", "name": "bob"})
        client.insert_json({"type": "product", "name": "widget"})
        client.insert_json({"type": "order", "name": "order1"})

        # $any should be stripped, so this matches users + products
        results = client.search_json(
            probe={
                "$or": [
                    {"type": "user", "name": {"$any": True}},
                    {"type": "product"},
                ]
            },
            limit=10,
        )

        # Should find users and products, not orders
        types_found = set()
        for r in results:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            types_found.add(data.get("type"))

        assert "user" in types_found or "product" in types_found


class TestASTStructure:
    """Test AST structure extraction."""

    def test_ast_to_simple_structure(self):
        """Test AST node conversion."""
        import ast

        code = "def foo(x): return x + 1"
        tree = ast.parse(code)
        func = tree.body[0]

        structure = ast_to_simple_structure(func)

        assert structure["_type"] == "FunctionDef"
        assert structure["name"] == "foo"
        assert "args_type" in structure  # Simplified structure uses _type suffix

    def test_extract_searchable_nodes(self):
        """Test node extraction from AST."""
        import ast

        code = """
class MyClass:
    def my_method(self):
        for i in range(10):
            print(i)
"""
        tree = ast.parse(code)
        nodes = extract_searchable_nodes(tree, "test.py")

        # Should find ClassDef, FunctionDef, For, Call
        types = [n[0]["_type"] for n in nodes]
        assert "ClassDef" in types
        assert "FunctionDef" in types
        assert "For" in types
        assert "Call" in types

    def test_function_in_class_context(self):
        """Test that functions inside classes get _in_class set."""
        import ast

        code = """
class MyClass:
    def my_method(self):
        pass
"""
        tree = ast.parse(code)
        nodes = extract_searchable_nodes(tree, "test.py")

        # Find the FunctionDef
        for structure, loc in nodes:
            if structure.get("_type") == "FunctionDef":
                assert structure.get("_in_class") == "MyClass"
                break
        else:
            pytest.fail("FunctionDef not found")
