#!/usr/bin/env python3
"""
Code Structure Search Demo

Ingest Python code as AST structures with coordinates,
then search for patterns with wildcards.

This tests:
1. Deep nesting (AST can be very deep)
2. $any wildcards for partial matching
3. Returning coordinates (file, line, column)
4. Real-world structured data

Run: ./scripts/run_with_venv.sh python scripts/demos/code_structure_search.py
"""

import sys
import os
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from holon.cpu_store import CPUStore
from holon.client import HolonClient


@dataclass
class CodeLocation:
    """Coordinate for a code structure."""
    file: str
    line: int
    col: int
    end_line: Optional[int] = None
    end_col: Optional[int] = None
    path: Optional[str] = None  # AST path like "body[0].value.args[1]"


def ast_to_simple_structure(node: ast.AST) -> Dict[str, Any]:
    """Convert AST node to a FLAT searchable structure.

    Only includes immediate fields, not nested children.
    Context (class, function) is tracked separately in coordinates.
    """
    result = {
        "_type": node.__class__.__name__,
    }

    # Add location if available
    if hasattr(node, 'lineno'):
        result["_line"] = node.lineno
    if hasattr(node, 'col_offset'):
        result["_col"] = node.col_offset

    # Only include simple scalar fields, not nested AST nodes
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            # Just note the type, don't recurse
            result[f"{field}_type"] = value.__class__.__name__
        elif isinstance(value, list):
            # For lists, just store count and types
            if value and isinstance(value[0], ast.AST):
                result[f"{field}_count"] = len(value)
                result[f"{field}_types"] = list(set(v.__class__.__name__ for v in value if isinstance(v, ast.AST)))[:5]
            else:
                # Scalar list (like decorator names)
                result[field] = value[:5] if len(value) > 5 else value
        elif value is not None:
            # Scalar values
            if isinstance(value, (str, int, float, bool)):
                result[field] = value
            elif value is Ellipsis or value is ...:
                result[field] = "..."
            else:
                result[field] = str(value)

    return result


def extract_searchable_nodes(tree: ast.AST, filepath: str) -> List[Tuple[Dict, CodeLocation]]:
    """Extract all interesting nodes with their locations.

    Context (current class, function) is tracked as coordinate metadata,
    not as nested structure. This keeps each encoded item small and fast.
    """
    nodes = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None
            self.current_function = None

        def visit_ClassDef(self, node):
            # Record class definition (flat structure)
            structure = ast_to_simple_structure(node)
            loc = self._make_location(node, None, None)
            nodes.append((structure, loc))

            # Set context for children
            old_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old_class

        def visit_FunctionDef(self, node):
            structure = ast_to_simple_structure(node)
            loc = self._make_location(node, self.current_class, None)
            nodes.append((structure, loc))

            # Set context for children
            old_func = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = old_func

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)  # Same handling

        def _make_location(self, node, class_name, func_name):
            """Create location with context."""
            return CodeLocation(
                file=filepath,
                line=getattr(node, 'lineno', 0),
                col=getattr(node, 'col_offset', 0),
                end_line=getattr(node, 'end_lineno', None),
                end_col=getattr(node, 'end_col_offset', None),
                path=f"{class_name or ''}.{func_name or ''}".strip('.')
            )

        def generic_visit(self, node):
            # Only record "interesting" nodes
            interesting = (ast.Call, ast.For, ast.While, ast.If, ast.With,
                          ast.Try, ast.Import, ast.ImportFrom, ast.Assign,
                          ast.Return, ast.Raise, ast.Assert,
                          ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)

            if isinstance(node, interesting):
                structure = ast_to_simple_structure(node)
                # Add context to structure
                if self.current_class:
                    structure["_in_class"] = self.current_class
                if self.current_function:
                    structure["_in_function"] = self.current_function

                loc = self._make_location(node, self.current_class, self.current_function)
                nodes.append((structure, loc))

            # Continue visiting children
            for field, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    self.visit(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)

    Visitor().visit(tree)
    return nodes


def ingest_python_file(client: HolonClient, filepath: str) -> int:
    """Ingest a Python file's AST structures."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Skip {filepath}: {e}")
        return 0

    nodes = extract_searchable_nodes(tree, filepath)

    for structure, location in nodes:
        # Add location to the structure for storage
        structure["_location"] = asdict(location)
        client.insert_json(structure)

    return len(nodes)


def ingest_directory(client: HolonClient, directory: str) -> int:
    """Ingest all Python files in a directory (caller should manage bulk mode)."""
    total = 0
    for path in Path(directory).rglob("*.py"):
        count = ingest_python_file(client, str(path))
        if count > 0:
            print(f"  {path}: {count} nodes")
        total += count
    return total


def search_pattern(client: HolonClient, pattern: Dict, description: str, limit: int = 10):
    """Search for a pattern and display results with locations."""
    print(f"\n--- Search: {description} ---")
    print(f"Pattern: {json.dumps(pattern, indent=2)[:200]}...")

    results = client.search_json(probe=pattern, limit=limit)

    print(f"\nFound {len(results)} matches:")
    for i, r in enumerate(results[:10]):
        data = json.loads(r['data']) if isinstance(r['data'], str) else r['data']
        loc = data.get('_location', {})
        node_type = data.get('_type', 'unknown')

        file_short = loc.get('file', 'unknown')
        if len(file_short) > 40:
            file_short = "..." + file_short[-37:]

        print(f"  [{i+1}] {node_type:20} @ {file_short}:{loc.get('line', '?')}")

        # Show some context
        if node_type == 'FunctionDef':
            print(f"       def {data.get('name', '?')}(...)")
        elif node_type == 'ClassDef':
            print(f"       class {data.get('name', '?')}")
        elif node_type == 'Call':
            func = data.get('func', {})
            if isinstance(func, dict):
                if func.get('_type') == 'Name':
                    print(f"       {func.get('id', '?')}(...)")
                elif func.get('_type') == 'Attribute':
                    print(f"       ...{func.get('attr', '?')}(...)")
        elif node_type == 'Import':
            names = data.get('names', [])
            if names:
                print(f"       import {names[0].get('name', '?') if isinstance(names[0], dict) else names[0]}")


def demo_holon_codebase():
    """Demo searching the Holon codebase itself."""
    print("="*70)
    print("CODE STRUCTURE SEARCH DEMO")
    print("Indexing Holon's own codebase")
    print("="*70)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Ingest the holon library (with bulk mode for speed)
    print("\nIngesting holon/ directory...")
    holon_dir = Path(__file__).parent.parent.parent / "holon"
    store.start_bulk_insert()
    total = ingest_directory(client, str(holon_dir))
    store.end_bulk_insert()
    print(f"ANN index built in bulk mode")
    print(f"\nTotal: {total} AST nodes indexed")

    # Search 1: Find all function definitions
    search_pattern(client, {
        "_type": "FunctionDef"
    }, "All function definitions")

    # Search 2: Find functions with specific name pattern
    search_pattern(client, {
        "_type": "FunctionDef",
        "name": "insert"
    }, "Functions named 'insert'")

    # Search 3: Find all class definitions
    search_pattern(client, {
        "_type": "ClassDef"
    }, "All class definitions")

    # Search 4: Find calls to specific function (using $any for args)
    search_pattern(client, {
        "_type": "Call",
        "func": {
            "_type": "Attribute",
            "attr": "prototype"
        }
    }, "Calls to .prototype(...)")

    # Search 5: Find for loops
    search_pattern(client, {
        "_type": "For"
    }, "All for loops")

    # Search 6: Find imports
    search_pattern(client, {
        "_type": "ImportFrom"
    }, "All 'from X import Y' statements")

    # Search 7: Find list comprehensions
    search_pattern(client, {
        "_type": "ListComp"
    }, "All list comprehensions")

    # Search 8: Find try/except blocks
    search_pattern(client, {
        "_type": "Try"
    }, "All try/except blocks")


def demo_larger_codebase():
    """Demo with a larger codebase - the entire scripts directory."""
    print("\n" + "="*70)
    print("LARGER CODEBASE TEST")
    print("Indexing all scripts/ and tests/")
    print("="*70)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Ingest multiple directories with bulk mode
    base = Path(__file__).parent.parent.parent

    # Start bulk mode for all ingestion
    store.start_bulk_insert()

    print("\nIngesting scripts/...")
    total = ingest_directory(client, str(base / "scripts"))

    print("\nIngesting tests/...")
    total += ingest_directory(client, str(base / "tests"))

    print("\nIngesting holon/...")
    total += ingest_directory(client, str(base / "holon"))

    # End bulk mode and build ANN index
    print("\nBuilding ANN index...")
    store.end_bulk_insert()

    print(f"\nTotal: {total} AST nodes indexed")

    # More interesting searches
    search_pattern(client, {
        "_type": "FunctionDef",
        "name": "test"  # Fuzzy match for test functions
    }, "Functions with 'test' in name (fuzzy)")

    search_pattern(client, {
        "_type": "Call",
        "func": {
            "_type": "Attribute",
            "attr": "search_json"
        }
    }, "Calls to .search_json(...)")

    search_pattern(client, {
        "_type": "Assign"
    }, "All assignments (top 10)")

    # Pattern with wildcard - find any call with 2 arguments
    # (This tests the structure matching, not exact counts)
    search_pattern(client, {
        "_type": "Call",
        "func": {
            "_type": "Name"  # Direct function call, not method
        }
    }, "Direct function calls (not method calls)")


def main():
    demo_holon_codebase()
    demo_larger_codebase()

    print("\n" + "="*70)
    print("CODE STRUCTURE SEARCH: COMPLETE")
    print("="*70)
    print("""
This demonstrates:
1. Deep AST structure indexing
2. Pattern matching with partial structures
3. Location/coordinate tracking (file:line:col)
4. Real-world code search use case

Future enhancements:
- EDN/Clojure expression search
- $any wildcard in patterns (e.g., [a b $any d])
- Path-based filtering
- Cross-file relationship tracking
""")


if __name__ == "__main__":
    main()
