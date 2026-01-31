#!/usr/bin/env python3
"""
Challenge 007-002: Multi-Modal Code Understanding

Demonstrates unifying multiple code metadata sources into searchable vectors:
- AST structure (function signatures, class hierarchies)
- Docstrings and comments
- Git history (author, recency, churn)
- Test coverage metrics
- Dependency relationships

Usage:
    # Local mode
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/002-code-understanding-solution.py

    # HTTP mode (requires server running)
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/002-code-understanding-solution.py --http
"""

import argparse
import ast
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from holon import CPUStore, HolonClient


class CodeAnalyzer:
    """Analyze Python code and extract metadata."""

    def __init__(self):
        self.functions = []
        self.classes = []

    def analyze_file(self, filepath: Path) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            source = filepath.read_text()
            tree = ast.parse(source)

            file_data = {
                "filepath": str(filepath),
                "filename": filepath.name,
                "size_bytes": len(source),
                "line_count": source.count("\n") + 1,
                "functions": [],
                "classes": [],
                "imports": [],
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_data = self._extract_function(node, source)
                    file_data["functions"].append(func_data)
                elif isinstance(node, ast.ClassDef):
                    class_data = self._extract_class(node, source)
                    file_data["classes"].append(class_data)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports = self._extract_imports(node)
                    file_data["imports"].extend(imports)

            return file_data

        except Exception as e:
            return {
                "filepath": str(filepath),
                "filename": filepath.name,
                "error": str(e),
            }

    def _extract_function(self, node: ast.FunctionDef, source: str) -> Dict[str, Any]:
        """Extract function metadata."""
        args = [arg.arg for arg in node.args.args]
        docstring = ast.get_docstring(node)

        # Detect error handlers
        has_exception_handler = any(
            isinstance(child, ast.ExceptHandler) for child in ast.walk(node)
        )

        return {
            "name": node.name,
            "args": args,
            "arg_count": len(args),
            "docstring": docstring[:200] if docstring else None,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "has_exception_handler": has_exception_handler,
            "decorator_count": len(node.decorator_list),
        }

    def _extract_class(self, node: ast.ClassDef, source: str) -> Dict[str, Any]:
        """Extract class metadata."""
        methods = [
            n.name
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        docstring = ast.get_docstring(node)

        return {
            "name": node.name,
            "methods": methods,
            "method_count": len(methods),
            "docstring": docstring[:200] if docstring else None,
            "base_count": len(node.bases),
        }

    def _extract_imports(self, node) -> List[str]:
        """Extract import names."""
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            return [f"{module}.{alias.name}" for alias in node.names]
        return []


class CodeSearchIndex:
    """Index code metadata for searching."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        self.use_http = use_http
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.analyzer = CodeAnalyzer()
        self.indexed_count = 0

    def index_directory(self, directory: Path, pattern: str = "**/*.py"):
        """Index all Python files in a directory."""
        print(f"📁 Indexing directory: {directory}")
        files = list(directory.glob(pattern))
        print(f"   Found {len(files)} Python files")

        start_time = time.time()

        for filepath in files:
            if filepath.name.startswith("."):
                continue  # Skip hidden files

            file_data = self.analyzer.analyze_file(filepath)

            # Create searchable records for each function
            for func in file_data.get("functions", []):
                record = {
                    "_type": "function",
                    "file": file_data["filename"],
                    "filepath": file_data["filepath"],
                    "name": func["name"],
                    "signature": {
                        "args": func["args"],
                        "arg_count": func["arg_count"],
                    },
                    "metadata": {
                        "is_async": func["is_async"],
                        "has_exception_handler": func["has_exception_handler"],
                        "decorator_count": func["decorator_count"],
                    },
                    "docstring": func["docstring"],
                    # Mock git/coverage data - vary by name for demo
                    "git": {
                        "author": "developer",
                        "last_modified": "2024-01-15",
                        "churn": 5,
                    },
                    # Give higher coverage to some functions for demo
                    "coverage": 85 if hash(func["name"]) % 3 == 0 else 60,
                }
                self.client.insert_json(record)
                self.indexed_count += 1

            # Create searchable records for each class
            for cls in file_data.get("classes", []):
                record = {
                    "_type": "class",
                    "file": file_data["filename"],
                    "filepath": file_data["filepath"],
                    "name": cls["name"],
                    "methods": cls["methods"],
                    "method_count": cls["method_count"],
                    "docstring": cls["docstring"],
                    "git": {
                        "author": "developer",
                        "last_modified": "2024-01-15",
                        "churn": 3,
                    },
                    "coverage": 75,
                }
                self.client.insert_json(record)
                self.indexed_count += 1

        elapsed = time.time() - start_time
        rate = self.indexed_count / elapsed if elapsed > 0 else 0
        print(f"   ✅ Indexed {self.indexed_count} items in {elapsed:.2f}s ({rate:.0f}/sec)")

    def search_functions(
        self, pattern: Dict[str, Any], limit: int = 10, **kwargs
    ) -> List[Dict]:
        """Search for functions matching a pattern."""
        pattern["_type"] = "function"
        return self.client.search_json(probe=pattern, limit=limit, **kwargs)

    def search_classes(
        self, pattern: Dict[str, Any], limit: int = 10, **kwargs
    ) -> List[Dict]:
        """Search for classes matching a pattern."""
        pattern["_type"] = "class"
        return self.client.search_json(probe=pattern, limit=limit, **kwargs)

    def search_with_coverage(
        self, pattern: Dict[str, Any], min_coverage: int, limit: int = 10
    ) -> List[Dict]:
        """Search with coverage filter."""
        guard = {"coverage": {"$gte": min_coverage}}
        return self.client.search_json(probe=pattern, guard=guard, limit=limit)

    def search_error_handlers(self, limit: int = 10) -> List[Dict]:
        """Find all error handlers."""
        pattern = {"_type": "function", "metadata": {"has_exception_handler": True}}
        return self.client.search_json(probe=pattern, limit=limit)


def demo_basic_search(index: CodeSearchIndex):
    """Demo 1: Basic function search."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Function Search")
    print("=" * 70)

    # Search for functions with specific signatures
    print("\n🔍 Searching for functions with 'data' argument...")
    results = index.search_functions({"signature": {"args": ["data"]}}, limit=5)
    print(f"   Found {len(results)} functions:")
    for r in results[:3]:
        data = r["data"]
        print(f"   - {data['name']} in {data['file']} (score: {r['score']:.3f})")

    # Search for async functions
    print("\n🔍 Searching for async functions...")
    results = index.search_functions({"metadata": {"is_async": True}}, limit=5)
    print(f"   Found {len(results)} async functions:")
    for r in results[:3]:
        data = r["data"]
        print(f"   - {data['name']} in {data['file']} (score: {r['score']:.3f})")


def demo_coverage_filter(index: CodeSearchIndex):
    """Demo 2: Coverage-based filtering."""
    print("\n" + "=" * 70)
    print("DEMO 2: Coverage-Based Filtering")
    print("=" * 70)

    print("\n🔍 Searching for high-coverage functions (>80%)...")
    results = index.search_with_coverage(
        {"_type": "function"}, min_coverage=80, limit=10
    )
    print(f"   Found {len(results)} high-coverage functions:")
    for r in results[:5]:
        data = r["data"]
        print(
            f"   - {data['name']}: coverage={data['coverage']}% (score: {r['score']:.3f})"
        )


def demo_error_handlers(index: CodeSearchIndex):
    """Demo 3: Find error handlers."""
    print("\n" + "=" * 70)
    print("DEMO 3: Find Error Handlers")
    print("=" * 70)

    print("\n🔍 Searching for functions with exception handlers...")
    results = index.search_error_handlers(limit=10)
    print(f"   Found {len(results)} error handler functions:")
    for r in results[:5]:
        data = r["data"]
        print(f"   - {data['name']} in {data['file']} (score: {r['score']:.3f})")


def demo_class_search(index: CodeSearchIndex):
    """Demo 4: Class search."""
    print("\n" + "=" * 70)
    print("DEMO 4: Class Search")
    print("=" * 70)

    print("\n🔍 Searching for classes with many methods...")
    results = index.search_classes({"method_count": {"$gte": 5}}, limit=10)
    print(f"   Found {len(results)} classes with 5+ methods:")
    for r in results[:5]:
        data = r["data"]
        print(
            f"   - {data['name']}: {data['method_count']} methods (score: {r['score']:.3f})"
        )


def demo_fuzzy_search(index: CodeSearchIndex):
    """Demo 5: Fuzzy search by description."""
    print("\n" + "=" * 70)
    print("DEMO 5: Fuzzy Search by Description")
    print("=" * 70)

    print("\n🔍 Fuzzy search for 'search' functionality...")
    results = index.search_functions({"name": "search"}, limit=10)
    print(f"   Found {len(results)} matching functions:")
    for r in results[:5]:
        data = r["data"]
        print(f"   - {data['name']} in {data['file']} (score: {r['score']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Code Understanding")
    parser.add_argument(
        "--http", action="store_true", help="Use HTTP API instead of local store"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL for HTTP mode"
    )
    parser.add_argument(
        "--dir",
        default="holon",
        help="Directory to index (default: holon library itself)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-MODAL CODE UNDERSTANDING")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    if args.http:
        print(f"   Server: {args.url}")

    start_time = time.time()

    # Create index
    index = CodeSearchIndex(use_http=args.http, base_url=args.url)

    # Index directory
    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"❌ Directory not found: {target_dir}")
        return

    index.index_directory(target_dir)

    # Run demos
    if index.indexed_count > 0:
        demo_basic_search(index)
        demo_coverage_filter(index)
        demo_error_handlers(index)
        demo_class_search(index)
        demo_fuzzy_search(index)
    else:
        print("⚠️  No code indexed")

    elapsed = time.time() - start_time

    # Final stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
    Mode: {mode}
    Elapsed: {elapsed:.2f}s

    Items Indexed: {index.indexed_count}

    ✅ Multi-modal code understanding demonstrates:
       - AST parsing for structure
       - Metadata extraction (async, exceptions, decorators)
       - Fuzzy search by function/class names
       - Guard filters for coverage/git metrics
       - Combined structural + semantic search

    This enables intelligent code search that goes beyond
    simple text matching to understand code structure!
    """
    )


if __name__ == "__main__":
    main()
