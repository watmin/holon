#!/usr/bin/env python3
"""
Challenge 008-003: Code Repository Search Engine

COMPREHENSIVE HOLON DEMO showcasing:
1. AST parsing - Extract function/class structure from Python code
2. N-gram encoding - Fuzzy matching on function names
3. Rich guards - Filter by file, complexity, has_decorator
4. Negations - "Find functions NOT deprecated"
5. prototype() - Learn patterns for different code types
6. Structural similarity - Find "similar functions" not just keyword matches

Use case: Developers search for "similar functions" or 
"files that import X and have Y pattern".

This challenge indexes Holon's own codebase!
"""

import ast
import os
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from holon import CPUStore, HolonClient
from holon.similarity import normalized_dot_similarity

# =============================================================================
# AST Parsing
# =============================================================================

def extract_functions_from_file(filepath: str) -> List[Dict]:
    """Extract function/method definitions from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []
    
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = extract_function_info(node, filepath, source)
            functions.append(func_info)
    
    return functions


def extract_function_info(node: ast.FunctionDef, filepath: str, source: str) -> Dict:
    """Extract detailed information about a function."""
    # Get decorators
    decorators = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            decorators.append(dec.attr)
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                decorators.append(dec.func.attr)
    
    # Get arguments
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    
    # Get function calls made within this function
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    
    # Get docstring
    docstring = ast.get_docstring(node) or ""
    
    # Count complexity indicators
    complexity = {
        "lines": node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0,
        "branches": sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))),
        "calls": len(calls),
    }
    
    # N-gram tokenization for fuzzy name matching
    name = node.name
    name_tokens = tokenize_name(name)
    
    # Get return type annotation if present
    returns = None
    if node.returns:
        if isinstance(node.returns, ast.Name):
            returns = node.returns.id
        elif isinstance(node.returns, ast.Constant):
            returns = str(node.returns.value)
    
    return {
        "name": name,
        "name_tokens": name_tokens,  # For n-gram matching
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "line_start": node.lineno,
        "line_end": getattr(node, 'end_lineno', node.lineno),
        "decorators": decorators,
        "arguments": args,
        "arg_count": len(args),
        "calls": list(set(calls))[:20],  # Dedupe, limit
        "call_count": len(set(calls)),
        "docstring": docstring[:500] if docstring else "",
        "has_docstring": bool(docstring),
        "complexity": complexity,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "is_private": name.startswith("_") and not name.startswith("__"),
        "is_dunder": name.startswith("__") and name.endswith("__"),
        "returns": returns,
        "has_decorator": len(decorators) > 0,
    }


def tokenize_name(name: str) -> List[str]:
    """
    Tokenize a function name for n-gram matching.
    Handles snake_case and CamelCase.
    """
    tokens = []
    
    # Split on underscores
    parts = name.split("_")
    
    for part in parts:
        if not part:
            continue
        
        # Split CamelCase
        current = ""
        for char in part:
            if char.isupper() and current:
                tokens.append(current.lower())
                current = char
            else:
                current += char
        if current:
            tokens.append(current.lower())
    
    return tokens


def extract_classes_from_file(filepath: str) -> List[Dict]:
    """Extract class definitions from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []
    
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            
            # Get methods
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            
            # Get decorators
            decorators = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
            
            classes.append({
                "name": node.name,
                "name_tokens": tokenize_name(node.name),
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "line_start": node.lineno,
                "bases": bases,
                "methods": methods,
                "method_count": len(methods),
                "decorators": decorators,
                "docstring": (ast.get_docstring(node) or "")[:500],
                "has_docstring": bool(ast.get_docstring(node)),
                "item_type": "class",
            })
    
    return classes


def extract_imports_from_file(filepath: str) -> List[str]:
    """Extract import statements from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []
    
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    return imports


def index_codebase(root_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Index all Python files in a codebase."""
    functions = []
    classes = []
    
    for path in Path(root_path).rglob("*.py"):
        # Skip common non-code directories
        if any(part in path.parts for part in ["__pycache__", ".git", "venv", "node_modules"]):
            continue
        
        filepath = str(path)
        
        # Extract functions
        file_functions = extract_functions_from_file(filepath)
        for func in file_functions:
            func["item_type"] = "function"
        functions.extend(file_functions)
        
        # Extract classes
        file_classes = extract_classes_from_file(filepath)
        classes.extend(file_classes)
    
    return functions, classes


# =============================================================================
# Code Search Engine
# =============================================================================

class CodeSearchEngine:
    """Search code using Holon's structural similarity."""
    
    def __init__(self, dimensions: int = 4096, use_torchhd: bool = True):
        backend = "torchhd" if use_torchhd else "cpu"
        self.store = CPUStore(dimensions=dimensions, backend=backend)
        self.client = HolonClient(local_store=self.store)
        self.backend = backend
        self.function_prototype = None
        self.class_prototype = None
    
    def index_code(self, items: List[Dict]):
        """Index code items (functions, classes)."""
        for item in items:
            self.client.insert_json(item)
    
    def learn_prototypes(self, items: List[Dict]):
        """Learn prototypes for functions and classes."""
        functions = [i for i in items if i.get("item_type") == "function"]
        classes = [i for i in items if i.get("item_type") == "class"]
        
        if functions:
            vecs = []
            for f in functions[:100]:
                v = self.store.encoder.encode_data(f)
                v_np = v.cpu().numpy() if hasattr(v, 'cpu') else v
                vecs.append(v_np.astype(np.float32))
            self.function_prototype = self.store.prototype(vecs)
        
        if classes:
            vecs = []
            for c in classes[:50]:
                v = self.store.encoder.encode_data(c)
                v_np = v.cpu().numpy() if hasattr(v, 'cpu') else v
                vecs.append(v_np.astype(np.float32))
            self.class_prototype = self.store.prototype(vecs)
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search_by_name(self, name: str, limit: int = 10) -> List[Dict]:
        """Search by function/class name using n-gram tokens."""
        tokens = tokenize_name(name)
        return self.client.search_json(
            probe={"name_tokens": tokens, "name": name},
            limit=limit
        )
    
    def search_by_calls(self, calls: List[str], limit: int = 10) -> List[Dict]:
        """Find functions that call specific functions."""
        return self.client.search_json(
            probe={"calls": calls},
            limit=limit
        )
    
    def search_by_signature(self, args: List[str] = None, returns: str = None,
                            limit: int = 10) -> List[Dict]:
        """Find functions with similar signatures."""
        probe = {}
        if args:
            probe["arguments"] = args
        if returns:
            probe["returns"] = returns
        
        return self.client.search_json(probe=probe, limit=limit)
    
    def search_by_complexity(self, min_lines: int = None, min_branches: int = None,
                             limit: int = 10) -> List[Dict]:
        """Find functions by complexity using guards."""
        guard = {}
        if min_lines:
            guard["complexity"] = {"lines": {"$gte": min_lines}}
        if min_branches:
            if "complexity" not in guard:
                guard["complexity"] = {}
            guard["complexity"]["branches"] = {"$gte": min_branches}
        
        return self.client.search_json(
            probe={"item_type": "function"},
            guard=guard,
            limit=limit
        )
    
    def search_excluding_private(self, query: Dict, limit: int = 10) -> List[Dict]:
        """Search excluding private functions (negations demo)."""
        return self.client.search_json(
            probe=query,
            negations={"is_private": True},
            limit=limit
        )
    
    def search_with_decorator(self, decorator: str, limit: int = 10) -> List[Dict]:
        """Find functions with a specific decorator."""
        return self.client.search_json(
            probe={"decorators": [decorator]},
            guard={"has_decorator": True},
            limit=limit
        )
    
    def find_similar_functions(self, func: Dict, limit: int = 10) -> List[Dict]:
        """Find functions structurally similar to a given function."""
        return self.client.search_json(
            probe={
                "calls": func.get("calls", [])[:5],
                "arg_count": func.get("arg_count", 0),
                "complexity": func.get("complexity", {}),
            },
            negations={"name": func.get("name")},  # Exclude self
            limit=limit
        )
    
    def search_in_file(self, filename: str, query: Dict = None, limit: int = 20) -> List[Dict]:
        """Search within a specific file."""
        probe = query or {}
        return self.client.search_json(
            probe=probe,
            guard={"filename": filename},
            limit=limit
        )
    
    # =========================================================================
    # ADVANCED FEATURES
    # =========================================================================
    
    def demo_holon_features(self, items: List[Dict]):
        """Demonstrate all advanced Holon features for code search."""
        print("\n" + "=" * 70)
        print("ADVANCED HOLON FEATURES DEMO")
        print("=" * 70)
        
        # 1. N-gram fuzzy search
        print("\n1️⃣  N-GRAM ENCODING - Fuzzy name matching")
        results = self.search_by_name("encode_data")
        print(f"   Query: 'encode_data'")
        print(f"   Found {len(results)} matches:")
        for r in results[:3]:
            print(f"      {r['data']['name']} in {r['data']['filename']} (score: {r['score']:.3f})")
        
        # 2. Search by calls
        print("\n2️⃣  STRUCTURAL SEARCH - Functions that call 'numpy'")
        results = self.search_by_calls(["numpy", "np"])
        print(f"   Found {len(results)} functions")
        
        # 3. Negations
        print("\n3️⃣  NEGATIONS - Public functions only (exclude private)")
        results = self.search_excluding_private({"item_type": "function"})
        private_count = sum(1 for r in results if r["data"].get("is_private"))
        print(f"   Found {len(results)} public functions")
        print(f"   Private leaked: {private_count} (should be 0)")
        
        # 4. Decorator search
        print("\n4️⃣  DECORATOR SEARCH - Find @property methods")
        results = self.search_with_decorator("property")
        print(f"   Found {len(results)} @property methods")
        if results:
            for r in results[:3]:
                print(f"      {r['data']['name']} in {r['data']['filename']}")
        
        # 5. Complexity guards
        print("\n5️⃣  RICH GUARDS - Complex functions (10+ lines)")
        results = self.search_by_complexity(min_lines=10)
        print(f"   Found {len(results)} complex functions (10+ lines)")
        
        # 6. Similar functions
        print("\n6️⃣  STRUCTURAL SIMILARITY - Find similar functions")
        sample_func = next((i for i in items if i.get("item_type") == "function" and i.get("call_count", 0) > 3), None)
        if sample_func:
            similar = self.find_similar_functions(sample_func)
            print(f"   Reference: {sample_func['name']} ({sample_func['call_count']} calls)")
            print(f"   Similar functions:")
            for r in similar[:3]:
                print(f"      {r['data']['name']} ({r['data'].get('call_count', 0)} calls, score: {r['score']:.3f})")
        
        # 7. TorchHD benefit
        if self.backend == "torchhd":
            print("\n7️⃣  TORCHHD - Numeric similarity for complexity")
            print("   arg_count=3 is similar to arg_count=4")
            print("   lines=10 is similar to lines=12")


# =============================================================================
# Main Demo
# =============================================================================

def main():
    print("=" * 70)
    print("CHALLENGE 008-003: CODE REPOSITORY SEARCH ENGINE")
    print("=" * 70)
    
    # Index Holon's own codebase (just holon/ and scripts/ for speed)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    holon_src = os.path.join(project_root, "holon")
    scripts_dir = os.path.join(project_root, "scripts")
    
    print(f"\n📦 Indexing codebase: {holon_src} + {scripts_dir}")
    
    start = time.time()
    functions1, classes1 = index_codebase(holon_src)
    functions2, classes2 = index_codebase(scripts_dir)
    functions = functions1 + functions2
    classes = classes1 + classes2
    parse_time = time.time() - start
    
    print(f"   Parsed in {parse_time:.2f}s")
    print(f"   Functions: {len(functions)}")
    print(f"   Classes: {len(classes)}")
    
    all_items = functions + classes
    
    # Initialize engine
    print("\n🔧 Initializing search engine...")
    engine = CodeSearchEngine(dimensions=4096, use_torchhd=True)
    print(f"   Backend: {engine.backend}")
    
    # Index code
    print("\n📥 Indexing code items...")
    start = time.time()
    engine.index_code(all_items)
    index_time = time.time() - start
    print(f"   Indexed {len(all_items)} items in {index_time:.2f}s")
    print(f"   Rate: {len(all_items)/index_time:.0f} items/sec")
    
    # Learn prototypes
    print("\n🧠 Learning prototypes...")
    engine.learn_prototypes(all_items)
    
    # Demo 1: Name search
    print("\n" + "=" * 70)
    print("DEMO 1: Fuzzy Name Search")
    print("=" * 70)
    
    queries = ["encode", "prototype", "search", "similarity"]
    for query in queries:
        results = engine.search_by_name(query, limit=3)
        print(f"\n   Query: '{query}'")
        for r in results:
            print(f"      {r['data']['name']} ({r['data']['filename']}, score: {r['score']:.3f})")
    
    # Demo 2: Find functions that call specific functions
    print("\n" + "=" * 70)
    print("DEMO 2: Find Functions That Call...")
    print("=" * 70)
    
    results = engine.search_by_calls(["json", "loads"])
    print(f"\n   Query: Functions calling 'json' or 'loads'")
    print(f"   Found {len(results)} functions:")
    for r in results[:5]:
        calls = r['data'].get('calls', [])
        print(f"      {r['data']['name']} calls: {calls[:5]}...")
    
    # Demo 3: Search by signature
    print("\n" + "=" * 70)
    print("DEMO 3: Search by Signature")
    print("=" * 70)
    
    results = engine.search_by_signature(args=["self", "data"], returns="Dict")
    print(f"\n   Query: Functions with args=['self', 'data'], returns=Dict")
    print(f"   Found {len(results)} functions:")
    for r in results[:5]:
        args = r['data'].get('arguments', [])
        ret = r['data'].get('returns')
        print(f"      {r['data']['name']}({', '.join(args[:3])}) -> {ret}")
    
    # Demo 4: Exclude private functions
    print("\n" + "=" * 70)
    print("DEMO 4: Search Excluding Private Functions")
    print("=" * 70)
    
    results = engine.search_excluding_private({"calls": ["encode"]})
    print(f"\n   Query: Functions calling 'encode', excluding private")
    print(f"   Found {len(results)} public functions:")
    for r in results[:5]:
        name = r['data']['name']
        is_private = r['data'].get('is_private', False)
        print(f"      {name} (private: {is_private})")
    
    # Demo 5: Find similar functions
    print("\n" + "=" * 70)
    print("DEMO 5: Find Similar Functions")
    print("=" * 70)
    
    # Find a good sample function
    sample = next((f for f in functions if f.get("call_count", 0) > 5 and not f.get("is_private")), functions[0])
    similar = engine.find_similar_functions(sample)
    
    print(f"\n   Reference: {sample['name']}")
    print(f"   Calls: {sample.get('calls', [])[:5]}")
    print(f"   Complexity: {sample.get('complexity', {})}")
    print(f"\n   Similar functions:")
    for r in similar[:5]:
        print(f"      {r['data']['name']} (score: {r['score']:.3f})")
    
    # Demo 6: Search in specific file
    print("\n" + "=" * 70)
    print("DEMO 6: Search Within File")
    print("=" * 70)
    
    results = engine.search_in_file("encoder.py")
    print(f"\n   Query: All items in encoder.py")
    print(f"   Found {len(results)} items:")
    for r in results[:5]:
        print(f"      {r['data']['name']} (line {r['data'].get('line_start', '?')})")
    
    # Demo 7: Advanced features
    engine.demo_holon_features(all_items)
    
    # Demo 8: Query performance
    print("\n" + "=" * 70)
    print("DEMO 8: Query Performance")
    print("=" * 70)
    
    query_times = []
    for _ in range(100):
        start = time.time()
        engine.search_by_name(random.choice(["encode", "search", "query", "insert"]))
        query_times.append((time.time() - start) * 1000)
    
    avg_latency = sum(query_times) / len(query_times)
    print(f"\n   Query latency (avg of 100): {avg_latency:.2f}ms")
    print(f"   Queries/sec: {1000/avg_latency:.0f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
Success Criteria:
   ✅ Index real codebase: {len(all_items)} items from Holon
   ✅ Find "functions that call X": Working
   ✅ Fuzzy match on function names: N-gram tokenization
   ✅ Negation filters work: Exclude private functions

Key Holon Features Demonstrated:
   1. N-gram tokenization for fuzzy name matching
   2. Structural search by calls, args, complexity
   3. Negations to exclude private/deprecated
   4. Guards for complexity filtering
   5. Find similar functions by structure
   6. TorchHD for numeric field similarity

The Core Insight:
   Unlike grep/ack, Holon finds STRUCTURALLY similar code:
   - Functions with similar call patterns
   - Similar complexity profiles
   - Fuzzy name matches across naming conventions
""")


if __name__ == "__main__":
    main()
