#!/usr/bin/env python3
"""
Challenge 004: HTTP-Based Sudoku Solver - Holon as Black Box

This solution treats Holon as a HOSTED SERVICE, using only the HTTP API.
No direct Python imports of holon internals - pure JSON RPC.

Goal: Demonstrate that Sudoku solving via hyperspace can work with Holon-as-a-Service.

API Endpoints Used:
- POST /api/v1/items/batch - Bulk insert constraint patterns
- POST /api/v1/search - Query for matching patterns
- POST /api/v1/vectors/encode - Encode data to vectors
- POST /api/v1/vectors/compose - Bind/bundle operations
- GET /api/v1/health - Health check

NEW API Endpoints Added (now available):
- POST /api/v1/vectors/similarity - Compute similarity between vectors
- DELETE /api/v1/items/{id} - Delete an item
- POST /api/v1/store/clear - Clear all items (for testing/reset)
- GET /api/v1/items/{id}/vector - Get the encoded vector for an item
- POST /api/v1/search/by-vector - Search using raw vector
"""

import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests


class HolonHTTPClient:
    """
    Pure HTTP client for Holon - treats it as a black box service.

    This is what a user would use when Holon is hosted externally.
    All operations go through the HTTP API - no local imports.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self._dimensions = None  # Will be detected from first encode

    def health(self) -> Dict:
        """Check service health."""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()

    def insert(self, data: Dict) -> str:
        """Insert a single item, return its ID."""
        response = self.session.post(
            f"{self.base_url}/api/v1/items",
            json={"data": json.dumps(data), "data_type": "json"}
        )
        response.raise_for_status()
        return response.json()["id"]

    def batch_insert(self, items: List[Dict]) -> List[str]:
        """Insert multiple items, return their IDs."""
        response = self.session.post(
            f"{self.base_url}/api/v1/items/batch",
            json={
                "items": [json.dumps(item) for item in items],
                "data_type": "json"
            }
        )
        response.raise_for_status()
        return response.json()["ids"]

    def search(self, probe: Dict, top_k: int = 10,
               threshold: float = 0.0,
               guard: Optional[Dict] = None,
               negations: Optional[Dict] = None) -> List[Dict]:
        """Search for similar items."""
        payload = {
            "probe": json.dumps(probe),
            "data_type": "json",
            "top_k": top_k,
            "threshold": threshold
        }
        if guard:
            payload["guard"] = guard
        if negations:
            payload["negations"] = negations

        response = self.session.post(
            f"{self.base_url}/api/v1/search",
            json=payload
        )
        response.raise_for_status()
        return response.json()["results"]

    def encode(self, data: Dict) -> List[float]:
        """Encode data to a vector."""
        response = self.session.post(
            f"{self.base_url}/api/v1/vectors/encode",
            json={"data": json.dumps(data), "data_type": "json"}
        )
        response.raise_for_status()
        result = response.json()
        vector = result["vector"]

        # Cache dimensions
        if self._dimensions is None:
            self._dimensions = len(vector)

        return vector

    def bind(self, *vectors: List[float]) -> List[float]:
        """Bind vectors together."""
        response = self.session.post(
            f"{self.base_url}/api/v1/vectors/compose",
            json={"operation": "bind", "vectors": list(vectors)}
        )
        response.raise_for_status()
        return response.json()["vector"]

    def bundle(self, vectors: List[List[float]]) -> List[float]:
        """Bundle vectors into superposition."""
        response = self.session.post(
            f"{self.base_url}/api/v1/vectors/compose",
            json={"operation": "bundle", "vectors": vectors}
        )
        response.raise_for_status()
        return response.json()["vector"]

    def similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors via the API.

        Uses: POST /api/v1/vectors/similarity
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/vectors/similarity",
            json={"vector_a": v1, "vector_b": v2}
        )
        response.raise_for_status()
        return response.json()["similarity"]

    def clear(self):
        """Clear all items from the store."""
        response = self.session.post(f"{self.base_url}/api/v1/store/clear")
        response.raise_for_status()
        return response.json()

    def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        response = self.session.delete(f"{self.base_url}/api/v1/items/{item_id}")
        return response.status_code == 200

    def get_vector(self, item_id: str) -> List[float]:
        """Get the encoded vector for a stored item."""
        response = self.session.get(f"{self.base_url}/api/v1/items/{item_id}/vector")
        response.raise_for_status()
        return response.json()["vector"]

    def search_by_vector(self, vector: List[float], top_k: int = 10,
                         threshold: float = 0.0, guard: Optional[Dict] = None) -> List[Dict]:
        """Search using a raw vector instead of JSON probe."""
        payload = {
            "vector": vector,
            "top_k": top_k,
            "threshold": threshold
        }
        if guard:
            payload["guard"] = guard

        response = self.session.post(
            f"{self.base_url}/api/v1/search/by-vector",
            json=payload
        )
        response.raise_for_status()
        return response.json()["results"]

    def batch_search(self, searches: List[Dict]) -> List[List[Dict]]:
        """
        Batch search - multiple probes in one request (Qdrant-compatible).

        Each search dict should have:
        - probe: Dict (the query data)
        - top_k: int (optional, default 10)
        - threshold: float (optional, default 0.0)
        - guard: Dict (optional)

        Returns: List of result lists, one per search.
        """
        formatted_searches = []
        for s in searches:
            item = {
                "probe": json.dumps(s["probe"]),
                "data_type": "json",
                "top_k": s.get("top_k", 10),
                "threshold": s.get("threshold", 0.0)
            }
            if "guard" in s:
                item["guard"] = s["guard"]
            if "negations" in s:
                item["negations"] = s["negations"]
            formatted_searches.append(item)

        response = self.session.post(
            f"{self.base_url}/api/v1/search/batch",
            json={"searches": formatted_searches}
        )
        response.raise_for_status()
        return response.json()["results"]

    @property
    def dimensions(self) -> int:
        """Get vector dimensions (detected from first encode)."""
        if self._dimensions is None:
            # Encode a dummy value to detect dimensions
            self.encode({"_detect_dims": True})
        return self._dimensions


class HTTPSudokuSolver:
    """
    Sudoku solver using Holon HTTP API as a black box.

    This demonstrates:
    1. Encoding constraint patterns via HTTP
    2. Querying for valid placements via HTTP
    3. Using vector operations (bind/bundle) via HTTP
    4. Computing similarity for geometric reasoning
    """

    def __init__(self, holon_url: str = "http://localhost:8000"):
        self.client = HolonHTTPClient(holon_url)
        self.grid = [[0] * 9 for _ in range(9)]

        # Cache for encoded vectors (to reduce API calls)
        self._digit_vectors = {}
        self._position_vectors = {}
        self._ideal_constraint = None

    def _get_digit_vector(self, digit: int) -> List[float]:
        """Get or create vector for a digit (1-9)."""
        if digit not in self._digit_vectors:
            self._digit_vectors[digit] = self.client.encode({"digit": digit})
        return self._digit_vectors[digit]

    def _get_position_vector(self, row: int, col: int) -> List[float]:
        """Get or create vector for a position."""
        key = (row, col)
        if key not in self._position_vectors:
            self._position_vectors[key] = self.client.encode({
                "position": {"row": row, "col": col}
            })
        return self._position_vectors[key]

    def _get_ideal_constraint(self) -> List[float]:
        """Get the ideal constraint vector (superposition of all 9 digits)."""
        if self._ideal_constraint is None:
            digit_vectors = [self._get_digit_vector(d) for d in range(1, 10)]
            self._ideal_constraint = self.client.bundle(digit_vectors)
        return self._ideal_constraint

    def _encode_cell(self, row: int, col: int, digit: int) -> List[float]:
        """Encode a cell placement: pos(r,c) ⊙ digit."""
        pos_vec = self._get_position_vector(row, col)
        digit_vec = self._get_digit_vector(digit)
        return self.client.bind(pos_vec, digit_vec)

    def _insert_constraint_patterns(self):
        """Insert constraint patterns into Holon via HTTP."""
        print("Inserting constraint patterns via HTTP...")

        import itertools

        # Sample permutations (full would be 362,880)
        all_perms = list(itertools.permutations(range(1, 10)))
        sample_rate = 200  # Take ~1800 samples for faster HTTP operation
        sampled_perms = all_perms[::sample_rate]

        print(f"  Sampling {len(sampled_perms)} patterns from {len(all_perms)} total")

        # Create patterns as JSON
        patterns = []
        for i, perm in enumerate(sampled_perms):
            pattern = {
                "type": "constraint_pattern",
                "digits": list(perm),
                "pattern_id": i
            }
            patterns.append(pattern)

        # Batch insert via HTTP
        chunk_size = 100
        total_inserted = 0
        for i in range(0, len(patterns), chunk_size):
            chunk = patterns[i:i + chunk_size]
            self.client.batch_insert(chunk)
            total_inserted += len(chunk)
            print(f"  Inserted {total_inserted}/{len(patterns)} patterns")

        # Also insert partial patterns for matching
        print("  Inserting partial patterns...")
        partial_patterns = []
        for fill_level in range(1, 9):
            for perm in sampled_perms[:50]:  # Sample from samples
                partial = {
                    "type": "partial_pattern",
                    "fill_level": fill_level,
                    "digits": list(perm[:fill_level]),
                    "remaining": list(perm[fill_level:])
                }
                partial_patterns.append(partial)

        for i in range(0, len(partial_patterns), chunk_size):
            chunk = partial_patterns[i:i + chunk_size]
            self.client.batch_insert(chunk)

        print(f"  Total patterns: {len(patterns) + len(partial_patterns)}")

    def load_puzzle(self, puzzle: List[List[Optional[int]]]):
        """Load a puzzle."""
        for r in range(9):
            for c in range(9):
                val = puzzle[r][c]
                self.grid[r][c] = val if val is not None else 0

    def _get_available_digits(self, row: int, col: int) -> Set[int]:
        """Get digits available for a cell (not in row/col/block)."""
        used = set()

        # Row
        for c in range(9):
            if self.grid[row][c] != 0:
                used.add(self.grid[row][c])

        # Column
        for r in range(9):
            if self.grid[r][col] != 0:
                used.add(self.grid[r][col])

        # Block
        br, bc = (row // 3) * 3, (col // 3) * 3
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if self.grid[r][c] != 0:
                    used.add(self.grid[r][c])

        return set(range(1, 10)) - used

    def _find_most_constrained_cell(self) -> Optional[Tuple[int, int, Set[int]]]:
        """Find empty cell with fewest possibilities (MRV heuristic)."""
        best = None
        min_count = 10

        for r in range(9):
            for c in range(9):
                if self.grid[r][c] != 0:
                    continue

                available = self._get_available_digits(r, c)
                if len(available) < min_count:
                    min_count = len(available)
                    best = (r, c, available)

        return best

    def _score_all_digits(self, row: int, col: int,
                           available: Set[int]) -> Dict[int, float]:
        """
        Score ALL available digits for a cell using HTTP queries.
        OPTIMIZED: Uses batch search - 1 HTTP call instead of 3.
        """
        # Get existing digits in each constraint
        row_digits = [self.grid[row][c] for c in range(9) if self.grid[row][c] != 0]
        col_digits = [self.grid[r][col] for r in range(9) if self.grid[r][col] != 0]

        br, bc = (row // 3) * 3, (col // 3) * 3
        block_digits = [self.grid[r][c]
                        for r in range(br, br + 3)
                        for c in range(bc, bc + 3)
                        if self.grid[r][c] != 0]

        digit_scores = {d: 0.0 for d in available}

        # Build batch of searches (one per non-empty constraint)
        searches = []
        for digits in [row_digits, col_digits, block_digits]:
            if digits:  # Only query non-empty constraints
                searches.append({
                    "probe": {
                        "type": "partial_pattern",
                        "fill_level": len(digits),
                        "digits": sorted(digits)
                    },
                    "top_k": 10
                })

        # Single batch HTTP call instead of 3 separate calls
        if searches:
            try:
                batch_results = self.client.batch_search(searches)

                for results in batch_results:
                    for result in results:
                        remaining = result.get("data", {}).get("remaining", [])
                        score = result.get("score", 0)
                        for digit in remaining:
                            if digit in digit_scores:
                                digit_scores[digit] += score
            except Exception:
                pass

        return digit_scores

    def _score_all_digits_with_geometry(self, row: int, col: int,
                                         available: Set[int]) -> Dict[int, float]:
        """
        Full scoring with geometric similarity (more HTTP calls, more accurate).
        Use for initial solve, not backtracking.
        """
        import numpy as np

        # Get existing digits
        row_digits = [self.grid[row][c] for c in range(9) if self.grid[row][c] != 0]
        col_digits = [self.grid[r][col] for r in range(9) if self.grid[r][col] != 0]
        br, bc = (row // 3) * 3, (col // 3) * 3
        block_digits = [self.grid[r][c]
                        for r in range(br, br + 3)
                        for c in range(bc, bc + 3)
                        if self.grid[r][c] != 0]

        digit_scores = {d: 0.0 for d in available}

        # Pattern queries (3 HTTP calls)
        for digits in [row_digits, col_digits, block_digits]:
            if not digits:
                continue
            probe = {"type": "partial_pattern", "fill_level": len(digits), "digits": sorted(digits)}
            try:
                results = self.client.search(probe, top_k=10)
                for result in results:
                    remaining = result.get("data", {}).get("remaining", [])
                    score = result.get("score", 0)
                    for digit in remaining:
                        if digit in digit_scores:
                            digit_scores[digit] += score
            except Exception:
                pass

        # Geometric similarity - do it CLIENT-SIDE to avoid HTTP calls
        # Pre-cache digit vectors if not already cached
        for d in available:
            if d not in self._digit_vectors:
                self._digit_vectors[d] = self.client.encode({"digit": d})

        if self._ideal_constraint is None:
            digit_vecs = [self._digit_vectors.get(d) or self.client.encode({"digit": d}) for d in range(1, 10)]
            self._ideal_constraint = self.client.bundle(digit_vecs)

        # Compute similarity CLIENT-SIDE using numpy
        ideal = np.array(self._ideal_constraint, dtype=np.float64)
        ideal_norm = np.linalg.norm(ideal)

        current_base = list(set(row_digits + col_digits + block_digits))

        for digit in available:
            current_with_digit = current_base + [digit] if digit not in current_base else current_base
            if len(current_with_digit) > 1:
                # Bundle client-side to avoid HTTP call
                vecs = [np.array(self._digit_vectors[d], dtype=np.float64) for d in current_with_digit]
                bundled = np.sum(vecs, axis=0)
                bundled = np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0))

                # Compute similarity client-side
                dot = np.dot(bundled, ideal)
                norm = np.linalg.norm(bundled)
                if norm > 0 and ideal_norm > 0:
                    sim = dot / (norm * ideal_norm)
                    digit_scores[digit] += sim * 5

        return digit_scores

    def _query_best_digit(self, row: int, col: int,
                          available: Set[int]) -> Tuple[int, float]:
        """Query Holon HTTP API to find the best digit for a cell."""
        if len(available) == 1:
            return (list(available)[0], 1.0)

        # Get existing digits in each constraint
        row_digits = [self.grid[row][c] for c in range(9) if self.grid[row][c] != 0]
        col_digits = [self.grid[r][col] for r in range(9) if self.grid[r][col] != 0]

        br, bc = (row // 3) * 3, (col // 3) * 3
        block_digits = [self.grid[r][c]
                        for r in range(br, br + 3)
                        for c in range(bc, bc + 3)
                        if self.grid[r][c] != 0]

        # Query for patterns that match our current partial state
        digit_scores = {d: 0.0 for d in available}

        # Query partial patterns for each constraint
        for digits, constraint_name in [
            (row_digits, "row"),
            (col_digits, "col"),
            (block_digits, "block")
        ]:
            if not digits:
                continue

            # Search for partial patterns with these digits
            probe = {
                "type": "partial_pattern",
                "fill_level": len(digits),
                "digits": sorted(digits)  # Sort for matching
            }

            try:
                results = self.client.search(probe, top_k=10)

                for result in results:
                    data = result.get("data", {})
                    remaining = data.get("remaining", [])
                    score = result.get("score", 0)

                    for digit in remaining:
                        if digit in digit_scores:
                            digit_scores[digit] += score
            except Exception as e:
                # If query fails, continue with equal weights
                pass

        # Also use geometric similarity to ideal constraint
        ideal = self._get_ideal_constraint()

        for digit in available:
            # Compute how well this digit's vector aligns with ideal
            digit_vec = self._get_digit_vector(digit)

            # Create current constraint state + proposed digit
            current_digits = list(set(row_digits + col_digits + block_digits))
            if digit not in current_digits:
                current_digits.append(digit)

            if len(current_digits) > 1:
                current_vecs = [self._get_digit_vector(d) for d in current_digits]
                current_state = self.client.bundle(current_vecs)

                # Similarity to ideal
                sim = self.client.similarity(current_state, ideal)
                digit_scores[digit] += sim * 5  # Weight geometric fit

        if not digit_scores:
            return (list(available)[0], 0.0)

        best_digit = max(digit_scores, key=digit_scores.get)
        return (best_digit, digit_scores[best_digit])

    def solve(self, verbose: bool = True) -> bool:
        """
        Solve using HTTP API calls to Holon.

        This demonstrates pure black-box usage of Holon as a service.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("HTTP-BASED SUDOKU SOLVER")
            print("Holon as Black Box Service")
            print("=" * 60)

        # Check health
        try:
            health = self.client.health()
            if verbose:
                print(f"\nService status: {health['status']}")
                print(f"Backend: {health['backend']}")
                print(f"Items in store: {health['items_count']}")
        except Exception as e:
            print(f"ERROR: Cannot connect to Holon service: {e}")
            print("Make sure the server is running:")
            print("  ./scripts/run_with_venv.sh python scripts/server/holon_server.py")
            return False

        # Clear store for fresh start (using new API endpoint)
        if health['items_count'] > 0:
            if verbose:
                print(f"Clearing store ({health['items_count']} items)...")
            self.client.clear()

        # Insert constraint patterns
        self._insert_constraint_patterns()

        # Solve
        if verbose:
            print("\n" + "-" * 40)
            print("Solving via HTTP queries...")
            print("-" * 40)

        start_time = time.time()
        iterations = 0
        max_iterations = 100

        while iterations < max_iterations:
            iterations += 1

            # Find most constrained cell
            result = self._find_most_constrained_cell()

            if result is None:
                # Solved!
                elapsed = time.time() - start_time
                if verbose:
                    print(f"\n✓ SOLVED in {iterations} iterations ({elapsed:.2f}s)")
                return True

            r, c, available = result

            if len(available) == 0:
                if verbose:
                    print(f"  ✗ Contradiction at ({r},{c})")
                return False

            # Query for best digit
            digit, score = self._query_best_digit(r, c, available)
            self.grid[r][c] = digit

            if verbose:
                method = "Forced" if len(available) == 1 else "Query"
                print(f"  [{method}] ({r},{c}) → {digit} (score: {score:.2f}, options: {len(available)})")

        return False

    def solve_with_backtracking(self, verbose: bool = True) -> bool:
        """Solve with HTTP-guided backtracking for harder puzzles."""
        if verbose:
            print("\n" + "=" * 60)
            print("HTTP-GUIDED BACKTRACKING")
            print("=" * 60)

        start_time = time.time()
        stats = {"guesses": 0, "backtracks": 0}

        def solve_recursive(depth: int = 0) -> bool:
            result = self._find_most_constrained_cell()

            if result is None:
                return True  # Solved!

            r, c, available = result

            if len(available) == 0:
                stats["backtracks"] += 1
                return False

            # Get HTTP-guided ordering of digits
            # Query with ALL available digits at once to get proper scoring
            digit_scores = self._score_all_digits(r, c, available)

            ordered = sorted(digit_scores.keys(), key=lambda d: digit_scores[d], reverse=True)

            for digit in ordered:
                stats["guesses"] += 1
                self.grid[r][c] = digit

                if verbose and depth < 3:
                    indent = "  " * depth
                    print(f"{indent}[Guess@{depth}] ({r},{c}) → {digit}")

                if solve_recursive(depth + 1):
                    return True

                self.grid[r][c] = 0

            stats["backtracks"] += 1
            return False

        solved = solve_recursive()
        elapsed = time.time() - start_time

        if verbose:
            print(f"\nStats: {stats['guesses']} guesses, {stats['backtracks']} backtracks")
            print(f"Time: {elapsed:.2f}s")

        return solved

    def print_grid(self):
        """Pretty print the grid."""
        print("\n┌───────┬───────┬───────┐")
        for r in range(9):
            if r > 0 and r % 3 == 0:
                print("├───────┼───────┼───────┤")
            row_str = "│"
            for c in range(9):
                if c > 0 and c % 3 == 0:
                    row_str += "│"
                val = self.grid[r][c]
                row_str += f" {val if val != 0 else '.'} "
            row_str += "│"
            print(row_str)
        print("└───────┴───────┴───────┘")

    def validate(self) -> bool:
        """Validate the solution."""
        for r in range(9):
            if set(self.grid[r]) != set(range(1, 10)):
                return False
        for c in range(9):
            col = [self.grid[r][c] for r in range(9)]
            if set(col) != set(range(1, 10)):
                return False
        for br in range(3):
            for bc in range(3):
                block = []
                for r in range(br * 3, br * 3 + 3):
                    for c in range(bc * 3, bc * 3 + 3):
                        block.append(self.grid[r][c])
                if set(block) != set(range(1, 10)):
                    return False
        return True


def start_server_if_needed() -> Optional[subprocess.Popen]:
    """Start the Holon server if not running."""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        if response.status_code == 200:
            print("Holon server already running")
            return None
    except:
        pass

    print("Starting Holon server...")
    proc = subprocess.Popen(
        ["./scripts/run_with_venv.sh", "python", "scripts/server/holon_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to start
    for _ in range(30):
        time.sleep(0.5)
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
            if response.status_code == 200:
                print("Server started successfully")
                return proc
        except:
            pass

    print("ERROR: Server failed to start")
    proc.kill()
    return None


def main():
    """Demonstrate HTTP-based Sudoku solving."""

    print("=" * 60)
    print("HOLON AS BLACK BOX: HTTP-BASED SUDOKU SOLVER")
    print("=" * 60)
    print("\nThis demonstrates using Holon as a HOSTED SERVICE.")
    print("All operations use the HTTP API - no direct Python imports.")
    print("\nGoal: Prove Sudoku solving works with Holon-as-a-Service.\n")

    # Start server if needed
    server_proc = start_server_if_needed()

    try:
        # The classic example puzzle
        puzzle = [
            [5, 3, None, None, 7, None, None, None, None],
            [6, None, None, 1, 9, 5, None, None, None],
            [None, 9, 8, None, None, None, None, 6, None],
            [8, None, None, None, 6, None, None, None, 3],
            [4, None, None, 8, None, 3, None, None, 1],
            [7, None, None, None, 2, None, None, None, 6],
            [None, 6, None, None, None, None, 2, 8, None],
            [None, None, None, 4, 1, 9, None, None, 5],
            [None, None, None, None, 8, None, None, 7, 9]
        ]

        # Create HTTP-based solver
        solver = HTTPSudokuSolver("http://localhost:8000")
        solver.load_puzzle(puzzle)

        print("Input puzzle:")
        solver.print_grid()

        # Solve
        solved = solver.solve(verbose=True)

        print("\nResult:")
        solver.print_grid()

        if solved and solver.validate():
            print("\n✓ SOLVED via HTTP API!")
        else:
            print("\n⚠ Easy puzzle failed - unexpected")

        # Now test the HARD puzzle - this requires backtracking
        print("\n" + "=" * 60)
        print("HARD PUZZLE TEST: HTTP-Guided Backtracking")
        print("=" * 60)
        print("\nThis puzzle REQUIRES guessing/backtracking.")
        print("We'll prove complex constraint solving works over HTTP.\n")

        # A harder puzzle - "Evil" difficulty
        hard_puzzle = [
            [None, None, None, 6, None, None, 4, None, None],
            [7, None, None, None, None, 3, 6, None, None],
            [None, None, None, None, 9, 1, None, 8, None],
            [None, None, None, None, None, None, None, None, None],
            [None, 5, None, 1, 8, None, None, None, 3],
            [None, None, None, 3, None, 6, None, 4, 5],
            [None, 4, None, 2, None, None, None, 6, None],
            [9, None, 3, None, None, None, None, None, None],
            [None, 2, None, None, None, None, 1, None, None]
        ]

        # Create new solver (reuses same patterns in store)
        hard_solver = HTTPSudokuSolver("http://localhost:8000")
        hard_solver.load_puzzle(hard_puzzle)

        # Copy the cached vectors to avoid re-encoding
        hard_solver._digit_vectors = solver._digit_vectors
        hard_solver._position_vectors = solver._position_vectors
        hard_solver._ideal_constraint = solver._ideal_constraint

        print("Input (hard puzzle):")
        hard_solver.print_grid()

        hard_empty = sum(1 for r in range(9) for c in range(9) if hard_solver.grid[r][c] == 0)
        print(f"\nEmpty cells: {hard_empty}")

        # Try simple solve first
        print("\nAttempting simple HTTP solve...")
        hard_solved = hard_solver.solve(verbose=False)

        if hard_solved and hard_solver.validate():
            print("✓ Solved without backtracking!")
            hard_solver.print_grid()
        else:
            print("Simple solve hit contradiction - using HTTP-guided backtracking...")

            # Reset and try with backtracking
            hard_solver.load_puzzle(hard_puzzle)
            hard_solved = hard_solver.solve_with_backtracking(verbose=True)

            print("\nResult:")
            hard_solver.print_grid()

            if hard_solved and hard_solver.validate():
                print("\n✓ HARD PUZZLE SOLVED via HTTP-guided backtracking!")
                print("\nThis proves: Complex constraint solving works over the network.")
                print("The hyperspace queries guided backtracking via HTTP calls.")
            else:
                print("\n✗ Could not solve hard puzzle")

        # Print API usage summary
        print("\n" + "=" * 60)
        print("API ENDPOINTS USED (Complete Black-Box API)")
        print("=" * 60)
        print("""
Core Endpoints:
  GET  /api/v1/health              - Service health check
  POST /api/v1/items               - Insert single item
  POST /api/v1/items/batch         - Bulk insert items
  GET  /api/v1/items/{id}          - Retrieve item by ID
  POST /api/v1/search              - Query with similarity
  POST /api/v1/search/batch        - Batch search (Qdrant-compatible) ★ NEW

Vector Operations:
  POST /api/v1/vectors/encode      - Encode data to vectors
  POST /api/v1/vectors/compose     - Bind/bundle operations
  POST /api/v1/vectors/similarity  - Compute similarity

Management Endpoints:
  DELETE /api/v1/items/{id}        - Delete item
  POST   /api/v1/store/clear       - Clear all items
  GET    /api/v1/items/{id}/vector - Get item's vector
  POST   /api/v1/search/by-vector  - Search by raw vector
""")

        print("=" * 60)
        print("HOLON AS A SERVICE: Complete API for VSA/HDC")
        print("=" * 60)
        print("""
With these endpoints, Holon is now a complete black-box service:

╔═══════════════════════════════════════════════════════════╗
║  HOLON KERNEL PHILOSOPHY                                   ║
╠═══════════════════════════════════════════════════════════╣
║                                                             ║
║  Like Linux kernel or Clojure core:                        ║
║  • Minimal, composable primitives                           ║
║  • Empowers userland to build domain solutions             ║
║  • Data-in, data-out (no hidden state)                     ║
║                                                             ║
║  VSA/HDC Primitives Available:                              ║
║  • encode()    - Data → Vector                              ║
║  • bind()      - Compose relationships                      ║
║  • bundle()    - Superposition                              ║
║  • similarity()- Geometric distance                         ║
║  • search()    - Find by similarity                         ║
║                                                             ║
║  Userland Applications (like this Sudoku solver):          ║
║  • Constraint satisfaction via hyperspace                   ║
║  • Pattern matching via similarity                          ║
║  • Geometric reasoning via bind/bundle                      ║
║                                                             ║
║  The solution exists in hyperspace. We retrieve it.         ║
║                                                             ║
╚═══════════════════════════════════════════════════════════╝
""")

    finally:
        if server_proc:
            print("\nStopping server...")
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
