#!/usr/bin/env python3
"""
Approach 22: Hierarchical Encoding

Exploit Holon's recursive data encoding that we haven't been using.

Instead of flat encoding:
    grid_vec = bundle([bind(pos, digit) for all cells])

Use hierarchical structure:
    {
        "rows": {0: {c: digit, ...}, 1: {...}, ...},
        "cols": {0: {r: digit, ...}, ...},
        "blocks": {0: {idx: digit, ...}, ...},
        "grid": [[digits...], ...]
    }

This creates nested bindings:
    grid_vec = bind("rows", bundle([
        bind(0, bundle([bind(c, digit), ...])),
        bind(1, ...),
        ...
    ]))

The hierarchical structure should:
1. Separate row/col/block information for targeted queries
2. Allow unbinding at different levels
3. Preserve constraint unit relationships
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Dict, Tuple, Any
import numpy as np

from holon import HolonClient, CPUStore
from holon.encoder import Encoder, ListEncodeMode

from common import (
    bind,
    bundle,
    similarity,
    Timer,
    print_grid_9x9,
    validate_9x9,
    count_empty,
    get_available_digits_9x9,
    PUZZLE_9x9_HARD,
)


def create_store(dimensions: int = 16384):
    """Create Holon store."""
    store = CPUStore(dimensions=dimensions)
    return store


def grid_to_hierarchical(grid: List[List[Optional[int]]]) -> Dict[str, Any]:
    """
    Convert grid to hierarchical structure for Holon's recursive encoding.
    """
    # Row-centric view: row -> {col: digit}
    rows = {}
    for r in range(9):
        row_data = {}
        for c in range(9):
            if grid[r][c] is not None:
                row_data[f"c{c}"] = grid[r][c]
        if row_data:
            rows[f"r{r}"] = row_data

    # Column-centric view: col -> {row: digit}
    cols = {}
    for c in range(9):
        col_data = {}
        for r in range(9):
            if grid[r][c] is not None:
                col_data[f"r{r}"] = grid[r][c]
        if col_data:
            cols[f"c{c}"] = col_data

    # Block-centric view: block -> {idx: digit}
    blocks = {}
    for b in range(9):
        block_data = {}
        br, bc = (b // 3) * 3, (b % 3) * 3
        idx = 0
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if grid[r][c] is not None:
                    block_data[f"i{idx}"] = grid[r][c]
                idx += 1
        if block_data:
            blocks[f"b{b}"] = block_data

    return {
        "rows": rows,
        "cols": cols,
        "blocks": blocks,
    }


def encode_hierarchical(store: CPUStore, grid: List[List[Optional[int]]]) -> np.ndarray:
    """
    Encode grid using Holon's recursive data encoding.
    """
    hierarchical = grid_to_hierarchical(grid)
    return store.encoder.encode_data(hierarchical)


def encode_flat(store: CPUStore, grid: List[List[Optional[int]]]) -> np.ndarray:
    """
    Encode grid using flat pos-digit binding (our previous approach).
    """
    vm = store.vector_manager
    vectors = []

    for r in range(9):
        for c in range(9):
            if grid[r][c] is not None:
                pos_vec = vm.get_vector(f"pos_{r}_{c}")
                digit_vec = vm.get_vector(f"digit_{grid[r][c]}")
                vectors.append(pos_vec * digit_vec)

    if not vectors:
        return np.zeros(store.dimensions, dtype=np.int8)

    bundled = np.sum(vectors, axis=0)
    return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)


def compare_encodings():
    """Compare flat vs hierarchical encoding."""
    print("=" * 70)
    print("COMPARING FLAT VS HIERARCHICAL ENCODING")
    print("=" * 70)

    store = create_store()

    # Test puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    print(f"\nPuzzle: {81 - count_empty(grid)} filled cells")

    # Encode both ways
    flat_vec = encode_flat(store, grid)
    hier_vec = encode_hierarchical(store, grid)

    print(f"\nFlat encoding norm: {np.linalg.norm(flat_vec):.1f}")
    print(f"Hierarchical encoding norm: {np.linalg.norm(hier_vec):.1f}")

    # Similarity between the two
    sim = similarity(flat_vec, hier_vec)
    print(f"\nSimilarity between flat and hierarchical: {sim:.4f}")

    # Test: can we query row information from hierarchical?
    print("\n" + "-" * 50)
    print("QUERYING HIERARCHICAL ENCODING")
    print("-" * 50)

    # Get the "rows" key vector
    rows_key = store.vector_manager.get_vector("rows")

    # Unbind to get row information
    row_info = hier_vec * rows_key  # unbind

    print(f"\nUnbound 'rows' from hierarchical:")
    print(f"  Norm: {np.linalg.norm(row_info):.1f}")

    # Try to find which row vectors match
    for r in range(9):
        row_key = store.vector_manager.get_vector(f"r{r}")
        row_unbind = row_info * row_key
        # Check similarity to known digits
        best_sim = 0
        for d in range(1, 10):
            digit_vec = store.vector_manager.get_vector(str(d))
            s = similarity(row_unbind, digit_vec)
            if abs(s) > abs(best_sim):
                best_sim = s
        print(f"  Row {r}: best digit similarity = {best_sim:.4f}")


def test_constraint_queries():
    """Test querying constraint units from hierarchical encoding."""
    print("\n" + "=" * 70)
    print("CONSTRAINT UNIT QUERIES")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Encode
    hier_vec = encode_hierarchical(store, grid)

    # Build constraint unit prototypes
    # A "complete row" would have digits 1-9
    complete_row = np.zeros(store.dimensions, dtype=np.float32)
    for d in range(1, 10):
        complete_row += store.vector_manager.get_vector(str(d))
    complete_row = np.where(complete_row > 0, 1, np.where(complete_row < 0, -1, 0)).astype(np.int8)

    print(f"\n'Complete row' prototype norm: {np.linalg.norm(complete_row):.1f}")

    # Query each row's completion status
    rows_key = store.vector_manager.get_vector("rows")
    row_info = hier_vec * rows_key

    print("\nRow completion similarity to 'complete row' prototype:")
    for r in range(9):
        row_key = store.vector_manager.get_vector(f"r{r}")
        row_content = row_info * row_key

        # How similar is this row's content to a complete row?
        sim = similarity(row_content, complete_row)
        actual_count = sum(1 for c in range(9) if grid[r][c] is not None)
        print(f"  Row {r}: sim={sim:.4f}, filled={actual_count}/9")


def test_hierarchical_for_solving():
    """Test if hierarchical encoding helps with solving decisions."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL ENCODING FOR SOLVING DECISIONS")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find first decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) with options {options}")

                    # Encode current state
                    current_vec = encode_hierarchical(store, grid)

                    # For each option, encode the resulting state
                    print("\nHierarchical encoding after each choice:")
                    for digit in options:
                        test_grid = [[cell for cell in row] for row in grid]
                        test_grid[r][c] = digit

                        new_vec = encode_hierarchical(store, test_grid)

                        # How much did the encoding change?
                        delta = new_vec - current_vec
                        delta_norm = np.linalg.norm(delta)

                        # Similarity to current state
                        sim = similarity(current_vec, new_vec)

                        print(f"  Digit {digit}: delta_norm={delta_norm:.1f}, sim_to_current={sim:.4f}")

                    return


def test_row_col_block_separation():
    """
    Test if we can use hierarchical encoding to query row/col/block separately.
    """
    print("\n" + "=" * 70)
    print("ROW/COL/BLOCK SEPARATION")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Encode separately
    hierarchical = grid_to_hierarchical(grid)

    rows_vec = store.encoder.encode_data(hierarchical["rows"])
    cols_vec = store.encoder.encode_data(hierarchical["cols"])
    blocks_vec = store.encoder.encode_data(hierarchical["blocks"])

    print(f"\nSeparate encodings:")
    print(f"  Rows:   norm={np.linalg.norm(rows_vec):.1f}")
    print(f"  Cols:   norm={np.linalg.norm(cols_vec):.1f}")
    print(f"  Blocks: norm={np.linalg.norm(blocks_vec):.1f}")

    # Similarities between views
    print(f"\nCross-view similarities:")
    print(f"  Rows vs Cols:   {similarity(rows_vec, cols_vec):.4f}")
    print(f"  Rows vs Blocks: {similarity(rows_vec, blocks_vec):.4f}")
    print(f"  Cols vs Blocks: {similarity(cols_vec, blocks_vec):.4f}")

    # Test: does a choice affect views differently?
    print("\n" + "-" * 50)
    print("CHOICE IMPACT ON DIFFERENT VIEWS")
    print("-" * 50)

    # Find first decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) with options {options}")
                    print(f"In row {r}, col {c}, block {(r//3)*3 + c//3}")

                    for digit in options:
                        test_grid = [[cell for cell in row] for row in grid]
                        test_grid[r][c] = digit

                        new_hier = grid_to_hierarchical(test_grid)
                        new_rows = store.encoder.encode_data(new_hier["rows"])
                        new_cols = store.encoder.encode_data(new_hier["cols"])
                        new_blocks = store.encoder.encode_data(new_hier["blocks"])

                        row_delta = np.linalg.norm(new_rows - rows_vec)
                        col_delta = np.linalg.norm(new_cols - cols_vec)
                        block_delta = np.linalg.norm(new_blocks - blocks_vec)

                        print(f"\n  Digit {digit}:")
                        print(f"    Row delta:   {row_delta:.1f}")
                        print(f"    Col delta:   {col_delta:.1f}")
                        print(f"    Block delta: {block_delta:.1f}")

                    return


def test_nested_unbinding():
    """
    Test nested unbinding to query specific information.
    """
    print("\n" + "=" * 70)
    print("NESTED UNBINDING QUERIES")
    print("=" * 70)

    store = create_store()

    # Simple test case: known grid position
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find a filled cell
    test_r, test_c, test_d = None, None, None
    for r in range(9):
        for c in range(9):
            if grid[r][c] is not None:
                test_r, test_c, test_d = r, c, grid[r][c]
                break
        if test_r is not None:
            break

    print(f"\nTest cell: ({test_r},{test_c}) = {test_d}")

    # Encode hierarchically
    hier_vec = encode_hierarchical(store, grid)

    # Try to unbind: grid -> rows -> r{test_r} -> c{test_c} -> digit
    rows_key = store.vector_manager.get_vector("rows")
    row_key = store.vector_manager.get_vector(f"r{test_r}")
    col_key = store.vector_manager.get_vector(f"c{test_c}")

    # Step-by-step unbinding
    step1 = hier_vec * rows_key  # unbind "rows"
    step2 = step1 * row_key      # unbind specific row
    step3 = step2 * col_key      # unbind specific col

    print(f"\nUnbinding chain:")
    print(f"  After unbind 'rows': norm={np.linalg.norm(step1):.1f}")
    print(f"  After unbind 'r{test_r}': norm={np.linalg.norm(step2):.1f}")
    print(f"  After unbind 'c{test_c}': norm={np.linalg.norm(step3):.1f}")

    # Check similarity to each digit
    print(f"\nSimilarity to each digit (should be highest for {test_d}):")
    best_digit = None
    best_sim = -1
    for d in range(1, 10):
        digit_vec = store.vector_manager.get_vector(str(d))
        sim = similarity(step3, digit_vec)
        print(f"  Digit {d}: {sim:.4f}" + (" <-- expected" if d == test_d else ""))
        if sim > best_sim:
            best_sim = sim
            best_digit = d

    print(f"\nPredicted digit: {best_digit} (actual: {test_d})")
    print(f"Correct: {best_digit == test_d}")


def query_digit_at_position(store: CPUStore, hier_vec: np.ndarray, r: int, c: int) -> Tuple[int, float]:
    """
    Query the digit at position (r,c) using nested unbinding.
    Returns (predicted_digit, confidence).
    """
    rows_key = store.vector_manager.get_vector("rows")
    row_key = store.vector_manager.get_vector(f"r{r}")
    col_key = store.vector_manager.get_vector(f"c{c}")

    # Unbind chain
    step1 = hier_vec * rows_key
    step2 = step1 * row_key
    step3 = step2 * col_key

    # Find best matching digit
    best_digit = None
    best_sim = -1

    for d in range(1, 10):
        digit_vec = store.vector_manager.get_vector(str(d))
        sim = similarity(step3, digit_vec)
        if sim > best_sim:
            best_sim = sim
            best_digit = d

    return best_digit, best_sim


def test_holographic_recall():
    """
    Test holographic recall on all filled cells.
    Can we correctly recall digits from hierarchical encoding?
    """
    print("\n" + "=" * 70)
    print("HOLOGRAPHIC RECALL TEST")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Encode hierarchically
    hier_vec = encode_hierarchical(store, grid)

    correct = 0
    total = 0
    high_confidence = 0

    print("\nRecalling all filled cells:")
    for r in range(9):
        for c in range(9):
            if grid[r][c] is not None:
                actual = grid[r][c]
                predicted, confidence = query_digit_at_position(store, hier_vec, r, c)

                total += 1
                if predicted == actual:
                    correct += 1
                if confidence > 0.05:
                    high_confidence += 1

    print(f"\nRecall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"High confidence (>0.05): {high_confidence}/{total}")


def test_query_empty_cells():
    """
    What happens when we query empty cells?
    Does the encoding give us any signal about what SHOULD be there?
    """
    print("\n" + "=" * 70)
    print("QUERYING EMPTY CELLS")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Encode hierarchically
    hier_vec = encode_hierarchical(store, grid)

    print("\nQuerying first 5 empty cells:")
    count = 0

    for r in range(9):
        for c in range(9):
            if grid[r][c] is None and count < 5:
                options = list(get_available_digits_9x9(grid, r, c))

                # Query via unbinding
                predicted, confidence = query_digit_at_position(store, hier_vec, r, c)

                # Get all digit similarities
                rows_key = store.vector_manager.get_vector("rows")
                row_key = store.vector_manager.get_vector(f"r{r}")
                col_key = store.vector_manager.get_vector(f"c{c}")

                step1 = hier_vec * rows_key
                step2 = step1 * row_key
                step3 = step2 * col_key

                print(f"\nCell ({r},{c}): valid options = {options}")
                print(f"  Similarities:")
                for d in range(1, 10):
                    digit_vec = store.vector_manager.get_vector(str(d))
                    sim = similarity(step3, digit_vec)
                    marker = "*" if d in options else ""
                    print(f"    Digit {d}: {sim:+.4f} {marker}")

                count += 1


def test_constraint_completion_query():
    """
    Query: "What's missing from this row/col/block?"
    """
    print("\n" + "=" * 70)
    print("CONSTRAINT COMPLETION QUERY")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Build "has digit d" detectors for each digit
    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

    # Encode rows separately
    hierarchical = grid_to_hierarchical(grid)
    rows_vec = store.encoder.encode_data(hierarchical["rows"])

    print("\nRow-by-row digit presence detection:")
    for r in range(9):
        # Actual digits in row
        actual = set(grid[r][c] for c in range(9) if grid[r][c] is not None)
        missing = set(range(1, 10)) - actual

        # Unbind to get row content
        row_key = store.vector_manager.get_vector(f"r{r}")
        row_content = rows_vec * row_key

        # Check each digit
        detected = set()
        for d in range(1, 10):
            sim = similarity(row_content, digit_vecs[d])
            if sim > 0.02:  # Threshold for "present"
                detected.add(d)

        print(f"\n  Row {r}:")
        print(f"    Actual:   {sorted(actual)}")
        print(f"    Missing:  {sorted(missing)}")
        print(f"    Detected: {sorted(detected)}")
        print(f"    Correct detection: {detected == actual}")


def solve_with_hierarchical_queries():
    """
    Attempt to solve using hierarchical queries as guidance.
    """
    print("\n" + "=" * 70)
    print("SOLVING WITH HIERARCHICAL QUERIES")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Propagate forced moves first
    def propagate(g):
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if len(opts) == 1:
                            g[r][c] = opts[0]
                            changed = True
                        elif len(opts) == 0:
                            return False
        return True

    propagate(grid)
    print(f"After propagation: {count_empty(grid)} empty cells")

    # For each empty cell, query via hierarchical encoding
    print("\nQuerying empty cells with hierarchical encoding:")

    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) <= 3:  # Only for cells with few options
                    # Encode current state
                    hier_vec = encode_hierarchical(store, grid)

                    # Query what "fits" at this position
                    predicted, confidence = query_digit_at_position(store, hier_vec, r, c)

                    # Also score each valid option
                    rows_key = store.vector_manager.get_vector("rows")
                    row_key = store.vector_manager.get_vector(f"r{r}")
                    col_key = store.vector_manager.get_vector(f"c{c}")

                    step1 = hier_vec * rows_key
                    step2 = step1 * row_key
                    step3 = step2 * col_key

                    scores = {}
                    for d in options:
                        digit_vec = store.vector_manager.get_vector(str(d))
                        scores[d] = similarity(step3, digit_vec)

                    print(f"\n  Cell ({r},{c}): options={options}")
                    print(f"    Scores: {scores}")
                    if options:
                        best = max(options, key=lambda d: scores[d])
                        print(f"    Best: {best} (score={scores[best]:.4f})")


def test_duplicate_detection():
    """
    Can hierarchical encoding detect when we add a DUPLICATE digit?
    """
    print("\n" + "=" * 70)
    print("DUPLICATE DETECTION VIA HIERARCHICAL ENCODING")
    print("=" * 70)

    store = create_store()

    # Create a simple test: row with some digits
    test_grid = [[None]*9 for _ in range(9)]
    test_grid[0][0] = 5
    test_grid[0][1] = 3

    print("\nTest grid row 0: [5, 3, _, _, _, _, _, _, _]")

    # Encode
    hier_vec = encode_hierarchical(store, test_grid)

    # Now add different digits at (0,2)
    print("\nAdding digits at (0,2) and measuring encoding change:")

    for new_digit in [1, 3, 5, 7]:  # 3 and 5 would be duplicates
        test_copy = [[cell for cell in row] for row in test_grid]
        test_copy[0][2] = new_digit

        new_hier = encode_hierarchical(store, test_copy)

        # Measure change
        delta = np.linalg.norm(new_hier - hier_vec)
        sim = similarity(hier_vec, new_hier)

        is_dup = new_digit in [5, 3]
        print(f"  Digit {new_digit}: delta={delta:.1f}, sim={sim:.4f} {'<-- DUPLICATE' if is_dup else ''}")


def test_constraint_coherence():
    """
    Measure "coherence" of constraint units.

    Idea: A valid row should have 9 different digits.
    A row with duplicates is "less coherent" in some measurable way.
    """
    print("\n" + "=" * 70)
    print("CONSTRAINT COHERENCE MEASUREMENT")
    print("=" * 70)

    store = create_store()

    # Create test rows with varying coherence
    test_cases = [
        ("Valid: 1-9",       [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("One duplicate",    [1, 2, 3, 4, 5, 6, 7, 8, 8]),
        ("Two duplicates",   [1, 2, 3, 4, 5, 6, 7, 7, 7]),
        ("Many duplicates",  [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ]

    for name, row in test_cases:
        # Encode row as dict
        row_dict = {f"c{c}": row[c] for c in range(9)}
        row_vec = store.encoder.encode_data(row_dict)

        # Measure properties
        norm = np.linalg.norm(row_vec)

        # Count unique non-zero components (proxy for dimensionality)
        nonzero = np.sum(row_vec != 0)

        # Check similarity to "complete row" prototype
        complete = np.zeros(store.dimensions, dtype=np.float32)
        for d in range(1, 10):
            complete += store.vector_manager.get_vector(str(d))
        complete = np.where(complete > 0, 1, np.where(complete < 0, -1, 0)).astype(np.int8)

        sim_to_complete = similarity(row_vec, complete)

        print(f"\n{name}:")
        print(f"  Row: {row}")
        print(f"  Norm: {norm:.1f}")
        print(f"  Non-zero components: {nonzero}")
        print(f"  Similarity to complete: {sim_to_complete:.4f}")


def test_dimensionality_signal():
    """
    Test if effective dimensionality reveals duplicates.

    Hypothesis: Bundles with duplicates have LOWER effective dimensionality
    because repeated vectors don't add new orthogonal components.
    """
    print("\n" + "=" * 70)
    print("DIMENSIONALITY AS COHERENCE SIGNAL")
    print("=" * 70)

    store = create_store()

    def effective_dim(vec):
        """Estimate effective dimensionality."""
        nonzero = np.sum(vec != 0)
        # Could also use variance or other measures
        return nonzero

    # Test rows
    test_cases = [
        ("9 unique", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("8 unique", [1, 2, 3, 4, 5, 6, 7, 8, 8]),
        ("7 unique", [1, 2, 3, 4, 5, 6, 7, 7, 7]),
        ("6 unique", [1, 2, 3, 4, 5, 6, 6, 6, 6]),
        ("3 unique", [1, 1, 1, 2, 2, 2, 3, 3, 3]),
        ("1 unique", [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ]

    print("\nRow dimensionality vs uniqueness:")
    for name, row in test_cases:
        row_dict = {f"c{c}": row[c] for c in range(9)}
        row_vec = store.encoder.encode_data(row_dict)

        dim = effective_dim(row_vec)
        norm = np.linalg.norm(row_vec)
        unique_count = len(set(row))

        print(f"  {name:12s}: dim={dim:5d}, norm={norm:.1f}, unique={unique_count}")

    # Test: Can we detect when adding a digit REDUCES dimensionality?
    print("\n" + "-" * 50)
    print("INCREMENTAL DIMENSIONALITY TEST")
    print("-" * 50)

    # Start with partial row
    base_row = {f"c{c}": c+1 for c in range(5)}  # [1,2,3,4,5]
    base_vec = store.encoder.encode_data(base_row)
    base_dim = effective_dim(base_vec)

    print(f"\nBase row [1,2,3,4,5]: dim={base_dim}")

    # Add different digits at position 5
    print("\nAdding digit at c5:")
    for digit in range(1, 10):
        test_row = {f"c{c}": c+1 for c in range(5)}
        test_row["c5"] = digit

        test_vec = store.encoder.encode_data(test_row)
        test_dim = effective_dim(test_vec)
        delta_dim = test_dim - base_dim

        is_new = digit > 5
        marker = "NEW" if is_new else "DUP"
        print(f"  Digit {digit}: dim={test_dim}, delta={delta_dim:+d} [{marker}]")


def test_constraint_template_matching():
    """
    Match partial rows against complete row templates.

    Idea: Encode all valid complete rows (permutations of 1-9).
    For a partial row, find which completions are most similar.
    """
    print("\n" + "=" * 70)
    print("CONSTRAINT TEMPLATE MATCHING")
    print("=" * 70)

    store = create_store()

    # Build templates for "row contains digits {d}"
    # Instead of all 9! permutations, use "digit present" signatures
    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

    def encode_digit_set(digits: set) -> np.ndarray:
        """Encode a set of digits as their bundle."""
        if not digits:
            return np.zeros(store.dimensions, dtype=np.int8)
        vecs = [digit_vecs[d] for d in digits]
        bundled = np.sum(vecs, axis=0)
        return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

    # Test: partial row with some digits
    partial_digits = {1, 2, 3, 4, 5}
    partial_vec = encode_digit_set(partial_digits)

    print(f"\nPartial row has digits: {sorted(partial_digits)}")
    print(f"Missing: {sorted(set(range(1,10)) - partial_digits)}")

    # For each potential next digit, how much does it "complete" the row?
    print("\nSimilarity of partial + next_digit to complete set {1..9}:")

    complete_vec = encode_digit_set(set(range(1, 10)))

    for next_d in range(1, 10):
        # Add next_d to partial
        test_set = partial_digits | {next_d}
        test_vec = encode_digit_set(test_set)

        sim = similarity(test_vec, complete_vec)
        is_new = next_d not in partial_digits
        marker = "NEW" if is_new else "DUP"

        print(f"  +{next_d}: sim to complete = {sim:.4f} [{marker}]")


class TemplateMatchingSolver:
    """
    Solver using template matching at constraint level.

    Key insight: NEW digits bring us closer to complete template.
    Use this to ORDER choices at decision points.
    """

    def __init__(self, dimensions: int = 16384, verbose: bool = True):
        self.store = create_store(dimensions)
        self.verbose = verbose
        self.backtracks = 0

        # Pre-compute complete template
        digit_vecs = [self.store.vector_manager.get_vector(str(d)) for d in range(1, 10)]
        bundled = np.sum(digit_vecs, axis=0)
        self.complete_template = np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

        # Cache digit vectors
        self.digit_vecs = {d: self.store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

    def encode_digit_set(self, digits: set) -> np.ndarray:
        """Encode a set of digits."""
        if not digits:
            return np.zeros(self.store.dimensions, dtype=np.int8)
        vecs = [self.digit_vecs[d] for d in digits]
        bundled = np.sum(vecs, axis=0)
        return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

    def score_choice(self, grid: List[List[Optional[int]]], r: int, c: int, digit: int) -> float:
        """
        Score a choice by template matching.

        Measures how much closer we get to complete templates for all 3 constraint units.
        """
        # Get current digits in row/col/block
        row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
        col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}

        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = set()
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if grid[rr][cc] is not None:
                    block_digits.add(grid[rr][cc])

        # Check validity
        if digit in row_digits or digit in col_digits or digit in block_digits:
            return -1000

        # Score: similarity to complete template after adding digit
        new_row = self.encode_digit_set(row_digits | {digit})
        new_col = self.encode_digit_set(col_digits | {digit})
        new_block = self.encode_digit_set(block_digits | {digit})

        row_sim = similarity(new_row, self.complete_template)
        col_sim = similarity(new_col, self.complete_template)
        block_sim = similarity(new_block, self.complete_template)

        return row_sim + col_sim + block_sim

    def propagate(self, grid: List[List[Optional[int]]]) -> bool:
        """Propagate forced moves."""
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if not opts:
                            return False
                        if len(opts) == 1:
                            grid[r][c] = opts[0]
                            changed = True
        return True

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        grid = [[cell for cell in row] for row in puzzle]

        if not self.propagate(grid):
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, depth=0)

        if result:
            return True, result
        else:
            return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid: List[List[Optional[int]]], depth: int) -> Optional[List[List[int]]]:
        if count_empty(grid) == 0:
            return grid

        # Find cell with fewest options (MRV)
        best = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        # Score each option by template matching
        scored = []
        for digit in options:
            score = self.score_choice(grid, r, c, digit)
            if score > -500:  # Valid
                scored.append((score, digit))

        # Sort by score (highest first = closest to complete template)
        scored.sort(reverse=True)

        if self.verbose and depth < 3:
            print(f"  Depth {depth}: Cell ({r},{c}) options={options}")
            for score, digit in scored:
                print(f"    Digit {digit}: template_score={score:.4f}")

        for score, digit in scored:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            if not self.propagate(test_grid):
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result

            self.backtracks += 1

        return None


class UltimateHierarchicalSolver:
    """
    Combines:
    1. Template matching (from hierarchical encoding)
    2. Simulation rejection (fast negation)
    3. Chain length ordering (tie-breaker)
    """

    def __init__(self, dimensions: int = 16384, sim_depth: int = 10, verbose: bool = True):
        self.store = create_store(dimensions)
        self.sim_depth = sim_depth
        self.verbose = verbose
        self.backtracks = 0
        self.simulation_rejections = 0

        # Pre-compute complete template
        digit_vecs = [self.store.vector_manager.get_vector(str(d)) for d in range(1, 10)]
        bundled = np.sum(digit_vecs, axis=0)
        self.complete_template = np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

        self.digit_vecs = {d: self.store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

    def encode_digit_set(self, digits: set) -> np.ndarray:
        if not digits:
            return np.zeros(self.store.dimensions, dtype=np.int8)
        vecs = [self.digit_vecs[d] for d in digits]
        bundled = np.sum(vecs, axis=0)
        return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

    def template_score(self, grid, r, c, digit):
        """Score by template matching."""
        row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
        col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}

        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = set()
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if grid[rr][cc] is not None:
                    block_digits.add(grid[rr][cc])

        new_row = self.encode_digit_set(row_digits | {digit})
        new_col = self.encode_digit_set(col_digits | {digit})
        new_block = self.encode_digit_set(block_digits | {digit})

        row_sim = similarity(new_row, self.complete_template)
        col_sim = similarity(new_col, self.complete_template)
        block_sim = similarity(new_block, self.complete_template)

        return row_sim + col_sim + block_sim

    def propagate(self, grid):
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if not opts:
                            return False
                        if len(opts) == 1:
                            grid[r][c] = opts[0]
                            changed = True
        return True

    def simulate_survives(self, grid, row, col, digit, depth=10):
        """Check if choice survives simulation."""
        test_grid = [[cell for cell in r] for r in grid]
        test_grid[row][col] = digit

        forced_count = 0

        for _ in range(depth):
            if not self.propagate(test_grid):
                return False, forced_count

            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if test_grid[r][c] is None:
                        opts = list(get_available_digits_9x9(test_grid, r, c))
                        if not opts:
                            return False, forced_count
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                return True, forced_count

            if best_count == 1:
                r, c, opts = best
                test_grid[r][c] = opts[0]
                forced_count += 1
            else:
                break

        return True, forced_count

    def solve(self, puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        if not self.propagate(grid):
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, 0)
        if result:
            return True, result
        return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid, depth):
        if count_empty(grid) == 0:
            return grid

        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        # Score each option
        scored = []
        for digit in options:
            survives, chain_len = self.simulate_survives(grid, r, c, digit, self.sim_depth)
            if survives:
                template_sc = self.template_score(grid, r, c, digit)
                # Combined: template matching + chain length bonus
                combined = template_sc + chain_len * 0.01
                scored.append((combined, chain_len, digit))
            else:
                self.simulation_rejections += 1

        scored.sort(reverse=True)

        if not scored:
            scored = [(0, 0, d) for d in options]

        for combined, chain_len, digit in scored:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit
            if not self.propagate(test_grid):
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result
            self.backtracks += 1

        return None


def compare_with_template_solver():
    """Compare template matching solver with others."""
    print("\n" + "=" * 70)
    print("TEMPLATE MATCHING SOLVER COMPARISON")
    print("=" * 70)

    import time
    from approach_10_global import SimulationGuidedSolver
    from approach_19_opportunistic import HybridSolver

    solvers = [
        ("Standard (sim-guided)", SimulationGuidedSolver(verbose=False)),
        ("Hybrid (sim + chain)", HybridSolver(sim_depth=10, verbose=False)),
        ("Template Matching", TemplateMatchingSolver(verbose=False)),
        ("ULTIMATE (template + sim + chain)", UltimateHierarchicalSolver(verbose=False)),
    ]

    for name, solver in solvers:
        start = time.time()
        success, result = solver.solve(PUZZLE_9x9_HARD)
        elapsed = time.time() - start

        print(f"\n{name}:")
        print(f"  Solved: {success}")
        print(f"  Backtracks: {solver.backtracks}")
        if hasattr(solver, 'simulation_rejections'):
            print(f"  Sim rejections: {solver.simulation_rejections}")
        print(f"  Time: {elapsed:.3f}s")
        if success:
            print(f"  Valid: {validate_9x9(result)}")


def test_hierarchical_solver():
    """
    Build a solver that uses hierarchical encoding for constraint checking.
    """
    print("\n" + "=" * 70)
    print("HIERARCHICAL CONSTRAINT-BASED SOLVER")
    print("=" * 70)

    store = create_store()

    # Build digit vectors once
    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

    def encode_row(grid, r):
        """Encode just one row."""
        row_dict = {}
        for c in range(9):
            if grid[r][c] is not None:
                row_dict[f"c{c}"] = grid[r][c]
        return store.encoder.encode_data(row_dict) if row_dict else np.zeros(store.dimensions, dtype=np.int8)

    def check_row_has_digit(row_vec, digit):
        """Check if row already contains digit."""
        digit_vec = digit_vecs[digit]
        sim = similarity(row_vec, digit_vec)
        return sim > 0.05  # Threshold for "present"

    def hierarchical_valid(grid, r, c, digit):
        """Check if placing digit at (r,c) is valid using hierarchical encoding."""
        # Quick constraint check via encoding
        row_vec = encode_row(grid, r)

        # Does row already have this digit?
        if check_row_has_digit(row_vec, digit):
            return False

        # Could add column and block checks similarly
        return True

    # Test on puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find first empty cell
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))

                print(f"\nCell ({r},{c}): valid options = {options}")

                # Check each digit
                for d in range(1, 10):
                    std_valid = d in options
                    hier_valid = hierarchical_valid(grid, r, c, d)

                    match = "✓" if std_valid == hier_valid else "✗"
                    print(f"  Digit {d}: standard={std_valid}, hierarchical={hier_valid} {match}")

                return


def main():
    # Key tests
    test_nested_unbinding()
    test_holographic_recall()
    test_constraint_template_matching()
    test_dimensionality_signal()

    # Main comparison
    compare_with_template_solver()


if __name__ == "__main__":
    main()
