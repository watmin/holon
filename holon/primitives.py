"""
VSA/HDC Algebraic Primitives

Core vector operations for hyperdimensional computing.
Mirrors holon-rs/src/primitives.rs for cross-language parity.

## Quick Reference

### Core Algebra (Binding & Bundling)
- `bind(A, B)` → Element-wise multiply (AND-like association)
- `unbind(AB, A)` → Retrieve B from bound vector (inverse of bind)
- `bundle(vectors)` → Sum + threshold (OR-like superposition)
- `negate(ABC, B)` → Remove B's influence from superposition
- `amplify(ABC, B, strength)` → Strengthen B's presence

### Pattern Extraction
- `prototype(vectors)` → Extract consensus pattern
- `prototype_add(proto, new, n)` → Incremental prototype update
- `resonance(vec, ref)` → Extract agreeing dimensions
- `difference(before, after)` → Compute delta/change
- `blend(A, B, alpha)` → Weighted interpolation
- `permute(vec, k)` → Circular shift for positional encoding
- `cleanup(noisy, codebook)` → Find closest match in codebook

### Extended Algebra
- `similarity_profile(A, B)` → Similarity as vector (not scalar)
- `attend(query, memory, strength)` → Soft attention / weighted resonance
- `analogy(A, B, C)` → A:B::C:? relational transfer
- `project(vec, subspace)` → Project onto exemplar subspace
- `conditional_bind(A, B, gate)` → Gated/conditional binding
- `segment(stream, window)` → Find structural breakpoints
- `invert(vec, codebook)` → Reconstruct components from vector
- `complexity(vec)` → Entropy/mixture measure (0.0 to 1.0)
"""

from typing import List, Tuple, Union

import numpy as np

from .distance import cosine_similarity


def threshold_bipolar(vector: np.ndarray) -> np.ndarray:
    """Threshold summed vector to bipolar {-1, 0, 1}."""
    return np.where(vector > 0, 1, np.where(vector < 0, -1, 0)).astype(np.int8)


# =============================================================================
# Core Algebra (Binding & Bundling)
# =============================================================================


def bind(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Bind two vectors using element-wise multiplication.

    Binding creates associations: bind(key, value) stores value under key.
    For bipolar vectors, this is self-inverse: bind(bind(A, B), A) ≈ B.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Bound vector (element-wise product)
    """
    return vec1 * vec2


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Unbind a key from a bound vector to retrieve the value.

    For bipolar vectors with element-wise multiplication binding,
    unbind is identical to bind (self-inverse property):
        unbind(bind(A, B), A) = B

    Args:
        bound: The bound vector (e.g., A ⊙ B)
        key: The key to remove (e.g., A)

    Returns:
        The unbound value (e.g., B)
    """
    return bind(bound, key)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Bundle multiple vectors by summing and thresholding.

    Creates a superposition that is similar to all inputs.
    The result is the "consensus" vector.

    Args:
        vectors: List of vectors to bundle

    Returns:
        Bundled vector (bipolar)
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list")
    bundled = np.sum(np.stack(vectors), axis=0)
    return threshold_bipolar(bundled)


def negate(
    superposition: np.ndarray, component: np.ndarray, method: str = "subtract"
) -> np.ndarray:
    """
    Remove a component's influence from a superposition (NOT operation).

    Args:
        superposition: The vector to remove from (e.g., bundle([A, B, C]))
        component: The vector to remove (e.g., B)
        method: "subtract" (default), "project", or "flip"

    Returns:
        Vector with component's influence removed
    """
    sup = superposition.astype(float)
    comp = component.astype(float)

    if method == "subtract":
        result = sup - comp
    elif method in ("project", "orthogonalize"):
        comp_norm = np.linalg.norm(comp)
        if comp_norm < 1e-10:
            return superposition
        comp_unit = comp / comp_norm
        projection = np.dot(sup, comp_unit) * comp_unit
        result = sup - projection
    elif method == "flip":
        result = sup.copy()
        mask = comp > 0
        result[mask] = -result[mask]
    else:
        raise ValueError(f"Unknown negation method: {method}")

    return threshold_bipolar(result)


def amplify(
    superposition: np.ndarray, component: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    """
    Strengthen a component's presence in a superposition.

    Args:
        superposition: The vector containing multiple components
        component: The component to amplify
        strength: How much to boost (1.0 = double, 2.0 = triple, etc.)

    Returns:
        Vector with component's influence strengthened
    """
    result = superposition.astype(float) + strength * component.astype(float)
    return threshold_bipolar(result)


# =============================================================================
# Pattern Extraction
# =============================================================================


def prototype(vectors: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Extract the common pattern from a set of vectors.

    Keeps only dimensions where a majority of vectors agree.

    Args:
        vectors: List of vectors to find consensus from
        threshold: Fraction of vectors that must agree (0.5 = majority)

    Returns:
        Vector representing the common pattern
    """
    if not vectors:
        raise ValueError("Cannot compute prototype of empty list")

    stacked = np.stack([v.astype(np.float32) for v in vectors])
    total = np.sum(stacked, axis=0)

    n = len(vectors)
    agreement_threshold = n * threshold

    result = np.zeros_like(total)
    result[total > agreement_threshold] = 1
    result[total < -agreement_threshold] = -1

    return result.astype(np.int8)


def prototype_add(proto: np.ndarray, example: np.ndarray, count: int) -> np.ndarray:
    """
    Incrementally update a prototype with a new example.

    Args:
        proto: Existing prototype vector
        example: New example to incorporate
        count: Number of examples already in prototype (before this one)

    Returns:
        Updated prototype incorporating the new example
    """
    weighted = proto.astype(np.float32) * count + example.astype(np.float32)
    averaged = weighted / (count + 1)
    return threshold_bipolar(averaged)


def resonance(vec: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Extract the part of vec that resonates with reference.

    Keeps only dimensions where both vectors agree (same sign).

    Args:
        vec: Vector to filter
        reference: Reference pattern to resonate with

    Returns:
        Vector containing only the resonating components
    """
    v = vec.astype(float)
    r = reference.astype(float)

    agree = (v * r) > 0
    result = np.zeros_like(v)
    result[agree] = v[agree]

    return threshold_bipolar(result)


def difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Compute what changed between two states.

    Args:
        before: The original state
        after: The new state

    Returns:
        Vector highlighting what was added (positive) or removed (negative)
    """
    delta = after.astype(float) - before.astype(float)
    return threshold_bipolar(delta)


def blend(vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Weighted interpolation between two vectors.

    Args:
        vec1: First vector (alpha=0 returns this)
        vec2: Second vector (alpha=1 returns this)
        alpha: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated vector
    """
    result = (1 - alpha) * vec1.astype(float) + alpha * vec2.astype(float)
    return threshold_bipolar(result)


def permute(vec: np.ndarray, k: int) -> np.ndarray:
    """
    Circular shift (permutation) of vector dimensions.

    Used for positional encoding in sequences.

    Args:
        vec: Input vector
        k: Shift amount (positive = right, negative = left)

    Returns:
        Shifted vector
    """
    return np.roll(vec, k)


def cleanup(noisy: np.ndarray, codebook: List[np.ndarray]) -> np.ndarray:
    """
    Find the closest vector in codebook to the noisy input.

    Args:
        noisy: Noisy or composed input vector
        codebook: List of clean/known vectors to match against

    Returns:
        The codebook vector with highest similarity to noisy
    """
    if not codebook:
        return noisy

    best_vec = codebook[0]
    best_sim = -float("inf")

    for vec in codebook:
        sim = cosine_similarity(noisy, vec)
        if sim > best_sim:
            best_sim = sim
            best_vec = vec

    return best_vec


# =============================================================================
# Extended Algebra
# =============================================================================


def similarity_profile(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """
    Return similarity as a VECTOR, not a scalar.

    Preserves dimension-wise agreement pattern.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Similarity profile vector (int8, bipolar)
    """
    return (vec_a * vec_b).astype(np.int8)


def attend(
    query: np.ndarray,
    memory: np.ndarray,
    strength: float = 1.0,
    mode: str = "soft",
) -> np.ndarray:
    """
    Weighted resonance - soft attention in VSA algebra.

    Modes:
        - "soft": Smooth weighting based on agreement
        - "hard": Binary resonance (same as resonance())
        - "amplify": Boost agreeing dimensions proportionally

    Args:
        query: The attention query vector
        memory: The memory to attend over
        strength: Attention strength multiplier
        mode: "soft", "hard", or "amplify"

    Returns:
        Attended vector with query-relevant parts emphasized
    """
    q = query.astype(np.float64)
    m = memory.astype(np.float64)

    if mode == "hard":
        agree = (q * m) > 0
        result = np.zeros_like(m)
        result[agree] = m[agree]

    elif mode == "soft":
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        m_norm = m / (np.linalg.norm(m) + 1e-10)
        agreement = q_norm * m_norm
        weights = (1 + np.tanh(strength * agreement)) / 2
        result = m * weights

    elif mode == "amplify":
        agree = (q * m) > 0
        result = m.copy()
        result[agree] = result[agree] * (1 + strength)

    else:
        raise ValueError(f"Unknown attention mode: {mode}")

    return threshold_bipolar(result)


def analogy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Relational transfer: A is to B as C is to ?

    Computes: C + difference(A, B)

    Args:
        a: Source concept
        b: Target of source relation
        c: New source to transfer to

    Returns:
        Predicted target
    """
    delta = difference(b, a)
    result = c.astype(np.float64) + delta.astype(np.float64)
    return threshold_bipolar(result)


def project(
    vec: np.ndarray,
    subspace: List[np.ndarray],
    orthogonalize: bool = True,
) -> np.ndarray:
    """
    Project vector onto subspace defined by exemplars.

    Args:
        vec: Vector to project
        subspace: List of exemplar vectors defining the subspace
        orthogonalize: Whether to orthogonalize subspace first (Gram-Schmidt)

    Returns:
        Projection of vec onto the subspace
    """
    if not subspace:
        return np.zeros(len(vec), dtype=np.int8)

    v = vec.astype(np.float64)
    basis = [u.astype(np.float64) for u in subspace]

    if orthogonalize and len(basis) > 1:
        ortho_basis = []
        for u in basis:
            for prev in ortho_basis:
                prev_norm = np.linalg.norm(prev)
                if prev_norm > 1e-10:
                    proj = np.dot(u, prev) / (prev_norm**2) * prev
                    u = u - proj
            if np.linalg.norm(u) > 1e-10:
                ortho_basis.append(u)
        basis = ortho_basis

    projection = np.zeros_like(v)
    for u in basis:
        norm_u = np.linalg.norm(u)
        if norm_u > 1e-10:
            coeff = np.dot(v, u) / (norm_u**2)
            projection += coeff * u

    return threshold_bipolar(projection)


def conditional_bind(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    gate: np.ndarray,
    mode: str = "positive",
) -> np.ndarray:
    """
    Bind only where condition is met (gated binding).

    Modes:
        - "positive": Bind where gate > 0
        - "negative": Bind where gate < 0
        - "nonzero": Bind where gate != 0
        - "strong": Bind where |gate| > threshold

    Args:
        vec_a: First vector to bind
        vec_b: Second vector to bind
        gate: Gate/condition vector
        mode: Gating mode

    Returns:
        Conditionally bound vector
    """
    a = vec_a.astype(np.float64)
    b = vec_b.astype(np.float64)
    g = gate.astype(np.float64)

    bound = a * b

    if mode == "positive":
        mask = g > 0
    elif mode == "negative":
        mask = g < 0
    elif mode == "nonzero":
        mask = g != 0
    elif mode == "strong":
        threshold_val = np.percentile(np.abs(g), 75)
        mask = np.abs(g) > threshold_val
    else:
        raise ValueError(f"Unknown gating mode: {mode}")

    result = np.zeros_like(bound)
    result[mask] = bound[mask]

    return threshold_bipolar(result)


def segment(
    stream: List[np.ndarray],
    window: int = 100,
    threshold: float = 0.3,
    method: str = "prototype",
    decay_factor: float = 0.9,
) -> List[int]:
    """
    Find structural breakpoints in a vector stream.

    Methods:
        - "prototype": Compare to running prototype (default)
        - "diff": Compare consecutive vectors
        - "accumulator": Compare to running accumulator

    Args:
        stream: List of vectors (chronological order)
        window: Lookback window for baseline computation
        threshold: Similarity drop to trigger segment (0.0-1.0)
        method: Segmentation method
        decay_factor: Decay factor for accumulator method

    Returns:
        List of indices where segments begin
    """
    if len(stream) < 2:
        return [0] if stream else []

    breakpoints = [0]

    if method == "prototype":
        baseline_vecs = [stream[0]]

        for i in range(1, len(stream)):
            start = max(0, i - window)
            baseline = prototype(stream[start:i])
            sim = cosine_similarity(stream[i], baseline)

            if sim < threshold:
                breakpoints.append(i)
                baseline_vecs = [stream[i]]
            else:
                baseline_vecs.append(stream[i])
                if len(baseline_vecs) > window:
                    baseline_vecs.pop(0)

    elif method == "diff":
        for i in range(1, len(stream)):
            sim = cosine_similarity(stream[i], stream[i - 1])
            if sim < threshold:
                breakpoints.append(i)

    elif method == "accumulator":
        # Import here to avoid circular dependency
        from .accumulator import (
            accumulate,
            create_accumulator,
            decay,
            normalize_accumulator,
        )

        accum = create_accumulator(len(stream[0]))
        accum = accumulate(accum, stream[0])

        for i in range(1, len(stream)):
            baseline = normalize_accumulator(accum)
            sim = cosine_similarity(stream[i], baseline)

            if sim < threshold:
                breakpoints.append(i)
                accum = create_accumulator(len(stream[0]))

            accum = accumulate(accum, stream[i])

            if i % window == 0:
                accum = decay(accum, factor=decay_factor)

    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    return breakpoints


def complexity(vec: np.ndarray) -> float:
    """
    Measure the "complexity" or "mixedness" of a vector.

    Low complexity = clean signal (single concept)
    High complexity = superposition of many things

    Returns a value between 0.0 (minimal) and 1.0 (maximal).

    Args:
        vec: Vector to measure

    Returns:
        Complexity score (0.0 to 1.0)
    """
    v = vec.astype(np.float64)

    nnz = np.sum(vec != 0)
    density = nnz / len(vec)

    pos = np.sum(vec > 0)
    neg = np.sum(vec < 0)
    total_active = pos + neg
    if total_active == 0:
        balance = 0.0
    else:
        ratio = min(pos, neg) / total_active
        balance = ratio * 2

    if vec.dtype in [np.float32, np.float64]:
        abs_v = np.abs(v)
        total = np.sum(abs_v)
        if total > 1e-10:
            probs = abs_v / total
            probs = probs[probs > 1e-10]
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(v))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0.0
    else:
        normalized_entropy = density * balance

    complexity_score = 0.4 * density + 0.3 * balance + 0.3 * normalized_entropy
    return float(np.clip(complexity_score, 0.0, 1.0))


def invert(
    vec: np.ndarray,
    codebook: List[Union[Tuple[str, np.ndarray], np.ndarray]] = None,
    top_k: int = 5,
    threshold: float = 0.3,
) -> List[Tuple[str, Union[float, dict]]]:
    """
    Reconstruct representative structure from a vector.

    If codebook is provided, returns matching components.
    Otherwise, returns a structural analysis.

    Args:
        vec: Vector to invert/analyze
        codebook: Optional list of (name, vector) tuples or just vectors
        top_k: Number of top matches to return
        threshold: Minimum similarity threshold

    Returns:
        List of (component, similarity) tuples, sorted by similarity
    """
    if codebook is None:
        v = vec.astype(np.float64)
        analysis = {
            "complexity": complexity(vec),
            "density": float(np.sum(vec != 0) / len(vec)),
            "pos_ratio": float(np.sum(vec > 0) / max(np.sum(vec != 0), 1)),
            "magnitude": float(np.linalg.norm(v)),
        }
        return [("_analysis", analysis)]

    results = []
    for entry in codebook:
        if isinstance(entry, tuple):
            name, code_vec = entry
        else:
            name = f"vec_{len(results)}"
            code_vec = entry

        sim = cosine_similarity(vec, code_vec)
        if sim >= threshold:
            results.append((name, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
