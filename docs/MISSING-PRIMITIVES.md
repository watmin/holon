# Missing Primitives: What Holon-rs Should Provide

**Brainstorm of vector operations that belong in the library, not userland.**

Last updated: February 2026

The test: "Would a user reasonably implement this themselves, or would they
expect the library to provide it?" If two different applications would write
the same 10 lines of code, it belongs in the library.

---

## Tier 1 — ✅ COMPLETE

All Tier 1 primitives have been implemented in both Python and Rust.

| Primitive | Status | Location |
|---|---|---|
| `sparsify(vec, k)` | ✅ | `primitives.rs` / `primitives.py` |
| `capacity(acc, codebook_size)` | ✅ | `accumulator.rs` / `accumulator.py` |
| `centroid(vectors)` | ✅ | `primitives.rs` / `primitives.py` |
| `flip(vec)` | ✅ | `primitives.rs` / `primitives.py` |
| `topk_similar(query, candidates, k)` | ✅ | `primitives.rs` / `primitives.py` |
| `similarity_matrix(vectors)` | ✅ | `primitives.rs` / `primitives.py` |

---

## Tier 2 — ✅ COMPLETE

All Tier 2 primitives have been implemented in both Python and Rust.

| Primitive | Status | Location |
|---|---|---|
| `entropy(vec)` | ✅ | `primitives.rs` / `primitives.py` |
| `random_project(vec, dims, seed)` | ✅ | `primitives.rs` / `primitives.py` |
| `power(vec, exponent)` | ✅ | `primitives.rs` / `primitives.py` |
| `autocorrelate(stream, lag)` | ✅ | `primitives.rs` / `primitives.py` |
| `cross_correlate(a, b, lag)` | ✅ | `primitives.rs` / `primitives.py` |

---

## Quantum-inspired — ✅ COMPLETE

From `QUANTUM.md`, the accumulator-level concentration measures have been implemented.

| Primitive | Status | Location |
|---|---|---|
| `purity(acc)` | ✅ | `accumulator.rs` / `accumulator.py` |
| `participation_ratio(acc)` | ✅ | `accumulator.rs` / `accumulator.py` |

---

## Tier 3: Won't Implement

Reviewed February 2026. These were originally proposed as "niche but
potentially powerful." On closer analysis, none justify inclusion.

### ~~`decompose(vec, max_components)`~~ — Unsolvable without codebook

Blind source separation from a single accumulated vector is an
underspecified problem. The iterative approach (probe with random vectors,
negate strongest, repeat) is unreliable in high dimensions — you can't
distinguish true components from noise artifacts. Real ICA/NMF requires
many independent observations, not a single vector. With a codebook this
is just `invert`, which is already implemented.

### ~~`mutual_information(vec_a, vec_b)`~~ — Redundant with cosine

For bipolar vectors with 3 discrete values (+1, -1, 0), the joint
distribution is a 3×3 contingency table. The MI over this correlates
almost perfectly with cosine similarity because the element values are
already discrete — there are no nonlinear dependencies to capture beyond
what linear agreement measures. You'd implement it, find it tracks
cosine 1:1, and never use it.

### ~~`context_gate(vec, context, temperature)`~~ — Redundant with attend

This is `attend(context, vec, temperature, AttendMode::Soft)` with the
arguments reordered. The existing `attend` primitive already supports
soft/hard/amplify modes with a strength parameter. Adding a separate
function with identical semantics would only create API confusion.

### ~~`kronecker(vec_a, vec_b)`~~ — Wrong algebra for bipolar VSA

Kronecker/tensor product expands to N² dimensions then requires a lossy
fold back to N. The fold step is arbitrary and destroys the properties
that make VSA work (self-inverse binding, dimension-independence). This
is relevant to HRR (holographic reduced representations) with continuous
complex vectors, not bipolar MAP. Element-wise `bind` already captures
the interactions that matter for our use cases.

---

## Summary

| Tier | Status | Count |
|---|---|---|
| Tier 1 (Core) | ✅ Complete | 6/6 |
| Tier 2 (Strong) | ✅ Complete | 5/5 |
| Quantum-inspired | ✅ Complete | 2/2 |
| Tier 3 (Niche) | ❌ Won't implement | 0/4 |

All actionable primitives from this document have been implemented.
Tier 3 items were evaluated and rejected with rationale above.
