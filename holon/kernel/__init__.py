"""
Holon Kernel - Foundational VSA/HDC Primitives

The kernel layer provides the minimal, stable foundation for all Holon operations:
- VSA primitives (bind, bundle, unbind, negate, etc.)
- Structured data encoding (Encoder)
- Store backends (CPUStore, etc.)
- Deterministic vector management
- Zero-allocation encoding (Walkable)

This layer has no dependencies on memory/ or highlevel/ layers.
"""

# Core VSA primitives
from .primitives import (
    # Core algebra
    bind,
    unbind,
    bundle,
    permute,
    negate,
    amplify,
    difference,
    threshold_bipolar,
    # Pattern extraction
    prototype,
    prototype_add,
    resonance,
    blend,
    cleanup,
    # Extended algebra
    similarity_profile,
    attend,
    analogy,
    project,
    conditional_bind,
    segment,
    invert,
    complexity,
    # Vector operations
    sparsify,
    centroid,
    flip,
    topk_similar,
    similarity_matrix,
    entropy,
    random_project,
    power,
    autocorrelate,
    cross_correlate,
    # Advanced operations
    reject,
    bundle_with_confidence,
    coherence,
    grover_amplify,
    drift_rate,
)

# Accumulator primitives
from .accumulator import (
    create_accumulator,
    accumulate,
    accumulate_weighted,
    normalize_accumulator,
    decay,
    clear_accumulator,
    merge_accumulators,
    threshold_accumulator,
    capacity,
    purity,
    participation_ratio,
)

# Scalar encoding
from .scalar import (
    encode_scalar,
    encode_scalar_log,
    decode_scalar_log,
    encode_circular,
    encode_positional,
)

# Encoder
from .encoder import (
    Encoder,
    ListEncodeMode,
    MathematicalPrimitive,
    TimeResolution,
)

# Store backends
from .store import (
    CPUStore,
    Store,
)

# Vector management
from .vector_manager import (
    VectorManager,
    DeterministicVectorManager,
)

# Walkable interface for zero-serialization encoding
from .walkable import (
    Walkable,
    WalkableDict,
    WalkableList,
    WalkableScalar,
    WalkableSet,
    WalkType,
    LinearScale,
    LogScale,
    as_walkable,
    is_walkable,
    register_walkable,
    register_walkable_adapter,
    walk_iter,
)

# Distance metrics
from .distance import (
    DistanceEngine,
    DistanceMetric,
    cosine_similarity,
    compare_metrics,
    get_recommended_metric,
    significance,
)

# Similarity metrics
from .similarity import (
    SimilarityMetric,
    AdvancedSimilarityEngine,
    find_similar_vectors,
    normalized_dot_similarity,
)

__all__ = [
    # VSA primitives
    "bind",
    "unbind",
    "bundle",
    "permute",
    "negate",
    "amplify",
    "difference",
    "threshold_bipolar",
    "prototype",
    "prototype_add",
    "resonance",
    "blend",
    "cleanup",
    "similarity_profile",
    "attend",
    "analogy",
    "project",
    "conditional_bind",
    "segment",
    "invert",
    "complexity",
    "sparsify",
    "centroid",
    "flip",
    "topk_similar",
    "similarity_matrix",
    "entropy",
    "random_project",
    "power",
    "autocorrelate",
    "cross_correlate",
    "reject",
    "bundle_with_confidence",
    "coherence",
    "grover_amplify",
    "drift_rate",
    # Accumulators
    "create_accumulator",
    "accumulate",
    "accumulate_weighted",
    "normalize_accumulator",
    "decay",
    "clear_accumulator",
    "merge_accumulators",
    "threshold_accumulator",
    "capacity",
    "purity",
    "participation_ratio",
    # Scalar encoding
    "encode_scalar",
    "encode_scalar_log",
    "decode_scalar_log",
    "encode_circular",
    "encode_positional",
    # Encoder
    "Encoder",
    "ListEncodeMode",
    "MathematicalPrimitive",
    "TimeResolution",
    # Store
    "CPUStore",
    "Store",
    # Vector management
    "VectorManager",
    "DeterministicVectorManager",
    # Walkable
    "Walkable",
    "WalkableDict",
    "WalkableList",
    "WalkableScalar",
    "WalkableSet",
    "WalkType",
    "LinearScale",
    "LogScale",
    "as_walkable",
    "is_walkable",
    "register_walkable",
    "register_walkable_adapter",
    "walk_iter",
    # Distance
    "DistanceEngine",
    "DistanceMetric",
    "cosine_similarity",
    "compare_metrics",
    "get_recommended_metric",
    "significance",
    # Similarity
    "SimilarityMetric",
    "AdvancedSimilarityEngine",
    "find_similar_vectors",
    "normalized_dot_similarity",
]
