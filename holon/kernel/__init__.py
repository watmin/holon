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

# Accumulator primitives
from .accumulator import (
    accumulate,
    accumulate_weighted,
    capacity,
    clear_accumulator,
    create_accumulator,
    decay,
    merge_accumulators,
    normalize_accumulator,
    participation_ratio,
    purity,
    threshold_accumulator,
)

# Distance metrics
from .distance import (
    DistanceEngine,
    DistanceMetric,
    compare_metrics,
    cosine_similarity,
    get_recommended_metric,
    significance,
)

# Encoder
from .encoder import Encoder, ListEncodeMode, MathematicalPrimitive, TimeResolution

# Core VSA primitives
from .primitives import (  # Core algebra; Pattern extraction; Extended algebra; Vector operations; Advanced operations
    amplify,
    analogy,
    attend,
    autocorrelate,
    bind,
    blend,
    bundle,
    bundle_with_confidence,
    centroid,
    cleanup,
    coherence,
    complexity,
    conditional_bind,
    cross_correlate,
    difference,
    drift_rate,
    entropy,
    flip,
    grover_amplify,
    invert,
    negate,
    permute,
    power,
    project,
    prototype,
    prototype_add,
    random_project,
    reject,
    resonance,
    segment,
    similarity_matrix,
    similarity_profile,
    sparsify,
    threshold_bipolar,
    topk_similar,
    unbind,
)

# Scalar encoding
from .scalar import (
    decode_scalar_log,
    encode_circular,
    encode_positional,
    encode_scalar,
    encode_scalar_log,
)

# Similarity metrics
from .similarity import (
    AdvancedSimilarityEngine,
    SimilarityMetric,
    find_similar_vectors,
    normalized_dot_similarity,
)

# Store backends
from .store import CPUStore, Store

# Vector management
from .vector_manager import DeterministicVectorManager, VectorManager

# Walkable interface for zero-serialization encoding
from .walkable import (
    LinearScale,
    LogScale,
    TimeScale,
    Walkable,
    WalkableDict,
    WalkableList,
    WalkableScalar,
    WalkableSet,
    WalkableSpread,
    WalkType,
    as_walkable,
    is_walkable,
    register_walkable,
    register_walkable_adapter,
    walk_iter,
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
    "WalkableSpread",
    "WalkType",
    "LinearScale",
    "LogScale",
    "TimeScale",
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
