# Holon: Programmatic Neural Memory
# Version: 0.1.0
#
# Import anything directly from the top level:
#
#   from holon import bind, bundle, Encoder, OnlineSubspace, HolonClient
#
# The implementation is organized into three layers, accessible when you
# want to be explicit or pull in a single subsystem:
#
#   holon.kernel   - VSA/HDC primitives (bind, bundle, encode, store, walkable)
#   holon.memory   - Neural memory (OnlineSubspace, Engram, EngramLibrary)
#   holon.highlevel - Convenience facade (HolonClient)

__version__ = "0.1.0"

# ============================================================================
# Layer 1: Kernel - Foundational primitives
# ============================================================================

from .highlevel import HolonClient
from .kernel import (  # VSA primitives; Accumulators; Scalar encoding; Encoder; Store; Vector management; Walkable; Distance; Similarity metrics
    AdvancedSimilarityEngine,
    CPUStore,
    DeterministicVectorManager,
    DistanceEngine,
    DistanceMetric,
    Encoder,
    LinearScale,
    ListEncodeMode,
    LogScale,
    MathematicalPrimitive,
    SimilarityMetric,
    Store,
    TimeResolution,
    TimeScale,
    VectorManager,
    Walkable,
    WalkableDict,
    WalkableList,
    WalkableScalar,
    WalkableSet,
    WalkableSpread,
    WalkType,
    accumulate,
    accumulate_weighted,
    amplify,
    analogy,
    as_walkable,
    attend,
    autocorrelate,
    bind,
    blend,
    bundle,
    bundle_with_confidence,
    capacity,
    centroid,
    cleanup,
    clear_accumulator,
    coherence,
    compare_metrics,
    complexity,
    conditional_bind,
    cosine_similarity,
    create_accumulator,
    cross_correlate,
    decay,
    decode_scalar_log,
    difference,
    drift_rate,
    encode_circular,
    encode_positional,
    encode_scalar,
    encode_scalar_log,
    entropy,
    find_similar_vectors,
    flip,
    get_recommended_metric,
    grover_amplify,
    invert,
    is_walkable,
    merge_accumulators,
    negate,
    normalize_accumulator,
    normalized_dot_similarity,
    participation_ratio,
    permute,
    power,
    project,
    prototype,
    prototype_add,
    purity,
    random_project,
    register_walkable,
    register_walkable_adapter,
    reject,
    resonance,
    segment,
    significance,
    similarity_matrix,
    similarity_profile,
    sparsify,
    threshold_accumulator,
    threshold_bipolar,
    topk_similar,
    unbind,
    walk_iter,
)
from .memory import Engram, EngramLibrary, OnlineSubspace, StripedSubspace

# ============================================================================
# Layer 2: Memory - Programmatic neural memory
# ============================================================================


# ============================================================================
# Layer 3: High-Level - Convenience API
# ============================================================================


# ============================================================================
# Optional backends
# ============================================================================

# Optional Qdrant backend (requires qdrant-client)
try:
    from .qdrant_store import QdrantStore
except ImportError:
    QdrantStore = None  # type: ignore


# ============================================================================
# Convenience function
# ============================================================================


def create_client(dimensions: int = 4096) -> HolonClient:
    """Create a standalone HolonClient with default settings.

    Args:
        dimensions: Vector dimensionality (default 4096)

    Returns:
        Configured HolonClient

    Example:
        >>> from holon import create_client
        >>> client = create_client()
        >>> vec = client.encode({"type": "billing"})
    """
    return HolonClient(dimensions=dimensions)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    # Kernel - VSA primitives
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
    # Kernel - Accumulators
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
    # Kernel - Scalar encoding
    "encode_scalar",
    "encode_scalar_log",
    "decode_scalar_log",
    "encode_circular",
    "encode_positional",
    # Kernel - Encoder
    "Encoder",
    "ListEncodeMode",
    "MathematicalPrimitive",
    "TimeResolution",
    # Kernel - Store
    "CPUStore",
    "Store",
    "QdrantStore",
    # Kernel - Vector management
    "VectorManager",
    "DeterministicVectorManager",
    # Kernel - Walkable
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
    # Kernel - Distance
    "DistanceEngine",
    "DistanceMetric",
    "cosine_similarity",
    "compare_metrics",
    "get_recommended_metric",
    "significance",
    # Kernel - Similarity
    "SimilarityMetric",
    "AdvancedSimilarityEngine",
    "find_similar_vectors",
    "normalized_dot_similarity",
    # Memory layer
    "OnlineSubspace",
    "StripedSubspace",
    "Engram",
    "EngramLibrary",
    # High-level
    "HolonClient",
    # Convenience
    "create_client",
]
