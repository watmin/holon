"""
Distance Metrics for Holon VSA/HDC

Comprehensive distance and similarity metrics optimized for bipolar vectors.
Includes Qdrant-compatible metrics plus VSA-specific metrics.

## Metric Categories

1. **Qdrant-Native** (can be used directly in Qdrant queries):
   - Cosine, Dot Product, Euclidean, Manhattan

2. **VSA-Specific** (client-side, optimal for bipolar vectors):
   - Hamming, Overlap, Agreement

3. **Advanced** (for specialized use cases):
   - Chebyshev (L∞), Minkowski, Weighted

## Recommendations by Use Case

| Use Case | Recommended Metric |
|----------|-------------------|
| General semantic similarity | Cosine |
| Normalized vectors | Dot Product |
| Bipolar VSA vectors | Hamming or Overlap |
| Outlier sensitivity | Chebyshev |
| Spatial/geometric | Euclidean |
| Feature importance | Weighted |
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class DistanceMetric(str, Enum):
    """
    All supported distance/similarity metrics.

    Qdrant-Compatible (can be used server-side):
    - COSINE: Angular similarity (direction-only)
    - DOT_PRODUCT: Raw dot product (direction + magnitude)
    - EUCLIDEAN: L2 distance
    - MANHATTAN: L1 distance

    VSA-Specific (client-side, optimal for bipolar):
    - HAMMING: Bit differences (natural for bipolar)
    - OVERLAP: Count of matching positions
    - AGREEMENT: Normalized overlap

    Advanced:
    - CHEBYSHEV: L∞ (max difference)
    - MINKOWSKI: Generalized Lp distance
    - WEIGHTED_COSINE: Per-dimension weighted cosine
    """

    # Qdrant-compatible
    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    # VSA-specific
    HAMMING = "hamming"
    OVERLAP = "overlap"
    AGREEMENT = "agreement"

    # Advanced
    CHEBYSHEV = "chebyshev"
    MINKOWSKI = "minkowski"
    WEIGHTED_COSINE = "weighted_cosine"
    WEIGHTED_EUCLIDEAN = "weighted_euclidean"


# =============================================================================
# Core Distance Functions
# =============================================================================


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Cosine similarity: measures angular similarity.

    Range: [-1, 1] for general vectors, typically [0, 1] for bipolar
    Higher = more similar

    Best for: Semantic similarity, direction-only comparison
    Qdrant: Yes (native)
    """
    vec1 = vec1.astype(np.float64)
    vec2 = vec2.astype(np.float64)

    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Raw dot product similarity.

    Range: Unbounded (depends on vector magnitudes)
    Higher = more similar

    Best for: Normalized vectors, ranking by relevance
    Qdrant: Yes (native)
    """
    return float(np.dot(vec1.astype(np.float64), vec2.astype(np.float64)))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Euclidean (L2) distance.

    Range: [0, ∞)
    Lower = more similar

    Best for: Geometric/spatial relationships
    Qdrant: Yes (native)
    """
    return float(np.linalg.norm(vec1.astype(np.float64) - vec2.astype(np.float64)))


def euclidean_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Euclidean distance converted to similarity.

    Range: (0, 1]
    Higher = more similar
    """
    dist = euclidean_distance(vec1, vec2)
    # Use exponential decay: exp(-d) maps [0,∞) to (0,1]
    # Scale by sqrt(dim) for dimension-independent behavior
    scale = np.sqrt(len(vec1))
    return float(np.exp(-dist / scale))


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Manhattan (L1) distance.

    Range: [0, ∞)
    Lower = more similar

    Best for: Grid-like spaces, feature counting
    Qdrant: Yes (native)
    """
    return float(np.sum(np.abs(vec1.astype(np.float64) - vec2.astype(np.float64))))


def manhattan_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Manhattan distance converted to similarity.

    Range: (0, 1]
    Higher = more similar
    """
    dist = manhattan_distance(vec1, vec2)
    # Normalize by dimension and max possible distance per dim
    max_dist = 2 * len(vec1)  # For bipolar vectors: max diff is 2 per dim
    return float(1.0 - dist / max_dist) if max_dist > 0 else 1.0


# =============================================================================
# VSA-Specific Metrics (Optimal for Bipolar Vectors)
# =============================================================================


def hamming_distance(vec1: np.ndarray, vec2: np.ndarray) -> int:
    """
    Hamming distance: count of positions where vectors differ.

    Range: [0, D] where D is dimension
    Lower = more similar

    Best for: Bipolar vectors, binary classification
    Qdrant: No (client-side only)

    Note: For bipolar vectors (-1, +1), this counts mismatches.
    For vectors with zeros, positions where both are 0 count as matches.
    """
    # For bipolar, compare signs
    # Positions match if product > 0 (same sign) or both are 0
    same = (vec1 * vec2 > 0) | ((vec1 == 0) & (vec2 == 0))
    return int(np.sum(~same))


def hamming_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Normalized Hamming similarity.

    Range: [0, 1]
    Higher = more similar

    Formula: 1 - (hamming_distance / dimension)
    """
    dist = hamming_distance(vec1, vec2)
    return float(1.0 - dist / len(vec1))


def overlap_count(vec1: np.ndarray, vec2: np.ndarray) -> int:
    """
    Count positions where vectors agree (same sign).

    Range: [0, D]
    Higher = more similar

    Best for: Counting shared features in bipolar vectors
    """
    return int(np.sum((vec1 * vec2) > 0))


def overlap_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Normalized overlap similarity.

    Range: [0, 1]
    Higher = more similar

    Note: Similar to Hamming similarity but handles zeros differently.
    """
    count = overlap_count(vec1, vec2)
    # Count non-zero positions in either vector
    active = np.sum((vec1 != 0) | (vec2 != 0))
    return float(count / active) if active > 0 else 0.0


def agreement_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Agreement similarity: (agreements - disagreements) / dimension.

    Range: [-1, 1]
    Higher = more similar

    Best for: Bipolar vectors where you want a balanced view.
    """
    agree = np.sum(vec1 * vec2 > 0)
    disagree = np.sum(vec1 * vec2 < 0)
    return float((agree - disagree) / len(vec1))


# =============================================================================
# Advanced Metrics
# =============================================================================


def chebyshev_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Chebyshev (L∞) distance: maximum absolute difference.

    Range: [0, ∞)
    Lower = more similar

    Best for: Outlier-sensitive comparisons, worst-case analysis
    Qdrant: No (client-side only)
    """
    return float(np.max(np.abs(vec1.astype(np.float64) - vec2.astype(np.float64))))


def chebyshev_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Chebyshev distance converted to similarity.

    Range: (0, 1]
    Higher = more similar
    """
    dist = chebyshev_distance(vec1, vec2)
    # For bipolar vectors, max possible distance is 2
    return float(1.0 - dist / 2.0) if dist <= 2.0 else 0.0


def minkowski_distance(vec1: np.ndarray, vec2: np.ndarray, p: float = 2.0) -> float:
    """
    Minkowski (Lp) distance: generalized distance metric.

    Range: [0, ∞)
    Lower = more similar

    Special cases:
    - p=1: Manhattan distance
    - p=2: Euclidean distance
    - p→∞: Chebyshev distance

    Best for: Tunable distance sensitivity
    Qdrant: No (client-side only)
    """
    diff = np.abs(vec1.astype(np.float64) - vec2.astype(np.float64))
    return float(np.sum(diff**p) ** (1.0 / p))


def weighted_cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted cosine similarity: per-dimension importance weighting.

    Range: [-1, 1]
    Higher = more similar

    Best for: Feature importance, learned weights from training
    Qdrant: No (client-side, but can pre-weight vectors before storage)
    """
    w = weights.astype(np.float64)
    v1 = vec1.astype(np.float64) * w
    v2 = vec2.astype(np.float64) * w

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def weighted_euclidean_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted Euclidean distance: per-dimension importance weighting.

    Range: [0, ∞)
    Lower = more similar

    Best for: Feature importance, non-uniform dimension scaling
    Qdrant: No (client-side, but can pre-weight vectors before storage)
    """
    w = weights.astype(np.float64)
    diff = (vec1.astype(np.float64) - vec2.astype(np.float64)) * np.sqrt(w)
    return float(np.linalg.norm(diff))


# =============================================================================
# Unified Distance/Similarity Engine
# =============================================================================


class DistanceEngine:
    """
    Unified engine for computing distances and similarities.

    Provides a consistent interface for all metrics with:
    - Automatic distance ↔ similarity conversion
    - Qdrant compatibility information
    - Batch computation support
    """

    # Metrics that return DISTANCE (lower = more similar)
    DISTANCE_METRICS = {
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
        DistanceMetric.HAMMING,
        DistanceMetric.CHEBYSHEV,
        DistanceMetric.MINKOWSKI,
        DistanceMetric.WEIGHTED_EUCLIDEAN,
    }

    # Metrics that return SIMILARITY (higher = more similar)
    SIMILARITY_METRICS = {
        DistanceMetric.COSINE,
        DistanceMetric.DOT_PRODUCT,
        DistanceMetric.OVERLAP,
        DistanceMetric.AGREEMENT,
        DistanceMetric.WEIGHTED_COSINE,
    }

    # Metrics supported natively by Qdrant
    QDRANT_NATIVE = {
        DistanceMetric.COSINE,
        DistanceMetric.DOT_PRODUCT,
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
    }

    def __init__(self, default_metric: DistanceMetric = DistanceMetric.COSINE):
        self.default_metric = default_metric

        # Map metrics to their functions
        self._distance_funcs: Dict[DistanceMetric, Callable] = {
            DistanceMetric.EUCLIDEAN: euclidean_distance,
            DistanceMetric.MANHATTAN: manhattan_distance,
            DistanceMetric.HAMMING: hamming_distance,
            DistanceMetric.CHEBYSHEV: chebyshev_distance,
        }

        self._similarity_funcs: Dict[DistanceMetric, Callable] = {
            DistanceMetric.COSINE: cosine_similarity,
            DistanceMetric.DOT_PRODUCT: dot_product_similarity,
            DistanceMetric.OVERLAP: overlap_similarity,
            DistanceMetric.AGREEMENT: agreement_similarity,
            DistanceMetric.HAMMING: hamming_similarity,
            DistanceMetric.EUCLIDEAN: euclidean_similarity,
            DistanceMetric.MANHATTAN: manhattan_similarity,
            DistanceMetric.CHEBYSHEV: chebyshev_similarity,
        }

    def distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: DistanceMetric = None,
        **kwargs,
    ) -> float:
        """
        Compute distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Distance metric to use
            **kwargs: Additional arguments (e.g., p for Minkowski, weights)

        Returns:
            Distance value (lower = more similar)
        """
        metric = metric or self.default_metric

        if metric == DistanceMetric.MINKOWSKI:
            p = kwargs.get("p", 2.0)
            return minkowski_distance(vec1, vec2, p)

        elif metric == DistanceMetric.WEIGHTED_EUCLIDEAN:
            weights = kwargs.get("weights")
            if weights is None:
                raise ValueError("weighted_euclidean requires 'weights' argument")
            return weighted_euclidean_distance(vec1, vec2, weights)

        elif metric in self._distance_funcs:
            return self._distance_funcs[metric](vec1, vec2)

        else:
            # Convert similarity to distance
            sim = self.similarity(vec1, vec2, metric, **kwargs)
            return 1.0 - sim

    def similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: DistanceMetric = None,
        **kwargs,
    ) -> float:
        """
        Compute similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric to use
            **kwargs: Additional arguments (e.g., weights)

        Returns:
            Similarity value (higher = more similar)
        """
        metric = metric or self.default_metric

        if metric == DistanceMetric.WEIGHTED_COSINE:
            weights = kwargs.get("weights")
            if weights is None:
                raise ValueError("weighted_cosine requires 'weights' argument")
            return weighted_cosine_similarity(vec1, vec2, weights)

        elif metric in self._similarity_funcs:
            return self._similarity_funcs[metric](vec1, vec2)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_nearest(
        self,
        query: np.ndarray,
        candidates: List[Tuple[str, np.ndarray]],
        k: int = 10,
        metric: DistanceMetric = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors from candidates.

        Args:
            query: Query vector
            candidates: List of (id, vector) tuples
            k: Number of neighbors to return
            metric: Distance metric to use
            **kwargs: Additional arguments for metric

        Returns:
            List of (id, score) tuples, sorted by similarity (desc)
        """
        metric = metric or self.default_metric

        scores = []
        for cand_id, cand_vec in candidates:
            sim = self.similarity(query, cand_vec, metric, **kwargs)
            scores.append((cand_id, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def is_qdrant_native(self, metric: DistanceMetric) -> bool:
        """Check if metric is supported natively by Qdrant."""
        return metric in self.QDRANT_NATIVE

    def get_qdrant_distance(self, metric: DistanceMetric) -> Optional[str]:
        """Get Qdrant Distance enum name for native metrics."""
        mapping = {
            DistanceMetric.COSINE: "Cosine",
            DistanceMetric.DOT_PRODUCT: "Dot",
            DistanceMetric.EUCLIDEAN: "Euclid",
            DistanceMetric.MANHATTAN: "Manhattan",
        }
        return mapping.get(metric)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_recommended_metric(use_case: str) -> DistanceMetric:
    """
    Get recommended metric for common use cases.

    Use cases:
    - "semantic": General semantic similarity
    - "bipolar": VSA bipolar vector comparison
    - "normalized": Pre-normalized vectors
    - "geometric": Spatial/geometric relationships
    - "outlier": Outlier-sensitive comparison
    - "weighted": Feature importance weighting
    """
    recommendations = {
        "semantic": DistanceMetric.COSINE,
        "bipolar": DistanceMetric.HAMMING,
        "normalized": DistanceMetric.DOT_PRODUCT,
        "geometric": DistanceMetric.EUCLIDEAN,
        "outlier": DistanceMetric.CHEBYSHEV,
        "weighted": DistanceMetric.WEIGHTED_COSINE,
        "grid": DistanceMetric.MANHATTAN,
    }
    return recommendations.get(use_case, DistanceMetric.COSINE)


def compare_metrics(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metrics: List[DistanceMetric] = None,
) -> Dict[str, float]:
    """
    Compare multiple metrics for the same vector pair.

    Useful for understanding metric behavior on your data.
    """
    if metrics is None:
        metrics = [
            DistanceMetric.COSINE,
            DistanceMetric.DOT_PRODUCT,
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.MANHATTAN,
            DistanceMetric.HAMMING,
            DistanceMetric.OVERLAP,
            DistanceMetric.AGREEMENT,
            DistanceMetric.CHEBYSHEV,
        ]

    engine = DistanceEngine()
    results = {}

    for metric in metrics:
        try:
            sim = engine.similarity(vec1, vec2, metric)
            results[metric.value] = round(sim, 4)
        except Exception as e:
            results[metric.value] = f"Error: {e}"

    return results
