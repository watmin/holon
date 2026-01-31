"""
Similarity Primitives for Holon

Core similarity functions and advanced multi-metric approaches.
Designed to work within Qdrant vector database constraints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


# =============================================================================
# Core Similarity Functions
# =============================================================================


def normalized_dot_similarity(
    vec1: Union[np.ndarray, "cp.ndarray"], vec2: Union[np.ndarray, "cp.ndarray"]
) -> float:
    """
    Compute normalized dot product similarity (dot product divided by dimension).

    :param vec1: First vector.
    :param vec2: Second vector.
    :return: Normalized similarity score (0 to 1 range).
    """
    D = len(vec1)

    # Handle mixed CPU/GPU arrays
    if CUPY_AVAILABLE and isinstance(vec1, cp.ndarray):
        dot = cp.dot(vec1.astype(cp.float32), vec2.astype(cp.float32))
        return float(cp.asnumpy(dot) / D)
    else:
        dot = np.dot(vec1.astype(float), vec2.astype(float))
        return dot / D


def find_similar_vectors(
    probe_vector: Union[np.ndarray, "cp.ndarray"],
    stored_vectors: Dict[str, Any],
    top_k: int = 10,
    threshold: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Find top-k similar vectors to the probe vector using heap optimization.

    :param probe_vector: The query vector.
    :param stored_vectors: Dict of id to vector.
    :param top_k: Number of top results.
    :param threshold: Minimum similarity.
    :return: List of (id, similarity) tuples, sorted descending.
    """
    return _find_similar_vectors_single(probe_vector, stored_vectors, top_k, threshold)


def _find_similar_vectors_single(
    probe_vector: Union[np.ndarray, "cp.ndarray"],
    stored_vectors: Dict[str, Any],
    top_k: int,
    threshold: float,
) -> List[Tuple[str, float]]:
    """Single-threaded similarity search with heap optimization."""
    import heapq

    heap = []
    for data_id, vec in stored_vectors.items():
        sim = normalized_dot_similarity(probe_vector, vec)
        if sim >= threshold:
            if len(heap) < top_k:
                heapq.heappush(heap, (sim, data_id))
            elif sim > heap[0][0]:
                heapq.heapreplace(heap, (sim, data_id))

    results = [(data_id, score) for score, data_id in heap]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _find_similar_vectors_parallel(
    probe_vector: Union[np.ndarray, "cp.ndarray"],
    stored_vectors: Dict[str, Any],
    top_k: int,
    threshold: float,
) -> List[Tuple[str, float]]:
    """Parallel similarity search using thread pools."""
    import concurrent.futures
    import heapq

    # Split work into chunks for parallel processing
    items = list(stored_vectors.items())
    n_workers = min(14, len(items) // 1000 + 1)  # Scale workers with data size
    chunk_size = len(items) // n_workers

    def process_chunk(chunk_items):
        """Process a chunk of vectors and return top-k results."""
        local_heap = []
        for data_id, vec in chunk_items:
            sim = normalized_dot_similarity(probe_vector, vec)
            if sim >= threshold:
                if len(local_heap) < top_k:
                    heapq.heappush(local_heap, (sim, data_id))
                elif sim > local_heap[0][0]:
                    heapq.heapreplace(local_heap, (sim, data_id))
        return local_heap

    # Process chunks in parallel
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        local_results = list(executor.map(process_chunk, chunks))

    # Merge results from all chunks
    global_heap = []
    for local_heap in local_results:
        for sim, data_id in local_heap:
            if len(global_heap) < top_k:
                heapq.heappush(global_heap, (sim, data_id))
            elif sim > global_heap[0][0]:
                heapq.heapreplace(global_heap, (sim, data_id))

    results = [(data_id, score) for score, data_id in global_heap]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =============================================================================
# Advanced Similarity Metrics
# =============================================================================


class SimilarityMetric(str, Enum):
    """Supported similarity metrics compatible with Qdrant."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class AdvancedSimilarityEngine:
    """
    Advanced similarity engine with multiple geometric approaches.
    All primitives designed to work within Qdrant vector database constraints.
    """

    def __init__(self):
        self.metrics = {
            SimilarityMetric.COSINE: self._cosine_similarity,
            SimilarityMetric.DOT_PRODUCT: self._dot_product_similarity,
            SimilarityMetric.EUCLIDEAN: self._euclidean_similarity,
            SimilarityMetric.MANHATTAN: self._manhattan_similarity,
        }

    def multi_metric_similarity(
        self,
        query_vector: np.ndarray,
        target_vector: np.ndarray,
        weights: Optional[Dict[SimilarityMetric, float]] = None,
    ) -> float:
        """
        Multi-metric similarity combining multiple distance measures.
        Compatible with Qdrant's vector operations.

        Args:
            query_vector: Query vector
            target_vector: Target vector
            weights: Weights for each similarity metric

        Returns:
            Combined similarity score (0-1 scale)
        """
        if weights is None:
            weights = {
                SimilarityMetric.COSINE: 0.4,
                SimilarityMetric.DOT_PRODUCT: 0.3,
                SimilarityMetric.EUCLIDEAN: 0.2,
                SimilarityMetric.MANHATTAN: 0.1,
            }

        # Validate weights sum to 1.0 for predictable behavior
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:  # Allow small floating point errors
            raise ValueError(
                f"Multi-metric weights must sum to 1.0, got {total_weight}. "
                f"Current weights: {weights}"
            )

        similarities = {}
        for metric, weight in weights.items():
            if weight > 0:
                sim_func = self.metrics[metric]
                raw_sim = sim_func(query_vector, target_vector)
                # Normalize to 0-1 scale for all metrics
                normalized_sim = self._normalize_similarity(raw_sim, metric)
                similarities[metric] = normalized_sim * weight

        return sum(similarities.values())

    def contextual_similarity(
        self,
        query_vector: np.ndarray,
        target_vector: np.ndarray,
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Contextual similarity that incorporates metadata information.
        Qdrant-compatible by using metadata for score adjustment.

        Args:
            query_vector: Query vector
            target_vector: Target vector
            context_metadata: Metadata about the query/target context

        Returns:
            Context-adjusted similarity score
        """
        # Base geometric similarity
        base_similarity = self._cosine_similarity(query_vector, target_vector)

        # Context adjustments (can be stored as metadata in Qdrant)
        context_boost = 1.0

        if context_metadata:
            # Length-based adjustment
            query_length = context_metadata.get("query_length", 1)
            target_length = context_metadata.get("target_length", 1)

            # Penalize length mismatches for short queries
            if query_length < 3 and abs(query_length - target_length) > 1:
                context_boost *= 0.8

            # Boost for domain matches
            if context_metadata.get("domain_match"):
                context_boost *= 1.1

            # Temporal relevance (recency boost)
            if context_metadata.get("temporal_relevance"):
                context_boost *= 1.05

        return min(base_similarity * context_boost, 1.0)

    def hierarchical_similarity(
        self,
        query_vector: np.ndarray,
        target_vector: np.ndarray,
        hierarchy_levels: int = 3,
    ) -> float:
        """
        Hierarchical similarity using vector subspace analysis.
        Compatible with Qdrant's vector operations.

        Args:
            query_vector: Query vector
            target_vector: Target vector
            hierarchy_levels: Number of hierarchical levels to analyze

        Returns:
            Hierarchical similarity score
        """
        total_similarity = 0.0
        weights = [0.5, 0.3, 0.2]  # Weights for different levels

        vector_dim = len(query_vector)
        subspace_size = vector_dim // (2**hierarchy_levels)

        for level in range(min(hierarchy_levels, len(weights))):
            # Analyze different subspaces
            start_idx = level * subspace_size
            end_idx = min((level + 1) * subspace_size, vector_dim)

            query_sub = query_vector[start_idx:end_idx]
            target_sub = target_vector[start_idx:end_idx]

            if len(query_sub) > 0 and len(target_sub) > 0:
                sub_similarity = self._cosine_similarity(query_sub, target_sub)
                total_similarity += sub_similarity * weights[level]

        return total_similarity

    def ensemble_similarity(
        self,
        query_vector: np.ndarray,
        target_vector: np.ndarray,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Ensemble similarity combining multiple encoding strategies.
        Qdrant-compatible by treating different encodings as separate vectors.

        Args:
            query_vector: Primary query vector
            target_vector: Primary target vector
            ensemble_weights: Weights for different similarity approaches

        Returns:
            Ensemble similarity score
        """
        if ensemble_weights is None:
            ensemble_weights = {
                "cosine": 0.5,
                "hierarchical": 0.3,
                "contextual": 0.2,
            }

        similarities = {}

        # Cosine similarity
        similarities["cosine"] = self._cosine_similarity(query_vector, target_vector)

        # Hierarchical similarity
        similarities["hierarchical"] = self.hierarchical_similarity(
            query_vector, target_vector
        )

        # Contextual similarity (minimal context for basic version)
        context = {"query_length": len(query_vector) // 100}  # Rough estimate
        similarities["contextual"] = self.contextual_similarity(
            query_vector, target_vector, context
        )

        # Weighted combination
        ensemble_score = sum(
            similarities[approach] * weight
            for approach, weight in ensemble_weights.items()
        )

        return min(ensemble_score, 1.0)

    def _infer_query_type(self, vector: np.ndarray) -> str:
        """Infer query type from vector characteristics."""
        vector_norm = np.linalg.norm(vector)
        sparsity = np.sum(np.abs(vector) < 0.1) / len(vector)

        if vector_norm < 10:  # Low magnitude = short query
            return "short"
        elif sparsity > 0.8:  # High sparsity = exact match
            return "exact"
        elif vector_norm > 50:  # High magnitude = long/complex query
            return "long"
        else:
            return "fuzzy"

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity (0-1 scale)."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return (dot_product / (norm1 * norm2) + 1) / 2  # Normalize to 0-1

    def _dot_product_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Dot product similarity (0-1 scale)."""
        dot_product = np.dot(vec1, vec2)
        # Normalize by geometric mean of norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        max_possible = norm1 * norm2

        if max_possible == 0:
            return 0.0

        return (dot_product / max_possible + 1) / 2

    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Euclidean distance converted to similarity (0-1 scale)."""
        distance = np.linalg.norm(vec1 - vec2)
        # Convert distance to similarity (closer = more similar)
        return np.exp(-distance / 10)  # Exponential decay

    def _manhattan_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Manhattan distance converted to similarity (0-1 scale)."""
        distance = np.sum(np.abs(vec1 - vec2))
        # Convert distance to similarity
        return np.exp(-distance / (10 * len(vec1)))  # Normalized exponential decay

    def _normalize_similarity(self, raw_sim: float, metric: SimilarityMetric) -> float:
        """Normalize different metrics to 0-1 similarity scale."""
        if metric == SimilarityMetric.COSINE:
            # Holon uses normalized dot product (already in 0-1 range)
            return raw_sim
        elif metric == SimilarityMetric.DOT_PRODUCT:
            # Assume dot product is roughly -N to N, normalize to 0-1
            return (raw_sim + 100) / 200  # Rough normalization
        elif metric == SimilarityMetric.EUCLIDEAN:
            # Distance is already converted to similarity in the function
            return raw_sim
        elif metric == SimilarityMetric.MANHATTAN:
            # Distance is already converted to similarity in the function
            return raw_sim
        else:
            return raw_sim
