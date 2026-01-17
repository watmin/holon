import numpy as np
from typing import List, Tuple, Dict, Any, Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def normalized_dot_similarity(vec1: Union[np.ndarray, 'cp.ndarray'],
                            vec2: Union[np.ndarray, 'cp.ndarray']) -> float:
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
    probe_vector: np.ndarray,
    stored_vectors: Dict[str, np.ndarray],
    top_k: int = 10,
    threshold: float = 0.0
) -> List[Tuple[str, float]]:
    """
    Find top-k similar vectors to the probe vector.

    :param probe_vector: The query vector.
    :param stored_vectors: Dict of id to vector.
    :param top_k: Number of top results.
    :param threshold: Minimum similarity.
    :return: List of (id, similarity) tuples, sorted descending.
    """
    similarities = []
    for data_id, vec in stored_vectors.items():
        sim = normalized_dot_similarity(probe_vector, vec)
        if sim >= threshold:
            similarities.append((data_id, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]