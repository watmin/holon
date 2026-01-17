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
    probe_vector: Union[np.ndarray, 'cp.ndarray'],
    stored_vectors: Dict[str, Any],
    top_k: int = 10,
    threshold: float = 0.0
) -> List[Tuple[str, float]]:
    """
    Find top-k similar vectors to the probe vector using heap optimization.

    :param probe_vector: The query vector.
    :param stored_vectors: Dict of id to vector.
    :param top_k: Number of top results.
    :param threshold: Minimum similarity.
    :return: List of (id, similarity) tuples, sorted descending.
    """
    # Use single-threaded optimized approach
    return _find_similar_vectors_single(probe_vector, stored_vectors, top_k, threshold)

def _find_similar_vectors_single(
    probe_vector: Union[np.ndarray, 'cp.ndarray'],
    stored_vectors: Dict[str, Any],
    top_k: int,
    threshold: float
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
    probe_vector: Union[np.ndarray, 'cp.ndarray'],
    stored_vectors: Dict[str, Any],
    top_k: int,
    threshold: float
) -> List[Tuple[str, float]]:
    """Parallel similarity search using thread pools."""
    import heapq
    import concurrent.futures
    import threading

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
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
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