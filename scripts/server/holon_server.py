#!/usr/bin/env python3

"""
Holon HTTP API Server
Provides REST API for VSA/HDC neural memory operations.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from holon import CPUStore
from holon.atomizer import parse_data
from holon.encoder import MathematicalPrimitive

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Holon Neural Memory API",
    description="VSA/HDC-based neural memory system for structured data",
    version="0.1.0",
)

# Global store instance (ephemeral for now)
store = CPUStore(backend="auto")

# Configuration
MAX_QUERY_RESULTS = 100  # System maximum for top_k
DEFAULT_QUERY_RESULTS = 10


def is_subset(guard: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Check if guard dict is a subset of data dict (recursive, with $any support)."""
    for key, value in guard.items():
        if key not in data:
            return False
        if isinstance(value, dict):
            if not isinstance(data[key], dict):
                return False
            if not is_subset(value, data[key]):
                return False
        elif isinstance(value, list):
            if not isinstance(data[key], list) or len(value) != len(data[key]):
                return False
            for g_item, d_item in zip(value, data[key]):
                if isinstance(g_item, dict) and "$any" in g_item:
                    continue  # any_marker matches anything
                elif g_item != d_item:
                    return False
        # For other types, check exact match
        elif data[key] != value:
            return False
    return True


# Configuration
MAX_QUERY_RESULTS = 100  # System maximum for top_k
DEFAULT_QUERY_RESULTS = 10


class InsertRequest(BaseModel):
    data: str = Field(..., description="Data to insert (JSON or EDN string)")
    data_type: str = Field("json", description="Data format: 'json' or 'edn'")


class BatchInsertRequest(BaseModel):
    items: List[str] = Field(..., description="List of data strings to insert")
    data_type: str = Field("json", description="Data format for all items")


class QueryRequest(BaseModel):
    probe: str = Field(..., description="Query probe (JSON or EDN string)")
    data_type: str = Field("json", description="Data format: 'json' or 'edn'")
    top_k: int = Field(
        DEFAULT_QUERY_RESULTS, description="Number of top results to return"
    )
    threshold: float = Field(0.0, description="Similarity threshold (0-1)")
    guard: Optional[Dict[str, Any]] = Field(
        None, description="Guard condition as dict (pattern match)"
    )
    negations: Optional[Dict[str, Any]] = Field(
        None, description="Negation filters as dict {key: value_to_exclude}"
    )
    any_marker: str = Field("$any", description="Marker for wildcards in probe/guard")


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(
        ..., description="Query results with id, score, data"
    )


class InsertResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for inserted data")


class BatchInsertResponse(BaseModel):
    ids: List[str] = Field(..., description="List of unique identifiers")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    backend: str = Field(..., description="Storage backend type")
    items_count: int = Field(..., description="Number of stored items")


class EncodeRequest(BaseModel):
    data: str = Field(..., description="Data to encode (JSON or EDN string)")
    data_type: str = Field("json", description="Data format: 'json' or 'edn'")


class MathematicalEncodeRequest(BaseModel):
    primitive: str = Field(..., description="Mathematical primitive to encode")
    value: Union[int, float] = Field(..., description="Value for mathematical primitive")


class MathematicalComposeRequest(BaseModel):
    operation: str = Field(..., description="Mathematical operation: 'bind' or 'bundle'")
    vectors: List[List[float]] = Field(..., description="Vectors for composition operations")


class EncodeResponse(BaseModel):
    vector: List[float] = Field(..., description="Encoded vector as list of floats")
    encoding_type: str = Field(..., description="Type of encoding performed")


class SimilarityRequest(BaseModel):
    vector_a: List[float] = Field(..., description="First vector")
    vector_b: List[float] = Field(..., description="Second vector")


class SimilarityResponse(BaseModel):
    similarity: float = Field(..., description="Cosine similarity between vectors")


class VectorSearchRequest(BaseModel):
    vector: List[float] = Field(..., description="Query vector")
    top_k: int = Field(DEFAULT_QUERY_RESULTS, description="Number of results")
    threshold: float = Field(0.0, description="Similarity threshold")
    guard: Optional[Dict[str, Any]] = Field(None, description="Guard condition")
    negations: Optional[Dict[str, Any]] = Field(None, description="Negation filters")


class BatchSearchItem(BaseModel):
    """Single search request within a batch (Qdrant-compatible)."""
    probe: str = Field(..., description="Query probe (JSON or EDN string)")
    data_type: str = Field("json", description="Data format")
    top_k: int = Field(DEFAULT_QUERY_RESULTS, description="Number of results")
    threshold: float = Field(0.0, description="Similarity threshold")
    guard: Optional[Dict[str, Any]] = Field(None, description="Guard condition")
    negations: Optional[Dict[str, Any]] = Field(None, description="Negation filters")
    any_marker: str = Field("$any", description="Marker for wildcards")


class BatchSearchRequest(BaseModel):
    """Batch search request - Qdrant search_batch compatible."""
    searches: List[BatchSearchItem] = Field(..., description="List of search requests")




@app.get("/api/v1/health")
async def health_v1():
    """Health check with v1 API structure."""
    return HealthResponse(
        status="healthy",
        backend=store.backend,
        items_count=len(store.stored_data)
    )


@app.get("/api/v1/diagnostics")
async def diagnostics_v1():
    """Get diagnostic information about the store and query performance."""
    return {
        "items_count": len(store.stored_data),
        "vectors_count": len(store.stored_vectors),
        "ann_index_active": store.ann_index is not None,
        "ann_threshold": 1000,  # ANN_THRESHOLD from cpu_store
        "dimensions": store.dimensions,
        "query_stats": query_stats,
        "encode_stats": encode_stats,
    }


@app.post("/api/v1/items")
async def create_item_v1(request: InsertRequest):
    """Create a single item (v1 API)."""
    try:
        data_id = store.insert(request.data, request.data_type)
        logger.info(f"Inserted item {data_id}")
        return {"id": data_id, "created": True}
    except Exception as e:
        logger.error(f"Insert failed: {e}")
        raise HTTPException(status_code=400, detail=f"Insert failed: {str(e)}")


@app.post("/api/v1/items/batch")
async def create_items_batch_v1(request: BatchInsertRequest):
    """Create multiple items (v1 API)."""
    try:
        ids = store.batch_insert(request.items, request.data_type)
        logger.info(f"Batch inserted {len(ids)} items")
        return {"ids": ids, "created": len(ids)}
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        raise HTTPException(status_code=400, detail=f"Batch insert failed: {str(e)}")


@app.get("/api/v1/items/{item_id}")
async def get_item_v1(item_id: str):
    """Retrieve a specific item by ID (v1 API)."""
    try:
        data = store.get(item_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"id": item_id, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get item failed: {e}")
        raise HTTPException(status_code=400, detail=f"Get item failed: {str(e)}")


# Query timing stats
query_stats = {"count": 0, "total_time": 0.0, "last_logged": 0}

@app.post("/api/v1/search")
async def search_items_v1(request: QueryRequest):
    """Search items using vector similarity (v1 API)."""
    import time
    start_time = time.time()

    try:
        # Validate top_k
        if request.top_k > MAX_QUERY_RESULTS:
            raise HTTPException(
                status_code=400,
                detail=f"top_k exceeds maximum allowed ({MAX_QUERY_RESULTS})",
            )
        if request.top_k < 1:
            raise HTTPException(status_code=400, detail="top_k must be >= 1")

        # Validate threshold
        if not (0.0 <= request.threshold <= 1.0):
            raise HTTPException(
                status_code=400, detail="threshold must be between 0.0 and 1.0"
            )

        # Execute query
        results = store.query(
            probe=request.probe,
            data_type=request.data_type,
            top_k=request.top_k,
            threshold=request.threshold,
            guard=request.guard,
            negations=request.negations,
            any_marker=request.any_marker,
        )

        # Format response
        formatted_results = [
            {"id": data_id, "score": score, "data": data}
            for data_id, score, data in results
        ]

        # Track timing
        elapsed = time.time() - start_time
        query_stats["count"] += 1
        query_stats["total_time"] += elapsed

        # Log every 100 queries
        if query_stats["count"] % 100 == 0:
            avg = query_stats["total_time"] / query_stats["count"]
            logger.info(f"QUERY_STATS: {query_stats['count']} queries, avg={avg*1000:.2f}ms, total={query_stats['total_time']:.2f}s")

        return {"results": formatted_results, "count": len(formatted_results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Search failed: {str(e)}")


# Encode timing stats
encode_stats = {"count": 0, "total_time": 0.0}

@app.post("/api/v1/vectors/encode")
async def encode_data_v1(request: EncodeRequest):
    """Encode structured data to vector (v1 API)."""
    import time
    start_time = time.time()

    try:
        parsed = parse_data(request.data, request.data_type)
        encoded_vector = store.encoder.encode_data(parsed)
        cpu_vector = store.vector_manager.to_cpu(encoded_vector)
        vector_list = cpu_vector.tolist()

        # Track timing
        elapsed = time.time() - start_time
        encode_stats["count"] += 1
        encode_stats["total_time"] += elapsed

        if encode_stats["count"] % 100 == 0:
            avg = encode_stats["total_time"] / encode_stats["count"]
            logger.info(f"ENCODE_STATS: {encode_stats['count']} encodes, avg={avg*1000:.2f}ms")

        return {"vector": vector_list, "encoding_type": f"structural_{request.data_type}"}
    except Exception as e:
        logger.error(f"Vector encoding failed: {e}")
        raise HTTPException(status_code=400, detail=f"Vector encoding failed: {str(e)}")


@app.post("/api/v1/vectors/encode/mathematical")
async def encode_mathematical_v1(request: MathematicalEncodeRequest):
    """Encode mathematical primitives (v1 API)."""
    try:
        try:
            primitive = MathematicalPrimitive(request.primitive)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown mathematical primitive: {request.primitive}")

        encoded_vector = store.encoder.encode_mathematical_primitive(primitive, request.value)
        cpu_vector = store.vector_manager.to_cpu(encoded_vector)
        vector_list = cpu_vector.tolist()

        return {"vector": vector_list, "encoding_type": f"mathematical_{request.primitive}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mathematical encoding failed: {e}")
        raise HTTPException(status_code=400, detail=f"Mathematical encoding failed: {str(e)}")


@app.post("/api/v1/vectors/compose")
async def compose_vectors_v1(request: MathematicalComposeRequest):
    """Compose vectors using mathematical operations (v1 API)."""
    try:
        import numpy as np

        np_vectors = []
        for vec_list in request.vectors:
            np_vec = np.array(vec_list, dtype=np.int8)
            np_vectors.append(np_vec)

        if request.operation == "bind":
            result_vector = store.encoder.mathematical_bind(*np_vectors)
            encoding_type = f"compose_bind_{len(np_vectors)}_vectors"
        elif request.operation == "bundle":
            result_vector = store.encoder.mathematical_bundle(np_vectors)
            encoding_type = f"compose_bundle_{len(np_vectors)}_vectors"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")

        cpu_result = store.vector_manager.to_cpu(result_vector)
        result_list = cpu_result.tolist()

        return {"vector": result_list, "encoding_type": encoding_type}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector composition failed: {e}")
        raise HTTPException(status_code=400, detail=f"Vector composition failed: {str(e)}")


# ============================================================================
# NEW API ENDPOINTS: Full black-box support for advanced VSA/HDC applications
# ============================================================================

@app.delete("/api/v1/items/{item_id}")
async def delete_item_v1(item_id: str):
    """Delete a specific item by ID (v1 API)."""
    try:
        success = store.delete(item_id)
        if not success:
            raise HTTPException(status_code=404, detail="Item not found")
        logger.info(f"Deleted item {item_id}")
        return {"id": item_id, "deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete item failed: {e}")
        raise HTTPException(status_code=400, detail=f"Delete failed: {str(e)}")


@app.post("/api/v1/store/clear")
async def clear_store_v1():
    """Clear all items from the store (v1 API)."""
    try:
        count = len(store.stored_data)
        store.clear()
        logger.info(f"Cleared store ({count} items removed)")
        return {"cleared": True, "items_removed": count}
    except Exception as e:
        logger.error(f"Clear store failed: {e}")
        raise HTTPException(status_code=400, detail=f"Clear failed: {str(e)}")


@app.get("/api/v1/items/{item_id}/vector")
async def get_item_vector_v1(item_id: str):
    """Retrieve the encoded vector for a stored item (v1 API)."""
    try:
        if item_id not in store.stored_vectors:
            raise HTTPException(status_code=404, detail="Item not found")

        vector = store.stored_vectors[item_id]
        cpu_vector = store.vector_manager.to_cpu(vector)
        vector_list = cpu_vector.tolist()

        return {"id": item_id, "vector": vector_list}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get item vector failed: {e}")
        raise HTTPException(status_code=400, detail=f"Get vector failed: {str(e)}")


@app.post("/api/v1/vectors/similarity")
async def compute_similarity_v1(request: SimilarityRequest):
    """Compute cosine similarity between two vectors (v1 API)."""
    try:
        import numpy as np

        v1 = np.array(request.vector_a, dtype=np.float64)
        v2 = np.array(request.vector_b, dtype=np.float64)

        if len(v1) != len(v2):
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimensions must match ({len(v1)} vs {len(v2)})"
            )

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = float(dot / (norm1 * norm2))

        return {"similarity": similarity}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Similarity failed: {str(e)}")


@app.post("/api/v1/search/batch")
async def batch_search_v1(request: BatchSearchRequest):
    """
    Batch search - multiple probes in one request (Qdrant-compatible).

    Returns results for each search in the same order as the input.
    Uses concurrent execution via thread pool for efficiency.

    Design mirrors Qdrant's search_batch:
    - POST /collections/{collection}/points/search/batch
    - Takes list of SearchRequest objects
    - Returns list of result lists
    """
    import concurrent.futures
    import time

    start_time = time.time()

    def execute_single_search(search: BatchSearchItem) -> List[Dict[str, Any]]:
        """Execute a single search synchronously."""
        try:
            top_k = min(max(1, search.top_k), MAX_QUERY_RESULTS)
            threshold = max(0.0, min(1.0, search.threshold))

            results = store.query(
                probe=search.probe,
                data_type=search.data_type,
                top_k=top_k,
                threshold=threshold,
                guard=search.guard,
                negations=search.negations,
                any_marker=search.any_marker,
            )

            return [
                {"id": data_id, "score": score, "data": data}
                for data_id, score, data in results
            ]
        except Exception as e:
            logger.error(f"Batch search item failed: {e}")
            return []

    try:
        # Execute searches concurrently via thread pool
        # This mirrors how Qdrant processes batch searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(execute_single_search, search)
                for search in request.searches
            ]

            all_results = []
            for future in futures:
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Batch search future error: {e}")
                    all_results.append([])

        # Track timing
        elapsed = time.time() - start_time
        query_stats["count"] += len(request.searches)
        query_stats["total_time"] += elapsed

        if query_stats["count"] % 100 == 0:
            avg = query_stats["total_time"] / query_stats["count"]
            logger.info(f"QUERY_STATS: {query_stats['count']} queries, avg={avg*1000:.2f}ms")

        return {
            "results": all_results,
            "count": len(all_results),
            "searches_executed": len(request.searches),
            "elapsed_ms": elapsed * 1000
        }

    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Batch search failed: {str(e)}")


@app.post("/api/v1/search/by-vector")
async def search_by_vector_v1(request: VectorSearchRequest):
    """Search using a raw vector instead of JSON probe (v1 API)."""
    try:
        import numpy as np
        from holon.similarity import find_similar_vectors

        # Validate parameters
        if request.top_k > MAX_QUERY_RESULTS:
            raise HTTPException(
                status_code=400,
                detail=f"top_k exceeds maximum allowed ({MAX_QUERY_RESULTS})"
            )
        if request.top_k < 1:
            raise HTTPException(status_code=400, detail="top_k must be >= 1")

        if not (0.0 <= request.threshold <= 1.0):
            raise HTTPException(
                status_code=400, detail="threshold must be between 0.0 and 1.0"
            )

        # Convert to numpy array
        probe_vector = np.array(request.vector, dtype=np.int8)

        # Use similarity search
        similar_ids_scores = find_similar_vectors(
            probe_vector, store.stored_vectors, request.top_k, request.threshold
        )

        # Apply guards and negations if provided
        results = []
        for data_id, score in similar_ids_scores:
            data_dict = store.stored_data[data_id]

            # Apply guard
            if request.guard and not is_subset(request.guard, data_dict):
                continue

            # Apply negations (simplified)
            if request.negations:
                skip = False
                for key, value in request.negations.items():
                    if isinstance(value, dict) and "$not" in value:
                        not_val = value["$not"]
                        if key in data_dict and data_dict[key] == not_val:
                            skip = True
                            break
                if skip:
                    continue

            results.append({"id": data_id, "score": score, "data": data_dict})

        return {"results": results, "count": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Vector search failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Holon Neural Memory API starting up")
    logger.info(f"Using backend: {store.backend}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Holon Neural Memory API shutting down")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
