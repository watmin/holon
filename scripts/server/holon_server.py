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




@app.get("/api/v1/health")
async def health_v1():
    """Health check with v1 API structure."""
    return HealthResponse(
        status="healthy",
        backend=store.backend,
        items_count=len(store.stored_data)
    )


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


@app.post("/api/v1/search")
async def search_items_v1(request: QueryRequest):
    """Search items using vector similarity (v1 API)."""
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
            request.probe,
            request.data_type,
            request.top_k,
            request.threshold,
            guard=request.guard,
            negations=request.negations,
            any_marker=request.any_marker,
        )

        # Format response
        formatted_results = [
            {"id": data_id, "score": score, "data": data}
            for data_id, score, data in results
        ]

        return {"results": formatted_results, "count": len(formatted_results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Search failed: {str(e)}")


@app.post("/api/v1/vectors/encode")
async def encode_data_v1(request: EncodeRequest):
    """Encode structured data to vector (v1 API)."""
    try:
        parsed = parse_data(request.data, request.data_type)
        encoded_vector = store.encoder.encode_data(parsed)
        cpu_vector = store.vector_manager.to_cpu(encoded_vector)
        vector_list = cpu_vector.tolist()

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
