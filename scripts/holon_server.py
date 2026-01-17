#!/usr/bin/env python3

"""
Holon HTTP API Server
Provides REST API for VSA/HDC neural memory operations.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from holon import CPUStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Holon Neural Memory API",
    description="VSA/HDC-based neural memory system for structured data",
    version="0.1.0"
)

# Global store instance (ephemeral for now)
store = CPUStore(backend='auto')

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
    top_k: int = Field(DEFAULT_QUERY_RESULTS, description="Number of top results to return")
    threshold: float = Field(0.0, description="Similarity threshold (0-1)")

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Query results with id, score, data")

class InsertResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for inserted data")

class BatchInsertResponse(BaseModel):
    ids: List[str] = Field(..., description="List of unique identifiers")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    backend: str = Field(..., description="Storage backend type")
    items_count: int = Field(..., description="Number of stored items")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        backend=store.backend,
        items_count=len(store.stored_data)
    )

@app.post("/insert", response_model=InsertResponse)
async def insert_item(request: InsertRequest, req: Request):
    """Insert a single data item."""
    try:
        data_id = store.insert(request.data, request.data_type)
        logger.info(f"Inserted item {data_id}")
        return InsertResponse(id=data_id)
    except Exception as e:
        logger.error(f"Insert failed: {e}")
        raise HTTPException(status_code=400, detail=f"Insert failed: {str(e)}")

@app.post("/batch_insert", response_model=BatchInsertResponse)
async def batch_insert_items(request: BatchInsertRequest, req: Request):
    """Insert multiple data items."""
    try:
        ids = []
        for item in request.items:
            data_id = store.insert(item, request.data_type)
            ids.append(data_id)
        logger.info(f"Batch inserted {len(ids)} items")
        return BatchInsertResponse(ids=ids)
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        raise HTTPException(status_code=400, detail=f"Batch insert failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_items(request: QueryRequest, req: Request, res: Response):
    """Query the store with a probe."""
    try:
        # Validate top_k
        if request.top_k > MAX_QUERY_RESULTS:
            raise HTTPException(
                status_code=400,
                detail=f"top_k exceeds maximum allowed ({MAX_QUERY_RESULTS})"
            )
        if request.top_k < 1:
            raise HTTPException(status_code=400, detail="top_k must be >= 1")

        # Validate threshold
        if not (0.0 <= request.threshold <= 1.0):
            raise HTTPException(status_code=400, detail="threshold must be between 0.0 and 1.0")

        # Execute query
        results = store.query(
            request.probe,
            request.data_type,
            request.top_k,
            request.threshold
        )

        # Format response - convert EDN types to JSON-compatible
        def convert_for_json(obj):
            """Convert EDN types to JSON-compatible Python types."""
            if isinstance(obj, dict):
                return {convert_for_json(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, frozenset):
                return list(obj)
            elif hasattr(obj, 'name'):  # EDN Keyword/Symbol
                return obj.name if hasattr(obj, 'name') else str(obj)
            else:
                return obj

        formatted_results = [
            {"id": data_id, "score": score, "data": convert_for_json(data)}
            for data_id, score, data in results
        ]

        logger.info(f"Query returned {len(formatted_results)} results")
        return QueryResponse(results=formatted_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e)}")

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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")