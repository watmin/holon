import pytest
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from fastapi.testclient import TestClient
from holon_server import app, store  # Import the FastAPI app and store

client = TestClient(app)


class TestHTTPAPI:
    def setup_method(self, method):
        store.clear()

    def test_insert_json(self):
        data = {"user": "alice", "action": "login"}
        response = client.post("/insert", json={"data": json.dumps(data)})
        assert response.status_code == 200
        result = response.json()
        assert "id" in result
        assert result["id"] is not None

    def test_insert_edn(self):
        data = "{:user \"bob\" :action \"logout\"}"
        response = client.post("/insert", json={"data": data, "data_type": "edn"})
        assert response.status_code == 200
        result = response.json()
        assert "id" in result

    def test_query_without_guard(self):
        # Insert some data first
        data1 = {"user": "alice", "status": "success"}
        data2 = {"user": "bob", "status": "failed"}
        client.post("/insert", json={"data": json.dumps(data1)})
        client.post("/insert", json={"data": json.dumps(data2)})

        # Query
        probe = {"user": "alice"}
        response = client.post("/query", json={"probe": json.dumps(probe), "top_k": 10})
        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert len(result["results"]) >= 1  # At least alice's data

    def test_query_with_guard(self):
        # Insert data
        data1 = {"user": "alice", "status": "success"}
        data2 = {"user": "alice", "status": "failed"}
        client.post("/insert", json={"data": json.dumps(data1)})
        client.post("/insert", json={"data": json.dumps(data2)})

        # Query with guard
        probe = {"user": "alice"}
        guard = {"status": None}  # Presence of status
        response = client.post("/query", json={
            "probe": json.dumps(probe),
            "top_k": 10,
            "guard": json.dumps(guard)
        })
        assert response.status_code == 200
        result = response.json()
        # Should return both since both have status
        assert len(result["results"]) >= 2

    def test_query_with_negations(self):
        # Insert data
        data1 = {"user": "alice", "status": "success"}
        data2 = {"user": "alice", "status": "failed"}
        client.post("/insert", json={"data": json.dumps(data1)})
        client.post("/insert", json={"data": json.dumps(data2)})

        # Query with negations
        probe = {"user": "alice"}
        negations = {"status": "failed"}
        response = client.post("/query", json={
            "probe": json.dumps(probe),
            "top_k": 10,
            "negations": negations
        })
        assert response.status_code == 200
        result = response.json()
        # Should exclude failed
        assert len(result["results"]) >= 1
        for res in result["results"]:
            assert res["data"]["status"] != "failed"



    def test_query_invalid_guard(self):
        probe = {"user": "alice"}
        response = client.post("/query", json={
            "probe": json.dumps(probe),
            "guard": "invalid json"
        })
        assert response.status_code == 400
        assert "Invalid guard" in response.json()["detail"]

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "backend" in data
        assert "items_count" in data