import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "server"))
import pytest
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
        data = '{:user "bob" :action "logout"}'
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
        guard = {"status": "success"}  # Exact match for success
        response = client.post(
            "/query",
            json={
                "probe": json.dumps(probe),
                "top_k": 10,
                "guard": guard,  # Send as dict, not JSON string
            },
        )
        assert response.status_code == 200
        result = response.json()
        # Should return only the success result
        assert len(result["results"]) >= 1
        for res in result["results"]:
            assert res["data"]["status"] == "success"

    def test_query_with_negations(self):
        # Insert data
        data1 = {"user": "alice", "status": "success"}
        data2 = {"user": "alice", "status": "failed"}
        client.post("/insert", json={"data": json.dumps(data1)})
        client.post("/insert", json={"data": json.dumps(data2)})

        # Query with negations
        probe = {"user": "alice"}
        negations = {"status": "failed"}
        response = client.post(
            "/query",
            json={"probe": json.dumps(probe), "top_k": 10, "negations": negations},
        )
        assert response.status_code == 200
        result = response.json()
        # Should exclude failed
        assert len(result["results"]) >= 1
        for res in result["results"]:
            assert res["data"]["status"] != "failed"

    def test_query_invalid_guard(self):
        probe = {"user": "alice"}
        # Send invalid type for guard (should be dict or None)
        response = client.post(
            "/query",
            json={
                "probe": json.dumps(probe),
                "guard": "invalid json",  # Pydantic will reject this
            },
        )
        # Pydantic validation error
        assert response.status_code == 422

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "backend" in data
        assert "items_count" in data

    def test_encode_endpoint(self):
        """Test the new /encode endpoint for vector bootstrapping"""
        # Test encoding JSON data
        data = {
            "words": {
                "_encode_mode": "ngram",
                "sequence": ["test", "vector", "bootstrap"],
            },
            "metadata": {"type": "test_data"},
        }

        response = client.post("/encode", json={"data": json.dumps(data)})
        assert response.status_code == 200
        result = response.json()
        assert "vector" in result
        assert isinstance(result["vector"], list)
        assert len(result["vector"]) == 16000  # Default dimensions

        # Verify vector contains valid bipolar values
        vector = result["vector"]
        assert all(val in [-1, 0, 1] for val in vector[:10])  # Check first 10 values

    def test_encode_endpoint_edn(self):
        """Test the /encode endpoint with EDN data"""
        edn_data = '{:words {:_encode_mode "chained" :sequence ["edn" "test"]} :type "edn_test"}'

        response = client.post("/encode", json={"data": edn_data, "data_type": "edn"})
        assert response.status_code == 200
        result = response.json()
        assert "vector" in result
        assert len(result["vector"]) == 16000

    def test_encode_invalid_data(self):
        """Test /encode endpoint with invalid data"""
        response = client.post("/encode", json={"data": "invalid json{"})
        assert response.status_code == 400
        assert "Encode failed" in response.json()["detail"]
