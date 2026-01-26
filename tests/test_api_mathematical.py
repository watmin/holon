"""
Unit tests for mathematical API endpoints.

Tests the HTTP API endpoints that expose mathematical primitives.
"""

import json

import pytest
import requests
from fastapi.testclient import TestClient

from holon import CPUStore
from holon.semantic_encoder import SemanticEncoder
from holon.vector_manager import VectorManager
from scripts.server.holon_server import app


class TestMathematicalAPIEndpoints:
    """Test mathematical API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with semantic encoder."""
        # Override the global store with semantic-enabled store
        vector_manager = VectorManager(dimensions=1000)
        semantic_encoder = SemanticEncoder(vector_manager)
        global_store = CPUStore(dimensions=1000)
        global_store.encoder = semantic_encoder

        # Replace the global store in the app
        import scripts.server.holon_server as server_module

        server_module.store = global_store

        with TestClient(app) as test_client:
            yield test_client

    def test_encode_mathematical_primitive_endpoint(self, client):
        """Test mathematical primitive encoding via /encode/mathematical endpoint."""
        response = client.post(
            "/encode/mathematical", json={"primitive": "convergence_rate", "value": 0.8}
        )

        assert response.status_code == 200
        data = response.json()
        assert "vector" in data
        assert "encoding_type" in data
        assert data["encoding_type"] == "mathematical_convergence_rate"
        assert isinstance(data["vector"], list)
        assert len(data["vector"]) > 0

    def test_mathematical_bind_endpoint(self, client):
        """Test mathematical binding via /encode/compose endpoint."""
        # First create some vectors
        vec1_response = client.post(
            "/encode/mathematical", json={"primitive": "convergence_rate", "value": 0.8}
        )
        vec1 = vec1_response.json()["vector"]

        vec2_response = client.post(
            "/encode/mathematical",
            json={"primitive": "iteration_complexity", "value": 50},
        )
        vec2 = vec2_response.json()["vector"]

        # Bind them via compose endpoint
        bind_response = client.post(
            "/encode/compose", json={"operation": "bind", "vectors": [vec1, vec2]}
        )

        assert bind_response.status_code == 200
        data = bind_response.json()
        assert "vector" in data
        assert "encoding_type" in data
        assert "compose_bind_2_vectors" in data["encoding_type"]
        assert isinstance(data["vector"], list)
        assert len(data["vector"]) > 0

    def test_mathematical_bundle_endpoint(self, client):
        """Test mathematical bundling via /encode/compose endpoint."""
        # Create vectors
        vec1_response = client.post(
            "/encode/mathematical", json={"primitive": "frequency_domain", "value": 2.5}
        )
        vec1 = vec1_response.json()["vector"]

        vec2_response = client.post(
            "/encode/mathematical", json={"primitive": "amplitude_scale", "value": 0.8}
        )
        vec2 = vec2_response.json()["vector"]

        # Bundle them via compose endpoint
        bundle_response = client.post(
            "/encode/compose", json={"operation": "bundle", "vectors": [vec1, vec2]}
        )

        assert bundle_response.status_code == 200
        data = bundle_response.json()
        assert "vector" in data
        assert "encoding_type" in data
        assert "compose_bundle_2_vectors" in data["encoding_type"]
        assert isinstance(data["vector"], list)

    def test_invalid_mathematical_primitive(self, client):
        """Test invalid primitive returns error via mathematical endpoint."""
        response = client.post(
            "/encode/mathematical",
            json={"primitive": "invalid_primitive", "value": 1.0},
        )

        assert response.status_code == 400
        assert "Unknown mathematical primitive" in response.json()["detail"]

    def test_mathematical_bind_empty_vectors(self, client):
        """Test mathematical bind with empty vector list via compose endpoint."""
        response = client.post(
            "/encode/compose", json={"operation": "bind", "vectors": []}
        )

        # Should succeed and return zero vector (handled gracefully)
        assert response.status_code == 200
        data = response.json()
        assert "vector" in data
        # Empty bind should return zero vector
        assert all(v == 0 for v in data["vector"])

    def test_all_mathematical_primitives(self, client):
        """Test all available mathematical primitives via unified endpoint."""
        primitives = [
            "convergence_rate",
            "iteration_complexity",
            "frequency_domain",
            "amplitude_scale",
            "power_law_exponent",
            "clustering_coefficient",
            "topological_distance",
            "self_similarity",
        ]

        for primitive in primitives:
            response = client.post(
                "/encode/mathematical", json={"primitive": primitive, "value": 1.0}
            )

            assert response.status_code == 200, f"Failed for {primitive}"
            data = response.json()
            assert "vector" in data, f"No vector returned for {primitive}"
            assert "encoding_type" in data, f"No encoding type for {primitive}"
            assert len(data["vector"]) > 0, f"Empty vector for {primitive}"


class TestMathematicalAPIIntegration:
    """Test mathematical API integration with data storage."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with TestClient(app) as test_client:
            yield test_client

    def test_store_and_query_with_mathematical_properties(self, client):
        """Test storing data with mathematical properties and querying."""
        # Create test data with mathematical properties
        test_data = {
            "matrix-id": "api_test_matrix",
            "rule": "fractal",
            "panels": {"row1-col1": {"shapes": ["circle"], "count": 1}},
            "mathematical_properties": {
                "convergence_rate": 0.8,
                "iteration_complexity": 45,
                "self_similarity": 0.85,
            },
        }

        # Store it
        store_response = client.post(
            "/insert", json={"data": json.dumps(test_data), "data_type": "json"}
        )
        assert store_response.status_code == 200

        # Query for fractal matrices
        query_response = client.post(
            "/query",
            json={"probe": '{"rule": "fractal"}', "data_type": "json", "top_k": 5},
        )
        assert query_response.status_code == 200

        results = query_response.json()["results"]
        assert len(results) >= 1

        # Check that mathematical properties are preserved
        found_matrix = None
        for result in results:
            if result["data"].get("matrix-id") == "api_test_matrix":
                found_matrix = result["data"]
                break

        assert found_matrix is not None
        assert "mathematical_properties" in found_matrix
        assert found_matrix["mathematical_properties"]["convergence_rate"] == 0.8
