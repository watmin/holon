#!/usr/bin/env python3
"""
HTTP-adapted Bug Report Memory & Duplicate Finder

This demonstrates how Challenge 3 (Bug Reports) can work with Holon's HTTP API.
Instead of local CPUStore, it uses HTTP requests to a running Holon server.
"""

import json
import time
from typing import Any, Dict, List, Tuple

import requests


class HTTPBugReportStore:
    """HTTP client version of BugReportStore that works with Holon server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.bug_reports = (
            {}
        )  # Local cache for demo (in production, server would store this)

    def insert_bug_report(self, bug_report: Dict[str, Any]) -> str:
        """Insert a bug report via HTTP API."""
        # Prepare data for JSON serialization
        json_ready_bug = self._prepare_for_json(bug_report)

        # HTTP POST to /insert endpoint
        response = requests.post(
            f"{self.base_url}/insert",
            json={"data": json.dumps(json_ready_bug), "data_type": "json"},
        )

        if response.status_code != 200:
            raise Exception(f"HTTP insert failed: {response.text}")

        result = response.json()
        vector_id = result["id"]

        # Cache locally for this demo
        self.bug_reports[vector_id] = bug_report

        return vector_id

    def find_similar_bugs(
        self, probe_bug: Dict[str, Any], top_k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict]]:
        """Find bugs similar to the probe via HTTP API."""
        json_probe = json.dumps(self._prepare_for_json(probe_bug))

        # HTTP POST to /query endpoint
        response = requests.post(
            f"{self.base_url}/query",
            json={
                "probe": json_probe,
                "data_type": "json",
                "top_k": top_k,
                "threshold": threshold,
            },
        )

        if response.status_code != 200:
            raise Exception(f"HTTP query failed: {response.text}")

        results = response.json()["results"]

        # Return with original bug report data from cache
        return [
            (r["id"], r["score"], self.bug_reports.get(r["id"], r["data"]))
            for r in results
        ]

    def query_with_filters(
        self,
        probe: Dict[str, Any] = None,
        guard: Dict[str, Any] = None,
        negations: Dict[str, Any] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict]]:
        """Advanced query with guards and negations via HTTP."""
        json_probe = json.dumps(self._prepare_for_json(probe or {}))

        # HTTP POST to /query with advanced parameters
        query_payload = {
            "probe": json_probe,
            "data_type": "json",
            "top_k": top_k,
            "threshold": 0.0,
        }

        # Add guard as JSON string if provided
        if guard:
            query_payload["guard"] = json.dumps(guard)

        # Note: Current HTTP API doesn't support negations yet, but could be added
        if negations:
            print(f"⚠️  Negations not yet supported in HTTP API: {negations}")

        response = requests.post(f"{self.base_url}/query", json=query_payload)

        if response.status_code != 200:
            raise Exception(f"HTTP query failed: {response.text}")

        results = response.json()["results"]

        return [
            (r["id"], r["score"], self.bug_reports.get(r["id"], r["data"]))
            for r in results
        ]

    def _prepare_for_json(self, data: Any) -> Any:
        """Convert sets to lists for JSON serialization."""
        if isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        else:
            return data


def demo_http_bug_reports():
    """Demonstrate HTTP-adapted bug report system."""
    print("🐛 HTTP Bug Report Memory & Duplicate Finder")
    print("Using Holon HTTP API instead of local CPUStore")
    print("=" * 55)
    print()

    # Check if server is running
    try:
        health_response = requests.get(
            f"{HTTPBugReportStore().base_url}/health", timeout=2
        )
        if health_response.status_code != 200:
            print(
                "❌ Holon server not running. Start with: python scripts/server/holon_server.py"
            )
            return
        print("✅ Holon server is running")
    except Exception:
        print(
            "❌ Cannot connect to Holon server. Start with: python scripts/server/holon_server.py"
        )
        return

    # Initialize HTTP client
    bug_store = HTTPBugReportStore()

    # Insert sample bug report via HTTP
    sample_bug = {
        "id": "http-demo-001",
        "title": "Login crash on mobile Safari",
        "component": ":auth",
        "severity": ":critical",
        "stacktrace": "TypeError: Cannot read property 'auth' of null",
        "environment": {"os": ":ios", "browser": ":safari", "version": "14.0.1"},
        "labels": {"blocking", "mobile", "regression"},
        "reported_at": "2024-01-15T10:30:00Z",
    }

    print("📤 Inserting bug report via HTTP...")
    bug_id = bug_store.insert_bug_report(sample_bug)
    print(f"✅ Inserted with ID: {bug_id}")

    # Query via HTTP
    print("🔍 Querying similar bugs via HTTP...")
    probe = {
        "title": "crash on login",
        "component": ":auth",
        "environment": {"browser": ":safari"},
    }

    results = bug_store.find_similar_bugs(probe, top_k=3)
    print(f"📥 Found {len(results)} similar bugs:")
    for i, (bug_id, score, bug) in enumerate(results[:3], 1):
        print(f"  {i}. Score: {score:.3f}")

    print()
    print("✨ HTTP integration working! Our solutions can be adapted to use:")
    print("   • REST API instead of local CPUStore")
    print("   • Network calls for insert/query operations")
    print("   • Distributed Holon servers")
    print("   • Load balancing and scaling")


if __name__ == "__main__":
    demo_http_bug_reports()
