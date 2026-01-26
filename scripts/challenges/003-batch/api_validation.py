#!/usr/bin/env python3
"""
API Validation for Quote Finder - Test vector bootstrapping through HTTP API
Similar to Challenge 4's black-box HTTP validation approach.
"""

import json
import time
import requests
import subprocess
import signal
import os
from typing import Dict, List, Any
from pathlib import Path


class QuoteFinderAPIValidator:
    """API validator for quote finder using Challenge 4's HTTP validation approach."""

    def __init__(self):
        self.server_process = None
        self.api_base = "http://localhost:8000"
        self.test_results = {}

    def start_server(self) -> bool:
        """Start Holon server for API testing (like Challenge 4's HTTP validation)."""
        print("🚀 Starting Holon server for API validation...")

        try:
            # Start server in background
            server_script = Path(__file__).parent.parent / "server" / "holon_server.py"
            self.server_process = subprocess.Popen(
                ["python", str(server_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )

            # Wait for server to start
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{self.api_base}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ Holon server started successfully")
                        return True
                except:
                    pass

                print(f"   Waiting for server... ({attempt + 1}/{max_attempts})")
                time.sleep(1)

            print("❌ Server failed to start")
            return False

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                print("✅ Server stopped")
            except:
                self.server_process.kill()
                print("⚠️  Server force-killed")

    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test all relevant API endpoints for quote finding."""
        results = {
            "health_check": False,
            "encode_endpoint": False,
            "query_endpoint": False,
            "bootstrap_tests": [],
            "search_tests": [],
            "overall_success": False
        }

        print("\n🔍 Testing API Endpoints:")

        # 1. Health check
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            results["health_check"] = response.status_code == 200
            print(f"   Health check: {'✅' if results['health_check'] else '❌'}")
        except Exception as e:
            print(f"   Health check: ❌ ({e})")

        if not results["health_check"]:
            return results

        # First, insert some test data so we have something to query
        print("   Inserting test data...")
        test_data = [
            {"words": {"_encode_mode": "ngram", "sequence": ["everything", "depends", "upon", "relative", "minuteness"]}},
            {"words": {"_encode_mode": "ngram", "sequence": ["integration", "is", "reverse", "of", "differentiation"]}},
            {"words": {"_encode_mode": "ngram", "sequence": ["depends", "on", "relative", "smallness"]}},
        ]

        insert_ids = []
        for i, data in enumerate(test_data):
            try:
                api_request = {
                    "data": json.dumps(data),
                    "data_type": "json"
                }
                response = requests.post(
                    f"{self.api_base}/insert",
                    json=api_request,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    insert_ids.append(result["id"])
                    print(f"     Inserted test data {i+1}: ✅ ({result['id']})")
                else:
                    print(f"     Inserted test data {i+1}: ❌ ({response.status_code})")
            except Exception as e:
                print(f"     Inserted test data {i+1}: ❌ ({e})")

        # 2. Test /encode endpoint (vector bootstrapping)
        print("   Testing /encode endpoint...")
        encode_tests = [
            {"words": {"_encode_mode": "ngram", "sequence": ["everything", "depends", "upon", "relative", "minuteness"]}},
            {"words": {"_encode_mode": "ngram", "sequence": ["integration", "is", "reverse", "of", "differentiation"]}},
            {"words": {"_encode_mode": "ngram", "sequence": ["depends", "on", "relative", "smallness"]}},
        ]

        for i, test_data in enumerate(encode_tests):
            try:
                # API expects data as JSON string in "data" field
                api_request = {
                    "data": json.dumps(test_data),
                    "data_type": "json"
                }
                response = requests.post(
                    f"{self.api_base}/encode",
                    json=api_request,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )

                success = response.status_code == 200
                results["encode_endpoint"] = results["encode_endpoint"] or success

                if success:
                    vector_data = response.json()
                    vector_length = len(vector_data.get("vector", []))
                    print(f"     Test {i+1}: ✅ ({vector_length} dimensions)")
                    results["bootstrap_tests"].append({
                        "test": f"encode_test_{i+1}",
                        "success": True,
                        "vector_length": vector_length
                    })
                else:
                    print(f"     Test {i+1}: ❌ ({response.status_code})")
                    results["bootstrap_tests"].append({
                        "test": f"encode_test_{i+1}",
                        "success": False,
                        "error": response.status_code
                    })

            except Exception as e:
                print(f"     Test {i+1}: ❌ ({e})")
                results["bootstrap_tests"].append({
                    "test": f"encode_test_{i+1}",
                    "success": False,
                    "error": str(e)
                })

        # 3. Test /query endpoint with bootstrapped vectors
        print("   Testing /query endpoint...")
        query_tests = [
            {
                "name": "exact_match",
                "probe": {"words": {"_encode_mode": "ngram", "sequence": ["everything", "depends", "upon", "relative", "minuteness"]}},
                "expected_min_results": 1
            },
            {
                "name": "fuzzy_match",
                "probe": {"words": {"_encode_mode": "ngram", "sequence": ["depends", "on", "relative", "smallness"]}},
                "expected_min_results": 1
            },
            {
                "name": "partial_match",
                "probe": {"words": {"_encode_mode": "ngram", "sequence": ["integration"]}},
                "expected_min_results": 1
            }
        ]

        for test in query_tests:
            try:
                # API expects probe as JSON string in "probe" field
                api_request = {
                    "probe": json.dumps(test["probe"]),
                    "data_type": "json",
                    "top_k": 5
                }
                response = requests.post(
                    f"{self.api_base}/query",
                    json=api_request,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )

                success = response.status_code == 200
                results["query_endpoint"] = results["query_endpoint"] or success

                if success:
                    query_results = response.json()
                    num_results = len(query_results.get("results", []))
                    expected_min = test["expected_min_results"]

                    test_success = num_results >= expected_min
                    print(f"     {test['name']}: {'✅' if test_success else '❌'} ({num_results} results, expected ≥{expected_min})")

                    results["search_tests"].append({
                        "test": test["name"],
                        "success": test_success,
                        "results_count": num_results,
                        "expected_min": expected_min
                    })
                else:
                    print(f"     {test['name']}: ❌ ({response.status_code})")
                    results["search_tests"].append({
                        "test": test["name"],
                        "success": False,
                        "error": response.status_code
                    })

            except Exception as e:
                print(f"     {test['name']}: ❌ ({e})")
                results["search_tests"].append({
                    "test": test["name"],
                    "success": False,
                    "error": str(e)
                })

        # Overall assessment
        bootstrap_success = all(test["success"] for test in results["bootstrap_tests"])
        search_success = all(test["success"] for test in results["search_tests"])

        results["overall_success"] = (
            results["health_check"] and
            results["encode_endpoint"] and
            results["query_endpoint"] and
            bootstrap_success and
            search_success
        )

        return results

    def run_api_validation(self) -> Dict[str, Any]:
        """Run complete API validation like Challenge 4."""
        print("🌐 API VALIDATION: Testing Vector Bootstrapping Over HTTP")
        print("=" * 60)

        # Assume server is already running
        print("✅ Assuming server is already running...")

        # Run tests
        results = self.test_api_endpoints()

        # Print summary
        self._print_api_summary(results)

        return results

    def _print_api_summary(self, results: Dict[str, Any]):
        """Print API validation summary like Challenge 4."""
        print("\n" + "="*60)
        print("🌐 API VALIDATION SUMMARY")
        print("="*60)

        print("🔗 Endpoint Status:")
        print(f"   Health Check: {'✅' if results['health_check'] else '❌'}")
        print(f"   /encode (Bootstrap): {'✅' if results['encode_endpoint'] else '❌'}")
        print(f"   /query (Search): {'✅' if results['query_endpoint'] else '❌'}")

        print("\n🧪 Bootstrap Tests:")
        for test in results["bootstrap_tests"]:
            status = "✅" if test["success"] else "❌"
            if test["success"]:
                print(f"   {test['test']}: {status} ({test['vector_length']}D vector)")
            else:
                print(f"   {test['test']}: {status} ({test.get('error', 'unknown error')})")

        print("\n🔍 Search Tests:")
        for test in results["search_tests"]:
            status = "✅" if test["success"] else "❌"
            if test["success"]:
                print(f"   {test['test']}: {status} ({test['results_count']} results ≥ {test['expected_min']})")
            else:
                print(f"   {test['test']}: {status} ({test.get('error', 'unknown error')})")

        # Challenge 4 style assessment
        if results["overall_success"]:
            assessment = "🎉 EXCELLENT - Full HTTP API compatibility!"
            detail = "Vector bootstrapping works in deployed environment"
        elif results["encode_endpoint"] and results["query_endpoint"]:
            assessment = "✅ GOOD - Core API functions work"
            detail = "Some tests failed but basic functionality confirmed"
        elif results["health_check"]:
            assessment = "⚠️  PARTIAL - Server runs but API issues"
            detail = "Server healthy but endpoint functionality needs work"
        else:
            assessment = "❌ FAILED - Cannot test API"
            detail = "Server startup or basic connectivity issues"

        print(f"\n🏆 Overall Assessment: {assessment}")
        print(f"   {detail}")

        print("\n📊 Challenge 4 Comparison:")
        print("   ✅ HTTP API testing (like Challenge 4)")
        print("   ✅ Black-box validation approach")
        print("   ✅ Deployed environment testing")
        print("   ✅ Bootstrap + search integration")


def main():
    """Run API validation."""
    validator = QuoteFinderAPIValidator()
    results = validator.run_api_validation()

    # Return success status for CI/testing
    return results.get("overall_success", False)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)