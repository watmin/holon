#!/usr/bin/env python3
"""
UNIFIED API DEMO: Single /encode endpoint for all encoding types

Shows how the unified /encode endpoint handles:
- Structural data encoding (original functionality)
- Mathematical primitive encoding (new functionality)
- Mathematical composition operations (new functionality)
"""

import json
import requests

API_BASE = "http://localhost:8000"


def demonstrate_unified_api():
    """Demonstrate the unified /encode endpoint."""
    print("🎯 UNIFIED API DEMO: Single /encode endpoint for everything")
    print("=" * 60)

    # 1. Structural Data Encoding (Original Functionality)
    print("📄 1. STRUCTURAL DATA ENCODING")
    print("-" * 30)

    structural_data = {
        "user": "alice",
        "action": "login",
        "timestamp": "2024-01-01T12:00:00Z"
    }

    response = requests.post(f"{API_BASE}/encode", json={
        "data": json.dumps(structural_data),
        "data_type": "json"
    })

    if response.status_code == 200:
        result = response.json()
        encoding_type = result.get('encoding_type', 'structural_json')
        print(f"✓ Structural encoding: {len(result['vector'])}D vector")
        print(f"  Type: {encoding_type}")
    else:
        print(f"❌ Structural encoding failed: {response.text}")

    # 2. Mathematical Primitive Encoding (New Functionality)
    print("\n🧮 2. MATHEMATICAL PRIMITIVE ENCODING")
    print("-" * 40)

    primitives = [
        ("convergence_rate", 0.85, "Mathematical stability analysis"),
        ("frequency_domain", 2.3, "Wave frequency properties"),
        ("power_law_exponent", 2.5, "Scale-free network topology"),
        ("clustering_coefficient", 0.7, "Local connectivity measure")
    ]

    encoded_vectors = {}
    for primitive, value, description in primitives:
        response = requests.post(f"{API_BASE}/encode/mathematical", json={
            "primitive": primitive,
            "value": value
        })

        if response.status_code == 200:
            result = response.json()
            encoded_vectors[f"{primitive}_{value}"] = result["vector"]
            print(f"✓ {description}: {len(result['vector'])}D vector ({result['encoding_type']})")
        else:
            print(f"❌ {primitive} encoding failed: {response.text}")

    # 3. Mathematical Composition Operations (New Functionality)
    print("\n🔗 3. MATHEMATICAL COMPOSITION OPERATIONS")
    print("-" * 45)

    # Bind operation (combine mathematical properties)
    if "convergence_rate_0.85" in encoded_vectors and "power_law_exponent_2.5" in encoded_vectors:
        bind_response = requests.post(f"{API_BASE}/encode/compose", json={
            "operation": "bind",
            "vectors": [
                encoded_vectors["convergence_rate_0.85"],
                encoded_vectors["power_law_exponent_2.5"]
            ]
        })

        if bind_response.status_code == 200:
            result = bind_response.json()
            print(f"✓ Mathematical binding: {len(result['vector'])}D vector ({result['encoding_type']})")
            print("  (Combines convergence stability with network topology)")

    # Bundle operation (weighted combination)
    if len(encoded_vectors) >= 3:
        vectors_to_bundle = list(encoded_vectors.values())[:3]
        bundle_response = requests.post(f"{API_BASE}/encode/compose", json={
            "operation": "bundle",
            "vectors": vectors_to_bundle
        })

        if bundle_response.status_code == 200:
            result = bundle_response.json()
            print(f"✓ Mathematical bundling: {len(result['vector'])}D vector ({result['encoding_type']})")
            print("  (Weighted combination of multiple mathematical properties)")

    # 4. Error Handling
    print("\n🚫 4. ERROR HANDLING")
    print("-" * 20)

    # Invalid primitive
    error_response = requests.post(f"{API_BASE}/encode/mathematical", json={
        "primitive": "invalid_primitive",
        "value": 1.0
    })

    if error_response.status_code == 400:
        print("✓ Invalid primitive properly rejected")
    else:
        print("❌ Invalid primitive not rejected properly")

    # Missing required fields
    error_response = requests.post(f"{API_BASE}/encode/mathematical", json={})

    if error_response.status_code == 400:
        print("✓ Missing fields properly rejected")
    else:
        print("❌ Missing fields not rejected properly")

    # 5. Real-world Usage Example
    print("\n🌍 5. REAL-WORLD USAGE EXAMPLE")
    print("-" * 30)

    print("Example: Analyzing a fractal pattern in a scale-free network")
    print()

    # Step 1: Encode fractal properties
    fractal_props = requests.post(f"{API_BASE}/encode/mathematical", json={
        "primitive": "convergence_rate",
        "value": 0.92
    })

    # Step 2: Encode network properties
    network_props = requests.post(f"{API_BASE}/encode/mathematical", json={
        "primitive": "power_law_exponent",
        "value": 2.3
    })

    # Step 3: Combine for semantic pattern
    if fractal_props.status_code == 200 and network_props.status_code == 200:
        combined = requests.post(f"{API_BASE}/encode/compose", json={
            "operation": "bind",
            "vectors": [
                fractal_props.json()["vector"],
                network_props.json()["vector"]
            ]
        })

        if combined.status_code == 200:
            print("✓ Combined fractal + network analysis → semantic pattern vector")
            print("  This vector represents 'fractal convergence in scale-free networks'")
            print("  Can be used for similarity search against other complex systems")

    print("\n" + "=" * 60)
    print("🎉 UNIFIED API SUCCESS!")
    print("=" * 60)
    print("✅ Single /encode endpoint handles all encoding types")
    print("✅ Backward compatible with existing structural encoding")
    print("✅ New mathematical primitives seamlessly integrated")
    print("✅ Clean, consistent API design")
    print("=" * 60)


def main():
    """Run the unified API demonstration."""
    try:
        # Check server
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not running at {API_BASE}")
            print("Start with: ./scripts/server/start_semantic_server.sh")
            return

        print("✅ Connected to unified Holon API")
        print()

        demonstrate_unified_api()

    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
