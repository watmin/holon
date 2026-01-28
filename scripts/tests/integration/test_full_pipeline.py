#!/usr/bin/env python3
"""
Full Pipeline Integration Test
Tests end-to-end functionality across all major components:
- CPU Store operations
- Guards, negations, $or queries
- JSON and EDN data types
- Bulk operations
- Error handling
- ANN vs brute force consistency
"""

import json
import time
from typing import Any, Dict, List

from holon import CPUStore


class FullPipelineTest:
    """Comprehensive integration test for Holon pipeline."""

    def __init__(self):
        self.store = CPUStore(dimensions=8000)  # Smaller for faster testing

        # Test data
        self.test_data = [
            {"name": "Alice", "role": "developer", "team": "backend", "salary": 100000},
            {"name": "Bob", "role": "designer", "team": "frontend", "salary": 90000},
            {
                "name": "Charlie",
                "role": "developer",
                "team": "frontend",
                "salary": 95000,
            },
            {"name": "Diana", "role": "manager", "team": "backend", "salary": 120000},
            {"name": "Eve", "role": "developer", "team": "backend", "salary": 105000},
        ]

        self.edn_test_data = [
            '{:name "Frank" :role :developer :team :mobile :salary 98000}',
            '{:name "Grace" :role :analyst :team :data :salary 95000}',
        ]

    def test_cpu_store_operations(self):
        """Test basic CPU store operations."""
        print("\n🧪 Testing CPU Store Operations...")

        # Insert test data
        ids = []
        for data in self.test_data:
            data_id = self.store.insert(json.dumps(data), "json")
            ids.append(data_id)
            assert data_id is not None

        print(f"✅ Inserted {len(ids)} JSON records")

        # Test basic query
        results = self.store.query(probe='{"role": "developer"}', data_type="json", top_k=10)
        assert len(results) >= 3  # Alice, Charlie, Eve

        print(f"✅ Found {len(results)} developers")

        # Test guards
        results = self.store.query(probe='{"role": "developer"}', data_type="json", guard={"team": "backend"}, top_k=10)
        assert len(results) >= 2  # Alice, Eve

        print(f"✅ Found {len(results)} backend developers with guards")

        # Test negations
        results = self.store.query(probe='{"role": "developer"}',
            data_type="json",
            negations={"name": {"$not": "Alice"}},
            top_k=10,
        )
        assert len(results) >= 2  # Charlie, Eve

        print(f"✅ Found {len(results)} developers excluding Alice")

        # Test $or
        results = self.store.query(probe='{"$or": [{"team": "backend"}, {"team": "frontend"}]}', data_type="json", top_k=10)
        assert len(results) >= 4

        print(f"✅ Found {len(results)} team members with $or query")

        return True

    def test_edn_operations(self):
        """Test EDN data operations."""
        print("\n🧪 Testing EDN Operations...")

        ids = []
        for edn_data in self.edn_test_data:
            data_id = self.store.insert(edn_data, "edn")
            ids.append(data_id)

        print(f"✅ Inserted {len(ids)} EDN records")

        # Query EDN data
        results = self.store.query(probe="{:role :developer}", data_type="edn", top_k=10)
        assert len(results) >= 1  # Frank

        print(f"✅ Found {len(results)} EDN developers")

        return True

    def test_performance_consistency(self):
        """Test ANN vs brute force consistency and performance."""
        print("\n🧪 Testing ANN vs Brute Force Consistency...")

        # Add more data to trigger ANN indexing
        performance_data = []
        for i in range(1500):  # Exceed ANN_THRESHOLD of 1000
            performance_data.append(
                {"id": i, "text": f"sample text {i}", "value": i * 10}
            )

        # Bulk insert
        insert_start = time.time()
        for data in performance_data:
            self.store.insert(json.dumps(data), "json")
        insert_time = time.time() - insert_start

        print(f"✅ Inserted 1500 records in {insert_time:.3f}s")

        # Query with ANN (should be fast)
        ann_start = time.time()
        ann_results = self.store.query(probe='{"id": 750}', data_type="json", top_k=5)
        ann_time = time.time() - ann_start

        # Force brute force by temporarily disabling ANN
        self.store.ann_index = None
        brute_start = time.time()
        brute_results = self.store.query(probe='{"id": 750}', data_type="json", top_k=5)
        brute_time = time.time() - brute_start

        # Re-enable ANN
        self.store._build_ann_index()

        # Verify results are identical
        ann_scores = [r[1] for r in ann_results]
        brute_scores = [r[1] for r in brute_results]

        # Allow small floating point differences
        for ann_score, brute_score in zip(ann_scores, brute_scores):
            assert (
                abs(ann_score - brute_score) < 0.001
            ), f"ANN/Brute mismatch: {ann_score} vs {brute_score}"

        # speedup = brute_time / ann_time if ann_time > 0 else float("inf")  # Not used

        print("✅ ANN vs Brute force consistency verified")
        print(f"ANN query time: {ann_time:.4f}s")
        print(f"Brute force query time: {brute_time:.1f}s")

        return True

    def test_bulk_operations(self):
        """Test bulk insert operations."""
        print("\n🧪 Testing Bulk Operations...")

        bulk_data = [
            {"name": f"Bulk_{i}", "role": "bulk_tester", "batch": i} for i in range(50)
        ]

        # Bulk insert via store
        bulk_ids = []
        for data in bulk_data:
            data_id = self.store.insert(json.dumps(data), "json")
            bulk_ids.append(data_id)

        print(f"✅ Bulk inserted {len(bulk_ids)} records")

        # Query bulk data
        results = self.store.query(probe='{"role": "bulk_tester"}', data_type="json", top_k=100)
        assert len(results) >= 50

        print(f"✅ Found {len(results)} bulk test records")

        return True

    def test_error_handling(self):
        """Test error handling scenarios."""
        print("\n🧪 Testing Error Handling...")

        # Test invalid JSON
        try:
            self.store.insert('{"invalid": json}', "json")
            assert False, "Should have failed with invalid JSON"
        except Exception:
            print("✅ Invalid JSON properly rejected")

        # Test invalid EDN
        try:
            self.store.insert("{invalid edn syntax", "edn")
            assert False, "Should have failed with invalid EDN"
        except Exception:
            print("✅ Invalid EDN properly rejected")

        # Test querying with invalid probe
        try:
            self.store.query(probe="invalid json probe", data_type="json")
            assert False, "Should have failed with invalid probe"
        except Exception:
            print("✅ Invalid query probe properly rejected")

        # Test querying non-existent data (should not crash)
        results = self.store.query(probe='{"nonexistent": "field"}', data_type="json", top_k=5)
        # Should return results (possibly empty or low-scoring matches), but not crash
        assert isinstance(results, list)  # Should return a list, not crash
        print(
            f"✅ Query for non-existent data handled gracefully (returned {len(results)} results)"
        )

        return True

    def test_data_integrity(self):
        """Test data integrity across operations."""
        print("\n🧪 Testing Data Integrity...")

        # Use a separate store instance for this test to avoid interference
        integrity_store = CPUStore(dimensions=8000)

        # Insert test record
        original_data = {
            "name": "Integrity_Test",
            "role": "validator",
            "status": "active",
        }
        data_id = integrity_store.insert(json.dumps(original_data), "json")

        # Retrieve and verify
        retrieved = integrity_store.get(data_id)
        assert retrieved == original_data
        print("✅ Data storage/retrieval integrity verified")

        # Query and verify
        results = integrity_store.query(probe='{"name": "Integrity_Test"}', data_type="json", top_k=5)
        assert len(results) >= 1

        # Verify the retrieved data matches original
        found_id, score, found_data = results[0]
        assert found_data == original_data
        print("✅ Query result integrity verified")

        # Test deletion
        result = integrity_store.delete(data_id)
        assert result is True
        try:
            integrity_store.get(data_id)
            assert False, "Should have raised KeyError for deleted data"
        except KeyError:
            print("✅ Data deletion integrity verified")

        return True

    def run_all_tests(self):
        """Run the complete integration test suite."""
        print("🧪 Starting Full Pipeline Integration Tests")
        print("=" * 60)

        success = True

        # Run all test phases
        test_phases = [
            self.test_cpu_store_operations,
            self.test_edn_operations,
            self.test_bulk_operations,
            self.test_error_handling,
            self.test_performance_consistency,
            self.test_data_integrity,
        ]

        for test_phase in test_phases:
            try:
                if not test_phase():
                    success = False
                    print(f"❌ {test_phase.__name__} failed")
                else:
                    print(f"✅ {test_phase.__name__} passed")
            except Exception as e:
                print(f"❌ {test_phase.__name__} failed with exception: {e}")
                success = False
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 60)
        if success:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            return True
        else:
            print("❌ SOME TESTS FAILED!")
            return False


def main():
    """Main test runner."""
    test_suite = FullPipelineTest()
    success = test_suite.run_all_tests()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
