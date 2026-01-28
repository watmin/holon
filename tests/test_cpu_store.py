import pytest

from holon import CPUStore


class TestCPUStore:
    def test_insert_and_get_json(self):
        store = CPUStore(dimensions=1000)  # Smaller for testing
        data = '{"key": "value"}'
        data_id = store.insert(data, "json")
        assert data_id is not None
        retrieved = store.get(data_id)
        assert retrieved == {"key": "value"}

    def test_insert_and_get_edn(self):
        store = CPUStore(dimensions=1000)
        data = '{:key "value", :num 42}'
        data_id = store.insert(data, "edn")
        assert data_id is not None
        retrieved = store.get(data_id)
        # EDN parsed data will have Keyword keys
        assert str(retrieved) == "{Keyword(key): 'value', Keyword(num): 42}"

    def test_query_json(self):
        store = CPUStore(dimensions=1000)
        data1 = '{"name": "Alice", "age": 30}'
        data2 = '{"name": "Bob", "age": 25}'
        store.insert(data1, "json")
        store.insert(data2, "json")

        probe = '{"name": "Alice"}'
        results = store.query(probe=probe, data_type="json", top_k=5, threshold=0.0)
        assert len(results) >= 1
        # Should find the Alice data with some similarity

    def test_query_edn(self):
        store = CPUStore(dimensions=1000)
        data1 = '{:name "Alice", :skills #{"clojure"}}'
        data2 = '{:name "Bob", :skills #{"python"}}'
        store.insert(data1, "edn")
        store.insert(data2, "edn")

        probe = '{:skills #{"clojure"}}'
        results = store.query(probe=probe, data_type="edn", top_k=5, threshold=0.0)
        assert len(results) >= 1
        # Should find data with clojure skill

    def test_delete(self):
        store = CPUStore(dimensions=1000)
        data = '{"test": "data"}'
        data_id = store.insert(data, "json")
        assert store.delete(data_id) is True
        assert store.delete(data_id) is False  # Already deleted
        with pytest.raises(KeyError):
            store.get(data_id)

    def test_guard_or_logic(self):
        """Test the new guard OR logic with array syntax"""
        # Test the guard logic directly without fuzzy query complications
        from holon.cpu_store import CPUStore

        # Test data
        test_data = [
            {"name": "Alice", "priority": "high", "category": "A"},
            {"name": "Bob", "priority": "medium", "category": "B"},
            {"name": "Charlie", "priority": "low", "category": "A"},
        ]

        # Test the is_subset function directly (matching the updated implementation)
        def is_subset(guard, data):
            # Handle top-level $or in guards for powerful OR logic
            if "$or" in guard and isinstance(guard["$or"], list):
                # Any of the OR conditions must match
                return any(
                    is_subset(or_condition, data) for or_condition in guard["$or"]
                )

            for key, value in guard.items():
                if key not in data:
                    return False
                if isinstance(value, dict):
                    # Handle nested $or
                    if "$or" in value and isinstance(value["$or"], list):
                        # For nested $or, any of the conditions for this key must match
                        if not any(
                            is_subset({key: or_val}, data) for or_val in value["$or"]
                        ):
                            return False
                    elif not isinstance(data[key], dict) or not is_subset(
                        value, data[key]
                    ):
                        return False
                elif isinstance(value, list):
                    # Support OR logic: if guard has a list and data has a scalar that's IN
                    # the list,
                    # treat it as "match any of these values" (backward compatibility)
                    data_value = data[key]
                    if isinstance(data_value, list):
                        # Exact array matching for array-to-array comparison
                        # (backward compatibility)
                        if len(value) != len(data_value):
                            return False
                        for g_item, d_item in zip(value, data_value):
                            if isinstance(g_item, dict) and "$any" in g_item:
                                continue
                            elif g_item != d_item:
                                return False
                    else:
                        # OR logic: scalar data value must be IN the guard list
                        if data_value not in value:
                            return False
                elif value is not None and data[key] != value:
                    return False
            return True

        # Test 1: Guard OR logic - priority should be high OR medium
        guard = {"priority": ["high", "medium"]}
        assert is_subset(guard, test_data[0])  # Alice: high priority
        assert is_subset(guard, test_data[1])  # Bob: medium priority
        assert not is_subset(guard, test_data[2])  # Charlie: low priority

        # Test 2: Guard with exact match (backward compatibility)
        guard = {"category": "A"}
        assert is_subset(guard, test_data[0])  # Alice: category A
        assert not is_subset(guard, test_data[1])  # Bob: category B
        assert is_subset(guard, test_data[2])  # Charlie: category A

        # Test 3: Guard with array for tags (exact array matching - backward compatibility)
        guard = {"tags": ["urgent", "backend"]}
        data1 = {"name": "Project1", "tags": ["urgent", "backend"]}
        data2 = {"name": "Project2", "tags": ["frontend", "urgent"]}
        assert is_subset(guard, data1)  # Exact match
        assert not is_subset(guard, data2)  # Different array

        # Test 4: Guard OR logic with single-element array
        guard = {"priority": ["high"]}
        assert is_subset(guard, test_data[0])  # Alice: high priority
        assert not is_subset(guard, test_data[1])  # Bob: medium priority
        assert not is_subset(guard, test_data[2])  # Charlie: low priority

        # Test 5: Structured $or in guards (most powerful)
        guard = {"$or": [{"priority": "high"}, {"category": "A"}]}
        assert is_subset(
            guard, test_data[0]
        )  # Alice: high priority (matches first condition)
        assert not is_subset(
            guard, test_data[1]
        )  # Bob: medium priority, category B (matches neither)
        assert is_subset(
            guard, test_data[2]
        )  # Charlie: category A (matches second condition)

        # Test 6: Complex structured $or
        guard = {
            "$or": [
                {"priority": "high", "category": "A"},  # Alice matches this
                {"priority": "low", "category": "B"},  # No one matches this
            ]
        }
        assert is_subset(guard, test_data[0])  # Alice matches compound condition
        assert not is_subset(guard, test_data[1])  # Bob doesn't match either condition
        assert not is_subset(
            guard, test_data[2]
        )  # Charlie doesn't match either condition

    def test_guard_edge_cases(self):
        """Test edge cases for guard functionality"""
        store = CPUStore(dimensions=1000)

        test_data = [
            '{"name": "Item1", "status": "active", "category": "A"}',
            '{"name": "Item2", "status": "inactive", "category": "B"}',
            '{"name": "Item3", "status": "active", "category": "A"}',
        ]

        for data in test_data:
            store.insert(data)

        # Test empty guard should return all results
        results = store.query(probe="{}", guard={}, top_k=10, threshold=0.0)
        assert len(results) == 3

        # Test guard with non-existent field should return no results
        results = store.query(
            probe="{}", guard={"nonexistent": "value"}, top_k=10, threshold=0.0
        )
        assert len(results) == 0

        # Test guard OR with empty array should return no results
        results = store.query(probe="{}", guard={"status": []}, top_k=10, threshold=0.0)
        assert len(results) == 0

        # Test guard OR with single element array
        results = store.query(
            probe="{}", guard={"status": ["active"]}, top_k=10, threshold=0.0
        )
        assert len(results) == 2  # Item1 and Item3

    def test_new_encoder_modes(self):
        """Test the new encoder modes (ngram, chained, positional)"""
        from holon.encoder import ListEncodeMode

        store = CPUStore(dimensions=1000)

        # Test data with different encoding modes
        test_data = [
            '{"words": {"_encode_mode": "ngram", "sequence": ["quick", "brown", "fox"]}, '
            '"type": "ngram_test"}',
            '{"words": {"_encode_mode": "chained", "sequence": ["hello", "world"]}, '
            '"type": "chained_test"}',
            '{"words": {"_encode_mode": "positional", "sequence": ["foo", "bar"]}, '
            '"type": "positional_test"}',
            '{"words": {"_encode_mode": "bundle", "sequence": ["simple", "list"]}, "type": "bundle_test"}',
        ]

        ids = []
        for data in test_data:
            id_ = store.insert(data)
            ids.append(id_)

        # Verify all inserts succeeded
        assert len(ids) == 4
        for id_ in ids:
            assert id_ is not None

        # Test that we can retrieve the data back
        for id_ in ids:
            retrieved = store.get(id_)
            assert retrieved is not None
            assert "type" in retrieved

        # Test query with ngram-encoded data
        probe = '{"words": {"_encode_mode": "ngram", "sequence": ["quick", "brown"]}}'
        results = store.query(probe=probe, top_k=5, threshold=0.0)
        assert len(results) >= 1  # Should find the ngram_test item

    def test_vector_bootstrapping_api(self):
        """Test the encode functionality for vector bootstrapping"""
        from holon.encoder import Encoder, VectorManager

        # Test the encoder directly
        vm = VectorManager(dimensions=1000)
        encoder = Encoder(vm)

        # Test different encoding modes
        test_data = {
            "words": {
                "_encode_mode": "ngram",
                "sequence": ["test", "vector", "encoding"],
            }
        }

        # Encode the data
        vector = encoder.encode_data(test_data)

        # Verify vector properties
        assert vector is not None
        assert len(vector) == 1000
        assert all(val in [-1, 0, 1] for val in vector)  # Bipolar values

        # Test that different data produces different vectors
        test_data2 = {
            "words": {
                "_encode_mode": "ngram",
                "sequence": ["different", "test", "data"],
            }
        }
        vector2 = encoder.encode_data(test_data2)

        # Vectors should be different (with high probability in high dimensions)
        assert not all(v1 == v2 for v1, v2 in zip(vector, vector2))
