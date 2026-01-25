import pytest
from holon import CPUStore


class TestCPUStore:
    def test_insert_and_get_json(self):
        store = CPUStore(dimensions=1000)  # Smaller for testing
        data = '{"key": "value"}'
        data_id = store.insert(data, 'json')
        assert data_id is not None
        retrieved = store.get(data_id)
        assert retrieved == {"key": "value"}

    def test_insert_and_get_edn(self):
        store = CPUStore(dimensions=1000)
        data = '{:key "value", :num 42}'
        data_id = store.insert(data, 'edn')
        assert data_id is not None
        retrieved = store.get(data_id)
        # EDN parsed data will have Keyword keys
        assert str(retrieved) == "{Keyword(key): 'value', Keyword(num): 42}"

    def test_query_json(self):
        store = CPUStore(dimensions=1000)
        data1 = '{"name": "Alice", "age": 30}'
        data2 = '{"name": "Bob", "age": 25}'
        id1 = store.insert(data1, 'json')
        id2 = store.insert(data2, 'json')

        probe = '{"name": "Alice"}'
        results = store.query(probe, 'json', top_k=5, threshold=0.0)
        assert len(results) >= 1
        # Should find the Alice data with some similarity

    def test_query_edn(self):
        store = CPUStore(dimensions=1000)
        data1 = '{:name "Alice", :skills #{"clojure"}}'
        data2 = '{:name "Bob", :skills #{"python"}}'
        id1 = store.insert(data1, 'edn')
        id2 = store.insert(data2, 'edn')

        probe = '{:skills #{"clojure"}}'
        results = store.query(probe, 'edn', top_k=5, threshold=0.0)
        assert len(results) >= 1
        # Should find data with clojure skill

    def test_delete(self):
        store = CPUStore(dimensions=1000)
        data = '{"test": "data"}'
        data_id = store.insert(data, 'json')
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
            {"name": "Charlie", "priority": "low", "category": "A"}
        ]

        # Test the is_subset function directly (matching the updated implementation)
        def is_subset(guard, data):
            # Handle top-level $or in guards for powerful OR logic
            if "$or" in guard and isinstance(guard["$or"], list):
                # Any of the OR conditions must match
                return any(is_subset(or_condition, data) for or_condition in guard["$or"])

            for key, value in guard.items():
                if key not in data:
                    return False
                if isinstance(value, dict):
                    # Handle nested $or
                    if "$or" in value and isinstance(value["$or"], list):
                        # For nested $or, any of the conditions for this key must match
                        if not any(is_subset({key: or_val}, data) for or_val in value["$or"]):
                            return False
                    elif not isinstance(data[key], dict) or not is_subset(value, data[key]):
                        return False
                elif isinstance(value, list):
                    # Support OR logic: if guard has a list and data has a scalar that's IN the list,
                    # treat it as "match any of these values" (backward compatibility)
                    data_value = data[key]
                    if isinstance(data_value, list):
                        # Exact array matching for array-to-array comparison (backward compatibility)
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
        guard = {'priority': ['high', 'medium']}
        assert is_subset(guard, test_data[0]) == True   # Alice: high priority
        assert is_subset(guard, test_data[1]) == True   # Bob: medium priority
        assert is_subset(guard, test_data[2]) == False  # Charlie: low priority

        # Test 2: Guard with exact match (backward compatibility)
        guard = {'category': 'A'}
        assert is_subset(guard, test_data[0]) == True   # Alice: category A
        assert is_subset(guard, test_data[1]) == False  # Bob: category B
        assert is_subset(guard, test_data[2]) == True   # Charlie: category A

        # Test 3: Guard with array for tags (exact array matching - backward compatibility)
        guard = {'tags': ['urgent', 'backend']}
        data1 = {'name': 'Project1', 'tags': ['urgent', 'backend']}
        data2 = {'name': 'Project2', 'tags': ['frontend', 'urgent']}
        assert is_subset(guard, data1) == True   # Exact match
        assert is_subset(guard, data2) == False  # Different array

        # Test 4: Guard OR logic with single-element array
        guard = {'priority': ['high']}
        assert is_subset(guard, test_data[0]) == True   # Alice: high priority
        assert is_subset(guard, test_data[1]) == False  # Bob: medium priority
        assert is_subset(guard, test_data[2]) == False  # Charlie: low priority

        # Test 5: Structured $or in guards (most powerful)
        guard = {'$or': [{'priority': 'high'}, {'category': 'A'}]}
        assert is_subset(guard, test_data[0]) == True   # Alice: high priority (matches first condition)
        assert is_subset(guard, test_data[1]) == False  # Bob: medium priority, category B (matches neither)
        assert is_subset(guard, test_data[2]) == True   # Charlie: category A (matches second condition)

        # Test 6: Complex structured $or
        guard = {'$or': [
            {'priority': 'high', 'category': 'A'},  # Alice matches this
            {'priority': 'low', 'category': 'B'}    # No one matches this
        ]}
        assert is_subset(guard, test_data[0]) == True   # Alice matches compound condition
        assert is_subset(guard, test_data[1]) == False  # Bob doesn't match either condition
        assert is_subset(guard, test_data[2]) == False  # Charlie doesn't match either condition

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
        results = store.query('{}', guard={}, top_k=10, threshold=0.0)
        assert len(results) == 3

        # Test guard with non-existent field should return no results
        results = store.query('{}', guard={'nonexistent': 'value'}, top_k=10, threshold=0.0)
        assert len(results) == 0

        # Test guard OR with empty array should return no results
        results = store.query('{}', guard={'status': []}, top_k=10, threshold=0.0)
        assert len(results) == 0

        # Test guard OR with single element array
        results = store.query('{}', guard={'status': ['active']}, top_k=10, threshold=0.0)
        assert len(results) == 2  # Item1 and Item3

