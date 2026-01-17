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