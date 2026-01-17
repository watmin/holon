#!/usr/bin/env python3
"""
Test corner cases: empty data, deep nesting, large structures, invalid inputs.
"""

import json
import pytest
from holon import CPUStore

def test_empty_data():
    store = CPUStore()
    # Empty dict
    id_ = store.insert("{}")
    result = store.query("{}")
    assert len(result) == 1

def test_deep_nesting():
    store = CPUStore()
    # Deep nested structure
    deep = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
    store.insert(json.dumps(deep))
    results = store.query(json.dumps({"a": {"b": {"c": {"d": {"e": "value"}}}}}))
    assert len(results) == 1

def test_large_lists():
    store = CPUStore()
    large_list = {"sequence": list(range(1000))}
    store.insert(json.dumps(large_list))
    results = store.query(json.dumps({"sequence": [0, 1, 2, {"$any": True}, 4]}))
    assert len(results) == 1  # Should match due to positional encoding

def test_invalid_probe():
    store = CPUStore()
    with pytest.raises(Exception):
        store.query("invalid json")

def test_no_results():
    store = CPUStore()
    store.insert('{"user": "alice"}')
    results = store.query('{"user": "bob"}')
    assert len(results) == 0

def test_wildcard_edge():
    store = CPUStore()
    store.insert('{"a": {"b": "c"}}')
    results = store.query('{"a": {"$any": true}}')  # Match any in a
    assert len(results) == 1

def test_negation_edge():
    store = CPUStore()
    store.insert('{"status": "success"}')
    results = store.query('{"status": "success"}', negations={"status": {"$not": "success"}})
    assert len(results) == 0  # Excluded

def test_or_empty():
    store = CPUStore()
    store.insert('{"user": "alice"}')
    results = store.query('{"$or": []}')
    assert len(results) == 0  # No subqueries

def test_guard_exact():
    store = CPUStore()
    store.insert('{"tags": ["a", "b", "c"]}')
    # Guard for exact list
    def guard(d):
        return d.get("tags") == ["a", "b", "c"]
    results = store.query('{"tags": ["a", "b", "c"]}', guard=guard)
    assert len(results) == 1

def test_concurrent_like():
    # Simulate multiple queries
    store = CPUStore()
    for i in range(10):
        store.insert(json.dumps({"unique": f"user_{i}"}))
    results = []
    for i in range(10):
        res = store.query(json.dumps({"unique": f"user_{i}"}))
        results.append(len(res))
    assert all(r == 1 for r in results)