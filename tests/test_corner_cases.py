#!/usr/bin/env python3
"""
Test corner cases: empty data, deep nesting, large structures, invalid inputs.
"""

import json

import edn_format
import numpy as np
import pytest

from holon import CPUStore


def test_empty_data():
    store = CPUStore()
    # Empty dict
    store.insert("{}")
    result = store.query(probe="{}")
    assert len(result) == 1


def test_deep_nesting():
    store = CPUStore()
    # Deep nested structure
    deep = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
    store.insert(json.dumps(deep))
    results = store.query(probe=json.dumps({"a": {"b": {"c": {"d": {"e": "value"}}}}}))
    assert len(results) == 1


def test_large_lists():
    store = CPUStore()
    large_list = {"sequence": list(range(1000))}
    store.insert(json.dumps(large_list))
    results = store.query(probe=json.dumps({"sequence": [0, 1, 2, {"$any": True}, 4]}))
    assert len(results) == 1  # Should match due to positional encoding


def test_invalid_probe():
    store = CPUStore()
    with pytest.raises(Exception):
        store.query(probe="invalid json")


def test_no_results():
    store = CPUStore()
    store.insert('{"user": "alice"}')
    results = store.query(probe='{"user": "bob"}')
    assert len(results) == 0


def test_wildcard_edge():
    store = CPUStore()
    store.insert('{"a": {"b": "c"}}')
    results = store.query(probe='{"a": {"$any": true}}')  # Match any in a
    assert len(results) == 1


def test_negation_edge():
    store = CPUStore()
    store.insert('{"status": "success"}')
    results = store.query(
        probe='{"status": "success"}', negations={"status": {"$not": "success"}}
    )
    assert len(results) == 0  # Excluded


def test_or_empty():
    store = CPUStore()
    store.insert('{"user": "alice"}')
    results = store.query(probe='{"$or": []}')
    assert len(results) == 0  # No subqueries


def test_guard_exact():
    store = CPUStore()
    store.insert('{"tags": ["a", "b", "c"]}')
    # Guard as data structure
    guard = {"tags": ["a", "b", "c"]}
    results = store.query(probe='{"tags": ["a", "b", "c"]}', guard=guard)
    assert len(results) == 1


def test_concurrent_like():
    # Simulate multiple queries
    store = CPUStore()
    for i in range(10):
        store.insert(json.dumps({"id": i, "data": "x" * (i + 1)}))  # Make data distinct
    results = []
    for i in range(10):
        res = store.query(probe=json.dumps({"id": i, "data": "x" * (i + 1)}))
        results.append(len(res))
    assert all(r >= 1 for r in results)  # At least the inserted one


def test_atomize_json():
    from holon.atomizer import atomize

    data = {"user": "alice", "count": 42, "active": True}
    atoms = atomize(data)
    expected = {"user", "alice", "count", "42", "active", "True"}
    assert atoms == expected


def test_atomize_edn():
    from holon.atomizer import atomize

    data = edn_format.loads('{:user "bob" :count 24 :active false}')
    atoms = atomize(data)
    expected = {":user", "bob", ":count", "24", ":active", "False"}
    assert atoms == expected


def test_atomize_nested():
    from holon.atomizer import atomize

    data = {"nested": {"key": "value", "list": [1, 2, "three"]}}
    atoms = atomize(data)
    expected = {"nested", "key", "value", "list", "1", "2", "three"}
    assert atoms == expected


def test_normalized_dot_similarity():
    from holon.similarity import normalized_dot_similarity

    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    sim = normalized_dot_similarity(vec1, vec2)
    assert sim == 0.0

    vec3 = np.array([1.0, 1.0, 0.0])
    sim = normalized_dot_similarity(vec1, vec3)
    expected = np.dot(vec1, vec3) / len(vec1)  # dot / dimension
    assert abs(sim - expected) < 1e-6


def test_find_similar_vectors_small():
    from holon.similarity import find_similar_vectors

    stored_vectors = {
        f"id_{i}": np.random.rand(100).astype(np.float32) for i in range(10)
    }
    query = np.random.rand(100).astype(np.float32)
    results = find_similar_vectors(query, stored_vectors, top_k=5)
    assert len(results) == 5
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
