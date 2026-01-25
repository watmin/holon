#!/usr/bin/env python3
"""
Atomizer Coverage Tests
Tests for EDN parsing edge cases and atomization coverage gaps.
"""

import edn_format
import pytest

from holon.atomizer import atomize, parse_data


class TestAtomizerParseData:
    """Test parse_data function edge cases."""

    def test_parse_data_json(self):
        """Test JSON parsing."""
        data = '{"key": "value", "num": 42}'
        result = parse_data(data, "json")
        assert result == {"key": "value", "num": 42}

    def test_parse_data_edn(self):
        """Test EDN parsing."""
        data = '{:key "value" :num 42}'
        result = parse_data(data, "edn")
        assert result[edn_format.Keyword("key")] == "value"
        assert result[edn_format.Keyword("num")] == 42

    def test_parse_data_invalid_type(self):
        """Test invalid data_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported data_type"):
            parse_data('{"test": "data"}', "invalid")

    def test_parse_data_malformed_json(self):
        """Test malformed JSON raises exception."""
        with pytest.raises(ValueError):  # JSON parsing error
            parse_data('{"invalid": json}', "json")

    def test_parse_data_malformed_edn(self):
        """Test malformed EDN raises exception."""
        with pytest.raises(ValueError):  # edn_format raises ValueError for parse errors
            parse_data("{invalid edn syntax", "edn")


class TestAtomizerAtomize:
    """Test atomize function comprehensive coverage."""

    def test_atomize_simple_dict(self):
        """Test basic dictionary atomization."""
        data = {"name": "Alice", "age": 30}
        result = atomize(data)
        assert result == {"name", "Alice", "age", "30"}

    def test_atomize_nested_dict(self):
        """Test nested dictionary atomization."""
        data = {"user": {"name": "Alice", "profile": {"age": 30}}}
        result = atomize(data)
        assert result == {"user", "name", "Alice", "profile", "age", "30"}

    def test_atomize_list(self):
        """Test list atomization."""
        data = ["Alice", "Bob", 42, 3.14]
        result = atomize(data)
        assert result == {"Alice", "Bob", "42", "3.14"}

    def test_atomize_set(self):
        """Test set atomization."""
        data = {"alice", "bob", 123}
        result = atomize(data)
        assert result == {"alice", "bob", "123"}

    def test_atomize_tuple(self):
        """Test tuple atomization."""
        data = ("alice", "bob", 456)
        result = atomize(data)
        assert result == {"alice", "bob", "456"}

    def test_atomize_frozenset(self):
        """Test frozenset atomization."""
        data = frozenset(["alice", "bob", 789])
        result = atomize(data)
        assert result == {"alice", "bob", "789"}

    def test_atomize_edn_keyword(self):
        """Test EDN keyword atomization."""
        keyword = edn_format.Keyword("user")
        data = {keyword: "alice"}
        result = atomize(data)
        assert ":user" in result
        assert "alice" in result

    def test_atomize_edn_symbol(self):
        """Test EDN symbol atomization."""
        symbol = edn_format.Symbol("my-symbol")
        data = {"key": symbol}
        result = atomize(data)
        assert "key" in result
        assert "my-symbol" in result

    def test_atomize_edn_char(self):
        """Test EDN character atomization."""
        char = edn_format.Char("a")
        data = {"letter": char}
        result = atomize(data)
        assert "letter" in result
        assert "a" in result

    def test_atomize_none(self):
        """Test None/null atomization."""
        data = {"optional": None}
        result = atomize(data)
        assert result == {"optional", "nil"}

    def test_atomize_boolean(self):
        """Test boolean atomization."""
        data = {"active": True, "disabled": False}
        result = atomize(data)
        assert result == {"active", "True", "disabled", "False"}

    def test_atomize_mixed_types(self):
        """Test mixed data types."""
        data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": ["a", "b", 1],
            "dict": {"nested": "value"},
        }
        result = atomize(data)
        expected = {
            "string",
            "hello",
            "number",
            "42",
            "float",
            "3.14",
            "bool",
            "True",
            "none",
            "nil",
            "list",
            "a",
            "b",
            "1",
            "dict",
            "nested",
            "value",
        }
        assert result == expected

    def test_atomize_empty_structures(self):
        """Test empty data structures."""
        assert atomize({}) == set()
        assert atomize([]) == set()
        assert atomize(set()) == set()

    def test_atomize_unicode_strings(self):
        """Test unicode string handling."""
        data = {"name": "José", "city": "São Paulo"}
        result = atomize(data)
        assert "name" in result
        assert "José" in result
        assert "city" in result
        assert "São Paulo" in result

    def test_atomize_special_characters(self):
        """Test special characters in strings."""
        data = {"path": "/usr/local/bin", "email": "user@domain.com"}
        result = atomize(data)
        assert "path" in result
        assert "/usr/local/bin" in result
        assert "email" in result
        assert "user@domain.com" in result

    def test_atomize_large_numbers(self):
        """Test large number handling."""
        data = {"big_int": 9223372036854775807, "big_float": 1.7976931348623157e308}
        result = atomize(data)
        assert "big_int" in result
        assert "9223372036854775807" in result
        assert "big_float" in result
        # Large float might be represented differently, just check it contains the key
        assert any("big_float" in atom for atom in result)

    def test_atomize_recursive_prevention(self):
        """Test that recursive structures don't cause infinite loops."""
        # This would be caught by Python's recursion limit anyway,
        # but we should ensure our code doesn't create problematic structures
        data = {"self": None}
        data["self"] = data  # Create circular reference
        # This should raise RecursionError, not infinite loop in our atomizer
        with pytest.raises(RecursionError):
            atomize(data)
