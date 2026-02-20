"""Tests for the Walkable interface and zero-serialization encoding."""

import numpy as np
import pytest

from holon import (
    Encoder,
    VectorManager,
    Walkable,
    WalkableDict,
    WalkableList,
    WalkableScalar,
    WalkableSet,
    WalkType,
    as_walkable,
    is_walkable,
    register_walkable,
    register_walkable_adapter,
    walk_iter,
)


@pytest.fixture
def encoder():
    """Create an encoder for testing."""
    vm = VectorManager(dimensions=1024, global_seed=42)
    return Encoder(vm)


class TestWalkType:
    """Test WalkType enum."""

    def test_walk_types_exist(self):
        assert WalkType.SCALAR.value == "scalar"
        assert WalkType.MAP.value == "map"
        assert WalkType.LIST.value == "list"
        assert WalkType.SET.value == "set"


class TestAsWalkable:
    """Test the as_walkable() function."""

    def test_dict_becomes_walkable_dict(self):
        d = {"name": "Alice", "age": 30}
        w = as_walkable(d)
        assert isinstance(w, WalkableDict)
        assert w.walk_type() == WalkType.MAP

    def test_list_becomes_walkable_list(self):
        lst = [1, 2, 3]
        w = as_walkable(lst)
        assert isinstance(w, WalkableList)
        assert w.walk_type() == WalkType.LIST

    def test_tuple_becomes_walkable_list(self):
        tpl = (1, 2, 3)
        w = as_walkable(tpl)
        assert isinstance(w, WalkableList)
        assert w.walk_type() == WalkType.LIST

    def test_set_becomes_walkable_set(self):
        s = {1, 2, 3}
        w = as_walkable(s)
        assert isinstance(w, WalkableSet)
        assert w.walk_type() == WalkType.SET

    def test_frozenset_becomes_walkable_set(self):
        fs = frozenset([1, 2, 3])
        w = as_walkable(fs)
        assert isinstance(w, WalkableSet)
        assert w.walk_type() == WalkType.SET

    def test_scalar_string_becomes_walkable_scalar(self):
        w = as_walkable("hello")
        assert isinstance(w, WalkableScalar)
        assert w.walk_type() == WalkType.SCALAR
        assert w.walk_scalar_value() == "hello"

    def test_scalar_int_becomes_walkable_scalar(self):
        w = as_walkable(42)
        assert isinstance(w, WalkableScalar)
        assert w.walk_scalar_value() == 42

    def test_scalar_float_becomes_walkable_scalar(self):
        w = as_walkable(3.14)
        assert isinstance(w, WalkableScalar)
        assert w.walk_scalar_value() == 3.14

    def test_scalar_bool_becomes_walkable_scalar(self):
        w = as_walkable(True)
        assert isinstance(w, WalkableScalar)
        assert w.walk_scalar_value() is True

    def test_scalar_none_becomes_walkable_scalar(self):
        w = as_walkable(None)
        assert isinstance(w, WalkableScalar)
        assert w.walk_scalar_value() is None

    def test_walkable_returned_as_is(self):
        original = WalkableScalar("test")
        result = as_walkable(original)
        assert result is original


class TestWalkableDict:
    """Test WalkableDict behavior."""

    def test_walk_map_items(self):
        d = {"a": 1, "b": 2, "c": 3}
        w = WalkableDict(d)
        items = list(w.walk_map_items())
        assert len(items) == 3
        assert ("a", 1) in items
        assert ("b", 2) in items
        assert ("c", 3) in items

    def test_empty_dict(self):
        w = WalkableDict({})
        items = list(w.walk_map_items())
        assert items == []


class TestWalkableList:
    """Test WalkableList behavior."""

    def test_walk_list_items(self):
        lst = [1, 2, 3]
        w = WalkableList(lst)
        items = list(w.walk_list_items())
        assert items == [1, 2, 3]

    def test_empty_list(self):
        w = WalkableList([])
        items = list(w.walk_list_items())
        assert items == []


class TestWalkableSet:
    """Test WalkableSet behavior."""

    def test_walk_set_items(self):
        s = {1, 2, 3}
        w = WalkableSet(s)
        items = set(w.walk_set_items())
        assert items == {1, 2, 3}

    def test_empty_set(self):
        w = WalkableSet(set())
        items = list(w.walk_set_items())
        assert items == []


class TestIsWalkable:
    """Test the is_walkable() function."""

    def test_native_types_are_walkable(self):
        assert is_walkable({})
        assert is_walkable([])
        assert is_walkable(set())
        assert is_walkable("hello")
        assert is_walkable(42)
        assert is_walkable(3.14)
        assert is_walkable(True)
        assert is_walkable(None)

    def test_walkable_subclass_is_walkable(self):
        class CustomWalkable(Walkable):
            def walk_type(self):
                return WalkType.SCALAR

            def walk_scalar_value(self):
                return "custom"

        assert is_walkable(CustomWalkable())


class TestCustomWalkable:
    """Test custom Walkable implementations."""

    def test_custom_scalar(self):
        class MyValue(Walkable):
            def __init__(self, v):
                self._v = v

            def walk_type(self):
                return WalkType.SCALAR

            def walk_scalar_value(self):
                return self._v

        mv = MyValue("test")
        assert mv.walk_type() == WalkType.SCALAR
        assert mv.walk_scalar_value() == "test"

    def test_custom_map(self):
        class Person(Walkable):
            def __init__(self, name, age):
                self.name = name
                self.age = age

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "name", self.name
                yield "age", self.age

        person = Person("Alice", 30)
        assert person.walk_type() == WalkType.MAP
        items = list(person.walk_map_items())
        assert items == [("name", "Alice"), ("age", 30)]

    def test_custom_list(self):
        class TimeSeries(Walkable):
            def __init__(self, values):
                self._values = values

            def walk_type(self):
                return WalkType.LIST

            def walk_list_items(self):
                for v in self._values:
                    yield v

        ts = TimeSeries([1.0, 2.0, 3.0])
        assert ts.walk_type() == WalkType.LIST
        assert list(ts.walk_list_items()) == [1.0, 2.0, 3.0]

    def test_custom_set(self):
        class TagSet(Walkable):
            def __init__(self, tags):
                self._tags = tags

            def walk_type(self):
                return WalkType.SET

            def walk_set_items(self):
                for t in self._tags:
                    yield t

        tags = TagSet(["python", "rust", "holon"])
        assert tags.walk_type() == WalkType.SET
        assert set(tags.walk_set_items()) == {"python", "rust", "holon"}


class TestEncodeWalkable:
    """Test the encode_walkable() method."""

    def test_encode_simple_dict(self, encoder):
        data = {"name": "Alice", "age": 30}
        vec = encoder.encode_walkable(data)
        assert vec.shape == (1024,)
        assert vec.dtype == np.int8

    def test_encode_simple_list(self, encoder):
        data = [1, 2, 3]
        vec = encoder.encode_walkable(data)
        assert vec.shape == (1024,)
        assert vec.dtype == np.int8

    def test_encode_simple_set(self, encoder):
        data = {"a", "b", "c"}
        vec = encoder.encode_walkable(data)
        assert vec.shape == (1024,)
        assert vec.dtype == np.int8

    def test_encode_scalar(self, encoder):
        vec = encoder.encode_walkable("hello")
        assert vec.shape == (1024,)

    def test_encode_nested_structure(self, encoder):
        data = {
            "user": {"name": "Alice", "email": "alice@example.com"},
            "scores": [95, 87, 92],
            "tags": {"python", "data-science"},
        }
        vec = encoder.encode_walkable(data)
        assert vec.shape == (1024,)

    def test_encode_empty_dict(self, encoder):
        vec = encoder.encode_walkable({})
        assert vec.shape == (1024,)
        # Empty dict should produce zero vector
        assert np.all(vec == 0)

    def test_encode_empty_list(self, encoder):
        vec = encoder.encode_walkable([])
        assert vec.shape == (1024,)
        assert np.all(vec == 0)

    def test_walkable_matches_traditional_encoding(self, encoder):
        """Verify that encode_walkable produces similar results to encode_data."""
        data = {"type": "billing", "amount": 100, "status": "paid"}

        vec_traditional = encoder.encode_data(data)
        vec_walkable = encoder.encode_walkable(data)

        # They should be identical for native types
        assert np.array_equal(vec_traditional, vec_walkable)

    def test_walkable_matches_for_lists(self, encoder):
        data = ["apple", "banana", "cherry"]
        vec_traditional = encoder.encode_data(data)
        vec_walkable = encoder.encode_walkable(data)
        assert np.array_equal(vec_traditional, vec_walkable)

    def test_walkable_matches_for_sets(self, encoder):
        # Note: sets might not be identical due to iteration order
        # but the vector should be the same regardless of order
        data = frozenset(["a", "b", "c"])
        vec_traditional = encoder.encode_data(data)
        vec_walkable = encoder.encode_walkable(data)
        assert np.array_equal(vec_traditional, vec_walkable)

    def test_encode_custom_walkable(self, encoder):
        """Test encoding a custom Walkable type."""

        class Order(Walkable):
            def __init__(self, order_id, items, total):
                self.order_id = order_id
                self.items = items
                self.total = total

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "order_id", self.order_id
                yield "items", self.items
                yield "total", self.total

        order = Order("ORD-123", ["widget", "gadget"], 99.99)
        vec = encoder.encode_walkable(order)

        assert vec.shape == (1024,)
        assert vec.dtype == np.int8
        # Should have non-zero components
        assert np.sum(np.abs(vec)) > 0

    def test_nested_custom_walkable(self, encoder):
        """Test encoding nested custom Walkable types."""

        class Address(Walkable):
            def __init__(self, city, country):
                self.city = city
                self.country = country

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "city", self.city
                yield "country", self.country

        class Person(Walkable):
            def __init__(self, name, address):
                self.name = name
                self.address = address

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "name", self.name
                yield "address", self.address

        person = Person("Alice", Address("NYC", "USA"))
        vec = encoder.encode_walkable(person)
        assert vec.shape == (1024,)


class TestWalkIter:
    """Test the walk_iter() utility function."""

    def test_walk_simple_dict(self):
        data = {"a": 1, "b": 2}
        paths = [(path, val) for path, val, _ in walk_iter(data)]

        # Root + 2 values = 3 nodes (keys are part of path, not separate nodes)
        assert len(paths) == 3
        assert ("", data) in paths
        assert ("a", 1) in paths
        assert ("b", 2) in paths

    def test_walk_nested_dict(self):
        data = {"user": {"name": "Alice"}}
        paths = [path for path, _, _ in walk_iter(data)]

        assert "" in paths  # root
        assert "user" in paths
        assert "user.name" in paths

    def test_walk_list(self):
        data = ["a", "b", "c"]
        paths = [path for path, _, _ in walk_iter(data)]

        assert "" in paths  # root
        assert "0" in paths
        assert "1" in paths
        assert "2" in paths

    def test_walk_scalar(self):
        data = "hello"
        paths = list(walk_iter(data))

        assert len(paths) == 1
        assert paths[0][0] == ""
        assert paths[0][1] == "hello"


class TestRegistration:
    """Test the adapter registration system."""

    def test_register_walkable_adapter(self):
        class MyCustomType:
            def __init__(self, value):
                self.value = value

        class MyCustomAdapter(Walkable):
            def __init__(self, obj):
                self._obj = obj

            def walk_type(self):
                return WalkType.SCALAR

            def walk_scalar_value(self):
                return self._obj.value

        register_walkable_adapter(MyCustomType, MyCustomAdapter)

        obj = MyCustomType("test-value")
        w = as_walkable(obj)

        assert isinstance(w, MyCustomAdapter)
        assert w.walk_scalar_value() == "test-value"

    def test_register_walkable_decorator(self):
        class AnotherType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        @register_walkable(AnotherType)
        class AnotherTypeAdapter(Walkable):
            def __init__(self, obj):
                self._obj = obj

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "x", self._obj.x
                yield "y", self._obj.y

        obj = AnotherType(10, 20)
        w = as_walkable(obj)

        assert isinstance(w, AnotherTypeAdapter)
        assert list(w.walk_map_items()) == [("x", 10), ("y", 20)]


class TestSimilarity:
    """Test that walkable encoding preserves semantic similarity."""

    @pytest.fixture
    def large_encoder(self):
        """Create a larger encoder for similarity tests (needs more dimensions)."""
        vm = VectorManager(dimensions=8192, global_seed=42)
        return Encoder(vm)

    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def test_similar_dicts_have_high_similarity(self, large_encoder):
        """Dicts with shared keys/values should have higher similarity."""
        # Use structures with more shared content for clearer signal
        d1 = {"type": "billing", "category": "finance", "status": "active"}
        d2 = {"type": "billing", "category": "finance", "status": "pending"}
        d3 = {"type": "shipping", "weight": 5, "carrier": "ups"}

        v1 = large_encoder.encode_walkable(d1)
        v2 = large_encoder.encode_walkable(d2)
        v3 = large_encoder.encode_walkable(d3)

        # d1 and d2 share "type": "billing" and "category": "finance"
        sim_12 = self.cosine_similarity(v1, v2)
        sim_13 = self.cosine_similarity(v1, v3)

        # With more shared structure, d1-d2 should clearly be more similar
        assert sim_12 > sim_13

    def test_custom_walkable_preserves_similarity(self, large_encoder):
        """Custom walkable should produce same vector as equivalent dict."""

        class Record(Walkable):
            def __init__(self, **kwargs):
                self._data = kwargs

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                for k, v in self._data.items():
                    yield k, v

        dict_data = {"type": "billing", "status": "active"}
        record_data = Record(type="billing", status="active")

        vec_dict = large_encoder.encode_walkable(dict_data)
        vec_record = large_encoder.encode_walkable(record_data)

        # Should be identical
        assert np.array_equal(vec_dict, vec_record)


class TestMagnitudeAwareScalars:
    """Tests for LogScale and LinearScale wrappers in walkable encoding."""

    @pytest.fixture
    def encoder(self):
        from holon import Encoder
        from holon.vector_manager import VectorManager

        vm = VectorManager(dimensions=4096, global_seed=42)
        return Encoder(vm)

    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        # Cast to float64 to avoid int8 overflow in dot product
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def test_log_scale_basic(self, encoder):
        """LogScale wrapper produces valid vectors and self-similarity is 1.0."""
        from holon import LogScale, Walkable, WalkType

        # Use only the rate field to isolate the log encoding effect
        class RateRecord(Walkable):
            def __init__(self, rate: float):
                self.rate = rate

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "rate", LogScale(self.rate)

        r1 = RateRecord(1000)
        r2 = RateRecord(1000)  # Same as r1

        v1 = encoder.encode_walkable(r1)
        v2 = encoder.encode_walkable(r2)

        # Self-similarity should be 1.0
        self_sim = self.cosine_similarity(v1, v2)
        assert (
            abs(self_sim - 1.0) < 0.001
        ), f"Same values should be identical, got {self_sim}"

        # Vector should be non-zero and have correct dimensions
        assert v1.shape == (4096,)
        assert np.any(v1 != 0), "Vector should be non-zero"

    def test_linear_scale_basic(self, encoder):
        """LinearScale wrapper encodes values with linear similarity."""
        from holon import LinearScale, Walkable, WalkType

        class Measurement(Walkable):
            def __init__(self, sensor: str, temp: float):
                self.sensor = sensor
                self.temp = temp

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "sensor", self.sensor
                yield "temp", LinearScale(self.temp)

        m1 = Measurement("room_a", 20.0)
        m2 = Measurement("room_a", 22.0)  # Close to m1
        m3 = Measurement("room_a", 50.0)  # Far from m1

        v1 = encoder.encode_walkable(m1)
        v2 = encoder.encode_walkable(m2)
        v3 = encoder.encode_walkable(m3)

        sim_close = self.cosine_similarity(v1, v2)
        sim_far = self.cosine_similarity(v1, v3)

        # Similar temperatures should produce higher similarity
        assert sim_close > sim_far, f"Expected {sim_close} > {sim_far}"

    def test_log_scale_ratio_preservation(self, encoder):
        """Log encoding: equal ratios should produce similar similarity drops."""
        from holon import LogScale, Walkable, WalkType

        class RateRecord(Walkable):
            def __init__(self, rate: float):
                self.rate = rate

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "rate", LogScale(self.rate)

        # 10x ratios
        r100 = RateRecord(100)
        r1000 = RateRecord(1000)
        r10000 = RateRecord(10000)

        v100 = encoder.encode_walkable(r100)
        v1000 = encoder.encode_walkable(r1000)
        v10000 = encoder.encode_walkable(r10000)

        sim_100_1000 = self.cosine_similarity(v100, v1000)
        sim_1000_10000 = self.cosine_similarity(v1000, v10000)

        # 10x ratios should produce approximately equal similarity drops
        diff = abs(sim_100_1000 - sim_1000_10000)
        assert diff < 0.15, f"Expected similar drops for 10x ratios, got diff={diff}"

    def test_log_vs_string_encoding(self, encoder):
        """LogScale and string encoding produce different vectors."""
        from holon import LogScale, Walkable, WalkType

        class LogRecord(Walkable):
            def __init__(self, rate: float):
                self.rate = rate

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "rate", LogScale(self.rate)

        # Log encoding should produce different vectors than string encoding
        v_log = encoder.encode_walkable(LogRecord(100))
        v_str = encoder.encode_walkable({"rate": 100})

        # They should be different (log encoding vs string "100")
        sim = self.cosine_similarity(v_log, v_str)
        assert sim < 0.9, f"Log and string encoding should differ, got similarity {sim}"

        # But both should be valid non-zero vectors
        assert np.any(v_log != 0), "Log vector should be non-zero"
        assert np.any(v_str != 0), "String vector should be non-zero"

    def test_log_scale_with_custom_scale(self, encoder):
        """Higher scale parameter should produce higher similarity for same ratio."""
        from holon import LogScale, Walkable, WalkType

        class ScaledRecord(Walkable):
            def __init__(self, rate: float, scale: float):
                self.rate = rate
                self.scale = scale

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "rate", LogScale(self.rate, scale=self.scale)

        # Default scale
        r1_default = ScaledRecord(100, scale=1000)
        r2_default = ScaledRecord(1000, scale=1000)

        # Higher scale
        r1_high = ScaledRecord(100, scale=5000)
        r2_high = ScaledRecord(1000, scale=5000)

        v1_d = encoder.encode_walkable(r1_default)
        v2_d = encoder.encode_walkable(r2_default)
        v1_h = encoder.encode_walkable(r1_high)
        v2_h = encoder.encode_walkable(r2_high)

        sim_default = self.cosine_similarity(v1_d, v2_d)
        sim_high = self.cosine_similarity(v1_h, v2_h)

        # Higher scale should produce higher similarity for same ratio
        assert (
            sim_high > sim_default
        ), f"Expected high scale > default: {sim_high} > {sim_default}"

    # -------------------------------------------------------------------------
    # TimeScale tests
    # -------------------------------------------------------------------------

    def test_time_scale_parity_with_dict_marker(self, encoder):
        """TimeScale(ts) should produce the same vector as {'$time': ts}."""
        from holon import TimeScale

        ts = 1_700_000_000

        v_wrapper = encoder.encode_data({"ts": TimeScale(ts)})
        v_marker = encoder.encode_data({"ts": {"$time": ts}})

        np.testing.assert_array_equal(
            v_wrapper,
            v_marker,
            err_msg="TimeScale wrapper and $time marker should encode identically",
        )

    def test_time_scale_resolution_parity(self, encoder):
        """TimeScale with explicit resolution should match $time_resolution marker."""
        from holon import TimeScale

        ts = 1_700_000_000

        v_wrapper = encoder.encode_data({"ts": TimeScale(ts, resolution="minute")})
        v_marker = encoder.encode_data(
            {"ts": {"$time": ts, "$time_resolution": "minute"}}
        )

        np.testing.assert_array_equal(
            v_wrapper,
            v_marker,
            err_msg="TimeScale(resolution='minute') and $time_resolution='minute' should match",
        )

    def test_time_scale_default_resolution_is_hour(self, encoder):
        """TimeScale default resolution should match $time_resolution='hour'."""
        from holon import TimeScale

        ts = 1_700_000_000

        v_default = encoder.encode_data({"ts": TimeScale(ts)})
        v_hour = encoder.encode_data({"ts": {"$time": ts, "$time_resolution": "hour"}})

        np.testing.assert_array_equal(
            v_default,
            v_hour,
            err_msg="TimeScale() default should equal $time_resolution='hour'",
        )

    def test_time_scale_near_times_more_similar(self, encoder):
        """Close timestamps should yield higher similarity than distant ones."""
        from holon import TimeScale

        base_ts = 1_700_000_000

        v_base = encoder.encode_data({"ts": TimeScale(base_ts)})
        v_near = encoder.encode_data({"ts": TimeScale(base_ts + 3600)})  # 1 hour
        v_far = encoder.encode_data(
            {"ts": TimeScale(base_ts + 180 * 86400)}
        )  # 180 days

        sim_near = self.cosine_similarity(v_base, v_near)
        sim_far = self.cosine_similarity(v_base, v_far)

        assert (
            sim_near > sim_far
        ), f"Near time should be more similar: {sim_near:.3f} > {sim_far:.3f}"

    def test_time_scale_iso_string(self, encoder):
        """TimeScale should accept ISO 8601 string timestamps."""
        from holon import TimeScale

        v = encoder.encode_data({"ts": TimeScale("2024-01-29T10:30:00Z")})
        assert np.any(v != 0), "ISO string TimeScale should produce non-zero vector"

    def test_time_scale_walkable_path(self, encoder):
        """TimeScale works through encode_walkable as well as encode_data."""
        from holon import TimeScale, Walkable, WalkType

        ts = 1_700_000_000

        class Event(Walkable):
            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "ts", TimeScale(ts)

        v_walkable = encoder.encode_walkable(Event())
        v_data = encoder.encode_data({"ts": TimeScale(ts)})

        np.testing.assert_array_equal(
            v_walkable,
            v_data,
            err_msg="TimeScale should produce same vector via encode_walkable and encode_data",
        )

    def test_time_scale_non_zero(self, encoder):
        """TimeScale encoding should produce a non-zero vector."""
        from holon import TimeScale

        v = encoder.encode_data({"ts": TimeScale(1_700_000_000)})
        assert np.any(v != 0), "TimeScale should produce a non-zero vector"


class TestWrapperEncodeDatePath:
    """Tests that LogScale, LinearScale, and TimeScale work correctly through encode_data."""

    @pytest.fixture
    def encoder(self):
        from holon import Encoder
        from holon.vector_manager import VectorManager

        vm = VectorManager(dimensions=4096, global_seed=42)
        return Encoder(vm)

    def cosine_similarity(self, a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def test_log_scale_encode_data_not_string(self, encoder):
        """LogScale in encode_data must NOT fall through to string encoding."""
        from holon import LogScale

        v_log = encoder.encode_data({"rate": LogScale(1000)})
        v_str = encoder.encode_data({"rate": 1000})

        # LogScale encoding differs from plain integer string encoding
        sim = self.cosine_similarity(v_log, v_str)
        assert (
            sim < 0.95
        ), f"LogScale should not fall through to str encoding (sim={sim:.3f})"
        assert np.any(v_log != 0), "LogScale vector should be non-zero"

    def test_log_scale_encode_data_ratio_preservation(self, encoder):
        """LogScale via encode_data should preserve equal-ratio similarity."""
        from holon import LogScale

        v100 = encoder.encode_data({"rate": LogScale(100)})
        v1000 = encoder.encode_data({"rate": LogScale(1000)})
        v10000 = encoder.encode_data({"rate": LogScale(10000)})

        sim_100_1000 = self.cosine_similarity(v100, v1000)
        sim_1000_10000 = self.cosine_similarity(v1000, v10000)

        diff = abs(sim_100_1000 - sim_1000_10000)
        assert diff < 0.15, (
            f"encode_data LogScale: equal 10x ratios should have equal similarity drops, "
            f"got diff={diff:.3f}"
        )

    def test_log_scale_encode_data_matches_encode_walkable(self, encoder):
        """LogScale via encode_data should match encode_walkable."""
        from holon import LogScale

        v_data = encoder.encode_data({"rate": LogScale(500)})
        v_walkable = encoder.encode_walkable({"rate": LogScale(500)})

        np.testing.assert_array_equal(
            v_data,
            v_walkable,
            err_msg="LogScale should produce same vector via encode_data and encode_walkable",
        )

    def test_linear_scale_encode_data_not_string(self, encoder):
        """LinearScale in encode_data must NOT fall through to string encoding."""
        from holon import LinearScale

        v_linear = encoder.encode_data({"temp": LinearScale(20.0)})
        v_str = encoder.encode_data({"temp": 20.0})

        sim = self.cosine_similarity(v_linear, v_str)
        assert (
            sim < 0.95
        ), f"LinearScale should not fall through to str encoding (sim={sim:.3f})"

    def test_linear_scale_encode_data_matches_encode_walkable(self, encoder):
        """LinearScale via encode_data should match encode_walkable."""
        from holon import LinearScale

        v_data = encoder.encode_data({"temp": LinearScale(25.0)})
        v_walkable = encoder.encode_walkable({"temp": LinearScale(25.0)})

        np.testing.assert_array_equal(
            v_data,
            v_walkable,
            err_msg="LinearScale should produce same vector via encode_data and encode_walkable",
        )
