"""
Walkable Protocol for In-Memory Data Traversal

This module provides the Walkable interface that allows Holon to encode
arbitrary in-memory data structures without serialization to JSON/EDN.

Design Goals:
- Native Python types (dict, list, set, scalars) work automatically
- Custom types can implement Walkable for custom traversal
- Extensible via registration for third-party types
- Zero-copy where possible - we walk references, not copies

Usage:

    # Native types just work
    encoder.encode_walkable({"name": "Alice", "age": 30})
    encoder.encode_walkable([1, 2, 3])

    # Custom types implement Walkable
    class MyRecord(Walkable):
        def __init__(self, data):
            self._data = data

        def walk_type(self) -> WalkType:
            return WalkType.MAP

        def walk_map_items(self):
            for k, v in self._data.items():
                yield k, v

    # Or register an adapter for existing types
    @register_walkable(pandas.DataFrame)
    class DataFrameWalker(Walkable):
        def __init__(self, df):
            self._df = df
        ...
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterator, Tuple, Type, Union

# Try importing edn_format types for compatibility
try:
    import edn_format
    from edn_format.immutable_dict import ImmutableDict

    HAS_EDN = True
except ImportError:
    edn_format = None
    ImmutableDict = None
    HAS_EDN = False


class WalkType(Enum):
    """The structural type of a walkable value.

    These correspond to the fundamental EDN/JSON types:
    - SCALAR: Atomic values (str, int, float, bool, None, keywords, symbols)
    - MAP: Key-value pairs (dict, records, objects)
    - LIST: Ordered sequences (list, tuple, arrays) — composed into one vector
    - SET: Unordered unique items (set, frozenset) — composed into one vector
    - SPREAD: Independent indexed items — each element fans out into its own
      leaf binding (fan-out). Use when per-element attribution or rule crafting
      is needed (e.g., TLS cipher order, HTTP header list).
    """

    SCALAR = "scalar"
    MAP = "map"
    LIST = "list"
    SET = "set"
    SPREAD = "spread"


class Walkable(ABC):
    """Abstract base class for walkable data structures.

    Implement this to make custom types traversable by the Holon encoder.
    Only implement the methods relevant to your type's WalkType.

    Example - A custom map type:

        class Person(Walkable):
            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age

            def walk_type(self) -> WalkType:
                return WalkType.MAP

            def walk_map_items(self):
                yield "name", self.name
                yield "age", self.age

    Example - A custom list type:

        class TimeSeries(Walkable):
            def __init__(self, values: List[float]):
                self.values = values

            def walk_type(self) -> WalkType:
                return WalkType.LIST

            def walk_list_items(self):
                for v in self.values:
                    yield v
    """

    @abstractmethod
    def walk_type(self) -> WalkType:
        """Return the structural type of this value."""
        pass

    def walk_scalar_value(self) -> Any:
        """Return the scalar value.

        Override for SCALAR types. The returned value will be converted to
        a string for vector lookup (similar to how JSON/EDN values work).

        Returns:
            The scalar value (str, int, float, bool, None, etc.)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is type {self.walk_type()}, "
            "walk_scalar_value() only valid for SCALAR types"
        )

    def walk_map_items(self) -> Iterator[Tuple[Any, Any]]:
        """Yield (key, value) pairs for this map.

        Override for MAP types. Keys are treated as scalars, values can be
        any walkable type (including nested structures).

        Yields:
            Tuple[key, value] pairs
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is type {self.walk_type()}, "
            "walk_map_items() only valid for MAP types"
        )

    def walk_list_items(self) -> Iterator[Any]:
        """Yield items in order for this list.

        Override for LIST types. Items can be any walkable type.

        Yields:
            Items in order
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is type {self.walk_type()}, "
            "walk_list_items() only valid for LIST types"
        )

    def walk_set_items(self) -> Iterator[Any]:
        """Yield items for this set (order not guaranteed).

        Override for SET types. Items can be any walkable type.

        Yields:
            Set items
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is type {self.walk_type()}, "
            "walk_set_items() only valid for SET types"
        )

    def walk_spread_items(self) -> Iterator[Any]:
        """Yield items for fan-out encoding.

        Override for SPREAD types. Each item becomes an independent leaf
        binding with its own indexed path (e.g., field.[0], field.[1]).
        Unlike LIST, these are not positionally composed — each element
        is treated as a separate leaf for per-element attribution.

        Use when per-element attribution or rule crafting is needed.

        Yields:
            Items in order (index assigned by position in iteration)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is type {self.walk_type()}, "
            "walk_spread_items() only valid for SPREAD types"
        )


# =============================================================================
# Built-in Wrappers for Native Python Types
# =============================================================================


class WalkableScalar(Walkable):
    """Wrapper for scalar values (str, int, float, bool, None, etc.)."""

    __slots__ = ("_value",)

    def __init__(self, value: Any):
        self._value = value

    def walk_type(self) -> WalkType:
        return WalkType.SCALAR

    def walk_scalar_value(self) -> Any:
        return self._value


# =============================================================================
# Magnitude-Aware Numeric Scalars
# =============================================================================


class LogScale:
    """Wrapper for log-scale numeric encoding.

    Equal ratios produce equal similarity drops:
    - 100 → 1000 (10x) has same drop as 1000 → 10000 (10x)

    Use for: packet rates, file sizes, frequencies, byte counts.

    Example:
        class TrafficRecord(Walkable):
            def __init__(self, rate: float):
                self.rate = rate

            def walk_type(self) -> WalkType:
                return WalkType.MAP

            def walk_map_items(self):
                yield "type", "traffic"
                yield "rate", LogScale(self.rate)  # Log-encoded
    """

    __slots__ = ("value", "scale")

    def __init__(self, value: float, scale: float = 1000.0):
        """Create a log-scale numeric wrapper.

        Args:
            value: The numeric value to encode (must be > 0)
            scale: Controls similarity decay rate (default 1000.0).
                   Higher scale = slower similarity decay for same ratio.
        """
        self.value = float(value)
        self.scale = float(scale)

    def __repr__(self) -> str:
        return f"LogScale({self.value}, scale={self.scale})"


class LinearScale:
    """Wrapper for linear positional encoding.

    Equal absolute differences produce equal similarity drops:
    - 10 → 20 (+10) has same drop as 100 → 110 (+10)

    Use for: temperatures, positions, timestamps, coordinates.

    Example:
        class Measurement(Walkable):
            def __init__(self, temp: float):
                self.temp = temp

            def walk_type(self) -> WalkType:
                return WalkType.MAP

            def walk_map_items(self):
                yield "sensor", "room_a"
                yield "temp", LinearScale(self.temp)  # Linear-encoded
    """

    __slots__ = ("value", "scale")

    def __init__(self, value: float, scale: float = 1000.0):
        """Create a linear-scale numeric wrapper.

        Args:
            value: The numeric value to encode
            scale: Controls similarity decay rate (default 1000.0).
                   Higher scale = slower similarity decay for same difference.
        """
        self.value = float(value)
        self.scale = float(scale)

    def __repr__(self) -> str:
        return f"LinearScale({self.value}, scale={self.scale})"


class TimeScale:
    """Wrapper for time-aware encoding with circular + positional components.

    Decomposes a timestamp into circular components (hour-of-day, day-of-week,
    month-of-year) plus a positional component, so temporally close events
    produce similar vectors while preserving periodic structure.

    Equal absolute time differences produce equal similarity drops (positional),
    and periodic structure is preserved across day/week/year boundaries (circular).

    Use for: timestamps, event times, log times, any Unix epoch value.

    Example:
        class LogEntry(Walkable):
            def __init__(self, ts: float, msg: str):
                self.ts = ts
                self.msg = msg

            def walk_type(self) -> WalkType:
                return WalkType.MAP

            def walk_map_items(self):
                yield "ts", TimeScale(self.ts)
                yield "msg", self.msg
    """

    __slots__ = ("value", "resolution")

    def __init__(self, value, resolution: str = "hour"):
        """Create a time-aware encoding wrapper.

        Args:
            value: Unix timestamp (float or int) or ISO 8601 string.
                   Same formats accepted by the {"$time": ...} marker.
            resolution: Controls the positional component granularity.
                        One of "second", "minute", "hour" (default), "day".
                        Higher resolution = finer positional discrimination.
        """
        self.value = value
        self.resolution = resolution

    def __repr__(self) -> str:
        return f"TimeScale({self.value!r}, resolution={self.resolution!r})"


class WalkableDict(Walkable):
    """Wrapper for dict-like objects."""

    __slots__ = ("_data",)

    def __init__(self, data: dict):
        self._data = data

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self) -> Iterator[Tuple[Any, Any]]:
        for k, v in self._data.items():
            yield k, v


class WalkableList(Walkable):
    """Wrapper for list/tuple-like sequences."""

    __slots__ = ("_data",)

    def __init__(self, data: Union[list, tuple]):
        self._data = data

    def walk_type(self) -> WalkType:
        return WalkType.LIST

    def walk_list_items(self) -> Iterator[Any]:
        for item in self._data:
            yield item


class WalkableSet(Walkable):
    """Wrapper for set/frozenset-like collections."""

    __slots__ = ("_data",)

    def __init__(self, data: Union[set, frozenset]):
        self._data = data

    def walk_type(self) -> WalkType:
        return WalkType.SET

    def walk_set_items(self) -> Iterator[Any]:
        for item in self._data:
            yield item


class WalkableSpread(Walkable):
    """Wrapper for fan-out sequences where each element is an independent leaf.

    Unlike WalkableList (which composes items into a single positionally-encoded
    vector), WalkableSpread fans out — each element gets its own indexed path
    and leaf binding. Use when per-element attribution matters.

    In the standard (non-striped) :meth:`~holon.kernel.Encoder.encode_walkable`
    path, SPREAD is encoded identically to LIST (positional binding + bundle),
    so existing workloads are unaffected. The fan-out semantics only apply
    in :meth:`~holon.kernel.Encoder.encode_walkable_striped`.

    Example::

        tls_ciphers = WalkableSpread(["AES256-GCM", "AES128-GCM", "CHACHA20"])
        # In striped encoding, each cipher becomes an independent leaf:
        # cipher.[0]=AES256-GCM, cipher.[1]=AES128-GCM, ...
    """

    __slots__ = ("_data",)

    def __init__(self, data: Union[list, tuple]):
        self._data = data

    def walk_type(self) -> WalkType:
        return WalkType.SPREAD

    def walk_spread_items(self) -> Iterator[Any]:
        for item in self._data:
            yield item

    def walk_list_items(self) -> Iterator[Any]:
        """Fallback for the standard encode path (treats Spread like a List)."""
        for item in self._data:
            yield item


# =============================================================================
# Type Registry for Extensibility
# =============================================================================

# Maps type -> adapter factory function
_walkable_adapters: Dict[Type, Callable[[Any], Walkable]] = {}


def register_walkable_adapter(
    target_type: Type, adapter_factory: Callable[[Any], Walkable]
) -> None:
    """Register an adapter for a type to make it walkable.

    Args:
        target_type: The type to register an adapter for
        adapter_factory: A callable that takes an instance and returns a Walkable

    Example:
        # Make dataclasses walkable as maps
        @dataclass
        class Person:
            name: str
            age: int

        class DataclassWalker(Walkable):
            def __init__(self, obj):
                self._obj = obj
                self._fields = fields(obj)

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                for f in self._fields:
                    yield f.name, getattr(self._obj, f.name)

        register_walkable_adapter(Person, DataclassWalker)
    """
    _walkable_adapters[target_type] = adapter_factory


def register_walkable(target_type: Type) -> Callable[[Type], Type]:
    """Decorator to register a Walkable adapter class for a type.

    Example:
        @register_walkable(MyCustomType)
        class MyCustomTypeWalker(Walkable):
            def __init__(self, obj):
                self._obj = obj

            def walk_type(self):
                return WalkType.MAP

            def walk_map_items(self):
                yield "field1", self._obj.field1
                yield "field2", self._obj.field2
    """

    def decorator(adapter_class: Type) -> Type:
        register_walkable_adapter(target_type, adapter_class)
        return adapter_class

    return decorator


# =============================================================================
# The Main Entry Point: as_walkable()
# =============================================================================


def as_walkable(value: Any) -> Walkable:
    """Convert any value to a Walkable.

    This is the main entry point for walking data structures. It handles:
    1. Values already implementing Walkable - returned as-is
    2. Native Python types - wrapped with built-in adapters
    3. Registered types - wrapped with registered adapters
    4. Unknown types - attempts to treat as scalar

    Args:
        value: Any value to make walkable

    Returns:
        A Walkable instance for the value

    Examples:
        # Native types
        w = as_walkable({"name": "Alice", "age": 30})
        assert w.walk_type() == WalkType.MAP

        w = as_walkable([1, 2, 3])
        assert w.walk_type() == WalkType.LIST

        w = as_walkable({1, 2, 3})
        assert w.walk_type() == WalkType.SET

        w = as_walkable("hello")
        assert w.walk_type() == WalkType.SCALAR

        # Custom types implementing Walkable
        class MyType(Walkable):
            ...
        w = as_walkable(MyType())  # Returns the instance itself
    """
    # Already walkable - return as-is
    if isinstance(value, Walkable):
        return value

    # Check registered adapters first (allows overriding built-in handling)
    value_type = type(value)
    if value_type in _walkable_adapters:
        return _walkable_adapters[value_type](value)

    # Check for subclass matches in registered adapters
    for registered_type, adapter_factory in _walkable_adapters.items():
        if isinstance(value, registered_type):
            return adapter_factory(value)

    # Built-in handling for native Python types
    if isinstance(value, dict):
        return WalkableDict(value)

    # Handle EDN ImmutableDict if available
    if HAS_EDN and ImmutableDict is not None and isinstance(value, ImmutableDict):
        return WalkableDict(dict(value))

    if isinstance(value, (list, tuple)):
        return WalkableList(value)

    if isinstance(value, (set, frozenset)):
        return WalkableSet(value)

    # Everything else is a scalar
    return WalkableScalar(value)


def is_walkable(value: Any) -> bool:
    """Check if a value can be walked (either implements Walkable or is a known type)."""
    if isinstance(value, Walkable):
        return True
    if isinstance(value, (dict, list, tuple, set, frozenset, str, int, float, bool)):
        return True
    if value is None:
        return True
    if HAS_EDN:
        if isinstance(value, (edn_format.Keyword, edn_format.Symbol)):
            return True
        if ImmutableDict is not None and isinstance(value, ImmutableDict):
            return True
    return type(value) in _walkable_adapters


# =============================================================================
# Utility: Walk Iterator
# =============================================================================


def walk_iter(value: Any) -> Iterator[Tuple[str, Any, Walkable]]:
    """Iterate over a walkable structure, yielding (path, value, walkable) tuples.

    Useful for debugging or inspecting the walk traversal.

    Args:
        value: Any walkable value

    Yields:
        (path, original_value, walkable) tuples
        - path: dot-notation path like "users.0.name"
        - original_value: the original Python value
        - walkable: the Walkable wrapper for the value
    """

    def _walk(v: Any, path: str) -> Iterator[Tuple[str, Any, Walkable]]:
        w = as_walkable(v)
        yield (path, v, w)

        wtype = w.walk_type()
        if wtype == WalkType.MAP:
            for k, child in w.walk_map_items():
                child_path = f"{path}.{k}" if path else str(k)
                yield from _walk(child, child_path)
        elif wtype == WalkType.LIST:
            for i, child in enumerate(w.walk_list_items()):
                child_path = f"{path}.{i}" if path else str(i)
                yield from _walk(child, child_path)
        elif wtype == WalkType.SET:
            for child in w.walk_set_items():
                # Sets don't have indices, use value as path component
                child_path = f"{path}.{{{child}}}" if path else f"{{{child}}}"
                yield from _walk(child, child_path)
        elif wtype == WalkType.SPREAD:
            for i, child in enumerate(w.walk_spread_items()):
                child_path = f"{path}.[{i}]" if path else f"[{i}]"
                yield from _walk(child, child_path)
        # SCALAR has no children

    yield from _walk(value, "")
