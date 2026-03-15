# Extending the Walkable Interface

The Walkable interface provides zero-serialization encoding for in-memory data structures. This guide shows how to extend it for custom types.

## Quick Reference

| I want to... | Solution |
|-------------|----------|
| Encode a custom class | Implement `Walkable` |
| Encode a third-party class I can't modify | Use `@register_walkable` |
| Handle a custom scalar type | Return it from `walk_scalar_value()` |
| Change how a scalar encodes to vector | Subclass `Encoder` |

## The Five Structural Types

Walkable has five types that cover all data structures:

```python
class WalkType(Enum):
    SCALAR = "scalar"  # Atomic values (str, int, float, bool, None, etc.)
    MAP = "map"        # Key-value pairs (dict, objects, records)
    LIST = "list"      # Ordered sequences (list, tuple, array)
    SET = "set"        # Unordered unique items
    SPREAD = "spread"  # Fan-out: each element gets its own indexed leaf binding
```

Choose based on **structure**, not semantics:
- Your `Person` class with `name` and `age` fields? → `MAP`
- Your `TimeSeries` with ordered values? → `LIST`
- Your `TagSet` with unique tags? → `SET`
- Your `Money` amount? → `SCALAR`
- Your `CipherSuite` list you want per-element attribution on? → `SPREAD`

`SPREAD` behaves identically to `LIST` in the standard single-vector encode path. The
distinction only matters in **striped encoding**: `LIST` contributes one aggregate leaf
binding for the whole sequence, while `SPREAD` fans out to N indexed bindings
(`field.[0]`, `field.[1]`, ...) that land in different stripes and can be attributed
individually.

## Implementing Walkable for Custom Types

### Map-like Types (most common)

```python
from holon import Walkable, WalkType

class Order(Walkable):
    def __init__(self, order_id: str, items: list, total: float):
        self.order_id = order_id
        self.items = items
        self.total = total

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "order_id", self.order_id
        yield "items", self.items      # Nested list - works automatically
        yield "total", self.total

# Usage
order = Order("ORD-123", ["widget", "gadget"], 99.99)
vec = client.encode_walkable(order)
```

### List-like Types

```python
class TimeSeries(Walkable):
    def __init__(self, values: list[float]):
        self._values = values

    def walk_type(self) -> WalkType:
        return WalkType.LIST

    def walk_list_items(self):
        for v in self._values:
            yield v
```

### Set-like Types

```python
class TagSet(Walkable):
    def __init__(self, tags: set[str]):
        self._tags = tags

    def walk_type(self) -> WalkType:
        return WalkType.SET

    def walk_set_items(self):
        for tag in self._tags:
            yield tag
```

### Custom Scalars

```python
from decimal import Decimal

class Money(Walkable):
    def __init__(self, amount: Decimal, currency: str):
        self.amount = amount
        self.currency = currency

    def walk_type(self) -> WalkType:
        return WalkType.SCALAR

    def walk_scalar_value(self):
        # Return a string representation for encoding
        return f"{self.currency}:{self.amount}"
```

## Nested Walkable Types

Walkable types can nest arbitrarily. Return your nested Walkable from `walk_map_items()`, `walk_list_items()`, or `walk_set_items()`:

```python
class Address(Walkable):
    def __init__(self, city: str, country: str):
        self.city = city
        self.country = country

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "city", self.city
        yield "country", self.country


class Customer(Walkable):
    def __init__(self, name: str, address: Address):
        self.name = name
        self.address = address

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "name", self.name
        yield "address", self.address  # Nested Walkable - just works!
```

## Adapters for Third-Party Types

Can't modify a class? Register an adapter:

```python
from holon import register_walkable, Walkable, WalkType

# Third-party class you can't modify
class ExternalRecord:
    def __init__(self, id: str, data: dict):
        self.id = id
        self.data = data

# Register an adapter
@register_walkable(ExternalRecord)
class ExternalRecordAdapter(Walkable):
    def __init__(self, obj: ExternalRecord):
        self._obj = obj

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "id", self._obj.id
        for k, v in self._obj.data.items():
            yield k, v

# Now it works automatically
record = ExternalRecord("EXT-001", {"type": "event"})
vec = client.encode_walkable(record)  # Uses the adapter
```

## Scalar Encoding Details

The encoder handles scalar values through `_encode_scalar()`:

```python
# Built-in handling (in order of precedence)
str          → vector for that string
int, float   → vector for str(value)
edn.Keyword  → vector for ":name"
edn.Symbol   → vector for "name"
bool         → vector for "true" or "false"
None         → vector for "nil"
(fallback)   → vector for str(value)
```

**Your custom scalars just need to return something the encoder can handle.** Options:

1. **Return a string** (most common):
   ```python
   def walk_scalar_value(self):
       return f"{self.currency}:{self.amount}"
   ```

2. **Return a known type** (EDN keyword, etc.):
   ```python
   def walk_scalar_value(self):
       return edn_format.Keyword(self.tag_name)
   ```

3. **Extend the encoder** (for special vector encoding):
   ```python
   class MyEncoder(Encoder):
       def _encode_scalar(self, data):
           if isinstance(data, Money):
               # Custom encoding logic
               return self._encode_money(data)
           return super()._encode_scalar(data)
   ```

## Common Patterns

### Dataclasses

```python
from dataclasses import dataclass, fields

@dataclass
class Person:
    name: str
    age: int

@register_walkable(Person)
class DataclassWalker(Walkable):
    def __init__(self, obj):
        self._obj = obj

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        for f in fields(self._obj):
            yield f.name, getattr(self._obj, f.name)
```

### Named Tuples

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

@register_walkable(Point)
class PointWalker(Walkable):
    def __init__(self, obj):
        self._obj = obj

    def walk_type(self) -> WalkType:
        return WalkType.MAP  # Treat as map with field names

    def walk_map_items(self):
        for name in self._obj._fields:
            yield name, getattr(self._obj, name)
```

### Pydantic Models

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str

@register_walkable(User)
class PydanticWalker(Walkable):
    def __init__(self, obj):
        self._obj = obj

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        for k, v in self._obj.model_dump().items():
            yield k, v
```

### SQLAlchemy Models

```python
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.inspection import inspect

@register_walkable(DeclarativeBase)
class SQLAlchemyWalker(Walkable):
    def __init__(self, obj):
        self._obj = obj

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        mapper = inspect(self._obj.__class__)
        for col in mapper.columns:
            yield col.key, getattr(self._obj, col.key)
```

## Debugging with walk_iter

Use `walk_iter()` to inspect how your structure will be traversed:

```python
from holon import walk_iter

customer = Customer("Alice", Address("NYC", "USA"))

for path, value, walkable in walk_iter(customer):
    wtype = walkable.walk_type()
    print(f"{path or 'root'}: {wtype.value}")

# Output:
# root: map
# name: scalar
# address: map
# address.city: scalar
# address.country: scalar
```

## Performance Tips

1. **Use `__slots__`** in your Walkable wrappers:
   ```python
   class MyWalker(Walkable):
       __slots__ = ('_obj',)
       def __init__(self, obj):
           self._obj = obj
   ```

2. **Avoid copies** - yield references, don't build intermediate structures:
   ```python
   # Good - yields directly
   def walk_map_items(self):
       yield "name", self._obj.name

   # Bad - creates intermediate dict
   def walk_map_items(self):
       d = {"name": self._obj.name, ...}
       yield from d.items()
   ```

3. **Register adapters once** at import time, not per-call.

## Using SPREAD for Striped Encoding

`WalkableSpread` is a ready-made wrapper for sequences whose elements you want to
attribute individually when using `encode_walkable_striped`:

```python
from holon import Encoder, WalkableSpread, StripedSubspace

encoder = Encoder(dim=4096, seed=42)
n_stripes = 8

# Each cipher lands in a deterministic stripe by FNV-1a hash of "ciphers.[0]", etc.
record = {
    "version": "TLS1.3",
    "ciphers": WalkableSpread(["AES256-GCM", "AES128-GCM", "CHACHA20"]),
}

stripes = encoder.encode_walkable_striped(record, n_stripes)
# stripes[i] is the vector contribution from fields that hash to stripe i

subspace = StripedSubspace(dim=4096, k=32, n_stripes=n_stripes)
subspace.update(stripes)
profile = subspace.residual_profile(stripes)  # per-stripe anomaly scores
```

Use a plain `list` (or `WalkType.LIST`) when you want the whole sequence to behave as a
single encoded unit. Use `WalkableSpread` when individual elements should be isolatable
in residual profiling and drilldown attribution.

## Summary

- **5 structural types**: SCALAR, MAP, LIST, SET, SPREAD
- **Implement Walkable** for your own types
- **Use `@register_walkable`** for third-party types
- **Scalars return any value** - encoder handles encoding
- **Nesting works automatically** - just yield Walkables
- **Use `WalkableSpread`** for per-element attribution in striped encoding
