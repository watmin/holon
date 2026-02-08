#!/usr/bin/env python3
"""
Walkable Interface Demo

This demonstrates the zero-serialization encoding path in Holon.
Instead of converting data to JSON/EDN strings, you can encode
in-memory data structures directly.

Run with: ./scripts/run_with_venv.sh python examples/walkable_demo.py
"""

import numpy as np

from holon import (
    HolonClient,
    Walkable,
    WalkType,
    as_walkable,
    register_walkable,
    walk_iter,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# Example 1: Native Python types work automatically
# =============================================================================


def demo_native_types():
    print("=" * 60)
    print("Demo 1: Native Python Types")
    print("=" * 60)

    client = HolonClient(dimensions=4096)

    # Dicts, lists, sets, and scalars all work automatically
    dict_data = {"type": "billing", "amount": 100, "status": "paid"}
    list_data = ["apple", "banana", "cherry"]
    set_data = {"python", "rust", "holon"}

    # Encode using walkable (no serialization needed!)
    vec_dict = client.encode_walkable(dict_data)
    vec_list = client.encode_walkable(list_data)
    vec_set = client.encode_walkable(set_data)

    print(f"Dict vector shape: {vec_dict.shape}")
    print(f"List vector shape: {vec_list.shape}")
    print(f"Set vector shape:  {vec_set.shape}")

    # Compare with traditional encode() - should be identical
    vec_traditional = client.encode(dict_data)
    assert np.array_equal(vec_dict, vec_traditional)
    print("\n✓ encode_walkable() produces identical vectors to encode()")


# =============================================================================
# Example 2: Custom types implementing Walkable
# =============================================================================


class Order(Walkable):
    """A custom order type that implements Walkable."""

    def __init__(self, order_id: str, customer: str, items: list, total: float):
        self.order_id = order_id
        self.customer = customer
        self.items = items
        self.total = total

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "order_id", self.order_id
        yield "customer", self.customer
        yield "items", self.items  # Nested list works automatically
        yield "total", self.total


def demo_custom_walkable():
    print("\n" + "=" * 60)
    print("Demo 2: Custom Walkable Types")
    print("=" * 60)

    client = HolonClient(dimensions=4096)

    # Create custom objects
    order1 = Order("ORD-001", "Alice", ["widget", "gadget"], 99.99)
    order2 = Order("ORD-002", "Alice", ["widget", "tool"], 149.99)
    order3 = Order("ORD-003", "Bob", ["book", "lamp"], 45.00)

    # Encode directly - no JSON serialization!
    vec1 = client.encode_walkable(order1)
    vec2 = client.encode_walkable(order2)
    vec3 = client.encode_walkable(order3)

    print(f"Order 1 vector: {vec1.shape}, non-zero: {np.count_nonzero(vec1)}")

    # Similar orders have higher similarity
    sim_12 = cosine_similarity(vec1, vec2)
    sim_13 = cosine_similarity(vec1, vec3)

    print(f"\nSimilarity (Order1 vs Order2): {sim_12:.4f}")
    print(f"Similarity (Order1 vs Order3): {sim_13:.4f}")
    print("\n✓ Orders with shared customer/items are more similar")


# =============================================================================
# Example 3: Nested custom types
# =============================================================================


class Address(Walkable):
    """A nested walkable type."""

    def __init__(self, city: str, country: str):
        self.city = city
        self.country = country

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "city", self.city
        yield "country", self.country


class Customer(Walkable):
    """A walkable with nested walkable."""

    def __init__(self, name: str, email: str, address: Address):
        self.name = name
        self.email = email
        self.address = address  # Nested Walkable

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "name", self.name
        yield "email", self.email
        yield "address", self.address  # Nested walkable works!


def demo_nested_walkable():
    print("\n" + "=" * 60)
    print("Demo 3: Nested Walkable Types")
    print("=" * 60)

    client = HolonClient(dimensions=4096)

    # Create nested structure
    addr = Address("NYC", "USA")
    customer = Customer("Alice", "alice@example.com", addr)

    vec = client.encode_walkable(customer)
    print(f"Customer vector: {vec.shape}")

    # Walk and inspect the structure
    print("\nWalk traversal:")
    for path, value, walkable in walk_iter(customer):
        wtype = walkable.walk_type()
        if wtype == WalkType.SCALAR:
            print(f"  {path or 'root'}: {value} (scalar)")
        else:
            print(f"  {path or 'root'}: <{wtype.value}>")


# =============================================================================
# Example 4: Registering adapters for third-party types
# =============================================================================


# Imagine this is a third-party class you can't modify
class ExternalRecord:
    """A class from an external library."""

    def __init__(self, id: str, data: dict):
        self.id = id
        self.data = data


# Register an adapter for it
@register_walkable(ExternalRecord)
class ExternalRecordWalker(Walkable):
    """Adapter to make ExternalRecord walkable."""

    def __init__(self, obj: ExternalRecord):
        self._obj = obj

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self):
        yield "id", self._obj.id
        for k, v in self._obj.data.items():
            yield k, v


def demo_adapter_registration():
    print("\n" + "=" * 60)
    print("Demo 4: Adapter Registration")
    print("=" * 60)

    client = HolonClient(dimensions=4096)

    # External record is now walkable!
    record = ExternalRecord("EXT-001", {"type": "event", "source": "webhook"})

    # as_walkable() automatically uses the registered adapter
    walkable = as_walkable(record)
    print(f"Adapter class: {walkable.__class__.__name__}")

    # Encode works seamlessly
    vec = client.encode_walkable(record)
    print(f"Encoded vector: {vec.shape}")


# =============================================================================
# Example 5: Performance comparison
# =============================================================================


def demo_performance():
    print("\n" + "=" * 60)
    print("Demo 5: Performance Comparison")
    print("=" * 60)

    import json
    import time

    client = HolonClient(dimensions=4096)

    # Create test data
    data = {
        "type": "transaction",
        "amount": 1234.56,
        "currency": "USD",
        "merchant": {"name": "Acme Corp", "category": "retail"},
        "items": [{"sku": f"ITEM-{i}", "qty": i} for i in range(10)],
    }

    iterations = 1000

    # Traditional path: dict -> JSON string -> parse -> encode
    start = time.perf_counter()
    for _ in range(iterations):
        json_str = json.dumps(data)
        _ = client.encode(json_str)
    traditional_time = time.perf_counter() - start

    # Walkable path: dict -> encode directly
    start = time.perf_counter()
    for _ in range(iterations):
        _ = client.encode_walkable(data)
    walkable_time = time.perf_counter() - start

    print(f"Traditional (via JSON): {traditional_time:.3f}s for {iterations} encodes")
    print(f"Walkable (direct):      {walkable_time:.3f}s for {iterations} encodes")
    speedup = traditional_time / walkable_time
    print(f"\nSpeedup: {speedup:.2f}x faster with walkable path")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_native_types()
    demo_custom_walkable()
    demo_nested_walkable()
    demo_adapter_registration()
    demo_performance()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)
