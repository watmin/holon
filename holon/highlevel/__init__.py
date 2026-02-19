"""
Holon High-Level API - Convenience and Composition

The highlevel layer provides opinionated, convenient interfaces:
- HolonClient: Unified facade for encoding, searching, and memory operations
- Query DSL with guards, negations, and $markers
- Integration with stores and backends

This layer depends on both holon.kernel and holon.memory.
"""

from .client import HolonClient

__all__ = [
    "HolonClient",
]
