"""
Holon Memory - Programmatic Neural Memory System

The memory layer provides the novel memory primitives built on top of the kernel:
- OnlineSubspace: CCIPCA-based manifold learning for anomaly detection
- Engram/EngramLibrary: Stored memory traces of learned patterns

This layer depends on holon.kernel but not on holon.highlevel.
"""

from .subspace import OnlineSubspace
from .engram import Engram, EngramLibrary

__all__ = [
    "OnlineSubspace",
    "Engram",
    "EngramLibrary",
]
