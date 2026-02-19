"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.memory.subspace import OnlineSubspace

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .memory.subspace import *  # noqa: F401, F403

__all__ = ['OnlineSubspace']
