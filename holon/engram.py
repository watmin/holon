"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.memory.engram import Engram, EngramLibrary

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .memory.engram import *  # noqa: F401, F403

__all__ = ['Engram', 'EngramLibrary']
