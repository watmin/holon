"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.highlevel.client import HolonClient

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .highlevel.client import *  # noqa: F401, F403

__all__ = ['HolonClient']
