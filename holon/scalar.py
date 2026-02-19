"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.kernel.scalar import *

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .kernel.scalar import *  # noqa: F401, F403

__all__ = [
    'encode_scalar',
    'encode_scalar_log',
    'decode_scalar_log',
    'encode_circular',
    'encode_positional',
]
