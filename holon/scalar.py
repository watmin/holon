"""Continuous scalar value encoding (linear, log, circular).

Alias for holon.kernel.scalar — both paths are supported.
"""

from .kernel.scalar import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "encode_scalar",
    "encode_scalar_log",
    "decode_scalar_log",
    "encode_circular",
    "encode_positional",
]
