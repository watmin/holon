"""Accumulator primitives for frequency-preserving streaming.

Alias for holon.kernel.accumulator — both paths are supported.
"""

from .kernel.accumulator import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "create_accumulator",
    "accumulate",
    "accumulate_weighted",
    "normalize_accumulator",
    "decay",
    "clear_accumulator",
    "merge_accumulators",
    "threshold_accumulator",
    "capacity",
    "purity",
    "participation_ratio",
]
