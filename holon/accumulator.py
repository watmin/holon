"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.kernel.accumulator import *

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .kernel.accumulator import *  # noqa: F401, F403

__all__ = ['create_accumulator', 'accumulate', 'accumulate_weighted',
           'normalize_accumulator', 'decay', 'clear_accumulator',
           'merge_accumulators', 'threshold_accumulator', 'capacity',
           'purity', 'participation_ratio']
