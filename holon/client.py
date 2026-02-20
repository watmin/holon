"""HolonClient convenience facade.

Alias for holon.highlevel.client — both paths are supported.
"""

from .highlevel.client import *  # noqa: F401, F403

__all__ = ["HolonClient"]  # noqa: F405
