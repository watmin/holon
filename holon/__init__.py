# Holon: Programmatic Neural Memory
# Version: 0.1.0

from .client import HolonClient  # Unified local/remote client
from .cpu_store import CPUStore  # Supports both CPU and GPU backends
from .store import Store

__version__ = "0.1.0"
