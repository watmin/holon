import numpy as np
from numba import jit
from typing import Dict, Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class VectorManager:
    def __init__(self, dimensions: int = 16000, backend: str = 'cpu'):
        self.dimensions = dimensions
        self.backend = backend
        self.atom_vectors: Dict[str, Union[np.ndarray, 'cp.ndarray']] = {}

        if backend == 'cpu':
            self.np = np
            self.rng = np.random.RandomState(42)
        elif backend == 'gpu' and CUPY_AVAILABLE:
            self.np = cp
            self.rng = cp.random.RandomState(42)
        else:
            raise ValueError(f"Backend {backend} not supported or CuPy not available")

    def get_vector(self, atom: str) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Get or create a vector for an atom.
        Vectors are bipolar: {-1, 0, 1}.

        :param atom: The atom string.
        :return: The vector.
        """
        if atom not in self.atom_vectors:
            # Generate random bipolar vector
            vector = self.rng.choice([-1, 0, 1], size=self.dimensions).astype(self.np.int8)
            self.atom_vectors[atom] = vector
        return self.atom_vectors[atom]

    def to_cpu(self, vector: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Convert vector to CPU numpy array."""
        if self.backend == 'gpu' and CUPY_AVAILABLE:
            return cp.asnumpy(vector)
        return vector

    def to_backend(self, vector: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Convert CPU vector to backend array."""
        if self.backend == 'gpu' and CUPY_AVAILABLE:
            return cp.asarray(vector)
        return vector

    def clear(self):
        """Clear all stored vectors."""
        self.atom_vectors.clear()