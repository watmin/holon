"""
Online Subspace Learning for Anomaly Detection (CCIPCA).

Learns the low-dimensional manifold that "familiar" hypervectors occupy,
then flags anything that doesn't project cleanly onto it.

Even though holon vectors live in 4096D, structured encodings from similar
schemas cluster on a much lower-dimensional manifold (k << d). CCIPCA
learns that manifold incrementally — O(k*d) per vector — without building
the covariance matrix or doing batch SVD.

Three outputs:
  - residual: ||x - proj(x)|| — distance from subspace (anomaly score)
  - projection: coefficients in basis coordinates (for clustering/viz)
  - threshold: adaptive cutoff via EMA of residual statistics

Algorithm: Candid Covariance-free Incremental PCA (Weng et al., 2003).
Tracks unnormalized eigenvectors whose norms approximate eigenvalues.

Usage:
    from holon.subspace import OnlineSubspace

    sub = OnlineSubspace(dim=4096, k=64)
    for vec in stream:
        residual = sub.update(vec)
        if residual > sub.threshold:
            print("anomaly")
"""

from typing import Optional, Tuple

import numpy as np


class OnlineSubspace:
    """Online subspace learner using CCIPCA.

    Args:
        dim: Vector dimensionality (must match input vectors).
        k: Number of principal components to track. Lower = faster,
           higher = tighter boundary. Typical: 32-128 for 4096D.
        amnesia: Forgetting exponent (>1 forgets old data faster).
            2.0 = moderate forgetting, 3.0 = aggressive. Set to 1.0
            for pure averaging (no forgetting).
        ema_alpha: EMA decay for threshold tracking (0.01 = slow, 0.1 = fast).
        sigma_mult: Number of standard deviations for threshold.
        reorth_interval: Re-orthogonalize basis every N updates.
            Prevents drift from accumulated numerical error.
    """

    def __init__(
        self,
        dim: int = 4096,
        k: int = 64,
        amnesia: float = 2.0,
        ema_alpha: float = 0.01,
        sigma_mult: float = 3.5,
        reorth_interval: int = 500,
    ):
        self.dim = dim
        self.k = min(k, dim)
        self.amnesia = amnesia
        self.ema_alpha = ema_alpha
        self.sigma_mult = sigma_mult
        self.reorth_interval = reorth_interval

        self.mean = np.zeros(dim, dtype=np.float64)
        self._components = np.zeros((self.k, dim), dtype=np.float64)
        self.n = 0

        self._res_ema = 0.0
        self._res_var_ema = 0.0
        self._initialized = False

    @property
    def threshold(self) -> float:
        """Adaptive anomaly threshold: EMA(residual) + sigma_mult * std."""
        if self.n < 2:
            return float("inf")
        return self._res_ema + self.sigma_mult * np.sqrt(max(self._res_var_ema, 0.0))

    @property
    def eigenvalues(self) -> np.ndarray:
        """Approximate eigenvalues (norms of unnormalized components)."""
        return np.array([np.linalg.norm(self._components[i]) for i in range(self.k)])

    @property
    def explained_ratio(self) -> float:
        """Fraction of variance explained by the subspace (0-1).

        Estimated from recent residuals: 1 - (mean_residual^2 / input_variance).
        Only meaningful after sufficient updates.
        """
        if self.n < 10:
            return 0.0
        total_var = self.dim  # unit-normalized input has variance ≈ dim
        unexplained = self._res_ema**2
        return max(0.0, 1.0 - unexplained / total_var)

    def _basis_vectors(self) -> np.ndarray:
        """Return unit-normalized basis vectors (k x dim)."""
        basis = np.empty_like(self._components)
        for i in range(self.k):
            norm = np.linalg.norm(self._components[i])
            if norm > 1e-10:
                basis[i] = self._components[i] / norm
            else:
                basis[i] = 0.0
        return basis

    def update(self, x: np.ndarray) -> float:
        """Update subspace with a new vector and return its residual.

        This is the main entry point for streaming use. Each call:
        1. Computes the residual with the *current* subspace (pre-update)
        2. Updates the running mean
        3. Updates all k principal components via CCIPCA
        4. Updates the adaptive threshold

        The residual is computed *before* the CCIPCA update so that it
        matches what residual() would return. This ensures the adaptive
        threshold is calibrated against the same distribution as test-time
        scoring.

        Args:
            x: Input vector (dim,). Can be int8 bipolar or float.

        Returns:
            Residual norm (anomaly score). Higher = more anomalous.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        assert x.shape[0] == self.dim, f"Expected dim={self.dim}, got {x.shape[0]}"

        # Compute residual BEFORE updating (matches test-time residual())
        if self._initialized:
            res = self.residual(x)
        else:
            res = float(np.linalg.norm(x))

        self.n += 1
        n = self.n
        amn = self.amnesia

        # Update running mean
        self.mean = ((n - 1) / n) * self.mean + (1.0 / n) * x

        # Center (for CCIPCA update)
        x_c = x - self.mean

        if not self._initialized and n == 1:
            self._components[0] = x_c.copy()
            self._initialized = True
            return res

        # CCIPCA update for each component
        for i in range(self.k):
            v = self._components[i]
            v_norm = np.linalg.norm(v)

            if v_norm < 1e-10:
                if np.linalg.norm(x_c) > 1e-10:
                    self._components[i] = x_c * ((1 + amn) / n)
            else:
                u = v / v_norm
                x_c_proj = np.dot(x_c, u)

                # Weng et al. CCIPCA update: track unnormalized eigenvectors
                self._components[i] = ((n - 1 - amn) / n) * v + (
                    (1 + amn) / n
                ) * x_c_proj * x_c

            # Deflate for next component
            v_new = self._components[i]
            v_new_norm = np.linalg.norm(v_new)
            if v_new_norm > 1e-10:
                u_new = v_new / v_new_norm
                x_c = x_c - np.dot(x_c, u_new) * u_new

        residual = res

        # Update adaptive threshold via EMA
        alpha = self.ema_alpha
        if self.n <= 1.0 / alpha:
            # Use simple average during warmup for stability
            alpha = 1.0 / self.n
        delta = residual - self._res_ema
        self._res_ema += alpha * delta
        self._res_var_ema = (1 - alpha) * self._res_var_ema + alpha * delta**2

        # Periodic re-orthogonalization to prevent basis drift
        if self.reorth_interval > 0 and self.n % self.reorth_interval == 0:
            self._reorthogonalize()

        return residual

    def residual(self, x: np.ndarray) -> float:
        """Compute residual without updating the subspace.

        Args:
            x: Input vector (dim,).

        Returns:
            Residual norm (anomaly score).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        x_c = x - self.mean

        for i in range(self.k):
            v = self._components[i]
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                continue
            u = v / v_norm
            x_c = x_c - np.dot(x_c, u) * u

        return float(np.linalg.norm(x_c))

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project vector onto the learned subspace.

        Returns coefficients in basis coordinates (k,). Useful for
        low-dimensional visualization and clustering.

        Args:
            x: Input vector (dim,).

        Returns:
            Projection coefficients (k,).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        x_c = x - self.mean
        basis = self._basis_vectors()
        return basis @ x_c

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct vector from its subspace projection.

        Args:
            x: Input vector (dim,).

        Returns:
            Reconstructed vector (dim,). The difference x - reconstruct(x)
            is the anomalous component.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        coeffs = self.project(x)
        basis = self._basis_vectors()
        return (coeffs @ basis) + self.mean

    def anomalous_component(self, x: np.ndarray) -> np.ndarray:
        """Extract the anomalous (out-of-subspace) component of a vector.

        This is x minus its subspace reconstruction — the signal that
        doesn't belong. Analogous to reject() but learned from data.

        Args:
            x: Input vector (dim,).

        Returns:
            Anomalous component vector (dim,).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        return x - self.reconstruct(x)

    def update_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Update subspace with a batch of vectors.

        Convenience method that calls update() on each vector.

        Args:
            vectors: Array of shape (n, dim).

        Returns:
            Array of residuals (n,).
        """
        vectors = np.atleast_2d(vectors)
        residuals = np.empty(len(vectors))
        for i, v in enumerate(vectors):
            residuals[i] = self.update(v)
        return residuals

    def _reorthogonalize(self):
        """Re-orthogonalize components via modified Gram-Schmidt.

        Preserves component norms (which approximate eigenvalues)
        while correcting angular drift between components.
        """
        norms = np.array([np.linalg.norm(self._components[i]) for i in range(self.k)])

        for i in range(self.k):
            if norms[i] < 1e-10:
                continue
            for j in range(i):
                if norms[j] < 1e-10:
                    continue
                u_j = self._components[j] / norms[j]
                self._components[i] -= np.dot(self._components[i], u_j) * u_j

            # Restore original norm (eigenvalue estimate)
            new_norm = np.linalg.norm(self._components[i])
            if new_norm > 1e-10:
                self._components[i] *= norms[i] / new_norm

    def snapshot(self) -> dict:
        """Export subspace state for persistence or shipping.

        Returns a compact dict that can be used to reconstruct the
        subspace on another node without the full training data.
        """
        return {
            "dim": self.dim,
            "k": self.k,
            "n": self.n,
            "mean": self.mean.copy(),
            "components": self._components.copy(),
            "res_ema": self._res_ema,
            "res_var_ema": self._res_var_ema,
            "threshold": self.threshold,
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "OnlineSubspace":
        """Restore subspace from a snapshot."""
        sub = cls(dim=snap["dim"], k=snap["k"])
        sub.n = snap["n"]
        sub.mean = snap["mean"].copy()
        sub._components = snap["components"].copy()
        sub._res_ema = snap["res_ema"]
        sub._res_var_ema = snap["res_var_ema"]
        sub._initialized = True
        return sub

    def __repr__(self) -> str:
        eigs = self.eigenvalues
        active = int(np.sum(eigs > 1e-6))
        return (
            f"OnlineSubspace(dim={self.dim}, k={self.k}, n={self.n}, "
            f"active_components={active}, threshold={self.threshold:.4f})"
        )
