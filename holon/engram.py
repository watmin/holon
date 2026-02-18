"""
Engram: learned manifold snapshots for pattern recognition and replay.

An Engram is the stored memory of a learned subspace — the mathematical
trace of a pattern encountered in data. Named after the neuroscience
concept of a physical memory trace stored in neural tissue.

Engrams capture:
  - The subspace manifold (mean + principal components)
  - An eigenvalue signature for cheap pre-filtering
  - An optional surprise profile (per-field anomaly magnitudes)
  - Arbitrary metadata (labels, rules, timestamps, tags)

EngramLibrary manages a collection of engrams with two-tier matching:
  1. Fast eigenvalue cosine pre-filter — O(library_size * k)
  2. Full residual verification on top candidates — O(candidates * k * d)

Usage:
    from holon.engram import Engram, EngramLibrary
    from holon.subspace import OnlineSubspace

    library = EngramLibrary()

    # Train a subspace, then mint an engram
    sub = OnlineSubspace(dim=4096, k=64)
    for vec in attack_stream:
        sub.update(vec)
    library.add("dns_amp", sub, rule="((and (= dst_port 53)) => (drop))")

    # Later, match new traffic against the library
    matches = library.match(new_vec, top_k=3)
    if matches:
        name, score = matches[0]
        print(f"Matched engram '{name}' with score {score:.4f}")
"""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .subspace import OnlineSubspace


class Engram:
    """A stored memory trace of a learned subspace manifold.

    Args:
        name: Human-readable label for this engram.
        snapshot: Subspace state from OnlineSubspace.snapshot().
        eigenvalue_signature: Normalized eigenvalue spectrum (k,).
        surprise_profile: Optional per-field anomaly magnitudes.
        metadata: Arbitrary key-value pairs (rules, timestamps, tags).
    """

    def __init__(
        self,
        name: str,
        snapshot: dict,
        eigenvalue_signature: np.ndarray,
        surprise_profile: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self._snapshot = snapshot
        self.eigenvalue_signature = eigenvalue_signature
        self.surprise_profile = surprise_profile or {}
        self.metadata = metadata or {}
        self._subspace: Optional[OnlineSubspace] = None

    @property
    def subspace(self) -> OnlineSubspace:
        """Lazily reconstruct the OnlineSubspace from the stored snapshot."""
        if self._subspace is None:
            self._subspace = OnlineSubspace.from_snapshot(self._snapshot)
        return self._subspace

    def residual(self, vec: np.ndarray) -> float:
        """Compute how well a vector fits this engram's manifold."""
        return self.subspace.residual(vec)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        snap = self._snapshot
        return {
            "name": self.name,
            "snapshot": {
                "dim": snap["dim"],
                "k": snap["k"],
                "n": snap["n"],
                "mean": _ndarray_to_b64(snap["mean"]),
                "components": _ndarray_to_b64(snap["components"]),
                "res_ema": snap["res_ema"],
                "res_var_ema": snap["res_var_ema"],
                "threshold": snap["threshold"],
            },
            "eigenvalue_signature": _ndarray_to_b64(self.eigenvalue_signature),
            "surprise_profile": self.surprise_profile,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Engram":
        """Deserialize from a JSON-compatible dict."""
        snap_data = data["snapshot"]
        snapshot = {
            "dim": snap_data["dim"],
            "k": snap_data["k"],
            "n": snap_data["n"],
            "mean": _b64_to_ndarray(snap_data["mean"]),
            "components": _b64_to_ndarray(snap_data["components"]),
            "res_ema": snap_data["res_ema"],
            "res_var_ema": snap_data["res_var_ema"],
            "threshold": snap_data["threshold"],
        }
        return cls(
            name=data["name"],
            snapshot=snapshot,
            eigenvalue_signature=_b64_to_ndarray(data["eigenvalue_signature"]),
            surprise_profile=data.get("surprise_profile", {}),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        n = self._snapshot.get("n", 0)
        k = self._snapshot.get("k", 0)
        meta_keys = list(self.metadata.keys())
        return f"Engram('{self.name}', n={n}, k={k}, " f"metadata_keys={meta_keys})"


class EngramLibrary:
    """A collection of engrams with two-tier matching.

    Tier 1 (fast): Eigenvalue cosine similarity pre-filter.
    Tier 2 (full): Residual computation on candidate subspaces.

    Args:
        dim: Expected vector dimensionality. Used for validation only.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self._engrams: Dict[str, Engram] = {}

    def add(
        self,
        name: str,
        subspace: OnlineSubspace,
        surprise_profile: Optional[Dict[str, float]] = None,
        **metadata: Any,
    ) -> Engram:
        """Mint and store an engram from a trained subspace.

        Args:
            name: Unique label for this engram.
            subspace: Trained OnlineSubspace to snapshot.
            surprise_profile: Optional per-field anomaly magnitudes.
            **metadata: Arbitrary key-value pairs stored with the engram.

        Returns:
            The minted Engram.
        """
        eig = subspace.eigenvalues
        eig_norm = np.linalg.norm(eig)
        sig = eig / eig_norm if eig_norm > 1e-10 else eig

        engram = Engram(
            name=name,
            snapshot=subspace.snapshot(),
            eigenvalue_signature=sig,
            surprise_profile=surprise_profile,
            metadata=metadata,
        )
        self._engrams[name] = engram
        return engram

    def match(
        self,
        vec: np.ndarray,
        top_k: int = 3,
        prefilter_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Two-tier matching against all stored engrams.

        Tier 1: Rank all engrams by eigenvalue cosine similarity (cheap).
        Tier 2: Compute full residual on top prefilter_k candidates.

        Lower residual = better match (vector fits the engram's manifold).

        Args:
            vec: Input vector to match.
            top_k: Number of best matches to return.
            prefilter_k: Number of candidates to pass to tier-2.

        Returns:
            List of (name, residual) tuples, sorted by residual ascending.
            Empty list if library is empty.
        """
        if not self._engrams:
            return []

        vec = np.asarray(vec, dtype=np.float64).ravel()

        candidates = list(self._engrams.values())
        if len(candidates) > prefilter_k:
            candidates = self._prefilter(candidates, prefilter_k)

        results = []
        for engram in candidates:
            res = engram.residual(vec)
            results.append((engram.name, res))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def match_spectrum(
        self,
        eigenvalues: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Eigenvalue-only pre-filter for bulk screening.

        Compares an eigenvalue spectrum against all stored engrams using
        cosine similarity. No full residual computation — very fast.

        Args:
            eigenvalues: Eigenvalue spectrum to compare (k,).
            top_k: Number of best matches to return.

        Returns:
            List of (name, cosine_similarity) tuples, sorted descending.
        """
        if not self._engrams:
            return []

        eig = np.asarray(eigenvalues, dtype=np.float64)
        eig_norm = np.linalg.norm(eig)
        if eig_norm < 1e-10:
            return []
        eig_normed = eig / eig_norm

        results = []
        for engram in self._engrams.values():
            sig = engram.eigenvalue_signature
            min_len = min(len(eig_normed), len(sig))
            cos = float(np.dot(eig_normed[:min_len], sig[:min_len]))
            results.append((engram.name, cos))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _prefilter(
        self,
        candidates: List[Engram],
        top_k: int,
    ) -> List[Engram]:
        """Rank candidates by eigenvalue signature similarity."""
        if len(candidates) <= top_k:
            return candidates

        scored = []
        for eng in candidates:
            energy = float(np.sum(eng.eigenvalue_signature**2))
            scored.append((energy, eng))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [eng for _, eng in scored[:top_k]]

    def remove(self, name: str) -> bool:
        """Remove an engram by name. Returns True if it existed."""
        return self._engrams.pop(name, None) is not None

    def names(self) -> List[str]:
        """List all engram names."""
        return list(self._engrams.keys())

    def get(self, name: str) -> Optional[Engram]:
        """Retrieve an engram by name."""
        return self._engrams.get(name)

    def __len__(self) -> int:
        return len(self._engrams)

    def __contains__(self, name: str) -> bool:
        return name in self._engrams

    def save(self, path: str) -> None:
        """Persist library to a JSON file.

        Numpy arrays are base64-encoded for portability.
        """
        data = {
            "dim": self.dim,
            "engrams": [eng.to_dict() for eng in self._engrams.values()],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "EngramLibrary":
        """Load library from a JSON file."""
        data = json.loads(Path(path).read_text())
        lib = cls(dim=data["dim"])
        for eng_data in data["engrams"]:
            engram = Engram.from_dict(eng_data)
            lib._engrams[engram.name] = engram
        return lib

    def __repr__(self) -> str:
        names = ", ".join(f"'{n}'" for n in list(self._engrams.keys())[:5])
        if len(self._engrams) > 5:
            names += ", ..."
        return f"EngramLibrary(dim={self.dim}, count={len(self)}, engrams=[{names}])"


def _ndarray_to_b64(arr: np.ndarray) -> dict:
    """Encode a numpy array as a base64 string with dtype/shape metadata."""
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _b64_to_ndarray(obj: dict) -> np.ndarray:
    """Decode a numpy array from base64 representation."""
    raw = base64.b64decode(obj["data"])
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
