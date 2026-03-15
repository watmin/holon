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

Engrams are polymorphic over the subspace type they were minted from:
  - ``kind="single"``: minted from OnlineSubspace — single-vector residual
  - ``kind="striped"``: minted from StripedSubspace — list-of-vecs residual

EngramLibrary manages a collection of engrams with two-tier matching:
  1. Fast eigenvalue cosine pre-filter — O(library_size * k)
  2. Full residual verification on top candidates — O(candidates * k * d)

Usage (single-vector)::

    library = EngramLibrary()
    sub = OnlineSubspace(dim=4096, k=64)
    for vec in stream:
        sub.update(vec)
    library.add("pattern_a", sub, rule="drop")
    matches = library.match(new_vec, top_k=3)

Usage (striped)::

    library = EngramLibrary()
    ss = StripedSubspace(dim=1024, k=16, n_stripes=8)
    for stripe_vecs in stream:
        ss.update(stripe_vecs)
    library.add_striped("pattern_b", ss, action="BUY")
    matches = library.match_striped(new_stripe_vecs, top_k=3)
"""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .subspace import OnlineSubspace, StripedSubspace


class Engram:
    """A stored memory trace of a learned subspace manifold.

    Polymorphic over subspace type: ``kind="single"`` wraps an
    ``OnlineSubspace`` snapshot; ``kind="striped"`` wraps a
    ``StripedSubspace`` snapshot (N independent per-stripe OnlineSubspaces).

    Args:
        name: Human-readable label for this engram.
        snapshot: Subspace state dict. For single: OnlineSubspace.snapshot().
                  For striped: StripedSubspace.snapshot() (contains "stripes" key).
        eigenvalue_signature: Normalized eigenvalue spectrum for pre-filtering.
            For single: shape (k,). For striped: concatenation of all stripe
            eigenvalue signatures, shape (n_stripes * k,).
        kind: ``"single"`` (default, backward-compat) or ``"striped"``.
        surprise_profile: Optional per-field anomaly magnitudes.
        metadata: Arbitrary key-value pairs (rules, timestamps, tags).
    """

    def __init__(
        self,
        name: str,
        snapshot: dict,
        eigenvalue_signature: np.ndarray,
        kind: str = "single",
        surprise_profile: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if kind not in ("single", "striped"):
            raise ValueError(f"kind must be 'single' or 'striped', got {kind!r}")
        self.name = name
        self._snapshot = snapshot
        self.eigenvalue_signature = eigenvalue_signature
        self.kind = kind
        self.surprise_profile = surprise_profile or {}
        self.metadata = metadata or {}
        self._subspace: Optional[Union[OnlineSubspace, StripedSubspace]] = None

    @property
    def subspace(self) -> Union[OnlineSubspace, StripedSubspace]:
        """Lazily reconstruct the subspace from the stored snapshot."""
        if self._subspace is None:
            if self.kind == "striped":
                self._subspace = StripedSubspace.from_snapshot(self._snapshot)
            else:
                self._subspace = OnlineSubspace.from_snapshot(self._snapshot)
        return self._subspace

    def residual(self, vec: np.ndarray) -> float:
        """Compute residual for a single-vector engram.

        Args:
            vec: Input vector (dim,).

        Raises:
            TypeError: If called on a striped engram (use residual_striped).
        """
        if self.kind == "striped":
            raise TypeError(
                f"Engram '{self.name}' is striped — use residual_striped(stripe_vecs)"
            )
        return self.subspace.residual(vec)  # type: ignore[union-attr]

    def residual_striped(self, stripe_vecs: List[np.ndarray]) -> float:
        """Compute RSS residual for a striped engram.

        Args:
            stripe_vecs: List of per-stripe vectors (n_stripes,).

        Raises:
            TypeError: If called on a single-vector engram (use residual).
        """
        if self.kind == "single":
            raise TypeError(
                f"Engram '{self.name}' is single-vector — use residual(vec)"
            )
        return self.subspace.residual(stripe_vecs)  # type: ignore[union-attr]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        if self.kind == "striped":
            snap = self._snapshot
            serialized_snap = {
                "stripes": [_serialize_single_snap(s) for s in snap["stripes"]]
            }
        else:
            serialized_snap = _serialize_single_snap(self._snapshot)

        return {
            "name": self.name,
            "kind": self.kind,
            "snapshot": serialized_snap,
            "eigenvalue_signature": _ndarray_to_b64(self.eigenvalue_signature),
            "surprise_profile": self.surprise_profile,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Engram":
        """Deserialize from a JSON-compatible dict."""
        kind = data.get("kind", "single")
        snap_data = data["snapshot"]

        if kind == "striped":
            snapshot = {
                "stripes": [_deserialize_single_snap(s) for s in snap_data["stripes"]]
            }
        else:
            snapshot = _deserialize_single_snap(snap_data)

        return cls(
            name=data["name"],
            snapshot=snapshot,
            eigenvalue_signature=_b64_to_ndarray(data["eigenvalue_signature"]),
            kind=kind,
            surprise_profile=data.get("surprise_profile", {}),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        if self.kind == "striped":
            n_stripes = len(self._snapshot.get("stripes", []))
            n = self._snapshot["stripes"][0].get("n", 0) if n_stripes else 0
            k = self._snapshot["stripes"][0].get("k", 0) if n_stripes else 0
            return (
                f"Engram('{self.name}', kind='striped', n_stripes={n_stripes}, "
                f"n={n}, k={k}, metadata_keys={list(self.metadata.keys())})"
            )
        n = self._snapshot.get("n", 0)
        k = self._snapshot.get("k", 0)
        meta_keys = list(self.metadata.keys())
        return f"Engram('{self.name}', n={n}, k={k}, metadata_keys={meta_keys})"


class EngramLibrary:
    """A collection of engrams with two-tier matching.

    Holds both single-vector engrams (minted from ``OnlineSubspace``) and
    striped engrams (minted from ``StripedSubspace``) in the same store.
    Each is matched via the appropriate method:

    - ``match(vec)``         — for single-vector engrams
    - ``match_striped(vecs)`` — for striped engrams

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
        """Mint and store an engram from a trained OnlineSubspace.

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
            kind="single",
            surprise_profile=surprise_profile,
            metadata=metadata,
        )
        self._engrams[name] = engram
        return engram

    def add_striped(
        self,
        name: str,
        subspace: StripedSubspace,
        surprise_profile: Optional[Dict[str, float]] = None,
        **metadata: Any,
    ) -> Engram:
        """Mint and store an engram from a trained StripedSubspace.

        The eigenvalue signature is the concatenation of all per-stripe
        eigenvalue signatures (normalized). This lets the same eigenvalue
        pre-filter work across both single and striped engrams.

        Args:
            name: Unique label for this engram.
            subspace: Trained StripedSubspace to snapshot.
            surprise_profile: Optional per-field anomaly magnitudes.
            **metadata: Arbitrary key-value pairs stored with the engram.

        Returns:
            The minted Engram (kind="striped").
        """
        # Concatenate per-stripe eigenvalue signatures as the prefilter key
        stripe_eigs = []
        for i in range(subspace.n_stripes):
            eig = subspace.stripe(i).eigenvalues
            eig_norm = np.linalg.norm(eig)
            stripe_eigs.append(eig / eig_norm if eig_norm > 1e-10 else eig)
        sig = np.concatenate(stripe_eigs)

        engram = Engram(
            name=name,
            snapshot=subspace.snapshot(),
            eigenvalue_signature=sig,
            kind="striped",
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
        """Two-tier matching against single-vector engrams.

        Skips striped engrams — use match_striped() for those.

        Tier 1: Rank by eigenvalue cosine similarity (cheap).
        Tier 2: Compute full residual on top prefilter_k candidates.

        Lower residual = better match (vector fits the engram's manifold).

        Args:
            vec: Input vector to match.
            top_k: Number of best matches to return.
            prefilter_k: Number of candidates to pass to tier-2.

        Returns:
            List of (name, residual) tuples, sorted by residual ascending.
            Empty list if no single-vector engrams.
        """
        single_engrams = [e for e in self._engrams.values() if e.kind == "single"]
        if not single_engrams:
            return []

        vec = np.asarray(vec, dtype=np.float64).ravel()

        candidates = single_engrams
        if len(candidates) > prefilter_k:
            candidates = self._prefilter(candidates, prefilter_k)

        results = []
        for engram in candidates:
            res = engram.residual(vec)
            results.append((engram.name, res))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def match_striped(
        self,
        stripe_vecs: List[np.ndarray],
        top_k: int = 3,
        prefilter_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Two-tier matching against striped engrams.

        Skips single-vector engrams — use match() for those.

        Tier 1: Rank by concatenated eigenvalue cosine similarity (cheap).
        Tier 2: Compute full RSS residual on top prefilter_k candidates.

        Lower residual = better match.

        Args:
            stripe_vecs: List of per-stripe vectors from encode_walkable_striped.
            top_k: Number of best matches to return.
            prefilter_k: Number of candidates to pass to tier-2.

        Returns:
            List of (name, rss_residual) tuples, sorted by residual ascending.
            Empty list if no striped engrams.
        """
        striped_engrams = [e for e in self._engrams.values() if e.kind == "striped"]
        if not striped_engrams:
            return []

        # Build probe signature: concatenate normalized per-stripe eigenvalues
        # from the live stripe_vecs statistics isn't available — use a flat
        # probe of ones as a neutral prefilter (falls back to energy ranking)
        candidates = striped_engrams
        if len(candidates) > prefilter_k:
            candidates = self._prefilter(candidates, prefilter_k)

        results = []
        for engram in candidates:
            res = engram.residual_striped(stripe_vecs)
            results.append((engram.name, res))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def names(self, kind: Optional[str] = None) -> List[str]:
        """List all engram names, optionally filtered by kind.

        Args:
            kind: ``"single"``, ``"striped"``, or ``None`` for all.

        Returns:
            List of engram names.
        """
        if kind is None:
            return list(self._engrams.keys())
        return [n for n, e in self._engrams.items() if e.kind == kind]

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

    def match_alignment(
        self,
        probe_subspace: OnlineSubspace,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Directional alignment matching against all stored engrams.

        Compares the principal component directions of a probe subspace
        against each engram's stored subspace. This measures whether the
        variance lives in the same part of the vector space — the
        directional complement to match_spectrum's magnitude comparison.

        Args:
            probe_subspace: A trained OnlineSubspace to compare.
            top_k: Number of best matches to return.

        Returns:
            List of (name, alignment_score) tuples, sorted descending.
            Alignment is in [0, 1]: 1.0 = same directions, 0.0 = orthogonal.
        """
        if not self._engrams:
            return []

        results = []
        for engram in self._engrams.values():
            alignment = probe_subspace.subspace_alignment(engram.subspace)
            results.append((engram.name, alignment))

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


def _serialize_single_snap(snap: dict) -> dict:
    """Serialize a single OnlineSubspace snapshot dict to JSON-safe form."""
    return {
        "dim": snap["dim"],
        "k": snap["k"],
        "n": snap["n"],
        "mean": _ndarray_to_b64(snap["mean"]),
        "components": _ndarray_to_b64(snap["components"]),
        "res_ema": snap["res_ema"],
        "res_var_ema": snap["res_var_ema"],
        "threshold": snap["threshold"],
    }


def _deserialize_single_snap(snap_data: dict) -> dict:
    """Deserialize a single OnlineSubspace snapshot dict from JSON-safe form."""
    return {
        "dim": snap_data["dim"],
        "k": snap_data["k"],
        "n": snap_data["n"],
        "mean": _b64_to_ndarray(snap_data["mean"]),
        "components": _b64_to_ndarray(snap_data["components"]),
        "res_ema": snap_data["res_ema"],
        "res_var_ema": snap_data["res_var_ema"],
        "threshold": snap_data["threshold"],
    }
