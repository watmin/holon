"""
Tests for subspace_alignment and match_alignment.

Validates the directional measurement primitive that complements
match_spectrum's magnitude signal. Principal angles via SVD of
the basis inner-product matrix.
"""

import numpy as np
import pytest

from holon.memory.engram import EngramLibrary
from holon.memory.subspace import OnlineSubspace

DIM = 512
K = 16


def _make_subspace(dim=DIM, k=K, seed=0, n_samples=200):
    """Train a subspace on random data from a specific distribution."""
    rng = np.random.default_rng(seed)
    sub = OnlineSubspace(dim=dim, k=k, amnesia=2.0)
    for _ in range(n_samples):
        v = rng.choice([-1, 1], size=dim).astype(np.float64)
        sub.update(v)
    return sub


def _make_subspace_from_directions(directions, dim=DIM, k=K, n_samples=200, seed=0):
    """Train a subspace where variance concentrates along given directions."""
    rng = np.random.default_rng(seed)
    sub = OnlineSubspace(dim=dim, k=k, amnesia=2.0)
    for _ in range(n_samples):
        v = np.zeros(dim)
        for d in directions:
            v += rng.normal() * 5.0 * d
        v += rng.normal(size=dim) * 0.1
        sub.update(v)
    return sub


class TestSubspaceAlignment:
    """Tests for OnlineSubspace.subspace_alignment()."""

    def test_self_alignment_is_high(self):
        sub = _make_subspace(seed=42)
        score = sub.subspace_alignment(sub)
        assert score > 0.95, f"Self-alignment should be ~1.0, got {score}"

    def test_same_data_alignment_is_high(self):
        sub_a = _make_subspace(seed=42)
        sub_b = _make_subspace(seed=42)
        score = sub_a.subspace_alignment(sub_b)
        assert score > 0.95, f"Same-data alignment should be ~1.0, got {score}"

    def test_different_data_alignment_is_lower(self):
        sub_a = _make_subspace(seed=42)
        sub_b = _make_subspace(seed=999)
        same_score = sub_a.subspace_alignment(sub_a)
        diff_score = sub_a.subspace_alignment(sub_b)
        assert diff_score < same_score, (
            f"Different data should have lower alignment: "
            f"same={same_score:.4f}, diff={diff_score:.4f}"
        )

    def test_orthogonal_subspaces_low_alignment(self):
        rng = np.random.default_rng(0)
        all_dirs = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]

        dirs_a = all_dirs[:5]
        dirs_b = all_dirs[5:10]

        sub_a = _make_subspace_from_directions(dirs_a, seed=10)
        sub_b = _make_subspace_from_directions(dirs_b, seed=20)

        score = sub_a.subspace_alignment(sub_b)
        assert (
            score < 0.5
        ), f"Orthogonal directions should have low alignment, got {score}"

    def test_same_directions_high_alignment(self):
        rng = np.random.default_rng(0)
        dirs = rng.normal(size=(5, DIM))
        for i in range(5):
            dirs[i] /= np.linalg.norm(dirs[i])

        sub_a = _make_subspace_from_directions(dirs, seed=10, n_samples=300)
        sub_b = _make_subspace_from_directions(dirs, seed=20, n_samples=300)

        score = sub_a.subspace_alignment(sub_b)
        assert score > 0.8, f"Same directions should have high alignment, got {score}"

    def test_symmetry(self):
        sub_a = _make_subspace(seed=1)
        sub_b = _make_subspace(seed=2)
        ab = sub_a.subspace_alignment(sub_b)
        ba = sub_b.subspace_alignment(sub_a)
        assert abs(ab - ba) < 0.05, (
            f"Alignment should be approximately symmetric: "
            f"a→b={ab:.4f}, b→a={ba:.4f}"
        )

    def test_returns_float_in_unit_interval(self):
        sub_a = _make_subspace(seed=1)
        sub_b = _make_subspace(seed=2)
        score = sub_a.subspace_alignment(sub_b)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score must be in [0,1], got {score}"

    def test_empty_subspace_returns_zero(self):
        sub_a = OnlineSubspace(dim=DIM, k=K)
        sub_b = _make_subspace(seed=1)
        assert sub_a.subspace_alignment(sub_b) == 0.0
        assert sub_b.subspace_alignment(sub_a) == 0.0

    def test_top_angles_parameter(self):
        sub_a = _make_subspace(seed=1)
        sub_b = _make_subspace(seed=2)

        score_all = sub_a.subspace_alignment(sub_b, top_angles=K)
        score_top3 = sub_a.subspace_alignment(sub_b, top_angles=3)
        assert score_top3 >= score_all, (
            f"Fewer top angles should give >= mean: "
            f"top3={score_top3:.4f}, all={score_all:.4f}"
        )

    def test_different_k_values(self):
        sub_a = _make_subspace(dim=DIM, k=8, seed=1)
        sub_b = _make_subspace(dim=DIM, k=16, seed=1)
        score = sub_a.subspace_alignment(sub_b)
        assert 0.0 <= score <= 1.0


class TestMatchAlignment:
    """Tests for EngramLibrary.match_alignment()."""

    def _build_library(self):
        lib = EngramLibrary(dim=DIM)
        rng = np.random.default_rng(0)
        all_dirs = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]

        for i, name in enumerate(["alpha", "beta", "gamma"]):
            dirs = all_dirs[i * 5 : (i + 1) * 5]
            sub = _make_subspace_from_directions(dirs, seed=i * 100, n_samples=300)
            lib.add(name, sub)
        return lib, all_dirs

    def test_best_match_is_correct(self):
        lib, all_dirs = self._build_library()
        dirs_alpha = all_dirs[:5]
        probe = _make_subspace_from_directions(dirs_alpha, seed=999, n_samples=300)

        matches = lib.match_alignment(probe, top_k=3)
        assert (
            matches[0][0] == "alpha"
        ), f"Expected 'alpha' as best match, got '{matches[0][0]}'"

    def test_returns_sorted_descending(self):
        lib, all_dirs = self._build_library()
        probe = _make_subspace_from_directions(all_dirs[:5], seed=999, n_samples=300)
        matches = lib.match_alignment(probe, top_k=3)
        scores = [s for _, s in matches]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self):
        lib, all_dirs = self._build_library()
        probe = _make_subspace_from_directions(all_dirs[:5], seed=999, n_samples=300)
        matches = lib.match_alignment(probe, top_k=1)
        assert len(matches) == 1

    def test_empty_library_returns_empty(self):
        lib = EngramLibrary(dim=DIM)
        probe = _make_subspace(seed=0)
        assert lib.match_alignment(probe) == []

    def test_unknown_probe_scores_lower_than_known(self):
        lib, all_dirs = self._build_library()
        dirs_alpha = all_dirs[:5]
        dirs_unknown = all_dirs[15:20]

        known_probe = _make_subspace_from_directions(
            dirs_alpha, seed=888, n_samples=300
        )
        unknown_probe = _make_subspace_from_directions(
            dirs_unknown, seed=777, n_samples=300
        )

        known_best = lib.match_alignment(known_probe, top_k=1)[0][1]
        unknown_best = lib.match_alignment(unknown_probe, top_k=1)[0][1]

        assert known_best > unknown_best, (
            f"Known should score higher: known={known_best:.4f}, "
            f"unknown={unknown_best:.4f}"
        )
