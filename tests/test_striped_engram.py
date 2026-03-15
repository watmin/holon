"""Tests for StripedSubspace integration with EngramLibrary.

Covers:
- Engram kind="striped" construction and residual dispatch
- EngramLibrary.add_striped() minting
- EngramLibrary.match_striped() two-tier matching
- EngramLibrary.names(kind=...) filtering
- Serialization round-trip (to_dict / from_dict, save / load)
- Mixed library (single + striped engrams coexist)
- Error paths: wrong residual method called on wrong kind
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from holon.memory import EngramLibrary, OnlineSubspace, StripedSubspace
from holon.memory.engram import Engram

DIM = 64
K = 4
N_STRIPES = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vecs(n_stripes=N_STRIPES, dim=DIM, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return [
        rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=dim).astype(np.float64)
        for _ in range(n_stripes)
    ]


def _trained_striped(n=30, n_stripes=N_STRIPES, dim=DIM, k=K, seed=0):
    ss = StripedSubspace(dim=dim, k=k, n_stripes=n_stripes)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        ss.update(_random_vecs(n_stripes=n_stripes, dim=dim, rng=rng))
    return ss


def _trained_single(n=30, dim=DIM, k=K, seed=0):
    sub = OnlineSubspace(dim=dim, k=k)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        v = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=dim).astype(np.float64)
        sub.update(v)
    return sub


# ---------------------------------------------------------------------------
# Engram kind dispatch
# ---------------------------------------------------------------------------


class TestEngramKind:
    def test_default_kind_is_single(self):
        sub = _trained_single()
        snap = sub.snapshot()
        eig = sub.eigenvalues / (np.linalg.norm(sub.eigenvalues) + 1e-10)
        eng = Engram("x", snap, eig)
        assert eng.kind == "single"

    def test_invalid_kind_raises(self):
        sub = _trained_single()
        snap = sub.snapshot()
        eig = np.ones(K)
        with pytest.raises(ValueError, match="kind must be"):
            Engram("x", snap, eig, kind="bogus")

    def test_single_residual_returns_float(self):
        sub = _trained_single()
        lib = EngramLibrary(dim=DIM)
        lib.add("s", sub)
        eng = lib.get("s")
        vec = np.ones(DIM, dtype=np.float64)
        assert isinstance(eng.residual(vec), float)

    def test_striped_residual_returns_float(self):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("st", ss)
        eng = lib.get("st")
        vecs = _random_vecs()
        assert isinstance(eng.residual_striped(vecs), float)

    def test_residual_on_striped_raises(self):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("st", ss)
        eng = lib.get("st")
        with pytest.raises(TypeError, match="striped"):
            eng.residual(np.ones(DIM))

    def test_residual_striped_on_single_raises(self):
        sub = _trained_single()
        lib = EngramLibrary(dim=DIM)
        lib.add("s", sub)
        eng = lib.get("s")
        with pytest.raises(TypeError, match="single-vector"):
            eng.residual_striped(_random_vecs())

    def test_striped_engram_repr(self):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("mystripe", ss)
        eng = lib.get("mystripe")
        r = repr(eng)
        assert "striped" in r
        assert "mystripe" in r
        assert f"n_stripes={N_STRIPES}" in r


# ---------------------------------------------------------------------------
# EngramLibrary.add_striped
# ---------------------------------------------------------------------------


class TestAddStriped:
    def test_add_striped_stores_engram(self):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        eng = lib.add_striped("a", ss, action="BUY", score=0.5)
        assert eng.kind == "striped"
        assert lib.get("a") is eng

    def test_add_striped_metadata(self):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("a", ss, action="SELL", confidence=0.9)
        eng = lib.get("a")
        assert eng.metadata["action"] == "SELL"
        assert eng.metadata["confidence"] == 0.9

    def test_add_striped_surprise_profile(self):
        ss = _trained_striped()
        profile = {"t0.ohlcv.open": 0.8, "t0.rsi": 0.3}
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("a", ss, surprise_profile=profile)
        eng = lib.get("a")
        assert eng.surprise_profile == profile

    def test_eigenvalue_signature_length(self):
        ss = _trained_striped(n_stripes=4, k=4)
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("a", ss)
        eng = lib.get("a")
        # Concatenated per-stripe signatures: n_stripes * k
        assert len(eng.eigenvalue_signature) == N_STRIPES * K

    def test_add_striped_len(self):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("a", ss)
        lib.add_striped("b", ss)
        assert len(lib) == 2


# ---------------------------------------------------------------------------
# EngramLibrary.match_striped
# ---------------------------------------------------------------------------


class TestMatchStriped:
    def test_empty_library_returns_empty(self):
        lib = EngramLibrary(dim=DIM)
        assert lib.match_striped(_random_vecs()) == []

    def test_match_returns_tuples(self):
        ss = _trained_striped(seed=0)
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("a", ss)
        results = lib.match_striped(_random_vecs(rng=np.random.default_rng(1)))
        assert len(results) == 1
        name, score = results[0]
        assert isinstance(name, str)
        assert isinstance(score, float)

    def test_match_sorted_ascending(self):
        rng = np.random.default_rng(42)
        lib = EngramLibrary(dim=DIM)
        for i in range(5):
            ss = _trained_striped(seed=i)
            lib.add_striped(f"eng{i}", ss)
        results = lib.match_striped(_random_vecs(rng=rng), top_k=3)
        scores = [r[1] for r in results]
        assert scores == sorted(scores)

    def test_match_top_k_respected(self):
        lib = EngramLibrary(dim=DIM)
        for i in range(6):
            lib.add_striped(f"e{i}", _trained_striped(seed=i))
        results = lib.match_striped(_random_vecs(), top_k=2)
        assert len(results) <= 2

    def test_trained_pattern_scores_lower_than_noise(self):
        """A striped subspace should score its own training distribution lower
        than a completely different random distribution."""
        rng_train = np.random.default_rng(0)
        ss = StripedSubspace(dim=DIM, k=K, n_stripes=N_STRIPES)
        # Train on all-positive vectors
        for _ in range(40):
            vecs = [
                np.ones(DIM, dtype=np.float64) + rng_train.normal(0, 0.1, DIM)
                for _ in range(N_STRIPES)
            ]
            ss.update(vecs)

        lib = EngramLibrary(dim=DIM)
        lib.add_striped("familiar", ss)

        # Familiar: similar to training
        familiar_vecs = [np.ones(DIM, dtype=np.float64) for _ in range(N_STRIPES)]
        # Novel: all-negative, very different
        novel_vecs = [-np.ones(DIM, dtype=np.float64) for _ in range(N_STRIPES)]

        familiar_score = lib.match_striped(familiar_vecs)[0][1]
        novel_score = lib.match_striped(novel_vecs)[0][1]
        assert familiar_score < novel_score


# ---------------------------------------------------------------------------
# EngramLibrary.names(kind=...)
# ---------------------------------------------------------------------------


class TestNamesFiltering:
    def test_names_all(self):
        sub = _trained_single()
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add("single_one", sub)
        lib.add_striped("striped_one", ss)
        assert set(lib.names()) == {"single_one", "striped_one"}

    def test_names_single(self):
        sub = _trained_single()
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add("single_one", sub)
        lib.add_striped("striped_one", ss)
        assert lib.names(kind="single") == ["single_one"]

    def test_names_striped(self):
        sub = _trained_single()
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add("single_one", sub)
        lib.add_striped("striped_one", ss)
        assert lib.names(kind="striped") == ["striped_one"]

    def test_names_no_filter_none(self):
        lib = EngramLibrary(dim=DIM)
        lib.add("s", _trained_single())
        assert lib.names(kind=None) == lib.names()


# ---------------------------------------------------------------------------
# Mixed library — single and striped coexist
# ---------------------------------------------------------------------------


class TestMixedLibrary:
    def test_match_only_hits_single(self):
        sub = _trained_single()
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add("single", sub)
        lib.add_striped("striped", ss)

        # match() should only return single engrams
        results = lib.match(np.ones(DIM))
        names = [r[0] for r in results]
        assert "striped" not in names
        assert "single" in names

    def test_match_striped_only_hits_striped(self):
        sub = _trained_single()
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add("single", sub)
        lib.add_striped("striped", ss)

        # match_striped() should only return striped engrams
        results = lib.match_striped(_random_vecs())
        names = [r[0] for r in results]
        assert "single" not in names
        assert "striped" in names


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_engram_to_dict_from_dict_single(self):
        sub = _trained_single()
        lib = EngramLibrary(dim=DIM)
        lib.add("orig", sub, action="HOLD")
        orig = lib.get("orig")

        d = orig.to_dict()
        assert d["kind"] == "single"
        restored = Engram.from_dict(d)
        assert restored.kind == "single"
        assert restored.name == "orig"
        assert restored.metadata["action"] == "HOLD"

        vec = np.ones(DIM, dtype=np.float64)
        assert abs(orig.residual(vec) - restored.residual(vec)) < 1e-10

    def test_engram_to_dict_from_dict_striped(self):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("orig", ss, action="BUY")
        orig = lib.get("orig")

        d = orig.to_dict()
        assert d["kind"] == "striped"
        assert "stripes" in d["snapshot"]
        assert len(d["snapshot"]["stripes"]) == N_STRIPES

        restored = Engram.from_dict(d)
        assert restored.kind == "striped"
        assert restored.name == "orig"
        assert restored.metadata["action"] == "BUY"

        vecs = _random_vecs()
        assert (
            abs(orig.residual_striped(vecs) - restored.residual_striped(vecs)) < 1e-10
        )

    def test_library_save_load_mixed(self, tmp_path):
        sub = _trained_single()
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add("single_eng", sub, label="A")
        lib.add_striped("striped_eng", ss, label="B", action="SELL")

        path = str(tmp_path / "library.json")
        lib.save(path)

        lib2 = EngramLibrary.load(path)
        assert set(lib2.names()) == {"single_eng", "striped_eng"}
        assert lib2.get("single_eng").kind == "single"
        assert lib2.get("striped_eng").kind == "striped"
        assert lib2.get("striped_eng").metadata["action"] == "SELL"

    def test_library_save_load_residuals_preserved(self, tmp_path):
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("s", ss)
        path = str(tmp_path / "lib.json")
        lib.save(path)

        lib2 = EngramLibrary.load(path)
        vecs = _random_vecs()
        r1 = lib.get("s").residual_striped(vecs)
        r2 = lib2.get("s").residual_striped(vecs)
        assert abs(r1 - r2) < 1e-10

    def test_json_is_human_readable(self, tmp_path):
        """Saved file should be valid JSON with readable structure."""
        ss = _trained_striped()
        lib = EngramLibrary(dim=DIM)
        lib.add_striped("test", ss)
        path = str(tmp_path / "lib.json")
        lib.save(path)

        raw = json.loads(Path(path).read_text())
        assert "engrams" in raw
        eng_data = raw["engrams"][0]
        assert eng_data["kind"] == "striped"
        assert "stripes" in eng_data["snapshot"]

    def test_old_format_loads_as_single(self, tmp_path):
        """Libraries saved before kind was introduced should load as single."""
        sub = _trained_single()
        lib = EngramLibrary(dim=DIM)
        lib.add("legacy", sub)
        path = str(tmp_path / "legacy.json")
        lib.save(path)

        # Manually strip the kind field to simulate old format
        data = json.loads(Path(path).read_text())
        del data["engrams"][0]["kind"]
        Path(path).write_text(json.dumps(data))

        lib2 = EngramLibrary.load(path)
        assert lib2.get("legacy").kind == "single"
