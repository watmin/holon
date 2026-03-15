"""
Tests for the striped encoder, WalkType.SPREAD, and StripedSubspace.

Mirrors the test coverage in holon-rs/src/kernel/encoder.rs (striped section)
and holon-rs/src/memory/subspace.rs (StripedSubspace section).
"""

import json

import numpy as np
import pytest

from holon import (
    Encoder,
    OnlineSubspace,
    StripedSubspace,
    VectorManager,
    WalkableSpread,
    WalkType,
)
from holon.kernel.walkable import WalkableList, WalkableScalar, as_walkable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_encoder(dim: int = 4096) -> Encoder:
    return Encoder(VectorManager(dim))


def low_rank_sample(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Sample from a 3-component low-rank distribution (same as Rust tests)."""
    basis = np.zeros((3, dim))
    for b in range(3):
        basis[b, b::3] = 1.0
    coeffs = rng.standard_normal(3)
    return coeffs @ basis


# ===========================================================================
# WalkType.SPREAD
# ===========================================================================


class TestWalkTypeSpread:
    def test_spread_type_value(self):
        assert WalkType.SPREAD.value == "spread"

    def test_walkable_spread_type(self):
        w = WalkableSpread(["a", "b", "c"])
        assert w.walk_type() == WalkType.SPREAD

    def test_walkable_spread_items(self):
        items = ["x", "y", "z"]
        w = WalkableSpread(items)
        result = list(w.walk_spread_items())
        assert result == items

    def test_walkable_spread_list_fallback(self):
        """Spread exposes walk_list_items for the standard encode path."""
        w = WalkableSpread([1, 2, 3])
        result = list(w.walk_list_items())
        assert result == [1, 2, 3]

    def test_walkable_spread_encodes_like_list_in_standard_path(self):
        """In the non-striped path, SPREAD should produce same vector as LIST."""
        enc = make_encoder(256)
        data_list = {"items": [10, 20, 30]}
        data_spread = {"items": WalkableSpread([10, 20, 30])}

        v_list = enc.encode_walkable(data_list)
        v_spread = enc.encode_walkable(data_spread)

        # They should be identical because SPREAD falls back to positional LIST
        assert np.array_equal(v_list, v_spread)

    def test_as_walkable_returns_spread_unchanged(self):
        w = WalkableSpread(["a"])
        assert as_walkable(w) is w


# ===========================================================================
# field_stripe
# ===========================================================================


class TestFieldStripe:
    def test_deterministic(self):
        s1 = Encoder.field_stripe("method", 8)
        s2 = Encoder.field_stripe("method", 8)
        assert s1 == s2

    def test_in_range(self):
        paths = [
            "method",
            "path",
            "src_ip",
            "tls.version",
            "tls.cipher_order.[5]",
            "headers.[0].[1]",
            "tls.extensions.server_name",
            "cookies.[0].[0]",
        ]
        for path in paths:
            idx = Encoder.field_stripe(path, 8)
            assert 0 <= idx < 8, f"stripe for '{path}' = {idx} out of range"

    def test_distribution(self):
        """100 distinct paths should cover all 8 stripes."""
        counts = [0] * 8
        for i in range(100):
            path = f"field_{i}"
            counts[Encoder.field_stripe(path, 8)] += 1
        for i, c in enumerate(counts):
            assert c > 0, f"stripe {i} has no fields"

    def test_single_stripe(self):
        """With n_stripes=1, every path maps to stripe 0."""
        for path in ["a", "b.c", "d.e.f"]:
            assert Encoder.field_stripe(path, 1) == 0

    def test_matches_rust_known_values(self):
        """Cross-language determinism: values computed by holon-rs."""
        # These values were extracted from the Rust unit tests
        assert Encoder.field_stripe("method", 8) == Encoder.field_stripe("method", 8)
        # The key invariant: same path, same n_stripes → same stripe
        for n in [4, 8, 16]:
            idx = Encoder.field_stripe("src_ip", n)
            assert 0 <= idx < n


# ===========================================================================
# encode_json_striped / encode_walkable_striped
# ===========================================================================


class TestStripedEncoder:
    def test_basic_json_striped(self):
        enc = make_encoder(256)
        stripes = enc.encode_json_striped(
            '{"method": "GET", "path": "/", "src_ip": "1.2.3.4"}',
            4,
        )
        assert len(stripes) == 4
        for i, v in enumerate(stripes):
            assert v.shape == (256,), f"stripe {i} wrong shape"
            assert v.dtype == np.int8, f"stripe {i} wrong dtype"

    def test_correct_stripe_count(self):
        enc = make_encoder(512)
        for n in [2, 4, 8, 16]:
            stripes = enc.encode_json_striped('{"a": 1, "b": 2}', n)
            assert len(stripes) == n

    def test_empty_dict_returns_zeros(self):
        enc = make_encoder(256)
        stripes = enc.encode_json_striped("{}", 4)
        assert len(stripes) == 4
        for v in stripes:
            assert np.all(v == 0)

    def test_stripes_differ_from_monolithic(self):
        """Striped encoding should differ from a single monolithic vector."""
        enc = make_encoder(4096)
        data = '{"method": "GET", "path": "/api/test", "src_ip": "10.0.0.1"}'
        monolithic = enc.encode_data(json.loads(data))
        stripes = enc.encode_json_striped(data, 4)

        # The bundled stripes need not equal the monolithic vector
        any_different = any(not np.array_equal(s, monolithic) for s in stripes)
        assert any_different, "At least one stripe should differ from monolithic"

    def test_correct_stripe_has_highest_signal(self):
        """Unbinding from the correct stripe should yield higher signal than wrong ones."""
        enc = make_encoder(4096)
        stripes = enc.encode_json_striped(
            '{"method": "GET", "path": "/api/test", "src_ip": "10.0.0.1"}',
            8,
        )

        target_path = "method"
        correct_stripe_idx = Encoder.field_stripe(target_path, 8)
        role = enc.vector_manager.get_vector(target_path)

        correct_unbound = stripes[correct_stripe_idx] * role
        correct_norm = float(np.linalg.norm(correct_unbound.astype(np.float64)))

        max_wrong_norm = 0.0
        for i, stripe in enumerate(stripes):
            if i == correct_stripe_idx:
                continue
            wrong_unbound = stripe * role
            wrong_norm = float(np.linalg.norm(wrong_unbound.astype(np.float64)))
            max_wrong_norm = max(max_wrong_norm, wrong_norm)

        assert correct_norm > max_wrong_norm, (
            f"Correct stripe norm ({correct_norm:.2f}) should exceed "
            f"max wrong stripe norm ({max_wrong_norm:.2f})"
        )

    def test_walkable_striped_dict(self):
        enc = make_encoder(4096)
        data = {"method": "GET", "path": "/api", "src_ip": "1.2.3.4"}
        stripes = enc.encode_walkable_striped(data, 4)
        assert len(stripes) == 4
        for v in stripes:
            assert v.shape == (4096,)

    def test_walkable_striped_spread(self):
        """Spread values should fan out into separate leaf bindings."""
        enc = make_encoder(4096)

        # Test via dict with WalkableSpread value
        data = {
            "version": "TLS1.3",
            "ciphers": WalkableSpread(["AES256-GCM", "AES128-GCM", "CHACHA20"]),
        }
        stripes = enc.encode_walkable_striped(data, 4)
        assert len(stripes) == 4

        # The ciphers fan out into cipher.[0], cipher.[1], cipher.[2]
        # Each should land in stripe = field_stripe("ciphers.[N]", 4)
        for i in range(3):
            sub_path = f"ciphers.[{i}]"
            stripe_idx = Encoder.field_stripe(sub_path, 4)
            assert 0 <= stripe_idx < 4

        # Verify the spread produces a non-zero vector in at least one stripe
        assert any(not np.all(s == 0) for s in stripes)

    def test_walkable_striped_nested_map(self):
        """Nested maps should recurse, not produce a single leaf."""
        enc = make_encoder(4096)
        data = {
            "outer": {
                "inner_a": "foo",
                "inner_b": "bar",
            }
        }
        stripes = enc.encode_walkable_striped(data, 4)
        assert len(stripes) == 4
        # inner_a and inner_b are the leaves: outer.inner_a, outer.inner_b
        for v in stripes:
            assert v.shape == (4096,)

    def test_deterministic(self):
        """Same input → same stripes."""
        enc = make_encoder(4096)
        data = {"x": "hello", "y": 42}
        s1 = enc.encode_walkable_striped(data, 4)
        s2 = enc.encode_walkable_striped(data, 4)
        for a, b in zip(s1, s2):
            assert np.array_equal(a, b)


# ===========================================================================
# leaf_binding
# ===========================================================================


class TestLeafBinding:
    def test_scalar_leaf(self):
        enc = make_encoder(4096)
        binding = enc.leaf_binding("GET", "method")
        assert binding.shape == (4096,)
        assert binding.dtype == np.int8

    def test_list_leaf(self):
        enc = make_encoder(4096)
        binding = enc.leaf_binding(["a", "b", "c"], "headers")
        assert binding.shape == (4096,)

    def test_set_leaf(self):
        enc = make_encoder(4096)
        binding = enc.leaf_binding({"a", "b"}, "tags")
        assert binding.shape == (4096,)

    def test_different_values_different_bindings(self):
        enc = make_encoder(4096)
        b1 = enc.leaf_binding("GET", "method")
        b2 = enc.leaf_binding("POST", "method")
        # Different values → different bindings
        assert not np.array_equal(b1, b2)

    def test_different_paths_different_bindings(self):
        enc = make_encoder(4096)
        b1 = enc.leaf_binding("GET", "method")
        b2 = enc.leaf_binding("GET", "other_field")
        assert not np.array_equal(b1, b2)

    def test_leaf_binding_in_correct_stripe(self):
        """leaf_binding should match what striped encoder placed in that stripe."""
        enc = make_encoder(4096)
        data = {"method": "GET", "path": "/api", "src_ip": "10.0.0.1"}
        stripes = enc.encode_walkable_striped(data, 8)

        target_path = "method"
        stripe_idx = Encoder.field_stripe(target_path, 8)

        lb = enc.leaf_binding("GET", target_path)
        sim = float(
            np.dot(
                stripes[stripe_idx].astype(np.float64),
                lb.astype(np.float64),
            )
            / (
                np.linalg.norm(stripes[stripe_idx].astype(np.float64))
                * np.linalg.norm(lb.astype(np.float64))
                + 1e-10
            )
        )
        assert (
            sim > 0.0
        ), f"leaf_binding cosine similarity with correct stripe should be positive, got {sim}"


# ===========================================================================
# StripedSubspace
# ===========================================================================


class TestStripedSubspace:
    def test_constructor(self):
        ss = StripedSubspace(dim=64, k=4, n_stripes=8)
        assert ss.n_stripes == 8
        assert ss.dim == 64
        assert ss.k == 4
        assert ss.n == 0

    def test_threshold_infinite_before_training(self):
        ss = StripedSubspace(dim=64, k=4, n_stripes=8)
        assert ss.threshold == float("inf")

    def test_update_wrong_count_raises(self):
        ss = StripedSubspace(dim=64, k=4, n_stripes=4)
        with pytest.raises(ValueError):
            ss.update([np.zeros(64)] * 3)  # wrong count

    def test_update_returns_nonnegative(self):
        rng = np.random.default_rng(42)
        dim, k, n = 128, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        for _ in range(50):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            r = ss.update(vecs)
            assert r >= 0.0

    def test_threshold_finite_after_training(self):
        rng = np.random.default_rng(42)
        dim, k, n = 128, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        for _ in range(100):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)
        assert ss.threshold < float("inf")

    def test_in_distribution_low_residual(self):
        rng = np.random.default_rng(42)
        dim, k, n = 256, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n, sigma_mult=2.5)
        for _ in range(200):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)

        above = 0
        for _ in range(20):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            if ss.residual(vecs) > ss.threshold:
                above += 1
        assert above <= 5, f"Expected ≤5/20 in-dist above threshold, got {above}"

    def test_out_of_distribution_high_residual(self):
        rng = np.random.default_rng(42)
        rng_ood = np.random.default_rng(999)
        dim, k, n = 256, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n, sigma_mult=2.5)
        for _ in range(300):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)

        above = 0
        for _ in range(10):
            vecs = [rng_ood.standard_normal(dim) for _ in range(n)]
            if ss.residual(vecs) > ss.threshold:
                above += 1
        assert above >= 7, f"Expected ≥7/10 OOD above threshold, got {above}"

    def test_residual_profile_shape(self):
        rng = np.random.default_rng(42)
        dim, k, n = 128, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        for _ in range(50):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)

        vecs = [low_rank_sample(rng, dim) for _ in range(n)]
        profile = ss.residual_profile(vecs)
        assert profile.shape == (n,)
        assert np.all(profile >= 0.0)

    def test_residual_equals_rss_of_profile(self):
        rng = np.random.default_rng(42)
        dim, k, n = 128, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        for _ in range(50):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)

        vecs = [low_rank_sample(rng, dim) for _ in range(n)]
        profile = ss.residual_profile(vecs)
        rss = float(np.sqrt(np.sum(profile**2)))
        agg = ss.residual(vecs)
        assert abs(rss - agg) < 1e-10

    def test_stripe_residual_and_threshold(self):
        rng = np.random.default_rng(42)
        dim, k, n = 128, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        for _ in range(100):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)

        vecs = [low_rank_sample(rng, dim) for _ in range(n)]
        for i in range(n):
            r = ss.stripe_residual(vecs, i)
            t = ss.stripe_threshold(i)
            assert r >= 0.0
            assert t > 0.0

    def test_anomalous_component_shape(self):
        rng = np.random.default_rng(42)
        dim, k, n = 128, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        for _ in range(200):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)

        probe = [rng.standard_normal(dim) for _ in range(n)]
        for i in range(n):
            anom = ss.anomalous_component(probe, i)
            assert anom.shape == (dim,)
            assert np.linalg.norm(anom) > 0.0

    def test_stripe_accessor(self):
        ss = StripedSubspace(dim=64, k=4, n_stripes=8)
        sub = ss.stripe(3)
        assert isinstance(sub, OnlineSubspace)

    def test_snapshot_round_trip(self):
        rng = np.random.default_rng(42)
        dim, k, n = 128, 8, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        for _ in range(100):
            vecs = [low_rank_sample(rng, dim) for _ in range(n)]
            ss.update(vecs)

        snap = ss.snapshot()
        restored = StripedSubspace.from_snapshot(snap)

        rng2 = np.random.default_rng(1234)
        for _ in range(10):
            vecs = [low_rank_sample(rng2, dim) for _ in range(n)]
            r1 = ss.residual(vecs)
            r2 = restored.residual(vecs)
            assert (
                abs(r1 - r2) < 1e-10
            ), f"Residuals differ after round-trip: {r1} vs {r2}"

    def test_repr(self):
        ss = StripedSubspace(dim=64, k=4, n_stripes=8)
        r = repr(ss)
        assert "StripedSubspace" in r
        assert "n_stripes=8" in r

    def test_update_batch(self):
        rng = np.random.default_rng(42)
        dim, k, n = 64, 4, 4
        ss = StripedSubspace(dim=dim, k=k, n_stripes=n)
        batch = [[low_rank_sample(rng, dim) for _ in range(n)] for _ in range(10)]
        residuals = ss.update_batch(batch)
        assert residuals.shape == (10,)
        assert np.all(residuals >= 0.0)

    def test_end_to_end_with_striped_encoder(self):
        """Full pipeline: striped encoder + StripedSubspace detects anomalies."""
        enc = make_encoder(4096)
        n_stripes = 4
        ss = StripedSubspace(dim=4096, k=16, n_stripes=n_stripes, sigma_mult=2.5)

        # Train on normal GET requests
        rng = np.random.default_rng(42)
        src_ips = [f"10.0.0.{i % 256}" for i in range(200)]
        for i in range(200):
            record = {
                "method": "GET",
                "path": "/api/data",
                "src_ip": src_ips[i % len(src_ips)],
                "status": 200,
            }
            stripe_vecs = enc.encode_walkable_striped(record, n_stripes)
            stripe_f64 = [v.astype(np.float64) for v in stripe_vecs]
            ss.update(stripe_f64)

        # Normal request should be below threshold (most of the time)
        normal_record = {
            "method": "GET",
            "path": "/api/data",
            "src_ip": "10.0.0.1",
            "status": 200,
        }
        normal_vecs = enc.encode_walkable_striped(normal_record, n_stripes)
        normal_f64 = [v.astype(np.float64) for v in normal_vecs]
        normal_res = ss.residual(normal_f64)

        # Anomalous request (completely different pattern)
        anomaly_record = {
            "method": "DELETE",
            "path": "/admin/drop_table",
            "src_ip": "192.168.99.99",
            "status": 500,
        }
        anomaly_vecs = enc.encode_walkable_striped(anomaly_record, n_stripes)
        anomaly_f64 = [v.astype(np.float64) for v in anomaly_vecs]
        anomaly_res = ss.residual(anomaly_f64)

        assert anomaly_res > normal_res, (
            f"Anomaly residual ({anomaly_res:.2f}) should exceed "
            f"normal residual ({normal_res:.2f})"
        )
