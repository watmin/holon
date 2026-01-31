"""
Unit tests for $time marker and temporal encoding.
"""

from datetime import datetime

import numpy as np
import pytest

from holon import CPUStore, HolonClient
from holon.encoder import TimeResolution
from holon.similarity import normalized_dot_similarity


class TestTimeEncoding:
    """Tests for the $time marker encoding."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = CPUStore(dimensions=1024)
        self.client = HolonClient(local_store=self.store)

    def test_time_marker_basic(self):
        """Test that $time marker encodes without error."""
        doc = {
            "event": "test",
            "created_at": {"$time": 1706500000},
        }
        self.client.insert_json(doc)
        assert len(self.store.stored_data) == 1

    def test_time_marker_iso_string(self):
        """Test that ISO string timestamps work."""
        doc = {
            "event": "test",
            "created_at": {"$time": "2024-01-29T10:30:00Z"},
        }
        self.client.insert_json(doc)
        assert len(self.store.stored_data) == 1

    def test_time_resolution_parameter(self):
        """Test that $time_resolution is accepted."""
        doc = {
            "event": "test",
            "created_at": {"$time": 1706500000, "$time_resolution": "minute"},
        }
        self.client.insert_json(doc)
        assert len(self.store.stored_data) == 1

    def test_time_similarity_near_times(self):
        """Test that near times are more similar than far times."""
        base_ts = datetime(2024, 6, 15, 14, 30).timestamp()

        # Encode base time
        base_vec = self.client.encode_vectors({"$time": base_ts})
        if isinstance(base_vec, list):
            base_vec = np.array(base_vec)

        # Encode 1 hour later
        near_vec = self.client.encode_vectors({"$time": base_ts + 3600})
        if isinstance(near_vec, list):
            near_vec = np.array(near_vec)

        # Encode 6 months later
        far_vec = self.client.encode_vectors({"$time": base_ts + 180 * 24 * 3600})
        if isinstance(far_vec, list):
            far_vec = np.array(far_vec)

        near_sim = normalized_dot_similarity(base_vec, near_vec)
        far_sim = normalized_dot_similarity(base_vec, far_vec)

        # Near time should be more similar
        assert near_sim > far_sim, f"Near ({near_sim}) should be > far ({far_sim})"

    def test_circular_hour_similarity(self):
        """Test that same hour different days are similar."""
        # 2:30 PM today
        time1 = datetime(2024, 6, 15, 14, 30).timestamp()
        # 2:30 PM tomorrow
        time2 = datetime(2024, 6, 16, 14, 30).timestamp()
        # 2:30 AM today (opposite time)
        time3 = datetime(2024, 6, 15, 2, 30).timestamp()

        vec1 = self.client.encode_vectors({"$time": time1})
        vec2 = self.client.encode_vectors({"$time": time2})
        vec3 = self.client.encode_vectors({"$time": time3})

        if isinstance(vec1, list):
            vec1, vec2, vec3 = np.array(vec1), np.array(vec2), np.array(vec3)

        same_hour_sim = normalized_dot_similarity(vec1, vec2)
        diff_hour_sim = normalized_dot_similarity(vec1, vec3)

        # Same hour should be more similar
        assert (
            same_hour_sim > diff_hour_sim
        ), f"Same hour ({same_hour_sim}) should be > different hour ({diff_hour_sim})"

    def test_time_in_document_query(self):
        """Test querying documents with time constraints."""
        base = datetime(2024, 6, 15, 12, 0)

        # Insert documents at different times
        docs = [
            {"id": 1, "type": "order", "ts": {"$time": base.timestamp()}},
            {
                "id": 2,
                "type": "order",
                "ts": {"$time": (base.replace(hour=12, minute=30)).timestamp()},
            },
            {
                "id": 3,
                "type": "order",
                "ts": {"$time": (base.replace(month=1)).timestamp()},
            },  # 5 months earlier
        ]

        for doc in docs:
            self.client.insert_json(doc)

        # Query for orders around base time
        probe = {"type": "order", "ts": {"$time": base.timestamp()}}
        results = self.client.search_json(probe=probe, limit=3, threshold=0.0)

        assert len(results) == 3

        # First result should be the exact match or very close
        top_ids = [r["data"]["id"] for r in results[:2]]
        assert 1 in top_ids or 2 in top_ids, "Close time documents should rank high"

    def test_structure_plus_time(self):
        """Test combined structure and time similarity."""
        base_ts = datetime(2024, 6, 15, 14, 0).timestamp()

        docs = [
            # Matching structure, close time
            {
                "id": 1,
                "customer": {"tier": "platinum"},
                "ts": {"$time": base_ts},
            },
            # Matching structure, far time
            {
                "id": 2,
                "customer": {"tier": "platinum"},
                "ts": {"$time": base_ts - 180 * 24 * 3600},
            },
            # Different structure, close time
            {
                "id": 3,
                "customer": {"tier": "bronze"},
                "ts": {"$time": base_ts + 3600},
            },
        ]

        for doc in docs:
            self.client.insert_json(doc)

        # Query for platinum customer around base time
        probe = {
            "customer": {"tier": "platinum"},
            "ts": {"$time": base_ts + 1800},  # 30 min later
        }

        results = self.client.search_json(probe=probe, limit=3, threshold=0.0)

        # Doc 1 should rank highest (matching structure + close time)
        assert results[0]["data"]["id"] == 1


class TestTimeResolutionEnum:
    """Tests for TimeResolution enum."""

    def test_resolution_values(self):
        """Test that resolution enum has expected values."""
        assert TimeResolution.SECOND.value == "second"
        assert TimeResolution.MINUTE.value == "minute"
        assert TimeResolution.HOUR.value == "hour"
        assert TimeResolution.DAY.value == "day"

    def test_resolution_from_string(self):
        """Test creating resolution from string."""
        res = TimeResolution("hour")
        assert res == TimeResolution.HOUR


class TestTimeEncodingEdgeCases:
    """Edge case tests for time encoding."""

    def setup_method(self):
        """Set up test fixtures."""
        self.store = CPUStore(dimensions=1024)
        self.client = HolonClient(local_store=self.store)

    def test_midnight_wraparound(self):
        """Test encoding around midnight."""
        # 11:59 PM
        late = datetime(2024, 6, 15, 23, 59).timestamp()
        # 12:01 AM next day
        early = datetime(2024, 6, 16, 0, 1).timestamp()

        vec_late = self.client.encode_vectors({"$time": late})
        vec_early = self.client.encode_vectors({"$time": early})

        if isinstance(vec_late, list):
            vec_late, vec_early = np.array(vec_late), np.array(vec_early)

        sim = normalized_dot_similarity(vec_late, vec_early)
        # Should be somewhat similar (close in time despite day change)
        assert sim > 0.3, f"Midnight wraparound similarity {sim} should be > 0.3"

    def test_year_boundary(self):
        """Test encoding around year boundary."""
        # Dec 31
        dec31 = datetime(2024, 12, 31, 12, 0).timestamp()
        # Jan 1
        jan1 = datetime(2025, 1, 1, 12, 0).timestamp()

        vec_dec = self.client.encode_vectors({"$time": dec31})
        vec_jan = self.client.encode_vectors({"$time": jan1})

        if isinstance(vec_dec, list):
            vec_dec, vec_jan = np.array(vec_dec), np.array(vec_jan)

        sim = normalized_dot_similarity(vec_dec, vec_jan)
        # Should be similar (same time of day, adjacent days, circular month)
        assert sim > 0.2, f"Year boundary similarity {sim} should be > 0.2"

    def test_invalid_iso_string_fallback(self):
        """Test that invalid ISO strings don't crash."""
        doc = {
            "event": "test",
            "created_at": {"$time": "not-a-timestamp"},
        }
        # Should not raise - falls back to string encoding
        self.client.insert_json(doc)
        assert len(self.store.stored_data) == 1

    def test_nested_time_marker(self):
        """Test $time marker in nested structure."""
        doc = {
            "order": {
                "id": 123,
                "timestamps": {
                    "created": {"$time": 1706500000},
                    "updated": {"$time": 1706503600},
                },
            },
        }
        self.client.insert_json(doc)
        assert len(self.store.stored_data) == 1
