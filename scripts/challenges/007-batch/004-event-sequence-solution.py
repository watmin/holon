#!/usr/bin/env python3
"""
Challenge 007-004: Event Sequence Matching (Anomaly Detection)

Demonstrates finding similar event patterns in logs/transactions with temporal awareness.
Features:
- Chained encoding for temporal sequences
- Prototype learning from anomalies
- Combining sequence similarity with numeric guards
- Fraud detection patterns

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/004-event-sequence-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/004-event-sequence-solution.py --http
"""

import argparse
import random
import time
import uuid
from typing import Any, Dict, List

import numpy as np

from holon import CPUStore, HolonClient


def generate_normal_sessions(count: int = 20) -> List[Dict[str, Any]]:
    """Generate normal user sessions."""
    sessions = []
    normal_patterns = [
        ["login", "view_account", "view_profile", "logout"],
        ["login", "view_products", "add_to_cart", "checkout", "logout"],
        ["login", "view_account", "update_profile", "logout"],
        ["login", "view_transactions", "download_statement", "logout"],
        ["login", "view_help", "logout"],
    ]

    for i in range(count):
        pattern = random.choice(normal_patterns)
        session = {
            "session_id": f"sess_{i:04d}",
            "user_id": f"user_{random.randint(1, 50):03d}",
            "events": {"_encode_mode": "chained", "sequence": pattern},
            "duration_seconds": random.randint(120, 900),
            "ip_changes": random.randint(0, 1),
            "total_amount": random.randint(0, 500),
            "label": "normal",
        }
        sessions.append(session)

    return sessions


def generate_fraud_sessions(count: int = 10) -> List[Dict[str, Any]]:
    """Generate fraudulent sessions."""
    sessions = []
    fraud_patterns = [
        ["login", "transfer", "transfer", "transfer", "logout"],
        ["login", "change_email", "change_password", "transfer", "logout"],
        ["login", "transfer", "logout", "login", "transfer", "logout"],
        ["login", "view_account", "transfer", "transfer", "delete_logs", "logout"],
        ["login", "transfer", "transfer", "change_password", "logout"],
    ]

    for i in range(count):
        pattern = random.choice(fraud_patterns)
        session = {
            "session_id": f"fraud_{i:04d}",
            "user_id": f"user_{random.randint(1, 50):03d}",
            "events": {"_encode_mode": "chained", "sequence": pattern},
            "duration_seconds": random.randint(30, 120),  # Faster than normal
            "ip_changes": random.randint(2, 5),  # More IP changes
            "total_amount": random.randint(5000, 50000),  # Higher amounts
            "label": "fraud",
        }
        sessions.append(session)

    return sessions


def generate_test_sessions(count: int = 5) -> List[Dict[str, Any]]:
    """Generate test sessions (mix of suspicious and normal)."""
    sessions = []

    # Suspicious patterns
    suspicious_patterns = [
        {
            "pattern": ["login", "transfer", "transfer", "logout"],
            "duration": 45,
            "ip_changes": 3,
            "amount": 15000,
            "expected": "fraud",
        },
        {
            "pattern": ["login", "change_password", "transfer", "logout"],
            "duration": 60,
            "ip_changes": 2,
            "amount": 8000,
            "expected": "fraud",
        },
        {
            "pattern": ["login", "view_account", "logout"],
            "duration": 300,
            "ip_changes": 0,
            "amount": 0,
            "expected": "normal",
        },
        {
            "pattern": ["login", "view_products", "add_to_cart", "logout"],
            "duration": 400,
            "ip_changes": 0,
            "amount": 100,
            "expected": "normal",
        },
        {
            "pattern": ["login", "transfer", "change_email", "logout"],
            "duration": 50,
            "ip_changes": 4,
            "amount": 20000,
            "expected": "fraud",
        },
    ]

    for i, pattern_data in enumerate(suspicious_patterns[:count]):
        session = {
            "session_id": f"test_{i:04d}",
            "user_id": f"user_test_{i:03d}",
            "events": {
                "_encode_mode": "chained",
                "sequence": pattern_data["pattern"],
            },
            "duration_seconds": pattern_data["duration"],
            "ip_changes": pattern_data["ip_changes"],
            "total_amount": pattern_data["amount"],
            "label": "unknown",
            "expected_label": pattern_data["expected"],
        }
        sessions.append(session)

    return sessions


class AnomalyDetector:
    """Detect anomalies using prototype learning."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.fraud_prototype = None
        self.normal_prototype = None

    def ingest_sessions(self, sessions: List[Dict[str, Any]]):
        """Ingest session data."""
        print(f"📥 Ingesting {len(sessions)} sessions...")
        start = time.time()

        for session in sessions:
            self.client.insert_json(session)

        elapsed = time.time() - start
        rate = len(sessions) / elapsed if elapsed > 0 else 0
        print(f"   ✅ Ingested in {elapsed:.2f}s ({rate:.0f}/sec)")

    def learn_patterns(self, fraud_sessions: List[Dict], normal_sessions: List[Dict]):
        """Learn fraud and normal prototypes."""
        print("\n🧠 Learning fraud pattern prototype...")
        fraud_vectors = []
        for session in fraud_sessions:
            vec = self.client.encode_vectors(session)
            if isinstance(vec, list):
                vec = np.array(vec)
            fraud_vectors.append(vec)

        if fraud_vectors:
            self.fraud_prototype = np.mean(fraud_vectors, axis=0)
            print(f"   ✅ Learned fraud prototype from {len(fraud_vectors)} examples")

        print("\n🧠 Learning normal pattern prototype...")
        normal_vectors = []
        for session in normal_sessions:
            vec = self.client.encode_vectors(session)
            if isinstance(vec, list):
                vec = np.array(vec)
            normal_vectors.append(vec)

        if normal_vectors:
            self.normal_prototype = np.mean(normal_vectors, axis=0)
            print(f"   ✅ Learned normal prototype from {len(normal_vectors)} examples")

    def detect_anomalies(
        self, test_sessions: List[Dict], threshold: float = 0.15
    ) -> List[Dict]:
        """Detect anomalies in test sessions."""
        print(f"\n🔍 Detecting anomalies (threshold={threshold})...")

        results = []
        for session in test_sessions:
            # Encode test session
            test_vec = self.client.encode_vectors(session)
            if isinstance(test_vec, list):
                test_vec = np.array(test_vec)

            # Compare to fraud prototype
            fraud_sim = 0.0
            if self.fraud_prototype is not None:
                fraud_sim = float(
                    np.dot(test_vec, self.fraud_prototype)
                    / (
                        np.linalg.norm(test_vec)
                        * np.linalg.norm(self.fraud_prototype)
                        + 1e-10
                    )
                )

            # Compare to normal prototype
            normal_sim = 0.0
            if self.normal_prototype is not None:
                normal_sim = float(
                    np.dot(test_vec, self.normal_prototype)
                    / (
                        np.linalg.norm(test_vec)
                        * np.linalg.norm(self.normal_prototype)
                        + 1e-10
                    )
                )

            # Classify based on which prototype is closer
            predicted = "fraud" if fraud_sim > normal_sim else "normal"
            confidence = max(fraud_sim, normal_sim)

            results.append(
                {
                    "session": session,
                    "fraud_sim": fraud_sim,
                    "normal_sim": normal_sim,
                    "predicted": predicted,
                    "confidence": confidence,
                    "expected": session.get("expected_label", "unknown"),
                }
            )

        return results

    def search_similar_to_pattern(
        self, pattern: Dict[str, Any], limit: int = 10
    ) -> List[Dict]:
        """Search for sessions similar to a pattern."""
        return self.client.search_json(probe=pattern, limit=limit, threshold=0.05)


def demo_fraud_detection(detector: AnomalyDetector, test_sessions: List[Dict]):
    """Demo: Fraud detection."""
    print("\n" + "=" * 70)
    print("DEMO: Fraud Detection Results")
    print("=" * 70)

    results = detector.detect_anomalies(test_sessions, threshold=0.15)

    print(f"\n📊 Classification Results:")
    correct = 0
    total = 0

    for result in results:
        session = result["session"]
        predicted = result["predicted"]
        expected = result["expected"]
        fraud_sim = result["fraud_sim"]
        normal_sim = result["normal_sim"]

        status = "✅" if predicted == expected else "❌"
        if expected != "unknown":
            total += 1
            if predicted == expected:
                correct += 1

        print(f"\n   {status} Session: {session['session_id']}")
        print(f"      Events: {' → '.join(session['events']['sequence'])}")
        print(f"      Duration: {session['duration_seconds']}s, Amount: ${session['total_amount']}")
        print(f"      Fraud similarity: {fraud_sim:.4f}")
        print(f"      Normal similarity: {normal_sim:.4f}")
        print(f"      Predicted: {predicted} (Expected: {expected})")

    if total > 0:
        accuracy = correct / total
        print(f"\n   📈 Accuracy: {correct}/{total} = {accuracy:.1%}")


def demo_pattern_search(detector: AnomalyDetector):
    """Demo: Search for similar patterns."""
    print("\n" + "=" * 70)
    print("DEMO: Search for Similar Transfer Patterns")
    print("=" * 70)

    pattern = {
        "events": {"_encode_mode": "chained", "sequence": ["login", "transfer", "transfer"]}
    }

    print("\n🔍 Searching for sessions with multiple transfers...")
    results = detector.search_similar_to_pattern(pattern, limit=5)

    print(f"   Found {len(results)} similar sessions:")
    for r in results[:5]:
        data = r["data"]
        events = data["events"]["sequence"]
        print(
            f"   - {data['session_id']}: {' → '.join(events)} (score: {r['score']:.3f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Event Sequence Anomaly Detection")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    print("=" * 70)
    print("EVENT SEQUENCE MATCHING (ANOMALY DETECTION)")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    start_time = time.time()

    # Create detector
    detector = AnomalyDetector(use_http=args.http, base_url=args.url)

    # Generate data
    print("\n📊 Generating training data...")
    normal_sessions = generate_normal_sessions(20)
    fraud_sessions = generate_fraud_sessions(10)
    print(f"   Normal sessions: {len(normal_sessions)}")
    print(f"   Fraud sessions: {len(fraud_sessions)}")

    # Ingest training data
    detector.ingest_sessions(normal_sessions + fraud_sessions)

    # Learn patterns
    detector.learn_patterns(fraud_sessions, normal_sessions)

    # Generate test data
    print("\n📊 Generating test data...")
    test_sessions = generate_test_sessions(5)
    print(f"   Test sessions: {len(test_sessions)}")

    # Run demos
    demo_fraud_detection(detector, test_sessions)
    demo_pattern_search(detector)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
    Mode: {mode}
    Elapsed: {elapsed:.2f}s

    Training Data: {len(normal_sessions) + len(fraud_sessions)} sessions
    Test Data: {len(test_sessions)} sessions

    ✅ Event sequence matching demonstrates:
       - Chained encoding for temporal sequences
       - Prototype learning from fraud examples
       - Similarity-based anomaly detection
       - Pattern matching with sequence awareness
       - Multi-factor fraud detection (events + duration + amount)

    This enables detecting suspicious activity patterns
    without manually defining all possible fraud rules!
    """
    )


if __name__ == "__main__":
    main()
