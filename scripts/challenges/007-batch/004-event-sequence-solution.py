#!/usr/bin/env python3
"""
Challenge 007-004: Event Sequence Matching (Anomaly Detection)

Demonstrates finding similar event patterns in logs/transactions with temporal awareness.
Features:
- Chained encoding for temporal sequences
- k-NN style classification via Holon search (HTTP-compatible)
- No local vector operations - all similarity computed server-side
- Fraud detection patterns

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/004-event-sequence-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/004-event-sequence-solution.py --http
"""

import argparse
import json
import random
import time
from typing import Any, Dict, List

from holon import CPUStore, HolonClient


def parse_data(data):
    """Parse data field - may be string in HTTP mode."""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return data


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
            "session_id": f"normal_{i:04d}",
            "user_id": f"user_{random.randint(1, 50):03d}",
            "events": {"$mode": "chained", "sequence": pattern},
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
            "events": {"$mode": "chained", "sequence": pattern},
            "duration_seconds": random.randint(30, 120),  # Faster than normal
            "ip_changes": random.randint(2, 5),  # More IP changes
            "total_amount": random.randint(5000, 50000),  # Higher amounts
            "label": "fraud",
        }
        sessions.append(session)

    return sessions


def generate_test_sessions() -> List[Dict[str, Any]]:
    """Generate test sessions (mix of suspicious and normal)."""
    test_cases = [
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

    sessions = []
    for i, tc in enumerate(test_cases):
        session = {
            "session_id": f"test_{i:04d}",
            "user_id": f"user_test_{i:03d}",
            "events": {"$mode": "chained", "sequence": tc["pattern"]},
            "duration_seconds": tc["duration"],
            "ip_changes": tc["ip_changes"],
            "total_amount": tc["amount"],
            "label": "unknown",
            "expected_label": tc["expected"],
        }
        sessions.append(session)

    return sessions


class AnomalyDetector:
    """
    Detect anomalies using k-NN classification via Holon search.

    This implementation is HTTP-compatible - all similarity computation
    happens server-side via search_json(). No local vector operations.
    """

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.use_http = use_http

    def ingest_training_data(
        self, fraud_sessions: List[Dict], normal_sessions: List[Dict]
    ):
        """Ingest labeled training data into Holon."""
        print(f"📥 Ingesting training data...")
        print(f"   Fraud sessions: {len(fraud_sessions)}")
        print(f"   Normal sessions: {len(normal_sessions)}")
        start = time.time()

        # Insert all training sessions (they already have 'label' field)
        for session in fraud_sessions + normal_sessions:
            self.client.insert_json(session)

        elapsed = time.time() - start
        total = len(fraud_sessions) + len(normal_sessions)
        rate = total / elapsed if elapsed > 0 else 0
        print(f"   ✅ Ingested {total} sessions in {elapsed:.2f}s ({rate:.0f}/sec)")

    def classify_session(self, session: Dict, k: int = 5) -> Dict:
        """
        Classify a session using k-NN via Holon search.

        All similarity computation happens server-side.
        We just count labels of the k nearest neighbors.
        """
        # Create probe from the session (without test-specific fields)
        probe = {
            "events": session["events"],
            "duration_seconds": session["duration_seconds"],
            "ip_changes": session["ip_changes"],
            "total_amount": session["total_amount"],
        }

        # Search for similar sessions in training data
        results = self.client.search_json(probe=probe, limit=k, threshold=0.0)

        # Count labels from nearest neighbors
        fraud_count = 0
        normal_count = 0
        fraud_scores = []
        normal_scores = []

        for r in results:
            data = parse_data(r["data"])
            label = data.get("label", "unknown") if isinstance(data, dict) else "unknown"
            score = r["score"]

            if label == "fraud":
                fraud_count += 1
                fraud_scores.append(score)
            elif label == "normal":
                normal_count += 1
                normal_scores.append(score)

        # Classify based on majority vote, weighted by similarity
        avg_fraud_sim = sum(fraud_scores) / len(fraud_scores) if fraud_scores else 0.0
        avg_normal_sim = (
            sum(normal_scores) / len(normal_scores) if normal_scores else 0.0
        )

        # Use weighted voting: count * average_similarity
        fraud_weight = fraud_count * (1 + avg_fraud_sim)
        normal_weight = normal_count * (1 + avg_normal_sim)

        predicted = "fraud" if fraud_weight > normal_weight else "normal"

        return {
            "session": session,
            "predicted": predicted,
            "expected": session.get("expected_label", "unknown"),
            "k_neighbors": k,
            "fraud_neighbors": fraud_count,
            "normal_neighbors": normal_count,
            "avg_fraud_sim": avg_fraud_sim,
            "avg_normal_sim": avg_normal_sim,
            "top_match": results[0] if results else None,
        }

    def detect_anomalies(self, test_sessions: List[Dict], k: int = 5) -> List[Dict]:
        """Classify all test sessions."""
        print(f"\n🔍 Classifying {len(test_sessions)} sessions (k={k} neighbors)...")

        results = []
        for session in test_sessions:
            result = self.classify_session(session, k=k)
            results.append(result)

        return results

    def search_similar_to_pattern(
        self, pattern: Dict[str, Any], limit: int = 10
    ) -> List[Dict]:
        """Search for sessions similar to a pattern."""
        return self.client.search_json(probe=pattern, limit=limit, threshold=0.0)


def demo_fraud_detection(detector: AnomalyDetector, test_sessions: List[Dict]):
    """Demo: Fraud detection with k-NN classification."""
    print("\n" + "=" * 70)
    print("DEMO: Fraud Detection via k-NN (HTTP-Compatible)")
    print("=" * 70)

    results = detector.detect_anomalies(test_sessions, k=5)

    print(f"\n📊 Classification Results:")
    correct = 0
    total = 0

    for result in results:
        session = result["session"]
        predicted = result["predicted"]
        expected = result["expected"]

        status = "✅" if predicted == expected else "❌"
        if expected != "unknown":
            total += 1
            if predicted == expected:
                correct += 1

        print(f"\n   {status} Session: {session['session_id']}")
        print(f"      Events: {' → '.join(session['events']['sequence'])}")
        print(f"      Duration: {session['duration_seconds']}s, Amount: ${session['total_amount']}")
        print(f"      Neighbors: {result['fraud_neighbors']} fraud, {result['normal_neighbors']} normal")
        print(f"      Avg similarity: fraud={result['avg_fraud_sim']:.3f}, normal={result['avg_normal_sim']:.3f}")
        print(f"      Predicted: {predicted} (Expected: {expected})")

        if result["top_match"]:
            top = result["top_match"]
            top_data = parse_data(top["data"]) if isinstance(top["data"], str) else top["data"]
            if isinstance(top_data, dict):
                print(f"      Top match: {top_data.get('session_id', '?')} ({top_data.get('label', '?')}) score={top['score']:.3f}")

    if total > 0:
        accuracy = correct / total
        print(f"\n   📈 Accuracy: {correct}/{total} = {accuracy:.1%}")

    return correct, total


def demo_pattern_search(detector: AnomalyDetector):
    """Demo: Search for similar patterns."""
    print("\n" + "=" * 70)
    print("DEMO: Search for Similar Transfer Patterns")
    print("=" * 70)

    # Search for sessions with transfer patterns
    pattern = {
        "events": {"$mode": "chained", "sequence": ["login", "transfer", "transfer", "logout"]}
    }

    print("\n🔍 Searching for sessions with multiple transfers...")
    results = detector.search_similar_to_pattern(pattern, limit=5)

    print(f"   Found {len(results)} similar sessions:")
    for r in results[:5]:
        data = parse_data(r["data"])
        if not isinstance(data, dict):
            continue
        events = data.get("events", {}).get("sequence", [])
        label = data.get("label", "?")
        session_id = data.get("session_id", "?")
        print(f"   - {session_id} [{label}]: {' → '.join(events)} (score: {r['score']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Event Sequence Anomaly Detection")
    parser.add_argument("--http", action="store_true", help="Use HTTP API")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN")
    args = parser.parse_args()

    print("=" * 70)
    print("EVENT SEQUENCE MATCHING (ANOMALY DETECTION)")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")
    print(f"   k-NN neighbors: {args.k}")

    if args.http:
        print(f"   Server: {args.url}")
        print("   ⚠️  Ensure server is running")

    start_time = time.time()

    # Create detector
    detector = AnomalyDetector(use_http=args.http, base_url=args.url)

    # Generate and ingest training data
    print("\n📊 Generating training data...")
    normal_sessions = generate_normal_sessions(20)
    fraud_sessions = generate_fraud_sessions(10)

    detector.ingest_training_data(fraud_sessions, normal_sessions)

    # Generate test data
    print("\n📊 Generating test data...")
    test_sessions = generate_test_sessions()
    print(f"   Test sessions: {len(test_sessions)}")

    # Run demos
    correct, total = demo_fraud_detection(detector, test_sessions)
    demo_pattern_search(detector)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    accuracy_pct = (correct / total * 100) if total > 0 else 0
    print(f"""
    Mode: {mode}
    Elapsed: {elapsed:.2f}s

    Training Data: {len(normal_sessions) + len(fraud_sessions)} sessions
    Test Data: {len(test_sessions)} sessions
    Accuracy: {correct}/{total} = {accuracy_pct:.0f}%

    ✅ HTTP-Compatible Implementation:
       - All similarity computed server-side via search_json()
       - No local vector operations (numpy)
       - k-NN classification using neighbor labels
       - Chained encoding for temporal sequences

    This works identically in local and HTTP modes!
    """)


if __name__ == "__main__":
    main()
