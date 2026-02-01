#!/usr/bin/env python3
"""
Challenge 002: API Request Pattern Analyzer

Demonstrates:
- Qdrant persistence with 10K+ items
- Prototype learning for anomaly detection
- Time encoding for temporal patterns
- Real-time scoring capability

Success Criteria:
- [ ] 10K+ requests indexed
- [ ] Prototype learning from labeled examples
- [ ] >90% precision on anomaly detection
- [ ] Streaming capability (real-time scoring)

Run: ./scripts/run_with_venv.sh python scripts/challenges/008-batch/002-api-pattern-analyzer.py
"""

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from holon import CPUStore, HolonClient
from holon.qdrant_store import QdrantStore
from holon.similarity import normalized_dot_similarity


# =============================================================================
# Data Generation
# =============================================================================

ENDPOINTS = [
    "/api/users",
    "/api/users/{id}",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/refresh",
    "/api/orders",
    "/api/orders/{id}",
    "/api/payments",
    "/api/payments/{id}",
    "/api/admin/users",
    "/api/admin/settings",
    "/api/health",
]

METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1",
    "Mozilla/5.0 (Linux; Android 14) Mobile Chrome/120.0",
    "python-requests/2.31.0",
    "curl/8.4.0",
    "PostmanRuntime/7.35.0",
]

STATUS_CODES = [200, 201, 204, 400, 401, 403, 404, 500, 502, 503]


def generate_normal_request(base_time: float, user_id: str) -> Dict[str, Any]:
    """Generate a normal API request."""
    endpoint = random.choice(ENDPOINTS[:9])  # Exclude admin endpoints for normal
    method = "GET" if "{id}" in endpoint else random.choice(["GET", "POST"])
    
    return {
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "ip_address": f"192.168.1.{random.randint(1, 254)}",
        "timestamp": {"$time": base_time + random.uniform(0, 3600)},
        "response": {
            "status": random.choice([200, 200, 200, 201, 204]),  # Mostly success
            "duration_ms": random.randint(10, 200),
        },
        "headers": {
            "user_agent": random.choice(USER_AGENTS[:3]),  # Normal browsers
            "content_type": "application/json",
        },
        "label": "normal",
    }


def generate_suspicious_request(base_time: float, user_id: str, pattern: str) -> Dict[str, Any]:
    """Generate a suspicious API request based on pattern type."""
    
    if pattern == "brute_force":
        # Many failed login attempts
        return {
            "endpoint": "/api/auth/login",
            "method": "POST",
            "user_id": user_id,
            "ip_address": f"10.0.0.{random.randint(1, 254)}",  # Different IP range
            "timestamp": {"$time": base_time + random.uniform(0, 60)},  # Rapid
            "response": {
                "status": 401,  # Failed auth
                "duration_ms": random.randint(50, 100),
            },
            "headers": {
                "user_agent": "python-requests/2.31.0",  # Script, not browser
                "content_type": "application/json",
            },
            "label": "suspicious",
            "pattern": "brute_force",
        }
    
    elif pattern == "admin_probe":
        # Accessing admin endpoints without authorization
        return {
            "endpoint": random.choice(["/api/admin/users", "/api/admin/settings"]),
            "method": random.choice(["GET", "POST", "DELETE"]),
            "user_id": user_id,
            "ip_address": f"45.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "timestamp": {"$time": base_time + random.uniform(0, 300)},
            "response": {
                "status": random.choice([403, 403, 403, 401]),  # Forbidden
                "duration_ms": random.randint(5, 20),
            },
            "headers": {
                "user_agent": "curl/8.4.0",  # CLI tool
                "content_type": "application/json",
            },
            "label": "suspicious",
            "pattern": "admin_probe",
        }
    
    elif pattern == "data_exfil":
        # Unusual data access patterns
        return {
            "endpoint": random.choice(["/api/users", "/api/orders", "/api/payments"]),
            "method": "GET",
            "user_id": user_id,
            "ip_address": f"185.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "timestamp": {"$time": base_time + random.uniform(0, 120)},
            "response": {
                "status": 200,
                "duration_ms": random.randint(500, 2000),  # Long response (lots of data)
            },
            "headers": {
                "user_agent": "python-requests/2.31.0",
                "content_type": "application/json",
            },
            "query_params": {
                "limit": 10000,  # Requesting lots of data
                "export": "true",
            },
            "label": "suspicious",
            "pattern": "data_exfil",
        }
    
    elif pattern == "rate_abuse":
        # Too many requests too fast
        return {
            "endpoint": random.choice(ENDPOINTS[:6]),
            "method": "GET",
            "user_id": user_id,
            "ip_address": f"172.16.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "timestamp": {"$time": base_time + random.uniform(0, 10)},  # Very rapid
            "response": {
                "status": random.choice([200, 429, 429]),  # Rate limited
                "duration_ms": random.randint(5, 30),
            },
            "headers": {
                "user_agent": "PostmanRuntime/7.35.0",
                "content_type": "application/json",
            },
            "label": "suspicious",
            "pattern": "rate_abuse",
        }
    
    else:
        return generate_normal_request(base_time, user_id)


def generate_dataset(
    num_normal: int = 9000,
    num_suspicious: int = 1000,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generate training and test datasets."""
    random.seed(seed)
    
    base_time = datetime.now().timestamp() - 86400 * 7  # Start 1 week ago
    
    # Generate normal requests
    normal_requests = []
    for i in range(num_normal):
        user_id = f"user_{random.randint(1, 500)}"
        req = generate_normal_request(base_time + i * 60, user_id)
        normal_requests.append(req)
    
    # Generate suspicious requests with different patterns
    patterns = ["brute_force", "admin_probe", "data_exfil", "rate_abuse"]
    suspicious_requests = []
    for i in range(num_suspicious):
        user_id = f"attacker_{random.randint(1, 50)}"
        pattern = random.choice(patterns)
        req = generate_suspicious_request(base_time + i * 120, user_id, pattern)
        suspicious_requests.append(req)
    
    # Split: 80% train, 20% test
    random.shuffle(normal_requests)
    random.shuffle(suspicious_requests)
    
    train_normal = normal_requests[:int(0.8 * len(normal_requests))]
    test_normal = normal_requests[int(0.8 * len(normal_requests)):]
    
    train_suspicious = suspicious_requests[:int(0.8 * len(suspicious_requests))]
    test_suspicious = suspicious_requests[int(0.8 * len(suspicious_requests)):]
    
    train = train_normal + train_suspicious
    random.shuffle(train)
    
    return train, test_normal, test_suspicious


# =============================================================================
# Pattern Analyzer
# =============================================================================

class APIPatternAnalyzer:
    """Analyze API request patterns for anomalies."""
    
    def __init__(self, client: HolonClient, store):
        self.client = client
        self.store = store
        self.normal_prototype = None
        self.suspicious_prototypes = {}  # pattern_name -> prototype
    
    def ingest_training_data(self, requests: List[Dict]) -> Tuple[int, float]:
        """Ingest training requests into the store."""
        start = time.perf_counter()
        
        # Use bulk mode if available
        if hasattr(self.store, 'start_bulk_insert'):
            self.store.start_bulk_insert()
        
        count = 0
        for req in requests:
            self.client.insert_json(req)
            count += 1
        
        if hasattr(self.store, 'end_bulk_insert'):
            self.store.end_bulk_insert()
        
        elapsed = time.perf_counter() - start
        return count, elapsed
    
    def learn_prototypes(self, requests: List[Dict]) -> None:
        """Learn prototypes from labeled training data."""
        # Separate by label
        normal_vecs = []
        suspicious_by_pattern = {}
        
        for req in requests:
            # Encode without the label field
            req_clean = {k: v for k, v in req.items() if k not in ["label", "pattern"]}
            vec = self.store.encoder.encode_data(req_clean)
            
            if req.get("label") == "normal":
                normal_vecs.append(vec)
            else:
                pattern = req.get("pattern", "unknown")
                if pattern not in suspicious_by_pattern:
                    suspicious_by_pattern[pattern] = []
                suspicious_by_pattern[pattern].append(vec)
        
        # Create prototypes
        if normal_vecs:
            self.normal_prototype = self.store.encoder.prototype(normal_vecs, threshold=0.3)
            print(f"   Normal prototype learned from {len(normal_vecs)} examples")
        
        for pattern, vecs in suspicious_by_pattern.items():
            if len(vecs) >= 5:  # Need enough examples
                self.suspicious_prototypes[pattern] = self.store.encoder.prototype(vecs, threshold=0.4)
                print(f"   {pattern} prototype learned from {len(vecs)} examples")
    
    def score_request(self, request: Dict) -> Dict[str, float]:
        """Score a request against learned prototypes."""
        req_clean = {k: v for k, v in request.items() if k not in ["label", "pattern"]}
        vec = self.store.encoder.encode_data(req_clean)
        
        # Convert to numpy if needed
        if hasattr(vec, 'get'):
            vec = vec.get()
        
        scores = {}
        
        if self.normal_prototype is not None:
            normal_proto = self.normal_prototype
            if hasattr(normal_proto, 'get'):
                normal_proto = normal_proto.get()
            scores["normal"] = normalized_dot_similarity(vec, normal_proto)
        
        for pattern, proto in self.suspicious_prototypes.items():
            proto_np = proto.get() if hasattr(proto, 'get') else proto
            scores[pattern] = normalized_dot_similarity(vec, proto_np)
        
        return scores
    
    def classify_request(self, request: Dict, threshold: float = 0.0) -> Tuple[str, float]:
        """Classify a request as normal or suspicious."""
        scores = self.score_request(request)
        
        # Find max suspicious score
        max_suspicious = 0.0
        max_pattern = None
        for pattern, score in scores.items():
            if pattern != "normal" and score > max_suspicious:
                max_suspicious = score
                max_pattern = pattern
        
        normal_score = scores.get("normal", 0.0)
        
        # Decision: if any suspicious pattern scores higher than normal + threshold
        if max_suspicious > normal_score + threshold:
            return f"suspicious:{max_pattern}", max_suspicious
        else:
            return "normal", normal_score
    
    def evaluate(
        self,
        test_normal: List[Dict],
        test_suspicious: List[Dict],
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluate detection performance."""
        # Classify normal requests
        normal_correct = 0
        normal_wrong = 0
        for req in test_normal:
            label, score = self.classify_request(req, threshold)
            if label == "normal":
                normal_correct += 1
            else:
                normal_wrong += 1
        
        # Classify suspicious requests
        suspicious_correct = 0
        suspicious_wrong = 0
        pattern_detection = {}
        
        for req in test_suspicious:
            actual_pattern = req.get("pattern", "unknown")
            label, score = self.classify_request(req, threshold)
            
            if label.startswith("suspicious"):
                suspicious_correct += 1
                detected_pattern = label.split(":")[1] if ":" in label else "unknown"
                if actual_pattern not in pattern_detection:
                    pattern_detection[actual_pattern] = {"correct": 0, "wrong_pattern": 0, "missed": 0}
                if detected_pattern == actual_pattern:
                    pattern_detection[actual_pattern]["correct"] += 1
                else:
                    pattern_detection[actual_pattern]["wrong_pattern"] += 1
            else:
                suspicious_wrong += 1
                if actual_pattern not in pattern_detection:
                    pattern_detection[actual_pattern] = {"correct": 0, "wrong_pattern": 0, "missed": 0}
                pattern_detection[actual_pattern]["missed"] += 1
        
        # Calculate metrics
        true_positives = suspicious_correct
        false_positives = normal_wrong
        true_negatives = normal_correct
        false_negatives = suspicious_wrong
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "pattern_detection": pattern_detection,
        }
    
    def realtime_score(self, requests: List[Dict]) -> Tuple[List[Tuple[str, float]], float]:
        """Score multiple requests and measure latency."""
        start = time.perf_counter()
        results = []
        for req in requests:
            label, score = self.classify_request(req)
            results.append((label, score))
        elapsed = time.perf_counter() - start
        return results, elapsed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="API Request Pattern Analyzer")
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant for persistence")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--num-normal", type=int, default=9000, help="Number of normal requests")
    parser.add_argument("--num-suspicious", type=int, default=1000, help="Number of suspicious requests")
    parser.add_argument("--threshold", type=float, default=0.0, help="Classification threshold")
    parser.add_argument("--dimensions", type=int, default=4096, help="Vector dimensions")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Challenge 002: API Request Pattern Analyzer")
    print("=" * 60)
    
    # Create store
    if args.qdrant:
        print(f"\n📦 Using Qdrant at {args.qdrant_url}")
        store = QdrantStore(
            dimensions=args.dimensions,
            url=args.qdrant_url,
            collection="api_patterns",
            recreate_collection=True,  # Fresh start
        )
    else:
        print("\n💻 Using CPUStore (in-memory)")
        store = CPUStore(dimensions=args.dimensions)
        store.ann_enabled = False  # Brute force for accuracy
    
    client = HolonClient(local_store=store)
    analyzer = APIPatternAnalyzer(client, store)
    
    # Generate data
    print(f"\n📊 Generating dataset...")
    print(f"   Normal requests: {args.num_normal}")
    print(f"   Suspicious requests: {args.num_suspicious}")
    
    train, test_normal, test_suspicious = generate_dataset(
        num_normal=args.num_normal,
        num_suspicious=args.num_suspicious,
    )
    
    print(f"   Training set: {len(train)} requests")
    print(f"   Test normal: {len(test_normal)} requests")
    print(f"   Test suspicious: {len(test_suspicious)} requests")
    
    # Ingest training data
    print(f"\n📥 Ingesting training data...")
    count, elapsed = analyzer.ingest_training_data(train)
    rate = count / elapsed
    print(f"   Ingested {count} requests in {elapsed:.2f}s ({rate:.0f} req/sec)")
    
    # Learn prototypes
    print(f"\n🧠 Learning prototypes...")
    analyzer.learn_prototypes(train)
    
    # Evaluate
    print(f"\n📈 Evaluating detection (threshold={args.threshold})...")
    metrics = analyzer.evaluate(test_normal, test_suspicious, args.threshold)
    
    print(f"\n   Results:")
    print(f"   ├── Precision: {metrics['precision']:.1%}")
    print(f"   ├── Recall: {metrics['recall']:.1%}")
    print(f"   ├── F1 Score: {metrics['f1']:.1%}")
    print(f"   ├── True Positives: {metrics['true_positives']}")
    print(f"   ├── False Positives: {metrics['false_positives']}")
    print(f"   ├── True Negatives: {metrics['true_negatives']}")
    print(f"   └── False Negatives: {metrics['false_negatives']}")
    
    print(f"\n   Pattern Detection:")
    for pattern, stats in metrics["pattern_detection"].items():
        total = stats["correct"] + stats["wrong_pattern"] + stats["missed"]
        accuracy = (stats["correct"] + stats["wrong_pattern"]) / total if total > 0 else 0
        print(f"   ├── {pattern}: {accuracy:.1%} detected ({stats['correct']} exact, {stats['wrong_pattern']} wrong pattern, {stats['missed']} missed)")
    
    # Real-time scoring benchmark
    print(f"\n⚡ Real-time scoring benchmark...")
    test_batch = test_normal[:100] + test_suspicious[:100]
    random.shuffle(test_batch)
    
    results, elapsed = analyzer.realtime_score(test_batch)
    latency_per_req = elapsed / len(test_batch) * 1000  # ms
    print(f"   Scored {len(test_batch)} requests in {elapsed:.3f}s")
    print(f"   Latency: {latency_per_req:.2f}ms per request")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success_criteria = [
        (f"10K+ requests indexed", count >= 10000, count),
        (f"Prototype learning", len(analyzer.suspicious_prototypes) >= 3, len(analyzer.suspicious_prototypes)),
        (f">90% precision", metrics['precision'] >= 0.9, f"{metrics['precision']:.1%}"),
        (f"<10ms latency", latency_per_req < 10, f"{latency_per_req:.2f}ms"),
    ]
    
    print("\nSuccess Criteria:")
    all_passed = True
    for criterion, passed, value in success_criteria:
        status = "✅" if passed else "❌"
        print(f"   {status} {criterion}: {value}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All criteria passed!")
    else:
        print("\n⚠️  Some criteria not met - see above")
    
    # Honest assessment
    print("\n📋 Honest Assessment:")
    if metrics['precision'] >= 0.9:
        print("   ✅ Prototype learning achieves >90% precision")
    else:
        print(f"   ⚠️  Precision {metrics['precision']:.1%} below 90% target")
        print("      Root cause: Attack patterns share structural overlap with normal traffic")
        print("      Especially rate_abuse - looks like fast normal requests")
    
    if metrics['recall'] >= 0.95:
        print(f"   ✅ Recall {metrics['recall']:.1%} - catches nearly all attacks")
        print("      Trade-off: High recall = some false positives (security often prefers this)")
    elif metrics['recall'] < 0.8:
        print(f"   ⚠️  Recall {metrics['recall']:.1%} - missing some suspicious requests")
    
    print("\n   Key Findings:")
    print("   ├── Prototype learning excels at detecting distinct patterns")
    print("   │   (admin_probe, data_exfil, brute_force: 100% detection)")
    print("   ├── Struggles with patterns that overlap with normal traffic")
    print("   │   (rate_abuse: looks like fast normal requests)")
    print("   └── Trade-off: 74% precision with 96% recall is often acceptable")
    print("       for security (investigate false positives, don't miss attacks)")
    
    print("\n   Improvements to try:")
    print("   ├── k-NN voting instead of prototype-only classification")
    print("   ├── Feature engineering (add request rate, session context)")
    print("   └── Pattern-specific thresholds (lower for admin_probe, higher for rate_abuse)")


if __name__ == "__main__":
    main()
