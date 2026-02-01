#!/usr/bin/env python3
"""
Compare TorchHD backend vs original Holon encoder on Challenge 002.

This tests whether torchhd's Level embeddings and GPU acceleration
improve the API pattern detection task.
"""

import sys
import time
import json
import random
import argparse
from typing import Dict, List, Any, Tuple

# Add holon to path
sys.path.insert(0, "/home/john/work/holon")

from holon.torchhd_encoder import TorchHDStore, TorchHDEncoder

# Same data generation as original
ENDPOINTS = [
    "/api/users/{id}",
    "/api/products/{id}",
    "/api/orders/{id}",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/search",
    "/api/upload",
    "/api/download/{id}",
    "/api/webhooks",
    "/api/admin/users",
    "/api/admin/config",
    "/api/admin/logs",
]

USER_AGENTS = [
    "Mozilla/5.0 Chrome/120",
    "Mozilla/5.0 Firefox/121",
    "Mozilla/5.0 Safari/17",
    "curl/8.0",
    "python-requests/2.31",
    "PostmanRuntime/7.35",
]

def generate_normal_request(base_time: float, user_id: str) -> Dict[str, Any]:
    endpoint = random.choice(ENDPOINTS[:9])
    method = "GET" if "{id}" in endpoint else random.choice(["GET", "POST"])
    
    return {
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "ip_address": f"192.168.1.{random.randint(1, 254)}",
        "timestamp": {"$time": base_time + random.uniform(0, 3600)},
        "response": {
            "status": random.choice([200, 200, 200, 201, 204]),
            "duration_ms": random.randint(10, 200),
        },
        "headers": {
            "user_agent": random.choice(USER_AGENTS[:3]),
            "content_type": "application/json",
        },
        "label": "normal",
    }


def generate_suspicious_request(base_time: float, pattern: str) -> Dict[str, Any]:
    if pattern == "brute_force":
        return {
            "endpoint": "/api/auth/login",
            "method": "POST",
            "user_id": f"attacker_{random.randint(1, 10)}",
            "ip_address": f"45.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "timestamp": {"$time": base_time + random.uniform(0, 60)},
            "response": {
                "status": 401,
                "duration_ms": random.randint(5, 20),
            },
            "headers": {
                "user_agent": random.choice(USER_AGENTS[3:]),
                "content_type": "application/json",
            },
            "label": "suspicious",
            "pattern": pattern,
        }
    elif pattern == "data_exfil":
        return {
            "endpoint": "/api/download/{id}",
            "method": "GET",
            "user_id": f"user_{random.randint(1, 100)}",
            "ip_address": f"10.0.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "timestamp": {"$time": base_time + random.uniform(0, 300)},
            "response": {
                "status": 200,
                "duration_ms": random.randint(500, 5000),
            },
            "headers": {
                "user_agent": random.choice(USER_AGENTS[3:]),
                "content_type": "application/octet-stream",
            },
            "label": "suspicious",
            "pattern": pattern,
        }
    elif pattern == "admin_probe":
        return {
            "endpoint": random.choice(ENDPOINTS[9:]),
            "method": random.choice(["GET", "POST", "DELETE"]),
            "user_id": f"user_{random.randint(1, 100)}",
            "ip_address": f"203.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "timestamp": {"$time": base_time + random.uniform(0, 600)},
            "response": {
                "status": random.choice([403, 403, 403, 401]),
                "duration_ms": random.randint(5, 50),
            },
            "headers": {
                "user_agent": random.choice(USER_AGENTS),
                "content_type": "application/json",
            },
            "label": "suspicious",
            "pattern": pattern,
        }
    else:  # rate_abuse
        return {
            "endpoint": random.choice(ENDPOINTS[:6]),
            "method": random.choice(["GET", "POST"]),
            "user_id": f"bot_{random.randint(1, 5)}",
            "ip_address": f"185.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "timestamp": {"$time": base_time + random.uniform(0, 10)},
            "response": {
                "status": random.choice([200, 429, 429]),
                "duration_ms": random.randint(1, 10),
            },
            "headers": {
                "user_agent": random.choice(USER_AGENTS[3:]),
                "content_type": "application/json",
            },
            "label": "suspicious",
            "pattern": pattern,
        }


def generate_dataset(num_normal: int = 9000, num_suspicious: int = 1000):
    random.seed(42)
    base_time = 1700000000
    
    train = []
    test_normal = []
    test_suspicious = []
    
    # Generate normal requests
    for i in range(num_normal):
        req = generate_normal_request(base_time, f"user_{random.randint(1, 1000)}")
        if i < int(num_normal * 0.8):
            train.append(req)
        else:
            test_normal.append(req)
    
    # Generate suspicious requests
    patterns = ["brute_force", "data_exfil", "admin_probe", "rate_abuse"]
    for i in range(num_suspicious):
        pattern = patterns[i % len(patterns)]
        req = generate_suspicious_request(base_time, pattern)
        if i < int(num_suspicious * 0.8):
            train.append(req)
        else:
            test_suspicious.append(req)
    
    random.shuffle(train)
    return train, test_normal, test_suspicious


class TorchHDPatternAnalyzer:
    """API Pattern analyzer using TorchHD backend."""
    
    def __init__(self, store: TorchHDStore):
        self.store = store
        self.normal_prototype = None
        self.suspicious_prototypes: Dict[str, Any] = {}
    
    def ingest_training_data(self, requests: List[Dict]) -> Tuple[int, float]:
        start = time.time()
        count = 0
        for req in requests:
            req_clean = {k: v for k, v in req.items() if k not in ["label", "pattern"]}
            self.store.insert(req_clean)
            count += 1
        elapsed = time.time() - start
        return count, elapsed
    
    def learn_prototypes(self, requests: List[Dict], use_advanced: bool = True):
        normal_vecs = []
        suspicious_by_pattern = {}
        
        for req in requests:
            req_clean = {k: v for k, v in req.items() if k not in ["label", "pattern"]}
            vec = self.store.encoder.encode_data(req_clean)
            
            if req.get("label") == "normal":
                normal_vecs.append(vec)
            else:
                pattern = req.get("pattern", "unknown")
                if pattern not in suspicious_by_pattern:
                    suspicious_by_pattern[pattern] = []
                suspicious_by_pattern[pattern].append(vec)
        
        if normal_vecs:
            self.normal_prototype = self.store.encoder.prototype(normal_vecs)
            print(f"   Normal prototype learned from {len(normal_vecs)} examples")
        
        for pattern, vecs in suspicious_by_pattern.items():
            if len(vecs) >= 5:
                raw_prototype = self.store.encoder.prototype(vecs)
                
                if use_advanced and self.normal_prototype is not None:
                    attack_diff = self.store.encoder.difference(self.normal_prototype, raw_prototype)
                    enhanced = self.store.encoder.amplify(raw_prototype, attack_diff, strength=0.5)
                    self.suspicious_prototypes[pattern] = enhanced
                    print(f"   {pattern} prototype learned from {len(vecs)} examples (enhanced)")
                else:
                    self.suspicious_prototypes[pattern] = raw_prototype
                    print(f"   {pattern} prototype learned from {len(vecs)} examples")
    
    def score_request(self, request: Dict) -> Tuple[str, float, str]:
        req_clean = {k: v for k, v in request.items() if k not in ["label", "pattern"]}
        vec = self.store.encoder.encode_data(req_clean)
        
        normal_sim = 0.0
        if self.normal_prototype is not None:
            normal_sim = self.store.encoder.similarity(vec, self.normal_prototype)
        
        best_attack = None
        best_attack_sim = -1.0
        
        for pattern, proto in self.suspicious_prototypes.items():
            sim = self.store.encoder.similarity(vec, proto)
            if sim > best_attack_sim:
                best_attack_sim = sim
                best_attack = pattern
        
        attack_score = best_attack_sim - normal_sim
        
        return "suspicious" if attack_score > 0.05 else "normal", attack_score, best_attack or "none"
    
    def evaluate(self, test_normal: List[Dict], test_suspicious: List[Dict], threshold: float = 0.05):
        tp, fp, tn, fn = 0, 0, 0, 0
        pattern_results = {}
        
        for req in test_normal:
            label, score, pattern = self.score_request(req)
            if score > threshold:
                fp += 1
            else:
                tn += 1
        
        for req in test_suspicious:
            true_pattern = req.get("pattern", "unknown")
            label, score, detected_pattern = self.score_request(req)
            
            if true_pattern not in pattern_results:
                pattern_results[true_pattern] = {"correct": 0, "wrong_pattern": 0, "missed": 0}
            
            if score > threshold:
                tp += 1
                if detected_pattern == true_pattern:
                    pattern_results[true_pattern]["correct"] += 1
                else:
                    pattern_results[true_pattern]["wrong_pattern"] += 1
            else:
                fn += 1
                pattern_results[true_pattern]["missed"] += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "patterns": pattern_results,
        }


def main():
    parser = argparse.ArgumentParser(description="TorchHD API Pattern Analyzer Comparison")
    parser.add_argument("--dimensions", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--advanced", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Challenge 002: TorchHD Backend Comparison")
    print("=" * 60)
    
    # Create TorchHD store
    store = TorchHDStore(dimensions=args.dimensions)
    analyzer = TorchHDPatternAnalyzer(store)
    
    # Generate data
    print(f"\n📊 Generating dataset...")
    train, test_normal, test_suspicious = generate_dataset(num_normal=9000, num_suspicious=1000)
    print(f"   Training: {len(train)}, Test normal: {len(test_normal)}, Test suspicious: {len(test_suspicious)}")
    
    # Ingest
    print(f"\n📥 Ingesting training data...")
    count, elapsed = analyzer.ingest_training_data(train)
    rate = count / elapsed
    print(f"   Ingested {count} requests in {elapsed:.2f}s ({rate:.0f} req/sec)")
    
    # Learn prototypes
    print(f"\n🧠 Learning prototypes...")
    analyzer.learn_prototypes(train, use_advanced=args.advanced)
    
    # Evaluate
    print(f"\n📈 Evaluating (threshold={args.threshold})...")
    metrics = analyzer.evaluate(test_normal, test_suspicious, args.threshold)
    
    print(f"\n   Results:")
    print(f"   ├── Precision: {metrics['precision']*100:.1f}%")
    print(f"   ├── Recall: {metrics['recall']*100:.1f}%")
    print(f"   ├── F1 Score: {metrics['f1']*100:.1f}%")
    print(f"   ├── TP: {metrics['tp']}, FP: {metrics['fp']}")
    print(f"   └── TN: {metrics['tn']}, FN: {metrics['fn']}")
    
    print(f"\n   Pattern Detection:")
    for pattern, stats in metrics['patterns'].items():
        total = stats['correct'] + stats['wrong_pattern'] + stats['missed']
        detected = stats['correct'] + stats['wrong_pattern']
        rate = detected / total * 100 if total > 0 else 0
        print(f"   ├── {pattern}: {rate:.1f}% detected")
    
    # Benchmark scoring latency
    print(f"\n⚡ Latency benchmark...")
    import torch
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start = time.time()
    for req in test_normal[:200]:
        analyzer.score_request(req)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    latency = elapsed / 200 * 1000
    print(f"   Latency: {latency:.2f}ms per request")
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nTorchHD Backend Results:")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall: {metrics['recall']*100:.1f}%")
    print(f"  F1 Score: {metrics['f1']*100:.1f}%")
    print(f"  Latency: {latency:.2f}ms")
    print(f"\nOriginal Holon (from earlier):")
    print(f"  Precision: 95.9% (with advanced primitives)")
    print(f"  Recall: 81.5%")
    print(f"  F1 Score: 88.1%")
    print(f"  Latency: 0.23ms")
    
    print(f"\n📊 Analysis:")
    if metrics['precision'] > 0.959:
        print(f"  ✅ TorchHD beats original on precision")
    if metrics['recall'] > 0.815:
        print(f"  ✅ TorchHD beats original on recall")
    if latency > 0.23:
        print(f"  ⚠️  TorchHD slower per-request (GPU transfer overhead)")
    print(f"\n💡 Key insight: TorchHD's Level embeddings give numeric similarity")
    print(f"   (close status codes/durations = similar vectors)")


if __name__ == "__main__":
    main()
