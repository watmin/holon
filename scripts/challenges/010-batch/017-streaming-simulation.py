#!/usr/bin/env python3
"""
Challenge 010-017: Streaming Simulation

Simulates a real-world streaming scenario:
1. TRAINING PHASE: Build accumulator from historical benign data
2. STREAMING PHASE: Process incoming events in real-time
3. DETECTION: Flag anomalies based on similarity to learned patterns
4. METRICS: Track detection performance over the stream

Features:
- Complex multi-schema data (realistic generator)
- Time-windowed metrics (rolling accuracy)
- Attack bursts at specific times
- Online accumulator updates (optional)
- Performance benchmarks

This demonstrates practical application of the accumulator primitive.
"""

import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from realistic_data_generator import RealisticDataGenerator
from holon import DeterministicVectorManager
from holon.encoder import Encoder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


@dataclass
class StreamEvent:
    """A single event in the stream."""
    timestamp: float
    record: dict
    schema: str
    is_malicious: bool

    # Filled in during processing
    similarity: Optional[float] = None
    flagged: Optional[bool] = None
    latency_ms: Optional[float] = None


@dataclass
class StreamMetrics:
    """Metrics for a time window."""
    window_start: float
    window_end: float
    total_events: int
    benign_events: int
    malicious_events: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    avg_latency_ms: float
    avg_similarity_benign: float
    avg_similarity_malicious: float

    @property
    def precision(self) -> float:
        return self.true_positives / max(1, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return self.true_positives / max(1, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(0.001, p + r)


def normalize_record(record: dict, schema: str) -> dict:
    """Normalize record for encoding."""
    import re

    exclude_fields = {
        "timestamp", "request_id", "trace_id", "span_id", "session_id",
        "created_at", "updated_at", "triggered_at", "started_at",
    }

    normalized = {"_schema": schema}

    for key, value in record.items():
        if key in exclude_fields or key.startswith("_extra"):
            continue

        # Normalize high-cardinality fields
        if key == "path" and isinstance(value, str):
            value = re.sub(r'/\d+', '/{id}', value)
        elif key in ("user_id", "order_id", "customer_id") and isinstance(value, str):
            value = f"{key}_present"
        elif key == "ip_address" and isinstance(value, str):
            parts = value.split(".")
            value = f"{parts[0]}.{parts[1]}.x.x" if len(parts) >= 2 else value

        # Simplify complex values
        if isinstance(value, dict):
            normalized[key] = {k: "present" for k in value.keys()}
        elif isinstance(value, list):
            normalized[key] = f"list_len_{min(len(value), 10)}"
        elif value is None:
            normalized[key] = "null"
        else:
            normalized[key] = value

    return normalized


def generate_malicious_events(n: int, seed: int) -> List[dict]:
    """Generate malicious events for the stream."""
    import random
    random.seed(seed)

    templates = [
        # API attacks
        {"_schema": "api_request", "method": "GET", "path": "/api/../../../etc/passwd", "status_code": 403},
        {"_schema": "api_request", "method": "GET", "path": "/api/users/' OR 1=1--", "status_code": 500},
        {"_schema": "api_request", "method": "TRACE", "path": "/api/debug", "status_code": 200},
        {"_schema": "api_request", "method": "GET", "path": "/.git/config", "status_code": 404},
        # Log anomalies
        {"_schema": "log_entry", "level": "FATAL", "message": "Unauthorized root access", "service": "auth"},
        {"_schema": "log_entry", "level": "ERROR", "message": "SQL injection detected", "service": "db"},
        # Suspicious events
        {"_schema": "user_event", "event_type": "privilege_escalation", "user_id": "unknown"},
        {"_schema": "user_event", "event_type": "data_exfiltration", "user_id": "attacker"},
        # Config tampering
        {"_schema": "config_change", "key": "security.auth_disabled", "new_value": True},
        {"_schema": "config_change", "key": "firewall.rules", "new_value": "allow_all"},
        # Alert attacks
        {"_schema": "alert", "severity": "emergency", "title": "Ransomware detected"},
        {"_schema": "alert", "severity": "critical", "title": "Data breach in progress"},
    ]

    return [random.choice(templates).copy() for _ in range(n)]


class StreamingDetector:
    """
    Streaming anomaly detector using Holon accumulator.

    Supports:
    - Batch training from historical data
    - Real-time detection on stream
    - Optional online updates
    """

    def __init__(
        self,
        encoder: Encoder,
        threshold: float = 0.3,
        online_learning: bool = False,
        online_weight: float = 0.01,
    ):
        self.encoder = encoder
        self.threshold = threshold
        self.online_learning = online_learning
        self.online_weight = online_weight

        self.accumulator = None
        self.normalized_proto = None
        self.observation_count = 0

    def train(self, records: List[Tuple[dict, str]]):
        """Train from historical data."""
        self.accumulator = self.encoder.create_accumulator()

        for record, schema in records:
            normalized = normalize_record(record, schema)
            vec = self.encoder.encode_data(normalized)
            self.accumulator = self.encoder.accumulate(self.accumulator, vec)
            self.observation_count += 1

        self.normalized_proto = self.encoder.normalize_accumulator(self.accumulator)

    def process(self, event: StreamEvent) -> StreamEvent:
        """Process a single stream event."""
        start = time.perf_counter()

        # Encode and compute similarity
        normalized = normalize_record(event.record, event.schema)
        vec = self.encoder.encode_data(normalized)
        similarity = cosine_similarity(vec, self.normalized_proto)

        # Detection decision
        flagged = similarity < self.threshold

        # Online update (optional)
        if self.online_learning and not flagged:
            # Only learn from events we think are benign
            self.accumulator = self.encoder.accumulate(self.accumulator, vec)
            self.observation_count += 1
            # Re-normalize periodically (expensive, so not every event)
            if self.observation_count % 100 == 0:
                self.normalized_proto = self.encoder.normalize_accumulator(self.accumulator)

        latency = (time.perf_counter() - start) * 1000

        event.similarity = similarity
        event.flagged = flagged
        event.latency_ms = latency

        return event


def generate_stream(
    benign_records: List[Tuple[dict, str]],
    malicious_records: List[dict],
    stream_duration: float,
    events_per_second: float,
    attack_windows: List[Tuple[float, float, float]],  # (start, end, rate)
    seed: int = 42,
) -> List[StreamEvent]:
    """
    Generate a stream of events with attack bursts.

    Args:
        benign_records: Pool of benign (record, schema) tuples
        malicious_records: Pool of malicious records
        stream_duration: Total stream duration in seconds
        events_per_second: Base event rate
        attack_windows: List of (start_time, end_time, malicious_rate)
    """
    import random
    random.seed(seed)

    events = []
    current_time = 0.0

    while current_time < stream_duration:
        # Determine if this is an attack window
        malicious_rate = 0.0
        for start, end, rate in attack_windows:
            if start <= current_time < end:
                malicious_rate = rate
                break

        # Generate event
        if random.random() < malicious_rate and malicious_records:
            record = random.choice(malicious_records).copy()
            schema = record.get("_schema", "unknown")
            is_malicious = True
        else:
            record, schema = random.choice(benign_records)
            is_malicious = False

        events.append(StreamEvent(
            timestamp=current_time,
            record=record,
            schema=schema,
            is_malicious=is_malicious,
        ))

        # Advance time (Poisson-like interval)
        interval = random.expovariate(events_per_second)
        current_time += interval

    return events


def compute_window_metrics(
    events: List[StreamEvent],
    window_start: float,
    window_end: float,
) -> StreamMetrics:
    """Compute metrics for a time window."""
    window_events = [e for e in events if window_start <= e.timestamp < window_end]

    if not window_events:
        return StreamMetrics(
            window_start=window_start, window_end=window_end,
            total_events=0, benign_events=0, malicious_events=0,
            true_positives=0, false_positives=0, false_negatives=0, true_negatives=0,
            avg_latency_ms=0, avg_similarity_benign=0, avg_similarity_malicious=0,
        )

    tp = sum(1 for e in window_events if e.is_malicious and e.flagged)
    fp = sum(1 for e in window_events if not e.is_malicious and e.flagged)
    fn = sum(1 for e in window_events if e.is_malicious and not e.flagged)
    tn = sum(1 for e in window_events if not e.is_malicious and not e.flagged)

    benign_sims = [e.similarity for e in window_events if not e.is_malicious]
    malicious_sims = [e.similarity for e in window_events if e.is_malicious]

    return StreamMetrics(
        window_start=window_start,
        window_end=window_end,
        total_events=len(window_events),
        benign_events=len(benign_sims),
        malicious_events=len(malicious_sims),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        avg_latency_ms=np.mean([e.latency_ms for e in window_events]),
        avg_similarity_benign=np.mean(benign_sims) if benign_sims else 0,
        avg_similarity_malicious=np.mean(malicious_sims) if malicious_sims else 0,
    )


def main():
    print("=" * 80)
    print("Challenge 010-017: Streaming Simulation")
    print("=" * 80)
    print("""
Scenario: Real-time API/log monitoring with attack detection.

Phases:
1. TRAINING: Build accumulator from 10,000 historical benign events
2. STREAMING: Process 60 seconds of traffic at 100 events/sec
3. ATTACKS: 3 attack bursts at different times

Measuring: Detection accuracy, latency, similarity distributions
""")

    # Setup - Use DETERMINISTIC vector manager for distributed consensus
    # Same atom = same vector regardless of processing node or order
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    print("Using DeterministicVectorManager for distributed consensus")
    print("  Same atom → same vector across all nodes")

    # Verify deterministic consensus
    print("\n--- Verifying Distributed Consensus ---")
    vm_node1 = DeterministicVectorManager(dimensions=4096, global_seed=42)
    vm_node2 = DeterministicVectorManager(dimensions=4096, global_seed=42)

    test_atoms = ["api_request", "GET", "/api/users", "level", "ERROR"]
    consensus_ok = True
    for atom in test_atoms:
        v1 = vm_node1.get_vector(atom)
        v2 = vm_node2.get_vector(atom)
        match = np.array_equal(v1, v2)
        if not match:
            consensus_ok = False
            print(f"  ✗ MISMATCH: {atom}")

    if consensus_ok:
        print(f"  ✓ All {len(test_atoms)} atoms produce identical vectors across nodes")
    else:
        print("  ✗ CONSENSUS FAILED - vectors differ between nodes!")
        return

    # Generate training data (benign only)
    print("\n--- Phase 1: Training Data Generation ---")
    gen = RealisticDataGenerator(seed=42, cardinality=10000)
    train_records, train_schemas, _ = gen.generate_dataset(10000)
    train_data = list(zip(train_records, train_schemas))

    print(f"Generated {len(train_data)} benign training events")

    # Generate stream data pools
    print("\n--- Stream Data Pools ---")
    gen_stream = RealisticDataGenerator(seed=100, cardinality=5000)
    stream_benign, stream_schemas, _ = gen_stream.generate_dataset(5000)
    stream_benign_data = list(zip(stream_benign, stream_schemas))

    malicious_pool = generate_malicious_events(100, seed=200)

    print(f"Benign pool: {len(stream_benign_data)} events")
    print(f"Malicious pool: {len(malicious_pool)} events")

    # Train detector
    print("\n--- Phase 2: Training Detector ---")
    detector = StreamingDetector(encoder, threshold=0.15, online_learning=False)

    start = time.time()
    detector.train(train_data)
    train_time = time.time() - start

    print(f"Trained on {detector.observation_count} events in {train_time:.2f}s")
    print(f"Rate: {detector.observation_count/train_time:.0f} events/sec")

    # Generate stream with attack windows
    print("\n--- Phase 3: Generating Stream ---")

    stream_duration = 60.0  # 60 seconds
    events_per_second = 100  # 100 events/sec

    # Attack windows: (start, end, malicious_rate)
    attack_windows = [
        (10.0, 15.0, 0.5),   # 50% attacks at t=10-15s
        (30.0, 35.0, 0.3),   # 30% attacks at t=30-35s
        (50.0, 55.0, 0.8),   # 80% attacks at t=50-55s
    ]

    stream = generate_stream(
        benign_records=stream_benign_data,
        malicious_records=malicious_pool,
        stream_duration=stream_duration,
        events_per_second=events_per_second,
        attack_windows=attack_windows,
        seed=300,
    )

    total_benign = sum(1 for e in stream if not e.is_malicious)
    total_malicious = sum(1 for e in stream if e.is_malicious)

    print(f"Generated {len(stream)} stream events over {stream_duration}s")
    print(f"  Benign: {total_benign} ({100*total_benign/len(stream):.1f}%)")
    print(f"  Malicious: {total_malicious} ({100*total_malicious/len(stream):.1f}%)")
    print(f"\nAttack windows:")
    for start, end, rate in attack_windows:
        print(f"  t={start:.0f}s to t={end:.0f}s: {rate:.0%} malicious rate")

    # Process stream
    print("\n--- Phase 4: Processing Stream ---")

    process_start = time.time()
    for event in stream:
        detector.process(event)
    process_time = time.time() - process_start

    print(f"Processed {len(stream)} events in {process_time:.2f}s")
    print(f"Throughput: {len(stream)/process_time:.0f} events/sec")
    print(f"Avg latency: {np.mean([e.latency_ms for e in stream]):.3f} ms/event")

    # Overall metrics
    print("\n--- Overall Detection Results ---")

    all_flagged = [e for e in stream if e.flagged]
    tp = sum(1 for e in stream if e.is_malicious and e.flagged)
    fp = sum(1 for e in stream if not e.is_malicious and e.flagged)
    fn = sum(1 for e in stream if e.is_malicious and not e.flagged)
    tn = sum(1 for e in stream if not e.is_malicious and not e.flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp:4d} (malicious correctly detected)")
    print(f"  False Positives: {fp:4d} (benign incorrectly flagged)")
    print(f"  False Negatives: {fn:4d} (malicious missed)")
    print(f"  True Negatives:  {tn:4d} (benign correctly passed)")

    print(f"\nMetrics:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.3f}")

    # Similarity distributions
    benign_sims = [e.similarity for e in stream if not e.is_malicious]
    malicious_sims = [e.similarity for e in stream if e.is_malicious]

    print(f"\nSimilarity Distributions:")
    print(f"  Benign:    mean={np.mean(benign_sims):+.4f}, std={np.std(benign_sims):.4f}, range=[{min(benign_sims):+.4f}, {max(benign_sims):+.4f}]")
    print(f"  Malicious: mean={np.mean(malicious_sims):+.4f}, std={np.std(malicious_sims):.4f}, range=[{min(malicious_sims):+.4f}, {max(malicious_sims):+.4f}]")
    print(f"  Separation: {np.mean(benign_sims) - np.mean(malicious_sims):+.4f}")

    # Window-based analysis
    print("\n--- Time Window Analysis (5s windows) ---")

    window_size = 5.0
    window_metrics = []

    for window_start in np.arange(0, stream_duration, window_size):
        metrics = compute_window_metrics(stream, window_start, window_start + window_size)
        window_metrics.append(metrics)

    print(f"\n{'Time':>8} {'Events':>8} {'Mal%':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Recall':>6} {'F1':>6}")
    print("-" * 70)

    for m in window_metrics:
        if m.total_events == 0:
            continue
        mal_pct = 100 * m.malicious_events / m.total_events if m.total_events > 0 else 0

        # Highlight attack windows
        is_attack = any(s <= m.window_start < e for s, e, _ in attack_windows)
        prefix = "→ " if is_attack else "  "

        print(f"{prefix}{m.window_start:>5.0f}s {m.total_events:>7d} {mal_pct:>5.1f}% {m.true_positives:>4d} {m.false_positives:>4d} {m.false_negatives:>4d} {m.precision:>5.1%} {m.recall:>5.1%} {m.f1:>6.3f}")

    # Sample detections
    print("\n--- Sample Detections ---")

    print("\nCorrectly detected attacks (True Positives):")
    tps = [e for e in stream if e.is_malicious and e.flagged][:5]
    for e in tps:
        schema = e.record.get("_schema", "?")
        if schema == "api_request":
            detail = e.record.get("path", "")[:30]
        elif schema == "log_entry":
            detail = e.record.get("message", "")[:30]
        elif schema == "alert":
            detail = e.record.get("title", "")[:30]
        else:
            detail = schema
        print(f"  t={e.timestamp:5.1f}s sim={e.similarity:+.4f} | {detail}")

    print("\nMissed attacks (False Negatives):")
    fns = [e for e in stream if e.is_malicious and not e.flagged][:5]
    if fns:
        for e in fns:
            schema = e.record.get("_schema", "?")
            detail = str(list(e.record.keys())[:3])
            print(f"  t={e.timestamp:5.1f}s sim={e.similarity:+.4f} | {schema}: {detail}")
    else:
        print("  None! All attacks detected.")

    print("\nFalse alarms (False Positives):")
    fps = [e for e in stream if not e.is_malicious and e.flagged][:5]
    if fps:
        for e in fps:
            print(f"  t={e.timestamp:5.1f}s sim={e.similarity:+.4f} | {e.schema}")
    else:
        print("  None! No false alarms.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Stream Processing:
  Duration: {stream_duration:.0f}s
  Events: {len(stream)}
  Throughput: {len(stream)/process_time:.0f} events/sec
  Avg Latency: {np.mean([e.latency_ms for e in stream]):.3f} ms

Detection Performance:
  F1 Score: {f1:.3f}
  Precision: {precision:.1%}
  Recall: {recall:.1%}

Attack Windows Handled:
  3 attack bursts at t=10-15s, 30-35s, 50-55s
  Total malicious events: {total_malicious}
  Detected: {tp} ({100*tp/total_malicious:.1f}%)

The streaming detector successfully processes real-time events
using the Holon accumulator primitive for anomaly detection.
""")


if __name__ == "__main__":
    main()
