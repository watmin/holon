#!/usr/bin/env python3
"""
Challenge 011-003: N-gram Text Encoding for HTTP

Use n-gram (character or token level) encoding for HTTP text fields
to capture sequence patterns that structural fingerprints miss.

Key insight: Attack payloads have distinctive character sequences:
- SQL: ' OR ', '--', '; DROP'
- XSS: '<script', 'javascript:', 'onerror='
- Command: '; cat', '| ls', '`whoami`'

N-gram encoding captures these patterns even when the structural
fingerprint (bitmask, length) looks similar to normal.

Approaches:
1. Character n-grams: sliding window of N chars
2. Token n-grams: sliding window of N tokens (split on special chars)
3. Positional encoding: n-gram position matters

Comparison to batch 010's bitmask approach.
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from urllib.parse import unquote

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.9995
WARMUP_REQUESTS = 500


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# N-GRAM EXTRACTION
# =============================================================================

def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character-level n-grams from text."""
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def extract_token_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Extract token-level n-grams.

    Tokens are split on special characters to capture meaningful units.
    """
    import re
    # Split on special chars but keep them as tokens
    tokens = re.findall(r"[a-zA-Z0-9]+|[^\s\w]", text)

    if len(tokens) < n:
        return ["_".join(tokens)] if tokens else []

    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def extract_suspicious_patterns(text: str) -> Set[str]:
    """
    Extract known suspicious patterns.

    These are common attack signatures that n-grams should capture.
    """
    patterns = set()
    text_lower = text.lower()

    # SQL patterns
    if "' or " in text_lower or "'or " in text_lower:
        patterns.add("sqli_or")
    if "--" in text:
        patterns.add("sqli_comment")
    if "union" in text_lower and "select" in text_lower:
        patterns.add("sqli_union")
    if "drop" in text_lower or "delete" in text_lower:
        patterns.add("sqli_destructive")

    # XSS patterns
    if "<script" in text_lower:
        patterns.add("xss_script")
    if "javascript:" in text_lower:
        patterns.add("xss_javascript")
    if "onerror" in text_lower or "onload" in text_lower:
        patterns.add("xss_event")

    # Command injection
    if "; " in text and any(c in text for c in "cat|ls|whoami|rm"):
        patterns.add("cmd_injection")
    if "`" in text or "$(" in text:
        patterns.add("cmd_subst")

    # Path traversal
    if ".." in text:
        patterns.add("traversal")

    return patterns


# =============================================================================
# N-GRAM ENCODER
# =============================================================================

class NgramEncoder:
    """
    Encode text using n-gram based VSA vectors.

    Instead of encoding the whole string as one atom, encode each n-gram
    and bundle them together. This captures sequence patterns.
    """

    def __init__(
        self,
        vm: DeterministicVectorManager,
        char_n: int = 3,
        token_n: int = 2,
        use_positions: bool = False,
    ):
        self.vm = vm
        self.char_n = char_n
        self.token_n = token_n
        self.use_positions = use_positions

        # Track seen n-grams for analysis
        self.seen_char_ngrams: Set[str] = set()
        self.seen_token_ngrams: Set[str] = set()

    def encode_char_ngrams(self, text: str) -> np.ndarray:
        """Encode text as bundle of character n-grams."""
        ngrams = extract_char_ngrams(text, self.char_n)
        if not ngrams:
            return np.zeros(self.vm.dimensions, dtype=np.int8)

        vectors = []
        for i, ng in enumerate(ngrams):
            self.seen_char_ngrams.add(ng)
            vec = self.vm.get_vector(f"char_{ng}")

            if self.use_positions:
                # Bind with position vector
                pos_vec = self.vm.get_position_vector(i)
                vec = np.sign(vec.astype(np.float32) * pos_vec.astype(np.float32)).astype(np.int8)

            vectors.append(vec)

        # Bundle: majority vote per dimension
        stacked = np.stack(vectors)
        bundled = np.sign(np.sum(stacked.astype(np.float32), axis=0)).astype(np.int8)
        return bundled

    def encode_token_ngrams(self, text: str) -> np.ndarray:
        """Encode text as bundle of token n-grams."""
        ngrams = extract_token_ngrams(text, self.token_n)
        if not ngrams:
            return np.zeros(self.vm.dimensions, dtype=np.int8)

        vectors = []
        for i, ng in enumerate(ngrams):
            self.seen_token_ngrams.add(ng)
            vec = self.vm.get_vector(f"token_{ng}")

            if self.use_positions:
                pos_vec = self.vm.get_position_vector(i)
                vec = np.sign(vec.astype(np.float32) * pos_vec.astype(np.float32)).astype(np.int8)

            vectors.append(vec)

        stacked = np.stack(vectors)
        bundled = np.sign(np.sum(stacked.astype(np.float32), axis=0)).astype(np.int8)
        return bundled

    def encode_combined(self, text: str) -> np.ndarray:
        """Combine character and token n-gram encodings."""
        char_vec = self.encode_char_ngrams(text)
        token_vec = self.encode_token_ngrams(text)

        # Bundle both
        combined = np.sign(char_vec.astype(np.float32) + token_vec.astype(np.float32)).astype(np.int8)
        return combined


# =============================================================================
# DECAYING ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float = DECAY_FACTOR):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0):
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


# =============================================================================
# N-GRAM DETECTOR
# =============================================================================

@dataclass
class NgramDetectionResult:
    """Detection result using n-gram encoding."""
    text: str
    is_flagged: bool
    char_similarity: float
    token_similarity: float
    combined_similarity: float
    suspicious_patterns: Set[str]
    is_warmup: bool = False


class NgramHttpDetector:
    """
    HTTP anomaly detector using n-gram encoding.

    Encodes URL/body text as n-grams to capture attack patterns.
    """

    def __init__(
        self,
        char_n: int = 3,
        token_n: int = 2,
        decay: float = DECAY_FACTOR,
        threshold: float = 0.50,
        warmup: int = WARMUP_REQUESTS,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.ngram_encoder = NgramEncoder(self.vm, char_n, token_n)

        self.char_accumulator = DecayingAccumulator(DIMENSIONS, decay)
        self.token_accumulator = DecayingAccumulator(DIMENSIONS, decay)
        self.combined_accumulator = DecayingAccumulator(DIMENSIONS, decay)

        self.threshold = threshold
        self.warmup = warmup
        self.requests_seen = 0

    def process(self, text: str) -> NgramDetectionResult:
        """Process text and detect anomalies via n-gram analysis."""
        self.requests_seen += 1
        is_warmup = self.requests_seen <= self.warmup

        # Encode with n-grams
        char_vec = self.ngram_encoder.encode_char_ngrams(text)
        token_vec = self.ngram_encoder.encode_token_ngrams(text)
        combined_vec = self.ngram_encoder.encode_combined(text)

        # Get similarities
        char_sim = cosine_similarity(char_vec, self.char_accumulator.get_normalized()) if self.requests_seen > 1 else 1.0
        token_sim = cosine_similarity(token_vec, self.token_accumulator.get_normalized()) if self.requests_seen > 1 else 1.0
        combined_sim = cosine_similarity(combined_vec, self.combined_accumulator.get_normalized()) if self.requests_seen > 1 else 1.0

        # Extract suspicious patterns for analysis
        suspicious = extract_suspicious_patterns(text)

        # Detection: use combined similarity
        is_flagged = combined_sim < self.threshold if not is_warmup else False

        # Update accumulators
        weight = 0.1 if is_flagged else 1.0
        self.char_accumulator.update(char_vec, weight)
        self.token_accumulator.update(token_vec, weight)
        self.combined_accumulator.update(combined_vec, weight)

        return NgramDetectionResult(
            text=text[:50] + "..." if len(text) > 50 else text,
            is_flagged=is_flagged,
            char_similarity=char_sim,
            token_similarity=token_sim,
            combined_similarity=combined_sim,
            suspicious_patterns=suspicious,
            is_warmup=is_warmup,
        )


# =============================================================================
# REQUEST GENERATOR
# =============================================================================

class RequestGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        self.normal_urls = [
            "/api/users/123",
            "/api/orders/456",
            "/api/products/789",
            "/api/search?q=laptop",
            "/api/search?q=phone",
            "/api/auth/login",
            "/api/cart/items",
            "/api/checkout",
        ]

        self.normal_bodies = [
            '{"name": "john", "email": "john@example.com"}',
            '{"product": "laptop", "qty": 1}',
            '{"search": "electronics"}',
            '{"limit": 10, "offset": 0}',
        ]

    def generate_benign(self) -> Tuple[str, Optional[str]]:
        """Returns (text, None) for benign."""
        if self.rng.random() < 0.7:
            return self.rng.choice(self.normal_urls), None
        else:
            return self.rng.choice(self.normal_bodies), None

    def generate_malicious(self) -> Tuple[str, str]:
        """Returns (text, attack_type)."""
        attack = self.rng.choice([
            "sqli_simple", "sqli_union", "sqli_comment",
            "xss_script", "xss_event", "xss_javascript",
            "cmd_injection", "traversal",
        ])

        if attack == "sqli_simple":
            payloads = ["' OR '1'='1", "' OR 1=1--", "admin' OR '1'='1"]
            return f"/api/users/{self.rng.choice(payloads)}", attack

        elif attack == "sqli_union":
            return "/api/users/1 UNION SELECT username,password FROM users--", attack

        elif attack == "sqli_comment":
            return "/api/auth/login?user=admin'--&pass=x", attack

        elif attack == "xss_script":
            return "/api/search?q=<script>alert(1)</script>", attack

        elif attack == "xss_event":
            return "/api/search?q=<img src=x onerror=alert(1)>", attack

        elif attack == "xss_javascript":
            return "/api/redirect?url=javascript:alert(document.cookie)", attack

        elif attack == "cmd_injection":
            payloads = ["; cat /etc/passwd", "| ls -la", "`whoami`", "$(id)"]
            return f"/api/exec?cmd=test{self.rng.choice(payloads)}", attack

        else:  # traversal
            return "/api/files/../../../etc/passwd", attack

    def generate_stream(self, n: int, malicious_ratio: float = 0.02):
        stream = []
        for _ in range(n):
            if self.rng.random() < malicious_ratio:
                text, attack = self.generate_malicious()
                stream.append((text, attack))
            else:
                text, _ = self.generate_benign()
                stream.append((text, None))
        return stream


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 011-003: N-gram Text Encoding")
    print("=" * 80)
    print("""
N-gram encoding captures sequence patterns in text:

Character n-grams (n=3):
  "admin'--" → ["adm", "dmi", "min", "in'", "n'-", "'-", "--"]

Token n-grams (n=2):
  "admin'--" → ["admin_'", "'_--"]

Attack signatures become distinct n-grams:
  SQL: "' O", " OR", "R '", etc.
  XSS: "<sc", "scr", "rip", "ipt", etc.

Hypothesis: N-grams capture attack semantics better than structural fingerprints.
""")

    generator = RequestGenerator(seed=42)
    stream = generator.generate_stream(10000, malicious_ratio=0.03)

    actual_malicious = sum(1 for _, a in stream if a is not None)
    print(f"Stream: {len(stream)} requests, {actual_malicious} malicious ({100*actual_malicious/len(stream):.1f}%)")

    # Test different n-gram sizes
    configs = [
        {"char_n": 2, "token_n": 2, "name": "char=2, token=2"},
        {"char_n": 3, "token_n": 2, "name": "char=3, token=2"},
        {"char_n": 4, "token_n": 3, "name": "char=4, token=3"},
    ]

    results = {}

    for config in configs:
        detector = NgramHttpDetector(
            char_n=config["char_n"],
            token_n=config["token_n"],
            threshold=0.45,
            warmup=WARMUP_REQUESTS,
        )

        detection_results = []
        start = time.time()

        for text, attack in stream:
            result = detector.process(text)
            detection_results.append((result, attack))

        elapsed = time.time() - start
        throughput = len(stream) / elapsed

        # Metrics
        post_warmup = [(r, a) for r, a in detection_results if not r.is_warmup]

        tp = sum(1 for r, a in post_warmup if a is not None and r.is_flagged)
        fp = sum(1 for r, a in post_warmup if a is None and r.is_flagged)
        fn = sum(1 for r, a in post_warmup if a is not None and not r.is_flagged)
        tn = sum(1 for r, a in post_warmup if a is None and not r.is_flagged)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        print(f"\n--- {config['name']} ---")
        print(f"  Throughput: {throughput:.0f} req/sec")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.3f}")
        print(f"  Unique char n-grams: {len(detector.ngram_encoder.seen_char_ngrams)}")
        print(f"  Unique token n-grams: {len(detector.ngram_encoder.seen_token_ngrams)}")

        # By attack type
        attack_stats = {}
        for r, a in post_warmup:
            if a:
                if a not in attack_stats:
                    attack_stats[a] = {"detected": 0, "missed": 0}
                if r.is_flagged:
                    attack_stats[a]["detected"] += 1
                else:
                    attack_stats[a]["missed"] += 1

        print(f"  By attack type:")
        for attack, stats in sorted(attack_stats.items()):
            total = stats["detected"] + stats["missed"]
            rate = stats["detected"] / total if total else 0
            print(f"    {attack}: {stats['detected']}/{total} ({rate:.0%})")

        results[config["name"]] = {"f1": f1, "precision": precision, "recall": recall}

    # Show n-gram examples
    print("\n--- N-gram Examples ---")
    detector = NgramHttpDetector(char_n=3, token_n=2)

    examples = [
        "/api/users/123",
        "/api/users/' OR '1'='1",
        "<script>alert(1)</script>",
        "; cat /etc/passwd",
    ]

    for text in examples:
        char_ngrams = extract_char_ngrams(text, 3)
        token_ngrams = extract_token_ngrams(text, 2)
        suspicious = extract_suspicious_patterns(text)

        print(f"\n  Text: {text}")
        print(f"  Char 3-grams (first 10): {char_ngrams[:10]}")
        print(f"  Token 2-grams (first 5): {token_ngrams[:5]}")
        print(f"  Suspicious patterns: {suspicious}")

    # Similarity distribution
    print("\n--- Similarity Distribution ---")
    detector = NgramHttpDetector(char_n=3, token_n=2, threshold=0.45)

    # Process stream
    for text, attack in stream:
        detector.process(text)

    # Sample final similarities
    benign_sims = []
    malicious_sims = []

    for text, attack in stream[-1000:]:
        result = detector.process(text)
        if attack:
            malicious_sims.append(result.combined_similarity)
        else:
            benign_sims.append(result.combined_similarity)

    if benign_sims and malicious_sims:
        print(f"  Benign:    min={min(benign_sims):.3f}, mean={np.mean(benign_sims):.3f}, max={max(benign_sims):.3f}")
        print(f"  Malicious: min={min(malicious_sims):.3f}, mean={np.mean(malicious_sims):.3f}, max={max(malicious_sims):.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    best = max(results.items(), key=lambda x: x[1]["f1"])
    print(f"""
Best config: {best[0]} (F1={best[1]['f1']:.3f})

Key observations:

1. N-grams capture attack signatures directly
   - "' OR " → distinct character sequence
   - "<script>" → distinct token sequence

2. Character n-grams provide fine-grained patterns
   - SQL injection: "' O", " OR", "R '"
   - XSS: "<sc", "scr", "ipt"

3. Token n-grams capture semantic structure
   - "admin_'" = username followed by quote
   - "'_--" = quote followed by comment

4. Trade-off: Larger n = more specific but higher cardinality

Comparison to structural fingerprints (batch 010):
- Fingerprints: catch abnormal char classes (bit 16)
- N-grams: catch specific attack sequences
- Best: combine both approaches
""")


if __name__ == "__main__":
    main()
