#!/usr/bin/env python3
"""
Challenge 010-031: Explainability - WHY is this request suspicious?

Given:
  - reference_vec: The learned "normal" pattern (accumulator)
  - request: The actual HTTP request data
  - request_vec: The encoded request vector

Can we deduce WHY it was flagged?

Approaches:
1. Component-level similarity - encode each field separately, find the outlier
2. Resonance analysis - which dimensions agree vs disagree
3. Difference vector - extract the "delta" from normal
4. Unbind to extract specific signals

Key insight: VSA encoding is compositional. The total vector is a superposition
of component vectors. We can isolate contributions by encoding components
separately and comparing.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import unquote

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# CHARACTER CLASS BITMASK (from 025)
# =============================================================================

def char_class_bitmask(s: str) -> int:
    mask = 0
    normal_special = set("-_./@:,= ")
    for c in s:
        if c.islower():
            mask |= 1
        elif c.isupper():
            mask |= 2
        elif c.isdigit():
            mask |= 4
        elif c in normal_special:
            mask |= 8
        else:
            mask |= 16  # Abnormal
    return mask


def describe_bitmask(mask: int) -> str:
    parts = []
    if mask & 1: parts.append("lower")
    if mask & 2: parts.append("upper")
    if mask & 4: parts.append("digit")
    if mask & 8: parts.append("normal")
    if mask & 16: parts.append("ABNORMAL")
    return "+".join(parts) if parts else "empty"


# =============================================================================
# STRUCTURAL FEATURE EXTRACTION
# =============================================================================

def bucket_length(length: int) -> int:
    if length == 0: return 0
    elif length <= 3: return 1
    elif length <= 6: return 2
    elif length <= 12: return 3
    elif length <= 25: return 4
    else: return 5


def bucket_count(count: int) -> int:
    if count == 0: return 0
    elif count <= 2: return 1
    elif count <= 5: return 2
    else: return 3


def extract_features(url: str, method: str = "GET") -> dict:
    """Extract structural features from URL."""
    url = unquote(url)

    if "?" in url:
        path_part, query_part = url.split("?", 1)
    else:
        path_part, query_part = url, ""

    path_segments = [seg for seg in path_part.split("/") if seg]

    query_pairs = []
    if query_part:
        for param in query_part.split("&"):
            if "=" in param:
                k, v = param.split("=", 1)
                query_pairs.append((k, v))
            else:
                query_pairs.append((param, ""))

    # Collect bitmasks
    path_bitmasks = [char_class_bitmask(seg) for seg in path_segments]
    query_bitmasks = [char_class_bitmask(v) for k, v in query_pairs]
    all_bitmasks = path_bitmasks + query_bitmasks

    features = {
        "method": method,
        "bitmasks": sorted(set(all_bitmasks)),
        "max_bitmask": max(all_bitmasks) if all_bitmasks else 0,
        "path_depth": bucket_count(len(path_segments)),
        "query_count": bucket_count(len(query_pairs)),
        "path_lengths": [bucket_length(len(seg)) for seg in path_segments],
        "query_lengths": [bucket_length(len(v)) for k, v in query_pairs],
    }

    # Store raw components for explainability
    features["_raw_path_segments"] = path_segments
    features["_raw_query_pairs"] = query_pairs

    return features


# =============================================================================
# DECAYING ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float = 0.9995):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)

    def update(self, vector: np.ndarray, weight: float = 1.0):
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


# =============================================================================
# EXPLAINABILITY ANALYSIS
# =============================================================================

@dataclass
class ComponentContribution:
    """Analysis of a single component's contribution to suspicion."""
    component_name: str
    component_value: Any
    similarity_to_reference: float
    is_suspicious: bool
    reason: str


@dataclass
class DimensionAnalysis:
    """Analysis of vector dimensions - where do they agree/disagree?"""
    agreeing_dims: int       # Both positive or both negative
    disagreeing_dims: int    # Opposite signs
    neutral_dims: int        # One or both zero
    strongest_agreement: List[int]    # Top dimension indices
    strongest_disagreement: List[int]  # Top dimension indices


@dataclass
class ExplainabilityResult:
    """Full explainability analysis of a suspicious request."""
    url: str
    overall_similarity: float
    is_flagged: bool
    threshold: float
    boosted_threshold: float  # Adjusted threshold based on component analysis
    component_analysis: List[ComponentContribution]
    resonance_ratio: float  # What fraction of dimensions agree
    dimension_analysis: DimensionAnalysis
    top_suspicious_components: List[str]
    explanation: str


class ExplainableDetector:
    """
    Anomaly detector with explainability.

    Can answer: WHY is this request suspicious?

    Features:
    1. Component-level analysis - which parts deviate
    2. Component-based boosting - suspicious components lower the threshold
    3. Resonance/difference analysis - which dimensions agree/disagree
    """

    def __init__(
        self,
        global_seed: int = GLOBAL_SEED,
        decay: float = 0.9995,
        threshold: float = 0.60,
        boost_factor: float = 0.15,  # How much to lower threshold per suspicious component
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)
        self.accumulator = DecayingAccumulator(DIMENSIONS, decay)
        self.threshold = threshold
        self.boost_factor = boost_factor
        self.seen = 0

        # Track what we've learned (for explainability)
        self.learned_methods = set()
        self.learned_bitmasks = set()
        self.learned_path_depths = set()

    def train(self, urls: List[str], method: str = "GET"):
        """Train on benign examples."""
        for url in urls:
            features = extract_features(url, method)
            vec = self.encoder.encode_data(self._encodable_features(features))
            self.accumulator.update(vec)
            self.seen += 1

            # Track learned patterns
            self.learned_methods.add(features["method"])
            self.learned_bitmasks.update(features["bitmasks"])
            self.learned_path_depths.add(features["path_depth"])

    def _encodable_features(self, features: dict) -> dict:
        """Remove raw components, keep only encodable features."""
        return {k: v for k, v in features.items() if not k.startswith("_")}

    def analyze(self, url: str, method: str = "GET") -> ExplainabilityResult:
        """
        Analyze a request and explain WHY it's suspicious.

        Returns detailed breakdown of:
        - Overall similarity
        - Each component's contribution
        - Resonance analysis
        - Human-readable explanation
        """
        features = extract_features(url, method)
        full_vec = self.encoder.encode_data(self._encodable_features(features))
        reference = self.accumulator.get_normalized()

        overall_sim = cosine_similarity(full_vec, reference)
        is_flagged = overall_sim < self.threshold

        # Component-level analysis
        component_analysis = []

        # 1. Method contribution
        method_features = {"method": features["method"]}
        method_vec = self.encoder.encode_data(method_features)
        method_sim = cosine_similarity(method_vec, reference)
        is_method_suspicious = features["method"] not in self.learned_methods
        component_analysis.append(ComponentContribution(
            component_name="method",
            component_value=features["method"],
            similarity_to_reference=method_sim,
            is_suspicious=is_method_suspicious,
            reason=f"Method '{features['method']}' {'not seen in training' if is_method_suspicious else 'is common'}"
        ))

        # 2. Bitmask contribution (KEY for attack detection)
        novel_bitmasks = set(features["bitmasks"]) - self.learned_bitmasks
        has_abnormal = any(m & 16 for m in features["bitmasks"])
        bitmask_features = {"bitmasks": features["bitmasks"], "max_bitmask": features["max_bitmask"]}
        bitmask_vec = self.encoder.encode_data(bitmask_features)
        bitmask_sim = cosine_similarity(bitmask_vec, reference)
        is_bitmask_suspicious = bool(novel_bitmasks) or has_abnormal

        if has_abnormal:
            bitmask_reason = f"Contains ABNORMAL chars (bitmask {features['max_bitmask']})"
        elif novel_bitmasks:
            bitmask_reason = f"Novel bitmasks: {novel_bitmasks}"
        else:
            bitmask_reason = "All bitmasks seen in training"

        component_analysis.append(ComponentContribution(
            component_name="bitmasks",
            component_value=features["bitmasks"],
            similarity_to_reference=bitmask_sim,
            is_suspicious=is_bitmask_suspicious,
            reason=bitmask_reason
        ))

        # 3. Path depth contribution
        depth_features = {"path_depth": features["path_depth"]}
        depth_vec = self.encoder.encode_data(depth_features)
        depth_sim = cosine_similarity(depth_vec, reference)
        is_depth_suspicious = features["path_depth"] not in self.learned_path_depths
        component_analysis.append(ComponentContribution(
            component_name="path_depth",
            component_value=features["path_depth"],
            similarity_to_reference=depth_sim,
            is_suspicious=is_depth_suspicious,
            reason=f"Path depth {features['path_depth']} {'unusual' if is_depth_suspicious else 'normal'}"
        ))

        # 4. Path lengths contribution
        lengths_features = {"path_lengths": features["path_lengths"]}
        lengths_vec = self.encoder.encode_data(lengths_features)
        lengths_sim = cosine_similarity(lengths_vec, reference)
        has_huge_segment = any(l >= 4 for l in features["path_lengths"])
        component_analysis.append(ComponentContribution(
            component_name="path_lengths",
            component_value=features["path_lengths"],
            similarity_to_reference=lengths_sim,
            is_suspicious=has_huge_segment,
            reason=f"Has {'unusually long' if has_huge_segment else 'normal'} path segments"
        ))

        # 5. Query contribution
        query_features = {"query_count": features["query_count"], "query_lengths": features["query_lengths"]}
        query_vec = self.encoder.encode_data(query_features)
        query_sim = cosine_similarity(query_vec, reference)
        has_huge_query = any(l >= 4 for l in features.get("query_lengths", []))
        component_analysis.append(ComponentContribution(
            component_name="query",
            component_value=query_features,
            similarity_to_reference=query_sim,
            is_suspicious=has_huge_query,
            reason=f"Has {'unusually long' if has_huge_query else 'normal'} query values"
        ))

        # Resonance analysis - where do vectors agree?
        resonance = self.encoder.resonance(full_vec, reference)
        resonance_nonzero = np.count_nonzero(resonance)
        resonance_ratio = resonance_nonzero / DIMENSIONS

        # Difference analysis - what's the delta from normal?
        difference = self.encoder.difference(reference, full_vec)

        # Dimension-level analysis
        dimension_analysis = self._analyze_dimensions(full_vec, reference, difference)

        # Find suspicious components
        suspicious = [c for c in component_analysis if c.is_suspicious]
        top_suspicious = [c.component_name for c in sorted(
            suspicious,
            key=lambda c: c.similarity_to_reference
        )]

        # COMPONENT-BASED BOOSTING
        # Each suspicious component lowers the effective threshold
        n_suspicious = len(suspicious)
        boosted_threshold = self.threshold + (n_suspicious * self.boost_factor)
        boosted_threshold = min(boosted_threshold, 0.95)  # Cap at 0.95

        # Decision with boosted threshold
        is_flagged = overall_sim < boosted_threshold

        # Generate human-readable explanation
        if not is_flagged:
            explanation = f"Request appears NORMAL (similarity {overall_sim:.3f} >= boosted threshold {boosted_threshold:.3f})"
        else:
            reasons = []
            for c in suspicious:
                reasons.append(f"  - {c.component_name}: {c.reason}")

            if reasons:
                explanation = f"Request is SUSPICIOUS (similarity {overall_sim:.3f} < boosted threshold {boosted_threshold:.3f})\n"
                explanation += f"Base threshold: {self.threshold}, boosted by {n_suspicious} suspicious component(s)\n"
                explanation += "Reasons:\n" + "\n".join(reasons)
            else:
                explanation = f"Request is SUSPICIOUS (similarity {overall_sim:.3f} < threshold {self.threshold})\n"
                explanation += "Reason: Overall pattern doesn't match learned normal traffic"

        return ExplainabilityResult(
            url=url,
            overall_similarity=overall_sim,
            is_flagged=is_flagged,
            threshold=self.threshold,
            boosted_threshold=boosted_threshold,
            component_analysis=component_analysis,
            resonance_ratio=resonance_ratio,
            dimension_analysis=dimension_analysis,
            top_suspicious_components=top_suspicious,
            explanation=explanation,
        )

    def _analyze_dimensions(
        self,
        request_vec: np.ndarray,
        reference: np.ndarray,
        difference: np.ndarray
    ) -> DimensionAnalysis:
        """
        Analyze which dimensions agree, disagree, or are neutral.

        This tells us WHERE in the vector space the request deviates.
        """
        req = request_vec.astype(np.float64)
        ref = reference.astype(np.float64)

        # Count agreeing/disagreeing dimensions
        agreeing = 0
        disagreeing = 0
        neutral = 0

        agreement_strength = []
        disagreement_strength = []

        for i in range(len(req)):
            if req[i] == 0 or ref[i] == 0:
                neutral += 1
            elif (req[i] > 0 and ref[i] > 0) or (req[i] < 0 and ref[i] < 0):
                agreeing += 1
                agreement_strength.append((i, abs(req[i] * ref[i])))
            else:
                disagreeing += 1
                disagreement_strength.append((i, abs(req[i] * ref[i])))

        # Find strongest agreement/disagreement dimensions
        agreement_strength.sort(key=lambda x: -x[1])
        disagreement_strength.sort(key=lambda x: -x[1])

        return DimensionAnalysis(
            agreeing_dims=agreeing,
            disagreeing_dims=disagreeing,
            neutral_dims=neutral,
            strongest_agreement=[idx for idx, _ in agreement_strength[:10]],
            strongest_disagreement=[idx for idx, _ in disagreement_strength[:10]],
        )


# =============================================================================
# DEMO
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 010-031: Explainability - WHY is this suspicious?")
    print("=" * 80)
    print("""
Given:
  - Reference vector (learned from normal traffic)
  - Suspicious request

Can we deduce WHY it was flagged?

Approach:
  1. Component-level similarity - encode each feature separately
  2. Compare each component to the reference
  3. Identify which components deviate from learned patterns
  4. Generate human-readable explanation
""")

    # Create detector and train on normal traffic
    detector = ExplainableDetector(threshold=0.55)

    print("--- Training on Normal Traffic ---")
    normal_urls = [
        "/api/users",
        "/api/users/123",
        "/api/users/456",
        "/api/orders",
        "/api/orders/789",
        "/api/products",
        "/api/products/101",
        "/api/search?q=laptop",
        "/api/search?q=phone",
        "/api/search?q=tablet",
        "/api/auth/login",
        "/api/auth/logout",
    ] * 100  # Repeat to build strong signal

    detector.train(normal_urls)
    print(f"  Trained on {len(normal_urls)} requests")
    print(f"  Learned methods: {detector.learned_methods}")
    print(f"  Learned bitmasks: {sorted(detector.learned_bitmasks)}")
    print(f"  Learned path depths: {detector.learned_path_depths}")

    # Test cases
    print("\n" + "=" * 80)
    print("ANALYSIS OF SUSPICIOUS REQUESTS")
    print("=" * 80)

    test_cases = [
        # Normal
        ("/api/users/123", "Should be NORMAL - typical API call"),

        # SQL injection in path
        ("/api/users/' OR '1'='1", "SQL injection in path"),

        # SQL injection in query
        ("/api/search?q=' OR '1'='1", "SQL injection in query param"),

        # XSS
        ("/api/search?q=<script>alert(1)</script>", "XSS attack"),

        # Path traversal
        ("/api/files/../../../etc/passwd", "Path traversal"),

        # Hidden file access
        ("/.git/config", "Hidden file access"),

        # Command injection
        ("/api/exec?cmd=; cat /etc/passwd", "Command injection"),

        # Normal but unusual method
        ("/api/users/123", "Normal URL, unusual method"),
    ]

    for i, (url, description) in enumerate(test_cases):
        method = "DEBUG" if i == len(test_cases) - 1 else "GET"

        print(f"\n{'─' * 80}")
        print(f"Test {i+1}: {description}")
        print(f"  URL: {url}")
        print(f"  Method: {method}")
        print(f"{'─' * 80}")

        result = detector.analyze(url, method)

        # Show overall result
        status = "🚨 FLAGGED" if result.is_flagged else "✅ ALLOWED"
        print(f"\n  {status}")
        print(f"  Overall similarity: {result.overall_similarity:.3f}")
        print(f"  Base threshold: {result.threshold:.3f}, Boosted threshold: {result.boosted_threshold:.3f}")

        # Show component breakdown
        print(f"\n  Component Analysis:")
        for c in result.component_analysis:
            flag = "❌" if c.is_suspicious else "✓"
            print(f"    {flag} {c.component_name}: sim={c.similarity_to_reference:.3f}")
            print(f"        value: {c.component_value}")
            print(f"        {c.reason}")

        # Show dimension analysis
        da = result.dimension_analysis
        print(f"\n  Dimension Analysis:")
        print(f"    Agreeing:    {da.agreeing_dims:4d} dims ({100*da.agreeing_dims/DIMENSIONS:.1f}%)")
        print(f"    Disagreeing: {da.disagreeing_dims:4d} dims ({100*da.disagreeing_dims/DIMENSIONS:.1f}%)")
        print(f"    Neutral:     {da.neutral_dims:4d} dims ({100*da.neutral_dims/DIMENSIONS:.1f}%)")
        if da.strongest_disagreement:
            print(f"    Top disagreeing dims: {da.strongest_disagreement[:5]}")

        # Show explanation
        print(f"\n  Explanation:")
        for line in result.explanation.split("\n"):
            print(f"    {line}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Explainability + Boosting")
    print("=" * 80)
    print("""
THREE KEY FEATURES:

1. COMPONENT-LEVEL ANALYSIS
   - Encode each feature separately (method, bitmasks, path_depth, etc.)
   - Compare each to the reference vector
   - Identify which specific components deviate

2. COMPONENT-BASED BOOSTING
   - Each suspicious component RAISES the effective threshold
   - Base threshold: 0.55
   - With 1 suspicious component: 0.55 + 0.15 = 0.70
   - With 2 suspicious components: 0.55 + 0.30 = 0.85

   Effect: Requests that are "borderline" on similarity BUT have
   suspicious components (like ABNORMAL bitmasks) get flagged.

   Before boosting: SQL injection in path ALLOWED (sim=0.614 >= 0.55)
   After boosting:  SQL injection in path FLAGGED (sim=0.614 < 0.70)

3. DIMENSION ANALYSIS
   - Resonance: which dimensions AGREE (both positive or both negative)
   - Disagreement: which dimensions have OPPOSITE signs
   - Neutral: where one or both vectors are zero

   Dimension-level analysis tells us WHERE in the vector space
   the request diverges from normal. High disagreement = anomaly.

WHAT WE CAN DEDUCE FROM (reference_vec, request, request_vec):

  ✓ Which CHARACTER CLASSES are unusual (bitmask bit 16 = ABNORMAL)
  ✓ Which STRUCTURAL features deviate (path depth, lengths)
  ✓ Whether the METHOD is unexpected
  ✓ How many dimensions agree vs disagree
  ✓ Which specific dimensions disagree most strongly

  ✗ CANNOT see specific payload content (we encode structure)
  ✗ CANNOT explain attack semantics (requires domain knowledge)

BOOSTING FORMULA:
  boosted_threshold = base_threshold + (n_suspicious * boost_factor)

  Where:
    base_threshold = 0.55 (tunable)
    boost_factor = 0.15 per suspicious component
    n_suspicious = count of components flagged as suspicious

PRACTICAL VALUE:
  - Analyst sees: "ABNORMAL chars in bitmask" → check for SQL/XSS
  - Analyst sees: "Path depth unusual" → check for traversal
  - Analyst sees: "Method not in training" → check for DEBUG/TRACE
  - Dimension analysis: "1200 disagreeing dims" → significant anomaly
""")


if __name__ == "__main__":
    main()
