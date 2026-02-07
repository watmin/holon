#!/usr/bin/env python3
"""
HolonClient Demo: The Primary Interface for Holon

This demonstrates HolonClient as the single, clean interface for all Holon
operations. This API is designed for easy portability to Rust:

    struct Holon {
        fn encode(&self, data: Value) -> Vector { ... }
        fn bind(&self, a: &Vector, b: &Vector) -> Vector { ... }
        fn bundle(&self, vecs: &[Vector]) -> Vector { ... }
        fn accumulate(&self, acc: &mut Accumulator, vec: &Vector) { ... }
        ...
    }

Run: ./scripts/run_with_venv.sh python examples/holon_client_demo.py
"""

from holon import HolonClient, create_client


def demo_basic_encoding():
    """Basic data encoding."""
    print("=" * 60)
    print("DEMO 1: Basic Data Encoding")
    print("=" * 60)

    # Create standalone client (simplest way)
    client = HolonClient()
    # Or: client = create_client()

    # Encode structured data
    billing = client.encode({"type": "billing", "amount": 100})
    technical = client.encode({"type": "technical", "issue": "bug"})

    print(f"Vector dimensions: {client.dimensions}")
    print(f"Billing vector shape: {billing.shape}, dtype: {billing.dtype}")
    print(f"Technical vector shape: {technical.shape}")

    # Similarity between different concepts
    sim = client.similarity(billing, technical)
    print(f"Similarity (billing vs technical): {sim:.3f}")

    # Same data = same vector (deterministic)
    billing2 = client.encode({"type": "billing", "amount": 100})
    sim_self = client.similarity(billing, billing2)
    print(f"Similarity (billing vs billing): {sim_self:.3f}")


def demo_vsa_primitives():
    """VSA/HDC core operations."""
    print("\n" + "=" * 60)
    print("DEMO 2: VSA Primitives (The Core Operations)")
    print("=" * 60)

    client = HolonClient()

    # Get base vectors for atoms
    A = client.get_vector("concept_A")
    B = client.get_vector("concept_B")
    C = client.get_vector("concept_C")

    print("\n--- BIND (AND-like association) ---")
    AB = client.bind(A, B)  # "A associated with B"
    print(f"bind(A, B) similarity to A: {client.similarity(AB, A):.3f}")
    print(f"bind(A, B) similarity to B: {client.similarity(AB, B):.3f}")
    print("  → Binding creates something dissimilar to both inputs")

    # Unbinding to recover
    B_recovered = client.unbind(AB, A)
    print(f"unbind(AB, A) similarity to B: {client.similarity(B_recovered, B):.3f}")
    print("  → Unbinding recovers the other component")

    print("\n--- BUNDLE (OR-like superposition) ---")
    ABC = client.bundle([A, B, C])
    print(f"bundle([A,B,C]) similarity to A: {client.similarity(ABC, A):.3f}")
    print(f"bundle([A,B,C]) similarity to B: {client.similarity(ABC, B):.3f}")
    print(f"bundle([A,B,C]) similarity to C: {client.similarity(ABC, C):.3f}")
    print("  → Bundle is similar to ALL inputs")

    print("\n--- NEGATE (NOT operation) ---")
    AC = client.negate(ABC, B)
    print(f"After negating B, similarity to B: {client.similarity(AC, B):.3f}")
    print(f"After negating B, similarity to A: {client.similarity(AC, A):.3f}")
    print("  → Negate removes component's influence")

    print("\n--- AMPLIFY (Strengthen component) ---")
    boosted = client.amplify(ABC, B, strength=2.0)
    print(f"Original similarity to B: {client.similarity(ABC, B):.3f}")
    print(f"After amplify(2.0) similarity to B: {client.similarity(boosted, B):.3f}")
    print("  → Amplify strengthens component's presence")

    print("\n--- PROTOTYPE (Extract common pattern) ---")
    # Create vectors with shared component
    v1 = client.bundle([A, client.get_vector("unique1")])
    v2 = client.bundle([A, client.get_vector("unique2")])
    v3 = client.bundle([A, client.get_vector("unique3")])
    proto = client.prototype([v1, v2, v3])
    print(
        f"Prototype similarity to shared component A: {client.similarity(proto, A):.3f}"
    )
    print("  → Prototype captures what's common")

    print("\n--- DIFFERENCE (What changed?) ---")
    before = client.bundle([A, B])
    after = client.bundle([A, B, C])
    delta = client.difference(before, after)
    print(f"Difference similarity to C (added): {client.similarity(delta, C):.3f}")
    print("  → Difference highlights what's new")


def demo_accumulators():
    """Streaming operations with accumulators."""
    print("\n" + "=" * 60)
    print("DEMO 3: Accumulators (Streaming / Continuous Learning)")
    print("=" * 60)

    client = HolonClient()

    # Create accumulator for learning patterns
    accum = client.create_accumulator()
    print(f"New accumulator: shape={accum.shape}, dtype={accum.dtype}")

    # Simulate streaming - 90 "normal" events, 10 "rare" events
    normal = client.encode({"type": "normal", "status": "ok"})
    rare = client.encode({"type": "rare", "status": "anomaly"})

    print("\nStreaming 90 normal + 10 rare events...")
    for _ in range(90):
        accum = client.accumulate(accum, normal)
    for _ in range(10):
        accum = client.accumulate(accum, rare)

    # Normalize for similarity queries
    baseline = client.normalize_accumulator(accum)
    print(f"Normalized baseline: dtype={baseline.dtype}")

    # Check similarity - frequent patterns should have higher similarity
    sim_normal = client.similarity(normal, baseline)
    sim_rare = client.similarity(rare, baseline)

    print("\nSimilarity to baseline:")
    print(f"  Normal (90 occurrences): {sim_normal:.3f}")
    print(f"  Rare (10 occurrences):   {sim_rare:.3f}")
    print("  → Frequency is preserved in accumulator")

    # New unknown pattern
    unknown = client.encode({"type": "attack", "payload": "malicious"})
    sim_unknown = client.similarity(unknown, baseline)
    print(f"  Unknown pattern:         {sim_unknown:.3f}")
    print("  → Unknown patterns have low similarity (anomaly!)")


def demo_continuous_encoding():
    """Continuous scalar encoding."""
    print("\n" + "=" * 60)
    print("DEMO 4: Continuous Scalar Encoding")
    print("=" * 60)

    client = HolonClient()

    print("\n--- LINEAR ENCODING ---")
    v100 = client.encode_scalar(100)
    v110 = client.encode_scalar(110)
    v200 = client.encode_scalar(200)
    v1000 = client.encode_scalar(1000)

    print("Similarity matrix for linear values:")
    print(f"  100 vs 110:  {client.similarity(v100, v110):.3f} (close)")
    print(f"  100 vs 200:  {client.similarity(v100, v200):.3f} (moderate)")
    print(f"  100 vs 1000: {client.similarity(v100, v1000):.3f} (far)")
    print("  → Nearby values have similar vectors")

    print("\n--- LOG-SCALE ENCODING ---")
    v100_log = client.encode_scalar_log(100)
    v1000_log = client.encode_scalar_log(1000)
    v10000_log = client.encode_scalar_log(10000)

    sim_100_1000 = client.similarity(v100_log, v1000_log)
    sim_1000_10000 = client.similarity(v1000_log, v10000_log)

    print("Log-scale similarity (equal ratios = equal similarity):")
    print(f"  100 vs 1000:   {sim_100_1000:.3f} (10x ratio)")
    print(f"  1000 vs 10000: {sim_1000_10000:.3f} (10x ratio)")
    print("  → Equal ratios have approximately equal similarity drops")

    print("\n--- CIRCULAR ENCODING ---")
    h0 = client.encode_scalar(0, mode="circular", period=24)
    h6 = client.encode_scalar(6, mode="circular", period=24)
    h12 = client.encode_scalar(12, mode="circular", period=24)
    h23 = client.encode_scalar(23, mode="circular", period=24)

    print("Hour of day (24-hour cycle):")
    print(f"  0:00 vs 6:00:  {client.similarity(h0, h6):.3f}")
    print(f"  0:00 vs 12:00: {client.similarity(h0, h12):.3f}")
    print(f"  0:00 vs 23:00: {client.similarity(h0, h23):.3f}")
    print("  → Circular encoding wraps (23:00 close to 0:00)")


def demo_sequence_encoding():
    """Sequence encoding with different modes."""
    print("\n" + "=" * 60)
    print("DEMO 5: Sequence Encoding")
    print("=" * 60)

    client = HolonClient()

    items = ["login", "view", "purchase"]
    items_reversed = ["purchase", "view", "login"]

    print("\n--- POSITIONAL MODE (order matters) ---")
    seq1 = client.encode_sequence(items, mode="positional")
    seq2 = client.encode_sequence(items_reversed, mode="positional")
    sim_pos = client.similarity(seq1, seq2)
    print(f"  [login, view, purchase] vs reversed: {sim_pos:.3f}")
    print("  → Different orders produce different vectors")

    print("\n--- BUNDLE MODE (bag-of-words, order ignored) ---")
    seq1_bundle = client.encode_sequence(items, mode="bundle")
    seq2_bundle = client.encode_sequence(items_reversed, mode="bundle")
    sim_bundle = client.similarity(seq1_bundle, seq2_bundle)
    print(f"  [login, view, purchase] vs reversed: {sim_bundle:.3f}")
    print("  → Same elements = same vector (order ignored)")

    print("\n--- NGRAM MODE (local patterns) ---")
    text1 = ["quick", "brown", "fox"]
    text2 = ["quick", "brown", "dog"]
    ngram1 = client.encode_sequence(text1, mode="ngram")
    ngram2 = client.encode_sequence(text2, mode="ngram")
    sim_ngram = client.similarity(ngram1, ngram2)
    print(f"  'quick brown fox' vs 'quick brown dog': {sim_ngram:.3f}")
    print("  → Partial overlap in n-grams")


def demo_anomaly_detection():
    """Complete anomaly detection workflow."""
    print("\n" + "=" * 60)
    print("DEMO 6: Anomaly Detection Workflow")
    print("=" * 60)

    client = HolonClient()

    print("\n1. Learning phase (build baseline from normal traffic)...")
    pattern_accum = client.create_accumulator()
    rate_accum = client.create_accumulator()

    # Simulate normal traffic
    for i in range(100):
        # Pattern: TCP to port 80/443
        pattern = client.encode(
            {
                "protocol": "TCP",
                "dst_port": 80 if i % 2 == 0 else 443,
                "flags": "PA" if i % 3 != 0 else "A",
            }
        )
        # Rate: ~100 pps
        rate = client.encode_scalar_log(100 + (i % 20))

        pattern_accum = client.accumulate(pattern_accum, pattern)
        rate_accum = client.accumulate(rate_accum, rate)

    pattern_baseline = client.normalize_accumulator(pattern_accum)
    rate_baseline = client.normalize_accumulator(rate_accum)
    print("   Baselines learned from 100 normal packets")

    print("\n2. Detection phase...")

    # Test normal packet
    normal_pattern = client.encode({"protocol": "TCP", "dst_port": 80, "flags": "PA"})
    normal_rate = client.encode_scalar_log(105)

    normal_pattern_sim = client.similarity(normal_pattern, pattern_baseline)
    normal_rate_sim = client.similarity(normal_rate, rate_baseline)

    print("   Normal packet:")
    print(f"     Pattern similarity: {normal_pattern_sim:.3f}")
    print(f"     Rate similarity:    {normal_rate_sim:.3f}")
    print("     Verdict: NORMAL (both high)")

    # Test attack packet (DNS reflection)
    attack_pattern = client.encode(
        {"protocol": "UDP", "src_port": 53, "dst_port": 49152}
    )
    attack_rate = client.encode_scalar_log(100000)  # 1000x normal rate

    attack_pattern_sim = client.similarity(attack_pattern, pattern_baseline)
    attack_rate_sim = client.similarity(attack_rate, rate_baseline)

    print("\n   Attack packet (DNS reflection at 100k pps):")
    print(f"     Pattern similarity: {attack_pattern_sim:.3f}")
    print(f"     Rate similarity:    {attack_rate_sim:.3f}")
    print("     Verdict: ANOMALY (both low)")

    print("\n3. Using difference() to explain...")
    # What changed?
    delta = client.difference(pattern_baseline, attack_pattern)
    print("   The difference vector highlights what's new in the attack")


def demo_storage():
    """Data storage and search."""
    print("\n" + "=" * 60)
    print("DEMO 7: Data Storage and Search")
    print("=" * 60)

    client = HolonClient()

    print("\n1. Inserting data...")
    items = [
        {"type": "billing", "amount": 100, "customer": "alice"},
        {"type": "billing", "amount": 250, "customer": "bob"},
        {"type": "technical", "issue": "login bug", "priority": "high"},
        {"type": "technical", "issue": "slow query", "priority": "medium"},
        {"type": "event", "action": "purchase", "customer": "alice"},
    ]

    ids = client.insert_batch(items)
    print(f"   Inserted {len(ids)} items")

    print("\n2. Searching for billing items...")
    results = client.search(probe={"type": "billing"}, limit=5)
    print(f"   Found {len(results)} results:")
    for r in results[:3]:
        print(f"     - Score: {r['score']:.3f}, Data: {r['data']}")

    print("\n3. Searching with guard (filter)...")
    results = client.search(
        probe={"type": "technical"}, guard={"priority": "high"}, limit=5
    )
    print(f"   High priority technical items: {len(results)}")

    print("\n4. Retrieving by ID...")
    item = client.get(ids[0])
    print(f"   Retrieved: {item}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("HOLON CLIENT: THE PRIMARY INTERFACE")
    print("=" * 60)
    print(
        """
This demo showcases HolonClient as the single, clean interface for:

  - Data encoding and storage
  - VSA primitives (bind, bundle, negate, etc.)
  - Streaming operations (accumulators)
  - Continuous value encoding
  - Similarity computation

Designed for easy portability to Rust:

    struct Holon {
        fn encode(&self, data: Value) -> Vector;
        fn bind(&self, a: &Vector, b: &Vector) -> Vector;
        fn bundle(&self, vecs: &[Vector]) -> Vector;
        fn accumulate(&mut self, acc: &Accumulator, vec: &Vector);
        fn similarity(&self, a: &Vector, b: &Vector) -> f32;
        ...
    }
"""
    )

    demo_basic_encoding()
    demo_vsa_primitives()
    demo_accumulators()
    demo_continuous_encoding()
    demo_sequence_encoding()
    demo_anomaly_detection()
    demo_storage()

    print("\n" + "=" * 60)
    print("SUMMARY: HolonClient provides a clean, unified interface")
    print("=" * 60)
    print(
        """
Key methods available:

  ENCODING:
    encode(data)              - Structured data to vector
    encode_scalar(value)      - Continuous linear/circular encoding
    encode_scalar_log(value)  - Log-scale encoding (rates, frequencies)
    encode_sequence(items)    - Sequence with positional/bundle/ngram modes

  VSA PRIMITIVES:
    bind(a, b)                - Association (AND-like)
    bundle([vectors])         - Superposition (OR-like)
    negate(sup, component)    - Remove component (NOT)
    unbind(bound, key)        - Recover value from binding
    amplify(sup, component)   - Strengthen component
    prototype([vectors])      - Extract common pattern
    difference(before, after) - What changed?
    blend(a, b, alpha)        - Interpolate
    resonance(vec, ref)       - Extract agreeing parts
    permute(vec, k)           - Circular shift

  STREAMING:
    create_accumulator()      - Initialize for streaming
    accumulate(acc, vec)      - Add observation (preserves frequency)
    normalize_accumulator()   - Get unit vector for queries
    threshold_accumulator()   - Convert to bipolar

  SIMILARITY:
    similarity(a, b, metric)  - Cosine, hamming, dot, euclidean, etc.

  STORAGE:
    insert(data), insert_batch(items)
    search(probe, ...), get(id)
"""
    )


if __name__ == "__main__":
    main()
