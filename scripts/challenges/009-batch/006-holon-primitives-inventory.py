#!/usr/bin/env python3
"""
Challenge 009-006: Holon Primitives Inventory

Catalog ALL available Holon encoders and primitives to ensure
Phase 3 (Full Program Synthesis) uses everything at our disposal.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def inventory_primitives():
    """List all available primitives and their purposes."""

    print("=" * 70)
    print("HOLON PRIMITIVES INVENTORY")
    print("=" * 70)

    # =========================================================================
    # ENCODERS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ENCODERS")
    print("=" * 70)

    encoders = {
        "Encoder": {
            "file": "holon/encoder.py",
            "purpose": "Base encoder with all VSA primitives",
            "used_in_009": True,
        },
        "TorchHDEncoder": {
            "file": "holon/torchhd_encoder.py",
            "purpose": "GPU-accelerated with Level/Circular embeddings",
            "used_in_009": False,
            "key_feature": "Numeric similarity (200 ≈ 201 ≠ 500)",
        },
        "EnhancedEncoder": {
            "file": "holon/enhanced_encoder.py",
            "purpose": "Extended list encoding with N-gram modes",
            "used_in_009": False,
            "key_feature": "TF-IDF style weighting, subsequence alignment",
        },
        "SemanticEncoder": {
            "file": "holon/semantic_encoder.py",
            "purpose": "Domain-specific pattern recognition",
            "used_in_009": False,
            "key_feature": "Auto-detects and enhances math/graph patterns",
        },
        "MathematicalPatternEncoder": {
            "file": "holon/domain_encoders.py",
            "purpose": "Fractal, wave, polynomial pattern encoding",
            "used_in_009": False,
        },
        "GraphTopologyEncoder": {
            "file": "holon/domain_encoders.py",
            "purpose": "Scale-free, small-world, random topology encoding",
            "used_in_009": False,
        },
    }

    for name, info in encoders.items():
        status = "✓ USED" if info.get("used_in_009") else "✗ NOT USED"
        print(f"\n{name} [{status}]")
        print(f"  File: {info['file']}")
        print(f"  Purpose: {info['purpose']}")
        if "key_feature" in info:
            print(f"  Key feature: {info['key_feature']}")

    # =========================================================================
    # PRIMITIVES (Operations on vectors)
    # =========================================================================
    print("\n" + "=" * 70)
    print("VSA PRIMITIVES")
    print("=" * 70)

    primitives = {
        # Core VSA operations
        "bind(a, b)": {
            "purpose": "Compose two vectors (role-filler binding)",
            "used_in_009": True,
            "use_case": "Field interactions, structure preservation",
        },
        "bundle(vectors)": {
            "purpose": "Superposition of vectors (logical OR)",
            "used_in_009": True,
            "use_case": "Combine multiple components",
        },

        # Signal manipulation
        "negate(superpos, component)": {
            "purpose": "Remove component from superposition",
            "used_in_009": False,
            "use_case": "Exclude patterns, anomaly detection",
        },
        "amplify(superpos, component, strength)": {
            "purpose": "Boost component in superposition",
            "used_in_009": False,
            "use_case": "Emphasize important features",
        },
        "resonance(vec, reference)": {
            "purpose": "Extract part of vec that matches reference",
            "used_in_009": False,
            "use_case": "Signal extraction, filtering",
        },

        # Learning/comparison
        "prototype(vectors, threshold)": {
            "purpose": "Extract common pattern from examples",
            "used_in_009": True,
            "use_case": "Learn category signatures",
        },
        "prototype_add(proto, example, count)": {
            "purpose": "Incrementally update prototype",
            "used_in_009": False,
            "use_case": "Online learning, streaming data",
        },
        "difference(before, after)": {
            "purpose": "Compute what changed between states",
            "used_in_009": False,
            "use_case": "Change detection, drift detection",
        },
        "blend(a, b, alpha)": {
            "purpose": "Weighted interpolation between vectors",
            "used_in_009": False,
            "use_case": "Fuzzy queries, hybrid concepts",
        },

        # Sequence operations
        "permute(vec, k)": {
            "purpose": "Circular shift for position encoding",
            "used_in_009": False,
            "use_case": "Sequence order, 'what comes after X'",
        },
        "cleanup(noisy, codebook)": {
            "purpose": "Find closest clean vector",
            "used_in_009": False,
            "use_case": "Denoising, classification",
        },
    }

    for name, info in primitives.items():
        status = "✓ USED" if info.get("used_in_009") else "✗ NOT USED"
        print(f"\n{name} [{status}]")
        print(f"  Purpose: {info['purpose']}")
        print(f"  Use case: {info['use_case']}")

    # =========================================================================
    # LIST ENCODING MODES
    # =========================================================================
    print("\n" + "=" * 70)
    print("LIST ENCODING MODES")
    print("=" * 70)

    list_modes = {
        "positional": {
            "purpose": "Bind items to position vectors",
            "used_in_009": False,
            "use_case": "Ordered sequences, arrays",
        },
        "chained": {
            "purpose": "Chain items for suffix/prefix operations",
            "used_in_009": False,
            "use_case": "Sequence matching, n-grams",
        },
        "ngram": {
            "purpose": "N-gram encoding with configurable sizes",
            "used_in_009": False,
            "use_case": "Text matching, fuzzy search",
        },
        "bundle": {
            "purpose": "Pure bundling (multiset, no order)",
            "used_in_009": False,
            "use_case": "Bag-of-words, unordered sets",
        },
        "ngram_configurable": {
            "purpose": "Choose which N sizes to use",
            "used_in_009": False,
            "use_case": "Custom n-gram combinations",
        },
        "ngram_weighted": {
            "purpose": "TF-IDF style bigram weighting",
            "used_in_009": False,
            "use_case": "Text ranking, importance weighting",
        },
        "subsequence_aligned": {
            "purpose": "Sliding window encoding",
            "used_in_009": False,
            "use_case": "Substring matching",
        },
    }

    for name, info in list_modes.items():
        status = "✓ USED" if info.get("used_in_009") else "✗ NOT USED"
        print(f"\n{name} [{status}]")
        print(f"  Purpose: {info['purpose']}")
        print(f"  Use case: {info['use_case']}")

    # =========================================================================
    # SPECIAL FEATURES
    # =========================================================================
    print("\n" + "=" * 70)
    print("SPECIAL FEATURES")
    print("=" * 70)

    features = {
        "$time encoding": {
            "purpose": "Circular + positional time encoding",
            "used_in_009": False,
            "use_case": "Temporal similarity, 'around that time'",
        },
        "Level embeddings (TorchHD)": {
            "purpose": "Numeric similarity (close values → similar vectors)",
            "used_in_009": False,
            "use_case": "Status codes, prices, metrics",
        },
        "Circular embeddings (TorchHD)": {
            "purpose": "Cyclical value similarity (hour, day of week)",
            "used_in_009": False,
            "use_case": "Time-of-day, angles, periodic data",
        },
        "Mathematical primitives": {
            "purpose": "Convergence, frequency, amplitude encoding",
            "used_in_009": False,
            "use_case": "Scientific data, signal processing",
        },
        "Graph topology encoding": {
            "purpose": "Scale-free, small-world network patterns",
            "used_in_009": False,
            "use_case": "Network analysis, social graphs",
        },
    }

    for name, info in features.items():
        status = "✓ USED" if info.get("used_in_009") else "✗ NOT USED"
        print(f"\n{name} [{status}]")
        print(f"  Purpose: {info['purpose']}")
        print(f"  Use case: {info['use_case']}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: What Phase 3 Should Use")
    print("=" * 70)

    print("""
Currently using in Challenge 009:
- encode_data() - basic encoding
- bind() - for interactions
- bundle() - via weighted sum
- prototype() - via averaging examples

NOT using (opportunities for Phase 3):

1. SIGNAL MANIPULATION
   - negate(): Exclude noise patterns from encoding
   - amplify(): Boost discriminative features
   - resonance(): Extract only the matching part

2. COMPARISON OPERATIONS
   - difference(): Find what distinguishes classes
   - blend(): Create hybrid queries for fuzzy matching

3. SEQUENCE OPERATIONS
   - permute(): Position-aware encoding
   - cleanup(): Denoise before classification

4. BETTER ENCODERS
   - TorchHDEncoder: Level embeddings for numeric fields
   - EnhancedEncoder: N-gram modes for list fields

5. INCREMENTAL LEARNING
   - prototype_add(): Online learning without rebuilding

PHASE 3 PROGRAM SYNTHESIS should search over compositions of ALL these!
    """)


def demonstrate_underutilized_primitives():
    """Show examples of primitives we're not using."""

    print("\n" + "=" * 70)
    print("DEMONSTRATION: Underutilized Primitives")
    print("=" * 70)

    from holon import CPUStore
    import numpy as np

    store = CPUStore(dimensions=4096)
    encoder = store.encoder

    # Example data
    billing = encoder.encode_data({"type": "billing", "keywords": ["refund"]})
    tech = encoder.encode_data({"type": "technical", "keywords": ["crash"]})
    noise = encoder.encode_data({"noise": "random123"})

    # Prototype from examples
    billing_examples = [
        encoder.encode_data({"type": "billing", "keywords": ["refund"]}),
        encoder.encode_data({"type": "billing", "keywords": ["charge"]}),
        encoder.encode_data({"type": "billing", "keywords": ["invoice"]}),
    ]
    billing_proto = encoder.prototype(billing_examples)

    def cosine_sim(a, b):
        return float(np.dot(a.astype(float), b.astype(float)) /
                     (np.linalg.norm(a) * np.linalg.norm(b)))

    print("\n1. negate() - Remove noise from encoding")
    mixed = encoder.bundle([billing, noise])
    cleaned = encoder.negate(mixed, noise)
    print(f"   sim(mixed, billing) = {cosine_sim(mixed, billing):.3f}")
    print(f"   sim(cleaned, billing) = {cosine_sim(cleaned, billing):.3f}")
    print(f"   → Negation removes noise, increases signal")

    print("\n2. amplify() - Boost important feature")
    type_vec = encoder.encode_data({"type": "billing"})
    base = encoder.encode_data({"type": "billing", "channel": "email"})
    amplified = encoder.amplify(base, type_vec, strength=2.0)
    print(f"   sim(base, type_vec) = {cosine_sim(base, type_vec):.3f}")
    print(f"   sim(amplified, type_vec) = {cosine_sim(amplified, type_vec):.3f}")
    print(f"   → Amplification boosts specific component")

    print("\n3. difference() - What distinguishes classes")
    diff = encoder.difference(billing_proto, tech)
    print(f"   sim(diff, billing_proto) = {cosine_sim(diff, billing_proto):.3f}")
    print(f"   sim(diff, tech) = {cosine_sim(diff, tech):.3f}")
    print(f"   → Difference captures what's unique to each")

    print("\n4. resonance() - Extract matching part")
    query = encoder.encode_data({"type": "billing"})
    mixed_signal = encoder.bundle([billing, tech])
    resonated = encoder.resonance(mixed_signal, query)
    print(f"   sim(mixed_signal, billing) = {cosine_sim(mixed_signal, billing):.3f}")
    print(f"   sim(resonated, billing) = {cosine_sim(resonated, billing):.3f}")
    print(f"   → Resonance extracts the billing component")

    print("\n5. blend() - Fuzzy hybrid query")
    hybrid = encoder.blend(billing, tech, alpha=0.5)
    print(f"   sim(hybrid, billing) = {cosine_sim(hybrid, billing):.3f}")
    print(f"   sim(hybrid, tech) = {cosine_sim(hybrid, tech):.3f}")
    print(f"   → Blend creates query matching both categories")

    print("\n6. cleanup() - Find closest known prototype")
    noisy_billing = encoder.bundle([billing, noise, noise])
    codebook = [billing_proto, tech]
    cleaned = encoder.cleanup(noisy_billing, codebook)
    print(f"   Noisy input → closest codebook entry")
    print(f"   sim(noisy, billing_proto) = {cosine_sim(noisy_billing, billing_proto):.3f}")
    print(f"   cleanup() returned billing_proto: {np.array_equal(cleaned, billing_proto)}")


if __name__ == "__main__":
    inventory_primitives()
    demonstrate_underutilized_primitives()
