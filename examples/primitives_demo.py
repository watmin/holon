#!/usr/bin/env python3
"""
Holon Primitives Demo

Demonstrates all kernel primitives available in Holon.
Run with: ./scripts/run_with_venv.sh python examples/primitives_demo.py
"""

from holon import CPUStore, HolonClient
import numpy as np

def main():
    print("=" * 60)
    print("HOLON PRIMITIVES DEMO")
    print("=" * 60)
    
    # Initialize store
    store = CPUStore(dimensions=2048, backend="cpu")
    client = HolonClient(local_store=store)
    
    # =========================================================================
    # 1. BASIC ENCODING
    # =========================================================================
    print("\n1. BASIC ENCODING")
    print("-" * 40)
    
    # Encode structured data
    vec = store.encoder.encode_data({"name": "Alice", "role": "developer"})
    print(f"   encode_data: {vec.shape} vector")
    
    # =========================================================================
    # 2. SEQUENCE ENCODING
    # =========================================================================
    print("\n2. SEQUENCE ENCODING")
    print("-" * 40)
    
    events = ["login", "view", "purchase", "logout"]
    
    # Positional - preserves order
    pos_vec = store.encode_sequence(events, mode="positional")
    print(f"   positional: order-aware encoding")
    
    # N-gram - fuzzy substring matching  
    ngram_vec = store.encode_sequence(events, mode="ngram")
    print(f"   ngram: fuzzy partial matching")
    
    # Chained - for prefix/suffix operations
    chain_vec = store.encode_sequence(events, mode="chained")
    print(f"   chained: prefix/suffix operations")
    
    # Bundle - unordered (bag of items)
    bundle_vec = store.encode_sequence(events, mode="bundle")
    print(f"   bundle: unordered set")
    
    # =========================================================================
    # 3. BIND / UNBIND
    # =========================================================================
    print("\n3. BIND / UNBIND")
    print("-" * 40)
    
    key = store.encoder.encode_data("role")
    value = store.encoder.encode_data("developer")
    
    # Bind creates a combined representation
    bound = store.bind(key, value)
    print(f"   bind(key, value) = combined vector")
    
    # Unbind retrieves the value
    recovered = store.unbind(bound, key)
    similarity = np.dot(recovered, value) / (np.linalg.norm(recovered) * np.linalg.norm(value))
    print(f"   unbind(bound, key) ≈ value (sim: {similarity:.3f})")
    
    # =========================================================================
    # 4. BUNDLE
    # =========================================================================
    print("\n4. BUNDLE")
    print("-" * 40)
    
    vec1 = store.encoder.encode_data({"skill": "python"})
    vec2 = store.encoder.encode_data({"skill": "rust"})
    vec3 = store.encoder.encode_data({"skill": "go"})
    
    bundled = store.bundle([vec1, vec2, vec3])
    print(f"   bundle([v1, v2, v3]) = superposition")
    
    # =========================================================================
    # 5. PROTOTYPE
    # =========================================================================
    print("\n5. PROTOTYPE")
    print("-" * 40)
    
    # Learn from examples
    examples = [
        store.encoder.encode_data({"lang": "python", "type": "scripting"}),
        store.encoder.encode_data({"lang": "ruby", "type": "scripting"}),
        store.encoder.encode_data({"lang": "javascript", "type": "scripting"}),
    ]
    proto = store.prototype(examples)
    print(f"   prototype([examples]) = common pattern")
    
    # =========================================================================
    # 6. PROTOTYPE_ADD (Incremental)
    # =========================================================================
    print("\n6. PROTOTYPE_ADD")
    print("-" * 40)
    
    new_example = store.encoder.encode_data({"lang": "perl", "type": "scripting"})
    updated_proto = store.prototype_add(proto, new_example, count=3)
    print(f"   prototype_add(proto, new, n=3) = updated prototype")
    
    # =========================================================================
    # 7. DIFFERENCE
    # =========================================================================
    print("\n7. DIFFERENCE")
    print("-" * 40)
    
    before = store.encoder.encode_data({"version": "1.0", "debug": False})
    after = store.encoder.encode_data({"version": "2.0", "debug": True})
    
    delta = store.difference(before, after)
    print(f"   difference(before, after) = what changed")
    
    # =========================================================================
    # 8. AMPLIFY
    # =========================================================================
    print("\n8. AMPLIFY")
    print("-" * 40)
    
    base = store.encoder.encode_data({"topic": "security", "level": "high"})
    signal = store.encoder.encode_data({"topic": "security"})
    
    boosted = store.amplify(base, signal, strength=2.0)
    print(f"   amplify(base, signal, 2.0) = boosted security component")
    
    # =========================================================================
    # 9. NEGATE
    # =========================================================================
    print("\n9. NEGATE")
    print("-" * 40)
    
    full = store.encoder.encode_data({"status": "active", "role": "admin"})
    unwanted = store.encoder.encode_data({"role": "admin"})
    
    filtered = store.negate(full, unwanted, method="orthogonalize")
    print(f"   negate(full, unwanted) = removed admin component")
    
    # =========================================================================
    # 10. BLEND
    # =========================================================================
    print("\n10. BLEND")
    print("-" * 40)
    
    concept_a = store.encoder.encode_data({"style": "formal"})
    concept_b = store.encoder.encode_data({"style": "casual"})
    
    hybrid = store.blend(concept_a, concept_b, alpha=0.7)
    print(f"   blend(a, b, 0.7) = 70% formal, 30% casual")
    
    # =========================================================================
    # 11. PERMUTE
    # =========================================================================
    print("\n11. PERMUTE")
    print("-" * 40)
    
    seq = store.encode_sequence(["a", "b", "c"], mode="positional")
    shifted = store.permute(seq, k=1)
    print(f"   permute(seq, 1) = shifted for sequence ops")
    
    # =========================================================================
    # 12. CLEANUP
    # =========================================================================
    print("\n12. CLEANUP")
    print("-" * 40)
    
    codebook = [
        store.encoder.encode_data({"type": "A"}),
        store.encoder.encode_data({"type": "B"}),
        store.encoder.encode_data({"type": "C"}),
    ]
    noisy = store.bundle([codebook[0], store.encoder.encode_data({"noise": True})])
    
    clean = store.cleanup(noisy, codebook)
    print(f"   cleanup(noisy, codebook) = closest clean vector")
    
    # =========================================================================
    # 13. RESONANCE
    # =========================================================================
    print("\n13. RESONANCE")
    print("-" * 40)
    
    mixed = store.encoder.encode_data({"a": 1, "b": 2, "c": 3})
    filter_vec = store.encoder.encode_data({"a": 1})
    
    resonated = store.resonance(mixed, filter_vec)
    print(f"   resonance(mixed, filter) = extract agreement")
    
    # =========================================================================
    # 14. MARKER PREFIX
    # =========================================================================
    print("\n14. MARKER PREFIX")
    print("-" * 40)
    
    # Custom prefix for when data has $time as real field
    store_custom = CPUStore(dimensions=1024, marker_prefix="@@")
    print(f"   marker_prefix='@@' → use @@time instead of $time")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("PRIMITIVE SUMMARY")
    print("=" * 60)
    print("""
Encoding:
  encode_data(data)              - JSON → vector
  encode_sequence(items, mode)   - List → vector (positional/chained/ngram/bundle)

VSA Operations:
  bind(a, b)                     - Combine (reversible)
  unbind(ab, a)                  - Retrieve b
  bundle([vecs])                 - Superposition
  permute(vec, k)                - Circular shift

Learning:
  prototype([examples])          - Common pattern
  prototype_add(proto, ex, n)    - Incremental update
  cleanup(noisy, codebook)       - Find closest match

Manipulation:
  difference(before, after)      - What changed
  amplify(base, signal, str)     - Boost component
  negate(full, unwanted)         - Remove component
  blend(a, b, alpha)             - Weighted mix
  resonance(vec, ref)            - Extract agreement

Configuration:
  marker_prefix="$"              - Configurable markers
""")


if __name__ == "__main__":
    main()
