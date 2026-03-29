# Challenge 019: Forward-Chaining Inference Engine over VSA

## Idea

Use holon's VSA primitives as the substrate for a production rule /
forward-chaining inference engine. Facts are vectors, rules derive
new facts from existing ones, and the entire inference chain operates
in continuous vector space rather than discrete symbols.

## Why This Is Different

Traditional Rete/forward-chaining engines require exact pattern
matches on discrete symbols. A holon-based engine gets:

- **Fuzzy matching**: cosine similarity instead of equality — "this
  situation is *similar to* a known rule antecedent"
- **Compositionality**: `bind()` encodes structure, `bundle()`
  encodes co-occurrence, same algebra as the rule derivations
- **Explainability**: `invert(conclusion, fact_codebook)` decodes
  any derived vector back into the ranked list of contributing facts
- **Learning-optional**: can run as a pure rule engine (no feedback
  loop) or as a self-supervised learner (trading lab style)

## Architecture Sketch

```
Pass 1 (Alpha): raw inputs → boolean/fuzzy gates → first-order facts
Pass 2 (Beta):  first-order facts × rules → second-order facts
Pass N:         repeat until fixpoint (no new facts derived)

Rules are typed compositions:
  confirms  : (Fact, Fact) → Fact          via resonance()
  when      : (Fact, Fact, Fact) → Fact    via conditional_bind()
  context   : (Fact, Fact, Fact) → Fact    via analogy()
```

## Target Domains

- Compliance / policy evaluation (GDPR, SOC2, HIPAA)
- Configuration drift detection and remediation
- Medical/diagnostic reasoning
- Access control with contextual permissions
- Any domain where symbolic inference + fuzzy matching has value

## Primitives Already Proven (in trading lab)

- `bind/bundle` for S-expression encoding
- `invert` for explanation/debug
- `resonance` for confirmation between facts
- `coherence/entropy` for signal quality
- Background removal for always-true fact filtering
- Entropy-derived recognition gates for self-calibrating thresholds

## Status

Concept only. Primitives validated in holon-lab-trading.
No implementation started.
