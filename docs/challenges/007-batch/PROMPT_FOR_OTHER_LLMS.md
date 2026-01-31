# Prompt for Other LLMs

Use this prompt with Grok, Gemini, GPT-4, or other reasoning models to get their input on Holon challenges.

---

## The Prompt

```
I'm working on a project called Holon - a hyperdimensional computing (VSA/HDC) system for fuzzy structured data retrieval. I need help designing complex challenges that showcase its strengths and find its limits.

## What Holon Does

Holon encodes structured data (JSON, EDN) into high-dimensional vectors (~16,000 dimensions) using:
- **Binding**: Combines key-value pairs (key ⊗ value)
- **Bundling**: Aggregates multiple bindings (sum + normalize)
- **N-gram encoding**: Preserves sequence/substring information
- **Chained encoding**: Captures temporal order

Queries use vector similarity (normalized dot product) plus post-filters:
- **Guards**: Exact field matching with operators ($gt, $lt, $contains, $in, $exists)
- **Negations**: Exclude matches ($not)
- **$or**: Disjunction via VSA superposition (single query, not N queries)
- **$any**: Wildcard markers

It also supports:
- **Prototype learning**: Average vectors to create category signatures
- **Blend/Amplify/Negate**: Vector algebra for fuzzy queries
- **FAISS ANN**: Approximate nearest neighbor for scale

## What Holon Is Good At

1. **Fuzzy structured retrieval**: Query `{"role": "dev"}` finds `{"role": "developer"}`
2. **Prototype classification**: 100% accuracy on graph topology, topic classification
3. **N-gram text matching**: Partial phrase matches in documents
4. **Anomaly detection**: Distance from "normal" prototype
5. **Complex queries**: $or + guards + negations compose cleanly

## What Holon Is NOT Good At

1. **NP-hard problems**: Can't solve Sudoku, SAT, etc. geometrically
2. **Global constraints**: Only local similarity, no global coherence
3. **Exact matching**: Fuzzy by design
4. **Very long sequences**: N-gram signal dilutes at >1000 words

## Current Benchmarks

- 123x brute-force speedup after optimization
- 6ms for 999 items (brute force)
- 8ms for 2000 items (with ANN)
- 100% prototype classification up to 200 categories
- Memory: ~250KB per item at 16k dimensions

## What I Need From You

### 1. Complex Challenge Ideas

Design 3-5 challenges that would showcase Holon's unique capabilities. Good challenges should:
- Require fuzzy matching (not just keyword search)
- Benefit from structural encoding (not just text embeddings)
- Combine multiple query features ($or, guards, prototypes)
- Have clear success metrics

For each challenge, specify:
- Data structure (JSON schema)
- Example queries
- What VSA feature it tests
- How to measure success

### 2. Scale Limit Experiments

What experiments would reveal Holon's breaking points? I want to find:
- At what N categories do prototypes overlap too much?
- At what N similar items does the target get lost in noise?
- At what nesting depth does signal degrade?
- At what field count do important fields get drowned?

For each experiment, specify:
- Hypothesis
- Method
- Expected outcome
- How to measure

### 3. Unique VSA Applications

What problems are uniquely suited for hyperdimensional computing that:
- Neural networks struggle with (sample efficiency, interpretability)
- Traditional databases can't do (fuzzy + structured)
- Embedding models miss (structure, not just semantics)

### 4. Benchmark Design

If you were designing a benchmark suite to compare Holon against Elasticsearch, PostgreSQL JSONB, and vector databases like Pinecone:
- What queries would highlight Holon's strengths?
- What queries would expose its weaknesses?
- What metrics matter most?

## Context

This is for a real open-source project. The goal is honest assessment - finding both capabilities and limits. We've already proven Holon can't solve Sudoku (after 55 approaches). We want more challenges like that - ones that either succeed impressively or fail instructively.

Don't hold back on edge cases or adversarial scenarios. Finding the limits is as valuable as finding the wins.
```

---

## How to Use This Prompt

1. **Copy the prompt above** (everything in the code block)
2. **Paste into Grok, Gemini, GPT-4, Claude, etc.**
3. **Collect responses** in separate files
4. **Compare and synthesize** the best ideas

## Expected Output

Each LLM should provide:
- 3-5 challenge ideas with data schemas and queries
- 3-5 scale limit experiments with methods
- Unique VSA application ideas
- Benchmark design suggestions

## Saving Responses

Save each LLM's response as:
- `docs/challenges/007-batch/GROK_RESPONSE.md`
- `docs/challenges/007-batch/GEMINI_RESPONSE.md`
- `docs/challenges/007-batch/GPT4_RESPONSE.md`
- etc.

Then create a synthesis document:
- `docs/challenges/007-batch/SYNTHESIS.md`

---

*Generated from Claude session, Jan 2026*
