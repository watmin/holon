# Compositional Recall

Algebraic queries over incident memory — one operation per query, no
re-indexing, no pipeline.

## The Problem

You have a library of past incident reports. A new incident arrives and you want
to ask: "find me similar incidents, but exclude database issues," or "boost
results with production impact," or "what would this backend incident look like
if it happened in the auth layer?" These are not SQL queries. They are
algebraic transformations of meaning — and vector databases don't give you that.

## What Holon Does

Every incident is encoded into a vector. The full library lives in memory as
vectors and subspace engrams. Queries compose algebraically:

- `negate(query, db_proto)` — subtract the database structural signal
- `amplify(query, prod_proto)` — strengthen production-environment dimensions
- `analogy(backend, security, query)` — transfer the relationship between layers
- `attend(query, security_proto)` — extract the security-resonant components

Each operation is a single function call. The result is a new query vector,
which you pass to `topk_similar`. No re-indexing, no query rewrite, no
intermediate pipeline.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.compositional_recall.showcase
```

## Output

16 incidents encoded across 4 service layers. A probe is constructed: a
high-severity latency incident in backend/prod:

```
Encoding 16 incidents across 4 service layers...
  Layer subspaces: database, frontend, backend, security
```

**Basic recall** — what any vector database gives you:

```
  Query   : Basic recall (cosine similarity)
    1. INC-005  api           backend     high      latency     env=prod     sim=0.694
    2. INC-007  api           backend     critical  outage      env=prod     sim=0.576
    3. INC-008  api           backend     low       latency     env=staging  sim=0.465
    4. INC-006  api           backend     medium    crash       env=prod     sim=0.428
```

Top results are backend/prod latency and outage incidents. Expected.

**Negation** — structurally remove database signal from the query:

```
  Query   : negate(probe, db_proto)  — exclude database incidents
    1. INC-005  api           backend     high      latency     env=prod     sim=0.629
    2. INC-007  api           backend     critical  outage      env=prod     sim=0.538
    3. INC-008  api           backend     low       latency     env=staging  sim=0.416
    4. INC-006  api           backend     medium    crash       env=prod     sim=0.377
```

Similarity scores drop slightly — the database signal has been projected out.
Database incidents would now rank lower even if they were structurally similar.
No filter clause, no exclusion list.

**Amplification** — boost production-environment signal in the query:

```
  Query   : amplify(probe, prod_proto)  — prioritise production impact
    1. INC-005  api           backend     high      latency     env=prod     sim=0.703
    2. INC-007  api           backend     critical  outage      env=prod     sim=0.596
    3. INC-006  api           backend     medium    crash       env=prod     sim=0.482
    4. INC-008  api           backend     low       latency     env=staging  sim=0.445
```

INC-006 (medium crash, prod) jumps above INC-008 (low latency, staging) —
production incidents are re-ranked without touching the index.

**Attention** — extract the security-resonant signal from the probe:

```
  Query   : attend(probe, security_proto)  — surface security-resonant signal
    1. INC-013  auth          security    critical  intrusion   env=prod     sim=0.720
    2. INC-014  auth          security    high      intrusion   env=prod     sim=0.698
    3. INC-015  api           security    critical  intrusion   env=prod     sim=0.679
    4. INC-016  auth          security    medium    crash       env=staging  sim=0.638
```

The query was a backend latency incident. After `attend` extracts the dimensions
that resonate with security patterns, the results pivot entirely to intrusion
incidents. The probe hasn't been replaced — it's been filtered through a
security lens.

**EngramLibrary** — which structural manifold does the probe belong to?

```
  Query   : EngramLibrary.match — which layer manifold fits best?
    1. layer='backend'  residual=35.98  (lower = better fit)
    2. layer='security'  residual=60.07  (lower = better fit)
    3. layer='database'  residual=60.29  (lower = better fit)
    4. layer='frontend'  residual=60.48  (lower = better fit)
```

```
Each query is one algebraic operation on the same encoded library.
No re-indexing. No query rewrite. No pipeline.
```

## Without Holon

Negation requires a separate exclusion list and a keyword or metadata filter.
Amplification requires a re-ranking step with a separate scoring model.
Analogy is not available in any standard vector database — it would require
embedding a new query from scratch using a language model.
Attention requires extracting the overlap between two embeddings as a separate
post-processing step.

Each of these is a pipeline. In Holon, each is a single line.

## Try

- Compose operations: `negate(amplify(probe, prod_proto), db_proto)` to boost
  prod and exclude db in one query.
- Build a `resolved=True` prototype and use `negate(probe, resolved_proto)` to
  find only open incidents.
- Add more incidents (50+) to see analogy produce cleaner cross-layer transfers.
