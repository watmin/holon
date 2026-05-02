# Clara-style querying as its own crate

The user's mid-scratch refinement (2026-05-01):

> "i think this also means we need to lift and shift the clara-style
> querying to its own crate and it deps on the sqlite crate for its
> own tests?....
>
> telemetry then deps on the query interface?..."

Right. The recognition surfaces a substrate refactor that this
scratch arc didn't trigger but does crystallize. Clara-style
querying — predicate-driven discrimination over stored data, the
Rete lineage from Forgy → Brush → wat — is a generic capability
the substrate's telemetry layer happens to use. It deserves its
own home.

## The reorganization

```
BEFORE                              AFTER

wat-sqlite                          wat-sqlite
                                       ↓
wat-telemetry                       wat-query              (NEW)
  (queries inline,                     │ (uses wat-sqlite for its own tests)
   not exported)                       │
                                       ├→ wat-telemetry
                                       │    (now consumes the query interface)
                                       │
                                       └→ wat-memory       (this scratch arc's
                                            (recall via         eventual home)
                                             query +
                                             hologram)
```

The dependency direction is the right one: storage primitive at
the bottom, generic query interface in the middle, application
consumers at the top. Each crate has one job.

## What lives in `wat-query`

Predicate-driven querying over typed data. Three shapes settled
in the trading lab's existing telemetry-query work:

**1. Exact-match predicates (SQL-shaped).**
The wat side declares predicates as Holon AST forms over named
fields. The query crate compiles the predicate AST into a
prepared SQL statement against a wat-sqlite table. The consumer
writes wat predicates; the substrate generates the SQL.

```scheme
(:wat::query::where memory
  (= type :feedback)
  (= topic :wat))
;; → compiles to a prepared SELECT against the memory table
```

**2. Semantic predicates (cosine-shaped).**
The wat side declares predicates as `coincident?` queries against
projection vectors stored in the same SQLite tables (or a
companion HologramStore built from them). The query crate routes
these through the algebra layer.

```scheme
(:wat::query::coincident-with memory
  (:wat::query::projection-of memory)
  scope-ast
  :top-n 5)
;; → materializes projections from SQLite, builds (or reuses)
;;   the in-process hologram, runs cosine, returns top-N
```

**3. Composed predicates.**
SQL filter ANDed with cosine ranking, evaluated in two phases:
SQL filter narrows the row set (cheap, keeps the DB fast); the
hologram cosine ranks within the filtered set (semantic, returns
the top-N).

```scheme
(:wat::query::compose
  (:wat::query::where memory
    (= type :feedback))
  (:wat::query::coincident-with memory
    (:wat::query::projection-of memory)
    scope-ast
    :top-n 3))
;; → SQL filters to feedback memories, then cosine ranks within
;;   that subset, returning top 3 most relevant
```

The crate exposes these three shapes plus the AST grammar that
combines them. Consumers (telemetry, memory, future MTG state,
future truth engine) write queries in this grammar; the crate
decides how to evaluate them against the underlying storage.

## Why this is a substrate move, not a memory-arc move

The memory hologram surfaces the need, but the need exists across
multiple consumers. Lifting it to a crate means:

- **Telemetry's existing query code becomes the wat-query crate's
  first migration.** The trading lab already does Clara-style
  querying internally for run-DB analysis; that code moves to
  wat-query and the lab depends on the new crate.
- **Memory-as-hologram is the first NEW consumer.** It builds on
  the migrated query interface; it doesn't have to invent its
  own.
- **Future consumers inherit.** The truth engine that's been
  sitting in BOOK foreshadowing as the third domain after trading
  + MTG would consume the same query crate. Same shape; different
  data.
- **The substrate stays one substrate.** The discipline named in
  CONVENTIONS.md (each `:wat::*` sub-prefix one cohesive concern)
  enforces the split. Query is a concern. SQLite is a concern.
  Telemetry is a concern. Memory is a concern. Each gets its own
  namespace, none collides with the others.

## The dependency edge for memory-as-hologram

Memory-as-hologram cannot ship cleanly until wat-query exists.
The hologram + recall function in `architecture.md` was written as
if memory would consume substrate primitives directly; the
correct shape is **memory consumes wat-query, which consumes
wat-sqlite + the algebra layer.**

This pushes the slice plan: before memory-as-hologram opens as a
real wat-rs arc, the wat-query extraction has to happen. That's
its own arc — probably a few slices to:

1. Audit the trading lab's existing query code; identify what's
   generic vs. telemetry-specific
2. Lift the generic parts into a new `crates/wat-query/` crate
3. Migrate the telemetry crate to consume wat-query
4. Migrate the trading lab's telemetry consumer call sites
5. Ship the crate with its own tests against wat-sqlite

After wat-query lands, memory-as-hologram becomes a clean
five-slice consumer arc with no substrate refactor in its path.

## What this doesn't change

The five locked decisions in `README.md` and `INDEX.yaml` stay
intact:

1. Memories ARE HolonASTs ✓
2. Storage is SQLite + file system + in-process hologram ✓
3. Recall is one cosine via coincident? ✓ — but now expressed
   through wat-query's grammar, not raw substrate calls
4. Persistence via canonical EDN in SQLite rows ✓
5. MCP delivery via wat-mcp's one-tool surface ✓ — the recall
   function inside the hosted wat program now calls wat-query

The architecture in `architecture.md` is right; the
implementation now goes through one more layer (wat-query)
between the consumer and the substrate primitives.

## Naming the crate

The user said *"clara-style querying."* Two candidate names:

- `wat-clara` — pays direct homage to Brush's Clara Rules; honest
  to the lineage; risks a tight name binding to one specific
  prior art
- `wat-query` — generic, honest about what it does, doesn't claim
  Rete-completeness (the crate may not implement full Rete; it
  ships the query subset)

Lean: `wat-query`. The crate's behavior is queryable predicates
over typed data; full Rete (forward-chaining inference, working
memory, conflict resolution) might be a future extension or a
separate crate. Don't promise more than the first version
delivers.

If the gaze ward later finds the Clara homage meaningful enough
to ship, renaming is cheap — workspace-member crates inside
`crates/wat-*/` move under reserved-prefix rules without
breaking external consumers (only internal dep references need
to update).

## Status of this implication

- **Surfaced by:** memory-as-hologram architecture work, 2026-05-01
- **Blocks memory arc:** yes — the memory arc cannot ship its
  recall function cleanly without wat-query existing
- **Blocks anything else:** the trading lab's telemetry queries
  work today; the lift is a refactor for cleanliness, not a
  blocker for ongoing trading work
- **Owner:** wat-rs, when the user is ready to open the arc
- **Slot in user's queue:** TBD — sequenced before
  memory-as-hologram, after current wat-mcp + arc 109 work
  completes
