# Sift — the query interface gets its name

The Clara-style query interface that fell out of the
memory-as-hologram architecture got the gaze treatment 2026-05-01.
The user landed on **Sift**.

## The name

```
crate:        wat-sift
namespace:    :wat::sift::*
verb:         (sift stream constraint*) -> stream
```

The verb the interface performs — pass a stream through cumulative
narrowing predicates. Each predicate sifts the stream further. The
output is the survivors of the cumulative sift.

## What Sift is (and what it isn't)

**It is:** stream + predicates, applied cumulatively. Single-fact
predicates only. The richest query is *"sift the memory stream by
type, then by topic, then by coincident? against the scope vector,
return top 5."* Each step narrows.

**It is not:** Rete. There is no working memory, no rule firing,
no forward chaining, no cross-fact joins. Brush's Clara is the
emotional ancestor; Sift is the descendant that omits the
machinery the Memory hologram doesn't need.

The user's gaze settled it: *"WorkQuery / QueryDsl / ClaraStyle
would mumble or lie. Sift / Probe / Scry don't."*

## The one universally-indexable predicate: time

Per the user's design call: **TIME is the only predicate that
pushes down to SQL.** Reasons:

- Every SQL backend indexes BTREE on integer columns; epoch time
  is universally cheap.
- Time-window queries are the most common cursor-shape in
  practice (the trading lab's run-DB analysis is almost all
  *"events between T1 and T2"*).
- Pushing other predicates down means knowing the schema; Sift
  is schema-agnostic by design.

So: a sift query with a time-window constraint hits SQL with that
window, then everything else (type filter, topic filter, cosine
predicate) runs in-memory over the streamed result set.

```
sift(memory)
  | time-window(:after some-epoch)         <- pushes to SQL WHERE
  | type-equals(:feedback)                 <- in-memory after stream
  | topic-includes(:wat)                   <- in-memory
  | coincident-with(scope-ast, :top 5)     <- in-memory cosine
```

The first predicate is the cheap one (it filters by SQL index).
Everything after is rich-shape in-memory pruning.

## The Constraint generalization

Per the user's note: *"TimeConstraint generalizes to
:wat::sift::Constraint."* The trading lab's existing telemetry
query code has a `TimeConstraint` type. Lifting to wat-sift means
generalizing it:

```scheme
(:wat::core::enum :wat::sift::Constraint
  (TimeWindow      ((after :i64) (before :i64)))
  (FieldEquals     ((field :String) (value :String)))
  (FieldIncludes   ((field :String) (any-of :Vec<String>)))
  (Coincident      ((scope :wat::holon::HolonAST) (top-n :i64)))
  (Custom          ((predicate :fn(Stream::Item) -> :bool))))
```

`TimeWindow` is the special-cased SQL-pushdown variant. The rest
are evaluated in-memory on the stream after SQL returns.

The `Custom` variant is the escape hatch — any consumer with a
genuinely application-specific predicate writes it as a closure;
Sift evaluates it like any other in-memory filter.

## The dependency direction

```
wat-sqlite                  — storage primitive (already shipped)
   ↓
wat-sift                    — Sift crate (NEW)
   │ tests use wat-sqlite as the integration target
   │
   ├→ wat-telemetry         — re-exports cursor primitives,
   │                          consumes Sift for queries
   │
   └→ wat-memory             — this scratch arc's eventual home
        (uses Sift for recall, layers a HologramStore on top
         for the in-memory cosine dimension)
```

`wat-telemetry` becomes the FIRST migration of Sift. The trading
lab's existing query code lives there today; lifting to wat-sift
extracts the generic capability and leaves telemetry as a
consumer that re-exports the cursor-shaped primitives it owns
(per-tail-cursor, follow, batch-flush).

`wat-memory` becomes the SECOND consumer. The recall function
constructs Sift queries; the cosine predicate is a `Coincident`
constraint; the SQL stream is the memory-table SELECT.

## Why "Sift" beat the runners-up

Per the user's gaze:

- **Sift** — describes the mechanism (stream → predicate → stream).
  Carries no false lineage. Verb-shaped, fits the wat family
  (bind, bundle, cosine, sift). Decouples cleanly.
- **Probe** — picks up the pry/gdb framing. Same Level-1-clear /
  Level-2-strong; choice was taste. Probe describes intent;
  Sift describes mechanism. **Mechanism is the durable thing.**
- **Scry** — collides with the existing `/scry` ward. Disambiguation
  cost not worth it.
- **WorkQuery / QueryDsl / ClaraStyle** — mumble or lie. WorkQuery
  reads as a place; QueryDsl announces a sub-language without
  warrant; ClaraStyle imports machinery wat-sift deliberately
  omits.

The Level-1 (Lies) check passes for Sift. The Level-2 (Mumbles)
check passes for Sift. The Level-3 (Taste) tier is where Sift vs
Probe lived; the gaze doesn't chase taste, but in this case the
mechanism-vs-intent distinction tipped to Sift.

## What this changes upstream

**The architecture.md design now goes through Sift.** Where it
previously named *"wat-query"* or *"the Clara-style query
interface,"* substitute *"Sift"* / *"wat-sift"* / *"`:wat::sift::*`"*.
The recall function in the memory wat program calls Sift; Sift
calls wat-sqlite; results flow through the cumulative predicates;
the surviving stream is what the MCP returns as `Vec<MemoryRef>`.

**The storage.md schema design stays correct.** Three tables,
multiple-table compose, EDN payloads, file-system content refs.
No change. Sift sits between the memory wat program and the
SQLite backing store; the schema doesn't care which crate
queries it.

**The slice plan (slice-plan.md) gains a hard dependency on
wat-sift's extraction landing first.** Memory recall cannot
cleanly express the *"SQL filter + cosine rank"* composition
without Sift. So:

1. wat-mcp ships its slice 1 (basic MCP eval) — separate arc
2. wat-sift extracts from wat-telemetry — separate arc
3. wat-memory consumes both — this arc opens

## What's still open about Sift itself

- **The Custom predicate's closure-capture.** Sift accepts
  `:fn(Stream::Item) -> :bool`. What can the closure capture?
  Bare data, yes. Other Sift queries? Probably not in v1
  (would force nested-stream semantics). Lean: keep Custom
  closures local-scope only in v1; revisit if a real consumer
  surfaces nested needs.
- **Stream materialization.** Does Sift materialize the SQL
  result set in memory, or does it stream lazily through the
  predicates? Lean: lazy stream (Iterator-shaped) for
  composability; eager materialization only when a Sift
  predicate genuinely requires it (e.g., the Coincident
  predicate needs the projection vectors to be loaded).
- **Pushdown extension.** Today only `TimeWindow` pushes to
  SQL. Tomorrow's question: should `FieldEquals` push down too?
  Lean: yes for known-indexed columns; the consumer declares
  which fields are indexed at sift-construction time. But not
  in v1 — let the actual consumer pain decide.

These are small open questions; none block opening the arc.

## Status

- **Named:** 2026-05-01 via gaze ward (per the speaker's note)
- **Crate:** `wat-sift` planned; not yet extracted
- **Migration source:** existing telemetry query code in
  `holon-lab-trading/wat/io/telemetry/` and wat-rs's
  `crates/wat-telemetry/`
- **First consumer after migration:** `wat-memory` (this arc)
- **Sequencing:** must ship before this arc's slice 3 (recall
  function); can ship in parallel with this arc's slices 1 + 2
  (vocab + storage)
