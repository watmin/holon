# Open questions

Not blockers. The arc opens cleanly with the locked decisions and
the slice plan. Each question below has a lean where the design
has one; resolution surfaces in the slice that needs the answer.

## Vocabulary (3)

**Q1 — Which axes ship in slice 1?**
Lean: four (type, topic, handle, recency). The other five
candidates from `memory-vocabulary.md` (reference-type,
specificity, relevance-context, valence, body-hash) extend the
surface as the recall protocol's actual usage surfaces a need
for them. No principled reason to ship more in v1.

**Q2 — Scalar bucketing for recency.**
The recency axis encodes via Thermometer over (oldest_epoch,
current_epoch). What's the bucketing? Per the substrate's
`set-presence-sigma!` discipline, this is data-driven — the
sigma derives from the dimension, not from a hardcoded bucket
count. But the Thermometer bounds are still a choice: sliding
window (e.g., last 90 days) vs. all-time. **Lean: all-time.**
Reason: memory persistence is by definition long-term; trimming
to a window throws away signal the recall might want. If
recall against an all-time-bounded recency axis turns out to
under-weight recent memories, the fix is a recency-weighted
combinator at recall time, not a smaller Thermometer window.

**Q3 — Set vs. list for tags.**
For the topic and handle axes (multi-valued per memory), is the
encoding a Set (sorted, deduplicated) or a List (positional,
duplicates allowed)? **Lean: Set.** Per arc 074's
container-surface discipline (`get` / `assoc` / `conj` /
`contains?` polymorphic). Memory tags don't have positional
semantics; ordering is noise; deduplication is correct. Sets
also get the substrate's set-based comparison primitives for
free.

## Recall protocol (3)

**Q4 — Top-N default.**
Lean: 5 (per `recall-protocol.md`). Configurable per-call.
First-prototype heuristic; subjective tuning during slice 5.

**Q5 — Scope-AST shape for free-form queries.**
The `like` combinator from `recall-protocol.md` accepts a free-form
description string; the body extractor parses it into axis values.
What if the extractor returns no axis values (description is
generic, e.g., *"the user's pattern of refusing orthodox tooling"*
parses to nothing structured)? **Lean: fall through to a
text-similarity backup** — the recall encodes the description
itself as a body-hash bundle and finds memories whose body hashes
to nearby coordinates. This is a degenerate cosine but better than
empty results. Open whether this backup is in v1 or punted to a
future slice.

**Q6 — Recall across compaction.**
The hologram is per-process. When the wat-memory MCP server is
restarted, the hologram rebuilds from SQLite (cheap; <1 second on
80 memories). When a Claude session compacts and a new session
connects, the recall protocol works identically — the MCP server
doesn't know the agent compacted. This is handled. The open
question is **whether the new session knows the memory MCP exists
at all**. Lean: addressed by the BOOTSTRAP.md update (to mention
the recall MCP tool as part of the operating discipline). Not a
substrate question; a documentation question.

## Persistence (1)

**Q7 — Write-time AST canonicalization.**
When the memory's body lowers to a HolonAST and serializes to
EDN for SQLite storage, what's the canonicalization rule? Two
choices:
- (a) The wat-edn shipping default (sorts map keys; uniform
  whitespace; stable across processes)
- (b) A memory-specific canonicalization that strips position
  metadata, normalizes scalar precision to N digits, etc.

**Lean: (a).** Use the substrate's default canonicalization;
don't invent a memory-specific shape. If the recall hologram
needs a normalized representation (e.g., to make scalars cluster
properly), normalization is at extraction time (the vocab module
quantizes the scalar to N digits before binding), not at
serialization time. Keeps the EDN round-trip lossless.

## Delivery (1)

**Q8 — MCP error-shape for "no memories matched."**
When the recall returns zero results (e.g., the scope encodes
to a coordinate too far from any stored projection), is that
an error or a successful empty response? **Lean: empty
response.** Zero is a valid recall outcome. The agent receives
`Vec<MemoryRef>` with `len=0` and decides what to do. Errors
are reserved for *actual* failures: SQLite connection lost,
hologram build failed, malformed scope-AST, etc.

What about *"no memories matched above a threshold"*? This is
the same shape: top-N returns up to N matches; if all matches
are below a sane similarity floor, the agent can read the
`score` field on each ref and choose to ignore them. The
recall function doesn't filter on score; the agent does.

## From Sift (1)

**Q9 — Closure capture for the Custom predicate.**
Sift's `Custom` constraint takes `:fn(Stream::Item) -> :bool`.
What can the closure capture? **Lean: bare data only in v1.**
Closures over other Sift queries would require nested-stream
semantics; not worth complicating the v1 surface. Revisit if a
real consumer surfaces a genuine nested need.

## What's NOT a question (settled)

- Memories ARE HolonASTs. Settled in `architecture.md`.
- Storage is SQLite + file system + in-process hologram. Settled
  in `storage.md`.
- MCP returns refs, not bytes. Settled in `architecture.md`.
- Time is metadata, not the primary index. Settled in
  `storage.md`.
- The query interface is Sift, not WorkQuery / QueryDsl /
  Clara. Settled in `sift.md` per the user's gaze call.
- The arc cannot ship until wat-sift extracts. Settled in
  `slice-plan.md`.

## Status

Eight active open questions. None blocks opening the arc. Each
has a lean documented; each gets resolved in the slice that
needs the answer. The persistence layer holds the design; the
substrate work follows when it surfaces.
