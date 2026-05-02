# Slice plan

Five slices to first running prototype of memory-as-hologram,
plus the two prerequisite arcs that have to ship before this one
opens. Sized for the user's prompting cadence; ~1 week of focused
work after the prerequisites land.

## Prerequisite arcs (must ship first)

These are not part of the memory-as-hologram arc; they are
substrate work that this arc depends on.

### Prerequisite 1 — wat-mcp slice 1

Source: `scratch/2026/04/006-wat-mcp/`. The basic MCP eval
surface — wat programs as MCP servers, one tool (`wat-eval`),
EDN payload inside JSON-RPC envelope. Discovery via wat
introspection. Already designed; not yet shipped. ~3-4 days
when it surfaces.

### Prerequisite 2 — wat-sift extraction

Source: `scratch/2026/05/001-memory-as-hologram/sift.md`. Lift
the Clara-style query interface from the trading lab's
telemetry crate to its own `crates/wat-sift/` workspace member.
Generalize `TimeConstraint` to `:wat::sift::Constraint` (with
`TimeWindow`, `FieldEquals`, `FieldIncludes`, `Coincident`,
`Custom` variants). Tests against `wat-sqlite`. ~5-7 days
including the telemetry-crate migration.

The two prerequisites are **independent of each other**; can
ship in parallel or in either order.

## Memory arc slices (after prerequisites land)

### Slice 1 — Memory vocab + extractor (~1.5 days)

**Ships:**
- `wat/memory/vocab.wat` — the `extract-facts` function from
  `memory-vocabulary.md`. Reads a `MemoryRecord`; emits
  `Vec<HolonAST>` (the four-axis minimum: type / topic / handle
  / recency).
- `wat/memory/extractor.wat` — the body-extraction primitives
  (regex for backtick-quoted symbols, proper-noun extraction,
  whatever else the first cut needs).
- The hybrid mechanism (mechanism C from memory-vocabulary.md):
  frontmatter axes win; body extractor fills gaps.

**Tests:** unit tests over a representative sample of the
existing 80+ memories; assert each emits expected axis values.
Green-bar before slice 2.

### Slice 2 — SQLite schema + indexer (~1.5 days)

**Ships:**
- `wat/memory/schema.wat` — `MemoryRecord` / `MemoryAxis` /
  `MemoryLink` struct declarations. Auto-derived schema via
  `Sqlite/auto-spawn` (arc 085 substrate primitive).
- `wat/memory/indexer.wat` — reads the existing memory tree
  (`/home/watmin/.claude/projects/.../memory/`); for each
  memory file, runs slice 1's extractor; populates SQLite via
  `wat-sqlite::execute`.
- A backfill driver that runs the indexer once over the existing
  ~80 memories.

**Tests:** post-indexing, the `memory` table has 80+ rows; the
`memory_axis` table has the expected number of axis values; one
end-to-end SELECT produces the expected memory's path.

### Slice 3 — Recall function (~2 days)

**Ships:**
- `wat/memory/recall.wat` — the recall function from
  `recall-protocol.md`. Takes a scope AST; encodes via slice 1's
  vocab; runs Sift query (Coincident constraint against the
  memory projections); returns `Vec<MemoryRef>`.
- `wat/memory/combinators.wat` — the helper combinators (`about`,
  `referencing`, `feedback-about`, `similar-to`, `like`).
- Hologram build at startup from SQLite (per `architecture.md`
  layer 3 — in-process; rebuilt from SQLite on startup; warm
  cache after).

**Tests:** known-answer queries — recall about wat-pry surfaces
the wat-pry scratch arc memory; recall about Hickey surfaces the
designer-pattern memories; recall by similar-to(known-memory)
surfaces structural neighbors.

**Hard dependency:** Slice 3 cannot ship without `wat-sift` and
the algebra layer's HologramStore both available.

### Slice 4 — wat-mcp tool publish (~1 day)

**Ships:**
- `wat/memory/main.wat` — the wat program that hosts the memory
  store as a long-running service.
- The MCP tool registration: `(:wat::memory::recall scope-edn n)`
  exposed via the wat-mcp single-tool surface.
- The tool wraps slice 3's recall function; deserializes the
  scope-edn argument back to a HolonAST; serializes the
  `Vec<MemoryRef>` return back to EDN for JSON-RPC.

**Tests:** end-to-end via an MCP client harness. JSON-RPC
request goes in; JSON-RPC response with refs comes out. Latency
acceptable for interactive use (target: < 100ms per recall on
the 80-memory corpus).

### Slice 5 — Live integration test (~1 day)

**Ships:**
- A test harness that spins up the wat-memory MCP server,
  connects a Claude session to it via stdio, runs a
  representative set of recall queries (drawn from real
  conversation patterns), and inspects the returned refs for
  subjective quality.
- Documentation of the deploy story: how the user (or a future
  CI) starts the wat-memory MCP server, where its SQLite lives,
  how it picks up memory file changes (lean: file mtime check
  on startup; reindex on mismatch).

**Tests:** subjective sanity check ("did the recall surface the
memories I'd want for this scope?") + latency benchmarks
+ smoke test for restart/reindex.

**Acceptance bar for the whole arc:** Slice 5's harness
demonstrates that a new Claude session connected to the memory
MCP server, given a representative scope, recalls the memories
that the prior-session-aware speaker would have recalled. The
discipline propagates through the substrate layer; the agent's
memory is no longer flat.

## Sequencing visualization

```
[Now]                                                 [Memory arc opens]
  │                                                          │
  ├─→ wat-mcp slice 1   (parallel)   ────────────────────────┤
  │                                                          │
  ├─→ wat-sift extract  (parallel)   ────────────────────────┤
  │                                                          │
  └─────────────────────────────────────────────────[memory arc starts]
                                                             │
   slice 1 (vocab+extractor) ──→ slice 2 (sqlite+indexer) ──→ slice 3 (recall+sift)
                                                             │
                                          slice 4 (mcp tool) ──→ slice 5 (live test)
```

Total path length from "now" to "shipping prototype":
- Prerequisites: ~5-10 days (parallel; longest path is wat-sift)
- Memory arc itself: ~7 days

If prerequisites and arc-1 of memory ship in parallel where the
slices permit, total ~10-14 days from arc opening to first
running prototype against the live MEMORY.md tree.

## What lands AFTER first prototype (not in initial scope)

- **Live memory updates.** Today's plan: indexer runs on startup;
  reindex on file mtime change. Live watcher is a follow-up.
- **Cross-project memory.** The current MEMORY.md is per-project
  (Claude's auto-memory is scoped to this holon project). A
  global memory layer is a different arc entirely.
- **Memory promotion / archive.** Old memories that no longer
  apply could be archived to a slow-tier hologram. Not in v1.
- **Multi-agent shared memory.** Multiple Claude sessions
  hitting the same MCP server is supported by construction
  (stateless stdio); shared-write semantics are a future arc.
- **Vocabulary growth.** The 4 first-cut axes extend to all 9
  candidate axes from `memory-vocabulary.md` as the recall
  surface needs them. This is incremental, not a slice.
- **Public X-shaped post.** The "MCP-shaped memory for Claude
  using my substrate" angle is a strong public follow-up to the
  DEFCON CFP. Lives outside the substrate arc itself.

## Status

- Sized at ~10-14 days from the moment prerequisites land.
- Five slices defined with explicit ships + tests per slice.
- Hard dependency: wat-sift + wat-mcp slice 1 must ship first.
- No new substrate primitives required beyond Sift; the rest is
  composition of already-shipped pieces (HologramStore,
  HashBundle, coincident?, wat-edn, wat-sqlite, wat-mcp).
