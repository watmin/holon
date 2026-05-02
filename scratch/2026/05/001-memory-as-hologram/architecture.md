# Architecture

The system has four layers, each composing a substrate primitive
that already ships. The user's storage refinement (2026-05-01,
mid-scratch) settled the layering cleanly: **content lives on the
file system; structure and index live in SQLite as canonical EDN
forms; the hologram is built from the SQLite-resident projections;
the MCP tool returns file refs, not content.**

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4 — MCP delivery                                     │
│  wat-mcp tool: (recall scope-ast n) -> Vec<MemoryRef>       │
│  Returns FILE PATHS plus minimal metadata, not file bytes.  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — Hologram traversal                               │
│  HologramStore<MemoryProjection> on the substrate.          │
│  coincident? selects top-N matches against the scope.       │
│  In-process; rebuilt from SQLite at startup.                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — SQLite (structure + index, queryable, REMOTE-ABLE)│
│  wat-sqlite tables, EDN-formatted payloads. Multiple tables │
│  composed by foreign-key references. NOT time-indexed —     │
│  semantic structure is the primary key. Same telemetry query│
│  tooling the trading lab uses for its run DB.               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — File system (content, large, opaque to query)    │
│  The original .md memory files stay where they are.         │
│  SQLite holds path refs; file system holds bytes.           │
└─────────────────────────────────────────────────────────────┘
```

Read top-down (interface → content) or bottom-up (storage →
delivery). Each layer can be replaced without touching the others;
the substrate already enforces that discipline elsewhere.

## Why this layering matters

**Content stays on the file system because content is large.** A
memory file has frontmatter (a few hundred bytes) and a body
(potentially several KB of prose). Loading every body into a
HologramStore at startup is wasteful — the hologram works on the
*projection* of the memory, not the body itself. The body matters
only when the agent decides to actually read the matched file.

**Structure lives in SQLite because structure needs to be queried
from anywhere.** A wat-sqlite table is queryable by any process
with database access, including a remote one. The SQLite file can
sit on disk locally (default) or be replaced with a remote-DB
adapter (Postgres-shaped, Turso-shaped, libsql-shaped) without
the hologram or MCP layers caring.

**The hologram lives in-process because the cosine math is fast
when local.** Building the hologram once at startup from the
SQLite projection rows, then keeping it warm in memory, lets
recall be one cosine. Persistence + queryability stay in SQLite;
recall speed stays in the in-process hologram.

**The MCP returns refs because the MCP should be cheap.** Every
recall is one MCP call; the response is a list of file paths
plus minimal metadata (memory name, type, one-line description).
The agent loads the matched files itself with its existing file
read tooling. The MCP doesn't have to ship body bytes; the agent
doesn't have to invent a new "memory content" type.

## The MCP interface, named

```
Tool name: wat-eval (the existing wat-mcp single tool from scratch 006)
Inside the agent's call: (:wat::memory::recall scope-edn n)

Argument scope-edn: an EDN-serialized HolonAST representing the
                    agent's current recall query. Constructed
                    deliberately by the agent (recall-protocol.md
                    holds the design conversation on this).

Argument n:         top-N count. Default 3-5.

Return value: a wat Vec of MemoryRef structs. Each ref has:
  - path     :String  (relative to the memory root; agent reads
                       with its own file ops)
  - name     :String  (the memory's frontmatter name)
  - type     :Enum    (user / feedback / project / reference)
  - hook     :String  (the one-line description from MEMORY.md)
  - score    :f64     (the cosine similarity, for sorting / debug)
```

The agent receives the Vec, picks which files to actually load
(probably all of them for top-3; maybe filtered for top-5+), reads
them with its standard file-read tooling, and continues the
conversation with the loaded context.

This is the "which file should I read" interface the user named.
The MCP doesn't recall content; it points at content. The agent
reads what it points at.

## The substrate primitives this composes from

Every layer is built from a primitive that already ships in
wat-rs or holon-rs. Nothing new at the substrate level.

| Layer | Primitive | Source arc |
|---|---|---|
| MCP delivery | `wat-eval` MCP tool | `scratch/2026/04/006-wat-mcp/` (designed) |
| Hologram traversal | `HologramStore<V>` + `coincident?` | wat-rs arcs 074, 023, 024 |
| Hologram traversal | `eval-coincident?` family (for AST equivalence on demand) | wat-rs arc 026 |
| SQLite layer | `wat-sqlite` crate + `Db` primitives | wat-rs arc 083 |
| SQLite layer | `execute` + `Param` enum (typed parameter binding) | wat-rs arc 084 |
| SQLite layer | `Sqlite/auto-spawn` reflects on consumer's enum | wat-rs arc 085 |
| EDN payload | `wat::edn::write` / `read` round-trip | wat-rs arcs 079, 086 |
| Service shape | `wat::telemetry::Service<E,G>` | wat-rs arc 080 |
| Service shape | `Reporter` + `MetricsCadence` contract | wat-rs arc 078 |
| File I/O | `IOWriter::open-file` etc. | wat-rs arc 088 |

The arc isn't building a new substrate; it's building a
**second consumer of the substrate the trading lab already uses**.
The trading lab's run DB writes telemetry events as auto-derived
SQLite schemas with EDN payloads. The memory hologram does the
same shape against memory projections.

## What "second real consumer" means

The trading lab is the first real consumer of the substrate.
Memory-as-hologram becomes the second. That matters because:

- Two consumers force the substrate to generalize honestly.
  Whatever pattern the lab forced into wat-sqlite, the memory
  layer will probe again — and any place it doesn't fit, the
  substrate gets sharpened.
- The same telemetry-query tooling now answers two questions:
  *"what did the trader observe?"* (lab's run DB) and *"what
  does the agent know?"* (memory hologram). Same pattern, two
  domains. Same six primitives shape recurs at the application
  layer.
- A future third consumer (MTG state? truth engine?) inherits
  the proven pattern.

The arc is not just "build memory for Claude." It is "operate the
substrate in a second domain so the substrate proves it
generalizes."

## What's NOT in this architecture

Some moves the user explicitly ruled out or that the design
deliberately excludes:

- **Time as the primary index.** Time is metadata on a memory
  (added/updated dates), but the recall surface is semantic, not
  chronological. The user named this directly. *"A time index
  isn't likely the thing."* Right call — the agent recalls *about
  X*, not *recent.*
- **One big SQLite table.** Multiple tables compose by foreign
  key. Schema design lives in `storage.md`.
- **Body content in the database.** Bodies stay on disk; SQLite
  rows hold path refs.
- **Embedding-vector storage.** This is a VSA hologram, not an
  embedding-vector store. The substrate's own algebra is the
  retrieval mechanism; no neural embedding layer required.
- **Special MCP recall primitive.** The recall is just another
  wat call inside the MCP-hosted program. The MCP surface stays
  at one tool (`wat-eval`); the recall function is wat-side.
- **The hologram cached to disk.** The hologram is built at
  startup from SQLite. Cached eagerly in memory; never serialized
  back. SQLite is the source of truth; the hologram is the
  in-process index.

## Reading order of the rest of this scratch

- `storage.md` — the deeper dive on the SQLite + file system layering. Schema candidates. Remote-DB adapter shape. The user's "few tables that compose" principle worked through.
- `memory-vocabulary.md` — open problem #1: how a markdown file lowers to `Vec<Fact>`.
- `recall-protocol.md` — open problem #2: what scope-AST the agent constructs.
- `slice-plan.md` — five proposed slices.
- `open-questions.md` — everything not settled.
