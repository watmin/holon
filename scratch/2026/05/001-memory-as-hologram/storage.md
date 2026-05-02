# Storage

The user's mid-scratch refinement (2026-05-01):

> "we need to identify the storage.... sqlite?.... how can we
> pull this off with an interface that could be a remote db?.....
>
> like... we use edn forms... in sqlite... we use the telemetry
> query tooling?... we can do smart traversal of forms and then
> do work on them?....
>
> a time index isn't likely the thing... maybe there's a few
> tables at work here.. and they compose with file system refs
> for actual content to utilize.....
>
> we ask the mcp which file i should read... that's the
> interface?..."

That settles every storage question that was open in
`architecture.md`. SQLite for structure. EDN payloads in SQLite
rows. The telemetry-query tooling pattern from arcs 080-088
applies. Multiple tables that compose. Time is not the primary
index. File system holds content; SQLite holds refs and
projections. MCP returns refs.

This file is the deeper design pass.

## Why SQLite specifically

Three reasons, each load-bearing:

**1. The substrate already ships it.** wat-sqlite (arc 083) is a
first-party workspace crate. `Sqlite/auto-spawn` (arc 085) reflects
on a consumer enum and derives schemas + INSERTs + binders. The
trading lab uses this for its run DB. Memory-as-hologram inherits
the proven pattern; no new substrate work required.

**2. EDN payloads compose.** Arc 086 shipped `wat::edn::read`
round-trip via tag dispatch through the SymbolTable. Any HolonAST
that compiles in the user's wat program serializes to EDN and
back. SQLite holds the EDN as a TEXT column; the wat side
deserializes on read.

**3. SQLite is remote-able.** A SQLite database file can be:
- local (default; `runs/memory.db`)
- replicated by Litestream / Turso for read-replicas
- replaced wholesale by libsql (SQLite-protocol-compatible remote
  DB) without changing the SQL or the schema
- replaced by a Postgres adapter at the wat-sqlite layer, if the
  substrate ever surfaces that abstraction (currently it doesn't,
  but the schema is portable)

The user named the remote-DB question explicitly. SQLite + libsql
solves it without forcing a new architectural commitment today.

## Why "a few tables that compose" rather than one table

The user's framing rejects the antipattern of *"one big memories
table with a TEXT column called body."* That table grows
linearly in row count and indexes badly. The decomposition the
substrate prefers:

```
memory                    -- the canonical record
  id            INTEGER PRIMARY KEY
  name          TEXT     -- the frontmatter name (uniqueness key)
  type          TEXT     -- user / feedback / project / reference
  path          TEXT     -- file system ref (relative to memory root)
  hook          TEXT     -- one-line description from MEMORY.md
  added_at      INTEGER  -- unix epoch (metadata, NOT primary index)
  updated_at    INTEGER  -- unix epoch (metadata, NOT primary index)
  projection    TEXT     -- canonical EDN of the memory's HolonAST projection
                         -- (the lowered Vec<Fact> bundle, encoded)

memory_axis               -- one row per (memory, axis-name, axis-value)
                          -- the searchable structural decomposition
  id            INTEGER PRIMARY KEY
  memory_id     INTEGER  REFERENCES memory(id)
  axis_name     TEXT     -- e.g. "topic", "type", "referenced-handle"
  axis_value    TEXT     -- the value at that axis (atom name)

memory_link               -- declared cross-references between memories
                          -- the "see also" graph
  id            INTEGER PRIMARY KEY
  from_memory   INTEGER  REFERENCES memory(id)
  to_memory     INTEGER  REFERENCES memory(id)
  link_type     TEXT     -- "supersedes" / "refines" / "depends-on" / etc.
```

Three tables. Compose by foreign key. The `memory` table is the
canonical record; `memory_axis` is the decomposition that the
hologram builds against; `memory_link` is the optional graph
overlay.

A query like *"give me all feedback memories about wat that
reference watmin's prior session"* becomes:

```sql
SELECT m.path, m.name, m.hook
FROM memory m
JOIN memory_axis a1 ON m.id = a1.memory_id AND a1.axis_name = 'type' AND a1.axis_value = 'feedback'
JOIN memory_axis a2 ON m.id = a2.memory_id AND a2.axis_name = 'topic' AND a2.axis_value = 'wat'
JOIN memory_axis a3 ON m.id = a3.memory_id AND a3.axis_name = 'referenced-handle' AND a3.axis_value = 'watmin';
```

This SQL is the *exact-match* query path. The hologram's
`coincident?` query is the *fuzzy-match* query path. Both run
against the same SQLite tables; they just use different
mechanisms (SQL JOIN vs. cosine on the `projection` EDN
materialized into the hologram).

## How the schema is auto-derived (arc 085 pattern)

The user-side wat code declares a `MemoryRecord` enum (or struct,
post-arc-085 typing). The substrate's `Sqlite/auto-spawn` reflects
on it:

```scheme
(:wat::core::struct :user::memory::Record
  ((name        :wat::core::String)
   (type        :user::memory::Type)
   (path        :wat::core::String)
   (hook        :wat::core::String)
   (added-at    :wat::core::i64)
   (updated-at  :wat::core::i64)
   (projection  :wat::holon::HolonAST)))

(:wat::core::struct :user::memory::Axis
  ((memory-id   :wat::core::i64)
   (axis-name   :wat::core::String)
   (axis-value  :wat::core::String)))

(:wat::core::struct :user::memory::Link
  ((from-memory :wat::core::i64)
   (to-memory   :wat::core::i64)
   (link-type   :wat::core::String)))

(:wat::sqlite::Sqlite/auto-spawn
  "memory.db"
  (:wat::core::vec :wat::core::Type
    :user::memory::Record
    :user::memory::Axis
    :user::memory::Link)
  reporter
  metrics-cadence)
```

The substrate derives:
- The `CREATE TABLE` statements
- The INSERT prepared statements with typed Param binders
- The SELECT scaffolding (consumer writes the WHERE clauses)
- The HolonAST round-trip for the `projection` column via
  `wat::edn::write` / `read`

This is the same shape the trading lab's run DB uses. Memory is
the second consumer of the same substrate primitive.

## Time as metadata, not index

Per the user's direction. `added_at` and `updated_at` are
columns; they are NOT the primary lookup. Reasons:

- **The agent recalls about X, not about recent.** A six-month-old
  memory about wat-rs arc 023 is more useful for arc 023 work
  than yesterday's memory about something else.
- **Recency-decay is a property of the discriminant, not the
  index.** If recency matters for a particular query, the
  recall-protocol can construct a query AST that *bundles*
  recency-axis facts with topic facts. The cosine then naturally
  weights by both.
- **Time-indexing seduces the agent into temporal narratives.**
  *"Show me my last 10 memories"* is a query that produces a
  story, not an answer. Memories are points on a sphere, not
  events on a timeline.

Time stays available for filtering when explicitly asked.
*"Project memories about wat that I added this week"* is a real
query. But the SELECT path uses the explicit `WHERE
added_at > N`, not a primary-index scan.

## File system refs vs. embedded content

The `memory.path` column points at the .md file in the original
memory tree. The agent reads the file with its standard file
ops after the recall returns. The MCP doesn't ship body bytes;
the agent reads what the MCP tells it to read.

Why this split:
- The .md file is human-readable, version-controllable, editable
  by hand. Keeping the canonical content there preserves the
  ergonomics of the existing system.
- The DB doesn't have to hold the body. Bodies are large and
  duplicated wouldn't add value.
- The MCP response stays small (refs + minimal metadata). One
  recall is one MCP call; the call returns a list of file paths.
- Updates to a memory file flow naturally: edit the .md, re-run
  the indexer (new SQLite row), the next hologram rebuild picks
  it up.

The `memory.projection` column DOES hold a serialized form — but
the projection is the lowered HolonAST (the
`Vec<Fact>`-bundled vector representation), not the body text.
The projection is small; the body is large.

## Remote DB shape

The user named the question. The answer in three pieces:

**1. SQLite-protocol compatibility.** wat-sqlite uses rusqlite under
the hood; rusqlite supports SQLite, libsql (Turso's
SQLite-compatible network DB), and Litestream-replicated SQLite.
The DB connection string is the only thing that changes.

**2. Schema portability.** The auto-derived schema is plain SQL;
it ports to Postgres / MySQL / any RDBMS that supports TEXT and
INTEGER. If a future arc surfaces a Postgres adapter at the
wat-sqlite-namespace level, the wat code reads `:wat::sqlite::Db`
the same way regardless of the backend.

**3. The hologram is per-process.** Each consumer process builds
its own hologram from the (possibly remote) SQLite. The hologram
itself doesn't need to be remote — multiple agents can share the
same backing DB and each compute their own in-process hologram.
For very large memory trees the hologram would chunk by domain
(per-project memory holograms) or by tier (frequent vs. archive).

For the first prototype, default is local SQLite + local file
system + local wat program hosting the MCP. Remote shapes are
opt-in by config, not a different architecture.

## Telemetry-query tooling for "smart traversal"

The user's phrase: *"we can do smart traversal of forms and then
do work on them."* The substrate's existing pattern from arc 080
(telemetry::Service<E,G>) plus arc 085 (Sqlite/auto-spawn)
already supports this. A consumer can:

- Read all memories matching a SQL filter
- Materialize the EDN-stored `projection` column back into
  HolonASTs
- Run wat algebra over them (cosine, presence?, coincident?,
  unbind, custom predicates)
- Return the result set to the MCP caller

This composes the SQLite query path (exact match, fast) with the
HolonAST algebra (semantic match, expressive). The agent picks
which to use based on the query shape — *"all feedback memories
that mention X"* is SQL; *"memories that feel related to my
current scope"* is hologram cosine.

## What's still open

- **Schema versioning.** The `MemoryRecord` enum will evolve.
  The substrate's auto-derived schema migration path is not yet
  named; arc 085 ships the initial-schema derivation but updates
  are an open question. Probably the same migration pattern any
  consumer of wat-sqlite would adopt.
- **Index updates on file changes.** Editing a memory file
  on disk requires re-running the indexer. Detection: file mtime
  vs. SQLite `updated_at`. Reindex on mismatch. Could be
  on-startup (cheap) or via a watcher process (more responsive,
  more code).
- **Concurrent writes.** Multiple agents writing to the same
  remote DB need either single-writer enforcement or proper
  conflict resolution. SQLite's default is single-writer;
  libsql relaxes this; Postgres-shaped adapters bring real
  multi-writer concurrency. First prototype: single writer.
  Future: open question.
- **Reindex granularity.** Does adding one new memory rebuild
  the whole hologram or update one slot? Substrate-wise, the
  HologramStore supports incremental updates (arc 074); the
  per-slot update is the right answer.
