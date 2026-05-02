# Scratch arc — memory as hologram

The wat substrate already has every primitive needed to host
content-addressed agent memory on a hypersphere. The auto-memory
system Claude currently uses (a flat `MEMORY.md` index pointing
at ~80 leaf markdown files) is an Index pattern that scales
linearly in cognitive load and gets brittle past ~100 memories.

The proposed pattern is the **Lattice** the wat machine is built
on: each memory is a coordinate; recall is a cosine; the entrypoint
is geometric, not enumerative; capacity scales by `√d` per
HashBundle layer; the agent walks the hologram instead of grepping
the index.

Delivered as an MCP tool, this becomes substrate-native memory for
**any** frontier LLM that speaks MCP — not just Claude. wat-mcp's
"one tool, EDN payload, agent talks wat directly" architecture is
the perfect delivery vehicle.

The recognition emerged on **2026-05-01**, the night the DEFCON
submission shipped, in the conversation thread that produced the
submission package itself. Captured here so the recognition
persists past the conversation's compaction.

---

## Files in this arc

| File | What it captures |
|---|---|
| `INDEX.yaml` | Beat-by-beat capture, conventions, status. Read this first if you want the meta-shape. |
| `README.md` | This file. Top-level orientation. |
| `the-recognition.md` | The user's framing, verbatim. The connection to existing substrate beats. The strange-loop nature of agents using wat to host their own memory of working on wat. |
| `architecture.md` | The proposed system shape. Memory as AST. HologramStore as the cache. Recall as one cosine. Persistence via canonical EDN. Inventory of existing substrate primitives the design composes from. |
| `memory-vocabulary.md` | Open problem #1: how does a markdown memory file lower to `Vec<Fact>`? What axes does the surface have? Five-to-ten named axes is the first-cut shape. |
| `recall-protocol.md` | Open problem #2: what's the *current scope* the agent encodes for the cosine query? Plausible candidates and tradeoffs. |
| `slice-plan.md` | Five proposed slices to first running prototype. Sequencing relative to wat-mcp's main shipping arc. The `feedback_no_pre_existing_excuse` discipline applied to memory writes. |
| `open-questions.md` | Everything not yet settled, named with leans where the design has them. Nothing here blocks opening the wat-rs arc when the work surfaces. |

## Locked decisions

Five load-bearing decisions banked from the recognition conversation:

1. **Memories ARE HolonASTs.** Not "represented as." The frontmatter is a struct; the body is text that gets lowered through a memory-vocab encoder. The file IS the AST.
2. **Storage is HologramStore<MemoryAST>.** Arc 074's substrate-shipped primitive. The index dissolves; the hologram replaces it.
3. **Recall is one cosine.** The agent encodes its current scope; the substrate returns top-N coincident? memories. Top-N is configurable; default likely 3-5.
4. **Persistence is canonical EDN on disk.** Round-trip via `:wat::edn::write` / `:wat::edn::read` (arcs 079, 086). One file per memory; the file IS the AST in serialized form.
5. **MCP is the delivery vehicle.** wat-mcp's "one tool" architecture publishes `(:wat::memory::recall scope-edn n) -> Vec<MemoryAST-as-EDN>` as the agent-facing surface. No special MCP primitive; the recall is just another wat call inside the running program.

## Status

**Open.** The recognition is captured. The architecture is sketched. Open
problems are named. No substrate work has shipped yet.

The arc opens for real when:

- wat-mcp's main shipping arc has at least slice 1 landed (the basic
  MCP eval surface) — this arc rides on it as the first real consumer
- Or the user decides this is more interesting than wat-mcp and
  promotes it ahead of the current sequence

Either way, the persistence layer holds the recognition until then.
