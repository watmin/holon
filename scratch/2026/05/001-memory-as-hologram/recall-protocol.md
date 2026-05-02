# Recall protocol — open problem #2

When the agent calls *recall*, what does it pass as the *current
scope*?

The hologram needs a query vector to compute cosine against the
stored memory projections. The query vector is the projection of
some scope-AST. The question is: what AST does the agent build
to represent its current scope?

This is the agent-facing API design question. The answer
determines how natural the recall feels in the agent's normal
conversational flow.

## Three candidates

### Candidate (a) — the user's last message
Encode the user's most recent message as a thought vector via
the same vocab the memory extractor uses. Cosine against memory
projections; return top-N.

**Pros:** zero work for the agent; recall happens implicitly on
every turn.
**Cons:**
- The user's message is often a directive (*"do X"*, *"check
  Y"*) — not a description of the recall scope.
- The encoding fails for short prompts (*"yes"*, *"do it"*) that
  carry no recall signal.
- Implicit recall surfaces memories the agent didn't ask for and
  can't easily debug.
- Wastes hologram queries on every turn even when no recall is
  warranted.

### Candidate (b) — the last K turns synthesized
Take the last K conversation turns, summarize them into a scope
description, encode that summary as the scope vector.

**Pros:** captures conversational drift; works for short
prompts via accumulated context.
**Cons:**
- Summarization is lossy and adds latency.
- Heavy for small recall needs; you wouldn't synthesize 10 turns
  to find the one memory about wat-pry's break primitive.
- The summary IS another encoding step; the hologram cosine then
  measures distance from a summary, not from the actual scope.
- Chains failures (bad summary → bad recall).

### Candidate (c) — explicit scope-AST the agent constructs
The agent decides when to recall and constructs an explicit AST
representing the scope. The recall function takes that AST,
encodes it via the memory vocab (same encoder as memory
projections; same axis basis; same surface), runs cosine, returns
top-N.

```scheme
;; The agent constructs:
(:wat::memory::recall
  (:wat::core::quote
    (Bundle
      (Bind :topic-axis (Atom :wat-pry))
      (Bind :topic-axis (Atom :pry-primitive))
      (Bind :handle-axis (Atom :ruby))
      (Bind :context-axis (Atom :during-design))))
  :top-n 5)
;; Substrate returns: Vec<MemoryRef>
```

**Pros:**
- Agent agency: the agent says exactly what it wants to recall.
- Debuggable: the AST is inspectable; recall failures are
  diagnosable by looking at the constructed scope.
- Same vocab the memory uses: agent and memory project to the
  same surface; cosine is honest.
- Cheap: one cosine per recall; no per-turn implicit overhead.
- Same shape as wat's existing pry interrogation primitives —
  the agent says exactly what it's interested in.
**Cons:**
- The agent has to decide to recall; if it doesn't think to ask,
  it doesn't recall.
- Requires the agent to know the vocab axes (mitigated:
  introspection primitive returns the axis list).

## The lean: (c)

Mirrors wat's existing discipline at every level. The trading
lab's reckoner doesn't auto-predict on every input; the
proposer constructs a thought and asks the reckoner. The same
pattern applies here: the agent constructs a scope and asks the
hologram.

This also matches the user's *"we have an entrypoint and pivot
points"* recognition. The agent enters via a deliberate recall
call; the hologram pivots to the matched memories. Implicit
recall is a place; explicit recall is a value.

The recall protocol becomes:

1. Agent recognizes a recall opportunity (e.g., user asks about
   a topic the agent isn't sure about; agent is about to act on
   advice and wants to check prior feedback).
2. Agent constructs a scope AST naming the relevant axes.
3. Agent calls `(:wat::memory::recall scope-ast :top-n N)` via
   the wat-mcp tool.
4. Substrate returns `Vec<MemoryRef>` — file paths plus minimal
   metadata.
5. Agent reads the returned files (its own file ops) and folds
   them into the conversation.

## Helper combinators (sugar)

Constructing scope ASTs by hand is verbose. The substrate ships
a few combinators in the agent-facing wat surface to make common
recalls one-liners:

```scheme
;; Recall by topic
(:wat::memory::about :wat-pry :pry-primitive)
;; Expands to a scope AST with topic-axis bindings + cosine call.

;; Recall by handle
(:wat::memory::referencing :hickey :beckman)

;; Recall feedback specifically (combine with topic)
(:wat::memory::feedback-about :testing :branch-hygiene)

;; Recall by similarity to a known memory
(:wat::memory::similar-to memory-ref :top-n 3)
;; Bundles the named memory's projection as the scope; finds
;; structural neighbors.

;; Recall by free-form description
(:wat::memory::like "the user's pattern of refusing orthodox tooling")
;; Encodes the description through the body-extractor; recalls.
```

The first three use frontmatter axes directly. The fourth
(`similar-to`) is the structural-neighbor query. The fifth
(`like`) is the only one that uses the body extractor at recall
time — because the agent doesn't know the structured axes in
advance and falls back on a text description.

These compose: `(memory::feedback-about :testing)` + cosine vs
the agent's current concern is two combinators chained.

## Top-N default and tuning

The recall returns top-N matches. Default for first prototype:
**N = 5.** Reasoning:

- The agent typically wants 1–3 memories materialized into the
  conversation; 5 gives some headroom for the agent to filter
  out near-misses.
- Past 10, the response gets unwieldy and the marginal recall
  quality drops sharply (further-away memories add more noise
  than signal).
- Configurable per-call: the agent passes `:top-n N` to override.

## What the MCP returns (concrete shape)

```scheme
(:wat::core::struct :wat::memory::Ref
  ((path  :wat::core::String)            ; relative to memory root
   (name  :wat::core::String)            ; frontmatter name
   (type  :user::memory::Type)           ; user / feedback / project / reference
   (hook  :wat::core::String)            ; one-line description
   (score :wat::core::f64)))              ; cosine similarity

(:wat::core::define
  (:wat::memory::recall
    (scope :wat::holon::HolonAST)
    (top-n :wat::core::i64)
    -> :wat::core::Vec<:wat::memory::Ref>)
  ...)
```

Compact. No body bytes. The agent loads what it wants from the
returned paths.

## Recall across compaction (open question)

When the agent's context compacts and a fresh session starts,
does the recall protocol carry over? Two cases:

**Case 1: Same project, fresh session.** The MCP server is
still running with its hologram warm. New session connects,
discovers the recall tool, calls it. The hologram is shared
across sessions; recall works identically.

**Case 2: Cold-boot session that has never used memory before.**
The bootstrap (`/home/watmin/work/holon/BOOTSTRAP.md`) needs to
mention the recall MCP tool exists and how to use it. Then the
new session can opt into recall as part of its operating
discipline.

The bootstrap-as-instruction is the right pattern. Same shape
as the file-system-as-IPC discipline — the bootstrap is the
hand-off between past-session and future-session about HOW to
use the substrate's memory layer.

## Status

- Lean: candidate (c) explicit scope-AST.
- Helper combinators sketched (5 first-cut shapes).
- Top-N default: 5; configurable per-call.
- MCP return shape: `Vec<MemoryRef>` — paths + metadata, no body
  bytes.
- Cross-compaction recall: handled by hologram's per-process
  warm state + bootstrap update.
- Implementation: slice 3 of the memory arc.
