# Memory vocabulary — open problem #1

How does a markdown memory file lower to `Vec<Fact>`?

The substrate's hologram traversal needs each memory to project
onto the unit sphere as a vector. The vector is the bundle of
facts the memory's body + frontmatter encodes. The vocab decides
which dimensions the surface has — and the dimensions ARE the
recall surface.

Same shape as the trading lab's vocab modules (read structured
input, emit `Vec<Fact>`). The difference: the input here is
unstructured prose plus structured frontmatter, not a candle
window.

## What the vocab determines

The recall surface is the set of axes the hologram can
discriminate on. If the vocab only emits a single fact per memory
(*"this is a memory"*), every memory projects to the same
coordinate and recall is useless. If the vocab emits ten honest
axes, recall can find memories by their actual structural
similarity to the scope.

**The vocab is the model.** Same recognition the trading lab
arrived at: *"the vocabulary is the model. The discriminant is
learned. The flip is derived."* For memory, the vocabulary
determines what kinds of similarity are computable. *"Find
memories about wat"* works iff the vocab encodes a topic axis.
*"Find memories I added recently"* works iff there's a recency
scalar.

## First-cut axis proposal (5–10 axes)

These come from inspecting the existing 80+ memory files. Each
axis is named, sourced (frontmatter vs body extraction), and
sized for first-prototype scope.

### Axis 1 — Type (categorical)
- Source: frontmatter `type:` field (already present)
- Values: `user` / `feedback` / `project` / `reference` (the
  current set; extensible)
- Encoding: `Bind(:type-axis, Atom(:user|:feedback|:project|:reference))`
- Why it matters: the agent's recall almost always wants
  type-filtered results. *"What did I learn from him about
  wat?"* is `type=feedback` ∩ `topic=wat`.

### Axis 2 — Topic (set of categorical)
- Source: extracted from frontmatter `description:` and body
  text (proper nouns + named concepts)
- Values: open vocabulary; substrate's atoms grow with the
  vocabulary
- Encoding: `Bundle(Bind(:topic-axis, Atom(:wat)),
  Bind(:topic-axis, Atom(:trading)), ...)`
- Why it matters: the most common recall dimension. *"About X."*
- Open: extracted via what mechanism? See "Extraction methods"
  below.

### Axis 3 — Referenced handle (set of categorical)
- Source: extracted from body — proper nouns of people, repos,
  projects, AWS service names, language names
- Values: open vocabulary
- Encoding: `Bundle(Bind(:handle-axis, Atom(:watmin)),
  Bind(:handle-axis, Atom(:wat-rs)), ...)`
- Why it matters: distinct from topic. *"Memories that
  reference Hickey"* vs. *"Memories about Clojure."* The same
  Hickey reference might appear in memories about wat AND about
  the lab AND about the architecture — handle is the cross-
  cutting concern.

### Axis 4 — Reference type (categorical, optional)
- Source: extracted from body — does the memory name a file
  path? a URL? a commit hash? a person's handle?
- Values: `file-ref` / `url-ref` / `commit-ref` / `handle-ref`
  / `arc-ref` / `chapter-ref`
- Encoding: as Bind under `:ref-type-axis`
- Why it matters: the agent often recalls *"the memory that
  pointed at X file."* Distinct from topic and handle.

### Axis 5 — Recency (continuous scalar)
- Source: frontmatter `added_at` and `updated_at` (need to add
  these if not present; today's MEMORY.md doesn't carry them
  on every entry — see "Open problem: backfill" below)
- Encoding: `Bind(:recency-axis, Thermometer(epoch, oldest, now))`
- Why it matters: recency is a tiebreaker, not a primary index.
  When two memories are equally relevant by topic, the more
  recent one is usually preferred. The Thermometer encoding
  lets the cosine weight by recency naturally without hard-
  coding a recency cutoff.

### Axis 6 — Specificity (continuous scalar)
- Source: derived from body length + topic count; short and
  topic-narrow → high specificity; long and topic-broad → low
- Values: f64 in [0, 1]
- Encoding: `Bind(:specificity-axis, Thermometer(score, 0, 1))`
- Why it matters: when the agent's scope is narrow, prefer
  high-specificity memories. When the scope is broad, prefer
  low-specificity (overview) memories. Same shape as the trading
  lab's conviction-as-scalar discipline.

### Axis 7 — Relevance context (set of categorical)
- Source: frontmatter `description:` parsing for "when does
  this matter" cues (could be explicit `applies_to:` field)
- Values: open vocabulary; usually phrases like
  `during-build` / `during-debug` / `during-design` /
  `during-conversation` / `during-prompting`
- Encoding: as Bind under `:context-axis`
- Why it matters: distinct from topic. A memory about wat
  might apply during design (architectural recognition) or
  during build (a specific syntax gotcha). Different recall
  contexts.

### Axis 8 — Sentiment / valence (categorical, optional)
- Source: extracted (a positive recognition vs. a "do not do
  this" warning has different recall characteristics)
- Values: `positive` (worked, ship it) / `negative` (don't do
  this) / `neutral` (information)
- Encoding: as Bind under `:valence-axis`
- Why it matters: `feedback` memories often carry warnings
  (*"do not do X"*); the agent recalling about a domain wants
  to surface both the positive examples and the warnings.
  Without this axis, warnings risk being out-ranked by more
  abundant positive memories.

### Axis 9 — Hash of canonical body (opaque identity)
- Source: hash of the canonical EDN of the memory's full body
- Encoding: `Bind(:identity-axis, Atom(hash))`
- Why it matters: distinguishes memories that rhyme on every
  other axis but are different documents. Acts as the last-
  resort discriminator when the memory hologram has near-
  collisions.

That's nine axes. **Lean: ship slice 1 with axes 1, 2, 3, 5
(type, topic, handle, recency).** Those four are the most
reliably extractable from the existing memory files and cover
the most common recall shapes. The other five extend the
surface as the recall protocol's actual usage surfaces a need
for them.

## Extraction methods

Which mechanism extracts which axis? Three honest candidates:

**Mechanism A: Frontmatter declaration.**
The memory file declares the axes explicitly:

```markdown
---
name: feedback_no_pre_existing_excuse
description: When a test fails on your branch, investigate the rot
type: feedback
topics: [testing, accountability, branch-hygiene]
handles: [watmin]
applies_to: [during-debug, during-pr-review]
valence: negative
---

(body text)
```

**Pros:** explicit, deterministic, fast (no NLP needed),
debuggable.
**Cons:** requires backfill of existing 80+ memories with
topic/handle/applies_to/valence frontmatter; new memories
need discipline to declare the axes.

**Mechanism B: Body extraction via vocab-extractor wat program.**
A wat program reads the body, extracts named concepts (regex
for backtick-quoted symbols + spacy-style proper noun
extraction + whatever else), produces the axis values.

**Pros:** works on existing memories without backfill; new
memories don't need discipline.
**Cons:** harder to debug; harder to make deterministic; one
more wat module to maintain; quality depends on the extractor.

**Mechanism C: Hybrid — frontmatter wins, body extracts as
fallback.**
The memory file CAN declare axes in frontmatter (and they
override). When axes are missing, the extractor fills them.

**Lean: C.** New memories declare axes deliberately (the
discipline is good); existing memories get extracted on
first-pass index; either source produces the same axis values
for the hologram.

## Open problem: backfill of the existing 80+ memories

Today's `MEMORY.md` and the leaf files don't have `topics:` /
`handles:` / `applies_to:` / `valence:` frontmatter on every
entry. Slice 1 has to either:

- Run mechanism B (body extractor) over the existing tree once
  to populate the SQLite tables (so the hologram has axis
  values to work with from day one)
- Or accept that the first-pass hologram is type+recency-only
  on existing memories until they're hand-backfilled

**Lean: run the extractor.** The extractor is a useful primitive
regardless; running it once over the existing tree validates
the extraction logic and produces the immediate-bootstrap
hologram.

## What this implies for the SQLite schema

`storage.md` proposed three tables: `memory`, `memory_axis`,
`memory_link`. The vocab work confirms `memory_axis` is the
right shape — one row per (memory, axis_name, axis_value). All
nine candidate axes fit this shape; the only special-case is
the recency Thermometer (which stores as the raw epoch in
`memory.added_at` and the projection re-derives the Thermometer
on hologram build).

## Proposed first-cut commit

For slice 1 of the memory arc, the vocab module exposes:

```scheme
(:wat::core::define
  (:user::memory::extract-facts
    (memory-record :user::memory::Record)
    -> :wat::holon::Holons)
  (:wat::core::vec :wat::holon::HolonAST
    ;; Type axis (frontmatter)
    (:wat::holon::Bind :type-axis (memory-record/type memory-record))
    ;; Topic axis (set; extracted)
    (:wat::core::map (lambda (topic)
      (:wat::holon::Bind :topic-axis topic))
      (extract-topics (memory-record/body memory-record)))
    ;; Handle axis (set; extracted)
    (:wat::core::map (lambda (handle)
      (:wat::holon::Bind :handle-axis handle))
      (extract-handles (memory-record/body memory-record)))
    ;; Recency scalar
    (:wat::holon::Bind :recency-axis
      (:wat::holon::Thermometer
        (memory-record/added-at memory-record)
        oldest-epoch
        current-epoch))))
```

That's the four-axis minimum. The five other axes extend the
function as their extractors prove out.

## Status

- Vocab axes proposed; not yet implemented
- Extraction mechanism: lean on hybrid (mechanism C)
- First-cut covers 4 axes; extends to 9 as the recall protocol
  surfaces need
- Backfill of existing memories: extractor runs once at
  prototype build
- Schema: confirms `storage.md`'s three-table design
