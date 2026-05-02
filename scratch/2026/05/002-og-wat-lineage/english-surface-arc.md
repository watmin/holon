# English-surface arc — wat-english consumer crate

The English-like SVO statement surface from OG wat is the one
piece of the original spec that did NOT migrate to current wat
as language primitives. Current wat's substrate doesn't need it;
SVO is domain-specific. But it IS shippable today as a wat
consumer crate — `wat-english` — that publishes OG wat's surface
as macros over current Holon Bind/Bundle primitives.

This file sketches that consumer crate. Not opened as a real
arc. Lives here as the thought; sized so the user knows what
it would cost if they decide to ship it.

## What the crate publishes

A namespace `:wat::english::*` (workspace-member crate, claims
the substrate-tier prefix per CONVENTIONS.md's first-party
rules) with macros that desugar OG wat's primitives into Holon
algebra.

### Macros

```scheme
;; OG: (Statement subject verb object)
;; Compiles to: a Bundle with three Bind pairs on the role axes
(:wat::english::Statement subject verb object)

;; OG: (Statement subject verb object :adverb adv :time t :number n)
;; Compiles to: a Bundle with role pairs + modifier pairs
(:wat::english::Statement subject verb object
  :adverb adv :time t :number n)

;; OG: (every subject stmt)
;; Compiles to: a Bundle bearing :quantifier-axis = :every plus
;;   the binding for the bound variable + the inner statement
(:wat::english::every subject stmt)

;; OG: (some subject stmt)
(:wat::english::some subject stmt)

;; OG: (at time stmt)
;; Compiles to: a Bundle with the inner statement bound to
;;   :time-axis = the time atom
(:wat::english::at time stmt)

;; OG: (before t1 t2)
;; Returns a precedence Statement
(:wat::english::before t1 t2)

;; OG: (during stmt t1 t2)
(:wat::english::during stmt t1 t2)

;; OG: (between t1 t2)
;; Returns a Time entity representing the range
(:wat::english::between t1 t2)

;; OG: (that subject stmt)
;; Returns a subject qualified by a relative clause
(:wat::english::that subject stmt)

;; OG: (passive object verb subject)
(:wat::english::passive object verb subject)
```

### Atoms / role axes

The crate also publishes the role-axis atoms that OG wat
implicitly assumed:

```scheme
:wat::english::role-subject
:wat::english::role-verb
:wat::english::role-object
:wat::english::adverb-axis
:wat::english::time-axis
:wat::english::number-axis
:wat::english::quantifier-axis
:wat::english::quantifier-every
:wat::english::quantifier-some
```

These are atoms (deterministic vector projections); they get
bundled into the macros' expansions to mark the structural
roles.

### Type wrappers (sugar)

```scheme
;; OG: (Subject value)
(:wat::english::Subject value)
;; Expands to: (Bind :role-subject (Atom value))

;; Same shape for Object, Verb, Adverb, Time, Adjective, Pronoun
```

## What the crate does NOT do

- **No new substrate primitives.** Every macro expands to existing
  Holon `:wat::holon::Bind` / `:wat::holon::Bundle` /
  `:wat::holon::Atom` calls. The crate is pure macros + atom
  registrations.
- **No type system extension.** The OG trait system
  (Relatable / Adverbial / Timeable / etc.) is NOT re-published
  as substrate traits. The crate provides macros that produce
  well-formed Holon shapes; type discipline is at the macro
  level (the macro's parameters carry the type expectations).
- **No new evaluator semantics.** OG wat's `(assert ...)` is not
  in this crate — assertion is a substrate-level concept that
  current wat handles differently. The crate publishes
  STATEMENT CONSTRUCTION; truth-of-statement is the consumer's
  problem (which is the right architectural call).

## Consumer experience

A user who wants to write English-like wat:

```scheme
;; In their wat code:
(:wat::core::use! :wat::english)

(:wat::core::let* (((dog :wat::holon::HolonAST) (:wat::english::Subject "dog"))
                   ((toy :wat::holon::HolonAST) (:wat::english::Object "toy"))
                   ((chases :wat::holon::HolonAST) (:wat::english::Verb "chases"))
                   ((quickly :wat::holon::HolonAST) (:wat::english::Adverb "quickly"))
                   ((t-0 :wat::holon::HolonAST) (:wat::english::Time "t-0")))
  (:wat::english::Statement dog chases toy :adverb quickly :time t-0))
;; -> a Holon Bundle with all six bindings on the appropriate axes;
;;    cosineable, presence-able, coincident-able with anything else
;;    on the substrate
```

The Lisp on Rust hosts the English-like Lisp on top.

## Shape of the work (slices)

If opened as a real arc:

### Slice 1 — Atom registration + Subject/Object/Verb wrappers
- Register the role-axis atoms in the substrate's symbol table
  via the crate's `register()` function
- Publish the type-wrapper macros (`Subject`, `Object`, `Verb`,
  `Adverb`, `Time`, `Adjective`, `Pronoun`)
- Tests: each wrapper produces the expected Holon Bind shape

### Slice 2 — Statement macro
- Publish `:wat::english::Statement` with its `:adverb` / `:time` /
  `:number` keyword arguments
- Tests: a Statement compiles to the expected Bundle of Binds;
  cosine between two structurally similar Statements is high

### Slice 3 — Temporal forms
- Publish `at`, `before`, `during`, `between`, `after`
- Tests: temporal forms produce well-formed Bundles; nested
  temporal forms compose

### Slice 4 — Quantification + relative clauses
- Publish `every`, `some`, `that`
- Tests: quantification forms carry the quantifier-axis binding;
  relative clauses qualify their subject correctly

### Slice 5 — Passive voice + tests against OG wat's full example
- Publish `passive`
- End-to-end test: the OG wat full example
  (the big-dog-chases-toy-at-t-0 walk) compiles and produces
  a substrate-recognizable Holon

Total: ~1 week of focused work; could ship as a community
crate (under `:user::watmin::wat-english::*`) or as a first-party
workspace member (under `:wat::english::*`). The first-party
status would be honest — the user IS the one who originated the
design — but `:user::*` is the safer convention until the crate
proves itself with real consumers.

## Why this is interesting

Three reasons:

1. **Public-narrative completion.** Shipping wat-english is the
   visible closure of the *"three years of haunting"* arc. The
   user can post: *"I designed a Lisp years ago. The substrate
   to host it didn't exist yet. I built the substrate over the
   last three months. Today I shipped the original design as a
   consumer crate on top of the substrate. Same Lisp; different
   layer."* That post writes itself.

2. **The strange loop closure.** OG wat had English-like
   statements as a top-level surface; current wat has Holon
   algebra at the top level; wat-english publishes English-like
   statements again, on top of the algebra. The recursion is
   correct — the SUBSTRATE got the algebra; the SURFACE got the
   English. Both layers exist; the user moves between them as
   the work demands.

3. **Demonstrates the wat-as-platform thesis.** The DEFCON
   submission's claim is that anyone whose native cognition
   mismatches the platform's demanded paradigm can build a
   language to think in. Shipping wat-english on top of wat IS
   that thesis instantiated twice — once at the wat-on-Rust
   layer, once at the wat-english-on-wat layer. Same move; two
   levels.

## Why NOT to ship it now

- The DEFCON CFP is in flight; the next ~3-6 weeks are about
  waiting for the response, not opening new arcs.
- Memory-as-hologram (`scratch/2026/05/001-memory-as-hologram/`)
  is the OTHER recognition that came out of the post-CFP burst,
  and it's structurally more interesting (a substrate-tier
  capability with an MCP delivery).
- wat-english is a capstone, not a substrate move. Ship it after
  the substrate has matured a little more (wat-mcp, wat-sift,
  the memory layer).

## Status

- **Recognized:** 2026-05-01 in this scratch arc.
- **Sized:** 5 slices, ~1 week of focused work.
- **Not opened:** lives here as the shape; opens when the user
  decides this is the next arc.
- **Reading list for that opening:** og-wat-spec.md (the full
  OG spec) + analysis.md (what survived / transformed / latent).
