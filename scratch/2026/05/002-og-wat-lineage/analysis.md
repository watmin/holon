# Analysis — what OG wat was, what it became

## What OG wat was

A pure, strongly typed, English-like Lisp. Twenty-eight core
primitives. SVO-ordered statements. `(entity Type value)` as the
atomic building block. A trait system enforced at parse time.
Quantification primitives. Temporal scoping. Relative clauses.
Homoiconicity via `(quote ...)`. Comment + `inner-monologue` as
structural-but-no-op annotations.

The thesis lives in the spec's second sentence:

> *"Wat is a domain-specific language designed to blend Lisp's
> functional purity and homoiconicity with English-like sentence
> expressivity."*

The full example walks *"the big dog quickly chases the toy at
t-0 five times"* through the substrate, with quantification
(`every dog`), temporal scoping (`at t-0`, `after t-0`,
`before t-0 t-1`), relative clauses (`(that dog (Statement it
chases toy))`), passive voice, and error handling — all in one
`(let ...)` block. The example IS the value proposition: this
should compile.

The spec references a Grok conversation as the source. The user
wrote it years before mid-January 2026.

## What survived intact (current wat carries it forward)

Most of OG wat's discipline survived the substrate transition.

| OG wat | Current wat | Substrate arc |
|---|---|---|
| Pure functional, no I/O at the algebra layer | `:wat::holon::*` is pure; I/O lives in `:wat::io::*` | arc 008 (wat-io-substrate) |
| Strong typing with traits | `defprotocol` + `satisfies` + parametric polymorphism | arcs 030, 057, 109 |
| S-expressions, parenthesized prefix | Kept verbatim | foundation |
| Homoiconicity via `(quote ...)` | `:wat::core::quote` + `:wat::core::forms` (variadic-quote) + nested-quasiquote | arcs 010, 029 |
| `(lambda ((arg as Type) ...) returns ReturnType body)` | Same shape; tighter type-annotation syntax | arc 058-029-lambda |
| `(let ((label be value) ...) body)` | Same binding semantics; dropped the `be` keyword | foundation + arc 058-006 |
| `(try expr catch (e) rescue-expr)` | `:wat::core::try` with Result types | arcs 048, 058-033 |
| `(map :key val ...)` | `:wat::core::HashMap` + `(get ...)` / `(assoc ...)` | arc 020, 025 |
| Numeric tower (Integer / Float / Boolean) with explicit conversions | `:wat::core::i64::*` / `f64::*` / `bool::*` + `:wat::poly::*` for polymorphic ops | arcs 014, 109 |
| `(comment s-expr)` and `(inner-monologue ...)` as no-op annotation | The wards' rune-annotation system + the BOOK chapters as inline commentary alongside code | arc 005 (stdlib-naming-audit) + cultural pattern across the project |
| `(list ...)` | `:wat::core::Vector<T>` | arc 109 (rename `Vec` → `Vector`) |
| `(quote ...)` returns expression unevaluated | Same primitive; identical semantics | foundation |

The discipline that mattered to the user was not the SVO surface
specifically; it was *Lisp + strong typing + traits + pure
functional + homoiconicity + the ability to express thoughts as
data*. All of that survived.

## What transformed when the substrate landed

Three structural moves the substrate forced.

### Statements became Holon Bundle-of-Binds

OG wat:
```
(Statement dog chases toy :adverb quickly :time t-0)
```

Current wat:
```
(:wat::holon::Bundle
  (:wat::core::vec :wat::holon::HolonAST
    (:wat::holon::Bind :role-subject (:wat::holon::Atom "dog"))
    (:wat::holon::Bind :role-verb    (:wat::holon::Atom "chases"))
    (:wat::holon::Bind :role-object  (:wat::holon::Atom "toy"))
    (:wat::holon::Bind :adverb       (:wat::holon::Atom "quickly"))
    (:wat::holon::Bind :time         (:wat::holon::Atom "t-0"))))
```

The OG `Statement` was hardcoded SVO. The current `Bundle` of
`Bind` pairs is the **generalization** — any role-filler
composition, not just SVO. The trade: OG wat read like English;
current wat reads like algebra. The substrate gained: arbitrary
composition, cosine recall on the hypersphere, no role-locking
in the type system.

The English-readability is recoverable as a wat library on top of
current Holon primitives. See `english-surface-arc.md`.

### Named traits became polymorphic dispatch

OG wat had named traits with English semantics: `Relatable`,
`Adverbial`, `Timeable`, `StringValued`, `Numeric`, `Assertable`,
`Listable`, `Mappable`, `Describable`. Each was an `(impl ...)`
declaration enforced at parse time.

Current wat has:
- `defprotocol` for the abstract trait declaration
- `satisfies` for the structural conformance proof
- `:wat::poly::*` for runtime-polymorphic dispatch (arc 109)

The mechanism is more powerful (covers parametric polymorphism +
runtime dispatch + structural typing). The English names didn't
survive — they would have been carried as user-domain traits, not
substrate-shipped traits. The OG wat traits were already
domain-specific (Adverbial only matters if you're doing English-
like statements); current wat's discipline correctly ships only
the substrate-generic traits at the language level.

### Quantification + temporal forms became vocab patterns

OG wat had `(every subject stmt)`, `(some subject stmt)`,
`(at time stmt)`, `(before t1 t2)`, `(during stmt t1 t2)`,
`(that subject stmt)` as language-level primitives.

Current wat doesn't ship these as primitives. They live as
patterns in the trading lab's vocab modules — `(at zone candle-state ...)`
forms emitted by the regime / persistence / ichimoku vocabs.
That's the right call: the substrate doesn't need to know what
"every" or "during" mean; the vocab modules express them as
domain-specific facts that bundle with other facts on the
hologram.

If the English-like SVO surface ships as a consumer crate, these
quantification + temporal forms come back as macros in that
crate.

## What's still latent

**The English-like SVO statement surface is shippable today as a
wat consumer crate.** Current wat's substrate doesn't want SVO
baked into the algebra (rightly — it's domain-specific), but a
crate can re-publish OG wat's surface as macros over Holon
primitives without touching the substrate.

The candidate crate is `wat-english`, sized in
`english-surface-arc.md`. It would publish:

- `:wat::english::Statement subject verb object` macro
- `:wat::english::every` / `some` for quantification
- `:wat::english::at` / `before` / `during` / `between` for time
- `:wat::english::that` for relative clauses
- `:wat::english::passive` for voice transformation

All expressible as wat-level macros over `:wat::holon::Bind` /
`Bundle`. No substrate change. Sits at the same architectural
tier as `wat-lru` (first-party workspace member) or as a community
`:user::*` crate.

The decision to ship wat-english is the user's. The recognition
that it's possible is what this arc preserves.

## What this contextualizes about the lineage

The DEFCON submission's `Speaker Perspective` answer says:

> *"For nine years inside AWS I tried to convince anyone who
> would listen... I wrote them on my body in Latin in college
> because I needed a persistence layer that wouldn't let me
> forget."*

The OG wat spec is the SECOND persistence layer. The first was
the Latin tattoos; the second was the spec on disk. Both held
the same discipline through years of *"the substrate to host
this didn't exist yet."*

The DEFCON timeline (3y haunting / 1mo rest / 3mo building) is
honest. The OG wat file shows the haunting was not formless —
it was a fully-specified Lisp waiting for the substrate.
The 3 months of building was substantively **the realization of
a years-old design** on the substrate that finally landed.

This also sharpens BOOK Chapter 5's *"the wat language had been
living on my GitHub as a relic for about a year. Grok
conversation links, a proof-of-concept continuation function.
The ideas couldn't be built yet."* The OG wat spec IS that
relic, preserved here for the next session.

## What to do with this recognition

Three options, all defensible:

1. **Preserve only.** This scratch arc captures the OG spec + the
   analysis. No action; the recognition lives on disk for the
   next session or the future BOOK chapter.

2. **Write a BOOK chapter** (Chapter 82 territory). *"What OG wat
   was."* Public-facing documentation of the lineage. Connects
   the DEFCON submission's *"3 years of haunting"* claim to the
   concrete artifact that grounds it.

3. **Open the wat-english arc.** Ship the OG wat English-like
   surface as a consumer crate on top of current wat. Real
   substrate work; ~4-5 slices; would also be a powerful public
   demo (*"I shipped my years-old language design on the
   substrate I built three months ago."*)

User decides. The persistence layer holds either way.
