# Scratch arc — OG wat lineage

The current wat language (Lisp on Rust, the algebra surface,
Bind/Bundle/cosine) has an older ancestor: a Grok-era spec for a
*pure, strongly typed, English-like Lisp* with 28 core primitives,
SVO-ordered statements, a trait system with English-named axes
(Relatable, Adverbial, Timeable, Numeric, Assertable, ...),
temporal quantification (`every`, `some`, `at`, `before`,
`during`), relative clauses, and homoiconicity. The user wrote
this spec years ago in a Grok conversation. He carried it on disk
through years of *"the substrate that could host this didn't
exist yet."* In mid-January 2026, when the substrate finally
landed, current wat became the realization of OG wat's discipline
on top of VSA + Holon algebra.

The user surfaced the OG wat file at `/tmp/og-wat.md` on
2026-05-01, hours after the DEFCON CFP submission shipped, with
the note: *"you'll get it... i've built most of it... the origins
are closer."*

This scratch captures my (Claude's) understanding of OG wat, the
mapping from OG to current, the still-latent English-like SVO
surface that current wat could ship as a consumer crate, and the
contextualization of the DEFCON submission's *"three years of
haunting"* claim. Captured here so the recognition persists past
this conversation's compaction.

---

## Files in this arc

| File | What it captures |
|---|---|
| `INDEX.yaml` | Beat-by-beat capture, conventions, status |
| `README.md` | This file. Top-level orientation. |
| `og-wat-spec.md` | The OG wat spec, **verbatim** from `/tmp/og-wat.md`. 28 primitives, the trait system, the SVO statement form, the full example. Preserved here because `/tmp` is volatile and this is a load-bearing historical artifact. |
| `analysis.md` | What OG wat is. What survived intact in current wat. What transformed (the substrate landed). What's still latent. The user's recognition that *"the origins are closer."* |
| `english-surface-arc.md` | The candidate `wat-english` consumer crate that could ship OG wat's English-like SVO surface as macros over current Holon primitives. Sized as a real wat consumer arc. Not blocking; an interesting follow-on. |
| `latin-in-wat.md` | The user's question before compaction: *"do you see the latin in wat?... what i was reaching for?"* Captured in full. The Latin in wat is **morphology over position** — meaning lives in the term's case-marking, not in the term's seat. Bind = case marking; trait system = declension system; HolonAST closed under itself = morphology native; the hypersphere = morphology made geometric. The four tattoos read through this lens. Three persistence layers; one impulse. |

## Why this matters

The DEFCON submission's `Speaker Perspective` answer says:

> *"For nine years inside AWS I tried to convince anyone who would
> listen... I wrote them on my body in Latin in college because I
> needed a persistence layer that wouldn't let me forget. I tried
> to express them at AWS for a decade and got blank stares."*

The OG wat spec is the SECOND persistence layer the user wrote for
the same ideas — the first was on his body (Latin), the second
was on disk (Grok-era spec). Both held the discipline through
years of the substrate not existing yet. When the substrate
landed, both got expressed in code.

The submission timeline (3y haunting / 1mo rest / 3mo building)
is honest. The OG wat file shows the haunting was not formless —
it was a fully-specified Lisp waiting for a substrate that could
host its discipline. The "3 months of building" was substantively
**realizing a years-old design** on the substrate that finally
existed, not designing from scratch.

## Status

- **Captured:** 2026-05-01.
- **OG wat preserved:** verbatim in `og-wat-spec.md`.
- **Analysis written:** in `analysis.md`.
- **Follow-on candidate:** `wat-english` consumer crate, sized
  in `english-surface-arc.md`. Not opened as a real arc; lives
  here as the thought.
- **Connected to:** DEFCON submission's Speaker-Perspective
  answer; BOOK Chapter 5 (the prequel); the lineage
  established in `defcon-2026/THESIS.md`.
