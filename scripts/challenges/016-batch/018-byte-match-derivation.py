#!/usr/bin/env python3
"""
Byte Match Rule Derivation Algorithm

Given VSA-detected unfamiliar payload positions, algorithmically derive the
optimal set of l4-match rules respecting the real eBPF filter constraints.

eBPF FILTER CONSTRAINTS:
========================
  1-4 byte patterns  → custom dim slots (7 total, arbitrary L4 offset)
  5-64 byte patterns → PatternGuard (L4 offset + length ≤ 64 only)
  Max 32 l4-match rules per destination scope
  Masks: 0xFF (exact byte match) or 0x00 (wildcard / don't-care)
  l4-match offset N  ↔  payload byte at position (N - 8)

RULE COST MODEL:
================
  Tier 1 (cheapest):  1-4 byte custom dim — O(1) tree fan-out, any offset
  Tier 2 (moderate):  PatternGuard — linear scan, near-header only (offset<64)
  Multiple rules at the SAME (offset, length) share one custom dim slot.

ATTACK PROFILES:
================
  A) Dumb:     constant payload every time — trivially matchable
  B) Limited:  3-4 payload variants — small rule set covers all
  C) Rotating: cycle through variations per position — partial coverage
  D) Random:   uniformly random unfamiliar bytes — unmatchable by exact bytes
              (but still VSA-detectable by similarity drop)

ALGORITHM:
==========
  1. Analyze constancy: for each unfamiliar position, measure how concentrated
     the attack byte distribution is (entropy / constancy score)
  2. Generate candidates: 1-byte, 2-byte, 4-byte rules at all positions,
     plus PatternGuard for near-header regions
  3. Validate: compute TP and FP for every candidate
  4. Rank: by (TP desc, cost asc)
  5. Select: greedy knapsack within resource budget
"""

import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity


# =====================================================================
# Config
# =====================================================================

DIMENSIONS = 4096
PAYLOAD_SIZE = 1500
WINDOW_SIZE = 64
NUM_WINDOWS = (PAYLOAD_SIZE + WINDOW_SIZE - 1) // WINDOW_SIZE
UDP_HDR = 8

MAX_CUSTOM_DIM_SLOTS = 7
MAX_RULES_PER_SCOPE = 32
PATTERN_DATA_WINDOW = 64  # PatternGuard limited to first 64 bytes of L4


# =====================================================================
# Windowed Payload Analyzer (reused from 017)
# =====================================================================

class WindowedPayloadAnalyzer:
    def __init__(self, dimensions=4096, window_size=64, payload_size=1500):
        self.client = HolonClient(dimensions=dimensions)
        self.window_size = window_size
        self.payload_size = payload_size
        self.num_windows = (payload_size + window_size - 1) // window_size
        self.accumulators = [
            self.client.create_accumulator() for _ in range(self.num_windows)
        ]
        self.baselines = [None] * self.num_windows
        self.packet_count = 0

    def _window_dict(self, payload, window_idx):
        start = window_idx * self.window_size
        end = min(start + self.window_size, len(payload))
        return {f"p{i - start}": f"0x{payload[i]:02x}" for i in range(start, end)}

    def learn(self, payload):
        for w in range(self.num_windows):
            d = self._window_dict(payload, w)
            vec = self.client.encode(d)
            self.accumulators[w] = self.client.accumulate(self.accumulators[w], vec)
        self.packet_count += 1

    def freeze(self):
        for w in range(self.num_windows):
            self.baselines[w] = self.client.normalize_accumulator(
                self.accumulators[w]
            )

    def score_windows(self, payload):
        results = []
        for w in range(self.num_windows):
            d = self._window_dict(payload, w)
            vec = self.client.encode(d)
            sim = cosine_similarity(vec, self.baselines[w])
            results.append((w, sim))
        return results

    def drill_down_window(self, payload, window_idx):
        baseline = self.baselines[window_idx]
        start = window_idx * self.window_size
        end = min(start + self.window_size, len(payload))
        results = []
        for i in range(start, end):
            local_pos = i - start
            field = f"p{local_pos}"
            value = f"0x{payload[i]:02x}"
            role_vec = self.client.get_vector(field)
            val_vec = self.client.get_vector(value)
            bound = role_vec * val_vec
            sim = cosine_similarity(bound, baseline)
            results.append((i, payload[i], sim))
        return results


# =====================================================================
# Rule Derivation
# =====================================================================

@dataclass
class RuleCandidate:
    """A candidate l4-match rule."""
    l4_offset: int
    match_bytes: list
    mask_bytes: list
    tp: int = 0
    fp: int = 0
    tp_rate: float = 0.0
    description: str = ""

    @property
    def length(self):
        return len(self.match_bytes)

    @property
    def active_bytes(self):
        return sum(1 for m in self.mask_bytes if m != 0x00)

    @property
    def cost_tier(self):
        if self.length <= 4:
            return 1  # custom dim slot
        return 2  # PatternGuard

    @property
    def cost_label(self):
        if self.length <= 4:
            return f"{self.length}B custom dim"
        return f"{self.length}B PatternGuard"

    @property
    def slot_key(self):
        """Custom dim slot identity: (offset, length). Rules sharing this
        key use the same slot."""
        return (self.l4_offset, self.length)

    def to_edn(self):
        mh = "".join(f"{b:02X}" for b in self.match_bytes)
        mk = "".join(f"{b:02X}" for b in self.mask_bytes)
        return f'(l4-match {self.l4_offset} "{mh}" "{mk}")'

    def matches(self, payload):
        start_pos = self.l4_offset - UDP_HDR
        for i in range(self.length):
            if self.mask_bytes[i] == 0x00:
                continue
            pos = start_pos + i
            if pos >= len(payload):
                return False
            if (payload[pos] & self.mask_bytes[i]) != self.match_bytes[i]:
                return False
        return True


def position_entropy(byte_values):
    """Shannon entropy of byte distribution (bits). 0 = constant, 8 = uniform."""
    counts = Counter(byte_values)
    total = len(byte_values)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def constancy_score(byte_values):
    """How concentrated is the distribution? 1.0 = single value, 0.0 = uniform.
    Defined as 1 - (entropy / max_possible_entropy)."""
    n_unique = len(set(byte_values))
    if n_unique <= 1:
        return 1.0
    ent = position_entropy(byte_values)
    max_ent = math.log2(n_unique)
    if max_ent == 0:
        return 1.0
    return 1.0 - (ent / max_ent)


def derive_rules(unfamiliar_positions, attack_payloads, legit_payloads):
    """
    Derive optimal l4-match rules from unfamiliar positions.

    Returns: list of RuleCandidate, sorted by (cost_tier asc, tp desc)
    """
    n_attack = len(attack_payloads)
    n_legit = len(legit_payloads)
    pos_set = set(unfamiliar_positions)

    # Phase 1: Per-position analysis
    pos_info = {}
    for pos in sorted(pos_set):
        attack_bytes = [p[pos] for p in attack_payloads]
        legit_bytes_set = set(p[pos] for p in legit_payloads)

        byte_counts = Counter(attack_bytes)
        unfam_vals = {v: c for v, c in byte_counts.items() if v not in legit_bytes_set}

        pos_info[pos] = {
            "entropy": position_entropy(attack_bytes),
            "constancy": constancy_score(attack_bytes),
            "byte_counts": byte_counts,
            "legit_set": legit_bytes_set,
            "unfam_vals": unfam_vals,
            "coverage": sum(unfam_vals.values()) / n_attack if unfam_vals else 0,
        }

    candidates = []

    # Phase 2a: 1-byte rules at every position
    for pos in sorted(pos_set):
        info = pos_info[pos]
        for val, count in info["unfam_vals"].items():
            r = RuleCandidate(
                l4_offset=UDP_HDR + pos,
                match_bytes=[val],
                mask_bytes=[0xFF],
                description=f"1B @pos {pos} = 0x{val:02X}",
            )
            candidates.append(r)

    # Phase 2b: 2-byte rules at consecutive positions
    sorted_positions = sorted(pos_set)
    for i, pos in enumerate(sorted_positions):
        if pos + 1 in pos_set:
            atk_combos = Counter(
                (p[pos], p[pos + 1]) for p in attack_payloads
            )
            legit_combos = set(
                (p[pos], p[pos + 1]) for p in legit_payloads
            )
            for combo, count in atk_combos.most_common():
                if combo not in legit_combos:
                    r = RuleCandidate(
                        l4_offset=UDP_HDR + pos,
                        match_bytes=list(combo),
                        mask_bytes=[0xFF, 0xFF],
                        description=f"2B @pos {pos}-{pos+1}",
                    )
                    candidates.append(r)

    # Phase 2c: 4-byte rules at consecutive positions
    for pos in sorted_positions:
        if all(pos + j in pos_set for j in range(4)):
            atk_combos = Counter(
                tuple(p[pos + j] for j in range(4)) for p in attack_payloads
            )
            legit_combos = set(
                tuple(p[pos + j] for j in range(4)) for p in legit_payloads
            )
            for combo, count in atk_combos.most_common():
                if combo not in legit_combos:
                    r = RuleCandidate(
                        l4_offset=UDP_HDR + pos,
                        match_bytes=list(combo),
                        mask_bytes=[0xFF] * 4,
                        description=f"4B @pos {pos}-{pos+3}",
                    )
                    candidates.append(r)

    # Phase 2d: PatternGuard for near-header positions
    # Only positions where L4 offset + length ≤ 64
    near_header = [p for p in sorted_positions if UDP_HDR + p < PATTERN_DATA_WINDOW]
    if len(near_header) >= 2:
        first = near_header[0]
        last = near_header[-1]
        span = last - first + 1
        l4_off = UDP_HDR + first

        # Only if the span fits in the 64-byte window
        if l4_off + span <= PATTERN_DATA_WINDOW and span >= 5:
            # Build the most common combo across all near-header positions
            def extract_near(payload):
                result = []
                for p in range(first, last + 1):
                    result.append(payload[p])
                return tuple(result)

            atk_combos = Counter(extract_near(p) for p in attack_payloads)
            legit_combos = set(extract_near(p) for p in legit_payloads)

            for combo, count in atk_combos.most_common(20):
                if combo in legit_combos:
                    continue
                match_b = []
                mask_b = []
                for idx, pos in enumerate(range(first, last + 1)):
                    if pos in pos_set:
                        match_b.append(combo[idx])
                        mask_b.append(0xFF)
                    else:
                        match_b.append(0x00)
                        mask_b.append(0x00)

                r = RuleCandidate(
                    l4_offset=l4_off,
                    match_bytes=match_b,
                    mask_bytes=mask_b,
                    description=f"PatternGuard @pos {first}-{last} "
                                f"({sum(1 for m in mask_b if m)}/"
                                f"{span} active)",
                )
                candidates.append(r)

    # Phase 3: Validate all candidates
    for r in candidates:
        r.tp = sum(1 for p in attack_payloads if r.matches(p))
        r.fp = sum(1 for p in legit_payloads if r.matches(p))
        r.tp_rate = r.tp / n_attack

    # Filter: zero FP only
    candidates = [r for r in candidates if r.fp == 0]

    # Sort: cost tier ascending, then TP descending
    candidates.sort(key=lambda r: (r.cost_tier, -r.tp))

    return candidates, pos_info


def select_optimal_rules(candidates, attack_payloads,
                         max_slots=MAX_CUSTOM_DIM_SLOTS,
                         max_rules=MAX_RULES_PER_SCOPE):
    """
    Coverage-aware greedy selection: pick rules that maximize coverage
    within budget, stopping as soon as 100% TP is reached.

    Budget:
      - max_slots custom dim slot keys (unique offset+length combos for 1-4B)
      - PatternGuard entries don't consume custom dim slots
      - Total rules ≤ max_rules

    Returns: selected rules, used slots, coverage progression
    """
    selected = []
    used_slots = set()
    covered = set()  # indices of attack packets already matched
    progression = []  # (rule_count, coverage_pct)
    n = len(attack_payloads)

    # Pre-compute which packets each candidate matches
    candidate_hits = []
    for r in candidates:
        hits = set(
            i for i, p in enumerate(attack_payloads) if r.matches(p)
        )
        candidate_hits.append(hits)

    for idx, r in enumerate(candidates):
        if len(selected) >= max_rules:
            break

        # Check custom dim slot budget
        if r.cost_tier == 1:
            if r.slot_key not in used_slots and len(used_slots) >= max_slots:
                continue

        # How many NEW packets does this rule cover?
        new_hits = candidate_hits[idx] - covered
        if not new_hits:
            continue  # skip redundant rules

        selected.append(r)
        covered |= new_hits
        if r.cost_tier == 1:
            used_slots.add(r.slot_key)

        progression.append((len(selected), len(covered) / n * 100))

        if len(covered) == n:
            break  # 100% coverage reached

    return selected, used_slots, progression


# =====================================================================
# Payload Generators
# =====================================================================

FAMILIAR = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30]
UNFAMILIAR = [0xFF, 0xAC, 0xFB, 0xCA, 0xDE, 0xAD, 0xBE, 0xEF]


def make_normal(rng, size=PAYLOAD_SIZE):
    return [int(rng.choice(FAMILIAR)) for _ in range(size)]


def make_attack_dumb(rng, size=PAYLOAD_SIZE):
    """Constant attack: always the same payload at anomalous positions."""
    payload = [int(rng.choice(FAMILIAR)) for _ in range(size)]
    # Fixed signature at positions 10-13 and 700-703
    for i, val in zip(range(10, 14), [0xDE, 0xAD, 0xBE, 0xEF]):
        payload[i] = val
    for i, val in zip(range(700, 704), [0xCA, 0xFE, 0xBA, 0xBE]):
        payload[i] = val
    return payload


def make_attack_limited(rng, size=PAYLOAD_SIZE):
    """Limited variation: 3 attack tool variants, each with a distinct but
    consistent signature."""
    payload = [int(rng.choice(FAMILIAR)) for _ in range(size)]
    variant = int(rng.integers(3))
    signatures = [
        ([0xDE, 0xAD, 0xBE, 0xEF], [0xCA, 0xFE, 0xBA, 0xBE]),
        ([0xFF, 0xAC, 0xFB, 0xCA], [0xAD, 0xDE, 0xEF, 0xFF]),
        ([0xBE, 0xEF, 0xCA, 0xFE], [0xDE, 0xAD, 0x00, 0xFF]),
    ]
    near, deep = signatures[variant]
    for i, val in enumerate(near):
        payload[10 + i] = val
    for i, val in enumerate(deep):
        payload[700 + i] = val
    return payload


def make_attack_rotating(rng, size=PAYLOAD_SIZE):
    """Rotating: each position cycles through 3-4 values independently."""
    payload = [int(rng.choice(FAMILIAR)) for _ in range(size)]
    rotation_sets = {
        10: [0xDE, 0xFF, 0xBE],
        11: [0xAD, 0xAC],
        12: [0xBE, 0xFB, 0xCA, 0xEF],
        13: [0xEF, 0xCA],
        700: [0xCA, 0xAD],
        701: [0xFE, 0xDE, 0xBE],
        702: [0xBA, 0xEF],
        703: [0xBE, 0xFF, 0xAC],
    }
    for pos, vals in rotation_sets.items():
        payload[pos] = vals[int(rng.integers(len(vals)))]
    return payload


def make_attack_random(rng, size=PAYLOAD_SIZE):
    """Fully random unfamiliar bytes — unmatchable by exact byte rules."""
    payload = [int(rng.choice(FAMILIAR)) for _ in range(size)]
    for pos in list(range(10, 14)) + list(range(700, 704)):
        payload[pos] = int(rng.choice(UNFAMILIAR))
    return payload


# =====================================================================
# Detection pipeline
# =====================================================================

def detect_unfamiliar(analyzer, sample, legit_ref, expected_positions):
    """Run windowed detection + drill-down, return unfamiliar positions."""
    atk_scores = analyzer.score_windows(sample)
    leg_scores = analyzer.score_windows(legit_ref)

    anomalous_windows = []
    for w in range(analyzer.num_windows):
        drop = leg_scores[w][1] - atk_scores[w][1]
        if drop > 0.02:
            anomalous_windows.append(w)

    all_unfamiliar = []
    for w in anomalous_windows:
        drill = analyzer.drill_down_window(sample, w)
        for pos, byte_val, sim in drill:
            if sim < 0.005:
                all_unfamiliar.append(pos)

    return sorted(set(all_unfamiliar)), anomalous_windows


# =====================================================================
# Main
# =====================================================================

def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("BYTE MATCH RULE DERIVATION: Algorithmic Optimization")
    print("=" * 80)

    # Train the analyzer
    print("\nPhase 1: Training windowed analyzer on 500 normal payloads...")
    analyzer = WindowedPayloadAnalyzer(DIMENSIONS, WINDOW_SIZE, PAYLOAD_SIZE)
    for _ in range(500):
        analyzer.learn(make_normal(rng))
    analyzer.freeze()
    print(f"  {analyzer.packet_count} packets, {NUM_WINDOWS} windows frozen")

    legit_payloads = [make_normal(rng) for _ in range(200)]
    legit_ref = legit_payloads[0]

    # Attack profiles
    attacks = [
        (
            "DUMB (constant payload)",
            make_attack_dumb,
            list(range(10, 14)) + list(range(700, 704)),
        ),
        (
            "LIMITED (3 tool variants)",
            make_attack_limited,
            list(range(10, 14)) + list(range(700, 704)),
        ),
        (
            "ROTATING (per-position cycling)",
            make_attack_rotating,
            list(range(10, 14)) + list(range(700, 704)),
        ),
        (
            "RANDOM (uniform unfamiliar bytes)",
            make_attack_random,
            list(range(10, 14)) + list(range(700, 704)),
        ),
    ]

    for attack_name, attack_fn, expected_positions in attacks:
        print(f"\n{'=' * 80}")
        print(f"ATTACK PROFILE: {attack_name}")
        print(f"{'=' * 80}")

        attack_payloads = [attack_fn(rng) for _ in range(200)]

        # Detect
        sample = attack_payloads[0]
        detected_positions, anom_windows = detect_unfamiliar(
            analyzer, sample, legit_ref, expected_positions
        )

        expected_set = set(expected_positions)
        detected_set = set(detected_positions)
        hit = detected_set & expected_set
        missed = expected_set - detected_set
        false_alarm = detected_set - expected_set

        print(f"\n  Detection: {len(hit)}/{len(expected_set)} positions found"
              f"  (missed: {len(missed)}, false: {len(false_alarm)})")
        if missed:
            print(f"    Missed: {sorted(missed)}")
        print(f"  Anomalous windows: {anom_windows}")

        if not detected_positions:
            print("  No unfamiliar positions → no rules to generate.")
            continue

        # Derive rules
        candidates, pos_info = derive_rules(
            detected_positions, attack_payloads, legit_payloads
        )

        # Per-position constancy analysis
        print(f"\n  Position constancy analysis:")
        print(f"  {'Pos':>6} {'L4 Off':>7} {'Entropy':>8} {'Constancy':>10} "
              f"{'Coverage':>9} {'#Unique':>8} {'Top Byte':>9}")
        print(f"  {'-' * 63}")

        for pos in sorted(detected_positions):
            if pos in pos_info:
                info = pos_info[pos]
                if info["unfam_vals"]:
                    top_val = max(info["unfam_vals"], key=info["unfam_vals"].get)
                    top_str = f"0x{top_val:02X}"
                else:
                    top_str = "—"
                print(f"  {pos:>6} {UDP_HDR + pos:>7} "
                      f"{info['entropy']:>8.2f} "
                      f"{info['constancy']:>10.2f} "
                      f"{info['coverage']:>8.0%} "
                      f"{len(info['byte_counts']):>8} "
                      f"{top_str:>9}")

        # Show all candidate rules by tier
        print(f"\n  Candidate rules: {len(candidates)} total (0% FP)")

        # Group by tier
        tier1 = [r for r in candidates if r.cost_tier == 1]
        tier2 = [r for r in candidates if r.cost_tier == 2]

        if tier1:
            # Sub-group by length
            for width in [1, 2, 4]:
                group = [r for r in tier1 if r.length == width]
                if not group:
                    continue
                print(f"\n  Tier 1 — {width}B custom dim rules "
                      f"(top 10 of {len(group)}):")
                print(f"  {'Rule':>50} {'TP':>5} {'TP%':>6} {'Slot':>20}")
                print(f"  {'-' * 85}")
                for r in group[:10]:
                    print(f"  {r.to_edn():>50} "
                          f"{r.tp:>5} {r.tp_rate:>5.0%} "
                          f"{str(r.slot_key):>20}")

        if tier2:
            print(f"\n  Tier 2 — PatternGuard rules (top 5 of {len(tier2)}):")
            print(f"  {'Rule':>65} {'TP':>5} {'TP%':>6}")
            print(f"  {'-' * 80}")
            for r in tier2[:5]:
                print(f"  {r.to_edn():>65} {r.tp:>5} {r.tp_rate:>5.0%}")

        # Optimal selection (coverage-aware)
        selected, used_slots, progression = select_optimal_rules(
            candidates, attack_payloads
        )

        # Compute combined TP/FP
        combined_tp = sum(
            1 for p in attack_payloads
            if any(r.matches(p) for r in selected)
        )
        combined_fp = sum(
            1 for p in legit_payloads
            if any(r.matches(p) for r in selected)
        )

        print(f"\n  OPTIMAL RULE SET (greedy selection):")
        print(f"  {'─' * 70}")

        slot_groups = {}
        pg_rules = []
        for r in selected:
            if r.cost_tier == 1:
                slot_groups.setdefault(r.slot_key, []).append(r)
            else:
                pg_rules.append(r)

        slot_idx = 0
        for key, rules in sorted(slot_groups.items()):
            slot_idx += 1
            print(f"\n  Custom dim slot {slot_idx}/7 "
                  f"(offset={key[0]}, length={key[1]}):")
            for r in rules:
                print(f"    {r.to_edn()}  "
                      f"TP={r.tp}/{len(attack_payloads)} ({r.tp_rate:.0%})")

        if pg_rules:
            print(f"\n  PatternGuard entries:")
            for r in pg_rules:
                print(f"    {r.to_edn()}")
                print(f"      TP={r.tp}/{len(attack_payloads)} ({r.tp_rate:.0%})")

        print(f"\n  COMBINED RESULT:")
        print(f"    Rules:       {len(selected)}/{MAX_RULES_PER_SCOPE} budget")
        print(f"    Dim slots:   {len(used_slots)}/{MAX_CUSTOM_DIM_SLOTS} budget")
        print(f"    Attack TP:   {combined_tp}/{len(attack_payloads)} "
              f"({combined_tp / len(attack_payloads) * 100:.0f}%)")
        print(f"    Legit FP:    {combined_fp}/{len(legit_payloads)} "
              f"({combined_fp / len(legit_payloads) * 100:.0f}%)")

        # Verdict
        if combined_tp == len(attack_payloads):
            verdict = "PERFECT — 100% of attack caught"
        elif combined_tp / len(attack_payloads) >= 0.9:
            verdict = "STRONG — 90%+ coverage"
        elif combined_tp / len(attack_payloads) >= 0.5:
            verdict = "PARTIAL — significant but incomplete"
        else:
            verdict = "WEAK — rely on rate-limiting instead"
        print(f"    Verdict:     {verdict}")

        # Coverage efficiency curve
        if progression:
            print(f"\n  COVERAGE EFFICIENCY:")
            print(f"  {'Rules':>6} {'Coverage':>10}")
            print(f"  {'-' * 18}")
            for n_rules, cov in progression:
                bar = "#" * int(cov / 5)
                print(f"  {n_rules:>6} {cov:>9.1f}% {bar}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print("""
  Dumb attacks:     Multi-byte exact match → 1 rule, 100% TP
  Limited attacks:  One rule per variant   → 3 rules, 100% TP
  Rotating attacks: Enumerate combinations → ~8 rules, 100% TP
  Random attacks:   Enumerate per-position → 8 rules at 1 offset, 100% TP

  KEY INSIGHT: As long as attack bytes are NEVER seen in legit traffic,
  even fully random attacks are matchable by enumerating the unfamiliar
  values. The only truly unmatchable case is when attackers use byte
  values that overlap with legitimate traffic (evasion).

  COST HIERARCHY (the algorithm biases cheap → expensive):
    1. 1-byte custom dim at high-constancy position (cheapest)
    2. 2-byte or 4-byte custom dim at consecutive constant positions
    3. Multiple 1-byte rules at same offset (shared slot)
    4. PatternGuard for near-header sparse patterns (most expensive)
""")


if __name__ == "__main__":
    main()
