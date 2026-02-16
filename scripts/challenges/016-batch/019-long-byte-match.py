#!/usr/bin/env python3
"""
Long Byte Match Derivation: Finding 32-byte signatures

Can the algorithm sniff out a long l4-match rule from VSA drill-down?

SCENARIOS:
==========
  A) 32-byte constant shellcode at near-header (payload 0-31, L4 8-39)
     → should produce a single 32-byte PatternGuard with all FF masks

  B) 32-byte constant at deep offset (payload 200-231, L4 208-239)
     → PatternGuard can't reach, must decompose to custom dims

  C) 32-byte "protocol header" with 24 constant + 8 variable bytes
     → PatternGuard with sparse mask (FF at constant, 00 at variable)

  D) 32-byte signature split: 16 constant near-header + 16 constant deep
     → near-header gets PatternGuard, deep gets custom dims

eBPF CONSTRAINTS RECAP:
=======================
  PatternGuard: 5-64 bytes, L4 offset + length ≤ 64
  Custom dim:   1-4 bytes, arbitrary offset, 7 slots total
  Max rules:    32 per destination scope
  Masks:        0xFF (exact) or 0x00 (wildcard)
"""

import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity


DIMENSIONS = 4096
PAYLOAD_SIZE = 512
WINDOW_SIZE = 64
NUM_WINDOWS = (PAYLOAD_SIZE + WINDOW_SIZE - 1) // WINDOW_SIZE
UDP_HDR = 8
PATTERN_DATA_WINDOW = 64
MAX_CUSTOM_DIM_SLOTS = 7
MAX_RULES_PER_SCOPE = 32


# =====================================================================
# Windowed Analyzer (compact version from 017/018)
# =====================================================================

class WindowedPayloadAnalyzer:
    def __init__(self):
        self.client = HolonClient(dimensions=DIMENSIONS)
        self.num_windows = NUM_WINDOWS
        self.accumulators = [
            self.client.create_accumulator() for _ in range(NUM_WINDOWS)
        ]
        self.baselines = [None] * NUM_WINDOWS
        self.count = 0

    def _window_dict(self, payload, w):
        start = w * WINDOW_SIZE
        end = min(start + WINDOW_SIZE, len(payload))
        return {f"p{i - start}": f"0x{payload[i]:02x}" for i in range(start, end)}

    def learn(self, payload):
        for w in range(self.num_windows):
            vec = self.client.encode(self._window_dict(payload, w))
            self.accumulators[w] = self.client.accumulate(
                self.accumulators[w], vec
            )
        self.count += 1

    def freeze(self):
        for w in range(self.num_windows):
            self.baselines[w] = self.client.normalize_accumulator(
                self.accumulators[w]
            )

    def detect(self, payload, legit_ref):
        """Return list of unfamiliar (position, byte_value) tuples."""
        atk_scores = self.score_windows(payload)
        leg_scores = self.score_windows(legit_ref)

        unfamiliar = []
        for w in range(self.num_windows):
            drop = leg_scores[w][1] - atk_scores[w][1]
            if drop > 0.015:
                for pos, byte_val, sim in self.drill_down(payload, w):
                    if sim < 0.005:
                        unfamiliar.append((pos, byte_val))
        return unfamiliar

    def score_windows(self, payload):
        results = []
        for w in range(self.num_windows):
            vec = self.client.encode(self._window_dict(payload, w))
            sim = cosine_similarity(vec, self.baselines[w])
            results.append((w, sim))
        return results

    def drill_down(self, payload, w):
        baseline = self.baselines[w]
        start = w * WINDOW_SIZE
        end = min(start + WINDOW_SIZE, len(payload))
        results = []
        for i in range(start, end):
            field = f"p{i - start}"
            value = f"0x{payload[i]:02x}"
            bound = self.client.get_vector(field) * self.client.get_vector(value)
            sim = cosine_similarity(bound, baseline)
            results.append((i, payload[i], sim))
        return results


# =====================================================================
# Rule derivation — extended for long patterns
# =====================================================================

@dataclass
class Rule:
    l4_offset: int
    match_bytes: list
    mask_bytes: list
    tp: int = 0
    fp: int = 0
    description: str = ""

    @property
    def length(self):
        return len(self.match_bytes)

    @property
    def active(self):
        return sum(1 for m in self.mask_bytes if m != 0x00)

    @property
    def cost_label(self):
        if self.length <= 4:
            return f"{self.length}B custom-dim"
        return f"{self.length}B PatternGuard"

    @property
    def is_pattern_guard(self):
        return self.length > 4

    @property
    def enforceable(self):
        if self.length <= 4:
            return True  # custom dim, any offset
        return self.l4_offset + self.length <= PATTERN_DATA_WINDOW

    def to_edn(self):
        mh = "".join(f"{b:02X}" for b in self.match_bytes)
        mk = "".join(f"{b:02X}" for b in self.mask_bytes)
        return f'(l4-match {self.l4_offset} "{mh}" "{mk}")'

    def matches(self, payload):
        start = self.l4_offset - UDP_HDR
        for i in range(self.length):
            if self.mask_bytes[i] == 0x00:
                continue
            p = start + i
            if p >= len(payload):
                return False
            if (payload[p] & self.mask_bytes[i]) != self.match_bytes[i]:
                return False
        return True


def find_constant_runs(positions, attack_payloads, legit_payloads):
    """Find runs of consecutive positions where attack bytes are highly
    consistent (constant or near-constant) and unfamiliar."""
    pos_set = set(positions)
    sorted_pos = sorted(pos_set)
    if not sorted_pos:
        return []

    # Build runs — allow small gaps (≤3) to be bridged with wildcards
    runs = []
    current_run = [sorted_pos[0]]

    for pos in sorted_pos[1:]:
        gap = pos - current_run[-1]
        if gap == 1:
            current_run.append(pos)
        elif gap <= 4:
            for bridge in range(current_run[-1] + 1, pos):
                current_run.append(bridge)
            current_run.append(pos)
        else:
            runs.append(current_run)
            current_run = [pos]
    runs.append(current_run)

    # Per-position analysis — including bridged gap positions
    all_run_positions = set()
    for run in runs:
        all_run_positions.update(run)

    pos_data = {}
    for pos in sorted(all_run_positions):
        atk_bytes = [p[pos] for p in attack_payloads]
        leg_set = set(p[pos] for p in legit_payloads)
        counts = Counter(atk_bytes)
        top_val, top_count = counts.most_common(1)[0]
        constancy = top_count / len(atk_bytes)

        pos_data[pos] = {
            "top_val": top_val,
            "constancy": constancy,
            "n_unique": len(counts),
            "is_constant": constancy >= 0.95,
            "unfam_vals": {v for v in counts if v not in leg_set},
            "leg_set": leg_set,
            "counts": counts,
        }

    return runs, pos_data


def derive_long_rules(positions, attack_payloads, legit_payloads):
    """Derive rules including long PatternGuard candidates."""
    n_atk = len(attack_payloads)
    n_leg = len(legit_payloads)

    runs, pos_data = find_constant_runs(
        positions, attack_payloads, legit_payloads
    )

    candidates = []

    for run in runs:
        first = run[0]
        last = run[-1]
        run_len = len(run)
        l4_off = UDP_HDR + first

        # Always generate 1-byte rules for each position
        for pos in run:
            info = pos_data[pos]
            for val in info["unfam_vals"]:
                r = Rule(
                    l4_offset=UDP_HDR + pos,
                    match_bytes=[val],
                    mask_bytes=[0xFF],
                    description=f"1B @{pos}",
                )
                candidates.append(r)

        # 2-byte rules
        for i in range(len(run) - 1):
            p1, p2 = run[i], run[i + 1]
            if p2 != p1 + 1:
                continue
            combos = Counter(
                (p[p1], p[p2]) for p in attack_payloads
            )
            legit_combos = set(
                (p[p1], p[p2]) for p in legit_payloads
            )
            for combo, cnt in combos.most_common(5):
                if combo not in legit_combos:
                    r = Rule(
                        l4_offset=UDP_HDR + p1,
                        match_bytes=list(combo),
                        mask_bytes=[0xFF, 0xFF],
                        description=f"2B @{p1}-{p1+1}",
                    )
                    candidates.append(r)

        # 4-byte rules
        for i in range(len(run) - 3):
            positions_4 = run[i:i + 4]
            if positions_4[-1] - positions_4[0] != 3:
                continue
            combos = Counter(
                tuple(p[q] for q in positions_4) for p in attack_payloads
            )
            legit_combos = set(
                tuple(p[q] for q in positions_4) for p in legit_payloads
            )
            for combo, cnt in combos.most_common(5):
                if combo not in legit_combos:
                    r = Rule(
                        l4_offset=UDP_HDR + positions_4[0],
                        match_bytes=list(combo),
                        mask_bytes=[0xFF] * 4,
                        description=f"4B @{positions_4[0]}-{positions_4[-1]}",
                    )
                    candidates.append(r)

        # LONG RULES: 8, 16, 32 bytes and full run length
        # Try spans of increasing length, anchored at constant sub-runs
        for target_len in [8, 16, 32, run_len]:
            if target_len < 5 or target_len > run_len:
                continue

            # Sliding window over the run
            for start_idx in range(len(run) - target_len + 1):
                span_positions = run[start_idx:start_idx + target_len]
                span_first = span_positions[0]
                span_last = span_positions[-1]
                span_l4 = UDP_HDR + span_first
                actual_span = span_last - span_first + 1

                # Build match/mask using most common attack byte per position
                # Use 0x00 mask for positions that are variable or familiar
                match_b = []
                mask_b = []
                constant_count = 0

                for pos in range(span_first, span_last + 1):
                    if pos in pos_data:
                        info = pos_data[pos]
                        if info["is_constant"] and info["unfam_vals"]:
                            match_b.append(info["top_val"])
                            mask_b.append(0xFF)
                            constant_count += 1
                        elif info["unfam_vals"]:
                            # Variable but unfamiliar — try most common
                            top = max(
                                info["unfam_vals"],
                                key=lambda v: info["counts"][v],
                            )
                            match_b.append(top)
                            mask_b.append(0xFF)
                            constant_count += 1
                        else:
                            match_b.append(0x00)
                            mask_b.append(0x00)
                    else:
                        match_b.append(0x00)
                        mask_b.append(0x00)

                # Only generate if we have enough active bytes
                active = sum(1 for m in mask_b if m != 0x00)
                if active < 3:
                    continue

                r = Rule(
                    l4_offset=span_l4,
                    match_bytes=match_b,
                    mask_bytes=mask_b,
                    description=f"{actual_span}B @{span_first}-{span_last} "
                                f"({active}/{actual_span} active)",
                )
                candidates.append(r)

    # Validate all candidates
    for r in candidates:
        r.tp = sum(1 for p in attack_payloads if r.matches(p))
        r.fp = sum(1 for p in legit_payloads if r.matches(p))

    # Filter zero FP
    candidates = [r for r in candidates if r.fp == 0 and r.tp > 0]

    # Sort: enforceable first, then by (length desc for same TP, TP desc)
    candidates.sort(key=lambda r: (-r.enforceable, -r.tp, -r.length))

    return candidates, runs, pos_data


def select_best(candidates, attack_payloads):
    """Coverage-aware greedy selection."""
    selected = []
    covered = set()
    used_slots = set()
    n = len(attack_payloads)

    hit_cache = []
    for r in candidates:
        hits = set(i for i, p in enumerate(attack_payloads) if r.matches(p))
        hit_cache.append(hits)

    for idx, r in enumerate(candidates):
        if len(selected) >= MAX_RULES_PER_SCOPE:
            break

        # Custom dim slot budget
        if r.length <= 4:
            key = (r.l4_offset, r.length)
            if key not in used_slots and len(used_slots) >= MAX_CUSTOM_DIM_SLOTS:
                continue

        new_hits = hit_cache[idx] - covered
        if not new_hits:
            continue

        selected.append(r)
        covered |= new_hits
        if r.length <= 4:
            used_slots.add((r.l4_offset, r.length))

        if len(covered) == n:
            break

    return selected, len(covered) / n * 100


# =====================================================================
# Payload Generators
# =====================================================================

FAMILIAR = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30]


def make_normal(rng):
    return [int(rng.choice(FAMILIAR)) for _ in range(PAYLOAD_SIZE)]


# Scenario A: 32 constant bytes near header
SHELLCODE_32 = [
    0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x90, 0x90, 0x90, 0x90, 0xCC, 0xCC, 0xCC, 0xCC,
    0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8,
]


def make_attack_near32(rng):
    payload = make_normal(rng)
    for i, val in enumerate(SHELLCODE_32):
        payload[i] = val
    return payload


# Scenario B: 32 constant bytes deep in payload (offset 200)
def make_attack_deep32(rng):
    payload = make_normal(rng)
    for i, val in enumerate(SHELLCODE_32):
        payload[200 + i] = val
    return payload


# Scenario C: 32-byte protocol header — 24 constant + 8 variable
PROTO_CONSTANT = [
    0xAA, 0xBB, 0xCC, 0xDD,  # magic
    0x00, 0x00,                # variable: sequence number (uses familiar bytes)
    0xEE, 0xFF,               # version
    0x55, 0x66, 0x77, 0x88,   # session id (constant)
    0x00, 0x00, 0x00, 0x00,   # variable: timestamp (familiar bytes)
    0x99, 0xAA, 0xBB, 0xCC,   # command id (constant)
    0xDD, 0xEE, 0xFF, 0x11,   # auth token prefix (constant)
    0x00, 0x00, 0x00, 0x00,   # variable: nonce (familiar bytes)
    0x22, 0x33, 0x44, 0x55,   # checksum (constant)
]

PROTO_VARIABLE_POS = [4, 5, 12, 13, 14, 15, 24, 25, 26, 27]


def make_attack_sparse32(rng):
    payload = make_normal(rng)
    for i, val in enumerate(PROTO_CONSTANT):
        if i in PROTO_VARIABLE_POS:
            payload[i] = int(rng.choice(FAMILIAR))
        else:
            payload[i] = val
    return payload


# Scenario D: Split — 16 constant near + 16 constant deep
NEAR_16 = SHELLCODE_32[:16]
DEEP_16 = SHELLCODE_32[16:]


def make_attack_split(rng):
    payload = make_normal(rng)
    for i, val in enumerate(NEAR_16):
        payload[i] = val
    for i, val in enumerate(DEEP_16):
        payload[300 + i] = val
    return payload


# =====================================================================
# Main
# =====================================================================

def run_scenario(name, attack_fn, expected_positions, analyzer, legit_payloads,
                 rng):
    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {name}")
    print(f"{'=' * 80}")

    attack_payloads = [attack_fn(rng) for _ in range(200)]
    legit_ref = legit_payloads[0]

    # Multi-sample detection: check multiple attack samples
    all_detected = set()
    for sample in attack_payloads[:5]:
        detected = analyzer.detect(sample, legit_ref)
        all_detected.update(pos for pos, _ in detected)

    expected_set = set(expected_positions)
    hit = all_detected & expected_set
    missed = expected_set - all_detected

    print(f"\n  VSA detection: {len(hit)}/{len(expected_set)} positions found"
          f"  (missed: {len(missed)})")
    if missed and len(missed) <= 10:
        print(f"    Missed: {sorted(missed)}")
    elif missed:
        print(f"    Missed: {len(missed)} positions "
              f"(first: {sorted(missed)[:5]}...)")

    if not all_detected:
        print("  No unfamiliar positions found — skipping rule derivation.")
        return

    # GAP PROBING: extend detected runs by checking neighboring positions
    # The VSA drill-down may miss some unfamiliar positions (borderline
    # similarity). We probe positions BETWEEN and ADJACENT to detected
    # runs to see if the attack bytes there are also unfamiliar.
    legit_byte_sets = {}
    for pos in range(PAYLOAD_SIZE):
        legit_byte_sets[pos] = set(p[pos] for p in legit_payloads)

    probed = set()
    sorted_detected = sorted(all_detected)
    if sorted_detected:
        # Find the bounding region: extend ±4 around detected positions
        lo = max(0, sorted_detected[0] - 4)
        hi = min(PAYLOAD_SIZE - 1, sorted_detected[-1] + 4)

        for pos in range(lo, hi + 1):
            if pos in all_detected:
                continue
            # Check if attack bytes at this position are unfamiliar
            atk_bytes = set(p[pos] for p in attack_payloads[:20])
            unfam = atk_bytes - legit_byte_sets[pos]
            if unfam:
                probed.add(pos)

    extended = all_detected | probed
    new_hit = extended & expected_set
    new_missed = expected_set - extended

    if probed:
        print(f"  Gap probing: +{len(probed)} positions recovered → "
              f"{len(new_hit)}/{len(expected_set)} total")
        if new_missed and len(new_missed) <= 10:
            print(f"    Still missed: {sorted(new_missed)}")
    all_detected = extended

    # Derive rules
    candidates, runs, pos_data = derive_long_rules(
        list(all_detected), attack_payloads, legit_payloads
    )

    # Show runs
    print(f"\n  Consecutive runs found: {len(runs)}")
    for run in runs:
        first, last = run[0], run[-1]
        length = len(run)
        const = sum(1 for p in run if pos_data[p]["is_constant"])
        print(f"    pos {first:>4}-{last:<4} ({length:>2} bytes, "
              f"{const} constant)")

    # Show constancy per position (condensed for long runs)
    print(f"\n  Position constancy (showing first/last of each run):")
    print(f"  {'Pos':>6} {'L4':>5} {'Const':>7} {'#Uniq':>6} "
          f"{'Top':>6} {'Coverage':>9}")
    print(f"  {'-' * 47}")

    for run in runs:
        show = set()
        if len(run) <= 8:
            show = set(run)
        else:
            show = set(run[:3]) | set(run[-3:])

        for pos in run:
            if pos not in show:
                continue
            info = pos_data[pos]
            top = info["top_val"]
            c = "YES" if info["is_constant"] else "no"
            print(f"  {pos:>6} {UDP_HDR + pos:>5} {c:>7} "
                  f"{info['n_unique']:>6} "
                  f"0x{top:02X}  {info['constancy']:>8.0%}")

        if len(run) > 8:
            omitted = len(run) - 6
            print(f"  {'...':>6} {'':>5} {'':>7} {'':>6} "
                  f"{'':>6} ({omitted} more)")

    # Show candidate rules grouped by type
    short_rules = [r for r in candidates if r.length <= 4]
    long_rules = [r for r in candidates if r.length > 4]

    if short_rules:
        print(f"\n  Short rules (1-4B, custom dim): {len(short_rules)} "
              f"candidates (showing top 5)")
        for r in short_rules[:5]:
            print(f"    {r.to_edn()}")
            print(f"      {r.description}  TP={r.tp}/{len(attack_payloads)} "
                  f"({r.tp / len(attack_payloads) * 100:.0f}%)")

    if long_rules:
        print(f"\n  Long rules (5+B, PatternGuard): {len(long_rules)} "
              f"candidates")
        print(f"  {'─' * 70}")

        for r in long_rules[:10]:
            enforced = "ENFORCEABLE" if r.enforceable else "TOO DEEP"
            print(f"\n    {r.to_edn()}")
            print(f"      {r.description}")
            print(f"      TP={r.tp}/{len(attack_payloads)} "
                  f"({r.tp / len(attack_payloads) * 100:.0f}%)  "
                  f"Active: {r.active}/{r.length}B  "
                  f"[{enforced}]")

    # Select best
    selected, coverage = select_best(candidates, attack_payloads)

    print(f"\n  OPTIMAL SELECTION:")
    print(f"  {'─' * 70}")

    for r in selected:
        enforced = "OK" if r.enforceable else "NEEDS DECOMPOSITION"
        print(f"    {r.to_edn()}")
        print(f"      {r.cost_label}  TP={r.tp}  [{enforced}]")

    print(f"\n    Coverage: {coverage:.1f}%")
    print(f"    Rules: {len(selected)}")


def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("LONG BYTE MATCH DERIVATION: 32-byte Signatures")
    print("=" * 80)

    # Train
    print("\nTraining on 500 normal payloads...")
    analyzer = WindowedPayloadAnalyzer()
    for _ in range(500):
        analyzer.learn(make_normal(rng))
    analyzer.freeze()
    print(f"  {analyzer.count} packets, {NUM_WINDOWS} windows")

    legit_payloads = [make_normal(rng) for _ in range(200)]

    # Scenario A: 32 constant bytes near header
    run_scenario(
        "A) 32-byte constant shellcode at payload[0-31] (L4 offset 8-39)",
        make_attack_near32,
        list(range(0, 32)),
        analyzer, legit_payloads, rng,
    )

    # Scenario B: 32 constant bytes deep
    run_scenario(
        "B) 32-byte constant at payload[200-231] (L4 offset 208-239)",
        make_attack_deep32,
        list(range(200, 232)),
        analyzer, legit_payloads, rng,
    )

    # Scenario C: 32-byte sparse (24 constant + 8 variable/familiar)
    expected_c = [i for i in range(32) if i not in PROTO_VARIABLE_POS]
    run_scenario(
        "C) 32-byte protocol header (22 constant + 10 variable)",
        make_attack_sparse32,
        expected_c,
        analyzer, legit_payloads, rng,
    )

    # Scenario D: split near + deep
    run_scenario(
        "D) Split: 16B near header [0-15] + 16B deep [300-315]",
        make_attack_split,
        list(range(0, 16)) + list(range(300, 316)),
        analyzer, legit_payloads, rng,
    )

    print(f"\n{'=' * 80}")
    print("TAKEAWAYS")
    print(f"{'=' * 80}")
    print("""
  Near-header constant → single 32-byte PatternGuard, 100% TP, 1 entry
  Deep constant         → decomposes to 1-4B custom dims, still effective
  Sparse (constant+var) → PatternGuard with 00 masks at variable positions
  Split near+deep       → PatternGuard for near, custom dim for deep

  The algorithm naturally discovers the LONGEST enforceable pattern that
  maximizes TP. Longer patterns are more specific (less collision risk in
  production), while shorter patterns at deep offsets use custom dim slots.
""")


if __name__ == "__main__":
    main()
