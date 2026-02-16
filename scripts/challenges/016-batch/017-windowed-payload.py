#!/usr/bin/env python3
"""
Windowed Payload Analysis: Full MTU Coverage at d=4096

APPROACH:
=========
Split the payload into 64-byte windows. Each window gets its own
accumulator (one 4096-dim vector each — super cheap).

  Window 0:  bytes 0-63    → accum_0
  Window 1:  bytes 64-127  → accum_1
  ...
  Window 23: bytes 1472-1499 → accum_23

Detection:
  1. Score each window against its accumulator
  2. Low-scoring windows = anomalous region
  3. Drill-down within the anomalous window (64 fields, well within √4096=64)
  4. Generate l4-match rule from unfamiliar positions

SCENARIO:
=========
1500-byte payload. Normal traffic has familiar bytes everywhere.
Attack hides anomalous bytes at positions 1490-1499 — the very end.
This is the worst case: the signal is buried in the last 10 bytes.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity


DIMENSIONS = 4096
PAYLOAD_SIZE = 1500
WINDOW_SIZE = 64
NUM_WINDOWS = (PAYLOAD_SIZE + WINDOW_SIZE - 1) // WINDOW_SIZE  # 24 windows


class WindowedPayloadAnalyzer:
    """Full-MTU payload analysis using windowed accumulators."""

    def __init__(self, dimensions=4096, window_size=64, payload_size=1500):
        self.client = HolonClient(dimensions=dimensions)
        self.window_size = window_size
        self.payload_size = payload_size
        self.num_windows = (payload_size + window_size - 1) // window_size

        # One accumulator per window
        self.accumulators = [
            self.client.create_accumulator() for _ in range(self.num_windows)
        ]
        self.baselines = [None] * self.num_windows
        self.packet_count = 0

    def _window_dict(self, payload, window_idx):
        """Encode a window's bytes as a map."""
        start = window_idx * self.window_size
        end = min(start + self.window_size, len(payload))
        return {
            f"p{i - start}": f"0x{payload[i]:02x}"
            for i in range(start, end)
        }

    def learn(self, payload):
        """Accumulate a payload into all window baselines."""
        for w in range(self.num_windows):
            d = self._window_dict(payload, w)
            vec = self.client.encode(d)
            self.accumulators[w] = self.client.accumulate(self.accumulators[w], vec)
        self.packet_count += 1

    def freeze(self):
        """Normalize all accumulators into baselines."""
        for w in range(self.num_windows):
            self.baselines[w] = self.client.normalize_accumulator(self.accumulators[w])

    def score_windows(self, payload):
        """Score each window against its baseline. Returns list of (window, sim)."""
        results = []
        for w in range(self.num_windows):
            d = self._window_dict(payload, w)
            vec = self.client.encode(d)
            sim = cosine_similarity(vec, self.baselines[w])
            results.append((w, sim))
        return results

    def drill_down_window(self, payload, window_idx):
        """Drill-down within a specific window to find unfamiliar positions."""
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
# Payload Generators
# =====================================================================

def make_normal(rng, size=1500):
    """Normal payload: familiar bytes at every position."""
    familiar = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30]
    return [int(rng.choice(familiar)) for _ in range(size)]


def make_attack_tail(rng, size=1500):
    """Attack with anomalous bytes at positions 1490-1499 ONLY."""
    familiar = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30]
    unfamiliar = [0xFF, 0xAC, 0xFB, 0xCA, 0xDE, 0xAD, 0xBE, 0xEF]
    payload = [int(rng.choice(familiar)) for _ in range(size)]
    for i in range(1490, min(1500, size)):
        payload[i] = int(rng.choice(unfamiliar))
    return payload


def make_attack_scattered(rng, size=1500):
    """Attack with anomalous bytes scattered: positions 10-13, 750-753, 1496-1499."""
    familiar = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30]
    unfamiliar = [0xFF, 0xAC, 0xFB, 0xCA, 0xDE, 0xAD, 0xBE, 0xEF]
    payload = [int(rng.choice(familiar)) for _ in range(size)]
    for i in list(range(10, 14)) + list(range(750, 754)) + list(range(1496, 1500)):
        payload[i] = int(rng.choice(unfamiliar))
    return payload


def make_attack_middle(rng, size=1500):
    """Attack with anomalous bytes at positions 700-709 only."""
    familiar = [0x00, 0x01, 0x02, 0x10, 0x20, 0x30]
    unfamiliar = [0xFF, 0xAC, 0xFB, 0xCA, 0xDE, 0xAD, 0xBE, 0xEF]
    payload = [int(rng.choice(familiar)) for _ in range(size)]
    for i in range(700, 710):
        payload[i] = int(rng.choice(unfamiliar))
    return payload


# =====================================================================
# Main
# =====================================================================

def main():
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("WINDOWED PAYLOAD ANALYSIS: Full MTU at d=4096")
    print("=" * 80)
    print()
    print(f"  Payload size: {PAYLOAD_SIZE} bytes")
    print(f"  Window size:  {WINDOW_SIZE} bytes")
    print(f"  Num windows:  {NUM_WINDOWS}")
    print(f"  Dimensions:   {DIMENSIONS}")
    print(f"  Memory:       {NUM_WINDOWS} accumulators × 4096 dims × 8 bytes = "
          f"{NUM_WINDOWS * DIMENSIONS * 8 / 1024:.1f} KB")
    print()

    # ================================================================
    # LEARN
    # ================================================================
    print("PHASE 1: Learning from 500 normal payloads...")
    analyzer = WindowedPayloadAnalyzer(DIMENSIONS, WINDOW_SIZE, PAYLOAD_SIZE)
    for _ in range(500):
        analyzer.learn(make_normal(rng))
    analyzer.freeze()
    print(f"  Trained on {analyzer.packet_count} packets")
    print(f"  {NUM_WINDOWS} baselines frozen")

    # ================================================================
    # TEST ATTACKS
    # ================================================================
    attacks = [
        ("Tail (bytes 1490-1499)", make_attack_tail, list(range(1490, 1500))),
        ("Middle (bytes 700-709)", make_attack_middle, list(range(700, 710))),
        ("Scattered (10-13, 750-753, 1496-1499)", make_attack_scattered,
         list(range(10, 14)) + list(range(750, 754)) + list(range(1496, 1500))),
    ]

    legit_payloads = [make_normal(rng) for _ in range(100)]

    for attack_name, attack_fn, expected_positions in attacks:
        print(f"\n{'=' * 80}")
        print(f"ATTACK: {attack_name}")
        print(f"{'=' * 80}")

        attack_payloads = [attack_fn(rng) for _ in range(100)]
        sample = attack_payloads[0]

        # Step 1: Score all windows
        window_scores = analyzer.score_windows(sample)

        # Also score a legit payload for reference
        legit_scores = analyzer.score_windows(legit_payloads[0])

        expected_windows = set()
        for pos in expected_positions:
            expected_windows.add(pos // WINDOW_SIZE)

        print(f"\n  Window scores (anomalous windows marked):")
        print(f"  {'Win':>4} {'Range':>14} {'LegitSim':>9} {'AtkSim':>9} {'Drop':>6} {'Status':>10}")
        print(f"  {'-' * 58}")

        anomalous_windows = []
        for w in range(NUM_WINDOWS):
            start = w * WINDOW_SIZE
            end = min(start + WINDOW_SIZE, PAYLOAD_SIZE)
            l_sim = legit_scores[w][1]
            a_sim = window_scores[w][1]
            drop = l_sim - a_sim

            if w in expected_windows:
                status = "ANOMALOUS" if drop > 0.02 else "MISSED"
                marker = " ← "
            else:
                status = "normal"
                marker = "   "

            if drop > 0.02:
                anomalous_windows.append(w)

            # Only print windows that are interesting
            if w in expected_windows or drop > 0.02 or w < 2 or w >= NUM_WINDOWS - 2:
                print(f"  {w:>4} {start:>6}-{end-1:<6} {l_sim:>9.4f} {a_sim:>9.4f} "
                      f"{drop:>6.3f} {status:>10}{marker}")

        # Show how many windows are clean
        clean = NUM_WINDOWS - len(anomalous_windows)
        print(f"  ... ({clean} other windows: normal)")

        # Step 2: Drill-down on anomalous windows
        print(f"\n  Drill-down on {len(anomalous_windows)} anomalous window(s):")
        all_unfamiliar = []

        for w in anomalous_windows:
            start = w * WINDOW_SIZE
            end = min(start + WINDOW_SIZE, PAYLOAD_SIZE)
            drill = analyzer.drill_down_window(sample, w)

            print(f"\n    Window {w} (bytes {start}-{end-1}):")
            print(f"    {'Pos':>6} {'Byte':>6} {'Sim':>8} {'Verdict':>12}")
            print(f"    {'-' * 36}")

            for pos, byte_val, sim in drill:
                expected = pos in expected_positions
                if sim < -0.005:
                    verdict = "UNFAMILIAR"
                elif sim < 0.01:
                    verdict = "borderline"
                else:
                    verdict = "familiar"

                # Only print unfamiliar or expected positions
                if expected or sim < 0.01:
                    marker = " ✓" if expected and sim < 0.01 else " ✗" if expected else ""
                    print(f"    {pos:>6} 0x{byte_val:02x} {sim:>8.4f} {verdict:>12}{marker}")

                if sim < 0.005:
                    all_unfamiliar.append((pos, byte_val, sim))

        # Step 3: Generate l4-match rules — respecting eBPF constraints
        #
        # CONSTRAINTS (from the real filter):
        #   - l4-match offset N  ↔  payload byte at position (N - 8)
        #   - Masks are always 0xFF (exact byte match) or 0x00 (wildcard)
        #   - 1-4 byte patterns → custom dim slots (7 total, arbitrary offset)
        #   - 5-64 byte patterns → PatternGuard (offset + len must fit in
        #     64-byte pattern_data window, i.e. L4 offset < 64)
        #   - Max 32 l4-match rules per destination scope
        #
        # STRATEGY:
        #   For each unfamiliar position, find the most common attack byte
        #   that never appears in legit traffic. Generate 1-byte exact-match
        #   rules at the best positions. Multiple rules at the same offset
        #   with different byte values share one custom dim slot.

        print(f"\n  Rule generation (exact byte match, 0xFF masks):")

        if not all_unfamiliar:
            print(f"    No unfamiliar positions detected.")
        else:
            unfam_positions = sorted(set(p for p, _, _ in all_unfamiliar))

            legit_byte_sets = {
                pos: set(p[pos] for p in legit_payloads)
                for pos in unfam_positions
            }

            # For each position, find ALL unfamiliar byte values and their
            # frequency across attack samples
            pos_analysis = []
            for pos in unfam_positions:
                attack_bytes = [p[pos] for p in attack_payloads]
                legit_set = legit_byte_sets[pos]

                # Find unfamiliar byte values (appear in attacks, not in legit)
                byte_counts = Counter(attack_bytes)
                unfamiliar_hits = {
                    val: count
                    for val, count in byte_counts.items()
                    if val not in legit_set
                }

                if not unfamiliar_hits:
                    continue

                # Total coverage: fraction of attacks with ANY unfamiliar byte
                total_unfam = sum(unfamiliar_hits.values())
                coverage = total_unfam / len(attack_bytes)

                # Best single byte: highest individual TP
                best_val, best_count = max(
                    unfamiliar_hits.items(), key=lambda x: x[1]
                )

                pos_analysis.append({
                    "pos": pos,
                    "l4_offset": 8 + pos,
                    "best_val": best_val,
                    "best_tp": best_count / len(attack_bytes),
                    "coverage": coverage,
                    "unfam_vals": unfamiliar_hits,
                })

            if not pos_analysis:
                print(f"    No matchable positions found.")
            else:
                # Rank positions by coverage (what fraction of attacks have
                # ANY unfamiliar byte there)
                pos_analysis.sort(key=lambda p: -p["coverage"])

                # Show per-position analysis
                print(f"\n    Position analysis ({len(pos_analysis)} "
                      f"matchable positions):")
                print(f"    {'Pos':>6} {'L4 Off':>7} {'Coverage':>9} "
                      f"{'Best Byte':>10} {'Best TP':>8} {'# Unfam':>8}")
                print(f"    {'-' * 54}")

                for pa in pos_analysis:
                    print(f"    {pa['pos']:>6} {pa['l4_offset']:>7} "
                          f"{pa['coverage']:>8.0%} "
                          f"  0x{pa['best_val']:02X}     "
                          f"{pa['best_tp']:>7.0%} "
                          f"{len(pa['unfam_vals']):>8}")

                # Generate rules: pick the best position, then generate
                # one rule per unfamiliar byte value there
                best_pos = pos_analysis[0]
                rules_generated = []

                print(f"\n    Best position: payload byte {best_pos['pos']} "
                      f"(L4 offset {best_pos['l4_offset']})")
                print(f"    Coverage: {best_pos['coverage']:.0%} of attacks "
                      f"have an unfamiliar byte here")
                print(f"\n    Rules (1-byte exact match, 1 custom dim slot):")

                # Generate a rule for each unfamiliar byte value at this
                # position (they all share the same custom dim slot)
                for val, count in sorted(
                    best_pos["unfam_vals"].items(),
                    key=lambda x: -x[1]
                ):
                    tp_rate = count / len(attack_payloads)
                    offset = best_pos["l4_offset"]
                    rule_str = (
                        f'(l4-match {offset} '
                        f'"{val:02X}" "FF")'
                    )

                    # Validate
                    tp = sum(
                        1 for p in attack_payloads
                        if p[best_pos["pos"]] == val
                    )
                    fp = sum(
                        1 for p in legit_payloads
                        if p[best_pos["pos"]] == val
                    )
                    rules_generated.append({
                        "rule": rule_str, "tp": tp, "fp": fp, "val": val
                    })

                    print(f"      {rule_str}  "
                          f"TP={tp}/{len(attack_payloads)} "
                          f"({tp_rate:.0%})  FP={fp}/{len(legit_payloads)}")

                # Combined coverage
                combined_tp = sum(
                    1 for p in attack_payloads
                    if p[best_pos["pos"]] not in legit_byte_sets[best_pos["pos"]]
                )
                combined_fp = 0  # All values are unfamiliar by construction

                print(f"\n    Combined (all {len(rules_generated)} rules at "
                      f"same offset, 1 custom dim slot):")
                print(f"      Attack: {combined_tp}/{len(attack_payloads)} "
                      f"({combined_tp / len(attack_payloads) * 100:.0f}% TP)")
                print(f"      Legit:  {combined_fp}/{len(legit_payloads)} "
                      f"(0% FP)")
                print(f"      Cost:   1 custom dim slot, "
                      f"{len(rules_generated)} rules")

    # ================================================================
    # RESOURCE SUMMARY
    # ================================================================
    print(f"\n{'=' * 80}")
    print("RESOURCE SUMMARY")
    print(f"{'=' * 80}")
    print()
    print(f"  Per-destination memory:")
    print(f"    {NUM_WINDOWS} windows × 4096 dims × 8 bytes = "
          f"{NUM_WINDOWS * DIMENSIONS * 8 / 1024:.1f} KB (accumulators)")
    print(f"    {NUM_WINDOWS} windows × 4096 dims × 1 byte  = "
          f"{NUM_WINDOWS * DIMENSIONS / 1024:.1f} KB (baselines)")
    print(f"    Total: {NUM_WINDOWS * DIMENSIONS * 9 / 1024:.1f} KB per destination")
    print()
    print(f"  Per-packet compute:")
    print(f"    {NUM_WINDOWS} window encodings × {WINDOW_SIZE} binds = "
          f"{NUM_WINDOWS * WINDOW_SIZE} bind ops")
    print(f"    {NUM_WINDOWS} cosine similarities (4096-dim)")
    print(f"    Drill-down: only on flagged windows ({WINDOW_SIZE} ops each)")
    print()
    print(f"  Compared to single-vector approach:")
    print(f"    Single: 1 × 1500 binds + 1 cosine = fast but drill-down fails")
    print(f"    Windowed: 24 × 64 binds + 24 cosines = same work, drill-down works")
    print()


if __name__ == "__main__":
    main()
