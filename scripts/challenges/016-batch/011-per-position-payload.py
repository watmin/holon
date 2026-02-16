#!/usr/bin/env python3
"""
Per-Position Payload Analysis: Surgical Byte Anomaly Detection

THE IDEA:
=========
A payload is a map: {position: byte_value}

  {0: 0x47, 1: 0x4D, 2: 0x01, 3: 0xAB, ...}

Encode each byte position INDEPENDENTLY. During learning, each position gets
its own accumulator that learns the frequency distribution of byte values seen
at that position. During detection, each position is scored independently —
no cross-talk, no bundling noise.

TWO-TIER SYSTEM:
================
Tier 1 (fast): whole-payload vector (map encoding) → is this packet anomalous?
  - Single vector comparison
  - Fast screening: YES/NO

Tier 2 (surgical): per-position scoring → WHICH bytes are anomalous?
  - 32 independent comparisons
  - Pinpoints exact byte positions
  - Generates surgical l4-match rules

WHAT THE ACCUMULATOR GIVES US:
==============================
At each position, the accumulator is a float vector that preserves frequency.
If position 0 sees 0x47 in 95% of packets, the accumulator at position 0 is
dominated by the vector for "0x47". When a new packet has 0x90 at position 0,
the similarity to that accumulator is LOW → anomalous.

This is BETTER than a frequency table because:
- Values with similar byte patterns have similar vectors → graceful degradation
- The accumulator naturally handles multi-modal distributions
- Works with the existing holon primitive set (accumulate, normalize, cosine_similarity)

CONSTRAINT:
===========
Stateless, per-window, unidirectional. No flow tracking.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import coherence, prototype


# =====================================================================
# Simulated Game Protocol (same as 010)
# =====================================================================

GAME_MSG_TYPES = {
    "move":       bytes([0x47, 0x4D, 0x01, 0x00]),
    "shoot":      bytes([0x47, 0x4D, 0x02, 0x00]),
    "chat":       bytes([0x47, 0x4D, 0x03, 0x00]),
    "heartbeat":  bytes([0x47, 0x4D, 0x04, 0x00]),
    "spawn":      bytes([0x47, 0x4D, 0x05, 0x00]),
    "inventory":  bytes([0x47, 0x4D, 0x06, 0x00]),
}


def make_legit_payload(rng, length=32):
    msg_type = rng.choice(list(GAME_MSG_TYPES.keys()))
    header = bytearray(GAME_MSG_TYPES[msg_type])
    # Sequence number varies per packet
    header[2] = rng.integers(0, 256)
    header[3] = rng.integers(0, 256)

    if msg_type == "move":
        body = bytearray(rng.integers(0, 256, size=12))
    elif msg_type == "shoot":
        body = bytearray(rng.integers(0, 256, size=8))
    elif msg_type == "chat":
        text = rng.choice(["gg", "nice", "help", "go go", "lol", "wp"])
        body = bytearray(text.encode("ascii"))
        body += bytearray(length - len(header) - len(body))
    elif msg_type == "heartbeat":
        body = bytearray([0x00] * 4 + list(rng.integers(0, 256, size=4)))
    elif msg_type == "spawn":
        body = bytearray(rng.integers(0, 256, size=16))
    else:
        body = bytearray(rng.integers(0, 64, size=8))

    payload = bytes(header + body)[:length]
    if len(payload) < length:
        payload += bytes(length - len(payload))
    return payload


def make_attack_overflow(rng, length=32):
    """NOP sled + overflow — completely wrong bytes at every position."""
    payload = bytearray([0x90] * 8 + [0x41] * 8)
    payload += bytearray(rng.integers(0x80, 0xFF, size=length - 16))
    return bytes(payload[:length])


def make_attack_spoofed(rng, length=32):
    """Right magic bytes, wrong message type + high-byte garbage."""
    header = bytearray([0x47, 0x4D, 0xFF, 0xFF])
    body = bytearray(rng.integers(0xC0, 0xFF, size=length - 4))
    return bytes(header + body)[:length]


def make_attack_subtle(rng, length=32):
    """Subtle: valid game header but unusual body pattern.
    Uses byte values in a range the game protocol rarely produces."""
    msg_type = rng.choice(list(GAME_MSG_TYPES.keys()))
    header = bytearray(GAME_MSG_TYPES[msg_type])
    header[2] = rng.integers(0, 256)
    header[3] = rng.integers(0, 256)
    # Subtly wrong body: uses bytes 0x70-0x7F exclusively
    body = bytearray(rng.integers(0x70, 0x80, size=length - 4))
    return bytes(header + body)[:length]


# =====================================================================
# Per-Position Accumulator System
# =====================================================================

class PositionalPayloadAnalyzer:
    """Per-position byte familiarity using independent accumulators."""

    def __init__(self, client, num_positions=32):
        self.client = client
        self.num_positions = num_positions
        self.accumulators = [client.create_accumulator() for _ in range(num_positions)]
        self.packet_count = 0

        # Pre-compute byte vectors for all 256 values (codebook)
        self.byte_vectors = {}
        for b in range(256):
            self.byte_vectors[b] = client.get_vector(f"0x{b:02x}")

        # Whole-payload accumulator
        self.whole_accum = client.create_accumulator()

    def learn(self, payload):
        """Add a payload to the familiarity model."""
        for i, byte_val in enumerate(payload[:self.num_positions]):
            self.accumulators[i] = self.client.accumulate(
                self.accumulators[i], self.byte_vectors[byte_val]
            )

        # Also accumulate whole-payload vector
        whole_vec = self._encode_whole(payload)
        self.whole_accum = self.client.accumulate(self.whole_accum, whole_vec)
        self.packet_count += 1

    def _encode_whole(self, payload):
        """Encode full payload as a map for tier-1 screening."""
        data = {str(i): f"0x{b:02x}" for i, b in enumerate(payload[:self.num_positions])}
        return self.client.encode(data)

    def score_whole(self, payload):
        """Tier 1: whole-payload anomaly score."""
        vec = self._encode_whole(payload)
        baseline = self.client.normalize_accumulator(self.whole_accum)
        return cosine_similarity(vec, baseline)

    def score_positions(self, payload):
        """Tier 2: per-position anomaly scores.

        Returns array of similarities: high = familiar, low = unfamiliar.
        """
        scores = np.zeros(min(len(payload), self.num_positions))
        for i, byte_val in enumerate(payload[:self.num_positions]):
            byte_vec = self.byte_vectors[byte_val]
            baseline_i = self.client.normalize_accumulator(self.accumulators[i])
            scores[i] = cosine_similarity(byte_vec, baseline_i)
        return scores

    def most_common_byte(self, position):
        """Find the most familiar byte at a given position using cleanup."""
        baseline_i = self.client.normalize_accumulator(self.accumulators[position])
        best_byte = None
        best_sim = -2.0
        for b in range(256):
            sim = cosine_similarity(self.byte_vectors[b], baseline_i)
            if sim > best_sim:
                best_sim = sim
                best_byte = b
        return best_byte, best_sim

    def top_k_bytes(self, position, k=5):
        """Find the top k most familiar bytes at a position."""
        baseline_i = self.client.normalize_accumulator(self.accumulators[position])
        scored = []
        for b in range(256):
            sim = cosine_similarity(self.byte_vectors[b], baseline_i)
            scored.append((b, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def generate_rules(self, payload, pos_scores, sim_threshold=0.0):
        """Generate l4-match rules for anomalous positions.

        For each position below threshold, use the actual byte from the
        attack payload as the match byte.
        """
        anomalous = []
        for i, (score, byte_val) in enumerate(zip(pos_scores, payload[:self.num_positions])):
            if score < sim_threshold:
                anomalous.append((i, byte_val, score))

        if not anomalous:
            return []

        # Group contiguous positions into rules
        rules = []
        i = 0
        while i < len(anomalous):
            start_pos = anomalous[i][0]
            run = [anomalous[i]]
            while (i + 1 < len(anomalous) and
                   anomalous[i + 1][0] == anomalous[i][0] + 1):
                i += 1
                run.append(anomalous[i])
            i += 1

            offset = 8 + start_pos  # 8-byte UDP header
            match_hex = "".join(f"{r[1]:02X}" for r in run)
            mask_hex = "FF" * len(run)

            rules.append({
                "rule": f'(l4-match {offset} "{match_hex}" "{mask_hex}")',
                "offset": offset,
                "positions": [r[0] for r in run],
                "bytes": [r[1] for r in run],
                "scores": [r[2] for r in run],
            })

        return rules


# =====================================================================
# Experiments
# =====================================================================

def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    # Build the analyzer and train on legit traffic
    analyzer = PositionalPayloadAnalyzer(client, num_positions=32)

    print("=" * 90)
    print("Training on 500 legitimate game protocol packets...")
    print("=" * 90)

    for _ in range(500):
        analyzer.learn(make_legit_payload(rng))
    print(f"  Trained on {analyzer.packet_count} packets")

    # ================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: Tier 1 — Whole-Payload Anomaly Scoring")
    print("=" * 90)
    print()

    test_sets = {
        "Legitimate": [make_legit_payload(rng) for _ in range(50)],
        "Attack: Overflow": [make_attack_overflow(rng) for _ in range(50)],
        "Attack: Spoofed Hdr": [make_attack_spoofed(rng) for _ in range(50)],
        "Attack: Subtle": [make_attack_subtle(rng) for _ in range(50)],
    }

    print(f"  {'Traffic':>22} {'Mean Sim':>9} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-' * 53}")
    for name, payloads in test_sets.items():
        scores = np.array([analyzer.score_whole(p) for p in payloads])
        print(f"  {name:>22} {np.mean(scores):>9.4f} {np.std(scores):>7.4f} "
              f"{np.min(scores):>7.4f} {np.max(scores):>7.4f}")

    # ================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: Tier 2 — Per-Position Anomaly Scoring")
    print("=" * 90)
    print()
    print("Each position scored independently. Shows WHERE the anomalous bytes are.")
    print()

    for name, payloads in test_sets.items():
        sample = payloads[0]
        pos_scores = analyzer.score_positions(sample)

        print(f"  {name}:")
        print(f"    Bytes: {' '.join(f'{b:02x}' for b in sample[:32])}")

        # Visual bar chart
        print(f"    Score: ", end="")
        for s in pos_scores:
            if s > 0.5:
                print("█", end="")    # very familiar
            elif s > 0.2:
                print("▓", end="")    # familiar
            elif s > 0.0:
                print("▒", end="")    # somewhat familiar
            elif s > -0.2:
                print("░", end="")    # unfamiliar
            else:
                print("·", end="")    # very unfamiliar
        print()

        print(f"    Vals:  ", end="")
        for s in pos_scores:
            print(f"{s:>5.2f} ", end="")
        print()

        # Most anomalous positions
        worst = np.argsort(pos_scores)[:5]
        print(f"    Most anomalous: ", end="")
        for w in worst:
            print(f"pos {w}=0x{sample[w]:02x} (sim={pos_scores[w]:.2f})  ", end="")
        print()
        print()

    # ================================================================
    print("=" * 90)
    print("EXPERIMENT 3: What SHOULD Be at Each Position?")
    print("=" * 90)
    print()
    print("Use the accumulator to recover the most familiar byte at each position.")
    print("This is the 'expected' protocol structure learned from traffic.")
    print()

    print("  Position  Top-1 (sim)   Top-2 (sim)   Top-3 (sim)   Coverage")
    print("  " + "-" * 72)

    for pos in range(min(16, 32)):  # First 16 positions
        top = analyzer.top_k_bytes(pos, k=3)
        coverage = ""
        if top[0][1] > 0.8:
            coverage = "FIXED"
        elif top[0][1] > 0.3:
            coverage = "COMMON"
        else:
            coverage = "VARIED"

        print(f"  {pos:>8}  0x{top[0][0]:02x} ({top[0][1]:>5.2f})  "
              f"0x{top[1][0]:02x} ({top[1][1]:>5.2f})  "
              f"0x{top[2][0]:02x} ({top[2][1]:>5.2f})  {coverage}")

    # ================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 4: Rule Generation from Per-Position Analysis")
    print("=" * 90)
    print()

    for attack_name, payloads in [
        ("Overflow", test_sets["Attack: Overflow"]),
        ("Spoofed Hdr", test_sets["Attack: Spoofed Hdr"]),
        ("Subtle", test_sets["Attack: Subtle"]),
    ]:
        # Score positions for a sample attack packet
        sample = payloads[0]
        pos_scores = analyzer.score_positions(sample)

        # Find anomalous positions
        rules = analyzer.generate_rules(sample, pos_scores, sim_threshold=0.0)

        print(f"  {attack_name}:")
        print(f"    Payload: {' '.join(f'{b:02x}' for b in sample[:32])}")
        print(f"    Anomalous positions: {sum(1 for s in pos_scores if s < 0.0)}/32")

        if rules:
            for rule in rules:
                print(f"    Rule: {rule['rule']}")
                print(f"      Positions: {rule['positions']}")
                print(f"      Scores: [{', '.join(f'{s:.2f}' for s in rule['scores'])}]")
            print()

            # Validate: false positive rate
            print(f"    Validation:")
            for rule in rules:
                legit_hits = sum(
                    1 for p in test_sets["Legitimate"]
                    if all(p[pos] == mb for pos, mb in zip(rule["positions"], rule["bytes"]))
                )
                attack_hits = sum(
                    1 for p in payloads
                    if all(p[pos] == mb for pos, mb in zip(rule["positions"], rule["bytes"]))
                )
                print(f"      {rule['rule']}:")
                print(f"        Legit: {legit_hits}/50 ({legit_hits*2}%)  "
                      f"Attack: {attack_hits}/50 ({attack_hits*2}%)")
        else:
            print(f"    No rules generated (no positions below threshold)")
        print()

    # ================================================================
    print("=" * 90)
    print("EXPERIMENT 5: Consensus Rule from Multiple Attack Samples")
    print("=" * 90)
    print()
    print("Score multiple attack packets and find positions that are")
    print("CONSISTENTLY anomalous with CONSISTENT byte values.")
    print()

    for attack_name, payloads in [
        ("Overflow", test_sets["Attack: Overflow"]),
        ("Spoofed Hdr", test_sets["Attack: Spoofed Hdr"]),
        ("Subtle", test_sets["Attack: Subtle"]),
    ]:
        # Score all attack packets
        all_pos_scores = np.array([analyzer.score_positions(p) for p in payloads])
        mean_pos_scores = np.mean(all_pos_scores, axis=0)
        std_pos_scores = np.std(all_pos_scores, axis=0)

        # Positions consistently anomalous (mean < 0 AND low variance)
        consistently_anomalous = []
        for pos in range(32):
            if mean_pos_scores[pos] < 0.0:
                # Check byte consensus at this position
                byte_vals = [p[pos] for p in payloads]
                unique_bytes = set(byte_vals)
                most_common = max(unique_bytes, key=lambda b: byte_vals.count(b))
                consensus_pct = byte_vals.count(most_common) / len(byte_vals)

                consistently_anomalous.append({
                    "pos": pos,
                    "mean_sim": mean_pos_scores[pos],
                    "std_sim": std_pos_scores[pos],
                    "byte": most_common,
                    "consensus": consensus_pct,
                    "unique_count": len(unique_bytes),
                })

        print(f"  {attack_name}:")
        print(f"    Consistently anomalous positions (mean sim < 0):")
        print(f"    {'Pos':>4} {'MeanSim':>8} {'Std':>6} {'Byte':>6} {'Consensus':>10} {'Unique':>7}")
        print(f"    {'-' * 45}")

        for ca in consistently_anomalous:
            print(f"    {ca['pos']:>4} {ca['mean_sim']:>8.3f} {ca['std_sim']:>6.3f} "
                  f"0x{ca['byte']:02x}{ca['consensus']:>9.0%} {ca['unique_count']:>7}")

        # Generate consensus rule from consistently anomalous + high consensus
        consensus_rule_bytes = {}
        for ca in consistently_anomalous:
            if ca["consensus"] > 0.5:  # >50% of attack packets agree on this byte
                consensus_rule_bytes[ca["pos"]] = ca["byte"]

        if consensus_rule_bytes:
            # Build contiguous rules
            positions = sorted(consensus_rule_bytes.keys())
            rules = []
            i = 0
            while i < len(positions):
                start = positions[i]
                run = [start]
                while i + 1 < len(positions) and positions[i + 1] == positions[i] + 1:
                    i += 1
                    run.append(positions[i])
                i += 1

                offset = 8 + start
                match_hex = "".join(f"{consensus_rule_bytes[p]:02X}" for p in run)
                mask_hex = "FF" * len(run)
                rule = f'(l4-match {offset} "{match_hex}" "{mask_hex}")'
                rules.append((rule, run))

            print(f"\n    Consensus rules:")
            for rule, positions_list in rules:
                print(f"      {rule}")
                # Validate
                legit_hits = sum(
                    1 for p in test_sets["Legitimate"]
                    if all(p[pos] == consensus_rule_bytes[pos] for pos in positions_list)
                )
                attack_hits = sum(
                    1 for p in payloads
                    if all(p[pos] == consensus_rule_bytes[pos] for pos in positions_list)
                )
                print(f"        Legit: {legit_hits}/50 ({legit_hits*2}%)  "
                      f"Attack: {attack_hits}/50 ({attack_hits*2}%)")
        print()

    # ================================================================
    print("=" * 90)
    print("EXPERIMENT 6: Full Pipeline — Mixed Window → Detect → Locate → Rule")
    print("=" * 90)
    print()

    # 75% legit, 25% attack overflow
    mixed_legit = [make_legit_payload(rng) for _ in range(60)]
    mixed_attack = [make_attack_overflow(rng) for _ in range(20)]
    mixed = mixed_legit + mixed_attack
    mixed_labels = ["legit"] * 60 + ["attack"] * 20
    indices = np.arange(len(mixed))
    rng.shuffle(indices)
    mixed = [mixed[i] for i in indices]
    mixed_labels = [mixed_labels[i] for i in indices]

    # Tier 1: whole-payload screening
    whole_scores = np.array([analyzer.score_whole(p) for p in mixed])

    # Adaptive threshold from training distribution
    legit_whole = np.array([analyzer.score_whole(p) for p in test_sets["Legitimate"]])
    whole_threshold = np.mean(legit_whole) - 3 * np.std(legit_whole)

    flagged = whole_scores < whole_threshold
    flagged_indices = np.where(flagged)[0]

    tp = sum(1 for i in flagged_indices if mixed_labels[i] == "attack")
    fp = sum(1 for i in flagged_indices if mixed_labels[i] == "legit")
    fn = sum(1 for i in range(len(mixed)) if not flagged[i] and mixed_labels[i] == "attack")

    print(f"  Step 1 — Tier 1 Screening (whole-payload sim < {whole_threshold:.3f}):")
    print(f"    Flagged: {len(flagged_indices)} packets")
    print(f"    True positive: {tp}, False positive: {fp}, False negative: {fn}")

    # Tier 2: per-position analysis on flagged packets
    if len(flagged_indices) > 0:
        flagged_payloads = [mixed[i] for i in flagged_indices]
        all_pos_scores = np.array([analyzer.score_positions(p) for p in flagged_payloads])
        mean_scores = np.mean(all_pos_scores, axis=0)

        print(f"\n  Step 2 — Tier 2 Per-Position Analysis (across {len(flagged_payloads)} flagged packets):")
        print(f"    Mean per-position similarity to baseline:")
        print(f"    Pos: " + " ".join(f"{i:>5}" for i in range(32)))
        print(f"    Sim: " + " ".join(f"{s:>5.2f}" for s in mean_scores))
        print(f"    Bar: ", end="")
        for s in mean_scores:
            if s > 0.5: print("█", end="")
            elif s > 0.2: print("▓", end="")
            elif s > 0.0: print("▒", end="")
            elif s > -0.2: print("░", end="")
            else: print("·", end="")
        print()

        # Positions where mean < 0 → consistently unfamiliar
        hot_positions = [i for i, s in enumerate(mean_scores) if s < 0.0]
        print(f"    Consistently unfamiliar positions: {hot_positions}")

        # Step 3: generate consensus rule
        consensus_bytes = {}
        for pos in hot_positions:
            byte_vals = [p[pos] for p in flagged_payloads]
            unique = set(byte_vals)
            most_common = max(unique, key=lambda b: byte_vals.count(b))
            pct = byte_vals.count(most_common) / len(byte_vals)
            if pct > 0.5:
                consensus_bytes[pos] = (most_common, pct)

        if consensus_bytes:
            positions = sorted(consensus_bytes.keys())
            print(f"\n  Step 3 — Consensus Byte Values at Hot Positions:")
            for pos in positions:
                byte_val, pct = consensus_bytes[pos]
                normal_byte, normal_sim = analyzer.most_common_byte(pos)
                print(f"    pos {pos:>2}: attack=0x{byte_val:02x} ({pct:.0%})  "
                      f"normal=0x{normal_byte:02x} (sim={normal_sim:.2f})")

            # Build rules
            rules = []
            i = 0
            while i < len(positions):
                start = positions[i]
                run = [start]
                while i + 1 < len(positions) and positions[i + 1] == positions[i] + 1:
                    i += 1
                    run.append(positions[i])
                i += 1

                offset = 8 + start
                match_hex = "".join(f"{consensus_bytes[p][0]:02X}" for p in run)
                mask_hex = "FF" * len(run)
                rule = f'(l4-match {offset} "{match_hex}" "{mask_hex}")'
                rules.append((rule, run))

            print(f"\n  Step 4 — Generated Rules:")
            for rule, run in rules:
                legit_hits = sum(
                    1 for p in mixed_legit
                    if all(p[pos] == consensus_bytes[pos][0] for pos in run)
                )
                attack_hits = sum(
                    1 for p in mixed_attack
                    if all(p[pos] == consensus_bytes[pos][0] for pos in run)
                )
                print(f"    {rule}")
                print(f"      Legit: {legit_hits}/{len(mixed_legit)} ({legit_hits/len(mixed_legit)*100:.0f}%)  "
                      f"Attack: {attack_hits}/{len(mixed_attack)} ({attack_hits/len(mixed_attack)*100:.0f}%)")

    print()


if __name__ == "__main__":
    main()
