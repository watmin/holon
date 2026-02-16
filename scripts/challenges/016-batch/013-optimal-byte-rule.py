#!/usr/bin/env python3
"""
Optimal Byte Match Rule Selection: Maximum Coverage, Minimum Resources

PROBLEM:
========
We can generate many l4-match rules from the drill-down, but we're resource
constrained:
  - 7 custom dim slots (1-4 byte matches, O(1) fan-out) — PRECIOUS
  - 32 byte matches per destination scope
  - Ideally: ONE rule that catches everything

GOAL:
=====
Find the single most effective byte match rule. Score every candidate rule by:
  - True positive rate (% of attack packets matched)
  - False positive rate (% of legit packets matched — must be 0%)
  - Rule length (shorter = cheaper)

Then try combining rules (AND logic via contiguous bytes) when single bytes
aren't sufficient.

APPROACH:
=========
1. VSA drill-down identifies candidate positions (unfamiliar bytes)
2. For each position: test exact match, masked match, and range match
3. Rank candidates by coverage with zero false positives
4. Try contiguous combinations for compound rules
5. Output the single best l4-match rule

MASK TRICK:
===========
Instead of matching exact byte 0x90, match with mask 0xF0 → catches
0x90-0x9F. Or mask 0x80 → catches 0x80-0xFF. This is more resilient
to attacker variation while still catching the anomaly.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity


# =====================================================================
# Game Protocol
# =====================================================================

GAME_MSG_TYPES = {
    "move":       bytes([0x47, 0x4D, 0x01, 0x00]),
    "shoot":      bytes([0x47, 0x4D, 0x02, 0x00]),
    "chat":       bytes([0x47, 0x4D, 0x03, 0x00]),
    "heartbeat":  bytes([0x47, 0x4D, 0x04, 0x00]),
    "spawn":      bytes([0x47, 0x4D, 0x05, 0x00]),
    "inventory":  bytes([0x47, 0x4D, 0x06, 0x00]),
}

NUM_PAYLOAD_BYTES = 16


def make_legit_payload(rng, length=32):
    msg_type = rng.choice(list(GAME_MSG_TYPES.keys()))
    header = bytearray(GAME_MSG_TYPES[msg_type])
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
    else:
        body = bytearray(rng.integers(0, 64, size=max(0, length - len(header))))
    payload = bytes(header + body)[:length]
    if len(payload) < length:
        payload += bytes(length - len(payload))
    return payload


def make_attack_overflow(rng, length=32):
    payload = bytearray([0x90] * 8 + [0x41] * 8)
    payload += bytearray(rng.integers(0x80, 0xFF, size=length - 16))
    return bytes(payload[:length])


def make_attack_varied(rng, length=32):
    """Attack with some byte variation — not perfectly uniform."""
    nop = rng.choice([0x90, 0x91, 0x92, 0x93])  # NOP variants
    pad = rng.choice([0x41, 0x42, 0x43])  # padding variants
    payload = bytearray([nop] * 8 + [pad] * 8)
    payload += bytearray(rng.integers(0x80, 0xFF, size=length - 16))
    return bytes(payload[:length])


def make_attack_polymorphic(rng, length=32):
    """Attack where only some positions are stable, others vary."""
    # Positions 0-1: always 0xEB 0xFE (short JMP -2, infinite loop)
    payload = bytearray([0xEB, 0xFE])
    # Positions 2-7: random high bytes (polymorphic decoder stub)
    payload += bytearray(rng.integers(0x80, 0xFF, size=6))
    # Positions 8-15: shellcode varies per sample
    payload += bytearray(rng.integers(0x00, 0xFF, size=8))
    payload += bytearray(rng.integers(0x00, 0xFF, size=length - 16))
    return bytes(payload[:length])


def make_packet_dict(rng, payload_bytes):
    d = {
        "src_ip": f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "src_port": str(rng.integers(1024, 65535)),
        "dst_port": "27015",
    }
    for i, b in enumerate(payload_bytes[:NUM_PAYLOAD_BYTES]):
        d[f"p{i}"] = f"0x{b:02x}"
    return d


def drill_down(client, pkt_dict, baseline):
    results = []
    for field, value in pkt_dict.items():
        role_vec = client.get_vector(field)
        val_vec = client.get_vector(value)
        bound = role_vec * val_vec
        sim = cosine_similarity(bound, baseline)
        results.append((field, value, sim))
    results.sort(key=lambda x: x[2])
    return results


# =====================================================================
# Rule Candidate Evaluation
# =====================================================================

def eval_exact_rule(legit_payloads, attack_payloads, position, byte_val):
    """Evaluate a single exact byte match at a position."""
    fp = sum(1 for p in legit_payloads if p[position] == byte_val)
    tp = sum(1 for p in attack_payloads if p[position] == byte_val)
    return tp / len(attack_payloads), fp / len(legit_payloads)


def eval_masked_rule(legit_payloads, attack_payloads, position, match_byte, mask):
    """Evaluate a masked byte match at a position."""
    fp = sum(1 for p in legit_payloads if (p[position] & mask) == match_byte)
    tp = sum(1 for p in attack_payloads if (p[position] & mask) == match_byte)
    return tp / len(attack_payloads), fp / len(legit_payloads)


def eval_multi_position_rule(legit_payloads, attack_payloads, positions, match_bytes, masks):
    """Evaluate a multi-byte rule (contiguous positions)."""
    def matches(payload):
        return all(
            (payload[pos] & mask) == mb
            for pos, mb, mask in zip(positions, match_bytes, masks)
        )
    fp = sum(1 for p in legit_payloads if matches(p))
    tp = sum(1 for p in attack_payloads if matches(p))
    return tp / len(attack_payloads), fp / len(legit_payloads)


def find_best_mask(attack_bytes_at_pos, legit_bytes_at_pos):
    """Find the best mask that catches all attack bytes but no legit bytes.

    Try masks from most specific to least specific:
    0xFF (exact), 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80
    """
    masks = [0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80]

    best = None
    for mask in masks:
        # Find the masked value that covers the most attack bytes
        attack_masked = Counter(b & mask for b in attack_bytes_at_pos)
        legit_masked = set(b & mask for b in legit_bytes_at_pos)

        for masked_val, count in attack_masked.most_common():
            tp_rate = count / len(attack_bytes_at_pos)
            if masked_val not in legit_masked:
                # Zero false positives with this mask+value
                if best is None or tp_rate > best["tp_rate"]:
                    best = {
                        "mask": mask,
                        "match": masked_val,
                        "tp_rate": tp_rate,
                        "fp_rate": 0.0,
                    }
                # Even if we found a good one, check if a wider mask catches more
                # but only if zero FP
    return best


# =====================================================================
# Main
# =====================================================================

def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    # Build baseline
    accum = client.create_accumulator()
    for _ in range(500):
        payload = make_legit_payload(rng)
        pkt = make_packet_dict(rng, payload)
        vec = client.encode(pkt)
        accum = client.accumulate(accum, vec)
    baseline = client.normalize_accumulator(accum)

    # Generate test data
    legit_payloads = [make_legit_payload(rng) for _ in range(200)]
    attack_sets = {
        "Overflow (uniform)": [make_attack_overflow(rng) for _ in range(100)],
        "Overflow (varied)": [make_attack_varied(rng) for _ in range(100)],
        "Polymorphic": [make_attack_polymorphic(rng) for _ in range(100)],
    }

    for attack_name, attack_payloads in attack_sets.items():
        print("=" * 80)
        print(f"ATTACK TYPE: {attack_name}")
        print("=" * 80)
        print()

        # ----- VSA drill-down to identify candidates -----
        sample_pkt = make_packet_dict(rng, attack_payloads[0])
        drill = drill_down(client, sample_pkt, baseline)

        payload_drill = [(f, v, s) for f, v, s in drill
                         if f.startswith("p") and f[1:].isdigit()]

        print(f"  VSA drill-down (payload positions, sorted by familiarity):")
        for f, v, s in payload_drill[:8]:
            marker = " ← CANDIDATE" if s < 0.005 else ""
            print(f"    {f:>4} = {v:>6}  sim={s:>7.4f}{marker}")
        print()

        # Candidate positions: unfamiliar ones first
        candidate_positions = [int(f[1:]) for f, v, s in payload_drill if s < 0.01]

        # ----- Evaluate every single-byte exact match -----
        print(f"  SINGLE-BYTE EXACT MATCHES (at VSA candidate positions):")
        print(f"  {'Pos':>4} {'Byte':>6} {'TP%':>6} {'FP%':>6} {'Rule':>35} {'Verdict':>10}")
        print(f"  {'-' * 72}")

        single_byte_candidates = []
        for pos in candidate_positions:
            byte_counts = Counter(p[pos] for p in attack_payloads)
            top_byte, top_count = byte_counts.most_common(1)[0]
            tp_rate, fp_rate = eval_exact_rule(legit_payloads, attack_payloads, pos, top_byte)

            offset = 8 + pos
            rule = f'(l4-match {offset} "{top_byte:02X}" "FF")'
            verdict = "PERFECT" if fp_rate == 0 and tp_rate == 1.0 else \
                      "GOOD" if fp_rate == 0 and tp_rate > 0.5 else \
                      "WEAK" if fp_rate == 0 else "FP!"

            print(f"  {pos:>4} 0x{top_byte:02x}{tp_rate:>5.0%}{fp_rate:>5.0%}   {rule:>35} {verdict:>10}")
            single_byte_candidates.append({
                "pos": pos, "byte": top_byte, "tp": tp_rate, "fp": fp_rate,
                "rule": rule, "verdict": verdict,
            })

        # ----- Find best single-byte rule -----
        zero_fp = [c for c in single_byte_candidates if c["fp"] == 0]
        if zero_fp:
            best_single = max(zero_fp, key=lambda c: (c["tp"], -c["pos"]))
            print(f"\n  BEST single-byte rule:")
            print(f"    {best_single['rule']}")
            print(f"    TP: {best_single['tp']:.0%}  FP: {best_single['fp']:.0%}")
        else:
            print(f"\n  No zero-FP single-byte rule found. Need masks or multi-byte.")
            best_single = None

        # ----- Masked matching for better coverage -----
        print(f"\n  MASKED MATCHES (relaxed byte matching for higher coverage):")
        print(f"  {'Pos':>4} {'Match':>6} {'Mask':>6} {'TP%':>6} {'FP%':>6} {'Coverage':>10}")
        print(f"  {'-' * 46}")

        masked_candidates = []
        for pos in candidate_positions:
            attack_bytes = [p[pos] for p in attack_payloads]
            legit_bytes = [p[pos] for p in legit_payloads]

            best_mask = find_best_mask(attack_bytes, legit_bytes)
            if best_mask and best_mask["tp_rate"] > 0:
                tp_rate, fp_rate = eval_masked_rule(
                    legit_payloads, attack_payloads, pos,
                    best_mask["match"], best_mask["mask"]
                )
                coverage = "FULL" if tp_rate == 1.0 else f"{tp_rate:.0%}"

                print(f"  {pos:>4} 0x{best_mask['match']:02x} 0x{best_mask['mask']:02x} "
                      f"{tp_rate:>5.0%} {fp_rate:>5.0%}   {coverage:>10}")

                masked_candidates.append({
                    "pos": pos, "match": best_mask["match"],
                    "mask": best_mask["mask"], "tp": tp_rate, "fp": fp_rate,
                })

        if masked_candidates:
            best_masked = max(masked_candidates, key=lambda c: c["tp"])
            if best_masked["tp"] > (best_single["tp"] if best_single else 0):
                offset = 8 + best_masked["pos"]
                rule = f'(l4-match {offset} "{best_masked["match"]:02X}" "{best_masked["mask"]:02X}")'
                print(f"\n  BEST masked rule (wider coverage):")
                print(f"    {rule}")
                print(f"    TP: {best_masked['tp']:.0%}  FP: {best_masked['fp']:.0%}")

        # ----- Multi-byte contiguous rules -----
        print(f"\n  BEST MULTI-BYTE CONTIGUOUS RULES:")
        print(f"  Testing 2-byte and 3-byte combinations at candidate positions...")
        print()

        multi_candidates = []
        # Try 2-byte combos
        for i in range(len(candidate_positions) - 1):
            p1, p2 = candidate_positions[i], candidate_positions[i + 1]
            if p2 != p1 + 1:
                continue
            # Get consensus bytes
            b1 = Counter(p[p1] for p in attack_payloads).most_common(1)[0]
            b2 = Counter(p[p2] for p in attack_payloads).most_common(1)[0]
            if b1[1] / len(attack_payloads) < 0.5 or b2[1] / len(attack_payloads) < 0.5:
                continue

            tp, fp = eval_multi_position_rule(
                legit_payloads, attack_payloads,
                [p1, p2], [b1[0], b2[0]], [0xFF, 0xFF]
            )
            if fp == 0 and tp > 0:
                offset = 8 + p1
                rule = f'(l4-match {offset} "{b1[0]:02X}{b2[0]:02X}" "FFFF")'
                multi_candidates.append({
                    "rule": rule, "tp": tp, "fp": fp, "length": 2,
                    "positions": [p1, p2],
                })

        # Try 3-byte combos
        for i in range(len(candidate_positions) - 2):
            p1, p2, p3 = candidate_positions[i], candidate_positions[i+1], candidate_positions[i+2]
            if p2 != p1 + 1 or p3 != p2 + 1:
                continue
            bytes_at = []
            skip = False
            for p in [p1, p2, p3]:
                bc = Counter(pp[p] for pp in attack_payloads).most_common(1)[0]
                if bc[1] / len(attack_payloads) < 0.5:
                    skip = True
                    break
                bytes_at.append(bc[0])
            if skip:
                continue

            tp, fp = eval_multi_position_rule(
                legit_payloads, attack_payloads,
                [p1, p2, p3], bytes_at, [0xFF, 0xFF, 0xFF]
            )
            if fp == 0 and tp > 0:
                offset = 8 + p1
                hex_match = "".join(f"{b:02X}" for b in bytes_at)
                rule = f'(l4-match {offset} "{hex_match}" "FFFFFF")'
                multi_candidates.append({
                    "rule": rule, "tp": tp, "fp": fp, "length": 3,
                    "positions": [p1, p2, p3],
                })

        if multi_candidates:
            multi_candidates.sort(key=lambda c: (-c["tp"], c["length"]))
            for mc in multi_candidates[:5]:
                print(f"    {mc['rule']}")
                print(f"      TP: {mc['tp']:.0%}  FP: {mc['fp']:.0%}  Length: {mc['length']} bytes")

        # ----- Summary: overall best rule -----
        print(f"\n  {'=' * 60}")
        print(f"  RECOMMENDATION FOR: {attack_name}")
        print(f"  {'=' * 60}")

        all_candidates = []
        if best_single and best_single["fp"] == 0:
            all_candidates.append({
                "rule": best_single["rule"], "tp": best_single["tp"],
                "fp": 0, "length": 1, "type": "exact 1-byte",
                "cost": "1 custom dim slot",
            })
        for mc in (masked_candidates or []):
            if mc["fp"] == 0 and mc["tp"] > 0:
                offset = 8 + mc["pos"]
                rule = f'(l4-match {offset} "{mc["match"]:02X}" "{mc["mask"]:02X}")'
                all_candidates.append({
                    "rule": rule, "tp": mc["tp"], "fp": 0,
                    "length": 1, "type": "masked 1-byte",
                    "cost": "1 custom dim slot",
                })
        for mc in (multi_candidates or []):
            cost = "1 custom dim slot" if mc["length"] <= 4 else "1 pattern guard"
            all_candidates.append({
                "rule": mc["rule"], "tp": mc["tp"], "fp": 0,
                "length": mc["length"], "type": f"exact {mc['length']}-byte",
                "cost": cost,
            })

        if all_candidates:
            # Sort: highest TP first, then shortest rule
            all_candidates.sort(key=lambda c: (-c["tp"], c["length"]))
            winner = all_candidates[0]

            print(f"\n  Winner: {winner['rule']}")
            print(f"    Type: {winner['type']}")
            print(f"    TP: {winner['tp']:.0%}  FP: {winner['fp']:.0%}")
            print(f"    Cost: {winner['cost']}")
            print(f"    Resource usage: {'MINIMAL' if winner['length'] <= 4 else 'MODERATE'}")

            if len(all_candidates) > 1:
                print(f"\n  Runner-up options:")
                for c in all_candidates[1:3]:
                    print(f"    {c['rule']}  TP:{c['tp']:.0%}  ({c['type']}, {c['cost']})")
        else:
            print(f"\n  No zero-FP rule found. This attack may need rate-based detection.")

        print()

    # ================================================================
    print("=" * 80)
    print("RESOURCE BUDGET SUMMARY")
    print("=" * 80)
    print()
    print("  Constraints:")
    print("    Custom dim slots: 7 total (1-4 byte matches, O(1))")
    print("    Byte matches per scope: 32")
    print("    Pattern guards: 65,536 (5-64 byte matches)")
    print()
    print("  With 3 attack types detected, we'd use:")
    print("    3 custom dim slots (1 per attack type) → 4 slots remaining")
    print("    3/32 byte matches per scope → 29 remaining")
    print("    0 pattern guards (all fits in custom dims)")
    print()
    print("  That leaves room for:")
    print("    4 more attack signatures via custom dims")
    print("    29 more byte match rules per tenant")
    print("    65,536 pattern guards for longer signatures")
    print()


if __name__ == "__main__":
    main()
