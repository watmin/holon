#!/usr/bin/env python3
"""
Sparse Mask: Skip Familiar Bytes in the Middle of an Attack Signature

SCENARIO:
=========
Good traffic has seen lots of byte variations: [00, 02, 20, 30, 34, 93, 11, 22, ...]
An attack rolls in with:         [FF, AC, FB, CA, 01, 02, 03, FF, FA, CB]
                                  ^^^ bad ^^^^  ~~ ok ~~  ^^^ bad ^^^^

Bytes 0-3 (FF AC FB CA) and 7-9 (FF FA CB) are unfamiliar.
Bytes 4-6 (01 02 03) happen to be values we've seen in normal traffic.

Instead of two separate rules for the two bad regions, we use ONE rule:
  match: FF AC FB CA 00 00 00 FF FA CB
  mask:  FF FF FF FF 00 00 00 FF FF FF

The 00 mask bytes mean "don't care" — skip those positions.
One rule catches the whole signature.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity


NUM_BYTES = 10


def make_normal_payload(rng):
    """Normal traffic: bytes drawn from a 'familiar' set at each position."""
    # Each position has its own distribution of common values
    familiar = {
        0: [0x00, 0x01, 0x02, 0x10, 0x20, 0x30],
        1: [0x00, 0x0A, 0x0B, 0x0C, 0x11, 0x22],
        2: [0x00, 0x01, 0x05, 0x10, 0x20, 0x34],
        3: [0x00, 0x02, 0x04, 0x08, 0x30, 0x40],
        4: [0x00, 0x01, 0x02, 0x03, 0x04, 0x05],
        5: [0x00, 0x01, 0x02, 0x03, 0x10, 0x20],
        6: [0x00, 0x01, 0x02, 0x03, 0x04, 0x06],
        7: [0x00, 0x07, 0x08, 0x09, 0x50, 0x60],
        8: [0x00, 0x0A, 0x0B, 0x33, 0x44, 0x55],
        9: [0x00, 0x01, 0x09, 0x11, 0x22, 0x93],
    }
    return bytes([int(rng.choice(familiar[i])) for i in range(NUM_BYTES)])


def make_attack_payload(rng):
    """Attack payload: unfamiliar bytes at 0-3 and 7-9, familiar at 4-6.

    [FF, AC, FB, CA, 01, 02, 03, FF, FA, CB]
     ^^^^^^^^^^^^^^  ^^^^^^^^  ^^^^^^^^^^^^
      unfamiliar     familiar   unfamiliar
    """
    return bytes([
        0xFF,                              # pos 0: unfamiliar
        0xAC,                              # pos 1: unfamiliar
        0xFB,                              # pos 2: unfamiliar
        0xCA,                              # pos 3: unfamiliar
        int(rng.choice([0x01, 0x02, 0x03])),  # pos 4: familiar value
        int(rng.choice([0x01, 0x02, 0x03])),  # pos 5: familiar value
        int(rng.choice([0x01, 0x02, 0x03])),  # pos 6: familiar value
        0xFF,                              # pos 7: unfamiliar
        0xFA,                              # pos 8: unfamiliar
        0xCB,                              # pos 9: unfamiliar
    ])


def make_attack_varied(rng):
    """Same attack but with slight variation in the unfamiliar bytes."""
    return bytes([
        int(rng.choice([0xFF, 0xFE, 0xFD])),  # pos 0
        int(rng.choice([0xAC, 0xAD, 0xAE])),  # pos 1
        int(rng.choice([0xFB, 0xFC])),         # pos 2
        0xCA,                                   # pos 3: always 0xCA
        int(rng.choice([0x01, 0x02, 0x03])),   # pos 4: familiar
        int(rng.choice([0x01, 0x02, 0x03])),   # pos 5: familiar
        int(rng.choice([0x01, 0x02, 0x03])),   # pos 6: familiar
        int(rng.choice([0xFF, 0xFE])),         # pos 7
        int(rng.choice([0xFA, 0xFB, 0xFC])),   # pos 8
        int(rng.choice([0xCB, 0xCC])),         # pos 9
    ])


def make_packet_dict(payload_bytes):
    """Encode payload bytes as a map."""
    return {f"p{i}": f"0x{b:02x}" for i, b in enumerate(payload_bytes)}


def drill_down(client, pkt_dict, baseline):
    results = []
    for field, value in pkt_dict.items():
        role_vec = client.get_vector(field)
        val_vec = client.get_vector(value)
        bound = role_vec * val_vec
        sim = cosine_similarity(bound, baseline)
        pos = int(field[1:])
        results.append((pos, value, sim))
    results.sort(key=lambda x: x[0])
    return results


def find_best_mask(attack_bytes, legit_bytes):
    """Find best mask for a position: most attack coverage with zero legit hits."""
    masks = [0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80]
    legit_set = set(legit_bytes)
    best = None
    for mask in masks:
        masked_legit = set(b & mask for b in legit_set)
        attack_masked = Counter(b & mask for b in attack_bytes)
        for val, count in attack_masked.most_common():
            tp = count / len(attack_bytes)
            if val not in masked_legit and (best is None or tp > best[2]):
                best = (val, mask, tp)
    return best


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    # ================================================================
    # LEARN
    # ================================================================
    print("=" * 70)
    print("LEARNING: 500 normal payloads")
    print("=" * 70)
    accum = client.create_accumulator()
    for _ in range(500):
        payload = make_normal_payload(rng)
        pkt = make_packet_dict(payload)
        vec = client.encode(pkt)
        accum = client.accumulate(accum, vec)
    baseline = client.normalize_accumulator(accum)

    legit_payloads = [make_normal_payload(rng) for _ in range(200)]

    print(f"  Familiar byte values per position:")
    for pos in range(NUM_BYTES):
        vals = sorted(set(p[pos] for p in legit_payloads))
        print(f"    p{pos}: {', '.join(f'0x{v:02x}' for v in vals)}")

    # ================================================================
    # DETECT
    # ================================================================
    for attack_name, attack_fn in [
        ("Uniform attack", make_attack_payload),
        ("Varied attack", make_attack_varied),
    ]:
        attack_payloads = [attack_fn(rng) for _ in range(100)]

        print(f"\n{'=' * 70}")
        print(f"ATTACK: {attack_name}")
        print(f"{'=' * 70}")

        sample = attack_payloads[0]
        print(f"\n  Attack sample: [{', '.join(f'{b:02X}' for b in sample)}]")
        print(f"                  {''.join('^^^^ ' if i not in [4,5,6] else ' ok  ' for i in range(NUM_BYTES))}")

        # Whole-payload similarity
        sample_vec = client.encode(make_packet_dict(sample))
        overall_sim = cosine_similarity(sample_vec, baseline)
        print(f"\n  Overall similarity to baseline: {overall_sim:.4f}")

        # Per-position drill-down
        drill = drill_down(client, make_packet_dict(sample), baseline)
        print(f"\n  Per-position drill-down:")
        print(f"  {'Pos':>4} {'Byte':>6} {'Sim':>8} {'Verdict':>12}")
        print(f"  {'-' * 34}")
        for pos, val, sim in drill:
            if sim < -0.005:
                verdict = "UNFAMILIAR"
            elif sim < 0.01:
                verdict = "borderline"
            else:
                verdict = "FAMILIAR"
            print(f"  {pos:>4} {val:>6} {sim:>8.4f} {verdict:>12}")

        # Consensus across all attack samples
        print(f"\n  Consensus across {len(attack_payloads)} attack samples:")
        print(f"  {'Pos':>4} {'TopByte':>8} {'Cons%':>6} {'MeanSim':>8} {'#Uniq':>6} {'Action':>12}")
        print(f"  {'-' * 50}")

        match_bytes = []
        mask_bytes = []
        for pos in range(NUM_BYTES):
            attack_bytes = [p[pos] for p in attack_payloads]
            legit_bytes_at = [p[pos] for p in legit_payloads]
            bc = Counter(attack_bytes)
            top_byte, top_count = bc.most_common(1)[0]
            consensus = top_count / len(attack_payloads)
            n_unique = len(bc)

            # Compute mean similarity for this position
            role_vec = client.get_vector(f"p{pos}")
            sims = []
            for p in attack_payloads[:30]:
                val_vec = client.get_vector(f"0x{p[pos]:02x}")
                bound = role_vec * val_vec
                sims.append(cosine_similarity(bound, baseline))
            mean_sim = np.mean(sims)

            # Decide: match or skip?
            if mean_sim > 0.01:
                # Familiar position — mask it off
                action = "SKIP (00)"
                match_bytes.append(0x00)
                mask_bytes.append(0x00)
            else:
                # Unfamiliar — find best mask
                best = find_best_mask(attack_bytes, legit_bytes_at)
                if best:
                    action = f"MATCH ({best[1]:02X})"
                    match_bytes.append(best[0])
                    mask_bytes.append(best[1])
                else:
                    action = "EXACT (FF)"
                    match_bytes.append(top_byte)
                    mask_bytes.append(0xFF)

            print(f"  {pos:>4} 0x{top_byte:02x}{consensus:>5.0%} {mean_sim:>8.4f} "
                  f"{n_unique:>6} {action:>12}")

        # Build the rule
        match_hex = "".join(f"{b:02X}" for b in match_bytes)
        mask_hex = "".join(f"{b:02X}" for b in mask_bytes)
        offset = 8  # after UDP header
        rule = f'(l4-match {offset} "{match_hex}" "{mask_hex}")'

        print(f"\n  {'=' * 50}")
        print(f"  GENERATED RULE:")
        print(f"  {rule}")
        print()

        # Visual breakdown
        print(f"  Position: {' '.join(f'{i:>5}' for i in range(NUM_BYTES))}")
        print(f"  Attack:   {' '.join(f' 0x{b:02x}' for b in sample)}")
        line = ""
        for i in range(NUM_BYTES):
            if mask_bytes[i] == 0x00:
                line += "    --"
            else:
                line += f"  0x{match_bytes[i]:02x}"
        print(f"  Match:   {line}")
        line = ""
        for i in range(NUM_BYTES):
            if mask_bytes[i] == 0x00:
                line += "  skip"
            else:
                line += f"  0x{mask_bytes[i]:02x}"
        print(f"  Mask:    {line}")
        line = ""
        for i in range(NUM_BYTES):
            if mask_bytes[i] == 0x00:
                line += "   .. "
            else:
                line += "  ^^^^"
        print(f"           {line}")

        # Validate
        def matches_rule(payload):
            for i in range(NUM_BYTES):
                if mask_bytes[i] == 0x00:
                    continue
                if (payload[i] & mask_bytes[i]) != match_bytes[i]:
                    return False
            return True

        tp = sum(1 for p in attack_payloads if matches_rule(p))
        fp = sum(1 for p in legit_payloads if matches_rule(p))

        print(f"\n  VALIDATION:")
        print(f"    Attack: {tp}/{len(attack_payloads)} ({tp/len(attack_payloads)*100:.0f}%)")
        print(f"    Legit:  {fp}/{len(legit_payloads)} ({fp/len(legit_payloads)*100:.0f}%)")
        print(f"    Cost:   1 pattern guard (out of 65,536)")
        print()


if __name__ == "__main__":
    main()
