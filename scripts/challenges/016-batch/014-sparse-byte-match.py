#!/usr/bin/env python3
"""
Sparse Byte Match: Masking Off Familiar Bytes in the Middle

THE IDEA:
=========
Attack payload has anomalous bytes at positions 0-3, then normal-looking
bytes at 4-13, then anomalous again at 14-19. Instead of two separate
rules, use ONE rule that spans the whole range with the mask zeroed out
for the familiar positions:

  Offset 8, length 20:
    match: "DEADBEEF 0000000000000000000000 C0FFEE BABE"
    mask:  "FFFFFFFF 0000000000000000000000 FFFFFF FFFF"

The zeros in the mask mean "don't care" — those positions are ignored.
One rule, one pattern guard entry, catches the whole signature.

This is how real IDS signatures work (Snort/Suricata content+offset+depth),
and our l4-match with masks supports it natively.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity


NUM_PAYLOAD_BYTES = 24  # encode first 24 bytes


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


def make_attack_sandwich(rng, length=32):
    """Attack with anomalous-normal-anomalous sandwich pattern.

    Positions 0-3:   anomalous (exploit header: 0xDE 0xAD 0xBE 0xEF)
    Positions 4-13:  normal-looking (valid game header + body to blend in)
    Positions 14-19: anomalous (shellcode stub: 0xC0 0xFF 0xEE 0xBA 0xBE 0xCC)
    Positions 20+:   padding
    """
    # Exploit header — clearly anomalous bytes
    payload = bytearray([0xDE, 0xAD, 0xBE, 0xEF])
    # Middle: looks like legit game traffic (valid magic + random body)
    msg = rng.choice(list(GAME_MSG_TYPES.values()))
    payload += bytearray(msg)  # 4 bytes of valid game header
    payload += bytearray(int(x) for x in rng.integers(0, 256, size=6))  # 6 bytes game body
    # Shellcode stub — clearly anomalous bytes again
    payload += bytearray([0xC0, 0xFF, 0xEE, 0xBA, 0xBE, 0xCC])
    # Padding
    payload += bytearray(int(x) for x in rng.integers(0, 256, size=max(0, length - 20)))
    return bytes(payload[:length])


def make_attack_sandwich_varied(rng, length=32):
    """Same sandwich but with slight byte variation in the anomalous parts."""
    h = bytearray([
        int(rng.choice([0xDE, 0xDF])),
        int(rng.choice([0xAD, 0xAE, 0xAF])),
        int(rng.choice([0xBE, 0xBF])),
        0xEF,
    ])
    # Middle: valid game header + random body
    msg = rng.choice(list(GAME_MSG_TYPES.values()))
    mid = bytearray(msg) + bytearray(int(x) for x in rng.integers(0, 256, size=6))
    # Tail varies
    t = bytearray([
        int(rng.choice([0xC0, 0xC1, 0xC2, 0xC3])),
        0xFF,
        int(rng.choice([0xEE, 0xEF])),
        int(rng.choice([0xBA, 0xBB])),
        int(rng.choice([0xBE, 0xBF])),
        0xCC,
    ])
    payload = bytes(h + mid + t)
    pad = bytes(int(x) for x in rng.integers(0, 256, size=max(0, length - len(payload))))
    return (payload + pad)[:length]


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


def find_best_mask_for_byte(attack_bytes, legit_bytes):
    """Find best mask for a single position."""
    masks = [0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80]
    best = None
    legit_set = set(legit_bytes)
    for mask in masks:
        masked_attack = Counter(b & mask for b in attack_bytes)
        masked_legit = set(b & mask for b in legit_set)
        for masked_val, count in masked_attack.most_common():
            tp = count / len(attack_bytes)
            if masked_val not in masked_legit and (best is None or tp > best[2]):
                best = (masked_val, mask, tp)
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

    legit_payloads = [make_legit_payload(rng) for _ in range(200)]

    for attack_name, attack_fn in [
        ("Sandwich (uniform)", make_attack_sandwich),
        ("Sandwich (varied)", make_attack_sandwich_varied),
    ]:
        attack_payloads = [attack_fn(rng) for _ in range(100)]

        print("=" * 80)
        print(f"ATTACK: {attack_name}")
        print("=" * 80)

        # Show the sandwich structure
        sample = attack_payloads[0]
        print(f"\n  Sample payload:")
        print(f"  Bytes: {' '.join(f'{b:02x}' for b in sample[:24])}")
        print(f"         ^^^^ anomalous  ^^^^^^^^^^^^^^^^^^^^^^^^^^ normal  "
              f"^^^^^^^^^^^^^^^^ anomalous")

        # ----- Step 1: VSA drill-down -----
        sample_pkt = make_packet_dict(rng, sample)
        sample_vec = client.encode(sample_pkt)
        overall_sim = cosine_similarity(sample_vec, baseline)
        print(f"\n  Overall similarity to baseline: {overall_sim:.4f}")

        drill = drill_down(client, sample_pkt, baseline)
        payload_drill = [(f, v, s) for f, v, s in drill
                         if f.startswith("p") and f[1:].isdigit()]

        print(f"\n  Per-position drill-down:")
        print(f"  {'Pos':>4} {'Byte':>6} {'Sim':>7} {'Status':>12}")
        print(f"  {'-' * 33}")
        for f, v, s in sorted(payload_drill, key=lambda x: int(x[0][1:])):
            pos = int(f[1:])
            if s < -0.005:
                status = "ANOMALOUS"
            elif s < 0.005:
                status = "borderline"
            elif s > 0.1:
                status = "familiar"
            else:
                status = "weak"
            print(f"  {pos:>4} {v:>6} {s:>7.4f} {status:>12}")

        # ----- Step 2: Consensus across attack samples -----
        print(f"\n  Consensus across {len(attack_payloads)} attack packets:")
        print(f"  {'Pos':>4} {'MeanSim':>8} {'TopByte':>8} {'Cons%':>6} {'#Uniq':>6} {'Zone':>12}")
        print(f"  {'-' * 50}")

        pos_analysis = {}
        for pos in range(NUM_PAYLOAD_BYTES):
            sims = []
            for p in attack_payloads:
                pkt = make_packet_dict(rng, p)
                role_vec = client.get_vector(f"p{pos}")
                val_vec = client.get_vector(f"0x{p[pos]:02x}")
                bound = role_vec * val_vec
                sim = cosine_similarity(bound, baseline)
                sims.append(sim)

            attack_bytes = [p[pos] for p in attack_payloads]
            bc = Counter(attack_bytes)
            top_byte, top_count = bc.most_common(1)[0]
            consensus = top_count / len(attack_payloads)
            n_unique = len(bc)
            mean_sim = np.mean(sims)

            if pos < 4:
                zone = "EXPLOIT HDR"
            elif pos < 14:
                zone = "normal mid"
            elif pos < 20:
                zone = "SHELLCODE"
            else:
                zone = "padding"

            print(f"  {pos:>4} {mean_sim:>8.4f} 0x{top_byte:02x}{consensus:>5.0%} "
                  f"{n_unique:>6} {zone:>12}")

            pos_analysis[pos] = {
                "mean_sim": mean_sim,
                "top_byte": top_byte,
                "consensus": consensus,
                "n_unique": n_unique,
                "zone": zone,
                "attack_bytes": attack_bytes,
            }

        # ----- Step 3: Build the sparse match rule -----
        print(f"\n  Building sparse byte match rule...")

        # Identify anomalous positions: mean_sim well below the "familiar" floor
        # Familiar positions (like 0x00 padding) have sim ~0.15-0.23
        # Anomalous positions have sim < 0.05
        familiar_sims = [pos_analysis[p]["mean_sim"] for p in range(NUM_PAYLOAD_BYTES)
                         if pos_analysis[p]["mean_sim"] > 0.1]
        if familiar_sims:
            anomaly_threshold = min(familiar_sims) * 0.5
        else:
            anomaly_threshold = 0.01

        anomalous_positions = []
        for pos in range(NUM_PAYLOAD_BYTES):
            pa = pos_analysis[pos]
            if pa["mean_sim"] < anomaly_threshold and pa["n_unique"] <= 10:
                anomalous_positions.append(pos)

        print(f"  Anomaly threshold: sim < {anomaly_threshold:.4f}")

        if not anomalous_positions:
            print(f"  No consistently anomalous positions found.")
            continue

        first_pos = min(anomalous_positions)
        last_pos = max(anomalous_positions)
        span = last_pos - first_pos + 1

        print(f"  Anomalous positions: {anomalous_positions}")
        print(f"  Span: position {first_pos} to {last_pos} ({span} bytes)")

        # Build match and mask arrays
        match_bytes = []
        mask_bytes = []
        for pos in range(first_pos, last_pos + 1):
            pa = pos_analysis[pos]
            if pos in anomalous_positions:
                # Find best mask for this position
                legit_bytes_at_pos = [p[pos] for p in legit_payloads]
                best = find_best_mask_for_byte(pa["attack_bytes"], legit_bytes_at_pos)
                if best:
                    match_bytes.append(best[0])
                    mask_bytes.append(best[1])
                else:
                    # Fall back to exact match of most common
                    match_bytes.append(pa["top_byte"])
                    mask_bytes.append(0xFF)
            else:
                # Normal position — mask it off (don't care)
                match_bytes.append(0x00)
                mask_bytes.append(0x00)

        offset = 8 + first_pos  # UDP header offset
        match_hex = "".join(f"{b:02X}" for b in match_bytes)
        mask_hex = "".join(f"{b:02X}" for b in mask_bytes)
        rule = f'(l4-match {offset} "{match_hex}" "{mask_hex}")'

        print(f"\n  SPARSE RULE:")
        print(f"    {rule}")
        print(f"    Length: {span} bytes")
        print()

        # Visual breakdown
        print(f"    Position: ", end="")
        for pos in range(first_pos, last_pos + 1):
            print(f" {pos:>4}", end="")
        print()

        print(f"    Match:    ", end="")
        for i, b in enumerate(match_bytes):
            pos = first_pos + i
            if pos in anomalous_positions:
                print(f" 0x{b:02x}", end="")
            else:
                print(f"   --", end="")
        print()

        print(f"    Mask:     ", end="")
        for i, b in enumerate(mask_bytes):
            pos = first_pos + i
            if pos in anomalous_positions:
                print(f" 0x{b:02x}", end="")
            else:
                print(f" 0x00", end="")
        print()

        print(f"    Zone:     ", end="")
        for pos in range(first_pos, last_pos + 1):
            z = pos_analysis[pos]["zone"]
            if "EXPLOIT" in z or "SHELL" in z:
                print(f"  ^^^", end="")
            else:
                print(f"  ...", end="")
        print()

        # ----- Step 4: Validate -----
        def matches_rule(payload):
            for i in range(span):
                pos = first_pos + i
                if mask_bytes[i] == 0x00:
                    continue
                if (payload[pos] & mask_bytes[i]) != match_bytes[i]:
                    return False
            return True

        tp = sum(1 for p in attack_payloads if matches_rule(p))
        fp = sum(1 for p in legit_payloads if matches_rule(p))

        print(f"\n    Validation:")
        print(f"      Attack TP: {tp}/{len(attack_payloads)} ({tp/len(attack_payloads)*100:.0f}%)")
        print(f"      Legit  FP: {fp}/{len(legit_payloads)} ({fp/len(legit_payloads)*100:.0f}%)")

        # ----- Step 5: Compare with separate rules -----
        print(f"\n  COMPARISON: Sparse rule vs separate rules")

        # Separate rules for each anomalous region
        regions = []
        current_region = [anomalous_positions[0]]
        for i in range(1, len(anomalous_positions)):
            if anomalous_positions[i] == anomalous_positions[i-1] + 1:
                current_region.append(anomalous_positions[i])
            else:
                regions.append(current_region)
                current_region = [anomalous_positions[i]]
        regions.append(current_region)

        print(f"\n    Option A: ONE sparse rule (mask zeroes in the middle)")
        print(f"      {rule}")
        cost_a = "1 pattern guard" if span > 4 else "1 custom dim slot"
        print(f"      Cost: {cost_a}")
        print(f"      TP: {tp/len(attack_payloads)*100:.0f}%  FP: {fp/len(legit_payloads)*100:.0f}%")

        print(f"\n    Option B: {len(regions)} separate rules (one per region)")
        total_slots = 0
        for region in regions:
            r_start = region[0]
            r_len = len(region)
            r_match = []
            r_mask = []
            for pos in region:
                idx = pos - first_pos
                r_match.append(match_bytes[idx])
                r_mask.append(mask_bytes[idx])

            r_offset = 8 + r_start
            r_match_hex = "".join(f"{b:02X}" for b in r_match)
            r_mask_hex = "".join(f"{b:02X}" for b in r_mask)
            r_rule = f'(l4-match {r_offset} "{r_match_hex}" "{r_mask_hex}")'

            r_cost = "1 custom dim slot" if r_len <= 4 else "1 pattern guard"
            total_slots += 1 if r_len <= 4 else 0
            print(f"      {r_rule}  ({r_cost})")

        print(f"\n    Verdict: Option A uses 1 resource. Option B uses {len(regions)}.")
        if span <= 4:
            print(f"    But span is only {span} bytes — fits in a custom dim slot either way!")
        elif span <= 64:
            print(f"    Span is {span} bytes — uses 1 pattern guard (out of 65,536).")
        print()

    # ================================================================
    print("=" * 80)
    print("SUMMARY: The Sparse Mask Trick")
    print("=" * 80)
    print()
    print("  When anomalous bytes are separated by normal bytes, use ONE rule")
    print("  with mask=0x00 for the normal positions in the middle.")
    print()
    print("  Before (2 rules, 2 resources):")
    print("    (l4-match 8  \"DEADBEEF\" \"FFFFFFFF\")")
    print("    (l4-match 22 \"C0FFEEBABE00\" \"FFFFFFFFFFFF\")")
    print()
    print("  After (1 rule, 1 resource):")
    print("    (l4-match 8  \"DEADBEEF00000000000000000000C0FFEEBABE00\" ")
    print("                 \"FFFFFFFF00000000000000000000FFFFFFFFFFFF\")")
    print()
    print("  The VSA drill-down tells you which positions to enforce (mask=FF)")
    print("  and which to skip (mask=00). One pattern guard covers it all.")
    print()


if __name__ == "__main__":
    main()
