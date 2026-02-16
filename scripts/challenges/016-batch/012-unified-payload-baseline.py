#!/usr/bin/env python3
"""
Unified Baseline with Payload Bytes: No New Machinery Needed

THE INSIGHT:
============
We already have:
  1. Encode packet as map → vector
  2. Accumulate into baseline
  3. Detect anomaly (similarity drop)
  4. Drill into fields to find what changed

Just ADD the payload bytes to the map. The existing pipeline does the rest.

Before:  {"src_ip": "10.0.1.5", "proto": "UDP", "dst_port": "27015"}
After:   {"src_ip": "10.0.1.5", "proto": "UDP", "dst_port": "27015",
          "p0": "0x47", "p1": "0x4d", "p2": "0x01", ...}

The accumulator naturally learns that position 0 is usually 0x47, position 1
is usually 0x4D, etc. When an attack arrives with 0x90 at those positions,
the similarity drops AND the drill-down pinpoints exactly which byte positions
are wrong.

NO new accumulators. NO new data structures. Just a wider encoding.

DRILL-DOWN METHOD:
==================
For each field f with value v in the anomalous packet:
  1. role_vec = get_vector(f)   — the key vector for this field
  2. val_vec = get_vector(v)    — the value vector for this value
  3. bound = role_vec * val_vec — the contribution this field:value makes
  4. sim = cosine(bound, baseline)
  5. If sim is low → this field:value is unfamiliar

This works because the accumulated baseline is a weighted sum of all
field:value bindings seen during learning. Familiar bindings have positive
contribution; unfamiliar ones have near-zero.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import coherence


# =====================================================================
# Game Protocol Simulation
# =====================================================================

GAME_MSG_TYPES = {
    "move":       bytes([0x47, 0x4D, 0x01, 0x00]),
    "shoot":      bytes([0x47, 0x4D, 0x02, 0x00]),
    "chat":       bytes([0x47, 0x4D, 0x03, 0x00]),
    "heartbeat":  bytes([0x47, 0x4D, 0x04, 0x00]),
    "spawn":      bytes([0x47, 0x4D, 0x05, 0x00]),
    "inventory":  bytes([0x47, 0x4D, 0x06, 0x00]),
}

NUM_PAYLOAD_BYTES = 16  # encode first 16 bytes of payload


def make_packet_dict(rng, payload_bytes, src_ip=None, dst_port="27015"):
    """Create a full packet dict with headers AND payload bytes."""
    d = {
        "src_ip": src_ip or f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": "192.168.1.100",
        "proto": "UDP",
        "src_port": str(rng.integers(1024, 65535)),
        "dst_port": dst_port,
    }
    # Add payload bytes as fields
    for i, b in enumerate(payload_bytes[:NUM_PAYLOAD_BYTES]):
        d[f"p{i}"] = f"0x{b:02x}"
    return d


def make_legit_payload(rng, length=32):
    msg_type = rng.choice(list(GAME_MSG_TYPES.keys()))
    header = bytearray(GAME_MSG_TYPES[msg_type])
    header[2] = rng.integers(0, 256)  # seq num varies
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


def make_attack_payload(rng, length=32):
    """NOP sled + overflow — wrong bytes everywhere."""
    payload = bytearray([0x90] * 8 + [0x41] * 8)
    payload += bytearray(rng.integers(0x80, 0xFF, size=length - 16))
    return bytes(payload[:length])


def make_attack_subtle(rng, length=32):
    """Valid magic bytes but unusual body bytes (0x70-0x7F range)."""
    header = bytearray([0x47, 0x4D])
    header += bytearray(rng.integers(0, 256, size=2))
    body = bytearray(rng.integers(0x70, 0x80, size=length - 4))
    return bytes(header + body)[:length]


# =====================================================================
# Drill-Down: Which Fields Are Unfamiliar?
# =====================================================================

def drill_down(client, packet_dict, baseline_norm):
    """For each field:value pair, check familiarity against baseline.

    Returns list of (field, value, similarity) sorted by similarity (ascending).
    """
    results = []
    for field, value in packet_dict.items():
        role_vec = client.get_vector(field)
        val_vec = client.get_vector(value)
        bound = role_vec * val_vec  # same binding as encode() uses
        sim = cosine_similarity(bound, baseline_norm)
        results.append((field, value, sim))
    results.sort(key=lambda x: x[2])
    return results


# =====================================================================
# Experiments
# =====================================================================

def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    # ================================================================
    print("=" * 80)
    print("LEARNING PHASE: 500 legitimate packets with headers + payload bytes")
    print("=" * 80)
    print()

    accum = client.create_accumulator()
    for _ in range(500):
        payload = make_legit_payload(rng)
        pkt = make_packet_dict(rng, payload)
        vec = client.encode(pkt)
        accum = client.accumulate(accum, vec)

    baseline = client.normalize_accumulator(accum)
    print(f"  Baseline built from 500 packets")
    print(f"  Encoding includes: {5} header fields + {NUM_PAYLOAD_BYTES} payload byte positions")
    print(f"  Total fields per packet: {5 + NUM_PAYLOAD_BYTES}")

    # ================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Similarity — Does the Baseline Catch Payload Anomalies?")
    print("=" * 80)
    print()

    legit_sims = []
    for _ in range(50):
        payload = make_legit_payload(rng)
        pkt = make_packet_dict(rng, payload)
        vec = client.encode(pkt)
        legit_sims.append(cosine_similarity(vec, baseline))

    attack_sims = []
    for _ in range(50):
        payload = make_attack_payload(rng)
        pkt = make_packet_dict(rng, payload)  # same headers!
        vec = client.encode(pkt)
        attack_sims.append(cosine_similarity(vec, baseline))

    subtle_sims = []
    for _ in range(50):
        payload = make_attack_subtle(rng)
        pkt = make_packet_dict(rng, payload)
        vec = client.encode(pkt)
        subtle_sims.append(cosine_similarity(vec, baseline))

    legit_sims = np.array(legit_sims)
    attack_sims = np.array(attack_sims)
    subtle_sims = np.array(subtle_sims)

    print(f"  {'Traffic':>20} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-' * 50}")
    print(f"  {'Legitimate':>20} {np.mean(legit_sims):>7.4f} {np.std(legit_sims):>7.4f} "
          f"{np.min(legit_sims):>7.4f} {np.max(legit_sims):>7.4f}")
    print(f"  {'Attack (overflow)':>20} {np.mean(attack_sims):>7.4f} {np.std(attack_sims):>7.4f} "
          f"{np.min(attack_sims):>7.4f} {np.max(attack_sims):>7.4f}")
    print(f"  {'Attack (subtle)':>20} {np.mean(subtle_sims):>7.4f} {np.std(subtle_sims):>7.4f} "
          f"{np.min(subtle_sims):>7.4f} {np.max(subtle_sims):>7.4f}")

    sep = np.mean(legit_sims) / np.mean(attack_sims)
    print(f"\n  Separation ratio (legit/attack): {sep:.1f}×")

    # ================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Drill-Down — Where Are the Unfamiliar Bytes?")
    print("=" * 80)
    print()
    print("When similarity drops, drill into each field to find what's wrong.")
    print("Fields sorted by familiarity (least familiar first).")
    print()

    # A legit packet
    legit_payload = make_legit_payload(rng)
    legit_pkt = make_packet_dict(rng, legit_payload)
    legit_vec = client.encode(legit_pkt)
    legit_sim = cosine_similarity(legit_vec, baseline)

    print(f"  LEGIT packet (overall sim: {legit_sim:.4f}):")
    drill = drill_down(client, legit_pkt, baseline)
    for field, value, sim in drill[:10]:  # show top 10 least familiar
        marker = " ← UNFAMILIAR" if sim < 0.0 else ""
        print(f"    {field:>10} = {value:>8}  sim = {sim:>7.4f}{marker}")
    print(f"    ... ({len(drill) - 10} more fields)")

    # An attack packet
    print()
    attack_payload = make_attack_payload(rng)
    attack_pkt = make_packet_dict(rng, attack_payload)
    attack_vec = client.encode(attack_pkt)
    attack_sim = cosine_similarity(attack_vec, baseline)

    print(f"  ATTACK packet (overall sim: {attack_sim:.4f}):")
    drill = drill_down(client, attack_pkt, baseline)
    for field, value, sim in drill[:15]:
        marker = " ← UNFAMILIAR" if sim < 0.0 else ""
        print(f"    {field:>10} = {value:>8}  sim = {sim:>7.4f}{marker}")
    print(f"    ... ({len(drill) - 15} more fields)")

    # Subtle attack
    print()
    subtle_payload = make_attack_subtle(rng)
    subtle_pkt = make_packet_dict(rng, subtle_payload)
    subtle_vec = client.encode(subtle_pkt)
    subtle_sim = cosine_similarity(subtle_vec, baseline)

    print(f"  SUBTLE ATTACK packet (overall sim: {subtle_sim:.4f}):")
    drill = drill_down(client, subtle_pkt, baseline)
    for field, value, sim in drill[:15]:
        marker = " ← UNFAMILIAR" if sim < 0.0 else ""
        print(f"    {field:>10} = {value:>8}  sim = {sim:>7.4f}{marker}")
    print(f"    ... ({len(drill) - 15} more fields)")

    # ================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Automatic Rule Extraction from Drill-Down")
    print("=" * 80)
    print()
    print("From the drill-down, extract payload positions where the byte value")
    print("is unfamiliar (sim < 0). Generate l4-match rules from those positions.")
    print()

    # Process multiple attack packets to find consensus
    attack_payloads = [make_attack_payload(rng) for _ in range(30)]

    # Per-position: track which byte values appear and their familiarity
    pos_data = {i: {"bytes": [], "sims": []} for i in range(NUM_PAYLOAD_BYTES)}

    for payload in attack_payloads:
        pkt = make_packet_dict(rng, payload)
        drill = drill_down(client, pkt, baseline)
        for field, value, sim in drill:
            if field.startswith("p") and field[1:].isdigit():
                pos = int(field[1:])
                byte_val = int(value, 16)
                pos_data[pos]["bytes"].append(byte_val)
                pos_data[pos]["sims"].append(sim)

    print(f"  Per-position analysis across 30 attack packets:")
    print(f"  {'Pos':>4} {'MeanSim':>8} {'StdSim':>7} {'TopByte':>8} {'Consensus':>10} {'Verdict':>12}")
    print(f"  {'-' * 55}")

    rule_candidates = {}
    for pos in range(NUM_PAYLOAD_BYTES):
        pd = pos_data[pos]
        if not pd["sims"]:
            continue
        mean_sim = np.mean(pd["sims"])
        std_sim = np.std(pd["sims"])

        # Most common byte
        from collections import Counter
        byte_counts = Counter(pd["bytes"])
        top_byte, top_count = byte_counts.most_common(1)[0]
        consensus = top_count / len(pd["bytes"])

        if mean_sim < 0.0:
            verdict = "UNFAMILIAR"
        elif mean_sim < 0.01:
            verdict = "borderline"
        else:
            verdict = "familiar"

        print(f"  {pos:>4} {mean_sim:>8.4f} {std_sim:>7.4f} "
              f"0x{top_byte:02x}{consensus:>9.0%} {verdict:>12}")

        if mean_sim < 0.0 and consensus > 0.5:
            rule_candidates[pos] = top_byte

    # Build contiguous rules
    if rule_candidates:
        positions = sorted(rule_candidates.keys())
        rules = []
        i = 0
        while i < len(positions):
            start = positions[i]
            run = [start]
            while i + 1 < len(positions) and positions[i + 1] == positions[i] + 1:
                i += 1
                run.append(positions[i])
            i += 1

            offset = 8 + start  # UDP header
            match_hex = "".join(f"{rule_candidates[p]:02X}" for p in run)
            mask_hex = "FF" * len(run)
            rule = f'(l4-match {offset} "{match_hex}" "{mask_hex}")'
            rules.append((rule, run))

        print(f"\n  Generated rules:")
        legit_test = [make_legit_payload(rng) for _ in range(100)]
        for rule, run in rules:
            legit_hits = sum(
                1 for p in legit_test
                if all(p[pos] == rule_candidates[pos] for pos in run)
            )
            attack_hits = sum(
                1 for p in attack_payloads
                if all(p[pos] == rule_candidates[pos] for pos in run)
            )
            print(f"    {rule}")
            print(f"      Legit: {legit_hits}/100 ({legit_hits}%)  "
                  f"Attack: {attack_hits}/30 ({attack_hits/30*100:.0f}%)")

    # ================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Header vs Payload Contribution")
    print("=" * 80)
    print()
    print("Attack packets have SAME headers as legit. Only payload differs.")
    print("The drill-down should show header fields as familiar, payload as not.")
    print()

    attack_payload = make_attack_payload(rng)
    attack_pkt = make_packet_dict(rng, attack_payload)
    drill = drill_down(client, attack_pkt, baseline)

    header_fields = []
    payload_fields = []
    for field, value, sim in drill:
        if field.startswith("p") and field[1:].isdigit():
            payload_fields.append((field, value, sim))
        else:
            header_fields.append((field, value, sim))

    header_mean = np.mean([s for _, _, s in header_fields])
    payload_mean = np.mean([s for _, _, s in payload_fields])

    print(f"  Header fields (mean sim: {header_mean:.4f}):")
    for field, value, sim in sorted(header_fields, key=lambda x: x[2]):
        print(f"    {field:>10} = {value:>20}  sim = {sim:>7.4f}")

    print(f"\n  Payload fields (mean sim: {payload_mean:.4f}):")
    for field, value, sim in sorted(payload_fields, key=lambda x: x[2]):
        marker = " ←" if sim < 0.0 else ""
        print(f"    {field:>10} = {value:>20}  sim = {sim:>7.4f}{marker}")

    print(f"\n  Header mean: {header_mean:.4f}  |  Payload mean: {payload_mean:.4f}")
    print(f"  The attack is in the payload, and the drill-down shows it.")

    # ================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Progressive Learning — Baseline Adapts to New Legit Patterns")
    print("=" * 80)
    print()
    print("Add a new legitimate message type to the game. The baseline should")
    print("adapt and stop flagging the new pattern as unfamiliar.")
    print()

    # New message type: "emote" with different header
    def make_emote_payload(rng, length=32):
        header = bytearray([0x47, 0x4D, 0x07, 0x00])  # type 0x07 = emote
        body = bytearray([0xE0] + list(rng.integers(0, 10, size=length - 5)))
        return bytes(header + body)[:length]

    # Before learning emotes
    emote_payload = make_emote_payload(rng)
    emote_pkt = make_packet_dict(rng, emote_payload)
    emote_vec = client.encode(emote_pkt)
    sim_before = cosine_similarity(emote_vec, baseline)

    drill_before = drill_down(client, emote_pkt, baseline)
    unfamiliar_before = sum(1 for _, _, s in drill_before if s < 0.0)

    print(f"  Before learning emotes:")
    print(f"    Similarity: {sim_before:.4f}")
    print(f"    Unfamiliar fields: {unfamiliar_before}/{len(drill_before)}")
    for f, v, s in drill_before[:5]:
        print(f"      {f:>10} = {v:>8}  sim = {s:.4f}")

    # Learn 200 emote packets
    for _ in range(200):
        payload = make_emote_payload(rng)
        pkt = make_packet_dict(rng, payload)
        vec = client.encode(pkt)
        accum = client.accumulate(accum, vec)

    baseline_updated = client.normalize_accumulator(accum)

    # After learning
    emote_payload2 = make_emote_payload(rng)
    emote_pkt2 = make_packet_dict(rng, emote_payload2)
    emote_vec2 = client.encode(emote_pkt2)
    sim_after = cosine_similarity(emote_vec2, baseline_updated)

    drill_after = drill_down(client, emote_pkt2, baseline_updated)
    unfamiliar_after = sum(1 for _, _, s in drill_after if s < 0.0)

    print(f"\n  After learning 200 emote packets:")
    print(f"    Similarity: {sim_after:.4f}")
    print(f"    Unfamiliar fields: {unfamiliar_after}/{len(drill_after)}")
    for f, v, s in drill_after[:5]:
        print(f"      {f:>10} = {v:>8}  sim = {s:.4f}")

    print(f"\n  Adaptation: {sim_before:.4f} → {sim_after:.4f} "
          f"(unfamiliar: {unfamiliar_before} → {unfamiliar_after})")

    print()


if __name__ == "__main__":
    main()
