#!/usr/bin/env python3
"""
Payload Anomaly Detection: Identifying Unfamiliar Bytes in Structurally Normal Traffic

PROBLEM:
========
Attackers send UDP traffic to a legitimate game server. Headers are
indistinguishable from real clients (same dst_ip, dst_port, proto=UDP,
varied src_ports). The only difference:
  1. Payload bytes are UNFAMILIAR (not matching known game protocol)
  2. Attack rate is HIGH (many packets from many sources)

Header-based coherence can't help — both legit and attack headers look similar.

HYPOTHESIS:
===========
Encode first N bytes of payload positionally into vectors. Accumulate
legitimate payloads into a "familiar" baseline. Attack payloads have
unfamiliar bytes → low similarity to baseline.

Among the low-similarity outliers, coherence is HIGH (same attack tool
sending the same bytes). This confirms a coordinated attack.

Finally: prototype the attack cluster to extract the consensus pattern,
identify which byte positions differ, and generate byte match rules.

PIPELINE:
=========
1. LEARN:    accumulate familiar payload patterns
2. DETECT:   flag low-similarity payloads
3. CLUSTER:  coherence among flagged packets confirms attack
4. EXTRACT:  prototype attack → identify anomalous byte positions
5. ACT:      generate (l4-match offset hex-match hex-mask) rules

VECTOR PROPERTIES EXPLOITED:
============================
- Payload content similarity (unfamiliar bytes → low similarity)
- Coherence among outliers (attack tool homogeneity)
- Prototype consensus (extract common attack pattern)
- Unbinding for byte-position identification (which positions are bad)

CONSTRAINTS:
============
- Stateless, per-window, unidirectional
- No flow tracking
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import coherence, prototype, difference


# =====================================================================
# Simulated Game Protocol
# =====================================================================

# Game protocol: first 2 bytes = message type, next 2 = sequence number,
# remaining = payload specific to message type
GAME_MSG_TYPES = {
    "move":       bytes([0x47, 0x4D, 0x01, 0x00]),  # GM\x01\x00
    "shoot":      bytes([0x47, 0x4D, 0x02, 0x00]),  # GM\x02\x00
    "chat":       bytes([0x47, 0x4D, 0x03, 0x00]),  # GM\x03\x00
    "heartbeat":  bytes([0x47, 0x4D, 0x04, 0x00]),  # GM\x04\x00
    "spawn":      bytes([0x47, 0x4D, 0x05, 0x00]),  # GM\x05\x00
    "inventory":  bytes([0x47, 0x4D, 0x06, 0x00]),  # GM\x06\x00
}


def make_legit_payload(rng, max_len=32):
    """Generate a legitimate game protocol payload."""
    msg_type = rng.choice(list(GAME_MSG_TYPES.keys()))
    header = bytearray(GAME_MSG_TYPES[msg_type])
    # Sequence number (varies per packet)
    header[2] = rng.integers(0, 256)
    header[3] = rng.integers(0, 256)

    # Payload varies by type but shares structure
    if msg_type == "move":
        # x, y, z coordinates (float-like bytes)
        body = bytearray(rng.integers(0, 256, size=12))
    elif msg_type == "shoot":
        # target id + angle
        body = bytearray(rng.integers(0, 256, size=8))
    elif msg_type == "chat":
        # ASCII text
        text = rng.choice(["gg", "nice", "help", "go go", "lol", "wp"])
        body = bytearray(text.encode("ascii"))
        body += bytearray(max_len - len(header) - len(body))  # pad
    elif msg_type == "heartbeat":
        # Timestamp-like
        body = bytearray([0x00] * 4 + list(rng.integers(0, 256, size=4)))
    elif msg_type == "spawn":
        body = bytearray(rng.integers(0, 256, size=16))
    elif msg_type == "inventory":
        # Item IDs
        body = bytearray(rng.integers(0, 64, size=8))
    else:
        body = bytearray(max_len - len(header))

    payload = bytes(header + body)[:max_len]
    # Pad to fixed length
    if len(payload) < max_len:
        payload = payload + bytes(max_len - len(payload))
    return payload


def make_attack_payload_a(rng, max_len=32):
    """Attack type A: buffer overflow attempt — no game protocol header."""
    # No "GM" magic bytes, just raw exploit bytes
    payload = bytearray([0x90] * 8)  # NOP sled
    payload += bytearray([0x41] * 8)  # 'AAAA...' overflow padding
    payload += bytearray(rng.integers(0x80, 0xFF, size=max_len - 16))  # shellcode-like
    return bytes(payload[:max_len])


def make_attack_payload_b(rng, max_len=32):
    """Attack type B: spoofed game header but wrong payload structure."""
    # Starts with GM but uses invalid message type and garbage body
    header = bytearray([0x47, 0x4D, 0xFF, 0xFF])  # GM + invalid type
    body = bytearray(rng.integers(0xC0, 0xFF, size=max_len - 4))  # high bytes
    return bytes(header + body)[:max_len]


def make_attack_payload_c(rng, max_len=32):
    """Attack type C: random flood — totally random bytes."""
    return bytes(rng.integers(0, 256, size=max_len))


# =====================================================================
# Payload Encoding
# =====================================================================

def encode_payload(client, payload_bytes, max_positions=32):
    """Encode payload bytes positionally into a vector.

    Each byte at position i is encoded as an atom 'pos_{i}_byte_{hex}'
    and bundled together.
    """
    data = {}
    for i, byte in enumerate(payload_bytes[:max_positions]):
        data[f"p{i}"] = f"0x{byte:02x}"
    return client.encode(data)


# =====================================================================
# Detection Pipeline
# =====================================================================

def build_baseline(client, legit_payloads):
    """Accumulate legitimate payloads into a baseline."""
    accum = client.create_accumulator()
    for payload in legit_payloads:
        vec = encode_payload(client, payload)
        accum = client.accumulate(accum, vec)
    return client.normalize_accumulator(accum), accum


def score_payloads(client, baseline, payloads):
    """Score each payload against baseline."""
    scores = []
    for payload in payloads:
        vec = encode_payload(client, payload)
        sim = cosine_similarity(vec, baseline)
        scores.append(sim)
    return np.array(scores)


def extract_attack_signature(client, attack_payloads, baseline):
    """Extract byte positions that distinguish attack from baseline."""
    # Encode all attack payloads
    attack_vecs = [encode_payload(client, p) for p in attack_payloads]

    # Get consensus attack pattern
    attack_proto = prototype(attack_vecs)

    # What changed from baseline?
    delta = difference(baseline, attack_proto)

    # Score each byte position individually
    position_scores = []
    for pos in range(len(attack_payloads[0])):
        # Get the role vector for this position
        role_key = f"p{pos}"
        role_vec = client.get_vector(role_key)

        # Unbind the position from both attack prototype and baseline
        attack_at_pos = attack_proto * role_vec  # unbind
        baseline_at_pos = baseline * role_vec

        # How different is this position between attack and baseline?
        pos_sim = cosine_similarity(attack_at_pos, baseline_at_pos)
        position_scores.append(pos_sim)

    return attack_proto, np.array(position_scores)


def generate_byte_match_rule(attack_payloads, position_scores, threshold=None,
                             legit_payloads=None):
    """Generate l4-match rule from attack signatures.

    Look at which byte positions have the most distinctive bytes between
    attack and legitimate payloads, then extract exact match rules.

    If legit_payloads is provided, filters out rules that match >10% of legit.
    """
    if threshold is None:
        # Use positions below median similarity score
        threshold = np.median(position_scores)

    # Find positions with low similarity (most anomalous)
    anomalous_positions = np.where(position_scores < threshold)[0]

    if len(anomalous_positions) == 0:
        # Fall back to positions below median
        median_score = np.median(position_scores)
        anomalous_positions = np.where(position_scores < median_score)[0]

    # For each anomalous position, find the most common byte value
    # across attack payloads
    match_bytes = {}
    for pos in anomalous_positions:
        byte_counts = {}
        for payload in attack_payloads:
            byte_val = payload[pos]
            byte_counts[byte_val] = byte_counts.get(byte_val, 0) + 1
        most_common = max(byte_counts, key=byte_counts.get)
        consensus_pct = byte_counts[most_common] / len(attack_payloads)
        if consensus_pct > 0.5:  # Only include if >50% agreement
            match_bytes[pos] = most_common

    if not match_bytes:
        return None

    # Find contiguous ranges for efficient rules
    positions = sorted(match_bytes.keys())
    rules = []

    # Generate rules for contiguous byte runs
    i = 0
    while i < len(positions):
        start = positions[i]
        run = [start]
        while i + 1 < len(positions) and positions[i + 1] == positions[i] + 1:
            i += 1
            run.append(positions[i])
        i += 1

        # Build the rule
        # Offset is relative to transport header (after UDP 8-byte header)
        offset = 8 + start  # 8 bytes UDP header
        match_hex = "".join(f"{match_bytes[p]:02X}" for p in run)
        mask_hex = "FF" * len(run)

        rule = f'(l4-match {offset} "{match_hex}" "{mask_hex}")'

        # If legit_payloads provided, filter rules that match too many
        if legit_payloads is not None:
            legit_matches = sum(
                1 for p in legit_payloads
                if all(p[pos] == match_bytes[pos] for pos in run if pos < len(p))
            )
            legit_match_rate = legit_matches / max(len(legit_payloads), 1)
            if legit_match_rate > 0.10:  # skip rules matching >10% of legit
                continue

        rules.append({
            "rule": rule,
            "offset": offset,
            "positions": run,
            "bytes": [match_bytes[p] for p in run],
        })

    return rules


# =====================================================================
# Experiments
# =====================================================================

def main():
    client = HolonClient(dimensions=4096)

    print("=" * 80)
    print("EXPERIMENT 1: Payload Similarity — Familiar vs Unfamiliar")
    print("=" * 80)
    print()
    print("Build baseline from 200 legitimate game packets, then score")
    print("legit and attack payloads against it.")
    print()

    rng = np.random.default_rng(42)

    # Build baseline from legitimate traffic
    legit_train = [make_legit_payload(rng) for _ in range(200)]
    baseline, accum = build_baseline(client, legit_train)

    # Test payloads
    legit_test = [make_legit_payload(rng) for _ in range(50)]
    attack_a = [make_attack_payload_a(rng) for _ in range(50)]
    attack_b = [make_attack_payload_b(rng) for _ in range(50)]
    attack_c = [make_attack_payload_c(rng) for _ in range(50)]

    legit_scores = score_payloads(client, baseline, legit_test)
    attack_a_scores = score_payloads(client, baseline, attack_a)
    attack_b_scores = score_payloads(client, baseline, attack_b)
    attack_c_scores = score_payloads(client, baseline, attack_c)

    print(f"  {'Traffic Type':>25} {'Mean Sim':>9} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-' * 56}")
    for name, scores in [
        ("Legitimate (test)", legit_scores),
        ("Attack A (overflow)", attack_a_scores),
        ("Attack B (spoofed hdr)", attack_b_scores),
        ("Attack C (random)", attack_c_scores),
    ]:
        print(f"  {name:>25} {np.mean(scores):>9.4f} {np.std(scores):>7.4f} "
              f"{np.min(scores):>7.4f} {np.max(scores):>7.4f}")

    print()
    # Adaptive threshold: mean - 2σ of legit scores
    legit_mean = np.mean(legit_scores)
    legit_std = np.std(legit_scores)
    adaptive_threshold = legit_mean - 3 * legit_std
    print(f"  Adaptive threshold (legit mean - 3σ): {adaptive_threshold:.4f}")
    print(f"    legit mean={legit_mean:.4f}, std={legit_std:.4f}")
    print()

    # Detection at various thresholds
    print(f"  Detection rates (flag if sim < threshold):")
    print(f"  {'Threshold':>10} {'Legit FP':>10} {'Atk A':>10} {'Atk B':>10} {'Atk C':>10}")
    print(f"  {'-' * 50}")
    for thresh in [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, adaptive_threshold]:
        fp = np.mean(legit_scores < thresh) * 100
        tp_a = np.mean(attack_a_scores < thresh) * 100
        tp_b = np.mean(attack_b_scores < thresh) * 100
        tp_c = np.mean(attack_c_scores < thresh) * 100
        marker = " ← adaptive" if abs(thresh - adaptive_threshold) < 0.001 else ""
        print(f"  {thresh:>10.4f} {fp:>9.0f}% {tp_a:>9.0f}% {tp_b:>9.0f}% {tp_c:>9.0f}%{marker}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Coherence Among Attack Payloads")
    print("=" * 80)
    print()
    print("Do attack payloads cluster together? (Same tool → same bytes)")
    print()

    legit_vecs = [encode_payload(client, p) for p in legit_test[:30]]
    attack_a_vecs = [encode_payload(client, p) for p in attack_a[:30]]
    attack_b_vecs = [encode_payload(client, p) for p in attack_b[:30]]
    attack_c_vecs = [encode_payload(client, p) for p in attack_c[:30]]

    print(f"  {'Group':>25} {'Coherence':>10}")
    print(f"  {'-' * 36}")
    print(f"  {'Legitimate':>25} {coherence(legit_vecs):>10.4f}")
    print(f"  {'Attack A (overflow)':>25} {coherence(attack_a_vecs):>10.4f}")
    print(f"  {'Attack B (spoofed hdr)':>25} {coherence(attack_b_vecs):>10.4f}")
    print(f"  {'Attack C (random)':>25} {coherence(attack_c_vecs):>10.4f}")

    # Mixed window (simulate real traffic mix)
    print()
    print("  Mixed windows (80% legit, 20% attack):")
    for attack_name, attack_set in [("Attack A", attack_a), ("Attack B", attack_b)]:
        mixed_payloads = legit_test[:40] + attack_set[:10]
        mixed_vecs = [encode_payload(client, p) for p in mixed_payloads]
        rng_shuffle = np.random.default_rng(42)
        indices = np.arange(len(mixed_vecs))
        rng_shuffle.shuffle(indices)
        mixed_vecs = [mixed_vecs[i] for i in indices]
        mixed_payloads_shuffled = [mixed_payloads[i] for i in indices]

        # Score all against baseline
        all_scores = score_payloads(client, baseline, mixed_payloads_shuffled)

        # Flag outliers using adaptive threshold
        flagged_mask = all_scores < adaptive_threshold
        flagged_vecs = [v for v, f in zip(mixed_vecs, flagged_mask) if f]

        if len(flagged_vecs) >= 2:
            flagged_coh = coherence(flagged_vecs)
        else:
            flagged_coh = 0.0

        print(f"    {attack_name}: {np.sum(flagged_mask)} flagged, "
              f"coherence among flagged: {flagged_coh:.4f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Attack Signature Extraction")
    print("=" * 80)
    print()
    print("Which byte positions distinguish attack from legitimate?")
    print()

    for attack_name, attack_set in [
        ("Attack A (overflow)", attack_a[:30]),
        ("Attack B (spoofed hdr)", attack_b[:30]),
    ]:
        _, pos_scores = extract_attack_signature(client, attack_set, baseline)

        print(f"  {attack_name}:")
        print(f"    Per-position similarity (attack vs baseline unbinding):")
        print(f"    Pos: ", end="")
        for i in range(min(32, len(pos_scores))):
            print(f"{i:>5}", end="")
        print()
        print(f"    Sim: ", end="")
        for i in range(min(32, len(pos_scores))):
            print(f"{pos_scores[i]:>5.2f}", end="")
        print()

        # Show the actual bytes at each position
        print(f"    Legit byte 0: ", end="")
        for i in range(min(32, len(legit_train[0]))):
            print(f" 0x{legit_train[0][i]:02x}", end="")
        print()
        print(f"    Atk byte 0:   ", end="")
        for i in range(min(32, len(attack_set[0]))):
            print(f" 0x{attack_set[0][i]:02x}", end="")
        print()
        print()

    print("=" * 80)
    print("EXPERIMENT 4: Byte Match Rule Generation")
    print("=" * 80)
    print()
    print("Generate (l4-match offset hex-match hex-mask) rules from attack patterns.")
    print()

    for attack_name, attack_set in [
        ("Attack A (overflow)", attack_a[:30]),
        ("Attack B (spoofed hdr)", attack_b[:30]),
    ]:
        _, pos_scores = extract_attack_signature(client, attack_set, baseline)
        rules = generate_byte_match_rule(attack_set, pos_scores,
                                         legit_payloads=legit_test)

        print(f"  {attack_name}:")
        if rules:
            for rule in rules:
                print(f"    Rule: {rule['rule']}")
                print(f"      Payload positions: {rule['positions']}")
                print(f"      Match bytes: {' '.join(f'0x{b:02x}' for b in rule['bytes'])}")
            print()

            # Validate: how many legit/attack payloads match each rule?
            print(f"    Validation (does rule match legit vs attack?):")
            for rule in rules:
                positions = rule["positions"]
                match_bytes_list = rule["bytes"]

                legit_match = 0
                for p in legit_test:
                    if all(p[pos] == mb for pos, mb in zip(positions, match_bytes_list)):
                        legit_match += 1

                attack_match = 0
                for p in attack_set:
                    if all(p[pos] == mb for pos, mb in zip(positions, match_bytes_list)):
                        attack_match += 1

                print(f"      {rule['rule']}:")
                print(f"        Legit matches: {legit_match}/{len(legit_test)} "
                      f"({legit_match/len(legit_test)*100:.0f}%)")
                print(f"        Attack matches: {attack_match}/{len(attack_set)} "
                      f"({attack_match/len(attack_set)*100:.0f}%)")
        else:
            print(f"    No rules generated (positions too noisy)")
        print()

    print("=" * 80)
    print("EXPERIMENT 5: Rate-Aware Detection")
    print("=" * 80)
    print()
    print("Combine payload anomaly + high rate for detection.")
    print("Attackers: many packets (high rate), unfamiliar bytes.")
    print("Legitimate: few packets (low rate), familiar bytes.")
    print()

    # Simulate a time window with mixed traffic
    # 10 legit clients at ~5 pps, 3 attack sources at ~50 pps each
    window_payloads = []
    window_sources = []
    for _ in range(50):  # 10 legit clients × 5 packets
        window_payloads.append(make_legit_payload(rng))
        window_sources.append(f"client_{rng.integers(0, 10)}")
    for _ in range(150):  # 3 attack sources × 50 packets
        window_payloads.append(make_attack_payload_a(rng))
        window_sources.append(f"attacker_{rng.integers(0, 3)}")

    # Shuffle
    indices = np.arange(len(window_payloads))
    rng.shuffle(indices)
    window_payloads = [window_payloads[i] for i in indices]
    window_sources = [window_sources[i] for i in indices]

    # Score all payloads
    all_scores = score_payloads(client, baseline, window_payloads)

    # Count source rates
    source_counts = {}
    for src in window_sources:
        source_counts[src] = source_counts.get(src, 0) + 1

    # Combined detection: anomalous payload + high rate source
    sim_threshold = adaptive_threshold
    rate_threshold = 20  # packets per window

    flagged_payloads = all_scores < sim_threshold
    flagged_sources = set()
    for i, (flagged, src) in enumerate(zip(flagged_payloads, window_sources)):
        if flagged and source_counts[src] > rate_threshold:
            flagged_sources.add(src)

    print(f"  Window: {len(window_payloads)} packets "
          f"({sum(1 for s in window_sources if s.startswith('client'))} legit, "
          f"{sum(1 for s in window_sources if s.startswith('attacker'))} attack)")
    print(f"  Payload anomaly threshold: sim < {sim_threshold}")
    print(f"  Rate threshold: > {rate_threshold} pkt/window")
    print()
    print(f"  {'Source':>15} {'Packets':>8} {'Anomalous':>10} {'Flagged':>8}")
    print(f"  {'-' * 44}")

    for src in sorted(set(window_sources)):
        src_indices = [i for i, s in enumerate(window_sources) if s == src]
        src_anomalous = sum(1 for i in src_indices if all_scores[i] < sim_threshold)
        is_flagged = src in flagged_sources
        print(f"  {src:>15} {len(src_indices):>8} {src_anomalous:>10} "
              f"{'YES' if is_flagged else 'no':>8}")

    print()
    legit_flagged = sum(1 for s in flagged_sources if s.startswith("client"))
    attack_flagged = sum(1 for s in flagged_sources if s.startswith("attacker"))
    print(f"  False positives: {legit_flagged} legit sources flagged")
    print(f"  True positives:  {attack_flagged}/3 attack sources detected")

    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Full Pipeline — Detect, Cluster, Extract, Rule")
    print("=" * 80)
    print()
    print("End-to-end: window of mixed traffic → l4-match rules.")
    print()

    # Re-score the mixed window
    pipeline_scores = score_payloads(client, baseline, window_payloads)
    anomalous_indices = [i for i in range(len(pipeline_scores)) if pipeline_scores[i] < sim_threshold]
    anomalous_payloads = [window_payloads[i] for i in anomalous_indices]
    anomalous_vecs = [encode_payload(client, p) for p in anomalous_payloads]

    if len(anomalous_vecs) >= 2:
        coh = coherence(anomalous_vecs)
        print(f"  Step 1: {len(anomalous_indices)} anomalous payloads detected (sim < {sim_threshold})")
        print(f"  Step 2: Coherence among anomalous = {coh:.4f}")

        if coh > 0.15:  # Above random-noise coherence
            print(f"  Step 3: High coherence → coordinated attack cluster confirmed")

            # Extract signature and generate rules
            _, pos_scores = extract_attack_signature(client, anomalous_payloads[:30], baseline)
            rules = generate_byte_match_rule(anomalous_payloads[:30], pos_scores,
                                             legit_payloads=legit_test)

            if rules:
                print(f"  Step 4: Generated {len(rules)} byte match rules:")
                for rule in rules:
                    print(f"    → {rule['rule']}")

                    # Validate
                    legit_matches = sum(
                        1 for p in legit_test
                        if all(p[pos] == mb for pos, mb in zip(rule["positions"], rule["bytes"]))
                    )
                    attack_matches = sum(
                        1 for p in anomalous_payloads
                        if all(p[pos] == mb for pos, mb in zip(rule["positions"], rule["bytes"]))
                    )
                    print(f"       Hit rate: {attack_matches}/{len(anomalous_payloads)} attack "
                          f"({attack_matches/len(anomalous_payloads)*100:.0f}%), "
                          f"{legit_matches}/{len(legit_test)} legit "
                          f"({legit_matches/len(legit_test)*100:.0f}%)")
            else:
                print(f"  Step 4: Could not generate clean rules (noisy positions)")
        else:
            print(f"  Step 3: Low coherence → likely noise, not coordinated attack")
    else:
        print(f"  Fewer than 2 anomalous payloads — no cluster to analyze")

    print()


if __name__ == "__main__":
    main()
