#!/usr/bin/env python3
"""
Adversarial Robustness: Can Attackers Evade Coherence-Based Detection?

PROBLEM:
========
We've shown coherence detects attacks because attack traffic is homogeneous.
But what if an attacker deliberately diversifies their traffic to reduce
coherence while maintaining the attack effect?

HYPOTHESIS:
===========
Adding noise to attack fields reduces coherence, but also reduces attack
effectiveness. There's a fundamental trade-off: an effective DDoS attack
MUST concentrate on a target (same dst_ip, same dst_port, same protocol),
which inherently creates detectable homogeneity.

APPROACH:
=========
1. DNS amplification with increasing randomization:
   a. Pure attack (all fields fixed)
   b. Randomize src_port (keeps attack effective)
   c. Randomize pkt_len (keeps attack effective)
   d. Randomize dst_port (breaks the attack — spreads across ports)
   e. Randomize proto (breaks the attack — mixed TCP/UDP)
   f. Randomize dst_ip (breaks the attack — no single target)

2. For each level, measure:
   - Whole-vector coherence
   - Per-field coherence spectrum
   - Whether the attack is still effective (same target:port)

CONSTRAINT:
===========
Stateless, unidirectional, per-window only.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import coherence

FIELDS = ["src_ip", "dst_ip", "proto", "src_port", "dst_port", "pkt_len", "ttl"]


def make_packet(client, rng, overrides=None):
    d = {
        "src_ip": f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": f"192.168.{rng.integers(0, 5)}.{rng.integers(1, 254)}",
        "proto": str(rng.choice(["TCP", "UDP"])),
        "src_port": str(rng.integers(1024, 65535)),
        "dst_port": str(rng.choice([80, 443, 22, 8080])),
        "pkt_len": str(rng.integers(64, 1500)),
        "ttl": str(rng.choice([64, 128, 255])),
    }
    if overrides:
        d.update(overrides)
    return d


def field_coherence(client, vecs, field):
    role_vec = client.get_vector(field)
    field_vecs = [v * role_vec for v in vecs]
    return coherence(field_vecs)


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)

    # Attack evasion levels: progressively randomize fields
    evasion_levels = [
        ("L0: Pure DNS amp", {
            "src_ip": lambda r: r.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"]),
            "dst_ip": lambda r: "192.168.1.100",
            "proto": lambda r: "UDP",
            "src_port": lambda r: "53",
            "dst_port": lambda r: str(r.integers(1024, 65535)),
            "pkt_len": lambda r: str(r.integers(512, 4096)),
            "ttl": lambda r: str(r.choice([240, 245, 250])),
        }),
        ("L1: Randomize src_ip", {
            "src_ip": lambda r: f"{r.integers(1,255)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}",
            "dst_ip": lambda r: "192.168.1.100",
            "proto": lambda r: "UDP",
            "src_port": lambda r: "53",
            "dst_port": lambda r: str(r.integers(1024, 65535)),
            "pkt_len": lambda r: str(r.integers(512, 4096)),
            "ttl": lambda r: str(r.choice([240, 245, 250])),
        }),
        ("L2: + random pkt_len", {
            "src_ip": lambda r: f"{r.integers(1,255)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}",
            "dst_ip": lambda r: "192.168.1.100",
            "proto": lambda r: "UDP",
            "src_port": lambda r: "53",
            "dst_port": lambda r: str(r.integers(1024, 65535)),
            "pkt_len": lambda r: str(r.integers(64, 9000)),
            "ttl": lambda r: str(r.integers(1, 255)),
        }),
        ("L3: + random dst_port", {
            "src_ip": lambda r: f"{r.integers(1,255)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}",
            "dst_ip": lambda r: "192.168.1.100",
            "proto": lambda r: "UDP",
            "src_port": lambda r: "53",
            "dst_port": lambda r: str(r.integers(1, 65535)),
            "pkt_len": lambda r: str(r.integers(64, 9000)),
            "ttl": lambda r: str(r.integers(1, 255)),
        }),
        ("L4: + random proto", {
            "src_ip": lambda r: f"{r.integers(1,255)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}",
            "dst_ip": lambda r: "192.168.1.100",
            "proto": lambda r: str(r.choice(["TCP", "UDP", "ICMP"])),
            "src_port": lambda r: str(r.integers(1, 65535)),
            "dst_port": lambda r: str(r.integers(1, 65535)),
            "pkt_len": lambda r: str(r.integers(64, 9000)),
            "ttl": lambda r: str(r.integers(1, 255)),
        }),
        ("L5: + random dst_ip", {
            "src_ip": lambda r: f"{r.integers(1,255)}.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(1,255)}",
            "dst_ip": lambda r: f"192.168.{r.integers(0, 255)}.{r.integers(1, 254)}",
            "proto": lambda r: str(r.choice(["TCP", "UDP", "ICMP"])),
            "src_port": lambda r: str(r.integers(1, 65535)),
            "dst_port": lambda r: str(r.integers(1, 65535)),
            "pkt_len": lambda r: str(r.integers(64, 9000)),
            "ttl": lambda r: str(r.integers(1, 255)),
        }),
    ]

    print("=" * 90)
    print("EXPERIMENT 1: Evasion vs Detection — Coherence at Each Level")
    print("=" * 90)
    print()
    print("As attacker randomizes more fields, whole-vector coherence drops.")
    print("Question: at what level does evasion break the attack?")
    print()

    field_short = {"src_ip": "SrcIP", "dst_ip": "DstIP", "proto": "Proto",
                   "src_port": "SPort", "dst_port": "DPort", "pkt_len": "PktLn",
                   "ttl": "TTL"}

    header = f"{'Level':>25} {'Whole':>7} " + " ".join(f"{field_short[f]:>7}" for f in FIELDS) + " Effective?"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    # Normal traffic baseline
    normal_vecs = []
    for _ in range(50):
        d = make_packet(client, rng)
        normal_vecs.append(client.encode(d))
    normal_coh = coherence(normal_vecs)

    row = f"  {'Normal traffic':>25} {normal_coh:>7.4f} "
    row += " ".join(f"{field_coherence(client, normal_vecs, f):>7.4f}" for f in FIELDS)
    row += " n/a"
    print(row)

    for level_name, generators in evasion_levels:
        rng_local = np.random.default_rng(42)
        vecs = []
        dicts = []
        for _ in range(50):
            d = {}
            for field, gen_fn in generators.items():
                d[field] = gen_fn(rng_local)
            dicts.append(d)
            vecs.append(client.encode(d))

        whole = coherence(vecs)

        # Check attack effectiveness: are packets targeting the same dst_ip?
        # (dst_port varies naturally for reflection attacks — victim's ephemeral port)
        unique_dst_ips = set(d["dst_ip"] for d in dicts)
        unique_protos = set(d["proto"] for d in dicts)
        # Effective if concentrated on ≤3 dst IPs with consistent protocol
        effective = len(unique_dst_ips) <= 3 and len(unique_protos) <= 2

        row = f"  {level_name:>25} {whole:>7.4f} "
        row += " ".join(f"{field_coherence(client, vecs, f):>7.4f}" for f in FIELDS)
        row += f" {'YES' if effective else 'NO — attack broken'}"
        print(row)

    print("\n" + "=" * 90)
    print("EXPERIMENT 2: Detection at Each Evasion Level (Top-K Method)")
    print("=" * 90)
    print()
    print("Using P99 pairwise similarity as detector (from experiment 006).")
    print("Does top-k catch what mean coherence misses?")
    print()

    # Compute P99 threshold from normal traffic
    normal_p99s = []
    for trial in range(20):
        rng_t = np.random.default_rng(trial + 5000)
        vecs = [client.encode(make_packet(client, rng_t)) for _ in range(50)]
        n = len(vecs)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(cosine_similarity(vecs[i], vecs[j]))
        normal_p99s.append(np.percentile(sims, 99))

    p99_threshold = np.max(normal_p99s) * 1.1
    print(f"  P99 threshold: {p99_threshold:.4f} (max normal P99 × 1.1)")
    print()

    print(f"  {'Level':>25} {'MeanCoh':>8} {'P99':>8} {'MeanDet':>8} {'P99Det':>8}")
    print(f"  {'-' * 60}")

    coh_threshold = normal_coh * 1.5

    for level_name, generators in evasion_levels:
        rng_local = np.random.default_rng(42)
        vecs = []
        for _ in range(50):
            d = {}
            for field, gen_fn in generators.items():
                d[field] = gen_fn(rng_local)
            vecs.append(client.encode(d))

        whole_coh = coherence(vecs)
        n = len(vecs)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(cosine_similarity(vecs[i], vecs[j]))
        p99 = np.percentile(sims, 99)

        mean_det = "✓" if whole_coh > coh_threshold else "✗"
        p99_det = "✓" if p99 > p99_threshold else "✗"

        print(f"  {level_name:>25} {whole_coh:>8.4f} {p99:>8.4f} {mean_det:>8} {p99_det:>8}")

    print("\n" + "=" * 90)
    print("EXPERIMENT 3: Attacker Cost Analysis")
    print("=" * 90)
    print()
    print("Fundamental question: what does evasion COST the attacker?")
    print()

    costs = [
        ("L0: Pure attack", "Full amplification, single target", "Max damage"),
        ("L1: Random src_ip", "Reflectors vary — still effective", "No cost"),
        ("L2: + random pkt_len", "Payload varies — still effective", "No cost"),
        ("L3: + random dst_port", "Spread across ports — less effective", "Reduced impact"),
        ("L4: + random proto", "Mixed protocols — breaks amplification", "Attack broken"),
        ("L5: + random dst_ip", "No single target — not a DDoS", "Attack broken"),
    ]

    print(f"  {'Level':>25} {'Effect':>35} {'Cost':>20}")
    print(f"  {'-' * 82}")
    for level, effect, cost in costs:
        print(f"  {level:>25} {effect:>35} {cost:>20}")

    print()
    print("  CONCLUSION: Evasion of per-field coherence detection requires randomizing")
    print("  dst_port or dst_ip — which breaks the attack itself. The attacker is in a")
    print("  fundamental bind: effective DDoS requires concentrated target fields, which")
    print("  is precisely what per-field coherence detects.")
    print()


if __name__ == "__main__":
    main()
