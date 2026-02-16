#!/usr/bin/env python3
"""
Coherence as Baseline-Free Attack Detection

HYPOTHESIS:
===========
coherence() measures mean pairwise cosine similarity of a vector window.
DDoS traffic is homogeneous (high coherence), normal traffic is diverse
(low coherence). This should detect attacks WITHOUT a baseline.

PRIMITIVES DEMONSTRATED:
========================
1. coherence()     - Mean pairwise similarity (cluster tightness)
2. significance()  - Convert coherence to z-score (kill magic thresholds)

SCENARIO:
=========
Three traffic phases:
  Phase 1: Normal diverse traffic (should have low coherence)
  Phase 2: DNS amplification attack (should have high coherence)
  Phase 3: Botnet SYN flood (many sources but same pattern — high coherence?)

We measure coherence with NO baseline. Can we detect both attack types
purely from window homogeneity?

VECTOR PROPERTIES EXPLOITED:
============================
- Pairwise similarity distribution (not point comparison to a reference)
- Statistical geometry of high dimensions (significance)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.distance import significance
from holon.primitives import coherence


def make_packet(client, src_ip, dst_ip, proto, src_port, dst_port, pkt_len, ttl):
    return client.encode(
        {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "proto": proto,
            "src_port": src_port,
            "dst_port": dst_port,
            "pkt_len": pkt_len,
            "ttl": ttl,
        }
    )


def generate_normal_window(client, rng, count=50):
    """Diverse normal traffic: varied IPs, ports, protocols."""
    vecs = []
    protos = ["TCP", "UDP", "TCP", "TCP"]
    dst_ports = [80, 443, 22, 8080, 3306]
    for _ in range(count):
        vecs.append(
            make_packet(
                client,
                src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                dst_ip=f"192.168.1.{rng.integers(1, 10)}",
                proto=rng.choice(protos),
                src_port=int(rng.integers(1024, 65535)),
                dst_port=int(rng.choice(dst_ports)),
                pkt_len=int(rng.integers(64, 1500)),
                ttl=int(rng.choice([64, 128, 255])),
            )
        )
    return vecs


def generate_dns_amplification_window(client, rng, count=50):
    """DNS amplification: few sources, one target, all UDP/53, large packets."""
    vecs = []
    reflectors = ["8.8.8.8", "1.1.1.1", "9.9.9.9"]
    for _ in range(count):
        vecs.append(
            make_packet(
                client,
                src_ip=rng.choice(reflectors),
                dst_ip="192.168.1.100",
                proto="UDP",
                src_port=53,
                dst_port=int(rng.integers(1024, 65535)),
                pkt_len=int(rng.integers(512, 4096)),
                ttl=int(rng.choice([240, 245, 250])),
            )
        )
    return vecs


def generate_botnet_syn_flood(client, rng, count=50):
    """Botnet SYN flood: many sources, one target, all TCP/80, small packets."""
    vecs = []
    for _ in range(count):
        vecs.append(
            make_packet(
                client,
                src_ip=f"{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                dst_ip="192.168.1.100",
                proto="TCP",
                src_port=int(rng.integers(1024, 65535)),
                dst_port=80,
                pkt_len=60,
                ttl=int(rng.choice([64, 128, 255])),
            )
        )
    return vecs


def generate_slow_ramp(client, rng, count=50, attack_fraction=0.0):
    """Mixed traffic: gradually increasing fraction of attack packets."""
    vecs = []
    for _ in range(count):
        if rng.random() < attack_fraction:
            vecs.append(
                make_packet(
                    client,
                    src_ip="8.8.8.8",
                    dst_ip="192.168.1.100",
                    proto="UDP",
                    src_port=53,
                    dst_port=int(rng.integers(1024, 65535)),
                    pkt_len=1400,
                    ttl=245,
                )
            )
        else:
            vecs.append(
                make_packet(
                    client,
                    src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                    dst_ip=f"192.168.1.{rng.integers(1, 10)}",
                    proto=rng.choice(["TCP", "UDP"]),
                    src_port=int(rng.integers(1024, 65535)),
                    dst_port=int(rng.choice([80, 443, 22])),
                    pkt_len=int(rng.integers(64, 1500)),
                    ttl=int(rng.choice([64, 128])),
                )
            )
    return vecs


def main():
    client = HolonClient(dimensions=4096)
    rng = np.random.default_rng(42)
    d = 4096

    print("=" * 70)
    print("EXPERIMENT 1: Coherence across traffic types (no baseline)")
    print("=" * 70)

    # Generate windows
    normal_vecs = generate_normal_window(client, rng, count=50)
    dns_amp_vecs = generate_dns_amplification_window(client, rng, count=50)
    botnet_vecs = generate_botnet_syn_flood(client, rng, count=50)

    c_normal = coherence(normal_vecs)
    c_dns = coherence(dns_amp_vecs)
    c_botnet = coherence(botnet_vecs)

    z_normal = significance(c_normal, d)
    z_dns = significance(c_dns, d)
    z_botnet = significance(c_botnet, d)

    print(f"\n{'Traffic Type':<25} {'Coherence':>10} {'Z-Score':>10} {'Detection':>12}")
    print("-" * 60)
    print(f"{'Normal (diverse)':<25} {c_normal:>10.4f} {z_normal:>10.2f} {'✓ NORMAL' if z_normal < 3.0 else '✗ FALSE POS':>12}")
    print(f"{'DNS Amplification':<25} {c_dns:>10.4f} {z_dns:>10.2f} {'✓ ATTACK' if z_dns > 3.0 else '✗ MISSED':>12}")
    print(f"{'Botnet SYN Flood':<25} {c_botnet:>10.4f} {z_botnet:>10.2f} {'✓ ATTACK' if z_botnet > 3.0 else '✗ MISSED':>12}")

    # Validate
    attack_detected = z_dns > 3.0 and z_botnet > 3.0
    no_false_pos = z_normal < 3.0
    print(f"\nAttacks detected (no baseline): {'PASS' if attack_detected else 'FAIL'}")
    print(f"No false positives:             {'PASS' if no_false_pos else 'FAIL'}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sensitivity — how much attack traffic triggers detection?")
    print("=" * 70)

    print(f"\n{'Attack %':>10} {'Coherence':>10} {'Z-Score':>10} {'Detected':>10}")
    print("-" * 45)

    for pct in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]:
        mixed = generate_slow_ramp(client, np.random.default_rng(int(pct * 1000) + 1), count=80, attack_fraction=pct)
        c = coherence(mixed)
        z = significance(c, d)
        detected = "YES" if z > 3.0 else "no"
        print(f"{pct * 100:>9.0f}% {c:>10.4f} {z:>10.2f} {detected:>10}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Window size sensitivity")
    print("=" * 70)

    print(f"\n{'Window Size':>12} {'Normal Coh':>12} {'Attack Coh':>12} {'Separation':>12}")
    print("-" * 52)

    for window_size in [10, 20, 30, 50, 80, 100]:
        rng_w = np.random.default_rng(42)
        n_vecs = generate_normal_window(client, rng_w, count=window_size)
        a_vecs = generate_dns_amplification_window(client, rng_w, count=window_size)
        c_n = coherence(n_vecs)
        c_a = coherence(a_vecs)
        sep = c_a - c_n
        print(f"{window_size:>12} {c_n:>12.4f} {c_a:>12.4f} {sep:>12.4f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Coherence vs similarity-to-baseline comparison")
    print("=" * 70)

    from holon.accumulator import accumulate, create_accumulator, normalize_accumulator

    # Build baseline from normal traffic
    acc = create_accumulator(4096)
    baseline_vecs = generate_normal_window(client, np.random.default_rng(99), count=200)
    for v in baseline_vecs:
        acc = accumulate(acc, v)
    baseline = normalize_accumulator(acc)

    print(f"\n{'Traffic Type':<25} {'Coherence':>10} {'Coh Z':>8} {'Sim→Base':>10} {'Sim Z':>8}")
    print("-" * 65)

    for label, gen_fn, seed in [
        ("Normal", generate_normal_window, 77),
        ("DNS Amplification", generate_dns_amplification_window, 77),
        ("Botnet SYN Flood", generate_botnet_syn_flood, 77),
    ]:
        vecs = gen_fn(client, np.random.default_rng(seed), count=50)
        c = coherence(vecs)
        z_c = significance(c, d)

        from holon.primitives import bundle

        window_vec = bundle(vecs)
        sim = cosine_similarity(window_vec, baseline)
        z_s = significance(sim, d)
        print(f"{label:<25} {c:>10.4f} {z_c:>8.2f} {sim:>10.4f} {z_s:>8.2f}")

    print()


if __name__ == "__main__":
    main()
