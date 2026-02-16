#!/usr/bin/env python3
"""
Reject for Novel Attack Isolation

HYPOTHESIS:
===========
reject(vec, subspace) extracts the orthogonal complement — what CANNOT be
explained by known patterns. After projecting onto known attack profiles,
the residual reveals novel attack vectors.

Combined with negate() for attack peeling, this enables layered attack
discovery: peel known attacks, examine what's left.

PRIMITIVES DEMONSTRATED:
========================
1. reject()         - Orthogonal complement of project
2. project()        - Subspace projection (for comparison)
3. negate()         - Attack peeling
4. coherence()      - Is the residual structured or noise?
5. significance()   - Is the residual statistically significant?

VECTOR PROPERTIES EXPLOITED:
============================
- Subspace orthogonality (what the vector ISN'T)
- Layered signal separation (iterative peeling)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.accumulator import accumulate, create_accumulator, normalize_accumulator
from holon.distance import significance
from holon.primitives import (
    bundle,
    coherence,
    negate,
    project,
    reject,
)


def make_packet(client, src_ip, dst_ip, proto, src_port, dst_port, pkt_len):
    return client.encode(
        {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "proto": proto,
            "src_port": src_port,
            "dst_port": dst_port,
            "pkt_len": pkt_len,
        }
    )


def build_attack_profile(client, rng, attack_type, count=100):
    """Build a profile vector for a known attack type."""
    vecs = []
    for _ in range(count):
        if attack_type == "dns_amp":
            vecs.append(
                make_packet(
                    client,
                    src_ip=rng.choice(["8.8.8.8", "1.1.1.1"]),
                    dst_ip="192.168.1.100",
                    proto="UDP",
                    src_port=53,
                    dst_port=int(rng.integers(1024, 65535)),
                    pkt_len=int(rng.integers(512, 4096)),
                )
            )
        elif attack_type == "syn_flood":
            vecs.append(
                make_packet(
                    client,
                    src_ip=f"{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                    dst_ip="192.168.1.100",
                    proto="TCP",
                    src_port=int(rng.integers(1024, 65535)),
                    dst_port=80,
                    pkt_len=60,
                )
            )
        elif attack_type == "ntp_amp":
            vecs.append(
                make_packet(
                    client,
                    src_ip=rng.choice(["216.239.35.0", "216.239.35.4"]),
                    dst_ip="192.168.1.100",
                    proto="UDP",
                    src_port=123,
                    dst_port=int(rng.integers(1024, 65535)),
                    pkt_len=int(rng.integers(400, 500)),
                )
            )
    return bundle(vecs)


def generate_traffic(client, rng, normal_count, attack_configs):
    """Generate mixed traffic with multiple potential attacks."""
    vecs = []

    # Normal traffic
    for _ in range(normal_count):
        vecs.append(
            make_packet(
                client,
                src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                dst_ip=f"192.168.1.{rng.integers(1, 10)}",
                proto=rng.choice(["TCP", "UDP"]),
                src_port=int(rng.integers(1024, 65535)),
                dst_port=int(rng.choice([80, 443, 22, 8080])),
                pkt_len=int(rng.integers(64, 1500)),
            )
        )

    # Attack traffic
    for attack_type, count in attack_configs:
        for _ in range(count):
            if attack_type == "dns_amp":
                vecs.append(
                    make_packet(
                        client,
                        src_ip=rng.choice(["8.8.8.8", "1.1.1.1"]),
                        dst_ip="192.168.1.100",
                        proto="UDP",
                        src_port=53,
                        dst_port=int(rng.integers(1024, 65535)),
                        pkt_len=1400,
                    )
                )
            elif attack_type == "ntp_amp":
                vecs.append(
                    make_packet(
                        client,
                        src_ip="216.239.35.0",
                        dst_ip="192.168.1.100",
                        proto="UDP",
                        src_port=123,
                        dst_port=int(rng.integers(1024, 65535)),
                        pkt_len=468,
                    )
                )
            elif attack_type == "novel_ssdp":
                vecs.append(
                    make_packet(
                        client,
                        src_ip=f"172.16.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                        dst_ip="192.168.1.100",
                        proto="UDP",
                        src_port=1900,
                        dst_port=int(rng.integers(1024, 65535)),
                        pkt_len=int(rng.integers(200, 400)),
                    )
                )

    rng.shuffle(vecs)
    return bundle(vecs), vecs


def main():
    client = HolonClient(dimensions=4096)
    d = 4096

    print("=" * 70)
    print("EXPERIMENT 1: Known attack detection via projection")
    print("=" * 70)

    # Build known attack profiles
    dns_profile = build_attack_profile(client, np.random.default_rng(10), "dns_amp")
    syn_profile = build_attack_profile(client, np.random.default_rng(20), "syn_flood")
    ntp_profile = build_attack_profile(client, np.random.default_rng(30), "ntp_amp")

    known_attacks = [dns_profile, syn_profile, ntp_profile]

    # Scenario 1: Traffic with ONLY a known attack (DNS amp)
    traffic_known, _ = generate_traffic(
        client,
        np.random.default_rng(42),
        normal_count=100,
        attack_configs=[("dns_amp", 50)],
    )

    # Scenario 2: Traffic with a NOVEL attack (SSDP)
    traffic_novel, _ = generate_traffic(
        client,
        np.random.default_rng(42),
        normal_count=100,
        attack_configs=[("novel_ssdp", 50)],
    )

    # Scenario 3: LAYERED — known + novel attacks simultaneously
    traffic_layered, _ = generate_traffic(
        client,
        np.random.default_rng(42),
        normal_count=100,
        attack_configs=[("dns_amp", 40), ("novel_ssdp", 40)],
    )

    print(f"\n{'Scenario':<30} {'Proj→Known':>12} {'Residual Mag':>14} {'Novel Signal':>14}")
    print("-" * 75)

    for label, traffic in [
        ("Known attack (DNS)", traffic_known),
        ("Novel attack (SSDP)", traffic_novel),
        ("Layered (DNS + SSDP)", traffic_layered),
    ]:
        projected = project(traffic, known_attacks)
        rejected = reject(traffic, known_attacks)

        proj_sim = cosine_similarity(traffic, projected)
        residual_magnitude = np.sum(np.abs(rejected)) / d
        novel_signal = cosine_similarity(traffic, rejected)

        print(f"{label:<30} {proj_sim:>12.4f} {residual_magnitude:>14.4f} {novel_signal:>14.4f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Attack peeling with negate()")
    print("=" * 70)

    traffic_layered, _ = generate_traffic(
        client,
        np.random.default_rng(42),
        normal_count=100,
        attack_configs=[("dns_amp", 40), ("novel_ssdp", 40)],
    )

    print("\n  Step 1: Detect known attacks via projection")
    for label, profile in [("DNS Amp", dns_profile), ("SYN Flood", syn_profile), ("NTP Amp", ntp_profile)]:
        sim = cosine_similarity(traffic_layered, profile)
        z = significance(sim, d)
        detected = z > 2.0
        print(f"    {label:<15}: sim={sim:+.4f}, z={z:.2f} {'← DETECTED' if detected else ''}")

    print("\n  Step 2: Peel the detected attack (DNS)")
    peeled = negate(traffic_layered, dns_profile)
    sim_after_peel = cosine_similarity(peeled, dns_profile)
    print(f"    After peeling DNS: sim to DNS profile = {sim_after_peel:+.4f}")

    print("\n  Step 3: Reject known attacks from peeled traffic")
    residual = reject(peeled, known_attacks)
    residual_mag = np.sum(np.abs(residual)) / d

    # Build SSDP profile for validation
    ssdp_profile = build_attack_profile(client, np.random.default_rng(50), "ntp_amp")
    ssdp_vecs = []
    for _ in range(100):
        ssdp_vecs.append(
            make_packet(
                client,
                src_ip=f"172.16.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                dst_ip="192.168.1.100",
                proto="UDP",
                src_port=1900,
                dst_port=int(np.random.randint(1024, 65535)),
                pkt_len=int(np.random.randint(200, 400)),
            )
        )
    actual_ssdp_profile = bundle(ssdp_vecs)
    sim_to_novel = cosine_similarity(residual, actual_ssdp_profile)

    print(f"    Residual magnitude: {residual_mag:.4f}")
    print(f"    Residual similarity to SSDP pattern: {sim_to_novel:+.4f}")
    print(f"    → {'Novel attack signal detected!' if residual_mag > 0.5 else 'Residual is noise'}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Reject as anomaly isolation (baseline rejection)")
    print("=" * 70)

    # Build normal baseline
    normal_vecs = []
    rng = np.random.default_rng(42)
    for _ in range(200):
        normal_vecs.append(
            make_packet(
                client,
                src_ip=f"10.0.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                dst_ip=f"192.168.1.{rng.integers(1, 10)}",
                proto=rng.choice(["TCP", "UDP"]),
                src_port=int(rng.integers(1024, 65535)),
                dst_port=int(rng.choice([80, 443, 22])),
                pkt_len=int(rng.integers(64, 1500)),
            )
        )
    normal_baseline = bundle(normal_vecs)

    # Reject baseline from anomalous traffic → pure anomaly
    anomalous, _ = generate_traffic(
        client,
        np.random.default_rng(99),
        normal_count=80,
        attack_configs=[("dns_amp", 30)],
    )

    anomaly_signal = reject(anomalous, [normal_baseline])
    sim_to_dns = cosine_similarity(anomaly_signal, dns_profile)
    sim_to_baseline = cosine_similarity(anomaly_signal, normal_baseline)

    print(f"\n  Reject baseline from mixed traffic:")
    print(f"    Residual sim to DNS profile:  {sim_to_dns:+.4f}")
    print(f"    Residual sim to baseline:     {sim_to_baseline:+.4f}")
    print(f"    → Residual should correlate with attack, not baseline")

    print()


if __name__ == "__main__":
    main()
