#!/usr/bin/env python3
"""
Per-Field Coherence Spectrum: Structural Fingerprinting via Unbinding

PROBLEM:
========
Whole-vector coherence is a single number. It tells us that packets in a
window are similar, but NOT which fields are responsible. We can't distinguish
"all same dst_port" (SYN flood) from "all same src_ip" (amplification) from
"all same proto" (just UDP-heavy traffic).

HYPOTHESIS:
===========
Use unbinding to isolate individual field contributions, then measure
coherence per field. This produces a SPECTRUM — a vector of per-field
coherence values that fingerprints the attack TYPE, not just its presence.

APPROACH:
=========
1. For each field f in {src_ip, dst_ip, proto, src_port, dst_port, pkt_len, ttl}:
   a. Unbind each packet vector by the field's role vector: field_vec = unbind(packet, role_f)
   b. Compute coherence across all field_vecs in the window
2. The result is a coherence spectrum: [coh_src_ip, coh_dst_ip, coh_proto, ...]
3. Different attack types should produce distinctive spectra

VECTOR PROPERTIES EXPLOITED:
============================
- Unbinding reverses role binding → isolates field contributions
- Per-field coherence → structural decomposition of traffic
- Spectrum shape → attack type fingerprint (stateless)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import coherence

FIELDS = ["src_ip", "dst_ip", "proto", "src_port", "dst_port", "pkt_len", "ttl"]


def make_packet_dict(rng, src_ip=None, dst_ip=None, proto=None, src_port=None,
                     dst_port=None, pkt_len=None, ttl=None):
    """Generate a packet dict with optional fixed fields."""
    return {
        "src_ip": src_ip or f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "dst_ip": dst_ip or f"192.168.{rng.integers(0, 5)}.{rng.integers(1, 254)}",
        "proto": proto or str(rng.choice(["TCP", "UDP", "ICMP"])),
        "src_port": src_port or str(rng.integers(1024, 65535)),
        "dst_port": dst_port or str(rng.choice([80, 443, 22, 53, 8080, 3306])),
        "pkt_len": pkt_len or str(rng.integers(64, 1500)),
        "ttl": ttl or str(rng.choice([64, 128, 255])),
    }


def compute_field_spectrum(client, packets, field_names=FIELDS):
    """Compute per-field coherence by unbinding each field's role vector.

    For each field, we unbind the packet vector with the field's position
    vector to isolate that field's contribution, then measure coherence.
    """
    spectrum = {}

    for field in field_names:
        role_vec = client.get_vector(field)
        field_vecs = []
        for pkt_vec in packets:
            field_vec = pkt_vec * role_vec  # unbind = element-wise multiply for bipolar
            field_vecs.append(field_vec)

        if len(field_vecs) >= 2:
            spectrum[field] = coherence(field_vecs)
        else:
            spectrum[field] = 0.0

    return spectrum


def generate_traffic(client, rng, scenario, count=50):
    """Generate packets for different attack scenarios."""
    packets = []
    dicts = []

    if scenario == "normal":
        for _ in range(count):
            d = make_packet_dict(rng)
            dicts.append(d)
            packets.append(client.encode(d))

    elif scenario == "dns_amplification":
        for _ in range(count):
            d = make_packet_dict(
                rng,
                src_ip=str(rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"])),
                proto="UDP",
                src_port="53",
                pkt_len=str(rng.integers(512, 4096)),
                ttl=str(rng.choice([240, 245, 250])),
            )
            dicts.append(d)
            packets.append(client.encode(d))

    elif scenario == "syn_flood":
        for _ in range(count):
            d = make_packet_dict(
                rng,
                dst_ip="192.168.1.100",
                proto="TCP",
                dst_port="80",
                pkt_len="60",
                ttl=str(rng.choice([64, 128, 255])),
            )
            dicts.append(d)
            packets.append(client.encode(d))

    elif scenario == "ntp_amplification":
        for _ in range(count):
            d = make_packet_dict(
                rng,
                src_port="123",
                proto="UDP",
                pkt_len=str(rng.integers(440, 480)),
                ttl=str(rng.choice([50, 51, 52, 53])),
            )
            dicts.append(d)
            packets.append(client.encode(d))

    elif scenario == "ssdp_amplification":
        for _ in range(count):
            d = make_packet_dict(
                rng,
                src_port="1900",
                proto="UDP",
                pkt_len=str(rng.integers(300, 400)),
            )
            dicts.append(d)
            packets.append(client.encode(d))

    elif scenario == "icmp_flood":
        for _ in range(count):
            d = make_packet_dict(
                rng,
                proto="ICMP",
                src_port="0",
                dst_port="0",
                pkt_len=str(rng.integers(56, 1500)),
            )
            dicts.append(d)
            packets.append(client.encode(d))

    elif scenario == "port_scan":
        # Single src, single dst, sequencing through ports
        src = f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        dst = "192.168.1.100"
        for i in range(count):
            d = make_packet_dict(
                rng,
                src_ip=src,
                dst_ip=dst,
                proto="TCP",
                dst_port=str(20 + i),
                pkt_len="60",
            )
            dicts.append(d)
            packets.append(client.encode(d))

    return packets, dicts


def main():
    client = HolonClient(dimensions=4096)

    scenarios = [
        "normal",
        "dns_amplification",
        "syn_flood",
        "ntp_amplification",
        "ssdp_amplification",
        "icmp_flood",
        "port_scan",
    ]

    print("=" * 90)
    print("EXPERIMENT 1: Per-Field Coherence Spectrum")
    print("=" * 90)
    print()
    print("Each row is a scenario. Each column is per-field coherence after unbinding.")
    print("High values = that field is homogeneous across the window.")
    print()

    field_short = {"src_ip": "SrcIP", "dst_ip": "DstIP", "proto": "Proto",
                   "src_port": "SPort", "dst_port": "DPort", "pkt_len": "PktLn",
                   "ttl": "TTL"}

    header = f"{'Scenario':>22} " + " ".join(f"{field_short[f]:>7}" for f in FIELDS) + f" {'Whole':>7}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    spectra = {}
    for scenario in scenarios:
        vecs, _ = generate_traffic(client, np.random.default_rng(42), scenario, count=50)
        spectrum = compute_field_spectrum(client, vecs)
        whole = coherence(vecs)
        spectra[scenario] = spectrum

        row = f"  {scenario:>22} "
        row += " ".join(f"{spectrum[f]:>7.4f}" for f in FIELDS)
        row += f" {whole:>7.4f}"
        print(row)

    print("\n" + "=" * 90)
    print("EXPERIMENT 2: Attack-Type Classification from Spectrum Shape")
    print("=" * 90)
    print()
    print("Can we distinguish attacks by WHICH fields have high coherence?")
    print()

    # Build expected signatures (which fields should be high for each attack)
    expected_high = {
        "dns_amplification": {"src_ip", "proto", "src_port", "ttl"},
        "syn_flood": {"dst_ip", "proto", "dst_port", "pkt_len"},
        "ntp_amplification": {"proto", "src_port", "pkt_len", "ttl"},
        "ssdp_amplification": {"proto", "src_port"},
        "icmp_flood": {"proto", "src_port", "dst_port"},
        "port_scan": {"src_ip", "dst_ip", "proto", "pkt_len"},
    }

    normal_spectrum = spectra["normal"]
    normal_max = max(normal_spectrum.values())
    threshold = normal_max * 1.5

    print(f"  Threshold: {threshold:.4f} (1.5× max normal field coherence)")
    print()

    for scenario in scenarios[1:]:  # skip normal
        spectrum = spectra[scenario]
        detected_fields = {f for f in FIELDS if spectrum[f] > threshold}
        expected = expected_high.get(scenario, set())

        overlap = detected_fields & expected
        missed = expected - detected_fields
        extra = detected_fields - expected

        print(f"  {scenario}:")
        print(f"    Detected high fields: {', '.join(sorted(detected_fields)) or 'none'}")
        print(f"    Expected high fields: {', '.join(sorted(expected))}")
        print(f"    Overlap: {len(overlap)}/{len(expected)} ({len(overlap)/max(len(expected), 1)*100:.0f}%)")
        if missed:
            print(f"    Missed: {', '.join(sorted(missed))}")
        if extra:
            print(f"    Extra:  {', '.join(sorted(extra))}")
        print()

    print("=" * 90)
    print("EXPERIMENT 3: Mixed traffic — can spectra detect attack component?")
    print("=" * 90)
    print()
    print("20% attack traffic mixed with 80% normal. Which fields spike?")
    print()

    for attack_type in ["dns_amplification", "syn_flood", "ntp_amplification"]:
        rng = np.random.default_rng(42)
        normal_vecs, _ = generate_traffic(client, rng, "normal", count=40)
        attack_vecs, _ = generate_traffic(client, rng, attack_type, count=10)
        mixed = normal_vecs + attack_vecs
        rng.shuffle(mixed)

        spectrum = compute_field_spectrum(client, mixed)
        whole = coherence(mixed)

        print(f"  {attack_type} (20% attack, 80% normal):")
        for f in FIELDS:
            marker = " ←" if spectrum[f] > threshold else ""
            print(f"    {field_short[f]:>7}: {spectrum[f]:.4f}{marker}")
        print(f"    {'Whole':>7}: {whole:.4f}")
        print()

    print("=" * 90)
    print("EXPERIMENT 4: Stability across trials")
    print("=" * 90)
    print()
    print("20 trials of DNS amp, measuring per-field coherence variance.")
    print()

    field_trials = {f: [] for f in FIELDS}
    for trial in range(20):
        vecs, _ = generate_traffic(
            client, np.random.default_rng(trial + 3000), "dns_amplification", count=50
        )
        spectrum = compute_field_spectrum(client, vecs)
        for f in FIELDS:
            field_trials[f].append(spectrum[f])

    print(f"  {'Field':>7} {'Mean':>7} {'Std':>7} {'CV':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-' * 42}")
    for f in FIELDS:
        vals = field_trials[f]
        mean = np.mean(vals)
        std = np.std(vals)
        cv = std / mean if mean > 0 else 0
        print(f"  {field_short[f]:>7} {mean:>7.4f} {std:>7.4f} {cv:>6.1%} {np.min(vals):>7.4f} {np.max(vals):>7.4f}")

    print()


if __name__ == "__main__":
    main()
