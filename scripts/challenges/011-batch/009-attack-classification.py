#!/usr/bin/env python3
"""
Challenge 011-009: Attack Classification from Culprit Patterns

Maps detected culprit patterns to specific attack types automatically.

Attack Classification Strategy:
1. Build a signature library of attack "fingerprints" using VSA
2. When anomaly detected, compare culprit pattern to known attack signatures
3. Report most likely attack type with confidence

Attack Types to Classify:
- SYN Flood: high rate, TCP SYN only, low payload
- UDP Flood: high rate, UDP, various ports, small/fixed payloads
- ICMP Flood: high rate, ICMP echo, spoofed sources
- DNS Amplification: UDP 53 as source (reflection), large payloads
- NTP Amplification: UDP 123 as source (reflection)
- Slowloris: TCP, valid ports, minimal data rate
- Port Scan: many dst_ports, low packet count per port
- IP Sweep: many dst_ips, same port pattern
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import numpy as np

from scapy.all import IP, TCP, UDP, ICMP, Raw, Packet

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager

# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# ATTACK SIGNATURES
# =============================================================================

@dataclass
class AttackSignature:
    """Defines the fingerprint of an attack type."""
    name: str
    description: str

    # Required indicators (must match)
    required_culprits: List[str] = field(default_factory=list)  # e.g., ["high:dst_port", "high:flags"]

    # Protocol expectations
    expected_protocol: Optional[str] = None  # "tcp", "udp", "icmp", None=any

    # Pattern characteristics
    expect_reflection: bool = False  # src_port is well-known (< 1024)
    expect_high_src_cardinality: bool = False  # many source IPs (spoofed)
    expect_high_dst_cardinality: bool = False  # many destinations (spray)
    expect_low_dst_port_cardinality: bool = False  # single target port
    expect_high_dst_port_cardinality: bool = False  # port scan
    expect_small_payload: bool = False  # minimal payload
    expect_large_payload: bool = False  # amplification

    # Vector signature (learned from examples)
    signature_vector: Optional[np.ndarray] = None

    # Scoring weights
    base_confidence: float = 0.5


class AttackClassifier:
    """
    Classifies detected anomalies into specific attack types.

    Uses a combination of:
    1. Rule-based matching (culprit patterns)
    2. Vector similarity to learned attack signatures
    """

    def __init__(self, vm: DeterministicVectorManager):
        self.vm = vm
        self.signatures = self._build_signatures()

    def _build_signatures(self) -> Dict[str, AttackSignature]:
        """Build attack signature library."""
        return {
            "syn_flood": AttackSignature(
                name="SYN Flood",
                description="TCP SYN packets flooding target",
                required_culprits=["flags"],  # Unusual flag pattern (SYN only)
                expected_protocol="tcp",
                expect_high_src_cardinality=True,
                expect_low_dst_port_cardinality=True,
                expect_small_payload=True,
                signature_vector=self._encode_attack_pattern([
                    "proto:tcp", "flags:2", "payload_size:0", "high_rate", "many_sources"
                ]),
            ),

            "udp_flood": AttackSignature(
                name="UDP Flood",
                description="UDP packets flooding target",
                expected_protocol="udp",
                expect_high_src_cardinality=True,
                expect_low_dst_port_cardinality=True,
                signature_vector=self._encode_attack_pattern([
                    "proto:udp", "high_rate", "many_sources", "fixed_port"
                ]),
            ),

            "icmp_flood": AttackSignature(
                name="ICMP Flood",
                description="ICMP echo requests flooding target",
                expected_protocol="icmp",
                expect_high_src_cardinality=True,
                signature_vector=self._encode_attack_pattern([
                    "proto:icmp", "type:8", "high_rate", "many_sources"
                ]),
            ),

            "dns_amplification": AttackSignature(
                name="DNS Amplification",
                description="DNS reflection attack (src_port=53)",
                required_culprits=["src_port"],  # Well-known port as source
                expected_protocol="udp",
                expect_reflection=True,
                expect_large_payload=True,
                signature_vector=self._encode_attack_pattern([
                    "proto:udp", "src_port:53", "reflection", "large_payload"
                ]),
            ),

            "ntp_amplification": AttackSignature(
                name="NTP Amplification",
                description="NTP reflection attack (src_port=123)",
                required_culprits=["src_port"],
                expected_protocol="udp",
                expect_reflection=True,
                expect_large_payload=True,
                signature_vector=self._encode_attack_pattern([
                    "proto:udp", "src_port:123", "reflection", "large_payload"
                ]),
            ),

            "port_scan": AttackSignature(
                name="Port Scan",
                description="Reconnaissance scanning many ports",
                required_culprits=["dst_port"],  # Many unusual ports
                expect_high_dst_port_cardinality=True,
                expect_small_payload=True,
                signature_vector=self._encode_attack_pattern([
                    "many_dst_ports", "low_packets_per_port", "reconnaissance"
                ]),
            ),

            "ip_sweep": AttackSignature(
                name="IP Sweep",
                description="Reconnaissance scanning IP range",
                expect_high_dst_cardinality=True,
                expect_low_dst_port_cardinality=True,
                expect_small_payload=True,
                signature_vector=self._encode_attack_pattern([
                    "many_dst_ips", "same_port", "reconnaissance"
                ]),
            ),
        }

    def _encode_attack_pattern(self, indicators: List[str]) -> np.ndarray:
        """Encode attack indicators into a signature vector."""
        acc = np.zeros(DIMENSIONS, dtype=np.float64)
        for indicator in indicators:
            vec = self.vm.get_vector(f"attack_indicator:{indicator}")
            acc += vec
        # Normalize
        norm = np.linalg.norm(acc)
        if norm > 1e-10:
            acc = acc / norm
        return acc

    def encode_observed_pattern(
        self,
        culprits: List[dict],  # List of culprit dicts with field, severity, value
        protocol: Optional[str],
        src_port: Optional[int],
        payload_size: int,
        src_cardinality: float,
        dst_cardinality: float,
        dst_port_cardinality: float,
    ) -> np.ndarray:
        """Encode observed anomaly into a pattern vector."""
        indicators = []

        # Add culprit-based indicators
        for c in culprits:
            indicators.append(f"culprit:{c['severity']}:{c['field']}")
            if c.get('observed_value'):
                indicators.append(f"value:{c['field']}:{c['observed_value']}")

        # Protocol
        if protocol:
            indicators.append(f"proto:{protocol}")

        # Reflection detection
        if src_port and src_port < 1024:
            indicators.append("reflection")
            indicators.append(f"src_port:{src_port}")

        # Payload characteristics
        if payload_size == 0:
            indicators.append("payload_size:0")
        elif payload_size < 64:
            indicators.append("small_payload")
        elif payload_size > 512:
            indicators.append("large_payload")

        # Cardinality patterns
        if src_cardinality > 50:
            indicators.append("many_sources")
            indicators.append("high_rate")
        if dst_cardinality > 10:
            indicators.append("many_dst_ips")
        if dst_port_cardinality > 20:
            indicators.append("many_dst_ports")
        elif dst_port_cardinality < 3:
            indicators.append("fixed_port")

        return self._encode_attack_pattern(indicators)

    def classify(
        self,
        culprits: List[dict],
        protocol: Optional[str],
        src_port: Optional[int],
        payload_size: int,
        src_cardinality: float = 1.0,
        dst_cardinality: float = 1.0,
        dst_port_cardinality: float = 1.0,
    ) -> List[Tuple[str, float, str]]:
        """
        Classify an anomaly into attack types.

        Returns: List of (attack_name, confidence, description) sorted by confidence.
        """
        # Encode the observed pattern
        observed_vec = self.encode_observed_pattern(
            culprits, protocol, src_port, payload_size,
            src_cardinality, dst_cardinality, dst_port_cardinality
        )

        results = []

        for attack_id, sig in self.signatures.items():
            confidence = sig.base_confidence

            # Protocol match
            if sig.expected_protocol:
                if protocol == sig.expected_protocol:
                    confidence += 0.1
                else:
                    confidence -= 0.2

            # Reflection check
            if sig.expect_reflection:
                if src_port and src_port < 1024:
                    confidence += 0.2
                else:
                    confidence -= 0.3

            # Required culprits
            culprit_fields = {c['field'] for c in culprits}
            for req in sig.required_culprits:
                if req in culprit_fields:
                    confidence += 0.15

            # Cardinality patterns
            if sig.expect_high_src_cardinality and src_cardinality > 20:
                confidence += 0.1
            if sig.expect_high_dst_cardinality and dst_cardinality > 5:
                confidence += 0.1
            if sig.expect_high_dst_port_cardinality and dst_port_cardinality > 10:
                confidence += 0.15
            if sig.expect_low_dst_port_cardinality and dst_port_cardinality < 3:
                confidence += 0.1

            # Payload size
            if sig.expect_small_payload and payload_size < 64:
                confidence += 0.1
            if sig.expect_large_payload and payload_size > 256:
                confidence += 0.1

            # Vector similarity
            if sig.signature_vector is not None:
                sim = cosine_similarity(observed_vec, sig.signature_vector)
                confidence += sim * 0.3  # Weight vector match

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            if confidence > 0.3:  # Threshold to report
                results.append((sig.name, confidence, sig.description))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# =============================================================================
# PACKET GENERATORS FOR TESTING
# =============================================================================

def generate_syn_flood(n: int = 100) -> List[Packet]:
    """Generate SYN flood packets."""
    packets = []
    for i in range(n):
        src_ip = f"10.{(i // 65536) % 256}.{(i // 256) % 256}.{i % 256}"
        pkt = IP(src=src_ip, dst="192.168.1.100") / TCP(
            sport=40000 + (i % 20000),
            dport=80,
            flags="S"  # SYN only
        )
        packets.append(pkt)
    return packets


def generate_udp_flood(n: int = 100) -> List[Packet]:
    """Generate UDP flood packets."""
    packets = []
    for i in range(n):
        src_ip = f"10.{(i // 65536) % 256}.{(i // 256) % 256}.{i % 256}"
        pkt = IP(src=src_ip, dst="192.168.1.100") / UDP(
            sport=40000 + (i % 20000),
            dport=53
        ) / Raw(load=b"A" * 64)
        packets.append(pkt)
    return packets


def generate_dns_reflection(n: int = 100) -> List[Packet]:
    """Generate DNS amplification packets (reflected responses)."""
    packets = []
    for i in range(n):
        src_ip = f"8.8.{(i % 4)}.{(i % 256)}"  # "DNS servers"
        pkt = IP(src=src_ip, dst="192.168.1.100") / UDP(
            sport=53,  # Reflection indicator!
            dport=40000 + (i % 1000)
        ) / Raw(load=b"X" * 512)  # Large response
        packets.append(pkt)
    return packets


def generate_port_scan(n: int = 100) -> List[Packet]:
    """Generate port scan packets."""
    packets = []
    for i in range(n):
        pkt = IP(src="10.0.0.5", dst="192.168.1.100") / TCP(
            sport=40000,
            dport=1 + i,  # Scanning ports 1-100
            flags="S"
        )
        packets.append(pkt)
    return packets


def generate_ip_sweep(n: int = 100) -> List[Packet]:
    """Generate IP sweep packets."""
    packets = []
    for i in range(n):
        dst_ip = f"192.168.1.{i % 256}"
        pkt = IP(src="10.0.0.5", dst=dst_ip) / ICMP(type=8)
        packets.append(pkt)
    return packets


def generate_normal_traffic(n: int = 100) -> List[Packet]:
    """Generate normal-looking traffic."""
    import random
    packets = []
    for i in range(n):
        pkt_type = random.choice(["http", "https", "dns_query"])
        if pkt_type == "http":
            pkt = IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
                sport=random.randint(49152, 65535),
                dport=80,
                flags="PA"
            ) / Raw(load=b"GET / HTTP/1.1\r\n")
        elif pkt_type == "https":
            pkt = IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
                sport=random.randint(49152, 65535),
                dport=443,
                flags="PA"
            ) / Raw(load=b"\x16\x03\x01" + b"X" * 100)
        else:  # DNS query
            pkt = IP(src="192.168.1.50", dst="8.8.8.8") / UDP(
                sport=random.randint(49152, 65535),
                dport=53
            ) / Raw(load=b"\x00\x01\x00\x00" + b"example" + b"\x03com\x00")
        packets.append(pkt)
    return packets


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 80)
    print("CHALLENGE 011-009: ATTACK CLASSIFICATION")
    print("=" * 80)
    print()

    # Initialize
    vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
    classifier = AttackClassifier(vm)

    # Test scenarios
    scenarios = [
        ("SYN Flood", generate_syn_flood, {"protocol": "tcp", "flags_pattern": "SYN only"}),
        ("UDP Flood", generate_udp_flood, {"protocol": "udp"}),
        ("DNS Amplification", generate_dns_reflection, {"protocol": "udp", "reflection": True}),
        ("Port Scan", generate_port_scan, {"protocol": "tcp", "many_ports": True}),
        ("IP Sweep", generate_ip_sweep, {"protocol": "icmp", "many_ips": True}),
        ("Normal Traffic", generate_normal_traffic, {}),
    ]

    results_summary = []

    for scenario_name, generator, expected in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print("=" * 60)

        packets = generator(100)

        # Analyze packet characteristics
        src_ips = set()
        dst_ips = set()
        dst_ports = set()
        src_ports = set()
        protocols = Counter()
        payloads = []

        for pkt in packets:
            if IP in pkt:
                src_ips.add(pkt[IP].src)
                dst_ips.add(pkt[IP].dst)
            if TCP in pkt:
                protocols["tcp"] += 1
                src_ports.add(pkt[TCP].sport)
                dst_ports.add(pkt[TCP].dport)
            elif UDP in pkt:
                protocols["udp"] += 1
                src_ports.add(pkt[UDP].sport)
                dst_ports.add(pkt[UDP].dport)
            elif ICMP in pkt:
                protocols["icmp"] += 1
            if Raw in pkt:
                payloads.append(len(pkt[Raw].load))

        # Get dominant protocol
        if protocols:
            protocol = protocols.most_common(1)[0][0]
        else:
            protocol = None

        # Get common src_port (for reflection detection)
        src_port = None
        if src_ports:
            port_counts = Counter()
            for pkt in packets:
                if TCP in pkt:
                    port_counts[pkt[TCP].sport] += 1
                elif UDP in pkt:
                    port_counts[pkt[UDP].sport] += 1
            if port_counts:
                most_common_port, count = port_counts.most_common(1)[0]
                if count > len(packets) * 0.5:  # Dominant port
                    src_port = most_common_port

        avg_payload = np.mean(payloads) if payloads else 0

        # Build culprits based on observations
        culprits = []
        if len(src_ips) > 20:
            culprits.append({"field": "src_prefix", "severity": "medium", "observed_value": "many"})
        if len(dst_ports) > 20:
            culprits.append({"field": "dst_port", "severity": "high", "observed_value": "many"})
        if src_port and src_port < 1024:
            culprits.append({"field": "src_port", "severity": "high", "observed_value": str(src_port)})

        print(f"\nPacket Characteristics:")
        print(f"  Protocol: {protocol}")
        print(f"  Unique src IPs: {len(src_ips)}")
        print(f"  Unique dst IPs: {len(dst_ips)}")
        print(f"  Unique dst ports: {len(dst_ports)}")
        print(f"  Avg payload size: {avg_payload:.1f} bytes")
        print(f"  Culprits identified: {len(culprits)}")

        # Classify
        classifications = classifier.classify(
            culprits=culprits,
            protocol=protocol,
            src_port=src_port,
            payload_size=int(avg_payload),
            src_cardinality=len(src_ips),
            dst_cardinality=len(dst_ips),
            dst_port_cardinality=len(dst_ports),
        )

        print(f"\nClassification Results:")
        if classifications:
            for attack_name, confidence, description in classifications[:3]:
                print(f"  [{confidence:.0%}] {attack_name}: {description}")
            top_classification = classifications[0][0]
        else:
            print("  No attack patterns matched (likely normal traffic)")
            top_classification = "Normal"

        # Check if classification matches expected
        is_correct = (
            scenario_name == "Normal Traffic" and top_classification == "Normal"
        ) or (
            scenario_name != "Normal Traffic" and top_classification in scenario_name
        )

        results_summary.append({
            "scenario": scenario_name,
            "top_classification": top_classification,
            "correct": is_correct,
            "confidence": classifications[0][1] if classifications else 0.0,
        })

    # Summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    correct = sum(1 for r in results_summary if r["correct"])
    total = len(results_summary)

    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    print()
    for r in results_summary:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['scenario']:20s} → {r['top_classification']:20s} ({r['confidence']:.0%})")

    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print("""
1. Attack classification uses a hybrid approach:
   - Rule-based matching for clear indicators (reflection ports, cardinality)
   - Vector similarity for learned attack signatures

2. Key differentiators between attacks:
   - Reflection attacks: well-known port as SOURCE
   - Flood attacks: high source IP cardinality
   - Scans: high destination port/IP cardinality
   - Payload size helps distinguish amplification vs floods

3. The VSA approach enables:
   - Fast signature matching (single vector comparison)
   - Composable signatures (bundle multiple indicators)
   - Fuzzy matching (partial pattern matches still score)
""")


if __name__ == "__main__":
    main()
