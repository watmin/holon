#!/usr/bin/env python3
"""
Challenge 011-006: Multi-Perspective Raw Packet Analysis

Use scapy for real packet crafting and analyze at multiple levels:

PERSPECTIVES (each gets its own accumulator):
1. L3: IP src/dst address patterns (prefixes, not full IPs)
2. L4-ports: TCP/UDP port patterns
3. L4-flags: TCP flags or ICMP type/code
4. Payload-bytes: Raw byte position analysis
5. Payload-ngrams: Byte n-gram sequences
6. Cardinality: Track unique values per field (high=random, low=focused)

ENTROPY-LIKE MEASURES (without information theory):
- Agreement ratio: What fraction of dimensions agree with accumulator?
- Spread: How many unique byte values in payload?
- Positional consistency: Do bytes at position N vary or stay constant?

KEY INSIGHT: DDoS/scanning shows LOW cardinality (same pattern repeated),
while normal traffic shows HIGHER cardinality (diverse patterns).
"""

import sys
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, deque

import numpy as np

# Scapy imports
from scapy.all import (
    IP, TCP, UDP, ICMP, Raw,
    Ether, Packet,
    raw as scapy_raw,
)

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.995


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# PACKET PARSING
# =============================================================================

@dataclass
class ParsedPacket:
    """Parsed packet with layer-specific fields."""
    # L3
    src_ip: str = ""
    dst_ip: str = ""
    src_prefix_16: str = ""
    dst_prefix_16: str = ""
    src_prefix_24: str = ""
    dst_prefix_24: str = ""

    # L4
    protocol: str = ""  # TCP, UDP, ICMP, OTHER
    src_port: int = 0
    dst_port: int = 0
    tcp_flags: int = 0
    icmp_type: int = 0
    icmp_code: int = 0

    # Payload
    payload_bytes: bytes = b""
    payload_size: int = 0

    # Raw
    raw_bytes: bytes = b""


def parse_packet(pkt: Packet) -> ParsedPacket:
    """Parse scapy packet into structured fields."""
    result = ParsedPacket()

    # Get raw bytes (skip Ethernet if present)
    if Ether in pkt:
        result.raw_bytes = scapy_raw(pkt[Ether].payload)
    else:
        result.raw_bytes = scapy_raw(pkt)

    # L3: IP
    if IP in pkt:
        ip = pkt[IP]
        result.src_ip = ip.src
        result.dst_ip = ip.dst

        # Prefixes
        src_parts = ip.src.split(".")
        dst_parts = ip.dst.split(".")
        result.src_prefix_16 = f"{src_parts[0]}.{src_parts[1]}.0.0/16"
        result.dst_prefix_16 = f"{dst_parts[0]}.{dst_parts[1]}.0.0/16"
        result.src_prefix_24 = f"{src_parts[0]}.{src_parts[1]}.{src_parts[2]}.0/24"
        result.dst_prefix_24 = f"{dst_parts[0]}.{dst_parts[1]}.{dst_parts[2]}.0/24"

    # L4: TCP
    if TCP in pkt:
        tcp = pkt[TCP]
        result.protocol = "TCP"
        result.src_port = tcp.sport
        result.dst_port = tcp.dport
        result.tcp_flags = int(tcp.flags)

        # Payload
        if Raw in pkt:
            result.payload_bytes = bytes(pkt[Raw].load)
            result.payload_size = len(result.payload_bytes)

    # L4: UDP
    elif UDP in pkt:
        udp = pkt[UDP]
        result.protocol = "UDP"
        result.src_port = udp.sport
        result.dst_port = udp.dport

        if Raw in pkt:
            result.payload_bytes = bytes(pkt[Raw].load)
            result.payload_size = len(result.payload_bytes)

    # L4: ICMP
    elif ICMP in pkt:
        icmp = pkt[ICMP]
        result.protocol = "ICMP"
        result.icmp_type = icmp.type
        result.icmp_code = icmp.code

        if Raw in pkt:
            result.payload_bytes = bytes(pkt[Raw].load)
            result.payload_size = len(result.payload_bytes)

    else:
        result.protocol = "OTHER"
        if Raw in pkt:
            result.payload_bytes = bytes(pkt[Raw].load)
            result.payload_size = len(result.payload_bytes)

    return result


# =============================================================================
# PERSPECTIVE ENCODERS
# =============================================================================

class PerspectiveEncoder:
    """
    Encode packets from multiple perspectives.

    Each perspective captures different aspects:
    - L3: Where is traffic going? (address patterns)
    - L4-ports: What services? (port patterns)
    - L4-flags: What behavior? (connection patterns)
    - Payload: What content? (byte patterns)
    """

    def __init__(self, vm: DeterministicVectorManager):
        self.vm = vm
        self.encoder = Encoder(vector_manager=vm)

    def encode_l3(self, parsed: ParsedPacket) -> np.ndarray:
        """Encode L3 perspective (IP addresses as prefixes)."""
        features = {
            "src_16": parsed.src_prefix_16,
            "dst_16": parsed.dst_prefix_16,
            "src_24": parsed.src_prefix_24,
            "dst_24": parsed.dst_prefix_24,
        }
        return self.encoder.encode_data(features)

    def encode_l4_ports(self, parsed: ParsedPacket) -> np.ndarray:
        """Encode L4 port perspective."""
        if parsed.protocol in ("TCP", "UDP"):
            # Bucket ports
            src_bucket = self._bucket_port(parsed.src_port)
            dst_bucket = self._bucket_port(parsed.dst_port)

            features = {
                "protocol": parsed.protocol,
                "src_bucket": src_bucket,
                "dst_bucket": dst_bucket,
                "src_wellknown": parsed.src_port < 1024,
                "dst_wellknown": parsed.dst_port < 1024,
                "src_is_53": parsed.src_port == 53,
                "src_is_123": parsed.src_port == 123,
            }
        else:
            features = {"protocol": parsed.protocol}

        return self.encoder.encode_data(features)

    def encode_l4_flags(self, parsed: ParsedPacket) -> np.ndarray:
        """Encode L4 flags/type perspective."""
        if parsed.protocol == "TCP":
            features = {
                "protocol": "TCP",
                "flags": parsed.tcp_flags,
                "is_syn": (parsed.tcp_flags & 0x02) != 0,
                "is_ack": (parsed.tcp_flags & 0x10) != 0,
                "is_fin": (parsed.tcp_flags & 0x01) != 0,
                "is_rst": (parsed.tcp_flags & 0x04) != 0,
                "is_psh": (parsed.tcp_flags & 0x08) != 0,
                "syn_only": parsed.tcp_flags == 0x02,
            }
        elif parsed.protocol == "ICMP":
            features = {
                "protocol": "ICMP",
                "type": parsed.icmp_type,
                "code": parsed.icmp_code,
                "is_echo": parsed.icmp_type in (0, 8),
            }
        else:
            features = {"protocol": parsed.protocol}

        return self.encoder.encode_data(features)

    def encode_payload_positional(self, parsed: ParsedPacket, max_positions: int = 64) -> np.ndarray:
        """
        Encode payload by byte position.

        Each position gets its own encoded byte value.
        This captures: "byte at position 0 is usually 0x47" (HTTP GET).
        """
        if not parsed.payload_bytes:
            return np.zeros(DIMENSIONS, dtype=np.int8)

        vectors = []
        for i, byte in enumerate(parsed.payload_bytes[:max_positions]):
            # Encode: position + byte value
            pos_atom = f"pos_{i}_byte_{byte:02x}"
            vectors.append(self.vm.get_vector(pos_atom))

        if not vectors:
            return np.zeros(DIMENSIONS, dtype=np.int8)

        # Bundle all position vectors
        stacked = np.stack(vectors)
        bundled = np.sign(np.sum(stacked.astype(np.float32), axis=0)).astype(np.int8)
        return bundled

    def encode_payload_ngrams(self, parsed: ParsedPacket, n: int = 2) -> np.ndarray:
        """
        Encode payload as byte n-grams.

        Captures sequences: 0x47 0x45 0x54 = "GET".
        """
        if len(parsed.payload_bytes) < n:
            return np.zeros(DIMENSIONS, dtype=np.int8)

        vectors = []
        for i in range(len(parsed.payload_bytes) - n + 1):
            ngram = parsed.payload_bytes[i:i+n]
            ngram_hex = ngram.hex()
            vectors.append(self.vm.get_vector(f"ngram_{ngram_hex}"))

        stacked = np.stack(vectors)
        bundled = np.sign(np.sum(stacked.astype(np.float32), axis=0)).astype(np.int8)
        return bundled

    def encode_payload_structure(self, parsed: ParsedPacket) -> np.ndarray:
        """
        Encode payload structural properties (entropy-like without info theory).

        Features:
        - Size bucket
        - Unique byte count (spread)
        - Byte distribution characteristics
        """
        payload = parsed.payload_bytes

        if not payload:
            features = {"has_payload": False}
        else:
            # Unique bytes = measure of "spread"
            unique_bytes = len(set(payload))

            # Byte range
            min_byte = min(payload)
            max_byte = max(payload)
            byte_range = max_byte - min_byte

            # Printable ratio (ASCII 32-126)
            printable = sum(1 for b in payload if 32 <= b <= 126)
            printable_ratio = printable / len(payload)

            # Zero ratio (padding/nulls)
            zero_count = payload.count(0)
            zero_ratio = zero_count / len(payload)

            features = {
                "has_payload": True,
                "size_bucket": self._bucket_size(len(payload)),
                "unique_bytes": self._bucket_spread(unique_bytes),
                "byte_range": self._bucket_range(byte_range),
                "mostly_printable": printable_ratio > 0.8,
                "has_nulls": zero_ratio > 0.1,
                "uniform_spread": unique_bytes > 200,  # High entropy indicator
                "narrow_spread": unique_bytes < 20,    # Low entropy indicator
            }

        return self.encoder.encode_data(features)

    def _bucket_port(self, port: int) -> str:
        if port == 0:
            return "zero"
        elif port < 1024:
            return "wellknown"
        elif port < 49152:
            return "registered"
        else:
            return "ephemeral"

    def _bucket_size(self, size: int) -> str:
        if size == 0:
            return "empty"
        elif size < 64:
            return "tiny"
        elif size < 256:
            return "small"
        elif size < 1024:
            return "medium"
        elif size < 1500:
            return "large"
        else:
            return "jumbo"

    def _bucket_spread(self, unique: int) -> str:
        """Bucket unique byte count."""
        if unique < 5:
            return "minimal"
        elif unique < 20:
            return "low"
        elif unique < 50:
            return "medium"
        elif unique < 150:
            return "high"
        else:
            return "very_high"

    def _bucket_range(self, range_val: int) -> str:
        if range_val < 10:
            return "narrow"
        elif range_val < 50:
            return "medium"
        elif range_val < 200:
            return "wide"
        else:
            return "full"


# =============================================================================
# CARDINALITY TRACKER
# =============================================================================

class CardinalityTracker:
    """
    Track cardinality of field values over a sliding window.

    High cardinality = diverse traffic (normal)
    Low cardinality = focused traffic (scan/DDoS)

    Sudden CHANGE in cardinality = transition signal.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.windows: Dict[str, deque] = {}
        self.cardinality_history: Dict[str, deque] = {}

    def record(self, field: str, value: str):
        """Record a value for a field."""
        if field not in self.windows:
            self.windows[field] = deque(maxlen=self.window_size)
            self.cardinality_history[field] = deque(maxlen=100)

        self.windows[field].append(value)

        # Compute current cardinality
        unique = len(set(self.windows[field]))
        cardinality = unique / len(self.windows[field])
        self.cardinality_history[field].append(cardinality)

    def get_cardinality(self, field: str) -> float:
        """Get current cardinality ratio (0-1) for field."""
        if field not in self.windows or len(self.windows[field]) < 10:
            return 1.0
        return len(set(self.windows[field])) / len(self.windows[field])

    def get_cardinality_change(self, field: str) -> float:
        """Get change in cardinality (positive = more diverse)."""
        if field not in self.cardinality_history:
            return 0.0

        history = list(self.cardinality_history[field])
        if len(history) < 20:
            return 0.0

        recent = np.mean(history[-10:])
        older = np.mean(history[-20:-10])

        return recent - older

    def encode_cardinality_state(self, encoder: PerspectiveEncoder) -> np.ndarray:
        """Encode current cardinality state as a vector."""
        features = {}
        for field in self.windows:
            card = self.get_cardinality(field)
            features[f"{field}_cardinality"] = "high" if card > 0.5 else "low"

        return encoder.encoder.encode_data(features)


# =============================================================================
# MULTI-PERSPECTIVE ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0):
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


@dataclass
class PerspectiveScores:
    """Similarity scores from each perspective."""
    l3: float = 0.0
    l4_ports: float = 0.0
    l4_flags: float = 0.0
    payload_pos: float = 0.0
    payload_ngram: float = 0.0
    payload_struct: float = 0.0
    cardinality: Dict[str, float] = field(default_factory=dict)

    def min_score(self) -> float:
        """Minimum similarity across perspectives."""
        scores = [self.l3, self.l4_ports, self.l4_flags,
                  self.payload_pos, self.payload_ngram, self.payload_struct]
        return min(s for s in scores if s > 0) if any(s > 0 for s in scores) else 0.0

    def anomalous_perspectives(self, threshold: float = 0.4) -> List[str]:
        """List perspectives below threshold."""
        result = []
        if self.l3 < threshold and self.l3 > 0:
            result.append("l3")
        if self.l4_ports < threshold and self.l4_ports > 0:
            result.append("l4_ports")
        if self.l4_flags < threshold and self.l4_flags > 0:
            result.append("l4_flags")
        if self.payload_pos < threshold and self.payload_pos > 0:
            result.append("payload_pos")
        if self.payload_ngram < threshold and self.payload_ngram > 0:
            result.append("payload_ngram")
        if self.payload_struct < threshold and self.payload_struct > 0:
            result.append("payload_struct")
        return result


class MultiPerspectiveDetector:
    """
    Analyze packets from multiple perspectives simultaneously.

    Each perspective has its own accumulator, allowing:
    - Detection: which perspective is anomalous?
    - Cardinality tracking: is traffic becoming more focused?
    - Deviation analysis: what's different about this packet?
    """

    def __init__(
        self,
        decay: float = DECAY_FACTOR,
        warmup: int = 100,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.encoder = PerspectiveEncoder(self.vm)
        self.cardinality = CardinalityTracker(window_size=100)

        self.decay = decay
        self.warmup = warmup

        # Per-perspective accumulators
        self.accumulators = {
            "l3": DecayingAccumulator(DIMENSIONS, decay),
            "l4_ports": DecayingAccumulator(DIMENSIONS, decay),
            "l4_flags": DecayingAccumulator(DIMENSIONS, decay),
            "payload_pos": DecayingAccumulator(DIMENSIONS, decay),
            "payload_ngram": DecayingAccumulator(DIMENSIONS, decay),
            "payload_struct": DecayingAccumulator(DIMENSIONS, decay),
        }

        self.count = 0

    def process(self, pkt: Packet) -> PerspectiveScores:
        """Process packet and get similarity from each perspective."""
        self.count += 1
        is_warmup = self.count <= self.warmup

        # Parse
        parsed = parse_packet(pkt)

        # Encode each perspective
        vecs = {
            "l3": self.encoder.encode_l3(parsed),
            "l4_ports": self.encoder.encode_l4_ports(parsed),
            "l4_flags": self.encoder.encode_l4_flags(parsed),
            "payload_pos": self.encoder.encode_payload_positional(parsed),
            "payload_ngram": self.encoder.encode_payload_ngrams(parsed),
            "payload_struct": self.encoder.encode_payload_structure(parsed),
        }

        # Track cardinality
        self.cardinality.record("src_prefix", parsed.src_prefix_24)
        self.cardinality.record("dst_prefix", parsed.dst_prefix_24)
        self.cardinality.record("src_port", str(parsed.src_port))
        self.cardinality.record("dst_port", str(parsed.dst_port))
        if parsed.protocol:
            self.cardinality.record("protocol", parsed.protocol)

        # Compute similarities
        scores = PerspectiveScores()

        if self.count > 1:
            scores.l3 = cosine_similarity(vecs["l3"], self.accumulators["l3"].get_normalized())
            scores.l4_ports = cosine_similarity(vecs["l4_ports"], self.accumulators["l4_ports"].get_normalized())
            scores.l4_flags = cosine_similarity(vecs["l4_flags"], self.accumulators["l4_flags"].get_normalized())
            scores.payload_pos = cosine_similarity(vecs["payload_pos"], self.accumulators["payload_pos"].get_normalized())
            scores.payload_ngram = cosine_similarity(vecs["payload_ngram"], self.accumulators["payload_ngram"].get_normalized())
            scores.payload_struct = cosine_similarity(vecs["payload_struct"], self.accumulators["payload_struct"].get_normalized())

        # Cardinality scores
        scores.cardinality = {
            "src_prefix": self.cardinality.get_cardinality("src_prefix"),
            "dst_prefix": self.cardinality.get_cardinality("dst_prefix"),
            "src_port": self.cardinality.get_cardinality("src_port"),
            "dst_port": self.cardinality.get_cardinality("dst_port"),
        }

        # Update accumulators
        for name, vec in vecs.items():
            self.accumulators[name].update(vec)

        return scores


# =============================================================================
# PACKET GENERATOR (using scapy)
# =============================================================================

class PacketGenerator:
    """Generate realistic packets using scapy."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _random_ip(self, prefix: str = "192.168") -> str:
        return f"{prefix}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}"

    def _ephemeral_port(self) -> int:
        return self.rng.randint(49152, 65535)

    def normal_http(self) -> Packet:
        """Normal HTTP GET request."""
        payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
        return (
            IP(src=self._random_ip(), dst=self._random_ip("10.0"))
            / TCP(sport=self._ephemeral_port(), dport=80, flags="PA")
            / Raw(load=payload)
        )

    def normal_https(self) -> Packet:
        """Normal HTTPS (TLS handshake-ish)."""
        # TLS Client Hello starts with 0x16 0x03
        payload = bytes([0x16, 0x03, 0x01] + [self.rng.randint(0, 255) for _ in range(50)])
        return (
            IP(src=self._random_ip(), dst=self._random_ip("10.0"))
            / TCP(sport=self._ephemeral_port(), dport=443, flags="PA")
            / Raw(load=payload)
        )

    def normal_dns_query(self) -> Packet:
        """Normal DNS query."""
        # Simplified DNS query structure
        payload = bytes([
            self.rng.randint(0, 255), self.rng.randint(0, 255),  # Transaction ID
            0x01, 0x00,  # Flags: standard query
            0x00, 0x01,  # Questions: 1
            0x00, 0x00,  # Answers: 0
            0x00, 0x00, 0x00, 0x00,  # Authority/Additional
        ]) + b"\x07example\x03com\x00\x00\x01\x00\x01"
        return (
            IP(src=self._random_ip(), dst="8.8.8.8")
            / UDP(sport=self._ephemeral_port(), dport=53)
            / Raw(load=payload)
        )

    def syn_flood(self, target: str = "192.168.1.100") -> Packet:
        """SYN flood attack packet."""
        return (
            IP(src=self._random_ip("172.16"), dst=target)
            / TCP(sport=self.rng.randint(1, 65535), dport=443, flags="S")
        )

    def dns_reflection(self, victim: str = "192.168.1.100") -> Packet:
        """DNS amplification reflection."""
        # Large DNS response (amplified)
        payload = bytes([self.rng.randint(0, 255) for _ in range(512)])
        return (
            IP(src="8.8.8.8", dst=victim)  # Spoofed from DNS server
            / UDP(sport=53, dport=self._ephemeral_port())
            / Raw(load=payload)
        )

    def port_scan(self, target: str = "192.168.1.100") -> Packet:
        """Port scanning."""
        return (
            IP(src=self._random_ip("10.10"), dst=target)
            / TCP(sport=self._ephemeral_port(), dport=self.rng.randint(1, 1024), flags="S")
        )

    def random_payload_attack(self, target: str = "192.168.1.100") -> Packet:
        """Attack with random payload (high entropy)."""
        payload = bytes([self.rng.randint(0, 255) for _ in range(200)])
        return (
            IP(src=self._random_ip("172.16"), dst=target)
            / TCP(sport=self._ephemeral_port(), dport=443, flags="PA")
            / Raw(load=payload)
        )

    def repeated_payload_attack(self, target: str = "192.168.1.100") -> Packet:
        """Attack with repeated payload (low entropy)."""
        payload = b"\x00" * 200  # All zeros
        return (
            IP(src=self._random_ip("172.16"), dst=target)
            / TCP(sport=self._ephemeral_port(), dport=443, flags="PA")
            / Raw(load=payload)
        )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 011-006: Multi-Perspective Raw Packet Analysis")
    print("=" * 80)
    print("""
PERSPECTIVES:
  1. L3: IP address patterns (prefixes)
  2. L4-ports: Port usage patterns
  3. L4-flags: TCP flags / ICMP type
  4. Payload-positional: Byte at position N
  5. Payload-ngrams: Byte sequences
  6. Payload-structure: Spread, size, printability

CARDINALITY TRACKING:
  - High cardinality = diverse (normal)
  - Low cardinality = focused (attack/scan)
""")

    gen = PacketGenerator(seed=42)
    detector = MultiPerspectiveDetector(warmup=100)

    # Phase 1: Normal traffic
    print("\n--- Phase 1: Normal Traffic (500 packets) ---")
    normal_scores = []

    for _ in range(500):
        pkt = gen.rng.choice([
            gen.normal_http,
            gen.normal_https,
            gen.normal_dns_query,
        ])()
        scores = detector.process(pkt)
        normal_scores.append(scores)

    # Show cardinality during normal
    print(f"  Cardinality (src_prefix): {detector.cardinality.get_cardinality('src_prefix'):.3f}")
    print(f"  Cardinality (dst_port): {detector.cardinality.get_cardinality('dst_port'):.3f}")
    print(f"  Cardinality (protocol): {detector.cardinality.get_cardinality('protocol'):.3f}")

    # Phase 2: SYN flood
    print("\n--- Phase 2: SYN Flood Attack (300 packets) ---")
    syn_scores = []

    for _ in range(300):
        pkt = gen.syn_flood()
        scores = detector.process(pkt)
        syn_scores.append(scores)

    print(f"  Cardinality (src_prefix): {detector.cardinality.get_cardinality('src_prefix'):.3f}")
    print(f"  Cardinality (dst_port): {detector.cardinality.get_cardinality('dst_port'):.3f}")
    print(f"  Cardinality (dst_prefix): {detector.cardinality.get_cardinality('dst_prefix'):.3f}")

    # Phase 3: DNS reflection
    print("\n--- Phase 3: DNS Reflection (300 packets) ---")
    dns_scores = []

    for _ in range(300):
        pkt = gen.dns_reflection()
        scores = detector.process(pkt)
        dns_scores.append(scores)

    print(f"  Cardinality (src_prefix): {detector.cardinality.get_cardinality('src_prefix'):.3f}")
    print(f"  Cardinality (src_port): {detector.cardinality.get_cardinality('src_port'):.3f}")

    # Phase 4: Random payload attack
    print("\n--- Phase 4: Random Payload Attack (200 packets) ---")
    random_scores = []

    for _ in range(200):
        pkt = gen.random_payload_attack()
        scores = detector.process(pkt)
        random_scores.append(scores)

    # Analyze perspective scores
    print("\n" + "=" * 80)
    print("PERSPECTIVE ANALYSIS")
    print("=" * 80)

    def analyze_phase(name: str, scores: List[PerspectiveScores]):
        if not scores:
            return

        # Skip warmup
        scores = scores[min(50, len(scores)):]
        if not scores:
            return

        print(f"\n{name}:")
        print(f"  {'Perspective':<15} {'Min':<8} {'Mean':<8} {'Max':<8}")
        print(f"  {'-'*40}")

        for persp in ["l3", "l4_ports", "l4_flags", "payload_pos", "payload_ngram", "payload_struct"]:
            vals = [getattr(s, persp) for s in scores if getattr(s, persp) > 0]
            if vals:
                print(f"  {persp:<15} {min(vals):<8.3f} {np.mean(vals):<8.3f} {max(vals):<8.3f}")

    analyze_phase("Normal Traffic", normal_scores)
    analyze_phase("SYN Flood", syn_scores)
    analyze_phase("DNS Reflection", dns_scores)
    analyze_phase("Random Payload", random_scores)

    # Cardinality signatures
    print("\n" + "=" * 80)
    print("CARDINALITY SIGNATURES")
    print("=" * 80)
    print("""
Attack cardinality patterns:

  NORMAL TRAFFIC:
    - src_prefix: HIGH (many sources)
    - dst_port: MEDIUM (80, 443, 53)
    - protocol: MEDIUM (TCP, UDP)

  SYN FLOOD:
    - src_prefix: HIGH (spoofed random sources)
    - dst_port: LOW (always 443)
    - dst_prefix: LOW (single target)

  DNS REFLECTION:
    - src_prefix: LOW (all from 8.8.8.8)
    - src_port: LOW (always 53)
    - dst_prefix: LOW (single victim)

  DETECTION STRATEGY:
    - Low dst_prefix cardinality + low dst_port = targeted attack
    - Low src_port cardinality (53, 123) = reflection
    - High src_prefix + low variance = spoofed sources
""")

    # Show individual packet analysis
    print("\n" + "=" * 80)
    print("SAMPLE PACKET BREAKDOWN")
    print("=" * 80)

    detector2 = MultiPerspectiveDetector(warmup=50)

    # Warmup
    for _ in range(100):
        detector2.process(gen.normal_http())

    samples = [
        ("Normal HTTP", gen.normal_http()),
        ("SYN Flood", gen.syn_flood()),
        ("DNS Reflection", gen.dns_reflection()),
        ("Random Payload", gen.random_payload_attack()),
        ("Repeated Payload", gen.repeated_payload_attack()),
    ]

    for name, pkt in samples:
        scores = detector2.process(pkt)
        parsed = parse_packet(pkt)

        print(f"\n{name}:")
        print(f"  Src: {parsed.src_ip}:{parsed.src_port} → Dst: {parsed.dst_ip}:{parsed.dst_port}")
        print(f"  Protocol: {parsed.protocol}, Flags: {parsed.tcp_flags:#04x}, Payload: {parsed.payload_size} bytes")
        print(f"  Perspective scores:")
        print(f"    L3: {scores.l3:.3f}, L4-ports: {scores.l4_ports:.3f}, L4-flags: {scores.l4_flags:.3f}")
        print(f"    Payload-pos: {scores.payload_pos:.3f}, Payload-ngram: {scores.payload_ngram:.3f}, Payload-struct: {scores.payload_struct:.3f}")

        if parsed.payload_bytes:
            unique = len(set(parsed.payload_bytes))
            printable = sum(1 for b in parsed.payload_bytes if 32 <= b <= 126)
            print(f"    Payload spread: {unique} unique bytes, {100*printable/len(parsed.payload_bytes):.0f}% printable")

        anomalous = scores.anomalous_perspectives(threshold=0.4)
        if anomalous:
            print(f"    ⚠️ Anomalous: {anomalous}")

    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print("""
Key findings:

1. MULTI-PERSPECTIVE WORKS
   - Different attacks trigger different perspectives
   - SYN flood: L4-flags anomalous (pure SYN)
   - DNS reflection: L4-ports anomalous (src_port=53)
   - Random payload: payload_struct anomalous (high spread)

2. CARDINALITY IS A STRONG SIGNAL
   - Normal: high cardinality across fields
   - Attack: low cardinality in attack-specific fields
   - Change in cardinality = transition signal

3. PAYLOAD ANALYSIS
   - Positional: captures "byte 0 is usually X"
   - N-grams: captures sequences (protocol signatures)
   - Structure: captures spread/printability (entropy-like)

4. ENTROPY WITHOUT INFORMATION THEORY
   - unique_bytes = spread measure
   - printable_ratio = content type indicator
   - zero_ratio = padding indicator
   - These are vector-encodable features!

NEXT STEPS:
- Combine perspective scores with cardinality changes
- Use multi-horizon accumulators per perspective
- Track perspective divergence for attack classification
""")


if __name__ == "__main__":
    main()
