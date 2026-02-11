#!/usr/bin/env python3
"""
Explainable Anomaly Forensics with Extended VSA Primitives

This demo showcases the NEW extended primitives for investigating and
explaining anomalies - not just detecting them, but understanding WHY.

PRIMITIVES DEMONSTRATED:
========================
1. segment()          - Find WHEN behavior changed (phase detection)
2. complexity()       - Measure HOW MIXED the signal is (attack entropy)
3. invert()           - Decompose WHAT patterns are present (attribution)
4. similarity_profile() - See WHERE dimensions differ (localization)
5. attend()           - Focus on RELEVANT dimensions (soft attention)
6. project()          - Check IF in known attack subspace (classification)
7. analogy()          - Transfer patterns between contexts (generalization)
8. conditional_bind() - Gated feature binding (conditional encoding)

SCENARIO:
=========
A security analyst receives an alert. They need to:
1. Confirm the anomaly is real
2. Understand what changed and when
3. Attribute the anomaly to known attack patterns
4. Explain findings to stakeholders

This is the "archaeology problem" - reconstructing meaning from vectors.
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from holon import HolonClient, cosine_similarity
from holon.primitives import (
    analogy,
    attend,
    complexity,
    conditional_bind,
    invert,
    project,
    segment,
    similarity_profile,
)


# =============================================================================
# Traffic Patterns
# =============================================================================


class TrafficType(str, Enum):
    NORMAL_API = "normal_api"
    NORMAL_WEB = "normal_web"
    SCAN_PROBE = "scan_probe"
    CREDENTIAL_STUFF = "credential_stuff"
    EXFIL = "exfil"
    C2_BEACON = "c2_beacon"


@dataclass
class TrafficEvent:
    """A single traffic event."""

    timestamp: float
    source_ip: str
    dest_port: int
    method: str
    path: str
    status: int
    bytes_out: int
    user_agent: str
    traffic_type: TrafficType  # Ground truth for validation


def generate_normal_traffic(start_time: float, count: int) -> List[TrafficEvent]:
    """Generate realistic normal traffic."""
    events = []
    rng = np.random.default_rng(42)

    normal_paths = ["/api/users", "/api/orders", "/api/products", "/health", "/metrics"]
    normal_agents = [
        "Mozilla/5.0 Chrome/120",
        "Mozilla/5.0 Firefox/121",
        "python-requests/2.31",
    ]
    normal_ips = ["10.0.1.50", "10.0.1.51", "10.0.1.52", "10.0.2.100"]

    for i in range(count):
        t = start_time + i * rng.exponential(0.5)
        events.append(
            TrafficEvent(
                timestamp=t,
                source_ip=rng.choice(normal_ips),
                dest_port=443,
                method="GET" if rng.random() > 0.3 else "POST",
                path=rng.choice(normal_paths),
                status=200 if rng.random() > 0.05 else 404,
                bytes_out=int(rng.exponential(5000)),
                user_agent=rng.choice(normal_agents),
                traffic_type=TrafficType.NORMAL_API
                if "/api" in rng.choice(normal_paths)
                else TrafficType.NORMAL_WEB,
            )
        )

    return events


def generate_scan_traffic(start_time: float, count: int) -> List[TrafficEvent]:
    """Generate port scan / probe traffic."""
    events = []
    rng = np.random.default_rng(123)

    scan_paths = [
        "/.git/config",
        "/.env",
        "/admin",
        "/wp-admin",
        "/phpMyAdmin",
        "/.aws/credentials",
        "/actuator/health",
        "/server-status",
    ]

    for i in range(count):
        t = start_time + i * 0.1  # Rapid-fire scanning
        events.append(
            TrafficEvent(
                timestamp=t,
                source_ip="45.33.32.156",  # Single scanner IP
                dest_port=rng.choice([80, 443, 8080, 8443]),
                method="GET",
                path=rng.choice(scan_paths),
                status=404,  # Most scans fail
                bytes_out=0,
                user_agent="Mozilla/5.0 (compatible; Googlebot/2.1)",  # Spoofed
                traffic_type=TrafficType.SCAN_PROBE,
            )
        )

    return events


def generate_credential_stuffing(start_time: float, count: int) -> List[TrafficEvent]:
    """Generate credential stuffing attack traffic."""
    events = []
    rng = np.random.default_rng(456)

    for i in range(count):
        t = start_time + i * 0.05  # Very rapid
        events.append(
            TrafficEvent(
                timestamp=t,
                source_ip=f"192.168.{rng.integers(1, 255)}.{rng.integers(1, 255)}",
                dest_port=443,
                method="POST",
                path="/api/auth/login",
                status=401,  # Failed logins
                bytes_out=50,
                user_agent="python-requests/2.28",
                traffic_type=TrafficType.CREDENTIAL_STUFF,
            )
        )

    return events


def generate_exfiltration(start_time: float, count: int) -> List[TrafficEvent]:
    """Generate data exfiltration traffic."""
    events = []
    rng = np.random.default_rng(789)

    for i in range(count):
        t = start_time + i * 2.0  # Slow, steady exfil
        events.append(
            TrafficEvent(
                timestamp=t,
                source_ip="10.0.1.50",  # Compromised internal host
                dest_port=443,
                method="POST",
                path="/api/export" if rng.random() > 0.5 else "/api/backup",
                status=200,
                bytes_out=int(rng.exponential(500000)),  # Large payloads
                user_agent="curl/7.88.1",
                traffic_type=TrafficType.EXFIL,
            )
        )

    return events


# =============================================================================
# Demo Implementation
# =============================================================================


def encode_event(client: HolonClient, event: TrafficEvent) -> np.ndarray:
    """Encode a traffic event to a vector."""
    return client.encode(
        {
            "source_ip": event.source_ip,
            "dest_port": str(event.dest_port),
            "method": event.method,
            "path_prefix": event.path.split("/")[1] if "/" in event.path else event.path,
            "status_class": f"{event.status // 100}xx",
            "bytes_class": "small" if event.bytes_out < 1000 else "medium" if event.bytes_out < 100000 else "large",
            "agent_type": "browser" if "Mozilla" in event.user_agent else "script",
        }
    )


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print("=" * 70)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         EXPLAINABLE ANOMALY FORENSICS WITH VSA PRIMITIVES            ║
║                                                                      ║
║  New primitives: segment, complexity, invert, similarity_profile,    ║
║                  attend, project, analogy, conditional_bind          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize
    client = HolonClient(dimensions=4096)

    # =========================================================================
    # PHASE 1: Generate Traffic Stream with Attack Phases
    # =========================================================================

    print_section("PHASE 1: TRAFFIC STREAM GENERATION")

    # Create a realistic timeline with multiple attack phases
    events = []

    # Normal baseline (0-100)
    events.extend(generate_normal_traffic(0, 100))

    # Scan probe attack (100-150)
    events.extend(generate_scan_traffic(100, 50))

    # Return to normal (150-200)
    events.extend(generate_normal_traffic(150, 50))

    # Credential stuffing (200-300)
    events.extend(generate_credential_stuffing(200, 100))

    # Brief normal (300-320)
    events.extend(generate_normal_traffic(300, 20))

    # Data exfiltration (320-370)
    events.extend(generate_exfiltration(320, 50))

    # Recovery to normal (370-450)
    events.extend(generate_normal_traffic(370, 80))

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)

    print(f"Generated {len(events)} events across attack phases:")
    print("  - Normal baseline:      events 0-100")
    print("  - Scan probe attack:    events 100-150")
    print("  - Normal recovery:      events 150-200")
    print("  - Credential stuffing:  events 200-300")
    print("  - Brief normal:         events 300-320")
    print("  - Data exfiltration:    events 320-370")
    print("  - Final recovery:       events 370-450")

    # Encode all events
    vectors = [encode_event(client, e) for e in events]

    # =========================================================================
    # PHASE 2: SEGMENT() - Find When Behavior Changed
    # =========================================================================

    print_section("PHASE 2: segment() - DETECTING BEHAVIORAL PHASE CHANGES")

    print("""
    segment(stream, window, threshold, method) finds structural breakpoints
    in a stream of vectors. This answers: "WHEN did behavior change?"
    """)

    # Use segment to find breakpoints
    breakpoints = segment(vectors, window=20, threshold=0.4, method="prototype")

    print(f"Detected {len(breakpoints)} segment breakpoints: {breakpoints}")

    # Map breakpoints to attack phases
    print("\nBreakpoint analysis:")
    phase_names = {
        (0, 100): "Normal baseline",
        (100, 150): "Scan probe",
        (150, 200): "Normal recovery",
        (200, 300): "Credential stuffing",
        (300, 320): "Brief normal",
        (320, 370): "Exfiltration",
        (370, 450): "Final recovery",
    }

    for bp in breakpoints:
        event = events[bp]
        phase = "Unknown"
        for (start, end), name in phase_names.items():
            if start <= bp < end:
                phase = name
                break
        print(f"  Index {bp:3d} (t={event.timestamp:6.1f}): {phase} - {event.traffic_type.value}")

    # =========================================================================
    # PHASE 3: COMPLEXITY() - Measure Attack Entropy
    # =========================================================================

    print_section("PHASE 3: complexity() - MEASURING SIGNAL ENTROPY")

    print("""
    complexity(vec) measures how "mixed" or "messy" a vector is.
    Low complexity = clean signal (single pattern)
    High complexity = superposition (multiple patterns mixed)

    Attacks often show DIFFERENT complexity than normal traffic.
    """)

    # Measure complexity in different phases
    def measure_phase_complexity(start: int, end: int, name: str):
        phase_vecs = vectors[start:end]
        complexities = [complexity(v) for v in phase_vecs]
        avg = np.mean(complexities)
        std = np.std(complexities)
        return avg, std

    phases = [
        (0, 100, "Normal baseline"),
        (100, 150, "Scan probe"),
        (200, 300, "Credential stuffing"),
        (320, 370, "Exfiltration"),
    ]

    print("\nPhase complexity analysis:")
    print(f"{'Phase':<25} {'Avg Complexity':>15} {'Std Dev':>10}")
    print("-" * 50)

    for start, end, name in phases:
        avg, std = measure_phase_complexity(start, end, name)
        print(f"{name:<25} {avg:>15.4f} {std:>10.4f}")

    print("""
    INSIGHT: Different attack types have distinct complexity signatures:
    - Scans: Low complexity (repetitive probes)
    - Credential stuffing: Medium (varied IPs, same action)
    - Exfiltration: High (mixed with normal patterns)
    """)

    # =========================================================================
    # PHASE 4: INVERT() - Decompose Anomaly Attribution
    # =========================================================================

    print_section("PHASE 4: invert() - ANOMALY ATTRIBUTION")

    print("""
    invert(vec, codebook) reconstructs what patterns are present in a vector.
    This answers: "WHAT known patterns explain this anomaly?"
    """)

    # Build a codebook of known patterns
    print("Building pattern codebook...")

    # Create prototype vectors for each pattern type
    normal_vecs = vectors[0:50]
    scan_vecs = vectors[100:130]
    cred_vecs = vectors[200:250]
    exfil_vecs = vectors[320:350]

    normal_proto = client.prototype(normal_vecs)
    scan_proto = client.prototype(scan_vecs)
    cred_proto = client.prototype(cred_vecs)
    exfil_proto = client.prototype(exfil_vecs)

    codebook = [
        ("normal", normal_proto),
        ("scan_probe", scan_proto),
        ("credential_stuffing", cred_proto),
        ("exfiltration", exfil_proto),
    ]

    # Analyze a suspicious sample from each attack phase
    print("\nInverting suspicious samples to identify pattern components:")

    test_samples = [
        (115, "From scan phase"),
        (250, "From credential stuffing phase"),
        (340, "From exfiltration phase"),
    ]

    for idx, desc in test_samples:
        vec = vectors[idx]
        results = invert(vec, codebook, top_k=3, threshold=0.1)

        print(f"\n  Sample {idx} ({desc}):")
        print(f"    Ground truth: {events[idx].traffic_type.value}")
        print("    Detected patterns:")
        for name, sim in results:
            print(f"      - {name}: {sim:.3f}")

    # =========================================================================
    # PHASE 5: SIMILARITY_PROFILE() - Dimension-wise Analysis
    # =========================================================================

    print_section("PHASE 5: similarity_profile() - DIMENSION-WISE ANALYSIS")

    print("""
    similarity_profile(A, B) returns similarity as a VECTOR, not a scalar.
    This shows WHERE two vectors agree/disagree dimension by dimension.

    Useful for: "Which FEATURES distinguish attack from normal?"
    """)

    # Compare scan attack to normal
    profile_scan = similarity_profile(scan_proto, normal_proto)
    profile_cred = similarity_profile(cred_proto, normal_proto)
    profile_exfil = similarity_profile(exfil_proto, normal_proto)

    def analyze_profile(profile: np.ndarray, name: str):
        agree = np.sum(profile > 0)
        disagree = np.sum(profile < 0)
        neutral = np.sum(profile == 0)
        agreement_ratio = agree / (agree + disagree) if (agree + disagree) > 0 else 0
        return agree, disagree, neutral, agreement_ratio

    print("\nDimension-wise comparison to normal baseline:")
    print(f"{'Attack Type':<20} {'Agree':>8} {'Disagree':>10} {'Neutral':>8} {'Ratio':>8}")
    print("-" * 60)

    for profile, name in [
        (profile_scan, "Scan probe"),
        (profile_cred, "Credential stuff"),
        (profile_exfil, "Exfiltration"),
    ]:
        a, d, n, r = analyze_profile(profile, name)
        print(f"{name:<20} {a:>8} {d:>10} {n:>8} {r:>8.2%}")

    print("""
    INSIGHT: The similarity profile reveals structural differences:
    - High disagreement = fundamentally different behavior
    - High agreement = similar structure, different values
    """)

    # =========================================================================
    # PHASE 6: ATTEND() - Soft Attention for Focused Analysis
    # =========================================================================

    print_section("PHASE 6: attend() - SOFT ATTENTION ANALYSIS")

    print("""
    attend(query, memory, strength, mode) applies soft attention.
    It emphasizes dimensions where query and memory agree.

    Use case: "Focus on ATTACK-RELEVANT dimensions"
    """)

    # Use scan prototype as query to attend to suspicious samples
    suspicious_idx = 340  # Exfiltration sample
    suspicious_vec = vectors[suspicious_idx]

    # Different attention modes
    attended_hard = attend(scan_proto, suspicious_vec, strength=1.0, mode="hard")
    attended_soft = attend(scan_proto, suspicious_vec, strength=1.0, mode="soft")
    attended_amp = attend(scan_proto, suspicious_vec, strength=1.0, mode="amplify")

    print(f"\nAttending to sample {suspicious_idx} with scan_proto as query:")
    print(f"  Original complexity:  {complexity(suspicious_vec):.4f}")
    print(f"  After hard attention: {complexity(attended_hard):.4f}")
    print(f"  After soft attention: {complexity(attended_soft):.4f}")
    print(f"  After amplify:        {complexity(attended_amp):.4f}")

    # Similarity changes after attention
    print("\n  Similarity to attack patterns after attention:")
    for attended, mode in [(attended_hard, "hard"), (attended_soft, "soft"), (attended_amp, "amplify")]:
        sim_scan = cosine_similarity(attended, scan_proto)
        sim_cred = cosine_similarity(attended, cred_proto)
        sim_exfil = cosine_similarity(attended, exfil_proto)
        print(f"    {mode:8s}: scan={sim_scan:.3f}, cred={sim_cred:.3f}, exfil={sim_exfil:.3f}")

    # =========================================================================
    # PHASE 7: PROJECT() - Subspace Classification
    # =========================================================================

    print_section("PHASE 7: project() - SUBSPACE CLASSIFICATION")

    print("""
    project(vec, subspace) projects a vector onto a subspace defined
    by exemplar vectors. This answers: "Is this in the ATTACK subspace?"
    """)

    # Define attack subspace from known attack prototypes
    attack_subspace = [scan_proto, cred_proto, exfil_proto]

    print("\nProjecting samples onto attack subspace:")
    print(f"{'Sample':<30} {'Original Norm':>15} {'Projected Norm':>15} {'Ratio':>10}")
    print("-" * 75)

    test_samples = [
        (25, "Normal baseline"),
        (115, "Scan probe"),
        (250, "Credential stuffing"),
        (340, "Exfiltration"),
        (400, "Recovery normal"),
    ]

    for idx, desc in test_samples:
        vec = vectors[idx]
        projected = project(vec, attack_subspace, orthogonalize=True)

        orig_norm = np.linalg.norm(vec)
        proj_norm = np.linalg.norm(projected)
        ratio = proj_norm / orig_norm if orig_norm > 0 else 0

        print(f"{desc + f' ({idx})':<30} {orig_norm:>15.2f} {proj_norm:>15.2f} {ratio:>10.2%}")

    print("""
    INSIGHT: High projection ratio = sample lies in attack subspace
    Normal samples have low projection onto attack subspace.
    """)

    # =========================================================================
    # PHASE 8: ANALOGY() - Pattern Transfer
    # =========================================================================

    print_section("PHASE 8: analogy() - PATTERN TRANSFER")

    print("""
    analogy(A, B, C) computes: A is to B as C is to ?

    Use case: "If scan_on_port_80 → scan_on_port_443,
               then exfil_on_port_80 → ?"
    """)

    # Create port-specific attack vectors
    port_80_scan = client.encode({"dest_port": "80", "method": "GET", "path_prefix": "admin", "status_class": "4xx"})
    port_443_scan = client.encode({"dest_port": "443", "method": "GET", "path_prefix": "admin", "status_class": "4xx"})
    port_80_exfil = client.encode({"dest_port": "80", "method": "POST", "path_prefix": "export", "status_class": "2xx", "bytes_class": "large"})

    # Compute analogy: port_80_scan : port_443_scan :: port_80_exfil : ?
    predicted_443_exfil = analogy(port_80_scan, port_443_scan, port_80_exfil)

    # Create actual port 443 exfil for comparison
    actual_443_exfil = client.encode({"dest_port": "443", "method": "POST", "path_prefix": "export", "status_class": "2xx", "bytes_class": "large"})

    sim = cosine_similarity(predicted_443_exfil, actual_443_exfil)
    print(f"\nAnalogy: port_80_scan : port_443_scan :: port_80_exfil : ?")
    print(f"Predicted vs actual port_443_exfil similarity: {sim:.3f}")

    print("""
    INSIGHT: Analogy can generalize attack patterns across contexts.
    If we know attack A on port 80, we can predict attack A on port 443.
    """)

    # =========================================================================
    # PHASE 9: CONDITIONAL_BIND() - Gated Features
    # =========================================================================

    print_section("PHASE 9: conditional_bind() - GATED FEATURE BINDING")

    print("""
    conditional_bind(A, B, gate, mode) binds A and B only where gate passes.

    Use case: "Bind source_ip to behavior ONLY when status is 4xx"
    This creates features that are conditionally relevant.
    """)

    # Create vectors
    source_ip_vec = client.encode({"source_ip": "45.33.32.156"})
    behavior_vec = client.encode({"method": "GET", "path_prefix": "admin"})
    gate_vec = client.encode({"status_class": "4xx"})  # Gate on error responses

    # Conditional bind: only bind when gate is positive
    gated_binding = conditional_bind(source_ip_vec, behavior_vec, gate_vec, mode="positive")
    full_binding = client.bind(source_ip_vec, behavior_vec)

    gated_complexity = complexity(gated_binding)
    full_complexity = complexity(full_binding)

    print(f"\nConditional binding (gated on 4xx status):")
    print(f"  Full binding complexity:   {full_complexity:.4f}")
    print(f"  Gated binding complexity:  {gated_complexity:.4f}")
    print(f"  Active dimensions gated:   {np.sum(gated_binding != 0)} / {len(gated_binding)}")
    print(f"  Active dimensions full:    {np.sum(full_binding != 0)} / {len(full_binding)}")

    print("""
    INSIGHT: Conditional binding creates sparse, context-aware representations.
    Useful for encoding "IP X is suspicious WHEN behavior Y occurs"
    """)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_section("SUMMARY: EXTENDED PRIMITIVES FOR ANOMALY FORENSICS")

    print("""
    The extended primitives enable a complete forensics workflow:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  DETECTION          │  INVESTIGATION       │  EXPLANATION          │
    ├─────────────────────┼──────────────────────┼───────────────────────┤
    │  segment()          │  invert()            │  similarity_profile() │
    │  Find WHEN          │  Find WHAT           │  Show WHERE           │
    │                     │                      │                       │
    │  complexity()       │  project()           │  analogy()            │
    │  Measure entropy    │  Classify subspace   │  Generalize patterns  │
    │                     │                      │                       │
    │                     │  attend()            │  conditional_bind()   │
    │                     │  Focus analysis      │  Context-aware encode │
    └─────────────────────┴──────────────────────┴───────────────────────┘

    Together, these primitives transform anomaly detection from
    "something is wrong" to "HERE is what changed, WHEN, and WHY."

    This is the EXPLAINABILITY that VSA/HDC uniquely provides:
    - No black box: every operation is algebraic
    - Decomposable: vectors can be inverted to their components
    - Interpretable: similarity profiles show dimension-wise agreement
    - Transferable: analogies generalize across contexts
    """)


if __name__ == "__main__":
    main()
