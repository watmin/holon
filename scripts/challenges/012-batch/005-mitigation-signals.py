#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 005: Mitigation Signal Emission
=============================================================================

Extends challenge 004 to emit structured mitigation signals for downstream
consumers. Signals are pure data - no actual iptables/firewall rules.

A mitigation signal contains:
- WHAT: Which field-value pairs are causing the anomaly
- WHY: Concentration, novelty, or volumetric shift
- SEVERITY: Confidence level (0-1)
- ACTION: Suggested action type (block, rate_limit, monitor)
- SCOPE: How broad the mitigation should be

The consumer decides how to interpret and act on these signals.

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/005-mitigation-signals.py
"""

import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from enum import Enum
import numpy as np
import json

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
DECAY = 0.98
WARMUP_PACKETS = 300


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# MITIGATION SIGNAL TYPES
# =============================================================================

class SignalType(Enum):
    CONCENTRATION = "concentration"  # Field became concentrated on specific value
    NOVELTY = "novelty"              # Novel values appeared in field
    VOLUMETRIC = "volumetric"        # Overall traffic pattern shifted


class ActionType(Enum):
    BLOCK = "block"              # Block traffic matching this pattern
    RATE_LIMIT = "rate_limit"    # Rate limit matching traffic
    MONITOR = "monitor"          # Increase monitoring, don't act yet
    CLEAR = "clear"              # Remove previous mitigation


class Scope(Enum):
    EXACT = "exact"      # Match exact field=value
    FIELD = "field"      # Match any unusual value in field
    GLOBAL = "global"    # Affect all traffic


@dataclass
class MitigationSignal:
    """Structured mitigation signal for downstream consumers."""

    timestamp: int               # Packet number when signal was generated
    signal_type: SignalType      # What kind of anomaly
    action: ActionType           # Suggested action
    scope: Scope                 # How broad to apply
    severity: float              # 0.0 to 1.0

    # What to match
    field: Optional[str] = None
    value: Optional[Any] = None

    # Context for decision making
    concentration: float = 0.0   # How concentrated (0-1)
    prior_concentration: float = 0.0
    divergence: float = 0.0      # How different from baseline

    # Explanation for operators
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for consumers."""
        return {
            "timestamp": self.timestamp,
            "signal_type": self.signal_type.value,
            "action": self.action.value,
            "scope": self.scope.value,
            "severity": round(self.severity, 3),
            "field": self.field,
            "value": self.value,
            "concentration": round(self.concentration, 3),
            "prior_concentration": round(self.prior_concentration, 3),
            "divergence": round(self.divergence, 3),
            "reason": self.reason,
        }

    def __repr__(self):
        if self.field and self.value is not None:
            return f"Signal({self.action.value}: {self.field}={self.value}, sev={self.severity:.2f})"
        elif self.field:
            return f"Signal({self.action.value}: {self.field}=*, sev={self.severity:.2f})"
        else:
            return f"Signal({self.action.value}: GLOBAL, sev={self.severity:.2f})"


# =============================================================================
# LIFECYCLE PHASES
# =============================================================================

class Phase(Enum):
    WARMUP = "warmup"
    NORMAL = "normal"
    ATTACK = "attack"
    DRAIN = "drain"


@dataclass
class PhaseSpec:
    phase: Phase
    duration: int
    attack_fraction: float = 0.0
    attack_type: Optional[str] = None
    description: str = ""


# =============================================================================
# FIELD TRACKER
# =============================================================================

class FieldTracker:
    def __init__(self, field: str, encoder, dimensions: int):
        self.field = field
        self.encoder = encoder
        self.dimensions = dimensions

        self.prior_accum = encoder.create_accumulator()
        self.recent_accum = encoder.create_accumulator()

        self.prior_counts = {}
        self.prior_total = 0
        self.recent_counts = {}
        self.recent_total = 0

        self.frozen = False
        self.prior_concentration = 0.0
        self.prior_dominant = None

    def _encode(self, value: Any) -> np.ndarray:
        return self.encoder.encode_data({self.field: value})

    def observe(self, value: Any, is_warmup: bool):
        if value is None:
            return
        vec = self._encode(value)

        if is_warmup:
            self.prior_accum = self.encoder.accumulate(self.prior_accum, vec)
            self.prior_counts[value] = self.prior_counts.get(value, 0) + 1
            self.prior_total += 1
        else:
            self.recent_accum = DECAY * self.recent_accum + vec.astype(np.float64)
            self.recent_counts[value] = self.recent_counts.get(value, 0) + 1
            self.recent_total += 1

            if self.recent_total > 150:
                for k in list(self.recent_counts.keys()):
                    self.recent_counts[k] = self.recent_counts[k] // 2
                    if self.recent_counts[k] == 0:
                        del self.recent_counts[k]
                self.recent_total = sum(self.recent_counts.values())

    def freeze(self):
        self.frozen = True
        if self.prior_total > 0 and self.prior_counts:
            dom = max(self.prior_counts.items(), key=lambda x: x[1])
            self.prior_dominant = dom[0]
            self.prior_concentration = dom[1] / self.prior_total
        self.recent_accum = self.prior_accum.copy()
        self.recent_counts = dict(self.prior_counts)
        self.recent_total = self.prior_total

    def get_divergence(self) -> float:
        if not self.frozen:
            return 0.0
        prior_norm = self.encoder.normalize_accumulator(self.prior_accum)
        recent_norm = self.encoder.normalize_accumulator(self.recent_accum)
        return 1.0 - cosine_similarity(prior_norm, recent_norm)

    def get_concentration_info(self) -> tuple:
        if self.recent_total == 0 or not self.recent_counts:
            return None, 0.0, 0.0
        dom = max(self.recent_counts.items(), key=lambda x: x[1])
        concentration = dom[1] / self.recent_total
        change = concentration - self.prior_concentration
        return dom[0], concentration, change

    def is_novel_value(self, value: Any) -> bool:
        """Check if value was never seen during warmup."""
        return value not in self.prior_counts


# =============================================================================
# MITIGATION DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    packet_num: int
    phase: Phase
    is_anomalous: bool
    confidence: float
    traffic_divergence: float
    significant_fields: List[str]
    explanation: str
    signals: List[MitigationSignal] = field(default_factory=list)


class MitigationDetector:
    """Detector that emits mitigation signals."""

    FIELDS = ["protocol", "src_port", "dst_port", "tcp_flags", "icmp_type", "payload_size"]

    def __init__(self, warmup: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup = warmup
        self.packet_count = 0
        self.warmup_complete = False

        self.field_trackers = {f: FieldTracker(f, self.encoder, DIMENSIONS) for f in self.FIELDS}

        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()

        self._prior_norm = None
        self._recent_norm = None

        self.anomaly_history = deque(maxlen=30)
        self.baseline_sims = []
        self.baseline_mean = 0.7
        self.baseline_std = 0.1

        # Track active mitigations to emit CLEAR signals
        self.active_mitigations: Dict[Tuple[str, Any], MitigationSignal] = {}
        self.mitigation_cooldown: Dict[Tuple[str, Any], int] = {}

        # Deduplication - don't emit same signal repeatedly
        self.last_emitted: Dict[Tuple[str, Any], int] = {}
        self.EMIT_COOLDOWN = 50  # Packets between re-emitting same signal

    def _extract_fields(self, packet: dict) -> Dict[str, Any]:
        fields = {}
        fields["protocol"] = packet.get("protocol")
        fields["src_port"] = packet.get("src_port")
        fields["dst_port"] = packet.get("dst_port")
        if packet.get("protocol") == "TCP":
            fields["tcp_flags"] = packet.get("flags")
        if packet.get("protocol") == "ICMP":
            fields["icmp_type"] = packet.get("icmp_type")
        size = packet.get("payload_size", 0)
        fields["payload_size"] = "none" if size == 0 else "small" if size < 256 else "large"
        return fields

    def _update_caches(self):
        self._prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
        self._recent_norm = self.encoder.normalize_accumulator(self.recent_traffic)

    def _compute_severity(self, concentration: float, change: float, divergence: float) -> float:
        """Compute severity score from multiple factors."""
        # High concentration + high change + high divergence = high severity
        conc_factor = max(0, (concentration - 0.5) * 2)  # 0.5 → 0, 1.0 → 1
        change_factor = max(0, change * 2)                # 0.0 → 0, 0.5 → 1
        div_factor = min(1, divergence * 2)               # 0.0 → 0, 0.5 → 1

        return min(1.0, (conc_factor + change_factor + div_factor) / 2)

    def _determine_action(self, severity: float, concentration: float, is_novel: bool) -> ActionType:
        """Determine suggested action based on severity and context."""
        if severity > 0.7 and concentration > 0.8:
            return ActionType.BLOCK
        elif severity > 0.4 or concentration > 0.6:
            return ActionType.RATE_LIMIT
        else:
            return ActionType.MONITOR

    def _should_emit(self, key: Tuple[str, Any]) -> bool:
        """Check if we should emit a signal (deduplication)."""
        last = self.last_emitted.get(key, 0)
        return self.packet_count - last >= self.EMIT_COOLDOWN

    def _is_ephemeral_port(self, field: str, value: Any) -> bool:
        """Check if this is an ephemeral port (noise for mitigation)."""
        if field != "src_port":
            return False
        try:
            return int(value) >= 49152
        except (TypeError, ValueError):
            return False

    def _generate_signals(self, fields: Dict[str, Any], sig_fields: List[str],
                          traffic_div: float) -> List[MitigationSignal]:
        """Generate mitigation signals for current state."""
        signals = []
        current_time = self.packet_count

        # Check each significant field
        for field_name in sig_fields:
            tracker = self.field_trackers[field_name]
            dom_val, conc, change = tracker.get_concentration_info()
            div = tracker.get_divergence()

            if dom_val is None:
                continue

            # Skip ephemeral source ports (always "novel", not useful for mitigation)
            if self._is_ephemeral_port(field_name, dom_val):
                continue

            # Determine if this is a novel value
            is_novel = tracker.is_novel_value(dom_val)

            # Compute severity
            severity = self._compute_severity(conc, change, div)

            if severity < 0.3:
                continue

            # Deduplication - don't re-emit same signal
            key = (field_name, dom_val)
            if not self._should_emit(key):
                # Still track as active, just don't emit
                continue

            # Determine action
            action = self._determine_action(severity, conc, is_novel)

            # Determine signal type
            if is_novel:
                signal_type = SignalType.NOVELTY
                reason = f"Novel value {dom_val} appeared (never seen in baseline)"
            else:
                signal_type = SignalType.CONCENTRATION
                reason = f"Concentrated from {tracker.prior_concentration:.0%} to {conc:.0%}"

            # Create signal
            signal = MitigationSignal(
                timestamp=current_time,
                signal_type=signal_type,
                action=action,
                scope=Scope.EXACT,
                severity=severity,
                field=field_name,
                value=dom_val,
                concentration=conc,
                prior_concentration=tracker.prior_concentration,
                divergence=div,
                reason=reason,
            )

            signals.append(signal)
            self.last_emitted[key] = current_time

            # Track active mitigation
            self.active_mitigations[key] = signal
            self.mitigation_cooldown[key] = current_time

        # Check for volumetric signal (also deduplicated)
        vol_key = ("__volumetric__", None)
        if traffic_div > 0.35 and self._should_emit(vol_key):
            severity = min(1.0, traffic_div)
            action = ActionType.RATE_LIMIT if traffic_div < 0.5 else ActionType.BLOCK

            signal = MitigationSignal(
                timestamp=current_time,
                signal_type=SignalType.VOLUMETRIC,
                action=action,
                scope=Scope.GLOBAL,
                severity=severity,
                divergence=traffic_div,
                reason=f"Traffic pattern shifted {traffic_div:.0%} from baseline",
            )
            signals.append(signal)
            self.last_emitted[vol_key] = current_time

        # Check for CLEAR signals (previously active mitigations that are no longer needed)
        clear_keys = []
        for key, old_signal in self.active_mitigations.items():
            field_name, value = key

            # Don't clear if recently activated
            if current_time - self.mitigation_cooldown.get(key, 0) < 50:
                continue

            # Check if field is no longer significant
            if field_name not in sig_fields:
                tracker = self.field_trackers[field_name]
                div = tracker.get_divergence()

                # If divergence is low, traffic returned to normal
                if div < 0.15:
                    clear_signal = MitigationSignal(
                        timestamp=current_time,
                        signal_type=old_signal.signal_type,
                        action=ActionType.CLEAR,
                        scope=Scope.EXACT,
                        severity=0.0,
                        field=field_name,
                        value=value,
                        divergence=div,
                        reason=f"Traffic normalized (divergence={div:.0%})",
                    )
                    signals.append(clear_signal)
                    clear_keys.append(key)

        for key in clear_keys:
            del self.active_mitigations[key]
            del self.mitigation_cooldown[key]

        return signals

    def process(self, packet: dict, current_phase: Phase) -> DetectionResult:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup

        fields = self._extract_fields(packet)
        for field_name, value in fields.items():
            if field_name in self.field_trackers:
                self.field_trackers[field_name].observe(value, is_warmup)

        packet_vec = self.encoder.encode_data(packet)

        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)

            if self.packet_count == self.warmup:
                self.warmup_complete = True
                for t in self.field_trackers.values():
                    t.freeze()
                self.recent_traffic = self.prior_traffic.copy()
                self._update_caches()
                self.baseline_mean = np.mean(self.baseline_sims) if self.baseline_sims else 0.7
                self.baseline_std = np.std(self.baseline_sims) if len(self.baseline_sims) > 1 else 0.1
            elif self.packet_count > 50:
                prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
                sim = cosine_similarity(packet_vec, prior_norm)
                self.baseline_sims.append(sim)

            return DetectionResult(
                packet_num=self.packet_count,
                phase=current_phase,
                is_anomalous=False,
                confidence=0.0,
                traffic_divergence=0.0,
                significant_fields=[],
                explanation="Warming up...",
                signals=[],
            )

        self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        if self.packet_count % 10 == 0:
            self._update_caches()

        prior_sim = cosine_similarity(packet_vec, self._prior_norm)
        traffic_div = 1.0 - cosine_similarity(self._prior_norm, self._recent_norm)

        packet_is_normal = prior_sim >= self.baseline_mean - 1.5 * self.baseline_std

        sig_fields = []
        explanations = []

        for name, tracker in self.field_trackers.items():
            div = tracker.get_divergence()
            dom_val, conc, change = tracker.get_concentration_info()

            if div > 0.20 and (change > 0.15 or conc > 0.7):
                sig_fields.append(name)
                if conc > 0.7:
                    explanations.append(f"{name}→{dom_val}({conc:.0%})")

        concentration_matches = 0
        for name in sig_fields:
            tracker = self.field_trackers[name]
            dom_val, conc, change = tracker.get_concentration_info()
            if conc > 0.65 and change > 0.10:
                pkt_val = fields.get(name)
                if pkt_val == dom_val:
                    concentration_matches += 1

        matches_concentration = concentration_matches >= 2

        sim_threshold = self.baseline_mean - 2 * self.baseline_std
        is_anomalous = (
            prior_sim < sim_threshold or
            (len(sig_fields) >= 2 and matches_concentration) or
            (traffic_div > 0.25 and not packet_is_normal)
        )

        self.anomaly_history.append(is_anomalous)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = is_anomalous and anomaly_rate > 0.30

        confidence = max(0, (sim_threshold - prior_sim) / 0.3) if prior_sim < sim_threshold else 0
        confidence = max(confidence, traffic_div)

        # Generate mitigation signals
        signals = []
        if is_anomalous:
            signals = self._generate_signals(fields, sig_fields, traffic_div)
        elif len(self.active_mitigations) > 0:
            # Check if we should clear any active mitigations
            signals = self._generate_signals(fields, [], traffic_div)

        if not is_anomalous:
            explanation = "Normal"
        elif explanations:
            explanation = "; ".join(explanations[:3])
        else:
            explanation = f"Divergence={traffic_div:.0%}"

        return DetectionResult(
            packet_num=self.packet_count,
            phase=current_phase,
            is_anomalous=is_anomalous,
            confidence=confidence,
            traffic_divergence=traffic_div,
            significant_fields=sig_fields,
            explanation=explanation,
            signals=signals,
        )


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def gen_normal(rng: random.Random) -> dict:
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]
    if proto == "TCP":
        return {"protocol": "TCP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([80, 443, 8080, 22]),
                "flags": rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[0],
                "payload_size": rng.randint(0, 1500)}
    elif proto == "UDP":
        return {"protocol": "UDP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([53, 443, 123]), "payload_size": rng.randint(20, 512)}
    else:
        return {"protocol": "ICMP", "icmp_type": rng.choice([0, 8]), "payload_size": 64}


def gen_syn_flood(rng: random.Random) -> dict:
    return {"protocol": "TCP", "src_port": rng.randint(1, 65535), "dst_port": 80,
            "flags": "S", "payload_size": 0}


def gen_dns_reflection(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 53, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(256, 4096)}


def gen_ntp_amplification(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 123, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(300, 500)}


def gen_icmp_flood(rng: random.Random) -> dict:
    return {"protocol": "ICMP", "icmp_type": 8, "payload_size": 1400}


ATTACK_GENERATORS = {
    "syn_flood": gen_syn_flood,
    "dns_reflection": gen_dns_reflection,
    "ntp_amplification": gen_ntp_amplification,
    "icmp_flood": gen_icmp_flood,
}


# =============================================================================
# SIGNAL CONSUMER (Example)
# =============================================================================

class SignalConsumer:
    """Example consumer that tracks and reports mitigation signals."""

    def __init__(self):
        self.all_signals: List[MitigationSignal] = []
        self.active_blocks: Dict[Tuple[str, Any], MitigationSignal] = {}
        self.active_rate_limits: Dict[Tuple[str, Any], MitigationSignal] = {}
        self.signal_history: List[dict] = []

    def consume(self, signals: List[MitigationSignal]):
        """Process incoming signals."""
        for signal in signals:
            self.all_signals.append(signal)
            self.signal_history.append(signal.to_dict())

            key = (signal.field, signal.value) if signal.field else ("GLOBAL", None)

            if signal.action == ActionType.BLOCK:
                self.active_blocks[key] = signal
                self.active_rate_limits.pop(key, None)
            elif signal.action == ActionType.RATE_LIMIT:
                if key not in self.active_blocks:
                    self.active_rate_limits[key] = signal
            elif signal.action == ActionType.CLEAR:
                self.active_blocks.pop(key, None)
                self.active_rate_limits.pop(key, None)

    def get_active_rules(self) -> List[dict]:
        """Get current active mitigation rules."""
        rules = []
        for key, signal in self.active_blocks.items():
            rules.append({
                "action": "BLOCK",
                "match": {"field": signal.field, "value": signal.value},
                "severity": signal.severity,
                "reason": signal.reason,
            })
        for key, signal in self.active_rate_limits.items():
            rules.append({
                "action": "RATE_LIMIT",
                "match": {"field": signal.field, "value": signal.value},
                "severity": signal.severity,
                "reason": signal.reason,
            })
        return rules

    def summarize(self) -> dict:
        """Summarize signal activity."""
        by_type = {}
        by_action = {}
        for s in self.all_signals:
            by_type[s.signal_type.value] = by_type.get(s.signal_type.value, 0) + 1
            by_action[s.action.value] = by_action.get(s.action.value, 0) + 1

        return {
            "total_signals": len(self.all_signals),
            "by_type": by_type,
            "by_action": by_action,
            "active_blocks": len(self.active_blocks),
            "active_rate_limits": len(self.active_rate_limits),
        }


# =============================================================================
# LIFECYCLE SIMULATION
# =============================================================================

def run_lifecycle(attack_type: str, phases: List[PhaseSpec]):
    """Run a complete attack lifecycle simulation with signal emission."""

    print(f"\n{'='*70}")
    print(f"LIFECYCLE TEST: {attack_type}")
    print(f"{'='*70}")

    detector = MitigationDetector(warmup=WARMUP_PACKETS)
    consumer = SignalConsumer()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Build stream from phases
    stream = []
    for spec in phases:
        for _ in range(spec.duration):
            if spec.phase == Phase.WARMUP or spec.phase == Phase.NORMAL or spec.phase == Phase.DRAIN:
                stream.append((gen_normal(rng), spec.phase, "normal"))
            elif spec.phase == Phase.ATTACK:
                if rng.random() < spec.attack_fraction:
                    stream.append((attack_gen(rng), spec.phase, "attack"))
                else:
                    stream.append((gen_normal(rng), spec.phase, "normal"))

    # Process stream
    results = []
    signal_log = []
    for packet, phase, label in stream:
        result = detector.process(packet, phase)
        results.append((result, label))

        if result.signals:
            consumer.consume(result.signals)
            for sig in result.signals:
                signal_log.append((result.packet_num, phase.value, sig))

    # Analyze by phase
    print(f"\nPhase-by-Phase Analysis:")
    print("-" * 70)

    current_idx = 0
    for spec in phases:
        phase_results = results[current_idx:current_idx + spec.duration]
        current_idx += spec.duration

        if spec.phase == Phase.WARMUP:
            print(f"  {spec.phase.value:12} ({spec.duration:4} pkts): [warmup - learning baseline]")
            continue

        detected = sum(1 for r, _ in phase_results if r.is_anomalous)
        total = len(phase_results)

        attack_total = sum(1 for _, l in phase_results if l == "attack")
        attack_detected = sum(1 for r, l in phase_results if l == "attack" and r.is_anomalous)
        normal_total = sum(1 for _, l in phase_results if l == "normal")
        normal_detected = sum(1 for r, l in phase_results if l == "normal" and r.is_anomalous)

        # Count signals in this phase
        phase_signals = [s for pkt, ph, s in signal_log
                        if current_idx - spec.duration < pkt <= current_idx]
        block_signals = len([s for s in phase_signals if s.action == ActionType.BLOCK])
        rl_signals = len([s for s in phase_signals if s.action == ActionType.RATE_LIMIT])
        clear_signals = len([s for s in phase_signals if s.action == ActionType.CLEAR])

        if spec.phase == Phase.ATTACK:
            attack_recall = attack_detected / attack_total if attack_total > 0 else 0
            print(f"  {spec.description:12} ({spec.duration:4} pkts): "
                  f"Recall={attack_recall:.0%} | Signals: {block_signals} BLOCK, {rl_signals} RATE_LIMIT")
        else:
            fp_rate = normal_detected / normal_total if normal_total > 0 else 0
            status = "✓ CLEAN" if fp_rate < 0.1 else "⚠ FPs"
            print(f"  {spec.description:12} ({spec.duration:4} pkts): "
                  f"FP={fp_rate:.0%} | Signals: {clear_signals} CLEAR {status}")

    # Signal summary
    summary = consumer.summarize()
    print(f"\n  Signal Summary:")
    print(f"    Total signals: {summary['total_signals']}")
    print(f"    By type: {summary['by_type']}")
    print(f"    By action: {summary['by_action']}")

    # Sample signals
    print(f"\n  Sample Mitigation Signals:")
    sample_signals = [(pkt, ph, s) for pkt, ph, s in signal_log if s.action != ActionType.CLEAR][:5]
    for pkt, ph, sig in sample_signals:
        print(f"    [{pkt}] {sig.action.value.upper():10} {sig.field}={sig.value} "
              f"(sev={sig.severity:.2f}, {sig.signal_type.value})")

    # Sample CLEAR signals
    clear_samples = [(pkt, ph, s) for pkt, ph, s in signal_log if s.action == ActionType.CLEAR][:2]
    if clear_samples:
        print(f"\n  Sample CLEAR Signals:")
        for pkt, ph, sig in clear_samples:
            print(f"    [{pkt}] CLEAR {sig.field}={sig.value} ({sig.reason})")

    # Final active rules
    rules = consumer.get_active_rules()
    print(f"\n  Final Active Rules: {len(rules)}")
    for rule in rules[:3]:
        print(f"    {rule['action']}: {rule['match']} - {rule['reason']}")

    # Metrics
    post_warmup = [(r, l) for r, l in results if r.phase != Phase.WARMUP]
    all_tp = sum(1 for r, l in post_warmup if l == "attack" and r.is_anomalous)
    all_fp = sum(1 for r, l in post_warmup if l == "normal" and r.is_anomalous)
    all_fn = sum(1 for r, l in post_warmup if l == "attack" and not r.is_anomalous)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    attack_results = [(r, l) for r, l in post_warmup if r.phase == Phase.ATTACK]
    attack_pkts = sum(1 for _, l in attack_results if l == "attack")
    tp = sum(1 for r, l in attack_results if l == "attack" and r.is_anomalous)
    attack_recall = tp / attack_pkts if attack_pkts > 0 else 0

    normal_results = [(r, l) for r, l in post_warmup if r.phase in (Phase.NORMAL, Phase.DRAIN)]
    fp = sum(1 for r, l in normal_results if r.is_anomalous)
    fp_rate = fp / len(normal_results) if normal_results else 0

    return {
        "attack": attack_type,
        "f1": f1,
        "attack_recall": attack_recall,
        "fp_rate": fp_rate,
        "signals": summary,
    }


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 005: Mitigation Signal Emission")
    print("="*70)
    print("""
    Extends lifecycle detection to emit structured mitigation signals.

    Signal Types:
    - CONCENTRATION: Field became concentrated on specific value
    - NOVELTY: Novel values appeared in field
    - VOLUMETRIC: Overall traffic pattern shifted

    Action Types:
    - BLOCK: High severity, high concentration
    - RATE_LIMIT: Medium severity
    - MONITOR: Low severity, watch only
    - CLEAR: Remove previous mitigation

    Signals are pure data for downstream consumers.
    """)

    # Define lifecycle phases
    lifecycle = [
        PhaseSpec(Phase.WARMUP, 300, 0.0, None, "warmup"),
        PhaseSpec(Phase.NORMAL, 200, 0.0, None, "pre-attack"),
        PhaseSpec(Phase.ATTACK, 250, 0.90, None, "ATTACK-1"),
        PhaseSpec(Phase.DRAIN, 150, 0.0, None, "drain-1"),
        PhaseSpec(Phase.ATTACK, 200, 0.85, None, "ATTACK-2"),
        PhaseSpec(Phase.DRAIN, 150, 0.0, None, "drain-2"),
        PhaseSpec(Phase.ATTACK, 200, 0.80, None, "ATTACK-3"),
        PhaseSpec(Phase.DRAIN, 200, 0.0, None, "final-drain"),
    ]

    results = []

    for attack_type in ["dns_reflection", "syn_flood"]:
        for spec in lifecycle:
            if spec.phase == Phase.ATTACK:
                spec.attack_type = attack_type

        result = run_lifecycle(attack_type, lifecycle)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("MITIGATION SIGNAL SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'F1':>8} {'Recall':>10} {'Signals':>10} {'Blocks':>8} {'Clears':>8}")
    print("-"*70)
    for r in results:
        s = r['signals']
        blocks = s['by_action'].get('block', 0)
        clears = s['by_action'].get('clear', 0)
        print(f"{r['attack']:<20} {r['f1']:>8.3f} {r['attack_recall']:>10.1%} "
              f"{s['total_signals']:>10} {blocks:>8} {clears:>8}")

    print("""

MITIGATION SIGNAL DESIGN
========================

Signals are structured data for downstream consumers:

```json
{
  "timestamp": 512,
  "signal_type": "concentration",
  "action": "block",
  "scope": "exact",
  "severity": 0.85,
  "field": "src_port",
  "value": 53,
  "concentration": 0.96,
  "prior_concentration": 0.01,
  "divergence": 0.40,
  "reason": "Concentrated from 1% to 96%"
}
```

Consumer decides:
- Whether to act on the signal
- How to implement the action (iptables, ACL, null route)
- How long to maintain the mitigation
- Whether to escalate or de-escalate

Detector only provides:
- What changed (field, value)
- Why it matters (concentration, novelty)
- Suggested action (based on severity)
- When to clear (traffic normalized)
    """)


if __name__ == "__main__":
    main()
