"""
Streaming visualization for Holon - shows baseline vs incoming comparison.

Features:
- Left pane: Baseline accumulator built from training data
- Right pane: Incoming stream of new records
- Real-time similarity scoring
- SSE-based streaming updates
"""

import json
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from projector import HolonProjector

app = Flask(__name__, static_folder="static")


# =============================================================================
# Streaming State
# =============================================================================


@dataclass
class StreamState:
    """Streaming state for visualization.

    Detection approach inspired by holon-lab-ddos sidecar:
    - Warmup phase: Learn baseline with EMA, no anomaly detection
    - After warmup: Freeze baseline, detect deviations
    - Track field value concentrations for attack indicators
    """

    # Projector for visualization
    projector: HolonProjector = field(
        default_factory=lambda: HolonProjector(dimensions=4096)
    )

    # Baseline accumulator (float64 for frequency preservation)
    # Uses exponential moving average during warmup
    baseline_acc: Optional[np.ndarray] = None
    baseline_count: int = 0
    baseline_frozen: bool = False

    # Recent window accumulator (for drift detection)
    recent_acc: Optional[np.ndarray] = None
    recent_count: int = 0
    recent_window_size: int = 30  # Smaller window for faster drift detection

    # Warmup parameters (from DDoS lab)
    # veth-lab uses 60 windows * ~100 packets/window = 6000 packets
    # For the visualization demo, we use a shorter but still adequate warmup
    warmup_packets: int = 300  # Minimum packets before detection starts
    warmup_complete: bool = False
    ema_alpha: float = (
        0.15  # Exponential moving average weight (lower = smoother baseline)
    )

    # Field value concentration tracking
    field_counts: Dict = field(default_factory=dict)  # {"field:value": count}
    window_field_total: int = 0
    concentration_threshold: float = 0.4  # Flag if >40% of traffic from one value

    # Stream control
    is_streaming: bool = False
    stream_speed: float = 2.0  # Records per second (default faster)

    # Event queue for SSE
    event_queue: queue.Queue = field(default_factory=queue.Queue)

    # History for replay
    history: List[Dict] = field(default_factory=list)
    history_limit: int = 100

    # Statistics
    total_records: int = 0
    anomaly_count: int = 0
    similarity_threshold: float = (
        0.35  # Below this = anomaly (tuned based on observed distributions)
    )


state = StreamState()


def reset_state():
    """Reset to fresh state."""
    global state
    state = StreamState()


def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def process_record(record: Dict) -> Dict:
    """
    Process an incoming record:
    1. Encode it to a vector
    2. Compare to baseline
    3. Update accumulators
    4. Return visualization data with atoms
    """
    # Encode the record
    vector = state.projector.encoder.encode_data(record)
    position = state.projector.project_vector(vector)

    # Extract and project atoms + bindings for visualization
    atoms = []
    bindings = []
    for key, value in record.items():
        key_str = str(key)
        val_str = str(value)

        # Get atom vectors and project them
        key_vec = state.projector.vector_manager.get_vector(key_str)
        val_vec = state.projector.vector_manager.get_vector(val_str)

        key_pos = state.projector.project_vector(key_vec)
        val_pos = state.projector.project_vector(val_vec)

        # Compute binding (key ⊙ value) and project it
        bind_vec = state.projector.encoder.bind(key_vec, val_vec)
        bind_pos = state.projector.project_vector(bind_vec)

        atoms.append(
            {
                "id": f"key:{key_str}",
                "name": key_str,
                "type": "key",
                "position": list(key_pos),
            }
        )
        atoms.append(
            {
                "id": f"val:{key_str}:{val_str}",
                "name": val_str,
                "type": "value",
                "position": list(val_pos),
                "key": key_str,
            }
        )
        bindings.append(
            {
                "id": f"bind:{key_str}:{val_str}",
                "name": f"{key_str}={val_str}",
                "type": "binding",
                "position": list(bind_pos),
                "key": key_str,
                "value": val_str,
            }
        )

    # Initialize accumulators if needed
    dims = state.projector.dimensions
    if state.baseline_acc is None:
        state.baseline_acc = np.zeros(dims, dtype=np.float64)
    if state.recent_acc is None:
        state.recent_acc = np.zeros(dims, dtype=np.float64)

    # Track field value concentrations (for attack detection)
    for key, value in record.items():
        field_key = f"{key}:{value}"
        state.field_counts[field_key] = state.field_counts.get(field_key, 0) + 1
    state.window_field_total += 1

    # Compute similarity to baseline
    baseline_sim = 0.0
    if state.baseline_count > 0:
        baseline_normalized = state.baseline_acc / np.linalg.norm(state.baseline_acc)
        baseline_sim = compute_similarity(
            vector.astype(np.float64), baseline_normalized
        )

    # Check warmup status
    if not state.warmup_complete and state.baseline_count >= state.warmup_packets:
        state.warmup_complete = True
        state.baseline_frozen = True  # Freeze baseline after warmup

    # Check for anomaly (only after warmup complete)
    is_anomaly = state.warmup_complete and baseline_sim < state.similarity_threshold
    if is_anomaly:
        state.anomaly_count += 1

    # Ground truth: is this actually attack traffic?
    is_attack = "attack_type" in record

    # Assessment correctness (only meaningful after warmup)
    if not state.warmup_complete:
        assessment = "warmup"
        correct = None  # Not applicable during warmup
    elif is_attack and is_anomaly:
        assessment = "true_positive"  # Attack correctly detected
        correct = True
    elif not is_attack and not is_anomaly:
        assessment = "true_negative"  # Normal correctly passed
        correct = True
    elif not is_attack and is_anomaly:
        assessment = "false_positive"  # Normal incorrectly flagged
        correct = False
    else:  # is_attack and not is_anomaly
        assessment = "false_negative"  # Attack missed
        correct = False

    # Find concentrated field values (attack indicators)
    concentrated_fields = []
    if state.window_field_total > 10:
        for field_key, count in state.field_counts.items():
            concentration = count / state.window_field_total
            if concentration >= state.concentration_threshold:
                field, value = field_key.split(":", 1)
                concentrated_fields.append(
                    {
                        "field": field,
                        "value": value,
                        "concentration": round(concentration, 2),
                    }
                )

    # Update accumulators
    vec_float = vector.astype(np.float64)
    if not state.baseline_frozen:
        # During warmup: use EMA for smoother baseline learning
        if state.baseline_count == 0:
            state.baseline_acc = vec_float.copy()
        else:
            # Exponential moving average
            state.baseline_acc = (
                1 - state.ema_alpha
            ) * state.baseline_acc + state.ema_alpha * vec_float
        state.baseline_count += 1

    state.recent_acc += vec_float
    state.recent_count += 1

    # Reset recent window if full
    if state.recent_count >= state.recent_window_size:
        state.recent_acc = np.zeros(dims, dtype=np.float64)
        state.recent_count = 0
        # Also reset field concentration tracking per window
        state.field_counts.clear()
        state.window_field_total = 0

    state.total_records += 1

    # Compute drift (baseline vs recent)
    drift = 0.0
    if state.baseline_count > 0 and state.recent_count > 5:
        baseline_norm = state.baseline_acc / np.linalg.norm(state.baseline_acc)
        recent_norm = state.recent_acc / np.linalg.norm(state.recent_acc)
        drift = 1.0 - compute_similarity(baseline_norm, recent_norm)

    # Project baseline position - same as any other vector
    # With EMA, baseline_acc is already averaged (not sum), so project directly
    baseline_pos = [0, 0, 0]
    if state.baseline_count > 0:
        baseline_pos = list(state.projector.project_vector(state.baseline_acc))

    # Also project recent window position for comparison
    recent_pos = [0, 0, 0]
    if state.recent_count > 0:
        recent_normalized = state.recent_acc / (
            np.linalg.norm(state.recent_acc) + 1e-10
        )
        coords = (
            recent_normalized.astype(np.float32) @ state.projector.projection_matrix
        )
        recent_pos = [float(c) * 300 for c in coords]

    # Build result
    result = {
        "type": "record",
        "timestamp": time.time(),
        "record_id": state.total_records,
        "data": record,
        "position": list(position),
        "atoms": atoms,  # Atom positions for graph visualization
        "bindings": bindings,  # Binding positions (key ⊙ value)
        "baseline_position": baseline_pos,
        "similarity": baseline_sim,
        "is_anomaly": is_anomaly,
        "is_attack": is_attack,  # Ground truth
        "assessment": assessment,  # true_positive, true_negative, false_positive, false_negative
        "correct": correct,  # Was detection correct?
        "drift": drift,
        "concentrated_fields": concentrated_fields,  # Attack indicators
        "stats": {
            "total_records": state.total_records,
            "baseline_count": state.baseline_count,
            "baseline_frozen": state.baseline_frozen,
            "warmup_complete": state.warmup_complete,
            "warmup_progress": min(1.0, state.baseline_count / state.warmup_packets),
            "recent_count": state.recent_count,
            "anomaly_count": state.anomaly_count,
            "threshold": state.similarity_threshold,
        },
    }

    # Add to history
    state.history.append(result)
    if len(state.history) > state.history_limit:
        state.history.pop(0)

    return result


# =============================================================================
# Traffic Generation (matching veth-lab patterns from holon-lab-ddos)
# =============================================================================

# Server IPs in our "datacenter" (target of traffic)
SERVER_IPS = [
    "10.100.0.1",  # Primary server
    "10.100.0.2",  # Secondary server
]

# Normal client subnets (sources of legitimate traffic)
# Matching veth-lab: random IPs from 192.168.x.x range
CLIENT_SUBNETS = [
    "192.168.1",  # Office network
    "192.168.2",  # Branch office
]


def generate_normal_packet() -> Dict:
    """Generate normal network traffic matching veth-lab patterns.

    Key characteristics (from veth-lab):
    - Random source IPs from 192.168.x.x range (varied, not concentrated)
    - Normal port (8888) - different from attack port
    - Mix of TCP/UDP protocols
    """
    # Source: random client from 192.168.x.x (matching veth-lab)
    subnet = random.choice(CLIENT_SUBNETS)
    src_ip = f"{subnet}.{random.randint(1, 254)}"

    # Destination: one of our servers
    dst_ip = random.choice(SERVER_IPS)

    # Protocol mix (mostly TCP, some UDP)
    protocol = random.choices(["tcp", "udp"], weights=[80, 20])[0]

    # Normal destination port (from veth-lab)
    dst_port = NORMAL_PORT

    # TCP flags for normal traffic
    flags = None
    if protocol == "tcp":
        flags = random.choice(["SYN", "ACK", "PSH+ACK", "FIN+ACK"])

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.randint(32768, 60999),
        "dst_port": dst_port,
        "protocol": protocol,
        "size": random.randint(64, 1400),
        "flags": flags,
        "ttl": random.choice([64, 128, 255]),
    }


# =============================================================================
# Attack patterns from veth-lab (matching holon-lab-ddos)
# =============================================================================
#
# The veth-lab uses simple but effective attack patterns:
# - SYN Flood: TCP SYN packets from spoofed IPs
# - UDP Flood: UDP packets for reflection/amplification
# - ICMP Flood: ICMP echo requests
#
# Key characteristics:
# - Attack traffic uses FIXED source IP (10.0.0.100) - easy to detect concentration
# - Attack traffic uses FIXED port (9999) - another concentration indicator
# - Normal traffic uses RANDOM source IPs (192.168.x.x)

ATTACK_PATTERNS = {
    "syn_flood": {
        "description": "TCP SYN Flood - half-open connections",
        "protocol": "tcp",
        "flags": "SYN",
        "size_range": (40, 60),  # Minimal SYN packet
    },
    "udp_flood": {
        "description": "UDP Flood - reflection/amplification",
        "protocol": "udp",
        "flags": None,
        "size_range": (64, 512),  # Amplified response size
    },
    "icmp_flood": {
        "description": "ICMP Flood - ping of death",
        "protocol": "icmp",
        "flags": None,
        "size_range": (64, 1500),  # Variable ICMP size
    },
}

# Attacker source IP - FIXED for concentration detection (from veth-lab)
ATTACKER_IP = "10.0.0.100"

# Attack destination port - FIXED (from veth-lab)
ATTACK_PORT = 9999

# Normal destination port
NORMAL_PORT = 8888


def generate_attack_packet(attack_type: str = None) -> Dict:
    """Generate attack traffic matching veth-lab patterns.

    Key difference from normal traffic:
    - Fixed source IP (10.0.0.100) - creates concentration
    - Fixed destination port (9999) - another concentration indicator
    """
    # Pick attack type
    if attack_type is None:
        attack_type = random.choice(list(ATTACK_PATTERNS.keys()))

    pattern = ATTACK_PATTERNS[attack_type]
    size_min, size_max = pattern["size_range"]

    # Attack traffic characteristics (from veth-lab):
    # - Fixed attacker IP (spoofed) - creates src_ip concentration
    # - Fixed attack port - creates dst_port concentration
    packet = {
        "src_ip": ATTACKER_IP,  # Fixed attacker IP (concentration!)
        "dst_ip": random.choice(SERVER_IPS[:2]),  # Target servers
        "src_port": random.randint(1024, 65535),
        "dst_port": ATTACK_PORT,  # Fixed attack port (concentration!)
        "protocol": pattern["protocol"],
        "size": random.randint(size_min, size_max),
        "ttl": random.choice([64, 128, 255]),
        "attack_type": attack_type,
    }

    if pattern["flags"]:
        packet["flags"] = pattern["flags"]

    return packet


# =============================================================================
# Background Streaming
# =============================================================================

stream_thread = None
stop_streaming = threading.Event()


def stream_worker():
    """Background worker that generates streaming data with realistic patterns.

    Traffic pattern inspired by DDoS lab scenarios:
    1. Warmup phase: Normal traffic only, baseline learning
    2. Normal phase: Mostly normal traffic with occasional outliers
    3. Attack phase: Bursts of concentrated attack traffic
    """
    attack_mode = False
    attack_start = None
    current_attack_type = None
    attack_types = list(ATTACK_PATTERNS.keys())
    attack_type_index = 0

    while not stop_streaming.is_set():
        if not state.is_streaming:
            time.sleep(0.1)
            continue

        # Use state's warmup tracking (set by process_record)
        warmup_complete = state.warmup_complete

        # During warmup: only normal traffic (baseline learning)
        if not warmup_complete:
            record = generate_normal_packet()
        else:
            # After warmup: cycle between normal and attack phases
            # Attack pattern: Normal period, then attack burst, repeat
            # Adjusted timing for better visibility:
            #   - Normal period: 25 records
            #   - Attack burst: 15 records (longer burst for clearer anomaly)
            attack_interval = 25
            attack_duration = 15

            # Calculate position in the attack cycle (start counting after warmup)
            records_since_warmup = state.total_records - state.warmup_packets
            cycle_position = records_since_warmup % (attack_interval + attack_duration)

            if cycle_position >= attack_interval:
                # Attack phase
                if not attack_mode:
                    attack_mode = True
                    attack_start = state.total_records
                    # Rotate through attack types
                    current_attack_type = attack_types[
                        attack_type_index % len(attack_types)
                    ]
                    attack_type_index += 1

                # 100% attack packets during attack phase (clean separation for demo)
                record = generate_attack_packet(current_attack_type)
            else:
                # Normal phase - 100% normal traffic (clean separation for demo)
                if attack_mode:
                    attack_mode = False
                    current_attack_type = None

                record = generate_normal_packet()

        # Process and emit
        result = process_record(record)

        # Add phase info to result
        result["phase"] = (
            "warmup" if not warmup_complete else ("attack" if attack_mode else "normal")
        )
        if current_attack_type:
            result["attack_type"] = current_attack_type

        state.event_queue.put(result)

        # Wait based on speed
        delay = 1.0 / max(0.1, state.stream_speed)
        time.sleep(delay)


def start_stream_thread():
    """Start the background streaming thread."""
    global stream_thread
    stop_streaming.clear()
    if stream_thread is None or not stream_thread.is_alive():
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()


# Start thread on import
start_stream_thread()


# =============================================================================
# Routes
# =============================================================================


@app.route("/")
def index():
    return send_from_directory("static", "streaming.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/api/stream/events")
def stream_events():
    """SSE endpoint for streaming events."""

    def generate():
        while True:
            try:
                event = state.event_queue.get(timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/stream/control", methods=["POST"])
def stream_control():
    """Control the stream (play/pause/speed)."""
    data = request.get_json() or {}

    if "action" in data:
        action = data["action"]
        if action == "play":
            state.is_streaming = True
        elif action == "pause":
            state.is_streaming = False
        elif action == "step":
            # Generate one record
            record = generate_normal_packet()
            result = process_record(record)
            state.event_queue.put(result)
            return jsonify({"status": "stepped", "record": result})
        elif action == "freeze_baseline":
            state.baseline_frozen = True
        elif action == "unfreeze_baseline":
            state.baseline_frozen = False
        elif action == "reset":
            reset_state()
            start_stream_thread()

    if "speed" in data:
        state.stream_speed = float(data["speed"])

    if "threshold" in data:
        state.similarity_threshold = float(data["threshold"])

    return jsonify(
        {
            "is_streaming": state.is_streaming,
            "speed": state.stream_speed,
            "baseline_frozen": state.baseline_frozen,
            "threshold": state.similarity_threshold,
        }
    )


@app.route("/api/stream/inject", methods=["POST"])
def inject_record():
    """Manually inject a record into the stream."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    record = data.get("record", data)
    result = process_record(record)
    state.event_queue.put(result)

    return jsonify({"status": "injected", "record": result})


@app.route("/api/stream/inject/attack", methods=["POST"])
def inject_attack():
    """Inject an attack pattern."""
    data = request.get_json() or {}
    count = min(data.get("count", 10), 50)

    results = []
    for _ in range(count):
        record = generate_attack_packet()
        result = process_record(record)
        state.event_queue.put(result)
        results.append(result)

    return jsonify({"status": "injected", "count": count})


@app.route("/api/stream/state")
def get_state():
    """Get current stream state."""
    baseline_pos = [0, 0, 0]
    if state.baseline_acc is not None and state.baseline_count > 0:
        # With EMA, baseline_acc is already averaged, project directly
        baseline_pos = list(state.projector.project_vector(state.baseline_acc))

    return jsonify(
        {
            "is_streaming": state.is_streaming,
            "speed": state.stream_speed,
            "baseline_frozen": state.baseline_frozen,
            "warmup_complete": state.warmup_complete,
            "warmup_packets": state.warmup_packets,
            "baseline_count": state.baseline_count,
            "baseline_position": baseline_pos,
            "recent_count": state.recent_count,
            "total_records": state.total_records,
            "anomaly_count": state.anomaly_count,
            "threshold": state.similarity_threshold,
            "history": state.history[-20:],  # Last 20 records
        }
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Holon Streaming Visualization")
    print("=" * 60)
    print("Open http://localhost:5051 in your browser")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5051, threaded=True)
