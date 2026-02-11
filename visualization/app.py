"""
Flask app for Holon 3D visualization.

Serves the Three.js frontend and provides REST API for
adding/querying points in the visualization.
"""

import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory
from projector import HolonProjector

app = Flask(__name__, static_folder="static")

# Global projector instance
projector = HolonProjector(dimensions=4096)


# ============================================================================
# Static file serving
# ============================================================================


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


# ============================================================================
# API endpoints
# ============================================================================


@app.route("/api/points", methods=["GET"])
def get_points():
    """Get all current points for visualization."""
    return jsonify(
        {
            "points": projector.get_all_points(),
            "edges": projector.get_edges(),
            "stats": projector.stats(),
            "timestamp": time.time(),
        }
    )


@app.route("/api/atom", methods=["POST"])
def add_atom():
    """Add an atom to the visualization."""
    data = request.get_json()
    atom_str = data.get("atom")
    metadata = data.get("metadata", {})

    if not atom_str:
        return jsonify({"error": "atom is required"}), 400

    point = projector.add_atom(atom_str, metadata)
    return jsonify({"point": point.to_dict()})


@app.route("/api/composite", methods=["POST"])
def add_composite():
    """Add a composite (encoded data structure) to the visualization."""
    data = request.get_json()
    composite_id = data.get("id")
    composite_data = data.get("data")
    metadata = data.get("metadata", {})

    if not composite_id or composite_data is None:
        return jsonify({"error": "id and data are required"}), 400

    point = projector.add_composite(composite_id, composite_data, metadata)
    return jsonify({"point": point.to_dict()})


@app.route("/api/accumulator", methods=["POST"])
def add_accumulator():
    """Add/update an accumulator state."""
    data = request.get_json()
    acc_id = data.get("id")
    # Accumulator vector as list of floats
    acc_vector = data.get("vector")
    metadata = data.get("metadata", {})

    if not acc_id or acc_vector is None:
        return jsonify({"error": "id and vector are required"}), 400

    import numpy as np

    acc_array = np.array(acc_vector, dtype=np.float64)
    point = projector.add_accumulator(acc_id, acc_array, metadata)
    return jsonify({"point": point.to_dict()})


@app.route("/api/touch/<path:point_id>", methods=["POST"])
def touch_point(point_id):
    """Update last_seen time for a point."""
    point = projector.touch(point_id)
    if point:
        return jsonify({"point": point.to_dict()})
    return jsonify({"error": "point not found"}), 404


@app.route("/api/similarity", methods=["GET"])
def get_similarity():
    """Get similarity between two points."""
    id1 = request.args.get("id1")
    id2 = request.args.get("id2")

    if not id1 or not id2:
        return jsonify({"error": "id1 and id2 are required"}), 400

    sim = projector.similarity(id1, id2)
    if sim is None:
        return jsonify({"error": "one or both points not found"}), 404

    return jsonify({"id1": id1, "id2": id2, "similarity": sim})


@app.route("/api/clear", methods=["POST"])
def clear_all():
    """Clear all points."""
    projector.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/cleanup", methods=["POST"])
def cleanup_old():
    """Remove points that haven't been seen recently."""
    data = request.get_json() or {}
    max_age = data.get("max_age", 60.0)
    removed = projector.remove_old(max_age)
    return jsonify({"removed": removed, "count": len(removed)})


@app.route("/api/demo/packet", methods=["POST"])
def demo_packet():
    """
    Demo endpoint: simulate a network packet with full encoding hierarchy.

    Creates three levels:
    - Level 0 (atoms): raw values like "src_ip", "10.0.0.1", "80"
    - Level 1 (bindings): bind(key, value) pairs
    - Level 2 (composite): bundle of all bindings
    """
    data = request.get_json()

    atoms_added = []
    bindings_added = []
    binding_ids = []

    for field in ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]:
        if field in data:
            value = data[field]
            value_str = str(value)

            # Level 0: Add raw atoms (the key and the value separately)
            key_atom = projector.add_atom(
                field,
                {"role": "key", "value": field},
            )
            atoms_added.append(key_atom.to_dict())

            value_atom = projector.add_atom(
                value_str,
                {"role": "value", "value": value_str},
            )
            atoms_added.append(value_atom.to_dict())

            # Level 1: Add binding (key ⊙ value)
            binding_id = f"{field}:{value}"
            binding = projector.add_binding(
                binding_id,
                key_atom=field,
                value_atom=value_str,
                metadata={
                    "field": field,
                    "value": value,
                    "encoding": f"bind({field}, {json.dumps(value)})",
                    "operation": "⊙ (element-wise multiply)",
                },
            )
            bindings_added.append(binding.to_dict())
            binding_ids.append(binding_id)

    # Level 2: Add composite (bundle of all bindings)
    packet_id = f"pkt_{int(time.time() * 1000)}"
    composite = projector.add_composite(
        packet_id,
        data,
        metadata={
            "type": "packet",
            "source_data": json.dumps(data, indent=2),
            "encoding": f"bundle([{', '.join(binding_ids)}])",
            "operation": "Σ + threshold (majority vote)",
        },
        components=binding_ids,
    )

    return jsonify(
        {
            "atoms": atoms_added,
            "bindings": bindings_added,
            "composite": composite.to_dict(),
            "hierarchy": {
                "level_0": "atoms (raw vectors)",
                "level_1": "bindings (key ⊙ value)",
                "level_2": "composite (bundle of bindings)",
            },
        }
    )


@app.route("/api/demo/random", methods=["POST"])
def demo_random():
    """Generate random demo data with full 3-level hierarchy."""
    import random

    data = request.get_json() or {}
    count = min(data.get("count", 10), 100)  # Cap at 100

    atoms_added = []
    bindings_added = []
    composites_added = []

    for i in range(count):
        # Random "document"
        doc = {
            "type": random.choice(["order", "user", "event", "log"]),
            "id": random.randint(1000, 9999),
            "status": random.choice(["active", "pending", "complete"]),
            "priority": random.randint(1, 5),
        }

        binding_ids = []

        # Create full hierarchy for each field
        for key, value in doc.items():
            value_str = str(value)

            # Level 0: Add raw atoms
            key_atom = projector.add_atom(
                key,
                {"role": "key", "value": key},
            )
            atoms_added.append(key_atom.to_dict())

            value_atom = projector.add_atom(
                value_str,
                {"role": "value", "value": value_str},
            )
            atoms_added.append(value_atom.to_dict())

            # Level 1: Add binding
            binding_id = f"{key}:{value}"
            binding = projector.add_binding(
                binding_id,
                key_atom=key,
                value_atom=value_str,
                metadata={
                    "field": key,
                    "value": value,
                    "encoding": f"bind({key}, {json.dumps(value)})",
                },
            )
            bindings_added.append(binding.to_dict())
            binding_ids.append(binding_id)

        # Level 2: Add composite
        comp = projector.add_composite(
            f"doc_{i}_{int(time.time() * 1000)}",
            doc,
            metadata={
                "type": "document",
                "source_data": json.dumps(doc, indent=2),
                "encoding": f"bundle([{', '.join(binding_ids[:2])}...])",
            },
            components=binding_ids,
        )
        composites_added.append(comp.to_dict())

    return jsonify(
        {
            "atoms_added": len(atoms_added),
            "bindings_added": len(bindings_added),
            "composites_added": len(composites_added),
        }
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Holon 3D Visualization Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=5050, help="Port to run on")
    args = parser.parse_args()

    print("=" * 60)
    print("Holon 3D Visualization Server")
    print("=" * 60)
    print(f"Dimensions: {projector.dimensions}")
    print(f"Open http://localhost:{args.port} in your browser")
    print("=" * 60)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
