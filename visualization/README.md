# Holon 3D Visualization

Interactive visualization of Holon's high-dimensional vector space projected to 3D.

## Quick Start

```bash
# Static explorer (port 5050)
./scripts/run_with_venv.sh python visualization/app.py

# Streaming demo (port 5051)
./scripts/run_with_venv.sh python visualization/streaming.py
```

## Components

### Static Explorer (`/` on port 5050)
- Explore encoded data structures in 3D space
- Hierarchical view: atoms → bindings → composites
- Click to highlight relationships
- Demo buttons for sample packet data

### Streaming Demo (`/` on port 5051)
- Real-time anomaly detection visualization
- Dual-pane: baseline evolution (left) + incoming traffic (right)
- Detection accuracy tracking (TP/TN/FP/FN)

**Detection approach** (inspired by veth-lab):
1. Warmup phase: Learn baseline with EMA (300 packets)
2. Freeze baseline after warmup
3. Flag anomalies when similarity < 0.35
4. Attack patterns: SYN flood, UDP flood, ICMP flood

## Technical Details

- **Projection**: Random orthogonal projection from 4096D → 3D
- **Rendering**: Three.js with OrbitControls
- **Backend**: Flask with SSE for streaming
- **Similarity**: Cosine similarity between vectors
