# Holon 3D Visualization

Interactive visualization of Holon's high-dimensional vector space projected to 3D.

https://github.com/user-attachments/assets/89e1524b-45de-4cda-9f36-dd14e8b53400

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

## Future: Engram Manifold Visualization

Visualize engram subspaces as 3D manifolds — see how attack signatures relate to the baseline traffic manifold and to each other.

**Ref**: [Grok thread on manifolds in 3D](https://grok.com/share/bGVnYWN5LWNvcHk_6441092c-4f08-487a-8df9-ff2e44b61504)

### Approach

The key insight: sample thousands of points **directly on each engram's k-dimensional linear manifold** (using the stored basis vectors and eigenvalues), then run **UMAP or PaCMAP once on the combined k-D coefficients** to produce 3D point clouds. This preserves the actual manifold geometry — no PCA smearing.

```
For each engram:
  1. Extract basis vectors (k × 4096) and eigenvalues from OnlineSubspace
  2. Sample coefficients: N(0, sqrt(eigenvalue) * scale) for each of k dims
  3. Optionally add small Gaussian noise for visible "thickness"
  4. These k-D coefficients ARE the manifold — no 4096D needed for viz

Combined:
  5. Stack all engrams' k-D coefficients
  6. UMAP(n_components=3) on the combined matrix → 3D point clouds
  7. Each engram is a distinct colored cloud; separation = distinctness
```

### What to visualize

- **Baseline manifold**: the learned normal subspace (OnlineSubspace principal components) as a colored surface/point cloud
- **Engram manifolds**: each stored attack subspace as a distinct colored cloud, showing how it diverges from baseline and from other attacks
- **Live traffic**: incoming vectors projected to coefficients and plotted as points (normal=green, anomalous=red, engram-matched=orange)
- **Rule attribution**: highlight which region of the anomalous manifold each auto-generated EDN rule targets — connecting the geometric view (VSA subspace divergence) to the symbolic view (constraint expressions)

### Data sources

Use the **Python holon reference** directly — no Rust bridge needed. Fabricate realistic traffic and attacks in Python, run through `OnlineSubspace` / `EngramLibrary`, mint engrams, and visualize. Same math, same subspace structure as the Rust implementation.

### Projection upgrade

The current `projector.py` and `streaming.py` use **random orthogonal projection** from 4096D → 3D. This throws away almost all structure — 4093 dimensions of information discarded randomly.

The better path: **4096D → k-D (subspace coefficients) → 3D (UMAP/PaCMAP)**. The k-D coefficients (where k=32 or similar) already capture the meaningful variance learned by OnlineSubspace. UMAP only has to untangle 32 dimensions instead of 4096, preserving neighborhood structure and manifold geometry.

This upgrade applies to both the existing streaming demo and the new engram viz — the existing demos should also switch to subspace-coefficient-based projection for dramatically better visual fidelity.

### Implementation options

1. **Plotly quick-win** — `engram_viz.py` fabricates engrams in Python, samples manifold points, UMAP → 3D, interactive Plotly scatter. Fastest path to seeing the shapes.
2. **Upgrade existing viz** — Update `projector.py` and `streaming.py` to use subspace-coefficient-based projection (k-D → UMAP → 3D) instead of random orthogonal projection. Same Three.js renderer, dramatically better visual fidelity.
3. **Three.js point clouds** — Export `engrams_3d.json`, load as PointCloud or InstancedMesh in the existing Three.js renderer.
4. **Voxel blobs** — Marching cubes / occupancy grid on the 3D points for filled volume rendering.
5. **Live streaming** — Extend `streaming.py` to show new engrams appearing in real time as attacks are detected and minted.

## Technical Details

- **Projection**: Random orthogonal projection from 4096D → 3D
- **Rendering**: Three.js with OrbitControls
- **Backend**: Flask with SSE for streaming
- **Similarity**: Cosine similarity between vectors
