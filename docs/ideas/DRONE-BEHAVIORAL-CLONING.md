# Autonomous Drone Flight via VSA Behavioral Cloning

*Captured: February 2026*
*Status: Idea — not yet prototyped*

## Core Idea

Use drone flight simulators (designed to train human pilots) as a training
ground for a VSA/HDC-based autonomous flight system. The human flies; the
system watches, learns, and remembers. Each flight situation is captured as
an **engram** — a stored memory trace binding what the drone saw + its flight
state to what the human did. At deployment, the system recognizes situations
from its engram library and replays the associated control actions.

This is behavioral cloning, but instead of training a neural network on
thousands of demonstrations, the system builds a library of engram memories
that match via subspace projection — giving generalization, transparency,
and graceful failure.

## Why This Architecture (Not a Neural Net)

| Property | Neural Net (end-to-end) | VSA Engram Library |
|---|---|---|
| Training data | Thousands of demonstrations | Handful per maneuver |
| Novel situation | Always outputs something (hallucination) | Knows it doesn't know (high residual, no match) |
| Correction | Retrain entire model | Delete bad engram, add good one |
| Explainability | Black box | "Matched engram X from flight Y at time Z" |
| Generalization | Learned implicitly | Proven — 100% variant resilience across parameter changes |
| Deployment | GPU or specialized hardware | CPU only, embedded-friendly |

## Data Model

Each timestep in a training flight produces a structured record:

```python
{
    # What the drone sees (features extracted from camera frame)
    "patch_0_0": {"edge": "horizontal", "color": "green", "intensity": 0.7},
    "patch_0_1": {"edge": "none", "color": "blue", "intensity": 0.9},
    # ... NxN grid of patch descriptors

    # Flight state (device measurements / virtual IMU)
    "roll": 3.2,         # degrees
    "pitch": -1.5,       # degrees
    "yaw": 180.0,        # degrees (circular)
    "altitude": 45.0,    # meters
    "airspeed": 12.3,    # m/s
    "accel_x": 0.1,      # g
    "accel_y": -0.3,
    "accel_z": 9.8,
    "vertical_speed": -0.5,

    # What the human did (control inputs — the "label")
    "throttle": 0.7,
    "stick_roll": 0.1,
    "stick_pitch": -0.2,
    "stick_yaw": 0.0,
}
```

The human's control input is encoded into the same vector via VSA binding.
`bind(scene, action)` means the scene and the action are *associated*. The
subspace learns the manifold of human-piloted flight — not just flight states,
but flight states paired with what a skilled human does in those states.

## Encoding Strategy

Holon's existing primitives map directly:

- **Structured data** → `client.encode(record)` — the core encoding
- **Continuous values** → `encode_scalar_log()` for rates/accelerations,
  `encode_circular()` for heading/yaw/bearing (wraps at 360)
- **Spatial binding** → `bind(patch_features, position_vector)` — located features
- **Multi-modal fusion** → keep separate vectors per modality (see below)

### Multi-Modal Architecture (Array of Vectors)

Don't collapse everything into one vector. Maintain separate subspaces per modality:

```
IMU subspace     — learns normal flight dynamics
GPS subspace     — learns normal trajectory patterns
Scene subspace   — learns normal visual environments
Control subspace — learns normal human control patterns
```

Each modality has its own subspace and engram library. Anomaly in any one
triggers investigation. The *combination* tells you what's happening:

- IMU anomaly + scene normal → turbulence (known flight condition)
- IMU normal + scene anomaly → new visual landmark (learn it)
- All anomalous → unknown situation (hover, request human input)
- Scene matches engram + IMU matches engram → replay stored controls

The decision layer combines signals from all modalities — same pattern as
the Rete tree combining predicates in the DDoS lab.

## Operational Lifecycle

Identical to the DDoS engram lifecycle (experiment 018), with different
domain-specific actions:

```
Phase 1: Human flies simulator
         System encodes every timestep
         Subspaces learn normal flight manifold

Phase 2: Identify distinct maneuvers (clustering on control subspace)
         Mint engrams for each maneuver type
         Decorate with control input profiles

Phase 3: Autonomous mode
         Encode current scene + flight state (no control inputs)
         Match against engram library
         Known situation → replay stored control profile
         Unknown situation → hover + alert human + learn

Phase 4: Correction loop
         Human overrides bad decision → old engram deleted or deprioritized
         Human demonstrates correct maneuver → new engram minted
         No retraining, no gradient descent, immediate effect
```

## Parallel to DDoS Lab Architecture

The structural parallel between `holon-lab-ddos/veth-lab` and a drone system
is deep — it's the same architecture with different domain payloads:

| Component | DDoS Lab | Drone Lab |
|---|---|---|
| Input stream | Network packets (100k+ pps) | Sensor frames (30-60 fps) |
| Feature extraction | Packet fields → structured dict | Image patches + IMU → structured dict |
| Encoding | `client.encode(packet_dict)` | `client.encode(flight_dict)` |
| Detection subspace | Normal traffic manifold | Normal flight manifold |
| Engram library | Known attack patterns | Known flight situations |
| Engram metadata | EDN mitigation rule | Control input profile |
| Decision engine | Rete tree in eBPF/XDP | Rete tree in flight controller |
| Action | `(drop)`, `(rate-limit N)` | `(climb N)`, `(turn N)`, `(hover)` |
| Enforcement | XDP program at wire speed | Flight controller firmware |
| Atomic updates | Blue/green tree flip | Blue/green maneuver table swap |
| Failure mode | Drop suspicious packets | Hover and alert human |

## Why Starting with Simulators Is Right

1. **Structured telemetry for free** — simulators provide IMU, GPS, altitude,
   airspeed as structured data. No sensor noise, no calibration. Prove the
   decision loop works before dealing with real hardware.

2. **Controlled visual environment** — rendered graphics are consistent. The
   visual vocabulary is finite. Lighting is predictable. You're not solving
   ImageNet — you're distinguishing "approaching landing pad" from "cruising
   over terrain" from "obstacle ahead."

3. **Safe iteration** — wrong decision crashes a virtual drone, not a real one.
   Thousands of training flights cost nothing.

4. **Ground truth available** — simulators can provide depth maps, segmentation,
   object labels alongside RGB frames. Use these to validate that your feature
   extraction captures the right information.

## Image Processing Approach

Divide each camera frame into an NxN grid of patches. Per patch, compute
cheap CPU-based descriptors:

- Color histogram (dominant hue, saturation, brightness)
- Edge orientation (Sobel/Canny → dominant direction)
- Texture descriptor (LBP — local binary pattern)
- Mean intensity + variance

Each patch becomes a structured dict. The full frame is a dict of dicts.
Holon encodes nested structures natively.

Timing budget: outdoor drones run 10-30Hz control loops (33-100ms per tick).
Patch feature extraction: ~5-15ms. VSA encode + residual + engram match:
~3-5ms. Leaves 15-80ms headroom for decision logic and actuation.

## Validation Path (Before Building Anything)

### Step 0: Synthetic telemetry experiment

Generate realistic flight state sequences (IMU + altitude + airspeed) for
distinct maneuvers: takeoff, cruise, turn, approach, land, obstacle avoidance.
Prove the subspace + engram system discriminates between them from telemetry
alone. No simulator needed — just structured data like the 017-batch experiments.

If telemetry alone isn't enough, vision won't save it.

### Step 1: Simulator telemetry integration

Connect to a simulator (candidates: AirSim, Gazebo, jMAVSim, FlightGear).
Capture telemetry stream while human flies. Encode and learn in real-time.
Verify subspace convergence, maneuver engram formation.

### Step 2: Add visual features

Extract patch features from simulator camera frames alongside telemetry.
Verify that visual features sharpen discrimination (reduce residual overlap
between maneuver types).

### Step 3: Closed-loop autonomous

System controls the simulator using engram-matched control profiles. Start
with simple scenarios (straight flight, gentle turns). Expand to full
maneuver vocabulary.

### Step 4: Hardware transfer

If simulation works, the CPU-only architecture means deployment on embedded
hardware (Raspberry Pi, Jetson Nano in CPU-only mode, STM32 with sufficient
RAM) is viable. The engram library serializes to JSON, loads at boot.

## Memory Budget

At 4096D float64:
- One vector: 32KB
- One engram (k=32 components + mean + metadata): ~1MB
- 1,000 engrams: ~1GB
- 10,000 engrams: ~10GB

A drone with 2-4GB of RAM can carry thousands of flight situation memories.
For constrained scenarios (fixed flight paths, known environments), a few
hundred engrams may suffice.

## Open Questions

1. **Temporal context** — a single timestep doesn't capture "I've been
   descending for 3 seconds." Sequence encoding (`encode_sequence()`) or
   sliding window bundling could provide temporal context. How much history
   is needed?

2. **Continuous control interpolation** — engrams store discrete control
   snapshots. Between exact matches, do you interpolate between the two
   nearest engrams' control profiles? Blend by inverse residual?

3. **Multi-resolution features** — coarse patches (4x4) for fast scene
   recognition, fine patches (16x16) for precise obstacle localization.
   Multiple subspaces at different resolutions?

4. **Transfer between environments** — an engram minted in Simulator A
   (desert terrain). Does it match in Simulator B (forest)? The structural
   features (obstacle ahead, clear sky above) might transfer even if the
   visual textures don't. Needs testing.

5. **Reaction time** — DDoS rules can tolerate milliseconds of latency.
   Drone obstacle avoidance needs sub-100ms response. Is the engram matching
   pipeline fast enough, or do high-priority maneuvers (emergency climb,
   collision avoidance) need to bypass the library and use hardcoded rules?

## Relationship to Holon

All of this uses existing holon primitives — no new library code is needed
beyond what's already in `holon/subspace.py` and `holon/engram.py`. The
drone-specific logic (feature extraction, control mapping, simulator
integration) lives in experiment scripts and a future `holon-lab-drone/`
workspace, keeping the core library generic.

The Rete tree compiler and eBPF walker from `holon-lab-ddos/veth-lab/filter/`
could be adapted for flight controller rule enforcement, but that's a future
step after the decision loop is validated.
