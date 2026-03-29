# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical: Python Execution

**Always** use `./scripts/run_with_venv.sh` for any Python command. Never invoke `python` or `pytest` directly.

```bash
./scripts/run_with_venv.sh python scripts/challenges/010-batch/001-solution.py
./scripts/run_with_venv.sh pytest tests/
./scripts/run_with_venv.sh pytest tests/test_primitives.py -v
./scripts/run_with_venv.sh python -m examples.showcases.log_anomaly_memory.showcase
```

## Build & Test Commands

### Python (holon library)
```bash
./scripts/run_with_venv.sh pytest tests/                          # full test suite
./scripts/run_with_venv.sh black holon/ && isort holon/           # format
./scripts/run_with_venv.sh flake8 holon/                          # lint
./scripts/run_with_venv.sh mypy holon/                            # type check
./scripts/run_with_venv.sh python scripts/server/holon_server.py  # HTTP API server
```

### Rust (holon-rs)
```bash
cd holon-rs
cargo build --release
cargo build --release --features simd    # AVX2/NEON SIMD (5x similarity speedup)
cargo test
cargo bench
cargo clippy
cargo run --example zero_hardcode_detection --release --features simd
```

### DDoS Lab (holon-lab-ddos/veth-lab)
```bash
cd holon-lab-ddos/veth-lab
./scripts/build.sh
sudo ./scripts/setup.sh
sudo ./target/release/veth-sidecar --interface veth-filter --enforce --rate-limit
```

### Website (algebraic-intelligence.dev)
```bash
cd algebraic-intelligence.dev
npm run dev      # local dev
npm run build    # static build
```

## Architecture

Holon is a **Vector Symbolic Architecture (VSA) / Hyperdimensional Computing (HDC)** library for encoding structured JSON data into high-dimensional vectors and performing algebraic operations on them. It captures *structure* (keys, nesting, role-filler binding), not semantic meaning. The Python library is the reference implementation; `holon-rs` is the production Rust port (~12x faster).

### Three-Layer Design (both Python and Rust mirror this)

**Layer 1 — `holon.kernel`** (stable primitives, rarely changes)
- `primitives.py`: ~50 VSA operations — `bind/unbind` (reversible XOR-like), `bundle` (superposition), `difference`, `negate`, `amplify`, `blend`, `prototype`, `cleanup`, `segment`, `attend`, `analogy`
- `encoder.py`: JSON → vector via role-filler binding (structure-preserving)
- `vector_manager.py`: Deterministic atom→vector allocation (same seed → same vector everywhere, enabling distributed consensus)
- `scalar.py`: Continuous value encoding — `$log` (ratios/rates), `$linear` (differences), `$circular` (angles/time-of-day)
- `accumulator.py`: Streaming composition with frequency preservation and decay
- `walkable.py`: Zero-serialization encoding interface for custom types

**Layer 2 — `holon.memory`** (the novel contribution)
- `OnlineSubspace` (CCIPCA): Learns what "normal" looks like from a stream; `residual()` scores anomalies; `anomalous_component()` gives per-field attribution
- `Engram` / `EngramLibrary`: Learned pattern snapshots for 1-packet matching; supports `StripedSubspace` encoding for multi-modal libraries

**Layer 3 — `holon.highlevel`** (convenience, changes freely)
- `HolonClient`: Unified facade with query DSL
- Special `$markers` in JSON probes: `$time`, `$log`, `$linear`, `$mode`, `$or`
- Guards (post-query field filtering) and negations (algebraic exclusion)

### Encoding Mechanics

Role-filler binding: `encode({"key": "value"})` → `bind(role_vector("key"), filler_vector("value"))`. Nested structures: permute by depth. This makes two records with the same values in different keys *structurally distinct* — the foundation of all anomaly detection.

Scalars use magnitude encoding: `$log` for ratios (request rates, byte counts), `$linear` for absolute differences, `$circular` for time-of-day/angles.

### Dimensions
- 1024: Quick demos
- 4096: Default (good accuracy/speed balance)
- 8192: High-complexity structured data

### Labs

**`holon-lab-trading`** — Active development. Self-supervised BTC trader in Rust. See full section below.

**`holon-lab-ddos`** — Kernel-level DDoS detection:
- XDP programs (`veth-lab/filter-ebpf/`) run at line rate
- eBPF tail-call tree: 1M rules evaluated in O(tree depth) per packet
- `veth-lab/sidecar/`: Holon-rs anomaly engine → derives rules → compiles to eBPF tree
- Blue/green atomic tree deployment (zero-downtime rule updates)
- `veth-lab/` is the primary development env (reproducible veth pairs); `http-lab/` is web-based

**`holon-lab-baseline`** — Realistic traffic generation for training anomaly detectors (WordPress + 23 macvlan IPs + Playwright + Ollama LLM agents)

### Challenge Batches (`scripts/challenges/`, `docs/challenges/`)

18 numbered batches (001–018) demonstrating progressive capabilities. Each batch has solution scripts and a corresponding `docs/challenges/` write-up with learnings. Key milestones:
- 009: 1M records, 94.5% accuracy, 25k encode/sec
- 010–012: Network anomaly/DDoS, F1=1.000, zero-hardcode detection
- 013: Rate limiting derived purely from vector algebra
- 017: Online subspace (CCIPCA) with per-field attribution
- 018: Eigenvalue prefiltering, multimodal engram libraries, taxonomy clustering

### What VSA Cannot Do

Constraint satisfaction (Sudoku, SAT), exact matching, NP-hard optimization. Holon uses similarity over equality — this is both the capability and the limitation. The `docs/ASSESSMENT.md` documents this honestly.

### HTTP API Server

`scripts/server/holon_server.py` exposes:
- `POST /api/v1/items` — insert JSON
- `POST /api/v1/search` — search by probe JSON
- `POST /api/v1/vectors/encode` — get vector for data
- `POST /api/v1/vectors/prototype` — compute prototype from vectors

---

## Trading Lab (`holon-lab-trading/`)

**Active development focus.** Self-organizing BTC trading enterprise in Rust. See `holon-lab-trading/CLAUDE.md` for full details.

### Quick Reference

```bash
cd holon-lab-trading

./enterprise.sh build                                             # compile (release)
./enterprise.sh run --max-candles 5000 --asset-mode hold          # quick run
./enterprise.sh test 100000 --asset-mode hold --name my-run       # benchmark → runs/
./enterprise.sh kill                                              # kill switch
```

Kill switch: `touch holon-lab-trading/trader-stop`

### Architecture

Six primitives (atom, bind, bundle, cosine, journal, curve), two templates (prediction + reaction), one tree. Observers predict direction → manager aggregates opinions → risk modulates sizing → treasury executes. The `wat/` directory contains domain specifications. The `src/` module layout mirrors the enterprise tree.

### Key Files

- `Cargo.toml`: `enterprise` crate at repo root, depends on `holon` (local `../holon-rs`, `features = ["simd"]`)
- `src/bin/enterprise.rs`: The heartbeat — orchestrates modules, doesn't define encoding
- `src/thought/`: Layer 0 (candle → thoughts via PELT + vocabulary)
- `src/market/`: Manager encoding + Observer struct
- `src/risk/`: Risk branch subspaces
- `src/vocab/`: Thought vocabulary modules (oscillators, flow, persistence)
- `BOOK.md`: The full story — architecture, philosophy, results
- `runs/`: Run ledgers and logs (append-only, never delete)
- `data/analysis.db`: 652,608 5-minute BTC candles (Jan 2019–Mar 2025)
