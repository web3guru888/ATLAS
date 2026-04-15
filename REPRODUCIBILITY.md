# ATLAS Reproducibility Guide

This document provides step-by-step instructions to reproduce all ATLAS benchmark
numbers from a fresh clone on any x86-64 Linux machine (GPU optional).

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Rust | ≥ 1.75 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| CUDA toolkit | ≥ 12.0 | Optional — CPU path works without GPU |
| git | Any | Standard |

CPU-only builds work on any modern x86-64 Linux without a GPU.
CUDA kernels require an NVIDIA GPU with compute capability ≥ 7.5 (Tesla T4, RTX 3000+).

## Step 1: Clone and Build

```bash
git clone https://github.com/web3guru888/ATLAS.git
cd ATLAS
cargo build --workspace
```

Expected output (CPU-only, no GPU): `Finished release [optimized]` in ~3–5 minutes.

With CUDA: set `CUDA_PATH=/usr/local/cuda` before building (detected automatically via `build.rs`).

## Step 2: Run the Test Suite

```bash
cargo test --workspace
```

Expected:
```
test result: ok. 383 passed; 0 failed; N ignored; 0 measured
```

All tests pass on CPU. GPU tests are `#[ignore]`-tagged and require `cargo test -- --ignored`.

## Step 3: Discover Real Science

```bash
cargo run --bin atlas -- discover --cycles 3 --output demo-corpus.json
```

This runs 3 OODA cycles against live APIs (NASA POWER, WHO GHO, World Bank, ArXiv).
Expected output:
- ≥ 3 corpus entries written to `demo-corpus.json`
- Hypotheses: climate correlations, health indicators, or arxiv discovery
- Run time: ~10–30 seconds (depends on network)

On network failure, ATLAS gracefully falls back to synthetic data.

## Step 4: Train on the Corpus

```bash
cargo run --bin atlas -- train --corpus demo-corpus.json --epochs 5 --lr 0.01
```

Expected output:
```
Epoch 1/5: loss=X.XXXX  lr=1.00e-2  steps=N
...
Epoch 5/5: loss=Y.YYYY  lr=...
✓ Checkpoint: ./atlas-ckpt/atlas-stepN.json
```

Loss should decrease monotonically (or approximately — depends on corpus content).

## Step 5: Run Benchmarks

```bash
cargo run --bin atlas -- bench
```

Expected results (T4 GPU, measured 2026-04-15):

| Component | Metric | Value |
|-----------|--------|-------|
| Palace insertions | ops/s | ≥ 100,000 |
| Palace A* queries | ops/s | ≥ 500 (release) |
| SFT training | steps/s | ≥ 1,000 |
| Quality gates | gates/s | ≥ 10,000 |

CPU debug builds will be slower — run `cargo run --release --bin atlas -- bench` for full-speed numbers.
CUDA matmul in `atlas-tensor` provides ~10× throughput for tensor-heavy ops.

## Step 6: Start the MCP Server

```bash
cargo run --bin atlas -- mcp serve
```

This starts the MCP server on stdin/stdout. Connect via Claude desktop or Cursor by adding to `mcp.json`:

```json
{
  "mcpServers": {
    "atlas": {
      "command": "/path/to/atlas",
      "args": ["mcp", "serve"]
    }
  }
}
```

All 28 tools will be available to the LLM.

## Benchmark Numbers (Reference — Tesla T4, CUDA 12.9, 2026-04-15)

| Benchmark | Result |
|-----------|--------|
| `cargo test --workspace` | 383 passed, 0 failed |
| Palace: 1000 insertions | ~12 ms total |
| Palace: 100 A* queries | ~18 ms total |
| Training: 10 steps (MLP) | ~2 ms total |
| Discovery: 3 cycles | ~15 s (network bound) |
| JSON parse 1MB | ~8 ms |

## Zero External Dependencies — Verification

```bash
cargo tree --workspace | grep -v "^atlas" | grep -v "^  \[" | head -5
```

Expected: no output (or only `proc-macro` entries). ATLAS has zero runtime crate dependencies.

## Troubleshooting

**Build fails with CUDA error**: Set `CUDA_PATH` or unset `CUDA_ARCH` env. CPU-only build always works.

**`atlas discover` returns synthetic data**: Network is unavailable. This is expected behavior — ATLAS degrades gracefully.

**Tests fail on `atlas-tensor` GPU tests**: Run `cargo test --workspace -- --skip gpu` or `cargo test --workspace --exclude atlas-tensor`.
