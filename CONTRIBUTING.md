# Contributing to ATLAS

Thank you for your interest in ATLAS! This document explains how to build, test, and contribute to the project.

> **ATLAS is an open source contribution from [OpenHub Research](https://openhubresearch.org/) (Thailand)** — authored by Robin Dey. Contributions are welcome from the community under the terms below.

## Prerequisites

- **Rust 1.75+** (stable toolchain)
- **CUDA 12.x + nvcc** (optional — only needed for `atlas-tensor` GPU tests)
- **Git** for version control

## Quick Start

```bash
# Clone the repo
git clone https://github.com/web3guru888/ATLAS.git
cd ATLAS

# Build all CPU crates (excludes atlas-tensor which needs CUDA)
cargo build --workspace --exclude atlas-tensor

# Run all tests
cargo test --workspace --exclude atlas-tensor

# Run with GPU support (requires CUDA)
cargo test --workspace
```

## Project Architecture

ATLAS is a pure-Rust workspace with **zero external crate dependencies** (the SQLite principle). All algorithms are implemented from scratch.

```
crates/
├��─ atlas-core/      # Foundation: error types, traits, config, bench harness
├── atlas-tensor/    # Tensor engine: f32/f16, matmul, CUDA FFI
├── atlas-grad/      # Autograd: reverse-mode AD tape
├── atlas-optim/     # AdamW optimizer, cosine LR scheduler
├── atlas-quant/     # INT4/INT8 quantization, QLoRA
├── atlas-model/     # Transformer: RMSNorm, RoPE, GQA, SwiGLU, OLMo 3 arch
├── atlas-tokenize/  # BPE tokenizer (sentencepiece port)
├── atlas-palace/    # GraphPalace memory: pheromones, A*, agents, Active Inference
├── atlas-mcp/       # MCP server: 28 palace tools via JSON-RPC 2.0 stdio
├── atlas-trm/       # TRM-CausalValidator (7M params)
├── atlas-causal/    # PC/FCI causal inference
├── atlas-bayes/     # Bayesian confidence scoring
├── atlas-http/      # HTTP client (raw libc syscalls)
├── atlas-json/      # JSON parser (zero-dep, recursive descent)
├── atlas-astra/     # ASTRA OODA discovery engine
├── atlas-corpus/    # LiveDiscoveryCorpus + pheromone sampler
├── atlas-zk/        # ZK Schnorr proofs: provenance chain
└── atlas-cli/       # CLI: train, discover, eval, prove
```

For the full architecture, see [CHARTER.md](CHARTER.md).

## Build Order

The crates have a dependency order (7 stages):

1. **Stage 1**: `atlas-core` → `atlas-tensor` → `atlas-grad` → `atlas-optim` → `atlas-quant`
2. **Stage 2**: `atlas-model` → `atlas-tokenize`
3. **Stage 3**: `atlas-palace`
4. **Stage 4**: `atlas-trm`
5. **Stage 5**: `atlas-http` → `atlas-json` → `atlas-bayes` → `atlas-causal` → `atlas-zk` → `atlas-astra`
6. **Stage 6**: `atlas-corpus`
7. **Stage 7**: `atlas-cli`

## Code Style

### Formatting

We use standard `rustfmt`. Before submitting, run:

```bash
cargo fmt --all
```

### Linting

All code must pass clippy with zero warnings:

```bash
cargo clippy --workspace --exclude atlas-tensor -- -D warnings
```

### Documentation

- All public items must have doc comments (`#![warn(missing_docs)]`)
- Use `///` for item docs, `//!` for module docs
- Include examples in doc comments where practical

### The Zero-Dependency Rule

**Do not add external crate dependencies.** This is a core design principle, not a suggestion.

If you need functionality from an external crate, port the algorithm from its original source (paper, reference implementation) and give proper attribution. If you're unsure, open an issue first.

CUDA is the one exception — it's called via raw FFI from `build.rs` + `kernels/*.cu` (a system dependency, not a Rust crate).

## Testing

### Run all tests

```bash
# CPU only (excludes atlas-tensor)
cargo test --workspace --exclude atlas-tensor

# All tests (requires CUDA GPU)
cargo test --workspace
```

### Run specific crate tests

```bash
cargo test -p atlas-palace
cargo test -p atlas-zk
```

### Run benchmarks

Benchmarks are `#[ignore]` tests that measure performance. They don't run in normal CI:

```bash
# All benchmarks
cargo test --workspace --exclude atlas-tensor -- --ignored --nocapture

# Specific crate benchmarks
cargo test -p atlas-palace --test benchmarks -- --ignored --nocapture
cargo test -p atlas-json --test benchmarks -- --ignored --nocapture
cargo test -p atlas-zk --test benchmarks -- --ignored --nocapture
cargo test -p atlas-model -- --ignored --nocapture
```

### Writing benchmarks

Use the built-in `atlas_core::bench::Bench` harness:

```rust
use atlas_core::bench::Bench;

#[test]
#[ignore]
fn bench_my_operation() {
    let b = Bench::run("my_op", 1000, || {
        std::hint::black_box(/* your operation */);
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for any new functionality
3. **Run the full test suite** to ensure nothing breaks
4. **Run clippy and fmt** to ensure code quality
5. **Update documentation** if you change public APIs
6. **Submit a PR** with a clear description of what and why

### PR Title Convention

```
[crate-name] Brief description of change

Examples:
[atlas-palace] Add batch pheromone deposit method
[atlas-model] Fix RoPE at position 0 edge case
[atlas-core] Add Bench::run_with_setup for benchmarks needing state
```

### What We Look For

- **Does it compile with zero warnings?** (`clippy -D warnings`)
- **Are there tests?** We aim for >80% coverage.
- **Does it follow the zero-dep rule?** No new external crates.
- **Is the documentation clear?** Doc comments on all public items.
- **Is the commit history clean?** Squash fixup commits.

## Reporting Issues

Please include:
- **Environment**: OS, Rust version (`rustc --version`), GPU (if relevant)
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Error messages** (full output)

## License

By contributing, you agree that your contributions will be licensed under:
- **Apache 2.0** for code (see [LICENSE-CODE](LICENSE-CODE))
- **CC BY 4.0** for documentation (see [LICENSE](LICENSE))

## Questions?

Open an issue or reach out to the maintainers. We're happy to help!

---

ATLAS is maintained by **OpenHub Research** (Thailand) — https://openhubresearch.org/  
Project website: https://atlasagi.org
