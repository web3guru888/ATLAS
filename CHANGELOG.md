# Changelog

All notable changes to ATLAS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
ATLAS uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [3.0.0-alpha.1] — 2026-04-16

### Added
- **`atlas-api`** — OpenAI-compatible HTTP inference endpoint (Issue #4); `atlas api serve` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`; SSE streaming, CORS, echo mode; 40 tests
- **`PalaceBackend` trait** in atlas-palace — pluggable storage architecture; `Palace` implements `PalaceBackend`; trait object safe (`dyn PalaceBackend`); enables LadybugDB / alternative backends without API changes
- **GPU-resident forward pass** in atlas-model — hidden state stays in VRAM between tokens; reduced PCIe transfers from 211 to 2 per token; pre-pinned weight upload at model load time
- **GPU AdamW kernel** (`atlas_adamw_step`) — CUDA kernel for optimizer step; weight update entirely on GPU
- **CUDA ops kernels** — `rmsnorm_forward`, `rope_forward`, `silu_mul_forward` all implemented as CUDA kernels; `GpuVec` activation buffer; `sgemm_vec` zero-copy path
- **SmolLM2-135M GPU inference validated** — 29.3 tok/s on Tesla T4 (vs 5.3 tok/s CPU = 5.8× speedup); 694 MiB VRAM; 71–80% GPU utilization
- **CUDA portability fix** — use `rsqrtf()` instead of `__rsqrtf()` across all kernels
- **`McpConnectionPool`** in atlas-mcp — lazy pool (max 5 connections, 5-min idle eviction); `acquire()`/`release()` API; prevents connection leaks across concurrent MCP clients; 5 tests
- **`session_id`** on `Drawer` in atlas-palace — cross-session knowledge retrieval; sessions tag every pheromone deposit for replay / deduplication
- **`PalaceConfig`** struct in atlas-palace — centralized palace configuration (decay rates, MMAS ceiling, A* weights, session management); `calibrated_decay_rate()` API
- **`DeepSupervisionTrainer`** in atlas-corpus — N_sup=4..16 forward passes per batch; summed loss across all supervision points; latent carry between passes; `loss_trace` telemetry; TRM arXiv:2510.04871 validated (>75% gain from N_sup); 10 tests
- **Horn-clause tractable safety constitution** in atlas-safety — 8 safety principles across 4 non-overlapping domains (capability, data, deployment, reasoning); Young 2026 NP-hardness validated (arXiv:2501.15446); ≤12 principles ensures polynomial tractability; 6 tests
- **OpenHub Research attribution** — all docs updated (Issue #5); author = Robin Dey, institution = OpenHub Research (Thailand)

### Tests: 400 → 426 (+26)

## [2.0.0] — 2026-04-15

### Added
- **`CasDecayCalibrator`** in atlas-palace — 4-regime adaptive pheromone decay calibration (exploration/exploitation/success/traversal); `calibrated_decay_rate()` API; pheromone field stabilization
- **`OodaFeedback`** in atlas-astra — closed-loop OODA control; adaptive `explore_ratio` [0.1, 0.9]; sliding-window performance tracking; feedback-driven curriculum steering
- **`SampleStrategy::Stigmergic`** in atlas-corpus — softmax pheromone sampling with configurable `temperature`; stigmergic curriculum beats random baseline
- **GPU dispatch** in atlas-model — `forward()` routes to `GpuMatrix::sgemm` (VRAM-resident weights); CPU fallback if CUDA absent; 29 tok/s on Tesla T4
- **`GpuMatrix`** in atlas-tensor — pre-upload weights to VRAM; `sgemm()` calls GPU kernel per forward pass; 695 MiB VRAM for SmolLM2-135M
- **`atlas-mcp` systemd services** — `atlas-mcp.service` + `atlas-discover.service` on astra-01 (TCP port 8765, 30-min OODA cycles)

### Tests: 383 → 400 (+17)

## [0.7.0] — 2026-04-15

### Added
- **`CHANGELOG.md`** — full version history from v0.1.0
- **`REPRODUCIBILITY.md`** — step-by-step reproduction guide with reference benchmark numbers
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — CPU test matrix (Rust 1.75 + stable), fmt check, clippy, GPU runner stub, benchmark artifact upload
- **`atlas bench` subcommand** — end-to-end benchmark suite: palace insertions/A* queries, SFT training throughput, quality gate throughput

## [0.6.0] — 2026-04-15

### Added
- **`atlas-safety`** — 5-state FSM safety controller (`BOOT → NOMINAL → DEGRADED → SAFE_MODE → EMERGENCY_STOP`) with `CircuitBreaker`, append-only `AuditEntry` audit trail, and configurable thresholds
- **`atlas-bridge`** — Rings↔Ethereum ZK bridge interface: `AtlasBridge` with deposit/withdraw/confirm, `Groth16Claim` attached to every transaction, deterministic `sha256_pub` transaction hashing
- **`atlas-zk` Groth16 stub** — Full SHA-256 (FIPS 180-4) + HMAC-SHA256 implemented from scratch; `Groth16Claim` + `groth16_prove()` / `groth16_verify()` — interface-identical to the full BLS12-381 system in asi-build
- 37 new tests across atlas-safety (11), atlas-bridge (8), atlas-zk Groth16 (7)

## [0.5.0] — 2026-04-15

### Added
- **LoRA adapters** in `atlas-quant` — `LoraAdapter` (rank=8, alpha=16, kaiming-A, zero-B), `LoraConfig`, batched forward pass, 7 tests
- **Gradient accumulation** in `SftTrainer` — configurable `accum_steps`, scaled loss, gradient summing across micro-batches
- **Gradient clipping** in `atlas-optim` — `clip_grad_norm(max_norm: f32)` with L2 global norm, 2 tests
- **Safetensors checkpoint roundtrip** — `SftTrainer::save_safetensors()` + `load_safetensors()`, MLP weights persisted as safetensors binary, roundtrip test

## [0.4.0] — 2026-04-15

### Added
- **`atlas-model` safetensors** — `SafetensorsFile` binary parser with F16/BF16→F32 conversion, `build_f32()` serializer, 19 tests
- **Model configs** — `ModelConfig::olmo3_1b()` (1.24B params, GQA 32/8 heads), `ModelConfig::llama32_1b()` (1B params)
- **`atlas-astra` live connectors** — NASA POWER daily API, WHO GHO, World Bank v2, ArXiv Atom XML with graceful HTTP fallback, 29 tests

## [0.3.0] — 2026-04-15

### Added
- **Live API discovery** — Real HTTP data fetchers via `atlas-http`, hypothesis lifecycle (Generate→Queue→Investigate→Evaluate→Validate/Refute)
- **Discovery memory** — JSON persistence, 30-minute TTL, dedup detection, adaptive strategy
- **Statistical tests** — Granger causality, Benjamini-Hochberg FDR, CUSUM change-point, Cohen's d effect sizes (all pure Rust)

## [0.2.0] — 2026-04-15

### Added
- **`atlas-palace` upgrade** — real 5-type pheromone system (exploitation/exploration/success/traversal/recency), MMAS ceiling, exponential/linear/sigmoid decay, A* semantic pathfinding (α·C_semantic + β·C_pheromone + γ·C_structural, 40/30/30), HNSW-style approximate search, Active Inference agents (EFE, Bayesian beliefs, softmax action selection, 5 archetypes), swarm coordination
- **`atlas-mcp`** — MCP server, JSON-RPC 2.0, 28 tools over stdin/stdout, bearer token auth

## [0.1.0] — 2026-04-15

### Added
- Initial release: 17 crates, zero external dependencies
- **`atlas-tensor`** — f32 + CUDA matmul kernels (sm_75/T4), CUDA FFI via build.rs, INT4/INT8 quantization
- **`atlas-grad`** — autograd tape, VJPs for matmul/relu/add, `backward()` + `backward_with_grad()`
- **`atlas-optim`** — AdamW with weight decay, `CosineScheduler` with warmup, gradient accumulation support
- **`atlas-quant`** — INT4/INT8 quantization, QLoRA integration, dequantize/requantize, LoRA adapters
- **`atlas-model`** — RMSNorm, RoPE, SwiGLU, Grouped Query Attention (GQA), safetensors loader
- **`atlas-tokenize`** — GPT-2 BPE tokenizer (merge-table loader, encode/decode, round-trip)
- **`atlas-palace`** — Stigmergic memory palace (wings/rooms/drawers), pheromone system, A* pathfinding, knowledge graph, Active Inference agents, diaries
- **`atlas-trm`** — TRM-CausalValidator (z = net(x, y, z)×6 Bayesian causal validation)
- **`atlas-causal`** — PC algorithm, Fisher-Z independence test
- **`atlas-bayes`** — BetaPrior, BayesNetwork, BIC model comparison
- **`atlas-zk`** — Schnorr proofs, `ProvenanceChain` with cryptographic commitments
- **`atlas-astra`** — OODA discovery engine stub
- **`atlas-corpus`** — `LiveDiscoveryCorpus`, 5 quality gates, pheromone sampler, `SftTrainer`
- **`atlas-http`** — HTTP/1.1 client (TCP + TLS), connection pooling, retry
- **`atlas-json`** — Recursive-descent JSON parser, zero-copy string interning
- **`atlas-mcp`** — MCP server (JSON-RPC 2.0, 28 tools, bearer auth)
- **`atlas-cli`** — Full pipeline binary: `atlas discover | train | eval | prove | palace | status | bench | mcp`
- Tesla T4 (sm_75, CUDA 12.9) validated — all 186 tests passing
