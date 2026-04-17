# Changelog

All notable changes to ATLAS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
ATLAS uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [4.0.3] — 2026-04-17

### Fixed
- **Issue #11 — Bug 1: `CanonicalPheromoneUpdate` λ decay formula** (`atlas-palace`):
  The Champagnat canonical equation `λ = base_rate × (1 − ½·er·σ²·n̄·|∇f|)` exits its valid
  domain (goes internally negative) when `canonical_term > 1`, which is routinely exceeded
  during early training when gradients are steep. The old `.clamp()` masked the output but left
  a dead gradient at the clamp boundary and would have been unsafe to spec into the v6 ASIC
  decay circuit. **Fix**: replace with `λ = base_rate × exp(−canonical_term)`. Properties:
  always positive, zero-gradient fidelity (output = `base_rate` at `canonical_term = 0`),
  first-order match to the original formula (`exp(−x) ≈ 1 − x + …`), smooth everywhere,
  no gradient discontinuity.
- **Issue #11 — Bug 2: `InvasionFitnessScorer` negative competition coefficients** (`atlas-corpus`):
  Raw `cosine_sim ∈ [−1, 1]` was used as the Lotka-Volterra competition coefficient α_{ij}.
  Negative cosine similarity (anti-correlated embeddings) produced negative competition — adding
  a fitness bonus for semantic opposites, equivalent to mutualism. Lotka-Volterra coexistence
  requires α_{ij} ≥ 0. **Fix**: `α_{ij} = ReLU(cos_sim − competition_threshold)` with
  configurable `competition_threshold` (default 0.2 ≈ 4σ above noise floor in d=384 embedding
  space). Anti-correlated and near-orthogonal morphs produce zero competition; only genuinely
  similar morphs compete. `competition_threshold` added to `InvasionFitnessConfig`.

### Tests: 532 / 532 (+4 new regression tests)
- `canonical_update_never_negative_across_random_inputs` — 10,000 random (er, temp, avg, grad) samples; all rates ≥ 0
- `canonical_update_smooth_at_old_failure_boundary` — non-zero finite difference at the old linear formula's zero-crossing
- `competition_never_negative_for_anticorrelated_embeddings` — anti-correlated resident cannot boost candidate fitness
- `competition_threshold_suppresses_near_orthogonal_noise` — cos_sim < 0.2 → zero competition effect

---

## [4.0.2] — 2026-04-17

### Added
- **BF16 GPU inference path** (Issue #9) — W16A32: weights stored in BF16 (2 bytes/elem → 14 GB), activations remain f32; halves VRAM footprint vs f32 (28 GB)
- **`GpuBufBf16`** in atlas-tensor — cudaMalloc/cudaMemcpy/cudaFree for `*mut u16` (BF16 = top 16 bits of f32, exact bit pattern, no precision loss)
- **`GpuBufKind`** enum — `{ F32(GpuBuf), BF16(GpuBufBf16) }` discriminated union; `GpuMatrix.buf` now stores either precision; `is_bf16()` predicate
- **`GpuMatrix::upload_bf16()`** — uploads raw BF16 bit patterns (`&[u16]`) to VRAM; returns `GpuBufKind::BF16`
- **`sgemv_bf16_kernel`** CUDA kernel — one-warp-per-row GEMV for N=1 autoregressive decode: dequantizes BF16 on the fly, warp-reduce via `__shfl_down_sync`, `__ldg` on activation vector
- **`sgemv_f32_kernel`** CUDA kernel — matching one-warp-per-row GEMV for F32 weights; fixes 32× inefficiency in tiled GEMM for N=1 (31/32 threads multiply against zero in TILE=32 grid)
- **`atlas_sgemm_bf16_f32()`** FFI wrapper — dispatches to `sgemv_bf16_kernel` (N=1) or `sgemm_bf16_kernel` (N>1)
- **`make_linear_bf16_aware()`** — if CUDA available and tensor is BF16-origin, converts f32 back to BF16 bit patterns and calls `upload_bf16()`; drops BF16 Vec after GPU upload to reclaim RAM
- **`load_model_from_dir()`** — `HashSet<String>` tracking BF16 tensor names across all shards; routes to `make_linear_bf16_aware()` for each weight tensor
- **`OlmoModel::gpu_weight_dtype_counts()`** — returns (bf16_count, f32_count) across all layer matrices for diagnostics
- **`gpu_benchmark_olmo3_7b_think_bf16`** test (ignored) — validates full BF16 path end-to-end on A100
- **`gpu_bf16_gemv_parity`** test (ignored) — validates BF16 GEMV kernel output matches CPU reference

### Fixed
- **Issue #9** — OLMo-3-7B-Think GPU throughput: 4.1 tok/s (CPU, f32 too large for pre-upload) → **19.9 tok/s** (GPU BF16, W16A32); **4.8× speedup** on A100-SXM4-40GB
- **Tiled GEMM for N=1**: added GEMV fast path in `atlas_matmul_f32` (was using 32× wasteful tiled kernel for single-token decode)
- **nvcc forward declaration**: `bf16u_to_f32` helper moved before GEMV kernels that reference it

### Performance
- OLMo-3-7B-Think on A100-SXM4-40GB: **4.1 → 19.9 tok/s** (4.8× speedup)
- 224/224 OLMo-3-7B weight matrices now resident in BF16 VRAM
- First-token latency: 33 ms | Load time: 109s

### Tests: 528 / 528 (unchanged — existing GPU tests still pass)

---

## [4.0.1] — 2026-04-17

### Changed
- Docs, README, CHANGELOG, and CITATION.cff updated to reflect v4.0.0 / Issue #7
- GPU quality gate test adjusted for tokenizer-agnostic validation

### Tests: 528 / 528

---

## [4.0.0] — 2026-04-17

### Added
- **`InvasionFitnessScorer`** in atlas-corpus — morphic fitness function f(y) = success − cost − Σcos_sim(y, xᵢ)·n̄ᵢ; replaces raw pheromone softmax; fixes pheromone saturation bug; 9 tests
- **`CanonicalPheromoneUpdate`** in atlas-palace — principled decay rate adaptation Δρ = ½·μ·σ²·n̄·∂₁s; derived from Champagnat-Méléard 2011 PTRF Theorem 3.1; replaces ad-hoc CAS tuning; 6 tests
- **`BarBovier2017Constraints`** in atlas-corpus — stability gate validating `explore_ratio × batch_size > 10` AND `temperature > 1/√batch_size`; grounded in Baar-Bovier-Champagnat 2017 AAP; 7 tests
- **`CognitiveBranching`** in atlas-astra — n-morphic OODA bifurcation detector; splits exploration thread when `explore_ratio` plateau + high `loss_trace` variance detected; 8 tests
- **`HJConcentrationPrior`** in atlas-trm — Hopf-Cole sharpening in TRM recursion: effective temperature T_eff(s) = T₀/(1+γs); enforces max(logits)=0 constraint; derived from Champagnat-Hass 2025 AAP; 7 tests
- **OLMo-3-7B-Think inference fix** (Issue #7) — three-part fix for degenerate inference (repeating "adidas adidas"):
  - **Fix A (SWA)**: `Attention::window_size: Option<usize>` per-layer banded causal mask; 24 of 32 OLMo-3 layers are sliding-window (window=4096), 8 are full-attention
  - **Fix B (YaRN RoPE)**: `RopeCache::new()` extended to 4-arg with Peng et al. 2023 Algorithm 1 — per-dimension frequency scaling partitioned into high/low/mid bands; `attn_factor=1.2079` applied to attention scale
  - **Fix C (config auto-patch)**: `patch_config_from_hf_json()` reads sibling `config.json`, auto-populates `layer_types`, `sliding_window`, `rope_scaling`, `rope_theta`, `rms_norm_eps` from HuggingFace model card
- **GPU quality verification** — `gpu_inference_olmo3_quality_sanity` test on A100-SXM4-40GB: logit spread 16.8, max_prob 0.96%, 10/10 unique generated tokens ✅
- **A100 OLMo-3-7B benchmark** — `gpu_benchmark_olmo3_7b_think`: 4.1 tok/s (CPU fallback; 28GB f32 exceeds pre-upload VRAM path)
- **SmolLM2 A100 benchmarks** — SmolLM2-135M: 37.7 tok/s; 360M: 25.4 tok/s; 1.7B: 12.6 tok/s; TinyLlama-1.1B: 20.9 tok/s

### Fixed
- **Issue #7** — OLMo-3-7B degenerate inference: SWA mask, YaRN RoPE, config.json auto-patch (commits 27de176→faa8bba)
- **GPU test binary** — `cargo test --release --no-run` now correctly rebuilds test binary when source changes

### Tests: 516 → 528 (+12)
- atlas-model: +12 tests (4 CPU unit tests for SWA/YaRN/config, Fix A+B+C GPU quality gate)

---

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
