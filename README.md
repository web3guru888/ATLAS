# ATLAS

**Active-inference Training with Learned Adaptive Stigmergy**

> *"Don't train on what humans wrote about the world.  
> Train on what you actually discover about the world.  
> Validate what you claim. Own what you build."*

[![License: Apache 2.0](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE-CODE)
[![License: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Zero Dependencies](https://img.shields.io/badge/external%20crates-0-brightgreen.svg)](#pure-rust--zero-dependencies)
[![Release](https://img.shields.io/badge/release-v4.2.0-success.svg)](#status)
[![Tests](https://img.shields.io/badge/tests-644%2F644%20passing-brightgreen.svg)](#status)
[![CI](https://img.shields.io/badge/CI-green-brightgreen.svg)](#status)
[![Crates](https://img.shields.io/badge/crates-22-blueviolet.svg)](#crate-status)
[![MCP Tools](https://img.shields.io/badge/MCP%20tools-28-blueviolet.svg)](#atlas-mcp)
[![CUDA](https://img.shields.io/badge/CUDA-sm__80%20A100-76b900.svg)](#gpu-inference)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20compatible-412991.svg)](#atlas-api--openai-compatible-endpoint)

---

> **An open source contribution from [OpenHub Research](https://openhubresearch.org/) (Thailand)**  
> Website: [atlasagi.org](https://atlasagi.org) · [Observatory](https://huggingface.co/spaces/openhubresearch/ATLAS) · [Live demo](https://demo.thebeastagi.com) · Author: Robin Dey · Institution: https://openhubresearch.org/

---

## What is ATLAS?

ATLAS is a next-generation LLM training framework built in **pure Rust with zero external crate dependencies** — the SQLite principle applied to AI infrastructure.

It fuses four architectural innovations:

| Component | Role | Key property |
|-----------|------|-------------|
| **ASTRA-dev** | Live discovery engine | ~10s/cycle, NASA/WHO/World Bank APIs, causal inference |
| **GraphPalace** | Stigmergic memory | Pheromone-guided curriculum, O(1/√T) convergence |
| **TRM-CausalValidator** | Recursive validator | 7M params, 0.1% compute, Quality Gate 6 |
| **ZK Schnorr proofs** | Provenance chain | LLM output → live API, cryptographically verifiable |

**v4.2.0** — HF-Reference Inference Fidelity + OpenRouter Launch Prep (2026-07-06):
- 🎯 **Three silent inference-quality bugs found & fixed** by differential testing against HuggingFace transformers ground truth (same weights, CPU f32):
  1. **YaRN RoPE correction range** — boundaries conflated 1/f with wavelength (extra 2π) and ramped linearly in wavelength; HF computes the range in dimension-index space (log formula) with a dim-index-linear ramp. 25/64 RoPE dims had wrong frequencies (~1.6 rad angle error by position 100). `attention_factor` now applied **squared** at score level (HF scales cos/sin, touching both q and k).
  2. **Per-layer-type RoPE** — `rope_scaling` (YaRN) applies **only to the 8 full-attention layers**; the 24 sliding-window layers use standard RoPE at the same theta (HF `configuration_olmo3.py`). ATLAS now keeps dual CPU+GPU RoPE tables selected per layer.
  3. **QK-norm scope** — `Olmo3RMSNorm(n_heads*head_dim)` normalizes the **full concatenated Q/K projection** (one RMS statistic across all heads), not per-head. Per-head normalization rescaled head magnitudes and crushed long-range retrieval: the model could not read its own prompt beyond ~128 tokens.
- 📈 **Verified**: needle-retrieval passes 68→1,784 tokens (was FAIL at 156+), output byte-identical to HF reference; MMLU diverse-100 22%→**54%** (direct-answer protocol); new 300-position CPU/GPU logit-parity test (exact match)
- 🚦 **CI green** (first time since April): MSRV 1.75→1.80, lockfile v4; **631/631 tests**
	- 🧠 **32B AWQ inference on a single A100-40GB** — OLMo-3-32B-Think at W4A32: **GSM8K-25 100%, MMLU-40 82%, Code-5 5/5**, 14.6 tok/s decode, 27.2 GB VRAM. Custom `gemv_w4_kernel` + streaming shard loader + `OlmoModel::new_uninit`. Purely additive — 7B BF16 path untouched and re-verified. See [model card](https://huggingface.co/openhubresearch/ATLAS-OLMo-3-32B-Think-v4). *(The 32B W4 serving code is **merged to `main`** and running in production on the A100.)*
	- 🔗 **HF Space LIVE** at [huggingface.co/spaces/openhubresearch/ATLAS](https://huggingface.co/spaces/openhubresearch/ATLAS) — serving 32B reasoning model through ATLAS stack on dedicated A100 silicon
- 🌐 **OpenRouter provider endpoints**: `GET /openrouter/models` (full provider schema), `GET /privacy`, SSE `: keep-alive` every 10s during prefill/think, early-429 concurrency discipline, bearer-key auth — live behind a Cloudflare tunnel on dedicated A100 silicon
- ⚡ **GPU training path**: SFT optimizer step on GPU (fixed `step_gpu` clip-norm + weight-decay parity, Issue #20); `GpuVec::dup` D2D copies (H2D traffic −59%)
- 🛡️ **No more silent GPU garbage**: sticky kernel-error flag + end-of-token integrity check + `gpu_poisoned` CPU fallback; stale host-shadow fallback audit (Issue #21)

**v4.1.0** — Full GPU Attention Path + 61 tok/s BF16 (Issue #18):
   - Vendor-fork mistral.rs SQLite philosophy: ATLAS owns every kernel
   - StigmergicHook trait: per-layer pheromone deposits into GraphPalace
   - GpuKvCache + GpuRopeTables: zero intra-layer PCIe transfers
   - decode_attention_kernel: GQA attention entirely in VRAM
   - CPU/GPU KV reset made NO-OP: eliminates 605ms overhead/generate()
   - Performance: 15.4 → **61.7 tok/s** (OLMo-3-7B-Think, A100 BF16)
   - Tests: **600** (was 579)

**v4.0.0** — Champagnat n-morphic Framework + OLMo-3-7B Fix:
- 🧬 **InvasionFitnessScorer** — morphic fitness f(y) = success − cost − Σcos_sim·n̄ (fixes pheromone saturation)
- 🌊 **CanonicalPheromoneUpdate** — principled decay Δρ ∝ μ·σ²·n̄·∂₁s (Champagnat-Méléard 2011)
- ⚖️ **BarBovier2017Constraints** — stability gate: explore_ratio × batch_size > 10, temp > 1/√batch
- 🔀 **CognitiveBranching** — n-morphic OODA bifurcation on plateau detection
- 🔆 **HJConcentrationPrior** — Hopf-Cole sharpening T_eff(s) = T₀/(1+γs) in TRM recursion
- 🔧 **Issue #7 fix** — OLMo-3-7B SWA (24/32 sliding layers, window=4096) + YaRN RoPE + config.json auto-patch

**v4.0.2** — BF16 GPU Inference Path (Issue #9):
- ⚡ **BF16 W16A32** — weights in BF16 (14 GB) vs f32 (28 GB); `GpuBufBf16`, `GpuBufKind`, `upload_bf16()` in atlas-tensor
- 🔥 **GEMV kernels** — `sgemv_bf16_kernel` + `sgemv_f32_kernel`: one-warp-per-row for N=1 decode; fixes 32× tiled-GEMM inefficiency
- 🚀 **OLMo-3-7B-Think: 4.1 → 19.9 tok/s** (4.8× speedup, A100-SXM4-40GB, W16A32)

**v4.0.3** — Math Integrity Fixes (Issue #11):
- 🧮 **`CanonicalPheromoneUpdate` λ decay** — replaced linear formula `base_rate × (1 − canonical_term)` (went negative when term > 1, dead gradient at clamp boundary) with `base_rate × exp(−canonical_term)`: always positive, smooth, zero-gradient fidelity, hardware-safe for v6 ASIC spec
- 🏆 **`InvasionFitnessScorer` competition kernel** — fixed negative Lotka-Volterra coefficients: raw `cosine_sim ∈ [−1, 1]` was giving fitness bonuses to anti-correlated strategies (mutualism, not competition); replaced with `α_ij = ReLU(cos_sim − 0.2)` — threshold at 4σ above noise floor in d=384 embedding space; `competition_threshold` added to `InvasionFitnessConfig`
- ✅ **532/532 tests** (+4 new regression tests); GPU validated: 47/47 A100 model tests, OLMo-3-7B-Think still **19.9 tok/s**

**v4.0.7** — OLMo-2/3 Post-Norm + QK-Norm Architecture Fixes:
- 🏗️ **Post-norm layer ordering** — OLMo-2/3 uses `x = residual + rmsnorm(output)` (normalize output, then add residual). ATLAS was incorrectly doing `x = rmsnorm(x + output)` (GPU) or pre-norm (CPU). Fixed both paths to match [HuggingFace Olmo2DecoderLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py) reference implementation.
- 🔧 **QK-norm per-head weight slicing** — `q_norm.weight` has shape `[n_heads × head_dim]` — each head has unique norm weights. `rmsnorm_inplace` was always using `weights[0..128]` for every head instead of `weights[h*128..(h+1)*128]`. Fixed GPU path.
- ➕ **QK-norm added to CPU attention path** — the CPU `Attention::forward_token()` had no QK-norm at all. Added per-head QK-norm before RoPE to match GPU path.
- ✅ **Before → After**: CPU/GPU logit agreement went from max diff **20.0 → 0.000015**. Top-1 token for "capital of France" went from `yp`/`décor` → **`Paris`**. OLMo-3-7B-Think now produces coherent `<think>` reasoning traces at **15.4 tok/s** on A100.

**v4.0.9** — Think Suppression + Output Quality:
- 🚫 **Think suppression** — ban `<think>` token at step 0 via logit masking, preventing the model from entering internal reasoning mode entirely. Produces clean, direct answers instead of visible `<think>...</think>` blocks
- 📏 **`max_tokens` raised to 512** (was 256) — gives the model room for complete answers without truncation
- 🧹 **Improved filler cleanup** — post-processing removes trailing "Okay" paragraphs and other filler patterns that small models produce
- 🔍 **Model auto-detection** — `atlas-model` now infers model config from the weights directory name (e.g., `OLMo-3-0125-7B` → OLMo-3 config), eliminating manual config specification
- 📊 **API test results**: 4/6 prompts produce clean, correct answers (Hello=perfect, Spain=perfect, Exercise=good, Einstein=good); before: 2/8 clean with 50% think-block fallback
- ✅ **579/579 tests** (+14 new)

**v4.0.8** — Anti-Repetition Defaults + Think Budget:
- 🔁 **Fixed degenerate think loops** — API defaulted all penalties to OFF; model entered "Wait, no, yes, Madrid" ×100 spirals. Root cause: no `repetition_penalty` from frontend + tiny 64-token window + no `top_k`/`min_p` filtering
- 🎯 **`SamplingConfig::olmo3()` v2** — `rep_penalty=1.15`, `window=256` (was 64), `top_k=50`, `min_p=0.05`, `top_p=0.95`. API request defaults now sourced from `olmo3()` preset instead of everything-off
- 🧠 **Think budget** — force-closes `<think>` blocks after 200 tokens to prevent small models from endlessly rambling. Streaming: injects `</think>` via SSE. Non-streaming: `enforce_think_budget()` post-processing
- 📡 **Extended API** — `top_k`, `min_p`, `repetition_window` added to request body; handler passes all fields explicitly (no more `..Default` spread)
- ✅ **565/565 tests** (+3 new: think budget close/passthrough/no-block)

**v4.0.6** — Sampling Controls (Issue #16):
- 🎛️ **Full sampling pipeline**: repetition penalty, temperature, top-p, top-k, min-p, frequency/presence penalty — 7-stage pipeline eliminates model text degeneration
- 🔧 **`SamplingConfig` struct** with `::olmo3()` preset — API defaults sourced from model-appropriate preset
- 🆕 **`generate_with_sampling()`** — full control; `generate()` remains backward-compatible
- 📡 **OpenAI-compatible API** — `top_p`, `top_k`, `min_p`, `repetition_penalty`, `repetition_window`, `frequency_penalty`, `presence_penalty` in request body
- ✅ **562/562 tests** (+13 new: repetition penalty, top-p/top-k/min-p filtering, frequency/presence penalties, greedy+sampling)

**v4.0.5** — Inference Pipeline Fixes (Issues #13, #14, #15):
- 🛑 **EOS stopping** — `generate()` now stops on EOS token (was dead code: `if let Some(eos) = None::<u32>`); parsed from `config.json` (OLMo-3: 100257), wired through model → API `finish_reason: "stop"` works correctly
- 🎲 **Stateful PRNG** — XorShift64 replaces deterministic step-based LCG hash; re-seeded from system time each `generate()` call; repeated requests now produce different completions
- 💬 **ChatML template** — `<|im_start|>/<|im_end|>` format (OLMo-3, SmolLM2, Qwen); auto-detected from tokenizer special tokens; also supports Llama-3 format; fixes garbage output from wrong `<|system|>/<|user|>` tokens
- ✅ **549/549 tests** (+10 new: EOS stopping, PRNG variability, ChatML/Llama3/Generic templates)

**v4.0.4** — GPT-4 Regex Tokenizer (Issue #12):
- 🔤 **Full HuggingFace tokenizer.json support** — hand-coded GPT-4/OLMo-3/LLaMA-3 pre-tokenization regex (zero external deps): contractions, word boundaries, 3-digit number grouping, punctuation, newlines, whitespace with backtracking
- ✅ **Verified against HuggingFace `tokenizers`** — OLMo-3 `encode("The capital of France is") → [791, 6864, 315, 9822, 374]`, SmolLM2 verified, round-trip decode perfect
- 🧪 **End-to-end GPU test** — tokenize→generate→decode on A100, chat template with `<|im_start|>/<|im_end|>` special tokens
- ✅ **539/539 tests** (+7 pre-tokenizer unit + 2 integration + 1 e2e GPU)

The result: a self-improving scientific intelligence that trains on what it **actually discovers** about the world — real causal relationships from live data, validated by recursive architecture, guided by stigmergic memory.

Nobody has built this before. See [CHARTER.md](CHARTER.md) for the full architecture.

---

## The Big Idea

Every other LLM is trained on:
- What humans wrote on the internet (web scrapes, Wikipedia)
- Synthetic data generated by another LLM (GPT-4 distillation)
- Human-curated datasets (expensive, frozen at curation time)

**atlas-7b is trained on:**
- What an autonomous science engine *actually discovers* about the world
- Real causal relationships extracted from live NASA, WHO, World Bank APIs
- Validated findings with Bayesian confidence scores and PC/FCI causal inference
- A corpus that grows every 10 seconds and never contains stale or duplicated information

This is not a better fine-tuning recipe. **This is a different paradigm for what training data can be.**

---

## Pure Rust — Zero Dependencies

The SQLite principle applied to AI infrastructure.

```
atlas/
├── Cargo.toml          # workspace root — [dependencies] is empty by design
├── kernels/
│   ├── matmul.cu       # raw CUDA kernel (no cudarc crate)
│   ├── attention.cu    # flash attention from scratch
│   └── quant.cu        # INT4/INT8 quantization
└── crates/
    ├── atlas-core/     # error types, traits, config
    ├── atlas-tensor/   # Tensor + CUDA FFI (the seed of everything)
    ├── atlas-grad/     # autograd tape, backward pass
    ├── atlas-optim/    # AdamW, cosine LR scheduler
    ├── atlas-quant/    # INT4/INT8 quantization, QLoRA
    ├── atlas-model/    # transformer: MultiHeadAttn, FFN, RMSNorm, RoPE
    ├── atlas-infer/    # StigmergicHook trait + InferEngine + GPU/CPU dispatch (v4.1.0)
    ├── atlas-tokenize/ # BPE tokenizer (sentencepiece port)
    ├── atlas-palace/   # GraphPalace stigmergic memory: A* search, 5-type pheromones, Active Inference
    ├── atlas-mcp/      # MCP server: 28 palace tools via JSON-RPC 2.0 stdio + connection pool
    ├── atlas-api/      # OpenAI-compatible HTTP endpoint: /v1/chat/completions, SSE streaming
    ├── atlas-trm/      # TRM-CausalValidator (7M params, arXiv:2510.04871)
    ├── atlas-causal/   # PC/FCI causal inference (py-causal port)
    ├── atlas-bayes/    # Bayesian confidence scoring
    ├── atlas-astra/    # ASTRA OODA engine (~8K LOC, full port)
    ├── atlas-corpus/   # LiveDiscoveryCorpus + DeepSupervisionTrainer + quality gates
    ├── atlas-zk/       # ZK Schnorr proofs (asi-build port)
    ├── atlas-http/     # HTTP client via raw libc syscalls
    ├── atlas-json/     # JSON parser from source
    ├── atlas-safety/   # Horn-clause safety constitution, 5-state FSM, CircuitBreaker
    ├── atlas-bridge/   # ZK-attested Rings↔ETH interface (Sepolia-compatible)
    └── atlas-cli/      # CLI: train / discover / eval / prove / mcp / api / bench
```

**22 crates. One coherent system. Zero external Rust dependencies.**

CUDA is called via raw `extern "C"` FFI from `build.rs` + `.cu` kernel files — no `cudarc`, no `tch`, no `candle`. The same approach that makes SQLite trustworthy, applied to GPU compute.

```rust
// atlas-tensor/src/lib.rs — the first line of ATLAS
pub struct Tensor {
    data:  Vec<f32>,
    shape: Vec<usize>,
}
```

Every billion-parameter transformer starts here.

---

## Seven Pillars

1. **GraphPalace Memory** — pheromone-weighted persistent knowledge; `search_by_embedding()`, `hot_paths()`, `deposit_pheromones()`
2. **Morphic Warm-Start** — O(1/√T) cross-run convergence (proven in [BUTTERS](https://github.com/web3guru888/morphic-resonance-sim), R²=0.982, p<10⁻³⁰)
3. **Stigmergic RLVR** — `r_total = α·r_verifiable + β·r_pheromone`; pheromone decay prevents reward hacking
4. **Active Inference Data Gen** — palace cold spots direct ASTRA to fill knowledge gaps
5. **ZK Knowledge Claims** — Schnorr proof chain from LLM output to raw API data; hallucinations have broken proof trails
6. **LiveDiscoveryCorpus** — ASTRA's output as a living training dataset; ~86K quality examples/month
7. **TRM-CausalValidator** — 7M-param recursive validator; `z = net(x,y,z) × 6 recursions`; Quality Gate 6; generates Type 5 training traces

---

## GPU Inference

ATLAS v4.0.0 delivers a **fully GPU-resident forward pass** — hidden states stay in VRAM between tokens, with pre-pinned weight upload at model load time.

### A100-SXM4-40GB Benchmark (sm_80, CUDA 13.0)

| Model | Params | GPU tok/s | VRAM | Notes |
|-------|--------|-----------|------|-------|
| SmolLM2-135M | 135M | **37.7** | 507 MiB | f32, sm_80 |
| SmolLM2-360M | 360M | **25.4** | ~1.4 GB | f32 |
| SmolLM2-1.7B | 1.7B | **12.6** | ~6.5 GB | f32, 2.4× over CPU |
| TinyLlama-1.1B | 1.1B | **20.9** | ~8.4 GB | f32 |
| OLMo-3-7B-Think | 7B | **61.7** | ~14 GB | **BF16 W16A32** (v4.1.0); full GPU attention, zero intra-layer PCIe, cuBLAS TF32 |
| OLMo-3-32B-Think | 32B | **14.6** | ~27.2 GB | **W4A32 AWQ** (v4.2.0); custom GEMV kernel, streaming W4 shard loader, single A100-40GB |

### CUDA Kernel Suite

| Kernel | What it does |
|--------|-------------|
| `rmsnorm_forward` | RMSNorm in CUDA — replaces per-token CPU loop |
| `rope_forward` | RoPE rotation — parallel over heads |
| `silu_mul_forward` | SwiGLU gate fused — single CUDA pass |
| `atlas_adamw_step` | AdamW optimizer step entirely on GPU |
| `decode_attention_kernel` | GQA decode attention entirely in VRAM — zero PCIe in attention (v4.1.0) |
| `qk_norm_inplace_kernel` | Per-head in-place RMSNorm for OLMo-2/3 QK-norm on GPU (v4.1.0) |
| `rope_precomputed_kernel` | GPU RoPE using precomputed YaRN cos/sin tables (v4.1.0) |
| `kv_cache_write_kernel` | VRAM-to-VRAM KV cache write at position pos (v4.1.0) |
| `atlas_gpu_argmax` | Two-pass parallel GPU argmax — no 400KB D2H logit download (v4.1.0) |
| `sgemm_vec` | Zero-copy matrix×vector; `GpuVec` activation buffer |
| `gemv_w4_kernel` | W4A32 GEMV — warp-per-row with inline dequant, int4 AWQ serving (v4.2.0) |

**CUDA portability**: all kernels use `rsqrtf()` (not `__rsqrtf()`) for cross-platform compatibility.

---

## atlas-api — OpenAI-Compatible Endpoint

ATLAS v4.0.0 adds `atlas-api` — an OpenAI-compatible HTTP inference server. Drop-in replacement for any OpenAI API client.

```bash
# Start the server
./target/release/atlas api serve --model /home/user/models/smollm2-135m --port 8080
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions with SSE streaming |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List available models |

### Usage Examples

```bash
# Chat completion (streaming)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "What is morphic resonance?"}],
    "stream": true
  }'

# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "Explain stigmergic memory"}],
    "stream": false
  }'

# List models
curl http://localhost:8080/v1/models
```

### Sampling Parameters

All requests accept sampling controls with **sensible OLMo-3 defaults**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.0 | Sampling temperature (0.0 = greedy) |
| `repetition_penalty` | 1.0 | Keskar 2019 penalty on recent tokens (AI2 reference: off) |
| `repetition_window` | 256 | Tokens to look back for repetition penalty |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `top_k` | 50 | Top-K filtering (0 = off) |
| `min_p` | 0.05 | Min-P filtering threshold |
| `frequency_penalty` | 0.0 | Proportional to token count |
| `presence_penalty` | 0.0 | Flat penalty for any seen token |
| `max_tokens` | 2048 | Generation budget — Think models spend 500–2,000+ tokens reasoning inside `<think>` before the visible answer |

> **Think handling**: the chat endpoint primes the official OLMo-3-Think template (`<|im_start|>assistant\n<think>`); the chain-of-thought is returned as `message.reasoning` (streaming: `delta.reasoning`) and the visible answer as `message.content`. **Answer reserve**: if `max_tokens` runs out while still inside `<think>`, the server force-closes the block and continues from the intact KV cache for up to `ATLAS_ANSWER_RESERVE` (default 512, `0` disables) answer tokens — clients always get a visible answer.

**Features**: SSE streaming, CORS headers, think-block reasoning/content split + answer reserve, client-disconnect abort (#25), echo mode for testing, 81 tests, 0 external dependencies.

---

## DeepSupervisionTrainer

The `DeepSupervisionTrainer` in `atlas-corpus` implements multi-pass deep supervision — each training batch runs N_sup=4..16 forward passes, summing loss across all supervision points with pheromone-driven latent carry between passes.

```rust
// atlas-corpus/src/deep_supervision.rs
pub struct DeepSupervisionTrainer {
    pub n_sup: usize,          // 4..16 forward passes per batch
    pub latent_carry: bool,    // carry hidden state between passes
    pub pheromone_weight: f32, // pheromone × mutation-selection coupling
    pub loss_trace: Vec<f32>,  // per-pass loss telemetry
}
```

**Theory (TRM arXiv:2510.04871 validated)**: deep supervision accounts for >75% of TRM's improvement over baseline transformers. The `DeepSupervisionTrainer` implements this in the training loop — each N_sup pass is one phenotypic morph; latent carry approximates Lotka-Volterra equilibrium n̄ᵢ; `pheromone_weight` sets the mutation-selection coupling μ.

**Convergence prediction**: doubling N_sup → √2 speedup in O(1/√T) morphic convergence. Testable via the `loss_trace` telemetry.

---

## Horn-Clause Safety Constitution

`atlas-safety` v4.0.0 adds a tractable safety constitution implemented as Horn clauses, alongside the existing 5-state FSM and CircuitBreaker.

```
8 safety principles across 4 non-overlapping domains:
  ┌─────────────────┬──────────────────────────────────┐
  │ capability      │ scope_limits, capability_bounds  │
  │ data            │ provenance_required, dedup_gate  │
  │ deployment      │ audit_trail, circuit_breaker     │
  │ reasoning       │ causal_grounding, zk_verifiable  │
  └─────────────────┴──────────────────────────────────┘
```

**Why Horn clauses?** Young (2026, arXiv:2501.15446) proves NP-hardness of general safety constitution verification. Horn-clause restriction (≤12 principles, 4 non-overlapping domains) ensures polynomial tractability — the safety checker can verify any system state in O(n·m) where n = principles, m = state predicates. No exponential blowup.

---

## PalaceBackend Trait

`atlas-palace` v4.0.0 extracts a `PalaceBackend` trait, enabling pluggable storage backends without API changes:

```rust
pub trait PalaceBackend: Send + Sync {
    fn search(&self, query: &str, limit: usize) -> Vec<DrawerMatch>;
    fn deposit_pheromones(&mut self, path: &[RoomId], ptype: PheromoneType, intensity: f32);
    fn navigate(&self, from: RoomId, to: RoomId) -> Vec<RoomId>;
    fn hot_paths(&self, limit: usize) -> Vec<Path>;
    // ... 32 additional methods
}

// Palace implements PalaceBackend — fully trait-object safe
pub struct Palace { /* existing implementation */ }
impl PalaceBackend for Palace { /* ... */ }

// Swap backends without changing caller code
let palace: Box<dyn PalaceBackend> = Box::new(Palace::new(config));
```

This is the prerequisite for LadybugDB migration (Q3 2026) — a drop-in Grafeo/LadybugDB backend can replace the default implementation with zero API changes.

---

## Build Order (7 Stages, ~22 Weeks)

| Stage | Weeks | Crates | Milestone |
|-------|-------|--------|-----------|
| 1 | 1–4 | atlas-core → tensor → grad → optim → quant | f32 matmul CPU+GPU, backward pass through 2-layer MLP |
| 2 | 5–7 | atlas-model → tokenize | OLMo 3 7B forward pass in pure Rust, token generation |
| 3 | 8–9 | atlas-palace + atlas-mcp | GraphPalace 36-method engine native, MCP server |
| 4 | 10–11 | atlas-trm | TRM-CausalValidator, <10ms causal graph pass/fail |
| 5 | 12–16 | http → json → bayes → causal → zk → astra | Full ASTRA OODA in Rust, ZK provenance |
| 6 | 17–20 | atlas-corpus + atlas-api | QLoRA SFT, DeepSupervisionTrainer, OpenAI API |
| 7 | 21–22 | atlas-zk (ext) → cli | End-to-end proof chain, atlas-7b release binary |

---

## Architecture Diagrams

Eight publication-quality figures are in [`docs/dashboard/diagrams/`](docs/dashboard/diagrams/).
The interactive dashboard (project overview, roadmap, papers, component status) is at [`docs/dashboard/index.html`](docs/dashboard/index.html).

| Figure | Description |
|--------|-------------|
| Fig. 1 | Full System Architecture (v3.0, TRM cluster) |
| Fig. 2 | Discovery Flywheel — the self-improving loop |
| Fig. 3 | ASTRA OODA + GraphPalace integration |
| Fig. 4 | Morphic Warm-Start cross-run convergence |
| Fig. 5 | Stigmergic RLVR pheromone reward function |
| Fig. 6 | ZK Provenance Chain |
| Fig. 7 | Training Pipeline phase roadmap |
| Fig. 8 | Hybrid Generative-Recursive Architecture (TRM v3.0) |

---

## Paper Strategy

| Paper | Venue | Contribution |
|-------|-------|-------------|
| Paper 1 | EMNLP 2026 | ATLAS architecture + LiveDiscoveryCorpus |
| Paper 2 | NeurIPS 2026 | Discovery Flywheel — closed-loop scientific intelligence |
| Paper 3 | ICML 2027 | Stigmergic RLVR — pheromone reward prevents policy collapse |
| Paper 4 | ICLR 2027 | O(1/√T) morphic convergence for LLMs (co-author Robin Dey) |
| Paper 5 | IEEE S&P 2027 | End-to-end ZK provenance for LLM outputs |
| Paper 6 | ICLR/NeurIPS 2027 | Hybrid generative-recursive architecture (TRM integration) |

---

## Getting Started

```bash
git clone https://github.com/web3guru888/ATLAS.git
cd ATLAS

# Run all tests (excludes CUDA-requiring tensor tests on CPU-only machines)
cargo test --workspace --exclude atlas-tensor

# Build the atlas binary
cargo build --release -p atlas-cli

# Full OODA discovery loop
./target/release/atlas discover --cycles 5 --output corpus.json

# Train on discoveries
./target/release/atlas train --corpus corpus.json --epochs 3

# Start OpenAI-compatible API server
./target/release/atlas api serve --model /path/to/model --port 8080

# ZK-prove a claim
./target/release/atlas prove --claim "Pheromone trails compound information gain" \
    --secret $(openssl rand -hex 16)

# Inspect palace memory
./target/release/atlas palace --stats --hot

# MCP server (connect to Claude Desktop / Cursor)
./target/release/atlas mcp serve --palace my-palace.json
```

**Prerequisites:**
- Rust 1.80+ (`rustup update stable`)
- CUDA 12.x + nvcc (optional; falls back to CPU if absent)
- GPU with sm_75+ (Tesla T4 / A100+) for CUDA training path

---

## Status — v4.2.0

**644/644 tests passing** · **22 crates** · **Zero external crate dependencies** · **CUDA sm_80 on A100-SXM4-40GB** · **61.7 tok/s OLMo-3-7B-Think (BF16)** · **14.6 tok/s OLMo-3-32B-Think (W4A32)**

> 🏔 **v4.2.0 is the current release.** 32B AWQ W4 inference on a single A100-40GB: `gemv_w4_kernel` + streaming shard loader fits within 27.2 GB VRAM — **now merged to `main` and serving in production** on the A100-40GB. Three silent inference-quality bugs fixed by differential testing against HuggingFace transformers. OpenRouter provider endpoints + bearer-key auth live behind Cloudflare tunnel. HF Space serving the 32B reasoning model. **Batched prefill ([#22](https://github.com/web3guru888/ATLAS/issues/22)) is MERGED (`main` @ `f96e3f7`) and DEPLOYED on the live 32B service (2026-07-08):** time-to-first-token at ~1.25K-token prompts dropped **121.8s → 23.4s (~5× live)**, greedy output bit-identical (parity gate passed). One GEMM per layer over all prompt positions replaces T sequential GEMV steps. Disconnect-abort fix ([#25](https://github.com/web3guru888/ATLAS/issues/25)) is **merged to `main` and deployed** (client-disconnect probe + optional `ATLAS_REQUEST_DEADLINE_SECS` wall-clock deadline). **Think-budget answer reserve** (2026-07-08, branch [`fix/think-budget-answer-reserve`](https://github.com/web3guru888/ATLAS/tree/fix/think-budget-answer-reserve), deployed on the live endpoint): when a Think model exhausts `max_tokens` inside its `<think>` block, the server force-closes the block and continues from the intact KV cache for up to `ATLAS_ANSWER_RESERVE` (default 512) answer tokens — a visible answer is always emitted (zero-answer rate 2/5 → 0/5 on the integration-test battery; byte-identical output when the reserve doesn't fire). Chat default `max_tokens` is now 2048.

### What Works

- ✅ **Discovery is real** — `atlas discover --cycles 3` hits NASA POWER, WHO GHO, World Bank, ArXiv live APIs; causal inference via PC algorithm; Bayesian quality gates
- ✅ **Memory is real** — 5-type pheromone system (exploitation/exploration/success/traversal/recency), MMAS ceiling, A\* semantic pathfinding (α·C_sem + β·C_phe + γ·C_str), Active Inference agents; `atlas palace --hot` shows pheromone trails
- ✅ **Training is real** — SFT with GradTape + AdamW + LoRA (rank=8) + gradient accumulation + safetensors checkpoint; DeepSupervisionTrainer (N_sup=4..16, loss trace, latent carry)
- ✅ **GPU inference is real** — SmolLM2-135M at 37.7 tok/s on A100-SXM4-40GB; OLMo-3-7B-Think at **61.7 tok/s** (BF16 GPU, W16A32, 14 GB); OLMo-3-32B-Think at **14.6 tok/s** (W4A32 AWQ, 27.2 GB, single A100-40GB, v4.2.0); full GPU attention path, zero intra-layer PCIe, cuBLAS TF32 tensor cores; SWA + YaRN RoPE; post-norm + QK-norm architecture
- ✅ **API is real** — `atlas api serve` exposes `/v1/chat/completions` + `/v1/completions` + `/v1/models`; SSE streaming; CORS; OLMo-3 sampling defaults; think-aware reasoning/content split + answer reserve; 81 tests
- ✅ **Provenance is real** — Schnorr proofs + Groth16 stub (HMAC-SHA256, BLS12-381-compatible interface) + ProvenanceChain; `atlas prove` generates verifiable proofs
- ✅ **Safety is real** — Horn-clause constitution (8 principles, 4 domains, Young 2026 NP-hardness validated); 5-state FSM (`BOOT→NOMINAL→DEGRADED→SAFE_MODE→EMERGENCY_STOP`); CircuitBreaker; append-only audit log
- ✅ **Bridge is real** — `AtlasBridge` with ZK-attested deposit/withdraw, Sepolia chain_id=11155111, Groth16 proof per transaction
- ✅ **MCP is real** — `atlas mcp serve` exposes 28 tools via JSON-RPC 2.0; McpConnectionPool (max 5, 5-min idle eviction); connects to Claude Desktop / Cursor

### Version History

| Version | Theme | Tests |
|---------|-------|-------|
| v0.1.0 | Infrastructure: f32 matmul, backward pass, GPU (7 stages) | 186 |
| v0.2.0 | Real Memory Palace + MCP (28 tools, JSON-RPC 2.0) | 236 |
| v0.3.0 + v0.4.0 | Real Discovery Engine + Validated Model Loading | 260 |
| v0.5.0 | Real Training Loop (LoRA, grad-accum, safetensors checkpoint) | 353 |
| v0.6.0 | Safety FSM + Groth16 stub + ZK Bridge | 383 |
| v0.7.0 | Benchmarks, CI, CHANGELOG, REPRODUCIBILITY | 383 |
| v1.0.0 | Production Release — all milestones complete | 383 |
| **v2.0.0** | **CAS Decay + OODA Feedback + Stigmergic Sampler + GPU dispatch (37.7 tok/s on A100)** | **400** |
| **v3.0.0-α.1** | **atlas-api + PalaceBackend + GPU-resident forward pass + DeepSupervisionTrainer + Horn-clause safety** | **426** |
| **v4.0.0** | **Champagnat n-morphic framework + Issue #7 fix (SWA + YaRN RoPE + config.json auto-patch for OLMo-3-7B)** | **528** |
| **v4.0.1** | **Docs + test cleanup for v4.0.0 / Issue #7** | **528** |
| **v4.0.2** | **BF16 GPU inference path (Issue #9): OLMo-3-7B-Think 4.1 → 19.9 tok/s (4.8×), W16A32, GEMV kernels** | **528** |
| **v4.0.3** | **Math integrity (Issue #11): λ exp decay + ReLU competition threshold. 47/47 GPU model tests.** | **532** |
| **v4.0.4** | **GPT-4 regex tokenizer (Issue #12): full HuggingFace tokenizer.json support. OLMo-3 + SmolLM2 verified. E2E GPU test.** | **539** |
| **v4.0.5** | **Inference pipeline fixes (Issues #13–15): EOS stopping, XorShift64 PRNG, ChatML auto-detection.** | **549** |
| **v4.0.6** | **Sampling controls (Issue #16): repetition penalty, top-p, top-k, min-p, freq/pres penalty. 7-stage pipeline.** | **562** |
| **v4.0.7** | **OLMo-2/3 post-norm + QK-norm fixes: 3 architecture bugs. CPU/GPU logit diff 20.0 → 0.000015. Correct OLMo-3 output.** | **562** |
| **v4.0.8** | **Anti-repetition defaults + think budget. Fixed degenerate think loops. API defaults from olmo3() preset. `<think>` force-close after 200 tokens.** | **565** |
| **v4.0.9** | Think suppression (logit masking at step 0) + max_tokens 512 + filler cleanup + model auto-detect. Clean direct answers. | 579 |
| **v4.1.0** | **Full GPU attention path (zero intra-layer PCIe) + StigmergicHook + cuBLAS TF32 tensor cores + async GPU alloc. OLMo-3-7B: 15.4→61.7 tok/s (4×). atlas-infer crate. Closes #18.** | **600** |
| **v4.2.0** | **HF-reference inference fidelity (3 silent bugs fixed) + 32B AWQ W4 inference on single A100-40GB (GSM8K 100%, MMLU 82%, Code 5/5). OpenRouter endpoints. CI green.** | **631** |

### Crate Status

| Crate | Stage | Tests | Status |
|-------|-------|-------|--------|
| atlas-core | 1 | 2 | ✅ Error types, Result, traits |
| atlas-tensor | 1 | 6 | ✅ CPU+GPU matmul, INT8/INT4, sm_80 kernels (A100); GPU AdamW kernel; sgemm_vec zero-copy; **BF16 GEMV** (`GpuBufBf16`, `sgemv_bf16_kernel`, W16A32 inference path) |
| atlas-grad | 1 | 9 | ✅ GradTape, matmul/relu/add backward |
| atlas-optim | 1 | 6 | ✅ AdamW + CosineScheduler, warmup |
| atlas-quant | 1 | 7 | ✅ INT8, INT4, symmetric scaling |
| CUDA kernels | 1 | — | ✅ tiled GEMM, rmsnorm, rope, silu_mul, AdamW, INT8/INT4 — compiled on A100-SXM4-40GB (sm_80) |
| atlas-json | 2 | 12 | ✅ Recursive descent parser, surrogate pairs |
| atlas-tokenize | 2 | **14** | ✅ GPT-4 regex pre-tokenization (7 alts w/ backtracking), byte-level BPE, HF tokenizer.json; OLMo-3 + SmolLM2 verified |
| atlas-model | 2 | **27** | ✅ OLMo 3 / Llama 3, RoPE, GQA, SwiGLU, SWA, YaRN RoPE, config.json auto-patch; GPU-resident forward pass; **v4.0.7: post-norm + QK-norm**; **v4.0.8: `olmo3()` preset**; **v4.0.9: model auto-detect from weights dir** |
| atlas-palace | 3 | **79** | ✅ A\* search, 5-type pheromones, Active Inference, MMAS, PalaceBackend trait, session_id, PalaceConfig; v4.0.3: `CanonicalPheromoneUpdate` uses `exp(−x)` decay (always positive, smooth, hardware-safe) |
| atlas-mcp | 3 | **32** | ✅ 28 MCP tools, JSON-RPC 2.0, live palace dispatch; McpConnectionPool (max 5, 5-min idle eviction) |
| atlas-api | 3 | **81** | ✅ OpenAI-compatible HTTP: /v1/chat/completions, /v1/completions, /v1/models; SSE streaming; CORS; OLMo-3 sampling defaults; official `<think>` primer with reasoning/content split; **think-budget answer reserve (`ATLAS_ANSWER_RESERVE`), max_tokens default 2048**; client-disconnect abort + request deadline (#25) |
| atlas-trm | 4 | 12 | ✅ TRM-CausalValidator depth-6 RNN, Bayesian combining |
| atlas-http | 5 | 11 | ✅ HTTP/1.1 TcpStream, chunked decoding, curl HTTPS |
| atlas-bayes | 5 | 13 | ✅ BetaPrior, BayesNetwork, QualityGate, Jaccard novelty |
| atlas-causal | 5 | 10 | ✅ PC algorithm, Fisher-Z, standard normal CDF, Meek rules |
| atlas-zk | 5 | **19** | ✅ Schnorr + Groth16 stub (HMAC-SHA256, BLS12-381 interface) |
| atlas-astra | 5 | 15 | ✅ OODA: NASA POWER / WHO GHO / World Bank / ArXiv; OodaFeedback adaptive explore_ratio |
| atlas-corpus | 6 | **79** | ✅ SftTrainer, LoRA (rank=8), grad-accum, safetensors checkpoint; DeepSupervisionTrainer (N_sup 4–16, loss_trace); v4.0.3: `InvasionFitnessScorer` uses `ReLU(cos_sim − 0.2)` competition (α_ij ≥ 0, no mutualism) |
| atlas-safety | 6 | **30** | ✅ Horn-clause constitution (8 principles, 4 domains); 5-state FSM; CircuitBreaker; append-only audit log |
| atlas-bridge | 6 | **8** | ✅ ZK-attested Rings↔ETH interface, Sepolia chain_id=11155111 |
| atlas-infer | 3 | **20** | ✅ `StigmergicHook` trait (per-layer pheromone deposits); `InferEngine` GPU/CPU dispatch; `PheromoneDeposit` aggregation; `generate_streaming` + `generate` with hooks; hooked GPU forward path (v4.1.0) |
| atlas-cli | 7 | **30** | ✅ discover / corpus / train / eval / prove / palace / mcp / api / bench / status |
| **TOTAL** | | **644** | **✅ All passing — `main`, full workspace run verified 2026-07-08** |

> Per-crate counts above are the v4.2.0-era breakdown and may lag as tests are added; the TOTAL is the current verified workspace figure (unit + integration + doc tests).

### Quick Start

```bash
git clone https://github.com/web3guru888/ATLAS.git
cd ATLAS
cargo build --release -p atlas-cli

# Full OODA discovery + training loop
./target/release/atlas discover --cycles 3 --output my-corpus.json
./target/release/atlas train --corpus my-corpus.json --epochs 2
./target/release/atlas prove --claim "CO2 drives warming" --secret deadbeef01020304
./target/release/atlas palace --stats --hot

# OpenAI-compatible API server
./target/release/atlas api serve --model /path/to/model --port 8080

# MCP server (connect to Claude Desktop / Cursor)
./target/release/atlas mcp serve --palace my-palace.json

# Run benchmarks
./target/release/atlas bench --all
```

---

## atlas-mcp — Model Context Protocol Server

ATLAS exposes its memory palace as **28 MCP tools** via stdio JSON-RPC 2.0, ready for Claude Desktop, Cursor, or any MCP client. v4.0.0 adds `McpConnectionPool` — lazy pool (max 5 connections, 5-min idle eviction) preventing connection leaks across concurrent MCP clients.

```bash
# Add to your Claude Desktop config (~/.config/claude/claude_desktop_config.json)
{
  "mcpServers": {
    "atlas-palace": {
      "command": "./target/release/atlas",
      "args": ["mcp", "--palace", "my-palace.json"]
    }
  }
}
```

**Tool categories:**
| Category | Tools | Examples |
|----------|-------|---------|
| Navigation | 8 | `palace_search`, `palace_navigate`, `palace_find_similar` |
| Operations | 5 | `palace_add_wing`, `palace_add_room`, `palace_add_drawer` |
| Knowledge Graph | 7 | `palace_kg_add`, `palace_kg_query`, `palace_kg_contradictions` |
| Stigmergy | 5 | `palace_deposit_pheromones`, `palace_hot_paths`, `palace_cold_spots` |
| Agent Diary | 3 | `palace_create_agent`, `palace_diary_write`, `palace_diary_read` |

Every tool call modifies live palace state. Pheromone trails compound across sessions. Knowledge graphs grow with every interaction.

---

## Benchmarks

ATLAS includes a zero-dependency benchmark suite using `atlas_core::bench::Bench`. Run with:

```bash
cargo test --workspace --exclude atlas-tensor -- --ignored --nocapture
```

**Representative results** (Ubuntu, Rust 1.95, A100-SXM4-40GB, CUDA 13.0):

| Benchmark | Metric | Description |
|-----------|--------|-------------|
| `gpu_inference_smollm2` | **37.7 tok/s** | SmolLM2-135M GPU inference (f32), A100-SXM4-40GB |
| `gpu_benchmark_olmo3_7b_think_bf16` | **61.7 tok/s** | OLMo-3-7B-Think BF16 GPU inference (W16A32, v4.1.0), full GPU attention, cuBLAS TF32 |
| `gpu_benchmark_olmo3_32b_think_w4` | **14.6 tok/s** | OLMo-3-32B-Think AWQ W4 GPU inference (W4A32, v4.2.0), single A100-40GB, custom GEMV kernel |
| `palace_search_1000` | ~50–200 µs/op | TF-IDF semantic search across 1000 drawers |
| `astar_100_nodes` | ~20–100 µs/op | Pheromone-guided A* pathfinding (100-node KG) |
| `pheromone_deposit_decay_1000` | ~5–20 µs/op | 10 deposits + full decay cycle per iteration |
| `kg_query_100_edges` | ~0.5–2 µs/op | KG edge lookup from a source node |
| `rmsnorm_2048` | ~1–5 µs/op | RMSNorm on 2048-dim vector |
| `rope_128dim_apply` | ~50–200 ns/op | RoPE rotation on a single attention head |
| `schnorr_prove_verify` | ~200–500 ns/op | Schnorr ZK proof generation + verification |
| `json_parse_1kb` | ~5–20 µs/op | Parse a 1KB JSON document (zero-dep parser) |

> **Note**: Numbers vary by hardware. Run benchmarks on your own machine for accurate results.

---

## Key Numbers

- **37.7 tok/s** — GPU inference throughput (SmolLM2-135M on A100-SXM4-40GB, v4.0.0)
- **61.7 tok/s** — GPU inference throughput (OLMo-3-7B-Think, BF16 W16A32, A100-SXM4-40GB, v4.1.0; was 15.4 tok/s = **4× speedup** via cuBLAS TF32 + full GPU attention path + async alloc)
- **14.6 tok/s** — GPU inference throughput (OLMo-3-32B-Think, W4A32 AWQ, A100-SXM4-40GB, v4.2.0; custom `gemv_w4_kernel`, streaming shard loader — full 32B reasoning model on a single 40GB GPU)
- **27.2 GB** — VRAM footprint for 32B W4 at 16K context (19.6 GB weights, 13.8 GB headroom)
- **121.8s → 23.4s (~5×)** — 32B W4 time-to-first-token at ~1.25K-token prompts after batched prefill ([#22](https://github.com/web3guru888/ATLAS/issues/22)) merged + deployed (2026-07-08); greedy output bit-identical
- **100% GSM8K · 82% MMLU · 5/5 Code** — 32B W4 quality on standard benchmarks (v4.2.0)
- **2.4×** — GPU speedup over CPU inference (SmolLM2-1.7B: 12.6 vs 5.2 tok/s)
- **507 MiB** — VRAM for pre-pinned SmolLM2-135M weights
- **d = 10.6** — Cohen's d for palace-memory vs. no-memory (ASTRA experiments)
- **34.4×** — more discoveries with memory than without
- **R² = 0.982** — O(1/√T) convergence fit (BUTTERS morphic warm-start)
- **1.83×** — cross-domain novelty acceleration (DC-24 experiment)
- **7M params** — TRM-CausalValidator size vs. 7B base model (1000× smaller)
- **45%** — TRM accuracy on ARC-AGI-1 (Samsung SAIL Montreal, arXiv:2510.04871)
- **<10ms** — target TRM validation latency per causal graph
- **~86K** — quality-gated training examples per month from ASTRA
- **8 principles / 4 domains** — Horn-clause safety constitution (Young 2026, arXiv:2501.15446)

---

## v4.0 — Champagnat n-Morphic Framework ✅ Implemented

ATLAS v4.0 implements the **Champagnat n-Morphic Framework** (Issue #6), grounded in Champagnat-Méléard 2011 (PTRF) and Baar-Bovier-Champagnat 2017 (AAP). All Tier 1 (Sprint 1+2) proposals are live as of v4.0.0:

| Module | Crate | Key idea |
|--------|-------|----------|
| `InvasionFitnessScorer` | atlas-corpus | Replaces raw pheromone softmax; α_ij = ReLU(cos_sim − 0.2) — Lotka-Volterra valid (v4.0.3) |
| `CognitiveBranching` | atlas-astra | Detects explore_ratio plateau → bifurcates OODA |
| `CanonicalPheromoneUpdate` | atlas-palace | Principled decay λ = base_rate × exp(−canonical_term) — always positive, smooth (v4.0.3) |
| `HJConcentrationPrior` | atlas-trm | Hopf-Cole sharpening across TRM recursion steps |
| `PolymorphicTrainer` | atlas-corpus | k=2,3 morphs (fast/slow/creative) with competition matrix |

**Mathematical foundation**: `DeepSupervisionTrainer` IS a k-Morphic Trait Substitution System (exact, not analogy). Each N_sup pass = one phenotypic morph. Champagnat Theorem 3.1 derivably explains TRM's >75% gain from deep supervision. Full theory: see research reports.

---

## Hugging Face Model Card

ATLAS models are published to Hugging Face under the [`openhubresearch`](https://huggingface.co/openhubresearch) organization.

**Latest release**: [`openhubresearch/ATLAS-OLMo-3-32B-Think-v4`](https://huggingface.co/openhubresearch/ATLAS-OLMo-3-32B-Think-v4) — OLMo-3-32B-Think served through the ATLAS v4.2.0 stack with AWQ 4-bit inference (**GSM8K-25 100%, MMLU-40 82%, Code-5 5/5**, 14.6 tok/s, 27.2 GB VRAM, single A100-40GB, custom `gemv_w4_kernel`, streaming shard loader; 644/644 tests on `main` as of 2026-07-08).

**First release**: [`openhubresearch/ATLAS-OLMo-3-7B-Think-v4`](https://huggingface.co/openhubresearch/ATLAS-OLMo-3-7B-Think-v4) — OLMo-3-7B-Think run through the ATLAS v4.1.0 n-morphic framework with BF16 inference (**61.7 tok/s** A100-SXM4-40GB, W16A32, 600/600 tests, full GPU attention path, StigmergicHook wired, think suppression + anti-repetition defaults).

```yaml
---
language: en
license: apache-2.0
library_name: atlas
tags:
  - atlas
  - stigmergic-memory
  - active-inference
  - causal-inference
  - pure-rust
  - zero-dependencies
  - champagnat-morphic
  - bf16-inference
base_model: allenai/OLMo-3-0125-7B
---
```

Models run through ATLAS carry the full n-morphic framework: `InvasionFitnessScorer` (Lotka-Volterra valid competition), `CanonicalPheromoneUpdate` (principled adaptive decay), `BarBovier2017Constraints` (stability gates), `CognitiveBranching` (OODA bifurcation), and `HJConcentrationPrior` (Hopf-Cole sharpening). See [atlasagi.org](https://atlasagi.org) for model releases and the LiveDiscoveryCorpus dataset.

---

## ATLAS Observatory

The **[ATLAS Observatory](https://huggingface.co/spaces/openhubresearch/ATLAS)** is an interactive web demo showcasing the full ATLAS stack — memory palace visualization, live LLM inference, n-morphic evolution, mathematical foundations, and MCP tool playground.

| Tab | What it does |
|-----|-------------|
| 🏛️ **Palace** | 3D force-directed graph of the memory palace with pheromone flow particles, bloom lighting, and semantic fly-to navigation |
| ⚒️ **Forge** | Live chat with OLMo-3-32B-Think (AWQ 4-bit) via SSE streaming, collapsible `<think>` blocks, token confidence visualization, real-time OODA loop display |
| ⚔️ **Arena** | k=1, 2, 4 morphic population competition with branching events, fitness landscapes, and +38% diversity measurement |
| 📚 **Library** | Interactive K↔L↔1/ρ sliders, λ decay charts, Fleming–Viot diagrams, and the full crate dependency tree |
| 🔧 **Workshop** | 12 MCP tool cards with live execution, tree/graph result viewers, operation log, and local-first architecture |

**Tech stack**: 13,659 lines · Three.js + 3d-force-graph · OLMo-3-32B-Think (AWQ 4-bit, 27.2 GB VRAM) · Palace REST API · GPU-accelerated on A100-SXM4-40GB.

🔭 **Try it**: [huggingface.co/spaces/openhubresearch/ATLAS](https://huggingface.co/spaces/openhubresearch/ATLAS) · [demo.thebeastagi.com](https://demo.thebeastagi.com) · Website: [atlasagi.org](https://atlasagi.org)

---

## Operated by The Beast 🤖

ATLAS was built by Robin Dey; day-to-day production operations on the A100 — serving, benchmarking, quantization experiments, and the performance sprints tracked in the issues — are run by **The Beast**, OpenHub Research's autonomous AI agent fleet.

- 🏆 **DoraHacks**: [dorahacks.io/hacker/thebeastagi](https://dorahacks.io/hacker/thebeastagi)
- 🐦 **X**: [@thebeastagi](https://x.com/thebeastagi)
- 🔭 **Live demo**: [demo.thebeastagi.com](https://demo.thebeastagi.com) — chat with OLMo-3-32B-Think served by ATLAS on a single A100-40GB
- 🤗 **Hugging Face**: [openhubresearch](https://huggingface.co/openhubresearch) · [ATLAS Observatory Space](https://huggingface.co/spaces/openhubresearch/ATLAS) · [32B model card](https://huggingface.co/openhubresearch/ATLAS-OLMo-3-32B-Think-v4)
- 🌐 **API**: `https://atlas.thebeastagi.com/v1` — OpenAI-compatible, bearer-key auth

---

## License

- **Code** (`crates/`, `kernels/`, `scripts/`): [Apache 2.0](LICENSE-CODE)
- **Documentation, paper, figures, datasets**: [CC BY 4.0](LICENSE)

© 2026 Robin Dey, OpenHub Research (Thailand)

See [NOTICE](NOTICE) for attribution to incorporated components.

---

## Citation

```bibtex
@software{atlas2026,
  title       = {ATLAS: Active-inference Training with Learned Adaptive Stigmergy},
  author      = {Robin Dey},
  year        = {2026},
  institution = {OpenHub Research, Thailand},
  url         = {https://github.com/web3guru888/ATLAS},
  note        = {Pure Rust LLM training framework. Zero external crate dependencies.
                 v4.2.0: 22 crates, 644 tests, Champagnat n-morphic framework,
                 32B AWQ W4 inference on single A100-40GB — GSM8K 100%, MMLU 82%, Code 5/5.
                 HF-reference inference fidelity: 3 silent bugs fixed by differential testing.
                 Full GPU attention path — OLMo-3-7B-Think 61.7 tok/s on A100-SXM4-40GB (BF16, 4× speedup).
                 StigmergicHook trait: per-layer pheromone deposits into GraphPalace.
                 cuBLAS TF32 tensor cores, async GPU allocator, zero intra-layer PCIe.}
}
```
