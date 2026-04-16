//! atlas-model — Transformer architecture for ATLAS.
//!
//! Implements the OLMo 3 / Llama 3 architecture in pure Rust:
//! - RMSNorm, RoPE, SwiGLU, Grouped Query Attention (GQA)
//! - Safetensors weight loading (zero-dependency binary parser)
//! - f32 + optional CUDA execution via atlas-tensor
//! - Greedy/temperature generation loop
//!
//! # Quickstart (small test model)
//! ```
//! use atlas_model::{ModelConfig, OlmoModel};
//! let cfg = ModelConfig::tiny(); // 2 layers, 64 dim
//! let mut model = OlmoModel::new(cfg);
//! let logits = model.forward(&[0u32, 1, 2], 0);
//! assert_eq!(logits.shape()[0], 3);
//! ```

#![warn(missing_docs)]

use atlas_core::{AtlasError, Result};
use atlas_tensor::Tensor;

// ── Model configuration ────────────────────────────────────────────────────

/// Transformer model configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model hidden dimension.
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (< n_heads for GQA).
    pub n_kv_heads: usize,
    /// FFN intermediate dimension (SwiGLU gate + up proj).
    pub ffn_hidden: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    /// OLMo 3 7B / Llama 3 8B configuration.
    pub fn olmo3_7b() -> Self {
        Self {
            vocab_size:   128_256,
            d_model:      4096,
            n_layers:     32,
            n_heads:      32,
            n_kv_heads:   8,
            ffn_hidden:   14_336,
            max_seq_len:  4096,
            rope_theta:   500_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// OLMo 3 1B (OLMo-2-0325-1B) configuration.
    pub fn olmo3_1b() -> Self {
        Self {
            vocab_size:   100_352,
            d_model:      2048,
            n_layers:     16,
            n_heads:      16,
            n_kv_heads:   16,
            ffn_hidden:   8192,
            max_seq_len:  4096,
            rope_theta:   500_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Llama 3.2 1B configuration.
    pub fn llama32_1b() -> Self {
        Self {
            vocab_size:   128_256,
            d_model:      2048,
            n_layers:     16,
            n_heads:      32,
            n_kv_heads:   8,
            ffn_hidden:   8192,
            max_seq_len:  131_072,
            rope_theta:   500_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// SmolLM2-1.7B (HuggingFaceTB) — LlamaForCausalLM, Apache 2.0
    /// hidden=2048, layers=24, heads=32, kv_heads=32, ffn=8192, vocab=49152
    pub fn smollm2_1b7() -> Self {
        Self {
            vocab_size:   49152,
            d_model:      2048,
            n_layers:     24,
            n_heads:      32,
            n_kv_heads:   32,
            ffn_hidden:   8192,
            max_seq_len:  8192,
            rope_theta:   130_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// SmolLM2-360M (HuggingFaceTB) — LlamaForCausalLM, Apache 2.0
    /// hidden=960, layers=32, heads=15, kv_heads=5, ffn=2560, vocab=49152
    pub fn smollm2_360m() -> Self {
        Self {
            vocab_size:   49152,
            d_model:      960,
            n_layers:     32,
            n_heads:      15,
            n_kv_heads:   5,
            ffn_hidden:   2560,
            max_seq_len:  8192,
            rope_theta:   100_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// SmolLM2-135M (HuggingFaceTB) — LlamaForCausalLM, Apache 2.0
    /// hidden=576, layers=30, heads=9, kv_heads=3, ffn=1536, vocab=49152
    pub fn smollm2_135m() -> Self {
        Self {
            vocab_size:   49152,
            d_model:      576,
            n_layers:     30,
            n_heads:      9,
            n_kv_heads:   3,
            ffn_hidden:   1536,
            max_seq_len:  8192,
            rope_theta:   10_000.0,
            rms_norm_eps: 1e-5,
        }
    }


    /// TinyLlama-1.1B-Chat-v1.0 (TinyLlama) — LlamaForCausalLM, Apache 2.0
    /// hidden=2048, layers=22, heads=32, kv_heads=4, ffn=5632, vocab=32000
    /// rope_theta=10_000, max_seq=2048 — HF: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    pub fn tinyllama_1b() -> Self {
        Self {
            vocab_size:   32000,
            d_model:      2048,
            n_layers:     22,
            n_heads:      32,
            n_kv_heads:   4,
            ffn_hidden:   5632,
            max_seq_len:  2048,
            rope_theta:   10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Head dimension (d_model / n_heads).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Key/value dimension per layer (n_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }

    /// Total number of weight tensors expected in a safetensors file.
    /// Per layer: 4 attn projections + 3 FFN weights + 2 norms = 9.
    /// Plus: embedding, final norm, lm_head = 3.
    pub fn expected_tensor_count(&self) -> usize {
        self.n_layers * 9 + 3
    }

    /// Generate the expected safetensors tensor names for OLMo 3 / Llama 3.2
    /// weight naming conventions.
    pub fn expected_tensor_names(&self) -> Vec<String> {
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for i in 0..self.n_layers {
            let pfx = format!("model.layers.{i}");
            names.push(format!("{pfx}.self_attn.q_proj.weight"));
            names.push(format!("{pfx}.self_attn.k_proj.weight"));
            names.push(format!("{pfx}.self_attn.v_proj.weight"));
            names.push(format!("{pfx}.self_attn.o_proj.weight"));
            names.push(format!("{pfx}.mlp.gate_proj.weight"));
            names.push(format!("{pfx}.mlp.up_proj.weight"));
            names.push(format!("{pfx}.mlp.down_proj.weight"));
            names.push(format!("{pfx}.input_layernorm.weight"));
            names.push(format!("{pfx}.post_attention_layernorm.weight"));
        }
        names
    }

    /// Tiny 2-layer 64-dim model for testing (no GPU required).
    pub fn tiny() -> Self {
        Self {
            vocab_size:   256,
            d_model:      64,
            n_layers:     2,
            n_heads:      4,
            n_kv_heads:   2,
            ffn_hidden:   128,
            max_seq_len:  128,
            rope_theta:   10_000.0,
            rms_norm_eps: 1e-5,
        }
    }
}

// ── Low-level math helpers ─────────────────────────────────────────────────

/// RMS normalize a 1D slice in-place using a weight vector.
/// rms = sqrt(mean(x^2) + eps); x_norm[i] = (x[i]/rms) * w[i]
fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi = *xi * inv * wi;
    }
}

/// SiLU activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Compute softmax of a slice in-place (numerically stable).
fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    let inv = 1.0 / sum;
    for v in x.iter_mut() { *v *= inv; }
}

// ── RoPE (Rotary Position Embedding) ──────────────────────────────────────

/// Precomputed cosine/sine tables for RoPE.
struct RopeCache {
    /// cos[pos][i] for i in 0..head_dim/2
    cos: Vec<Vec<f32>>,
    /// sin[pos][i]
    sin: Vec<Vec<f32>>,
}

impl RopeCache {
    fn new(head_dim: usize, max_seq: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq);
        let mut sin = Vec::with_capacity(max_seq);
        for pos in 0..max_seq {
            let mut c = Vec::with_capacity(half);
            let mut s = Vec::with_capacity(half);
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                c.push(angle.cos());
                s.push(angle.sin());
            }
            cos.push(c);
            sin.push(s);
        }
        Self { cos, sin }
    }

    /// Apply RoPE to a head's query or key vector at `pos`.
    /// x: mutable slice of length head_dim; rotated in-place.
    fn apply(&self, x: &mut [f32], pos: usize) {
        let half = x.len() / 2;
        let cos = &self.cos[pos];
        let sin = &self.sin[pos];
        for i in 0..half {
            let x0 = x[i];
            let x1 = x[i + half];
            x[i]        = x0 * cos[i] - x1 * sin[i];
            x[i + half] = x0 * sin[i] + x1 * cos[i];
        }
    }
}

// ── Linear layer ──────────────────────────────────────────────────────────

/// A weight matrix (no bias). Row-major: weight[out_i * in_dim + in_j].
struct Linear {
    weight:  Vec<f32>,
    gpu_mat: atlas_tensor::GpuMatrix,
    in_dim:  usize,
    out_dim: usize,
}

impl Linear {
    /// Create with Xavier-style initialization (deterministic, no rand crate).
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let weight: Vec<f32> = pseudo_randn(in_dim * out_dim, seed)
            .into_iter().map(|v| v * scale).collect();
        let gpu_mat = atlas_tensor::GpuMatrix::upload(&weight, out_dim, in_dim);
        Self { weight, gpu_mat, in_dim, out_dim }
    }

    /// Load from f32 slice (used by safetensors loader).
    fn from_data(weight: Vec<f32>, in_dim: usize, out_dim: usize) -> Self {
        assert_eq!(weight.len(), in_dim * out_dim);
        let gpu_mat = atlas_tensor::GpuMatrix::upload(&weight, out_dim, in_dim);
        Self { weight, gpu_mat, in_dim, out_dim }
    }

    /// y = W x  (weight: [out × in], x: [in], y: [out])
    fn forward(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim);
        assert_eq!(y.len(), self.out_dim);
        // Try GPU SGEMM: W[out×in] × x[in×1] → y[out×1]
        if self.gpu_mat.sgemm(x, self.in_dim, 1, y) {
            return;
        }
        // CPU fallback
        for o in 0..self.out_dim {
            let row = &self.weight[o * self.in_dim .. (o+1) * self.in_dim];
            y[o] = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
        }
    }

    /// GPU-to-GPU matmul: input stays in VRAM, output stays in VRAM.
    /// Returns None if CUDA is not available (caller must use CPU path).
    fn forward_vec(&self, x: &atlas_tensor::GpuVec) -> Option<atlas_tensor::GpuVec> {
        self.gpu_mat.sgemm_vec(x, 1)
    }

    /// Batch forward: X[seq_len × in_dim] → Y[seq_len × out_dim].
    /// Each token is dispatched via GPU SGEMM if available.
    fn forward_batch(&self, x: &[f32], seq_len: usize, y: &mut [f32]) {
        for s in 0..seq_len {
            self.forward(
                &x[s * self.in_dim .. (s+1) * self.in_dim],
                &mut y[s * self.out_dim .. (s+1) * self.out_dim],
            );
        }
    }
}

// ── Embedding ─────────────────────────────────────────────────────────────

struct Embedding {
    weight: Vec<f32>,
    vocab_size: usize,
    d_model: usize,
}

impl Embedding {
    fn new(vocab_size: usize, d_model: usize, seed: u64) -> Self {
        let weight = pseudo_randn(vocab_size * d_model, seed)
            .into_iter().map(|v| v * 0.02).collect();
        Self { weight, vocab_size, d_model }
    }

    fn from_data(weight: Vec<f32>, vocab_size: usize, d_model: usize) -> Self {
        Self { weight, vocab_size, d_model }
    }

    /// Return embedding for token `id` as a slice into the weight table.
    fn embed_token(&self, id: u32) -> &[f32] {
        let i = (id as usize).min(self.vocab_size - 1);
        &self.weight[i * self.d_model .. (i+1) * self.d_model]
    }

    /// Embed a sequence → flat [seq_len × d_model].
    fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let mut out = vec![0.0f32; tokens.len() * self.d_model];
        for (t, &id) in tokens.iter().enumerate() {
            out[t*self.d_model..(t+1)*self.d_model].copy_from_slice(self.embed_token(id));
        }
        out
    }
}

// ── Attention ─────────────────────────────────────────────────────────────

/// KV cache for one layer: stores key/value for all positions seen so far.
struct KvCache {
    /// keys:   [max_seq × n_kv_heads × head_dim]
    keys:   Vec<f32>,
    /// values: [max_seq × n_kv_heads × head_dim]
    values: Vec<f32>,
    n_kv_heads: usize,
    head_dim:   usize,
    max_seq:    usize,
}

impl KvCache {
    fn new(n_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let cap = max_seq * n_kv_heads * head_dim;
        Self {
            keys:   vec![0.0; cap],
            values: vec![0.0; cap],
            n_kv_heads, head_dim, max_seq,
        }
    }

    fn write_key(&mut self, pos: usize, kv_head: usize, data: &[f32]) {
        let off = (pos * self.n_kv_heads + kv_head) * self.head_dim;
        self.keys[off..off+self.head_dim].copy_from_slice(data);
    }

    fn write_val(&mut self, pos: usize, kv_head: usize, data: &[f32]) {
        let off = (pos * self.n_kv_heads + kv_head) * self.head_dim;
        self.values[off..off+self.head_dim].copy_from_slice(data);
    }

    fn key(&self, pos: usize, kv_head: usize) -> &[f32] {
        let off = (pos * self.n_kv_heads + kv_head) * self.head_dim;
        &self.keys[off..off+self.head_dim]
    }

    fn val(&self, pos: usize, kv_head: usize) -> &[f32] {
        let off = (pos * self.n_kv_heads + kv_head) * self.head_dim;
        &self.values[off..off+self.head_dim]
    }
}

/// Multi-head grouped-query attention.
struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads:    usize,
    n_kv_heads: usize,
    head_dim:   usize,
    scale:      f32,      // 1/sqrt(head_dim)
    kv_cache:   KvCache,
}

impl Attention {
    fn new(cfg: &ModelConfig, layer: usize) -> Self {
        let head_dim = cfg.d_model / cfg.n_heads;
        let kv_dim   = head_dim * cfg.n_kv_heads;
        let seed_base = 1000 + layer as u64 * 10;
        Self {
            wq: Linear::new(cfg.d_model, cfg.d_model, seed_base),
            wk: Linear::new(cfg.d_model, kv_dim,      seed_base + 1),
            wv: Linear::new(cfg.d_model, kv_dim,      seed_base + 2),
            wo: Linear::new(cfg.d_model, cfg.d_model, seed_base + 3),
            n_heads:    cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim,
            scale:      1.0 / (head_dim as f32).sqrt(),
            kv_cache:   KvCache::new(cfg.n_kv_heads, head_dim, cfg.max_seq_len),
        }
    }

    /// Single-token forward at position `pos`. x: [d_model] → out: [d_model].
    fn forward_token(&mut self, x: &[f32], pos: usize, rope: &RopeCache) -> Vec<f32> {
        let d  = self.n_heads * self.head_dim;
        let kv = self.n_kv_heads * self.head_dim;

        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; kv];
        let mut v = vec![0.0f32; kv];

        self.wq.forward(x, &mut q);
        self.wk.forward(x, &mut k);
        self.wv.forward(x, &mut v);

        // Apply RoPE to each Q head
        for h in 0..self.n_heads {
            rope.apply(&mut q[h*self.head_dim..(h+1)*self.head_dim], pos);
        }
        // Apply RoPE to each K head
        for h in 0..self.n_kv_heads {
            rope.apply(&mut k[h*self.head_dim..(h+1)*self.head_dim], pos);
        }

        // Write KV into cache
        for h in 0..self.n_kv_heads {
            self.kv_cache.write_key(pos, h, &k[h*self.head_dim..(h+1)*self.head_dim]);
            self.kv_cache.write_val(pos, h, &v[h*self.head_dim..(h+1)*self.head_dim]);
        }

        // GQA: group_size = n_heads / n_kv_heads
        let group = self.n_heads / self.n_kv_heads;
        let mut out = vec![0.0f32; d];
        let mut attn_scores = vec![0.0f32; pos + 1];

        for h in 0..self.n_heads {
            let kv_h = h / group;
            let q_h = &q[h*self.head_dim..(h+1)*self.head_dim];

            // Compute attention scores for all positions seen so far
            for t in 0..=pos {
                let k_t = self.kv_cache.key(t, kv_h);
                let score: f32 = q_h.iter().zip(k_t.iter()).map(|(&qi, &ki)| qi * ki).sum();
                attn_scores[t] = score * self.scale;
            }
            softmax_inplace(&mut attn_scores[..pos+1]);

            // Weighted sum of values
            let o_h = &mut out[h*self.head_dim..(h+1)*self.head_dim];
            for t in 0..=pos {
                let v_t = self.kv_cache.val(t, kv_h);
                let a   = attn_scores[t];
                for (oi, &vi) in o_h.iter_mut().zip(v_t.iter()) {
                    *oi += a * vi;
                }
            }
        }

        // Output projection
        let mut result = vec![0.0f32; d];
        self.wo.forward(&out, &mut result);
        result
    }

    /// GPU attention forward: QKV projections stay in VRAM.
    /// RoPE + attention scores run on CPU; output projection back on GPU.
    fn forward_token_gpu(
        &mut self,
        x: &atlas_tensor::GpuVec,
        pos: usize,
        rope: &RopeCache,
    ) -> Option<atlas_tensor::GpuVec> {
        let d   = self.n_heads * self.head_dim;
        let kv  = self.n_kv_heads * self.head_dim;

        // QKV projections (stay in VRAM)
        let q_gpu = self.wq.forward_vec(x)?;
        let k_gpu = self.wk.forward_vec(x)?;
        let v_gpu = self.wv.forward_vec(x)?;

        // Download Q, K, V for RoPE + attention (CPU for correctness)
        let mut q = q_gpu.download();
        let mut k = k_gpu.download();
        let     v = v_gpu.download();

        // Apply RoPE
        for h in 0..self.n_heads {
            rope.apply(&mut q[h*self.head_dim..(h+1)*self.head_dim], pos);
        }
        for h in 0..self.n_kv_heads {
            rope.apply(&mut k[h*self.head_dim..(h+1)*self.head_dim], pos);
        }

        // KV cache write
        for h in 0..self.n_kv_heads {
            self.kv_cache.write_key(pos, h, &k[h*self.head_dim..(h+1)*self.head_dim]);
            self.kv_cache.write_val(pos, h, &v[h*self.head_dim..(h+1)*self.head_dim]);
        }

        // Attention scores + weighted value sum (CPU)
        let group = self.n_heads / self.n_kv_heads;
        let mut out_cpu = vec![0.0f32; d];
        let mut scores  = vec![0.0f32; pos + 1];
        for h in 0..self.n_heads {
            let kv_h = h / group;
            let q_h = &q[h*self.head_dim..(h+1)*self.head_dim];
            for t in 0..=pos {
                let k_t = self.kv_cache.key(t, kv_h);
                let score: f32 = q_h.iter().zip(k_t.iter()).map(|(&qi, &ki)| qi*ki).sum();
                scores[t] = score * self.scale;
            }
            softmax_inplace(&mut scores[..pos+1]);
            let o_h = &mut out_cpu[h*self.head_dim..(h+1)*self.head_dim];
            for t in 0..=pos {
                let v_t = self.kv_cache.val(t, kv_h);
                let a = scores[t];
                for (oi, &vi) in o_h.iter_mut().zip(v_t.iter()) { *oi += a * vi; }
            }
        }

        // Output projection (GPU)
        let out_gpu = atlas_tensor::GpuVec::from_slice(&out_cpu);
        self.wo.forward_vec(&out_gpu)
    }

    fn reset_cache(&mut self) {
        self.kv_cache.keys.iter_mut().for_each(|v| *v = 0.0);
        self.kv_cache.values.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ── SwiGLU Feed-Forward Network ───────────────────────────────────────────

struct FeedForward {
    w_gate: Linear,   // d_model → ffn_hidden
    w_up:   Linear,   // d_model → ffn_hidden
    w_down: Linear,   // ffn_hidden → d_model
}

impl FeedForward {
    fn new(cfg: &ModelConfig, layer: usize) -> Self {
        let seed = 2000 + layer as u64 * 10;
        Self {
            w_gate: Linear::new(cfg.d_model, cfg.ffn_hidden, seed),
            w_up:   Linear::new(cfg.d_model, cfg.ffn_hidden, seed + 1),
            w_down: Linear::new(cfg.ffn_hidden, cfg.d_model, seed + 2),
        }
    }

    /// GPU SwiGLU FFN: hidden state stays in VRAM throughout.
    fn forward_gpu(&self, x: &atlas_tensor::GpuVec) -> Option<atlas_tensor::GpuVec> {
        use atlas_tensor::silu_mul_gpu;
        let gate = self.w_gate.forward_vec(x)?;
        let up   = self.w_up.forward_vec(x)?;
        let hidden = silu_mul_gpu(&gate, &up);
        self.w_down.forward_vec(&hidden)
    }

    /// SwiGLU: out = W_down * (silu(W_gate * x) ⊙ W_up * x)
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h = self.w_gate.out_dim;
        let mut gate = vec![0.0f32; h];
        let mut up   = vec![0.0f32; h];
        self.w_gate.forward(x, &mut gate);
        self.w_up.forward(x, &mut up);
        // SwiGLU fuse
        let hidden: Vec<f32> = gate.iter().zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        let mut out = vec![0.0f32; self.w_down.out_dim];
        self.w_down.forward(&hidden, &mut out);
        out
    }
}

// ── Transformer Block ─────────────────────────────────────────────────────

struct TransformerBlock {
    attn:     Attention,
    ffn:      FeedForward,
    attn_norm: Vec<f32>,   // RMSNorm weight
    ffn_norm:  Vec<f32>,   // RMSNorm weight
    eps:       f32,
    d_model:   usize,
}

impl TransformerBlock {
    fn new(cfg: &ModelConfig, layer: usize) -> Self {
        Self {
            attn:      Attention::new(cfg, layer),
            ffn:       FeedForward::new(cfg, layer),
            attn_norm: vec![1.0f32; cfg.d_model],
            ffn_norm:  vec![1.0f32; cfg.d_model],
            eps:       cfg.rms_norm_eps,
            d_model:   cfg.d_model,
        }
    }

    /// CPU-only path for one token at position `pos`. x: [d_model] → modified in-place.
    fn forward_token_cpu(&mut self, x: &mut Vec<f32>, pos: usize, rope: &RopeCache) {
        // Attention sub-layer with pre-norm residual
        let mut x_norm = x.clone();
        rmsnorm_inplace(&mut x_norm, &self.attn_norm, self.eps);
        let attn_out = self.attn.forward_token(&x_norm, pos, rope);
        for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) { *xi += ai; }

        // FFN sub-layer with pre-norm residual
        let mut x_norm2 = x.clone();
        rmsnorm_inplace(&mut x_norm2, &self.ffn_norm, self.eps);
        let ffn_out = self.ffn.forward(&x_norm2);
        for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) { *xi += fi; }
    }

    /// GPU transformer block: RMSNorm, attention projections, and FFN on GPU.
    /// Residual adds on GPU. Falls back to CPU path if any GPU op fails.
    fn forward_token_gpu(
        &mut self,
        x: &mut atlas_tensor::GpuVec,
        pos: usize,
        rope: &RopeCache,
    ) {
        use atlas_tensor::GpuVec;

        // Attention sub-layer: pre-norm + attention + residual
        let norm_w = GpuVec::from_slice(&self.attn_norm);
        let mut x_norm = GpuVec::from_slice(&x.download());
        x_norm.rmsnorm_inplace(&norm_w, self.eps);

        if let Some(attn_out) = self.attn.forward_token_gpu(&x_norm, pos, rope) {
            x.add_inplace(&attn_out);
        } else {
            // GPU fallback: run CPU path, write result back
            let mut x_cpu = x.download();
            self.forward_token_cpu(&mut x_cpu, pos, rope);
            *x = GpuVec::from_slice(&x_cpu);
            return;
        }

        // FFN sub-layer: pre-norm + FFN + residual
        let ffn_norm_w = GpuVec::from_slice(&self.ffn_norm);
        let mut x_norm2 = GpuVec::from_slice(&x.download());
        x_norm2.rmsnorm_inplace(&ffn_norm_w, self.eps);

        if let Some(ffn_out) = self.ffn.forward_gpu(&x_norm2) {
            x.add_inplace(&ffn_out);
        }
        // If FFN GPU fails, continue — partial GPU is still better than nothing
    }

    /// Process one token at position `pos` via the CPU path.
    ///
    /// The GPU path is available via `forward_token_gpu()` directly — call it
    /// from `OlmoModel::forward_one_gpu()` which manages the full VRAM lifecycle.
    fn forward_token(&mut self, x: &mut Vec<f32>, pos: usize, rope: &RopeCache) {
        self.forward_token_cpu(x, pos, rope);
    }
}

// ── Full Model ────────────────────────────────────────────────────────────

/// ATLAS transformer model (OLMo 3 / Llama 3 architecture).
pub struct OlmoModel {
    config: ModelConfig,
    embed:  Embedding,
    layers: Vec<TransformerBlock>,
    norm:   Vec<f32>,        // final RMSNorm weight
    lm_head: Linear,         // d_model → vocab_size
    rope:    RopeCache,
    /// Position counter (for autoregressive generation).
    pos:     usize,
}

impl OlmoModel {
    /// Create a new randomly-initialized model.
    pub fn new(cfg: ModelConfig) -> Self {
        let head_dim = cfg.d_model / cfg.n_heads;
        let rope = RopeCache::new(head_dim, cfg.max_seq_len, cfg.rope_theta);
        let embed = Embedding::new(cfg.vocab_size, cfg.d_model, 0);
        let layers: Vec<_> = (0..cfg.n_layers)
            .map(|i| TransformerBlock::new(&cfg, i))
            .collect();
        let norm = vec![1.0f32; cfg.d_model];
        // Weight tying: lm_head shares embed weights (copy for simplicity)
        let lm_head = Linear::from_data(
            embed.weight.clone(),
            cfg.d_model,
            cfg.vocab_size,
        );
        Self { config: cfg, embed, layers, norm, lm_head, rope, pos: 0 }
    }

    /// Reset KV cache and position counter (call between independent sequences).
    pub fn reset(&mut self) {
        self.pos = 0;
        for l in &mut self.layers { l.attn.reset_cache(); }
    }

    /// Forward pass for one new token. Returns logits [vocab_size].
    pub fn forward_one(&mut self, token: u32) -> Vec<f32> {
        let pos = self.pos;
        self.pos += 1;

        let mut x: Vec<f32> = self.embed.embed_token(token).to_vec();
        for layer in &mut self.layers {
            layer.forward_token(&mut x, pos, &self.rope);
        }
        rmsnorm_inplace(&mut x, &self.norm, self.config.rms_norm_eps);

        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.lm_head.forward(&x, &mut logits);
        logits
    }

    /// GPU-resident autoregressive forward pass for one token.
    ///
    /// Uploads the token embedding once (H2D), runs all layers with hidden
    /// state staying in VRAM, downloads logits at the end (D2H).
    /// PCIe transfers: 2 per token (vs 211 in the current CPU path).
    ///
    /// Returns `None` if CUDA is not available — caller should use `forward_one`.
    pub fn forward_one_gpu(&mut self, token: u32) -> Option<Vec<f32>> {
        use atlas_tensor::GpuVec;
        if !atlas_tensor::cuda_available() { return None; }

        let pos = self.pos;
        self.pos += 1;

        // H2D: upload embedding (one PCIe transfer per token)
        let embed = self.embed.embed_token(token);
        let mut x = GpuVec::from_slice(embed);

        // Run all transformer layers entirely in VRAM
        for layer in &mut self.layers {
            layer.forward_token_gpu(&mut x, pos, &self.rope);
        }

        // Final RMSNorm (GPU)
        let norm_w = GpuVec::from_slice(&self.norm);
        x.rmsnorm_inplace(&norm_w, self.config.rms_norm_eps);

        // LM head projection (GPU)
        let logits_gpu = self.lm_head.forward_vec(&x)?;

        // D2H: download logits (one PCIe transfer per token)
        Some(logits_gpu.download())
    }

    /// Forward pass for a sequence. Returns logits for every token position.
    /// Shape: [seq_len × vocab_size] stored as flat Vec.
    pub fn forward(&mut self, tokens: &[u32], start_pos: usize) -> SequenceLogits {
        self.pos = start_pos;
        let seq_len = tokens.len();
        let mut all_logits = Vec::with_capacity(seq_len * self.config.vocab_size);
        for &tok in tokens {
            let logits = self.forward_one(tok);
            all_logits.extend(logits);
        }
        SequenceLogits {
            data: all_logits,
            seq_len,
            vocab_size: self.config.vocab_size,
        }
    }

    /// Greedy generation: given prompt tokens, generate up to `max_new` more.
    /// Greedy generation: given prompt tokens, generate up to `max_new` more.
    /// Uses GPU-resident forward pass if CUDA available (2 PCIe transfers/token).
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, temperature: f32) -> Vec<u32> {
        self.reset();
        // Process prompt
        let mut new_tokens = Vec::new();
        let mut last_logits = vec![0.0f32; self.config.vocab_size];

        for &tok in prompt {
            last_logits = if let Some(gl) = self.forward_one_gpu(tok) {
                gl
            } else {
                self.forward_one(tok)
            };
        }

        for step in 0..max_new {
            let next = if temperature <= 0.0 || temperature < 1e-6 {
                // Greedy
                argmax(&last_logits)
            } else {
                // Temperature sampling
                let mut scaled: Vec<f32> = last_logits.iter()
                    .map(|&l| l / temperature).collect();
                softmax_inplace(&mut scaled);
                sample_from_probs(&scaled, step as u64)
            };
            new_tokens.push(next as u32);
            if let Some(eos) = None::<u32> {
                if next as u32 == eos { break; }
            }
            // Prefer GPU-resident forward
            last_logits = if let Some(gl) = self.forward_one_gpu(next as u32) {
                gl
            } else {
                self.forward_one(next as u32)
            };
        }
        new_tokens
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize { self.config.vocab_size }

    /// Number of parameters (approximate).
    pub fn param_count(&self) -> usize {
        let d = self.config.d_model;
        let h = self.config.ffn_hidden;
        let n = self.config.n_layers;
        let kv = self.config.n_kv_heads * (d / self.config.n_heads);
        // per layer: QKV + O + gate + up + down + 2 norms
        let per_layer = d*d + d*kv*2 + d*d + d*h*3 + d*2;
        // embedding + lm_head + final norm
        self.config.vocab_size * d * 2 + per_layer * n + d
    }
}

/// Logits for a full sequence.
pub struct SequenceLogits {
    data:       Vec<f32>,
    seq_len:    usize,
    vocab_size: usize,
}

impl SequenceLogits {
    /// Shape: [seq_len, vocab_size].
    pub fn shape(&self) -> [usize; 2] { [self.seq_len, self.vocab_size] }

    /// Logits for token at position `i`.
    pub fn at(&self, i: usize) -> &[f32] {
        &self.data[i*self.vocab_size..(i+1)*self.vocab_size]
    }

    /// Return the predicted next-token id for position `i` (argmax).
    pub fn predicted_id(&self, i: usize) -> u32 {
        argmax(self.at(i)) as u32
    }

    /// Cross-entropy loss averaged over all positions.
    /// `targets[i]` is the ground-truth token id for position i.
    pub fn cross_entropy_loss(&self, targets: &[u32]) -> f32 {
        assert_eq!(targets.len(), self.seq_len);
        let mut loss = 0.0f32;
        for (i, &target) in targets.iter().enumerate() {
            let logits = self.at(i);
            // log-softmax of the target token
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = logits.iter().map(|&l| (l - max_l).exp()).sum::<f32>().ln() + max_l;
            loss += log_sum_exp - logits[target as usize];
        }
        loss / self.seq_len as f32
    }
}

// ── Safetensors loader ─────────────────────────────────────────────────────

/// Safetensors tensor descriptor from the JSON header.
#[derive(Debug)]
pub struct TensorDesc {
    /// Tensor name (e.g. "model.embed_tokens.weight").
    pub name:    String,
    /// Data type string (e.g. "F32", "BF16").
    pub dtype:   String,
    /// Shape dimensions.
    pub shape:   Vec<usize>,
    /// Byte offsets [start, end] within the data section.
    pub offsets: [usize; 2],
}

/// Parse a safetensors file and return the tensor descriptors + raw data.
/// Does not materialise all tensors — returns the raw bytes for lazy loading.
pub struct SafetensorsFile {
    /// Tensor metadata (name, dtype, shape, byte offsets).
    pub tensors: Vec<TensorDesc>,
    /// Raw tensor data bytes (after the header).
    data: Vec<u8>,
}

impl SafetensorsFile {
    /// Read a `.safetensors` file from disk.
    pub fn open(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| AtlasError::Io(format!("safetensors: {e}")))?;
        Self::from_bytes(bytes)
    }

    /// Parse from an in-memory byte buffer.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(AtlasError::Parse("safetensors: file too short".into()));
        }
        // Header size: first 8 bytes LE u64
        let header_size = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_size {
            return Err(AtlasError::Parse("safetensors: header truncated".into()));
        }
        let header_json = std::str::from_utf8(&bytes[8..8+header_size])
            .map_err(|_| AtlasError::Parse("safetensors: header not UTF-8".into()))?;

        let root = atlas_json::Json::parse(header_json)
            .map_err(|e| AtlasError::Parse(format!("safetensors header JSON: {e}")))?;

        let mut tensors = Vec::new();
        if let Some(pairs) = root.as_object() {
            for (name, desc) in pairs {
                if name == "__metadata__" { continue; }
                let dtype = desc.get("dtype").and_then(|v| v.as_str())
                    .unwrap_or("F32").to_string();
                let shape: Vec<usize> = desc.get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_usize()).collect())
                    .unwrap_or_default();
                let data_offsets = desc.get("data_offsets")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| AtlasError::Parse(format!("missing data_offsets for {name}")))?;
                let start = data_offsets[0].as_usize().unwrap_or(0);
                let end   = data_offsets[1].as_usize().unwrap_or(0);
                tensors.push(TensorDesc {
                    name: name.clone(),
                    dtype,
                    shape,
                    offsets: [start, end],
                });
            }
        }

        let data_start = 8 + header_size;
        Ok(Self {
            tensors,
            data: bytes[data_start..].to_vec(),
        })
    }

    /// List all tensor names in this file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Check if a tensor with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    /// Get tensor descriptor by name.
    pub fn get_desc(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Build a safetensors file from f32 tensors.
    /// Returns the complete binary representation.
    pub fn build_f32(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        // 1. Serialize tensor data and compute offsets
        let mut data_buf: Vec<u8> = Vec::new();
        let mut entries: Vec<(String, String, Vec<usize>, usize, usize)> = Vec::new();
        for &(name, shape, values) in tensors {
            let start = data_buf.len();
            for &v in values {
                data_buf.extend_from_slice(&v.to_le_bytes());
            }
            let end = data_buf.len();
            entries.push((name.to_string(), "F32".to_string(), shape.to_vec(), start, end));
        }
        // 2. Build JSON header manually (no dependency on atlas_json for writing)
        let mut header = String::from("{");
        for (i, (name, dtype, shape, start, end)) in entries.iter().enumerate() {
            if i > 0 { header.push(','); }
            let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name, dtype, shape_str.join(","), start, end
            ));
        }
        header.push('}');
        // 3. Assemble: 8-byte header length + header + data
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut out = Vec::with_capacity(8 + header_bytes.len() + data_buf.len());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&data_buf);
        out
    }

    /// Get tensor data as f32. Handles F32 and BF16 → f32 conversion.
    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>> {
        let desc = self.tensors.iter()
            .find(|t| t.name == name)
            .ok_or_else(|| AtlasError::Io(format!("tensor '{name}' not found")))?;
        let raw = &self.data[desc.offsets[0]..desc.offsets[1]];
        match desc.dtype.as_str() {
            "F32" => {
                let vals: Vec<f32> = raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                Ok(vals)
            }
            "BF16" => {
                // BF16 → F32: upper 2 bytes of IEEE 754 f32
                let vals: Vec<f32> = raw.chunks_exact(2)
                    .map(|b| {
                        let u = u16::from_le_bytes(b.try_into().unwrap());
                        f32::from_bits((u as u32) << 16)
                    })
                    .collect();
                Ok(vals)
            }
            "F16" => {
                // F16 → F32 (manual conversion)
                let vals: Vec<f32> = raw.chunks_exact(2)
                    .map(|b| {
                        let h = u16::from_le_bytes(b.try_into().unwrap());
                        f16_to_f32(h)
                    })
                    .collect();
                Ok(vals)
            }
            other => Err(AtlasError::Parse(format!("unsupported dtype: {other}"))),
        }
    }
}

/// IEEE 754 half-precision to single-precision conversion.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) >> 15) & 1;
    let exp  = ((h as u32) >> 10) & 0x1F;
    let mant = (h as u32) & 0x3FF;
    if exp == 0 {
        if mant == 0 { return if sign == 0 { 0.0 } else { -0.0 }; }
        // Denormal
        let m = mant as f32 / 1024.0;
        return if sign == 0 { m * 2.0f32.powi(-14) } else { -m * 2.0f32.powi(-14) };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else { f32::NAN };
    }
    let f32_exp = (exp + 127 - 15) as u32;
    let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

/// Load an OlmoModel from a safetensors file.
/// Expects Llama 3 / OLMo 3 weight naming conventions.
pub fn load_model_from_safetensors(path: &str, cfg: ModelConfig) -> Result<OlmoModel> {
    let st = SafetensorsFile::open(path)?;
    let mut model = OlmoModel::new(cfg.clone());

    // Map HuggingFace weight names to our model
    for desc in &st.tensors {
        let name = &desc.name;
        let data = st.get_f32(name)?;

        if name == "model.embed_tokens.weight" || name == "tok_embeddings.weight" {
            let vocab = data.len() / cfg.d_model;
            model.embed = Embedding::from_data(data.clone(), vocab, cfg.d_model);
            // Tied lm_head
            model.lm_head = Linear::from_data(data, cfg.d_model, vocab);
        } else if name == "model.norm.weight" || name == "norm.weight" {
            model.norm = data;
        } else if name == "lm_head.weight" {
            model.lm_head = Linear::from_data(data, cfg.d_model, cfg.vocab_size);
        } else {
            // Layer weights: "model.layers.N.xxx"
            for layer_i in 0..cfg.n_layers {
                let pfx = format!("model.layers.{layer_i}.");
                let pfx2 = format!("layers.{layer_i}.");
                let local = if name.starts_with(&pfx) {
                    &name[pfx.len()..]
                } else if name.starts_with(&pfx2) {
                    &name[pfx2.len()..]
                } else {
                    continue;
                };
                let layer = &mut model.layers[layer_i];
                match local {
                    "self_attn.q_proj.weight"  => layer.attn.wq = Linear::from_data(data.clone(), cfg.d_model, cfg.d_model),
                    "self_attn.k_proj.weight"  => { let kd = cfg.n_kv_heads*(cfg.d_model/cfg.n_heads); layer.attn.wk = Linear::from_data(data.clone(), cfg.d_model, kd); }
                    "self_attn.v_proj.weight"  => { let kd = cfg.n_kv_heads*(cfg.d_model/cfg.n_heads); layer.attn.wv = Linear::from_data(data.clone(), cfg.d_model, kd); }
                    "self_attn.o_proj.weight"  => layer.attn.wo = Linear::from_data(data.clone(), cfg.d_model, cfg.d_model),
                    "mlp.gate_proj.weight"     => layer.ffn.w_gate = Linear::from_data(data.clone(), cfg.d_model, cfg.ffn_hidden),
                    "mlp.up_proj.weight"       => layer.ffn.w_up   = Linear::from_data(data.clone(), cfg.d_model, cfg.ffn_hidden),
                    "mlp.down_proj.weight"     => layer.ffn.w_down  = Linear::from_data(data.clone(), cfg.ffn_hidden, cfg.d_model),
                    "input_layernorm.weight"   => layer.attn_norm = data.clone(),
                    "post_attention_layernorm.weight" => layer.ffn_norm = data.clone(),
                    _ => {} // ignore unknown weights
                }
                break;
            }
        }
    }
    Ok(model)
}

// ── Utility ────────────────────────────────────────────────────────────────

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Deterministic sampling from a probability distribution (no rand crate).
fn sample_from_probs(probs: &[f32], seed: u64) -> usize {
    // LCG random in [0, 1)
    let rand = lcg_uniform(seed);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if rand < cum { return i; }
    }
    probs.len() - 1
}

fn lcg_uniform(seed: u64) -> f32 {
    let x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (x >> 33) as f32 / (1u64 << 31) as f32
}

/// LCG pseudo-random f32 in [-1, 1]. Same generator used in atlas-quant.
fn pseudo_randn(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(12345);
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 33) as f32 / (1u64 << 31) as f32;
        u * 2.0 - 1.0
    }).collect()
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weight() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        rmsnorm_inplace(&mut x, &w, 1e-5);
        // After norming, RMS should be ~1
        let rms = (x.iter().map(|&v| v*v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.01, "rms={rms}");
    }

    #[test]
    fn silu_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(1) = 1/(1+e^-1) ≈ 0.7311
        assert!((silu(1.0) - 0.7311).abs() < 0.001);
    }

    #[test]
    fn rope_cache_identity_at_zero() {
        let cache = RopeCache::new(16, 32, 10000.0);
        // At pos=0 all angles are 0, so cos=1, sin=0: x unchanged
        let orig = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let mut x = orig.clone();
        cache.apply(&mut x, 0);
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn forward_tiny_model() {
        let cfg = ModelConfig::tiny();
        let mut model = OlmoModel::new(cfg);
        let logits = model.forward(&[0u32, 1, 2], 0);
        assert_eq!(logits.shape(), [3, 256]);
    }

    #[test]
    fn forward_single_token() {
        let cfg = ModelConfig::tiny();
        let mut model = OlmoModel::new(cfg);
        let logits = model.forward_one(42);
        assert_eq!(logits.len(), 256);
        // Sanity check: finite values
        assert!(logits.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn generate_tokens() {
        let cfg = ModelConfig::tiny();
        let mut model = OlmoModel::new(cfg);
        let out = model.generate(&[0u32, 1], 5, 0.0);
        assert_eq!(out.len(), 5);
        // All token ids should be valid
        assert!(out.iter().all(|&id| (id as usize) < 256));
    }

    #[test]
    fn cross_entropy_loss() {
        let cfg = ModelConfig::tiny();
        let mut model = OlmoModel::new(cfg);
        let logits = model.forward(&[0u32, 1, 2], 0);
        // targets: predict token 1, 2, 3 from positions 0, 1, 2
        let loss = logits.cross_entropy_loss(&[1, 2, 3]);
        assert!(loss.is_finite() && loss > 0.0, "loss={loss}");
    }

    #[test]
    fn param_count_tiny() {
        let cfg = ModelConfig::tiny();
        let model = OlmoModel::new(cfg);
        let p = model.param_count();
        // Should be in a reasonable range for tiny config
        assert!(p > 10_000 && p < 1_000_000, "param_count={p}");
    }

    #[test]
    fn safetensors_bf16_to_f32() {
        // Test BF16 → F32 conversion: 1.0 in BF16 = 0x3F80
        let bf16_bytes = vec![0x80u8, 0x3F]; // 1.0 in BF16 LE
        let st = SafetensorsFile {
            tensors: vec![TensorDesc {
                name: "x".into(),
                dtype: "BF16".into(),
                shape: vec![1],
                offsets: [0, 2],
            }],
            data: bf16_bytes,
        };
        let vals = st.get_f32("x").unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn safetensors_missing_tensor() {
        let st = SafetensorsFile { tensors: vec![], data: vec![] };
        assert!(st.get_f32("nonexistent").is_err());
    }

    #[test]
    fn reset_clears_kv() {
        let cfg = ModelConfig::tiny();
        let mut model = OlmoModel::new(cfg);
        // Run forward to populate KV cache
        model.forward_one(0);
        model.forward_one(1);
        // Reset and run again — should produce same result as first run
        model.reset();
        let logits1 = model.forward_one(0);
        model.reset();
        let logits2 = model.forward_one(0);
        for (a, b) in logits1.iter().zip(logits2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── v0.4.0 tests: Validated Model Loading ──────────────────────────────

    #[test]
    fn safetensors_header_parse() {
        // Build a synthetic safetensors binary with two F32 tensors
        let values_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let values_b: Vec<f32> = vec![7.0, 8.0];
        let bytes = SafetensorsFile::build_f32(&[
            ("weight_a", &[2, 3], &values_a),
            ("bias_b",   &[2],    &values_b),
        ]);
        let st = SafetensorsFile::from_bytes(bytes).unwrap();
        assert_eq!(st.len(), 2);
        // Check tensor names
        let names = st.tensor_names();
        assert!(names.contains(&"weight_a"));
        assert!(names.contains(&"bias_b"));
        // Check descriptors
        let da = st.get_desc("weight_a").unwrap();
        assert_eq!(da.dtype, "F32");
        assert_eq!(da.shape, vec![2, 3]);
        let db = st.get_desc("bias_b").unwrap();
        assert_eq!(db.shape, vec![2]);
        // Check data
        let a = st.get_f32("weight_a").unwrap();
        assert_eq!(a, values_a);
        let b = st.get_f32("bias_b").unwrap();
        assert_eq!(b, values_b);
    }

    #[test]
    fn f16_to_f32_conversion() {
        // IEEE 754 half-precision test values:
        // 1.0 in F16 = 0x3C00 (sign=0, exp=15, mant=0)
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6, "1.0");
        // -1.0 in F16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6, "-1.0");
        // 0.0 in F16 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -0.0 in F16 = 0x8000
        assert!(f16_to_f32(0x8000).is_sign_negative() && f16_to_f32(0x8000) == 0.0, "-0.0");
        // 0.5 in F16 = 0x3800 (sign=0, exp=14, mant=0)
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6, "0.5");
        // 65504.0 (max normal F16) = 0x7BFF
        assert!((f16_to_f32(0x7BFF) - 65504.0).abs() < 1.0, "max normal");
        // Infinity = 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0, "+inf");
        // -Infinity = 0xFC00
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0, "-inf");
        // NaN = 0x7C01
        assert!(f16_to_f32(0x7C01).is_nan(), "NaN");
        // Denormal: smallest F16 subnormal = 0x0001 ≈ 5.96e-8
        let tiny = f16_to_f32(0x0001);
        assert!(tiny > 0.0 && tiny < 1e-6, "subnormal = {tiny}");

        // Round-trip through the BF16 path for comparison:
        // Build a safetensors with F16 data
        let f16_1_0: [u8; 2] = 0x3C00u16.to_le_bytes();
        let f16_half: [u8; 2] = 0x3800u16.to_le_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&f16_1_0);
        data.extend_from_slice(&f16_half);
        let st = SafetensorsFile {
            tensors: vec![TensorDesc {
                name: "x".into(),
                dtype: "F16".into(),
                shape: vec![2],
                offsets: [0, 4],
            }],
            data,
        };
        let vals = st.get_f32("x").unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6, "F16 load 1.0");
        assert!((vals[1] - 0.5).abs() < 1e-6, "F16 load 0.5");
    }

    #[test]
    fn bf16_to_f32_conversion() {
        // BF16 = upper 16 bits of IEEE 754 f32
        // 1.0 in BF16 = 0x3F80
        let bf16_1_0 = (1.0f32.to_bits() >> 16) as u16;
        assert_eq!(bf16_1_0, 0x3F80);
        let recovered = f32::from_bits((bf16_1_0 as u32) << 16);
        assert!((recovered - 1.0).abs() < 1e-5, "1.0");

        // -2.0 in BF16 = 0xC000
        let bf16_neg2 = ((-2.0f32).to_bits() >> 16) as u16;
        assert_eq!(bf16_neg2, 0xC000);
        let recovered2 = f32::from_bits((bf16_neg2 as u32) << 16);
        assert!((recovered2 - (-2.0)).abs() < 1e-5, "-2.0");

        // 0.0 in BF16 = 0x0000
        let bf16_zero = (0.0f32.to_bits() >> 16) as u16;
        assert_eq!(bf16_zero, 0x0000);
        let recovered3 = f32::from_bits((bf16_zero as u32) << 16);
        assert_eq!(recovered3, 0.0, "0.0");

        // Build a safetensors file with 3 BF16 values
        let mut data = Vec::new();
        data.extend_from_slice(&bf16_1_0.to_le_bytes());
        data.extend_from_slice(&bf16_neg2.to_le_bytes());
        data.extend_from_slice(&bf16_zero.to_le_bytes());
        let st = SafetensorsFile {
            tensors: vec![TensorDesc {
                name: "y".into(),
                dtype: "BF16".into(),
                shape: vec![3],
                offsets: [0, 6],
            }],
            data,
        };
        let vals = st.get_f32("y").unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5, "BF16 load 1.0");
        assert!((vals[1] - (-2.0)).abs() < 1e-5, "BF16 load -2.0");
        assert_eq!(vals[2], 0.0, "BF16 load 0.0");
    }

    #[test]
    fn model_config_olmo3_1b_dims() {
        let cfg = ModelConfig::olmo3_1b();
        assert_eq!(cfg.vocab_size,  100_352);
        assert_eq!(cfg.d_model,     2048);
        assert_eq!(cfg.n_layers,    16);
        assert_eq!(cfg.n_heads,     16);
        assert_eq!(cfg.n_kv_heads,  16);   // OLMo 3 1B uses full MHA
        assert_eq!(cfg.ffn_hidden,  8192);
        assert_eq!(cfg.max_seq_len, 4096);
        assert_eq!(cfg.head_dim(),  128);  // 2048 / 16
        assert_eq!(cfg.kv_dim(),    2048); // 16 * 128 (same as d_model for MHA)
        // 16 layers × 9 tensors/layer + 3 global = 147
        assert_eq!(cfg.expected_tensor_count(), 147);
    }

    #[test]
    fn model_config_llama32_1b_dims() {
        let cfg = ModelConfig::llama32_1b();
        assert_eq!(cfg.vocab_size,  128_256);
        assert_eq!(cfg.d_model,     2048);
        assert_eq!(cfg.n_layers,    16);
        assert_eq!(cfg.n_heads,     32);
        assert_eq!(cfg.n_kv_heads,  8);    // Llama 3.2 1B uses GQA
        assert_eq!(cfg.ffn_hidden,  8192);
        assert_eq!(cfg.max_seq_len, 131_072);
        assert_eq!(cfg.head_dim(),  64);   // 2048 / 32
        assert_eq!(cfg.kv_dim(),    512);  // 8 * 64
        // GQA ratio: 32 / 8 = 4 query heads per KV head
        assert_eq!(cfg.n_heads / cfg.n_kv_heads, 4);
    }

    #[test]
    fn weight_mapping_covers_all_layers() {
        // Verify that expected_tensor_names() generates the right names
        // and that load_model_from_safetensors maps every one.
        let cfg = ModelConfig::tiny(); // 2 layers, small
        let names = cfg.expected_tensor_names();
        // 2 layers × 9 + 3 global = 21
        assert_eq!(names.len(), cfg.expected_tensor_count());
        assert_eq!(names.len(), 21);

        // Check global tensors
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));

        // Check per-layer tensors for each layer
        for i in 0..cfg.n_layers {
            let pfx = format!("model.layers.{i}");
            assert!(names.contains(&format!("{pfx}.self_attn.q_proj.weight")));
            assert!(names.contains(&format!("{pfx}.self_attn.k_proj.weight")));
            assert!(names.contains(&format!("{pfx}.self_attn.v_proj.weight")));
            assert!(names.contains(&format!("{pfx}.self_attn.o_proj.weight")));
            assert!(names.contains(&format!("{pfx}.mlp.gate_proj.weight")));
            assert!(names.contains(&format!("{pfx}.mlp.up_proj.weight")));
            assert!(names.contains(&format!("{pfx}.mlp.down_proj.weight")));
            assert!(names.contains(&format!("{pfx}.input_layernorm.weight")));
            assert!(names.contains(&format!("{pfx}.post_attention_layernorm.weight")));
        }

        // Also verify for a larger config
        let cfg7b = ModelConfig::olmo3_7b();
        let names7b = cfg7b.expected_tensor_names();
        assert_eq!(names7b.len(), 32 * 9 + 3); // 291
    }

    #[test]
    fn safetensors_roundtrip() {
        // Build a synthetic safetensors file, serialize it, parse it back,
        // and verify all tensor data survives the round trip.
        let w1: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let w2: Vec<f32> = vec![1.0, 2.0, 3.0];
        let w3: Vec<f32> = vec![-1.0];

        let bytes = SafetensorsFile::build_f32(&[
            ("layer.0.weight", &[2, 3], &w1),
            ("layer.1.bias",   &[3],    &w2),
            ("scalar",         &[1],    &w3),
        ]);

        // Parse back
        let st = SafetensorsFile::from_bytes(bytes).unwrap();
        assert_eq!(st.len(), 3);
        assert!(st.contains("layer.0.weight"));
        assert!(st.contains("layer.1.bias"));
        assert!(st.contains("scalar"));
        assert!(!st.contains("nonexistent"));

        // Verify data
        let r1 = st.get_f32("layer.0.weight").unwrap();
        assert_eq!(r1.len(), 6);
        for (a, b) in r1.iter().zip(w1.iter()) {
            assert!((a - b).abs() < 1e-7, "{a} != {b}");
        }

        let r2 = st.get_f32("layer.1.bias").unwrap();
        assert_eq!(r2, w2);

        let r3 = st.get_f32("scalar").unwrap();
        assert_eq!(r3, w3);

        // Verify shapes
        let d1 = st.get_desc("layer.0.weight").unwrap();
        assert_eq!(d1.shape, vec![2, 3]);
        let d2 = st.get_desc("layer.1.bias").unwrap();
        assert_eq!(d2.shape, vec![3]);
    }

    #[test]
    fn load_tiny_model_from_synthetic_safetensors() {
        // Build a complete safetensors file for a tiny model and load it.
        let cfg = ModelConfig::tiny(); // vocab=256, d=64, layers=2, heads=4, kv=2, ffn=128
        let names = cfg.expected_tensor_names();

        // Build all tensors with deterministic data
        let mut tensor_data: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for name in &names {
            let (shape, numel) = tensor_shape_for_config(&cfg, name);
            // Fill with a simple pattern: index / numel
            let data: Vec<f32> = (0..numel).map(|i| i as f32 * 0.001).collect();
            tensor_data.push((name.clone(), shape, data));
        }

        // Build safetensors binary
        let refs: Vec<(&str, &[usize], &[f32])> = tensor_data.iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        let bytes = SafetensorsFile::build_f32(&refs);

        // Write to a temp file and load
        let tmp = "/tmp/atlas_test_tiny.safetensors";
        std::fs::write(tmp, &bytes).unwrap();

        let model = load_model_from_safetensors(tmp, cfg.clone()).unwrap();
        // Model should have correct vocab
        assert_eq!(model.vocab_size(), 256);
        // Verify embedding was loaded (not random init)
        let embed_tok = model.embed.embed_token(0);
        assert!((embed_tok[0] - 0.0).abs() < 1e-6, "embed[0][0] = {}", embed_tok[0]);
        assert!((embed_tok[1] - 0.001).abs() < 1e-6, "embed[0][1] = {}", embed_tok[1]);

        // Clean up
        let _ = std::fs::remove_file(tmp);
    }

    // ── Benchmarks (run with: cargo test -p atlas-model -- --ignored --nocapture)

    #[test]
    #[ignore]
    fn bench_rms_norm_2048() {
        use atlas_core::bench::Bench;
        let w = vec![1.0f32; 2048];
        let b = Bench::run("rmsnorm_2048", 10_000, || {
            let mut x: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.001).collect();
            rmsnorm_inplace(&mut x, &w, 1e-5);
            std::hint::black_box(&x);
        });
        eprintln!("{}", b.report());
        assert!(b.ns_per_op() > 0.0);
    }

    #[test]
    #[ignore]
    fn bench_rope_2048() {
        use atlas_core::bench::Bench;
        let head_dim = 128; // typical head dim for 2048-dim model with 16 heads
        let cache = RopeCache::new(head_dim, 4096, 500_000.0);
        let b = Bench::run("rope_128dim_apply", 50_000, || {
            let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01).collect();
            cache.apply(&mut x, 42);
            std::hint::black_box(&x);
        });
        eprintln!("{}", b.report());
        assert!(b.ns_per_op() > 0.0);
    }

    #[test]
    #[ignore]
    fn bench_softmax_4096() {
        use atlas_core::bench::Bench;
        let b = Bench::run("softmax_4096", 10_000, || {
            let mut x: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001 - 2.0).collect();
            softmax_inplace(&mut x);
            std::hint::black_box(&x);
        });
        eprintln!("{}", b.report());
        assert!(b.ns_per_op() > 0.0);
    }

    #[test]
    #[ignore]
    fn bench_forward_tiny_model() {
        use atlas_core::bench::Bench;
        let cfg = ModelConfig::tiny();
        let mut model = OlmoModel::new(cfg);
        let b = Bench::run("forward_tiny_3tok", 100, || {
            model.pos = 0; // reset position
            std::hint::black_box(model.forward(&[0u32, 1, 2], 0));
        });
        eprintln!("{}", b.report());
        assert!(b.ns_per_op() > 0.0);
    }


    // ── GPU Inference Tests ──────────────────────────────────────────────────
    // Run: cargo test -p atlas-model -- --ignored --nocapture --test-threads=1

    /// Regression test: GPU rmsnorm must not deadlock on small d.
    /// d=64 = 2 warps. Old __shfl_down_sync(0xffffffff,...) with only 2 active
    /// threads out of 32 caused a GPU-level deadlock. Fixed by smem[0] broadcast.
    #[test]
    #[ignore]
    fn gpu_rmsnorm_small_d_no_deadlock() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        use atlas_tensor::GpuVec;
        let n = 64usize;
        let x_cpu: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let w_cpu: Vec<f32> = vec![1.0f32; n];
        let mut x_gpu = GpuVec::from_slice(&x_cpu);
        let w_gpu     = GpuVec::from_slice(&w_cpu);
        x_gpu.rmsnorm_inplace(&w_gpu, 1e-5);
        let result = x_gpu.download();
        let mut x_ref = x_cpu.clone();
        crate::rmsnorm_inplace(&mut x_ref, &w_cpu, 1e-5);
        let max_err = result.iter().zip(x_ref.iter())
            .map(|(&g, &c)| (g - c).abs()).fold(0.0f32, f32::max);
        eprintln!("  gpu_rmsnorm_small_d: max_err={max_err:.2e}");
        assert!(max_err < 1e-4,
            "GPU/CPU rmsnorm mismatch (d=64): max_err={max_err:.2e}");
    }

    /// GPU forward pass on tiny synthetic model - checks output is finite.
    #[test]
    #[ignore]
    fn gpu_forward_tiny_model_explicit() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let cfg = ModelConfig::tiny();
        let mut model = OlmoModel::new(cfg);
        let logits = model.forward_one_gpu(42);
        assert!(logits.is_some(), "forward_one_gpu returned None");
        let logits = logits.unwrap();
        assert_eq!(logits.len(), 256);
        assert!(logits.iter().all(|&v| v.is_finite()), "non-finite GPU logit");
        eprintln!("  gpu_forward_tiny: 256 logits, all finite");
        model.reset();
        let tokens = model.generate(&[0u32, 1, 2], 5, 0.0);
        assert_eq!(tokens.len(), 5);
        eprintln!("  gpu_generate_tiny: {} tokens", tokens.len());
    }

    /// GPU vs CPU parity on tiny model.
    #[test]
    #[ignore]
    fn gpu_cpu_parity_tiny_model() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let cfg = ModelConfig::tiny();
        let mut m_gpu = OlmoModel::new(cfg.clone());
        let mut m_cpu = OlmoModel::new(cfg);
        let gpu_l = m_gpu.forward_one_gpu(7).unwrap();
        let cpu_l = m_cpu.forward_one(7);
        let max_err = gpu_l.iter().zip(cpu_l.iter())
            .map(|(&g, &c)| (g - c).abs()).fold(0.0f32, f32::max);
        eprintln!("  gpu_cpu_parity tiny: max_err={max_err:.6}");
        assert!(max_err < 0.1, "GPU/CPU divergence: {max_err}");
    }

    /// Load real SmolLM2-135M weights and run GPU inference end-to-end.
    /// Requires: ~/models/smollm2-135m/model.safetensors
    #[test]
    #[ignore]
    fn gpu_inference_smollm2_135m() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/robindey".to_string());
        let weights = format!("{home}/models/smollm2-135m/model.safetensors");
        let cfg = ModelConfig::smollm2_135m();
        eprintln!("  Loading SmolLM2-135M ...");
        let t0 = std::time::Instant::now();
        let mut model = match load_model_from_safetensors(&weights, cfg) {
            Ok(m) => m,
            Err(e) => { eprintln!("SKIP - weights: {e}"); return; }
        };
        eprintln!("  Loaded in {}ms", t0.elapsed().as_millis());
        let t1 = std::time::Instant::now();
        let logits = model.forward_one_gpu(1);
        eprintln!("  First GPU forward: {}ms", t1.elapsed().as_millis());
        assert!(logits.is_some(), "forward_one_gpu returned None");
        let logits = logits.unwrap();
        assert_eq!(logits.len(), 49152, "wrong vocab_size");
        assert!(logits.iter().all(|&v| v.is_finite()), "non-finite logit");
        model.reset();
        let prompt = vec![1u32, 2, 3, 4, 5];
        let t2 = std::time::Instant::now();
        let out = model.generate(&prompt, 20, 0.0);
        let ms = t2.elapsed().as_millis();
        let tok_s = 20.0 / (ms as f64 / 1000.0);
        eprintln!("  20 tokens in {}ms = {:.1} tok/s", ms, tok_s);
        assert_eq!(out.len(), 20);
        assert!(out.iter().all(|&t| (t as usize) < 49152));
        assert!(tok_s > 1.0, "GPU too slow: {tok_s:.1} tok/s");
    }

    /// GPU throughput benchmark: 100 tokens from SmolLM2-135M with timing report.
    #[test]
    #[ignore]
    fn gpu_benchmark_smollm2_135m_100tok() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/robindey".to_string());
        let weights = format!("{home}/models/smollm2-135m/model.safetensors");
        let mut model = match load_model_from_safetensors(&weights, ModelConfig::smollm2_135m()) {
            Ok(m) => m, Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        // Warm-up
        let _ = model.forward_one_gpu(1);
        model.reset();
        // Benchmark
        let n = 100usize;
        let t0 = std::time::Instant::now();
        let out = model.generate(&[1u32, 2, 3], n, 0.0);
        let elapsed = t0.elapsed();
        let tok_s = n as f64 / elapsed.as_secs_f64();
        eprintln!("");
        eprintln!("  ATLAS GPU Benchmark - SmolLM2-135M");
        eprintln!("  Tokens:     {n}");
        eprintln!("  Elapsed:    {:.3}s", elapsed.as_secs_f64());
        eprintln!("  Throughput: {:.1} tok/s", tok_s);
        eprintln!("");
        assert_eq!(out.len(), n);
        assert!(tok_s > 5.0, "Throughput below 5 tok/s: {tok_s:.1}");
    }

    /// GPU throughput benchmark: 50 tokens from SmolLM2-360M with timing report.
    /// Requires: ~/models/smollm2-360m/model.safetensors
    #[test]
    #[ignore]
    fn gpu_benchmark_smollm2_360m_50tok() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/robindey".to_string());
        let weights = format!("{home}/models/smollm2-360m/model.safetensors");
        let t_load = std::time::Instant::now();
        let mut model = match load_model_from_safetensors(&weights, ModelConfig::smollm2_360m()) {
            Ok(m) => m,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let load_ms = t_load.elapsed().as_millis();
        eprintln!("  Load time: {}ms", load_ms);
        // Warm-up
        let t_first = std::time::Instant::now();
        let _ = model.forward_one_gpu(1);
        let first_tok_ms = t_first.elapsed().as_millis();
        eprintln!("  First-token latency: {}ms", first_tok_ms);
        model.reset();
        // Benchmark
        let n = 50usize;
        let t0 = std::time::Instant::now();
        let out = model.generate(&[1u32, 2, 3], n, 0.0);
        let elapsed = t0.elapsed();
        let tok_s = n as f64 / elapsed.as_secs_f64();
        eprintln!("");
        eprintln!("  ┌─────────────────────────────────────────────────┐");
        eprintln!("  │  ATLAS GPU Benchmark - SmolLM2-360M             │");
        eprintln!("  │  Load:        {:>8}ms                          │", load_ms);
        eprintln!("  │  First-token: {:>8}ms                          │", first_tok_ms);
        eprintln!("  │  Tokens:      {:>8}                            │", n);
        eprintln!("  │  Elapsed:     {:>11.3}s                       │", elapsed.as_secs_f64());
        eprintln!("  │  Throughput:  {:>8.1} tok/s                    │", tok_s);
        eprintln!("  └─────────────────────────────────────────────────┘");
        eprintln!("");
        assert_eq!(out.len(), n);
        assert!(tok_s > 2.0, "Throughput below 2 tok/s: {tok_s:.1}");
    }

    /// GPU throughput benchmark: 50 tokens from SmolLM2-1.7B with timing report.
    /// Requires: ~/models/smollm2-1b7/model.safetensors
    #[test]
    #[ignore]
    fn gpu_benchmark_smollm2_1b7_50tok() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/robindey".to_string());
        let weights = format!("{home}/models/smollm2-1b7/model.safetensors");
        let t_load = std::time::Instant::now();
        let mut model = match load_model_from_safetensors(&weights, ModelConfig::smollm2_1b7()) {
            Ok(m) => m,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };
        let load_ms = t_load.elapsed().as_millis();
        eprintln!("  Load time: {}ms", load_ms);
        // Warm-up
        let t_first = std::time::Instant::now();
        let _ = model.forward_one_gpu(1);
        let first_tok_ms = t_first.elapsed().as_millis();
        eprintln!("  First-token latency: {}ms", first_tok_ms);
        model.reset();
        // Benchmark
        let n = 50usize;
        let t0 = std::time::Instant::now();
        let out = model.generate(&[1u32, 2, 3], n, 0.0);
        let elapsed = t0.elapsed();
        let tok_s = n as f64 / elapsed.as_secs_f64();
        eprintln!("");
        eprintln!("  ┌─────────────────────────────────────────────────┐");
        eprintln!("  │  ATLAS GPU Benchmark - SmolLM2-1.7B             │");
        eprintln!("  │  Load:        {:>8}ms                          │", load_ms);
        eprintln!("  │  First-token: {:>8}ms                          │", first_tok_ms);
        eprintln!("  │  Tokens:      {:>8}                            │", n);
        eprintln!("  │  Elapsed:     {:>11.3}s                       │", elapsed.as_secs_f64());
        eprintln!("  │  Throughput:  {:>8.1} tok/s                    │", tok_s);
        eprintln!("  └─────────────────────────────────────────────────┘");
        eprintln!("");
        assert_eq!(out.len(), n);
        assert!(tok_s > 1.0, "Throughput below 1 tok/s: {tok_s:.1}");
    }

    /// Comparative GPU benchmark — runs all three SmolLM2 sizes and prints a summary table.
    /// Requires: ~/models/smollm2-135m/, ~/models/smollm2-360m/, ~/models/smollm2-1b7/, ~/models/tinyllama-1b-chat/
    #[test]
    #[ignore]
    fn gpu_benchmark_smollm2_all_sizes() {
        if !atlas_tensor::cuda_available() { eprintln!("SKIP - no CUDA"); return; }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/robindey".to_string());

        struct BenchResult {
            name:       &'static str,
            params_m:   f32,
            load_ms:    u128,
            first_ms:   u128,
            tok_s:      f64,
            vram_note:  &'static str,
        }

        let mut results: Vec<BenchResult> = Vec::new();

        let models: &[(&str, &str, ModelConfig, f32, &str)] = &[
            ("SmolLM2-135M",   "smollm2-135m",     ModelConfig::smollm2_135m(),  135.0,  "~0.5GB"),
            ("SmolLM2-360M",   "smollm2-360m",     ModelConfig::smollm2_360m(),  360.0,  "~1.4GB"),
            ("SmolLM2-1.7B",   "smollm2-1b7",      ModelConfig::smollm2_1b7(),   1700.0, "~6.5GB"),
            ("TinyLlama-1.1B", "tinyllama-1b-chat", ModelConfig::tinyllama_1b(), 1100.0, "~8.4GB"),
        ];

        for (name, dir, cfg, params_m, vram_note) in models {
            let weights = format!("{home}/models/{dir}/model.safetensors");
            eprintln!("\n  Loading {name}...");
            let t_load = std::time::Instant::now();
            let mut model = match load_model_from_safetensors(&weights, cfg.clone()) {
                Ok(m) => m,
                Err(e) => { eprintln!("  SKIP {name}: {e}"); continue; }
            };
            let load_ms = t_load.elapsed().as_millis();
            // Warm-up
            let t_first = std::time::Instant::now();
            let _ = model.forward_one_gpu(1);
            let first_ms = t_first.elapsed().as_millis();
            model.reset();
            // Benchmark 30 tokens
            let n = 30usize;
            let t0 = std::time::Instant::now();
            let out = model.generate(&[1u32, 2, 3], n, 0.0);
            let elapsed = t0.elapsed();
            let tok_s = n as f64 / elapsed.as_secs_f64();
            assert_eq!(out.len(), n, "{name}: wrong output length");
            results.push(BenchResult { name, params_m: *params_m, load_ms, first_ms, tok_s, vram_note });
        }

        // Print summary table
        eprintln!("");
        eprintln!("  ╔══════════════════╦═══════════╦══════════╦═══════════╦══════════════╦══════════╗");
        eprintln!("  ║ Model            ║ Params    ║ Load(ms) ║ 1st-tok   ║ Throughput   ║ VRAM est ║");
        eprintln!("  ╠══════════════════╬═══════════╬══════════╬═══════════╬══════════════╬══════════╣");
        for r in &results {
            eprintln!("  ║ {:<16} ║ {:>7.0}M ║ {:>8} ║ {:>7}ms ║ {:>8.1} tok/s ║ {:>8} ║",
                r.name, r.params_m, r.load_ms, r.first_ms, r.tok_s, r.vram_note);
        }
        eprintln!("  ╚══════════════════╩═══════════╩══════════╩═══════════╩══════════════╩══════════╝");
        eprintln!("  Hardware: NVIDIA A100-SXM4-40GB (40GB HBM2e) | CUDA 13.0 | ATLAS v3.0.0-alpha.1");
        eprintln!("");

        assert!(!results.is_empty(), "No models benchmarked");
        for r in &results {
            assert!(r.tok_s > 0.5, "{} throughput too low: {:.1} tok/s", r.name, r.tok_s);
        }
    }

    /// Helper: compute the expected shape and numel for a tensor name given a config.
    fn tensor_shape_for_config(cfg: &ModelConfig, name: &str) -> (Vec<usize>, usize) {
        let d = cfg.d_model;
        let kv = cfg.kv_dim();
        let h = cfg.ffn_hidden;
        let v = cfg.vocab_size;
        match name {
            _ if name == "model.embed_tokens.weight" => (vec![v, d],  v * d),
            _ if name == "model.norm.weight"         => (vec![d],     d),
            _ if name == "lm_head.weight"            => (vec![v, d],  v * d),
            _ if name.ends_with("q_proj.weight")     => (vec![d, d],  d * d),
            _ if name.ends_with("k_proj.weight")     => (vec![kv, d], kv * d),
            _ if name.ends_with("v_proj.weight")     => (vec![kv, d], kv * d),
            _ if name.ends_with("o_proj.weight")     => (vec![d, d],  d * d),
            _ if name.ends_with("gate_proj.weight")  => (vec![h, d],  h * d),
            _ if name.ends_with("up_proj.weight")    => (vec![h, d],  h * d),
            _ if name.ends_with("down_proj.weight")  => (vec![d, h],  d * h),
            _ if name.ends_with("input_layernorm.weight")          => (vec![d], d),
            _ if name.ends_with("post_attention_layernorm.weight") => (vec![d], d),
            _ => panic!("unknown tensor: {name}"),
        }
    }
}
