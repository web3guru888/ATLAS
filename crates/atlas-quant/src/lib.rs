//! atlas-quant — INT4/INT8 quantization and QLoRA dequantization.
//! Stage 1. Zero external dependencies.
//!
//! Enables loading a 7B model in ~4GB (INT4) instead of ~28GB (f32),
//! making it fit on the T4's 15GB VRAM.
//!
//! # Quantization scheme
//! - INT8: per-row absmax quantization. scale = max(|x|) / 127.
//! - INT4: per-block absmax, packed two nibbles per byte. scale = max(|x|) / 7.
//!         Block size defaults to 64 (QLoRA standard).
//!
//! # QLoRA (Dettmers et al. 2023)
//! The base model weights are stored in INT4.
//! LoRA adapters (A, B matrices) are stored in f32/bf16.
//! Forward pass: dequantize block → matmul in f32 → add LoRA output.
//! This means we never need a full f32 copy of the base weights in VRAM.
//!
//! # Layout
//! - INT8: one `i8` per element + one `f32` scale per row.
//! - INT4: one nibble per element (two packed per byte) + one `f32` scale per block.

use atlas_core::{AtlasError, Result};
use atlas_tensor::Tensor;

pub mod lora;
pub use lora::LoraConfig;

// ── INT8 ──────────────────────────────────────────────────────────────────

/// A row-wise INT8-quantized weight matrix.
#[derive(Debug, Clone)]
pub struct Int8Tensor {
    /// Quantized values, row-major.
    pub data:   Vec<i8>,
    /// Per-row scale factors.
    pub scales: Vec<f32>,
    pub rows:   usize,
    pub cols:   usize,
}

impl Int8Tensor {
    /// Quantize a [rows × cols] f32 tensor to INT8 (per-row absmax).
    pub fn quantize(t: &Tensor) -> Result<Self> {
        if t.ndim() != 2 {
            return Err(AtlasError::Other("Int8Tensor::quantize requires 2D tensor".into()));
        }
        let (rows, cols) = (t.shape()[0], t.shape()[1]);
        let src = t.as_slice()?;
        let mut data   = vec![0i8;  rows * cols];
        let mut scales = vec![0.0f32; rows];

        for r in 0..rows {
            let row = &src[r * cols..(r + 1) * cols];
            let mx  = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let s   = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            scales[r] = s;
            let inv = 1.0 / s;
            for (j, &v) in row.iter().enumerate() {
                data[r * cols + j] = (v * inv).clamp(-127.0, 127.0).round() as i8;
            }
        }
        Ok(Self { data, scales, rows, cols })
    }

    /// Dequantize back to f32 Tensor.
    pub fn dequantize(&self) -> Tensor {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let s = self.scales[r];
            for j in 0..self.cols {
                out[r * self.cols + j] = self.data[r * self.cols + j] as f32 * s;
            }
        }
        Tensor::from_vec(out, vec![self.rows, self.cols]).unwrap()
    }

    /// Memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()                    // i8 data
            + self.scales.len() * 4        // f32 scales
    }

    /// Compression ratio vs f32.
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = self.rows * self.cols * 4;
        f32_bytes as f32 / self.memory_bytes() as f32
    }
}

// ── INT4 ──────────────────────────────────────────────────────────────────

/// Default INT4 block size (QLoRA standard).
pub const INT4_BLOCK_SIZE: usize = 64;

/// A block-wise INT4-quantized weight tensor (QLoRA style).
///
/// Layout: data is packed with two INT4 values per byte.
/// High nibble = even element, low nibble = odd element.
/// Each block of `block_size` elements shares one f32 scale.
#[derive(Debug, Clone)]
pub struct Int4Tensor {
    /// Packed nibbles: ceil(n_elements / 2) bytes.
    pub data:       Vec<u8>,
    /// Per-block scale factors.
    pub scales:     Vec<f32>,
    pub rows:       usize,
    pub cols:       usize,
    pub block_size: usize,
}

impl Int4Tensor {
    /// Quantize a 2D f32 tensor to INT4 with `block_size` blocks.
    pub fn quantize(t: &Tensor, block_size: usize) -> Result<Self> {
        if t.ndim() != 2 {
            return Err(AtlasError::Other("Int4Tensor::quantize requires 2D tensor".into()));
        }
        let (rows, cols) = (t.shape()[0], t.shape()[1]);
        let n = rows * cols;
        let n_blocks = (n + block_size - 1) / block_size;
        let src = t.as_slice()?;

        let mut scales = vec![0.0f32; n_blocks];
        let mut data   = vec![0u8; (n + 1) / 2];

        for b in 0..n_blocks {
            let start = b * block_size;
            let end   = (start + block_size).min(n);
            let block = &src[start..end];
            let mx    = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let s     = if mx > 0.0 { mx / 7.0 } else { 1.0 };
            scales[b] = s;
            let inv = 1.0 / s;
            for (j, &v) in block.iter().enumerate() {
                let q = (v * inv).clamp(-7.0, 7.0).round() as i8;
                let nibble = (q & 0x0F) as u8;
                let idx    = start + j;
                if idx % 2 == 0 {
                    data[idx / 2] = (data[idx / 2] & 0x0F) | (nibble << 4);
                } else {
                    data[idx / 2] = (data[idx / 2] & 0xF0) | nibble;
                }
            }
        }
        Ok(Self { data, scales, rows, cols, block_size })
    }

    /// Dequantize a single block (idx 0-based element index) to f32.
    #[inline]
    fn dequant_nibble(byte: u8, high: bool, scale: f32) -> f32 {
        let nibble = if high { byte >> 4 } else { byte & 0x0F };
        // Sign-extend 4-bit: values 0-7 → 0..7, 8-15 → -8..-1
        let signed = if nibble >= 8 { nibble as i8 - 16 } else { nibble as i8 };
        signed as f32 * scale
    }

    /// Dequantize back to f32 Tensor.
    pub fn dequantize(&self) -> Tensor {
        let n = self.rows * self.cols;
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let b     = i / self.block_size;
            let s     = self.scales[b];
            let byte  = self.data[i / 2];
            let high  = i % 2 == 0;
            out[i] = Self::dequant_nibble(byte, high, s);
        }
        Tensor::from_vec(out, vec![self.rows, self.cols]).unwrap()
    }

    /// Memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Compression ratio vs f32.
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = self.rows * self.cols * 4;
        f32_bytes as f32 / self.memory_bytes() as f32
    }
}

// ── QLoRA adapter ─────────────────────────────────────────────────────────

/// A QLoRA low-rank adapter: B×A in f32, applied to a quantized base weight.
///
/// Forward: out = x × dequant(W_int4) + x × A × B × scale
/// where scale = alpha / rank.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Down-projection [in_features × rank].
    pub a:     Vec<f32>,
    /// Up-projection [rank × out_features].
    pub b:     Vec<f32>,
    pub rank:  usize,
    pub alpha: f32,
    pub in_features:  usize,
    pub out_features: usize,
}

impl LoraAdapter {
    /// Create a LoRA adapter with A~N(0, 1/rank) init and B=0 init.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        // A init: Kaiming uniform-like for stability
        let scale = (2.0f32 / (in_features as f32 + rank as f32)).sqrt();
        let a: Vec<f32> = (0..in_features * rank)
            .map(|i| {
                // Deterministic pseudo-random for reproducibility (no rand crate!)
                let x = ((i as f32 * 6364136223846793005.0 + 1442695040888963407.0)
                    .to_bits() as f32) / u32::MAX as f32 - 0.5;
                x * 2.0 * scale
            })
            .collect();
        let b = vec![0.0f32; rank * out_features]; // B=0 so LoRA output starts at 0
        Self { a, b, rank, alpha, in_features, out_features }
    }

    /// Apply LoRA: lora_out = (x × A × B) × (alpha / rank)
    /// x: [batch × in_features] → out: [batch × out_features]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.numel() / self.in_features;
        let a_t = Tensor::from_vec(self.a.clone(), vec![self.in_features, self.rank])?;
        let b_t = Tensor::from_vec(self.b.clone(), vec![self.rank, self.out_features])?;
        let x2  = x.reshape(vec![batch, self.in_features])?;
        let xa  = x2.matmul(&a_t)?;     // [batch, rank]
        let xab = xa.matmul(&b_t)?;     // [batch, out_features]
        Ok(xab.scale(self.alpha / self.rank as f32))
    }

    /// Parameter count.
    pub fn param_count(&self) -> usize {
        self.a.len() + self.b.len()
    }
}

/// Estimate T4 VRAM usage for a model with QLoRA.
/// Returns (base_gb, lora_gb, total_gb).
pub fn estimate_vram_gb(
    param_count: usize,
    lora_rank: usize,
    n_lora_layers: usize,
    hidden_size: usize,
) -> (f32, f32, f32) {
    let base_gb  = (param_count as f32 * 0.5) / 1e9;  // INT4 = 0.5 bytes/param
    let lora_per = 2 * hidden_size * lora_rank * 4;     // A+B in f32
    let lora_gb  = (lora_per * n_lora_layers) as f32 / 1e9;
    (base_gb, lora_gb, base_gb + lora_gb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_tensor::Tensor;

    fn make_tensor(rows: usize, cols: usize, val: f32) -> Tensor {
        Tensor::full(&[rows, cols], val)
    }

    #[test]
    fn int8_roundtrip() {
        let t   = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();
        let q   = Int8Tensor::quantize(&t).unwrap();
        let out = q.dequantize();
        let s   = out.as_slice().unwrap();
        for (&orig, &deq) in [1.0f32,-2.0,3.0,-4.0].iter().zip(s.iter()) {
            assert!((orig - deq).abs() < 0.1, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn int8_compression_ratio() {
        let t = make_tensor(128, 128, 1.0);
        let q = Int8Tensor::quantize(&t).unwrap();
        // INT8 ratio vs f32: ~3.8× (not 4× due to scale storage)
        assert!(q.compression_ratio() > 3.5, "ratio={}", q.compression_ratio());
    }

    #[test]
    fn int4_roundtrip_small_error() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let t   = Tensor::from_vec(data.clone(), vec![1, 64]).unwrap();
        let q   = Int4Tensor::quantize(&t, INT4_BLOCK_SIZE).unwrap();
        let out = q.dequantize();
        let s   = out.as_slice().unwrap();
        let max_err = data.iter().zip(s.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // INT4 quantization error ≤ scale/2 = max(|x|)/7/2 ≈ 0.3 for this range
        assert!(max_err < 0.5, "max_err={max_err}");
    }

    #[test]
    fn int4_compression_ratio() {
        let t = make_tensor(128, 128, 1.0);
        let q = Int4Tensor::quantize(&t, INT4_BLOCK_SIZE).unwrap();
        // INT4 ratio vs f32: ~7.5× (not 8× due to scale storage)
        assert!(q.compression_ratio() > 7.0, "ratio={}", q.compression_ratio());
    }

    #[test]
    fn lora_forward_shape() {
        let lora = LoraAdapter::new(64, 32, 4, 16.0);
        let x = Tensor::zeros(&[2, 64]);
        let out = lora.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 32]);
    }

    #[test]
    fn lora_output_zero_init() {
        // B=0 init means LoRA output starts at 0 for any x
        let lora = LoraAdapter::new(32, 16, 4, 16.0);
        let x = Tensor::full(&[1, 32], 1.0);
        let out = lora.forward(&x).unwrap();
        let s = out.as_slice().unwrap();
        assert!(s.iter().all(|&v| v.abs() < 1e-6), "LoRA output should be ~0 with B=0 init");
    }

    #[test]
    fn vram_estimate_7b() {
        // OLMo 3 7B: ~7B params, 32 layers, hidden=4096, rank=16
        let (base, lora, total) = estimate_vram_gb(7_000_000_000, 16, 32, 4096);
        println!("7B QLoRA estimate: base={:.2}GB lora={:.2}GB total={:.2}GB", base, lora, total);
        // Should fit in T4 15GB
        assert!(total < 15.0, "Won't fit in T4! total={total}GB");
        assert!(base > 3.0, "Base model suspiciously small");
    }
}
