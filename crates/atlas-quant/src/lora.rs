//! LoRA (Low-Rank Adaptation) adapters for parameter-efficient fine-tuning.
//!
//! Based on Hu et al. 2021 (arXiv:2106.09685).
//! Pure Rust, zero external dependencies.

use atlas_core::{AtlasError, Result};

/// LoRA configuration.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Intrinsic rank of the adaptation. Default: 8.
    pub rank: usize,
    /// Alpha scaling factor. Effective scale = alpha / rank. Default: 16.0.
    pub alpha: f32,
    /// Dropout probability (0.0 = disabled). Default: 0.0.
    pub dropout: f32,
    /// Names of target weight matrices (informational). Default: ["q_proj", "v_proj"].
    pub target_names: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_names: vec!["q_proj".into(), "v_proj".into()],
        }
    }
}

/// LoRA adapter: adds low-rank update ΔW = (A @ B) * (alpha / rank) to a frozen weight matrix W.
///
/// - `lora_a`: shape [in_features, rank] — random init (scaled normal via Box-Muller + LCG)
/// - `lora_b`: shape [rank, out_features] — zero init
/// - `scale`:  alpha / rank
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// LoRA configuration.
    pub config: LoraConfig,
    /// A matrix: [in_features × rank], row-major.
    pub lora_a: Vec<f32>,
    /// B matrix: [rank × out_features], row-major.
    pub lora_b: Vec<f32>,
    /// in_features (columns of input).
    pub in_features: usize,
    /// out_features (columns of output).
    pub out_features: usize,
    /// Effective scale = alpha / rank.
    pub scale: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter.
    ///
    /// - `lora_a` is initialized with scaled normal distribution (std = 1 / sqrt(rank))
    ///   using Box-Muller + LCG PRNG (no external crate).
    /// - `lora_b` is initialized to zeros.
    pub fn new(config: LoraConfig, in_features: usize, out_features: usize, seed: u64) -> Self {
        let rank = config.rank;
        let scale = config.alpha / rank as f32;

        let a_size = in_features * rank;
        let b_size = rank * out_features;
        let mut lora_a = vec![0.0f32; a_size];

        // Simple LCG PRNG for reproducible init (no rand crate)
        let mut state = seed.wrapping_add(1);
        let std_a = 1.0 / (rank as f32).sqrt();
        for v in &mut lora_a {
            // Box-Muller transform using LCG
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 33) as f32 / u32::MAX as f32 + 1e-10;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 33) as f32 / u32::MAX as f32;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            *v = z * std_a;
        }

        let lora_b = vec![0.0f32; b_size];

        Self { config, lora_a, lora_b, in_features, out_features, scale }
    }

    /// Apply LoRA: output = x @ lora_a @ lora_b * scale
    ///
    /// Input `x`: flat array of length `batch * in_features` (treated as [batch, in_features]).
    /// Returns flat array [batch, out_features].
    pub fn forward(&self, x: &[f32], batch: usize) -> Result<Vec<f32>> {
        let rank = self.config.rank;
        if x.len() != batch * self.in_features {
            return Err(AtlasError::Other(format!(
                "LoRA forward: expected {} inputs, got {}",
                batch * self.in_features,
                x.len()
            )));
        }

        // Step 1: h = x @ lora_a  →  [batch, rank]
        let mut h = vec![0.0f32; batch * rank];
        for b in 0..batch {
            for r in 0..rank {
                let mut acc = 0.0f32;
                for i in 0..self.in_features {
                    acc += x[b * self.in_features + i] * self.lora_a[i * rank + r];
                }
                h[b * rank + r] = acc;
            }
        }

        // Step 2: out = h @ lora_b * scale  →  [batch, out_features]
        let mut out = vec![0.0f32; batch * self.out_features];
        for b in 0..batch {
            for o in 0..self.out_features {
                let mut acc = 0.0f32;
                for r in 0..rank {
                    acc += h[b * rank + r] * self.lora_b[r * self.out_features + o];
                }
                out[b * self.out_features + o] = acc * self.scale;
            }
        }

        Ok(out)
    }

    /// Number of trainable parameters (lora_a + lora_b).
    pub fn param_count(&self) -> usize {
        self.lora_a.len() + self.lora_b.len()
    }

    /// Update lora_b with gradients (SGD step for simplicity).
    /// In practice the optimizer handles this; this is for testing.
    pub fn update_lora_b(&mut self, grads: &[f32], lr: f32) {
        for (w, g) in self.lora_b.iter_mut().zip(grads.iter()) {
            *w -= lr * g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter() -> LoraAdapter {
        LoraAdapter::new(LoraConfig::default(), 64, 64, 42)
    }

    #[test]
    fn lora_init_a_nonzero_b_zero() {
        let adapter = make_adapter();
        // lora_b must be all zeros
        assert!(adapter.lora_b.iter().all(|&v| v == 0.0), "lora_b should be zero-initialized");
        // lora_a must have some nonzero values
        assert!(adapter.lora_a.iter().any(|&v| v != 0.0), "lora_a should be nonzero");
    }

    #[test]
    fn lora_param_count() {
        let adapter = LoraAdapter::new(
            LoraConfig { rank: 4, alpha: 8.0, ..Default::default() },
            32,
            64,
            0,
        );
        // lora_a: 32*4=128, lora_b: 4*64=256
        assert_eq!(adapter.param_count(), 128 + 256);
    }

    #[test]
    fn lora_scale_correct() {
        let cfg = LoraConfig { rank: 4, alpha: 8.0, ..Default::default() };
        let adapter = LoraAdapter::new(cfg, 8, 8, 0);
        assert!((adapter.scale - 2.0).abs() < 1e-6, "scale = alpha/rank = 8/4 = 2.0");
    }

    #[test]
    fn lora_forward_zero_b_gives_zero_output() {
        let adapter = make_adapter();
        // lora_b is zero → output must be zero regardless of input or lora_a
        let x = vec![1.0f32; 64];
        let out = adapter.forward(&x, 1).unwrap();
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&v| v.abs() < 1e-6), "zero lora_b → zero output");
    }

    #[test]
    fn lora_forward_after_b_update_nonzero() {
        let mut adapter = LoraAdapter::new(
            LoraConfig { rank: 2, alpha: 2.0, ..Default::default() },
            4,
            4,
            7,
        );
        // Manually set lora_b to all-ones
        adapter.lora_b = vec![1.0f32; 2 * 4];
        let x = vec![1.0f32; 4];
        let out = adapter.forward(&x, 1).unwrap();
        assert_eq!(out.len(), 4);
        // With lora_a nonzero and lora_b = 1, output should be nonzero
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "nonzero lora_b → nonzero output"
        );
    }

    #[test]
    fn lora_forward_output_shape_batch() {
        let adapter = LoraAdapter::new(
            LoraConfig { rank: 2, alpha: 2.0, ..Default::default() },
            8,
            16,
            1,
        );
        let x = vec![0.5f32; 3 * 8]; // batch=3, in=8
        let out = adapter.forward(&x, 3).unwrap();
        assert_eq!(out.len(), 3 * 16, "output shape [batch, out_features]");
    }

    #[test]
    fn lora_wrong_input_size_errors() {
        let adapter = make_adapter();
        let x = vec![1.0f32; 10]; // wrong size
        assert!(adapter.forward(&x, 1).is_err());
    }
}
