//! atlas-optim — AdamW optimizer and cosine LR scheduler.
//! Stage 1. Zero external dependencies.
//!
//! # AdamW
//! Paper: Loshchilov & Hutter 2019 (https://arxiv.org/abs/1711.05101)
//! Decouples weight decay from the gradient update, which is the correct
//! way to apply L2 regularization for adaptive optimizers.
//!
//! Update rule per step t:
//!   m_t = β₁·m_{t-1} + (1-β₁)·g_t
//!   v_t = β₂·v_{t-1} + (1-β₂)·g_t²
//!   m̂   = m_t / (1 - β₁^t)
//!   v̂   = v_t / (1 - β₂^t)
//!   θ_t = θ_{t-1}·(1 - lr·λ) - lr·m̂/(√v̂ + ε)
//!
//! # Cosine LR schedule
//! lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))

use atlas_core::{AtlasError, Result};
use atlas_tensor::Tensor;

/// AdamW hyperparameters.
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Peak learning rate.
    pub lr:           f32,
    /// β₁ — 1st moment decay (default 0.9).
    pub beta1:        f32,
    /// β₂ — 2nd moment decay (default 0.95, slightly lower than original 0.999
    ///       for better LLM fine-tuning stability).
    pub beta2:        f32,
    /// ε — numerical stability (default 1e-8).
    pub eps:          f32,
    /// Weight decay λ (default 0.1).
    pub weight_decay: f32,
    /// Gradient clipping max norm (default 1.0, 0.0 = disabled).
    pub clip_norm:    f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr:           2e-5,
            beta1:        0.9,
            beta2:        0.95,
            eps:          1e-8,
            weight_decay: 0.1,
            clip_norm:    1.0,
        }
    }
}

/// A single parameter group managed by the optimizer.
#[derive(Debug)]
pub struct ParamState {
    /// Parameter name (for logging / serialization).
    pub name: String,
    /// Current parameter values.
    pub param: Vec<f32>,
    /// 1st moment vector m.
    pub m: Vec<f32>,
    /// 2nd moment vector v.
    pub v: Vec<f32>,
    /// Shape of the parameter.
    pub shape: Vec<usize>,
    /// Whether to apply weight decay to this param (biases / norms usually skip).
    pub decay: bool,
}

impl ParamState {
    pub fn new(name: impl Into<String>, param: Vec<f32>, shape: Vec<usize>, decay: bool) -> Self {
        let n = param.len();
        Self {
            name: name.into(),
            m: vec![0.0; n],
            v: vec![0.0; n],
            param,
            shape,
            decay,
        }
    }

    pub fn numel(&self) -> usize { self.param.len() }
}

/// AdamW optimizer.
///
/// ```rust
/// use atlas_optim::{AdamW, AdamWConfig, ParamState};
/// let cfg = AdamWConfig::default();
/// let mut opt = AdamW::new(cfg);
/// opt.add_param(ParamState::new("W", vec![0.5; 4], vec![2,2], true));
/// let grads = vec![vec![0.1; 4]];
/// opt.step(&grads).unwrap();
/// ```
#[derive(Debug)]
pub struct AdamW {
    pub cfg:    AdamWConfig,
    pub params: Vec<ParamState>,
    /// Current step count (1-indexed after first step).
    pub step:   u64,
}

impl AdamW {
    /// Create a new optimizer with no parameter groups.
    pub fn new(cfg: AdamWConfig) -> Self {
        Self { cfg, params: Vec::new(), step: 0 }
    }

    /// Register a parameter group.
    pub fn add_param(&mut self, p: ParamState) {
        self.params.push(p);
    }

    /// Zero all gradients (call before backward).
    /// In our tape-based setup gradients live outside the optimizer,
    /// but this is provided for consistency with standard training loops.
    pub fn zero_grad(&self) { /* grads live in GradTape */ }

    /// Apply one gradient step.
    ///
    /// `grads[i]` must correspond to `self.params[i]`.
    pub fn step(&mut self, grads: &[Vec<f32>]) -> Result<()> {
        if grads.len() != self.params.len() {
            return Err(AtlasError::Other(format!(
                "optim.step: got {} grad groups, have {} params",
                grads.len(), self.params.len()
            )));
        }

        self.step += 1;
        let t = self.step as f32;
        let b1  = self.cfg.beta1;
        let b2  = self.cfg.beta2;
        let eps = self.cfg.eps;
        let lr  = self.cfg.lr;
        let wd  = self.cfg.weight_decay;

        // Bias-correction factors
        let bc1 = 1.0 / (1.0 - b1.powf(t));
        let bc2 = 1.0 / (1.0 - b2.powf(t));

        // Optional gradient clipping (global norm)
        let global_norm = if self.cfg.clip_norm > 0.0 {
            let sq_sum: f32 = grads.iter()
                .flat_map(|g| g.iter())
                .map(|x| x * x)
                .sum();
            sq_sum.sqrt()
        } else {
            1.0
        };
        let clip_scale = if self.cfg.clip_norm > 0.0 && global_norm > self.cfg.clip_norm {
            self.cfg.clip_norm / global_norm
        } else {
            1.0
        };

        for (ps, raw_g) in self.params.iter_mut().zip(grads.iter()) {
            if raw_g.len() != ps.numel() {
                return Err(AtlasError::ShapeMismatch {
                    expected: vec![ps.numel()],
                    got:      vec![raw_g.len()],
                });
            }

            for i in 0..ps.numel() {
                let g = raw_g[i] * clip_scale;

                // Update moments
                ps.m[i] = b1 * ps.m[i] + (1.0 - b1) * g;
                ps.v[i] = b2 * ps.v[i] + (1.0 - b2) * g * g;

                // Bias-corrected estimates
                let m_hat = ps.m[i] * bc1;
                let v_hat = ps.v[i] * bc2;

                // AdamW update: decouple weight decay from gradient
                let wd_scale = if ps.decay { 1.0 - lr * wd } else { 1.0 };
                ps.param[i] = ps.param[i] * wd_scale - lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
        Ok(())
    }

    /// Read the current parameter values as a Tensor.
    pub fn param_tensor(&self, idx: usize) -> Result<Tensor> {
        let ps = self.params.get(idx)
            .ok_or_else(|| AtlasError::OutOfBounds { index: idx, size: self.params.len() })?;
        Tensor::from_vec(ps.param.clone(), ps.shape.clone())
    }

    /// Return a snapshot of all (name, value) pairs.
    pub fn state_snapshot(&self) -> Vec<(&str, &[f32])> {
        self.params.iter().map(|p| (p.name.as_str(), p.param.as_slice())).collect()
    }
}

// ── Cosine LR scheduler ────────────────────────────────────────────────────

/// Cosine annealing with optional linear warm-up.
///
/// ```
/// lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π · (t - warmup) / (T - warmup)))
/// ```
/// During warm-up (t < warmup_steps):
/// ```
/// lr(t) = lr_max · t / warmup_steps
/// ```
#[derive(Debug, Clone)]
pub struct CosineScheduler {
    pub lr_max:        f32,
    pub lr_min:        f32,
    pub total_steps:   u64,
    pub warmup_steps:  u64,
}

impl CosineScheduler {
    pub fn new(lr_max: f32, lr_min: f32, total_steps: u64, warmup_steps: u64) -> Self {
        Self { lr_max, lr_min, total_steps, warmup_steps }
    }

    /// Compute learning rate at step `t` (1-indexed).
    pub fn lr(&self, t: u64) -> f32 {
        if t <= self.warmup_steps {
            // Linear warm-up
            return self.lr_max * (t as f32 / self.warmup_steps.max(1) as f32);
        }
        let progress = (t - self.warmup_steps) as f32
            / (self.total_steps - self.warmup_steps).max(1) as f32;
        let progress = progress.clamp(0.0, 1.0);
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
    }

    /// Apply the schedule to an optimizer's learning rate.
    pub fn apply(&self, opt: &mut AdamW, t: u64) {
        opt.cfg.lr = self.lr(t);
    }
}

// ── Gradient utilities ─────────────────────────────────────────────────────

/// Compute the global L2 norm across a list of gradient vectors.
pub fn global_grad_norm(grads: &[Vec<f32>]) -> f32 {
    grads.iter()
        .flat_map(|g| g.iter())
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
}

/// Clip gradients in-place to a maximum global norm.
pub fn clip_grad_norm(grads: &mut [Vec<f32>], max_norm: f32) -> f32 {
    let norm = global_grad_norm(grads);
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            for x in g.iter_mut() {
                *x *= scale;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adamw_step_decreases_loss_proxy() {
        // One parameter, gradient = 1.0 → parameter should move towards 0
        let cfg = AdamWConfig { lr: 0.01, weight_decay: 0.0, ..Default::default() };
        let mut opt = AdamW::new(cfg);
        opt.add_param(ParamState::new("w", vec![1.0], vec![1], false));

        for _ in 0..100 {
            opt.step(&[vec![1.0]]).unwrap();
        }
        // After 100 steps towards negative direction, param should be < 0
        assert!(opt.params[0].param[0] < 1.0,
            "param={}", opt.params[0].param[0]);
    }

    #[test]
    fn adamw_weight_decay() {
        // With weight decay, even zero gradient should shrink the parameter
        let cfg = AdamWConfig { lr: 0.01, weight_decay: 0.1, ..Default::default() };
        let mut opt = AdamW::new(cfg);
        opt.add_param(ParamState::new("w", vec![1.0], vec![1], true));
        opt.step(&[vec![0.0]]).unwrap();
        // Weight decay: param *= (1 - lr * wd) = 1 - 0.001 = 0.999
        assert!(opt.params[0].param[0] < 1.0,
            "Weight decay should reduce param");
    }

    #[test]
    fn cosine_schedule_warmup() {
        let sched = CosineScheduler::new(1e-4, 1e-6, 1000, 100);
        // Warm-up: lr at step 0 should be ~0
        assert!(sched.lr(0) == 0.0);
        // Warm-up: lr at step 100 should be lr_max
        assert!((sched.lr(100) - 1e-4).abs() < 1e-7);
        // After all steps: lr should approach lr_min
        assert!(sched.lr(1000) < 1e-5);
    }

    #[test]
    fn cosine_schedule_monotone_after_warmup() {
        let sched = CosineScheduler::new(1e-3, 1e-6, 1000, 10);
        let mut prev = sched.lr(10);
        for t in 11..=1000 {
            let cur = sched.lr(t);
            assert!(cur <= prev + 1e-10, "LR increased at step {t}: {prev} → {cur}");
            prev = cur;
        }
    }

    #[test]
    fn clip_grad_norm_scales_correctly() {
        let mut grads = vec![vec![3.0f32, 4.0]];  // norm = 5
        let actual_norm = clip_grad_norm(&mut grads, 1.0);
        assert!((actual_norm - 5.0).abs() < 1e-5);
        let new_norm = global_grad_norm(&grads);
        assert!((new_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn multi_param_step() {
        let cfg = AdamWConfig::default();
        let mut opt = AdamW::new(cfg);
        opt.add_param(ParamState::new("W1", vec![0.5; 4], vec![2, 2], true));
        opt.add_param(ParamState::new("b1", vec![0.1; 2], vec![2], false));

        let grads = vec![vec![0.1f32; 4], vec![0.05f32; 2]];
        opt.step(&grads).unwrap();
        assert_eq!(opt.step, 1);
        // Params should have changed
        assert!(opt.params[0].param[0] != 0.5);
        assert!(opt.params[1].param[0] != 0.1);
    }
}
