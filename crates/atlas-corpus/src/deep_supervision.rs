//! Deep Supervision Trainer — multi-pass training with latent carry-over.
//!
//! Validated by Samsung SAIL Montréal's TRM paper (arXiv:2510.04871):
//! running N_sup=16 supervised forward passes per batch with carried-over
//! latent state accounts for >75% of TRM's performance improvement on
//! structured reasoning tasks.
//!
//! The key insight: deep supervision encourages the model to make meaningful
//! predictions at each refinement depth, not just at the final layer.
//! When combined with ATLAS's pheromone curriculum (pheromone-weighted
//! corpus sampling), the warm-start latent acts as an external memory
//! signal for the training loop.
//!
//! # Architecture
//! ```text
//! PheromoneCorpus → batch → [forward_fn(x, target, latent_t)] × N_sup
//!                               ↑                         ↓
//!                               └──── latent_{t+1} ───────┘
//!                           Σ loss_t / N_sup → backward → AdamW
//! ```
//!
//! # Reference
//! Jolicoeur-Martineau, A. et al. (2025). "Tiny Recursive Model".
//! arXiv:2510.04871. Samsung SAIL Montréal.
//!
//! # Zero external crate dependencies.

/// Configuration for deep supervision training.
#[derive(Debug, Clone)]
pub struct DeepSupervisionConfig {
    /// Number of supervised forward passes per batch (N_sup in TRM paper).
    /// TRM uses 16; we default to 8 for faster iteration.
    pub n_sup: usize,
    /// Whether to carry latent state from pass t to pass t+1.
    /// When true, each pass can refine the previous pass's latent representation.
    pub carry_latent: bool,
    /// Weight applied to pheromone-warm-start latent initialization.
    /// 0.0 = cold start; 1.0 = fully pheromone-initialized.
    pub pheromone_weight: f32,
    /// Whether to normalize total loss by number of passes.
    pub normalize_loss: bool,
}

impl Default for DeepSupervisionConfig {
    fn default() -> Self {
        Self {
            n_sup: 8,
            carry_latent: true,
            pheromone_weight: 0.3,
            normalize_loss: true,
        }
    }
}

impl DeepSupervisionConfig {
    /// TRM-paper-equivalent configuration: N_sup=16, carry_latent=true.
    pub fn trm_equivalent() -> Self {
        Self {
            n_sup: 16,
            carry_latent: true,
            pheromone_weight: 0.0,
            normalize_loss: true,
        }
    }

    /// Fast ablation configuration: N_sup=1 (equivalent to standard single-pass training).
    pub fn single_pass() -> Self {
        Self {
            n_sup: 1,
            carry_latent: false,
            pheromone_weight: 0.0,
            normalize_loss: false,
        }
    }
}

/// Deep supervision trainer that runs N_sup forward passes per batch,
/// optionally carrying latent state between passes.
///
/// The `compute_loss` method abstracts over the underlying model:
/// any closure implementing `(input, target, Option<latent>) → (loss, new_latent)`
/// can be used, making this compatible with both the SftTrainer MLP and
/// future atlas-model LLM inference.
#[derive(Debug)]
pub struct DeepSupervisionTrainer {
    config: DeepSupervisionConfig,
}

impl DeepSupervisionTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: DeepSupervisionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default() -> Self {
        Self::new(DeepSupervisionConfig::default())
    }

    /// Number of supervised passes per batch.
    pub fn n_sup(&self) -> usize {
        self.config.n_sup
    }

    /// Whether latent is carried between passes.
    pub fn carries_latent(&self) -> bool {
        self.config.carry_latent
    }

    /// Pheromone warm-start weight.
    pub fn pheromone_weight(&self) -> f32 {
        self.config.pheromone_weight
    }

    /// Compute total (optionally normalized) loss across N_sup passes.
    ///
    /// `forward_fn(input, target, latent) → (loss, new_latent)`
    ///
    /// The closure is called N_sup times. If `carry_latent` is true,
    /// the `new_latent` from pass t is passed as `latent` to pass t+1.
    /// If `carry_latent` is false, `None` is always passed.
    pub fn compute_loss<F>(
        &self,
        input: &[f32],
        target: &[f32],
        mut forward_fn: F,
    ) -> f32
    where
        F: FnMut(&[f32], &[f32], Option<Vec<f32>>) -> (f32, Vec<f32>),
    {
        let mut total_loss = 0.0_f32;
        let mut latent: Option<Vec<f32>> = None;

        for _ in 0..self.config.n_sup {
            let (pass_loss, new_latent) =
                forward_fn(input, target, latent.clone());
            total_loss += pass_loss;
            latent = if self.config.carry_latent {
                Some(new_latent)
            } else {
                None
            };
        }

        if self.config.normalize_loss && self.config.n_sup > 0 {
            total_loss / self.config.n_sup as f32
        } else {
            total_loss
        }
    }

    /// Run N_sup passes and return per-pass losses (useful for ablation studies).
    pub fn compute_loss_trace<F>(
        &self,
        input: &[f32],
        target: &[f32],
        mut forward_fn: F,
    ) -> Vec<f32>
    where
        F: FnMut(&[f32], &[f32], Option<Vec<f32>>) -> (f32, Vec<f32>),
    {
        let mut losses = Vec::with_capacity(self.config.n_sup);
        let mut latent: Option<Vec<f32>> = None;

        for _ in 0..self.config.n_sup {
            let (pass_loss, new_latent) =
                forward_fn(input, target, latent.clone());
            losses.push(pass_loss);
            latent = if self.config.carry_latent {
                Some(new_latent)
            } else {
                None
            };
        }

        losses
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = DeepSupervisionConfig::default();
        assert_eq!(cfg.n_sup, 8);
        assert!(cfg.carry_latent);
        assert!((cfg.pheromone_weight - 0.3).abs() < 1e-6);
        assert!(cfg.normalize_loss);
    }

    #[test]
    fn trm_equivalent_config() {
        let cfg = DeepSupervisionConfig::trm_equivalent();
        assert_eq!(cfg.n_sup, 16);
        assert!(cfg.carry_latent);
    }

    #[test]
    fn single_pass_config() {
        let cfg = DeepSupervisionConfig::single_pass();
        assert_eq!(cfg.n_sup, 1);
        assert!(!cfg.carry_latent);
    }

    #[test]
    fn loss_normalized_across_passes() {
        let cfg = DeepSupervisionConfig { n_sup: 4, carry_latent: false,
            pheromone_weight: 0.0, normalize_loss: true };
        let trainer = DeepSupervisionTrainer::new(cfg);
        let loss = trainer.compute_loss(
            &[1.0, 2.0],
            &[1.0, 2.0],
            |_input, _target, _latent| (2.0_f32, vec![0.0_f32]),
        );
        // 4 passes × 2.0 / 4 = 2.0
        assert!((loss - 2.0).abs() < 1e-5, "normalized loss = {}", loss);
    }

    #[test]
    fn loss_unnormalized_when_configured() {
        let cfg = DeepSupervisionConfig { n_sup: 4, carry_latent: false,
            pheromone_weight: 0.0, normalize_loss: false };
        let trainer = DeepSupervisionTrainer::new(cfg);
        let loss = trainer.compute_loss(
            &[1.0],
            &[1.0],
            |_, _, _| (1.0_f32, vec![]),
        );
        // 4 passes × 1.0 = 4.0 (not normalized)
        assert!((loss - 4.0).abs() < 1e-5, "unnormalized loss = {}", loss);
    }

    #[test]
    fn pass_count_equals_n_sup() {
        let cfg = DeepSupervisionConfig { n_sup: 6, ..Default::default() };
        let trainer = DeepSupervisionTrainer::new(cfg);
        let mut count = 0_usize;
        trainer.compute_loss(
            &[],
            &[],
            |_, _, _| { count += 1; (0.0, vec![]) },
        );
        assert_eq!(count, 6);
    }

    #[test]
    fn latent_carried_between_passes() {
        let cfg = DeepSupervisionConfig { n_sup: 3, carry_latent: true,
            pheromone_weight: 0.0, normalize_loss: true };
        let trainer = DeepSupervisionTrainer::new(cfg);
        let mut received_latents: Vec<Option<Vec<f32>>> = Vec::new();
        trainer.compute_loss(
            &[],
            &[],
            |_, _, lat| {
                received_latents.push(lat.clone());
                (0.0, vec![99.0_f32])
            },
        );
        assert_eq!(received_latents.len(), 3);
        assert!(received_latents[0].is_none(),   "first pass should have no latent");
        assert_eq!(received_latents[1], Some(vec![99.0_f32]));
        assert_eq!(received_latents[2], Some(vec![99.0_f32]));
    }

    #[test]
    fn latent_not_carried_when_disabled() {
        let cfg = DeepSupervisionConfig { n_sup: 3, carry_latent: false,
            pheromone_weight: 0.0, normalize_loss: true };
        let trainer = DeepSupervisionTrainer::new(cfg);
        let mut received_latents: Vec<Option<Vec<f32>>> = Vec::new();
        trainer.compute_loss(
            &[],
            &[],
            |_, _, lat| {
                received_latents.push(lat.clone());
                (0.0, vec![42.0_f32])
            },
        );
        // all passes should receive None
        for lat in &received_latents {
            assert!(lat.is_none(), "should not carry latent: {:?}", lat);
        }
    }

    #[test]
    fn loss_trace_length_equals_n_sup() {
        let cfg = DeepSupervisionConfig { n_sup: 5, ..Default::default() };
        let trainer = DeepSupervisionTrainer::new(cfg);
        let trace = trainer.compute_loss_trace(
            &[],
            &[],
            |_, _, _| (1.0, vec![]),
        );
        assert_eq!(trace.len(), 5);
    }

    #[test]
    fn loss_trace_decreasing_simulated() {
        // Simulate a model that improves: loss decreases each pass
        let cfg = DeepSupervisionConfig { n_sup: 4, carry_latent: true,
            pheromone_weight: 0.0, normalize_loss: true };
        let trainer = DeepSupervisionTrainer::new(cfg);
        let mut pass = 0_f32;
        let trace = trainer.compute_loss_trace(
            &[],
            &[],
            |_, _, _| {
                let loss = 1.0 / (pass + 1.0);
                pass += 1.0;
                (loss, vec![pass])
            },
        );
        // trace[0] > trace[1] > trace[2] > trace[3]
        assert!(trace[0] > trace[1]);
        assert!(trace[1] > trace[2]);
        assert!(trace[2] > trace[3]);
    }
}
