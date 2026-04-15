//! SFT training loop — wires atlas-grad + atlas-optim for supervised fine-tuning.
//!
//! # Architecture
//!
//! ```text
//! LiveDiscoveryCorpus ──→ SimpleTokenizer ──→ TokenBatch
//!                                               │
//!                         TrainableMlp ←────────┘
//!                           │ forward on GradTape
//!                           ↓
//!                         cross_entropy(logits, target)
//!                           │ backward_with_grad
//!                           ↓
//!                         AdamW.step(grads) + CosineScheduler
//! ```
//!
//! The [`SftTrainer`] wires all ATLAS training crates into a single end-to-end
//! pipeline: corpus provides pheromone-weighted data, the tokenizer converts
//! text to token IDs, a 2-layer MLP runs forward on the [`GradTape`], and
//! [`AdamW`] with [`CosineScheduler`] updates the parameters.
//!
//! # Zero external crate dependencies.

use atlas_core::{AtlasError, Result};
use atlas_grad::GradTape;
use atlas_optim::{AdamW, AdamWConfig, CosineScheduler, ParamState};
use atlas_tensor::Tensor;

use crate::{LiveDiscoveryCorpus, SampleStrategy};

// ────────────────────────────────────────────────────────────────────────────
//  Configuration
// ────────────────────────────────────────────────────────────────────────────

/// SFT training configuration.
#[derive(Debug, Clone)]
pub struct SftConfig {
    /// Vocabulary size for the trainable MLP. Default: 100.
    pub vocab_size: usize,
    /// Hidden dimension for the 2-layer MLP. Default: 32.
    pub hidden_dim: usize,
    /// Maximum training epochs. Default: 10.
    pub max_epochs: usize,
    /// Corpus entries to sample per epoch. Default: 4.
    pub batch_size: usize,
    /// Peak learning rate. Default: 0.01.
    pub lr: f32,
    /// Minimum learning rate for cosine schedule. Default: 1e-5.
    pub lr_min: f32,
    /// Warm-up steps for cosine schedule. Default: 5.
    pub warmup_steps: u64,
    /// Weight decay coefficient. Default: 0.01.
    pub weight_decay: f32,
    /// Random seed for model initialization. Default: 42.
    pub seed: u64,
    /// Gradient accumulation steps. Effective batch = batch_size * accum_steps. Default: 1.
    pub accum_steps: usize,
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100,
            hidden_dim: 32,
            max_epochs: 10,
            batch_size: 4,
            lr: 0.01,
            lr_min: 1e-5,
            warmup_steps: 5,
            weight_decay: 0.01,
            seed: 42,
            accum_steps: 1,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Tokenizer
// ────────────────────────────────────────────────────────────────────────────

/// Byte-level tokenizer: maps each byte to `byte % vocab_size`.
///
/// This is intentionally simple — the real tokenizer lives in `atlas-tokenize`
/// (GPT-2 BPE). This one exists so the training loop can function without
/// requiring a BPE merge table.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    vocab_size: usize,
}

impl SimpleTokenizer {
    /// Create a tokenizer with the given vocabulary size (minimum 2).
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size: vocab_size.max(2) }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize { self.vocab_size }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| (b as u32) % self.vocab_size as u32).collect()
    }

    /// Create next-token prediction pairs from text.
    ///
    /// For text "abc" → inputs=[a,b], targets=[b,c].
    pub fn make_pairs(&self, text: &str) -> TokenBatch {
        let tokens = self.encode(text);
        if tokens.len() < 2 {
            return TokenBatch { inputs: vec![], targets: vec![] };
        }
        TokenBatch {
            inputs: tokens[..tokens.len() - 1].to_vec(),
            targets: tokens[1..].to_vec(),
        }
    }
}

/// A batch of (input, target) token pairs for next-token prediction.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    /// Input tokens (position i).
    pub inputs: Vec<u32>,
    /// Target tokens (position i+1).
    pub targets: Vec<u32>,
}

impl TokenBatch {
    /// Number of training pairs.
    pub fn len(&self) -> usize { self.inputs.len() }
    /// True if no pairs.
    pub fn is_empty(&self) -> bool { self.inputs.is_empty() }
}

// ────────────────────────────────────────────────────────────────────────────
//  Cross-entropy loss
// ────────────────────────────────────────────────────────────────────────────

/// Compute cross-entropy loss and its gradient w.r.t. logits.
///
/// Uses the numerically stable log-sum-exp trick.
///
/// Returns `(loss, gradient)` where `gradient = softmax(logits) − one_hot(target)`.
///
/// # Example
/// ```rust
/// use atlas_corpus::cross_entropy;
/// let logits = vec![1.0, 2.0, 0.5];
/// let (loss, grad) = cross_entropy(&logits, 1);
/// assert!(loss > 0.0);
/// assert!(grad[1] < 0.0); // target class has negative gradient
/// ```
pub fn cross_entropy(logits: &[f32], target: u32) -> (f32, Vec<f32>) {
    let t = (target as usize).min(logits.len().saturating_sub(1));

    // Numerically stable softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum_exp: f32 = exp.iter().sum();
    let softmax: Vec<f32> = exp.iter().map(|&e| e / sum_exp).collect();

    // Loss = −log(softmax[target])
    let loss = -(softmax[t].max(1e-10)).ln();

    // Gradient: softmax − one_hot(target)
    let mut grad = softmax;
    grad[t] -= 1.0;

    (loss, grad)
}

// ────────────────────────────────────────────────────────────────────────────
//  Trainable MLP
// ────────────────────────────────────────────────────────────────────────────

/// A 2-layer MLP for next-token prediction, trainable via [`GradTape`].
///
/// Architecture:
/// ```text
/// one_hot(token) × Embed  →  × W1  →  ReLU  →  × W2  →  logits
///   [1,V]×[V,H]=[1,H]     [1,H]×[H,H]=[1,H]  [1,H]×[H,V]=[1,V]
/// ```
///
/// All operations are recorded on the tape so that `backward_with_grad`
/// can compute exact gradients for Embed, W1, and W2.
pub struct TrainableMlp {
    /// Embedding matrix, flat `[vocab_size × hidden_dim]`.
    pub embed: Vec<f32>,
    /// Hidden-layer weight matrix, flat `[hidden_dim × hidden_dim]`.
    pub w1: Vec<f32>,
    /// Output-projection matrix, flat `[hidden_dim × vocab_size]`.
    pub w2: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl TrainableMlp {
    /// Construct with Xavier-uniform initialization.
    pub fn new(vocab_size: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = seed;
        let embed = xavier_init(vocab_size * hidden_dim, vocab_size, hidden_dim, &mut rng);
        let w1    = xavier_init(hidden_dim * hidden_dim, hidden_dim, hidden_dim, &mut rng);
        let w2    = xavier_init(hidden_dim * vocab_size, hidden_dim, vocab_size, &mut rng);
        Self { embed, w1, w2, vocab_size, hidden_dim }
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.embed.len() + self.w1.len() + self.w2.len()
    }

    /// Forward pass on a [`GradTape`].
    ///
    /// Returns `(logits_idx, embed_idx, w1_idx, w2_idx)` — the tape indices
    /// needed to extract gradients after backward.
    pub fn forward_on_tape(
        &self,
        tape: &mut GradTape,
        input_token: u32,
    ) -> Result<(usize, usize, usize, usize)> {
        let v = self.vocab_size;
        let h = self.hidden_dim;

        // 1. One-hot input [1, V]
        let mut one_hot = vec![0.0f32; v];
        one_hot[(input_token as usize) % v] = 1.0;
        let oh_idx = tape.push(Tensor::from_vec(one_hot, vec![1, v])?);

        // 2. Embedding matrix [V, H]
        let embed_idx = tape.push(Tensor::from_vec(self.embed.clone(), vec![v, h])?);

        // 3. Embedding lookup: one_hot × Embed → [1, H]
        let h0 = tape.matmul(oh_idx, embed_idx)?;

        // 4. W1 [H, H]
        let w1_idx = tape.push(Tensor::from_vec(self.w1.clone(), vec![h, h])?);

        // 5. Hidden: h0 × W1 → [1, H]
        let h1 = tape.matmul(h0, w1_idx)?;

        // 6. Activation: ReLU → [1, H]
        let h2 = tape.relu(h1);

        // 7. W2 [H, V]
        let w2_idx = tape.push(Tensor::from_vec(self.w2.clone(), vec![h, v])?);

        // 8. Output: h2 × W2 → [1, V] = logits
        let logits_idx = tape.matmul(h2, w2_idx)?;

        Ok((logits_idx, embed_idx, w1_idx, w2_idx))
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Metrics
// ────────────────────────────────────────────────────────────────────────────

/// Metrics for a single training step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Global step number (1-indexed).
    pub step: u64,
    /// Cross-entropy loss for this step.
    pub loss: f32,
    /// Learning rate used for this step.
    pub lr: f32,
    /// Global gradient L2 norm before clipping.
    pub grad_norm: f32,
}

/// Metrics for a full epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Epoch number (1-indexed).
    pub epoch: usize,
    /// Mean loss over all steps in this epoch.
    pub mean_loss: f32,
    /// Learning rate at the end of this epoch.
    pub final_lr: f32,
    /// Total gradient steps taken in this epoch.
    pub steps: usize,
    /// Corpus entries processed in this epoch.
    pub entries_processed: usize,
}

/// Full training history — all step and epoch metrics.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Per-step loss/lr/grad_norm history.
    pub step_history: Vec<StepMetrics>,
    /// Per-epoch aggregate metrics.
    pub epoch_history: Vec<EpochMetrics>,
}

// ────────────────────────────────────────────────────────────────────────────
//  SFT Trainer
// ────────────────────────────────────────────────────────────────────────────

/// Supervised Fine-Tuning trainer.
///
/// Wires the full ATLAS training pipeline:
/// - [`LiveDiscoveryCorpus`] → pheromone-weighted data sampling
/// - [`SimpleTokenizer`] → byte-level tokenization
/// - [`TrainableMlp`] → forward pass on [`GradTape`]
/// - [`cross_entropy`] → loss + analytical gradient
/// - [`GradTape::backward_with_grad`] → reverse-mode AD
/// - [`AdamW`] + [`CosineScheduler`] → parameter updates
///
/// # Example
/// ```rust
/// use atlas_corpus::{SftTrainer, SftConfig};
/// let config = SftConfig { vocab_size: 50, hidden_dim: 16, ..Default::default() };
/// let mut trainer = SftTrainer::new(config);
/// let step = trainer.train_step(5, 10).unwrap();
/// assert!(step.loss > 0.0);
/// ```
pub struct SftTrainer {
    /// The trainable 2-layer MLP.
    pub model: TrainableMlp,
    /// AdamW optimizer (owns moment estimates and param copies).
    pub optimizer: AdamW,
    /// Cosine learning-rate scheduler with linear warm-up.
    pub scheduler: CosineScheduler,
    /// Byte-level tokenizer.
    pub tokenizer: SimpleTokenizer,
    /// Training configuration.
    pub config: SftConfig,
    /// Accumulated training metrics.
    pub metrics: TrainingMetrics,
    /// Global step counter (total gradient steps taken).
    pub global_step: u64,
}

impl SftTrainer {
    /// Construct a new SFT trainer from configuration.
    ///
    /// Initialises the MLP with Xavier weights, registers parameters with
    /// AdamW, and configures the cosine LR schedule.
    pub fn new(config: SftConfig) -> Self {
        let model = TrainableMlp::new(config.vocab_size, config.hidden_dim, config.seed);
        let v = config.vocab_size;
        let h = config.hidden_dim;

        let mut optimizer = AdamW::new(AdamWConfig {
            lr: config.lr,
            weight_decay: config.weight_decay,
            ..Default::default()
        });
        optimizer.add_param(ParamState::new("embed", model.embed.clone(), vec![v, h], true));
        optimizer.add_param(ParamState::new("w1",    model.w1.clone(),    vec![h, h], true));
        optimizer.add_param(ParamState::new("w2",    model.w2.clone(),    vec![h, v], true));

        // Estimate total steps for the scheduler
        let estimated_steps = (config.max_epochs * config.batch_size * 20).max(100) as u64;
        let scheduler = CosineScheduler::new(
            config.lr,
            config.lr_min,
            estimated_steps,
            config.warmup_steps,
        );

        let tokenizer = SimpleTokenizer::new(config.vocab_size);

        Self {
            model,
            optimizer,
            scheduler,
            tokenizer,
            config,
            metrics: TrainingMetrics::default(),
            global_step: 0,
        }
    }

    /// Compute loss for a single (input, target) pair **without** updating weights.
    pub fn compute_loss(&self, input_token: u32, target_token: u32) -> Result<f32> {
        let mut tape = GradTape::new();
        let (logits_idx, _, _, _) = self.model.forward_on_tape(&mut tape, input_token)?;
        let logits = tape.tensors[logits_idx].as_slice()?;
        let (loss, _) = cross_entropy(logits, target_token);
        Ok(loss)
    }

    /// Execute one training step: forward → loss → backward → optimizer update.
    ///
    /// Returns metrics for this step.
    pub fn train_step(&mut self, input_token: u32, target_token: u32) -> Result<StepMetrics> {
        self.global_step += 1;

        // Apply cosine LR schedule
        self.scheduler.apply(&mut self.optimizer, self.global_step);

        // ── Forward ──────────────────────────────────────────
        let mut tape = GradTape::new();
        let (logits_idx, embed_idx, w1_idx, w2_idx) =
            self.model.forward_on_tape(&mut tape, input_token)?;

        // ── Loss ─────────────────────────────────────────────
        let logits = tape.tensors[logits_idx].as_slice()?;
        let (loss, d_logits) = cross_entropy(logits, target_token);

        // ── Backward ─────────────────────────────────────────
        let d_logits_t = Tensor::from_vec(d_logits, vec![1, self.config.vocab_size])?;
        tape.backward_with_grad(logits_idx, d_logits_t)?;

        // ── Extract gradients ────────────────────────────────
        let grad_embed = tape.grads[embed_idx]
            .as_ref()
            .ok_or_else(|| AtlasError::Other("no embed gradient".into()))?
            .as_slice()?.to_vec();
        let grad_w1 = tape.grads[w1_idx]
            .as_ref()
            .ok_or_else(|| AtlasError::Other("no w1 gradient".into()))?
            .as_slice()?.to_vec();
        let grad_w2 = tape.grads[w2_idx]
            .as_ref()
            .ok_or_else(|| AtlasError::Other("no w2 gradient".into()))?
            .as_slice()?.to_vec();

        let grads = vec![grad_embed, grad_w1, grad_w2];
        let grad_norm = atlas_optim::global_grad_norm(&grads);

        // ── Optimizer step ───────────────────────────────────
        self.optimizer.step(&grads)?;

        // ── Sync weights back to model ───────────────────────
        self.model.embed = self.optimizer.params[0].param.clone();
        self.model.w1    = self.optimizer.params[1].param.clone();
        self.model.w2    = self.optimizer.params[2].param.clone();

        let sm = StepMetrics {
            step: self.global_step,
            loss,
            lr: self.optimizer.cfg.lr,
            grad_norm,
        };
        self.metrics.step_history.push(sm.clone());
        Ok(sm)
    }

    /// Train on all next-token pairs in a text string.
    ///
    /// Returns per-step metrics for each (input, target) pair.
    pub fn train_on_text(&mut self, text: &str) -> Result<Vec<StepMetrics>> {
        let pairs = self.tokenizer.make_pairs(text);
        let mut steps = Vec::with_capacity(pairs.len());
        for i in 0..pairs.len() {
            let sm = self.train_step(pairs.inputs[i], pairs.targets[i])?;
            steps.push(sm);
        }
        Ok(steps)
    }

    /// Train one epoch over the corpus using pheromone-weighted sampling.
    ///
    /// Samples `config.batch_size` entries, tokenizes each, and runs gradient
    /// steps on all next-token pairs. Provides feedback to the corpus based on
    /// whether loss decreased (positive) or increased (negative).
    pub fn train_epoch(
        &mut self,
        corpus: &mut LiveDiscoveryCorpus,
    ) -> Result<EpochMetrics> {
        let batch = corpus.sample_batch(self.config.batch_size, SampleStrategy::Pheromone);
        let epoch_num = self.metrics.epoch_history.len() + 1;
        let entries_processed = batch.entries.len();
        let mut epoch_losses: Vec<f32> = Vec::new();

        for entry in &batch.entries {
            let text = entry.to_training_text();
            let pairs = self.tokenizer.make_pairs(&text);

            for i in 0..pairs.len() {
                let sm = self.train_step(pairs.inputs[i], pairs.targets[i])?;
                epoch_losses.push(sm.loss);
            }

            // Stigmergic feedback: reinforce entries that yield decreasing loss
            if epoch_losses.len() >= 2 {
                let recent   = epoch_losses[epoch_losses.len() - 1];
                let previous = epoch_losses[epoch_losses.len() - 2];
                if recent < previous {
                    corpus.feedback_positive(entry.id);
                } else {
                    corpus.feedback_negative(entry.id);
                }
            }
        }

        let mean_loss = if epoch_losses.is_empty() {
            0.0
        } else {
            epoch_losses.iter().sum::<f32>() / epoch_losses.len() as f32
        };

        let em = EpochMetrics {
            epoch: epoch_num,
            mean_loss,
            final_lr: self.optimizer.cfg.lr,
            steps: epoch_losses.len(),
            entries_processed,
        };
        self.metrics.epoch_history.push(em.clone());
        Ok(em)
    }

    /// Full training loop: run multiple epochs over the corpus.
    ///
    /// Returns a reference to the accumulated [`TrainingMetrics`].
    pub fn train(
        &mut self,
        corpus: &mut LiveDiscoveryCorpus,
        epochs: usize,
    ) -> Result<&TrainingMetrics> {
        for _ in 0..epochs {
            self.train_epoch(corpus)?;
        }
        Ok(&self.metrics)
    }

    /// Access the accumulated training metrics.
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Run forward + backward for one (input, target) pair.
    /// Returns `(loss, [grad_embed, grad_w1, grad_w2])`.
    /// Does NOT update optimizer state.
    fn forward_backward(
        &self,
        input_token: u32,
        target_token: u32,
    ) -> Result<(f32, Vec<Vec<f32>>)> {
        let mut tape = GradTape::new();
        let (logits_idx, embed_idx, w1_idx, w2_idx) =
            self.model.forward_on_tape(&mut tape, input_token)?;

        let logits = tape.tensors[logits_idx].as_slice()?;
        let (loss, d_logits) = cross_entropy(logits, target_token);

        let d_logits_t = Tensor::from_vec(d_logits, vec![1, self.config.vocab_size])?;
        tape.backward_with_grad(logits_idx, d_logits_t)?;

        let grad_embed = tape.grads[embed_idx]
            .as_ref()
            .ok_or_else(|| AtlasError::Other("no embed gradient".into()))?
            .as_slice()?.to_vec();
        let grad_w1 = tape.grads[w1_idx]
            .as_ref()
            .ok_or_else(|| AtlasError::Other("no w1 gradient".into()))?
            .as_slice()?.to_vec();
        let grad_w2 = tape.grads[w2_idx]
            .as_ref()
            .ok_or_else(|| AtlasError::Other("no w2 gradient".into()))?
            .as_slice()?.to_vec();

        Ok((loss, vec![grad_embed, grad_w1, grad_w2]))
    }

    // ── Checkpointing ─────────────────────────────────────────────────────────

    /// Save model weights to a safetensors binary file.
    ///
    /// Saves `embed`, `w1`, `w2` matrices with their logical shapes.
    pub fn save_safetensors(&self, path: &str) -> Result<()> {
        use atlas_model::SafetensorsFile;
        let embed_shape = vec![self.config.vocab_size, self.config.hidden_dim];
        let w1_shape    = vec![self.config.hidden_dim, self.config.hidden_dim];
        let w2_shape    = vec![self.config.hidden_dim, self.config.vocab_size];
        let bytes = SafetensorsFile::build_f32(&[
            ("embed", &embed_shape, &self.model.embed),
            ("w1",    &w1_shape,    &self.model.w1),
            ("w2",    &w2_shape,    &self.model.w2),
        ]);
        std::fs::write(path, &bytes)
            .map_err(|e| AtlasError::Io(e.to_string()))
    }

    /// Load a trainer from a safetensors checkpoint.
    ///
    /// Creates a new trainer from `config`, then overlays the saved weights.
    /// Any tensor not present in the file is left at its initialized value.
    pub fn load_safetensors(path: &str, config: SftConfig) -> Result<Self> {
        use atlas_model::SafetensorsFile;
        let bytes = std::fs::read(path)
            .map_err(|e| AtlasError::Io(e.to_string()))?;
        let st = SafetensorsFile::from_bytes(bytes)?;
        let mut trainer = Self::new(config);
        if let Ok(embed) = st.get_f32("embed") { trainer.model.embed = embed; }
        if let Ok(w1)    = st.get_f32("w1")    { trainer.model.w1    = w1;    }
        if let Ok(w2)    = st.get_f32("w2")    { trainer.model.w2    = w2;    }
        // Re-sync optimizer param snapshots from loaded weights
        trainer.optimizer.params[0].param = trainer.model.embed.clone();
        trainer.optimizer.params[1].param = trainer.model.w1.clone();
        trainer.optimizer.params[2].param = trainer.model.w2.clone();
        Ok(trainer)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────────────────────

/// LCG pseudo-random: returns a value in approximately [−1, 1).
fn lcg_next(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
}

/// Xavier-uniform initialization: `U(−√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))`.
fn xavier_init(n: usize, fan_in: usize, fan_out: usize, seed: &mut u64) -> Vec<f32> {
    let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
    (0..n).map(|_| lcg_next(seed) * scale).collect()
}

// ────────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GateConfig, LiveDiscoveryCorpus};
    use atlas_astra::Discovery;

    fn make_discovery(source: &str, title: &str, quality: f64) -> Discovery {
        Discovery {
            id: format!("train-test-{}", title.len()),
            source: source.to_string(),
            title: title.to_string(),
            description: String::new(),
            causal_claims: vec![],
            quality_score: quality,
            proof_commitment: 0,
            timestamp: 0,
            tags: vec![],
            provenance: None,
        }
    }

    fn make_test_corpus() -> LiveDiscoveryCorpus {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        c.add_discovery(make_discovery(
            "arxiv",
            "Neural networks can learn causal representations efficiently in modern architectures",
            0.80,
        ));
        c.add_discovery(make_discovery(
            "nasa_power",
            "Global temperature anomaly data strongly correlates with atmospheric CO2 concentration levels",
            0.85,
        ));
        c.add_discovery(make_discovery(
            "who_gho",
            "Vaccination coverage rates inversely correlated with infectious disease incidence worldwide",
            0.90,
        ));
        c
    }

    // ── 1. sft_trainer_constructs ─────────────────────────────────────────

    #[test]
    fn sft_trainer_constructs() {
        let config = SftConfig::default();
        let trainer = SftTrainer::new(config);
        assert!(trainer.model.param_count() > 0);
        assert_eq!(trainer.optimizer.params.len(), 3); // embed, w1, w2
        assert_eq!(trainer.global_step, 0);
        assert!(trainer.metrics.step_history.is_empty());
        assert!(trainer.metrics.epoch_history.is_empty());
        // Check param sizes match
        let v = trainer.config.vocab_size;
        let h = trainer.config.hidden_dim;
        assert_eq!(trainer.model.embed.len(), v * h);
        assert_eq!(trainer.model.w1.len(), h * h);
        assert_eq!(trainer.model.w2.len(), h * v);
        assert_eq!(trainer.model.param_count(), v * h + h * h + h * v);
    }

    // ── 2. cross_entropy_loss_correct ─────────────────────────────────────

    #[test]
    fn cross_entropy_loss_correct() {
        // Uniform logits → loss = ln(vocab_size)
        let logits = vec![0.0f32; 10];
        let (loss, grad) = cross_entropy(&logits, 3);
        let expected_loss = (10.0f32).ln();
        assert!(
            (loss - expected_loss).abs() < 1e-4,
            "uniform logits: loss={loss}, expected={expected_loss}"
        );
        // softmax of uniform = 1/N each
        // grad[target] = 1/N − 1 = −0.9, others = 1/N = 0.1
        assert!((grad[3] - (-0.9)).abs() < 1e-5, "grad[target]={}", grad[3]);
        assert!((grad[0] - 0.1).abs() < 1e-5, "grad[0]={}", grad[0]);

        // Sharp logits → low loss
        let mut sharp = vec![0.0f32; 10];
        sharp[5] = 10.0;
        let (loss2, _) = cross_entropy(&sharp, 5);
        assert!(loss2 < 0.01, "sharp logits: loss should be near 0, got {loss2}");

        // Wrong sharp logits → high loss
        let (loss3, _) = cross_entropy(&sharp, 0);
        assert!(loss3 > 5.0, "wrong target: loss should be high, got {loss3}");
    }

    // ── 3. pheromone_weighted_sampling ─────────────────────────────────────

    #[test]
    fn pheromone_weighted_sampling() {
        let mut c = make_test_corpus();
        let target_id = c.entries()[0].id;
        // Massively boost one entry's pheromone
        for _ in 0..30 {
            c.feedback_positive(target_id);
        }
        // Sample many times, count
        let mut target_count = 0usize;
        let trials = 60;
        for _ in 0..trials {
            let batch = c.sample_batch(1, SampleStrategy::Pheromone);
            if !batch.entries.is_empty() && batch.entries[0].id == target_id {
                target_count += 1;
            }
        }
        // With 3 entries and one having massively higher pheromone,
        // it should appear > 1/3 of the time
        let ratio = target_count as f64 / trials as f64;
        assert!(
            ratio > 0.33,
            "high-pheromone entry should be sampled more: {target_count}/{trials} = {ratio:.2}"
        );
    }

    // ── 4. train_single_step_decreases_loss ───────────────────────────────

    #[test]
    fn train_single_step_decreases_loss() {
        let config = SftConfig {
            vocab_size: 50,
            hidden_dim: 16,
            lr: 0.05,
            weight_decay: 0.0,
            warmup_steps: 0,
            ..Default::default()
        };
        let mut trainer = SftTrainer::new(config);
        let input = 5u32;
        let target = 10u32;

        let loss_before = trainer.compute_loss(input, target).unwrap();
        trainer.train_step(input, target).unwrap();
        let loss_after = trainer.compute_loss(input, target).unwrap();

        assert!(
            loss_after < loss_before,
            "loss should decrease after one step: {loss_before:.4} → {loss_after:.4}"
        );
    }

    // ── 5. train_epoch_processes_all_entries ───────────────────────────────

    #[test]
    fn train_epoch_processes_all_entries() {
        let mut corpus = make_test_corpus();
        let n_entries = corpus.len();
        let config = SftConfig {
            vocab_size: 50,
            hidden_dim: 16,
            batch_size: n_entries + 5, // request more than available
            ..Default::default()
        };
        let mut trainer = SftTrainer::new(config);
        let em = trainer.train_epoch(&mut corpus).unwrap();
        // All entries should be processed (capped at corpus size)
        assert_eq!(em.entries_processed, n_entries);
        // Should have taken at least 1 step per entry
        assert!(em.steps >= n_entries, "steps={} < entries={}", em.steps, n_entries);
    }

    // ── 6. learning_rate_decreases ────────────────────────────────────────

    #[test]
    fn learning_rate_decreases() {
        let config = SftConfig {
            vocab_size: 20,
            hidden_dim: 8,
            warmup_steps: 2,
            ..Default::default()
        };
        let mut trainer = SftTrainer::new(config);
        // Get past warmup
        for i in 0..10u32 {
            trainer.train_step(i % 20, (i + 1) % 20).unwrap();
        }
        let lr_early = trainer.optimizer.cfg.lr;
        for i in 10..40u32 {
            trainer.train_step(i % 20, (i + 1) % 20).unwrap();
        }
        let lr_late = trainer.optimizer.cfg.lr;
        assert!(
            lr_late <= lr_early + 1e-8,
            "LR should decrease: {lr_early:.6} → {lr_late:.6}"
        );
    }

    // ── 7. batch_assembly_correct_shapes ──────────────────────────────────

    #[test]
    fn batch_assembly_correct_shapes() {
        let tok = SimpleTokenizer::new(100);
        let text = "hello world test data";
        let batch = tok.make_pairs(text);

        // Basic invariants
        assert!(!batch.is_empty());
        assert_eq!(batch.inputs.len(), batch.targets.len());

        // Length = text_bytes − 1
        let all_tokens = tok.encode(text);
        assert_eq!(batch.len(), all_tokens.len() - 1);

        // First pair: inputs[0] = tokens[0], targets[0] = tokens[1]
        assert_eq!(batch.inputs[0], all_tokens[0]);
        assert_eq!(batch.targets[0], all_tokens[1]);

        // Last pair: inputs[n-1] = tokens[n-1], targets[n-1] = tokens[n]
        let n = batch.len();
        assert_eq!(batch.inputs[n - 1], all_tokens[n - 1]);
        assert_eq!(batch.targets[n - 1], all_tokens[n]);

        // All token IDs within vocab
        for &t in batch.inputs.iter().chain(batch.targets.iter()) {
            assert!((t as usize) < 100, "token {t} >= vocab_size 100");
        }
    }

    // ── 8. training_metrics_recorded ──────────────────────────────────────

    #[test]
    fn training_metrics_recorded() {
        let config = SftConfig {
            vocab_size: 30,
            hidden_dim: 8,
            ..Default::default()
        };
        let mut trainer = SftTrainer::new(config);
        let n_steps = 5u32;
        for i in 0..n_steps {
            trainer.train_step(i, i + 1).unwrap();
        }
        let metrics = trainer.metrics();
        assert_eq!(metrics.step_history.len(), n_steps as usize);
        for (i, sm) in metrics.step_history.iter().enumerate() {
            assert_eq!(sm.step, i as u64 + 1);
            assert!(sm.loss > 0.0, "step {}: loss should be positive", i);
            assert!(sm.lr > 0.0, "step {}: lr should be positive", i);
            assert!(sm.grad_norm >= 0.0, "step {}: grad_norm should be non-negative", i);
        }
    }

    // ── 9. train_on_text ──────────────────────────────────────────────────

    #[test]
    fn train_on_text_processes_all_pairs() {
        let config = SftConfig {
            vocab_size: 80,
            hidden_dim: 16,
            ..Default::default()
        };
        let mut trainer = SftTrainer::new(config);
        let text = "abcdefgh";
        let steps = trainer.train_on_text(text).unwrap();
        // "abcdefgh" = 8 bytes → 7 pairs
        assert_eq!(steps.len(), 7);
        // All losses should be positive
        for s in &steps {
            assert!(s.loss > 0.0);
        }
    }

    // ── 10. full_training_loop ────────────────────────────────────────────

    #[test]
    fn full_training_loop() {
        let mut corpus = make_test_corpus();
        let config = SftConfig {
            vocab_size: 50,
            hidden_dim: 16,
            batch_size: 3,
            max_epochs: 3,
            lr: 0.01,
            warmup_steps: 2,
            ..Default::default()
        };
        let mut trainer = SftTrainer::new(config);
        let metrics = trainer.train(&mut corpus, 3).unwrap();

        assert_eq!(metrics.epoch_history.len(), 3);
        for em in &metrics.epoch_history {
            assert!(em.mean_loss > 0.0);
            assert!(em.steps > 0);
            assert!(em.entries_processed > 0);
        }
        // Total steps should match step_history length
        let total_steps: usize = metrics.epoch_history.iter().map(|e| e.steps).sum();
        assert_eq!(metrics.step_history.len(), total_steps);
    }

    // ── 11. trainable_mlp_forward_shapes ──────────────────────────────────

    #[test]
    fn trainable_mlp_forward_shapes() {
        let model = TrainableMlp::new(50, 16, 42);
        let mut tape = GradTape::new();
        let (logits_idx, embed_idx, w1_idx, w2_idx) =
            model.forward_on_tape(&mut tape, 7).unwrap();

        // Logits shape: [1, 50]
        let logits = &tape.tensors[logits_idx];
        assert_eq!(logits.shape(), &[1, 50]);

        // Embed shape: [50, 16]
        assert_eq!(tape.tensors[embed_idx].shape(), &[50, 16]);
        // W1 shape: [16, 16]
        assert_eq!(tape.tensors[w1_idx].shape(), &[16, 16]);
        // W2 shape: [16, 50]
        assert_eq!(tape.tensors[w2_idx].shape(), &[16, 50]);
    }

    // ── 12. empty_text_no_crash ───────────────────────────────────────────

    #[test]
    fn empty_text_no_crash() {
        let tok = SimpleTokenizer::new(50);
        let batch = tok.make_pairs("");
        assert!(batch.is_empty());
        let batch2 = tok.make_pairs("x"); // single byte → no pairs
        assert!(batch2.is_empty());
    }
}
