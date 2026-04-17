//! InvasionFitnessScorer — Champagnat (2011) invasion fitness for stigmergic sampling.
//!
//! Replaces raw pheromone softmax in [`SampleStrategy::Stigmergic`] with the
//! principled invasion fitness function from adaptive dynamics:
//!
//!   f(y) = success_weight·success(y)
//!          − cost_weight·cost(y)
//!          − competition_weight · Σᵢ cos_sim(emb(y), emb(xᵢ)) · n̄ᵢ
//!
//! where n̄ᵢ is the equilibrium pheromone weight of resident xᵢ.
//! Positive f(y) → y can invade; negative → y is outcompeted.
//!
//! Reference: Champagnat-Méléard (2011) PTRF §3, eq. (3.2).

/// A single item that can be scored against a resident population.
#[derive(Debug, Clone)]
pub struct FitnessItem {
    /// Fraction of successful outcomes (∈ [0, 1]).
    pub success_rate: f32,
    /// Resource cost (≥ 0; higher = more expensive).
    pub cost: f32,
    /// Semantic embedding vector (any dimension, need not be unit-norm).
    pub embedding: Vec<f32>,
    /// Equilibrium pheromone weight n̄ᵢ (used as resident density proxy).
    pub pheromone_weight: f32,
}

/// Configuration for [`InvasionFitnessScorer`].
#[derive(Debug, Clone)]
pub struct InvasionFitnessConfig {
    /// Weight on success term (default: 1.0).
    pub success_weight: f32,
    /// Weight on cost term (default: 0.5).
    pub cost_weight: f32,
    /// Weight on competition term Σ α_{ij}·n̄ (default: 1.0).
    pub competition_weight: f32,
    /// Minimum cosine similarity below which competition is treated as zero.
    ///
    /// Default: 0.2 — approximately 4σ above the noise floor for random unit vectors
    /// in d=384 embedding space (std ≈ 1/√384 ≈ 0.051). Morphs with cos_sim below
    /// this threshold occupy sufficiently different semantic regions that competition
    /// is negligible. Negative cos_sim → α_{ij} = 0 (no mutualism).
    ///
    /// Lotka-Volterra coexistence requires α_{ij} ≥ 0. Negative competition coefficients
    /// (from raw cos_sim) modelled mutualism, not competition — always wrong here.
    /// Fix (Issue #11): α_{ij} = ReLU(cos_sim − competition_threshold).
    pub competition_threshold: f32,
}

impl Default for InvasionFitnessConfig {
    fn default() -> Self {
        Self {
            success_weight: 1.0,
            cost_weight: 0.5,
            competition_weight: 1.0,
            competition_threshold: 0.2,
        }
    }
}

/// Scores candidates using Champagnat invasion fitness.
///
/// # Example
/// ```rust
/// use atlas_corpus::invasion_fitness::{InvasionFitnessScorer, InvasionFitnessConfig, FitnessItem};
/// let scorer = InvasionFitnessScorer::new(InvasionFitnessConfig::default());
/// let candidate = FitnessItem { success_rate: 0.8, cost: 0.2,
///     embedding: vec![1.0, 0.0], pheromone_weight: 1.0 };
/// let residents = vec![
///     FitnessItem { success_rate: 0.5, cost: 0.3,
///         embedding: vec![0.0, 1.0], pheromone_weight: 0.5 },
/// ];
/// let fitness = scorer.score(&candidate, &residents);
/// // Orthogonal embeddings: cos_sim=0 < threshold=0.2 → zero competition → high fitness
/// assert!(fitness > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct InvasionFitnessScorer {
    config: InvasionFitnessConfig,
}

impl InvasionFitnessScorer {
    /// Create a new scorer with given config.
    pub fn new(config: InvasionFitnessConfig) -> Self { Self { config } }

    /// Create scorer with default config.
    pub fn default_scorer() -> Self { Self::new(InvasionFitnessConfig::default()) }

    /// Score a single candidate against a resident population.
    ///
    /// f(y) = success_weight·success(y)
    ///        − cost_weight·cost(y)
    ///        − competition_weight · Σᵢ α_{ij} · n̄ᵢ
    ///
    /// where α_{ij} = ReLU(cos_sim(emb(y), emb(xᵢ)) − competition_threshold).
    ///
    /// The ReLU threshold (default 0.2) ensures:
    /// - α_{ij} ≥ 0 always (Lotka-Volterra coexistence requirement).
    /// - Anti-correlated embeddings (cos_sim < 0) do NOT boost fitness (no mutualism).
    /// - Near-orthogonal noise-level similarity is suppressed (< 4σ in d=384 space).
    ///
    /// Issue #11 fix: the old formula used raw cos_sim ∈ [−1, 1], allowing negative
    /// α_{ij} which rewarded anti-correlated strategies — physically wrong.
    pub fn score(&self, candidate: &FitnessItem, residents: &[FitnessItem]) -> f32 {
        let success = self.config.success_weight * candidate.success_rate;
        let cost = self.config.cost_weight * candidate.cost;
        let competition: f32 = self.config.competition_weight
            * residents.iter()
                .map(|r| {
                    let sim = cosine_sim(&candidate.embedding, &r.embedding);
                    // α_{ij} = ReLU(cos_sim − τ): clip to zero below threshold.
                    // Guarantees α_{ij} ≥ 0 (Lotka-Volterra: no mutualism).
                    let alpha = (sim - self.config.competition_threshold).max(0.0);
                    alpha * r.pheromone_weight
                })
                .sum::<f32>();
        success - cost - competition
    }

    /// Score every item in `items` using every other item as residents.
    pub fn score_all(&self, items: &[FitnessItem]) -> Vec<f32> {
        (0..items.len())
            .map(|i| {
                let residents: Vec<&FitnessItem> =
                    items.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| x).collect();
                let residents_owned: Vec<FitnessItem> = residents.into_iter().cloned().collect();
                self.score(&items[i], &residents_owned)
            })
            .collect()
    }

    /// Return indices of top-k items by invasion fitness (highest first).
    pub fn top_k_indices(&self, items: &[FitnessItem], k: usize) -> Vec<usize> {
        let scores = self.score_all(items);
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Config accessor.
    pub fn config(&self) -> &InvasionFitnessConfig { &self.config }
}

/// Cosine similarity between two vectors. Returns 0.0 if either is zero-norm.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 { return 0.0; }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(success: f32, cost: f32, emb: Vec<f32>, weight: f32) -> FitnessItem {
        FitnessItem { success_rate: success, cost, embedding: emb, pheromone_weight: weight }
    }

    #[test]
    fn cosine_sim_orthogonal() {
        assert!((cosine_sim(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_identical() {
        assert!((cosine_sim(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_zero_vector() {
        assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn cosine_sim_opposite() {
        assert!((cosine_sim(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn invasion_fitness_positive_when_novel() {
        // candidate is orthogonal to all residents → zero competition → high fitness
        let scorer = InvasionFitnessScorer::default_scorer();
        let candidate = item(0.9, 0.1, vec![1.0, 0.0], 1.0);
        let residents = vec![item(0.5, 0.3, vec![0.0, 1.0], 0.5)];
        let f = scorer.score(&candidate, &residents);
        assert!(f > 0.0, "novel candidate should have positive fitness, got {f}");
    }

    #[test]
    fn invasion_fitness_negative_when_redundant() {
        // candidate is identical to a high-weight resident → max competition
        let scorer = InvasionFitnessScorer::default_scorer();
        let candidate = item(0.5, 0.1, vec![1.0, 0.0], 1.0);
        // resident has same embedding, very high pheromone
        let residents = vec![item(0.5, 0.1, vec![1.0, 0.0], 10.0)];
        let f = scorer.score(&candidate, &residents);
        assert!(f < 0.0, "redundant candidate should have negative fitness, got {f}");
    }

    #[test]
    fn score_all_returns_correct_length() {
        let scorer = InvasionFitnessScorer::default_scorer();
        let items = vec![
            item(0.8, 0.1, vec![1.0, 0.0], 1.0),
            item(0.6, 0.2, vec![0.0, 1.0], 0.8),
            item(0.7, 0.15, vec![0.7, 0.7], 0.6),
        ];
        let scores = scorer.score_all(&items);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn top_k_selects_highest_fitness() {
        let scorer = InvasionFitnessScorer::default_scorer();
        let items = vec![
            item(0.9, 0.05, vec![1.0, 0.0], 0.1),  // high success, low cost, low weight → high fitness
            item(0.1, 0.9, vec![0.0, 1.0], 5.0),   // low success, high cost → low fitness
            item(0.7, 0.2, vec![0.5, 0.5], 0.5),
        ];
        let top1 = scorer.top_k_indices(&items, 1);
        assert_eq!(top1[0], 0, "item 0 should have highest fitness");
    }

    #[test]
    fn empty_residents_uses_only_success_and_cost() {
        let scorer = InvasionFitnessScorer::default_scorer();
        let candidate = item(0.8, 0.2, vec![1.0, 0.0], 1.0);
        let f = scorer.score(&candidate, &[]);
        // expected = 1.0×0.8 − 0.5×0.2 − 0.0 = 0.7
        assert!((f - 0.7).abs() < 1e-6, "expected 0.7, got {f}");
    }

    // ── Issue #11 regression tests — Bug 2: competition coefficients ──────────

    /// Anti-correlated embeddings (cos_sim = −1) must NOT produce a fitness bonus.
    /// Old code: competition = −1.0 × weight → negative → ADDS to fitness (wrong).
    /// Fixed code: alpha = ReLU(−1 − 0.2) = 0 → zero competition.
    #[test]
    fn competition_never_negative_for_anticorrelated_embeddings() {
        let scorer = InvasionFitnessScorer::default_scorer();
        let candidate     = item(0.5, 0.1, vec![-1.0, 0.0], 1.0);
        let resident      = item(0.5, 0.1, vec![ 1.0, 0.0], 2.0); // exactly opposite

        let f_with_opposite = scorer.score(&candidate, &[resident]);
        let f_no_residents  = scorer.score(&candidate, &[]);

        // Anti-correlated resident must never BOOST fitness above the no-competition baseline.
        assert!(f_with_opposite <= f_no_residents + f32::EPSILON,
            "anti-correlated resident boosted fitness: with={f_with_opposite:.6}, \
             no_residents={f_no_residents:.6} — old mutualism bug still present");
    }

    /// Embeddings with cos_sim below the threshold (default 0.2) must produce
    /// exactly zero competition — they occupy different enough semantic regions.
    #[test]
    fn competition_threshold_suppresses_near_orthogonal_noise() {
        let scorer = InvasionFitnessScorer::default_scorer();
        // cos_sim([1,0,0,0], [0.1, 0.995, 0, 0]) ≈ 0.1/sqrt(1) = 0.1 < threshold 0.2
        let candidate = item(0.8, 0.1, vec![1.0, 0.0, 0.0, 0.0], 1.0);
        let resident  = item(0.5, 0.2, vec![0.1, 0.995, 0.0, 0.0], 1.0);

        let sim = cosine_sim(&candidate.embedding, &resident.embedding);
        assert!(sim < 0.2, "test setup: cos_sim={sim:.4} should be below threshold 0.2");

        let f_near_orthogonal = scorer.score(&candidate, &[resident]);
        let f_no_residents    = scorer.score(&candidate, &[]);

        // Below-threshold resident → zero alpha → no competition effect.
        assert!((f_near_orthogonal - f_no_residents).abs() < 1e-5,
            "below-threshold resident should not affect fitness: \
             near_orth={f_near_orthogonal:.6}, no_res={f_no_residents:.6}");
    }
}
