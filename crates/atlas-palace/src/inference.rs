//! Active Inference engine for ATLAS palace agents.
//!
//! Ports the GraphPalace Active Inference system (gp-agents, 5 modules, 1160 lines)
//! into pure Rust with **zero external dependencies**. Implements the full active
//! inference loop: predict → observe → update beliefs → select action.
//!
//! # Components
//!
//! - [`BeliefState`] — Gaussian belief (mean, precision) with Bayesian updates
//! - [`WelfordStats`] / [`GenerativeModel`] — Online statistics for prediction
//! - [`AgentArchetype`] — Explorer / Exploiter / Specialist / Generalist / Critic
//! - [`AnnealingSchedule`] — Linear / Exponential / Cosine temperature decay
//! - [`ActionPolicy`] — Softmax / ε-greedy / Greedy / UCB action selection
//! - [`InferenceAgent`] — Ties everything together; navigates palace via free energy
//!
//! # Wiring
//!
//! This module is self-contained. To activate, add `pub mod inference;` to `lib.rs`.
//! When wired in, the local `cosine_sim_f32` can be replaced with `super::cosine_sim`.

use std::collections::HashMap;

// ── Minimal PRNG ──────────────────────────────────────────────────────────

/// Xorshift64 PRNG — zero-dependency deterministic randomness.
///
/// Used for stochastic action selection (softmax sampling, ε-greedy).
/// Deterministic given the same seed, making tests reproducible.
#[derive(Debug, Clone)]
pub struct Xorshift64(u64);

impl Xorshift64 {
    /// Create with the given seed. Seed 0 is remapped to a default.
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed })
    }

    /// Generate the next u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ── Belief State ──────────────────────────────────────────────────────────

/// Default prior mean for unseen locations.
pub const DEFAULT_PRIOR_MEAN: f64 = 20.0;
/// Default prior precision — low confidence for unseen locations.
pub const DEFAULT_PRIOR_PRECISION: f64 = 0.1;

/// Gaussian belief parameterised by mean and precision (inverse variance).
///
/// Precision-weighted representation makes Bayesian updates a simple addition
/// in information space, matching the GraphPalace belief system (spec §6.3).
#[derive(Debug, Clone)]
pub struct BeliefState {
    /// Belief mean.
    pub mean: f64,
    /// Belief precision (1 / variance). Higher = more confident.
    pub precision: f64,
}

impl Default for BeliefState {
    fn default() -> Self {
        Self {
            mean: DEFAULT_PRIOR_MEAN,
            precision: DEFAULT_PRIOR_PRECISION,
        }
    }
}

impl BeliefState {
    /// New belief with explicit mean and precision.
    pub fn new(mean: f64, precision: f64) -> Self {
        Self { mean, precision }
    }

    /// Precision-weighted Bayesian update.
    ///
    /// Posterior precision = prior + observation precision.
    /// Posterior mean = precision-weighted average of prior and observation.
    pub fn update(&mut self, observation: f64, observation_precision: f64) {
        let prior_prec = self.precision;
        let prior_mean = self.mean;
        self.precision = prior_prec + observation_precision;
        self.mean = (prior_prec * prior_mean + observation_precision * observation) / self.precision;
    }

    /// Merge multiple beliefs via precision-weighted averaging.
    ///
    /// # Panics
    /// Panics if `beliefs` is empty.
    pub fn merge(beliefs: &[&BeliefState]) -> BeliefState {
        assert!(!beliefs.is_empty(), "cannot merge zero beliefs");
        let total_prec: f64 = beliefs.iter().map(|b| b.precision).sum();
        let merged_mean: f64 =
            beliefs.iter().map(|b| b.precision * b.mean).sum::<f64>() / total_prec;
        BeliefState {
            mean: merged_mean,
            precision: total_prec,
        }
    }

    /// Variance (1 / precision).
    pub fn variance(&self) -> f64 {
        1.0 / self.precision
    }
}

// ── Welford Statistics ────────────────────────────────────────────────────

/// Online statistics via Welford's algorithm — O(1) memory, numerically stable.
///
/// Maintains running mean and variance for a stream of observations.
#[derive(Debug, Clone)]
pub struct WelfordStats {
    /// Number of observations.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Sum of squared deviations from mean (M2 in Welford's notation).
    pub m2: f64,
}

impl WelfordStats {
    /// New empty tracker.
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    /// Update with a new observation.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Population variance (0 if count < 2).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Default for WelfordStats {
    fn default() -> Self {
        Self::new()
    }
}

// ── Generative Model ──────────────────────────────────────────────────────

/// Generative model for hierarchical prediction.
///
/// Maintains per-key running statistics and provides predictions (mean, variance).
/// Keys typically represent drawer IDs or room IDs in the palace.
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    /// Per-key Welford statistics.
    pub stats: HashMap<String, WelfordStats>,
}

impl GenerativeModel {
    /// New empty model.
    pub fn new() -> Self {
        Self { stats: HashMap::new() }
    }

    /// Record an observation for a key.
    pub fn observe(&mut self, key: &str, value: f64) {
        self.stats.entry(key.to_string()).or_default().update(value);
    }

    /// Predict (mean, variance) for a key, or `None` if never observed.
    pub fn predict(&self, key: &str) -> Option<(f64, f64)> {
        self.stats.get(key).map(|s| (s.mean, s.variance()))
    }

    /// Number of distinct keys tracked.
    pub fn num_keys(&self) -> usize {
        self.stats.len()
    }
}

impl Default for GenerativeModel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Agent Archetypes ──────────────────────────────────────────────────────

/// Predefined agent archetypes with characteristic exploration/exploitation profiles.
///
/// Each archetype defines a default temperature and action policy, matching
/// the GraphPalace archetype system (spec §6.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentArchetype {
    /// Temperature 1.0 — maximises epistemic value, no goal bias.
    Explorer,
    /// Temperature 0.1 — aggressively follows known rewarding paths.
    Exploiter,
    /// Temperature 0.3 — narrow domain focus, deep expertise.
    Specialist,
    /// Temperature 0.5 — balanced exploration across domains.
    Generalist,
    /// Temperature 0.2 — detects contradictions and inconsistencies.
    Critic,
}

impl AgentArchetype {
    /// Default softmax temperature for this archetype.
    pub fn default_temperature(&self) -> f64 {
        match self {
            Self::Explorer => 1.0,
            Self::Exploiter => 0.1,
            Self::Specialist => 0.3,
            Self::Generalist => 0.5,
            Self::Critic => 0.2,
        }
    }

    /// Default action policy for this archetype.
    pub fn default_policy(&self) -> ActionPolicy {
        match self {
            Self::Explorer => ActionPolicy::Softmax { temperature: 1.0 },
            Self::Exploiter => ActionPolicy::Greedy,
            Self::Specialist => ActionPolicy::Softmax { temperature: 0.3 },
            Self::Generalist => ActionPolicy::EpsilonGreedy { epsilon: 0.15 },
            Self::Critic => ActionPolicy::Softmax { temperature: 0.2 },
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Explorer => "High-temperature explorer: maximises epistemic value",
            Self::Exploiter => "Greedy exploiter: always follows best known path",
            Self::Specialist => "Narrow-focus specialist: deep expertise in one domain",
            Self::Generalist => "Broad-search generalist: balanced exploration",
            Self::Critic => "Contradiction detector: identifies inconsistencies",
        }
    }

    /// All five archetypes.
    pub fn all() -> &'static [AgentArchetype] {
        &[
            Self::Explorer,
            Self::Exploiter,
            Self::Specialist,
            Self::Generalist,
            Self::Critic,
        ]
    }
}

// ── Annealing Schedule ────────────────────────────────────────────────────

/// Temperature annealing schedule for controlling exploration over time.
#[derive(Debug, Clone)]
pub enum AnnealingSchedule {
    /// Linear: `start + progress * (end - start)`.
    Linear {
        /// Starting temperature.
        start: f64,
        /// Ending temperature.
        end: f64,
    },
    /// Exponential: `start * exp(-decay * progress)`.
    Exponential {
        /// Starting temperature.
        start: f64,
        /// Decay rate.
        decay: f64,
    },
    /// Cosine: smooth S-curve from start to end.
    Cosine {
        /// Starting temperature.
        start: f64,
        /// Ending temperature.
        end: f64,
    },
}

impl AnnealingSchedule {
    /// Compute temperature at `progress` ∈ [0, 1].
    pub fn anneal(&self, progress: f64) -> f64 {
        let p = progress.clamp(0.0, 1.0);
        match self {
            Self::Linear { start, end } => start - p * (start - end),
            Self::Exponential { start, decay } => start * (-decay * p).exp(),
            Self::Cosine { start, end } => {
                end + 0.5 * (start - end) * (1.0 + (std::f64::consts::PI * p).cos())
            }
        }
    }
}

// ── Free Energy ───────────────────────────────────────────────────────────

/// Variational Free Energy — measures surprise given current beliefs.
///
/// For a Gaussian belief with mean μ and precision τ:
///
///   VFE = 0.5 · τ · (observation − μ)² + 0.5 · ln(2π / τ)
///
/// Lower VFE = observation matches the belief better (less surprise).
pub fn variational_free_energy(belief: &BeliefState, observation: f64) -> f64 {
    let diff = observation - belief.mean;
    let tau = belief.precision.max(1e-10);
    0.5 * tau * diff * diff + 0.5 * (2.0 * std::f64::consts::PI / tau).ln()
}

/// Expected Free Energy for a navigation candidate.
///
/// EFE = −(epistemic + pragmatic + edge_quality)
///
/// - **Epistemic** (1/precision): uncertain locations are attractive.
/// - **Pragmatic** (goal similarity): goal-aligned locations preferred.
/// - **Edge quality**: 0.5 × exploitation − 0.3 × exploration pheromone.
///
/// Lower (more negative) EFE = better candidate.
pub fn expected_free_energy(
    belief_precision: f64,
    goal_similarity: f64,
    exploitation_pheromone: f64,
    exploration_pheromone: f64,
) -> f64 {
    let epistemic = 1.0 / belief_precision.max(1e-10);
    let pragmatic = goal_similarity.max(0.0);
    let edge_quality = 0.5 * exploitation_pheromone - 0.3 * exploration_pheromone;
    -(epistemic + pragmatic + edge_quality)
}

// ── Action Policy ─────────────────────────────────────────────────────────

/// Action selection policy for navigating the palace.
#[derive(Debug, Clone)]
pub enum ActionPolicy {
    /// Boltzmann softmax: probability ∝ exp(−EFE / temperature).
    Softmax {
        /// Temperature (higher = more exploratory).
        temperature: f64,
    },
    /// Epsilon-greedy: random with probability ε, best otherwise.
    EpsilonGreedy {
        /// Exploration probability ∈ [0, 1].
        epsilon: f64,
    },
    /// Always pick the candidate with lowest EFE.
    Greedy,
    /// Upper Confidence Bound: balances EFE with visit-count bonus.
    Ucb {
        /// Exploration constant (higher = more exploration).
        exploration_constant: f64,
    },
}

/// Compute softmax probabilities from (id, EFE) pairs.
///
/// Uses the log-sum-exp trick for numerical stability.
/// Lower EFE → higher probability (we negate EFE before exponentiating).
pub fn softmax_probabilities(
    candidates: &[(String, f64)],
    temperature: f64,
) -> Vec<(String, f64)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let temp = temperature.max(1e-10);
    let scaled: Vec<f64> = candidates.iter().map(|(_, efe)| -efe / temp).collect();
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scaled.iter().map(|v| (v - max_val).exp()).collect();
    let total: f64 = weights.iter().sum();

    candidates
        .iter()
        .zip(weights.iter())
        .map(|((id, _), w)| (id.clone(), w / total))
        .collect()
}

/// Select an action from candidates using the given policy.
///
/// For [`ActionPolicy::Ucb`], `visit_counts` and `total_visits` are used.
/// For other policies they are ignored.
pub fn select_action(
    candidates: &[(String, f64)],
    policy: &ActionPolicy,
    rng: &mut Xorshift64,
    visit_counts: &HashMap<String, u64>,
    total_visits: u64,
) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }

    match policy {
        ActionPolicy::Greedy => candidates
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id.clone()),

        ActionPolicy::Softmax { temperature } => {
            let probs = softmax_probabilities(candidates, *temperature);
            let u = rng.next_f64();
            let mut cumulative = 0.0;
            for (id, p) in &probs {
                cumulative += p;
                if u <= cumulative {
                    return Some(id.clone());
                }
            }
            probs.last().map(|(id, _)| id.clone())
        }

        ActionPolicy::EpsilonGreedy { epsilon } => {
            if rng.next_f64() < *epsilon {
                let idx = (rng.next_u64() as usize) % candidates.len();
                Some(candidates[idx].0.clone())
            } else {
                candidates
                    .iter()
                    .min_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(id, _)| id.clone())
            }
        }

        ActionPolicy::Ucb { exploration_constant } => {
            let ln_total = if total_visits > 0 {
                (total_visits as f64).ln()
            } else {
                1.0
            };
            candidates
                .iter()
                .min_by(|a, b| {
                    let n_a = *visit_counts.get(&a.0).unwrap_or(&0) as f64 + 1.0;
                    let n_b = *visit_counts.get(&b.0).unwrap_or(&0) as f64 + 1.0;
                    let ucb_a = a.1 - exploration_constant * (ln_total / n_a).sqrt();
                    let ucb_b = b.1 - exploration_constant * (ln_total / n_b).sqrt();
                    ucb_a.partial_cmp(&ucb_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| id.clone())
        }
    }
}

// ── Navigation Candidate ──────────────────────────────────────────────────

/// A candidate location for agent navigation.
///
/// Represents a drawer or room the agent might visit next.
#[derive(Debug, Clone)]
pub struct NavigationCandidate {
    /// Drawer or room ID.
    pub id: String,
    /// Embedding vector (for cosine similarity to agent's goal).
    pub embedding: Vec<f32>,
    /// Exploitation pheromone strength on the path to this location.
    pub exploitation_pheromone: f64,
    /// Exploration pheromone strength (high = already explored).
    pub exploration_pheromone: f64,
}

// ── Inference Diary Entry ─────────────────────────────────────────────────

/// Diary entry recording an inference agent's activity.
#[derive(Debug, Clone)]
pub struct InferenceDiaryEntry {
    /// Step number when the entry was written.
    pub step: u64,
    /// Entry text.
    pub text: String,
}

// ── Inference Agent ───────────────────────────────────────────────────────

/// Active Inference agent that navigates the palace by minimising free energy.
///
/// Combines beliefs, a generative model, and an action policy to implement
/// the full active inference loop. Links to a palace `Agent` via `agent_id`.
///
/// # Usage
///
/// ```no_run
/// # use atlas_palace::inference::*;
/// let mut agent = InferenceAgent::new("agent-1", "Scout", AgentArchetype::Explorer, 42);
/// // agent.set_goal_embedding(goal_vec);
///
/// // Evaluate candidates and navigate
/// // let selection = agent.select_navigation(&candidates);
///
/// // Observe the result and update beliefs
/// agent.observe_belief("drawer-17", 0.85, 2.0);
/// agent.observe_model("drawer-17", 0.85);
/// ```
#[derive(Debug, Clone)]
pub struct InferenceAgent {
    /// ID linking to the palace Agent record.
    pub agent_id: String,
    /// Agent name.
    pub name: String,
    /// Archetype (defines default behaviour profile).
    pub archetype: AgentArchetype,
    /// Per-location Gaussian beliefs (keyed by drawer/room ID).
    pub beliefs: HashMap<String, BeliefState>,
    /// Generative model for prediction.
    pub model: GenerativeModel,
    /// Goal embedding — direction in semantic space the agent is drawn to.
    pub goal_embedding: Vec<f32>,
    /// Preferred room IDs (pragmatic bias).
    pub preferred_rooms: Vec<String>,
    /// Current action policy.
    pub policy: ActionPolicy,
    /// Optional annealing schedule for temperature decay.
    pub annealing: Option<AnnealingSchedule>,
    /// Number of inference steps taken.
    pub step_count: u64,
    /// Per-location visit counts (for UCB policy).
    pub visit_counts: HashMap<String, u64>,
    /// Inference diary.
    diary: Vec<InferenceDiaryEntry>,
    /// Internal PRNG.
    rng: Xorshift64,
}

impl InferenceAgent {
    /// Create a new inference agent with archetype defaults.
    pub fn new(agent_id: &str, name: &str, archetype: AgentArchetype, seed: u64) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            name: name.to_string(),
            policy: archetype.default_policy(),
            archetype,
            beliefs: HashMap::new(),
            model: GenerativeModel::new(),
            goal_embedding: Vec::new(),
            preferred_rooms: Vec::new(),
            annealing: None,
            step_count: 0,
            visit_counts: HashMap::new(),
            diary: Vec::new(),
            rng: Xorshift64::new(seed),
        }
    }

    /// Set the agent's goal embedding.
    pub fn set_goal_embedding(&mut self, embedding: Vec<f32>) {
        self.goal_embedding = embedding;
    }

    /// Set preferred room IDs.
    pub fn set_preferred_rooms(&mut self, rooms: Vec<String>) {
        self.preferred_rooms = rooms;
    }

    /// Set a custom action policy (overrides archetype default).
    pub fn set_policy(&mut self, policy: ActionPolicy) {
        self.policy = policy;
    }

    /// Set an annealing schedule for automatic temperature decay.
    pub fn set_annealing(&mut self, schedule: AnnealingSchedule) {
        self.annealing = Some(schedule);
    }

    /// Get belief for a location, or the default prior if unseen.
    pub fn get_belief(&self, location_id: &str) -> BeliefState {
        self.beliefs
            .get(location_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Update belief for a location after observing a value with given precision.
    pub fn observe_belief(&mut self, location_id: &str, observation: f64, precision: f64) {
        self.beliefs
            .entry(location_id.to_string())
            .or_default()
            .update(observation, precision);
    }

    /// Record an observation in the generative model.
    pub fn observe_model(&mut self, key: &str, value: f64) {
        self.model.observe(key, value);
    }

    /// Predict (mean, variance) for a key from the generative model.
    pub fn predict(&self, key: &str) -> Option<(f64, f64)> {
        self.model.predict(key)
    }

    /// Compute the current effective temperature, accounting for annealing.
    pub fn effective_temperature(&self, total_steps: u64) -> f64 {
        if let Some(ref schedule) = self.annealing {
            let progress = if total_steps == 0 {
                0.0
            } else {
                (self.step_count as f64 / total_steps as f64).clamp(0.0, 1.0)
            };
            schedule.anneal(progress)
        } else {
            self.archetype.default_temperature()
        }
    }

    /// Evaluate EFE for a single navigation candidate.
    pub fn evaluate_candidate(&self, candidate: &NavigationCandidate) -> f64 {
        let belief = self.get_belief(&candidate.id);
        let similarity = cosine_sim_f32(&self.goal_embedding, &candidate.embedding) as f64;
        expected_free_energy(
            belief.precision,
            similarity,
            candidate.exploitation_pheromone,
            candidate.exploration_pheromone,
        )
    }

    /// Select the best navigation target from candidates.
    ///
    /// Returns the ID of the selected candidate, or `None` if candidates is empty.
    /// Increments step count and updates visit counts.
    pub fn select_navigation(&mut self, candidates: &[NavigationCandidate]) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }

        // Compute EFE for each candidate.
        let efe_pairs: Vec<(String, f64)> = candidates
            .iter()
            .map(|c| (c.id.clone(), self.evaluate_candidate(c)))
            .collect();

        // Apply annealing to softmax policies.
        let policy = if let Some(ref schedule) = self.annealing {
            let progress = (self.step_count as f64 / 100.0).clamp(0.0, 1.0);
            let temp = schedule.anneal(progress);
            match &self.policy {
                ActionPolicy::Softmax { .. } => ActionPolicy::Softmax { temperature: temp },
                other => other.clone(),
            }
        } else {
            self.policy.clone()
        };

        let selected = select_action(
            &efe_pairs,
            &policy,
            &mut self.rng,
            &self.visit_counts,
            self.step_count,
        );

        // Update state.
        self.step_count += 1;
        if let Some(ref id) = selected {
            *self.visit_counts.entry(id.clone()).or_insert(0) += 1;
        }

        selected
    }

    /// Write a diary entry.
    pub fn write_diary(&mut self, text: &str) {
        self.diary.push(InferenceDiaryEntry {
            step: self.step_count,
            text: text.to_string(),
        });
    }

    /// Read all diary entries.
    pub fn read_diary(&self) -> &[InferenceDiaryEntry] {
        &self.diary
    }

    /// Number of inference steps taken.
    pub fn steps(&self) -> u64 {
        self.step_count
    }

    /// Number of distinct locations visited.
    pub fn locations_visited(&self) -> usize {
        self.visit_counts.len()
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────

/// Cosine similarity for f32 vectors (local copy; use `super::cosine_sim` when wired in).
fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────

    fn make_candidate(id: &str, embedding: Vec<f32>, exploit: f64, explore: f64) -> NavigationCandidate {
        NavigationCandidate {
            id: id.to_string(),
            embedding,
            exploitation_pheromone: exploit,
            exploration_pheromone: explore,
        }
    }

    fn zero_embedding(dim: usize) -> Vec<f32> {
        vec![0.0; dim]
    }

    fn unit_embedding(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        if axis < dim {
            v[axis] = 1.0;
        }
        v
    }

    // ── Required tests ───────────────────────────────────────────────

    #[test]
    fn belief_update_shifts_probability() {
        let mut belief = BeliefState::default(); // mean=20, precision=0.1
        let old_mean = belief.mean;

        // Strong observation at 50
        belief.update(50.0, 2.0);

        // Mean should shift toward 50
        assert!(belief.mean > old_mean, "mean should shift toward observation");
        assert!(belief.mean < 50.0, "mean should not overshoot observation");

        // Precision should increase
        assert!(
            (belief.precision - 2.1).abs() < 1e-10,
            "precision should be prior + obs: {}",
            belief.precision
        );

        // Exact: (0.1*20 + 2.0*50) / 2.1
        let expected = (0.1 * 20.0 + 2.0 * 50.0) / 2.1;
        assert!(
            (belief.mean - expected).abs() < 1e-10,
            "mean should be precision-weighted average: got {}, expected {}",
            belief.mean,
            expected
        );
    }

    #[test]
    fn free_energy_decreases_on_correct_prediction() {
        // Belief: mean=30, precision=2.0
        let belief = BeliefState::new(30.0, 2.0);

        // Correct prediction: observation matches belief mean
        let vfe_correct = variational_free_energy(&belief, 30.0);

        // Incorrect prediction: observation is far from mean
        let vfe_wrong = variational_free_energy(&belief, 50.0);

        // VFE should be lower (less surprise) for correct prediction
        assert!(
            vfe_correct < vfe_wrong,
            "correct prediction should have lower VFE: correct={}, wrong={}",
            vfe_correct,
            vfe_wrong
        );

        // Even a slightly off prediction should have lower VFE than a very wrong one
        let vfe_close = variational_free_energy(&belief, 32.0);
        assert!(
            vfe_close < vfe_wrong,
            "close prediction should have lower VFE: close={}, wrong={}",
            vfe_close,
            vfe_wrong
        );
    }

    #[test]
    fn action_selection_softmax_explores() {
        // Three candidates with different EFE values
        let candidates = vec![
            ("a".to_string(), -3.0), // best
            ("b".to_string(), -1.0),
            ("c".to_string(), 0.0), // worst
        ];

        // With high temperature, all candidates should have non-trivial probability
        let probs = softmax_probabilities(&candidates, 10.0);

        for (id, p) in &probs {
            assert!(
                *p > 0.1,
                "at high temperature, candidate {id} should have substantial probability: {p}"
            );
        }

        // Probabilities should sum to 1
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "probabilities should sum to 1: {}",
            sum
        );

        // Even with moderate temperature, best should still have highest prob
        let probs_moderate = softmax_probabilities(&candidates, 1.0);
        let p_a = probs_moderate.iter().find(|(id, _)| id == "a").unwrap().1;
        let p_c = probs_moderate.iter().find(|(id, _)| id == "c").unwrap().1;
        assert!(p_a > p_c, "best candidate should have higher prob: a={p_a}, c={p_c}");
    }

    #[test]
    fn action_selection_greedy_exploits() {
        let candidates = vec![
            ("suboptimal".to_string(), -1.0),
            ("best".to_string(), -5.0), // lowest EFE = best
            ("worst".to_string(), 2.0),
        ];

        let mut rng = Xorshift64::new(42);
        let empty_visits = HashMap::new();

        // Greedy should always pick the lowest EFE
        for _ in 0..20 {
            let selected = select_action(
                &candidates,
                &ActionPolicy::Greedy,
                &mut rng,
                &empty_visits,
                0,
            );
            assert_eq!(
                selected,
                Some("best".to_string()),
                "greedy should always pick lowest EFE"
            );
        }
    }

    #[test]
    fn agent_navigates_toward_low_free_energy() {
        // Agent with a goal embedding pointing in direction [1, 0, 0, ...]
        let mut agent = InferenceAgent::new("nav-1", "Navigator", AgentArchetype::Exploiter, 42);
        agent.set_goal_embedding(unit_embedding(8, 0));
        agent.set_policy(ActionPolicy::Greedy);

        // Candidate A: aligned with goal, has exploitation pheromone
        let aligned = make_candidate("aligned", unit_embedding(8, 0), 1.0, 0.0);

        // Candidate B: orthogonal to goal, no pheromone
        let orthogonal = make_candidate("orthogonal", unit_embedding(8, 3), 0.0, 0.0);

        // Candidate C: anti-aligned, has exploration pheromone (discouraging)
        let mut anti_emb = zero_embedding(8);
        anti_emb[0] = -1.0;
        let anti = make_candidate("anti", anti_emb, 0.0, 2.0);

        let candidates = vec![aligned, orthogonal, anti];

        // Agent should consistently pick the aligned candidate
        let selected = agent.select_navigation(&candidates);
        assert_eq!(
            selected,
            Some("aligned".to_string()),
            "agent should navigate toward low free energy (goal-aligned + exploited)"
        );

        // Verify EFE ordering
        let efe_aligned = agent.evaluate_candidate(&candidates[0]);
        let efe_orthogonal = agent.evaluate_candidate(&candidates[1]);
        let efe_anti = agent.evaluate_candidate(&candidates[2]);

        assert!(
            efe_aligned < efe_orthogonal,
            "aligned EFE ({efe_aligned}) should be less than orthogonal ({efe_orthogonal})"
        );
        assert!(
            efe_orthogonal < efe_anti,
            "orthogonal EFE ({efe_orthogonal}) should be less than anti-aligned ({efe_anti})"
        );
    }

    #[test]
    fn generative_model_predicts_observation() {
        let mut model = GenerativeModel::new();

        // No data yet
        assert!(model.predict("temperature").is_none());

        // Feed observations
        model.observe("temperature", 20.0);
        model.observe("temperature", 22.0);
        model.observe("temperature", 21.0);

        let (mean, variance) = model.predict("temperature").unwrap();

        // Mean should be 21
        assert!(
            (mean - 21.0).abs() < 1e-10,
            "predicted mean should be 21: {}",
            mean
        );

        // Population variance: ((20-21)² + (22-21)² + (21-21)²) / 3 = 2/3
        assert!(
            (variance - 2.0 / 3.0).abs() < 1e-10,
            "predicted variance should be 2/3: {}",
            variance
        );

        // Multiple keys
        model.observe("humidity", 60.0);
        model.observe("humidity", 65.0);
        assert_eq!(model.num_keys(), 2);

        let (h_mean, _) = model.predict("humidity").unwrap();
        assert!((h_mean - 62.5).abs() < 1e-10);
    }

    #[test]
    fn agent_archetypes_have_different_policies() {
        let all = AgentArchetype::all();

        // All archetypes should have unique temperatures
        let temps: Vec<f64> = all.iter().map(|a| a.default_temperature()).collect();
        for i in 0..temps.len() {
            for j in (i + 1)..temps.len() {
                assert!(
                    (temps[i] - temps[j]).abs() > 1e-10,
                    "archetypes {:?} and {:?} should have different temperatures: {} vs {}",
                    all[i],
                    all[j],
                    temps[i],
                    temps[j]
                );
            }
        }

        // Explorer has highest temperature
        let explorer_temp = AgentArchetype::Explorer.default_temperature();
        for a in all {
            assert!(a.default_temperature() <= explorer_temp);
        }

        // Exploiter has lowest temperature
        let exploiter_temp = AgentArchetype::Exploiter.default_temperature();
        for a in all {
            assert!(a.default_temperature() >= exploiter_temp);
        }

        // Exploiter should use Greedy policy
        match AgentArchetype::Exploiter.default_policy() {
            ActionPolicy::Greedy => {} // expected
            other => panic!("Exploiter should use Greedy, got {:?}", other),
        }

        // Explorer should use Softmax with temperature 1.0
        match AgentArchetype::Explorer.default_policy() {
            ActionPolicy::Softmax { temperature } => {
                assert!((temperature - 1.0).abs() < 1e-10);
            }
            other => panic!("Explorer should use Softmax, got {:?}", other),
        }

        // Generalist should use EpsilonGreedy
        match AgentArchetype::Generalist.default_policy() {
            ActionPolicy::EpsilonGreedy { epsilon } => {
                assert!(epsilon > 0.0 && epsilon < 1.0);
            }
            other => panic!("Generalist should use EpsilonGreedy, got {:?}", other),
        }

        // All should have non-empty descriptions
        for a in all {
            assert!(!a.description().is_empty());
        }
    }

    #[test]
    fn inference_agent_diary_integration() {
        let mut agent = InferenceAgent::new("diary-1", "Diarist", AgentArchetype::Generalist, 99);

        // Initially empty
        assert!(agent.read_diary().is_empty());

        // Write some entries
        agent.write_diary("Observed high pheromone trail in wing-research");
        agent.write_diary("Contradiction detected between drawer-17 and drawer-42");

        // Navigate to increment step count
        let c = make_candidate("d1", vec![1.0, 0.0], 0.5, 0.1);
        agent.select_navigation(&[c]);

        agent.write_diary("Selected drawer-d1 for investigation");

        // Verify diary
        let diary = agent.read_diary();
        assert_eq!(diary.len(), 3);

        // First two entries at step 0 (before navigation)
        assert_eq!(diary[0].step, 0);
        assert_eq!(diary[1].step, 0);
        assert!(diary[0].text.contains("pheromone"));
        assert!(diary[1].text.contains("Contradiction"));

        // Third entry at step 1 (after one navigation)
        assert_eq!(diary[2].step, 1);
        assert!(diary[2].text.contains("drawer-d1"));

        // Verify step count
        assert_eq!(agent.steps(), 1);
    }

    // ── Additional tests for thorough coverage ───────────────────────

    #[test]
    fn belief_merge_precision_weighted() {
        let a = BeliefState::new(10.0, 2.0);
        let b = BeliefState::new(20.0, 3.0);
        let merged = BeliefState::merge(&[&a, &b]);

        assert!((merged.precision - 5.0).abs() < 1e-10);
        // (2*10 + 3*20) / 5 = 80/5 = 16
        assert!((merged.mean - 16.0).abs() < 1e-10);
    }

    #[test]
    fn welford_known_sequence() {
        let mut w = WelfordStats::new();
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.update(v);
        }
        assert_eq!(w.count, 8);
        assert!((w.mean - 5.0).abs() < 1e-10);
        assert!((w.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn annealing_linear_endpoints() {
        let s = AnnealingSchedule::Linear { start: 1.0, end: 0.1 };
        assert!((s.anneal(0.0) - 1.0).abs() < 1e-10);
        assert!((s.anneal(1.0) - 0.1).abs() < 1e-10);
        assert!((s.anneal(0.5) - 0.55).abs() < 1e-10);
    }

    #[test]
    fn annealing_exponential_decreases() {
        let s = AnnealingSchedule::Exponential { start: 1.0, decay: 3.0 };
        let t0 = s.anneal(0.0);
        let t_mid = s.anneal(0.5);
        let t_end = s.anneal(1.0);
        assert!(t0 > t_mid && t_mid > t_end);
    }

    #[test]
    fn annealing_cosine_smooth() {
        let s = AnnealingSchedule::Cosine { start: 1.0, end: 0.1 };
        assert!((s.anneal(0.0) - 1.0).abs() < 1e-10);
        assert!((s.anneal(1.0) - 0.1).abs() < 1e-10);
        // Monotone decreasing
        let mut prev = s.anneal(0.0);
        for i in 1..=100 {
            let curr = s.anneal(i as f64 / 100.0);
            assert!(curr <= prev + 1e-10);
            prev = curr;
        }
    }

    #[test]
    fn efe_exploitation_pheromone_lowers_efe() {
        let efe_plain = expected_free_energy(1.0, 0.0, 0.0, 0.0);
        let efe_exploit = expected_free_energy(1.0, 0.0, 2.0, 0.0);
        assert!(efe_exploit < efe_plain);
    }

    #[test]
    fn efe_exploration_pheromone_raises_efe() {
        let efe_plain = expected_free_energy(1.0, 0.0, 0.0, 0.0);
        let efe_explored = expected_free_energy(1.0, 0.0, 0.0, 2.0);
        assert!(efe_explored > efe_plain);
    }

    #[test]
    fn ucb_favours_unvisited() {
        let candidates = vec![
            ("visited".to_string(), -2.0),
            ("fresh".to_string(), -1.5), // slightly worse EFE
        ];
        let mut visits = HashMap::new();
        visits.insert("visited".to_string(), 50); // heavily visited
        // "fresh" has 0 visits → UCB bonus

        let mut rng = Xorshift64::new(1);
        let selected = select_action(
            &candidates,
            &ActionPolicy::Ucb { exploration_constant: 2.0 },
            &mut rng,
            &visits,
            51,
        );
        assert_eq!(
            selected,
            Some("fresh".to_string()),
            "UCB should favour unvisited candidate despite slightly worse EFE"
        );
    }

    #[test]
    fn xorshift_produces_distinct_values() {
        let mut rng = Xorshift64::new(42);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let v = rng.next_u64();
            assert!(seen.insert(v), "xorshift should produce distinct values");
        }
    }

    #[test]
    fn xorshift_f64_in_unit_interval() {
        let mut rng = Xorshift64::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "f64 should be in [0,1): {v}");
        }
    }

    #[test]
    fn cosine_sim_f32_identical_is_one() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert!((cosine_sim_f32(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_f32_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_sim_f32(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn agent_visit_counts_tracked() {
        let mut agent = InferenceAgent::new("v-1", "Visitor", AgentArchetype::Exploiter, 7);
        agent.set_policy(ActionPolicy::Greedy);
        agent.set_goal_embedding(unit_embedding(4, 0));

        let c1 = make_candidate("best", unit_embedding(4, 0), 1.0, 0.0);
        let c2 = make_candidate("other", unit_embedding(4, 2), 0.0, 0.0);

        // Greedy agent should always pick "best"
        for _ in 0..5 {
            agent.select_navigation(&[c1.clone(), c2.clone()]);
        }
        assert_eq!(agent.visit_counts.get("best"), Some(&5));
        assert_eq!(agent.visit_counts.get("other"), None);
        assert_eq!(agent.steps(), 5);
        assert_eq!(agent.locations_visited(), 1);
    }
}
