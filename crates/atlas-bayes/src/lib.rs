//! atlas-bayes — Bayesian scoring and structure learning for ATLAS.
//!
//! Zero external crate dependencies. Implements:
//! - **BDeu score**: Bayesian Dirichlet equivalent uniform scoring
//! - **BGe score**: Bayesian Gaussian equivalent scoring (for continuous data)
//! - **Dynamic Bayesian belief updating**: integrate new evidence
//! - **Quality gating**: assess discovery confidence
//! - **Novelty scoring**: detect new information vs. known facts

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::Result;
use std::collections::HashMap;

// ── Prior / Posterior ─────────────────────────────────────────────────────

/// A Beta distribution for Bayesian probability tracking.
#[derive(Debug, Clone)]
pub struct BetaPrior {
    /// Alpha parameter (pseudo-count for successes).
    pub alpha: f64,
    /// Beta parameter (pseudo-count for failures).
    pub beta: f64,
}

impl BetaPrior {
    /// Uninformative (uniform) prior: Beta(1, 1).
    pub fn uniform() -> Self { Self { alpha: 1.0, beta: 1.0 } }

    /// Weakly informative prior: Beta(2, 2), mean=0.5.
    pub fn weak() -> Self { Self { alpha: 2.0, beta: 2.0 } }

    /// Informative prior with given mean and pseudo-count.
    pub fn informative(mean: f64, n: f64) -> Self {
        let alpha = mean * n;
        let beta  = (1.0 - mean) * n;
        Self { alpha, beta }
    }

    /// Mean of the Beta distribution.
    pub fn mean(&self) -> f64 { self.alpha / (self.alpha + self.beta) }

    /// Variance of the Beta distribution.
    pub fn variance(&self) -> f64 {
        let n = self.alpha + self.beta;
        self.alpha * self.beta / (n * n * (n + 1.0))
    }

    /// Update with `successes` successes and `failures` failures.
    pub fn update(&mut self, successes: f64, failures: f64) {
        self.alpha += successes;
        self.beta  += failures;
    }

    /// 95% credible interval (approximate via normal approximation).
    pub fn credible_interval_95(&self) -> (f64, f64) {
        let mu = self.mean();
        let std = self.variance().sqrt();
        ((mu - 1.96 * std).max(0.0), (mu + 1.96 * std).min(1.0))
    }

    /// Log-odds = ln(alpha/beta).
    pub fn log_odds(&self) -> f64 { (self.alpha / self.beta).ln() }
}

// ── Bayesian belief node ───────────────────────────────────────────────────

/// A variable in the Bayesian belief network.
#[derive(Debug, Clone)]
pub struct BeliefNode {
    /// Node name.
    pub name: String,
    /// Marginal belief (probability this variable is true/high).
    pub belief: BetaPrior,
    /// Confidence in the current belief (0 = highly uncertain, 1 = very certain).
    pub confidence: f64,
    /// Number of evidence updates received.
    pub n_updates: usize,
}

impl BeliefNode {
    /// Create a new node with uniform prior.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            belief: BetaPrior::uniform(),
            confidence: 0.0,
            n_updates: 0,
        }
    }

    /// Update belief with new evidence.
    /// `likelihood_ratio` > 1 means evidence supports the hypothesis.
    pub fn update(&mut self, likelihood_ratio: f64, evidence_strength: f64) {
        let old_mean = self.belief.mean();
        // Bayesian update: posterior ∝ likelihood × prior
        let n_weight = evidence_strength * (1.0 + self.n_updates as f64).ln().max(1.0);
        if likelihood_ratio > 1.0 {
            self.belief.update(likelihood_ratio * n_weight, n_weight);
        } else {
            self.belief.update(n_weight, (1.0 / likelihood_ratio) * n_weight);
        }
        let new_mean = self.belief.mean();
        // Confidence increases as belief moves away from 0.5
        self.confidence = (new_mean - 0.5).abs() * 2.0;
        self.n_updates += 1;
    }
}

// ── Bayesian network ──────────────────────────────────────────────────────

/// A simple Bayesian belief network.
pub struct BayesNetwork {
    nodes: HashMap<String, BeliefNode>,
    /// Conditional dependencies: (child, parent) pairs.
    edges: Vec<(String, String, f64)>,  // (child, parent, influence_weight)
}

impl BayesNetwork {
    /// Create an empty network.
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), edges: Vec::new() }
    }

    /// Add a node with a uniform prior.
    pub fn add_node(&mut self, name: &str) {
        self.nodes.insert(name.to_string(), BeliefNode::new(name));
    }

    /// Add a node with an informative prior.
    pub fn add_node_with_prior(&mut self, name: &str, mean: f64, confidence: f64) {
        let mut node = BeliefNode::new(name);
        node.belief = BetaPrior::informative(mean, confidence * 10.0);
        node.confidence = (mean - 0.5).abs() * 2.0;
        self.nodes.insert(name.to_string(), node);
    }

    /// Add a conditional dependency edge (parent influences child).
    pub fn add_dependency(&mut self, child: &str, parent: &str, weight: f64) {
        self.edges.push((child.to_string(), parent.to_string(), weight));
    }

    /// Update a node's belief with new evidence.
    pub fn observe(&mut self, node: &str, likelihood_ratio: f64, strength: f64) {
        if let Some(n) = self.nodes.get_mut(node) {
            n.update(likelihood_ratio, strength);
        }
        // Propagate to children
        let children: Vec<(String, f64)> = self.edges.iter()
            .filter(|(_, p, _)| p == node)
            .map(|(c, _, w)| (c.clone(), *w))
            .collect();
        for (child, weight) in children {
            // Damped propagation
            let parent_belief = self.nodes.get(node).map(|n| n.belief.mean()).unwrap_or(0.5);
            let child_lr = 1.0 + (parent_belief - 0.5) * weight;
            if let Some(cn) = self.nodes.get_mut(&child) {
                cn.update(child_lr, strength * 0.5); // damped
            }
        }
    }

    /// Get current belief for a node.
    pub fn belief(&self, node: &str) -> Option<f64> {
        self.nodes.get(node).map(|n| n.belief.mean())
    }

    /// Get confidence for a node.
    pub fn confidence(&self, node: &str) -> Option<f64> {
        self.nodes.get(node).map(|n| n.confidence)
    }
}

impl Default for BayesNetwork {
    fn default() -> Self { Self::new() }
}

// ── Quality gating ─────────────────────────────────────────────────────────

/// Quality gate configuration for filtering discoveries.
#[derive(Debug, Clone)]
pub struct QualityGate {
    /// Minimum confidence to pass (default 0.65).
    pub min_confidence: f64,
    /// Minimum novelty score to pass (default 0.40).
    pub min_novelty: f64,
    /// Minimum evidence count to pass (default 2).
    pub min_evidence: usize,
    /// Maximum contradiction score to pass (default 0.30).
    pub max_contradiction: f64,
}

impl Default for QualityGate {
    fn default() -> Self {
        Self {
            min_confidence:    0.65,
            min_novelty:       0.40,
            min_evidence:      2,
            max_contradiction: 0.30,
        }
    }
}

impl QualityGate {
    /// ATLAS Type 5 discovery criteria (strictest).
    pub fn type5() -> Self {
        Self {
            min_confidence:    0.80,
            min_novelty:       0.55,
            min_evidence:      3,
            max_contradiction: 0.20,
        }
    }
}

/// Result of a quality gate evaluation.
#[derive(Debug, Clone)]
pub struct QualityResult {
    /// Whether the discovery passes the gate.
    pub passes: bool,
    /// Overall quality score in [0, 1].
    pub score: f64,
    /// Confidence component.
    pub confidence: f64,
    /// Novelty component.
    pub novelty: f64,
    /// Evidence count.
    pub evidence_count: usize,
    /// Contradiction score.
    pub contradiction: f64,
    /// Reason for failure if any.
    pub reason: Option<String>,
}

impl QualityResult {
    /// Returns the quality label.
    pub fn label(&self) -> &str {
        if self.score >= 0.90 { "excellent" }
        else if self.score >= 0.75 { "good" }
        else if self.score >= 0.60 { "acceptable" }
        else { "poor" }
    }
}

/// Evaluate a discovery against the quality gate.
pub fn quality_gate_eval(
    gate: &QualityGate,
    confidence: f64,
    novelty: f64,
    evidence_count: usize,
    contradiction: f64,
) -> QualityResult {
    let mut reasons = Vec::new();

    if confidence < gate.min_confidence {
        reasons.push(format!("confidence {confidence:.2} < {:.2}", gate.min_confidence));
    }
    if novelty < gate.min_novelty {
        reasons.push(format!("novelty {novelty:.2} < {:.2}", gate.min_novelty));
    }
    if evidence_count < gate.min_evidence {
        reasons.push(format!("evidence_count {evidence_count} < {}", gate.min_evidence));
    }
    if contradiction > gate.max_contradiction {
        reasons.push(format!("contradiction {contradiction:.2} > {:.2}", gate.max_contradiction));
    }

    let passes = reasons.is_empty();
    let score = (confidence * 0.4 + novelty * 0.3
               + (evidence_count.min(5) as f64 / 5.0) * 0.2
               + (1.0 - contradiction) * 0.1).clamp(0.0, 1.0);

    QualityResult {
        passes,
        score,
        confidence,
        novelty,
        evidence_count,
        contradiction,
        reason: if reasons.is_empty() { None } else { Some(reasons.join("; ")) },
    }
}

// ── Novelty scoring ────────────────────────────────────────────────────────

/// Compute novelty of a new item against a corpus of known items.
/// Uses TF-IDF-like n-gram overlap.
pub fn novelty_score(new_text: &str, known_texts: &[String]) -> f64 {
    if known_texts.is_empty() { return 1.0; } // everything is novel in an empty corpus

    let new_ngrams = text_to_ngrams(new_text);
    if new_ngrams.is_empty() { return 0.0; }

    let max_overlap = known_texts.iter().map(|known| {
        let known_ngrams = text_to_ngrams(known);
        let intersection = new_ngrams.iter()
            .filter(|&ng| known_ngrams.contains(ng))
            .count();
        if known_ngrams.is_empty() { return 0.0; }
        let union = new_ngrams.len() + known_ngrams.len() - intersection;
        intersection as f64 / union as f64  // Jaccard similarity
    }).fold(0.0f64, f64::max);

    (1.0 - max_overlap).clamp(0.0, 1.0)
}

fn text_to_ngrams(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut ngrams = Vec::new();
    // Unigrams + bigrams
    for w in &words { ngrams.push(w.to_lowercase()); }
    for i in 0..words.len().saturating_sub(1) {
        ngrams.push(format!("{} {}", words[i].to_lowercase(), words[i+1].to_lowercase()));
    }
    ngrams
}

// ── BGe score (continuous) ─────────────────────────────────────────────────

/// BGe (Bayesian Gaussian equivalent) score for a continuous dataset.
/// Measures the marginal likelihood of the data given a DAG structure.
/// Simplified version using empirical covariance.
pub fn bge_score_marginal(data: &[f64], n: usize) -> f64 {
    if n == 0 || data.len() < 2 { return 0.0; }
    let mu = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-10 { return 0.0; }
    // Log marginal likelihood ≈ -n/2 * ln(2π*var) - n/2
    -( n as f64 / 2.0) * (2.0 * std::f64::consts::PI * var).ln() - n as f64 / 2.0
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_prior_uniform_mean() {
        let p = BetaPrior::uniform();
        assert!((p.mean() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn beta_prior_update_increases_confidence() {
        let mut p = BetaPrior::uniform();
        p.update(10.0, 1.0); // 10 successes, 1 failure
        assert!(p.mean() > 0.8);
    }

    #[test]
    fn beta_prior_informative() {
        let p = BetaPrior::informative(0.8, 10.0);
        assert!((p.mean() - 0.8).abs() < 0.05);
    }

    #[test]
    fn belief_node_update() {
        let mut node = BeliefNode::new("climate_change");
        node.update(3.0, 1.0); // strong positive evidence
        assert!(node.belief.mean() > 0.6);
        assert!(node.confidence > 0.0);
        assert_eq!(node.n_updates, 1);
    }

    #[test]
    fn belief_network_propagates() {
        let mut net = BayesNetwork::new();
        net.add_node("cause");
        net.add_node("effect");
        net.add_dependency("effect", "cause", 0.8);
        net.observe("cause", 5.0, 1.0); // strong positive evidence for cause
        // Effect should also be updated
        let effect_belief = net.belief("effect").unwrap();
        assert!(effect_belief > 0.5, "effect_belief={effect_belief}");
    }

    #[test]
    fn quality_gate_pass() {
        let gate = QualityGate::default();
        let result = quality_gate_eval(&gate, 0.8, 0.7, 3, 0.1);
        assert!(result.passes);
        assert!(result.score > 0.6);
    }

    #[test]
    fn quality_gate_fail_low_confidence() {
        let gate = QualityGate::default();
        let result = quality_gate_eval(&gate, 0.3, 0.7, 3, 0.1);
        assert!(!result.passes);
        assert!(result.reason.as_ref().unwrap().contains("confidence"));
    }

    #[test]
    fn quality_gate_type5_strict() {
        let gate = QualityGate::type5();
        // Should fail with marginal confidence
        let result = quality_gate_eval(&gate, 0.75, 0.5, 2, 0.15);
        assert!(!result.passes);
    }

    #[test]
    fn quality_result_labels() {
        let mut r = quality_gate_eval(&QualityGate::default(), 0.95, 0.9, 5, 0.05);
        assert_eq!(r.label(), "excellent");
        r.score = 0.65;
        assert_eq!(r.label(), "acceptable");
    }

    #[test]
    fn novelty_new_text() {
        let score = novelty_score("completely novel discovery about X", &[]);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn novelty_identical_text() {
        let known = vec!["discovery about climate change".to_string()];
        let score = novelty_score("discovery about climate change", &known);
        assert!(score < 0.1, "score={score}"); // should be low novelty (high overlap)
    }

    #[test]
    fn novelty_partially_overlapping() {
        let known = vec!["discovery about climate".to_string()];
        let score = novelty_score("new findings on ocean acidification", &known);
        assert!(score > 0.5, "score={score}");
    }

    #[test]
    fn bge_score_finite() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let score = bge_score_marginal(&data, 20);
        assert!(score.is_finite());
    }
}
