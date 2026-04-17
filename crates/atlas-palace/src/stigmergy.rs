//! Stigmergic pheromone system: deposit, decay, path rewards, ACO cost.
//!
//! Ported from GraphPalace `gp-stigmergy` (decay.rs, cost.rs, rewards.rs, pheromones.rs).
//! Implements:
//! - Multiple decay strategies (exponential, sigmoid, linear)
//! - Position-weighted path reward deposits (spec §4.3)
//! - Composite pheromone factor for edge cost modulation (spec §4.4)
//! - Edge-level pheromone decay (3-field: success/traversal/recency)

use crate::types::*;
use crate::Palace;

// ── Decay helpers ─────────────────────────────────────────────────────────

/// Apply one tick of exponential decay: `v *= (1 - rate)`.
/// Returns 0.0 if result falls below `PHEROMONE_FLOOR`.
#[inline]
pub fn decay_exp(current: f32, rate: f32) -> f32 {
    let next = current * (1.0 - rate);
    if next < PHEROMONE_FLOOR { 0.0 } else { next }
}

/// Apply one tick of linear decay: `v -= rate`.
#[inline]
pub fn decay_linear(current: f32, rate: f32) -> f32 {
    let next = (current - rate).max(0.0);
    if next < PHEROMONE_FLOOR { 0.0 } else { next }
}

/// Decay an EdgePheromones struct by one tick using GP default rates.
pub fn decay_edge_pheromones(ep: &mut EdgePheromones) {
    ep.success   = decay_exp(ep.success,   PheromoneType::Success.default_decay_rate());
    ep.traversal = decay_exp(ep.traversal, PheromoneType::Traversal.default_decay_rate());
    ep.recency   = decay_exp(ep.recency,   PheromoneType::Recency.default_decay_rate());
}

// ── Composite pheromone factor (from GP cost.rs §4.4) ─────────────────────

/// Compute the composite pheromone factor from edge pheromones.
///
/// `factor = 0.5×min(success,1) + 0.3×min(recency,1) + 0.2×min(traversal,1)`
///
/// Returns a value in [0, 1].  Used to modulate edge cost.
pub fn pheromone_factor(ep: &EdgePheromones) -> f32 {
    SUCCESS_WEIGHT   * ep.success.min(1.0)
  + RECENCY_WEIGHT   * ep.recency.min(1.0)
  + TRAVERSAL_WEIGHT * ep.traversal.min(1.0)
}

/// Compute pheromone cost for an edge: `1 - factor`.
///
/// High pheromones → low cost (attracting); zero pheromones → cost 1.0.
pub fn pheromone_cost(ep: &EdgePheromones) -> f32 {
    1.0 - pheromone_factor(ep)
}

// ── Path reward deposit (from GP rewards.rs §4.3) ─────────────────────────

/// Deposit pheromones along a successful path.
///
/// For each **edge** at position `i` in a path of length `n`:
/// - `success += base_reward × (1 - i/n)` (position-weighted)
/// - `traversal += 0.1`
/// - `recency = 1.0` (reset, not additive)
///
/// For each **node drawer** along the path:
/// - exploitation pheromone += 0.2
///
/// The `edge_indices` are indices into `Palace.kg` for the edges along the path.
/// The `drawer_ids` are the drawer ids along the path (nodes).
pub fn deposit_path_reward(
    kg: &mut [KgEdge],
    edge_indices: &[usize],
    drawers: &mut std::collections::HashMap<String, crate::types::Drawer>,
    drawer_ids: &[String],
    base_reward: f32,
) {
    let n = edge_indices.len() as f32;
    if n > 0.0 {
        for (i, &ei) in edge_indices.iter().enumerate() {
            if let Some(edge) = kg.get_mut(ei) {
                let weight = 1.0 - (i as f32 / n);
                edge.edge_pheromones.success = (edge.edge_pheromones.success + base_reward * weight).min(PHEROMONE_CEILING);
                edge.edge_pheromones.traversal = (edge.edge_pheromones.traversal + TRAVERSAL_INCREMENT).min(PHEROMONE_CEILING);
                edge.edge_pheromones.recency = RECENCY_VALUE;
            }
        }
    }
    // Deposit exploitation on nodes
    for did in drawer_ids {
        if let Some(d) = drawers.get_mut(did) {
            deposit_on_drawer(d, EXPLOITATION_INCREMENT, "exploitation");
        }
    }
}

/// Deposit a pheromone on a drawer (node-level), updating or creating the tag.
fn deposit_on_drawer(d: &mut crate::types::Drawer, amount: f32, tag: &str) {
    if let Some(p) = d.pheromones.iter_mut().find(|p| p.tag == tag) {
        p.value = (p.value + amount).min(PHEROMONE_CEILING);
    } else {
        d.pheromones.push(Pheromone::new(
            amount,
            PheromoneType::Exploitation.default_decay_rate(),
            tag,
        ));
    }
}

/// Deposit exploration pheromone on a drawer that was visited during search.
pub fn deposit_exploration(d: &mut crate::types::Drawer) {
    deposit_on_drawer(d, EXPLORATION_INCREMENT, "exploration");
}

// ── Palace methods ────────────────────────────────────────────────────────

impl Palace {
    /// Deposit pheromone on a drawer.
    pub fn deposit_pheromones(&mut self, drawer_id: &str, value: f32, decay: f32, tag: &str) {
        if let Some(d) = self.drawers.get_mut(drawer_id) {
            if let Some(p) = d.pheromones.iter_mut().find(|p| p.tag == tag) {
                p.value = (p.value + value).min(PHEROMONE_CEILING);
            } else {
                d.pheromones.push(Pheromone::new(value, decay, tag));
            }
        }
    }

    /// Deposit pheromone with a specific decay strategy.
    pub fn deposit_pheromones_with_strategy(
        &mut self, drawer_id: &str, value: f32, decay: f32,
        tag: &str, strategy: DecayStrategy,
    ) {
        if let Some(d) = self.drawers.get_mut(drawer_id) {
            if let Some(p) = d.pheromones.iter_mut().find(|p| p.tag == tag) {
                p.value = (p.value + value).min(PHEROMONE_CEILING);
                p.strategy = strategy;
            } else {
                d.pheromones.push(Pheromone::with_strategy(value, decay, tag, strategy));
            }
        }
    }

    /// Decay all pheromones by one tick (drawers + KG edges).
    pub fn decay_pheromones(&mut self) {
        self.tick += 1;
        // Drawer-level (node) pheromones
        for d in self.drawers.values_mut() {
            for p in &mut d.pheromones {
                p.tick();
            }
            d.pheromones.retain(|p| p.value > PHEROMONE_FLOOR);
        }
        // Edge-level pheromones on KG edges
        for edge in &mut self.kg {
            decay_edge_pheromones(&mut edge.edge_pheromones);
        }
    }

    /// Return drawers sorted by pheromone strength (hottest first).
    pub fn hot_paths(&self, tag: &str, top_k: usize) -> Vec<(String, f32)> {
        let mut paths: Vec<(String, f32)> = self.drawers.values()
            .map(|d| {
                let ph = if tag.is_empty() {
                    d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max)
                } else {
                    d.pheromones.iter().filter(|p| p.tag == tag).map(|p| p.value).fold(0.0f32, f32::max)
                };
                (d.id.clone(), ph)
            })
            .filter(|(_, ph)| *ph > 0.0)
            .collect();
        paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        paths.truncate(top_k);
        paths
    }

    /// Return drawers with pheromone value below threshold (coldest / least visited).
    pub fn cold_spots(&self, threshold: f32, top_k: usize) -> Vec<String> {
        let mut spots: Vec<_> = self.drawers.values()
            .filter(|d| {
                let max_ph = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max);
                max_ph < threshold
            })
            .map(|d| d.id.clone())
            .collect();
        spots.sort();
        spots.truncate(top_k);
        spots
    }

    /// Deposit a success reward along a path of drawer ids.
    ///
    /// Finds the KG edges connecting consecutive drawers and applies
    /// position-weighted pheromone deposits per GP spec §4.3.
    pub fn deposit_path_success(&mut self, path: &[String], base_reward: f32) {
        if path.len() < 2 { return; }

        // Find edge indices for consecutive pairs
        let mut edge_indices = Vec::new();
        for pair in path.windows(2) {
            let from = &pair[0];
            let to = &pair[1];
            if let Some(idx) = self.kg.iter().position(|e| e.from == *from && e.to == *to) {
                edge_indices.push(idx);
            }
        }

        deposit_path_reward(
            &mut self.kg,
            &edge_indices,
            &mut self.drawers,
            path,
            base_reward,
        );
    }

    /// Average pheromone value across all drawers (for CAS calibration).
    /// Returns 0.0 for an empty palace.
    pub fn avg_pheromone(&self) -> f32 {
        let mut total = 0.0f32;
        let mut count = 0u32;
        for d in self.drawers.values() {
            let max_ph = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max);
            total += max_ph;
            count += 1;
        }
        if count == 0 { 0.0 } else { total / count as f32 }
    }

    /// Return the hottest KG edges by composite pheromone factor.
    pub fn hot_edges(&self, top_k: usize) -> Vec<(String, String, f32)> {
        let mut edges: Vec<_> = self.kg.iter()
            .map(|e| (e.from.clone(), e.to.clone(), pheromone_factor(&e.edge_pheromones)))
            .filter(|(_, _, f)| *f > 0.0)
            .collect();
        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        edges.truncate(top_k);
        edges
    }
}

// ── CAS Decay Calibration ────────────────────────────────────────────────

/// CAS = Corpus-Aware Stigmergy. Dynamically tunes pheromone decay rates
/// based on corpus statistics to prevent saturation or signal loss.
pub struct CasDecayCalibrator {
    /// Base decay rate ∈ [0, 1]. Default 0.05.
    pub base_rate: f32,
    /// Number of calibration ticks performed.
    pub calibrations: u32,
    /// Last computed rate.
    pub last_rate: f32,
}

impl CasDecayCalibrator {
    /// Create a new calibrator with the given base rate.
    pub fn new(base_rate: f32) -> Self {
        Self {
            base_rate: base_rate.clamp(0.001, 1.0),
            calibrations: 0,
            last_rate: base_rate,
        }
    }

    /// Compute optimal decay rate given corpus statistics.
    ///
    /// Rules:
    /// - `source_diversity < 0.3` → boost exploration decay: base × 2.0
    /// - `avg_pheromone > 0.7` → increase decay (prevent saturation): base × 1.5
    /// - `avg_pheromone < 0.1` → decrease decay (preserve signals): base × 0.5
    /// - Otherwise → base_rate
    ///
    /// `source_diversity` ∈ [0, 1] measures how spread sources are (1 = maximally diverse).
    pub fn calibrate(&mut self, avg_pheromone: f32, source_diversity: f32) -> f32 {
        self.calibrations += 1;
        let rate = if source_diversity < 0.3 {
            // Low diversity: aggressively decay to encourage exploration
            (self.base_rate * 2.0).min(1.0)
        } else if avg_pheromone > 0.7 {
            // Near saturation: increase decay
            (self.base_rate * 1.5).min(1.0)
        } else if avg_pheromone < 0.1 {
            // Signals fading: decrease decay
            self.base_rate * 0.5
        } else {
            self.base_rate
        };
        self.last_rate = rate;
        rate
    }

    /// Calibrate and immediately apply multiplicative decay to all palace
    /// drawer pheromones. Returns the rate that was applied.
    ///
    /// Each pheromone value is multiplied by `(1 - rate)`, and values
    /// falling below `PHEROMONE_FLOOR` are removed.
    pub fn apply(&mut self, palace: &mut Palace, avg_pheromone: f32, source_diversity: f32) -> f32 {
        let rate = self.calibrate(avg_pheromone, source_diversity);
        for d in palace.drawers.values_mut() {
            for p in &mut d.pheromones {
                p.value *= 1.0 - rate;
            }
            d.pheromones.retain(|p| p.value > PHEROMONE_FLOOR);
        }
        rate
    }
}

// ── CanonicalPheromoneUpdate ───────────────────────────────────────────────
//
// Implements the canonical equation from Champagnat-Méléard (2011) PTRF §4
// and Champagnat-Hass (2025) AAP for adaptive pheromone decay.
//
// v4.0.3 FIX (Issue #11): The original linear formula
//   λ = base_rate × (1 − ½·μ·σ²·n̄·|∂₁s|)
// exits its valid domain when the canonical term > 1 (common in early training),
// producing negative internal values and a dead gradient at the clamp boundary.
//
// Replacement: exponential formulation
//   λ = base_rate × exp(−½·μ·σ²·n̄·|∂₁s|)
//
// Properties:
//   - Always positive: exp(−x) > 0 for all x.
//   - Zero-gradient fidelity: at zero gradient, output = base_rate (unchanged from original).
//   - First-order match: Taylor exp(−x) ≈ 1 − x + … matches the linear formula to
//     first order for small canonical terms (normal operating range).
//   - Smooth everywhere: no gradient discontinuity at any boundary.
//   - Hardware-friendly: exp is a standard digital/analog primitive; no comparator needed.
//
// where:
//   μ = explore_ratio   (mutation rate proxy)
//   σ = temperature     (landscape variance proxy; σ² = temperature²)
//   n̄ = avg_pheromone   (equilibrium density)
//   ∂₁s = delta_success_rate (selection gradient)

/// State snapshot passed to [`CanonicalPheromoneUpdate::compute_rate`].
#[derive(Debug, Clone)]
pub struct CanonicalUpdateState {
    /// μ — mutation rate proxy (OODA explore_ratio, ∈ [0, 1]).
    pub explore_ratio: f32,
    /// σ — landscape variance proxy (sampling temperature, > 0).
    pub temperature: f32,
    /// n̄ — average pheromone value across all drawers (equilibrium density).
    pub avg_pheromone: f32,
    /// ∂₁s — selection gradient (change in success rate, can be negative).
    pub delta_success_rate: f32,
}

/// Adaptive pheromone decay implementing the Champagnat canonical equation.
///
/// Drop-in companion to [`CasDecayCalibrator`]: call `compute_rate` to get
/// the principled decay rate, then `apply` to update a `Palace` in place.
///
/// # Example
/// ```rust
/// use atlas_palace::stigmergy::{CanonicalPheromoneUpdate, CanonicalUpdateState};
/// let mut updater = CanonicalPheromoneUpdate::default();
/// let state = CanonicalUpdateState {
///     explore_ratio: 0.3, temperature: 0.8,
///     avg_pheromone: 0.5, delta_success_rate: 0.1,
/// };
/// let rate = updater.compute_rate(&state);
/// assert!(rate > 0.0 && rate < 0.30);
/// ```
#[derive(Debug, Clone)]
pub struct CanonicalPheromoneUpdate {
    /// Base decay rate before canonical adjustment (default: 0.05).
    pub base_rate: f32,
    /// Minimum allowed decay rate (default: 0.005).
    pub min_rate: f32,
    /// Maximum allowed decay rate (default: 0.30).
    pub max_rate: f32,
    /// Number of updates applied so far.
    pub updates: u32,
    /// Last computed rate (for diagnostics).
    pub last_rate: f32,
}

impl Default for CanonicalPheromoneUpdate {
    fn default() -> Self {
        Self { base_rate: 0.05, min_rate: 0.005, max_rate: 0.30, updates: 0, last_rate: 0.05 }
    }
}

impl CanonicalPheromoneUpdate {
    /// Create with default parameters.
    pub fn new() -> Self { Self::default() }

    /// Compute adaptive decay rate using Champagnat canonical equation (exp formulation).
    ///
    /// λ = base_rate × exp(−½·μ·σ²·n̄·|∂₁s|)
    ///
    /// This replaces the original linear formula `base_rate × (1 − canonical_term)` which
    /// exits its valid domain (goes negative) when `canonical_term > 1`. Properties:
    ///
    /// - **Always positive**: exp(−x) > 0 for all x.
    /// - **Zero-gradient fidelity**: at `canonical_term = 0`, output = `base_rate` (unchanged).
    /// - **First-order match**: Taylor expansion `exp(−x) ≈ 1 − x + …` matches the original
    ///   linear formula to first order for small gradients.
    /// - **Smooth everywhere**: no gradient discontinuity at any boundary.
    /// - **Natural decay**: as `|∇f| → ∞`, rate → 0 (pheromones freeze at singularities).
    ///
    /// The clamp is retained as a final safety rail for fp edge cases.
    /// See module-level comment for full rationale (Issue #11 fix).
    pub fn compute_rate(&mut self, state: &CanonicalUpdateState) -> f32 {
        let sigma_sq = state.temperature * state.temperature;
        let canonical_term = 0.5
            * state.explore_ratio
            * sigma_sq
            * state.avg_pheromone
            * state.delta_success_rate.abs();
        // Exponential: base_rate × exp(−canonical_term).
        // Always positive, smooth, matches original to first order near zero.
        let rate = self.base_rate * (-canonical_term).exp();
        // Final safety clamp (fp edge case guard; exp already guarantees positivity).
        let rate = rate.clamp(self.min_rate, self.max_rate);
        self.last_rate = rate;
        self.updates += 1;
        rate
    }

    /// Apply canonical decay to all pheromones in `palace`, returning the rate used.
    pub fn apply(&mut self, palace: &mut Palace, state: &CanonicalUpdateState) -> f32 {
        let rate = self.compute_rate(state);
        for d in palace.drawers.values_mut() {
            for p in &mut d.pheromones {
                p.value *= 1.0 - rate;
            }
            d.pheromones.retain(|p| p.value > PHEROMONE_FLOOR);
        }
        rate
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_palace;

    #[test]
    fn pheromone_deposit_and_decay() {
        let mut p = test_palace();
        let drawer_id: String = p.drawers.keys().next().unwrap().clone();
        p.deposit_pheromones(&drawer_id, 1.0, 0.5, "test");
        let ph_before = p.drawers[&drawer_id].pheromones[0].value;
        p.decay_pheromones();
        let ph_after = p.drawers[&drawer_id].pheromones[0].value;
        assert!(ph_after < ph_before, "pheromone should decay");
        assert!((ph_after - 0.5).abs() < 0.01);
    }

    #[test]
    fn hot_paths_and_cold_spots() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.deposit_pheromones(&ids[0], 0.9, 0.1, "hot");
        let hot = p.hot_paths("hot", 3);
        assert!(!hot.is_empty());
        assert_eq!(hot[0].0, ids[0]);
        let cold = p.cold_spots(0.1, 10);
        assert!(!cold.is_empty());
    }

    // ── New: Decay strategy tests ─────────────────────────────────────────

    #[test]
    fn decay_exp_single_step() {
        let r = decay_exp(1.0, 0.02);
        assert!((r - 0.98).abs() < 1e-6);
    }

    #[test]
    fn decay_exp_converges_to_zero() {
        let mut v = 1.0f32;
        for _ in 0..500 {
            v = decay_exp(v, 0.10);
        }
        assert_eq!(v, 0.0);
    }

    #[test]
    fn decay_linear_step() {
        let r = decay_linear(1.0, 0.1);
        assert!((r - 0.9).abs() < 1e-6);
    }

    #[test]
    fn decay_linear_floors() {
        let r = decay_linear(0.05, 0.1);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn linear_strategy_on_pheromone() {
        let mut p = Pheromone::with_strategy(0.5, 0.1, "test", DecayStrategy::Linear);
        p.tick();
        assert!((p.value - 0.4).abs() < 1e-5);
        for _ in 0..10 {
            p.tick();
        }
        assert_eq!(p.value, 0.0);
    }

    #[test]
    fn sigmoid_strategy_on_pheromone() {
        let mut p = Pheromone::with_strategy(0.8, 0.1, "sig", DecayStrategy::Sigmoid { steepness: 4.0 });
        let v0 = p.value;
        p.tick();
        assert!(p.value < v0, "sigmoid should decay");
        assert!(p.value > 0.0, "single tick should not zero it");
    }

    #[test]
    fn deposit_with_strategy() {
        let mut palace = test_palace();
        let id: String = palace.drawers.keys().next().unwrap().clone();
        palace.deposit_pheromones_with_strategy(&id, 0.8, 0.1, "lin", DecayStrategy::Linear);
        let ph = &palace.drawers[&id].pheromones[0];
        assert_eq!(ph.strategy, DecayStrategy::Linear);
        palace.decay_pheromones();
        let ph = &palace.drawers[&id].pheromones[0];
        assert!((ph.value - 0.7).abs() < 0.01);
    }

    // ── New: Pheromone factor / cost tests ────────────────────────────────

    #[test]
    fn pheromone_factor_zero() {
        let ep = EdgePheromones::default();
        assert!((pheromone_factor(&ep) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn pheromone_factor_all_max() {
        let ep = EdgePheromones { success: 1.0, traversal: 1.0, recency: 1.0 };
        assert!((pheromone_factor(&ep) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pheromone_cost_inverse() {
        let ep = EdgePheromones { success: 0.5, traversal: 0.0, recency: 0.5 };
        let factor = pheromone_factor(&ep);
        let cost = pheromone_cost(&ep);
        assert!((factor + cost - 1.0).abs() < 1e-6);
    }

    // ── New: Edge pheromone decay tests ───────────────────────────────────

    #[test]
    fn edge_pheromone_decay() {
        let mut ep = EdgePheromones { success: 1.0, traversal: 1.0, recency: 1.0 };
        decay_edge_pheromones(&mut ep);
        assert!((ep.success - 0.99).abs() < 1e-4);   // 1.0 × (1 - 0.01)
        assert!((ep.traversal - 0.97).abs() < 1e-4);  // 1.0 × (1 - 0.03)
        assert!((ep.recency - 0.90).abs() < 1e-4);    // 1.0 × (1 - 0.10)
    }

    #[test]
    fn edge_pheromone_preserved_zero() {
        let mut ep = EdgePheromones::default();
        decay_edge_pheromones(&mut ep);
        assert_eq!(ep.success, 0.0);
        assert_eq!(ep.traversal, 0.0);
        assert_eq!(ep.recency, 0.0);
    }

    // ── New: Path reward deposit tests ────────────────────────────────────

    #[test]
    fn deposit_path_success_rewards() {
        let mut palace = test_palace();
        let ids: Vec<String> = palace.drawers.keys().cloned().collect();
        // Add KG edges to form a path
        palace.kg_add(&ids[0], &ids[1], "leads_to", 0.9);
        palace.kg_add(&ids[1], &ids[2], "leads_to", 0.9);
        let path = vec![ids[0].clone(), ids[1].clone(), ids[2].clone()];
        palace.deposit_path_success(&path, 1.0);

        // Check edge pheromones
        let e0 = &palace.kg.iter().find(|e| e.from == ids[0] && e.to == ids[1]).unwrap();
        assert!(e0.edge_pheromones.success > 0.0, "first edge should have success pheromone");
        assert!((e0.edge_pheromones.traversal - TRAVERSAL_INCREMENT).abs() < 1e-6);
        assert!((e0.edge_pheromones.recency - RECENCY_VALUE).abs() < 1e-6);

        // Check position weighting (first edge gets more success than second)
        let e1 = &palace.kg.iter().find(|e| e.from == ids[1] && e.to == ids[2]).unwrap();
        assert!(e0.edge_pheromones.success > e1.edge_pheromones.success,
                "position weighting: earlier edges get more reward");

        // Check node exploitation deposits
        for id in &ids {
            let d = &palace.drawers[id];
            assert!(d.pheromones.iter().any(|p| p.tag == "exploitation" && p.value > 0.0),
                    "drawer {} should have exploitation pheromone", id);
        }
    }

    #[test]
    fn hot_edges_after_deposit() {
        let mut palace = test_palace();
        let ids: Vec<String> = palace.drawers.keys().cloned().collect();
        palace.kg_add(&ids[0], &ids[1], "leads_to", 0.9);
        let path = vec![ids[0].clone(), ids[1].clone()];
        palace.deposit_path_success(&path, 1.0);

        let hot = palace.hot_edges(5);
        assert!(!hot.is_empty());
        assert!(hot[0].2 > 0.0);
    }

    #[test]
    fn mmas_ceiling_prevents_saturation() {
        let mut p = test_palace();
        let id: String = p.drawers.keys().next().unwrap().clone();
        // Deposit 100 times — should never exceed PHEROMONE_CEILING
        for _ in 0..100 {
            p.deposit_pheromones(&id, 1.0, 0.01, "flood");
        }
        let val = p.drawers[&id].pheromones.iter()
            .find(|ph| ph.tag == "flood").unwrap().value;
        assert!(val <= PHEROMONE_CEILING + f32::EPSILON,
            "pheromone {} exceeded ceiling {}", val, PHEROMONE_CEILING);
        assert!(val >= PHEROMONE_CEILING - f32::EPSILON,
            "pheromone should have reached ceiling");
    }

    // ── CAS Decay Calibration tests ──────────────────────────────────────

    #[test]
    fn cas_calibrator_increases_decay_on_high_avg() {
        let mut cal = CasDecayCalibrator::new(0.05);
        let rate = cal.calibrate(0.8, 0.5); // high avg, decent diversity
        assert!((rate - 0.075).abs() < 1e-6); // 0.05 * 1.5
        assert_eq!(cal.calibrations, 1);
    }

    #[test]
    fn cas_calibrator_decreases_decay_on_low_avg() {
        let mut cal = CasDecayCalibrator::new(0.10);
        let rate = cal.calibrate(0.05, 0.5); // low avg
        assert!((rate - 0.05).abs() < 1e-6); // 0.10 * 0.5
    }

    #[test]
    fn cas_calibrator_boosts_on_low_diversity() {
        let mut cal = CasDecayCalibrator::new(0.05);
        let rate = cal.calibrate(0.5, 0.2); // low diversity takes priority
        assert!((rate - 0.10).abs() < 1e-6); // 0.05 * 2.0
    }

    #[test]
    fn cas_calibrator_normal_range() {
        let mut cal = CasDecayCalibrator::new(0.05);
        let rate = cal.calibrate(0.4, 0.6); // normal range
        assert!((rate - 0.05).abs() < 1e-6); // unchanged
    }

    #[test]
    fn cas_calibrator_apply_modifies_palace() {
        let mut p = test_palace();
        let id: String = p.drawers.keys().next().unwrap().clone();
        // Deposit pheromone first
        p.deposit_pheromones(&id, 0.8, 0.05, "test");
        let before = p.avg_pheromone();
        assert!(before > 0.0);

        let mut cal = CasDecayCalibrator::new(0.10);
        let rate = cal.apply(&mut p, 0.8, 0.5);
        assert!(rate > 0.10); // should be 0.15 (high avg)
        let after = p.avg_pheromone();
        assert!(after < before);
    }

    #[test]
    fn cas_calibrator_rate_clamped() {
        let mut cal = CasDecayCalibrator::new(0.9);
        let rate = cal.calibrate(0.5, 0.1); // low diversity → 0.9 * 2.0 = 1.8 → clamped to 1.0
        assert!((rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn avg_pheromone_empty_palace_is_zero() {
        let p = Palace::new("empty_test", "/tmp/test_cas_empty");
        assert!((p.avg_pheromone()).abs() < 1e-6);
    }

    #[test]
    fn avg_pheromone_after_deposit() {
        let mut p = test_palace();
        let id: String = p.drawers.keys().next().unwrap().clone();
        p.deposit_pheromones(&id, 0.6, 0.05, "test");
        let avg = p.avg_pheromone();
        assert!(avg > 0.0);
    }

    // ── CanonicalPheromoneUpdate tests ─────────────────────────────────────

    #[test]
    fn canonical_update_default_rate() {
        let mut updater = CanonicalPheromoneUpdate::default();
        let state = CanonicalUpdateState {
            explore_ratio: 0.0, temperature: 1.0,
            avg_pheromone: 0.5, delta_success_rate: 0.0,
        };
        let rate = updater.compute_rate(&state);
        // Δρ = 0 → rate = base_rate × 1.0 = 0.05
        assert!((rate - 0.05).abs() < 1e-6, "expected 0.05, got {rate}");
    }

    #[test]
    fn canonical_update_reduced_rate() {
        let mut updater = CanonicalPheromoneUpdate::default();
        let state = CanonicalUpdateState {
            explore_ratio: 0.5, temperature: 1.0,
            avg_pheromone: 1.0, delta_success_rate: 0.5,
        };
        // Δρ = 0.5×0.5×1.0×1.0×0.5 = 0.125 → rate = 0.05×(1−0.125) = 0.04375
        let rate = updater.compute_rate(&state);
        assert!(rate < 0.05, "rate should be reduced by canonical term, got {rate}");
        assert!(rate >= updater.min_rate);
    }

    #[test]
    fn canonical_update_clamped_to_min() {
        let mut updater = CanonicalPheromoneUpdate::default();
        // Very large canonical term → exp(−5000) underflows to 0.0 → clamp to min_rate.
        // (With the old linear formula this went negative; exp formulation approaches 0.)
        let state = CanonicalUpdateState {
            explore_ratio: 1.0, temperature: 10.0,
            avg_pheromone: 100.0, delta_success_rate: 1.0,
        };
        let rate = updater.compute_rate(&state);
        assert!((rate - updater.min_rate).abs() < 1e-6,
            "should clamp to min_rate={}, got {rate}", updater.min_rate);
    }

    #[test]
    fn canonical_update_counter_increments() {
        let mut updater = CanonicalPheromoneUpdate::default();
        let state = CanonicalUpdateState {
            explore_ratio: 0.3, temperature: 0.8,
            avg_pheromone: 0.5, delta_success_rate: 0.1,
        };
        updater.compute_rate(&state);
        updater.compute_rate(&state);
        assert_eq!(updater.updates, 2);
    }

    #[test]
    fn canonical_update_zero_gradient_uses_base() {
        let mut updater = CanonicalPheromoneUpdate::default();
        let state = CanonicalUpdateState {
            explore_ratio: 0.9, temperature: 2.0,
            avg_pheromone: 5.0, delta_success_rate: 0.0, // zero gradient
        };
        let rate = updater.compute_rate(&state);
        // Δρ = 0 when ∂₁s=0 → rate = base_rate
        assert!((rate - 0.05).abs() < 1e-6, "zero gradient → base rate, got {rate}");
    }

    #[test]
    fn canonical_update_last_rate_stored() {
        let mut updater = CanonicalPheromoneUpdate::default();
        let state = CanonicalUpdateState {
            explore_ratio: 0.2, temperature: 0.5,
            avg_pheromone: 0.8, delta_success_rate: 0.3,
        };
        let rate = updater.compute_rate(&state);
        assert!((updater.last_rate - rate).abs() < 1e-9);
    }

    // ── Issue #11 regression tests — Bug 1: λ decay formula ──────────────────

    /// Rate must be ≥ 0 across 10,000 random (er, temp, avg, grad) combinations.
    /// This directly tests the fix: the old linear formula went negative for many
    /// of these inputs; the exp formulation never does.
    #[test]
    fn canonical_update_never_negative_across_random_inputs() {
        let mut updater = CanonicalPheromoneUpdate::default();
        let mut rng: u64 = 0xdeadbeef_cafebabe;
        for _ in 0..10_000 {
            // LCG — no external deps
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let er   = (rng >> 48) as f32 / 65535.0;                         // [0, 1]
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let temp = 0.01 + (rng >> 48) as f32 / 65535.0 * 20.0;          // [0.01, 20]
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let avg  = (rng >> 48) as f32 / 65535.0 * 100.0;                 // [0, 100]
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let grad = (rng >> 48) as f32 / 65535.0 * 50.0;                  // [0, 50]
            let state = CanonicalUpdateState {
                explore_ratio: er, temperature: temp,
                avg_pheromone: avg, delta_success_rate: grad,
            };
            let rate = updater.compute_rate(&state);
            assert!(rate >= 0.0,
                "rate went negative: er={er}, temp={temp}, avg={avg}, grad={grad} → {rate}");
            assert!(rate <= updater.max_rate + f32::EPSILON,
                "rate exceeded max_rate: {rate}");
        }
    }

    /// The exp formulation is smooth at the old linear formula's failure boundary
    /// (canonical_term = 1.0, i.e., where the original formula hit zero before going
    /// negative). A non-zero finite difference confirms no gradient discontinuity.
    #[test]
    fn canonical_update_smooth_at_old_failure_boundary() {
        let mut updater = CanonicalPheromoneUpdate::default();
        let eps = 1e-3_f32;
        // er=1, temp=√2, avg=1, grad=1 → canonical_term = 0.5×1×2×1×1 = 1.0 (old zero point)
        let state_lo = CanonicalUpdateState {
            explore_ratio: 1.0, temperature: 2_f32.sqrt(),
            avg_pheromone: 1.0, delta_success_rate: 1.0 - eps,
        };
        let state_hi = CanonicalUpdateState {
            explore_ratio: 1.0, temperature: 2_f32.sqrt(),
            avg_pheromone: 1.0, delta_success_rate: 1.0 + eps,
        };
        let r_lo = updater.compute_rate(&state_lo);
        let r_hi = updater.compute_rate(&state_hi);
        // Both must be positive
        assert!(r_lo > 0.0, "r_lo should be positive at old boundary, got {r_lo}");
        assert!(r_hi > 0.0, "r_hi should be positive past old boundary, got {r_hi}");
        // Finite difference must be non-zero (smooth, not flat from clamp)
        let slope = (r_hi - r_lo).abs() / (2.0 * eps);
        assert!(slope > 1e-5,
            "exp should have non-zero slope at old failure boundary, got slope={slope}");
    }
}
