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
}
