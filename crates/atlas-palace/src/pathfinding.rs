//! Semantic A* pathfinding with pheromone-weighted costs and provenance.
//!
//! Ported from GraphPalace `gp-pathfinding` (astar.rs, heuristic.rs, edge_cost.rs, provenance.rs).
//!
//! ## Algorithm
//!
//! Uses A* search where:
//! - **g-cost**: composite of semantic distance, pheromone attraction, and structural cost
//! - **h-cost**: adaptive semantic heuristic that adjusts between cross-domain exploration
//!   and same-domain pursuit based on cosine similarity to the goal
//! - **Provenance**: every step records g, h, f costs for interpretability
//!
//! Zero external dependencies: uses `std::collections::BinaryHeap` with a custom
//! wrapper to achieve min-heap behavior without `ordered_float`.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::types::*;
use crate::util::cosine_sim;
use crate::Palace;

// ── Min-heap wrapper ──────────────────────────────────────────────────────

/// A* node for the priority queue.  Wraps f-cost in a reverse-ordered newtype
/// so `BinaryHeap` (a max-heap) behaves as a min-heap.
#[derive(Debug)]
struct AStarNode {
    node_id: String,
    f_cost: f32,
    g_cost: f32,
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
    }
}

impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed for min-heap: lower f_cost = higher priority
        other.f_cost.partial_cmp(&self.f_cost).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── Heuristic (from GP heuristic.rs §5.3) ────────────────────────────────

/// Adaptive semantic heuristic.
///
/// When `similarity < threshold` (cross-domain): 50/50 semantic + structural.
/// When `similarity >= threshold` (same-domain): 90/10 semantic-heavy.
///
/// The structural component penalises low-connectivity nodes (fewer KG edges).
pub fn semantic_heuristic(
    current_emb: &[f32],
    goal_emb: &[f32],
    current_degree: usize,
    cross_domain_threshold: f32,
) -> f32 {
    let sim = cosine_sim(current_emb, goal_emb);
    let h_semantic = 1.0 - sim;

    // Connectivity factor: well-connected nodes are easier to route through.
    let connectivity = (current_degree as f32 / 20.0).clamp(0.1, 1.0);
    let h_graph = (h_semantic / connectivity) * 0.5;

    if sim < cross_domain_threshold {
        // Cross-domain: broad exploration via graph structure
        0.5 * h_semantic + 0.5 * h_graph
    } else {
        // Same-domain: focused semantic pursuit
        0.9 * h_semantic + 0.1 * h_graph
    }
}

// ── Composite edge cost (from GP edge_cost.rs §5.1–5.2) ──────────────────

/// Composite edge cost: `α×C_semantic + β×C_pheromone + γ×C_structural`.
pub fn composite_edge_cost(
    target_emb: &[f32],
    goal_emb: &[f32],
    edge_pheromones: &EdgePheromones,
    relation: &str,
    weights: &CostWeights,
) -> f32 {
    let c_sem = 1.0 - cosine_sim(target_emb, goal_emb);
    let c_pher = crate::stigmergy::pheromone_cost(edge_pheromones);
    let c_struct = structural_cost(relation);

    weights.semantic * c_sem + weights.pheromone * c_pher + weights.structural * c_struct
}

// ── A* implementation ─────────────────────────────────────────────────────

impl Palace {
    /// Navigate from `start` to `goal` using pheromone-guided A*.
    ///
    /// Returns a path of drawer ids.  Falls back to greedy BFS if A* finds
    /// no path (e.g. disconnected graph).
    pub fn navigate(&self, start: &str, goal: &str, max_steps: usize) -> Vec<String> {
        if start == goal {
            return vec![start.to_string()];
        }

        // Try A* first
        let config = AStarConfig { max_iterations: max_steps * 10, ..Default::default() };
        if let Some(result) = self.astar_search(start, goal, &config, &CostWeights::default()) {
            return result.path;
        }

        // Fallback: greedy navigation (original behavior)
        self.greedy_navigate(start, goal, max_steps)
    }

    /// Full A* search with provenance tracking.
    ///
    /// Returns `None` if no path exists or `max_iterations` is exceeded.
    pub fn astar_search(
        &self,
        start: &str,
        goal: &str,
        config: &AStarConfig,
        weights: &CostWeights,
    ) -> Option<PathResult> {
        let start_drawer = self.drawers.get(start)?;
        let goal_drawer = self.drawers.get(goal)?;

        if start == goal {
            return Some(PathResult {
                path: vec![start.to_string()],
                edges: Vec::new(),
                total_cost: 0.0,
                iterations: 0,
                nodes_expanded: 0,
                provenance: vec![ProvenanceStep {
                    node_id: start.to_string(),
                    edge_type: String::new(),
                    g_cost: 0.0,
                    h_cost: 0.0,
                    f_cost: 0.0,
                }],
            });
        }

        let start_degree = self.drawer_degree(start);
        let h_start = semantic_heuristic(
            &start_drawer.embedding,
            &goal_drawer.embedding,
            start_degree,
            config.cross_domain_threshold,
        );

        let mut open = BinaryHeap::new();
        open.push(AStarNode {
            node_id: start.to_string(),
            f_cost: h_start,
            g_cost: 0.0,
        });

        // Best known g-cost to each node.
        let mut g_scores: HashMap<String, f32> = HashMap::new();
        g_scores.insert(start.to_string(), 0.0);

        // Backtracking: node_id → (parent_id, edge_relation, g_cost, h_cost)
        let mut came_from: HashMap<String, (String, String, f32, f32)> = HashMap::new();

        let mut iterations = 0usize;
        let mut nodes_expanded = 0usize;

        while let Some(current) = open.pop() {
            iterations += 1;
            if iterations > config.max_iterations {
                return None;
            }

            // Skip if we've found a better path already
            if let Some(&best_g) = g_scores.get(&current.node_id) {
                if current.g_cost > best_g + 1e-6 {
                    continue;
                }
            }

            nodes_expanded += 1;

            // Goal reached!
            if current.node_id == goal {
                return Some(self.reconstruct_path(
                    &came_from, start, goal,
                    &start_drawer.embedding, &goal_drawer.embedding,
                    start_degree,
                    current.g_cost, iterations, nodes_expanded,
                    config.cross_domain_threshold,
                ));
            }

            // Expand neighbors via KG edges
            let neighbors = self.kg_neighbors(&current.node_id);
            for (edge_idx, neighbor_id) in &neighbors {
                let edge = &self.kg[*edge_idx];
                let neighbor_drawer = match self.drawers.get(neighbor_id) {
                    Some(d) => d,
                    None => continue,
                };

                let edge_cost = composite_edge_cost(
                    &neighbor_drawer.embedding,
                    &goal_drawer.embedding,
                    &edge.edge_pheromones,
                    &edge.relation,
                    weights,
                );

                let tentative_g = current.g_cost + edge_cost;

                let is_better = match g_scores.get(neighbor_id) {
                    Some(&existing) => tentative_g < existing - 1e-6,
                    None => true,
                };

                if is_better {
                    let h = semantic_heuristic(
                        &neighbor_drawer.embedding,
                        &goal_drawer.embedding,
                        self.drawer_degree(neighbor_id),
                        config.cross_domain_threshold,
                    );

                    g_scores.insert(neighbor_id.clone(), tentative_g);
                    came_from.insert(
                        neighbor_id.clone(),
                        (current.node_id.clone(), edge.relation.clone(), tentative_g, h),
                    );

                    open.push(AStarNode {
                        node_id: neighbor_id.clone(),
                        f_cost: tentative_g + h,
                        g_cost: tentative_g,
                    });
                }
            }
        }

        // No path found
        None
    }

    /// Greedy navigation fallback (original BFS-like behavior).
    fn greedy_navigate(&self, start: &str, goal: &str, max_steps: usize) -> Vec<String> {
        let goal_emb = self.drawers.get(goal).map(|d| d.embedding.clone()).unwrap_or_default();
        let mut path = vec![start.to_string()];
        let mut visited = HashSet::new();
        visited.insert(start.to_string());

        let mut current = start.to_string();
        for _ in 0..max_steps {
            let neighbours: Vec<&str> = self.kg.iter()
                .filter(|e| e.from == current && !visited.contains(&e.to))
                .map(|e| e.to.as_str())
                .collect();

            if neighbours.is_empty() { break; }

            let best = neighbours.iter().min_by(|&&a, &&b| {
                let da = self.drawers.get(a)
                    .map(|d| {
                        let h = 1.0 - cosine_sim(&d.embedding, &goal_emb);
                        let ph = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max);
                        h - ph * 0.2
                    }).unwrap_or(1.0);
                let db = self.drawers.get(b)
                    .map(|d| {
                        let h = 1.0 - cosine_sim(&d.embedding, &goal_emb);
                        let ph = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max);
                        h - ph * 0.2
                    }).unwrap_or(1.0);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(&next) = best {
                path.push(next.to_string());
                visited.insert(next.to_string());
                if next == goal { break; }
                current = next.to_string();
            } else {
                break;
            }
        }
        path
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Count the degree (number of KG edges from/to) of a drawer.
    fn drawer_degree(&self, drawer_id: &str) -> usize {
        self.kg.iter()
            .filter(|e| e.from == drawer_id || e.to == drawer_id)
            .count()
    }

    /// Get all outgoing KG neighbors of a drawer: (edge_index, neighbor_id).
    fn kg_neighbors(&self, drawer_id: &str) -> Vec<(usize, String)> {
        self.kg.iter()
            .enumerate()
            .filter(|(_, e)| e.from == drawer_id)
            .map(|(i, e)| (i, e.to.clone()))
            .collect()
    }

    /// Reconstruct path from came_from map.
    fn reconstruct_path(
        &self,
        came_from: &HashMap<String, (String, String, f32, f32)>,
        _start: &str,
        goal: &str,
        start_emb: &[f32],
        goal_emb: &[f32],
        start_degree: usize,
        total_cost: f32,
        iterations: usize,
        nodes_expanded: usize,
        cross_domain_threshold: f32,
    ) -> PathResult {
        let mut path = Vec::new();
        let mut edges = Vec::new();
        let mut provenance = Vec::new();

        let mut current_id = goal.to_string();
        loop {
            if let Some((parent, edge_type, g, h)) = came_from.get(&current_id) {
                provenance.push(ProvenanceStep {
                    node_id: current_id.clone(),
                    edge_type: edge_type.clone(),
                    g_cost: *g,
                    h_cost: *h,
                    f_cost: g + h,
                });
                path.push(current_id.clone());
                edges.push(edge_type.clone());
                current_id = parent.clone();
            } else {
                // Start node
                let h_start = semantic_heuristic(
                    start_emb, goal_emb, start_degree, cross_domain_threshold,
                );
                provenance.push(ProvenanceStep {
                    node_id: current_id.clone(),
                    edge_type: String::new(),
                    g_cost: 0.0,
                    h_cost: h_start,
                    f_cost: h_start,
                });
                path.push(current_id);
                break;
            }
        }

        path.reverse();
        edges.reverse();
        provenance.reverse();

        PathResult {
            path,
            edges,
            total_cost,
            iterations,
            nodes_expanded,
            provenance,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_palace;

    #[test]
    fn navigate_follows_kg() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "leads_to", 0.9);
        p.kg_add(&ids[1], &ids[2], "leads_to", 0.9);
        let path = p.navigate(&ids[0], &ids[2], 5);
        assert!(path.len() >= 2);
    }

    #[test]
    fn navigate_self() {
        let p = test_palace();
        let id: String = p.drawers.keys().next().unwrap().clone();
        let path = p.navigate(&id, &id, 5);
        assert_eq!(path, vec![id]);
    }

    // ── New: A* search tests ──────────────────────────────────────────────

    #[test]
    fn astar_linear_path() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "leads_to", 0.9);
        p.kg_add(&ids[1], &ids[2], "leads_to", 0.9);

        let result = p.astar_search(&ids[0], &ids[2], &AStarConfig::default(), &CostWeights::default());
        assert!(result.is_some(), "A* should find path");

        let result = result.unwrap();
        assert_eq!(result.path.len(), 3);
        assert_eq!(result.path[0], ids[0]);
        assert_eq!(result.path[2], ids[2]);
        assert!(result.total_cost > 0.0);
        assert!(result.nodes_expanded >= 2);
        assert_eq!(result.provenance.len(), 3);
    }

    #[test]
    fn astar_no_path_returns_none() {
        let p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        // No KG edges → no path
        let result = p.astar_search(&ids[0], &ids[2], &AStarConfig::default(), &CostWeights::default());
        assert!(result.is_none());
    }

    #[test]
    fn astar_start_equals_goal() {
        let p = test_palace();
        let id: String = p.drawers.keys().next().unwrap().clone();
        let result = p.astar_search(&id, &id, &AStarConfig::default(), &CostWeights::default());
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.path, vec![id]);
        assert_eq!(result.total_cost, 0.0);
    }

    #[test]
    fn astar_max_iterations_respected() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "leads_to", 0.9);
        p.kg_add(&ids[1], &ids[2], "leads_to", 0.9);

        let config = AStarConfig { max_iterations: 1, ..Default::default() };
        let result = p.astar_search(&ids[0], &ids[2], &config, &CostWeights::default());
        // With only 1 iteration, can't reach node 2 hops away
        assert!(result.is_none(), "should fail with max_iterations=1");
    }

    #[test]
    fn astar_provenance_g_nondecreasing() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "leads_to", 0.9);
        p.kg_add(&ids[1], &ids[2], "leads_to", 0.9);

        let result = p.astar_search(&ids[0], &ids[2], &AStarConfig::default(), &CostWeights::default()).unwrap();
        for w in result.provenance.windows(2) {
            assert!(w[1].g_cost >= w[0].g_cost - 1e-6,
                    "g-cost should be non-decreasing: {} vs {}", w[0].g_cost, w[1].g_cost);
        }
    }

    #[test]
    fn astar_nonexistent_node_returns_none() {
        let p = test_palace();
        assert!(p.astar_search("MISSING", "ALSO_MISSING", &AStarConfig::default(), &CostWeights::default()).is_none());
    }

    // ── New: Heuristic tests ──────────────────────────────────────────────

    #[test]
    fn heuristic_identical_is_zero() {
        let emb = vec![1.0f32, 0.0, 0.0];
        let h = semantic_heuristic(&emb, &emb, 10, 0.3);
        assert!(h.abs() < 1e-4, "identical embeddings → h ≈ 0, got {h}");
    }

    #[test]
    fn heuristic_orthogonal_is_large() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let h = semantic_heuristic(&a, &b, 5, 0.3);
        assert!(h > 0.5, "orthogonal embeddings should have high h, got {h}");
    }

    #[test]
    fn heuristic_low_degree_higher() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 1.0];
        let h_low = semantic_heuristic(&a, &b, 1, 0.3);
        let h_high = semantic_heuristic(&a, &b, 20, 0.3);
        assert!(h_low > h_high, "low degree should give higher h: {h_low} vs {h_high}");
    }

    // ── New: Composite edge cost tests ────────────────────────────────────

    #[test]
    fn composite_cost_zero_pheromones() {
        let target = vec![1.0f32, 0.0];
        let goal = vec![1.0f32, 0.0]; // identical → semantic=0
        let ep = EdgePheromones::default(); // zero → pheromone_cost=1.0
        let cost = composite_edge_cost(&target, &goal, &ep, "leads_to", &CostWeights::default());
        // 0.4×0.0 + 0.3×1.0 + 0.3×0.4 = 0.42
        assert!((cost - 0.42).abs() < 0.01, "got {cost}");
    }

    #[test]
    fn composite_cost_max_pheromones() {
        let target = vec![1.0f32, 0.0];
        let goal = vec![0.0f32, 1.0]; // orthogonal → semantic≈1.0
        let ep = EdgePheromones { success: 1.0, traversal: 1.0, recency: 1.0 };
        let cost = composite_edge_cost(&target, &goal, &ep, "CONTAINS", &CostWeights::default());
        // 0.4×1.0 + 0.3×0.0 + 0.3×0.2 = 0.46
        assert!((cost - 0.46).abs() < 0.01, "got {cost}");
    }
}
