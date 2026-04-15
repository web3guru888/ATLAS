//! Knowledge graph operations: edges, queries, contradictions, similarity graph, tunnels.

use crate::types::*;
use crate::util::cosine_sim;
use crate::Palace;

impl Palace {
    /// Add a directed KG edge.
    pub fn kg_add(&mut self, from: &str, to: &str, relation: &str, confidence: f32) {
        self.kg.push(KgEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation: relation.to_string(),
            confidence,
            timestamp: None,
            edge_pheromones: EdgePheromones::default(),
        });
    }

    /// Add a temporal KG edge with timestamp.
    pub fn kg_add_temporal(&mut self, from: &str, to: &str, relation: &str,
                            confidence: f32, timestamp: u64) {
        self.kg.push(KgEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation: relation.to_string(),
            confidence,
            timestamp: Some(timestamp),
            edge_pheromones: EdgePheromones::default(),
        });
    }

    /// Add KG edge with explicit confidence tracking.
    pub fn kg_add_with_confidence(&mut self, from: &str, to: &str, relation: &str,
                                   confidence: f32) {
        self.kg_add(from, to, relation, confidence);
    }

    /// Query KG edges from a source node.
    pub fn kg_query(&self, from: &str) -> Vec<&KgEdge> {
        self.kg.iter().filter(|e| e.from == from).collect()
    }

    /// Detect contradictions: pairs of edges with opposing relations and overlapping confidence.
    pub fn kg_contradictions(&self, threshold: f32) -> Vec<(String, String, String)> {
        let opposites = [("supports", "contradicts"), ("confirms", "refutes"),
                         ("causes", "prevents"), ("increases", "decreases")];
        let mut result = Vec::new();
        for (i, e1) in self.kg.iter().enumerate() {
            for e2 in &self.kg[i+1..] {
                if e1.from == e2.from && e1.to == e2.to {
                    let is_opposite = opposites.iter().any(|(a, b)|
                        (e1.relation == *a && e2.relation == *b) ||
                        (e1.relation == *b && e2.relation == *a));
                    if is_opposite && (e1.confidence + e2.confidence) > threshold {
                        result.push((e1.from.clone(), e1.to.clone(),
                                     format!("{} vs {}", e1.relation, e2.relation)));
                    }
                }
            }
        }
        result
    }

    /// Invalidate (remove) all edges from/to a drawer id.
    pub fn kg_invalidate(&mut self, drawer_id: &str) {
        self.kg.retain(|e| e.from != drawer_id && e.to != drawer_id);
    }

    /// Build similarity edges between drawers with cosine > threshold.
    /// Returns number of edges added.
    pub fn build_similarity_graph(&mut self, threshold: f32) -> usize {
        let ids: Vec<String> = self.drawers.keys().cloned().collect();
        let mut added = 0;
        for i in 0..ids.len() {
            for j in i+1..ids.len() {
                let sim = {
                    let a = &self.drawers[&ids[i]].embedding;
                    let b = &self.drawers[&ids[j]].embedding;
                    cosine_sim(a, b)
                };
                if sim > threshold {
                    self.kg_add(&ids[i], &ids[j], "similar", sim);
                    added += 1;
                }
            }
        }
        added
    }

    /// Count similarity edges in KG.
    pub fn similarity_edge_count(&self) -> usize {
        self.kg.iter().filter(|e| e.relation == "similar").count()
    }

    /// Build "tunnels" — shortcut edges between high-pheromone drawers in different wings.
    pub fn build_tunnels(&mut self, min_pheromone: f32) -> usize {
        let hot: Vec<String> = self.drawers.values()
            .filter(|d| d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max) >= min_pheromone)
            .map(|d| d.id.clone())
            .collect();

        let mut added = 0;
        for i in 0..hot.len() {
            for j in i+1..hot.len() {
                let (w_i, w_j) = {
                    let ri = self.drawers[&hot[i]].room_id.clone();
                    let rj = self.drawers[&hot[j]].room_id.clone();
                    let wi = self.rooms.get(&ri).map(|r| r.wing_id.clone()).unwrap_or_default();
                    let wj = self.rooms.get(&rj).map(|r| r.wing_id.clone()).unwrap_or_default();
                    (wi, wj)
                };
                if w_i != w_j {
                    let sim = cosine_sim(
                        &self.drawers[&hot[i]].embedding.clone(),
                        &self.drawers[&hot[j]].embedding.clone(),
                    );
                    if sim > 0.3 {
                        self.kg_add(&hot[i], &hot[j], "tunnel", sim);
                        added += 1;
                    }
                }
            }
        }
        self.tunnel_count = added;
        added
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::test_palace;

    #[test]
    fn kg_add_and_query() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "supports", 0.8);
        let edges = p.kg_query(&ids[0]);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation, "supports");
    }

    #[test]
    fn kg_contradictions() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "supports",    0.8);
        p.kg_add(&ids[0], &ids[1], "contradicts", 0.7);
        let c = p.kg_contradictions(0.5);
        assert!(!c.is_empty());
    }

    #[test]
    fn build_similarity_graph() {
        let mut p = test_palace();
        let n = p.build_similarity_graph(0.0);
        assert!(n > 0);
        assert!(p.similarity_edge_count() > 0);
    }

    #[test]
    fn kg_invalidate_removes_edges() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add(&ids[0], &ids[1], "supports", 0.8);
        p.kg_add(&ids[1], &ids[0], "contradicts", 0.7);
        assert_eq!(p.kg.len(), 2);
        p.kg_invalidate(&ids[0]);
        assert_eq!(p.kg.len(), 0);
    }

    #[test]
    fn kg_temporal_edge() {
        let mut p = test_palace();
        let ids: Vec<String> = p.drawers.keys().cloned().collect();
        p.kg_add_temporal(&ids[0], &ids[1], "causes", 0.9, 1713168000);
        let edges = p.kg_query(&ids[0]);
        assert_eq!(edges[0].timestamp, Some(1713168000));
    }
}
