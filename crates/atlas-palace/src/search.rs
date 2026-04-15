//! Full-text, semantic, and pheromone-weighted search.

use crate::types::SearchResult;
use crate::util::{cosine_sim, tfidf_embedding};
use crate::Palace;

impl Palace {
    /// Full-text + pheromone-weighted search across all drawers.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let q_emb = tfidf_embedding(query);
        let q_lower = query.to_lowercase();
        let mut results: Vec<SearchResult> = self.drawers.values().map(|d| {
            let sem = cosine_sim(&q_emb, &d.embedding);
            // Keyword boost
            let kw_boost = if d.content.to_lowercase().contains(&q_lower)
                           || d.title.to_lowercase().contains(&q_lower) { 0.2 } else { 0.0 };
            // Pheromone boost
            let ph_boost = d.pheromones.iter().map(|p| p.value).fold(0.0f32, f32::max) * 0.1;
            let score = (sem + kw_boost + ph_boost).min(1.0);
            SearchResult {
                drawer_id: d.id.clone(),
                score,
                preview: d.content.chars().take(120).collect(),
            }
        }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Semantic search using embedding similarity only.
    pub fn search_by_embedding(&self, embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self.drawers.values().map(|d| {
            let score = cosine_sim(embedding, &d.embedding);
            SearchResult {
                drawer_id: d.id.clone(),
                score,
                preview: d.content.chars().take(120).collect(),
            }
        }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Find drawers similar to a given drawer_id.
    pub fn find_similar(&self, drawer_id: &str, top_k: usize) -> Vec<SearchResult> {
        let emb = self.drawers.get(drawer_id)
            .map(|d| d.embedding.clone())
            .unwrap_or_default();
        let mut results: Vec<SearchResult> = self.drawers.values()
            .filter(|d| d.id != drawer_id)
            .map(|d| {
                let score = cosine_sim(&emb, &d.embedding);
                SearchResult {
                    drawer_id: d.id.clone(),
                    score,
                    preview: d.content.chars().take(120).collect(),
                }
            }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::test_palace;

    #[test]
    fn search_finds_relevant() {
        let p = test_palace();
        let results = p.search("pheromone ants", 2);
        assert!(!results.is_empty());
        let titles: Vec<_> = results.iter()
            .filter_map(|r| p.drawers.get(&r.drawer_id))
            .map(|d| d.title.as_str())
            .collect();
        assert!(titles.iter().any(|&t| t.contains("ACO") || t.contains("DPPN")));
    }

    #[test]
    fn search_by_embedding_works() {
        let p = test_palace();
        let emb = crate::util::tfidf_embedding("pheromone");
        let results = p.search_by_embedding(&emb, 5);
        assert!(!results.is_empty());
    }
}
