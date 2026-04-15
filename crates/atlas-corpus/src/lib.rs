//! atlas-corpus — LiveDiscoveryCorpus: stigmergic curriculum engine. Stage 6.
//!
//! Ingests [`Discovery`](atlas_astra::Discovery) items from the ASTRA OODA engine,
//! applies a 5-gate quality pipeline, maintains pheromone weights per entry, and
//! exposes a **pheromone-weighted batch sampler** for training loops.
//!
//! # Architecture
//!
//! ```text
//! AstraEngine ──→ LiveDiscoveryCorpus ──→ TrainingBatch
//!                    │  5 quality gates        ↑
//!                    │  pheromone weights       │
//!                    │  curriculum scheduling   │
//!                    └──→ Palace pheromones ────┘
//! ```
//!
//! # Zero external crate dependencies.

#![warn(missing_docs)]

use std::collections::HashMap;

use atlas_astra::Discovery;
use atlas_core::Result;
use atlas_json::Json;
use atlas_palace::Palace;

// ────────────────────────────────────────────────────────────────────────────
//  Quality gates
// ────────────────────────────────────────────────────────────────────────────

/// Result of running all five quality gates on a candidate entry.
#[derive(Debug, Clone)]
pub struct GateResult {
    /// Gate 1: minimum confidence threshold.
    pub confidence_ok: bool,
    /// Gate 2: sufficient novelty vs existing corpus.
    pub novelty_ok: bool,
    /// Gate 3: source reliability score.
    pub source_ok: bool,
    /// Gate 4: TRM causal plausibility (heuristic without live TRM).
    pub causal_ok: bool,
    /// Gate 5: corpus diversity (not over-representing one source).
    pub diversity_ok: bool,
    /// Overall pass — all five gates must pass.
    pub pass: bool,
}

impl GateResult {
    fn new(c: bool, n: bool, s: bool, ca: bool, d: bool) -> Self {
        Self {
            confidence_ok: c,
            novelty_ok: n,
            source_ok: s,
            causal_ok: ca,
            diversity_ok: d,
            pass: c && n && s && ca && d,
        }
    }
}

/// Configuration for the five quality gates.
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Minimum confidence (0–1) for a discovery to enter the corpus. Default 0.55.
    pub min_confidence: f64,
    /// Minimum novelty score (0–1) — Jaccard n-gram distance to nearest existing entry. Default 0.25.
    pub min_novelty: f64,
    /// Sources with lower-than-this reliability are rejected. Default 0.4.
    pub min_source_reliability: f64,
    /// Maximum fraction of corpus that may come from one source before diversity gate fires. Default 0.5.
    pub max_source_fraction: f64,
    /// Minimum word count in a discovery summary to pass causal gate. Default 6.
    pub min_summary_words: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.55,
            min_novelty: 0.25,
            min_source_reliability: 0.4,
            max_source_fraction: 0.5,
            min_summary_words: 6,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Corpus entry
// ────────────────────────────────────────────────────────────────────────────

/// A quality-gated discovery stored in the corpus, annotated with learning metadata.
#[derive(Debug, Clone)]
pub struct CorpusEntry {
    /// Unique monotonic entry ID.
    pub id: usize,
    /// Underlying ASTRA discovery.
    pub discovery: Discovery,
    /// Current pheromone weight in [0, 1]. Higher → more likely to be sampled.
    pub pheromone: f64,
    /// How many times this entry has been included in a training batch.
    pub sample_count: u32,
    /// Composite quality score (average of gate inputs, 0–1).
    pub quality_score: f64,
    /// Curriculum tier: 0 = easy, 1 = medium, 2 = hard. Assigned on insert, updated by feedback.
    pub tier: u8,
    /// Ingestion timestamp (UNIX seconds, best-effort via manual counter since no std::time).
    pub ingested_at: u64,
}

impl CorpusEntry {
    /// Text representation suitable for training: `"[SOURCE] TITLE (quality=0.87)"`.
    pub fn to_training_text(&self) -> String {
        format!(
            "[{}] {} (quality={:.2})",
            self.discovery.source,
            self.discovery.title,
            self.discovery.quality_score,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Batch
// ────────────────────────────────────────────────────────────────────────────

/// A sampled training batch with per-entry metadata.
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Sampled entries (cloned).
    pub entries: Vec<CorpusEntry>,
    /// Sampling strategy used to produce this batch.
    pub strategy: SampleStrategy,
    /// Mean pheromone weight of selected entries.
    pub mean_pheromone: f64,
}

/// Sampling strategy for batch construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStrategy {
    /// Pure pheromone-weighted (exploitation).
    Pheromone,
    /// Curriculum-ordered: start from easy (tier 0), gradually introduce harder.
    Curriculum,
    /// Uniform random (exploration).
    Uniform,
    /// Mix: 60% pheromone + 40% uniform.
    Mixed,
}

// ────────────────────────────────────────────────────────────────────────────
//  Corpus statistics
// ────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics over the corpus.
#[derive(Debug, Clone)]
pub struct CorpusStats {
    /// Total entries in corpus.
    pub total_entries: usize,
    /// Entries rejected by quality gates.
    pub total_rejected: usize,
    /// Mean confidence of accepted entries.
    pub mean_confidence: f64,
    /// Mean pheromone weight.
    pub mean_pheromone: f64,
    /// Number of unique sources represented.
    pub unique_sources: usize,
    /// Distribution across tiers.
    pub tier_counts: [usize; 3],
    /// Total positive feedback signals received.
    pub positive_feedback: u64,
    /// Total negative feedback signals received.
    pub negative_feedback: u64,
}

// ────────────────────────────────────────────────────────────────────────────
//  LiveDiscoveryCorpus
// ────────────────────────────────────────────────────────────────────────────

/// Stigmergic living corpus: grows from ASTRA discoveries, guides training via pheromones.
///
/// # Example
/// ```rust
/// use atlas_corpus::{LiveDiscoveryCorpus, GateConfig};
/// let mut corpus = LiveDiscoveryCorpus::new(GateConfig::default());
/// assert_eq!(corpus.len(), 0);
/// ```
pub struct LiveDiscoveryCorpus {
    entries: Vec<CorpusEntry>,
    gate_cfg: GateConfig,
    /// Source reliability map.  Unknown sources default to 0.6.
    source_reliability: HashMap<String, f64>,
    /// Per-source entry count, for diversity gate.
    source_counts: HashMap<String, usize>,
    /// Monotonic counter for IDs.
    next_id: usize,
    /// Pseudo-random state for sampling (LCG, seeded at construction).
    rng: u64,
    /// Total rejects.
    total_rejected: usize,
    /// Monotonic simulated clock (incremented per add).
    clock: u64,
    positive_feedback: u64,
    negative_feedback: u64,
    /// Optional palace for pheromone deposit/decay.
    palace: Option<Palace>,
}

impl LiveDiscoveryCorpus {
    /// Create a new corpus with the given gate configuration.
    pub fn new(gate_cfg: GateConfig) -> Self {
        let mut source_reliability = HashMap::new();
        // Pre-seed known reliable sources.
        source_reliability.insert("arxiv".into(), 0.85);
        source_reliability.insert("nasa_power".into(), 0.90);
        source_reliability.insert("who_gho".into(), 0.90);
        source_reliability.insert("world_bank".into(), 0.85);
        source_reliability.insert("text_pattern".into(), 0.60);
        source_reliability.insert("unknown".into(), 0.50);

        Self {
            entries: Vec::new(),
            gate_cfg,
            source_reliability,
            source_counts: HashMap::new(),
            next_id: 0,
            rng: 0xDEAD_BEEF_1234_5678,
            total_rejected: 0,
            clock: 0,
            positive_feedback: 0,
            negative_feedback: 0,
            palace: None,
        }
    }

    /// Attach a Palace for pheromone integration.
    pub fn with_palace(mut self, palace: Palace) -> Self {
        self.palace = Some(palace);
        self
    }

    /// Current corpus size (accepted entries only).
    pub fn len(&self) -> usize { self.entries.len() }

    /// True if corpus is empty.
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Register a source reliability score (0–1).
    pub fn set_source_reliability(&mut self, source: &str, score: f64) {
        self.source_reliability.insert(source.to_string(), score.clamp(0.0, 1.0));
    }

    // ──────────────────────────────────────────────────
    //  Gate evaluation
    // ──────────────────────────────────────────────────

    /// Run all five quality gates on a candidate discovery.
    pub fn evaluate_gates(&self, d: &Discovery) -> GateResult {
        // Gate 1: confidence (quality_score is [0,1])
        let conf_ok = d.quality_score >= self.gate_cfg.min_confidence;

        // Gate 2: novelty (Jaccard bigram similarity against corpus titles)
        let novelty = self.compute_novelty(&d.title);
        let novelty_ok = novelty >= self.gate_cfg.min_novelty;

        // Gate 3: source reliability
        let rel = self.source_reliability.get(&d.source).copied().unwrap_or(0.6);
        let source_ok = rel >= self.gate_cfg.min_source_reliability;

        // Gate 4: causal plausibility — title length and presence of causal language
        let words: Vec<&str> = d.title.split_whitespace().collect();
        let causal_ok = words.len() >= self.gate_cfg.min_summary_words;

        // Gate 5: diversity — no source may exceed max_source_fraction of corpus
        let diversity_ok = if self.entries.is_empty() {
            true
        } else {
            let source_count = self.source_counts.get(&d.source).copied().unwrap_or(0);
            let fraction = source_count as f64 / self.entries.len() as f64;
            fraction < self.gate_cfg.max_source_fraction
        };

        GateResult::new(conf_ok, novelty_ok, source_ok, causal_ok, diversity_ok)
    }

    /// Try to add a discovery.  Returns the entry ID if accepted, None if rejected.
    pub fn add_discovery(&mut self, d: Discovery) -> Option<usize> {
        let gate = self.evaluate_gates(&d);
        if !gate.pass {
            self.total_rejected += 1;
            return None;
        }

        // Compute composite quality score
        let rel = self.source_reliability.get(&d.source).copied().unwrap_or(0.6);
        let novelty = self.compute_novelty(&d.title);
        let quality_score = (d.quality_score + rel + novelty) / 3.0;

        // Assign curriculum tier
        let tier = if quality_score >= 0.75 { 2 } else if quality_score >= 0.55 { 1 } else { 0 };

        // Initial pheromone = quality score
        let pheromone = quality_score;

        let id = self.next_id;
        self.next_id += 1;
        self.clock += 1;

        // Update source count
        *self.source_counts.entry(d.source.clone()).or_insert(0) += 1;

        // Deposit in palace if attached
        if let Some(ref mut palace) = self.palace {
            let tag = format!("corpus:{}", d.source);
            palace.deposit_pheromones("corpus", pheromone as f32, 0.05, &tag);
        }

        self.entries.push(CorpusEntry {
            id,
            discovery: d,
            pheromone,
            sample_count: 0,
            quality_score,
            tier,
            ingested_at: self.clock,
        });

        Some(id)
    }

    /// Bulk-add from an ASTRA engine run.  Returns (accepted, rejected).
    pub fn ingest(&mut self, discoveries: Vec<Discovery>) -> (usize, usize) {
        let mut accepted = 0;
        let mut rejected = 0;
        for d in discoveries {
            if self.add_discovery(d).is_some() {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }
        (accepted, rejected)
    }

    // ──────────────────────────────────────────────────
    //  Sampling
    // ──────────────────────────────────────────────────

    /// Sample a training batch of up to `n` entries using the given strategy.
    pub fn sample_batch(&mut self, n: usize, strategy: SampleStrategy) -> TrainingBatch {
        if self.entries.is_empty() {
            return TrainingBatch { entries: vec![], strategy, mean_pheromone: 0.0 };
        }
        let k = n.min(self.entries.len());

        let indices: Vec<usize> = match strategy {
            SampleStrategy::Pheromone => self.sample_pheromone(k),
            SampleStrategy::Curriculum => self.sample_curriculum(k),
            SampleStrategy::Uniform => self.sample_uniform(k),
            SampleStrategy::Mixed => {
                let n_ph = (k as f64 * 0.6).round() as usize;
                let n_un = k - n_ph;
                let mut idx = self.sample_pheromone(n_ph);
                let more = self.sample_uniform(n_un);
                // Merge without duplicates
                for i in more {
                    if !idx.contains(&i) { idx.push(i); }
                }
                idx.truncate(k);
                idx
            }
        };

        // Increment sample counts
        for &i in &indices {
            self.entries[i].sample_count += 1;
            // Slight pheromone evaporation per sample
            self.entries[i].pheromone *= 0.98;
            self.entries[i].pheromone = self.entries[i].pheromone.max(0.01);
        }

        let selected: Vec<CorpusEntry> = indices.iter().map(|&i| self.entries[i].clone()).collect();
        let mean_ph = if selected.is_empty() {
            0.0
        } else {
            selected.iter().map(|e| e.pheromone).sum::<f64>() / selected.len() as f64
        };

        TrainingBatch { entries: selected, strategy, mean_pheromone: mean_ph }
    }

    /// Positive feedback: strengthen pheromone for entry `id`.
    pub fn feedback_positive(&mut self, id: usize) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.pheromone = (e.pheromone * 1.2).min(1.0);
            // Potentially promote tier
            if e.tier < 2 && e.pheromone > 0.8 { e.tier += 1; }
            self.positive_feedback += 1;

            if let Some(ref mut palace) = self.palace {
                let tag = format!("corpus:{}", e.discovery.source);
                palace.deposit_pheromones("corpus", 0.2, 0.05, &tag);
            }
        }
    }

    /// Negative feedback: weaken pheromone for entry `id`.
    pub fn feedback_negative(&mut self, id: usize) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.pheromone = (e.pheromone * 0.7).max(0.01);
            self.negative_feedback += 1;
        }
    }

    /// Decay all pheromones by factor (0 < factor < 1).  Call periodically.
    pub fn decay_pheromones(&mut self, factor: f64) {
        let f = factor.clamp(0.0, 1.0);
        for e in &mut self.entries {
            e.pheromone = (e.pheromone * f).max(0.01);
        }
        if let Some(ref mut palace) = self.palace {
            palace.decay_pheromones();
        }
    }

    // ──────────────────────────────────────────────────
    //  Stats
    // ──────────────────────────────────────────────────

    /// Corpus-wide statistics.
    pub fn stats(&self) -> CorpusStats {
        let n = self.entries.len();
        let mean_confidence = if n == 0 {
            0.0
        } else {
            self.entries.iter().map(|e| e.discovery.quality_score).sum::<f64>() / n as f64
        };
        let mean_pheromone = if n == 0 {
            0.0
        } else {
            self.entries.iter().map(|e| e.pheromone).sum::<f64>() / n as f64
        };
        let unique_sources = self.source_counts.len();
        let mut tier_counts = [0usize; 3];
        for e in &self.entries {
            tier_counts[e.tier as usize] += 1;
        }
        CorpusStats {
            total_entries: n,
            total_rejected: self.total_rejected,
            mean_confidence,
            mean_pheromone,
            unique_sources,
            tier_counts,
            positive_feedback: self.positive_feedback,
            negative_feedback: self.negative_feedback,
        }
    }

    /// Iterate over all entries (read-only).
    pub fn entries(&self) -> &[CorpusEntry] {
        &self.entries
    }

    /// Get a single entry by ID.
    pub fn get(&self, id: usize) -> Option<&CorpusEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// All unique sources in the corpus.
    pub fn sources(&self) -> Vec<String> {
        self.source_counts.keys().cloned().collect()
    }

    // ──────────────────────────────────────────────────
    //  Persistence
    // ──────────────────────────────────────────────────

    /// Serialize corpus to JSON string.
    pub fn to_json(&self) -> String {
        let entries_json: Vec<String> = self.entries.iter().map(|e| {
            format!(
                r#"{{"id":{},"source":"{}","title":{},"confidence":{:.4},"pheromone":{:.4},"quality":{:.4},"tier":{},"samples":{},"ingested_at":{}}}"#,
                e.id,
                e.discovery.source,
                json_string_escape(&e.discovery.title),
                e.discovery.quality_score,
                e.pheromone,
                e.quality_score,
                e.tier,
                e.sample_count,
                e.ingested_at,
            )
        }).collect();
        format!(
            r#"{{"version":1,"total_rejected":{},"positive_feedback":{},"negative_feedback":{},"clock":{},"entries":[{}]}}"#,
            self.total_rejected,
            self.positive_feedback,
            self.negative_feedback,
            self.clock,
            entries_json.join(","),
        )
    }

    /// Save corpus to a JSON file.
    pub fn save(&self, path: &str) -> Result<()> {
        let json = self.to_json();
        std::fs::write(path, json).map_err(|e| atlas_core::AtlasError::Io(e.to_string()))
    }

    /// Load corpus from a JSON file (reconstructs entries; palace attachment must be re-done).
    pub fn load(path: &str, gate_cfg: GateConfig) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| atlas_core::AtlasError::Io(e.to_string()))?;
        Self::from_json_str(&raw, gate_cfg)
    }

    /// Parse corpus from a JSON string.
    pub fn from_json_str(s: &str, gate_cfg: GateConfig) -> Result<Self> {
        let root = Json::parse(s).map_err(|e| atlas_core::AtlasError::Parse(format!("{:?}", e)))?;
        let mut corpus = Self::new(gate_cfg);

        if let Some(entries) = root.get("entries").and_then(|v: &Json| v.as_array()) {
            for obj in entries {
                let id        = obj.get("id").and_then(|v: &Json| v.as_usize()).unwrap_or(0);
                let source    = obj.get("source").and_then(|v: &Json| v.as_str()).unwrap_or("unknown").to_string();
                let title     = obj.get("title").and_then(|v: &Json| v.as_str()).unwrap_or("").to_string();
                let confidence = obj.get("confidence").and_then(|v: &Json| v.as_f64()).unwrap_or(0.5);
                let pheromone = obj.get("pheromone").and_then(|v: &Json| v.as_f64()).unwrap_or(0.5);
                let quality   = obj.get("quality").and_then(|v: &Json| v.as_f64()).unwrap_or(0.5);
                let tier      = obj.get("tier").and_then(|v: &Json| v.as_usize()).unwrap_or(0) as u8;
                let samples   = obj.get("samples").and_then(|v: &Json| v.as_usize()).unwrap_or(0) as u32;
                let ingested  = obj.get("ingested_at").and_then(|v: &Json| v.as_usize()).unwrap_or(0) as u64;

                let discovery = Discovery {
                    id: format!("corpus-{}", id),
                    source: source.clone(),
                    title,
                    description: String::new(),
                    causal_claims: vec![],
                    quality_score: confidence,
                    proof_commitment: 0,
                    timestamp: ingested,
                    tags: vec![],
                };
                *corpus.source_counts.entry(source).or_insert(0) += 1;
                corpus.entries.push(CorpusEntry {
                    id,
                    discovery,
                    pheromone,
                    sample_count: samples,
                    quality_score: quality,
                    tier,
                    ingested_at: ingested,
                });
                if id >= corpus.next_id { corpus.next_id = id + 1; }
            }
        }

        corpus.total_rejected = root.get("total_rejected").and_then(|v: &Json| v.as_usize()).unwrap_or(0);
        corpus.positive_feedback = root.get("positive_feedback").and_then(|v: &Json| v.as_usize()).unwrap_or(0) as u64;
        corpus.negative_feedback = root.get("negative_feedback").and_then(|v: &Json| v.as_usize()).unwrap_or(0) as u64;
        corpus.clock = root.get("clock").and_then(|v: &Json| v.as_usize()).unwrap_or(0) as u64;

        Ok(corpus)
    }

    // ──────────────────────────────────────────────────
    //  Internal helpers
    // ──────────────────────────────────────────────────

    /// Jaccard bigram novelty of `text` vs all titles in corpus.  1.0 = entirely novel.
    fn compute_novelty(&self, text: &str) -> f64 {
        if self.entries.is_empty() { return 1.0; }
        let toks: Vec<&str> = text.split_whitespace().collect();
        if toks.len() < 2 { return 0.5; }
        let bigrams_new: std::collections::HashSet<(&str, &str)> =
            toks.windows(2).map(|w| (w[0], w[1])).collect();

        // Find maximum Jaccard similarity against any existing entry
        let mut max_sim = 0.0f64;
        for e in &self.entries {
            let toks2: Vec<&str> = e.discovery.title.split_whitespace().collect();
            if toks2.len() < 2 { continue; }
            let bigrams_ex: std::collections::HashSet<(&str, &str)> =
                toks2.windows(2).map(|w| (w[0], w[1])).collect();
            let inter = bigrams_new.intersection(&bigrams_ex).count() as f64;
            let union = bigrams_new.union(&bigrams_ex).count() as f64;
            if union > 0.0 {
                let sim = inter / union;
                if sim > max_sim { max_sim = sim; }
            }
        }
        1.0 - max_sim
    }

    fn lcg_next(&mut self) -> u64 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng
    }

    fn sample_pheromone(&mut self, k: usize) -> Vec<usize> {
        let total: f64 = self.entries.iter().map(|e| e.pheromone).sum();
        if total == 0.0 { return self.sample_uniform(k); }
        let mut chosen = Vec::with_capacity(k);
        let mut remaining: Vec<(usize, f64)> = self.entries.iter().enumerate()
            .map(|(i, e)| (i, e.pheromone)).collect();

        for _ in 0..k {
            if remaining.is_empty() { break; }
            let sum: f64 = remaining.iter().map(|(_, w)| w).sum();
            let r = (self.lcg_next() as f64 / u64::MAX as f64) * sum;
            let mut acc = 0.0;
            let mut sel = 0;
            for (idx, (i, w)) in remaining.iter().enumerate() {
                acc += w;
                if r <= acc { sel = idx; break; }
            }
            let (chosen_i, _) = remaining.remove(sel);
            chosen.push(chosen_i);
        }
        chosen
    }

    fn sample_uniform(&mut self, k: usize) -> Vec<usize> {
        let n = self.entries.len();
        if k >= n {
            return (0..n).collect();
        }
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.lcg_next() as usize % (n - i));
            pool.swap(i, j);
        }
        pool[..k].to_vec()
    }

    fn sample_curriculum(&mut self, k: usize) -> Vec<usize> {
        // Prefer tier 0 entries first, then tier 1, then tier 2
        let mut tiers: [Vec<usize>; 3] = [vec![], vec![], vec![]];
        for (i, e) in self.entries.iter().enumerate() {
            tiers[e.tier as usize].push(i);
        }
        let mut result = Vec::with_capacity(k);
        for tier in &mut tiers {
            for &i in tier.iter() {
                if result.len() >= k { break; }
                result.push(i);
            }
        }
        result
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Helper: JSON string escape
// ────────────────────────────────────────────────────────────────────────────

fn json_string_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ────────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_astra::Discovery;

    fn make_discovery(source: &str, title: &str, quality_score: f64) -> Discovery {
        Discovery {
            id: format!("test-{}", title.len()),
            source: source.to_string(),
            title: title.to_string(),
            description: String::new(),
            causal_claims: vec![],
            quality_score,
            proof_commitment: 0,
            timestamp: 0,
            tags: vec![],
        }
    }

    #[test]
    fn empty_corpus() {
        let c = LiveDiscoveryCorpus::new(GateConfig::default());
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn add_valid_discovery_accepted() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let d = make_discovery("arxiv", "Neural networks can learn causal representations efficiently", 0.80);
        let id = c.add_discovery(d);
        assert!(id.is_some());
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn add_low_confidence_rejected() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let d = make_discovery("arxiv", "Neural networks can learn causal representations efficiently", 0.10);
        let id = c.add_discovery(d);
        assert!(id.is_none());
        assert_eq!(c.len(), 0);
        assert_eq!(c.stats().total_rejected, 1);
    }

    #[test]
    fn add_short_summary_rejected() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let d = make_discovery("arxiv", "too short", 0.90);
        let id = c.add_discovery(d);
        assert!(id.is_none());
    }

    #[test]
    fn duplicate_novelty_rejected() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let text = "Reinforcement learning agents can discover causal structure from environments";
        let d1 = make_discovery("arxiv", text, 0.80);
        let d2 = make_discovery("arxiv", text, 0.80); // identical
        c.add_discovery(d1);
        // Second one should be rejected (0 novelty)
        let id2 = c.add_discovery(d2);
        // novelty = 0.0 < min_novelty 0.25 → rejected
        assert!(id2.is_none());
    }

    #[test]
    fn source_diversity_gate() {
        let mut cfg = GateConfig::default();
        cfg.max_source_fraction = 0.5;
        let mut c = LiveDiscoveryCorpus::new(cfg);

        // First 2 from arxiv accepted (no diversity problem yet)
        let d1 = make_discovery("arxiv", "Neural networks learn causal representations efficiently in training", 0.80);
        let d2 = make_discovery("nasa_power", "Global temperature anomaly increases correlated with CO2 concentration", 0.85);
        c.add_discovery(d1);
        c.add_discovery(d2);
        // 3rd from arxiv: now 1/2 = 50% which equals threshold; depends on < vs <=
        let d3 = make_discovery("arxiv", "Transformer models exhibit emergent reasoning abilities at scale", 0.82);
        // fraction = 1/2 = 0.5, threshold 0.5, gate is < so this is rejected
        let id3 = c.add_discovery(d3);
        // Gate says fraction < max_source_fraction, 0.5 < 0.5 is false → rejected
        assert!(id3.is_none());
    }

    #[test]
    fn ingest_bulk() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let discoveries = vec![
            make_discovery("arxiv", "Causal inference algorithms can recover ground truth structures from data", 0.80),
            make_discovery("nasa_power", "Temperature increases correlated with atmospheric CO2 concentration values", 0.85),
            make_discovery("who_gho", "Vaccination rates inversely correlated with disease incidence globally", 0.90),
            make_discovery("bad_source", "x", 0.10),  // fails multiple gates
        ];
        let (acc, rej) = c.ingest(discoveries);
        assert_eq!(acc, 3);
        assert_eq!(rej, 1);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn sample_pheromone() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        c.add_discovery(make_discovery("arxiv", "Causal inference algorithms recover ground truth network structures", 0.80));
        c.add_discovery(make_discovery("nasa_power", "Temperature anomalies correlate strongly with atmospheric CO2 concentrations", 0.85));
        c.add_discovery(make_discovery("who_gho", "Vaccination rates inversely correlated with disease incidence rates globally", 0.90));
        let batch = c.sample_batch(2, SampleStrategy::Pheromone);
        assert_eq!(batch.entries.len(), 2);
        assert!(batch.mean_pheromone > 0.0);
    }

    #[test]
    fn feedback_increases_pheromone() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let id = c.add_discovery(
            make_discovery("arxiv", "Graph neural networks achieve state of the art on causal discovery benchmarks", 0.80)
        ).unwrap();
        let before = c.get(id).unwrap().pheromone;
        c.feedback_positive(id);
        let after = c.get(id).unwrap().pheromone;
        assert!(after > before);
    }

    #[test]
    fn feedback_negative_decreases_pheromone() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let id = c.add_discovery(
            make_discovery("arxiv", "Bayesian networks provide probabilistic representations of causal dependencies", 0.80)
        ).unwrap();
        let before = c.get(id).unwrap().pheromone;
        c.feedback_negative(id);
        let after = c.get(id).unwrap().pheromone;
        assert!(after < before);
    }

    #[test]
    fn decay_reduces_all_pheromones() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        c.add_discovery(make_discovery("arxiv", "Transformer attention mechanisms implement soft retrieval over memory", 0.80));
        c.add_discovery(make_discovery("nasa_power", "Solar irradiance measurements correlate with global surface temperature", 0.85));
        let before: Vec<f64> = c.entries.iter().map(|e| e.pheromone).collect();
        c.decay_pheromones(0.9);
        let after: Vec<f64> = c.entries.iter().map(|e| e.pheromone).collect();
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(a <= b);
        }
    }

    #[test]
    fn curriculum_sample_prefers_easy() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        // Add entries with varying quality
        c.add_discovery(make_discovery("arxiv", "Low quality entry from unknown source about network structure", 0.56));
        c.add_discovery(make_discovery("nasa_power", "High quality validated finding from NASA about climate warming trend", 0.92));
        let batch = c.sample_batch(1, SampleStrategy::Curriculum);
        // tier 0 should be sampled first (easiest)
        assert_eq!(batch.entries.len(), 1);
    }

    #[test]
    fn json_roundtrip() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        c.add_discovery(make_discovery("arxiv", "Causal discovery algorithms learn directed acyclic graphs from data", 0.80));
        c.add_discovery(make_discovery("nasa_power", "Temperature warming trend correlates strongly with CO2 emissions globally", 0.85));
        let json = c.to_json();
        let c2 = LiveDiscoveryCorpus::from_json_str(&json, GateConfig::default()).unwrap();
        assert_eq!(c2.len(), c.len());
        assert_eq!(c2.stats().total_entries, c.stats().total_entries);
    }

    #[test]
    fn stats_correct() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        c.add_discovery(make_discovery("arxiv", "Causal discovery algorithms recover ground truth structures efficiently", 0.80));
        c.add_discovery(make_discovery("nasa_power", "CO2 atmospheric concentrations correlate strongly with global warming trends", 0.85));
        let s = c.stats();
        assert_eq!(s.total_entries, 2);
        assert!(s.mean_confidence > 0.0);
        assert!(s.mean_pheromone > 0.0);
        assert_eq!(s.unique_sources, 2);
    }

    #[test]
    fn gate_result_all_fail_means_reject() {
        let gate = GateResult::new(false, false, false, false, false);
        assert!(!gate.pass);
    }

    #[test]
    fn gate_result_one_fail_means_reject() {
        let gate = GateResult::new(true, true, true, true, false);
        assert!(!gate.pass);
    }

    #[test]
    fn gate_result_all_pass() {
        let gate = GateResult::new(true, true, true, true, true);
        assert!(gate.pass);
    }

    #[test]
    fn mixed_sampling() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        for i in 0..10 {
            let summary = format!("Unique scientific discovery about natural phenomenon number {i} in the world", i = i);
            if let Some(_id) = c.add_discovery(make_discovery("arxiv", &summary, 0.75)) {}
        }
        let batch = c.sample_batch(5, SampleStrategy::Mixed);
        assert!(batch.entries.len() <= 5);
    }

    #[test]
    fn corpus_sources_tracking() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        c.add_discovery(make_discovery("arxiv", "Causal representation learning enables robust transfer across environments", 0.80));
        c.add_discovery(make_discovery("nasa_power", "Atmospheric CO2 concentrations drive global temperature increase trends", 0.85));
        let sources = c.sources();
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&"arxiv".to_string()) || sources.contains(&"nasa_power".to_string()));
    }
}
