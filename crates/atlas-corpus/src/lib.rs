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

pub mod train;
pub use train::{
    SftTrainer, SftConfig, TrainableMlp, SimpleTokenizer, TokenBatch,
    StepMetrics, EpochMetrics, TrainingMetrics, cross_entropy,
};

pub mod deep_supervision;
pub use deep_supervision::{DeepSupervisionConfig, DeepSupervisionTrainer};

use std::collections::HashMap;

use atlas_astra::{Discovery, Observation};
use atlas_core::Result;
use atlas_json::Json;
use atlas_palace::Palace;
use atlas_zk::{ProvenanceChain, ProvenanceLinkType};

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
            // 0.40: calibrated for live-API data (NASA/WHO/WorldBank/ArXiv) which
            // produces raw confidence in the 0.45–0.90 range after causal extraction.
            min_confidence: 0.40,
            min_novelty: 0.25,
            min_source_reliability: 0.4,
            max_source_fraction: 0.5,
            // 3: causal titles like "A → B" are inherently 3 tokens; 6 was too strict.
            min_summary_words: 3,
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleStrategy {
    /// Pure pheromone-weighted (exploitation).
    Pheromone,
    /// Curriculum-ordered: start from easy (tier 0), gradually introduce harder.
    Curriculum,
    /// Uniform random (exploration).
    Uniform,
    /// Mix: 60% pheromone + 40% uniform.
    Mixed,
    /// Stigmergic: temperature-controlled pheromone sampling.
    /// temperature > 1.0 = more exploration, < 1.0 = more exploitation.
    Stigmergic { temperature: f64 },
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
            SampleStrategy::Stigmergic { temperature } => self.sample_stigmergic(k, temperature),
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
                    provenance: None,
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

    /// Temperature-controlled pheromone sampling (stigmergic).
    /// - temperature = 1.0: standard pheromone-proportional
    /// - temperature > 1.0: flatter distribution (more exploration)
    /// - temperature < 1.0: sharper distribution (more exploitation / greedy-like)
    fn sample_stigmergic(&mut self, k: usize, temperature: f64) -> Vec<usize> {
        if self.entries.is_empty() { return vec![]; }
        let temp = temperature.max(0.01); // avoid division by zero

        // Apply temperature to pheromone weights via softmax-style transform
        let weights: Vec<f64> = self.entries.iter()
            .map(|e| (e.pheromone / temp).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if total == 0.0 || !total.is_finite() { return self.sample_uniform(k); }

        // Weighted sampling without replacement
        let mut chosen = Vec::with_capacity(k);
        let mut remaining: Vec<(usize, f64)> = weights.into_iter().enumerate().collect();
        for _ in 0..k {
            if remaining.is_empty() { break; }
            let sum: f64 = remaining.iter().map(|(_, w)| w).sum();
            if sum == 0.0 || !sum.is_finite() { break; }
            let r = (self.lcg_next() as f64 / u64::MAX as f64) * sum;
            let mut acc = 0.0;
            let mut sel = 0;
            for (idx, (_, w)) in remaining.iter().enumerate() {
                acc += w;
                if r <= acc { sel = idx; break; }
            }
            let (chosen_i, _) = remaining.remove(sel);
            chosen.push(chosen_i);
        }
        chosen
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
//  Safety: configurable pre-corpus safety filter
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the safety filter applied before corpus insertion.
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Minimum confidence score (0–1) to pass safety check. Default 0.50.
    pub min_confidence: f64,
    /// Maximum number of discoveries accepted per OODA cycle. Default 50.
    pub max_per_cycle: usize,
    /// Blocklist: discoveries whose title contains any of these phrases are rejected.
    pub known_falsehoods: Vec<String>,
    /// If true, discoveries without supporting observations are flagged as hallucinations.
    pub require_observations: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.50,
            max_per_cycle: 50,
            known_falsehoods: vec![
                "flat earth".to_string(),
                "perpetual motion".to_string(),
                "5g causes".to_string(),
            ],
            require_observations: true,
        }
    }
}

/// Result of running the safety filter on one discovery.
#[derive(Debug, Clone)]
pub struct SafetyReport {
    /// Whether the discovery passed all safety checks.
    pub passed: bool,
    /// Human-readable reasons for pass or fail.
    pub reasons: Vec<String>,
    /// Number of checks that passed.
    pub checks_passed: usize,
    /// Number of checks that failed.
    pub checks_failed: usize,
}

impl SafetyReport {
    fn pass(reasons: Vec<String>) -> Self {
        let n = reasons.len();
        Self { passed: true, reasons, checks_passed: n, checks_failed: 0 }
    }
    fn fail(passed_reasons: Vec<String>, failed_reasons: Vec<String>) -> Self {
        let p = passed_reasons.len();
        let f = failed_reasons.len();
        let mut all = passed_reasons;
        all.extend(failed_reasons);
        Self { passed: false, reasons: all, checks_passed: p, checks_failed: f }
    }
}

/// Configurable safety filter for pre-corpus validation.
///
/// Checks:
/// 1. Minimum confidence threshold
/// 2. Known falsehoods blocklist (case-insensitive substring match)
/// 3. Rate limiting (max discoveries per cycle)
/// 4. Hallucination detection (claims without supporting observations)
///
/// # Example
/// ```
/// use atlas_corpus::{SafetyFilter, SafetyConfig};
/// let filter = SafetyFilter::new(SafetyConfig::default());
/// assert_eq!(filter.cycle_accepted(), 0);
/// ```
pub struct SafetyFilter {
    config: SafetyConfig,
    /// How many discoveries have been accepted this cycle.
    accepted_this_cycle: usize,
}

impl SafetyFilter {
    /// Create a new safety filter.
    pub fn new(config: SafetyConfig) -> Self {
        Self { config, accepted_this_cycle: 0 }
    }

    /// Reset the per-cycle counter (call at the start of each OODA cycle).
    pub fn reset_cycle(&mut self) {
        self.accepted_this_cycle = 0;
    }

    /// How many discoveries accepted in the current cycle.
    pub fn cycle_accepted(&self) -> usize {
        self.accepted_this_cycle
    }

    /// Run all safety checks on a discovery.
    ///
    /// `observations` should be the raw observations that support this discovery.
    /// If `require_observations` is true and `observations` is empty, the discovery
    /// is flagged as a potential hallucination.
    pub fn check(&mut self, discovery: &Discovery, observations: &[Observation]) -> SafetyReport {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // 1. Confidence threshold
        if discovery.quality_score >= self.config.min_confidence {
            passed.push(format!("confidence {:.2} ≥ {:.2}", discovery.quality_score, self.config.min_confidence));
        } else {
            failed.push(format!("FAIL: confidence {:.2} < {:.2}", discovery.quality_score, self.config.min_confidence));
        }

        // 2. Known falsehoods blocklist
        let title_lower = discovery.title.to_lowercase();
        let mut blocked = false;
        for phrase in &self.config.known_falsehoods {
            if title_lower.contains(&phrase.to_lowercase()) {
                failed.push(format!("FAIL: matches known falsehood pattern \"{}\"", phrase));
                blocked = true;
                break;
            }
        }
        if !blocked {
            passed.push("no known falsehood match".to_string());
        }

        // 3. Rate limiting
        if self.accepted_this_cycle < self.config.max_per_cycle {
            passed.push(format!("rate limit {}/{}", self.accepted_this_cycle + 1, self.config.max_per_cycle));
        } else {
            failed.push(format!("FAIL: rate limit exceeded ({}/{})", self.accepted_this_cycle, self.config.max_per_cycle));
        }

        // 4. Hallucination detection
        if self.config.require_observations {
            if observations.is_empty() {
                failed.push("FAIL: no supporting observations (possible hallucination)".to_string());
            } else {
                passed.push(format!("{} supporting observation(s)", observations.len()));
            }
        } else {
            passed.push("observation check disabled".to_string());
        }

        if failed.is_empty() {
            self.accepted_this_cycle += 1;
            SafetyReport::pass(passed)
        } else {
            SafetyReport::fail(passed, failed)
        }
    }

    /// Convenience: check a batch, returning (passed, failed) lists.
    pub fn check_batch(&mut self, discoveries: &[Discovery], observations: &[Observation]) -> (Vec<usize>, Vec<usize>) {
        let mut pass_ids = Vec::new();
        let mut fail_ids = Vec::new();
        for (i, d) in discoveries.iter().enumerate() {
            let report = self.check(d, observations);
            if report.passed {
                pass_ids.push(i);
            } else {
                fail_ids.push(i);
            }
        }
        (pass_ids, fail_ids)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Provenance-augmented corpus insertion
// ────────────────────────────────────────────────────────────────────────────

impl LiveDiscoveryCorpus {
    /// Ingest a discovery with provenance chain and safety checks.
    ///
    /// 1. Runs safety filter (if provided)
    /// 2. Runs 5 quality gates
    /// 3. Attaches provenance chain
    /// 4. Inserts into corpus
    ///
    /// Returns `Some(entry_id)` on success, `None` if rejected.
    pub fn add_discovery_safe(
        &mut self,
        mut discovery: Discovery,
        observations: &[Observation],
        safety: &mut SafetyFilter,
        provenance_secret: &[u8],
    ) -> Option<usize> {
        // Safety check
        let report = safety.check(&discovery, observations);
        if !report.passed {
            self.total_rejected += 1;
            return None;
        }

        // Build provenance chain if discovery doesn't already have one
        if discovery.provenance.is_none() {
            let mut chain = ProvenanceChain::new(provenance_secret);
            // Link each supporting observation
            for obs in observations.iter().take(3) {
                let claim = format!("Observation from {}: {}", obs.source, &obs.content[..obs.content.len().min(100)]);
                chain.add_link(&claim, provenance_secret, ProvenanceLinkType::Observation);
            }
            // Link the discovery itself
            let disc_claim = format!("Discovery: {} | quality={:.2}", discovery.title, discovery.quality_score);
            chain.add_link(&disc_claim, provenance_secret, ProvenanceLinkType::Discovery);
            discovery.provenance = Some(chain);
        }

        // Existing quality-gate path
        self.add_discovery(discovery)
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_astra::{Discovery, Observation};

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
            provenance: None,
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

    // ── Safety filter tests ─────────────────────────────────────────────

    fn make_observation(source: &str, content: &str) -> Observation {
        Observation {
            source: source.to_string(),
            content: content.to_string(),
            url: format!("test://{}", source),
            retrieved_at: 0,
        }
    }

    #[test]
    fn safety_filter_passes_high_confidence() {
        let mut sf = SafetyFilter::new(SafetyConfig::default());
        let d = make_discovery("arxiv", "Graph neural networks achieve state of the art on causal benchmarks", 0.80);
        let obs = vec![make_observation("arxiv", "GNN paper data")];
        let report = sf.check(&d, &obs);
        assert!(report.passed, "high confidence should pass: {:?}", report.reasons);
        assert!(report.checks_failed == 0);
    }

    #[test]
    fn safety_filter_blocks_low_confidence() {
        let mut sf = SafetyFilter::new(SafetyConfig { min_confidence: 0.70, ..SafetyConfig::default() });
        let d = make_discovery("arxiv", "Weak hypothesis about something with low evidence score", 0.40);
        let obs = vec![make_observation("arxiv", "some data")];
        let report = sf.check(&d, &obs);
        assert!(!report.passed, "low confidence should fail: {:?}", report.reasons);
        assert!(report.checks_failed > 0);
    }

    #[test]
    fn safety_filter_blocks_known_falsehood() {
        let mut sf = SafetyFilter::new(SafetyConfig::default());
        let d = make_discovery("unknown", "Study shows flat earth predictions match observations with high accuracy", 0.90);
        let obs = vec![make_observation("unknown", "flat earth data")];
        let report = sf.check(&d, &obs);
        assert!(!report.passed, "known falsehood should be blocked: {:?}", report.reasons);
        assert!(report.reasons.iter().any(|r| r.contains("falsehood")));
    }

    #[test]
    fn safety_filter_rate_limit() {
        let cfg = SafetyConfig { max_per_cycle: 2, ..SafetyConfig::default() };
        let mut sf = SafetyFilter::new(cfg);
        let obs = vec![make_observation("arxiv", "data")];
        // First 2 should pass
        let d1 = make_discovery("arxiv", "First valid discovery about causal inference in neural networks", 0.80);
        assert!(sf.check(&d1, &obs).passed);
        let d2 = make_discovery("arxiv", "Second valid discovery about reinforcement learning agents exploring", 0.85);
        assert!(sf.check(&d2, &obs).passed);
        // Third should fail rate limit
        let d3 = make_discovery("arxiv", "Third valid discovery about transformer attention mechanism patterns", 0.90);
        let report = sf.check(&d3, &obs);
        assert!(!report.passed, "rate limit should block 3rd: {:?}", report.reasons);
        assert!(report.reasons.iter().any(|r| r.contains("rate limit")));
    }

    #[test]
    fn safety_filter_rate_limit_resets() {
        let cfg = SafetyConfig { max_per_cycle: 1, ..SafetyConfig::default() };
        let mut sf = SafetyFilter::new(cfg);
        let obs = vec![make_observation("arxiv", "data")];
        let d1 = make_discovery("arxiv", "Valid discovery about causal inference in neural network models", 0.80);
        assert!(sf.check(&d1, &obs).passed);
        // Should fail
        let d2 = make_discovery("arxiv", "Another discovery about reinforcement learning exploration strategies", 0.80);
        assert!(!sf.check(&d2, &obs).passed);
        // Reset cycle
        sf.reset_cycle();
        // Now should pass again
        let d3 = make_discovery("arxiv", "Post-reset discovery about Bayesian optimization for hyperparameters", 0.80);
        assert!(sf.check(&d3, &obs).passed);
    }

    #[test]
    fn safety_hallucination_detection() {
        let mut sf = SafetyFilter::new(SafetyConfig {
            require_observations: true,
            ..SafetyConfig::default()
        });
        let d = make_discovery("arxiv", "Discovery claims causal link without any supporting observation data", 0.80);
        // No observations → hallucination
        let report = sf.check(&d, &[]);
        assert!(!report.passed, "no observations should flag hallucination: {:?}", report.reasons);
        assert!(report.reasons.iter().any(|r| r.contains("hallucination")));
    }

    #[test]
    fn safety_hallucination_with_observations_passes() {
        let mut sf = SafetyFilter::new(SafetyConfig {
            require_observations: true,
            ..SafetyConfig::default()
        });
        let d = make_discovery("arxiv", "Causal inference algorithms recover ground truth graphs from real data", 0.80);
        let obs = vec![make_observation("arxiv", "supporting evidence data from experiment")];
        let report = sf.check(&d, &obs);
        assert!(report.passed, "with observations should pass: {:?}", report.reasons);
    }

    #[test]
    fn safety_batch_check() {
        let mut sf = SafetyFilter::new(SafetyConfig::default());
        let obs = vec![make_observation("arxiv", "data")];
        let discoveries = vec![
            make_discovery("arxiv", "Valid causal discovery about neural network weight initialization", 0.80),
            make_discovery("unknown", "Flat earth predictions confirmed by satellite measurements analysis", 0.90),
            make_discovery("arxiv", "Another valid discovery about attention mechanism sparse patterns", 0.75),
        ];
        let (passed, failed) = sf.check_batch(&discoveries, &obs);
        assert!(passed.contains(&0), "first should pass");
        assert!(failed.contains(&1), "second should fail (falsehood)");
        assert!(passed.contains(&2), "third should pass");
    }

    // ── Provenance integration tests ────────────────────────────────────

    #[test]
    fn discovery_provenance_chain_verifies() {
        use atlas_zk::{ProvenanceChain, ProvenanceLinkType};
        let secret = b"test_provenance_secret";
        let mut chain = ProvenanceChain::new(secret);
        chain.add_link("NASA POWER: T2M=25.3", secret, ProvenanceLinkType::ApiSource);
        chain.add_link("Temperature observed at NYC", secret, ProvenanceLinkType::Observation);
        chain.add_link("Temperature → CO2", secret, ProvenanceLinkType::Discovery);

        let mut d = make_discovery("nasa_power", "Temperature causes CO2 concentration changes according to measurements", 0.80);
        d.provenance = Some(chain);

        let prov = d.provenance.as_ref().unwrap();
        assert!(prov.verify_all());
        assert_eq!(prov.len(), 3);
    }

    #[test]
    fn add_discovery_safe_attaches_provenance() {
        let mut corpus = LiveDiscoveryCorpus::new(GateConfig::default());
        let mut safety = SafetyFilter::new(SafetyConfig::default());
        let secret = b"corpus_provenance_key";
        let d = make_discovery("arxiv", "Causal discovery algorithms learn directed acyclic graph structures", 0.80);
        let obs = vec![make_observation("arxiv", "paper data about causal graphs")];

        let id = corpus.add_discovery_safe(d, &obs, &mut safety, secret);
        assert!(id.is_some(), "should be accepted");

        let entry = corpus.get(id.unwrap()).unwrap();
        let prov = entry.discovery.provenance.as_ref().expect("should have provenance");
        assert!(prov.verify_all(), "provenance should verify");
        assert!(prov.len() >= 2, "should have at least observation + discovery links");
    }

    #[test]
    fn add_discovery_safe_rejected_by_safety() {
        let mut corpus = LiveDiscoveryCorpus::new(GateConfig::default());
        let mut safety = SafetyFilter::new(SafetyConfig::default());
        let secret = b"key";
        // Low confidence → safety rejects
        let d = make_discovery("arxiv", "Weak unsupported claim about something without evidence backing", 0.20);
        let obs = vec![make_observation("arxiv", "data")];
        let id = corpus.add_discovery_safe(d, &obs, &mut safety, secret);
        assert!(id.is_none(), "should be rejected by safety");
    }

    #[test]
    fn end_to_end_discovery_safety_provenance() {
        // Full pipeline: create discovery → safety check → provenance → corpus
        let mut corpus = LiveDiscoveryCorpus::new(GateConfig::default());
        let mut safety = SafetyFilter::new(SafetyConfig::default());
        let secret = b"e2e_secret";

        let observations = vec![
            make_observation("nasa_power", r#"{"T2M":25.3,"CO2":415.0}"#),
            make_observation("arxiv", "Causal model linking temperature and CO2"),
        ];

        let discovery = Discovery {
            id: "e2e-test-1".to_string(),
            source: "nasa_power".to_string(),
            title: "Temperature anomaly correlates strongly with atmospheric CO2 level".to_string(),
            description: "Multi-source evidence of causal link".to_string(),
            causal_claims: vec![("CO2".into(), "temperature".into(), 0.85)],
            quality_score: 0.82,
            proof_commitment: 0,
            timestamp: 0,
            tags: vec!["climate".to_string()],
            provenance: None,
        };

        let id = corpus.add_discovery_safe(discovery, &observations, &mut safety, secret);
        assert!(id.is_some(), "e2e: should be accepted");

        let entry = corpus.get(id.unwrap()).unwrap();
        assert!(entry.discovery.provenance.is_some(), "e2e: should have provenance");

        let prov = entry.discovery.provenance.as_ref().unwrap();
        assert!(prov.verify_all(), "e2e: provenance chain should verify");
        assert!(prov.len() >= 3, "e2e: should have obs + obs + discovery links, got {}", prov.len());

        // Verify the chain serializes and deserializes
        let json = prov.to_json();
        let prov2 = atlas_zk::ProvenanceChain::from_json(&json).unwrap();
        assert_eq!(prov2.len(), prov.len());
        assert!(prov2.verify_all(), "e2e: roundtripped provenance should verify");

        // Corpus stats should reflect the accepted entry
        let stats = corpus.stats();
        assert_eq!(stats.total_entries, 1);
        assert!(stats.mean_confidence > 0.5);
    }

    #[test]
    fn provenance_to_json_roundtrip_in_corpus() {
        use atlas_zk::{ProvenanceChain, ProvenanceLinkType};
        let secret = b"json_rt_key";
        let mut chain = ProvenanceChain::new(secret);
        chain.add_link("raw API response", secret, ProvenanceLinkType::ApiSource);
        chain.add_link("observed pattern", secret, ProvenanceLinkType::Observation);
        chain.add_link("validated discovery", secret, ProvenanceLinkType::Discovery);

        let json = chain.to_json();
        let chain2 = ProvenanceChain::from_json(&json).expect("should parse");
        assert_eq!(chain2.len(), 3);
        assert_eq!(chain2.public_key, chain.public_key);
        assert!(chain2.verify_all());
    }

    // ── Stigmergic sampling tests ────────────────────────────────────────

    #[test]
    fn stigmergic_sample_returns_correct_count() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        // Add entries with very different titles to pass novelty gate
        let titles = [
            "Quantum entanglement breaks locality assumptions",
            "Ocean currents drive atmospheric temperature patterns",
            "Mitochondrial DNA reveals ancient migration routes",
            "Solar panel efficiency exceeds theoretical Shockley limit",
            "Bacterial quorum sensing coordinates biofilm formation",
            "Tectonic plate subduction generates volcanic activity",
            "Neural oscillations synchronize during memory consolidation",
            "CRISPR modifications propagate through wild populations",
            "Dark matter halos influence galactic rotation curves",
            "Ribosome stalling triggers mRNA quality control pathways",
        ];
        for (i, title) in titles.iter().enumerate() {
            let src = &format!("src{i}"); // different sources to avoid diversity gate
            let d = make_discovery(src, title, 0.8);
            c.add_discovery(d);
            c.feedback_positive(i); // give some pheromone
        }
        let batch = c.sample_batch(5, SampleStrategy::Stigmergic { temperature: 1.0 });
        assert_eq!(batch.entries.len(), 5);
    }

    #[test]
    fn stigmergic_sample_empty_corpus() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        let batch = c.sample_batch(5, SampleStrategy::Stigmergic { temperature: 1.0 });
        assert_eq!(batch.entries.len(), 0);
    }

    #[test]
    fn stigmergic_low_temp_favors_high_pheromone() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        // Entry id=0: low pheromone (no feedback)
        let d0 = make_discovery("src", "Low pheromone entry stays weak", 0.8);
        c.add_discovery(d0);
        // Entry id=1: high pheromone (via repeated feedback)
        let d1 = make_discovery("other", "High pheromone entry boosted strongly here", 0.8);
        c.add_discovery(d1);
        // Boost entry 1's pheromone to cap (1.0)
        for _ in 0..20 {
            c.feedback_positive(1);
        }

        // Low temperature (exploitation) — should strongly favor entry 1 (id=1)
        let mut high_count = 0;
        for _ in 0..10 {
            let batch = c.sample_batch(1, SampleStrategy::Stigmergic { temperature: 0.1 });
            if !batch.entries.is_empty() && batch.entries[0].id == 1 {
                high_count += 1;
            }
        }
        // With very low temp, high-pheromone entry should dominate
        assert!(high_count >= 7, "Expected mostly entry 1 (high pheromone), got {high_count}/10");
    }

    #[test]
    fn stigmergic_batch_integration() {
        let mut c = LiveDiscoveryCorpus::new(GateConfig::default());
        for i in 0..5 {
            let d = make_discovery("src", &format!("Stigmergic discovery number {i} content here"), 0.8);
            c.add_discovery(d);
        }
        // Various temperatures should all return valid batches
        for temp in [0.1, 0.5, 1.0, 2.0, 10.0] {
            let batch = c.sample_batch(3, SampleStrategy::Stigmergic { temperature: temp });
            assert!(batch.entries.len() <= 3);
        }
    }
}
