//! InferEngine — high-level inference facade with optional StigmergicHook.
//!
//! Wraps [`OlmoModel`] and provides a clean API that:
//! - Routes to the un-hooked `generate_with_sampling` path (zero overhead) when
//!   no hook is attached.
//! - Routes to `generate_with_sampling_hooked` when a hook is present, returning
//!   [`PheromoneDeposit`] data per generated token.
//!
//! # Example
//!
//! ```rust
//! use atlas_infer::{InferEngine, StigmergicHook};
//! use atlas_model::{OlmoModel, ModelConfig, SamplingConfig};
//! use std::sync::Arc;
//!
//! struct LogHook;
//! impl StigmergicHook for LogHook {
//!     fn on_layer(&self, layer_idx: usize, hidden: &[f32]) -> Option<f32> {
//!         if layer_idx >= 28 {
//!             let mag = hidden.iter().map(|v| v.abs()).sum::<f32>()
//!                       / hidden.len().max(1) as f32;
//!             Some(mag * 0.01)
//!         } else {
//!             None
//!         }
//!     }
//! }
//!
//! let cfg = ModelConfig::tiny();
//! let model = OlmoModel::new(cfg);
//! let mut engine = InferEngine::new(model).with_hook(Arc::new(LogHook));
//! let cfg_s = SamplingConfig::default();
//! let (tokens, deposits) = engine.generate(&[0u32, 1, 2], 5, &cfg_s);
//! assert!(!tokens.is_empty());
//! ```

use std::sync::Arc;

use atlas_model::{GenEvent, OlmoModel, SamplingConfig};

use crate::hook::{ArcHook, StigmergicHook};

// ── PheromoneDeposit ──────────────────────────────────────────────────────

/// Aggregated pheromone deposit for one generated token.
///
/// The engine accumulates per-layer hook signals and returns one
/// `PheromoneDeposit` per token so callers can drive GraphPalace in
/// a single deposit call per token rather than 32 calls (one per layer).
#[derive(Debug, Clone)]
pub struct PheromoneDeposit {
    /// Token that was generated.
    pub token_id: u32,
    /// Sum of all `Some(delta)` values returned by the hook during this token's
    /// layer forward passes.  Zero means no layer deposited.
    pub total_delta: f32,
    /// Number of layers that returned `Some(delta)`.
    pub layers_fired: u32,
}

// ── InferEngine ───────────────────────────────────────────────────────────

/// High-level inference engine with optional StigmergicHook.
///
/// Call [`InferEngine::new`] to create an engine from a loaded [`OlmoModel`],
/// then optionally attach a hook with [`with_hook`] / [`set_hook`].
///
/// [`with_hook`]: InferEngine::with_hook
/// [`set_hook`]: InferEngine::set_hook
pub struct InferEngine {
    /// Underlying transformer model (public for direct access when needed).
    pub model: OlmoModel,
    /// Optional pheromone hook.  `None` = zero overhead fast path.
    hook: Option<ArcHook>,
}

impl InferEngine {
    /// Create a new engine wrapping `model`.  No hook is attached by default.
    pub fn new(model: OlmoModel) -> Self {
        Self { model, hook: None }
    }

    /// Attach a [`StigmergicHook`].  Returns `self` for builder-style chaining.
    ///
    /// ```rust
    /// # use atlas_infer::{InferEngine, StigmergicHook};
    /// # use atlas_model::{OlmoModel, ModelConfig};
    /// # use std::sync::Arc;
    /// # struct NoopHook; impl StigmergicHook for NoopHook { fn on_layer(&self,_:usize,_:&[f32])->Option<f32>{None} }
    /// let engine = InferEngine::new(OlmoModel::new(ModelConfig::tiny()))
    ///     .with_hook(Arc::new(NoopHook));
    /// assert!(engine.has_hook());
    /// ```
    pub fn with_hook(mut self, hook: Arc<dyn StigmergicHook>) -> Self {
        self.hook = Some(hook);
        self
    }

    /// Replace (or clear) the hook at runtime.
    ///
    /// Passing `None` removes the hook and restores zero-overhead performance.
    pub fn set_hook(&mut self, hook: Option<Arc<dyn StigmergicHook>>) {
        self.hook = hook;
    }

    /// Returns `true` if a hook is currently attached.
    pub fn has_hook(&self) -> bool { self.hook.is_some() }

    /// Reset the KV cache and position counter (call between independent prompts).
    pub fn reset(&mut self) { self.model.reset(); }

    // ── Public: generate ──────────────────────────────────────────────────

    /// Autoregressive generation.
    ///
    /// Given `prompt` tokens, generates up to `max_new` more tokens using the
    /// `cfg` sampling configuration.
    ///
    /// # Returns
    /// `(generated_tokens, deposits)` where:
    /// - `generated_tokens` — the newly generated token ids (not including prompt).
    /// - `deposits` — one [`PheromoneDeposit`] per generated token, **only
    ///   populated when a hook is attached**.  Empty when `has_hook() == false`.
    ///
    /// # Performance
    ///
    /// When `has_hook() == false`, delegates entirely to `OlmoModel::generate_with_sampling`
    /// which uses the GPU-resident forward pass (2 PCIe transfers/token).
    ///
    /// When `has_hook() == true`, uses the CPU forward path so that the hook
    /// can inspect hidden states without extra VRAM→RAM copies between layers.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
    ) -> (Vec<u32>, Vec<PheromoneDeposit>) {
        match &self.hook {
            None => {
                // Zero-overhead: use OlmoModel's optimised generate path (GPU-resident).
                let tokens = self.model.generate_with_sampling(prompt, max_new, cfg);
                (tokens, Vec::new())
            }
            Some(hook) => {
                let hook = Arc::clone(hook);
                let (tokens, raw) = self.model.generate_with_sampling_hooked(
                    prompt,
                    max_new,
                    cfg,
                    |layer_idx, hidden| hook.on_layer(layer_idx, hidden),
                );
                let deposits = raw
                    .into_iter()
                    .map(|(token_id, total_delta, layers_fired)| PheromoneDeposit {
                        token_id,
                        total_delta,
                        layers_fired,
                    })
                    .collect();
                (tokens, deposits)
            }
        }
    }

    /// Streaming generation — calls `on_token(token_id, deposit)` for each token.
    ///
    /// Returns the number of tokens generated.  `deposit` is `None` when no hook
    /// is attached.
    ///
    /// Return `false` from the callback to stop generation early.
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
        mut on_token: F,
    ) -> usize
    where
        F: FnMut(u32, Option<PheromoneDeposit>) -> bool,
    {
        self.generate_streaming_events(prompt, max_new, cfg, |ev, deposit| match ev {
            GenEvent::Token(tok) => on_token(tok, deposit),
            GenEvent::Prefill { .. } => true,
        })
    }

    /// Streaming generation with progress events — like [`generate_streaming`]
    /// but the callback also receives [`GenEvent::Prefill`] progress during
    /// prompt processing (no-hook path only), so API layers can emit SSE
    /// keep-alives during long prefills.
    ///
    /// The hooked path generates in bulk and emits only `Token` events.
    /// Return `false` from the callback to stop generation early.
    ///
    /// [`generate_streaming`]: Self::generate_streaming
    pub fn generate_streaming_events<F>(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
        mut on_event: F,
    ) -> usize
    where
        F: FnMut(GenEvent, Option<PheromoneDeposit>) -> bool,
    {
        match &self.hook {
            None => {
                // Zero-overhead streaming path with prefill progress.
                let mut count = 0usize;
                self.model.generate_streaming_events(prompt, max_new, cfg, |ev| {
                    if matches!(ev, GenEvent::Token(_)) { count += 1; }
                    on_event(ev, None)
                });
                count
            }
            Some(hook) => {
                let hook = Arc::clone(hook);
                // Hooked generate then call on_token for each; could be
                // improved to true streaming in a future iteration.
                let (tokens, raw) = self.model.generate_with_sampling_hooked(
                    prompt,
                    max_new,
                    cfg,
                    |layer_idx, hidden| hook.on_layer(layer_idx, hidden),
                );
                let mut count = 0usize;
                for (tok, (_, total_delta, layers_fired)) in
                    tokens.iter().zip(raw.into_iter())
                {
                    let deposit = Some(PheromoneDeposit {
                        token_id: *tok,
                        total_delta,
                        layers_fired,
                    });
                    let keep_going = on_event(GenEvent::Token(*tok), deposit);
                    count += 1;
                    if !keep_going { break; }
                }
                count
            }
        }
    }

    /// Continue generation from the CURRENT KV state without resetting:
    /// force-feed `forced` tokens, then decode up to `max_new` more.
    ///
    /// Used by the API layer's think-budget answer reserve: when a Think
    /// model exhausts `max_tokens` inside its `<think>` block, the handler
    /// injects a `</think>` close and lets the model produce the visible
    /// answer from the intact KV cache — no re-prefill.
    ///
    /// No-hook path only (the API server attaches no hook); with a hook
    /// configured this is a no-op returning 0.
    pub fn continue_streaming_events<F>(
        &mut self,
        forced: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
        mut on_event: F,
    ) -> usize
    where
        F: FnMut(GenEvent, Option<PheromoneDeposit>) -> bool,
    {
        if self.hook.is_some() {
            return 0;
        }
        let mut count = 0usize;
        self.model.continue_streaming_events(forced, max_new, cfg, |ev| {
            if matches!(ev, GenEvent::Token(_)) { count += 1; }
            on_event(ev, None)
        });
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_model::ModelConfig;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn tiny_engine() -> InferEngine {
        let mut m = OlmoModel::new(ModelConfig::tiny());
        // Legacy per-token prefill cadence — the prefill-event test below
        // asserts one event per prompt token (#22 chunked prefill would
        // otherwise coalesce them into one event per chunk).
        m.set_prefill_chunk_tokens(1);
        InferEngine::new(m)
    }

    #[test]
    fn no_hook_streaming_events_include_prefill() {
        let mut e = tiny_engine();
        let mut prefills = 0usize;
        let mut toks = 0usize;
        let n = e.generate_streaming_events(
            &[0u32, 1, 2], 4, &SamplingConfig::default(),
            |ev, deposit| {
                assert!(deposit.is_none(), "no hook → no deposits");
                match ev {
                    GenEvent::Prefill { .. } => prefills += 1,
                    GenEvent::Token(_) => toks += 1,
                }
                true
            });
        assert_eq!(prefills, 3, "one prefill event per prompt token");
        assert_eq!(n, toks, "returned count must match token events");
    }

    /// Answer-reserve continuation (think-budget recovery): after a capped
    /// generate, `continue_streaming_events` must resume greedily and emit
    /// exactly the tokens the uninterrupted run would have produced.
    #[test]
    fn no_hook_continue_streaming_events_resumes_exactly() {
        let greedy = SamplingConfig { temperature: 0.0, ..Default::default() };
        let prompt = [0u32, 1, 2];

        // Reference: uninterrupted 8-token generation.
        let mut e1 = tiny_engine();
        let mut full: Vec<u32> = Vec::new();
        e1.generate_streaming_events(&prompt, 8, &greedy, |ev, _| {
            if let GenEvent::Token(t) = ev { full.push(t); }
            true
        });
        assert!(full.len() >= 4);

        // Split: cap at 3, force-feed the 4th reference token, continue.
        let mut e2 = tiny_engine();
        let mut head: Vec<u32> = Vec::new();
        e2.generate_streaming_events(&prompt, 3, &greedy, |ev, _| {
            if let GenEvent::Token(t) = ev { head.push(t); }
            true
        });
        assert_eq!(head[..], full[..3]);
        let mut tail: Vec<u32> = Vec::new();
        let n = e2.continue_streaming_events(&[full[3]], full.len() - 4, &greedy, |ev, dep| {
            assert!(dep.is_none(), "no hook → no deposits");
            if let GenEvent::Token(t) = ev { tail.push(t); }
            true
        });
        assert_eq!(n, tail.len());
        assert_eq!(tail[..], full[4..], "continuation must be exact (greedy)");
    }

    #[test]
    fn no_hook_generate_returns_tokens() {
        let mut e = tiny_engine();
        let (tokens, deposits) = e.generate(&[0u32, 1, 2], 5, &SamplingConfig::default());
        assert!(!tokens.is_empty(), "should generate at least one token");
        assert!(tokens.len() <= 5);
        assert!(deposits.is_empty(), "no hook → deposits must be empty");
    }

    #[test]
    fn hook_fires_for_each_generated_token() {
        struct CountHook(AtomicU32);
        impl StigmergicHook for CountHook {
            fn on_layer(&self, _: usize, _: &[f32]) -> Option<f32> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Some(1.0)
            }
        }

        let hook = Arc::new(CountHook(AtomicU32::new(0)));
        let mut e = tiny_engine()
            .with_hook(Arc::clone(&hook) as Arc<dyn StigmergicHook>);

        let cfg = SamplingConfig { temperature: 0.0, ..Default::default() };
        let (tokens, deposits) = e.generate(&[0u32, 1, 2], 3, &cfg);

        assert!(!tokens.is_empty());
        assert!(!deposits.is_empty(), "hook attached → deposits must be non-empty");
        // Hook fires: (prefill_len + generated_len) × num_layers
        // tiny config: 2 layers, prefill=3, generated=tokens.len()
        let expected_min = tokens.len() * 2; // generated tokens × layers (tiny=2)
        assert!(
            hook.0.load(Ordering::Relaxed) as usize >= expected_min,
            "hook fired {} times, expected ≥ {}",
            hook.0.load(Ordering::Relaxed),
            expected_min
        );
    }

    #[test]
    fn pheromone_deposit_fields_correct() {
        struct FixedHook;
        impl StigmergicHook for FixedHook {
            fn on_layer(&self, _: usize, _: &[f32]) -> Option<f32> { Some(2.0) }
        }

        let mut e = tiny_engine()
            .with_hook(Arc::new(FixedHook) as Arc<dyn StigmergicHook>);
        let cfg = SamplingConfig { temperature: 0.0, ..Default::default() };
        let (tokens, deposits) = e.generate(&[0u32], 1, &cfg);
        assert!(!tokens.is_empty());
        let d = &deposits[0];
        assert_eq!(d.token_id, tokens[0]);
        // tiny config: 2 layers, each deposits 2.0 → total = 4.0
        assert!(
            (d.total_delta - 4.0).abs() < 1e-5,
            "expected 4.0 total_delta, got {}",
            d.total_delta
        );
        assert_eq!(d.layers_fired, 2, "all 2 tiny layers should fire");
    }

    #[test]
    fn selective_hook_deposits_on_threshold() {
        struct ThresholdHook { threshold: usize }
        impl StigmergicHook for ThresholdHook {
            fn on_layer(&self, layer_idx: usize, _: &[f32]) -> Option<f32> {
                if layer_idx >= self.threshold { Some(1.0) } else { None }
            }
        }

        // tiny config has 2 layers (0 and 1). Threshold at 1 → only layer 1 fires.
        let hook = Arc::new(ThresholdHook { threshold: 1 });
        let mut e = tiny_engine()
            .with_hook(hook as Arc<dyn StigmergicHook>);
        let cfg = SamplingConfig { temperature: 0.0, ..Default::default() };
        let (tokens, deposits) = e.generate(&[0u32], 1, &cfg);
        if !tokens.is_empty() {
            let d = &deposits[0];
            assert_eq!(d.layers_fired, 1, "only 1 of 2 tiny layers fires above threshold=1");
            assert!((d.total_delta - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn set_hook_none_removes_hook() {
        struct AlwaysHook;
        impl StigmergicHook for AlwaysHook {
            fn on_layer(&self, _: usize, _: &[f32]) -> Option<f32> { Some(1.0) }
        }
        let mut e = tiny_engine()
            .with_hook(Arc::new(AlwaysHook) as Arc<dyn StigmergicHook>);
        assert!(e.has_hook());
        e.set_hook(None);
        assert!(!e.has_hook());
        let (_, deps) = e.generate(&[0u32], 2, &SamplingConfig::default());
        assert!(deps.is_empty(), "after hook removed, deposits must be empty");
    }

    #[test]
    fn streaming_no_hook_deposit_is_none() {
        let mut e = tiny_engine();
        let mut saw_none = true;
        e.generate_streaming(&[0u32, 1], 3, &SamplingConfig::default(), |_tok, dep| {
            if dep.is_some() { saw_none = false; }
            true
        });
        assert!(saw_none, "no hook → all deposits must be None");
    }

    #[test]
    fn streaming_with_hook_deposit_is_some() {
        struct AlwaysHook;
        impl StigmergicHook for AlwaysHook {
            fn on_layer(&self, _: usize, _: &[f32]) -> Option<f32> { Some(0.5) }
        }
        let mut e = tiny_engine()
            .with_hook(Arc::new(AlwaysHook) as Arc<dyn StigmergicHook>);
        let cfg = SamplingConfig { temperature: 0.0, ..Default::default() };
        let mut all_some = true;
        let count = e.generate_streaming(&[0u32], 2, &cfg, |_tok, dep| {
            if dep.is_none() { all_some = false; }
            true
        });
        if count > 0 {
            assert!(all_some, "hook attached → all deposits must be Some");
        }
    }

    #[test]
    fn reset_between_calls() {
        let cfg = SamplingConfig { temperature: 0.0, ..Default::default() };
        let mut e = tiny_engine();
        let (t1, _) = e.generate(&[0u32, 1], 3, &cfg);
        e.reset();
        let (t2, _) = e.generate(&[0u32, 1], 3, &cfg);
        // Greedy + same seed → deterministic output
        assert_eq!(t1, t2, "greedy output after reset must be identical");
    }

    #[test]
    fn engine_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<InferEngine>();
    }
}
