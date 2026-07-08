//! `spec_decode` — speculative-decoding prototype (Pathway 1/2, decode-throughput lever).
//!
//! Speculative decoding accelerates autoregressive generation by having a cheap
//! **drafter** propose `k` tokens, then verifying all of them in a *single*
//! batched target-model forward pass (the same batched machinery the shipped
//! `#22` batched-prefill path provides). With greedy verification the output is
//! provably **bit-identical** to plain greedy decoding — the win is pure latency,
//! never a quality change.
//!
//! This module is a self-contained, CPU-testable prototype of the *algorithm* and
//! its integration seam. It deliberately does **not** wire itself into the live
//! CUDA decode loop — that A/B belongs in an announced maintenance window. The
//! two traits below are the seam:
//! - [`Drafter`] — proposes speculative tokens. [`NgramDrafter`] is a zero-extra-model
//!   prompt-lookup drafter (Saxena 2023, "Prompt Lookup Decoding"): it finds the
//!   most recent recurrence of the last-n context tokens and proposes what followed.
//!   A real system can also plug in a small draft model / EAGLE head here.
//! - [`TargetModel`] — the full model. `verify(context, draft)` returns the greedy
//!   argmax token for each of the `draft.len() + 1` positions in one call, exactly
//!   what a batched forward produces.
//!
//! # Correctness invariant (tested)
//! `speculative_generate` with greedy verification returns the *same* token
//! sequence as [`greedy_generate`]. Speculation only changes *how many* target
//! forwards are needed, never the result.

/// Proposes up to `k` speculative continuation tokens for `context`.
pub trait Drafter {
    /// Return up to `k` proposed next tokens (may return fewer, including none).
    fn draft(&self, context: &[u32], k: usize) -> Vec<u32>;
}

/// The target (full) model under greedy decoding.
pub trait TargetModel {
    /// Greedy-verify `draft` against `context`.
    ///
    /// Returns `draft.len() + 1` tokens: for position `i`, the model's greedy
    /// argmax given `context` followed by `draft[..i]`. A batched forward computes
    /// all of these in one pass. Element `0` is the model's own next token for
    /// `context` (i.e. what plain greedy would emit), so an empty draft still
    /// yields exactly one token.
    fn verify(&mut self, context: &[u32], draft: &[u32]) -> Vec<u32>;
}

/// Prompt-lookup n-gram drafter — zero extra model, pure string matching.
#[derive(Debug, Clone)]
pub struct NgramDrafter {
    /// Length of the suffix n-gram to match against earlier context.
    pub n: usize,
}

impl NgramDrafter {
    /// New drafter matching on the last `n` tokens.
    pub fn new(n: usize) -> Self { Self { n: n.max(1) } }
}

impl Drafter for NgramDrafter {
    fn draft(&self, context: &[u32], k: usize) -> Vec<u32> {
        if context.len() <= self.n || k == 0 {
            return Vec::new();
        }
        let suffix = &context[context.len() - self.n..];
        // Search for the most recent earlier occurrence of `suffix`.
        let search_end = context.len() - self.n; // exclusive: don't match the suffix itself
        let mut best: Option<usize> = None;
        let mut i = 0usize;
        while i + self.n <= search_end {
            if &context[i..i + self.n] == suffix {
                best = Some(i + self.n); // tokens that followed this occurrence
            }
            i += 1;
        }
        match best {
            Some(start) => {
                let end = (start + k).min(context.len());
                context[start..end].to_vec()
            }
            None => Vec::new(),
        }
    }
}

/// Result of one speculative step.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecStep {
    /// Tokens committed this step (always ≥ 1: accepted draft prefix + 1 target token).
    pub committed: Vec<u32>,
    /// How many draft tokens were proposed.
    pub drafted: usize,
    /// How many draft tokens were accepted (matched the target's greedy choice).
    pub accepted: usize,
}

/// Run a single speculative step: draft `k`, verify, commit the accepted prefix
/// plus one guaranteed target token.
pub fn speculative_step<D: Drafter, T: TargetModel>(
    drafter: &D,
    target: &mut T,
    context: &[u32],
    k: usize,
) -> SpecStep {
    let draft = drafter.draft(context, k);
    let verified = target.verify(context, &draft); // len == draft.len() + 1
    // Accept the longest prefix where the drafter agreed with the target.
    let mut accepted = 0usize;
    while accepted < draft.len() && draft[accepted] == verified[accepted] {
        accepted += 1;
    }
    // Commit: the accepted draft tokens, then the target's token at the first
    // divergence (verified[accepted]) — this is the guaranteed +1 token.
    let mut committed: Vec<u32> = draft[..accepted].to_vec();
    committed.push(verified[accepted]);
    SpecStep { committed, drafted: draft.len(), accepted }
}

/// Aggregate metrics over a speculative-decoding run.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecMetrics {
    /// Total tokens produced.
    pub tokens: usize,
    /// Total target forward passes issued (each verifies a batch).
    pub target_calls: usize,
    /// Total draft tokens proposed.
    pub drafted: usize,
    /// Total draft tokens accepted.
    pub accepted: usize,
}

impl SpecMetrics {
    /// Draft acceptance rate ∈ [0, 1].
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted == 0 { 0.0 } else { self.accepted as f32 / self.drafted as f32 }
    }
    /// Tokens produced per target forward pass (the speedup proxy; plain greedy = 1.0).
    pub fn tokens_per_call(&self) -> f32 {
        if self.target_calls == 0 { 0.0 } else { self.tokens as f32 / self.target_calls as f32 }
    }
}

/// Speculative greedy generation. Produces `max_new` tokens (or until `stop`).
///
/// Returns `(tokens, metrics)`. **Invariant:** with greedy verification the
/// `tokens` are identical to [`greedy_generate`] for the same target model.
pub fn speculative_generate<D: Drafter, T: TargetModel>(
    drafter: &D,
    target: &mut T,
    prompt: &[u32],
    max_new: usize,
    k: usize,
    stop: Option<u32>,
) -> (Vec<u32>, SpecMetrics) {
    let mut context = prompt.to_vec();
    let mut out = Vec::new();
    let mut m = SpecMetrics { tokens: 0, target_calls: 0, drafted: 0, accepted: 0 };
    while out.len() < max_new {
        let step = speculative_step(drafter, target, &context, k);
        m.target_calls += 1;
        m.drafted += step.drafted;
        m.accepted += step.accepted;
        for &t in &step.committed {
            if out.len() >= max_new { break; }
            context.push(t);
            out.push(t);
            m.tokens += 1;
            if Some(t) == stop { return (out, m); }
        }
    }
    (out, m)
}

/// Plain greedy generation using the same [`TargetModel`] — the reference the
/// speculative path must match exactly.
pub fn greedy_generate<T: TargetModel>(
    target: &mut T,
    prompt: &[u32],
    max_new: usize,
    stop: Option<u32>,
) -> Vec<u32> {
    let mut context = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < max_new {
        let next = target.verify(&context, &[])[0]; // empty draft → 1 token
        context.push(next);
        out.push(next);
        if Some(next) == stop { break; }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic mock target: the next greedy token is a pure function of the
    /// last token (a fixed transition table), so plain and speculative decoding
    /// have a well-defined ground truth. Also exposes a repeating structure the
    /// n-gram drafter can exploit.
    struct MockTarget {
        /// next(tok) = transition[tok as usize % L]
        transition: Vec<u32>,
    }
    impl MockTarget {
        fn next_of(&self, ctx: &[u32]) -> u32 {
            let last = *ctx.last().unwrap_or(&0);
            self.transition[last as usize % self.transition.len()]
        }
    }
    impl TargetModel for MockTarget {
        fn verify(&mut self, context: &[u32], draft: &[u32]) -> Vec<u32> {
            // Position 0: next after context. Position i: next after context+draft[..i].
            let mut ctx = context.to_vec();
            let mut out = Vec::with_capacity(draft.len() + 1);
            out.push(self.next_of(&ctx));
            for &d in draft {
                ctx.push(d);
                out.push(self.next_of(&ctx));
            }
            out
        }
    }

    fn mock() -> MockTarget {
        // A 5-cycle: 0->1->2->3->4->0 ... highly predictable, ideal for lookup.
        MockTarget { transition: vec![1, 2, 3, 4, 0] }
    }

    #[test]
    fn speculative_matches_greedy_exactly() {
        let drafter = NgramDrafter::new(1);
        let prompt = vec![0u32];
        let reference = greedy_generate(&mut mock(), &prompt, 40, None);
        let (spec, _m) = speculative_generate(&drafter, &mut mock(), &prompt, 40, 4, None);
        assert_eq!(spec, reference, "greedy speculative decoding must be bit-identical");
        assert_eq!(spec.len(), 40);
    }

    #[test]
    fn speculation_reduces_target_calls_on_predictable_stream() {
        let drafter = NgramDrafter::new(1);
        let prompt = vec![0u32, 1, 2, 3, 4, 0, 1, 2, 3, 4]; // seeds the lookup table
        let (spec, m) = speculative_generate(&drafter, &mut mock(), &prompt, 40, 4, None);
        assert_eq!(spec.len(), 40);
        // On a fully predictable cycle the drafter should be accepted often, so we
        // produce many more tokens than target forward passes.
        assert!(m.tokens_per_call() > 1.5,
            "expected speedup > 1.5 tokens/call, got {} (accept rate {})",
            m.tokens_per_call(), m.acceptance_rate());
    }

    #[test]
    fn ngram_drafter_finds_repeat() {
        let d = NgramDrafter::new(2);
        // last two tokens [1,2] recurred earlier, followed by [3,4,0]
        let ctx = vec![1u32, 2, 3, 4, 0, 9, 9, 1, 2];
        let proposal = d.draft(&ctx, 3);
        assert_eq!(proposal, vec![3, 4, 0]);
    }

    #[test]
    fn ngram_drafter_empty_when_no_match() {
        let d = NgramDrafter::new(2);
        let ctx = vec![1u32, 2, 3];
        assert!(d.draft(&ctx, 4).is_empty());
    }

    #[test]
    fn empty_draft_commits_one_token() {
        struct Fixed;
        impl TargetModel for Fixed {
            fn verify(&mut self, _c: &[u32], draft: &[u32]) -> Vec<u32> {
                vec![7u32; draft.len() + 1]
            }
        }
        struct NoDraft;
        impl Drafter for NoDraft { fn draft(&self, _: &[u32], _: usize) -> Vec<u32> { Vec::new() } }
        let step = speculative_step(&NoDraft, &mut Fixed, &[0], 4);
        assert_eq!(step.committed, vec![7]);
        assert_eq!(step.accepted, 0);
        assert_eq!(step.drafted, 0);
    }
}
