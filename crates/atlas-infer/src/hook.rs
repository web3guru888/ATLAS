//! StigmergicHook trait — pheromone wiring point into the transformer forward pass.
//!
//! # Design rationale
//!
//! The hook is deliberately minimal:
//! - Called **after** each transformer layer (post-residual-add hidden state).
//! - Receives `layer_idx` so callers can gate on specific layers (e.g. only
//!   the last quarter of layers where semantics are most stable).
//! - Returns `Option<f32>` — the pheromone delta to deposit on the current
//!   GraphPalace path.  `None` means "no deposit this step".
//! - `Send + Sync` so the engine can be used across threads.
//!
//! # Performance
//!
//! When `InferEngine::hook` is `None`, the trait-object dispatch is skipped
//! entirely (no vtable overhead, no allocation).  When present, one virtual
//! call per layer per token is incurred — negligible compared with the GEMM
//! cost of each layer.
//!
//! # GraphPalace integration
//!
//! The atlas-palace crate (or any external caller) implements this trait to
//! connect pheromone deposits to palace room/edge weights.  A typical
//! implementation computes a scalar from the hidden state (e.g. L2 norm,
//! mean abs activation, a learned probe) and deposits it on the currently
//! active navigation path.
//!
//! ```rust
//! use atlas_infer::StigmergicHook;
//!
//! /// Hook that deposits mean-absolute-activation as pheromone.
//! pub struct MeanActHook {
//!     /// Only deposit on layers at or above this index.
//!     pub layer_threshold: usize,
//!     /// Scale factor applied to the raw activation magnitude.
//!     pub scale: f32,
//! }
//!
//! impl StigmergicHook for MeanActHook {
//!     fn on_layer(&self, layer_idx: usize, hidden: &[f32]) -> Option<f32> {
//!         if layer_idx < self.layer_threshold { return None; }
//!         let mag = hidden.iter().map(|v| v.abs()).sum::<f32>()
//!                   / hidden.len().max(1) as f32;
//!         Some(mag * self.scale)
//!     }
//! }
//! ```

use std::sync::Arc;

/// Pheromone wiring trait for GraphPalace integration.
///
/// Implement this trait and attach it to [`InferEngine`] via
/// [`InferEngine::with_hook`].  The engine will call `on_layer` after every
/// transformer layer during autoregressive generation.
///
/// [`InferEngine`]: crate::InferEngine
pub trait StigmergicHook: Send + Sync {
    /// Called after each transformer layer's forward pass.
    ///
    /// # Parameters
    /// - `layer_idx`: zero-based layer index in `0..num_layers`.
    /// - `hidden`: slice of the post-residual-add hidden state (f32, length =
    ///   `d_model`).
    ///
    /// # Returns
    /// `Some(delta)` — a non-negative pheromone increment to deposit on the
    /// active GraphPalace path edge.  The engine accumulates all deltas for
    /// the current token and makes one deposit call at the end of the token
    /// step.  `None` means no deposit this step.
    fn on_layer(&self, layer_idx: usize, hidden: &[f32]) -> Option<f32>;
}

/// A no-op hook that never deposits pheromone.  Used internally when
/// the caller provides no hook so the code path is identical.
pub(crate) struct NullHook;

impl StigmergicHook for NullHook {
    #[inline(always)]
    fn on_layer(&self, _layer_idx: usize, _hidden: &[f32]) -> Option<f32> {
        None
    }
}

/// Convenience type alias for a boxed, thread-safe hook.
pub type ArcHook = Arc<dyn StigmergicHook>;

/// Accumulate pheromone deltas from one token step.
///
/// Called by `InferEngine` with the hook output after each layer.
/// Accumulates the sum of all `Some(delta)` values returned by the hook.
/// Returns `None` if no layer produced a deposit.
pub(crate) fn accumulate(hook: &dyn StigmergicHook, layer_idx: usize, hidden: &[f32], acc: &mut f32) -> bool {
    if let Some(delta) = hook.on_layer(layer_idx, hidden) {
        *acc += delta.max(0.0); // clamp negative deltas
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountHook(std::sync::atomic::AtomicUsize);
    impl StigmergicHook for CountHook {
        fn on_layer(&self, _layer_idx: usize, _hidden: &[f32]) -> Option<f32> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(1.0)
        }
    }

    #[test]
    fn null_hook_returns_none() {
        let h = NullHook;
        assert_eq!(h.on_layer(0, &[1.0, 2.0, 3.0]), None);
        assert_eq!(h.on_layer(31, &[0.0; 4096]), None);
    }

    #[test]
    fn accumulate_sums_deltas() {
        let h = CountHook(std::sync::atomic::AtomicUsize::new(0));
        let mut acc = 0.0f32;
        let deposited = accumulate(&h, 0, &[1.0, 2.0], &mut acc);
        assert!(deposited);
        assert_eq!(acc, 1.0);
        accumulate(&h, 1, &[1.0], &mut acc);
        assert_eq!(acc, 2.0);
        assert_eq!(h.0.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn accumulate_clamps_negative() {
        struct NegHook;
        impl StigmergicHook for NegHook {
            fn on_layer(&self, _: usize, _: &[f32]) -> Option<f32> { Some(-5.0) }
        }
        let mut acc = 0.0f32;
        accumulate(&NegHook, 0, &[], &mut acc);
        assert_eq!(acc, 0.0, "negative deltas must be clamped to zero");
    }

    #[test]
    fn hook_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NullHook>();
    }

    #[test]
    fn mean_act_hook_example() {
        struct MeanActHook { threshold: usize, scale: f32 }
        impl StigmergicHook for MeanActHook {
            fn on_layer(&self, layer_idx: usize, hidden: &[f32]) -> Option<f32> {
                if layer_idx < self.threshold { return None; }
                let mag = hidden.iter().map(|v| v.abs()).sum::<f32>()
                          / hidden.len().max(1) as f32;
                Some(mag * self.scale)
            }
        }
        let hook = MeanActHook { threshold: 2, scale: 0.01 };
        // Below threshold — no deposit
        assert_eq!(hook.on_layer(0, &[1.0, 2.0, 3.0]), None);
        assert_eq!(hook.on_layer(1, &[1.0, 2.0, 3.0]), None);
        // At threshold — deposit mean_abs * scale = 2.0 * 0.01
        let delta = hook.on_layer(2, &[1.0, 2.0, 3.0]);
        assert!(delta.is_some());
        let v = delta.unwrap();
        assert!((v - 0.02).abs() < 1e-6, "expected ~0.02, got {v}");
    }
}
