//! atlas-infer — StigmergicHook trait + InferEngine facade.
//!
//! This crate provides the public inference API for ATLAS.  It wraps
//! `atlas-model`'s `OlmoModel` and threads a [`StigmergicHook`] through
//! every transformer layer's forward pass, enabling GraphPalace pheromone
//! signals to be wired into the model's hidden states without polluting
//! core model logic.
//!
//! # Architecture
//!
//! ```text
//! atlas-api
//!   └── atlas-infer  (this crate)
//!         ├── StigmergicHook trait   ← pheromone wiring point
//!         ├── InferEngine            ← wraps OlmoModel + hook
//!         └── atlas-model            ← transformer implementation
//!               └── atlas-tensor     ← CUDA kernels / CPU GEMM
//! ```
//!
//! # StigmergicHook
//!
//! Implement this trait to receive per-layer hidden-state signals during
//! autoregressive generation.  The hook is called **after every transformer
//! layer** (post-residual-add, pre-next-layer norm) with the layer index and
//! a slice of the hidden-state vector.  Returning `Some(delta)` deposits a
//! pheromone increment on the active GraphPalace path.
//!
//! ```rust
//! use atlas_infer::{StigmergicHook, InferEngine};
//! use atlas_model::{OlmoModel, ModelConfig, SamplingConfig};
//! use std::sync::Arc;
//!
//! struct MyHook;
//! impl StigmergicHook for MyHook {
//!     fn on_layer(&self, layer_idx: usize, hidden: &[f32]) -> Option<f32> {
//!         // E.g. compute mean activation magnitude as pheromone signal
//!         let mag = hidden.iter().map(|v| v.abs()).sum::<f32>() / hidden.len() as f32;
//!         Some(mag * 0.01)
//!     }
//! }
//!
//! let cfg = ModelConfig::tiny();
//! let model = OlmoModel::new(cfg);
//! let engine = InferEngine::new(model).with_hook(Arc::new(MyHook));
//! // engine.generate(&[0u32, 1, 2], 10, &SamplingConfig::default());
//! ```
//!
//! # Zero overhead when no hook
//!
//! When `hook` is `None` (the default), the generated token loop calls
//! `OlmoModel::forward_one_hooked` with a no-op path that compiles down
//! to the same machine code as the vanilla `forward_one`.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod engine;
pub mod hook;
pub mod streaming;

pub use engine::InferEngine;
pub use hook::StigmergicHook;
pub use streaming::StreamToken;

// Re-export the most commonly used types so callers only need `atlas-infer`.
pub use atlas_model::{
    ModelConfig, OlmoModel, SamplingConfig,
    load_model_from_dir, load_model_from_safetensors,
};
pub use atlas_tokenize::Tokenizer;
