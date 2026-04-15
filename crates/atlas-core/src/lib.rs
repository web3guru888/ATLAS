//! atlas-core — Error types, traits, and configuration primitives.
//!
//! This is the foundation crate. Everything else depends on it.
//! No external dependencies — intentional.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod bench;

/// ATLAS error type.
#[derive(Debug)]
pub enum AtlasError {
    /// Shape mismatch in tensor operations.
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    /// Index out of bounds.
    OutOfBounds { index: usize, size: usize },
    /// I/O error with message.
    Io(String),
    /// Parse error with message.
    Parse(String),
    /// Causal inference failure.
    CausalInference(String),
    /// ZK proof failure.
    ProofFailure(String),
    /// General error with message.
    Other(String),
}

impl std::fmt::Display for AtlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } =>
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got),
            Self::OutOfBounds { index, size } =>
                write!(f, "index {} out of bounds for size {}", index, size),
            Self::Io(msg)               => write!(f, "I/O error: {}", msg),
            Self::Parse(msg)            => write!(f, "parse error: {}", msg),
            Self::CausalInference(msg)  => write!(f, "causal inference: {}", msg),
            Self::ProofFailure(msg)     => write!(f, "ZK proof failure: {}", msg),
            Self::Other(msg)            => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for AtlasError {}

/// Convenience Result type for ATLAS operations.
pub type Result<T> = std::result::Result<T, AtlasError>;

/// Trait for objects that can be persisted to/from bytes.
pub trait Persist: Sized {
    /// Serialize to bytes.
    fn to_bytes(&self) -> Vec<u8>;
    /// Deserialize from bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

/// ATLAS global configuration.
#[derive(Debug, Clone)]
pub struct AtlasConfig {
    /// Path to model weights directory.
    pub weights_dir: String,
    /// Path to corpus storage.
    pub corpus_dir: String,
    /// Path to palace storage.
    pub palace_dir: String,
    /// CUDA device index (0 = first GPU, None = CPU).
    pub cuda_device: Option<u32>,
    /// Pheromone decay rate τ (default 0.1).
    pub pheromone_decay: f32,
    /// Minimum Bayesian confidence for corpus entry (default 0.65).
    pub min_confidence: f32,
    /// Minimum novelty score for corpus entry (default 0.55).
    pub min_novelty: f32,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            weights_dir:      "weights/".to_string(),
            corpus_dir:       "corpus/".to_string(),
            palace_dir:       "palace/".to_string(),
            cuda_device:      Some(0),
            pheromone_decay:  0.1,
            min_confidence:   0.65,
            min_novelty:      0.55,
        }
    }
}

/// Data type enum for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 16-bit float (brain float).
    BF16,
    /// 8-bit integer (quantized).
    I8,
    /// 4-bit integer (quantized, packed).
    I4,
}

/// Device enum for tensor placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU.
    Cpu,
    /// CUDA GPU by device index.
    Cuda(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = AtlasError::ShapeMismatch {
            expected: vec![3, 4],
            got: vec![3, 5],
        };
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn default_config() {
        let cfg = AtlasConfig::default();
        assert!((cfg.pheromone_decay - 0.1).abs() < 1e-6);
        assert!((cfg.min_confidence - 0.65).abs() < 1e-6);
        assert_eq!(cfg.cuda_device, Some(0));
    }
}
