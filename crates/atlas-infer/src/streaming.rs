//! Streaming token type for atlas-infer.

use crate::engine::PheromoneDeposit;

/// A single streaming token event emitted by [`InferEngine::generate_streaming`].
///
/// [`InferEngine::generate_streaming`]: crate::InferEngine::generate_streaming
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The generated token id.
    pub token_id: u32,
    /// Pheromone deposit for this token (present only when a hook is attached).
    pub deposit: Option<PheromoneDeposit>,
}

impl StreamToken {
    /// Create a new StreamToken.
    pub fn new(token_id: u32, deposit: Option<PheromoneDeposit>) -> Self {
        Self { token_id, deposit }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_token_no_deposit() {
        let t = StreamToken::new(42, None);
        assert_eq!(t.token_id, 42);
        assert!(t.deposit.is_none());
    }

    #[test]
    fn stream_token_with_deposit() {
        let dep = PheromoneDeposit { token_id: 42, total_delta: 3.14, layers_fired: 8 };
        let t = StreamToken::new(42, Some(dep.clone()));
        assert_eq!(t.token_id, 42);
        let d = t.deposit.unwrap();
        assert!((d.total_delta - 3.14).abs() < 1e-6);
        assert_eq!(d.layers_fired, 8);
    }
}
