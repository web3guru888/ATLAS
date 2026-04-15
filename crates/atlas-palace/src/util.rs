//! Internal utility functions: TF-IDF embeddings, cosine similarity, slugify.

/// Slugify a name for use as an id component.
pub fn slugify(s: &str) -> String {
    s.chars()
     .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
     .collect::<String>()
     .split('_')
     .filter(|s| !s.is_empty())
     .collect::<Vec<_>>()
     .join("_")
}

/// Compute a TF-IDF-like embedding vector for `text`.
/// Uses a fixed 256-dim projection based on character n-grams.
pub fn tfidf_embedding(text: &str) -> Vec<f32> {
    const DIM: usize = 256;
    let mut v = vec![0.0f32; DIM];
    if text.is_empty() { return v; }
    let lower = text.to_lowercase();
    let bytes = lower.as_bytes();
    // Character 2-grams + 3-grams hashed into DIM bins
    for i in 0..bytes.len().saturating_sub(1) {
        let h2 = (bytes[i] as usize * 31 + bytes[i+1] as usize) % DIM;
        v[h2] += 1.0;
    }
    for i in 0..bytes.len().saturating_sub(2) {
        let h3 = (bytes[i] as usize * 31 * 31
                + bytes[i+1] as usize * 31
                + bytes[i+2] as usize) % DIM;
        v[h3] += 1.0;
    }
    // L2 normalise
    let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
    if norm > 1e-8 { for vi in &mut v { *vi /= norm; } }
    v
}

/// Cosine similarity between two equal-length vectors.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { return 0.0; }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Return seconds since Unix epoch (wraps to 0 if syscall unavailable).
pub fn epoch_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tfidf_embedding_nonzero() {
        let e = tfidf_embedding("hello world");
        assert!(e.iter().any(|&v| v > 0.0));
        let norm = e.iter().map(|x| x*x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn tfidf_embedding_empty() {
        let e = tfidf_embedding("");
        assert!(e.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cosine_sim_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_sim(&a, &a) - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 0.001);
    }

    #[test]
    fn cosine_sim_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_sim(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("Hello World!"), "hello_world");
        assert_eq!(slugify("test--item"), "test_item");
    }
}
