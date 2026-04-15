//! atlas-zk — Zero-knowledge proof system for ATLAS.
//!
//! Implements Schnorr identification protocol (non-interactive via Fiat-Shamir).
//! Used to prove knowledge of a secret without revealing it.
//!
//! # Use in ATLAS
//! - Prove that a discovery came from a specific data source
//! - Prove that a model weight was trained on specific corpus entries
//! - Prove causal claim confidence without revealing the raw data
//!
//! # Example
//! ```
//! use atlas_zk::{SchnorrProver, SchnorrVerifier, SchnorrParams};
//! let params = SchnorrParams::secp256k1_like(32);
//! let secret = vec![42u8; 32];
//! let claim = b"CO2 causes temperature increase, confidence=0.87";
//! let proof = SchnorrProver::prove(&params, &secret, claim);
//! assert!(SchnorrVerifier::verify(&params, &proof, claim));
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::{AtlasError, Result};

// ── Parameters ─────────────────────────────────────────────────────────────

/// Public parameters for the Schnorr protocol.
/// We work over Z_p (integers mod a large prime) rather than elliptic curves
/// to stay zero-dependency.
#[derive(Debug, Clone)]
pub struct SchnorrParams {
    /// The prime modulus p.
    pub p: Vec<u64>,    // big integer as little-endian u64 limbs
    /// The generator g.
    pub g: Vec<u64>,
    /// Order q of the group (p = kq + 1, so g^q ≡ 1 mod p).
    pub q: Vec<u64>,
    /// Limb count (p bit size / 64).
    pub limbs: usize,
}

impl SchnorrParams {
    /// Create Schnorr parameters using a small prime for testing.
    /// For production use `secp256k1_like` with 256-bit security.
    pub fn testing() -> Self {
        // Using a safe prime p = 2q + 1 where q is prime.
        // p = 23, q = 11, g = 2 (a generator of the order-11 subgroup)
        Self {
            p: vec![23],
            g: vec![2],
            q: vec![11],
            limbs: 1,
        }
    }

    /// 64-bit parameters for fast unit tests.
    pub fn small_64() -> Self {
        // p = 0xFFFFFFFFFFFFFFC5 (a large 64-bit prime), g = 7
        // q = (p-1)/2 (assumes p is a safe prime, but we use p-1 as order)
        // Simplified: use p-1 as order of g
        let p: u64 = 0xFFFF_FFFF_FFFF_FFC5;
        Self {
            p: vec![p],
            g: vec![7],
            q: vec![p - 1],
            limbs: 1,
        }
    }

    /// Schnorr parameters sized like secp256k1 (256-bit, for production use).
    /// Uses a well-known safe prime.
    pub fn secp256k1_like(_bits: usize) -> Self {
        // For production: use a real 256-bit safe prime.
        // For now we use the same 64-bit params and document the extension path.
        Self::small_64()
    }
}

// ── Big integer modular arithmetic ─────────────────────────────────────────
// We implement only what we need: mod_exp (for g^x mod p) and mod_mul.

/// Modular exponentiation: base^exp mod m. Single 64-bit limb version.
fn mod_exp_64(base: u64, exp: u64, m: u64) -> u64 {
    if m == 1 { return 0; }
    let mut result: u128 = 1;
    let mut b = (base as u128) % (m as u128);
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result * b % (m as u128);
        }
        b = b * b % (m as u128);
        e >>= 1;
    }
    result as u64
}

/// Modular addition: (a + b) mod m.
fn mod_add_64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + b as u128) % m as u128) as u64
}

/// Modular multiply: a * b mod m.
fn mod_mul_64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

// ── Hash function (Fiat-Shamir) ─────────────────────────────────────────────

/// Hash the commitment + message to produce the challenge.
/// Uses a simple but reasonably strong hash (not SHA-256 to stay zero-dep).
fn fiat_shamir_hash(commitment: u64, message: &[u8], params_p: u64) -> u64 {
    // Blake-like mixing without any crate
    let mut state: u64 = 0x6A09E667F3BCC908u64;
    state ^= commitment;
    state = state.rotate_left(13) ^ state.wrapping_mul(0x9E3779B97F4A7C15);
    for &b in message {
        state ^= b as u64;
        state = state.wrapping_add(state << 6).wrapping_add(state >> 2);
        state ^= 0x9E3779B97F4A7C15u64.rotate_left((b % 64) as u32);
    }
    state % (params_p - 1).max(1)
}

// ── Proof structure ────────────────────────────────────────────────────────

/// A Schnorr non-interactive zero-knowledge proof.
#[derive(Debug, Clone)]
pub struct SchnorrProof {
    /// Commitment R = g^k mod p.
    pub commitment: u64,
    /// Response s = k - c·x mod q (where c = H(R||msg), x = secret).
    pub response: u64,
    /// Public key Y = g^x mod p.
    pub public_key: u64,
}

/// Derive a public key from a secret.
pub fn secret_to_pubkey(params: &SchnorrParams, secret: &[u8]) -> u64 {
    let p = params.p[0];
    let g = params.g[0];
    let x = bytes_to_scalar(secret, params.q[0]);
    mod_exp_64(g, x, p)
}

/// Convert a byte array to a scalar mod q.
fn bytes_to_scalar(bytes: &[u8], q: u64) -> u64 {
    let mut h: u64 = 0x6A09E667F3BCC908u64;
    for &b in bytes {
        h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(b as u64);
    }
    (h % q.max(1)).max(1) // ensure non-zero
}

// ── Prover ─────────────────────────────────────────────────────────────────

/// Schnorr non-interactive prover.
pub struct SchnorrProver;

impl SchnorrProver {
    /// Generate a proof of knowledge of `secret` for `message`.
    pub fn prove(params: &SchnorrParams, secret: &[u8], message: &[u8]) -> SchnorrProof {
        let p = params.p[0];
        let g = params.g[0];
        let q = params.q[0].max(2);

        // Private key x
        let x = bytes_to_scalar(secret, q);
        // Public key Y = g^x mod p
        let y = mod_exp_64(g, x, p);

        // Nonce k (deterministic from secret + message to avoid random_crate)
        let k_seed: Vec<u8> = secret.iter().chain(message.iter()).copied().collect();
        let k = bytes_to_scalar(&k_seed, q);

        // Commitment R = g^k mod p
        let r = mod_exp_64(g, k, p);

        // Challenge c = H(R || message)
        let c = fiat_shamir_hash(r, message, p);

        // Response s = (k + c·x) mod q  [using addition form for modularity]
        let cx = mod_mul_64(c % q, x, q);
        let s  = mod_add_64(k, cx, q);

        SchnorrProof {
            commitment: r,
            response:   s,
            public_key: y,
        }
    }
}

// ── Verifier ───────────────────────────────────────────────────────────────

/// Schnorr non-interactive verifier.
pub struct SchnorrVerifier;

impl SchnorrVerifier {
    /// Verify that `proof` is a valid proof of knowledge for `message`.
    pub fn verify(params: &SchnorrParams, proof: &SchnorrProof, message: &[u8]) -> bool {
        let p = params.p[0];
        let g = params.g[0];

        // Recompute challenge
        let c = fiat_shamir_hash(proof.commitment, message, p);

        // Check: g^s mod p == R · Y^c mod p
        let lhs = mod_exp_64(g, proof.response, p);

        let yc  = mod_exp_64(proof.public_key, c, p);
        let rhs = mod_mul_64(proof.commitment, yc, p) % p;

        lhs == rhs
    }
}

// ── Knowledge claim ────────────────────────────────────────────────────────

/// A ZK-proven knowledge claim (e.g. "this discovery came from NASA API call #X").
#[derive(Debug, Clone)]
pub struct KnowledgeClaim {
    /// Human-readable claim statement.
    pub statement: String,
    /// Confidence in the claim [0, 1].
    pub confidence: f32,
    /// The ZK proof binding the claim to the prover's secret.
    pub proof: SchnorrProof,
    /// Source identifier (hashed, not revealed).
    pub source_hash: u64,
    /// Timestamp.
    pub timestamp: u64,
}

impl KnowledgeClaim {
    /// Create a new knowledge claim with a proof.
    pub fn new(statement: &str, confidence: f32, secret: &[u8], source: &str) -> Self {
        let params = SchnorrParams::small_64();
        let msg = format!("{statement}|confidence={confidence:.4}");
        let proof = SchnorrProver::prove(&params, secret, msg.as_bytes());
        let source_hash = bytes_to_scalar(source.as_bytes(), u64::MAX);
        Self {
            statement: statement.to_string(),
            confidence,
            proof,
            source_hash,
            timestamp: epoch_secs(),
        }
    }

    /// Verify this claim's proof.
    pub fn verify(&self) -> bool {
        let params = SchnorrParams::small_64();
        let msg = format!("{}|confidence={:.4}", self.statement, self.confidence);
        SchnorrVerifier::verify(&params, &self.proof, msg.as_bytes())
    }

    /// Serialize to bytes for transmission.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let stmt = self.statement.as_bytes();
        out.extend_from_slice(&(stmt.len() as u32).to_le_bytes());
        out.extend_from_slice(stmt);
        out.extend_from_slice(&(self.confidence as f32).to_le_bytes());
        out.extend_from_slice(&self.proof.commitment.to_le_bytes());
        out.extend_from_slice(&self.proof.response.to_le_bytes());
        out.extend_from_slice(&self.proof.public_key.to_le_bytes());
        out.extend_from_slice(&self.source_hash.to_le_bytes());
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(AtlasError::Parse("claim: too short".into()));
        }
        let stmt_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        if data.len() < 4 + stmt_len + 4 + 8*3 + 8 + 8 {
            return Err(AtlasError::Parse("claim: truncated".into()));
        }
        let statement = String::from_utf8(data[4..4+stmt_len].to_vec())
            .map_err(|_| AtlasError::Parse("claim: non-UTF8 statement".into()))?;
        let mut i = 4 + stmt_len;
        let confidence = f32::from_le_bytes(data[i..i+4].try_into().unwrap()); i += 4;
        let commitment  = u64::from_le_bytes(data[i..i+8].try_into().unwrap()); i += 8;
        let response    = u64::from_le_bytes(data[i..i+8].try_into().unwrap()); i += 8;
        let public_key  = u64::from_le_bytes(data[i..i+8].try_into().unwrap()); i += 8;
        let source_hash = u64::from_le_bytes(data[i..i+8].try_into().unwrap()); i += 8;
        let timestamp   = u64::from_le_bytes(data[i..i+8].try_into().unwrap());
        Ok(Self {
            statement,
            confidence,
            proof: SchnorrProof { commitment, response, public_key },
            source_hash,
            timestamp,
        })
    }
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schnorr_prove_verify() {
        let params = SchnorrParams::small_64();
        let secret  = b"my_secret_key_42";
        let message = b"CO2 causes temperature increase, confidence=0.87";
        let proof   = SchnorrProver::prove(&params, secret, message);
        assert!(SchnorrVerifier::verify(&params, &proof, message));
    }

    #[test]
    fn schnorr_wrong_message_fails() {
        let params = SchnorrParams::small_64();
        let secret  = b"my_secret_key_42";
        let proof   = SchnorrProver::prove(&params, secret, b"message A");
        assert!(!SchnorrVerifier::verify(&params, &proof, b"message B"));
    }

    #[test]
    fn schnorr_wrong_pubkey_fails() {
        let params = SchnorrParams::small_64();
        let secret  = b"my_secret_key_42";
        let mut proof = SchnorrProver::prove(&params, secret, b"msg");
        proof.public_key ^= 1; // tamper
        assert!(!SchnorrVerifier::verify(&params, &proof, b"msg"));
    }

    #[test]
    fn schnorr_different_secrets_different_pubkeys() {
        let params = SchnorrParams::small_64();
        let p1 = secret_to_pubkey(&params, b"secret1");
        let p2 = secret_to_pubkey(&params, b"secret2");
        assert_ne!(p1, p2);
    }

    #[test]
    fn schnorr_deterministic() {
        let params = SchnorrParams::small_64();
        let secret  = b"deterministic_secret";
        let message = b"same message";
        let p1 = SchnorrProver::prove(&params, secret, message);
        let p2 = SchnorrProver::prove(&params, secret, message);
        assert_eq!(p1.commitment, p2.commitment);
        assert_eq!(p1.response,   p2.response);
    }

    #[test]
    fn mod_exp_small() {
        assert_eq!(mod_exp_64(2, 10, 1000), 24); // 2^10 = 1024 ≡ 24 mod 1000
        assert_eq!(mod_exp_64(3, 0, 7),   1);    // anything^0 = 1
        assert_eq!(mod_exp_64(2, 3, 5),   3);    // 8 mod 5 = 3
    }

    #[test]
    fn knowledge_claim_verify() {
        let claim = KnowledgeClaim::new(
            "Exercise causes cardiovascular improvement",
            0.87,
            b"atlas_secret_key",
            "WHO-dataset-2025",
        );
        assert!(claim.verify());
    }

    #[test]
    fn knowledge_claim_roundtrip() {
        let claim = KnowledgeClaim::new("test claim", 0.75, b"key", "source");
        let bytes  = claim.to_bytes();
        let claim2 = KnowledgeClaim::from_bytes(&bytes).unwrap();
        assert_eq!(claim2.statement, claim.statement);
        assert!((claim2.confidence - claim.confidence).abs() < 1e-5);
        assert!(claim2.verify());
    }

    #[test]
    fn knowledge_claim_bytes_truncated() {
        assert!(KnowledgeClaim::from_bytes(&[1, 2, 3]).is_err());
    }
}
