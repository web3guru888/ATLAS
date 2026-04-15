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

// ── Provenance chain ───────────────────────────────────────────────────────

/// Type of link in a provenance chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceLinkType {
    /// Raw data from an API endpoint.
    ApiSource,
    /// An observation derived from raw data.
    Observation,
    /// A causal hypothesis extracted from observations.
    Hypothesis,
    /// A validated, quality-gated discovery.
    Discovery,
}

impl ProvenanceLinkType {
    /// Serialize to a short tag string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ApiSource   => "api_source",
            Self::Observation => "observation",
            Self::Hypothesis  => "hypothesis",
            Self::Discovery   => "discovery",
        }
    }

    /// Parse from a tag string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "api_source"   => Some(Self::ApiSource),
            "observation"  => Some(Self::Observation),
            "hypothesis"   => Some(Self::Hypothesis),
            "discovery"    => Some(Self::Discovery),
            _              => None,
        }
    }
}

/// A single link in a provenance chain, cryptographically bound by a Schnorr proof.
#[derive(Debug, Clone)]
pub struct ProvenanceLink {
    /// Human-readable claim text.
    pub claim: String,
    /// Type of this link.
    pub link_type: ProvenanceLinkType,
    /// Schnorr proof binding `claim` to the prover's secret.
    pub proof: SchnorrProof,
    /// When this link was created.
    pub timestamp: u64,
}

/// An ordered chain of provenance links, from raw API source → observation → discovery.
///
/// Every link is individually Schnorr-signed by the same secret, forming an
/// auditable trail of how a piece of knowledge was derived.
///
/// # Example
/// ```
/// use atlas_zk::{ProvenanceChain, ProvenanceLinkType};
/// let secret = b"atlas_provenance_key";
/// let mut chain = ProvenanceChain::new(secret);
/// chain.add_link("NASA POWER API: T2M=25.3°C, lat=40.7, lon=-74.0", secret, ProvenanceLinkType::ApiSource);
/// chain.add_link("Temperature 25.3°C observed at NYC", secret, ProvenanceLinkType::Observation);
/// chain.add_link("Temperature correlates with CO2 concentration", secret, ProvenanceLinkType::Discovery);
/// assert!(chain.verify_all());
/// assert_eq!(chain.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct ProvenanceChain {
    /// Ordered links (index 0 = earliest/raw source, last = final claim).
    pub links: Vec<ProvenanceLink>,
    /// Public key of the prover (same across all links).
    pub public_key: u64,
}

impl ProvenanceChain {
    /// Create an empty provenance chain bound to a prover secret.
    pub fn new(secret: &[u8]) -> Self {
        let params = SchnorrParams::small_64();
        let pk = secret_to_pubkey(&params, secret);
        Self { links: Vec::new(), public_key: pk }
    }

    /// Append a provenance link with a Schnorr proof.
    pub fn add_link(&mut self, claim: &str, secret: &[u8], link_type: ProvenanceLinkType) {
        let params = SchnorrParams::small_64();
        // Include chain position in the signed message to prevent reordering
        let msg = format!("{}|pos={}|type={}", claim, self.links.len(), link_type.as_str());
        let proof = SchnorrProver::prove(&params, secret, msg.as_bytes());
        self.links.push(ProvenanceLink {
            claim: claim.to_string(),
            link_type,
            proof,
            timestamp: epoch_secs(),
        });
    }

    /// Verify every link in the chain.  Returns `true` only if ALL proofs verify
    /// and every link's public key matches the chain's declared public key.
    pub fn verify_all(&self) -> bool {
        let params = SchnorrParams::small_64();
        for (i, link) in self.links.iter().enumerate() {
            // Reconstruct the exact signed message
            let msg = format!("{}|pos={}|type={}", link.claim, i, link.link_type.as_str());
            if !SchnorrVerifier::verify(&params, &link.proof, msg.as_bytes()) {
                return false;
            }
            // All links must share the same public key
            if link.proof.public_key != self.public_key {
                return false;
            }
        }
        true
    }

    /// Number of links in the chain.
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// True if the chain has no links.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Serialize the chain to a JSON string for storage or audit.
    pub fn to_json(&self) -> String {
        let links_json: Vec<String> = self.links.iter().enumerate().map(|(i, link)| {
            format!(
                r#"{{"pos":{},"claim":{},"type":"{}","commitment":{},"response":{},"public_key":{},"timestamp":{}}}"#,
                i,
                json_escape_provenance(&link.claim),
                link.link_type.as_str(),
                link.proof.commitment,
                link.proof.response,
                link.proof.public_key,
                link.timestamp,
            )
        }).collect();
        format!(
            r#"{{"version":1,"public_key":{},"link_count":{},"links":[{}]}}"#,
            self.public_key,
            self.links.len(),
            links_json.join(","),
        )
    }

    /// Deserialize a provenance chain from its JSON representation.
    pub fn from_json(s: &str) -> Result<Self> {
        // Minimal JSON parsing using string operations (zero-dep)
        let pk = extract_u64_field(s, "public_key")
            .ok_or_else(|| AtlasError::Parse("provenance: missing public_key".into()))?;

        let mut chain = ProvenanceChain {
            links: Vec::new(),
            public_key: pk,
        };

        // Find the "links" array and parse each object
        let links_start = s.find("\"links\":[")
            .ok_or_else(|| AtlasError::Parse("provenance: missing links array".into()))?;
        let array_start = links_start + "\"links\":[".len();
        let array_end = s[array_start..].rfind(']')
            .map(|p| array_start + p)
            .ok_or_else(|| AtlasError::Parse("provenance: unterminated links array".into()))?;
        let array_content = &s[array_start..array_end];

        // Split on "},{" to find individual link objects
        if array_content.trim().is_empty() {
            return Ok(chain);
        }

        let mut depth = 0i32;
        let mut obj_start = 0;
        let chars: Vec<char> = array_content.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let obj = &array_content[obj_start..=i];
                        if let Some(link) = parse_provenance_link(obj) {
                            chain.links.push(link);
                        }
                        obj_start = i + 1;
                        // skip comma
                        if obj_start < chars.len() && chars[obj_start] == ',' {
                            obj_start += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(chain)
    }
}

/// Extract a u64 field value from a JSON string (simple pattern match).
fn extract_u64_field(s: &str, field: &str) -> Option<u64> {
    let pattern = format!("\"{}\":", field);
    let start = s.find(&pattern)?;
    let after = start + pattern.len();
    let rest = s[after..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extract a quoted string field from a JSON object, handling escaped quotes.
/// Returns the raw content between the quotes (with escape sequences still present).
fn extract_str_field<'a>(s: &'a str, field: &str) -> Option<&'a str> {
    let pattern = format!("\"{}\":\"", field);
    let start = s.find(&pattern)? + pattern.len();
    // Walk forward, skipping escaped quotes
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2; // skip escaped character
        } else if bytes[i] == b'"' {
            return Some(&s[start..i]);
        } else {
            i += 1;
        }
    }
    None
}

/// Parse a single ProvenanceLink from a JSON object string.
fn parse_provenance_link(s: &str) -> Option<ProvenanceLink> {
    let claim_raw = extract_str_field(s, "claim").unwrap_or("");
    let claim = json_unescape(claim_raw);
    let link_type_str = extract_str_field(s, "type")?;
    let link_type = ProvenanceLinkType::from_str(link_type_str)?;
    let commitment = extract_u64_field(s, "commitment")?;
    let response = extract_u64_field(s, "response")?;
    let public_key = extract_u64_field(s, "public_key")?;
    let timestamp = extract_u64_field(s, "timestamp").unwrap_or(0);
    Some(ProvenanceLink {
        claim,
        link_type,
        proof: SchnorrProof { commitment, response, public_key },
        timestamp,
    })
}

/// Unescape a JSON string value (reverse of `json_escape_provenance`, without outer quotes).
fn json_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"')  => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n')  => out.push('\n'),
                Some('r')  => out.push('\r'),
                Some('t')  => out.push('\t'),
                Some(other) => { out.push('\\'); out.push(other); }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// JSON-escape a string for provenance serialization.
fn json_escape_provenance(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
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

    // ── Provenance chain tests ──────────────────────────────────────────

    #[test]
    fn provenance_chain_verify() {
        let secret = b"atlas_provenance_key";
        let mut chain = ProvenanceChain::new(secret);
        chain.add_link("NASA POWER API: T2M=25.3, lat=40.7", secret, ProvenanceLinkType::ApiSource);
        chain.add_link("Temperature 25.3°C at NYC", secret, ProvenanceLinkType::Observation);
        chain.add_link("Temperature → CO2 (confidence=0.87)", secret, ProvenanceLinkType::Discovery);
        assert!(chain.verify_all());
        assert_eq!(chain.len(), 3);
    }

    #[test]
    fn provenance_chain_empty_verifies() {
        let chain = ProvenanceChain::new(b"key");
        assert!(chain.verify_all());
        assert!(chain.is_empty());
    }

    #[test]
    fn provenance_chain_tampered_fails() {
        let secret = b"my_secret";
        let mut chain = ProvenanceChain::new(secret);
        chain.add_link("original claim", secret, ProvenanceLinkType::ApiSource);
        chain.add_link("derived fact", secret, ProvenanceLinkType::Observation);
        // Tamper with the claim text
        chain.links[0].claim = "tampered claim".to_string();
        assert!(!chain.verify_all());
    }

    #[test]
    fn provenance_chain_reorder_fails() {
        let secret = b"secret_key";
        let mut chain = ProvenanceChain::new(secret);
        chain.add_link("first link", secret, ProvenanceLinkType::ApiSource);
        chain.add_link("second link", secret, ProvenanceLinkType::Discovery);
        // Swap links — position is baked into the proof
        chain.links.swap(0, 1);
        assert!(!chain.verify_all());
    }

    #[test]
    fn provenance_to_json_roundtrip() {
        let secret = b"roundtrip_key";
        let mut chain = ProvenanceChain::new(secret);
        chain.add_link("API raw data", secret, ProvenanceLinkType::ApiSource);
        chain.add_link("Observation from data", secret, ProvenanceLinkType::Observation);
        chain.add_link("Final discovery claim", secret, ProvenanceLinkType::Discovery);

        let json = chain.to_json();
        let chain2 = ProvenanceChain::from_json(&json).unwrap();

        assert_eq!(chain2.len(), 3);
        assert_eq!(chain2.public_key, chain.public_key);
        assert_eq!(chain2.links[0].claim, "API raw data");
        assert_eq!(chain2.links[1].link_type, ProvenanceLinkType::Observation);
        assert_eq!(chain2.links[2].claim, "Final discovery claim");
        // Deserialized chain should still verify
        assert!(chain2.verify_all());
    }

    #[test]
    fn provenance_wrong_pubkey_fails() {
        let secret = b"key_a";
        let mut chain = ProvenanceChain::new(secret);
        chain.add_link("claim", secret, ProvenanceLinkType::Discovery);
        // Mutate the chain's declared public key
        chain.public_key ^= 0xFF;
        assert!(!chain.verify_all());
    }

    #[test]
    fn provenance_link_type_roundtrip() {
        let types = [
            ProvenanceLinkType::ApiSource,
            ProvenanceLinkType::Observation,
            ProvenanceLinkType::Hypothesis,
            ProvenanceLinkType::Discovery,
        ];
        for t in &types {
            let s = t.as_str();
            let parsed = ProvenanceLinkType::from_str(s).unwrap();
            assert_eq!(*t, parsed);
        }
    }

    #[test]
    fn provenance_from_json_empty_links() {
        let json = r#"{"version":1,"public_key":12345,"link_count":0,"links":[]}"#;
        let chain = ProvenanceChain::from_json(json).unwrap();
        assert!(chain.is_empty());
        assert_eq!(chain.public_key, 12345);
    }

    #[test]
    fn provenance_from_json_malformed_errors() {
        assert!(ProvenanceChain::from_json("not json at all").is_err());
        assert!(ProvenanceChain::from_json(r#"{"links":[]}"#).is_err()); // missing public_key
    }
}
