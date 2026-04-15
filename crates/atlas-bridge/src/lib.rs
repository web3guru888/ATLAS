//! atlas-bridge — Rings↔Ethereum ZK bridge for ATLAS agent-to-agent payments.
//!
//! This is a local simulation interface. Real blockchain calls are made via the
//! asi-build bridge module deployed on Sepolia. The interface here is deliberately
//! compatible with that system.
//!
//! # Architecture
//! ```text
//! ATLAS agent → AtlasBridge::deposit() → BridgeTransaction (Pending)
//!                                              ↓
//!                               ZK proof attached (atlas-zk Groth16)
//!                                              ↓
//!                               Rings verification → Ethereum settlement
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::{AtlasError, Result};
use atlas_zk::{groth16_prove, groth16_verify, Groth16Claim};

/// Bridge network configuration.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// EVM chain ID (e.g. 11155111 for Sepolia).
    pub chain_id: u64,
    /// Verifier contract address (hex string).
    pub contract_address: String,
    /// RPC endpoint URL.
    pub rpc_url: String,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            chain_id: 11155111, // Sepolia
            contract_address: "0xBf6a13B2AeF32fa5E7948280aC1Baac0A6b78e2f".into(),
            rpc_url: "https://sepolia.infura.io/v3/placeholder".into(),
        }
    }
}

/// Bridge operation type.
#[derive(Debug, Clone)]
pub enum BridgeOp {
    /// Deposit `amount` (in wei) to `recipient` address.
    Deposit { amount: u64, recipient: String },
    /// Withdraw `amount` using a ZK proof.
    Withdraw { amount: u64, proof: Vec<u8> },
}

impl BridgeOp {
    /// Serialize to bytes for hashing.
    fn to_bytes(&self) -> Vec<u8> {
        match self {
            BridgeOp::Deposit { amount, recipient } => {
                let mut v = b"deposit:".to_vec();
                v.extend_from_slice(&amount.to_be_bytes());
                v.push(b':');
                v.extend_from_slice(recipient.as_bytes());
                v
            }
            BridgeOp::Withdraw { amount, proof } => {
                let mut v = b"withdraw:".to_vec();
                v.extend_from_slice(&amount.to_be_bytes());
                v.push(b':');
                v.extend_from_slice(proof);
                v
            }
        }
    }
}

/// Status of a bridge transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxStatus {
    /// Awaiting confirmation.
    Pending,
    /// Confirmed on-chain.
    Confirmed,
    /// Failed with reason.
    Failed(String),
}

impl TxStatus {
    /// True if the transaction is confirmed.
    pub fn is_confirmed(&self) -> bool { matches!(self, Self::Confirmed) }
}

/// A bridge transaction with ZK proof.
#[derive(Debug, Clone)]
pub struct BridgeTransaction {
    /// The operation being bridged.
    pub op: BridgeOp,
    /// Monotonically increasing nonce.
    pub nonce: u64,
    /// SHA-256(chain_id || nonce || op_bytes).
    pub hash: [u8; 32],
    /// Transaction status.
    pub status: TxStatus,
    /// ZK proof of the operation's validity.
    pub proof: Option<Groth16Claim>,
}

/// The ATLAS bridge: manages bridge transactions locally and interfaces with
/// the Rings↔Ethereum bridge contract.
pub struct AtlasBridge {
    /// Configuration.
    pub config: BridgeConfig,
    /// All transactions (pending + confirmed + failed).
    pub transactions: Vec<BridgeTransaction>,
    /// Current nonce (auto-increments).
    pub nonce: u64,
    /// Internal ZK key (32 bytes).
    key: [u8; 32],
}

impl AtlasBridge {
    /// Create a new bridge with the given config and ZK key.
    pub fn new(config: BridgeConfig, key: [u8; 32]) -> Self {
        Self {
            config,
            transactions: Vec::new(),
            nonce: 0,
            key,
        }
    }

    /// Create with default config and a derived key.
    pub fn default_with_key(key: [u8; 32]) -> Self {
        Self::new(BridgeConfig::default(), key)
    }

    /// Compute the transaction hash: SHA-256(chain_id_be || nonce_be || op_bytes).
    pub fn tx_hash(chain_id: u64, nonce: u64, op_bytes: &[u8]) -> [u8; 32] {
        let mut msg = Vec::new();
        msg.extend_from_slice(&chain_id.to_be_bytes());
        msg.extend_from_slice(&nonce.to_be_bytes());
        msg.extend_from_slice(op_bytes);
        atlas_zk::sha256_pub(&msg)
    }

    /// Submit a deposit transaction. Returns the pending transaction.
    pub fn deposit(&mut self, amount: u64, recipient: &str) -> BridgeTransaction {
        let op = BridgeOp::Deposit { amount, recipient: recipient.to_string() };
        self.submit(op)
    }

    /// Submit a withdrawal with ZK proof. Returns the pending transaction.
    pub fn withdraw(&mut self, amount: u64, proof_bytes: Vec<u8>) -> BridgeTransaction {
        let op = BridgeOp::Withdraw { amount, proof: proof_bytes };
        self.submit(op)
    }

    fn submit(&mut self, op: BridgeOp) -> BridgeTransaction {
        self.nonce += 1;
        let nonce = self.nonce;
        let op_bytes = op.to_bytes();
        let hash = Self::tx_hash(self.config.chain_id, nonce, &op_bytes);

        // Attach ZK proof
        let statement = format!("atlas-bridge:chain={},nonce={}", self.config.chain_id, nonce);
        let proof = groth16_prove(&statement, &op_bytes, &self.key);

        let tx = BridgeTransaction {
            op,
            nonce,
            hash,
            status: TxStatus::Pending,
            proof: Some(proof),
        };
        self.transactions.push(tx.clone());
        tx
    }

    /// Mark a transaction as confirmed (by hash).
    pub fn confirm(&mut self, hash: [u8; 32]) {
        for tx in &mut self.transactions {
            if tx.hash == hash {
                tx.status = TxStatus::Confirmed;
                return;
            }
        }
    }

    /// Mark a transaction as failed.
    pub fn fail(&mut self, hash: [u8; 32], reason: String) {
        for tx in &mut self.transactions {
            if tx.hash == hash {
                tx.status = TxStatus::Failed(reason);
                return;
            }
        }
    }

    /// Count pending transactions.
    pub fn pending_count(&self) -> usize {
        self.transactions.iter().filter(|t| t.status == TxStatus::Pending).count()
    }

    /// Count confirmed transactions.
    pub fn confirmed_count(&self) -> usize {
        self.transactions.iter().filter(|t| t.status.is_confirmed()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bridge() -> AtlasBridge {
        AtlasBridge::default_with_key([42u8; 32])
    }

    #[test]
    fn deposit_creates_pending_tx() {
        let mut b = make_bridge();
        let tx = b.deposit(1_000_000, "0xRecipient");
        assert_eq!(tx.status, TxStatus::Pending);
        assert_eq!(tx.nonce, 1);
        assert!(matches!(tx.op, BridgeOp::Deposit { amount: 1_000_000, .. }));
    }

    #[test]
    fn withdraw_creates_pending_tx() {
        let mut b = make_bridge();
        let tx = b.withdraw(500_000, vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(tx.status, TxStatus::Pending);
        assert!(matches!(tx.op, BridgeOp::Withdraw { amount: 500_000, .. }));
    }

    #[test]
    fn confirm_updates_status() {
        let mut b = make_bridge();
        let tx = b.deposit(100, "0xA");
        let hash = tx.hash;
        b.confirm(hash);
        assert_eq!(b.transactions[0].status, TxStatus::Confirmed);
    }

    #[test]
    fn tx_hash_deterministic() {
        let h1 = AtlasBridge::tx_hash(11155111, 1, b"deposit:100:0xA");
        let h2 = AtlasBridge::tx_hash(11155111, 1, b"deposit:100:0xA");
        assert_eq!(h1, h2);
    }

    #[test]
    fn bridge_nonce_increments() {
        let mut b = make_bridge();
        let t1 = b.deposit(1, "0xA");
        let t2 = b.deposit(2, "0xB");
        assert_eq!(t1.nonce, 1);
        assert_eq!(t2.nonce, 2);
    }

    #[test]
    fn pending_count_correct() {
        let mut b = make_bridge();
        b.deposit(1, "0xA");
        b.deposit(2, "0xB");
        assert_eq!(b.pending_count(), 2);
        let tx = b.deposit(3, "0xC");
        b.confirm(tx.hash);
        assert_eq!(b.pending_count(), 2);
        assert_eq!(b.confirmed_count(), 1);
    }

    #[test]
    fn deposit_hash_nonzero() {
        let mut b = make_bridge();
        let tx = b.deposit(999, "0xX");
        assert!(tx.hash.iter().any(|&v| v != 0));
    }

    #[test]
    fn withdraw_proof_attached() {
        let mut b = make_bridge();
        let tx = b.withdraw(100, vec![1, 2, 3]);
        assert!(tx.proof.is_some(), "proof should be attached");
        let p = tx.proof.unwrap();
        // verify with the same key
        assert!(groth16_verify(&p, &[42u8; 32]));
    }
}
