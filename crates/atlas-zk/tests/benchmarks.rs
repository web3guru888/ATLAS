//! ATLAS ZK benchmark tests.
//!
//! Run with: `cargo test -p atlas-zk --test benchmarks -- --ignored --nocapture`

use atlas_core::bench::Bench;
use atlas_zk::{SchnorrParams, SchnorrProver, SchnorrVerifier, KnowledgeClaim};

#[test]
#[ignore]
fn bench_schnorr_prove_verify() {
    let params = SchnorrParams::small_64();
    let secret = b"benchmark_secret_key_2026";
    let message = b"CO2 causes global temperature increase, confidence=0.87, source=NASA-GISS";

    let b_prove = Bench::run("schnorr_prove", 10_000, || {
        std::hint::black_box(SchnorrProver::prove(&params, secret, message));
    });
    eprintln!("{}", b_prove.report());

    let proof = SchnorrProver::prove(&params, secret, message);
    let b_verify = Bench::run("schnorr_verify", 10_000, || {
        std::hint::black_box(SchnorrVerifier::verify(&params, &proof, message));
    });
    eprintln!("{}", b_verify.report());

    let b_roundtrip = Bench::run("schnorr_prove_verify", 5_000, || {
        let p = SchnorrProver::prove(&params, secret, message);
        std::hint::black_box(SchnorrVerifier::verify(&params, &p, message));
    });
    eprintln!("{}", b_roundtrip.report());

    assert!(b_prove.ns_per_op() > 0.0);
    assert!(b_verify.ns_per_op() > 0.0);
    assert!(b_roundtrip.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_knowledge_claim_create_verify() {
    let b = Bench::run("knowledge_claim_create_verify", 2_000, || {
        let claim = KnowledgeClaim::new(
            "Exercise reduces cardiovascular disease risk",
            0.87,
            b"atlas_secret_key_for_bench",
            "WHO-dataset-2025",
        );
        std::hint::black_box(claim.verify());
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_knowledge_claim_serialization() {
    let claim = KnowledgeClaim::new(
        "A moderately long claim statement to test serialization throughput in the benchmark",
        0.92,
        b"atlas_key",
        "benchmark-source",
    );
    let bytes = claim.to_bytes();

    let b_ser = Bench::run("claim_to_bytes", 10_000, || {
        std::hint::black_box(claim.to_bytes());
    });
    eprintln!("{}", b_ser.report());

    let b_deser = Bench::run("claim_from_bytes", 10_000, || {
        std::hint::black_box(KnowledgeClaim::from_bytes(&bytes).unwrap());
    });
    eprintln!("{}", b_deser.report());

    assert!(b_ser.ns_per_op() > 0.0);
    assert!(b_deser.ns_per_op() > 0.0);
}
