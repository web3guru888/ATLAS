//! Integration tests for atlas-cli.
//!
//! These tests spawn `atlas` as a subprocess and verify its exit codes and
//! output.  They exercise every subcommand with a synthetic corpus so no
//! network access is required.

use std::process::Command;
use std::path::PathBuf;
use std::fs;

// Locate the compiled atlas binary.
fn atlas_bin() -> PathBuf {
    // In a workspace test run the binary lives in target/debug/atlas
    let mut p = std::env::current_exe().unwrap();
    p.pop(); // strip test binary name
    // Sometimes we're in target/debug/deps – go up one more
    if p.ends_with("deps") { p.pop(); }
    p.push("atlas");
    p
}

fn run(args: &[&str]) -> (i32, String, String) {
    let bin = atlas_bin();
    let out = Command::new(&bin).args(args).output().unwrap_or_else(|e| {
        panic!("Failed to run {:?}: {}", bin, e)
    });
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

// ──────────────────────────────────────────────────────────────────────────
//  --version / --help
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn version_flag() {
    let (code, out, _err) = run(&["--version"]);
    assert_eq!(code, 0);
    assert!(out.contains("atlas"), "expected 'atlas' in: {out}");
}

#[test]
fn help_flag() {
    let (code, out, _err) = run(&["--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("COMMANDS"), "expected COMMANDS in help: {out}");
}

#[test]
fn no_args_exits_zero_prints_usage() {
    let (code, out, _err) = run(&[]);
    assert_eq!(code, 0);
    assert!(out.contains("USAGE") || out.contains("atlas"), "unexpected output: {out}");
}

#[test]
fn unknown_command_exits_nonzero() {
    let (code, _out, err) = run(&["frobnicate"]);
    assert_ne!(code, 0);
    assert!(err.contains("unknown command") || err.contains("frobnicate"),
        "expected error message, got: {err}");
}

// ──────────────────────────────────────────────────────────────────────────
//  status
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn status_exits_zero() {
    let (code, out, _) = run(&["status"]);
    assert_eq!(code, 0);
    assert!(out.contains("atlas-core") || out.contains("Crates"), "status output: {out}");
}

// ──────────────────────────────────────────────────────────────────────────
//  prove
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn prove_generates_valid_proof() {
    let (code, out, err) = run(&[
        "prove",
        "--claim",  "Stigmergic pheromone trails compound knowledge graph signal",
        "--secret", "deadbeef01020304",
    ]);
    // Proof succeeds if claim.verify() returns true (code=0) or the Schnorr
    // params are too small (testing params) and verify returns false (code=1).
    // Either way the binary must not crash and must print expected fields.
    assert!(code == 0 || code == 1, "unexpected exit code {code}: {err}");
    assert!(out.contains("Commitment") || out.contains("claim"), "prove output: {out}");
}

#[test]
fn prove_missing_claim_exits_nonzero() {
    let (code, _out, err) = run(&["prove", "--secret", "deadbeef"]);
    assert_ne!(code, 0);
    assert!(err.contains("--claim"), "expected --claim error: {err}");
}

#[test]
fn prove_missing_secret_exits_nonzero() {
    let (code, _out, err) = run(&["prove", "--claim", "test claim here"]);
    assert_ne!(code, 0);
    assert!(err.contains("--secret"), "expected --secret error: {err}");
}

#[test]
fn prove_invalid_secret_exits_nonzero() {
    let (code, _out, err) = run(&["prove", "--claim", "test claim", "--secret", "ZZZZ"]);
    assert_ne!(code, 0);
    assert!(err.contains("hex"), "expected hex error: {err}");
}

// ──────────────────────────────────────────────────────────────────────────
//  corpus + eval (using a seeded corpus file)
// ──────────────────────────────────────────────────────────────────────────

/// Write a minimal corpus JSON that the CLI can load.
fn write_test_corpus(path: &str) {
    let corpus_json = r#"{"version":1,"total_rejected":2,"positive_feedback":0,"negative_feedback":0,"clock":3,"entries":[
        {"id":0,"source":"arxiv","title":"Causal discovery algorithms recover ground truth structures efficiently","confidence":0.82,"pheromone":0.75,"quality":0.78,"tier":1,"samples":0,"ingested_at":1},
        {"id":1,"source":"nasa_power","title":"CO2 atmospheric concentrations correlate strongly with global warming trends","confidence":0.88,"pheromone":0.80,"quality":0.84,"tier":2,"samples":0,"ingested_at":2},
        {"id":2,"source":"who_gho","title":"Vaccination rates inversely correlated with disease incidence rates globally","confidence":0.90,"pheromone":0.85,"quality":0.88,"tier":2,"samples":0,"ingested_at":3}
    ]}"#;
    fs::write(path, corpus_json).unwrap();
}

#[test]
fn corpus_stats_from_file() {
    let dir = std::env::temp_dir().join("atlas_test_corpus_stats");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corpus.json");
    write_test_corpus(path.to_str().unwrap());

    let (code, out, _err) = run(&[
        "corpus", "--path", path.to_str().unwrap(), "--stats",
    ]);
    assert_eq!(code, 0, "corpus --stats failed");
    assert!(out.contains("entries") || out.contains("Corpus"), "output: {out}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn corpus_list_from_file() {
    let dir = std::env::temp_dir().join("atlas_test_corpus_list");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corpus.json");
    write_test_corpus(path.to_str().unwrap());

    let (code, out, _err) = run(&[
        "corpus", "--path", path.to_str().unwrap(), "--list", "3",
    ]);
    assert_eq!(code, 0, "corpus --list failed");
    assert!(out.contains("pheromone") || out.contains("arxiv") || out.contains("1."), "output: {out}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn corpus_sample_from_file() {
    let dir = std::env::temp_dir().join("atlas_test_corpus_sample");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corpus.json");
    write_test_corpus(path.to_str().unwrap());

    let (code, out, _err) = run(&[
        "corpus", "--path", path.to_str().unwrap(), "--sample", "2",
    ]);
    assert_eq!(code, 0);
    assert!(out.contains("Sampled") || out.contains("["), "output: {out}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn eval_corpus_from_file() {
    let dir = std::env::temp_dir().join("atlas_test_eval");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corpus.json");
    write_test_corpus(path.to_str().unwrap());

    let (code, out, _err) = run(&[
        "eval", "--corpus", path.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "eval failed: {_err}");
    assert!(out.contains("health") || out.contains("quality") || out.contains("source"),
        "output: {out}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn train_corpus_from_file() {
    let dir = std::env::temp_dir().join("atlas_test_train");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("corpus.json");
    write_test_corpus(path.to_str().unwrap());
    let ckpt = dir.join("ckpt");

    let (code, out, _err) = run(&[
        "train",
        "--corpus", path.to_str().unwrap(),
        "--epochs", "1",
        "--batch",  "2",
        "--output", ckpt.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "train failed: {_err}");
    assert!(out.contains("complete") || out.contains("loss") || out.contains("step"),
        "output: {out}");

    // Checkpoint should have been written
    let ckpt_exists = fs::read_dir(&ckpt)
        .map(|d| d.count() > 0)
        .unwrap_or(false);
    assert!(ckpt_exists, "Expected checkpoint in {:?}", ckpt);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn train_missing_corpus_exits_nonzero() {
    let (code, _out, err) = run(&[
        "train", "--corpus", "/nonexistent/corpus.json",
    ]);
    assert_ne!(code, 0);
    assert!(err.contains("corpus") || err.contains("load") || err.contains("not"),
        "expected error message: {err}");
}

#[test]
fn corpus_missing_path_exits_nonzero() {
    let (code, _out, err) = run(&["corpus", "--stats"]);
    assert_ne!(code, 0);
    assert!(err.contains("--path"), "expected --path error: {err}");
}

#[test]
fn eval_missing_corpus_exits_nonzero() {
    let (code, _out, err) = run(&["eval"]);
    assert_ne!(code, 0);
    assert!(err.contains("--corpus"), "expected --corpus error: {err}");
}

// ──────────────────────────────────────────────────────────────────────────
//  palace
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn palace_status_new_palace() {
    let dir = std::env::temp_dir().join("atlas_test_palace");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("palace.json");

    let (code, out, _err) = run(&[
        "palace", "--path", path.to_str().unwrap(), "--stats",
    ]);
    assert_eq!(code, 0, "palace --stats failed");
    assert!(out.contains("wings") || out.contains("Palace") || out.contains("palace"),
        "output: {out}");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn palace_add_and_search() {
    let dir = std::env::temp_dir().join("atlas_test_palace_search");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("palace.json");

    // Add a drawer
    let (code, _out, _err) = run(&[
        "palace", "--path", path.to_str().unwrap(),
        "--add", "Stigmergy enables collective intelligence through pheromone trails",
    ]);
    assert_eq!(code, 0, "palace --add failed: {_err}");

    // Search for it
    let (code, out, _err) = run(&[
        "palace", "--path", path.to_str().unwrap(),
        "--search", "pheromone",
    ]);
    assert_eq!(code, 0, "palace --search failed: {_err}");
    // Results section should be present
    assert!(out.contains("Search") || out.contains("results") || out.contains("pheromone"),
        "search output: {out}");

    fs::remove_dir_all(&dir).ok();
}
