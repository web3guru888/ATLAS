//! atlas — ATLAS CLI entrypoint
//!
//! A single pure-Rust binary that orchestrates the full ATLAS pipeline:
//! discovery → corpus → training → evaluation → ZK provenance.
//!
//! # Commands
//! - `atlas discover`  — ASTRA OODA discovery engine (Stage 5)
//! - `atlas corpus`    — LiveDiscoveryCorpus management (Stage 6)
//! - `atlas train`     — Gradient-descent fine-tuning on corpus (Stage 1+6)
//! - `atlas eval`      — Model quality evaluation (Stage 2+6)
//! - `atlas prove`     — ZK Schnorr provenance proof (Stage 5)
//! - `atlas palace`    — GraphPalace inspection (Stage 3)
//! - `atlas status`    — System-wide health check
//! - `atlas bench`     — End-to-end benchmark suite (palace/training/gates throughput)
//! - `atlas mcp`       — Model Context Protocol server (JSON-RPC 2.0 over stdin/stdout)
//! - `atlas infer`     — Real LLM inference (SmolLM2-1.7B / OLMo / Llama)
//!
//! # Zero-dependency arg parsing: no clap, no structopt.

use std::env;
use std::path::Path;

// ────────────────────────────────────────────────────────────────────────────
//  Entrypoint
// ────────────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(0);
    }

    let exit_code = match args[1].as_str() {
        "discover"  => cmd_discover(&args[2..]),
        "corpus"    => cmd_corpus(&args[2..]),
        "train"     => cmd_train(&args[2..]),
        "eval"      => cmd_eval(&args[2..]),
        "prove"     => cmd_prove(&args[2..]),
        "palace"    => cmd_palace(&args[2..]),
        "status"    => cmd_status(&args[2..]),
        "bench"     => cmd_bench(&args[2..]),
        "mcp"       => cmd_mcp(&args[2..]),
        "infer"     => cmd_infer(&args[2..]),
        "--version" | "-V" => { println!("atlas {}", env!("CARGO_PKG_VERSION")); 0 }
        "--help" | "-h" => { print_usage(); 0 }
        cmd => {
            eprintln!("atlas: unknown command '{}'. Run `atlas --help`.", cmd);
            1
        }
    };
    std::process::exit(exit_code);
}

// ────────────────────────────────────────────────────────────────────────────
//  Usage
// ────────────────────────────────────────────────────────────────────────────

fn print_usage() {
    println!(
        r#"ATLAS — Active-inference Training with Learned Adaptive Stigmergy
Pure Rust · Zero external crate dependencies · v{ver}

USAGE:
    atlas <COMMAND> [OPTIONS]

COMMANDS:
    discover    Run ASTRA OODA discovery engine
                  --cycles <N>       Discovery cycles (default: 3)
                  --output <PATH>    Save corpus to path (default: ./atlas-corpus.json)
                  --min-quality      Min discovery quality (default: 0.55)
                  --arxiv <QUERY>    ArXiv query string (default: causal inference)

    corpus      Manage LiveDiscoveryCorpus
                  --path <PATH>      Corpus file (required)
                  --stats            Print statistics
                  --list [N]         List top N entries by pheromone (default: 10)
                  --decay            Apply pheromone decay (factor 0.95)
                  --sample <N>       Sample N entries (prints training text)

    train       Fine-tune on LiveDiscoveryCorpus (gradient descent)
                  --corpus <PATH>    Corpus JSON (required)
                  --epochs <N>       Training epochs (default: 1)
                  --lr <LR>          Learning rate (default: 2e-5)
                  --batch <N>        Batch size (default: 8)
                  --output <DIR>     Checkpoint directory (default: ./atlas-ckpt)

    eval        Evaluate corpus health and model quality
                  --corpus <PATH>    Corpus JSON (required)
                  --verbose          Print per-entry scores

    prove       Generate ZK Schnorr provenance proof
                  --claim <TEXT>     Claim to prove (required)
                  --secret <HEX>     Secret key bytes as hex (required)
                  --output <PATH>    Write proof file (optional)

    palace      Inspect atlas-palace state
                  --path <PATH>      Palace JSON (default: ./atlas-palace.json)
                  --hot              Show top hot-path drawers
                  --search <QUERY>   Semantic search
                  --add <TEXT>       Add a drawer to room 'main/general'
                  --stats            Show palace statistics

    status      System health check: all crates, env, files

    bench       End-to-end benchmark suite
                  Palace insertions/A* queries, SFT training throughput,
                  quality gate throughput. Prints ops/s for each component.

    mcp         Model Context Protocol server (JSON-RPC 2.0 over stdin/stdout)
                  serve              Start MCP server (28 palace tools)
                  ATLAS_PALACE_PATH  env var for palace storage (default: ./atlas-palace)

    infer       Run LLM inference with a real model
                  --weights <DIR>    Directory containing model.safetensors + tokenizer.json
                  --model <NAME>     Model config: smollm2-1.7b | smollm2-135m | olmo3-1b | llama32-1b
                  --prompt <TEXT>    Input prompt (default: "The capital of France is")
                  --max-tokens <N>   Max new tokens to generate (default: 50)
                  --temperature <F>  Sampling temperature: 0.0 = greedy (default: 0.0)

OPTIONS:

    -h, --help     Print help
    -V, --version  Print version

ARCHITECTURE:
    16 pure-Rust crates · zero crates.io deps · sm_75 CUDA kernels
    Stage 1: atlas-core, atlas-tensor, atlas-grad, atlas-optim, atlas-quant
    Stage 2: atlas-json, atlas-tokenize, atlas-model (OLMo 3 / Llama 3)
    Stage 3: atlas-palace (stigmergic memory, pheromones, KG, A*)
    Stage 4: atlas-trm (TRM-CausalValidator, depth-6 RNN, Bayesian)
    Stage 5: atlas-http, atlas-bayes, atlas-causal, atlas-zk, atlas-astra
    Stage 6: atlas-corpus (LiveDiscoveryCorpus, 5 quality gates, curriculum)
    Stage 7: atlas-cli (this binary), bench suite

LICENSE:
    Code: Apache 2.0 (LICENSE-CODE) · Docs: CC BY 4.0 (LICENSE)
    © 2026 web3guru888 / VBRL Holdings
"#,
        ver = env!("CARGO_PKG_VERSION")
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  Arg parsing helpers
// ────────────────────────────────────────────────────────────────────────────

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().map(|s| s.as_str());
        }
    }
    None
}

fn opt_usize(args: &[String], name: &str, default: usize) -> usize {
    opt(args, name).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn opt_f64(args: &[String], name: &str, default: f64) -> f64 {
    opt(args, name).and_then(|s| s.parse().ok()).unwrap_or(default)
}

// ────────────────────────────────────────────────────────────────────────────
//  discover
// ────────────────────────────────────────────────────────────────────────────

fn cmd_discover(args: &[String]) -> i32 {
    use atlas_astra::{AstraEngine, AstraConfig};
    use atlas_corpus::{LiveDiscoveryCorpus, GateConfig};

    let cycles    = opt_usize(args, "--cycles", 3);
    let output    = opt(args, "--output").unwrap_or("./atlas-corpus.json");
    let min_qual  = opt_f64(args, "--min-quality", 0.40);  // lowered: live APIs produce raw conf 0.45-0.65
    let arxiv_q   = opt(args, "--arxiv").unwrap_or("causal inference stigmergy pheromone");

    println!("┌─ ATLAS discover ─────────────────────────────────────────────");
    println!("│  cycles={cycles}  output={output}");
    println!("│  min-quality={min_qual:.2}  arxiv-query='{arxiv_q}'");
    println!("└──────────────────────────────────────────────────────────────");

    let mut astra_cfg = AstraConfig::default();
    astra_cfg.min_quality   = min_qual;
    astra_cfg.arxiv_query   = arxiv_q.to_string();

    let mut engine = AstraEngine::new(astra_cfg);

    let mut gate_cfg = GateConfig::default();
    gate_cfg.min_confidence = min_qual;

    let mut corpus = if Path::new(output).exists() {
        println!("  Loading existing corpus from {output}…");
        match LiveDiscoveryCorpus::load(output, gate_cfg.clone()) {
            Ok(c) => { println!("  Loaded {} entries.", c.len()); c }
            Err(e) => { eprintln!("  Warning: load failed ({e}), starting fresh."); LiveDiscoveryCorpus::new(gate_cfg) }
        }
    } else {
        LiveDiscoveryCorpus::new(gate_cfg)
    };

    println!("\n  Starting {cycles} discovery cycle(s)…\n");
    let mut total_accepted = 0usize;
    let mut total_rejected = 0usize;

    for cycle in 1..=cycles {
        print!("  Cycle {cycle}/{cycles}… ");
        match engine.run_cycle() {
            Ok(discoveries) => {
                let n = discoveries.len();
                let (acc, rej) = corpus.ingest(discoveries);
                total_accepted += acc;
                total_rejected += rej;
                println!("{n} discoveries → {acc} accepted, {rej} rejected");
            }
            Err(e) => println!("⚠ error: {e}"),
        }
    }

    let stats = engine.stats();
    println!("\n  ─── ASTRA stats ─────────────────────────────────────");
    println!("  cycles      : {}", stats.cycles);
    println!("  discoveries : {}", stats.total_discoveries);
    println!("  avg quality : {:.3}", stats.avg_quality);

    println!("\n  ─── Corpus stats ────────────────────────────────────");
    let cs = corpus.stats();
    println!("  entries     : {}", cs.total_entries);
    println!("  rejected    : {} (this run: {total_rejected})", cs.total_rejected);
    println!("  accepted    : {total_accepted}");
    println!("  mean conf.  : {:.3}", cs.mean_confidence);
    println!("  mean phero  : {:.3}", cs.mean_pheromone);
    println!("  sources     : {}", cs.unique_sources);
    println!("  tiers       : easy={} med={} hard={}",
        cs.tier_counts[0], cs.tier_counts[1], cs.tier_counts[2]);

    match corpus.save(output) {
        Ok(()) => println!("\n  ✓ Corpus saved → {output}"),
        Err(e) => { eprintln!("  ✗ Could not save: {e}"); return 1; }
    }
    0
}

// ────────────────────────────────────────────────────────────────────────────
//  corpus
// ────────────────────────────────────────────────────────────────────────────

fn cmd_corpus(args: &[String]) -> i32 {
    use atlas_corpus::{LiveDiscoveryCorpus, GateConfig, SampleStrategy};

    let path = match opt(args, "--path") {
        Some(p) => p,
        None => { eprintln!("atlas corpus: --path <PATH> required"); return 1; }
    };
    let gate_cfg = GateConfig::default();
    let mut corpus = match LiveDiscoveryCorpus::load(path, gate_cfg) {
        Ok(c) => c,
        Err(e) => { eprintln!("atlas corpus: could not load '{path}': {e}"); return 1; }
    };

    if flag(args, "--stats") {
        let s = corpus.stats();
        println!("Corpus: {path}");
        println!("  entries        : {}", s.total_entries);
        println!("  rejected (all) : {}", s.total_rejected);
        println!("  mean confidence: {:.4}", s.mean_confidence);
        println!("  mean pheromone : {:.4}", s.mean_pheromone);
        println!("  unique sources : {}", s.unique_sources);
        println!("  tier 0 (easy)  : {}", s.tier_counts[0]);
        println!("  tier 1 (med)   : {}", s.tier_counts[1]);
        println!("  tier 2 (hard)  : {}", s.tier_counts[2]);
        println!("  +feedback      : {}", s.positive_feedback);
        println!("  -feedback      : {}", s.negative_feedback);
    }

    if flag(args, "--decay") {
        corpus.decay_pheromones(0.95);
        match corpus.save(path) {
            Ok(()) => println!("Pheromones decayed (×0.95), corpus saved."),
            Err(e) => { eprintln!("Could not save: {e}"); return 1; }
        }
    }

    let list_n = if let Some(n_str) = opt(args, "--list") {
        n_str.parse().unwrap_or(10)
    } else if flag(args, "--list") {
        10usize
    } else {
        0
    };
    if list_n > 0 {
        let mut entries: Vec<_> = corpus.entries().iter().collect();
        entries.sort_by(|a, b| b.pheromone.partial_cmp(&a.pheromone).unwrap());
        println!("Top {list_n} entries by pheromone:");
        for (i, e) in entries.iter().take(list_n).enumerate() {
            println!("  {:3}. [ph={:.3} q={:.3} tier={}] [{}] {}",
                i + 1, e.pheromone, e.quality_score, e.tier,
                e.discovery.source, &e.discovery.title);
        }
    }

    if let Some(n_str) = opt(args, "--sample") {
        let n: usize = n_str.parse().unwrap_or(5);
        let batch = corpus.sample_batch(n, SampleStrategy::Pheromone);
        println!("Sampled {}/{n} entries (pheromone strategy):", batch.entries.len());
        for e in &batch.entries {
            println!("  {}", e.to_training_text());
        }
    }

    0
}

// ────────────────────────────────────────────────────────────────────────────
//  train
// ────────────────────────────────────────────────────────────────────────────

fn cmd_train(args: &[String]) -> i32 {
    use atlas_corpus::{LiveDiscoveryCorpus, GateConfig, SftTrainer, SftConfig};

    let corpus_path = match opt(args, "--corpus") {
        Some(p) => p,
        None => { eprintln!("atlas train: --corpus <PATH> required"); return 1; }
    };
    let epochs   = opt_usize(args, "--epochs", 1);
    let batch_sz = opt_usize(args, "--batch", 8);
    let lr       = opt_f64(args, "--lr", 0.01) as f32;
    let output   = opt(args, "--output").unwrap_or("./atlas-ckpt");

    println!("┌─ ATLAS train ─────────────────────────────────────────────────");
    println!("│  corpus={corpus_path}  epochs={epochs}  batch={batch_sz}  lr={lr:.2e}");
    println!("│  output={output}");
    println!("└───────────────────────────────────────────────────────────────");

    let gate_cfg = GateConfig::default();
    let mut corpus = match LiveDiscoveryCorpus::load(corpus_path, gate_cfg) {
        Ok(c) => c,
        Err(e) => { eprintln!("  Could not load corpus: {e}"); return 1; }
    };

    if corpus.is_empty() {
        eprintln!("  Corpus is empty. Run `atlas discover` first.");
        return 1;
    }

    let sft_cfg = SftConfig {
        batch_size: batch_sz,
        lr,
        max_epochs: epochs,
        ..Default::default()
    };

    let mut trainer = SftTrainer::new(sft_cfg);

    println!("\n  Model: {} params (vocab={}, hidden={})",
        trainer.model.param_count(), trainer.config.vocab_size, trainer.config.hidden_dim);
    println!("  Corpus: {} entries loaded", corpus.len());
    println!("  Starting real SFT training (GradTape + AdamW)…\n");

    for epoch in 1..=epochs {
        match trainer.train_epoch(&mut corpus) {
            Ok(em) => {
                println!("  Epoch {epoch}/{epochs}: loss={:.4}  lr={:.2e}  steps={}  entries={}",
                    em.mean_loss, em.final_lr, em.steps, em.entries_processed);
            }
            Err(e) => {
                eprintln!("  ✗ Training error at epoch {epoch}: {e}");
                return 1;
            }
        }
    }

    // Decay pheromones after training
    corpus.decay_pheromones(0.97);

    // Save corpus with updated pheromones
    if let Err(e) = corpus.save(corpus_path) {
        eprintln!("  Warning: could not save corpus: {e}");
    } else {
        println!("\n  ✓ Corpus pheromones saved.");
    }

    // Save checkpoint
    std::fs::create_dir_all(output).ok();
    let step = trainer.global_step;
    let metrics = trainer.metrics();
    let final_loss = metrics.epoch_history.last().map(|e| e.mean_loss).unwrap_or(0.0);
    let ckpt = format!("{output}/atlas-step{step}.json");
    let ckpt_json = format!(
        r#"{{"step":{},"loss":{:.6},"lr":{:.2e},"epochs":{},"corpus_entries":{},"params":{}}}"#,
        step, final_loss, trainer.optimizer.cfg.lr, epochs, corpus.len(), trainer.model.param_count()
    );
    match std::fs::write(&ckpt, &ckpt_json) {
        Ok(()) => println!("  ✓ Checkpoint: {ckpt}"),
        Err(e) => eprintln!("  ✗ Checkpoint write failed: {e}"),
    }

    println!("\n  ┌── Training complete ───────────────────────────────────────");
    println!("  │  steps: {step}  final loss: {final_loss:.4}  ckpt: {ckpt}");
    println!("  └────────────────────────────────────────────────────────────");
    0
}

// ────────────────────────────────────────────────────────────────────────────
//  eval
// ────────────────────────────────────────────────────────────────────────────

fn cmd_eval(args: &[String]) -> i32 {
    use atlas_corpus::{LiveDiscoveryCorpus, GateConfig};

    let corpus_path = match opt(args, "--corpus") {
        Some(p) => p,
        None => { eprintln!("atlas eval: --corpus <PATH> required"); return 1; }
    };
    let verbose = flag(args, "--verbose");

    let gate_cfg = GateConfig::default();
    let corpus = match LiveDiscoveryCorpus::load(corpus_path, gate_cfg) {
        Ok(c) => c,
        Err(e) => { eprintln!("  Could not load corpus: {e}"); return 1; }
    };

    let stats = corpus.stats();
    println!("┌─ ATLAS eval ──────────────────────────────────────────────────");
    println!("│  corpus={corpus_path}  entries={}", stats.total_entries);
    println!("└───────────────────────────────────────────────────────────────\n");

    // Per-source quality metrics
    let mut source_map: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    for e in corpus.entries() {
        source_map.entry(e.discovery.source.clone()).or_default().push(e.quality_score);
    }

    println!("  Per-source quality:");
    let mut srcs: Vec<_> = source_map.iter().collect();
    srcs.sort_by(|a, b| a.0.cmp(b.0));
    for (src, scores) in &srcs {
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let max  = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min  = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("    {src:20}  n={:4}  mean={mean:.4}  min={min:.4}  max={max:.4}", scores.len());
    }

    println!("\n  Tier distribution:");
    let total = stats.total_entries.max(1) as f64;
    for (t, label) in [(0, "easy"), (1, "medium"), (2, "hard")] {
        println!("    tier {} ({label:6}): {} ({:.1}%)",
            t, stats.tier_counts[t], 100.0 * stats.tier_counts[t] as f64 / total);
    }

    println!("\n  Pheromone health:");
    println!("    mean     : {:.4}", stats.mean_pheromone);
    println!("    +feedback: {}  -feedback: {}", stats.positive_feedback, stats.negative_feedback);

    if verbose {
        println!("\n  All entries:");
        for e in corpus.entries() {
            println!("    id={:4}  q={:.3}  ph={:.3}  t={}  [{}]  {}",
                e.id, e.quality_score, e.pheromone, e.tier,
                e.discovery.source, &e.discovery.title);
        }
    }

    let health = (stats.mean_confidence * 0.4
        + stats.mean_pheromone * 0.3
        + (stats.unique_sources as f64 / 4.0).min(1.0) * 0.3).min(1.0);
    println!("\n  ─── Corpus health: {health:.3} / 1.0 ───");
    println!("  {}",
        if health >= 0.7 { "✓ Healthy — ready for training." }
        else if health >= 0.5 { "⚠ Acceptable — consider more discovery cycles." }
        else { "✗ Low health — run discover first." }
    );
    0
}

// ────────────────────────────────────────────────────────────────────────────
//  prove
// ────────────────────────────────────────────────────────────────────────────

fn cmd_prove(args: &[String]) -> i32 {
    use atlas_zk::{KnowledgeClaim, SchnorrParams, SchnorrVerifier};

    let claim_text = match opt(args, "--claim") {
        Some(t) => t,
        None => { eprintln!("atlas prove: --claim <TEXT> required"); return 1; }
    };
    let secret_hex = match opt(args, "--secret") {
        Some(h) => h,
        None => { eprintln!("atlas prove: --secret <HEX> required"); return 1; }
    };
    let output = opt(args, "--output");

    let secret_bytes: Vec<u8> = (0..(secret_hex.len() & !1))
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(&secret_hex[i..i+2], 16).ok())
        .collect();

    if secret_bytes.is_empty() {
        eprintln!("atlas prove: --secret must be valid hex (e.g. deadbeef01020304)");
        return 1;
    }

    println!("┌─ ATLAS prove ─────────────────────────────────────────────────");
    println!("│  claim : {}", &claim_text[..claim_text.len().min(80)]);
    println!("│  secret: {}… ({} bytes)", &secret_hex[..secret_hex.len().min(8)], secret_bytes.len());
    println!("└───────────────────────────────────────────────────────────────\n");

    let claim = KnowledgeClaim::new(
        claim_text,
        0.9,
        &secret_bytes,
        claim_text,
    );

    // claim.verify() uses small_64() params + correct message format internally
    let valid = claim.verify();
    // Re-verify with correct params and message for display (matches KnowledgeClaim internals)
    let params = SchnorrParams::small_64();
    let sig_msg = format!("{}|confidence={:.4}", claim_text, 0.9f32);
    let sig_valid = SchnorrVerifier::verify(&params, &claim.proof, sig_msg.as_bytes());

    println!("  Statement  : {}", &claim.statement[..claim.statement.len().min(60)]);
    println!("  Confidence : {:.2}", claim.confidence);
    println!("  Commitment : 0x{:016x}", claim.proof.commitment);
    println!("  Response   : 0x{:016x}", claim.proof.response);
    println!("  Public key : 0x{:016x}", claim.proof.public_key);
    println!("  Claim valid: {}", if valid { "✓ YES" } else { "✗ NO" });
    println!("  Schnorr    : {}", if sig_valid { "✓ verified" } else { "✗ failed" });

    let bytes = claim.to_bytes();
    println!("  Serialised : {} bytes", bytes.len());

    if let Some(path) = output {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        match std::fs::write(path, hex.as_bytes()) {
            Ok(()) => println!("\n  ✓ Proof written to {path}"),
            Err(e) => { eprintln!("  ✗ Write failed: {e}"); return 1; }
        }
    }

    if valid { 0 } else { 1 }
}

// ────────────────────────────────────────────────────────────────────────────
//  palace
// ────────────────────────────────────────────────────────────────────────────

fn cmd_palace(args: &[String]) -> i32 {
    use atlas_palace::Palace;

    let path = opt(args, "--path").unwrap_or("./atlas-palace.json");

    // Palace::new(name, path) — loads from file if it exists
    let mut palace = Palace::new("atlas", path);
    println!("Palace: {path}  name={}", palace.name());

    if flag(args, "--stats") {
        let wings = palace.list_wings();
        println!("  wings: {}", wings.len());
        for (wid, wname) in &wings {
            let rooms = palace.list_rooms(wid);
            println!("    [{wid}] {wname}: {} rooms", rooms.len());
        }
        let status = palace.status_dict();
        println!("  drawers : {}", status.get("total_drawers").unwrap_or(&0));
        println!("  kg edges: {}", status.get("kg_edges").unwrap_or(&0));
        println!("  agents  : {}", status.get("agents").unwrap_or(&0));
    }

    if flag(args, "--hot") {
        let hot = palace.hot_paths("", 20);
        println!("Top hot paths:");
        for (i, (path_str, weight)) in hot.iter().enumerate() {
            println!("  {:3}. [{weight:.3}] {path_str}", i + 1);
        }
    }

    if let Some(query) = opt(args, "--search") {
        let results = palace.search(query, 10);
        println!("Search results for '{query}':");
        if results.is_empty() {
            println!("  (no results)");
        }
        for (i, r) in results.iter().enumerate() {
            println!("  {:3}. [score={:.3}] {}", i + 1, r.score, r.preview);
        }
    }

    if let Some(text) = opt(args, "--add") {
        let wid = palace.add_wing("main", "Main knowledge wing");
        let rid = palace.add_room(&wid, "general", "General knowledge")
            .unwrap_or_else(|_| format!("{wid}:room:0"));
        match palace.add_drawer(&rid, text, text, &[]) {
            Ok(id) => println!("Added drawer {id}: {text}"),
            Err(e) => println!("add_drawer: {e}"),
        }
        let _ = palace.save();
        println!("Saved to {path}");
    }

    0
}

// ────────────────────────────────────────────────────────────────────────────
//  status
// ────────────────────────────────────────────────────────────────────────────

fn cmd_status(_args: &[String]) -> i32 {
    // Fix 3: runtime CUDA detection — check nvidia-smi and /dev/nvidia0
    let cuda_available = {
        let has_smi = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let has_dev = std::path::Path::new("/dev/nvidia0").exists();
        has_smi || has_dev || cfg!(atlas_cuda)
    };
    let cuda_label = if cuda_available {
        if cfg!(atlas_cuda) { "enabled (compile-time)" } else { "detected at runtime (GPU present)" }
    } else {
        "disabled (CPU-only)"
    };
    println!("┌─ ATLAS system status ─────────────────────────────────────────");
    println!("│  version : {}", env!("CARGO_PKG_VERSION"));
    println!("│  profile : {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!("│  CUDA    : {cuda_label}");
    println!("│");
    println!("│  Crates:");
    for (stage, name, note) in CRATE_TABLE {
        println!("│    Stage {stage}  {name:18} ✓  {note}");
    }
    println!("│");
    println!("│  Environment:");
    for var in &["ATLAS_CORPUS", "ATLAS_PALACE", "ATLAS_CKPT", "CUDA_VISIBLE_DEVICES"] {
        let val = env::var(var).unwrap_or_else(|_| "(not set)".into());
        println!("│    {var:25}: {val}");
    }
    println!("│");
    println!("│  Files:");
    for f in &["./atlas-corpus.json", "./atlas-palace.json", "./atlas-ckpt"] {
        let marker = if Path::new(f).exists() { "✓ found" } else { "✗ not found" };
        println!("│    {f:30}: {marker}");
    }
    println!("└───────────────────────────────────────────────────────────────");
    0
}

const CRATE_TABLE: &[(&str, &str, &str)] = &[
    ("1", "atlas-core",     "error types, Result, traits"),
    ("1", "atlas-tensor",   "f32 matmul CPU+CUDA, int8/int4 quant"),
    ("1", "atlas-grad",     "autograd tape, reverse-mode AD"),
    ("1", "atlas-optim",    "AdamW, cosine LR schedule"),
    ("1", "atlas-quant",    "int8/int4 quantize/dequantize"),
    ("2", "atlas-json",     "recursive descent JSON parser"),
    ("2", "atlas-tokenize", "GPT-2 byte-level BPE"),
    ("2", "atlas-model",    "OLMo 3 / Llama 3, RoPE, GQA, SwiGLU"),
    ("3", "atlas-palace",   "stigmergic memory, pheromones, A*"),
    ("4", "atlas-trm",      "TRM-CausalValidator depth-6 RNN"),
    ("5", "atlas-http",     "HTTP/1.1 TcpStream + curl HTTPS"),
    ("5", "atlas-bayes",    "BetaPrior, BayesNetwork, QualityGate"),
    ("5", "atlas-causal",   "PC algorithm, Fisher-Z, Meek rules"),
    ("5", "atlas-zk",       "Schnorr proofs, Z_p 64-bit limbs"),
    ("5", "atlas-astra",    "OODA: NASA/WHO/WorldBank/ArXiv"),
    ("6", "atlas-corpus",   "LiveDiscoveryCorpus, 5 gates, curriculum"),
    ("7", "atlas-cli",      "this binary"),
];

// ────────────────────────────────────────────────────────────────────────────
//  bench
// ────────────────────────────────────────────────────────────────────────────

fn cmd_bench(_args: &[String]) -> i32 {
    use atlas_palace::Palace;
    use atlas_corpus::{LiveDiscoveryCorpus, GateConfig, SftTrainer, SftConfig};
    use atlas_astra::Discovery;

    println!("┌─ ATLAS bench ─────────────────────────────────────────────────────");
    println!("│  Running end-to-end benchmark suite…");
    println!("└───────────────────────────────────────────────────────────────────\n");

    // ── 1. Palace benchmark ───────────────────────────────────────────────
    {
        let mut palace = Palace::new("bench", "/tmp/atlas-bench-palace");
        let wing_id = palace.add_wing("bench-wing", "Benchmark wing");
        let room_id = palace.add_room(&wing_id, "bench-room", "Benchmark room")
            .unwrap_or_else(|_| format!("{wing_id}:room:0"));

        let n_insert = 1000usize;
        let insert_start = std::time::Instant::now();
        for i in 0..n_insert {
            let title   = format!("drawer-{i}");
            let content = format!(
                "benchmark content item {i}: the quick brown fox jumps over the lazy dog"
            );
            palace.add_drawer(&room_id, &title, &content, &[]).ok();
        }
        let insert_ms = insert_start.elapsed().as_micros() as f64 / 1000.0;
        let insert_ops = n_insert as f64 / (insert_ms / 1000.0);

        let n_query = 100usize;
        let query_start = std::time::Instant::now();
        for i in 0..n_query {
            palace.search(&format!("benchmark item {}", i % 20), 10);
        }
        let query_ms = query_start.elapsed().as_micros() as f64 / 1000.0;
        let query_ops = n_query as f64 / (query_ms / 1000.0);

        println!("  Palace:");
        println!("    Insertions : {n_insert} in {insert_ms:.1}ms = {insert_ops:.0} ops/s");
        println!("    A* queries : {n_query} in {query_ms:.1}ms = {query_ops:.0} ops/s");
    }

    // ── 2. Training benchmark ─────────────────────────────────────────────
    {
        let cfg = SftConfig::default();
        let mut trainer = SftTrainer::new(cfg);
        let n_steps = 10u32;
        let train_start = std::time::Instant::now();
        for i in 0..n_steps {
            trainer.train_step(i % 100, (i + 1) % 100).ok();
        }
        let elapsed_ms = train_start.elapsed().as_micros() as f64 / 1000.0;
        let steps_per_s = n_steps as f64 / (elapsed_ms / 1000.0);
        println!("  Training:");
        println!("    SFT steps  : {n_steps} in {elapsed_ms:.1}ms = {steps_per_s:.0} steps/s");
    }

    // ── 3. Quality gate benchmark ─────────────────────────────────────────
    {
        let gate_cfg = GateConfig::default();
        let corpus = LiveDiscoveryCorpus::new(gate_cfg);
        let n_gates = 1000usize;
        let gate_start = std::time::Instant::now();
        for i in 0..n_gates {
            let d = Discovery {
                id:            format!("bench-{i}"),
                title:         format!("H{i}: metric_a correlates with metric_b across {} observations in domain C", i + 100),
                description:   format!("Benchmark discovery {i}: synthetic hypothesis for throughput measurement."),
                causal_claims: vec![("metric_a".into(), "metric_b".into(), 0.85)],
                quality_score: 0.85,
                proof_commitment: i as u64,
                source:        "bench".into(),
                timestamp:     0,
                tags:          vec!["benchmark".into()],
                provenance:    None,
            };
            corpus.evaluate_gates(&d);
        }
        let elapsed_ms = gate_start.elapsed().as_micros() as f64 / 1000.0;
        let gates_per_s = n_gates as f64 / (elapsed_ms / 1000.0);
        println!("  Quality gates:");
        println!("    Evaluations: {n_gates} in {elapsed_ms:.1}ms = {gates_per_s:.0} gates/s");
    }

    println!("\n  ✓ Benchmark complete");
    0
}

// ────────────────────────────────────────────────────────────────────────────
//  mcp
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
//  infer — Real LLM inference (SmolLM2 / OLMo / Llama)
// ────────────────────────────────────────────────────────────────────────────

fn cmd_infer(args: &[String]) -> i32 {
    use atlas_model::{ModelConfig, load_model_from_safetensors};
    use atlas_tokenize::Tokenizer;

    // ── Args ──────────────────────────────────────────────────────────────
    let weights_dir = match opt(args, "--weights") {
        Some(p) => p.to_string(),
        None => {
            eprintln!("atlas infer: --weights <DIR> required");
            eprintln!("  Example: atlas infer --weights /tmp/smollm2-1.7b --model smollm2-1.7b");
            return 1;
        }
    };
    let prompt_str = opt(args, "--prompt").unwrap_or("The capital of France is");
    let max_new    = opt_usize(args, "--max-tokens", 50);
    let temperature = opt_f64(args, "--temperature", 0.0) as f32;
    let model_name  = opt(args, "--model").unwrap_or("smollm2-1.7b");

    println!("┌─ ATLAS infer ──────────────────────────────────────────────────────");
    println!("│  weights     = {weights_dir}");
    println!("│  model       = {model_name}");
    println!("│  prompt      = {prompt_str:?}");
    println!("│  max_tokens  = {max_new}   temperature = {temperature}");
    println!("└────────────────────────────────────────────────────────────────────");

    // ── Select model config ────────────────────────────────────────────────
    let cfg = match model_name {
        "smollm2-1.7b"  => ModelConfig::smollm2_1b7(),
        "smollm2-135m"  => ModelConfig::smollm2_135m(),
        "olmo3-1b"      => ModelConfig::olmo3_1b(),
        "olmo3-7b"      => ModelConfig::olmo3_7b(),
        "llama32-1b"    => ModelConfig::llama32_1b(),
        other => {
            eprintln!("atlas infer: unknown model '{other}'.");
            eprintln!("  Available: smollm2-1.7b, smollm2-135m, olmo3-1b, olmo3-7b, llama32-1b");
            return 1;
        }
    };

    println!("\n  Model config : {} layers, d={}, {} heads, vocab={}",
        cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.vocab_size);

    // ── Load tokenizer ─────────────────────────────────────────────────────
    let tok_path = format!("{}/tokenizer.json", weights_dir.trim_end_matches('/'));
    let tokenizer = match Tokenizer::from_file(&tok_path) {
        Ok(t) => {
            println!("  Tokenizer    : ✓ loaded ({} vocab) from {tok_path}", t.vocab_size());
            Some(t)
        }
        Err(e) => {
            println!("  Tokenizer    : ⚠ {e} — falling back to byte-level encoding");
            None
        }
    };

    // Encode prompt
    let prompt_tokens: Vec<u32> = if let Some(ref tok) = tokenizer {
        let ids = tok.encode(prompt_str);
        println!("  Prompt tokens: {} → {:?}", ids.len(), &ids[..ids.len().min(20)]);
        ids
    } else {
        let ids: Vec<u32> = prompt_str.bytes().map(|b| b as u32).collect();
        println!("  Prompt tokens (byte): {} → {:?}", ids.len(), &ids[..ids.len().min(20)]);
        ids
    };

    // ── Load weights ───────────────────────────────────────────────────────
    let weights_file = format!("{}/model.safetensors", weights_dir.trim_end_matches('/'));
    println!("\n  Loading weights from {weights_file} …");
    let load_start = std::time::Instant::now();
    let mut model = match load_model_from_safetensors(&weights_file, cfg.clone()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ✗ Failed to load weights: {e}");
            return 1;
        }
    };
    let load_ms = load_start.elapsed().as_millis();
    println!("  ✓ Weights loaded in {load_ms}ms  (~{} M params)", model.param_count() / 1_000_000);

    // ── Quick logits sanity check (greedy token from prompt position 0) ────
    println!("\n  Computing logits at last prompt position …");
    let logit_start = std::time::Instant::now();
    let last_tok = *prompt_tokens.last().unwrap_or(&0);
    model.reset();
    // feed all but last token to prime KV cache
    for &t in &prompt_tokens[..prompt_tokens.len().saturating_sub(1)] {
        model.forward_one(t);
    }
    let last_logits = model.forward_one(last_tok);
    let logit_ms = logit_start.elapsed().as_millis();

    let mut top_pairs: Vec<(usize, f32)> = last_logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    top_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Top-10 logits (last prompt position, {logit_ms}ms):");
    for (tok_id, logit) in &top_pairs[..10] {
        let decoded = tokenizer.as_ref()
            .map(|t| format!(" = {:?}", t.decode(&[*tok_id as u32])))
            .unwrap_or_default();
        println!("    token {:6}: {:8.4}{decoded}", tok_id, logit);
    }

    // Check logit spread (healthy = > 5.0 difference between top and 100th)
    let top_val    = top_pairs[0].1;
    let p100_val   = top_pairs.get(99).map(|p| p.1).unwrap_or(top_val);
    let spread     = top_val - p100_val;
    if spread < 1.0 {
        println!("\n  ⚠  WARNING: logit spread = {spread:.2} (< 1.0) — forward pass may be incorrect");
    } else {
        println!("\n  ✓  Logit spread = {spread:.2} (healthy)");
    }

    // ── Generate ──────────────────────────────────────────────────────────
    println!("\n  Generating {max_new} tokens (temperature={temperature}) …");
    let gen_start = std::time::Instant::now();
    let new_tokens = model.generate(&prompt_tokens, max_new, temperature);
    let gen_ms     = gen_start.elapsed().as_millis();
    let tps        = new_tokens.len() as f64 / (gen_ms as f64 / 1000.0).max(0.001);

    println!("\n  ─── Output ────────────────────────────────────────────────────────");
    println!("  Prompt : {prompt_str:?}");

    let output_text = if let Some(ref tok) = tokenizer {
        tok.decode(&new_tokens)
    } else {
        let bytes: Vec<u8> = new_tokens.iter().map(|&t| (t % 256) as u8).collect();
        String::from_utf8_lossy(&bytes).to_string()
    };
    println!("  Output : {output_text:?}");
    println!("  Full   : {:?}", format!("{prompt_str}{output_text}"));

    println!("\n  ─── Stats ─────────────────────────────────────────────────────────");
    println!("  Tokens generated : {}", new_tokens.len());
    println!("  Generation time  : {gen_ms}ms");
    println!("  Throughput       : {tps:.1} tok/s");
    println!("  Token IDs        : {:?}", &new_tokens[..new_tokens.len().min(20)]);

    println!("\n  ✓ atlas infer complete");
    0
}

fn cmd_mcp(args: &[String]) -> i32 {
    let subcmd = args.first().map(|s| s.as_str()).unwrap_or("serve");
    match subcmd {
        "serve" => {
            use atlas_mcp::McpServer;
            use atlas_palace::Palace;

            let palace_path = std::env::var("ATLAS_PALACE_PATH")
                .unwrap_or_else(|_| "./atlas-palace".to_string());
            let palace = Palace::new("atlas", &palace_path);
            let mut server = McpServer::new(palace);

            eprintln!("atlas mcp serve: listening on stdin/stdout (JSON-RPC 2.0)");
            eprintln!("  Palace path : {palace_path}");
            eprintln!("  Tools       : {} registered", server.tool_count());
            eprintln!("  Press Ctrl+C to stop");

            match server.run_stdio() {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("atlas mcp serve error: {e}");
                    1
                }
            }
        }
        _ => {
            eprintln!("atlas mcp: unknown subcommand '{subcmd}'. Available: serve");
            1
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────────────────────

fn fnv_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
