//! ATLAS Palace benchmark tests.
//!
//! Run with: `cargo test -p atlas-palace --test benchmarks -- --ignored --nocapture`

use atlas_core::bench::Bench;
use atlas_palace::Palace;

/// Build a palace with `n` drawers for benchmarking.
fn bench_palace(n: usize) -> Palace {
    let mut p = Palace::new("bench", "/tmp/atlas-bench");
    let w = p.add_wing("bench_wing", "Benchmark wing");
    let r = p.add_room(&w, "bench_room", "Benchmark room").unwrap();
    for i in 0..n {
        let content = format!(
            "drawer {i} content about topic_{} with keywords alpha beta gamma delta \
             epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma \
             tau upsilon phi chi psi omega variant_{}", i % 50, i % 7
        );
        p.add_drawer(
            &r,
            &format!("drawer_{i}"),
            &content,
            &["bench", &format!("tag_{}", i % 10)],
        ).unwrap();
    }
    p
}

/// Build a palace with KG edges forming a chain graph.
fn bench_palace_with_kg(n_nodes: usize) -> Palace {
    let mut p = bench_palace(n_nodes);
    let ids: Vec<String> = {
        // Collect drawer ids in a stable order
        let mut v: Vec<_> = p.status_dict().keys().cloned().collect();
        v.clear();
        // Use search to get IDs - grab them from status_dict won't work
        // Let's just use a simpler approach: search for each drawer
        let results = p.search("drawer", n_nodes);
        results.into_iter().map(|r| r.drawer_id).collect()
    };
    // Build a chain: 0→1→2→...→(n-1)
    for pair in ids.windows(2) {
        p.kg_add(&pair[0], &pair[1], "leads_to", 0.8);
    }
    // Add some cross-links for A* variety
    for i in (0..ids.len()).step_by(5) {
        if i + 3 < ids.len() {
            p.kg_add(&ids[i], &ids[i + 3], "similar", 0.6);
        }
    }
    p
}

#[test]
#[ignore]
fn bench_search_1000_drawers() {
    let p = bench_palace(1000);
    let b = Bench::run("palace_search_1000", 100, || {
        std::hint::black_box(p.search("pheromone stigmergy alpha", 10));
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_astar_100_nodes() {
    let p = bench_palace_with_kg(100);
    let results = p.search("drawer", 100);
    let ids: Vec<String> = results.into_iter().map(|r| r.drawer_id).collect();
    let start = ids.first().unwrap().clone();
    let goal = ids.last().unwrap().clone();

    let b = Bench::run("astar_100_nodes", 50, || {
        std::hint::black_box(p.navigate(&start, &goal, 200));
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_pheromone_deposit_decay_cycle() {
    let mut p = bench_palace(100);
    let ids: Vec<String> = {
        let results = p.search("drawer", 100);
        results.into_iter().map(|r| r.drawer_id).collect()
    };

    let b = Bench::run("pheromone_deposit_decay_1000", 1000, || {
        // Deposit on first 10 drawers
        for id in ids.iter().take(10) {
            p.deposit_pheromones(id, 0.5, 0.1, "bench");
        }
        // Decay all
        p.decay_pheromones();
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_kg_query_100_edges() {
    let p = bench_palace_with_kg(100);
    let results = p.search("drawer", 100);
    let ids: Vec<String> = results.into_iter().map(|r| r.drawer_id).collect();
    let query_id = ids.first().unwrap().clone();

    let b = Bench::run("kg_query_100_edges", 500, || {
        std::hint::black_box(p.kg_query(&query_id));
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_add_drawer_if_unique() {
    let mut p = bench_palace(500);
    let w = p.add_wing("unique_bench", "");
    let r = p.add_room(&w, "unique_room", "").unwrap();
    // Pre-populate with some drawers
    for i in 0..50 {
        p.add_drawer(&r, &format!("u_{i}"), &format!("unique content {i}"), &[]).unwrap();
    }

    let b = Bench::run("add_drawer_if_unique_50", 200, || {
        let _ = std::hint::black_box(
            p.add_drawer_if_unique(&r, "test", "unique content 25", &[], 0.8)
        );
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_build_similarity_graph() {
    let mut p = bench_palace(100);
    let b = Bench::run("similarity_graph_100", 10, || {
        // Reset KG before each run
        let n = std::hint::black_box(p.build_similarity_graph(0.3));
        std::hint::black_box(n);
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_hot_paths_1000() {
    let mut p = bench_palace(1000);
    let ids: Vec<String> = {
        let results = p.search("drawer", 100);
        results.into_iter().map(|r| r.drawer_id).collect()
    };
    // Deposit pheromones on some
    for (i, id) in ids.iter().enumerate().take(50) {
        p.deposit_pheromones(id, (i as f32 + 1.0) * 0.02, 0.05, "hot_bench");
    }

    let b = Bench::run("hot_paths_1000_drawers", 200, || {
        std::hint::black_box(p.hot_paths("hot_bench", 20));
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}
