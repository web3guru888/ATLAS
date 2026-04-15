//! Simple benchmark harness — zero external dependencies.
//!
//! Since ATLAS has no external crates (the SQLite principle), we implement
//! our own timing-based benchmark system using `std::time::Instant`.
//!
//! # Usage
//! ```rust
//! use atlas_core::bench::Bench;
//! let b = Bench::run("my_op", 1000, || {
//!     let _ = std::hint::black_box(42 * 42);
//! });
//! println!("{}", b.report());
//! ```

/// A single benchmark result.
#[derive(Debug, Clone)]
pub struct Bench {
    /// Benchmark name.
    pub name: String,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Total elapsed time in nanoseconds.
    pub elapsed_ns: u64,
}

impl Bench {
    /// Run a benchmark: execute `f` for `iterations` times and measure wall time.
    ///
    /// Uses `std::hint::black_box` on the closure result to prevent the compiler
    /// from optimizing away the measured work.
    pub fn run<F: FnMut()>(name: &str, iterations: usize, mut f: F) -> Self {
        // Warm-up: run 10% of iterations (min 1) to stabilise caches
        let warmup = (iterations / 10).max(1);
        for _ in 0..warmup {
            f();
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            f();
        }
        let elapsed = start.elapsed();

        Self {
            name: name.to_string(),
            iterations,
            elapsed_ns: elapsed.as_nanos() as u64,
        }
    }

    /// Operations per second.
    pub fn ops_per_sec(&self) -> f64 {
        if self.elapsed_ns == 0 {
            return f64::INFINITY;
        }
        self.iterations as f64 / (self.elapsed_ns as f64 / 1_000_000_000.0)
    }

    /// Nanoseconds per operation.
    pub fn ns_per_op(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        self.elapsed_ns as f64 / self.iterations as f64
    }

    /// Human-readable report string.
    pub fn report(&self) -> String {
        let ns = self.ns_per_op();
        let (value, unit) = if ns < 1_000.0 {
            (ns, "ns")
        } else if ns < 1_000_000.0 {
            (ns / 1_000.0, "µs")
        } else if ns < 1_000_000_000.0 {
            (ns / 1_000_000.0, "ms")
        } else {
            (ns / 1_000_000_000.0, "s")
        };
        format!(
            "{}: {:.1} {}/op ({:.0} ops/s, {} iters, {:.1}ms total)",
            self.name,
            value,
            unit,
            self.ops_per_sec(),
            self.iterations,
            self.elapsed_ns as f64 / 1_000_000.0,
        )
    }
}

/// Run a suite of benchmarks and return a combined report.
pub fn run_suite(benchmarks: Vec<Bench>) -> String {
    let mut lines = Vec::new();
    lines.push("╔══════════════════════════════════════════════════════════════╗".to_string());
    lines.push("║              ATLAS Benchmark Suite                          ║".to_string());
    lines.push("╠══════════════════════════════════════════════════════════════╣".to_string());
    for b in &benchmarks {
        lines.push(format!("║  {:<58}║", b.report()));
    }
    lines.push("╚══════════════════════════════════════════════════════════════╝".to_string());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_harness_works() {
        let b = Bench::run("noop", 100, || {
            std::hint::black_box(1 + 1);
        });
        assert_eq!(b.name, "noop");
        assert_eq!(b.iterations, 100);
        assert!(b.elapsed_ns > 0, "elapsed should be non-zero");
    }

    #[test]
    fn bench_harness_ops_per_sec() {
        let b = Bench {
            name: "test".to_string(),
            iterations: 1_000_000,
            elapsed_ns: 1_000_000_000, // 1 second
        };
        let ops = b.ops_per_sec();
        assert!((ops - 1_000_000.0).abs() < 1.0,
                "expected 1M ops/s, got {ops}");
    }

    #[test]
    fn bench_harness_ns_per_op() {
        let b = Bench {
            name: "test".to_string(),
            iterations: 1000,
            elapsed_ns: 1_000_000, // 1ms total
        };
        let ns = b.ns_per_op();
        assert!((ns - 1000.0).abs() < 0.001,
                "expected 1000 ns/op, got {ns}");
    }

    #[test]
    fn bench_report_format() {
        let b = Bench {
            name: "test_op".to_string(),
            iterations: 1000,
            elapsed_ns: 5_000_000, // 5ms
        };
        let report = b.report();
        assert!(report.contains("test_op"));
        assert!(report.contains("µs/op"));
        assert!(report.contains("ops/s"));
    }

    #[test]
    fn bench_suite_report() {
        let benchmarks = vec![
            Bench { name: "a".into(), iterations: 100, elapsed_ns: 1000 },
            Bench { name: "b".into(), iterations: 200, elapsed_ns: 2000 },
        ];
        let report = run_suite(benchmarks);
        assert!(report.contains("ATLAS Benchmark Suite"));
        assert!(report.contains("a:"));
        assert!(report.contains("b:"));
    }

    #[test]
    fn bench_zero_iterations() {
        let b = Bench {
            name: "zero".to_string(),
            iterations: 0,
            elapsed_ns: 0,
        };
        assert_eq!(b.ns_per_op(), 0.0);
    }
}
