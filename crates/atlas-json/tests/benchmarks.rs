//! ATLAS JSON parser benchmark tests.
//!
//! Run with: `cargo test -p atlas-json --test benchmarks -- --ignored --nocapture`

use atlas_core::bench::Bench;
use atlas_json::Json;

/// Build a ~1KB JSON document with realistic structure.
fn make_1kb_json() -> String {
    let mut entries = Vec::new();
    for i in 0..10 {
        let val = 400 + i;
        let frac = i * 3;
        entries.push(format!(
            r#"{{"id":{i},"name":"discovery_{i}","confidence":0.{i}7,"tags":["causal","validated"],"source":{{"api":"NASA","endpoint":"/climate/v2/data","params":{{"year":202{i},"metric":"co2_ppm"}}}},"result":{{"value":{val}.{frac},"unit":"ppm","trend":"increasing"}}}}"#
        ));
    }
    format!("[{}]", entries.join(","))
}

/// Build a ~10KB deeply nested JSON.
fn make_10kb_json() -> String {
    let mut obj = String::from("{");
    for i in 0..50 {
        if i > 0 { obj.push(','); }
        obj.push_str(&format!(
            r#""key_{i}":{{"value":{i},"nested":{{"a":[1,2,3,4,5],"b":"string value for testing {i}","c":null,"d":true,"e":{{"deep":{}}}}},"tags":["alpha","beta","gamma","delta","epsilon"]}}"#,
            i * 7
        ));
    }
    obj.push('}');
    obj
}

#[test]
#[ignore]
fn bench_json_parse_1kb() {
    let input = make_1kb_json();
    assert!(input.len() >= 900, "input should be ~1KB, got {} bytes", input.len());

    let b = Bench::run("json_parse_1kb", 5_000, || {
        std::hint::black_box(Json::parse(&input).unwrap());
    });
    eprintln!("{}", b.report());
    eprintln!("  → input size: {} bytes", input.len());
    eprintln!("  → throughput: {:.1} MB/s",
              input.len() as f64 / (b.ns_per_op() / 1e9) / 1e6);
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_json_parse_10kb() {
    let input = make_10kb_json();
    assert!(input.len() >= 5_000, "input should be ~10KB, got {} bytes", input.len());

    let b = Bench::run("json_parse_10kb", 1_000, || {
        std::hint::black_box(Json::parse(&input).unwrap());
    });
    eprintln!("{}", b.report());
    eprintln!("  → input size: {} bytes", input.len());
    eprintln!("  → throughput: {:.1} MB/s",
              input.len() as f64 / (b.ns_per_op() / 1e9) / 1e6);
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_json_serialize() {
    let input = make_1kb_json();
    let val = Json::parse(&input).unwrap();

    let b = Bench::run("json_serialize_1kb", 5_000, || {
        std::hint::black_box(val.to_json());
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}

#[test]
#[ignore]
fn bench_json_access_patterns() {
    let input = make_10kb_json();
    let val = Json::parse(&input).unwrap();

    let b = Bench::run("json_object_get_50_keys", 10_000, || {
        for i in 0..50 {
            let key = format!("key_{i}");
            std::hint::black_box(val.get(&key));
        }
    });
    eprintln!("{}", b.report());
    assert!(b.ns_per_op() > 0.0);
}
