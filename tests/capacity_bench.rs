// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! The capacity benchmark harness. One run anneals one model on one kernel
//! and writes one JSON object that says what the sweep cost per spin update.
//! The calibration of the chunk planner reads these records.
//!
//! ```sh
//! cargo test --release --test capacity_bench the_harness_writes_one_json_record
//! QUIP_BENCH_KERNEL=sa QUIP_BENCH_NODES=2000 QUIP_BENCH_DEGREE=20 \
//!   QUIP_BENCH_PRECISION=int8 QUIP_BENCH_OUT=run.jsonl \
//!   cargo test --release --test capacity_bench capacity_run -- --ignored --nocapture
//! ```
//!
//! # Variables
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `QUIP_BENCH_KERNEL` | `sa`, `gibbs` or `msa` | required |
//! | `QUIP_BENCH_NODES` | node count | required |
//! | `QUIP_BENCH_DEGREE` | neighbours per node, or `complete` | required |
//! | `QUIP_BENCH_PRECISION` | `binary`, `int8`, `int32` or `float` | `int8` |
//! | `QUIP_BENCH_STORAGE` | `auto`, `sparse` or `dense` | `auto` |
//! | `QUIP_BENCH_READS` | reads asked for, at most 256 | `256` |
//! | `QUIP_BENCH_SWEEPS` | sweeps of the whole schedule | `256` |
//! | `QUIP_BENCH_OUT` | file for the record | the record goes to stdout |
//!
//! `QUIP_BENCH_STORAGE=sparse` keeps the batch on CSR storage, which shows
//! the planner that ignores degree. `QUIP_BENCH_SWEEPS` caps the length of a
//! run whose planned chunks are long.
//!
//! # Record format
//!
//! The record is one JSON object on one line. When `QUIP_BENCH_OUT` names a
//! file, the line is appended to it, so the file is JSON Lines. Fields:
//!
//! - `status`: `ok`, or `refused` when the admission refused the model. A
//!   refused record carries only the identity fields and `error`.
//! - `kernel`: `sa`, `gibbs` or `msa`.
//! - `nodes`, `edges`: model size.
//! - `degree`: the requested neighbours per node as a number, or `"complete"`.
//! - `precision`: the requested coefficient precision.
//! - `storage_requested`: `auto`, `sparse` or `dense`.
//! - `storage`: the tier that ran, in the kernel's own names (`local`,
//!   `global`, `dense` for SA; `shared`, `spill`, `dense` for Gibbs and MSA).
//! - `reads_requested`, `reads_dispatched`: the reads asked for, and that
//!   count rounded up to a SIMD group of 32 and capped at 256, which is what
//!   the dispatch computes.
//! - `reads_returned`: reads that the call returned, which is `reads_requested`.
//! - `sweeps`: sweeps that the plan ran, per read.
//! - `chunks`: number of command buffers.
//! - `max_chunk_ms`: GPU time of the longest command buffer.
//! - `max_chunk_sweeps`: sweeps of the chunk that took the longest.
//! - `mean_ms_per_sweep`: `gpu_ms / sweeps`, the mean sweep time. The
//!   degree term of the planner calibrates on this value.
//! - `worst_ms_per_sweep`: the largest `ms / sweeps` over the chunks in
//!   `chunk_log`. The one-sweep refusal calibrates on this value, because the
//!   mean hides a slow sweep such as the hot start of SA.
//! - `chunk_log`: one object per command buffer, in order: `ms` (GPU time),
//!   `sweeps` (`beta_count * sweep_count`), `beta_start` and `sweep_start`.
//! - `gpu_ms`: GPU time of the whole run.
//! - `ns_per_update`: `gpu_ms` over `reads_dispatched * nodes * sweeps`, in
//!   nanoseconds. A spin update is one visit of one spin of one read.
//! - `wall_ms`: wall time of the call, which includes host scoring.
//! - `commit`: the commit SHA of the checkout, and `src_dirty`: whether
//!   `src`, `kernels` or `Cargo.toml` differ from that commit.

use std::collections::BTreeMap;
use std::env;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use quip_solver_core::SampleParams;
use quip_solver_metal::capacity::Storage;
use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{sample_ising_tuned, Kernel, Tuning};
use quip_solver_metal::IsingGraph;
use serde_json::{json, Value};

/// The dispatch cap on reads.
const MAX_READS: usize = 256;
/// Threads in a SIMD group.
const SIMD_WIDTH: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precision {
    Binary,
    Int8,
    Int32,
    Float,
}

impl Precision {
    fn parse(text: &str) -> Self {
        match text {
            "binary" => Self::Binary,
            "int8" => Self::Int8,
            "int32" => Self::Int32,
            "float" => Self::Float,
            other => panic!("QUIP_BENCH_PRECISION: unknown precision {other}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Degree {
    Regular(usize),
    Complete,
}

#[derive(Clone, Debug)]
struct Config {
    kernel: String,
    nodes: usize,
    degree: Degree,
    precision: Precision,
    storage: String,
    reads: usize,
    sweeps: usize,
}

impl Config {
    fn from_env() -> Self {
        let get = |name: &str| env::var(name).ok();
        let need = |name: &str| get(name).unwrap_or_else(|| panic!("{name} is required"));
        let degree = need("QUIP_BENCH_DEGREE");
        let number = |name: &str, text: &str| -> usize {
            text.parse()
                .unwrap_or_else(|_| panic!("{name}: {text} is not a number"))
        };
        Self {
            kernel: need("QUIP_BENCH_KERNEL"),
            nodes: number("QUIP_BENCH_NODES", &need("QUIP_BENCH_NODES")),
            degree: if degree == "complete" {
                Degree::Complete
            } else {
                Degree::Regular(number("QUIP_BENCH_DEGREE", &degree))
            },
            precision: Precision::parse(
                &get("QUIP_BENCH_PRECISION").unwrap_or_else(|| "int8".into()),
            ),
            storage: get("QUIP_BENCH_STORAGE").unwrap_or_else(|| "auto".into()),
            reads: get("QUIP_BENCH_READS").map_or(MAX_READS, |t| number("QUIP_BENCH_READS", &t)),
            sweeps: get("QUIP_BENCH_SWEEPS").map_or(256, |t| number("QUIP_BENCH_SWEEPS", &t)),
        }
    }
}

/// Coefficient magnitudes of `precision`, picked by `k`.
fn magnitude(precision: Precision, k: usize, field: bool) -> f64 {
    let (couplings, fields): (&[f64], &[f64]) = match precision {
        Precision::Binary => (&[1.0], &[1.0, 0.0]),
        Precision::Int8 => (&[0.5, 1.0, 1.5], &[0.5, 0.0, 1.0]),
        Precision::Int32 => (&[1.3, 0.127, 2.0], &[0.5, 0.0, 0.75]),
        Precision::Float => (&[1.300_65, 0.127_063_5, 2.001], &[0.500_25, 0.0, 0.750_375]),
    };
    let values = if field { fields } else { couplings };
    values[k % values.len()]
}

/// A deterministic mix of two node indices, so signs and magnitudes look
/// random but repeat from run to run.
fn mix(u: usize, v: usize) -> usize {
    let mut x = (u as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((v as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x >> 1) as usize
}

fn signed(precision: Precision, u: usize, v: usize, field: bool) -> f64 {
    let hash = mix(u, v);
    let value = magnitude(precision, hash >> 8, field);
    if hash & 1 == 0 {
        value
    } else {
        -value
    }
}

/// A frustrated model: random-sign couplings of `precision` on a circulant
/// graph (node `i` links to `i + 1 ..= i + degree / 2`), or on the complete
/// graph. Every node has `degree` neighbours when `nodes > degree`.
fn build_graph(nodes: usize, degree: Degree, precision: Precision) -> IsingGraph {
    let mut edges = Vec::new();
    match degree {
        Degree::Complete => {
            edges.reserve(nodes * nodes.saturating_sub(1) / 2);
            for u in 0..nodes {
                for v in u + 1..nodes {
                    edges.push((u, v));
                }
            }
        }
        Degree::Regular(degree) => {
            assert!(nodes > degree, "a degree-{degree} model needs more nodes");
            edges.reserve(nodes * (degree / 2));
            for i in 0..nodes {
                for d in 1..=degree / 2 {
                    edges.push((i, (i + d) % nodes));
                }
            }
        }
    }
    let j = edges
        .iter()
        .map(|&(u, v)| signed(precision, u, v, false))
        .collect();
    let h = (0..nodes)
        .map(|i| signed(precision, i, usize::MAX, true))
        .collect();
    IsingGraph::new(h, j, edges)
}

fn parse_kernel(name: &str) -> Kernel {
    match name {
        "sa" => Kernel::Sa,
        "gibbs" => Kernel::Gibbs,
        "msa" => Kernel::Msa,
        other => panic!("QUIP_BENCH_KERNEL: unknown kernel {other}"),
    }
}

fn git(args: &[&str]) -> Option<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
}

fn commit_sha() -> Value {
    git(&["rev-parse", "HEAD"])
        .filter(|o| o.status.success())
        .map_or(Value::Null, |o| {
            json!(String::from_utf8_lossy(&o.stdout).trim())
        })
}

fn src_dirty() -> Value {
    git(&[
        "status",
        "--porcelain",
        "--",
        "src",
        "kernels",
        "Cargo.toml",
    ])
    .filter(|o| o.status.success())
    .map_or(Value::Null, |o| json!(!o.stdout.is_empty()))
}

/// Runs one benchmark and returns its record.
fn run_bench(dev: &MetalDevice, cfg: &Config) -> Value {
    let kernel = parse_kernel(&cfg.kernel);
    let graph = build_graph(cfg.nodes, cfg.degree, cfg.precision);
    let tuning = match cfg.storage.as_str() {
        "auto" => Tuning::default(),
        "sparse" => Tuning {
            force_sparse: true,
            ..Tuning::default()
        },
        "dense" => Tuning {
            force_dense: true,
            ..Tuning::default()
        },
        other => panic!("QUIP_BENCH_STORAGE: unknown storage {other}"),
    };
    let reads_asked = cfg.reads.clamp(1, MAX_READS);
    // The same rule as `simd_rounded_reads` in `src/sampler.rs`, which is
    // private to the crate: the record would go stale if that rule changed,
    // so `the_harness_rounds_reads_to_a_simd_group` checks one case.
    let reads_dispatched = reads_asked
        .div_ceil(SIMD_WIDTH)
        .saturating_mul(SIMD_WIDTH)
        .min(MAX_READS);
    let params = SampleParams {
        num_reads: reads_asked,
        num_sweeps: cfg.sweeps,
        sweeps_per_beta: (cfg.sweeps / 64).max(1),
        seed: 9,
        ..SampleParams::default()
    };

    let mut record = BTreeMap::new();
    let mut put = |key: &str, value: Value| {
        record.insert(key.to_string(), value);
    };
    put("kernel", json!(cfg.kernel));
    put("nodes", json!(cfg.nodes));
    put("edges", json!(graph.edges.len()));
    put(
        "degree",
        match cfg.degree {
            Degree::Complete => json!("complete"),
            Degree::Regular(d) => json!(d),
        },
    );
    put(
        "precision",
        json!(format!("{:?}", cfg.precision).to_lowercase()),
    );
    put("storage_requested", json!(cfg.storage));
    put("reads_requested", json!(reads_asked));
    put("commit", commit_sha());
    put("src_dirty", src_dirty());

    let started = Instant::now();
    match sample_ising_tuned(dev, &graph, &params, kernel, tuning) {
        Ok(run) => {
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            let chunk_sweeps = |c: &quip_solver_metal::sampler::Chunk| {
                i64::from(c.beta_count) * i64::from(c.sweep_count)
            };
            let sweeps: i64 = run.plan.iter().map(chunk_sweeps).sum();
            let chunk_log: Vec<Value> = run
                .plan
                .iter()
                .zip(&run.chunk_us)
                .map(|(c, us)| {
                    json!({
                        "ms": *us as f64 / 1e3,
                        "sweeps": chunk_sweeps(c),
                        "beta_start": c.beta_start,
                        "sweep_start": c.sweep_start,
                    })
                })
                .collect();
            // The slowest chunk per sweep: the worst sweep time the plan met.
            let worst_ms_per_sweep = run
                .plan
                .iter()
                .zip(&run.chunk_us)
                .map(|(c, us)| *us as f64 / 1e3 / chunk_sweeps(c).max(1) as f64)
                .fold(0.0, f64::max);
            let longest_sweeps = run
                .plan
                .iter()
                .zip(&run.chunk_us)
                .max_by_key(|(_, us)| **us)
                .map_or(0, |(c, _)| chunk_sweeps(c));
            let updates = reads_dispatched as f64 * cfg.nodes as f64 * sweeps as f64;
            let ns_per_update = run.gpu_time_us as f64 * 1e3 / updates.max(1.0);
            put("status", json!("ok"));
            put("storage", json!(run.storage.name(kernel)));
            put("reads_dispatched", json!(reads_dispatched));
            put("reads_returned", json!(run.reads.len()));
            put("sweeps", json!(sweeps));
            put("chunks", json!(run.plan.len()));
            put("max_chunk_ms", json!(run.max_chunk_us as f64 / 1e3));
            put("max_chunk_sweeps", json!(longest_sweeps));
            put(
                "mean_ms_per_sweep",
                json!(run.gpu_time_us as f64 / 1e3 / sweeps.max(1) as f64),
            );
            put("worst_ms_per_sweep", json!(worst_ms_per_sweep));
            put("chunk_log", json!(chunk_log));
            put("gpu_ms", json!(run.gpu_time_us as f64 / 1e3));
            put("ns_per_update", json!(ns_per_update));
            put("wall_ms", json!(wall_ms));
        }
        Err(err) => {
            put("status", json!("refused"));
            put("error", json!(err.to_string()));
        }
    }
    Value::Object(record.into_iter().collect())
}

/// Appends `record` to `path`, or writes it to stdout without one.
fn emit(record: &Value, path: Option<&Path>) {
    let line = serde_json::to_string(record).expect("a record serializes");
    match path {
        Some(path) => {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("the output file opens");
            writeln!(file, "{line}").expect("the record is written");
        }
        None => writeln!(io::stdout().lock(), "{line}").expect("stdout takes the record"),
    }
}

/// Fields of an `ok` record.
const OK_FIELDS: [&str; 23] = [
    "status",
    "kernel",
    "nodes",
    "edges",
    "degree",
    "precision",
    "storage_requested",
    "storage",
    "reads_requested",
    "reads_dispatched",
    "reads_returned",
    "sweeps",
    "chunks",
    "max_chunk_ms",
    "max_chunk_sweeps",
    "mean_ms_per_sweep",
    "worst_ms_per_sweep",
    "chunk_log",
    "gpu_ms",
    "ns_per_update",
    "wall_ms",
    "commit",
    "src_dirty",
];

#[test]
fn the_harness_writes_one_json_record() {
    let dev = MetalDevice::open(0).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("bench.jsonl");
    let cfg = Config {
        kernel: "sa".into(),
        nodes: 64,
        degree: Degree::Regular(4),
        precision: Precision::Int8,
        storage: "auto".into(),
        reads: 64,
        sweeps: 16,
    };
    emit(&run_bench(&dev, &cfg), Some(&out));
    // A second run appends: the file is JSON Lines.
    emit(&run_bench(&dev, &cfg), Some(&out));
    let text = std::fs::read_to_string(&out).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    let record: Value = serde_json::from_str(lines[0]).unwrap();
    for field in OK_FIELDS {
        assert!(record.get(field).is_some(), "missing {field}: {record}");
    }
    assert_eq!(record["status"], "ok");
    assert_eq!(record["kernel"], "sa");
    assert_eq!(record["nodes"], 64);
    assert_eq!(record["degree"], 4);
    assert_eq!(record["storage"], "local");
    assert_eq!(record["reads_dispatched"], 64);
    assert_eq!(record["sweeps"], 16);
    assert!(record["ns_per_update"].as_f64().unwrap() > 0.0);
    assert!(record["max_chunk_ms"].as_f64().unwrap() > 0.0);
    assert!(record["worst_ms_per_sweep"].as_f64().unwrap() > 0.0);
    // The chunk log covers the whole plan.
    let log = record["chunk_log"].as_array().unwrap();
    assert_eq!(log.len() as u64, record["chunks"].as_u64().unwrap());
    let logged: i64 = log.iter().map(|c| c["sweeps"].as_i64().unwrap()).sum();
    assert_eq!(logged, 16);
    for field in ["ms", "sweeps", "beta_start", "sweep_start"] {
        assert!(log[0].get(field).is_some(), "chunk_log lacks {field}");
    }
    // The storage tier names follow the kernel.
    assert_eq!(Storage::Dense.name(Kernel::Sa), "dense");
}

fn small(kernel: &str, nodes: usize, degree: Degree, storage: &str, reads: usize) -> Config {
    Config {
        kernel: kernel.into(),
        nodes,
        degree,
        precision: Precision::Int8,
        storage: storage.into(),
        reads,
        sweeps: 8,
    }
}

#[test]
fn the_harness_records_a_refusal() {
    let dev = MetalDevice::open(0).unwrap();
    // The multi-spin sparse build holds at most 62 neighbours.
    let record = run_bench(&dev, &small("msa", 300, Degree::Regular(200), "sparse", 64));
    assert_eq!(record["status"], "refused");
    assert_eq!(record["kernel"], "msa");
    assert_eq!(record["degree"], 200);
    assert!(record["error"].as_str().unwrap().contains("62"), "{record}");
    assert!(record.get("ns_per_update").is_none());
}

#[test]
fn the_harness_forces_storage_and_runs_a_complete_model() {
    let dev = MetalDevice::open(0).unwrap();
    let dense = run_bench(&dev, &small("gibbs", 40, Degree::Complete, "dense", 32));
    assert_eq!(dense["status"], "ok");
    assert_eq!(dense["degree"], "complete");
    assert_eq!(dense["edges"], 40 * 39 / 2);
    assert_eq!(dense["storage"], "dense");
    let sparse = run_bench(&dev, &small("sa", 40, Degree::Complete, "sparse", 32));
    assert_eq!(sparse["storage"], "local");
}

#[test]
fn the_harness_rounds_reads_to_a_simd_group() {
    let dev = MetalDevice::open(0).unwrap();
    let record = run_bench(&dev, &small("sa", 64, Degree::Regular(4), "auto", 33));
    assert_eq!(record["reads_requested"], 33);
    assert_eq!(record["reads_dispatched"], 64);
    assert_eq!(record["reads_returned"], 33);
}

/// One run, configured by the `QUIP_BENCH_*` variables.
#[test]
#[ignore = "driven by QUIP_BENCH_* variables: see the module documentation"]
fn capacity_run() {
    let dev = MetalDevice::open(0).unwrap();
    let cfg = Config::from_env();
    let record = run_bench(&dev, &cfg);
    let out = env::var("QUIP_BENCH_OUT").ok();
    emit(&record, out.as_deref().map(Path::new));
}
