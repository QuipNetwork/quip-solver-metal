//! Zero-field conformance: the annealing binaries solve a model with no linear
//! field through `--solve`.
//!
//! The main model is a 16-spin antiferromagnetic ring. Its two ground states have
//! every bond satisfied, at energy `-16 * |J|`. A kernel that visits nodes in
//! one fixed order and accepts every flat move only shifts the ring's domain
//! walls in step, so its reads keep the energy of their random start.

use quip_solver_metal::metal_device::MetalDevice;
use quip_solver_metal::sampler::{sample_ising, sample_ising_tuned, Kernel, Tuning};
use quip_solver_metal::{IsingGraph, SampleParams};
use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

const NODES: usize = 16;
const READS: usize = 64;

/// Reads that must reach the ground state. Below `READS` so that a rare read
/// frozen with one wall pair does not fail the case; a stalled kernel reaches
/// the ground state on none.
const MIN_GROUND_READS: usize = 60;

/// The ring as a `--solve` request, with coupling `j` on every bond and field
/// `h0` on node 0.
fn ring(j: f64, h0: f64, beta_range: Option<[f64; 2]>) -> Value {
    let mut h = vec![0.0; NODES];
    h[0] = h0;
    let edges: Vec<[usize; 2]> = (0..NODES)
        .map(|i| {
            let k = (i + 1) % NODES;
            [i.min(k), i.max(k)]
        })
        .collect();
    json!({
        "h": h,
        "j": vec![j; NODES],
        "edges": edges,
        "num_reads": READS,
        "num_sweeps": 1000,
        "sweeps_per_beta": 1,
        "beta_range": beta_range,
        "seed": 7,
    })
}

/// Run `bin --solve` on `problem` and return its reads.
fn solve(bin: &str, problem: &Value) -> Vec<Value> {
    let mut child = Command::new(bin)
        .arg("--solve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn solver");
    child
        .stdin
        .take()
        .expect("solver stdin")
        .write_all(problem.to_string().as_bytes())
        .expect("write problem");
    let out = child.wait_with_output().expect("solver exit");
    assert!(
        out.status.success(),
        "{bin} --solve exited {:?}: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice::<Vec<Value>>(&out.stdout).expect("solutions are a JSON array")
}

fn energies(reads: &[Value]) -> Vec<i64> {
    reads
        .iter()
        .map(|r| r["energy_milli"].as_i64().expect("energy_milli"))
        .collect()
}

/// Assert `bin` reaches the ground state of the zero-field ring with coupling
/// `j`, under the default ladder and under an explicit one.
fn assert_anneals_zero_field_ring(bin: &str, j: f64) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "j is 1 or 1000 and the ring has 16 bonds; the product is far inside i64"
    )]
    let ground = -(NODES as i64) * (j * 1000.0) as i64;
    for beta_range in [None, Some([0.1, 10.0])] {
        let got = energies(&solve(bin, &ring(j, 0.0, beta_range)));
        assert_eq!(got.len(), READS, "{bin} j {j} beta {beta_range:?}");
        let at_ground = got.iter().filter(|&&e| e == ground).count();
        assert!(
            at_ground >= MIN_GROUND_READS,
            "{bin} j {j} beta {beta_range:?}: {at_ground}/{READS} reads at the ground energy \
             {ground}; energies {got:?}"
        );
    }
}

/// The ticket's ring: couplings of 1000.
#[test]
fn quip_metal_sa_anneals_the_zero_field_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-metal-sa"), 1000.0);
}

/// Unit couplings select the narrow kernel.
#[test]
fn quip_metal_sa_anneals_the_zero_field_unit_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-metal-sa"), 1.0);
}

#[test]
fn quip_metal_msa_anneals_the_zero_field_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-metal-msa"), 1000.0);
}

/// Unit couplings select the unit multi-spin kernel.
#[test]
fn quip_metal_msa_anneals_the_zero_field_unit_ring() {
    assert_anneals_zero_field_ring(env!("CARGO_BIN_EXE_quip-metal-msa"), 1.0);
}

/// One field on the ring picks one of the two orientations: the best read
/// gains the field's energy on top of the sixteen bonds.
#[test]
fn quip_metal_sa_still_solves_the_ring_with_one_field() {
    let got = energies(&solve(
        env!("CARGO_BIN_EXE_quip-metal-sa"),
        &ring(1000.0, 1000.0, None),
    ));
    assert_eq!(got.iter().min(), Some(&-17_000_000), "energies {got:?}");
}

/// Assert `bin` gives the same samples for the same seed, on the unit ring
/// that selects its own kernel.
fn assert_repeats_its_samples(bin: &str) {
    let problem = ring(1.0, 0.0, None);
    let spins =
        |reads: &[Value]| -> Vec<Value> { reads.iter().map(|r| r["spins"].clone()).collect() };
    assert_eq!(
        spins(&solve(bin, &problem)),
        spins(&solve(bin, &problem)),
        "{bin}"
    );
}

#[test]
fn quip_metal_sa_repeats_its_samples_on_the_zero_field_ring() {
    assert_repeats_its_samples(env!("CARGO_BIN_EXE_quip-metal-sa"));
}

#[test]
fn quip_metal_msa_repeats_its_samples_on_the_zero_field_ring() {
    assert_repeats_its_samples(env!("CARGO_BIN_EXE_quip-metal-msa"));
}

/// Side of the square lattice in the Max-Cut case.
const SIDE: usize = 6;

/// Max-Cut on a 6 by 6 periodic square lattice: antiferromagnetic unit
/// couplings and no fields. The lattice is bipartite, so the best cut takes
/// all 72 edges, at energy -72.
fn max_cut_torus() -> Value {
    let node = |x: usize, y: usize| (x % SIDE) * SIDE + (y % SIDE);
    let mut edges = Vec::new();
    for x in 0..SIDE {
        for y in 0..SIDE {
            for other in [node(x + 1, y), node(x, y + 1)] {
                let here = node(x, y);
                edges.push([here.min(other), here.max(other)]);
            }
        }
    }
    json!({
        "h": vec![0.0; SIDE * SIDE],
        "j": vec![1.0; edges.len()],
        "edges": edges,
        "num_reads": READS,
        "num_sweeps": 1000,
        "sweeps_per_beta": 1,
        "beta_range": null,
        "seed": 7,
    })
}

/// Assert `bin` finds the full cut of the lattice on at least one read.
fn assert_solves_max_cut_torus(bin: &str) {
    let got = energies(&solve(bin, &max_cut_torus()));
    assert_eq!(got.iter().min(), Some(&-72_000), "{bin}: energies {got:?}");
}

#[test]
fn quip_metal_sa_solves_zero_field_max_cut_on_a_lattice() {
    assert_solves_max_cut_torus(env!("CARGO_BIN_EXE_quip-metal-sa"));
}

#[test]
fn quip_metal_msa_solves_zero_field_max_cut_on_a_lattice() {
    assert_solves_max_cut_torus(env!("CARGO_BIN_EXE_quip-metal-msa"));
}

/// A one-node model has nothing to anneal around: every read takes the spin
/// its field prefers.
#[test]
fn quip_metal_msa_solves_a_single_node_with_a_field() {
    let problem = json!({
        "h": [1.0],
        "j": [],
        "edges": [],
        "num_reads": READS,
        "num_sweeps": 100,
        "sweeps_per_beta": 1,
        "beta_range": null,
        "seed": 7,
    });
    let got = energies(&solve(env!("CARGO_BIN_EXE_quip-metal-msa"), &problem));
    assert_eq!(got, vec![-1000; READS]);
}

/// Two unconnected nodes, each with a field, annealed for one cold sweep.
fn two_fields_one_sweep() -> Value {
    json!({
        "h": [1.0, 1.0],
        "j": [],
        "edges": [],
        "num_reads": READS,
        "num_sweeps": 1,
        "sweeps_per_beta": 1,
        "beta_range": [10.0, 10.0],
        "seed": 7,
    })
}

/// The last sweep of a read visits every node, so a one-sweep anneal still
/// settles each node against its field.
#[test]
fn quip_metal_sa_visits_every_node_in_a_one_sweep_anneal() {
    let got = energies(&solve(
        env!("CARGO_BIN_EXE_quip-metal-sa"),
        &two_fields_one_sweep(),
    ));
    assert_eq!(got, vec![-2000; READS]);
}

#[test]
fn quip_metal_msa_visits_every_node_in_a_one_sweep_anneal() {
    let got = energies(&solve(
        env!("CARGO_BIN_EXE_quip-metal-msa"),
        &two_fields_one_sweep(),
    ));
    assert_eq!(got, vec![-2000; READS]);
}

/// Coupling of the Float32 ring: off the milli grid, so the classifier
/// selects the Float32 build. The binaries refuse such a model, so the case
/// calls the library.
const FLOAT_J: f64 = 1.0005;

/// Assert `kernel`'s Float32 build reaches the ground state of the zero-field
/// ring, every bond satisfied at `-16 * FLOAT_J`.
fn assert_float_build_anneals_the_zero_field_ring(kernel: Kernel) {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let dev = MetalDevice::open(0).unwrap();
    let edges: Vec<(usize, usize)> = (0..NODES).map(|i| (i, (i + 1) % NODES)).collect();
    let graph = IsingGraph::new(vec![0.0; NODES], vec![FLOAT_J; NODES], edges.clone());
    let params = SampleParams {
        num_reads: READS,
        num_sweeps: 1000,
        sweeps_per_beta: 1,
        seed: 7,
        ..Default::default()
    };
    let reads = sample_ising(&dev, &graph, &params, kernel).unwrap();
    assert_eq!(reads.len(), READS, "{kernel:?}");
    let satisfied = |spins: &[i8]| edges.iter().filter(|&&(u, v)| spins[u] != spins[v]).count();
    let at_ground = reads
        .iter()
        .filter(|r| satisfied(&r.spins) == NODES)
        .count();
    assert!(
        at_ground >= MIN_GROUND_READS,
        "{kernel:?} Float32: {at_ground}/{READS} reads at the ground state"
    );
}

#[test]
fn float_sa_anneals_the_zero_field_ring() {
    assert_float_build_anneals_the_zero_field_ring(Kernel::Sa);
}

#[test]
fn float_msa_anneals_the_zero_field_ring() {
    assert_float_build_anneals_the_zero_field_ring(Kernel::Msa);
}

/// Assert the QUIP_MSA_SPILL build reaches the ground state of the zero-field
/// ring with coupling `j`, with `shared_spins_cap` words in threadgroup memory
/// and the rest spilled, and that sweep windows of one sweep return the same
/// reads. 1.5 takes the milli lane and `FLOAT_J` the Float32 lane.
fn assert_spill_build_anneals_the_zero_field_ring(j: f64) {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let dev = MetalDevice::open(0).unwrap();
    let edges: Vec<(usize, usize)> = (0..NODES).map(|i| (i, (i + 1) % NODES)).collect();
    let graph = IsingGraph::new(vec![0.0; NODES], vec![j; NODES], edges.clone());
    let params = SampleParams {
        num_reads: READS,
        num_sweeps: 256,
        sweeps_per_beta: 4,
        seed: 7,
        ..Default::default()
    };
    let run = |planning_rate| {
        let tuning = Tuning {
            shared_spins_cap: Some(NODES / 4),
            planning_rate,
            ..Tuning::default()
        };
        sample_ising_tuned(&dev, &graph, &params, Kernel::Msa, tuning).unwrap()
    };
    // One command buffer, then one sweep per command buffer.
    let whole = run(Some(1e18));
    assert_eq!(whole.plan.len(), 1, "j {j}");
    let windows = run(Some(1.0));
    assert_eq!(windows.plan.len(), params.num_sweeps, "j {j}");
    assert_eq!(
        windows.reads, whole.reads,
        "j {j}: windows change the reads"
    );
    let satisfied = |spins: &[i8]| edges.iter().filter(|&&(u, v)| spins[u] != spins[v]).count();
    let at_ground = whole
        .reads
        .iter()
        .filter(|r| satisfied(&r.spins) == NODES)
        .count();
    assert!(
        at_ground >= MIN_GROUND_READS,
        "spill j {j}: {at_ground}/{READS} reads at the ground state"
    );
}

#[test]
fn msa_spill_build_anneals_the_zero_field_ring() {
    assert_spill_build_anneals_the_zero_field_ring(1.5);
}

#[test]
fn msa_float_spill_build_anneals_the_zero_field_ring() {
    assert_spill_build_anneals_the_zero_field_ring(FLOAT_J);
}

/// Assert the QUIP_SA_GLOBAL build reaches the ground state of the zero-field
/// ring with coupling `j`, and that sweep windows of one sweep return the same
/// reads. 1 takes the int8 build, 1000 the wide build and `FLOAT_J` the
/// Float32 build.
fn assert_sa_global_build_anneals_the_zero_field_ring(j: f64) {
    if MetalDevice::device_count() == 0 {
        return;
    }
    let dev = MetalDevice::open(0).unwrap();
    let edges: Vec<(usize, usize)> = (0..NODES).map(|i| (i, (i + 1) % NODES)).collect();
    let graph = IsingGraph::new(vec![0.0; NODES], vec![j; NODES], edges.clone());
    let params = SampleParams {
        num_reads: READS,
        num_sweeps: 256,
        sweeps_per_beta: 4,
        seed: 7,
        ..Default::default()
    };
    let run = |planning_rate| {
        let tuning = Tuning {
            sa_global: true,
            planning_rate,
            ..Tuning::default()
        };
        sample_ising_tuned(&dev, &graph, &params, Kernel::Sa, tuning).unwrap()
    };
    // One command buffer, then one sweep per command buffer.
    let whole = run(Some(1e18));
    assert_eq!(whole.plan.len(), 1, "j {j}");
    let windows = run(Some(1.0));
    assert_eq!(windows.plan.len(), params.num_sweeps, "j {j}");
    assert_eq!(
        windows.reads, whole.reads,
        "j {j}: windows change the reads"
    );
    let satisfied = |spins: &[i8]| edges.iter().filter(|&&(u, v)| spins[u] != spins[v]).count();
    let at_ground = whole
        .reads
        .iter()
        .filter(|r| satisfied(&r.spins) == NODES)
        .count();
    assert!(
        at_ground >= MIN_GROUND_READS,
        "SA global j {j}: {at_ground}/{READS} reads at the ground state"
    );
}

#[test]
fn sa_global_build_anneals_the_zero_field_ring() {
    for j in [1.0, 1000.0, FLOAT_J] {
        assert_sa_global_build_anneals_the_zero_field_ring(j);
    }
}
