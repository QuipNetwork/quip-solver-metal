// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! MT-1: Metal miners at the generic family tier.
//!
//! No Metal binary expands family leases, so each declares the generic tier
//! (FAM-56): no `solver_id`, no generated or accepted family, and its own
//! kernel limits. A family lease is rejected without a fallback, and an
//! explicit job that carries a `FamilyBinding` is solved as the plain model
//! (FAM-69).
//!
//! Metal GPU tests: need a real device (Apple Silicon). Run with
//! `--test-threads=1`.

#[expect(
    dead_code,
    reason = "the shared lease coordinator has helpers this file does not call"
)]
mod support;

use quip_solver_conformance::driver::drive_miner;
use quip_solver_conformance::families::drive_miner_family;
use quip_solver_conformance::families::export_instances;
use quip_solver_conformance::families::layered::{P_SPEC, UNIFORM_V1};
use quip_solver_conformance::solve::CaseOutcome;
use quip_solver_core::ising_generators::descriptor::params_digest;
use quip_solver_core::ising_generators::family::BaseTopology;
use quip_solver_core::ising_generators::instance::Instance;
use quip_solver_core::quip_proto::v1::{
    coord_msg, ising_problem, Backend, Cancel, Capabilities, CoefficientEncoding, EdgeList,
    FamilyBinding, FamilyDescriptor, GeneratorAlgorithm, IsingProblem, Job, JobKind,
};
use quip_solver_core::quip_protocol::wire::{decode_spins_packed, encode_i32_le};
use quip_solver_core::BackendIdentity;
use quip_solver_metal::{METAL_GIBBS_IDENTITY, METAL_MSA_IDENTITY, METAL_SA_IDENTITY};
use serde_json::{json, Map, Value};
use std::process::Command;
use std::time::Duration;
use support::{target, Session};

const LONG: Duration = Duration::from_secs(120);

/// One Metal binary and the identity it is built from.
struct Bin {
    name: &'static str,
    path: &'static str,
    identity: &'static BackendIdentity,
}

fn bins() -> [Bin; 3] {
    [
        Bin {
            name: "quip-metal-sa",
            path: env!("CARGO_BIN_EXE_quip-metal-sa"),
            identity: &METAL_SA_IDENTITY,
        },
        Bin {
            name: "quip-metal-gibbs",
            path: env!("CARGO_BIN_EXE_quip-metal-gibbs"),
            identity: &METAL_GIBBS_IDENTITY,
        },
        Bin {
            name: "quip-metal-msa",
            path: env!("CARGO_BIN_EXE_quip-metal-msa"),
            identity: &METAL_MSA_IDENTITY,
        },
    ]
}

fn command_line_capabilities(path: &str) -> Value {
    let out = Command::new(path).arg("--capabilities").output().unwrap();
    assert!(out.status.success(), "{path} --capabilities failed");
    serde_json::from_slice(&out.stdout).unwrap()
}

/// The protobuf JSON name of an enum value with its type prefix removed, as
/// `--capabilities` prints `backend` and `algorithm`.
fn short_name(name: &str, prefix: &str) -> String {
    name.strip_prefix(prefix).unwrap().to_lowercase()
}

/// A session `Capabilities` in the shape `--capabilities` prints: the
/// protobuf JSON mapping, with empty strings, empty lists, and absent
/// optionals left out.
fn session_json(c: &Capabilities) -> Value {
    let names = |values: &[i32], name: fn(i32) -> Option<&'static str>| -> Value {
        values.iter().map(|&v| json!(name(v).unwrap())).collect()
    };
    let mut view = Map::new();
    view.insert(
        "supportedKinds".into(),
        names(&c.supported_kinds, |k| {
            JobKind::try_from(k).ok().map(|k| k.as_str_name())
        }),
    );
    view.insert("maxNodes".into(), json!(c.max_nodes));
    view.insert("maxEdges".into(), json!(c.max_edges));
    if !c.features.is_empty() {
        view.insert("features".into(), json!(c.features));
    }
    view.insert("protocolVersion".into(), json!(c.protocol_version));
    view.insert("streamWidth".into(), json!(c.stream_width));
    view.insert(
        "encodings".into(),
        names(&c.encodings, |e| {
            CoefficientEncoding::try_from(e)
                .ok()
                .map(|e| e.as_str_name())
        }),
    );
    view.insert(
        "generators".into(),
        names(&c.generators, |g| {
            GeneratorAlgorithm::try_from(g)
                .ok()
                .map(|g| g.as_str_name())
        }),
    );
    view.insert(
        "backend".into(),
        json!(short_name(c.backend().as_str_name(), "BACKEND_")),
    );
    view.insert(
        "algorithm".into(),
        json!(short_name(c.algorithm().as_str_name(), "ALGORITHM_")),
    );
    for (key, value) in [
        ("maxDegree", c.max_degree),
        ("maxLocalFieldMilli", c.max_local_field_milli),
        ("exactResolutionMilli", c.exact_resolution_milli),
    ] {
        if let Some(value) = value {
            view.insert(key.into(), json!(value));
        }
    }
    assert!(c.native_topology_hash.is_none(), "{c:?}");
    assert!(c.solver_id.is_empty(), "{c:?}");
    assert!(c.generated_families.is_empty(), "{c:?}");
    assert!(c.accepted_families.is_empty(), "{c:?}");
    Value::Object(view)
}

fn socket(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "unix:///tmp/{name}-family-{}-{nanos}.sock",
        std::process::id()
    )
}

#[test]
fn the_command_line_declares_the_generic_tier_with_each_kernel_limit() {
    for bin in bins() {
        let name = bin.name;
        let caps = command_line_capabilities(bin.path);
        for key in ["solverId", "generatedFamilies", "acceptedFamilies"] {
            assert!(caps.get(key).is_none(), "{name} lists {key}: {caps}");
        }
        let family_generator = GeneratorAlgorithm::FamilyChacha8V1.as_str_name();
        assert!(
            !caps["generators"]
                .as_array()
                .unwrap()
                .iter()
                .any(|g| g == family_generator),
            "{name} lists {family_generator} with no generated family (FAM-58): {caps}"
        );
        let identity = bin.identity;
        assert_eq!(caps["backend"], "metal", "{name}: {caps}");
        assert_eq!(
            caps["algorithm"],
            short_name(identity.algorithm.as_str_name(), "ALGORITHM_"),
            "{name}: {caps}"
        );
        assert_eq!(caps["maxNodes"], identity.max_nodes, "{name}: {caps}");
        assert_eq!(caps["maxEdges"], identity.max_edges, "{name}: {caps}");
        let limits = &identity.limits;
        for (key, value) in [
            ("maxDegree", limits.max_degree),
            ("maxLocalFieldMilli", limits.max_local_field_milli),
            ("exactResolutionMilli", limits.exact_resolution_milli),
        ] {
            let value = value.unwrap_or_else(|| panic!("{name} declares no {key}"));
            assert_eq!(caps[key], value, "{name} {key}: {caps}");
        }
    }
}

#[tokio::test]
async fn the_session_advertises_what_the_command_line_prints() {
    for bin in bins() {
        let name = bin.name;
        let caps = command_line_capabilities(bin.path);
        let report = drive_miner(bin.path, &socket(name)).await;
        assert!(report.handshake_ok, "{name}: handshake failed");
        let hello = report
            .hello
            .as_ref()
            .and_then(|hello| hello.capabilities.as_ref())
            .expect("Hello carries capabilities");
        let answered = report
            .capabilities_received
            .as_ref()
            .expect("a GetCapabilities answer");
        for (source, session) in [("Hello", hello), ("GetCapabilities", answered)] {
            assert_eq!(session.backend(), Backend::Metal, "{name} {source}");
            assert_eq!(
                session.algorithm(),
                bin.identity.algorithm,
                "{name} {source}"
            );
            assert_eq!(session_json(session), caps, "{name} {source}");
        }
    }
}

/// The SC-7 cases a generic binary must pass, not skip. The rest of the gate
/// covers generated and accepted revisions, which these binaries do not list.
const GENERIC_TIER_CASES: [&str; 5] = [
    "hello-protocol-version-2",
    "generators-list-family-iff-generated",
    "no-generated-families-rejects-family-lease",
    "credit-ledger-balanced",
    "clean-exit",
];

#[tokio::test]
async fn the_family_gate_rejects_family_leases_without_fallback() {
    for bin in bins() {
        let name = bin.name;
        let report = drive_miner_family(bin.path, &[], &socket(name)).await;
        let failed: Vec<String> = report
            .cases
            .iter()
            .filter(|case| matches!(case.outcome, CaseOutcome::Failed(_)))
            .map(|case| format!("{case:?}"))
            .collect();
        assert!(
            report.is_conformant(),
            "{name} failed the family gate (exit {}): {failed:#?}\n{}",
            report.exit_code,
            report.stderr
        );
        for wanted in GENERIC_TIER_CASES {
            let case = report
                .cases
                .iter()
                .find(|case| case.name == wanted)
                .unwrap_or_else(|| panic!("{name}: the gate ran no {wanted} case"));
            assert_eq!(case.outcome, CaseOutcome::Passed, "{name} {wanted}");
        }
    }
}

/// The canonical `vectors/uniform-field` revision 1 instance at `P_SPEC` on
/// the gate's `T_SPARSE` topology: sparse native ids, edges in instance
/// order, and a nonzero field on every node. The test limits skip some
/// salts, so this takes the first nonce that generates.
fn canonical_instance() -> Instance {
    let stored = quip_solver_conformance::t_sparse().unwrap();
    let topology = BaseTopology::new(stored.nodes, &stored.edges).unwrap();
    let nonces: Vec<[u8; 32]> = (0..=u8::MAX).map(|byte| [byte; 32]).collect();
    let instance = export_instances(&UNIFORM_V1, &P_SPEC, &topology, &nonces)
        .unwrap()
        .into_iter()
        .find_map(Result::ok)
        .expect("one of 256 nonces generates");
    assert!(
        instance.h().iter().all(|&h| h != 0),
        "P_SPEC sets a field on every node"
    );
    instance
}

/// The `FamilyBinding` a coordinator sends with `instance`.
fn binding(instance: &Instance) -> FamilyBinding {
    FamilyBinding {
        family: Some(FamilyDescriptor {
            family_id: UNIFORM_V1.family_id.to_owned(),
            revision: UNIFORM_V1.revision,
            params: P_SPEC.to_vec(),
            params_digest: params_digest(&P_SPEC).to_vec(),
        }),
        nonce: instance.nonce().unwrap().to_vec(),
        instance_digest: instance.digest().to_vec(),
        metadata: instance.metadata().to_vec(),
        topology_order_digest: instance.topology_order_digest().to_vec(),
        node_ids: instance.node_ids().to_vec(),
        offset_milli: instance.offset_milli(),
    }
}

/// An explicit job of `instance` at `generation`, in exact milli-units.
fn explicit_job(
    id: &[u8],
    generation: u64,
    instance: &Instance,
    family: Option<FamilyBinding>,
    num_sweeps: u32,
) -> Job {
    let (u, v) = instance.edges().iter().copied().unzip();
    Job {
        job_id: id.to_vec(),
        kind: JobKind::IsingSample as i32,
        generation,
        deadline_ms: 0,
        ising: Some(IsingProblem {
            encoding: CoefficientEncoding::I32 as i32,
            scale: 1000,
            h: encode_i32_le(instance.h()),
            j: encode_i32_le(instance.j()),
            family,
            num_reads: 16,
            num_sweeps,
            graph: Some(ising_problem::Graph::Edges(EdgeList { u, v })),
            ..Default::default()
        }),
        provenance: None,
        generator: None,
    }
}

/// The plain-model energy of `spins` in milli-units, without the instance
/// offset: a generic solver reports the model it was given (FAM-55).
fn plain_energy_milli(instance: &Instance, spins: &[i8]) -> i64 {
    let fields: i64 = instance
        .h()
        .iter()
        .zip(spins)
        .map(|(&h, &s)| i64::from(h) * i64::from(s))
        .sum();
    let couplings: i64 = instance
        .edges()
        .iter()
        .zip(instance.j())
        .map(|(&(u, v), &j)| {
            i64::from(j) * i64::from(spins[u as usize]) * i64::from(spins[v as usize])
        })
        .sum();
    fields + couplings
}

/// Every read of `id` reports the exact plain-model energy of its spins.
fn assert_exact(s: &Session, id: &[u8], instance: &Instance, name: &str) {
    let label = String::from_utf8_lossy(id);
    let results = s.log.results_for(id);
    assert_eq!(results.len(), 1, "{name} {label}: one result");
    let solutions = &results[0].solutions;
    assert_eq!(solutions.len(), 16, "{name} {label}: every read returns");
    let nodes = instance.node_ids().len();
    for solution in solutions {
        let spins = decode_spins_packed(&solution.spins, nodes).unwrap();
        assert_eq!(
            solution.energy_milli,
            plain_energy_milli(instance, &spins),
            "{name} {label}: reported energy is not the plain-model energy"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_family_bound_explicit_job_is_solved_as_the_plain_model() {
    let instance = canonical_instance();
    let mut forged = binding(&instance);
    forged.instance_digest[0] ^= 0xff;
    let jobs: [(&[u8], Option<FamilyBinding>); 3] = [
        (b"plain", None),
        (b"bound", Some(binding(&instance))),
        // A generic solver does not validate the binding (FAM-65, FAM-69).
        (b"forged-digest", Some(forged)),
    ];
    for bin in bins() {
        let name = bin.name;
        let mut s = Session::start(bin.path, 4, "").await;
        s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 64)))
            .await;
        for (id, family) in &jobs {
            let job = explicit_job(id, 1, &instance, family.clone(), 64);
            s.send(coord_msg::Msg::Job(job)).await;
        }
        s.until("explicit results", LONG, |log| {
            jobs.iter().all(|(id, _)| !log.results_for(id).is_empty())
        })
        .await;
        assert!(s.log.rejects.is_empty(), "{name}: {:?}", s.log.rejects);
        for (id, _) in &jobs {
            assert_exact(&s, id, &instance, name);
        }
        assert_eq!(s.shutdown(2_000).await, 0, "{name}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_family_bound_job_ends_with_a_result_or_a_refund() {
    let instance = canonical_instance();
    for bin in bins() {
        let name = bin.name;
        let mut s = Session::start(bin.path, 4, "").await;
        s.send(coord_msg::Msg::SetTarget(target(i64::MAX, 64)))
            .await;
        let job = explicit_job(
            b"cancelled",
            5,
            &instance,
            Some(binding(&instance)),
            1 << 20,
        );
        s.send(coord_msg::Msg::Job(job)).await;
        s.send(coord_msg::Msg::Cancel(Cancel { max_generation: 5 }))
            .await;
        s.until("cancel settles", LONG, |log| {
            log.refunds() >= 1 || !log.results_for(b"cancelled").is_empty()
        })
        .await;
        // The session still solves a live bound job after the cancel.
        let after = explicit_job(b"after", 6, &instance, Some(binding(&instance)), 64);
        s.send(coord_msg::Msg::Job(after)).await;
        s.until("job after cancel", LONG, |log| {
            !log.results_for(b"after").is_empty()
        })
        .await;
        assert!(s.log.rejects.is_empty(), "{name}: {:?}", s.log.rejects);
        let answered = s.log.results_for(b"cancelled").len();
        assert!(answered <= 1, "{name}: {answered} results for one job");
        assert_exact(&s, b"after", &instance, name);
        assert_eq!(s.shutdown(2_000).await, 0, "{name}");
    }
}
