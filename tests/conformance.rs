//! Protocol conformance: spawn SA, multi-spin SA and Gibbs miners against
//! quip-solver-conformance's scripted driver.
//!
//! Metal GPU tests: needs a real device (Apple Silicon).

use quip_solver_conformance::driver::{
    drive_miner, DriverReport, Terminal, CONFIGURED_SWEEPS, GIBBS_SWEEP_MULTIPLIER,
};
use quip_solver_conformance::solve::{drive_solve, CaseOutcome};
use quip_solver_core::quip_proto::v1::RejectReason;
use std::process::Command;

/// The jobs the driver expects a `Result` for, and the only ones a
/// conformant miner may answer with one. `job-seeded` carries warm-start
/// states, which this miner ignores and answers cold.
const SOLVABLE_JOBS: [&[u8]; 5] = [
    b"job-1",
    b"job-2",
    b"job-hash",
    b"job-sparse",
    b"job-seeded",
];

/// Grade one driven session against every axis of the miner protocol.
///
/// Per-axis assertions ahead of the driver's composite `is_conformant()`: a
/// bare composite failure says "not conformant" without saying which rule
/// broke. `expected_meta_sweeps` states this test's own expectation —
/// [`CONFIGURED_SWEEPS`] for SA and MSA, doubled for Gibbs — and is cross-checked
/// against the driver's derivation so a miner advertising the wrong
/// algorithm cannot make both sides agree by accident.
fn assert_conformant(bin: &str, report: &DriverReport, expected_meta_sweeps: u32) {
    // Handshake: Hello -> Welcome -> Configure -> Ready (SPEC.md
    // "Handshake"). Dispatch stays blocked until `Ready`.
    assert!(report.handshake_ok, "{bin}: handshake failed");
    assert!(report.ready_received, "{bin}: no Ready after Configure");

    // Credits: the miner grants a first batch, then returns one per terminal
    // outcome. A zero grant would deadlock dispatch ("Credits").
    assert!(
        !report.job_request_credits.is_empty(),
        "{bin}: miner granted no credits"
    );
    assert!(
        report.job_request_credits.iter().all(|&c| c > 0),
        "{bin}: zero-credit JobRequest: {:?}",
        report.job_request_credits
    );

    // Results: one per solvable job (job-1, job-2, job-hash, job-sparse), no
    // unexpected ones, each carrying solutions and a SamplerMeta whose
    // reported energy survives the driver's own re-score and whose sweeps
    // echo the resolved budget ("Servicing a job").
    for id in SOLVABLE_JOBS {
        assert!(
            report.results.iter().any(|r| r.job_id == id),
            "{bin}: missing Result for {}: {:?}",
            String::from_utf8_lossy(id),
            report.result_job_ids()
        );
    }
    for r in &report.results {
        assert!(
            SOLVABLE_JOBS.iter().any(|id| r.job_id == *id),
            "{bin}: unexpected Result for {:?}",
            r.job_id
        );
        assert!(
            !r.solution_energies_milli.is_empty(),
            "{bin}: result {:?} carried no solutions",
            r.job_id
        );
        assert!(r.meta_present, "{bin}: result {:?} had no meta", r.job_id);
        assert_eq!(
            r.meta_sweeps, expected_meta_sweeps,
            "{bin}: result {:?} meta.sweeps did not echo the resolved budget",
            r.job_id
        );
    }
    assert_eq!(
        report.expected_meta_sweeps(),
        expected_meta_sweeps,
        "{bin}: the driver derives a different sweep expectation from the \
         advertised algorithm than this test states"
    );
    assert!(
        report.energies_rescore_clean(),
        "{bin}: a reported energy did not survive the driver's re-score"
    );

    // Reject reasons: each drives a different coordinator response, so the
    // reason must be right, not merely present ("Servicing a job" table).
    for (job_id, reason) in [
        (&b"job-bad-h"[..], RejectReason::Malformed),
        (&b"job-bad-j"[..], RejectReason::Malformed),
        (&b"job-gate"[..], RejectReason::UnsupportedKind),
        (&b"job-old"[..], RejectReason::Expired),
    ] {
        assert!(
            report.has_reject(job_id, reason),
            "{bin}: missing {reason:?} reject for {}: {:?}",
            String::from_utf8_lossy(job_id),
            report.rejects
        );
    }

    // Salt leases (SPEC.md "Generated jobs"). The session advertises
    // ISING_GENERATE, so the driver runs its lease scenario and grades it.
    let lease = report
        .lease
        .as_ref()
        .unwrap_or_else(|| panic!("{bin}: the driver ran no lease scenario"));
    assert!(
        report.lease_conformant(),
        "{bin}: lease not conformant: {lease:?}"
    );

    // Cancel and Ping are each acknowledged with a Status ("Control-plane
    // pushes"); live cancellation is honoured (no Result/Reject for the
    // cancelled watermark); GetCapabilities answers with the same identity
    // Hello advertised.
    assert!(report.cancel_acked, "{bin}: Cancel not acked with Status");
    assert!(report.ping_acked, "{bin}: Ping not acked with Status");
    assert!(
        report.live_cancel_conformant(),
        "{bin}: live cancellation not honoured"
    );
    assert!(
        report.capabilities_conformant(),
        "{bin}: Capabilities disagrees with the Hello identity"
    );

    // Every dispatched job refunds exactly one credit, however it ends.
    assert!(
        report.credit_ledger_balanced(),
        "{bin}: credit ledger unbalanced: dispatched {}, refunded {}",
        report.jobs_dispatched,
        report.credits_refunded()
    );

    // Shutdown drains in-flight results, closes the stream cleanly with no
    // phase timing out, and exits 0 ("Exit codes").
    assert_eq!(
        report.terminal,
        Terminal::Closed,
        "{bin}: stream did not end cleanly"
    );
    assert!(
        report.timed_out_phases.is_empty(),
        "{bin}: phases timed out: {:?}",
        report.timed_out_phases
    );
    assert_eq!(report.exit_code, 0, "{bin}: clean shutdown expected");

    // The reference composite verdict. Algorithm-aware since
    // quip-solver-conformance 0.0.1-rc1, so a Gibbs miner passes it too.
    assert!(
        report.is_conformant(),
        "{bin}: not conformant per the reference verdict:\n{}",
        report.summary()
    );
}

/// Cross-package binary path (deps/ → profile/ → bin).
fn profile_bin(name: &str) -> String {
    let name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let mut p = std::env::current_exe().expect("test exe path");
    p.pop(); // deps/
    p.pop(); // <profile>/
    p.push(&name);
    p.to_string_lossy().into_owned()
}

fn ensure_built(package_bins: &[&str]) {
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-solver-metal"])
        .status()
        .expect("cargo build quip-solver-metal");
    assert!(status.success(), "failed to build quip-solver-metal");
    for b in package_bins {
        assert!(
            std::path::Path::new(&profile_bin(b)).exists(),
            "missing binary {b} at {}",
            profile_bin(b)
        );
    }
}

#[tokio::test]
async fn quip_metal_sa_passes_conformance() {
    ensure_built(&["quip-metal-sa"]);
    let miner = profile_bin("quip-metal-sa");
    let socket = format!(
        "/tmp/quip-metal-sa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert_conformant("quip-metal-sa", &report, CONFIGURED_SWEEPS);
}

#[tokio::test]
async fn quip_metal_gibbs_passes_conformance() {
    ensure_built(&["quip-metal-gibbs"]);
    let miner = profile_bin("quip-metal-gibbs");
    let socket = format!(
        "/tmp/quip-metal-gibbs-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    assert_conformant(
        "quip-metal-gibbs",
        &report,
        CONFIGURED_SWEEPS * GIBBS_SWEEP_MULTIPLIER,
    );
}

#[tokio::test]
async fn quip_metal_msa_passes_conformance() {
    ensure_built(&["quip-metal-msa"]);
    let miner = profile_bin("quip-metal-msa");
    let socket = format!(
        "/tmp/quip-metal-msa-conf-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let report = drive_miner(&miner, &format!("unix://{socket}")).await;
    // MSA always screens, but StreamResult has no completed-sweeps field.
    // The session builds SamplerMeta from the requested budget even when the
    // resident runner returns probe reads. Keep the exact metadata contract.
    assert_conformant("quip-metal-msa", &report, CONFIGURED_SWEEPS);
}

/// Grade a kernel against the `--solve` gate, and require the linear-schedule
/// case and the given limit cases to pass: a model at each advertised limit
/// solves, one past it is refused as capacity.
async fn assert_solve_conformant(bin: &str, limit_cases: &[&str]) {
    ensure_built(&[bin]);
    let report = drive_solve(&profile_bin(bin)).await;
    assert!(
        report.is_conformant(),
        "{bin}: --solve not conformant:\n{}",
        report.summary()
    );
    for case in ["linear-schedule"].iter().chain(limit_cases) {
        assert_eq!(
            report.outcome(case),
            Some(&CaseOutcome::Passed),
            "{bin} {case}:\n{}",
            report.summary()
        );
    }
}

const FIELD_LIMIT_CASES: &[&str] = &["field-limit-inside", "field-limit-outside"];

#[tokio::test]
async fn quip_metal_sa_passes_solve_conformance() {
    assert_solve_conformant("quip-metal-sa", FIELD_LIMIT_CASES).await;
}

#[tokio::test]
async fn quip_metal_gibbs_passes_solve_conformance() {
    assert_solve_conformant("quip-metal-gibbs", FIELD_LIMIT_CASES).await;
}

#[tokio::test]
async fn quip_metal_msa_passes_solve_conformance() {
    assert_solve_conformant(
        "quip-metal-msa",
        &[
            "degree-limit-inside",
            "degree-limit-outside",
            "field-limit-inside",
            "field-limit-outside",
        ],
    )
    .await;
}

#[test]
fn capabilities_and_version_and_check() {
    ensure_built(&["quip-metal-sa", "quip-metal-gibbs", "quip-metal-msa"]);

    for (bin, algo) in [
        ("quip-metal-sa", "sa"),
        ("quip-metal-msa", "msa"),
        ("quip-metal-gibbs", "gibbs"),
    ] {
        let path = profile_bin(bin);

        let out = Command::new(&path).arg("--capabilities").output().unwrap();
        assert!(out.status.success(), "{bin} --capabilities failed");
        let s = String::from_utf8(out.stdout).unwrap();
        assert!(s.contains("\"backend\":\"metal\""), "{bin}: {s}");
        assert!(
            s.contains(&format!("\"algorithm\":\"{algo}\"")),
            "{bin}: {s}"
        );
        assert!(s.contains("\"protocolVersion\":2"), "{bin}: {s}");
        assert!(s.contains("\"ISING_GENERATE\""), "{bin}: {s}");

        let out = Command::new(&path).arg("--version").output().unwrap();
        assert!(out.status.success());
        let version = String::from_utf8(out.stdout).unwrap();
        assert!(
            version.trim_end().ends_with("protocol 2"),
            "{bin}: --version must end with protocol 2: {version}"
        );

        // --check opens the GPU and compiles kernels.
        let status = Command::new(&path)
            .arg("--check")
            .arg("--device")
            .arg("0")
            .status();
        assert!(
            status.unwrap().success(),
            "{bin} --check must succeed when a GPU is present"
        );
    }
}

/// The core must reject an unknown `--log-level` before doing anything else.
/// In `quip-solver-core` 0.0.0 the validation moved from `logging::init`
/// (exit 64) into `CommonArgs` itself — a clap `PossibleValuesParser` — so a
/// bad level is a usage error (exit 2) and capabilities are never printed.
///
/// This crate installs a subscriber of its own, but that one does not validate
/// the level, so the check still detects a stale core: a core without the
/// value parser would print capabilities and exit 0 here.
#[test]
fn invalid_log_level_is_a_usage_error() {
    for bin in [
        env!("CARGO_BIN_EXE_quip-metal-sa"),
        env!("CARGO_BIN_EXE_quip-metal-gibbs"),
        env!("CARGO_BIN_EXE_quip-metal-msa"),
    ] {
        let out = Command::new(bin)
            .arg("--capabilities")
            .arg("--log-level")
            .arg("bogus")
            .env("QUIP_SESSION_TOKEN", "tok")
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(2),
            "{bin}: an unknown --log-level must be a clap usage error (got {:?}, stdout={}, stderr={})",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("bogus"),
            "{bin}: stderr must name the bad level, got {stderr}"
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).is_empty(),
            "{bin}: capabilities must not print on a bad --log-level"
        );
    }
}
