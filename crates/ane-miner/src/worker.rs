use std::io::{Read, Write};
use std::process::ExitCode;
use std::time::Duration;

use quip_solver_core::{IsingGraph, SampleParams};
use serde::{Deserialize, Serialize};

use crate::graph::MAX_LANES;
use crate::native::{self, AneProgram};
use crate::solver::{solve_in_process, RunOutput};
use crate::AneError;

pub(crate) const MAX_MESSAGE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(
    tag = "command",
    content = "job",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(crate) enum WorkerRequest {
    Check,
    Sample(RawJob),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawJob {
    pub(crate) h: Vec<f64>,
    pub(crate) j: Vec<f64>,
    pub(crate) edges: Vec<(usize, usize)>,
    pub(crate) num_reads: usize,
    pub(crate) num_sweeps: usize,
    pub(crate) sweeps_per_beta: usize,
    pub(crate) beta_range: Option<(f64, f64)>,
    pub(crate) schedule: quip_solver_core::BetaSchedule,
    pub(crate) seed: u64,
}

impl RawJob {
    pub(crate) fn from_parts(graph: &IsingGraph, params: &SampleParams) -> Self {
        Self {
            h: graph.h.clone(),
            j: graph.j.clone(),
            edges: graph.edges.to_vec(),
            num_reads: params.num_reads,
            num_sweeps: params.num_sweeps,
            sweeps_per_beta: params.sweeps_per_beta,
            beta_range: params.beta_range,
            schedule: params.schedule,
            seed: params.seed,
        }
    }

    pub(crate) fn into_parts(self) -> (IsingGraph, SampleParams) {
        (
            IsingGraph {
                h: self.h,
                j: self.j,
                edges: self.edges.into(),
            },
            SampleParams {
                num_reads: self.num_reads,
                num_sweeps: self.num_sweeps,
                sweeps_per_beta: self.sweeps_per_beta,
                beta_range: self.beta_range,
                schedule: self.schedule,
                seed: self.seed,
            },
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerReply {
    pub(crate) pid: u32,
    pub(crate) result: WorkerResult,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WorkerResult {
    Checked {
        dispatches: u64,
    },
    Solved {
        output: RunOutput,
    },
    Failed {
        kind: WorkerErrorKind,
        detail: String,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WorkerErrorKind {
    Capacity,
    Runtime,
}

pub(crate) fn read_message<T: serde::de::DeserializeOwned>(
    reader: impl Read,
) -> Result<T, AneError> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_MESSAGE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| AneError::Runtime(format!("read message: {error}")))?;
    if bytes.len() as u64 > MAX_MESSAGE_BYTES {
        return Err(AneError::Runtime("message exceeds size limit".into()));
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| AneError::Runtime(format!("parse message: {error}")))
}

struct BoundedBuffer(Vec<u8>);

impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_MESSAGE_BYTES as usize - self.0.len() {
            return Err(std::io::Error::other("message exceeds size limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn write_message<T: Serialize>(
    mut writer: impl Write,
    message: &T,
) -> Result<(), AneError> {
    let mut bytes = BoundedBuffer(Vec::new());
    serde_json::to_writer(&mut bytes, message)
        .map_err(|error| AneError::Runtime(format!("serialize message: {error}")))?;
    writer
        .write_all(&bytes.0)
        .and_then(|()| writer.flush())
        .map_err(|error| AneError::Runtime(format!("write message: {error}")))
}

fn check_device() -> Result<WorkerResult, AneError> {
    let graph = IsingGraph::new(vec![0.0; 2], vec![1.0], vec![(0, 1)]);
    let prepared = crate::graph::prepare(&graph)?;
    let mut program = AneProgram::compile(&prepared, MAX_LANES, crate::native::BLOCK_SWEEPS)?;
    program.reset(&vec![1; 32 * MAX_LANES])?;
    program.advance(&vec![0; 32 * MAX_LANES * crate::native::BLOCK_SWEEPS])?;
    let mut output = vec![0; 32 * MAX_LANES];
    program.read(&mut output)?;
    let valid = output[..MAX_LANES].iter().all(|&spin| spin == -1)
        && output[MAX_LANES..].iter().all(|&spin| spin == 1);
    program.close()?;
    if !valid {
        return Err(AneError::Runtime(
            "startup fused dispatch returned invalid ordered spin updates".into(),
        ));
    }
    Ok(WorkerResult::Checked { dispatches: 1 })
}

/// Run one private worker request, terminating if its owning parent disappears.
pub fn worker_main(parent_pid: u32) -> ExitCode {
    if parent_pid == 0 {
        return ExitCode::from(70);
    }
    if let Err(error) = std::thread::Builder::new()
        .name("ane-parent-watchdog".into())
        .spawn(move || loop {
            if native::parent_pid() != parent_pid {
                std::process::exit(70);
            }
            std::thread::sleep(Duration::from_millis(100));
        })
    {
        tracing::error!(%error, "start parent watchdog");
        return ExitCode::from(70);
    }
    let result = read_message(std::io::stdin().lock()).and_then(|request| match request {
        WorkerRequest::Check => check_device(),
        WorkerRequest::Sample(job) => {
            let (graph, params) = job.into_parts();
            solve_in_process(&graph, &params).map(|output| WorkerResult::Solved { output })
        }
    });
    let result = match result {
        Ok(result) => result,
        Err(AneError::Capacity(detail)) => WorkerResult::Failed {
            kind: WorkerErrorKind::Capacity,
            detail,
        },
        Err(AneError::Runtime(detail)) => WorkerResult::Failed {
            kind: WorkerErrorKind::Runtime,
            detail,
        },
    };
    let reply = WorkerReply {
        pid: std::process::id(),
        result,
    };
    match write_message(std::io::stdout().lock(), &reply) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "write worker reply");
            ExitCode::from(70)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_message_rejects_trailing_data() {
        let bytes = br#"{"command":"check"} {"command":"check"}"#;
        assert!(read_message::<WorkerRequest>(&bytes[..]).is_err());
    }

    #[test]
    fn internal_message_rejects_unknown_and_missing_fields() {
        for bytes in [
            r#"{"command":"unknown"}"#,
            r#"{"command":"sample"}"#,
            r#"{"command":"check","extra":0}"#,
            r#"{"command":"sample","job":{"h":[]}}"#,
        ] {
            assert!(read_message::<WorkerRequest>(bytes.as_bytes()).is_err());
        }
    }

    #[test]
    fn internal_message_rejects_oversize_input() {
        let bytes = vec![b' '; MAX_MESSAGE_BYTES as usize + 1];
        assert!(read_message::<WorkerRequest>(&bytes[..]).is_err());
    }

    #[test]
    fn internal_messages_round_trip_exactly() {
        let request = br#"{"command":"sample","job":{"h":[-1.0,1.0],"j":[1.0],"edges":[[0,1]],"num_reads":2,"num_sweeps":3,"sweeps_per_beta":1,"beta_range":[0.1,2.0],"schedule":"linear","seed":42}}"#;
        let parsed: WorkerRequest = read_message(&request[..]).unwrap();
        let mut bytes = Vec::new();
        write_message(&mut bytes, &parsed).unwrap();
        assert_eq!(bytes, request);
        let reply = br#"{"pid":123,"result":{"status":"checked","dispatches":1}}"#;
        let parsed: WorkerReply = read_message(&reply[..]).unwrap();
        bytes.clear();
        write_message(&mut bytes, &parsed).unwrap();
        assert_eq!(bytes, reply);
    }

    #[test]
    fn messages_preserve_job_fields_and_reject_nested_unknowns() {
        let graph = IsingGraph::new(vec![-1.0, 1.0], vec![1.0], vec![(0, 1)]);
        let params = SampleParams {
            num_reads: 2,
            num_sweeps: 3,
            sweeps_per_beta: 2,
            beta_range: Some((0.25, 4.0)),
            schedule: quip_solver_core::BetaSchedule::Linear,
            seed: 99,
        };
        let (graph_copy, params_copy) = RawJob::from_parts(&graph, &params).into_parts();
        assert_eq!(params_copy.schedule, quip_solver_core::BetaSchedule::Linear);
        assert_eq!(graph_copy.h, vec![-1.0, 1.0]);
        assert_eq!(graph_copy.j, vec![1.0]);
        assert_eq!(*graph_copy.edges, [(0, 1)]);
        assert_eq!(
            (
                params_copy.num_reads,
                params_copy.num_sweeps,
                params_copy.sweeps_per_beta
            ),
            (2, 3, 2)
        );
        assert_eq!(params_copy.beta_range, Some((0.25, 4.0)));
        assert_eq!(params_copy.seed, 99);
        for reply in [
            r#"{"pid":1,"result":{"status":"checked","dispatches":1,"extra":0}}"#,
            r#"{"pid":1,"result":{"status":"failed","kind":"runtime"}}"#,
            r#"{"pid":1,"result":{"status":"solved","output":{"spins":[],"stats":{"programs":0,"dispatches":0,"setup_us":0,"staging_us":0,"dispatch_us":0,"anneal_us":0,"extra":0}}}}"#,
        ] {
            assert!(read_message::<WorkerReply>(reply.as_bytes()).is_err());
        }
    }

    #[test]
    fn internal_message_accepts_exact_limit_and_trailing_whitespace() {
        let mut bytes = br#"{"command":"check"}"#.to_vec();
        bytes.resize(MAX_MESSAGE_BYTES as usize, b' ');
        assert!(read_message::<WorkerRequest>(&bytes[..]).is_ok());
    }

    #[test]
    fn internal_message_writer_rejects_oversize_without_writing() {
        let mut bytes = Vec::new();
        assert!(write_message(&mut bytes, &"x".repeat(MAX_MESSAGE_BYTES as usize)).is_err());
        assert!(bytes.is_empty());
    }
}
