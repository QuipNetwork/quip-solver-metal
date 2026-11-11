// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! Portable little-endian real-valued Ising model files.

use crate::IsingGraph;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// A real-valued Ising graph and its constant energy term.
#[derive(Clone, Debug)]
pub struct RealModel {
    /// Fields, couplings and undirected edges.
    pub graph: IsingGraph,
    /// Constant energy term, independent of the spins.
    pub offset: f64,
}

/// A model file error, including the input or output path.
#[derive(Debug, Error)]
#[error("model file {path}: {message}")]
pub struct ModelFileError {
    path: PathBuf,
    message: String,
}

/// Read a version 1 `.qmodel` file.
///
/// # Errors
/// Returns a path-bearing error for I/O failures or invalid model data.
pub fn read_model(path: &Path) -> Result<RealModel, ModelFileError> {
    let bytes = std::fs::read(path).map_err(|e| file_error(path, e))?;
    parse_model(&bytes).map_err(|e| file_error(path, e))
}

fn file_error(path: &Path, message: impl std::fmt::Display) -> ModelFileError {
    ModelFileError {
        path: path.to_owned(),
        message: message.to_string(),
    }
}

fn take<const N: usize>(bytes: &mut &[u8]) -> Result<[u8; N], String> {
    let value = bytes.get(..N).ok_or("truncated model")?;
    let value = value.try_into().map_err(|_| "truncated model")?;
    *bytes = &bytes[N..];
    Ok(value)
}

fn triangle(n: usize) -> Option<usize> {
    n.checked_mul(n.saturating_sub(1)).map(|v| v / 2)
}

fn parse_model(mut bytes: &[u8]) -> Result<RealModel, String> {
    if &take::<4>(&mut bytes)? != b"QMDL" {
        return Err("invalid magic".into());
    }
    if u32::from_le_bytes(take(&mut bytes)?) != 1 {
        return Err("unsupported version".into());
    }
    let flags = u32::from_le_bytes(take(&mut bytes)?);
    if flags & !1 != 0 {
        return Err("unknown flags".into());
    }
    let dense = flags & 1 != 0;
    let nodes = usize::try_from(u64::from_le_bytes(take(&mut bytes)?))
        .map_err(|_| "node count overflow")?;
    let couplers = usize::try_from(u64::from_le_bytes(take(&mut bytes)?))
        .map_err(|_| "coupler count overflow")?;
    let offset = f64::from_le_bytes(take(&mut bytes)?);
    if dense && triangle(nodes) != Some(couplers) {
        return Err("dense coupler count mismatch".into());
    }
    let expected = nodes
        .checked_mul(8)
        .and_then(|h| {
            couplers
                .checked_mul(if dense { 8 } else { 16 })
                .and_then(|j| h.checked_add(j))
        })
        .ok_or("model size overflow")?;
    if bytes.len() != expected {
        return Err("truncated model or trailing data".into());
    }
    let mut h = Vec::with_capacity(nodes);
    for _ in 0..nodes {
        h.push(f64::from_le_bytes(take(&mut bytes)?));
    }
    let mut edges = Vec::with_capacity(couplers);
    let mut j = Vec::with_capacity(couplers);
    if dense {
        for u in 0..nodes {
            for v in u + 1..nodes {
                edges.push((u, v));
                j.push(f64::from_le_bytes(take(&mut bytes)?));
            }
        }
    } else {
        for _ in 0..couplers {
            let u = u32::from_le_bytes(take(&mut bytes)?) as usize;
            let v = u32::from_le_bytes(take(&mut bytes)?) as usize;
            edges.push((u, v));
            j.push(f64::from_le_bytes(take(&mut bytes)?));
        }
    }
    let model = RealModel {
        graph: IsingGraph::new(h, j, edges),
        offset,
    };
    validate_model(&model)?;
    Ok(model)
}

fn validate_model(model: &RealModel) -> Result<(), String> {
    let graph = &model.graph;
    if graph.j.len() != graph.edges.len() {
        return Err("coupler count mismatch".into());
    }
    if !model.offset.is_finite() || graph.h.iter().chain(&graph.j).any(|v| !v.is_finite()) {
        return Err("non-finite coefficient or offset".into());
    }
    for &(u, v) in graph.edges.iter() {
        if u >= v || v >= graph.h.len() || u32::try_from(v).is_err() {
            return Err(format!(
                "invalid sparse edge ({u}, {v}), expected i < j < n within u32"
            ));
        }
    }
    Ok(())
}

/// Write a version 1 `.qmodel` file, using dense storage for a complete upper triangle.
///
/// # Errors
/// Returns a path-bearing error for I/O failures or invalid model data.
pub fn write_model(path: &Path, model: &RealModel) -> Result<(), ModelFileError> {
    validate_model(model).map_err(|e| file_error(path, e))?;
    write_valid_model(path, model).map_err(|e| file_error(path, e))
}

fn write_valid_model(path: &Path, model: &RealModel) -> std::io::Result<()> {
    use std::io::Write as _;
    let graph = &model.graph;
    let n = graph.h.len();
    let dense = triangle(n) == Some(graph.j.len())
        && graph
            .edges
            .iter()
            .copied()
            .eq((0..n).flat_map(|u| (u + 1..n).map(move |v| (u, v))));
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(b"QMDL")?;
    out.write_all(&1_u32.to_le_bytes())?;
    out.write_all(&u32::from(dense).to_le_bytes())?;
    out.write_all(&(n as u64).to_le_bytes())?;
    out.write_all(&(graph.j.len() as u64).to_le_bytes())?;
    out.write_all(&model.offset.to_le_bytes())?;
    for h in &graph.h {
        out.write_all(&h.to_le_bytes())?;
    }
    for (&(u, v), j) in graph.edges.iter().zip(&graph.j) {
        if !dense {
            out.write_all(&endpoint_bytes(u)?)?;
            out.write_all(&endpoint_bytes(v)?)?;
        }
        out.write_all(&j.to_le_bytes())?;
    }
    out.flush()
}

/// Encode a sparse edge endpoint as little-endian `u32`, failing instead of wrapping.
fn endpoint_bytes(index: usize) -> std::io::Result<[u8; 4]> {
    u32::try_from(index)
        .map(u32::to_le_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(dense: bool) -> RealModel {
        RealModel {
            graph: IsingGraph::new(
                vec![0.5, -0.125, std::f64::consts::PI],
                if dense {
                    vec![0.25, -0.75, 2.0]
                } else {
                    vec![0.25, 2.0]
                },
                if dense {
                    vec![(0, 1), (0, 2), (1, 2)]
                } else {
                    vec![(0, 1), (1, 2)]
                },
            ),
            offset: -1.25,
        }
    }

    #[test]
    fn model_file_round_trips_sparse_and_dense() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.qmodel");
        for dense in [false, true] {
            let original = model(dense);
            write_model(&path, &original).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(&bytes[..4], b"QMDL");
            assert_eq!(
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
                u32::from(dense)
            );
            let restored = read_model(&path).unwrap();
            assert_eq!(restored.graph.h, original.graph.h);
            assert_eq!(restored.graph.j, original.graph.j);
            assert_eq!(restored.graph.edges, original.graph.edges);
            assert_eq!(restored.offset.to_bits(), original.offset.to_bits());
        }
    }

    #[test]
    fn model_file_rejects_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.qmodel");
        for dense in [false, true] {
            write_model(&path, &model(dense)).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            for end in 0..bytes.len() {
                std::fs::write(&path, &bytes[..end]).unwrap();
                let err = read_model(&path).unwrap_err().to_string();
                assert!(err.contains(path.to_str().unwrap()), "{err}");
            }
        }
    }

    #[test]
    fn model_file_rejects_dense_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad-count.qmodel");
        write_model(&path, &model(true)).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[20..28].copy_from_slice(&2_u64.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(read_model(&path).unwrap_err().to_string().contains("dense"));
    }

    #[test]
    fn every_portfolio_fixture_parses() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/portfolio");
        let mut read = 0_usize;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "qmodel") {
                read_model(&path).unwrap();
                read += 1;
            }
        }
        assert!(read > 0, "no .qmodel files in {}", dir.display());
    }
}
