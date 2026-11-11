// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! `--probe-capacity`: the largest model each connectivity and precision
//! admits, from the capacity model alone. It opens no device and connects
//! nowhere.

use crate::capacity::{max_model, Connectivity, DeviceLimits, Precision};
use crate::sampler::Kernel;
use std::io::Write;

/// The largest model of one connectivity and precision.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeCell {
    /// `degree-20`, `degree-60`, or `complete`.
    pub connectivity: String,
    /// Coefficient type.
    pub precision: Precision,
    /// Nodes of the largest admitted model; 0 when none fits.
    pub nodes: usize,
    /// Couplers of that model; 0 when none fits.
    pub couplers: usize,
    /// Bytes one model needs in the chosen tier; 0 when none fits.
    pub bytes_per_instance: usize,
    /// Storage tier name, or `none` when no model fits.
    pub tier: &'static str,
    /// Measured slowdown against the threadgroup tier. Always `None`: the
    /// capacity model measures no speed.
    pub slowdown: Option<f64>,
}

/// Every precision, in declaration order.
const PRECISIONS: [Precision; 4] = [
    Precision::Binary,
    Precision::Int8,
    Precision::Int32,
    Precision::Float32,
];

/// Lowercase name of a precision, as the JSON output prints it.
const fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::Binary => "binary",
        Precision::Int8 => "int8",
        Precision::Int32 => "int32",
        Precision::Float32 => "float32",
    }
}

/// Name of a kernel, as the binaries and the JSON output print it.
const fn kernel_name(kernel: Kernel) -> &'static str {
    match kernel {
        Kernel::Sa => "sa",
        Kernel::Gibbs => "gibbs",
        Kernel::Msa => "msa",
    }
}

/// The twelve cells: degree 20, degree 60, then complete, each at the four
/// precisions.
#[must_use]
pub fn probe(kernel: Kernel, limits: &DeviceLimits, reads: usize) -> Vec<ProbeCell> {
    let connectivities = [
        ("degree-20", Connectivity::Degree(20)),
        ("degree-60", Connectivity::Degree(60)),
        ("complete", Connectivity::Complete),
    ];
    let mut cells = Vec::with_capacity(connectivities.len() * PRECISIONS.len());
    for (name, connectivity) in connectivities {
        for precision in PRECISIONS {
            let (nodes, couplers, bytes_per_instance, tier) =
                match max_model(kernel, connectivity, precision, limits, reads) {
                    Some((shape, footprint)) => (
                        shape.nodes,
                        shape.edges,
                        footprint.per_instance_bytes,
                        footprint.storage.name(kernel),
                    ),
                    None => (0, 0, 0, "none"),
                };
            cells.push(ProbeCell {
                connectivity: name.to_owned(),
                precision,
                nodes,
                couplers,
                bytes_per_instance,
                tier,
                slowdown: None,
            });
        }
    }
    cells
}

/// Print [`probe`] for `kernel` on stdout, as JSON or as a table. The reads
/// are the kernel's read cap, as the advertised limits use.
pub fn print_probe(kernel: Kernel, limits: &DeviceLimits, json: bool) {
    let text = render_probe(kernel, limits, json);
    if let Err(error) = std::io::stdout().write_all(text.as_bytes()) {
        tracing::warn!(%error, "could not write the capacity probe to stdout");
    }
}

/// The text [`print_probe`] prints.
fn render_probe(kernel: Kernel, limits: &DeviceLimits, json: bool) -> String {
    let reads = crate::streaming::max_reads(kernel) as usize;
    let cells = probe(kernel, limits, reads);
    if json {
        // Every string is a fixed ASCII name, so no escaping is needed.
        let cells: Vec<String> = cells
            .iter()
            .map(|cell| {
                format!(
                    "{{\"connectivity\":\"{}\",\"precision\":\"{}\",\"nodes\":{},\
                     \"couplers\":{},\"bytes_per_instance\":{},\"tier\":\"{}\",\
                     \"slowdown\":{}}}",
                    cell.connectivity,
                    precision_name(cell.precision),
                    cell.nodes,
                    cell.couplers,
                    cell.bytes_per_instance,
                    cell.tier,
                    cell.slowdown
                        .map_or_else(|| "null".to_owned(), |x| x.to_string()),
                )
            })
            .collect();
        return format!(
            "{{\"kernel\":\"{}\",\"cells\":[{}]}}\n",
            kernel_name(kernel),
            cells.join(",")
        );
    }
    let mut text = format!(
        "kernel {}: {} usable bytes\n{:<10} {:<8} {:>12} {:>16} {:>18} tier\n",
        kernel_name(kernel),
        limits.usable_bytes(),
        "graph",
        "type",
        "nodes",
        "couplers",
        "bytes/instance",
    );
    for cell in &cells {
        text.push_str(&format!(
            "{:<10} {:<8} {:>12} {:>16} {:>18} {}\n",
            cell.connectivity,
            precision_name(cell.precision),
            cell.nodes,
            cell.couplers,
            cell.bytes_per_instance,
            cell.tier
        ));
    }
    text
}
