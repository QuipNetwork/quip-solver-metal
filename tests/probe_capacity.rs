// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

//! `--probe-capacity` on each binary: twelve cells, no device, no
//! coordinator.

use serde_json::Value;
use std::process::Command;

fn probe(path: &str) -> Value {
    let out = Command::new(path)
        .args(["--probe-capacity", "--json"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{path} --probe-capacity --json failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn each_binary_prints_twelve_cells() {
    let bins = [
        ("sa", env!("CARGO_BIN_EXE_quip-metal-sa")),
        ("gibbs", env!("CARGO_BIN_EXE_quip-metal-gibbs")),
        ("msa", env!("CARGO_BIN_EXE_quip-metal-msa")),
    ];
    for (kernel, path) in bins {
        let report = probe(path);
        assert_eq!(report["kernel"], kernel, "{path}");
        let cells = report["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 12, "{path}");
        for cell in cells {
            assert!(cell["slowdown"].is_null(), "{path}: {cell}");
        }
        assert_eq!(cells[0]["connectivity"], "degree-20");
        assert_eq!(cells[0]["precision"], "binary");
        assert_eq!(cells[4]["connectivity"], "degree-60");
        assert_eq!(cells[8]["connectivity"], "complete");
    }
}

#[test]
fn each_binary_prints_the_probe_table_in_grid_order() {
    for (kernel, path) in [
        ("sa", env!("CARGO_BIN_EXE_quip-metal-sa")),
        ("gibbs", env!("CARGO_BIN_EXE_quip-metal-gibbs")),
        ("msa", env!("CARGO_BIN_EXE_quip-metal-msa")),
    ] {
        let output = Command::new(path).arg("--probe-capacity").output().unwrap();
        assert!(
            output.status.success(),
            "{path}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let table = String::from_utf8(output.stdout).unwrap();
        let lines: Vec<_> = table.lines().collect();
        assert_eq!(lines.len(), 14, "{kernel}: {table}");
        assert!(lines[0].starts_with(&format!("kernel {kernel}: ")));
        assert!(lines[1].contains("bytes/instance"));
        for (row, connectivity) in ["degree-20", "degree-60", "complete"]
            .into_iter()
            .enumerate()
        {
            for (column, precision) in ["binary", "int8", "int32", "float32"]
                .into_iter()
                .enumerate()
            {
                let cells: Vec<_> = lines[2 + row * 4 + column].split_whitespace().collect();
                assert_eq!(cells.len(), 6, "{kernel}: {cells:?}");
                assert_eq!(cells[0], connectivity);
                assert_eq!(cells[1], precision);
                for value in &cells[2..5] {
                    value.parse::<usize>().unwrap();
                }
                assert!(
                    ["local", "global", "shared", "spill", "dense", "none"].contains(&cells[5]),
                    "{kernel}: {cells:?}"
                );
            }
        }
    }
}
