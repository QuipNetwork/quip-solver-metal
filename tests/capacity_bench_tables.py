#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Turns the JSONL of tests/capacity_bench_grid.sh into the Markdown tables of
docs/perf/2026-11-metal-capacity.md.

Usage: python3 tests/capacity_bench_tables.py OUT.jsonl SECTION

SECTION is one of: sa, gibbs, msa (the results of one kernel), compare (dense
against sparse), over400 (runs with a chunk over 400 ms), spill (the spill and
global tiers past 20,000 nodes).

Points that ran more than once show the median of their runs. The spread
column shows the lowest and highest worst ms per sweep of the runs.
"""
import json
import re
import statistics
import sys
from collections import defaultdict

STORAGE_ORDER = {"auto": 0, "dense": 1, "sparse": 2}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def group(records):
    """Groups records of one point: the same model, storage and sweeps."""
    groups = defaultdict(list)
    for r in records:
        key = (r["kernel"], r["storage_requested"], r["degree"], r["precision"], r["nodes"])
        groups[key].append(r)
    return groups


def degree_key(degree):
    return 10**9 if degree == "complete" else degree


def sort_key(key):
    kernel, storage, degree, precision, nodes = key
    return (STORAGE_ORDER[storage], degree_key(degree), precision, nodes)


def med(runs, field):
    return statistics.median(r[field] for r in runs)


def spread(runs):
    values = [r["worst_ms_per_sweep"] for r in runs]
    return "-" if len(values) == 1 else f"{min(values):.3f} to {max(values):.3f}"


def results(records, kernel, min_nodes=0, refused=True):
    print("| Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | ns per update | Mean ms per sweep | Worst ms per sweep | Spread of worst | Longest chunk (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = group(r for r in records if r["kernel"] == kernel and r["status"] == "ok" and r["nodes"] >= min_nodes)
    for key in sorted(groups, key=sort_key):
        runs = groups[key]
        first = runs[0]
        print(
            f"| {key[1]} | {key[2]} | {key[3]} | {key[4]} | {first['storage']} | {len(runs)} | "
            f"{first['chunks']} | {first['sweeps']} | {med(runs, 'ns_per_update'):.3f} | "
            f"{med(runs, 'mean_ms_per_sweep'):.3f} | {med(runs, 'worst_ms_per_sweep'):.3f} | "
            f"{spread(runs)} | {med(runs, 'max_chunk_ms'):.3f} |"
        )
    if not refused:
        return
    print()
    print(f"Refused {kernel} runs:")
    print()
    print("| Storage | Degree | Precision | Nodes | Reason |")
    print("|---|---|---|---|---|")
    seen = set()
    for r in sorted((r for r in records if r["kernel"] == kernel and r["status"] == "refused"),
                    key=lambda r: sort_key((r["kernel"], r["storage_requested"], r["degree"], r["precision"], r["nodes"]))):
        key = (r["storage_requested"], r["degree"], r["precision"], r["nodes"])
        if key in seen:
            continue
        seen.add(key)
        reason = r["error"].replace("job exceeds Metal backend capacity: ", "")
        reason = re.sub(
            r"\w+ model of \d+ nodes: one dense-matrix sweep is estimated at ([\d.]+) ms, past the 400 ms limit of one chunk",
            lambda m: f"one dense sweep costs an estimated {float(m.group(1)):.0f} ms, over the 400 ms limit",
            reason,
        )
        reason = re.sub(
            r"\w+ model of \d+ nodes: (\d+) bytes needed; (\d+) bytes usable",
            lambda m: f"the model needs {int(m.group(1)):,} bytes and {int(m.group(2)):,} are usable",
            reason,
        )
        reason = re.sub(
            r"graph max degree (\d+) exceeds the multi-spin kernel's 62-neighbour budget",
            r"degree \1 is over the 62-neighbour limit of the multi-spin kernel",
            reason,
        )
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {reason} |")


def compare(records):
    print("| Kernel | Degree | Nodes | Precision | Auto tier | Auto ns | Sparse ns | Dense ns |")
    print("|---|---|---|---|---|---|---|---|")
    groups = group(r for r in records if r["status"] == "ok")
    for kernel in ("sa", "gibbs", "msa"):
        for degree in (62, 200):
            for nodes in (1000, 1414):
                for precision in ("int8", "int32"):
                    cells = {}
                    for storage in ("auto", "sparse", "dense"):
                        runs = groups.get((kernel, storage, degree, precision, nodes))
                        cells[storage] = runs
                    fmt = lambda runs: f"{med(runs, 'ns_per_update'):.3f}" if runs else "n/a"
                    tier = cells["auto"][0]["storage"] if cells["auto"] else "n/a"
                    print(f"| {kernel} | {degree} | {nodes} | {precision} | {tier} | {fmt(cells['auto'])} | {fmt(cells['sparse'])} | {fmt(cells['dense'])} |")


def over400(records):
    print("| Kernel | Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | Median longest chunk (ms) | Largest longest chunk (ms) | Median worst ms per sweep |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for key, runs in group(r for r in records if r["status"] == "ok").items():
        top = max(r["max_chunk_ms"] for r in runs)
        if top > 400:
            rows.append((top, key, runs))
    for top, key, runs in sorted(rows, key=lambda x: -x[0]):
        first = runs[0]
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {first['storage']} | {len(runs)} | "
            f"{first['chunks']} | {first['sweeps']} | {med(runs, 'max_chunk_ms'):.3f} | {top:.3f} | {med(runs, 'worst_ms_per_sweep'):.3f} |"
        )


def main():
    records = load(sys.argv[1])
    section = sys.argv[2]
    if section in ("sa", "gibbs", "msa"):
        results(records, section)
    elif section == "compare":
        compare(records)
    elif section == "over400":
        over400(records)
    elif section == "spill":
        for kernel in ("sa", "gibbs", "msa"):
            print(f"{kernel}:")
            print()
            results([r for r in records if r["status"] == "ok" and r["storage"] in ("global", "spill")], kernel, 20000, refused=False)
    else:
        sys.exit(f"unknown section {section}")


if __name__ == "__main__":
    main()
