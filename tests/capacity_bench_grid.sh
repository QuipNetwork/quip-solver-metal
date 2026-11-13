#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
#
# The measurement grid of docs/perf/2026-11-metal-capacity.md. It runs the
# ignored test `capacity_run` of tests/capacity_bench.rs once for each grid
# point, and appends one JSON record per run to OUT.
#
# Usage: tests/capacity_bench_grid.sh OUT.jsonl [SERIES...]
#
# SERIES are: auto, complete, sparse, dense, spill. Without SERIES the script
# runs all five. Run it from the repository root, on a Mac, with nothing else
# on the GPU. The full grid takes about one hour.
#
# The sweeps of a point come from one formula:
#
#   sweeps = clamp(BUDGET / (nodes * (degree + 1)), FLOOR, 256)
#
# where degree is the node count for a complete model. BUDGET and FLOOR are:
#
#   series             BUDGET        FLOOR   repeats
#   auto, complete     400000000     16      1 (3 at the boundary rows)
#   sparse (forced)    100000000     16      1
#   dense (forced)     400000000     16      1
#   spill              100000000     1       3
#
# The boundary rows are degrees 62 and 200 at the two sizes on each side of
# the SA local/global limit (4,500 and 4,700), the Gibbs shared/spill limit
# (4,700 and 4,900), and the MSA shared/spill limit (4,000 and 8,000).
set -u

if [ $# -lt 1 ]; then
	echo "usage: $0 OUT.jsonl [auto|complete|sparse|dense|spill ...]" >&2
	exit 2
fi
OUT=$1
shift
SERIES=("$@")
[ ${#SERIES[@]} -eq 0 ] && SERIES=(auto complete sparse dense spill)

# Find the test binary. `--no-run` builds it and prints its path as JSON.
BIN=$(cargo test --release --test capacity_bench --no-run --message-format=json 2>/dev/null |
	jq -r 'select(.executable != null and .target.name == "capacity_bench") | .executable' |
	tail -n 1)
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
	echo "cannot find the capacity_bench test binary" >&2
	exit 1
fi

export QUIP_BENCH_OUT=$OUT
export QUIP_BENCH_READS=256

# point KERNEL NODES DEGREE PRECISION STORAGE BUDGET FLOOR REPEATS
point() {
	local kernel=$1 nodes=$2 degree=$3 precision=$4 storage=$5 budget=$6 floor=$7 repeats=$8
	local eff=$degree
	[ "$degree" = complete ] && eff=$nodes
	local sweeps=$((budget / (nodes * (eff + 1))))
	[ "$sweeps" -gt 256 ] && sweeps=256
	[ "$sweeps" -lt "$floor" ] && sweeps=$floor
	local rep
	for ((rep = 1; rep <= repeats; rep++)); do
		echo "$(date +%T) $kernel nodes=$nodes degree=$degree $precision $storage sweeps=$sweeps rep=$rep"
		QUIP_BENCH_KERNEL=$kernel QUIP_BENCH_NODES=$nodes QUIP_BENCH_DEGREE=$degree \
			QUIP_BENCH_PRECISION=$precision QUIP_BENCH_STORAGE=$storage QUIP_BENCH_SWEEPS=$sweeps \
			timeout 180 "$BIN" capacity_run --ignored --nocapture 2>&1 | grep -E 'panicked|error|FAILED'
	done
}

sizes() {
	case $1 in
	sa) echo "1000 2000 4500 4700 10000 40000" ;;
	gibbs) echo "1000 2000 4700 4900 10000 40000" ;;
	msa) echo "1000 4000 8000 16000 40000" ;;
	esac
}
boundary() {
	case $1 in
	sa) echo "4500 4700" ;;
	gibbs) echo "4700 4900" ;;
	msa) echo "4000 8000" ;;
	esac
}
complete_sizes() {
	case $1 in
	sa) echo "500 1000 1414 2500 5000" ;;
	gibbs) echo "500 1000 1673 3000 5000" ;;
	msa) echo "1000 2500 4898 6000" ;;
	esac
}
# Series of the spill and global tiers, past the sizes of the auto series.
spill_points() { # prints "DEGREE NODES" lines for a kernel
	case $1 in
	sa) printf '62 20000\n62 80000\n' ;;
	gibbs) for d in 62 200; do for n in 100000 200000 400000; do echo "$d $n"; done; done ;;
	msa) for n in 100000 200000 400000; do echo "62 $n"; done ;;
	esac
}

run_auto() {
	local kernel=$1 degree nodes p precisions reps
	local -a list bound
	read -r -a list <<<"$(sizes "$kernel")"
	read -r -a bound <<<"$(boundary "$kernel")"
	for degree in 4 20 62 200; do
		for nodes in "${list[@]}"; do
			precisions=(int8 int32)
			[ "$degree" = 20 ] && precisions=(binary int8 int32 float)
			reps=1
			if [ "$degree" = 62 ] || [ "$degree" = 200 ]; then
				for b in "${bound[@]}"; do [ "$nodes" = "$b" ] && reps=3; done
			fi
			for p in "${precisions[@]}"; do point "$kernel" "$nodes" "$degree" "$p" auto 400000000 16 "$reps"; done
		done
	done
}
run_complete() {
	local kernel=$1 nodes p
	local -a list
	read -r -a list <<<"$(complete_sizes "$kernel")"
	for nodes in "${list[@]}"; do
		for p in binary int8 int32 float; do point "$kernel" "$nodes" complete "$p" auto 400000000 16 1; done
	done
}
run_sparse() {
	local kernel=$1 degree nodes p
	for degree in 62 200 complete; do
		for nodes in 1000 1414 2500; do
			for p in int8 int32; do point "$kernel" "$nodes" "$degree" "$p" sparse 100000000 16 1; done
		done
	done
}
run_dense() {
	local kernel=$1 degree nodes p
	for degree in 62 200; do
		for nodes in 1000 1414; do
			for p in int8 int32; do point "$kernel" "$nodes" "$degree" "$p" dense 400000000 16 1; done
		done
	done
}
run_spill() {
	local kernel=$1 degree nodes p
	while read -r degree nodes; do
		for p in int8 int32; do point "$kernel" "$nodes" "$degree" "$p" auto 100000000 1 3; done
	done < <(spill_points "$kernel")
}

for kernel in sa gibbs msa; do
	for s in "${SERIES[@]}"; do
		"run_$s" "$kernel"
	done
done
echo "GRID_DONE $(date +%T)"
