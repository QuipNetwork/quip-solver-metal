# Metal capacity measurements

These measurements record what one sweep costs on the Metal SA, Gibbs and MSA kernels, by model size, degree, precision, and storage tier. Task 5.3b uses them to calibrate the chunk planner by degree and to set a one-sweep refusal for each build. The raw records are in `docs/perf/2026-11-metal-capacity.jsonl`. The probe output is in `docs/perf/2026-11-metal-capacity-probe.jsonl`.

## Host

| Item | Value |
|---|---|
| Chip | Apple M4 Max, 16 CPU cores, 40 GPU cores |
| Memory | 128 GiB |
| macOS | 26.7.1 (Darwin 25.6.0) |
| Load average after the grid | 3.34, 3.43, 3.63 (1, 5 and 15 minutes) |
| Crate commit | `7a03688`, with a clean `src`, `kernels` and `Cargo.toml` in every record |

Other sessions shared the machine during the grid. The load average was 3 to 5 whenever someone sampled it. Nobody recorded the load average before the grid, so the table gives the value after it. Contention from those sessions may add to the GPU times. The grid took about 33 minutes. The harness wall times of its 418 runs add up to 32 minutes.

## What a record means

One run anneals one frustrated model with random-sign couplings. A regular model is a circulant graph. Node `i` links to the next `degree / 2` nodes, so each node has `degree` neighbours. A complete model couples every pair. Precision sets the coefficient values: `binary` uses couplings of plus or minus 1, `int8` uses multiples of 0.5, `int32` uses 1.3, 0.127 and 2.0, and `float` uses values off the milli grid.

- A spin update is one visit of one spin of one read. For MSA, a read is one of the 32 replicas of a word.
- `ns per update` is the GPU time of the whole run divided by `reads dispatched * nodes * sweeps`.
- Every run asks for 256 reads, so the dispatch runs 256 reads.
- `Sweeps` is the number of sweeps that the plan ran. It can be a little below the number that the script asked for, because the schedule has `sweeps / sweeps per beta` rungs of equal length.
- `Mean ms per sweep` is the GPU time of the whole run divided by its sweeps.
- `Worst ms per sweep` is the largest value of `chunk ms / chunk sweeps` over the chunks of the run. The record keeps every chunk in `chunk_log`, as `{ms, sweeps, beta_start, sweep_start}`.
- `Longest chunk` is the GPU time of the longest command buffer.
- A run that the admission refuses has `status: refused` and no timing.
- A point that ran three times shows the median of its runs. The spread column gives the lowest and highest worst ms per sweep of those runs.

The top of `tests/capacity_bench.rs` documents the format of a record.

### Which value to calibrate on

Task 5.3b calibrates the degree term of the planner on the mean, `gpu_ms / sweeps`. It calibrates the one-sweep refusal on the worst per-chunk `ms / sweeps`, because the mean hides the slowest sweep. The gap is largest on SA. The worst chunk per sweep took a median of 1.95 times the mean, and at most 3.38 times, over SA runs on auto storage with eight or more chunks. SA at degree 62, 20,000 nodes and `int32` has a mean of 74.8 ms per sweep and a worst of 210.8 ms. The first sweeps of an anneal are the slowest, because many spins flip while the schedule is hot. Gibbs has a median ratio of 1.23 and MSA of 1.07.

## Commands

The `QUIP_BENCH_*` variables drive the harness, which is one ignored test. The script `tests/capacity_bench_grid.sh` builds the release test binary and takes its path from the build output. It then runs the test once for each grid point. To run the whole grid, or one series of it:

```sh
export CARGO_TARGET_DIR=/Users/carback1/Code/quip/quip-miner-metal/target
tests/capacity_bench_grid.sh docs/perf/2026-11-metal-capacity.jsonl            # all series
tests/capacity_bench_grid.sh docs/perf/2026-11-metal-capacity.jsonl spill      # one series
```

The script finds the binary with `cargo test --release --test capacity_bench --no-run --message-format=json` and reads the `executable` field of the output. To run one point by hand, use the test name instead:

```sh
QUIP_BENCH_KERNEL=sa QUIP_BENCH_NODES=2000 QUIP_BENCH_DEGREE=20 \
  QUIP_BENCH_PRECISION=int8 QUIP_BENCH_STORAGE=auto QUIP_BENCH_READS=256 \
  QUIP_BENCH_SWEEPS=256 QUIP_BENCH_OUT=run.jsonl \
  cargo test --release --test capacity_bench capacity_run -- --ignored --nocapture
```

The tables below come from `python3 tests/capacity_bench_tables.py docs/perf/2026-11-metal-capacity.jsonl SECTION`, where `SECTION` is `sa`, `gibbs`, `msa`, `compare` or `over400`.

The sweeps of a point follow one formula, which the header of the script repeats:

```text
sweeps = clamp(BUDGET / (nodes * (degree + 1)), FLOOR, 256)
```

Here `degree` is the node count for a complete model. The script used these budgets, floors and repeats:

| Series | BUDGET | FLOOR | Repeats | Points |
|---|---|---|---|---|
| `auto` | 400,000,000 | 16 | 1, or 3 on the boundary rows | Degrees 4, 20, 62 and 200, auto storage. `int8` and `int32` everywhere, and `binary` and `float` as well at degree 20. |
| `complete` | 400,000,000 | 16 | 1 | Complete models, auto storage, all four precisions. |
| `sparse` | 100,000,000 | 16 | 1 | Forced sparse at degrees 62, 200 and complete, 1,000, 1,414 and 2,500 nodes, `int8` and `int32`. |
| `dense` | 400,000,000 | 16 | 1 | Forced dense at degrees 62 and 200, 1,000 and 1,414 nodes, `int8` and `int32`. |
| `spill` | 100,000,000 | 1 | 3 | Gibbs at degrees 62 and 200, and MSA at degree 62, for 100,000, 200,000 and 400,000 nodes. SA at degree 62 for 20,000 and 80,000 nodes. `int8` and `int32`. |

The sizes of the `auto` series are 1,000, 2,000, 4,500, 4,700, 10,000 and 40,000 nodes for SA, 1,000, 2,000, 4,700, 4,900, 10,000 and 40,000 for Gibbs, and 1,000, 4,000, 8,000, 16,000 and 40,000 for MSA. The boundary rows are degrees 62 and 200 at the two sizes next to each tier limit: 4,500 and 4,700 for SA (local build up to 4,593 nodes), 4,700 and 4,900 for Gibbs (4,800-spin local prefix), and 4,000 and 8,000 for MSA. The boundary rows and every `spill` row ran three times. The `complete` sizes are 500, 1,000, 1,414, 2,500 and 5,000 for SA, 500, 1,000, 1,673, 3,000 and 5,000 for Gibbs, and 1,000, 2,500, 4,898 and 6,000 for MSA. Each point ran with a timeout of 180 seconds.

The probe commands were:

```sh
target/release/quip-metal-sa --probe-capacity
target/release/quip-metal-gibbs --probe-capacity
target/release/quip-metal-msa --probe-capacity
target/release/quip-metal-sa --probe-capacity --json   # and the same for gibbs and msa
```

## Capacity probe output

### Simulated annealing probe

```text
kernel sa: 54975581388 usable bytes
graph      type            nodes         couplers     bytes/instance tier
degree-20  binary         269474          2694740          114529910 global
degree-20  int8           269474          2694740          114529910 global
degree-20  int32           91184           911840          114530180 global
degree-20  float32         91184           911840          114530180 global
degree-60  binary         183244          5497320          114530832 global
degree-60  int8           183244          5497320          114530832 global
degree-60  int32           72670          2180100          114531124 global
degree-60  float32         72670          2180100          114531124 global
complete   binary          29012        420833566          114528198 dense
complete   int8            10542         55561611          114520946 dense
complete   int32            5216         13600720          114525568 dense
complete   float32          5216         13600720          114525568 dense
```

### Gibbs probe

```text
kernel gibbs: 54975581388 usable bytes
graph      type            nodes         couplers     bytes/instance tier
degree-20  binary       23272346        232723460         9147081774 spill
degree-20  int8         23272346        232723460         9147081774 spill
degree-20  int32        20061782        200617820         9149222260 spill
degree-20  float32      20061782        200617820         9149222260 spill
degree-60  binary       15432140        462964200         9152308752 spill
degree-60  int8         15432140        462964200         9152308752 spill
degree-60  int32        11795983        353879490         9154732444 spill
degree-60  float32      11795983        353879490         9154732444 spill
complete   binary          65536       2147450880          556860416 dense
complete   int8            65536       2147450880         4314956800 dense
complete   int32           47821       1143400110         9162405592 dense
complete   float32         47821       1143400110         9162405592 dense
```

### Multi-spin probe

```text
kernel msa: 54975581388 usable bytes
graph      type            nodes         couplers     bytes/instance tier
degree-20  binary       15759832        157598320         1339717816 spill
degree-20  int8         15759832        157598320         1339717816 spill
degree-20  int32         9149516         91495160         1354260592 spill
degree-20  float32       9149516         91495160         1354260592 spill
degree-60  binary       10474521        314235630         1309447445 spill
degree-60  int8         10474521        314235630         1309447445 spill
degree-60  int32         4373829        131214870         1347271524 spill
degree-60  float32       4373829        131214870         1347271524 spill
complete   binary          65536       2147450880          541262848 dense
complete   int8            37038        685888203         1374353074 dense
complete   int32           18526        171597075         1374242632 dense
complete   float32         18526        171597075         1374242632 dense
```

## Results by kernel

In every table below, the storage column is the storage that the run asked for, and the tier column is the tier that ran. Each table holds the rows of all five series of its kernel, sorted by storage, degree, precision, and nodes.

### Simulated annealing results

`python3 tests/capacity_bench_tables.py docs/perf/2026-11-metal-capacity.jsonl sa`

| Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | ns per update | Mean ms per sweep | Worst ms per sweep | Spread of worst | Longest chunk (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 1000 | local | 1 | 8 | 256 | 1.992 | 0.510 | 0.781 | - | 28.119 |
| auto | 4 | int32 | 2000 | local | 1 | 16 | 256 | 2.030 | 1.039 | 1.628 | - | 26.042 |
| auto | 4 | int32 | 4500 | local | 1 | 32 | 256 | 2.116 | 2.437 | 3.923 | - | 31.387 |
| auto | 4 | int32 | 4700 | global | 1 | 64 | 256 | 2.084 | 2.507 | 4.378 | - | 17.514 |
| auto | 4 | int32 | 10000 | global | 1 | 256 | 256 | 2.620 | 6.707 | 17.005 | - | 17.005 |
| auto | 4 | int32 | 40000 | global | 1 | 256 | 256 | 2.819 | 28.863 | 70.816 | - | 70.816 |
| auto | 4 | int8 | 1000 | local | 1 | 8 | 256 | 2.767 | 0.708 | 1.611 | - | 57.984 |
| auto | 4 | int8 | 2000 | local | 1 | 16 | 256 | 2.321 | 1.188 | 1.626 | - | 26.010 |
| auto | 4 | int8 | 4500 | local | 1 | 32 | 256 | 2.409 | 2.775 | 3.914 | - | 31.310 |
| auto | 4 | int8 | 4700 | global | 1 | 64 | 256 | 2.360 | 2.840 | 4.301 | - | 17.205 |
| auto | 4 | int8 | 10000 | global | 1 | 256 | 256 | 2.901 | 7.426 | 16.232 | - | 16.232 |
| auto | 4 | int8 | 40000 | global | 1 | 256 | 256 | 3.021 | 30.931 | 67.201 | - | 67.201 |
| auto | 20 | binary | 1000 | local | 1 | 5 | 256 | 3.714 | 0.951 | 1.317 | - | 68.501 |
| auto | 20 | binary | 2000 | local | 1 | 11 | 256 | 3.732 | 1.911 | 2.766 | - | 66.384 |
| auto | 20 | binary | 4500 | local | 1 | 22 | 256 | 3.790 | 4.367 | 6.761 | - | 81.137 |
| auto | 20 | binary | 4700 | global | 1 | 64 | 256 | 4.786 | 5.758 | 8.465 | - | 33.861 |
| auto | 20 | binary | 10000 | global | 1 | 128 | 256 | 5.072 | 12.985 | 20.000 | - | 39.999 |
| auto | 20 | binary | 40000 | global | 1 | 256 | 256 | 5.826 | 59.659 | 115.233 | - | 115.233 |
| auto | 20 | float | 1000 | local | 1 | 7 | 256 | 3.098 | 0.793 | 1.338 | - | 53.516 |
| auto | 20 | float | 2000 | local | 1 | 13 | 256 | 3.059 | 1.566 | 2.712 | - | 54.234 |
| auto | 20 | float | 4500 | local | 1 | 32 | 256 | 3.228 | 3.718 | 6.247 | - | 49.979 |
| auto | 20 | float | 4700 | global | 1 | 64 | 256 | 4.626 | 5.567 | 16.424 | - | 65.698 |
| auto | 20 | float | 10000 | global | 1 | 128 | 256 | 4.981 | 12.750 | 26.460 | - | 52.921 |
| auto | 20 | float | 40000 | global | 1 | 256 | 256 | 5.576 | 57.100 | 157.051 | - | 157.051 |
| auto | 20 | int32 | 1000 | local | 1 | 8 | 256 | 3.303 | 0.845 | 1.396 | - | 50.273 |
| auto | 20 | int32 | 2000 | local | 1 | 16 | 256 | 3.275 | 1.677 | 2.826 | - | 45.218 |
| auto | 20 | int32 | 4500 | local | 1 | 32 | 256 | 3.413 | 3.932 | 6.683 | - | 53.463 |
| auto | 20 | int32 | 4700 | global | 1 | 64 | 256 | 4.281 | 5.151 | 9.115 | - | 36.459 |
| auto | 20 | int32 | 10000 | global | 1 | 256 | 256 | 5.232 | 13.393 | 30.987 | - | 30.987 |
| auto | 20 | int32 | 40000 | global | 1 | 256 | 256 | 5.763 | 59.016 | 136.023 | - | 136.023 |
| auto | 20 | int8 | 1000 | local | 1 | 8 | 256 | 4.104 | 1.051 | 1.491 | - | 53.681 |
| auto | 20 | int8 | 2000 | local | 1 | 16 | 256 | 4.110 | 2.104 | 3.088 | - | 49.414 |
| auto | 20 | int8 | 4500 | local | 1 | 32 | 256 | 4.199 | 4.837 | 7.176 | - | 57.407 |
| auto | 20 | int8 | 4700 | global | 1 | 64 | 256 | 4.891 | 5.884 | 9.026 | - | 36.104 |
| auto | 20 | int8 | 10000 | global | 1 | 256 | 256 | 5.664 | 14.499 | 28.913 | - | 28.913 |
| auto | 20 | int8 | 40000 | global | 1 | 256 | 256 | 6.026 | 61.709 | 147.595 | - | 147.595 |
| auto | 62 | int32 | 1000 | local | 1 | 8 | 256 | 8.167 | 2.091 | 3.683 | - | 132.576 |
| auto | 62 | int32 | 2000 | local | 1 | 16 | 256 | 8.178 | 4.187 | 7.539 | - | 120.626 |
| auto | 62 | int32 | 4500 | local | 3 | 32 | 256 | 8.429 | 9.710 | 17.960 | 17.959 to 18.015 | 143.682 |
| auto | 62 | int32 | 4700 | global | 3 | 64 | 256 | 11.175 | 13.446 | 24.995 | 24.950 to 44.928 | 99.982 |
| auto | 62 | int32 | 10000 | global | 1 | 256 | 256 | 13.398 | 34.298 | 85.867 | - | 85.867 |
| auto | 62 | int32 | 20000 | global | 3 | 79 | 79 | 14.604 | 74.770 | 210.846 | 205.514 to 211.393 | 210.846 |
| auto | 62 | int32 | 40000 | global | 1 | 158 | 158 | 14.623 | 149.740 | 416.839 | - | 416.839 |
| auto | 62 | int8 | 1000 | local | 1 | 8 | 256 | 10.486 | 2.685 | 4.031 | - | 145.126 |
| auto | 62 | int8 | 2000 | local | 1 | 16 | 256 | 10.376 | 5.312 | 8.241 | - | 131.853 |
| auto | 62 | int8 | 4500 | local | 3 | 32 | 256 | 10.640 | 12.257 | 19.346 | 19.296 to 19.408 | 154.766 |
| auto | 62 | int8 | 4700 | global | 3 | 64 | 256 | 12.647 | 15.217 | 24.709 | 24.647 to 24.719 | 98.836 |
| auto | 62 | int8 | 10000 | global | 1 | 256 | 256 | 13.890 | 35.559 | 76.556 | - | 76.556 |
| auto | 62 | int8 | 20000 | global | 3 | 79 | 79 | 15.050 | 77.057 | 182.237 | 158.815 to 185.982 | 182.237 |
| auto | 62 | int8 | 40000 | global | 1 | 158 | 158 | 15.000 | 153.600 | 355.651 | - | 355.651 |
| auto | 62 | int8 | 80000 | global | 3 | 19 | 19 | 17.690 | 362.287 | 674.195 | 654.089 to 689.267 | 674.195 |
| auto | 200 | int32 | 1000 | local | 1 | 8 | 256 | 34.119 | 8.735 | 17.009 | - | 612.339 |
| auto | 200 | int32 | 2000 | local | 1 | 16 | 256 | 34.886 | 17.862 | 36.256 | - | 580.097 |
| auto | 200 | int32 | 4500 | local | 3 | 32 | 256 | 35.272 | 40.633 | 83.730 | 83.716 to 84.055 | 669.837 |
| auto | 200 | int32 | 4700 | global | 3 | 64 | 256 | 51.166 | 61.563 | 137.538 | 131.573 to 163.805 | 550.152 |
| auto | 200 | int32 | 10000 | global | 1 | 198 | 198 | 53.146 | 136.055 | 373.122 | - | 373.122 |
| auto | 200 | int32 | 40000 | global | 1 | 49 | 49 | 57.274 | 586.483 | 1373.092 | - | 1373.092 |
| auto | 200 | int8 | 1000 | dense | 1 | 256 | 256 | 206.522 | 52.870 | 178.523 | - | 178.523 |
| auto | 200 | int8 | 2000 | local | 1 | 16 | 256 | 42.491 | 21.755 | 37.453 | - | 599.248 |
| auto | 200 | int8 | 4500 | local | 3 | 32 | 256 | 42.903 | 49.425 | 87.047 | 87.028 to 87.218 | 696.377 |
| auto | 200 | int8 | 4700 | global | 3 | 64 | 256 | 53.791 | 64.721 | 120.822 | 119.213 to 144.293 | 483.287 |
| auto | 200 | int8 | 10000 | global | 1 | 198 | 198 | 56.147 | 143.736 | 324.335 | - | 324.335 |
| auto | 200 | int8 | 40000 | global | 1 | 49 | 49 | 59.919 | 613.566 | 1264.947 | - | 1264.947 |
| auto | complete | binary | 500 | dense | 1 | 64 | 256 | 124.375 | 15.920 | 34.732 | - | 138.929 |
| auto | complete | binary | 1000 | dense | 1 | 256 | 256 | 245.367 | 62.814 | 161.336 | - | 161.336 |
| auto | complete | binary | 1414 | dense | 1 | 198 | 198 | 346.801 | 125.536 | 344.241 | - | 344.241 |
| auto | complete | float | 500 | dense | 1 | 64 | 256 | 55.457 | 7.099 | 15.191 | - | 60.764 |
| auto | complete | float | 1000 | dense | 1 | 256 | 256 | 118.101 | 30.234 | 101.352 | - | 101.352 |
| auto | complete | float | 1414 | dense | 1 | 198 | 198 | 167.032 | 60.463 | 194.726 | - | 194.726 |
| auto | complete | int32 | 500 | dense | 1 | 64 | 256 | 57.412 | 7.349 | 15.618 | - | 62.471 |
| auto | complete | int32 | 1000 | dense | 1 | 256 | 256 | 120.685 | 30.895 | 83.490 | - | 83.490 |
| auto | complete | int32 | 1414 | dense | 1 | 198 | 198 | 170.564 | 61.742 | 175.997 | - | 175.997 |
| auto | complete | int8 | 500 | dense | 1 | 64 | 256 | 113.470 | 14.524 | 28.009 | - | 112.035 |
| auto | complete | int8 | 1000 | dense | 1 | 256 | 256 | 229.809 | 58.831 | 182.523 | - | 182.523 |
| auto | complete | int8 | 1414 | dense | 1 | 198 | 198 | 324.127 | 117.329 | 342.407 | - | 342.407 |
| dense | 62 | int32 | 1000 | dense | 1 | 256 | 256 | 90.095 | 23.064 | 68.719 | - | 68.719 |
| dense | 62 | int32 | 1414 | dense | 1 | 256 | 256 | 133.380 | 48.281 | 151.895 | - | 151.895 |
| dense | 62 | int8 | 1000 | dense | 1 | 256 | 256 | 202.937 | 51.952 | 176.310 | - | 176.310 |
| dense | 62 | int8 | 1414 | dense | 1 | 256 | 256 | 287.416 | 104.040 | 292.391 | - | 292.391 |
| dense | 200 | int32 | 1000 | dense | 1 | 256 | 256 | 95.617 | 24.478 | 70.519 | - | 70.519 |
| dense | 200 | int32 | 1414 | dense | 1 | 256 | 256 | 136.144 | 49.282 | 150.455 | - | 150.455 |
| dense | 200 | int8 | 1000 | dense | 1 | 256 | 256 | 206.210 | 52.790 | 150.058 | - | 150.058 |
| dense | 200 | int8 | 1414 | dense | 1 | 256 | 256 | 287.941 | 104.230 | 297.796 | - | 297.796 |
| sparse | 62 | int32 | 1000 | local | 1 | 8 | 256 | 8.181 | 2.094 | 3.682 | - | 132.549 |
| sparse | 62 | int32 | 1414 | local | 1 | 11 | 256 | 8.237 | 2.982 | 5.261 | - | 126.276 |
| sparse | 62 | int32 | 2500 | local | 1 | 22 | 256 | 8.229 | 5.266 | 9.479 | - | 113.752 |
| sparse | 62 | int8 | 1000 | local | 1 | 8 | 256 | 10.877 | 2.784 | 4.705 | - | 169.395 |
| sparse | 62 | int8 | 1414 | local | 1 | 11 | 256 | 10.470 | 3.790 | 5.744 | - | 137.855 |
| sparse | 62 | int8 | 2500 | local | 1 | 22 | 256 | 10.517 | 6.731 | 10.396 | - | 124.752 |
| sparse | 200 | int32 | 1000 | local | 1 | 8 | 256 | 34.115 | 8.733 | 17.007 | - | 612.256 |
| sparse | 200 | int32 | 1414 | local | 1 | 11 | 256 | 34.774 | 12.588 | 24.437 | - | 586.495 |
| sparse | 200 | int32 | 2500 | local | 1 | 14 | 198 | 35.945 | 23.005 | 46.618 | - | 699.274 |
| sparse | 200 | int8 | 1000 | local | 1 | 8 | 256 | 42.670 | 10.923 | 17.864 | - | 643.109 |
| sparse | 200 | int8 | 1414 | local | 1 | 11 | 256 | 43.351 | 15.693 | 25.314 | - | 607.539 |
| sparse | 200 | int8 | 2500 | local | 1 | 14 | 198 | 42.571 | 27.246 | 45.286 | - | 679.288 |
| sparse | complete | int32 | 1000 | local | 1 | 3 | 99 | 250.962 | 64.246 | 122.210 | - | 4766.183 |
| sparse | complete | int32 | 1414 | local | 1 | 2 | 49 | 383.938 | 138.980 | 234.968 | - | 6344.125 |
| sparse | complete | int32 | 2500 | local | 1 | 2 | 16 | 956.068 | 611.883 | 638.497 | - | 9577.462 |
| sparse | complete | int8 | 1000 | local | 1 | 3 | 99 | 288.547 | 73.868 | 121.611 | - | 4742.837 |
| sparse | complete | int8 | 1414 | local | 1 | 2 | 49 | 435.403 | 157.609 | 245.618 | - | 6631.686 |
| sparse | complete | int8 | 2500 | local | 1 | 2 | 16 | 1011.351 | 647.265 | 677.190 | - | 10157.853 |

Refused sa runs:

| Storage | Degree | Precision | Nodes | Reason |
|---|---|---|---|---|
| auto | 62 | int32 | 80000 | the model needs 61,134,596,556 bytes and 54,975,581,388 are usable |
| auto | complete | binary | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | binary | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |
| auto | complete | float | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | float | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |
| auto | complete | int32 | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | int32 | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |
| auto | complete | int8 | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | int8 | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |

### Gibbs results

`python3 tests/capacity_bench_tables.py docs/perf/2026-11-metal-capacity.jsonl gibbs`

| Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | ns per update | Mean ms per sweep | Worst ms per sweep | Spread of worst | Longest chunk (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 1000 | shared | 1 | 1 | 256 | 0.043 | 0.011 | 0.011 | - | 2.801 |
| auto | 4 | int32 | 2000 | shared | 1 | 1 | 256 | 0.121 | 0.062 | 0.062 | - | 15.829 |
| auto | 4 | int32 | 4700 | shared | 1 | 2 | 256 | 0.063 | 0.075 | 0.083 | - | 12.304 |
| auto | 4 | int32 | 4900 | spill | 1 | 5 | 256 | 0.073 | 0.091 | 0.105 | - | 5.480 |
| auto | 4 | int32 | 10000 | spill | 1 | 11 | 256 | 0.080 | 0.205 | 0.232 | - | 5.566 |
| auto | 4 | int32 | 40000 | spill | 1 | 64 | 256 | 0.067 | 0.689 | 1.062 | - | 4.246 |
| auto | 4 | int8 | 1000 | shared | 1 | 1 | 256 | 0.037 | 0.009 | 0.009 | - | 2.409 |
| auto | 4 | int8 | 2000 | shared | 1 | 1 | 256 | 0.050 | 0.025 | 0.025 | - | 6.509 |
| auto | 4 | int8 | 4700 | shared | 1 | 2 | 256 | 0.094 | 0.113 | 0.137 | - | 23.584 |
| auto | 4 | int8 | 4900 | spill | 1 | 5 | 256 | 0.058 | 0.072 | 0.076 | - | 3.965 |
| auto | 4 | int8 | 10000 | spill | 1 | 11 | 256 | 0.085 | 0.217 | 0.267 | - | 6.410 |
| auto | 4 | int8 | 40000 | spill | 1 | 64 | 256 | 0.065 | 0.669 | 0.903 | - | 3.611 |
| auto | 20 | binary | 1000 | shared | 1 | 1 | 256 | 0.181 | 0.046 | 0.046 | - | 11.851 |
| auto | 20 | binary | 2000 | shared | 1 | 1 | 256 | 0.115 | 0.059 | 0.059 | - | 15.012 |
| auto | 20 | binary | 4700 | shared | 1 | 1 | 256 | 0.144 | 0.173 | 0.173 | - | 44.406 |
| auto | 20 | binary | 4900 | spill | 1 | 3 | 256 | 0.262 | 0.329 | 0.345 | - | 36.171 |
| auto | 20 | binary | 10000 | spill | 1 | 5 | 256 | 0.419 | 1.073 | 1.281 | - | 61.486 |
| auto | 20 | binary | 40000 | spill | 1 | 22 | 256 | 0.625 | 6.402 | 7.110 | - | 85.325 |
| auto | 20 | float | 1000 | shared | 1 | 1 | 256 | 0.177 | 0.045 | 0.045 | - | 11.622 |
| auto | 20 | float | 2000 | shared | 1 | 1 | 256 | 0.182 | 0.093 | 0.093 | - | 23.880 |
| auto | 20 | float | 4700 | shared | 1 | 1 | 256 | 0.140 | 0.168 | 0.168 | - | 43.019 |
| auto | 20 | float | 4900 | spill | 1 | 4 | 256 | 0.350 | 0.439 | 0.539 | - | 39.089 |
| auto | 20 | float | 10000 | spill | 1 | 7 | 256 | 0.419 | 1.073 | 1.280 | - | 47.722 |
| auto | 20 | float | 40000 | spill | 1 | 32 | 256 | 0.585 | 5.995 | 7.519 | - | 60.148 |
| auto | 20 | int32 | 1000 | shared | 1 | 1 | 256 | 0.150 | 0.038 | 0.038 | - | 9.830 |
| auto | 20 | int32 | 2000 | shared | 1 | 1 | 256 | 0.159 | 0.081 | 0.081 | - | 20.794 |
| auto | 20 | int32 | 4700 | shared | 1 | 2 | 256 | 0.154 | 0.185 | 0.186 | - | 31.797 |
| auto | 20 | int32 | 4900 | spill | 1 | 5 | 256 | 0.299 | 0.375 | 0.399 | - | 20.351 |
| auto | 20 | int32 | 10000 | spill | 1 | 11 | 256 | 0.390 | 0.999 | 1.058 | - | 25.401 |
| auto | 20 | int32 | 40000 | spill | 1 | 64 | 256 | 0.553 | 5.659 | 9.774 | - | 39.097 |
| auto | 20 | int8 | 1000 | shared | 1 | 1 | 256 | 0.182 | 0.046 | 0.046 | - | 11.902 |
| auto | 20 | int8 | 2000 | shared | 1 | 1 | 256 | 0.137 | 0.070 | 0.070 | - | 18.011 |
| auto | 20 | int8 | 4700 | shared | 1 | 2 | 256 | 0.153 | 0.184 | 0.188 | - | 32.320 |
| auto | 20 | int8 | 4900 | spill | 1 | 5 | 256 | 0.262 | 0.329 | 0.344 | - | 17.884 |
| auto | 20 | int8 | 10000 | spill | 1 | 11 | 256 | 0.364 | 0.933 | 0.964 | - | 23.144 |
| auto | 20 | int8 | 40000 | spill | 1 | 64 | 256 | 0.481 | 4.921 | 8.297 | - | 33.187 |
| auto | 62 | int32 | 1000 | shared | 1 | 1 | 256 | 0.516 | 0.132 | 0.132 | - | 33.844 |
| auto | 62 | int32 | 2000 | shared | 1 | 1 | 256 | 0.423 | 0.217 | 0.217 | - | 55.451 |
| auto | 62 | int32 | 4700 | shared | 3 | 2 | 256 | 0.655 | 0.788 | 0.811 | 0.801 to 0.863 | 134.321 |
| auto | 62 | int32 | 4900 | spill | 3 | 5 | 256 | 1.147 | 1.439 | 1.630 | 1.389 to 1.736 | 78.590 |
| auto | 62 | int32 | 10000 | spill | 1 | 11 | 256 | 1.317 | 3.372 | 3.859 | - | 92.622 |
| auto | 62 | int32 | 40000 | spill | 1 | 27 | 158 | 2.023 | 20.712 | 23.756 | - | 142.535 |
| auto | 62 | int32 | 100000 | spill | 3 | 8 | 15 | 1.556 | 39.833 | 47.620 | 45.364 to 52.485 | 94.446 |
| auto | 62 | int32 | 200000 | spill | 3 | 7 | 7 | 1.484 | 75.958 | 97.798 | 93.431 to 99.086 | 97.798 |
| auto | 62 | int32 | 400000 | spill | 3 | 3 | 3 | 1.547 | 158.399 | 175.915 | 173.600 to 197.128 | 175.915 |
| auto | 62 | int8 | 1000 | shared | 1 | 1 | 256 | 0.766 | 0.196 | 0.196 | - | 50.230 |
| auto | 62 | int8 | 2000 | shared | 1 | 1 | 256 | 0.440 | 0.225 | 0.225 | - | 57.656 |
| auto | 62 | int8 | 4700 | shared | 3 | 2 | 256 | 0.484 | 0.582 | 0.622 | 0.553 to 0.733 | 96.732 |
| auto | 62 | int8 | 4900 | spill | 3 | 5 | 256 | 1.033 | 1.296 | 1.309 | 1.295 to 1.457 | 68.062 |
| auto | 62 | int8 | 10000 | spill | 1 | 11 | 256 | 1.215 | 3.111 | 3.703 | - | 88.880 |
| auto | 62 | int8 | 40000 | spill | 1 | 27 | 158 | 1.905 | 19.510 | 23.131 | - | 138.787 |
| auto | 62 | int8 | 100000 | spill | 3 | 8 | 15 | 1.451 | 37.143 | 47.889 | 43.697 to 62.818 | 95.777 |
| auto | 62 | int8 | 200000 | spill | 3 | 7 | 7 | 1.399 | 71.613 | 89.347 | 89.164 to 96.102 | 89.347 |
| auto | 62 | int8 | 400000 | spill | 3 | 3 | 3 | 1.430 | 146.449 | 164.094 | 156.265 to 166.905 | 164.094 |
| auto | 200 | int32 | 1000 | shared | 1 | 1 | 256 | 4.458 | 1.141 | 1.141 | - | 292.145 |
| auto | 200 | int32 | 2000 | shared | 1 | 1 | 256 | 2.801 | 1.434 | 1.434 | - | 367.147 |
| auto | 200 | int32 | 4700 | shared | 3 | 2 | 256 | 1.806 | 2.173 | 3.074 | 2.350 to 3.139 | 300.279 |
| auto | 200 | int32 | 4900 | spill | 3 | 5 | 256 | 4.053 | 5.084 | 5.422 | 5.408 to 5.460 | 279.704 |
| auto | 200 | int32 | 10000 | spill | 1 | 8 | 198 | 4.398 | 11.260 | 13.409 | - | 362.034 |
| auto | 200 | int32 | 40000 | spill | 1 | 9 | 49 | 8.708 | 89.169 | 107.640 | - | 645.843 |
| auto | 200 | int32 | 100000 | spill | 3 | 2 | 4 | 5.438 | 139.208 | 144.779 | 144.415 to 145.954 | 289.557 |
| auto | 200 | int32 | 200000 | spill | 3 | 2 | 2 | 7.202 | 368.725 | 417.192 | 406.755 to 626.225 | 417.192 |
| auto | 200 | int32 | 400000 | spill | 3 | 1 | 1 | 10.542 | 1079.465 | 1079.465 | 1075.332 to 1289.163 | 1079.465 |
| auto | 200 | int8 | 1000 | dense | 1 | 256 | 256 | 23.822 | 6.099 | 19.184 | - | 19.184 |
| auto | 200 | int8 | 2000 | shared | 1 | 1 | 256 | 3.423 | 1.752 | 1.752 | - | 448.628 |
| auto | 200 | int8 | 4700 | shared | 3 | 2 | 256 | 1.982 | 2.385 | 2.796 | 2.631 to 2.827 | 373.206 |
| auto | 200 | int8 | 4900 | spill | 3 | 5 | 256 | 3.996 | 5.013 | 5.609 | 5.454 to 6.184 | 283.583 |
| auto | 200 | int8 | 10000 | spill | 1 | 8 | 198 | 5.412 | 13.854 | 20.112 | - | 543.016 |
| auto | 200 | int8 | 40000 | spill | 1 | 9 | 49 | 7.132 | 73.030 | 79.946 | - | 479.677 |
| auto | 200 | int8 | 100000 | spill | 3 | 2 | 4 | 5.155 | 131.974 | 139.194 | 138.440 to 220.338 | 278.388 |
| auto | 200 | int8 | 200000 | spill | 3 | 2 | 2 | 6.657 | 340.856 | 375.755 | 294.106 to 376.569 | 375.755 |
| auto | 200 | int8 | 400000 | spill | 3 | 1 | 1 | 7.022 | 719.047 | 719.047 | 570.369 to 844.717 | 719.047 |
| auto | complete | binary | 500 | dense | 1 | 64 | 256 | 38.867 | 4.975 | 9.501 | - | 38.003 |
| auto | complete | binary | 1000 | dense | 1 | 256 | 256 | 88.365 | 22.622 | 24.756 | - | 24.756 |
| auto | complete | binary | 1673 | dense | 1 | 142 | 142 | 149.532 | 64.043 | 83.117 | - | 83.117 |
| auto | complete | float | 500 | dense | 1 | 64 | 256 | 35.539 | 4.549 | 4.973 | - | 19.893 |
| auto | complete | float | 1000 | dense | 1 | 256 | 256 | 87.994 | 22.526 | 28.932 | - | 28.932 |
| auto | complete | float | 1673 | dense | 1 | 142 | 142 | 143.800 | 61.588 | 80.634 | - | 80.634 |
| auto | complete | int32 | 500 | dense | 1 | 64 | 256 | 40.891 | 5.234 | 5.367 | - | 21.469 |
| auto | complete | int32 | 1000 | dense | 1 | 256 | 256 | 97.019 | 24.837 | 46.039 | - | 46.039 |
| auto | complete | int32 | 1673 | dense | 1 | 142 | 142 | 169.356 | 72.533 | 95.768 | - | 95.768 |
| auto | complete | int8 | 500 | dense | 1 | 64 | 256 | 44.537 | 5.701 | 6.088 | - | 24.353 |
| auto | complete | int8 | 1000 | dense | 1 | 256 | 256 | 87.867 | 22.494 | 44.025 | - | 44.025 |
| auto | complete | int8 | 1673 | dense | 1 | 142 | 142 | 157.563 | 67.483 | 239.786 | - | 239.786 |
| dense | 62 | int32 | 1000 | dense | 1 | 256 | 256 | 4.000 | 1.024 | 1.113 | - | 1.113 |
| dense | 62 | int32 | 1414 | dense | 1 | 256 | 256 | 6.372 | 2.306 | 2.750 | - | 2.750 |
| dense | 62 | int8 | 1000 | dense | 1 | 256 | 256 | 5.027 | 1.287 | 5.524 | - | 5.524 |
| dense | 62 | int8 | 1414 | dense | 1 | 256 | 256 | 8.374 | 3.031 | 3.622 | - | 3.622 |
| dense | 200 | int32 | 1000 | dense | 1 | 256 | 256 | 24.063 | 6.160 | 6.591 | - | 6.591 |
| dense | 200 | int32 | 1414 | dense | 1 | 256 | 256 | 16.148 | 5.845 | 7.122 | - | 7.122 |
| dense | 200 | int8 | 1000 | dense | 1 | 256 | 256 | 29.962 | 7.670 | 28.710 | - | 28.710 |
| dense | 200 | int8 | 1414 | dense | 1 | 256 | 256 | 17.495 | 6.333 | 6.852 | - | 6.852 |
| sparse | 62 | int32 | 1000 | shared | 1 | 1 | 256 | 0.388 | 0.099 | 0.099 | - | 25.411 |
| sparse | 62 | int32 | 1414 | shared | 1 | 1 | 256 | 0.346 | 0.125 | 0.125 | - | 32.075 |
| sparse | 62 | int32 | 2500 | shared | 1 | 1 | 256 | 0.327 | 0.209 | 0.209 | - | 53.569 |
| sparse | 62 | int8 | 1000 | shared | 1 | 1 | 256 | 0.730 | 0.187 | 0.187 | - | 47.847 |
| sparse | 62 | int8 | 1414 | shared | 1 | 1 | 256 | 0.350 | 0.127 | 0.127 | - | 32.429 |
| sparse | 62 | int8 | 2500 | shared | 1 | 1 | 256 | 0.304 | 0.194 | 0.194 | - | 49.734 |
| sparse | 200 | int32 | 1000 | shared | 1 | 1 | 256 | 3.379 | 0.865 | 0.865 | - | 221.466 |
| sparse | 200 | int32 | 1414 | shared | 1 | 1 | 256 | 1.699 | 0.615 | 0.615 | - | 157.466 |
| sparse | 200 | int32 | 2500 | shared | 1 | 1 | 198 | 2.502 | 1.601 | 1.601 | - | 317.054 |
| sparse | 200 | int8 | 1000 | shared | 1 | 1 | 256 | 3.153 | 0.807 | 0.807 | - | 206.663 |
| sparse | 200 | int8 | 1414 | shared | 1 | 1 | 256 | 1.505 | 0.545 | 0.545 | - | 139.437 |
| sparse | 200 | int8 | 2500 | shared | 1 | 1 | 198 | 2.431 | 1.556 | 1.556 | - | 308.074 |
| sparse | complete | int32 | 1000 | shared | 1 | 1 | 99 | 80.016 | 20.484 | 20.484 | - | 2027.921 |
| sparse | complete | int32 | 1414 | shared | 1 | 1 | 49 | 165.779 | 60.009 | 60.009 | - | 2940.457 |
| sparse | complete | int32 | 2500 | shared | 1 | 1 | 16 | 257.452 | 164.769 | 164.769 | - | 2636.306 |
| sparse | complete | int8 | 1000 | shared | 1 | 1 | 99 | 103.829 | 26.580 | 26.580 | - | 2631.430 |
| sparse | complete | int8 | 1414 | shared | 1 | 1 | 49 | 163.203 | 59.077 | 59.077 | - | 2894.765 |
| sparse | complete | int8 | 2500 | shared | 1 | 1 | 16 | 341.741 | 218.714 | 218.714 | - | 3499.423 |

Refused gibbs runs:

| Storage | Degree | Precision | Nodes | Reason |
|---|---|---|---|---|
| auto | complete | binary | 3000 | one dense sweep costs an estimated 1286 ms, over the 400 ms limit |
| auto | complete | binary | 5000 | one dense sweep costs an estimated 3571 ms, over the 400 ms limit |
| auto | complete | float | 3000 | one dense sweep costs an estimated 1286 ms, over the 400 ms limit |
| auto | complete | float | 5000 | one dense sweep costs an estimated 3571 ms, over the 400 ms limit |
| auto | complete | int32 | 3000 | one dense sweep costs an estimated 1286 ms, over the 400 ms limit |
| auto | complete | int32 | 5000 | one dense sweep costs an estimated 3571 ms, over the 400 ms limit |
| auto | complete | int8 | 3000 | one dense sweep costs an estimated 1286 ms, over the 400 ms limit |
| auto | complete | int8 | 5000 | one dense sweep costs an estimated 3571 ms, over the 400 ms limit |

### Multi-spin results

`python3 tests/capacity_bench_tables.py docs/perf/2026-11-metal-capacity.jsonl msa`

| Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | ns per update | Mean ms per sweep | Worst ms per sweep | Spread of worst | Longest chunk (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 1000 | shared | 1 | 1 | 256 | 0.133 | 0.034 | 0.034 | - | 8.707 |
| auto | 4 | int32 | 4000 | shared | 1 | 1 | 256 | 0.110 | 0.113 | 0.113 | - | 28.961 |
| auto | 4 | int32 | 8000 | spill | 1 | 1 | 256 | 0.078 | 0.159 | 0.159 | - | 40.735 |
| auto | 4 | int32 | 16000 | spill | 1 | 2 | 256 | 0.060 | 0.244 | 0.246 | - | 56.536 |
| auto | 4 | int32 | 40000 | spill | 1 | 3 | 256 | 0.058 | 0.590 | 0.594 | - | 54.348 |
| auto | 4 | int8 | 1000 | shared | 1 | 1 | 256 | 0.399 | 0.102 | 0.102 | - | 26.180 |
| auto | 4 | int8 | 4000 | shared | 1 | 1 | 256 | 0.104 | 0.107 | 0.107 | - | 27.322 |
| auto | 4 | int8 | 8000 | spill | 1 | 1 | 256 | 0.099 | 0.203 | 0.203 | - | 51.860 |
| auto | 4 | int8 | 16000 | spill | 1 | 2 | 256 | 0.060 | 0.245 | 0.246 | - | 56.824 |
| auto | 4 | int8 | 40000 | spill | 1 | 3 | 256 | 0.058 | 0.592 | 0.599 | - | 54.494 |
| auto | 20 | binary | 1000 | shared | 1 | 1 | 256 | 0.163 | 0.042 | 0.042 | - | 10.684 |
| auto | 20 | binary | 4000 | shared | 1 | 1 | 256 | 0.053 | 0.055 | 0.055 | - | 13.965 |
| auto | 20 | binary | 8000 | spill | 1 | 1 | 256 | 0.100 | 0.206 | 0.206 | - | 52.651 |
| auto | 20 | binary | 16000 | spill | 1 | 2 | 256 | 0.117 | 0.478 | 0.483 | - | 110.845 |
| auto | 20 | binary | 40000 | spill | 1 | 3 | 256 | 0.143 | 1.460 | 1.460 | - | 134.281 |
| auto | 20 | float | 1000 | shared | 1 | 2 | 256 | 0.378 | 0.097 | 0.097 | - | 19.726 |
| auto | 20 | float | 4000 | shared | 1 | 2 | 256 | 0.141 | 0.144 | 0.145 | - | 33.966 |
| auto | 20 | float | 8000 | spill | 1 | 1 | 256 | 0.155 | 0.317 | 0.317 | - | 81.107 |
| auto | 20 | float | 16000 | spill | 1 | 2 | 256 | 0.186 | 0.763 | 0.763 | - | 115.956 |
| auto | 20 | float | 40000 | spill | 1 | 5 | 256 | 0.195 | 1.995 | 2.005 | - | 119.724 |
| auto | 20 | int32 | 1000 | shared | 1 | 1 | 256 | 0.286 | 0.073 | 0.073 | - | 18.716 |
| auto | 20 | int32 | 4000 | shared | 1 | 1 | 256 | 0.110 | 0.113 | 0.113 | - | 28.938 |
| auto | 20 | int32 | 8000 | spill | 1 | 1 | 256 | 0.100 | 0.205 | 0.205 | - | 52.590 |
| auto | 20 | int32 | 16000 | spill | 1 | 2 | 256 | 0.143 | 0.584 | 0.595 | - | 138.006 |
| auto | 20 | int32 | 40000 | spill | 1 | 3 | 256 | 0.149 | 1.527 | 1.646 | - | 151.476 |
| auto | 20 | int8 | 1000 | shared | 1 | 1 | 256 | 0.346 | 0.089 | 0.089 | - | 22.684 |
| auto | 20 | int8 | 4000 | shared | 1 | 1 | 256 | 0.119 | 0.122 | 0.122 | - | 31.263 |
| auto | 20 | int8 | 8000 | spill | 1 | 1 | 256 | 0.100 | 0.206 | 0.206 | - | 52.629 |
| auto | 20 | int8 | 16000 | spill | 1 | 2 | 256 | 0.117 | 0.479 | 0.483 | - | 110.908 |
| auto | 20 | int8 | 40000 | spill | 1 | 3 | 256 | 0.151 | 1.544 | 1.684 | - | 154.899 |
| auto | 62 | int32 | 1000 | shared | 1 | 13 | 256 | 2.765 | 0.708 | 0.715 | - | 14.297 |
| auto | 62 | int32 | 4000 | shared | 3 | 11 | 256 | 0.544 | 0.557 | 0.559 | 0.558 to 0.559 | 13.399 |
| auto | 62 | int32 | 8000 | spill | 3 | 11 | 256 | 0.434 | 0.889 | 0.892 | 0.892 to 0.893 | 21.402 |
| auto | 62 | int32 | 16000 | spill | 1 | 11 | 256 | 0.411 | 1.683 | 1.686 | - | 40.436 |
| auto | 62 | int32 | 40000 | spill | 1 | 16 | 158 | 0.638 | 6.537 | 8.360 | - | 83.603 |
| auto | 62 | int32 | 100000 | spill | 3 | 4 | 15 | 0.674 | 17.249 | 20.684 | 18.168 to 20.878 | 82.735 |
| auto | 62 | int32 | 200000 | spill | 3 | 4 | 7 | 0.704 | 36.051 | 46.160 | 44.828 to 46.472 | 92.319 |
| auto | 62 | int32 | 400000 | spill | 3 | 3 | 3 | 0.709 | 72.573 | 90.195 | 88.229 to 97.118 | 90.195 |
| auto | 62 | int8 | 1000 | shared | 1 | 13 | 256 | 2.757 | 0.706 | 0.716 | - | 14.315 |
| auto | 62 | int8 | 4000 | shared | 3 | 11 | 256 | 0.544 | 0.557 | 0.559 | 0.558 to 0.560 | 13.404 |
| auto | 62 | int8 | 8000 | spill | 3 | 11 | 256 | 0.434 | 0.889 | 0.892 | 0.892 to 0.894 | 21.414 |
| auto | 62 | int8 | 16000 | spill | 1 | 11 | 256 | 0.411 | 1.682 | 1.690 | - | 40.430 |
| auto | 62 | int8 | 40000 | spill | 1 | 16 | 158 | 0.643 | 6.584 | 9.149 | - | 91.494 |
| auto | 62 | int8 | 100000 | spill | 3 | 4 | 15 | 0.702 | 17.968 | 23.057 | 20.907 to 23.291 | 92.228 |
| auto | 62 | int8 | 200000 | spill | 3 | 4 | 7 | 0.699 | 35.801 | 43.789 | 43.303 to 45.856 | 87.579 |
| auto | 62 | int8 | 400000 | spill | 3 | 3 | 3 | 0.685 | 70.171 | 83.493 | 80.571 to 91.080 | 83.493 |
| auto | 200 | int32 | 1000 | dense | 1 | 22 | 256 | 34.144 | 8.741 | 9.778 | - | 106.086 |
| auto | 200 | int32 | 4000 | dense | 3 | 256 | 256 | 219.652 | 224.923 | 297.071 | 282.929 to 300.500 | 297.071 |
| auto | 200 | int8 | 1000 | dense | 1 | 22 | 256 | 34.580 | 8.852 | 10.690 | - | 128.274 |
| auto | 200 | int8 | 4000 | dense | 3 | 256 | 256 | 166.434 | 170.428 | 193.740 | 189.434 to 195.924 | 193.740 |
| auto | complete | binary | 1000 | dense | 1 | 22 | 256 | 17.558 | 4.495 | 6.884 | - | 82.602 |
| auto | complete | binary | 2500 | dense | 1 | 63 | 63 | 44.081 | 28.212 | 29.619 | - | 29.619 |
| auto | complete | binary | 4898 | dense | 1 | 16 | 16 | 86.431 | 108.374 | 132.197 | - | 132.197 |
| auto | complete | float | 1000 | dense | 1 | 22 | 256 | 28.964 | 7.415 | 9.527 | - | 114.327 |
| auto | complete | float | 2500 | dense | 1 | 63 | 63 | 115.488 | 73.912 | 100.677 | - | 100.677 |
| auto | complete | float | 4898 | dense | 1 | 16 | 16 | 244.553 | 306.642 | 321.698 | - | 321.698 |
| auto | complete | int32 | 1000 | dense | 1 | 22 | 256 | 35.518 | 9.093 | 10.858 | - | 130.301 |
| auto | complete | int32 | 2500 | dense | 1 | 63 | 63 | 127.736 | 81.751 | 107.177 | - | 107.177 |
| auto | complete | int32 | 4898 | dense | 1 | 16 | 16 | 270.579 | 339.276 | 350.504 | - | 350.504 |
| auto | complete | int8 | 1000 | dense | 1 | 22 | 256 | 34.668 | 8.875 | 11.148 | - | 133.775 |
| auto | complete | int8 | 2500 | dense | 1 | 63 | 63 | 98.296 | 62.909 | 112.287 | - | 112.287 |
| auto | complete | int8 | 4898 | dense | 1 | 16 | 16 | 204.849 | 256.858 | 277.609 | - | 277.609 |
| dense | 62 | int32 | 1000 | dense | 1 | 22 | 256 | 33.953 | 8.692 | 9.338 | - | 105.590 |
| dense | 62 | int32 | 1414 | dense | 1 | 64 | 256 | 51.069 | 18.486 | 19.357 | - | 77.428 |
| dense | 62 | int8 | 1000 | dense | 1 | 22 | 256 | 34.281 | 8.776 | 9.058 | - | 106.466 |
| dense | 62 | int8 | 1414 | dense | 1 | 64 | 256 | 49.617 | 17.961 | 18.531 | - | 74.125 |
| dense | 200 | int32 | 1000 | dense | 1 | 22 | 256 | 34.217 | 8.760 | 9.351 | - | 106.466 |
| dense | 200 | int32 | 1414 | dense | 1 | 64 | 256 | 51.148 | 18.515 | 19.525 | - | 78.100 |
| dense | 200 | int8 | 1000 | dense | 1 | 22 | 256 | 34.187 | 8.752 | 9.334 | - | 105.876 |
| dense | 200 | int8 | 1414 | dense | 1 | 64 | 256 | 49.927 | 18.073 | 18.441 | - | 73.763 |
| sparse | 62 | int32 | 1000 | shared | 1 | 13 | 256 | 2.752 | 0.705 | 0.715 | - | 14.292 |
| sparse | 62 | int32 | 1414 | shared | 1 | 13 | 256 | 1.855 | 0.672 | 0.682 | - | 13.638 |
| sparse | 62 | int32 | 2500 | shared | 1 | 13 | 256 | 1.019 | 0.652 | 0.662 | - | 13.231 |
| sparse | 62 | int8 | 1000 | shared | 1 | 13 | 256 | 3.165 | 0.810 | 2.012 | - | 40.239 |
| sparse | 62 | int8 | 1414 | shared | 1 | 13 | 256 | 1.857 | 0.672 | 0.682 | - | 13.642 |
| sparse | 62 | int8 | 2500 | shared | 1 | 13 | 256 | 1.019 | 0.652 | 0.662 | - | 13.226 |

Refused msa runs:

| Storage | Degree | Precision | Nodes | Reason |
|---|---|---|---|---|
| auto | 200 | int32 | 8000 | one dense sweep costs an estimated 1067 ms, over the 400 ms limit |
| auto | 200 | int32 | 16000 | one dense sweep costs an estimated 4267 ms, over the 400 ms limit |
| auto | 200 | int32 | 40000 | the model needs 256,114,564,036 bytes and 54,975,581,388 are usable |
| auto | 200 | int8 | 8000 | one dense sweep costs an estimated 1067 ms, over the 400 ms limit |
| auto | 200 | int8 | 16000 | one dense sweep costs an estimated 4267 ms, over the 400 ms limit |
| auto | 200 | int8 | 40000 | the model needs 256,114,564,036 bytes and 54,975,581,388 are usable |
| auto | complete | binary | 6000 | one dense sweep costs an estimated 600 ms, over the 400 ms limit |
| auto | complete | float | 6000 | one dense sweep costs an estimated 600 ms, over the 400 ms limit |
| auto | complete | int32 | 6000 | one dense sweep costs an estimated 600 ms, over the 400 ms limit |
| auto | complete | int8 | 6000 | one dense sweep costs an estimated 600 ms, over the 400 ms limit |
| sparse | 200 | int32 | 1000 | degree 200 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | 200 | int32 | 1414 | degree 200 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | 200 | int32 | 2500 | degree 200 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | 200 | int8 | 1000 | degree 200 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | 200 | int8 | 1414 | degree 200 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | 200 | int8 | 2500 | degree 200 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | complete | int32 | 1000 | degree 999 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | complete | int32 | 1414 | degree 1413 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | complete | int32 | 2500 | degree 2499 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | complete | int8 | 1000 | degree 999 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | complete | int8 | 1414 | degree 1413 is over the 62-neighbour limit of the multi-spin kernel |
| sparse | complete | int8 | 2500 | degree 2499 is over the 62-neighbour limit of the multi-spin kernel |

## Spill and global tiers at large sizes

The `spill` series extends the auto series past 40,000 nodes, so that Task 5.3b can place a one-sweep refusal. The table below lists the worst ms per sweep of every global or spill row from 20,000 nodes up. Medians and spreads cover the three repeats.

`python3 tests/capacity_bench_tables.py docs/perf/2026-11-metal-capacity.jsonl spill`

sa:

| Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | ns per update | Mean ms per sweep | Worst ms per sweep | Spread of worst | Longest chunk (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 40000 | global | 1 | 256 | 256 | 2.819 | 28.863 | 70.816 | - | 70.816 |
| auto | 4 | int8 | 40000 | global | 1 | 256 | 256 | 3.021 | 30.931 | 67.201 | - | 67.201 |
| auto | 20 | binary | 40000 | global | 1 | 256 | 256 | 5.826 | 59.659 | 115.233 | - | 115.233 |
| auto | 20 | float | 40000 | global | 1 | 256 | 256 | 5.576 | 57.100 | 157.051 | - | 157.051 |
| auto | 20 | int32 | 40000 | global | 1 | 256 | 256 | 5.763 | 59.016 | 136.023 | - | 136.023 |
| auto | 20 | int8 | 40000 | global | 1 | 256 | 256 | 6.026 | 61.709 | 147.595 | - | 147.595 |
| auto | 62 | int32 | 20000 | global | 3 | 79 | 79 | 14.604 | 74.770 | 210.846 | 205.514 to 211.393 | 210.846 |
| auto | 62 | int32 | 40000 | global | 1 | 158 | 158 | 14.623 | 149.740 | 416.839 | - | 416.839 |
| auto | 62 | int8 | 20000 | global | 3 | 79 | 79 | 15.050 | 77.057 | 182.237 | 158.815 to 185.982 | 182.237 |
| auto | 62 | int8 | 40000 | global | 1 | 158 | 158 | 15.000 | 153.600 | 355.651 | - | 355.651 |
| auto | 62 | int8 | 80000 | global | 3 | 19 | 19 | 17.690 | 362.287 | 674.195 | 654.089 to 689.267 | 674.195 |
| auto | 200 | int32 | 40000 | global | 1 | 49 | 49 | 57.274 | 586.483 | 1373.092 | - | 1373.092 |
| auto | 200 | int8 | 40000 | global | 1 | 49 | 49 | 59.919 | 613.566 | 1264.947 | - | 1264.947 |
gibbs:

| Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | ns per update | Mean ms per sweep | Worst ms per sweep | Spread of worst | Longest chunk (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 40000 | spill | 1 | 64 | 256 | 0.067 | 0.689 | 1.062 | - | 4.246 |
| auto | 4 | int8 | 40000 | spill | 1 | 64 | 256 | 0.065 | 0.669 | 0.903 | - | 3.611 |
| auto | 20 | binary | 40000 | spill | 1 | 22 | 256 | 0.625 | 6.402 | 7.110 | - | 85.325 |
| auto | 20 | float | 40000 | spill | 1 | 32 | 256 | 0.585 | 5.995 | 7.519 | - | 60.148 |
| auto | 20 | int32 | 40000 | spill | 1 | 64 | 256 | 0.553 | 5.659 | 9.774 | - | 39.097 |
| auto | 20 | int8 | 40000 | spill | 1 | 64 | 256 | 0.481 | 4.921 | 8.297 | - | 33.187 |
| auto | 62 | int32 | 40000 | spill | 1 | 27 | 158 | 2.023 | 20.712 | 23.756 | - | 142.535 |
| auto | 62 | int32 | 100000 | spill | 3 | 8 | 15 | 1.556 | 39.833 | 47.620 | 45.364 to 52.485 | 94.446 |
| auto | 62 | int32 | 200000 | spill | 3 | 7 | 7 | 1.484 | 75.958 | 97.798 | 93.431 to 99.086 | 97.798 |
| auto | 62 | int32 | 400000 | spill | 3 | 3 | 3 | 1.547 | 158.399 | 175.915 | 173.600 to 197.128 | 175.915 |
| auto | 62 | int8 | 40000 | spill | 1 | 27 | 158 | 1.905 | 19.510 | 23.131 | - | 138.787 |
| auto | 62 | int8 | 100000 | spill | 3 | 8 | 15 | 1.451 | 37.143 | 47.889 | 43.697 to 62.818 | 95.777 |
| auto | 62 | int8 | 200000 | spill | 3 | 7 | 7 | 1.399 | 71.613 | 89.347 | 89.164 to 96.102 | 89.347 |
| auto | 62 | int8 | 400000 | spill | 3 | 3 | 3 | 1.430 | 146.449 | 164.094 | 156.265 to 166.905 | 164.094 |
| auto | 200 | int32 | 40000 | spill | 1 | 9 | 49 | 8.708 | 89.169 | 107.640 | - | 645.843 |
| auto | 200 | int32 | 100000 | spill | 3 | 2 | 4 | 5.438 | 139.208 | 144.779 | 144.415 to 145.954 | 289.557 |
| auto | 200 | int32 | 200000 | spill | 3 | 2 | 2 | 7.202 | 368.725 | 417.192 | 406.755 to 626.225 | 417.192 |
| auto | 200 | int32 | 400000 | spill | 3 | 1 | 1 | 10.542 | 1079.465 | 1079.465 | 1075.332 to 1289.163 | 1079.465 |
| auto | 200 | int8 | 40000 | spill | 1 | 9 | 49 | 7.132 | 73.030 | 79.946 | - | 479.677 |
| auto | 200 | int8 | 100000 | spill | 3 | 2 | 4 | 5.155 | 131.974 | 139.194 | 138.440 to 220.338 | 278.388 |
| auto | 200 | int8 | 200000 | spill | 3 | 2 | 2 | 6.657 | 340.856 | 375.755 | 294.106 to 376.569 | 375.755 |
| auto | 200 | int8 | 400000 | spill | 3 | 1 | 1 | 7.022 | 719.047 | 719.047 | 570.369 to 844.717 | 719.047 |
msa:

| Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | ns per update | Mean ms per sweep | Worst ms per sweep | Spread of worst | Longest chunk (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 40000 | spill | 1 | 3 | 256 | 0.058 | 0.590 | 0.594 | - | 54.348 |
| auto | 4 | int8 | 40000 | spill | 1 | 3 | 256 | 0.058 | 0.592 | 0.599 | - | 54.494 |
| auto | 20 | binary | 40000 | spill | 1 | 3 | 256 | 0.143 | 1.460 | 1.460 | - | 134.281 |
| auto | 20 | float | 40000 | spill | 1 | 5 | 256 | 0.195 | 1.995 | 2.005 | - | 119.724 |
| auto | 20 | int32 | 40000 | spill | 1 | 3 | 256 | 0.149 | 1.527 | 1.646 | - | 151.476 |
| auto | 20 | int8 | 40000 | spill | 1 | 3 | 256 | 0.151 | 1.544 | 1.684 | - | 154.899 |
| auto | 62 | int32 | 40000 | spill | 1 | 16 | 158 | 0.638 | 6.537 | 8.360 | - | 83.603 |
| auto | 62 | int32 | 100000 | spill | 3 | 4 | 15 | 0.674 | 17.249 | 20.684 | 18.168 to 20.878 | 82.735 |
| auto | 62 | int32 | 200000 | spill | 3 | 4 | 7 | 0.704 | 36.051 | 46.160 | 44.828 to 46.472 | 92.319 |
| auto | 62 | int32 | 400000 | spill | 3 | 3 | 3 | 0.709 | 72.573 | 90.195 | 88.229 to 97.118 | 90.195 |
| auto | 62 | int8 | 40000 | spill | 1 | 16 | 158 | 0.643 | 6.584 | 9.149 | - | 91.494 |
| auto | 62 | int8 | 100000 | spill | 3 | 4 | 15 | 0.702 | 17.968 | 23.057 | 20.907 to 23.291 | 92.228 |
| auto | 62 | int8 | 200000 | spill | 3 | 4 | 7 | 0.699 | 35.801 | 43.789 | 43.303 to 45.856 | 87.579 |
| auto | 62 | int8 | 400000 | spill | 3 | 3 | 3 | 0.685 | 70.171 | 83.493 | 80.571 to 91.080 | 83.493 |

What the series shows:

- SA on the global build reaches a worst sweep of 355 to 417 ms at 40,000 nodes and degree 62. At 80,000 nodes the worst sweep is 674 ms (`int8`). The 400 ms point sits between 40,000 and 80,000 nodes at degree 62. At degree 200 and 40,000 nodes, one sweep takes 1.26 to 1.37 seconds in the worst chunk.
- The admission refuses SA with `int32` at 80,000 nodes and degree 62 for memory. The run needs 61.1 billion bytes and 55.0 billion are usable.
- Gibbs on the spill build at degree 200 reaches 376 to 417 ms per sweep at 200,000 nodes and 719 to 1,079 ms at 400,000 nodes, with one sweep in the only chunk. At degree 62 the worst sweep at 400,000 nodes is 164 to 176 ms, so the 400 ms point lies past 400,000 nodes. Linear growth with the node count would put it near 900,000 nodes, but that is an extrapolation and the grid did not measure it.
- MSA on the spill build at degree 62 reaches 83 to 90 ms per sweep at 400,000 nodes. The grid did not reach 400 ms on MSA.
- The three repeats of one point differ by up to a factor of 1.5 on the Gibbs rows (570 to 845 ms at degree 200, 400,000 nodes and `int8`). Contention from other sessions may explain part of that spread.

## Dense compared with sparse

For models that both storages hold, the table gives the time per spin update on the tier that auto storage chose. It then gives the time on forced sparse storage, and on forced dense storage. A dense update scans a whole matrix row, so its cost grows with the node count. At degrees 62 and 200 the dense build is slower than the sparse build in every row where the sparse build runs. For MSA, the sparse build refuses degrees over 62.

The routing of auto storage depends on precision. At degree 200 and 1,000 nodes, auto storage chose dense for `int8` on SA (206.5 ns, against 42.7 ns on forced sparse) and on Gibbs (23.8 ns against 3.2 ns). For `int32` at the same shape, auto storage chose the local or shared sparse build (34.1 ns on SA, 4.5 ns on Gibbs).

`python3 tests/capacity_bench_tables.py docs/perf/2026-11-metal-capacity.jsonl compare`

| Kernel | Degree | Nodes | Precision | Auto tier | Auto ns | Sparse ns | Dense ns |
|---|---|---|---|---|---|---|---|
| sa | 62 | 1000 | int8 | local | 10.486 | 10.877 | 202.937 |
| sa | 62 | 1000 | int32 | local | 8.167 | 8.181 | 90.095 |
| sa | 62 | 1414 | int8 | n/a | n/a | 10.470 | 287.416 |
| sa | 62 | 1414 | int32 | n/a | n/a | 8.237 | 133.380 |
| sa | 200 | 1000 | int8 | dense | 206.522 | 42.670 | 206.210 |
| sa | 200 | 1000 | int32 | local | 34.119 | 34.115 | 95.617 |
| sa | 200 | 1414 | int8 | n/a | n/a | 43.351 | 287.941 |
| sa | 200 | 1414 | int32 | n/a | n/a | 34.774 | 136.144 |
| gibbs | 62 | 1000 | int8 | shared | 0.766 | 0.730 | 5.027 |
| gibbs | 62 | 1000 | int32 | shared | 0.516 | 0.388 | 4.000 |
| gibbs | 62 | 1414 | int8 | n/a | n/a | 0.350 | 8.374 |
| gibbs | 62 | 1414 | int32 | n/a | n/a | 0.346 | 6.372 |
| gibbs | 200 | 1000 | int8 | dense | 23.822 | 3.153 | 29.962 |
| gibbs | 200 | 1000 | int32 | shared | 4.458 | 3.379 | 24.063 |
| gibbs | 200 | 1414 | int8 | n/a | n/a | 1.505 | 17.495 |
| gibbs | 200 | 1414 | int32 | n/a | n/a | 1.699 | 16.148 |
| msa | 62 | 1000 | int8 | shared | 2.757 | 3.165 | 34.281 |
| msa | 62 | 1000 | int32 | shared | 2.765 | 2.752 | 33.953 |
| msa | 62 | 1414 | int8 | n/a | n/a | 1.857 | 49.617 |
| msa | 62 | 1414 | int32 | n/a | n/a | 1.855 | 51.069 |
| msa | 200 | 1000 | int8 | dense | 34.580 | n/a | 34.187 |
| msa | 200 | 1000 | int32 | dense | 34.144 | n/a | 34.217 |
| msa | 200 | 1414 | int8 | n/a | n/a | n/a | 49.927 |
| msa | 200 | 1414 | int32 | n/a | n/a | n/a | 51.148 |

## Runs whose longest chunk passed 400 ms

The table lists every point with a run whose longest chunk passed 400 ms, sorted by the largest longest chunk. The target of the planner is 400 ms, so these runs show where the planner plans a chunk past it. The runs fall into three groups, none with a measured cause. A planner rate that does not depend on degree would explain the first two groups.

- Auto storage at degree 200 on SA and Gibbs plans chunks of 460 to 1,400 ms, on both the local or shared and the global or spill tiers.
- Forced sparse storage on complete models plans chunks of 2.0 to 10.2 seconds on SA and Gibbs. Forced sparse storage at degree 200 on SA plans chunks of 590 to 700 ms. Auto storage takes the dense build for complete models, and the dense chunks stay under 400 ms in this grid.
- The largest spill and global models have one sweep or few sweeps per chunk, so one sweep alone passes 400 ms. Examples are SA at 80,000 nodes (674 ms), SA at 40,000 nodes and degree 62 (417 ms for `int32`), and Gibbs at 400,000 nodes and degree 200 (719 ms or more).

`python3 tests/capacity_bench_tables.py docs/perf/2026-11-metal-capacity.jsonl over400`

| Kernel | Storage | Degree | Precision | Nodes | Tier | Runs | Chunks | Sweeps | Median longest chunk (ms) | Largest longest chunk (ms) | Median worst ms per sweep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sa | sparse | complete | int8 | 2500 | local | 1 | 2 | 16 | 10157.853 | 10157.853 | 677.190 |
| sa | sparse | complete | int32 | 2500 | local | 1 | 2 | 16 | 9577.462 | 9577.462 | 638.497 |
| sa | sparse | complete | int8 | 1414 | local | 1 | 2 | 49 | 6631.686 | 6631.686 | 245.618 |
| sa | sparse | complete | int32 | 1414 | local | 1 | 2 | 49 | 6344.125 | 6344.125 | 234.968 |
| sa | sparse | complete | int32 | 1000 | local | 1 | 3 | 99 | 4766.183 | 4766.183 | 122.210 |
| sa | sparse | complete | int8 | 1000 | local | 1 | 3 | 99 | 4742.837 | 4742.837 | 121.611 |
| gibbs | sparse | complete | int8 | 2500 | shared | 1 | 1 | 16 | 3499.423 | 3499.423 | 218.714 |
| gibbs | sparse | complete | int32 | 1414 | shared | 1 | 1 | 49 | 2940.457 | 2940.457 | 60.009 |
| gibbs | sparse | complete | int8 | 1414 | shared | 1 | 1 | 49 | 2894.765 | 2894.765 | 59.077 |
| gibbs | sparse | complete | int32 | 2500 | shared | 1 | 1 | 16 | 2636.306 | 2636.306 | 164.769 |
| gibbs | sparse | complete | int8 | 1000 | shared | 1 | 1 | 99 | 2631.430 | 2631.430 | 26.580 |
| gibbs | sparse | complete | int32 | 1000 | shared | 1 | 1 | 99 | 2027.921 | 2027.921 | 20.484 |
| sa | auto | 200 | int32 | 40000 | global | 1 | 49 | 49 | 1373.092 | 1373.092 | 1373.092 |
| gibbs | auto | 200 | int32 | 400000 | spill | 3 | 1 | 1 | 1079.465 | 1289.163 | 1079.465 |
| sa | auto | 200 | int8 | 40000 | global | 1 | 49 | 49 | 1264.947 | 1264.947 | 1264.947 |
| gibbs | auto | 200 | int8 | 400000 | spill | 3 | 1 | 1 | 719.047 | 844.717 | 719.047 |
| sa | sparse | 200 | int32 | 2500 | local | 1 | 14 | 198 | 699.274 | 699.274 | 46.618 |
| sa | auto | 200 | int8 | 4500 | local | 3 | 32 | 256 | 696.377 | 697.740 | 87.047 |
| sa | auto | 62 | int8 | 80000 | global | 3 | 19 | 19 | 674.195 | 689.267 | 674.195 |
| sa | sparse | 200 | int8 | 2500 | local | 1 | 14 | 198 | 679.288 | 679.288 | 45.286 |
| sa | auto | 200 | int32 | 4500 | local | 3 | 32 | 256 | 669.837 | 672.442 | 83.730 |
| sa | auto | 200 | int32 | 4700 | global | 3 | 64 | 256 | 550.152 | 655.222 | 137.538 |
| gibbs | auto | 200 | int32 | 40000 | spill | 1 | 9 | 49 | 645.843 | 645.843 | 107.640 |
| sa | sparse | 200 | int8 | 1000 | local | 1 | 8 | 256 | 643.109 | 643.109 | 17.864 |
| gibbs | auto | 200 | int32 | 200000 | spill | 3 | 2 | 2 | 417.192 | 626.225 | 417.192 |
| sa | auto | 200 | int32 | 1000 | local | 1 | 8 | 256 | 612.339 | 612.339 | 17.009 |
| sa | sparse | 200 | int32 | 1000 | local | 1 | 8 | 256 | 612.256 | 612.256 | 17.007 |
| sa | sparse | 200 | int8 | 1414 | local | 1 | 11 | 256 | 607.539 | 607.539 | 25.314 |
| sa | auto | 200 | int8 | 2000 | local | 1 | 16 | 256 | 599.248 | 599.248 | 37.453 |
| sa | sparse | 200 | int32 | 1414 | local | 1 | 11 | 256 | 586.495 | 586.495 | 24.437 |
| sa | auto | 200 | int32 | 2000 | local | 1 | 16 | 256 | 580.097 | 580.097 | 36.256 |
| sa | auto | 200 | int8 | 4700 | global | 3 | 64 | 256 | 483.287 | 577.172 | 120.822 |
| gibbs | auto | 200 | int8 | 10000 | spill | 1 | 8 | 198 | 543.016 | 543.016 | 20.112 |
| gibbs | auto | 200 | int8 | 40000 | spill | 1 | 9 | 49 | 479.677 | 479.677 | 79.946 |
| gibbs | auto | 200 | int8 | 2000 | shared | 1 | 1 | 256 | 448.628 | 448.628 | 1.752 |
| gibbs | auto | 200 | int8 | 4700 | shared | 3 | 2 | 256 | 373.206 | 444.365 | 2.796 |
| gibbs | auto | 200 | int8 | 100000 | spill | 3 | 2 | 4 | 278.388 | 440.676 | 139.194 |
| sa | auto | 62 | int32 | 40000 | global | 1 | 158 | 158 | 416.839 | 416.839 | 416.839 |

## Calibration of the planner and the one-sweep refusal

Task 5.3b uses these measurements in two places in `src/sampler.rs`. The SA and Gibbs chunk planners multiply their cost per spin update by a degree factor. The admission refuses any batch whose estimated one-sweep chunk passes 400 ms, on every tier. The doc comment of each constant names its source rows.

### Degree factor of the planner

The planner rates come from runs near degree 20, so the factor is 1 up to degree 20. Past degree 20, `sweep_degree_factor` interpolates `SA_DEGREE_FACTOR` and `GIBBS_DEGREE_FACTOR` on the mean degree, `2 * edges / nodes`. Each point is the largest ratio of the mean ns per update at a degree to the mean at degree 20, at the same nodes, precision, and tier.

| Mean degree | SA factor | Gibbs factor |
|---|---|---|
| 20 or less | 1 | 1 |
| 62 | 3.0 | 4.3 |
| 200 | 10.6 | 29.7 |
| 999 | 76 | 570 |
| 1,413 | 116 | 1,100 |
| 2,499 | 290 | 2,500 |

Past the last point, the factor follows the last segment. The Gibbs factor grows faster because a small model of high degree has few nodes in each color class, so the class steps bound its sweep. The multi-spin planner keeps its row chunks and color steps. The grid shows no planned multi-spin chunk past 160 ms, so Task 5.3b does not change it. The Advantage2 topology has a mean degree of 18.1, so its plans do not change.

### One-sweep estimate

`one_sweep_ms` covers every build. A dense-matrix sweep walks `n * n` columns at the kernel's column rate: 5.0 million per second for SA, 7.0 million for Gibbs and 60 million for MSA. These are the rates of the earlier dense checks, so the dense limits stay at 1,414, 1,673 and 4,898 nodes. A CSR sweep costs one problem of 256 reads at the worst measured ns per update of its build and mean degree. The table points are in `SA_LOCAL_SWEEP_NS`, `SA_GLOBAL_SWEEP_NS`, `GIBBS_SHARED_SWEEP_NS`, `GIBBS_SPILL_SWEEP_NS`, `MSA_SHARED_SWEEP_NS` and `MSA_SPILL_SWEEP_NS`. The estimate multiplies that cost by 1.15 for contention from other GPU work.

The grid ran most rows in chunks of 4 to 32 sweeps, and that hides two costs of a one-sweep chunk:

- The first chunk of an anneal initializes the state. On the SA local build, a one-sweep first chunk took 1.26 to 1.30 times the next chunk. The SA local points are the grid values times 1.3.
- The late sweeps of a Gibbs spill anneal run slower than the first ones, up to 2.2 times, and the cost per update grows with the node count. The grid ran 1 to 7 sweeps at large sizes. The Gibbs spill points at degrees 4, 20 and 62 come from runs of 16 one-sweep chunks instead.

The GPU runs a fixed number of reads side by side: one problem of 256 reads per core for SA, 6.4 reads per core for Gibbs, and one 32-replica word per core for MSA. Past that count, the estimate grows with the reads of the batch. Past it, a batch of two or more problems costs 2 times more on Gibbs and MSA. Every problem has a copy of its couplings, and the copies compete for the cache. A batch of fewer than 256 reads costs as much as 256 reads. One SA problem of 64 reads took about as long as one of 256.

The stream keeps two batches on the GPU at once, so its estimate counts the reads of both. Every single-batch path counts one batch, and the limits in the next table are for one batch. The stream also caps the problems of each batch at the largest count whose estimate fits 400 ms (`one_sweep_problem_cap`). The rest of the queue waits for the next batches. Here a job passes alone when one problem of it passes with one batch on the GPU, the rule that `sample_ising` applies. A job can pass alone but fail beside a second batch in flight. One such job is a Gibbs job of 256 reads at 4,800 nodes and a mean degree from about 733 to 1,966. The stream then runs that job as the only batch, after the batch in flight finishes. Two batches go in flight again after it. The stream refuses a job only when the job fails alone.

| Kernel | Advantage2 stream batch at 64 reads | At 256 reads |
|---|---|---|
| SA | 240 | 240 |
| Gibbs | 10 | 3 |
| MSA | 20 | 5 |

The cap leaves these batch sizes as they were before it.

The multi-problem and first-chunk runs come from the ignored test `batch_sweep_probe` in `src/sampler.rs`. Every chunk of that test is one sweep. Its records are in `docs/perf/2026-11-metal-sweep-probe.jsonl`, and the `run` field names the batch of runs. The command for one run is:

```sh
QUIP_PROBE_KERNEL=gibbs QUIP_PROBE_NODES=358494 QUIP_PROBE_DEGREE=62 \
  QUIP_PROBE_PROBLEMS=1 QUIP_PROBE_READS=256 QUIP_PROBE_SWEEPS=16 \
  cargo test --release --lib batch_sweep_probe -- --ignored --nocapture
```

The Gibbs tables come from the chromatic kernel. The limits are not calibrated for the sequential Gibbs kernel that the operator override `QUIP_METAL_GIBBS_SEQUENTIAL=1` selects, and that override also raises the Gibbs stream batch to 640 problems.

The probes ran on the same M4 Max as the grid, with other sessions on the machine. On small models, the first chunk of a run on an idle GPU took up to 38 ms more than later chunks, a cost that the estimate does not model and that matters only far below 400 ms.

### One-sweep limits

The table gives the largest node count that the admission accepts for one problem of 256 reads on 40 GPU cores. The test `each_build_refuses_one_node_past_its_one_sweep_limit` checks every row. "Measured" means that the grid or a probe has a point at or past the limit. "Extrapolated" means that the limit lies past the largest measured size, so one run at the limit verified it. The longest chunk column gives that run, with 16 sweeps of one sweep each.

| Build | Degree | Limit (nodes) | Basis | Longest chunk at the limit (ms) |
|---|---|---|---|---|
| SA local | complete | 1,476 | measured (1,414 and 2,000 nodes) | 347.6 (`int8`), 346.9 (`int32`) |
| SA global | 4 | 167,121 | extrapolated | 382.9, 301.0 |
| SA global | 20 | 88,572 | extrapolated | 293.4 |
| SA global | 62 | 32,906 | measured (20,000 and 40,000 nodes) | 362.7, 290.7 |
| SA global | 200 | 9,322 | measured (4,700 and 10,000 nodes) | 376.2 (`int8`), 340.9 (`int32`) |
| Gibbs shared | complete | 2,915 | measured (2,000 to 3,098 nodes) | 293.0 and 287.4 (`binary`), 251.5 and 261.8 (`int32`) |
| Gibbs spill | 4 | 5,684,919 | extrapolated | 365.4, 248.9 |
| Gibbs spill | 20 | 978,887 | extrapolated | 288.7, 301.3 |
| Gibbs spill | 62 | 358,494 | measured (400,000 nodes) | 288.3, 249.3, 346.0, 317.9, 305.1 (`int32`) |
| Gibbs spill | 200 | 107,927 | measured (100,000 and 200,000 nodes) | 286.3, 258.9 and 386.3 (`binary`), 293.9, 306.6, 277.4 and 331.0 (`int32`) |
| MSA spill | 4 | 13,724,198 | extrapolated | 357.0, 340.5 |
| MSA spill | 20 | 6,932,120 | extrapolated | 343.7 |
| MSA spill | 62 | 1,433,223 | extrapolated | 277.9 (`int32`) |
| SA dense | complete | 1,414 | measured | 344.7 (grid rerun) |
| Gibbs dense | complete | 1,673 | measured | 263.7 (grid rerun) |
| MSA dense | complete | 4,898 | measured | 358.1 (grid rerun) |

The SA local, Gibbs shared and multi-spin shared builds reach no limit at a regular degree up to 200, or up to 62 for the multi-spin kernel, below their node caps of 4,593, 4,800 and 6,016.

Five verification runs passed 400 ms. Their constants rose, and the runs repeated at the new limits:

- Gibbs spill at degree 62 and 553,665 nodes took 537.3 ms in its seventh chunk. The point rose from 2.454 to 3.79 ns.
- Gibbs spill at degree 20 and 1,170,280 nodes took 415.8 ms, and at degree 4 and 8,335,556 nodes it took 509.6 ms in the second of two runs. The points rose to 1.388 and 0.239 ns.
- SA global at degree 4 and 196,343 nodes took 408.4 ms in its first chunk. The point rose from 6.92 to 8.13 ns.
- Gibbs shared on a complete `binary` model of 3,098 nodes took 785.8 ms, with 11 chunks past 600 ms. The complete points from 2,000 nodes up now come from runs of 16 one-sweep chunks at 2,000, 2,500, 2,700 and 2,900 nodes, and the point at degree 3,097 is 991 ns. These runs are in `docs/perf/2026-11-metal-capacity-5.3b.jsonl`, after the grid rerun. In the same file, the `int32` run at 3,098 nodes reached 534.7 ms, and the tightened limit of 2,915 nodes now refuses that model.

The device tests `the_largest_admitted_high_degree_sweeps_stay_under_400_ms` in `tests/sa_capacity.rs` and `tests/gibbs_capacity.rs`, and `the_largest_admitted_spill_sweep_stays_under_400_ms` in `tests/msa_spill.rs`, run the high-degree limits with the production planner. They assert that the longest chunk is below 400 ms and that the admission refuses one node more. These tests are valid on a quiet GPU only:

```sh
cargo test --release --test sa_capacity the_largest_admitted -- --ignored --nocapture
cargo test --release --test gibbs_capacity the_largest_admitted -- --ignored --nocapture
cargo test --release --test msa_spill the_largest_admitted -- --ignored --nocapture
```

The planner plans each chunk against the 500 ms ceiling (`TARGET_DISPATCH_MS`), and a measured chunk on a quiet GPU must stay below 400 ms. Heavy GPU load from other processes made chunks longer than both values.

During Task 5.3b fix round 2, other sessions loaded the host. Six runs of the Gibbs limit test took 405 to 708 ms in their longest chunk. The load average is the one that `uptime` printed before the run.

| Shape | Longest chunk (ms) | Load average (1, 5 and 15 minutes) |
|---|---|---|
| Gibbs spill, degree 200, 107,927 nodes, `int32` | 508.6 | 16.24, 10.80, 11.50 |
| Gibbs spill, degree 200, 107,927 nodes, `int32` | 405.4 | 5.79, 9.35, 10.88 |
| Gibbs spill, degree 200, 107,927 nodes, `int32` | 408.2 | 7.51, 8.70, 10.40 |
| Gibbs spill, degree 200, 107,927 nodes, `int32` | 497.8 | 13.59, 13.53, 12.03 |
| Gibbs spill, degree 62, 358,494 nodes, `int32` | 470.2 | 8.47, 8.91, 10.08 |
| Gibbs spill, degree 62, 358,494 nodes, `int32` | 707.5 | 19.66, 12.10, 11.12 |

The same test at the earlier commit `84d6715` failed in the same way under that load, with 454.7 and 406.4 ms. The constants stay as they are.

The 500 ms ceiling does not cover these chunks. The margin to the macOS GPU watchdog prevents a fault at these times. The `TARGET_DISPATCH_MS` doc comment in `src/sampler.rs` says that the watchdog aborts a command buffer that runs more than a few seconds, and that MLX measured watchdog kills at about 1.2 s per operation with a display attached. The longest chunk here, 707.5 ms, is below 1.2 s.

### Grid rerun

The auto, complete, sparse and spill series ran again with the new planner and refusal. The records are in `docs/perf/2026-11-metal-capacity-5.3b.jsonl`, from the crate at `ef26712` with the Task 5.3b changes in the working tree. The command was:

```sh
tests/capacity_bench_grid.sh docs/perf/2026-11-metal-capacity-5.3b.jsonl auto complete sparse spill
```

No admitted run has a chunk past 400 ms. The longest chunk is 358.1 ms, one sweep of the dense MSA build at 4,898 nodes. The longest chunk that holds more than one sweep is 148.1 ms. The table compares the longest chunk of the high-degree rows that passed 400 ms before.

| Kernel | Storage | Degree | Precision | Nodes | Tier | Before (ms, chunks) | After (ms, chunks) |
|---|---|---|---|---|---|---|---|
| sa | sparse | complete | int8 | 2,500 | local | 10,157.9 (2) | refused |
| sa | sparse | complete | int8 | 1,414 | local | 6,631.7 (2) | 340.5 (49) |
| sa | sparse | complete | int32 | 1,000 | local | 4,766.2 (3) | 165.1 (99) |
| gibbs | sparse | complete | int8 | 2,500 | shared | 3,499.4 (1) | 208.6 (16) |
| gibbs | sparse | complete | int32 | 1,000 | shared | 2,027.9 (1) | 48.0 (99) |
| sa | auto | 200 | int32 | 40,000 | global | 1,373.1 (49) | refused |
| gibbs | auto | 200 | int32 | 400,000 | spill | 1,289.2 (1) | refused |
| sa | auto | 200 | int8 | 4,500 | local | 697.7 (32) | 131.0 (256) |
| sa | auto | 62 | int8 | 80,000 | global | 689.3 (19) | refused |
| sa | auto | 200 | int32 | 4,700 | global | 655.2 (64) | 175.9 (256) |
| gibbs | auto | 200 | int32 | 40,000 | spill | 645.8 (9) | 84.3 (49) |
| sa | sparse | 200 | int8 | 1,000 | local | 643.1 (8) | 59.3 (128) |
| gibbs | auto | 200 | int32 | 200,000 | spill | 626.2 (2) | refused |
| gibbs | auto | 200 | int8 | 10,000 | spill | 543.0 (8) | 14.1 (198) |
| gibbs | auto | 200 | int8 | 2,000 | shared | 448.6 (1) | 18.7 (22) |
| sa | auto | 62 | int32 | 40,000 | global | 416.8 (158) | refused |

A chunk count of 1 means that the old plan ran the whole anneal in one command buffer. The refusal also takes three rows that ran below 400 ms before, which are SA global at degree 200 and 10,000 nodes (373.1 ms before, estimated at 429.1 ms), SA global at degree 62 and 40,000 nodes `int8` (355.7 ms), and Gibbs spill at degree 62 and 400,000 nodes (197.1 ms over three sweeps, estimated at 446.3 ms).

## Limits of these measurements

- A point ran once, except the boundary rows and the spill rows, which ran three times. The load of the shared machine adds noise, which is larger for runs of a few milliseconds.
- A short run, such as Gibbs at degree 4, has a GPU time of a few milliseconds, so its time per update (0.03 ns) has a large relative error.
- The number of sweeps changes between grid points. Large models run 1 to 49 sweeps, so their chunks have few sweeps and may include start-up cost.
- The circulant graph has a regular neighbour pattern with good memory locality. A graph with random neighbours may cost more per update on the global and spill tiers.
- The harness measures `Tuning::default` planning, which is the production planner on this host.
- MSA ns per update counts one replica of a word as one read, so the value is far smaller than for SA and Gibbs.

## The sweep change of November 12

The grid predates QUI-2265 (merge request !33 on main) and was not run again. Since that change, every SA and MSA sweep of an anneal except the last leaves one node out. The change removes at most one spin update per sweep, so the SA and MSA times per update and per sweep in this file still bound the cost of a sweep. The Gibbs kernels stay the same.
