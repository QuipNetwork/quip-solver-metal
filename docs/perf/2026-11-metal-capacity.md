# Metal capacity measurements

These measurements record what one sweep costs on the Metal SA, Gibbs and MSA kernels, by model size, degree, precision, and storage tier. The next task uses them to calibrate the chunk planner by degree and to set a one-sweep refusal for each build. The raw records are in `docs/perf/2026-11-metal-capacity.jsonl`. The probe output is in `docs/perf/2026-11-metal-capacity-probe.jsonl`.

## Host

| Item | Value |
|---|---|
| Chip | Apple M4 Max, 16 CPU cores, 40 GPU cores |
| Memory | 128 GiB |
| macOS | 26.7.1 (Darwin 25.6.0) |
| Load average before the grid | 4.74, 5.41, 5.17 (1, 5 and 15 minutes) |
| Load average after the grid | 3.28, 3.48, 3.81 |
| Crate commit | `3c388ed` (the harness was not yet committed, and `src_dirty` is false in every record) |

Other sessions shared the machine during the grid, so the load average was 3 to 5 throughout. Contention from those sessions may add to the GPU times. The grid took about 19 minutes (1,134 seconds of task time, with 1,113 seconds of harness wall time).

## What a record means

One run anneals one frustrated model with random-sign couplings. A regular model is a circulant graph. Node `i` links to the next `degree / 2` nodes, so each node has `degree` neighbours. A complete model couples every pair. Precision sets the coefficient values: `binary` uses couplings of plus or minus 1, `int8` uses multiples of 0.5, `int32` uses 1.3, 0.127 and 2.0, and `float` uses values off the milli grid.

- A spin update is one visit of one spin of one read. For MSA, a read is one of the 32 replicas of a word.
- `ns per update` is the GPU time of the whole run divided by `reads dispatched * nodes * sweeps`.
- Every run asks for 256 reads, so the dispatch runs 256 reads.
- The number of sweeps comes from a budget of 4e8 node-degree products, with a floor of 16 sweeps and a cap of 256. Forced sparse runs use a budget of 1e8. The sweeps per beta rung are `max(1, sweeps / 64)`.
- `Longest chunk` is the GPU time of the longest command buffer. `Its sweeps` is the number of sweeps in the chunk with the most sweeps, which is the longest chunk in practice. The time of one sweep is the chunk time divided by those sweeps.
- A run that the admission refuses has `status: refused` and no timing.

The top of `tests/capacity_bench.rs` documents the format of a record.

## Commands

The harness is one ignored test that the `QUIP_BENCH_*` variables drive. The grid script built the release test binary once and ran it for each grid point:

```sh
export CARGO_TARGET_DIR=/Users/carback1/Code/quip/quip-miner-metal/target
cargo test --release --test capacity_bench --no-run

QUIP_BENCH_KERNEL=sa QUIP_BENCH_NODES=2000 QUIP_BENCH_DEGREE=20 \
  QUIP_BENCH_PRECISION=int8 QUIP_BENCH_STORAGE=auto QUIP_BENCH_READS=256 \
  QUIP_BENCH_SWEEPS=256 QUIP_BENCH_OUT=docs/perf/2026-11-metal-capacity.jsonl \
  target/release/deps/capacity_bench-<hash> capacity_run --ignored --nocapture
```

The grid covered these points for each kernel. Each point ran with a timeout of 180 seconds.

| Series | Nodes (SA) | Nodes (Gibbs) | Nodes (MSA) |
|---|---|---|---|
| Degrees 4, 20, 62 and 200, auto storage, `int8` and `int32` (`binary` and `float` as well at degree 20) | 1000, 2000, 4500, 4700, 10000, 40000 | 1000, 2000, 4700, 4900, 10000, 40000 | 1000, 4000, 8000, 16000, 40000 |
| Complete, auto storage, all four precisions | 500, 1000, 1414, 2500, 5000 | 500, 1000, 1673, 3000, 5000 | 1000, 2500, 4898, 6000 |
| Degrees 62, 200 and complete, forced sparse, `int8` and `int32` | 1000, 1414, 2500 | 1000, 1414, 2500 | 1000, 1414, 2500 |
| Degrees 62 and 200, forced dense, `int8` and `int32` | 1000, 1414 | 1000, 1414 | 1000, 1414 |

The sizes on each side of a tier boundary are 4,500 and 4,700 for SA (local build up to 4,593 nodes), and 4,700 and 4,900 for Gibbs (4,800-spin local prefix). The MSA boundary depends on the model, and the MSA series moves from `shared` to `spill` between 4,000 and 8,000 nodes. The admission refuses MSA models of 40,000 nodes at degree 200 for memory, not for time.

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

In every table below, the storage column is the storage that the run asked for, and the tier column is the tier that ran.

### Simulated annealing results

Each row is one run. The time of one sweep in the longest chunk is the chunk time divided by its sweeps.

| Storage | Degree | Precision | Nodes | Tier | Chunks | Sweeps | ns per update | Longest chunk (ms) | Its sweeps |
|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 1000 | local | 8 | 256 | 2.0 | 28.1 | 36 |
| auto | 4 | int32 | 2000 | local | 16 | 256 | 2.0 | 26.1 | 16 |
| auto | 4 | int32 | 4500 | local | 32 | 256 | 2.1 | 31.4 | 8 |
| auto | 4 | int32 | 4700 | global | 64 | 256 | 2.1 | 17.5 | 4 |
| auto | 4 | int32 | 10000 | global | 256 | 256 | 2.6 | 16.9 | 1 |
| auto | 4 | int32 | 40000 | global | 256 | 256 | 2.8 | 70.5 | 1 |
| auto | 4 | int8 | 1000 | local | 8 | 256 | 2.7 | 51.0 | 36 |
| auto | 4 | int8 | 2000 | local | 16 | 256 | 2.3 | 26.0 | 16 |
| auto | 4 | int8 | 4500 | local | 32 | 256 | 2.4 | 31.3 | 8 |
| auto | 4 | int8 | 4700 | global | 64 | 256 | 2.4 | 17.2 | 4 |
| auto | 4 | int8 | 10000 | global | 256 | 256 | 2.9 | 16.2 | 1 |
| auto | 4 | int8 | 40000 | global | 256 | 256 | 3.0 | 67.3 | 1 |
| auto | 20 | binary | 1000 | local | 5 | 256 | 3.7 | 68.5 | 52 |
| auto | 20 | binary | 2000 | local | 11 | 256 | 3.7 | 66.4 | 24 |
| auto | 20 | binary | 4500 | local | 22 | 256 | 3.8 | 80.8 | 12 |
| auto | 20 | binary | 4700 | global | 64 | 256 | 4.8 | 33.8 | 4 |
| auto | 20 | binary | 10000 | global | 128 | 256 | 5.1 | 41.3 | 2 |
| auto | 20 | binary | 40000 | global | 256 | 256 | 5.8 | 115.6 | 1 |
| auto | 20 | float | 1000 | local | 7 | 256 | 3.1 | 53.6 | 40 |
| auto | 20 | float | 2000 | local | 13 | 256 | 3.1 | 54.5 | 20 |
| auto | 20 | float | 4500 | local | 32 | 256 | 3.2 | 51.5 | 8 |
| auto | 20 | float | 4700 | global | 64 | 256 | 4.6 | 58.0 | 4 |
| auto | 20 | float | 10000 | global | 128 | 256 | 5.0 | 52.9 | 2 |
| auto | 20 | float | 40000 | global | 256 | 256 | 5.6 | 161.8 | 1 |
| auto | 20 | int32 | 1000 | local | 8 | 256 | 3.3 | 50.4 | 36 |
| auto | 20 | int32 | 2000 | local | 16 | 256 | 3.3 | 45.3 | 16 |
| auto | 20 | int32 | 4500 | local | 32 | 256 | 3.4 | 53.7 | 8 |
| auto | 20 | int32 | 4700 | global | 64 | 256 | 4.3 | 37.0 | 4 |
| auto | 20 | int32 | 10000 | global | 256 | 256 | 5.2 | 30.8 | 1 |
| auto | 20 | int32 | 40000 | global | 256 | 256 | 5.8 | 136.2 | 1 |
| auto | 20 | int8 | 1000 | local | 8 | 256 | 4.1 | 54.3 | 36 |
| auto | 20 | int8 | 2000 | local | 16 | 256 | 4.1 | 49.2 | 16 |
| auto | 20 | int8 | 4500 | local | 32 | 256 | 4.2 | 57.3 | 8 |
| auto | 20 | int8 | 4700 | global | 64 | 256 | 4.9 | 36.1 | 4 |
| auto | 20 | int8 | 10000 | global | 256 | 256 | 5.7 | 28.9 | 1 |
| auto | 20 | int8 | 40000 | global | 256 | 256 | 6.0 | 148.0 | 1 |
| auto | 62 | int32 | 1000 | local | 8 | 256 | 8.2 | 132.6 | 36 |
| auto | 62 | int32 | 2000 | local | 16 | 256 | 8.2 | 120.5 | 16 |
| auto | 62 | int32 | 4500 | local | 32 | 256 | 8.4 | 143.8 | 8 |
| auto | 62 | int32 | 4700 | global | 64 | 256 | 11.1 | 100.0 | 4 |
| auto | 62 | int32 | 10000 | global | 256 | 256 | 13.4 | 85.0 | 1 |
| auto | 62 | int32 | 40000 | global | 158 | 158 | 14.6 | 406.1 | 1 |
| auto | 62 | int8 | 1000 | local | 8 | 256 | 10.5 | 145.1 | 36 |
| auto | 62 | int8 | 2000 | local | 16 | 256 | 10.4 | 131.6 | 16 |
| auto | 62 | int8 | 4500 | local | 32 | 256 | 10.6 | 154.3 | 8 |
| auto | 62 | int8 | 4700 | global | 64 | 256 | 12.6 | 99.0 | 4 |
| auto | 62 | int8 | 10000 | global | 256 | 256 | 13.8 | 76.0 | 1 |
| auto | 62 | int8 | 40000 | global | 158 | 158 | 15.0 | 356.1 | 1 |
| auto | 200 | int32 | 1000 | local | 8 | 256 | 34.1 | 612.6 | 36 |
| auto | 200 | int32 | 2000 | local | 16 | 256 | 34.9 | 579.1 | 16 |
| auto | 200 | int32 | 4500 | local | 32 | 256 | 35.3 | 671.9 | 8 |
| auto | 200 | int32 | 4700 | global | 64 | 256 | 51.1 | 657.3 | 4 |
| auto | 200 | int32 | 10000 | global | 198 | 198 | 53.4 | 358.7 | 1 |
| auto | 200 | int32 | 40000 | global | 49 | 49 | 57.4 | 1385.8 | 1 |
| auto | 200 | int8 | 1000 | dense | 256 | 256 | 206.4 | 149.3 | 1 |
| auto | 200 | int8 | 2000 | local | 16 | 256 | 46.8 | 652.3 | 16 |
| auto | 200 | int8 | 4500 | local | 32 | 256 | 42.9 | 697.4 | 8 |
| auto | 200 | int8 | 4700 | global | 64 | 256 | 53.8 | 484.3 | 4 |
| auto | 200 | int8 | 10000 | global | 198 | 198 | 56.1 | 328.6 | 1 |
| auto | 200 | int8 | 40000 | global | 49 | 49 | 59.6 | 1268.0 | 1 |
| auto | complete | binary | 500 | dense | 64 | 256 | 124.4 | 140.9 | 4 |
| auto | complete | binary | 1000 | dense | 256 | 256 | 245.4 | 161.5 | 1 |
| auto | complete | binary | 1414 | dense | 198 | 198 | 347.2 | 352.8 | 1 |
| auto | complete | float | 500 | dense | 64 | 256 | 55.5 | 60.8 | 4 |
| auto | complete | float | 1000 | dense | 256 | 256 | 121.2 | 107.0 | 1 |
| auto | complete | float | 1414 | dense | 198 | 198 | 169.2 | 199.2 | 1 |
| auto | complete | int32 | 500 | dense | 64 | 256 | 57.4 | 62.5 | 4 |
| auto | complete | int32 | 1000 | dense | 256 | 256 | 120.8 | 110.4 | 1 |
| auto | complete | int32 | 1414 | dense | 198 | 198 | 173.6 | 198.8 | 1 |
| auto | complete | int8 | 500 | dense | 64 | 256 | 113.5 | 112.1 | 4 |
| auto | complete | int8 | 1000 | dense | 256 | 256 | 227.6 | 192.3 | 1 |
| auto | complete | int8 | 1414 | dense | 198 | 198 | 322.5 | 347.6 | 1 |
| dense | 62 | int32 | 1000 | dense | 256 | 256 | 90.4 | 67.2 | 1 |
| dense | 62 | int32 | 1414 | dense | 256 | 256 | 130.0 | 147.5 | 1 |
| dense | 62 | int8 | 1000 | dense | 256 | 256 | 204.2 | 165.5 | 1 |
| dense | 62 | int8 | 1414 | dense | 256 | 256 | 287.4 | 291.8 | 1 |
| dense | 200 | int32 | 1000 | dense | 256 | 256 | 95.0 | 71.6 | 1 |
| dense | 200 | int32 | 1414 | dense | 256 | 256 | 133.6 | 155.5 | 1 |
| dense | 200 | int8 | 1000 | dense | 256 | 256 | 208.2 | 149.4 | 1 |
| dense | 200 | int8 | 1414 | dense | 256 | 256 | 287.5 | 297.9 | 1 |
| sparse | 62 | int32 | 1000 | local | 8 | 256 | 8.1 | 132.2 | 36 |
| sparse | 62 | int32 | 1414 | local | 11 | 256 | 8.2 | 127.9 | 24 |
| sparse | 62 | int32 | 2500 | local | 22 | 256 | 8.2 | 114.3 | 12 |
| sparse | 62 | int8 | 1000 | local | 8 | 256 | 10.7 | 164.9 | 36 |
| sparse | 62 | int8 | 1414 | local | 11 | 256 | 10.4 | 138.2 | 24 |
| sparse | 62 | int8 | 2500 | local | 22 | 256 | 10.5 | 124.8 | 12 |
| sparse | 200 | int32 | 1000 | local | 8 | 256 | 34.1 | 612.0 | 36 |
| sparse | 200 | int32 | 1414 | local | 11 | 256 | 34.9 | 590.4 | 24 |
| sparse | 200 | int32 | 2500 | local | 14 | 198 | 35.2 | 684.2 | 15 |
| sparse | 200 | int8 | 1000 | local | 8 | 256 | 42.7 | 643.8 | 36 |
| sparse | 200 | int8 | 1414 | local | 11 | 256 | 42.4 | 607.1 | 24 |
| sparse | 200 | int8 | 2500 | local | 14 | 198 | 42.6 | 679.0 | 15 |
| sparse | complete | int32 | 1000 | local | 3 | 99 | 251.2 | 4773.1 | 39 |
| sparse | complete | int32 | 1414 | local | 2 | 49 | 383.3 | 6333.4 | 27 |
| sparse | complete | int32 | 2500 | local | 2 | 16 | 976.2 | 9805.9 | 15 |
| sparse | complete | int8 | 1000 | local | 3 | 99 | 291.2 | 4771.3 | 39 |
| sparse | complete | int8 | 1414 | local | 2 | 49 | 455.7 | 6987.5 | 27 |
| sparse | complete | int8 | 2500 | local | 2 | 16 | 1009.0 | 10133.8 | 15 |

Refused SA runs:

| Storage | Degree | Precision | Nodes | Reason |
|---|---|---|---|---|
| auto | complete | binary | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | binary | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |
| auto | complete | float | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | float | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |
| auto | complete | int32 | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | int32 | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |
| auto | complete | int8 | 2500 | one dense sweep costs an estimated 1250 ms, over the 400 ms limit |
| auto | complete | int8 | 5000 | one dense sweep costs an estimated 5000 ms, over the 400 ms limit |

### Gibbs results

Each row is one run. The time of one sweep in the longest chunk is the chunk time divided by its sweeps.

| Storage | Degree | Precision | Nodes | Tier | Chunks | Sweeps | ns per update | Longest chunk (ms) | Its sweeps |
|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 1000 | shared | 1 | 256 | 0.0 | 2.1 | 256 |
| auto | 4 | int32 | 2000 | shared | 1 | 256 | 0.0 | 3.9 | 256 |
| auto | 4 | int32 | 4700 | shared | 2 | 256 | 0.0 | 8.0 | 172 |
| auto | 4 | int32 | 4900 | spill | 5 | 256 | 0.0 | 2.2 | 52 |
| auto | 4 | int32 | 10000 | spill | 11 | 256 | 0.0 | 2.3 | 24 |
| auto | 4 | int32 | 40000 | spill | 64 | 256 | 0.1 | 12.1 | 4 |
| auto | 4 | int8 | 1000 | shared | 1 | 256 | 0.0 | 2.0 | 256 |
| auto | 4 | int8 | 2000 | shared | 1 | 256 | 0.0 | 3.8 | 256 |
| auto | 4 | int8 | 4700 | shared | 2 | 256 | 0.0 | 5.9 | 172 |
| auto | 4 | int8 | 4900 | spill | 5 | 256 | 0.0 | 1.9 | 52 |
| auto | 4 | int8 | 10000 | spill | 11 | 256 | 0.0 | 2.7 | 24 |
| auto | 4 | int8 | 40000 | spill | 64 | 256 | 0.0 | 2.3 | 4 |
| auto | 20 | binary | 1000 | shared | 1 | 256 | 0.1 | 7.8 | 256 |
| auto | 20 | binary | 2000 | shared | 1 | 256 | 0.1 | 14.3 | 256 |
| auto | 20 | binary | 4700 | shared | 1 | 256 | 0.1 | 36.5 | 256 |
| auto | 20 | binary | 4900 | spill | 3 | 256 | 0.2 | 32.3 | 108 |
| auto | 20 | binary | 10000 | spill | 5 | 256 | 0.4 | 50.7 | 52 |
| auto | 20 | binary | 40000 | spill | 22 | 256 | 0.6 | 82.9 | 12 |
| auto | 20 | float | 1000 | shared | 1 | 256 | 0.2 | 10.8 | 256 |
| auto | 20 | float | 2000 | shared | 1 | 256 | 0.2 | 20.2 | 256 |
| auto | 20 | float | 4700 | shared | 1 | 256 | 0.1 | 39.0 | 256 |
| auto | 20 | float | 4900 | spill | 4 | 256 | 0.3 | 34.3 | 84 |
| auto | 20 | float | 10000 | spill | 7 | 256 | 0.4 | 44.6 | 40 |
| auto | 20 | float | 40000 | spill | 32 | 256 | 0.6 | 59.5 | 8 |
| auto | 20 | int32 | 1000 | shared | 1 | 256 | 0.1 | 8.0 | 256 |
| auto | 20 | int32 | 2000 | shared | 1 | 256 | 0.1 | 16.1 | 256 |
| auto | 20 | int32 | 4700 | shared | 2 | 256 | 0.1 | 30.8 | 172 |
| auto | 20 | int32 | 4900 | spill | 5 | 256 | 0.3 | 18.6 | 52 |
| auto | 20 | int32 | 10000 | spill | 11 | 256 | 0.4 | 26.5 | 24 |
| auto | 20 | int32 | 40000 | spill | 64 | 256 | 0.6 | 39.9 | 4 |
| auto | 20 | int8 | 1000 | shared | 1 | 256 | 0.2 | 10.3 | 256 |
| auto | 20 | int8 | 2000 | shared | 1 | 256 | 0.1 | 16.0 | 256 |
| auto | 20 | int8 | 4700 | shared | 2 | 256 | 0.1 | 27.8 | 172 |
| auto | 20 | int8 | 4900 | spill | 5 | 256 | 0.3 | 33.0 | 52 |
| auto | 20 | int8 | 10000 | spill | 11 | 256 | 0.4 | 23.0 | 24 |
| auto | 20 | int8 | 40000 | spill | 64 | 256 | 0.5 | 30.5 | 4 |
| auto | 62 | int32 | 1000 | shared | 1 | 256 | 1.1 | 74.0 | 256 |
| auto | 62 | int32 | 2000 | shared | 1 | 256 | 0.6 | 72.6 | 256 |
| auto | 62 | int32 | 4700 | shared | 2 | 256 | 0.5 | 102.0 | 172 |
| auto | 62 | int32 | 4900 | spill | 5 | 256 | 1.3 | 93.7 | 52 |
| auto | 62 | int32 | 10000 | spill | 11 | 256 | 1.2 | 84.1 | 24 |
| auto | 62 | int32 | 40000 | spill | 27 | 158 | 2.2 | 203.5 | 6 |
| auto | 62 | int8 | 1000 | shared | 1 | 256 | 0.8 | 51.1 | 256 |
| auto | 62 | int8 | 2000 | shared | 1 | 256 | 0.8 | 103.3 | 256 |
| auto | 62 | int8 | 4700 | shared | 2 | 256 | 0.5 | 106.1 | 172 |
| auto | 62 | int8 | 4900 | spill | 5 | 256 | 1.1 | 78.6 | 52 |
| auto | 62 | int8 | 10000 | spill | 11 | 256 | 1.2 | 86.7 | 24 |
| auto | 62 | int8 | 40000 | spill | 27 | 158 | 1.9 | 134.8 | 6 |
| auto | 200 | int32 | 1000 | shared | 1 | 256 | 4.6 | 303.8 | 256 |
| auto | 200 | int32 | 2000 | shared | 1 | 256 | 2.7 | 359.5 | 256 |
| auto | 200 | int32 | 4700 | shared | 2 | 256 | 1.9 | 310.9 | 172 |
| auto | 200 | int32 | 4900 | spill | 5 | 256 | 4.0 | 317.3 | 52 |
| auto | 200 | int32 | 10000 | spill | 8 | 198 | 4.5 | 344.0 | 27 |
| auto | 200 | int32 | 40000 | spill | 9 | 49 | 7.2 | 575.2 | 6 |
| auto | 200 | int8 | 1000 | dense | 256 | 256 | 24.1 | 7.1 | 1 |
| auto | 200 | int8 | 2000 | shared | 1 | 256 | 3.5 | 460.8 | 256 |
| auto | 200 | int8 | 4700 | shared | 2 | 256 | 2.0 | 394.3 | 172 |
| auto | 200 | int8 | 4900 | spill | 5 | 256 | 4.0 | 291.8 | 52 |
| auto | 200 | int8 | 10000 | spill | 8 | 198 | 4.3 | 319.5 | 27 |
| auto | 200 | int8 | 40000 | spill | 9 | 49 | 7.1 | 477.1 | 6 |
| auto | complete | binary | 500 | dense | 64 | 256 | 39.3 | 40.6 | 4 |
| auto | complete | binary | 1000 | dense | 256 | 256 | 89.2 | 25.3 | 1 |
| auto | complete | binary | 1673 | dense | 142 | 142 | 155.6 | 84.7 | 1 |
| auto | complete | float | 500 | dense | 64 | 256 | 36.2 | 19.8 | 4 |
| auto | complete | float | 1000 | dense | 256 | 256 | 85.0 | 43.3 | 1 |
| auto | complete | float | 1673 | dense | 142 | 142 | 158.6 | 84.7 | 1 |
| auto | complete | int32 | 500 | dense | 64 | 256 | 41.8 | 24.1 | 4 |
| auto | complete | int32 | 1000 | dense | 256 | 256 | 106.7 | 123.4 | 1 |
| auto | complete | int32 | 1673 | dense | 142 | 142 | 160.9 | 83.8 | 1 |
| auto | complete | int8 | 500 | dense | 64 | 256 | 43.9 | 23.5 | 4 |
| auto | complete | int8 | 1000 | dense | 256 | 256 | 89.0 | 39.8 | 1 |
| auto | complete | int8 | 1673 | dense | 142 | 142 | 154.7 | 80.8 | 1 |
| dense | 62 | int32 | 1000 | dense | 256 | 256 | 4.1 | 1.2 | 1 |
| dense | 62 | int32 | 1414 | dense | 256 | 256 | 5.7 | 2.7 | 1 |
| dense | 62 | int8 | 1000 | dense | 256 | 256 | 5.3 | 5.5 | 1 |
| dense | 62 | int8 | 1414 | dense | 256 | 256 | 8.3 | 3.6 | 1 |
| dense | 200 | int32 | 1000 | dense | 256 | 256 | 23.7 | 6.7 | 1 |
| dense | 200 | int32 | 1414 | dense | 256 | 256 | 15.9 | 6.6 | 1 |
| dense | 200 | int8 | 1000 | dense | 256 | 256 | 24.8 | 8.2 | 1 |
| dense | 200 | int8 | 1414 | dense | 256 | 256 | 17.1 | 7.2 | 1 |
| sparse | 62 | int32 | 1000 | shared | 1 | 256 | 0.4 | 25.3 | 256 |
| sparse | 62 | int32 | 1414 | shared | 1 | 256 | 0.3 | 32.1 | 256 |
| sparse | 62 | int32 | 2500 | shared | 1 | 256 | 0.3 | 53.6 | 256 |
| sparse | 62 | int8 | 1000 | shared | 1 | 256 | 0.9 | 56.2 | 256 |
| sparse | 62 | int8 | 1414 | shared | 1 | 256 | 0.3 | 32.4 | 256 |
| sparse | 62 | int8 | 2500 | shared | 1 | 256 | 0.3 | 49.7 | 256 |
| sparse | 200 | int32 | 1000 | shared | 1 | 256 | 3.4 | 223.7 | 256 |
| sparse | 200 | int32 | 1414 | shared | 1 | 256 | 1.7 | 161.3 | 256 |
| sparse | 200 | int32 | 2500 | shared | 1 | 198 | 2.4 | 298.3 | 198 |
| sparse | 200 | int8 | 1000 | shared | 1 | 256 | 3.2 | 206.7 | 256 |
| sparse | 200 | int8 | 1414 | shared | 1 | 256 | 1.5 | 139.8 | 256 |
| sparse | 200 | int8 | 2500 | shared | 1 | 198 | 2.5 | 312.8 | 198 |
| sparse | complete | int32 | 1000 | shared | 1 | 99 | 172.6 | 4375.6 | 99 |
| sparse | complete | int32 | 1414 | shared | 1 | 49 | 135.6 | 2404.9 | 49 |
| sparse | complete | int32 | 2500 | shared | 1 | 16 | 292.6 | 2996.4 | 16 |
| sparse | complete | int8 | 1000 | shared | 1 | 99 | 98.5 | 2496.6 | 99 |
| sparse | complete | int8 | 1414 | shared | 1 | 49 | 143.9 | 2552.5 | 49 |
| sparse | complete | int8 | 2500 | shared | 1 | 16 | 342.9 | 3510.9 | 16 |

Refused Gibbs runs:

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

Each row is one run. The time of one sweep in the longest chunk is the chunk time divided by its sweeps.

| Storage | Degree | Precision | Nodes | Tier | Chunks | Sweeps | ns per update | Longest chunk (ms) | Its sweeps |
|---|---|---|---|---|---|---|---|---|---|
| auto | 4 | int32 | 1000 | shared | 1 | 256 | 0.1 | 5.9 | 256 |
| auto | 4 | int32 | 4000 | shared | 1 | 256 | 0.1 | 16.5 | 256 |
| auto | 4 | int32 | 8000 | spill | 1 | 256 | 0.1 | 33.1 | 256 |
| auto | 4 | int32 | 16000 | spill | 2 | 256 | 0.1 | 56.1 | 232 |
| auto | 4 | int32 | 40000 | spill | 3 | 256 | 0.1 | 53.6 | 92 |
| auto | 4 | int8 | 1000 | shared | 1 | 256 | 0.1 | 6.5 | 256 |
| auto | 4 | int8 | 4000 | shared | 1 | 256 | 0.1 | 16.9 | 256 |
| auto | 4 | int8 | 8000 | spill | 1 | 256 | 0.1 | 33.1 | 256 |
| auto | 4 | int8 | 16000 | spill | 2 | 256 | 0.1 | 56.1 | 232 |
| auto | 4 | int8 | 40000 | spill | 3 | 256 | 0.1 | 53.6 | 92 |
| auto | 20 | binary | 1000 | shared | 1 | 256 | 0.2 | 10.7 | 256 |
| auto | 20 | binary | 4000 | shared | 1 | 256 | 0.1 | 14.0 | 256 |
| auto | 20 | binary | 8000 | spill | 1 | 256 | 0.1 | 52.6 | 256 |
| auto | 20 | binary | 16000 | spill | 2 | 256 | 0.1 | 110.8 | 232 |
| auto | 20 | binary | 40000 | spill | 3 | 256 | 0.2 | 159.4 | 92 |
| auto | 20 | float | 1000 | shared | 2 | 256 | 0.4 | 19.7 | 204 |
| auto | 20 | float | 4000 | shared | 2 | 256 | 0.1 | 34.0 | 236 |
| auto | 20 | float | 8000 | spill | 1 | 256 | 0.2 | 81.1 | 256 |
| auto | 20 | float | 16000 | spill | 2 | 256 | 0.2 | 115.9 | 152 |
| auto | 20 | float | 40000 | spill | 5 | 256 | 0.2 | 144.4 | 60 |
| auto | 20 | int32 | 1000 | shared | 1 | 256 | 0.3 | 18.7 | 256 |
| auto | 20 | int32 | 4000 | shared | 1 | 256 | 0.1 | 28.9 | 256 |
| auto | 20 | int32 | 8000 | spill | 1 | 256 | 0.1 | 52.7 | 256 |
| auto | 20 | int32 | 16000 | spill | 2 | 256 | 0.1 | 110.9 | 232 |
| auto | 20 | int32 | 40000 | spill | 3 | 256 | 0.1 | 151.3 | 92 |
| auto | 20 | int8 | 1000 | shared | 1 | 256 | 0.3 | 22.7 | 256 |
| auto | 20 | int8 | 4000 | shared | 1 | 256 | 0.1 | 31.3 | 256 |
| auto | 20 | int8 | 8000 | spill | 1 | 256 | 0.1 | 52.6 | 256 |
| auto | 20 | int8 | 16000 | spill | 2 | 256 | 0.1 | 110.9 | 232 |
| auto | 20 | int8 | 40000 | spill | 3 | 256 | 0.2 | 153.7 | 92 |
| auto | 62 | int32 | 1000 | shared | 13 | 256 | 2.8 | 14.3 | 20 |
| auto | 62 | int32 | 4000 | shared | 11 | 256 | 0.5 | 13.4 | 24 |
| auto | 62 | int32 | 8000 | spill | 11 | 256 | 0.5 | 43.0 | 24 |
| auto | 62 | int32 | 16000 | spill | 11 | 256 | 0.4 | 63.8 | 24 |
| auto | 62 | int32 | 40000 | spill | 16 | 158 | 0.6 | 69.5 | 10 |
| auto | 62 | int8 | 1000 | shared | 13 | 256 | 3.2 | 42.0 | 20 |
| auto | 62 | int8 | 4000 | shared | 11 | 256 | 0.5 | 13.4 | 24 |
| auto | 62 | int8 | 8000 | spill | 11 | 256 | 0.4 | 21.4 | 24 |
| auto | 62 | int8 | 16000 | spill | 11 | 256 | 0.4 | 63.1 | 24 |
| auto | 62 | int8 | 40000 | spill | 16 | 158 | 0.6 | 64.4 | 10 |
| auto | 200 | int32 | 1000 | dense | 22 | 256 | 34.2 | 106.8 | 12 |
| auto | 200 | int32 | 4000 | dense | 256 | 256 | 219.5 | 279.6 | 1 |
| auto | 200 | int8 | 1000 | dense | 22 | 256 | 34.4 | 106.7 | 12 |
| auto | 200 | int8 | 4000 | dense | 256 | 256 | 166.5 | 230.6 | 1 |
| auto | complete | binary | 1000 | dense | 22 | 256 | 17.4 | 72.1 | 12 |
| auto | complete | binary | 2500 | dense | 63 | 63 | 44.8 | 56.9 | 1 |
| auto | complete | binary | 4898 | dense | 16 | 16 | 86.4 | 140.5 | 1 |
| auto | complete | float | 1000 | dense | 22 | 256 | 28.8 | 103.6 | 12 |
| auto | complete | float | 2500 | dense | 63 | 63 | 119.3 | 102.4 | 1 |
| auto | complete | float | 4898 | dense | 16 | 16 | 245.1 | 330.0 | 1 |
| auto | complete | int32 | 1000 | dense | 22 | 256 | 35.6 | 129.6 | 12 |
| auto | complete | int32 | 2500 | dense | 63 | 63 | 129.3 | 103.4 | 1 |
| auto | complete | int32 | 4898 | dense | 16 | 16 | 270.8 | 359.2 | 1 |
| auto | complete | int8 | 1000 | dense | 22 | 256 | 34.5 | 128.4 | 12 |
| auto | complete | int8 | 2500 | dense | 63 | 63 | 97.9 | 87.3 | 1 |
| auto | complete | int8 | 4898 | dense | 16 | 16 | 203.0 | 270.6 | 1 |
| dense | 62 | int32 | 1000 | dense | 22 | 256 | 33.9 | 105.6 | 12 |
| dense | 62 | int32 | 1414 | dense | 64 | 256 | 52.8 | 136.6 | 4 |
| dense | 62 | int8 | 1000 | dense | 22 | 256 | 34.4 | 107.2 | 12 |
| dense | 62 | int8 | 1414 | dense | 64 | 256 | 49.6 | 74.2 | 4 |
| dense | 200 | int32 | 1000 | dense | 22 | 256 | 34.3 | 108.8 | 12 |
| dense | 200 | int32 | 1414 | dense | 64 | 256 | 51.2 | 77.5 | 4 |
| dense | 200 | int8 | 1000 | dense | 22 | 256 | 34.2 | 107.9 | 12 |
| dense | 200 | int8 | 1414 | dense | 64 | 256 | 49.8 | 73.9 | 4 |
| sparse | 62 | int32 | 1000 | shared | 13 | 256 | 2.8 | 14.3 | 20 |
| sparse | 62 | int32 | 1414 | shared | 13 | 256 | 1.9 | 13.6 | 20 |
| sparse | 62 | int32 | 2500 | shared | 13 | 256 | 1.0 | 13.2 | 20 |
| sparse | 62 | int8 | 1000 | shared | 13 | 256 | 3.2 | 40.2 | 20 |
| sparse | 62 | int8 | 1414 | shared | 13 | 256 | 1.9 | 13.6 | 20 |
| sparse | 62 | int8 | 2500 | shared | 13 | 256 | 1.0 | 13.2 | 20 |

Refused MSA runs:

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
## Dense compared with sparse

For models that both storages hold, the table gives the time per spin update on the tier that auto storage chose. It then gives the time on forced sparse storage, and on forced dense storage. A dense update scans a whole matrix row, so its cost grows with the node count. At degrees 62 and 200 the dense build is slower than the sparse build in every row where the sparse build runs. At degree 200 and 1000 nodes the auto tier chose dense for SA at 206 ns against 43 ns sparse. It chose dense for Gibbs at 24 ns against 3 ns sparse. For MSA, the sparse build refuses degrees over 62.

| Kernel | Degree | Nodes | Precision | Auto tier | Auto ns | Sparse ns | Dense ns |
|---|---|---|---|---|---|---|---|
| sa | 62 | 1000 | int8 | local | 10.5 | 10.7 | 204.2 |
| sa | 62 | 1000 | int32 | local | 8.2 | 8.1 | 90.4 |
| sa | 62 | 1414 | int8 | n/a | n/a | 10.4 | 287.4 |
| sa | 62 | 1414 | int32 | n/a | n/a | 8.2 | 130.0 |
| sa | 200 | 1000 | int8 | dense | 206.4 | 42.7 | 208.2 |
| sa | 200 | 1000 | int32 | local | 34.1 | 34.1 | 95.0 |
| sa | 200 | 1414 | int8 | n/a | n/a | 42.4 | 287.5 |
| sa | 200 | 1414 | int32 | n/a | n/a | 34.9 | 133.6 |
| gibbs | 62 | 1000 | int8 | shared | 0.8 | 0.9 | 5.3 |
| gibbs | 62 | 1000 | int32 | shared | 1.1 | 0.4 | 4.1 |
| gibbs | 62 | 1414 | int8 | n/a | n/a | 0.3 | 8.3 |
| gibbs | 62 | 1414 | int32 | n/a | n/a | 0.3 | 5.7 |
| gibbs | 200 | 1000 | int8 | dense | 24.1 | 3.2 | 24.8 |
| gibbs | 200 | 1000 | int32 | shared | 4.6 | 3.4 | 23.7 |
| gibbs | 200 | 1414 | int8 | n/a | n/a | 1.5 | 17.1 |
| gibbs | 200 | 1414 | int32 | n/a | n/a | 1.7 | 15.9 |
| msa | 62 | 1000 | int8 | shared | 3.2 | 3.2 | 34.4 |
| msa | 62 | 1000 | int32 | shared | 2.8 | 2.8 | 33.9 |
| msa | 62 | 1414 | int8 | n/a | n/a | 1.9 | 49.6 |
| msa | 62 | 1414 | int32 | n/a | n/a | 1.9 | 52.8 |
| msa | 200 | 1000 | int8 | dense | 34.4 | n/a | 34.2 |
| msa | 200 | 1000 | int32 | dense | 34.2 | n/a | 34.3 |
| msa | 200 | 1414 | int8 | n/a | n/a | n/a | 49.8 |
| msa | 200 | 1414 | int32 | n/a | n/a | n/a | 51.2 |

## Runs whose longest chunk passed 400 ms

The list has every run with a longest chunk over 400 ms, by chunk time. The target of the planner is 400 ms, so these runs show where the planner missed. Two groups stand out. First, auto storage at degree 200 on SA and Gibbs plans chunks that run 460 to 1,400 ms, because the planner uses a rate that does not depend on degree. Second, forced sparse storage on complete models plans chunks of 2.4 to 10.1 seconds on SA and Gibbs. Forced sparse storage at degree 200 on SA plans chunks of 590 to 690 ms. Models of the same shape on auto storage take the dense build and stay under 400 ms. The one other run, SA at 40,000 nodes and degree 62 on the global build, has one sweep per chunk and a longest chunk of 406 ms, so one sweep alone is past the target.

| Kernel | Storage | Degree | Precision | Nodes | Tier | Chunks | Longest chunk (ms) | Its sweeps | ms per sweep |
|---|---|---|---|---|---|---|---|---|---|
| sa | sparse | complete | int8 | 2500 | local | 2 | 10134 | 15 | 675.6 |
| sa | sparse | complete | int32 | 2500 | local | 2 | 9806 | 15 | 653.7 |
| sa | sparse | complete | int8 | 1414 | local | 2 | 6988 | 27 | 258.8 |
| sa | sparse | complete | int32 | 1414 | local | 2 | 6333 | 27 | 234.6 |
| sa | sparse | complete | int32 | 1000 | local | 3 | 4773 | 39 | 122.4 |
| sa | sparse | complete | int8 | 1000 | local | 3 | 4771 | 39 | 122.3 |
| gibbs | sparse | complete | int32 | 1000 | shared | 1 | 4376 | 99 | 44.2 |
| gibbs | sparse | complete | int8 | 2500 | shared | 1 | 3511 | 16 | 219.4 |
| gibbs | sparse | complete | int32 | 2500 | shared | 1 | 2996 | 16 | 187.3 |
| gibbs | sparse | complete | int8 | 1414 | shared | 1 | 2552 | 49 | 52.1 |
| gibbs | sparse | complete | int8 | 1000 | shared | 1 | 2497 | 99 | 25.2 |
| gibbs | sparse | complete | int32 | 1414 | shared | 1 | 2405 | 49 | 49.1 |
| sa | auto | 200 | int32 | 40000 | global | 49 | 1386 | 1 | 1385.8 |
| sa | auto | 200 | int8 | 40000 | global | 49 | 1268 | 1 | 1268.0 |
| sa | auto | 200 | int8 | 4500 | local | 32 | 697 | 8 | 87.2 |
| sa | sparse | 200 | int32 | 2500 | local | 14 | 684 | 15 | 45.6 |
| sa | sparse | 200 | int8 | 2500 | local | 14 | 679 | 15 | 45.3 |
| sa | auto | 200 | int32 | 4500 | local | 32 | 672 | 8 | 84.0 |
| sa | auto | 200 | int32 | 4700 | global | 64 | 657 | 4 | 164.3 |
| sa | auto | 200 | int8 | 2000 | local | 16 | 652 | 16 | 40.8 |
| sa | sparse | 200 | int8 | 1000 | local | 8 | 644 | 36 | 17.9 |
| sa | auto | 200 | int32 | 1000 | local | 8 | 613 | 36 | 17.0 |
| sa | sparse | 200 | int32 | 1000 | local | 8 | 612 | 36 | 17.0 |
| sa | sparse | 200 | int8 | 1414 | local | 11 | 607 | 24 | 25.3 |
| sa | sparse | 200 | int32 | 1414 | local | 11 | 590 | 24 | 24.6 |
| sa | auto | 200 | int32 | 2000 | local | 16 | 579 | 16 | 36.2 |
| gibbs | auto | 200 | int32 | 40000 | spill | 9 | 575 | 6 | 95.9 |
| sa | auto | 200 | int8 | 4700 | global | 64 | 484 | 4 | 121.1 |
| gibbs | auto | 200 | int8 | 40000 | spill | 9 | 477 | 6 | 79.5 |
| gibbs | auto | 200 | int8 | 2000 | shared | 1 | 461 | 256 | 1.8 |
| sa | auto | 62 | int32 | 40000 | global | 158 | 406 | 1 | 406.1 |
## Limits of these measurements

- Each point ran once. The load of the shared machine adds noise, which is larger for runs of a few milliseconds.
- A short run, such as Gibbs at degree 4, has a GPU time of a few milliseconds, so its time per update (0.03 ns) has a large relative error.
- The number of sweeps changes between grid points. Large models run 16 to 49 sweeps, so their longest chunk has few sweeps and may include start-up cost.
- The circulant graph has a regular neighbour pattern with good memory locality. A graph with random neighbours may cost more per update on the global and spill tiers.
- The harness measures `Tuning::default` planning, which is the production planner on this host.
