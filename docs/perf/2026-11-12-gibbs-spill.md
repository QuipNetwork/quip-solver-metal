# Spill measurements for Gibbs sampling, 2026-11-12

The chromatic Gibbs kernel has a spill build, `QUIP_GIBBS_SPILL`, for models past its 4,800-spin threadgroup array. The build keeps the first spins of each read in threadgroup memory and the rest in the read's slice of the persistent spin buffer. On a model of 4,000 nodes the spill build updates spins 2.0 to 2.5 times slower than the threadgroup build. The chunk planner multiplies the Gibbs cost by `GIBBS_SPILL_COST_FACTOR = 3.0` on the spill build. The worst planned chunk was 117.8 ms, for a model of 20,000 nodes and degree 40 on the unit path, with 256 reads. Models of at most 4,800 nodes keep the threadgroup build, and the Gibbs digest `bc085a7fc384f073` does not change.

## Host and load

The host was an Apple M4 Max. Other agents used the host during the runs. The one-minute load average was 6.0 at the end of the run that produced the tables.

## Commands and method

```sh
cargo test --release --test gibbs_capacity -- --ignored --nocapture --test-threads=1 spill_rates
```

`spill_rates` in `tests/gibbs_capacity.rs` runs circulant models with 64 sweeps at 4 sweeps per rung, on four coefficient paths. The unit path runs the int8 build. Two milli paths run on `int` and `int8_t` storage. The last path is Float32. For each model it records the median GPU time of three runs in one command buffer, and the spin updates per second, where one update advances one read by one spin. It then runs the model with the production chunk plan and records the chunk count and the largest chunk's GPU time. A model of 4,000 nodes runs on both builds. For the spill build on this model, the test caps the threadgroup prefix at 1,000 spins, so 3,000 spins live in device memory. A larger model runs only on the spill build, with the full prefix of 4,800 spins.

Every model runs 256 reads, the kernel's read cap (`MAX_READS`), and the update count uses the reads that ran. The test asserts that all 256 reads came back.

## Rates

Rates are in billions of spin updates per second (Gupd/s). The ratio divides the local rate at 4,000 nodes by the spill rate of the same path.

| Nodes | Degree | Path | Local Gupd/s at 4,000 nodes | Spill Gupd/s | Ratio |
| --- | --- | --- | --- | --- | --- |
| 4,000 | 20 | unit | 6.1049 | 2.5527 | 2.39 |
| 4,000 | 20 | `int` | 4.6519 | 2.0319 | 2.29 |
| 4,000 | 20 | `int8_t` | 6.1850 | 2.4993 | 2.47 |
| 4,000 | 20 | Float32 | 4.1645 | 2.0468 | 2.03 |
| 20,000 | 20 | unit | 6.1049 | 3.3423 | 1.83 |
| 20,000 | 20 | `int` | 4.6519 | 2.4765 | 1.88 |
| 20,000 | 20 | `int8_t` | 6.1850 | 2.9795 | 2.08 |
| 20,000 | 20 | Float32 | 4.1645 | 2.2700 | 1.83 |
| 20,000 | 40 | unit | 6.1049 | 1.0112 | 6.04 |
| 20,000 | 40 | `int` | 4.6519 | 1.0884 | 4.27 |
| 20,000 | 40 | `int8_t` | 6.1850 | 1.1792 | 5.25 |
| 20,000 | 40 | Float32 | 4.1645 | 1.1167 | 3.73 |
| 60,000 | 20 | unit | 6.1049 | 1.7425 | 3.50 |
| 60,000 | 20 | `int` | 4.6519 | 1.7176 | 2.71 |
| 60,000 | 20 | `int8_t` | 6.1850 | 2.4535 | 2.52 |
| 60,000 | 20 | Float32 | 4.1645 | 1.7065 | 2.44 |
| 200,000 | 20 | unit | 6.1049 | 1.3241 | 4.61 |
| 200,000 | 20 | `int` | 4.6519 | 1.1011 | 4.23 |
| 200,000 | 20 | `int8_t` | 6.1850 | 1.5277 | 4.05 |
| 200,000 | 20 | Float32 | 4.1645 | 0.9726 | 4.28 |

The local rate varies between runs. An earlier run on a busier host measured 6.45, 5.36, 6.07 and 6.21 Gupd/s for the four paths, which gives ratios of 2.53, 2.45, 2.11 and 2.26 at 4,000 nodes. The factor of 3.0 rounds the largest ratio at 4,000 nodes, 2.53, up.

The factor does not bound the ratio at every size. At degree 20, the models of 60,000 and 200,000 nodes reach 3.50 and 4.61 on the unit path, and the models at degree 40 reach 6.04. Those models stay safe because the planner's base rate is pessimistic. `GIBBS_UPDATES_PER_SEC` times `GIBBS_THROUGHPUT_SAFETY` is 0.84 Gupd/s, against measured local rates of 4 to 6 Gupd/s. With the factor of 3.0 the planner sizes a chunk at 0.28 Gupd/s, and the lowest spill rate in the table is 0.97 Gupd/s (0.72 Gupd/s in the review probe). A later change must not raise the base rate or lower the safety margin without measuring the large spill models again.

## Chunks

The planned runs used the cost factor of 3.0. A rung splits into sweep windows when one rung exceeds a dispatch's budget, as on the SA and weighted multi-spin kernels.

| Nodes | Degree | Chunks | Largest chunk ms, by path (unit, `int`, `int8_t`, Float32) |
| --- | --- | --- | --- |
| 4,000 | 20 | 1 | 25.2, 25.9, 29.8, 26.9 |
| 20,000 | 20 | 3, 6, 6, 4 | 42.1, 25.4, 20.6, 50.5 |
| 20,000 | 40 | 3, 6, 6, 4 | 117.8, 58.0, 49.8, 101.5 |
| 60,000 | 20 | 8, 16, 16, 16 | 91.6, 29.6, 24.3, 54.4 |
| 200,000 | 20 | 32, 64, 64, 32 | 105.8, 44.4, 45.3, 110.8 |

The worst chunk was 117.8 ms, for 20,000 nodes at degree 40 on the unit path. A review probe at lower read counts measured 130.5 ms (200,000 nodes, Float32, 160 reads) and a lowest whole-run rate of 0.72 Gupd/s (20,000 nodes, degree 40, unit, 32 reads). Each chunk is below the 200 ms goal and the 400 ms watchdog margin. No planned chunk was a single sweep, so the run did not exercise the one-sweep floor. The slowest sweep measured was on the 200,000-node Float32 model, where 64 sweeps took 3,369.3 ms in one command buffer, or 52.6 ms per sweep. A model whose single sweep passes the 500 ms ceiling would pass it, because the planner does not split a sweep. A separate bead, `quip-miner-metal-dvg`, owns the admission check for that floor.

## Device energy

The spill build skips the device energy, which the host never reads because it rescores every sample. No flip decision uses the energy, so the reads do not change.

## Digests

The pm1 digests in the table below come from `cargo test --release --test pm1_rates -- --ignored --nocapture`, with `QUIP_PM1_SCALE=wide` and `QUIP_PM1_SCALE=float` for the weighted and Float32 multi-spin annealing jobs. Every digest equals its value before this change.

| Job set | Digest |
| --- | --- |
| ±1, unit SA | `e4e22f86b64cda5d` |
| ±1, Gibbs | `bc085a7fc384f073` |
| ±1.5, wide Gibbs | `e9069459b5895d15` |
| ±1.0005, Float32 Gibbs | `eabc348f9005570c` |
| ±1, unit multi-spin | `6ec8d35ff7c9f123` |
| ±1.5, weighted multi-spin | `4ee84dd42f5d5ade` |
| ±1.0005, Float32 multi-spin | `c103cce43c429e4b` |
