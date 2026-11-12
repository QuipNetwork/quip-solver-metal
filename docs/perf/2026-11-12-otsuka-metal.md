# Otsuka cluster anneal chunk sizing on Metal, 2026-11-12

The Otsuka cluster kernel, `cluster_anneal` in `kernels/cluster.metal`, runs one threadgroup of 512 threads per anneal. The host splits the sweeps into command buffers with `chunk_plan` on the simulated annealing (SA) occupancy curve. `CLUSTER_COST_FACTOR` in `src/cluster/device.rs` multiplies the cost of one node update. The factor is 7.0. With that factor, the longest measured command buffer was 304.8 ms, below the 400 ms limit. The plan does not let one command buffer pass the 500 ms ceiling.

## Host and load

The host was an Apple M4 Max with 128 GiB of memory, measured on 2026-11-12 between 13:00 and 13:25 local time. Other agents used the host during every run. The one-minute load average was between 10.6 and 20.0, more than 3 in every run, so the factor has a margin over the measured need.

## Commands and method

```sh
cargo test --release --test cluster_device -- --ignored --nocapture cluster_chunks
```

`cluster_chunks_stay_under_the_ceiling` in `tests/cluster_device.rs` generates a `quip/twisted-ice-dimer@1` instance on the Advantage2 working graph (`tests/fixtures/advantage2-system1.edges`, 4,577 nodes and 41,515 edges). The generator runs in adjacency mode with `weak_milli` 62 and 256 pairs. The tile and pair weights are 250 milli. For each anneal count from 1 to 256, the test runs 4,096 sweeps with the default schedule and the production chunk plan, then the final descent. It records the GPU time of each command buffer and the planned time of its sweeps. The ratio column divides the measured time of a chunk by its planned time. The test fails when a command buffer measures 400 ms or more.

## Measurements

At a factor of 1.0, the plan was too short at every anneal count. The ratio grew with the anneal count, because the SA curve credits more throughput between 128 and 256 threadgroups than the cluster kernel gets.

| Anneals | Factor | Chunks | Sweeps per chunk | Longest chunk, ms | Largest ratio |
| --- | --- | --- | --- | --- | --- |
| 8 | 1.0 | 2 | 3,058 | 695.7 | 1.498 |
| 32 | 1.0 | 3 | 1,658 | 620.5 | 1.241 |
| 64 | 1.0 | 3 | 1,400 | 748.2 | 1.497 |
| 128 | 1.0 | 5 | 954 | 1,179.1 | 2.360 |
| 256 | 1.0 | 7 | 610 | 1,509.2 | 3.125 |

A ratio of 3.125 needs a factor of at least 3.9 to keep a 500 ms plan below 400 ms. Two runs at a factor of 5.0 measured 465.1 ms and 344.7 ms at 256 anneals, with load averages of 10.9 and 20.0. The 465.1 ms chunk passes the limit, so a factor of 5.0 does not leave enough margin under this load.

Three runs at a factor of 7.0 gave the following longest chunks, in milliseconds. The one-minute load averages at the start of the runs were 13.86, 12.83, and 11.19.

| Anneals | Sweeps per chunk | Run 1 | Run 2 | Run 3 |
| --- | --- | --- | --- | --- |
| 1 | 436 | 108.9 | 101.7 | 102.4 |
| 8 | 436 | 107.2 | 107.2 | 113.7 |
| 32 | 236 | 87.8 | 83.9 | 106.4 |
| 64 | 200 | 120.6 | 112.5 | 118.1 |
| 128 | 136 | 304.8 | 189.3 | 198.4 |
| 256 | 87 | 248.7 | 265.9 | 283.5 |

Every greedy pass of the final descent took less than 1 ms. Each run converged in 3 passes.

## Limits of this result

- The factor covers 1 to 256 anneals at 4,577 nodes. Below 64 anneals, chunks run about 100 ms, so a run uses about four times as many command buffers as a quiet GPU needs. The planner has one factor and has no occupancy curve of its own for the cluster kernel.
- The measurement ran on one M4 Max. The SA curve scales with the GPU core count, but this report has no measurement from another GPU.
- The load average reports CPU load. GPU work from other agents changed the longest chunk at 128 anneals from 189.3 ms to 304.8 ms between runs.
