# Dense matrix Gibbs measurements, 2026-11-12

The dense-matrix Gibbs kernel, `kernels/gibbs_dense.metal`, reads one row of an `N * N` coupling matrix each time it updates a spin. It returns the same reads as the sparse chromatic Gibbs kernel for the same seed, at every precision. Every rate in this document uses one unit, the column walk. A column walk is the read of one matrix column during one spin update. A sweep of an `N`-node model makes `N * N` column walks, and that count does not depend on how many spins change.

The planner and the one-sweep admission limit use one constant, `GIBBS_MATRIX_COLUMNS_PER_SEC = 7.0e6`. The constant is a per-chunk worst case. The slowest single chunk measured was one sweep of a Float32 model of 1,024 nodes, 135.0 ms, which is 7.8 million column walks per second. The constant is 10 percent under that rate. The tables below show per-run averages, which are about five times higher, because most chunks run without interference. The slow chunks occurred on a shared host as single spikes, and their cause remains unknown.

A chunk holds at least one sweep, so the admission refuses a dense-matrix Gibbs model whose one sweep the planner estimates at more than 400 ms. That limit is 1,673 nodes at every precision and read count. At the limit, the longest measured chunk was 231.1 ms (Int8 model, one sweep). Where a chunk held more than one sweep, the longest measured chunk was 50.6 ms, at 64 nodes.

## Host and load

The host was an Apple M4 Max with 40 GPU cores. The runs took place on 2026-11-12 between 04:10 and 04:45 local time. Other agents and a node manager process used the host during the runs. The one-minute load average was between 3.6 and 6.5 around the runs.

## Commands and method

```sh
cargo test --release --test gibbs_dense dense_rates -- --ignored --nocapture
cargo test --release --test gibbs_dense the_largest_admitted_dense_sweep_runs -- --ignored --nocapture
cargo test --release --test gibbs_dense sparse_rates -- --ignored --nocapture
cargo test --release --test gibbs_dense sparse_high_degree_chunk -- --ignored --nocapture
```

`dense_rates` in `tests/gibbs_dense.rs` runs complete models with a planted ground state at each precision. Every run has 256 reads, the batch path's read cap (`MAX_READS`). The test checks that all 256 reads come back. It uses the production chunk plan and the default beta schedule, and it records the total GPU time, the chunk count, and the longest chunk. The "Rate" column divides `N * N * sweeps` by the total GPU time. The "Worst chunk rate" column divides the column walks of the largest chunk of the plan by the GPU time of the longest chunk.

The test also asserts that the longest chunk is 250 ms or less whenever every chunk of a shape holds more than one sweep. The planner aims at 200 ms on the slowest shape for such chunks.

## Rates and chunks at the planning constant

Rates are in millions of column walks per second.

| Precision | Nodes | Sweeps | GPU ms | Rate | Chunks | Longest chunk ms | Worst chunk rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Binary | 64 | 16,384 | 1,723.8 | 38.9 | 64 | 47.3 | 22.2 |
| Binary | 256 | 4,096 | 6,225.5 | 43.1 | 256 | 37.8 | 36.4 |
| Binary | 1,024 | 1,024 | 26,022.4 | 41.3 | 1,024 | 91.8 | 11.4 |
| Binary | 1,673 | 78 | 5,461.9 | 40.0 | 78 | 75.1 | 37.3 |
| Int8 | 64 | 16,384 | 1,908.6 | 35.2 | 64 | 50.6 | 20.7 |
| Int8 | 256 | 4,096 | 7,000.6 | 38.3 | 256 | 38.3 | 35.9 |
| Int8 | 1,024 | 1,024 | 27,419.1 | 39.2 | 1,024 | 29.6 | 35.5 |
| Int8 | 1,673 | 78 | 6,171.7 | 35.4 | 78 | 231.1 | 12.1 |
| Int32 | 64 | 16,384 | 1,675.1 | 40.1 | 64 | 46.1 | 22.7 |
| Int32 | 256 | 4,096 | 6,304.5 | 42.6 | 256 | 34.5 | 39.9 |
| Int32 | 1,024 | 1,024 | 28,509.4 | 37.7 | 1,024 | 34.1 | 30.8 |
| Int32 | 1,673 | 78 | 6,557.3 | 33.3 | 78 | 94.3 | 29.7 |
| Float32 | 64 | 16,384 | 1,288.5 | 52.1 | 64 | 40.3 | 26.0 |
| Float32 | 256 | 4,096 | 4,646.5 | 57.8 | 256 | 27.8 | 49.5 |
| Float32 | 1,024 | 1,024 | 25,008.2 | 42.9 | 1,024 | 135.0 | 7.8 |
| Float32 | 1,673 | 78 | 5,059.8 | 43.1 | 78 | 87.3 | 32.1 |

The shapes of 64 and 256 nodes hold 256 and 16 sweeps per chunk. The shapes of 1,024 and 1,673 nodes hold one sweep per chunk, so their worst chunk rate includes the fixed cost of one dispatch. A first run, before the constant was final, with a placeholder of 5.0e6 measured the same slow spikes: one sweep of an Int32 model of 1,024 nodes took 126.3 ms (8.3 million column walks per second). The two runs agree that the slowest chunk is about 8 million column walks per second.

At the limit, a run of an Int32 model of 1,673 nodes with 256 reads and two sweeps in `the_largest_admitted_dense_sweep_runs` ran two chunks of one sweep each. The longer chunk took 86.8 ms.

## One-sweep limit

The admission estimates one sweep as `N * N` column walks at 7.0e6 column walks per second, without the planning margin. Every threadgroup walks its rows at once, so the estimate does not depend on the read count. The estimate passes 400 ms past 1,673 nodes (1,673 nodes give 399.8 ms and 1,674 nodes give 400.325 ms). The admission refuses a model past the limit with `SampleError::TooLarge`, which the harness reports as `Capacity`, before any buffer exists. Auto routing and forced dense both refuse it. The sparse build is not a fallback. The unit test `a_dense_gibbs_sweep_past_one_chunk_is_refused_before_allocation` in `src/sampler.rs` checks both sides of the limit, and `a_model_past_the_sweep_limit_is_refused_under_auto_and_forced_dense` in `tests/gibbs_dense.rs` runs the refusal through the sampler.

The margin of the slowest chunk at the limit (231.1 ms) under 400 ms is 42 percent. It covers the spikes of the shared host, and it does not cover every load.

The limit is far below the 4,800 spins of the threadgroup array. The dense kernel keeps every spin in that array and has no spill build. A compile-time assertion next to the constant keeps the limit below the array.

## Color classes

A complete model has one node in each color class, so the kernel runs `N` color classes per sweep. One thread of each threadgroup walks the row of the node of a class, and the other threads wait at the barrier that closes the class. The kernel keeps the sparse kernel's threadgroup width and update order, so the reads match the sparse reads bit for bit. It also keeps the order of the additions of a row. The field starts from `h` and adds each column in ascending order, as the sparse walk adds the neighbors of a row. The test `the_dense_path_matches_the_sparse_path` pins this with a Float32 model whose fields round differently in another order. With the column loop of the kernel reversed, that test fails on that model. With the loop in ascending order, it passes.

The host builds the color classes with the sparse build's greedy coloring, from a transient CSR that it drops before upload. The coloring and the upload of a complete model of 1,673 nodes finished inside the 0.96 s of `the_largest_admitted_dense_sweep_runs`. The capacity model prices the color start and count arrays (`2 * N * 4` bytes) as shared buffers for the dense tier of Gibbs, as it does for the multi-spin kernel. The test `the_dense_tier_prices_the_colour_arrays_of_the_colour_kernels` in `src/capacity.rs` checks the sizes.

## Matrix pipelines

The sampler builds four dense Gibbs pipelines, one for each matrix element that a Gibbs model can reach:

- `gibbs_matrix_i8` holds an `int8_t` matrix of whole units, and its field is `float`. Unit models and whole-unit models share it. The Int8 tier covers a unit model for Gibbs (`sa_tier`), so no bit-matrix pipeline exists.
- `gibbs_matrix_wide_i8`: an `int8_t` matrix times one integer scale, with the wide build's integer field.
- `gibbs_matrix_int`: a milli `int` matrix.
- `gibbs_matrix_float`: a `float` matrix.

## Dense against sparse

The sparse Gibbs build ran the same complete models with the same seed. The sparse runs used `Tuning::force_sparse`, four sweeps, and one chunk. The command is the `sparse_rates` line in the commands section. Rates are in millions of column walks per second.

| Precision | Nodes | Sparse GPU ms | Sparse rate |
| --- | --- | --- | --- |
| Int8 | 1,024 | 97.4 | 43.1 |
| Int8 | 1,673 | 258.9 | 43.2 |
| Int32 | 1,024 | 92.4 | 45.4 |
| Int32 | 1,673 | 264.6 | 42.3 |

The dense and sparse rates agree within the spread between runs. Both builds update a node with a single thread walking one row, and memory latency limits both, so the dense kernel does not make a complete model faster. For an Int8 model it stores `N * N` bytes instead of the CSR arrays, which is smaller when the model has more than about a fifth of the possible couplers.

## Sparse Gibbs planner and high degree

Bead `quip-miner-metal-9sp` records that the sparse SA planner has no term for the degree of the model. The sparse Gibbs planner has the same gap: it plans in spin updates and does not count the columns of a row. One planned run of a complete model of 1,024 nodes with 256 reads, forced onto sparse storage (`sparse_high_degree_chunk`), put every sweep in one chunk. The longest chunk was 1,600.6 ms for 64 sweeps and 3,193.7 ms for 128 sweeps, which is about 25 ms per sweep. The planner budget at 500 ms and 0.84 billion updates per second is 4.2e8 updates, and one sweep of 256 reads is 262,144 updates, so the planner would put up to 1,602 sweeps in one chunk. At 25 ms per sweep, that chunk would take about 40 s. That figure is an extrapolation, and this change leaves the sparse planner unchanged.

## Effect on routing

The capacity model routes a Gibbs model to dense storage when the dense matrix takes fewer bytes than the CSR arrays. A routed model of more than 1,673 nodes now gets a `Capacity` refusal under Auto and under forced dense.

## Digests

The pm1 digests come from the commands below, with `QUIP_PM1_SCALE=wide` and `QUIP_PM1_SCALE=float` for the weighted and Float32 jobs. Every digest equals its value before this change.

```sh
cargo test --release --test pm1_rates -- --ignored --nocapture
QUIP_PM1_SCALE=wide cargo test --release --test pm1_rates -- --ignored --nocapture
QUIP_PM1_SCALE=float cargo test --release --test pm1_rates -- --ignored --nocapture
```

| Job set | Digest |
| --- | --- |
| Unit SA | `e4e22f86b64cda5d` |
| Unit Gibbs | `bc085a7fc384f073` |
| Wide Gibbs | `e9069459b5895d15` |
| Float32 Gibbs | `eabc348f9005570c` |
| Unit multi-spin | `6ec8d35ff7c9f123` |
| Weighted multi-spin | `4ee84dd42f5d5ade` |
| Float32 multi-spin | `c103cce43c429e4b` |
