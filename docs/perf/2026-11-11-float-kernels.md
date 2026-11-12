# Float32 kernel measurements, 2026-11-11

Each kernel with a `QUIP_FLOAT` build has a section in this file. Each section gives the cost factor that `chunk_plan` in `src/sampler.rs` uses for the Float32 build. The method is the method of `2026-10-11-wide-path.md`. The same Advantage2 jobs run on the int8 build and on the Float32 build. The factor is the ratio of the two median job times, rounded up to one decimal.

## Simulated annealing

Float32 SA takes 1.24 times the int8 SA time per job on the Advantage2 jobs. `FLOAT_SA_COST_FACTOR` is 1.3, the measured ratio rounded up.

### Host and load

The host was an Apple M4 Max with 16 logical CPUs and 128 GiB of memory (`hw.memsize` 137,438,953,472). Other sessions were active on the host, so other GPU and CPU work possibly ran during the timed runs.

`uptime` immediately before each timed run reported:

| Run | Uptime output |
| --- | --- |
| Int8 ±1 | `18:11  up 20 days,  4:17, 5 users, load averages: 20.91 10.17 9.81` |
| Float32 ±1.0005 | `18:15  up 20 days,  4:21, 5 users, load averages: 5.04 7.00 8.46` |

### Commands and method

The working tree was `c04a06c` with this change applied. The runs were sequential in one supervised script. Each result covers eight generated jobs on the 4,577-node, 41,515-edge Advantage2 topology, with 64 reads and 4,096 sweeps per job. `median_s` is the upper middle of the eight sorted job times. The digest is the `FNV-1a` hash of every read, as `2026-10-11-wide-path.md` describes.

`QUIP_PM1_SCALE=float` multiplies every coefficient of the ±1 jobs by 1.0005. Each coefficient is then off the milli grid, so the classifier selects Float32, and the jobs run on the `QUIP_FLOAT` build of `kernels/sa.metal`. The jobs keep the topology, the signs, and the read and sweep counts of the int8 run.

```sh
cargo test --release --test pm1_rates --no-run
uptime
cargo test --release --test pm1_rates -- --ignored --nocapture
uptime
QUIP_PM1_SCALE=float cargo test --release --test pm1_rates -- --ignored --nocapture
```

### Raw measurements

Int8 ±1, with this change:

```json
{"kernel":"Sa","scale":1,"median_s":30.2229,"mean_best_milli":-14954500.0,"digest":"e4e22f86b64cda5d"}
{"kernel":"Gibbs","scale":1,"median_s":0.3353,"mean_best_milli":-14952750.0,"digest":"bc085a7fc384f073"}
{"kernel":"Msa","scale":1,"median_s":0.1607,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
```

Float32 ±1.0005:

```json
{"kernel":"Sa","scale":1.0005,"median_s":37.6167,"mean_best_milli":-14977712.8,"digest":"04210f9bbe78e5d8"}
```

### Identical int8 reads

In `kernels/sa.metal`, the int8 build now sums the local field from `h` first, then the couplings. Integer sums do not depend on order, so this edit leaves the int8 reads unchanged.

For a check, a detached worktree at `c04a06c`, the commit before this change, repeated the int8 command after the preceding runs:

```sh
git worktree add --detach .worktrees/base-2.3a c04a06c
(cd .worktrees/base-2.3a && cargo test --release --test pm1_rates -- --ignored --nocapture)
```

```json
{"kernel":"Sa","scale":1,"median_s":30.5262,"mean_best_milli":-14954500.0,"digest":"e4e22f86b64cda5d"}
{"kernel":"Gibbs","scale":1,"median_s":0.5194,"mean_best_milli":-14952750.0,"digest":"bc085a7fc384f073"}
{"kernel":"Msa","scale":1,"median_s":0.1606,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
```

All three digests match the digests with this change. The int8 SA reads are identical. The digests differ from the digests in `2026-10-11-wide-path.md` because commit `4a30669` later changed how the host seed becomes the kernel seed.

### Cost ratio

Float32 SA divided by int8 SA is `37.6167 / 30.2229 = 1.2446`. Rounded up to one decimal, this sets `FLOAT_SA_COST_FACTOR` to `1.3`. Like the wide build, the Float32 build keeps a 4-byte delta for each spin and uses the wide dispatch geometry.

The load average before the int8 run was 20.91. If that load slowed the int8 run, the true ratio is higher than 1.2446. The int8 SA time of the base run, 30.5262 seconds, is within 1% of the int8 time of this run. This agreement makes a large error unlikely. With the base time, the ratio is 1.2323, which also rounds up to 1.3.

The mean best energy of the Float32 jobs is not comparable with the int8 value, because every coefficient is 1.0005 times larger.

## Gibbs

Float32 Gibbs takes 1.24 times the int8 Gibbs time per job on the Advantage2 jobs. `FLOAT_GIBBS_COST_FACTOR` is 1.3, the measured ratio rounded up. Both Gibbs entry points have a `QUIP_FLOAT` build: the sequential `block_gibbs_sampler` and the node-parallel `block_gibbs_parallel`, which is the default dispatch. Float32 uses the same dispatch geometry as the wide build in each case.

### Host and load

The host was an Apple M4 Max with 16 logical CPUs and 128 GiB of memory. Other sessions were active on the host, so other GPU and CPU work possibly ran during the timed runs.

`uptime` immediately before each timed run reported:

| Run | Uptime output |
| --- | --- |
| Int8 ±1 | `18:42  up 20 days,  4:49, 5 users, load averages: 13.17 8.04 8.13` |
| Float32 ±1.0005 | `18:47  up 20 days,  4:53, 5 users, load averages: 3.86 5.71 7.07` |

### Commands and method

The method is the method of the Simulated annealing section. The working tree was `0c31d25` with this change applied. The runs were sequential in one supervised script. `QUIP_PM1_SCALE=float` now runs both SA and Gibbs.

```sh
cargo test --release --test pm1_rates --no-run
uptime
cargo test --release --test pm1_rates -- --ignored --nocapture
uptime
QUIP_PM1_SCALE=float cargo test --release --test pm1_rates -- --ignored --nocapture
```

### Raw measurements

Int8 ±1, with this change:

```json
{"kernel":"Sa","scale":1,"median_s":30.1612,"mean_best_milli":-14954500.0,"digest":"e4e22f86b64cda5d"}
{"kernel":"Gibbs","scale":1,"median_s":0.5024,"mean_best_milli":-14952750.0,"digest":"bc085a7fc384f073"}
{"kernel":"Msa","scale":1,"median_s":0.1598,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
```

Float32 ±1.0005:

```json
{"kernel":"Sa","scale":1.0005,"median_s":37.6302,"mean_best_milli":-14977712.8,"digest":"04210f9bbe78e5d8"}
{"kernel":"Gibbs","scale":1.0005,"median_s":0.6207,"mean_best_milli":-14967702.8,"digest":"eabc348f9005570c"}
```

### Identical int8 reads

The int8 digests for SA, Gibbs, and MSA equal the digests that the SA section records for the commit before this change (`e4e22f86b64cda5d`, `bc085a7fc384f073`, `6ec8d35ff7c9f123`). The int8 Gibbs build keeps its arithmetic and its device energy.

### Cost ratio

Float32 Gibbs divided by int8 Gibbs is `0.6207 / 0.5024 = 1.2355`. Rounded up to one decimal, this sets `FLOAT_GIBBS_COST_FACTOR` to `1.3`. A Gibbs job takes about half a second on these jobs, so the ratio is less precise than the SA ratio. The Float32 SA ratio of 1.2446 agrees with it. The mean best energy of the Float32 jobs is not comparable with the int8 value, because every coefficient is 1.0005 times larger.

## Multi-spin

Float32 multi-spin annealing (MSA) takes 1.50 times the time of the weighted MSA lane per job on the same Advantage2 jobs. The weighted lane is the milli `int` build of `msa_anneal_weighted`. `FLOAT_MSA_COST_FACTOR` is 5.3: the measured ratio, 1.50, times `WEIGHTED_MSA_COST_FACTOR`, 3.5, rounded up to one decimal. Like `WEIGHTED_MSA_COST_FACTOR`, the factor is relative to the unit MSA kernel. The Float32 time is 5.05 times the unit time, so the factor is also larger than the direct ratio.

The Float32 build is the `QUIP_FLOAT` build of `msa_anneal_weighted` in `kernels/msa.metal`. It has `float` coefficients, a `float` local field for each lane, and a separate uniform draw for each of the 32 lanes. It has no threshold rows.

### Host and load

The host was an Apple M4 Max with 16 logical CPUs and 128 GiB of memory. Other sessions were active on the host, so other GPU and CPU work possibly ran during the timed runs.

Each script runs the unit, weighted, and Float32 jobs in sequence. `uptime` immediately before each timed run reported:

| Run | Uptime output |
| --- | --- |
| Unit ±1, first run | `19:09  up 20 days,  5:16, 5 users, load averages: 10.90 7.97 9.75` |
| Weighted ±1.5, first run | `19:09  up 20 days,  5:16, 5 users, load averages: 10.90 7.97 9.75` |
| Float32 ±1.0005, first run | `19:09  up 20 days,  5:16, 5 users, load averages: 10.35 7.90 9.71` |
| Unit ±1, second run | `19:10  up 20 days,  5:16, 5 users, load averages: 8.34 7.60 9.56` |
| Weighted ±1.5, second run | `19:10  up 20 days,  5:16, 5 users, load averages: 7.83 7.50 9.52` |
| Float32 ±1.0005, second run | `19:10  up 20 days,  5:16, 5 users, load averages: 7.36 7.41 9.47` |

The load averages before the weighted and the Float32 runs differ by less than 0.6 in each script.

### Commands and method

The working tree was `4f0aabf` with this change applied. The method is the method of the Simulated annealing section, with one addition. `QUIP_PM1_KERNEL=Msa` runs only the MSA kernel. `QUIP_PM1_SCALE=wide` multiplies every coefficient by 1.5, so the jobs run on the weighted lane. `QUIP_PM1_SCALE=float` multiplies every coefficient by 1.0005, so the jobs run on the Float32 build.

```sh
cargo test --release --test pm1_rates --no-run
uptime
QUIP_PM1_KERNEL=Msa cargo test --release --test pm1_rates -- --ignored --nocapture pm1_rates
uptime
QUIP_PM1_SCALE=wide QUIP_PM1_KERNEL=Msa cargo test --release --test pm1_rates -- --ignored --nocapture pm1_rates
uptime
QUIP_PM1_SCALE=float QUIP_PM1_KERNEL=Msa cargo test --release --test pm1_rates -- --ignored --nocapture pm1_rates
```

The script ran twice in sequence.

### Raw measurements

First run:

```json
{"kernel":"Msa","scale":1,"median_s":0.1585,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
{"kernel":"Msa","scale":1.5,"median_s":0.5337,"mean_best_milli":-22436250.0,"digest":"4ee84dd42f5d5ade"}
{"kernel":"Msa","scale":1.0005,"median_s":0.8007,"mean_best_milli":-14971206.2,"digest":"c103cce43c429e4b"}
```

Second run:

```json
{"kernel":"Msa","scale":1,"median_s":0.1585,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
{"kernel":"Msa","scale":1.5,"median_s":0.5342,"mean_best_milli":-22436250.0,"digest":"4ee84dd42f5d5ade"}
{"kernel":"Msa","scale":1.0005,"median_s":0.8008,"mean_best_milli":-14971206.2,"digest":"c103cce43c429e4b"}
```

### Identical unit and weighted reads

The same script ran at `4f0aabf` before the kernel change, with only the `QUIP_PM1_KERNEL` filter added to `tests/pm1_rates.rs`. The load average before its unit run was 8.11. The Float32 run failed there, because the MSA kernel then had no Float32 build.

```json
{"kernel":"Msa","scale":1,"median_s":0.1586,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
{"kernel":"Msa","scale":1.5,"median_s":0.5334,"mean_best_milli":-22436250.0,"digest":"4ee84dd42f5d5ade"}
```

The unit digest `6ec8d35ff7c9f123` and the weighted digest `4ee84dd42f5d5ade` are the same before and after the change. The unit digest also equals the MSA digest of the Simulated annealing section. The two Float32 runs have the same digest, `c103cce43c429e4b`.

### Cost ratio

Float32 MSA divided by weighted MSA is `0.8007 / 0.5337 = 1.5003` in the first run and `0.8008 / 0.5342 = 1.4991` in the second run. The factor is `1.5 * 3.5 = 5.25`, rounded up to `5.3`. Both times are wall time and include the host scoring of every read, because neither lane writes device energies.

The Float32 lane costs more than the weighted lane for two reasons. It takes a logarithm for each lane that moves uphill, where the weighted lane reads one threshold for the whole word. It also keeps 32 `float` fields while it walks the row in chunks of 20 neighbours. The mean best energy of the Float32 jobs is not comparable with the other values, because every coefficient is 1.0005 times larger.

### Rows past one chunk

The Advantage2 jobs that set `FLOAT_MSA_COST_FACTOR` fit each row in one chunk of 20 neighbours, and each color class of those jobs holds about 572 nodes. Two planning terms extend it to other models. Both apply to the Float32 lane only:

- The row term multiplies the factor by the row chunks of the largest degree, `ceil(degree / 20)`, at least one. A degree of 43 gives 3, and a degree of 62 gives 4.
- The color term multiplies it by `steps * 572 / n`, at least one. `steps` is the sum, over the color classes, of `ceil(class nodes / threads per group)`. Each class ends with a barrier, so a class of one node costs a full threadgroup step. A complete model of 44 nodes has 44 classes of one node, so its term is 572.

The row term alone did not hold the ceiling. With the row term and without the color term, the portfolio batch in the next table ran as one command buffer of 3,952.3 ms.

The ignored test `float_msa_high_degree_chunks_stay_under_the_ceiling` in `src/sampler.rs` runs three batches with both terms. Each batch has 128 reads and 8,192 sweeps in 64 rungs, at seed 42:

- `family-b-draw000-factor.qmodel`, a complete model of 44 nodes with degree 43.
- A circulant model of 6,016 nodes with degree 62, alone.
- Eight circulant models in one batch.

The host was the M4 Max of the preceding runs. `uptime` before the run reported `19:28  up 20 days,  5:35, 5 users, load averages: 11.45 6.70 6.59`.

```sh
cargo test --release --lib float_msa_high_degree_chunks --no-run
uptime
cargo test --release --lib float_msa_high_degree_chunks -- --ignored --nocapture
```

```json
{"batch":"portfolio x1","degree":43,"chunks":64,"max_chunk_ms":80.1,"total_gpu_ms":3951.9}
{"batch":"circulant x1","degree":62,"chunks":64,"max_chunk_ms":87.8,"total_gpu_ms":5598.7}
{"batch":"circulant x8","degree":62,"chunks":64,"max_chunk_ms":142.4,"total_gpu_ms":8174.0}
```

Each batch splits into one command buffer per rung, and the largest one takes 142.4 ms, under the 400 ms chunk target and the 500 ms ceiling. The portfolio batch takes the same total GPU time as the batch with one command buffer, 3,951.9 ms against 3,952.3 ms. The plan cannot split a rung, so a single rung of more sweeps than these runs use can still take longer.

## Digests after the sweep change of November 12

QUI-2265 (merge request !33 on main) changes the SA and MSA sweeps. Every sweep of an anneal except the last leaves one node out, so the SA and MSA samples change. The SA and MSA digests in the preceding sections come from the kernels before that change, while the Gibbs kernel and its digest stay the same.

After the merge of `feat/capacity-model` into `feat/precision-tiers`, the `pm1_rates` runs gave these digests:

| Run | Digest before QUI-2265 | Digest after QUI-2265 |
| --- | --- | --- |
| SA, int8 ±1 | `e4e22f86b64cda5d` | `e5ba74a87a6dadad` |
| Gibbs, int8 ±1 | `bc085a7fc384f073` | `bc085a7fc384f073` |
| MSA unit, ±1 | `6ec8d35ff7c9f123` | `a68757003e4eac09` |
| MSA weighted, ±1.5 | `4ee84dd42f5d5ade` | `be4c5e92185633b9` |
| MSA Float32, ±1.0005 | `c103cce43c429e4b` | `af08be780113298d` |

The Float32 SA digest `04210f9bbe78e5d8` also comes from the kernel before the change, and this file does not record a new value for it. The cost factors in this file were not measured again.
