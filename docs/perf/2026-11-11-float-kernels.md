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
