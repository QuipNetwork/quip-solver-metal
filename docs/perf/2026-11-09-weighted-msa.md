# Weighted MSA measurements, 2026-11-09

Weighted multi-spin annealing (MSA) takes 3.3 times the unit kernel's time per job on the Advantage2 ±1.5 jobs. The unit MSA kernel returns the same reads as the committed kernel on ±1 jobs. `WEIGHTED_MSA_COST_FACTOR` in `src/sampler.rs` is 3.5, the measured ratio rounded up.

## Host and load

The host was an Apple M4 Max with 16 logical CPUs and 128 GiB of memory. No miner, other benchmark, or build ran during the timed runs. These runs did not record `uptime`.

## Commands and method

The baseline is a `git archive` export of `8034ac0`, the branch head before this change, with the new `tests/pm1_rates.rs` copied in. The runs were sequential in one supervised script. Each result covers eight generated jobs on the 4,577-node, 41,515-edge Advantage2 topology, with 64 reads and 4,096 sweeps per job. `median_s` is the upper middle of the eight sorted job times. The digest is the `FNV-1a` hash of every read, described in `2026-10-11-wide-path.md`.

```sh
git archive 8034ac0 | tar -x -C base && cp tests/pm1_rates.rs base/tests/
(cd base && cargo test --release --test pm1_rates -- --ignored --nocapture)
cargo test --release --test pm1_rates -- --ignored --nocapture
QUIP_PM1_SCALE=wide cargo test --release --test pm1_rates -- --ignored --nocapture
(cd base && cargo test --release --test pm1_rates -- --ignored --nocapture)
```

## Raw measurements

Baseline ±1, first run:

```json
{"kernel":"Sa","scale":1,"median_s":30.8220,"mean_best_milli":-14954500.0,"digest":"e4e22f86b64cda5d"}
{"kernel":"Gibbs","scale":1,"median_s":0.6891,"mean_best_milli":-14952750.0,"digest":"bc085a7fc384f073"}
{"kernel":"Msa","scale":1,"median_s":0.1755,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
```

Branch ±1:

```json
{"kernel":"Sa","scale":1,"median_s":30.8377,"mean_best_milli":-14954500.0,"digest":"e4e22f86b64cda5d"}
{"kernel":"Gibbs","scale":1,"median_s":0.6724,"mean_best_milli":-14952750.0,"digest":"bc085a7fc384f073"}
{"kernel":"Msa","scale":1,"median_s":0.1690,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
```

Branch ±1.5:

```json
{"kernel":"Sa","scale":1.5,"median_s":39.6513,"mean_best_milli":-22436250.0,"digest":"ca3c528c95147c7d"}
{"kernel":"Gibbs","scale":1.5,"median_s":1.2796,"mean_best_milli":-22423500.0,"digest":"e9069459b5895d15"}
{"kernel":"Msa","scale":1.5,"median_s":0.5639,"mean_best_milli":-22436250.0,"digest":"4ee84dd42f5d5ade"}
```

Baseline ±1, second run:

```json
{"kernel":"Sa","scale":1,"median_s":30.6071,"mean_best_milli":-14954500.0,"digest":"e4e22f86b64cda5d"}
{"kernel":"Gibbs","scale":1,"median_s":0.5720,"mean_best_milli":-14952750.0,"digest":"bc085a7fc384f073"}
{"kernel":"Msa","scale":1,"median_s":0.1636,"mean_best_milli":-14961000.0,"digest":"6ec8d35ff7c9f123"}
```

## Results

| Check | Result |
| --- | --- |
| ±1 reads, all three kernels | Digests equal to the baseline |
| ±1 MSA time | 0.1690 s, baseline 0.1636 s and 0.1755 s |
| Weighted MSA time over unit MSA | 0.5639 / 0.1690 = 3.34 |
| Weighted MSA time over wide SA | 0.5639 / 39.6513 = 1 / 70 |
| Weighted MSA mean best | −22,436,250 milli, equal to wide SA |

The weighted ratio uses wall time, which includes host scoring of each weighted read, so the GPU share of the ratio is at most 3.34. The unit kernel's mean best, scaled by 1.5, is −22,441,500 milli. The weighted mean best is 5,250 milli higher, a difference of 0.02 percent.
