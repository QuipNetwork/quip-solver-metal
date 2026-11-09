# Probe cascade (`quip-metal-msa`)

The multi-spin miner keeps jobs that pass the screen in GPU slots through the
full sweep budget. It screens each job at short probe checkpoints. A job that
fails the screen returns its probe reads. A kept job continues from the same
slot state to the next checkpoint.

Metal MSA streams always use the resident runner. No key turns it off. An old
`cascade` key logs the unknown-field warning. The simulated annealing and
Gibbs binaries do not run the cascade. The Apple Neural Engine sampler runs
full-budget jobs on a separate path.

Jobs that are not unit models (couplings and fields other than −1, 0, and 1)
use the batch path, on the weighted MSA kernel. The runner
drains live slots, then runs each such job once at full budget. The batch path
rescores its reads on the host and returns one result. It does not screen
these jobs.

## Resident slots

Two slot pools take turns to keep job state on the GPU. Each slot holds one
job, including its coefficients, schedule, spin state, and random state. The
runner puts each ready job in a free slot. It steps live jobs through each
segment. A kept job keeps its state between checkpoints. When a job ends or
fails the screen, the runner decodes its reads and frees the slot.

Four preparation workers check jobs, prepare coefficients and topology, and
build schedules. Each worker caches the graph on the host and owns its queue.
The queues cap the number of jobs that wait for this work. The runner takes
ready jobs in input order and owns all Metal objects. Read decoding uses a
separate worker pool.

The runner drains both pools before a change in topology or read count. A job
that needs more schedule space also drains both pools before replacement. A
kept job stays in its slot.

## Segment schedule

`cascade_stages` gives the sweep budget of each probe checkpoint. Each probe
must fall below the job's sweep budget. The final checkpoint is the full sweep
budget. The runner advances each slot by at most the first stage's sweep count
per step, stopping at the next checkpoint.

Checkpoints of up to 256 sweeps share one anneal. The first runs the
standard schedule from the hot beta to the cold beta. Each later one reheats
the same spins to beta 0.25 and cools them to the cold beta again, so its
checkpoint falls at its stage's sweep count. Every longer checkpoint, the
full budget included, is a fresh anneal of its whole budget. A job kept for
it starts over from new random spins and a new random stream, and runs the
standard schedule over those sweeps. With stages `[32, 256]` and a budget of
1,000 sweeps, a job that passes both probes runs one 256-sweep anneal. That
anneal reheats at 32 sweeps. The job then runs a fresh 1,000-sweep anneal. The schedule computes one beta
range per job. If beta 0.25 falls outside it, the shared anneal follows one
standard schedule instead.

## Controller lifetime

The controller sets stage cutoffs from probe energies, audit results, and
load. About once per second, the runner supplies GPU busy time and live slot
counts per stage. Those counts and spare GPU time feed the cutoff updates.

The sampler saves the controller when a stream ends. The next stream on the
same sampler uses that state for as long as the process runs. The state stays
in memory. No file holds that state. A topology change resets the controller.
New stages or keep values reset the stage plan. Audit and yield settings have
their own update rules. Jobs already in slots keep their admitted plans.

## Keys

These keys enter through `backend_toml` from the coordinator. If stages, keep
values, audit values, or yield values fail their checks, the old values stay
in place. A warning names the failed check. The [calibration
report](perf/2026-09-23-cascade-calibration.md) explains the defaults.

| key | default | valid range | notes |
|-----|---------|-------------|-------|
| `cascade_stages` | `[32, 256]` | 1 to 3 increasing positive integers | sweep budget of each probe anneal |
| `cascade_keep` | `2000` | unsigned 32-bit integer, subject to group rule | probe-to-full denominator |
| `cascade_keep_min` | `1000` | unsigned 32-bit integer, subject to group rule | floor for the probe-to-full keep |
| `cascade_keep_max` | `30000` | unsigned 32-bit integer, subject to group rule | ceiling for the probe-to-full keep |
| `cascade_audit` | `200` | unsigned 32-bit integer, at least 2 | audit lane denominator |
| `cascade_yield_per_million` | none | finite and greater than 0 | expected nonces per million below target |
| `cascade_screen` | `true` | boolean | screen chain-gated lease salts eight models to a word |
| `cascade_prefilter` | `true` | boolean | reject chain lease salts on the frustration score (stage F) before the screen |
| `cascade_open_gates` | `false` | boolean | keep every job at every checkpoint, for gate measurement only |

The keep values pass as one group when `2 <= cascade_keep_min <= cascade_keep
<= cascade_keep_max`. If this check fails, the whole group keeps its old
values.

The keep values are probe-to-full denominators for the whole job. The
controller splits the keep factor across the probe stages. Each of `S` stages
uses the `S`-th root of that denominator. At the defaults, the stage keep
fraction is `1 / 2000^(1/2)`, about 1 in 44.7 per stage. The two default
stages aim to keep about 1 in 2,000 nonces through the full budget. Audits and
load changes can change that rate.

The target energy comes from the session target that the coordinator sets,
not from a key. The yield check runs only when a lease carries that target and
the coordinator sets `cascade_yield_per_million`. It checks target hits from
new lease units against the set yield.

## Screened-out results

A screened-out job returns one result with its probe reads. Its device time is
the sum of command time shares for all steps it runs. These shares include GPU
work and exclude host preparation.

`SamplerMeta.sweeps` reports the original requested budget. `quip-solver-core`
has no `sweeps_done` field, so screened-out jobs report more sweeps than they
run.

Under a salt lease, a screened-out salt sends nothing to the coordinator. The
session counts it in the lease's `LeaseDone` summary, and its probe energies
can set that summary's best energy.

## Model checks

Model checks let more jobs through when results differ from the model. Each
check waits for enough data before it acts. A `cascade model check` warning
names the stage, check, and action. It lists the measured and expected values.
The new keep and audit values appear on that line too.

- **Audit misses.** An audit lets a rejected probe run one level deeper.
  A miss means its next-stage best reaches or beats the kept jobs' median.
  Too many misses over time loosen the screen and raise the audit rate.
- **Distribution drift.** The stage-0 energy distribution has calibrated
  skew and excess kurtosis. A shift beyond 0.1 with enough data
  loosens the screen and doubles `k0`.
- **Yield.** The check compares target hits with a one-sided Poisson lower
  bound. Hits below that bound for a full window loosen the screen and
  raise the audit rate.
