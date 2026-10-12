# Engine selection

All three binaries accept these keys in the `backend_toml` sent by the coordinator. Both keys default to `true`.

```toml
enable_ane = true
enable_metal = true
```

| Binary | Effect |
| --- | --- |
| `quip-metal-msa` | Can use Metal, ANE, or both for separate jobs |
| `quip-metal-sa` | Uses Metal. `enable_ane` has no effect on jobs. |
| `quip-metal-gibbs` | Uses Metal. `enable_ane` has no effect on jobs. |

To use only Metal, set `enable_ane = false`. To use only ANE for MSA, set `enable_metal = false`.
Disabling Metal for SA or Gibbs is invalid because those algorithms have no ANE engine. Disabling both engines is also invalid.

The miner keeps one coordinator connection and one identity. Each job goes to one engine and returns at most one result.
For one `--solve` job, Metal takes priority when it can run the job. ANE handles jobs that cannot use Metal.
Metal keeps its batched stream. The router reserves at most one job for ANE.
A bounded queue lets ready Metal work pass a job that must wait for ANE.

Engines open on first use. An engine set to off stays closed. Metal also leaves its load probe off.
The miner reads the settings when it assigns each job. A later change does not move a job already assigned to an engine.
The existing `utilization` and `yielding` keys control Metal, including when it opens after the first config.

ANE accepts unit values from `{-1,0,1}` and at most 128 reads. Its limits are 16,384 nodes, 163,840 edges, and nonzero degree 20.
The ANE worker also checks its program and memory limits. Eligible jobs can still fail those later checks.
A job outside ANE limits can use Metal if the settings permit Metal and the job fits its limits. Jobs outside every enabled engine return a capacity error.
`quip-metal-msa` advertises Metal's 6,016-node cap, so the core refuses a larger model before it reaches either engine. ANE's own node limit applies only below that cap.

Invalid TOML or an invalid engine selection closes admission until valid settings arrive. The first affected job returns a device fault with a config reason.
The core `apply_config` interface returns no status, so the miner cannot reject the `Configure` frame itself.
A device fault follows the existing session error path. It can end the session before a replacement config arrives.

`--capabilities` opens neither engine. For MSA, `--check` tests both default engines. For SA and Gibbs, it tests Metal.
The MSA binary contains its own hidden ANE worker entry point. It needs no separate worker binary on the path.

## Measured workload policy

For finite batches of 300 Advantage2 jobs at 64 reads and 14,336 sweeps,
use `enable_metal = true` and `enable_ane = false`. Four balanced rounds
produced 4.609 valid proofs per second on Metal alone and 3.811 with both
engines. Each window includes queue drain. The combined runs sent 13 of
1200 jobs to ANE.

This policy applies to the measured batch mix. It does not establish a gain
for an endless stream or a different mix. The engine flags still default to
true. If a workload needs ANE, the router keeps its one-job slot and Metal
batching. See the [study report](perf/2026-09-21-coloring-router.md) for
uncertainty, per-window results, and limits.
