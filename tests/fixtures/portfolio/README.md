# Portfolio ground-state fixtures

These 108 fixtures are synthetic factor draws: 24 Family B models and 84 Family B2 models.
Each graph has 32 to 56 spins. Four draws omit two or three edges from the full graph.
File names keep the family and draw names.

The source files are `model.npz` files from these paths:

- `/home/carback1/quip-data/regimes/round3/portfolio-family-b/fixtures/draw*/`
- `/home/carback1/quip-data/regimes/round3/portfolio-family-b2/fixtures/*/`

`tools/npz_to_qmodel.py` keeps the f64 fields, couplings, and offset.
The energy convention is
`E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset`, with `s_i` in `{-1, +1}`.
Each edge counts once. The sampler returns energy without the offset.
The tests add the offset to match the ground energy.

`ground_states.json` records `file`, `n`, `energy`, `gap`, `status`, and `seconds` for each model.
Only records with `status == "optimal"` and `gap == 0` provide verified ground energies.
The tests skip all other records. The manifest keeps all results.
The SCIP solver uses floats. Its proof uses the default solver error bounds.

The solver uses binary bits with `s_i = 2 x_i - 1`.
For each edge, it replaces `x_i x_j` with `y_ij`, subject to
`0 <= y_ij <= 1`, `y_ij <= x_i`, `y_ij <= x_j`, and `y_ij >= x_i + x_j - 1`.
It seeks the least Ising energy plus the offset.
It adds no rules from the source problem. Each solve uses one thread and a 1,800-second time limit.
The batch runs at most 24 solves at once. It saves each result in a separate path.

Check the solver with all spin states of three random 16-spin models:

```sh
uv run --no-project --with pyscipopt --with numpy python -I tools/qubo_ground_states.py --self-test
```

Convert each source fixture on the host:

```sh
uv run --no-project --with numpy python -I tools/npz_to_qmodel.py IN OUT
```

Solve the source fixtures on the host from the root of this repo:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-project --with pyscipopt --with numpy \
  python -I tools/qubo_ground_states.py --workers 24 --time-limit 1800 \
  --partial-dir /tmp/portfolio-ground-partials \
  /home/carback1/quip-data/regimes/round3/portfolio-family-b/fixtures/draw*/model.npz \
  /home/carback1/quip-data/regimes/round3/portfolio-family-b2/fixtures/*/model.npz
```

The script writes the manifest once all solves finish. A restart loads saved results with the same source hash and time limit.
Supervise the batch with taskd.

The GPU tests use 128 reads, 8,192 sweeps, 64 beta steps, and seed 42 for every proven model.
`tests/real_models.rs` runs the SA, Gibbs, and MSA kernels on the automatic path.
Each node has at most 55 edges, so that path uses sparse storage.
`tests/sa_dense.rs`, `tests/gibbs_dense.rs`, and `tests/msa_dense.rs` force dense storage and check the storage tier.
The energy tolerance is `1e-6 * max(1, abs(proven_energy))`.

Run the tests on a Mac with a Metal device:

```sh
cargo test --test real_models --test sa_dense --test gibbs_dense --test msa_dense -- --test-threads=1
cargo test --features long-tests --test real_models --test sa_dense --test gibbs_dense --test msa_dense -- --test-threads=1
```

The second command adds the later SA and Gibbs fixtures, which take minutes.
The later MSA fixtures carry a plain `#[ignore]`, and `long-tests` does not run them.
Run them with this command:

```sh
cargo test --release --test msa_dense every_later_portfolio_fixture_runs_dense -- --ignored --test-threads=1
```
