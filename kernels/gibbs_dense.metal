// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// DENSE-MATRIX BLOCK GIBBS
// ==============================================================================
// The host appends this file to kernels/gibbs.metal, so it reuses that file's
// typedefs, `xoshiro128**` stream, seeding, and heat-bath helper. Each update
// reads one row of an N * N coupling matrix instead of a CSR row, so the
// kernel holds a complete or near-complete model of any degree. It is the
// chromatic kernel `block_gibbs_parallel` with the row walk replaced:
// one threadgroup per read, the threads of a colour class stride over its
// nodes, and a barrier orders one colour class before the next.
//
// Matrix element per build, row-major, one row per node:
// - QUIP_WIDE_I8: `int8_t` times the integer `coef_scale` (buffer 26).
// - QUIP_FLOAT: `float`.
// - QUIP_WIDE: milli `int`.
// - otherwise: `int8_t` whole units. A unit model runs here too, as the
//   sparse Gibbs kernels hold it.
// The diagonal is zero: the host routes a model with a self-loop or a
// repeated coupler to CSR storage.
//
// The reads equal the sparse chromatic kernel's for the same seed, bit for
// bit:
// - The host builds the colour classes the sparse kernel walks, with the
//   same threadgroup width, so each node is updated by the same thread and
//   draws from the same xoshiro128** stream. A complete model has one node
//   per colour class, so every update is one thread walking one row and the
//   other threads of the group wait at the barrier.
// - The field sums `h`, then every column in ascending order. The CSR rows
//   are sorted ascending, and a zero entry adds nothing, so the Float32 sum
//   takes the CSR walk's order of additions.
//   `the_dense_path_matches_the_sparse_path` (tests/gibbs_dense.rs) pins it
//   with a model whose fields round differently in another order.
// - The heat-bath step, the spin initialisation, and the carry-over of spins
//   and streams between chunks are the sparse kernel's.
// The spins stay in the threadgroup array: the host admits a dense Gibbs model
// only when one sweep fits a command buffer, which bounds N well below the
// 4,800 spins of that array (`GIBBS_MATRIX_COLUMNS_PER_SEC` in
// src/sampler.rs). The host scores every read, so no energy is kept on the
// device. The host binds heat-bath (update mode 0) for every Gibbs dispatch,
// so this kernel has no Metropolis arm.

typedef stored_t matrix_elem_t;

// The field of `var`: `h`, then the coupling of every column times its spin,
// in ascending column order. The loads of a group of eight columns issue
// together; the additions keep their order.
inline field_t matrix_field(
    int var,
    int n,
    threadgroup const int8_t* state,
    device const matrix_elem_t* row,
    device const stored_t* h_vals,
    int coef_scale
) {
    field_t h_eff = field_t(COEF(h_vals[var], coef_scale));
    int column = 0;
    for (; column + 8 <= n; column += 8) {
        coef_t coefficients[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            coefficients[k] = COEF(row[column + k], coef_scale);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            h_eff += field_t(coefficients[k]) * field_t(state[column + k]);
        }
    }
    for (; column < n; ++column) {
        coef_t coupling = COEF(row[column], coef_scale);
        h_eff += field_t(coupling) * field_t(state[column]);
    }
    return h_eff;
}

// The buffer slots of `block_gibbs_parallel`, with the matrix at buffer 2 and
// the CSR slots left unread.
kernel void block_gibbs_matrix(
    device const int* unused_row_ptr [[buffer(0)]],
    device const int* unused_col_ind [[buffer(1)]],
    device const matrix_elem_t* matrix [[buffer(2)]],      // [num_problems * N * N]
    device const int* unused_row_off [[buffer(3)]],
    device const int* unused_col_off [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    device const float* beta_schedule [[buffer(9)]],

    device int8_t* final_samples [[buffer(10)]],           // [num_threadgroups * (N+7)/8] - bit-packed
    device int* unused_energies [[buffer(11)]],            // the host scores every read

    constant int& num_threadgroups [[buffer(12)]],
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    device const stored_t* h_vals [[buffer(15)]],          // [num_problems * N]

    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],
    constant int& unused_update_mode [[buffer(19)]],
    constant int& num_colors [[buffer(20)]],

    constant int& beta_start [[buffer(21)]],
    constant int& beta_count [[buffer(22)]],
    device int8_t* persistent_state [[buffer(23)]],        // [num_threadgroups * N] unpacked spins
    device uint* persistent_rng [[buffer(24)]],            // [num_threadgroups * group_size * 4]

#ifdef QUIP_WIDE
    device const int* wide_rows [[buffer(25)]],            // [num_betas * WIDE_ROW] heat-bath thresholds
#endif
#ifdef QUIP_WIDE_I8
    constant int& coef_scale [[buffer(26)]],               // milli value of one stored unit
#endif

    constant int& sweep_start [[buffer(27)]],
    constant int& sweep_count [[buffer(28)]],

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
#ifndef QUIP_WIDE_I8
    const int coef_scale = 1;
#endif
    uint sample_id = threadgroup_pos.x;
    uint thread_in_group = thread_pos_in_group.x;
    uint group_size = threads_per_group.x;

    if (sample_id >= num_threadgroups) {
        return;
    }
    uint problem_id = sample_id / num_reads;

    // The matrix of one problem can pass 2^32 elements across a batch, so
    // its base is computed once in 64 bits; rows inside it index in 32.
    int n = N;
    device const matrix_elem_t* my_matrix =
        matrix + ulong(problem_id) * ulong(n) * ulong(n);
    device const stored_t* my_h_vals = &h_vals[ulong(problem_id) * ulong(n)];
    int packed_size = (n + 7) / 8;

    // Unpacked spins, one byte each. N stays below 4,800 (see above).
    threadgroup int8_t shared_state[4800];

    int sweep_end = min(sweep_start + sweep_count, sweeps_per_beta);

    RngState rng_state;
    if (beta_start == 0 && sweep_start == 0) {
        rng_state = seed_rng(
            (base_seed ? base_seed : 1u) ^ (sample_id * 2654435761u) ^ (thread_in_group * 2246822519u)
        );
        for (uint var = thread_in_group; var < n; var += group_size) {
            uint rand_val = xoshiro128starstar(rng_state);
            shared_state[var] = (rand_val & 1) ? -1 : 1;
        }
    } else {
        // Threadgroup memory does not survive dispatches, so the spins are
        // rebuilt from device memory every chunk.
        device const uint* src_rng = &persistent_rng[(sample_id * group_size + thread_in_group) * 4];
        rng_state.s0 = src_rng[0];
        rng_state.s1 = src_rng[1];
        rng_state.s2 = src_rng[2];
        rng_state.s3 = src_rng[3];

        device const int8_t* src_state = &persistent_state[sample_id * uint(n)];
        for (uint var = thread_in_group; var < n; var += group_size) {
            shared_state[var] = src_state[var];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int chunk_end = min(beta_start + beta_count, num_betas);
    for (int beta_idx = beta_start; beta_idx < chunk_end; beta_idx++) {
#ifdef QUIP_WIDE
        device const int* rung = &wide_rows[beta_idx * WIDE_ROW];
#else
        float beta = beta_schedule[beta_idx];
#endif

        for (int sweep = sweep_start; sweep < sweep_end; sweep++) {
            for (int color = 0; color < num_colors; color++) {
                int block_start = color_block_starts[color];
                int block_count = color_block_counts[color];

                for (uint i = thread_in_group; i < block_count; i += group_size) {
                    int var = color_node_indices[block_start + i];
                    field_t h_eff = matrix_field(
                        var, n, shared_state,
                        my_matrix + uint(var) * uint(n), my_h_vals, coef_scale);

                    int8_t new_spin;
#ifdef QUIP_WIDE
                    new_spin = gibbs_sample_wide(h_eff, rung, rng_state);
#else
#ifdef QUIP_FLOAT
                    float prob_plus = 1.0f / (1.0f + exp((2.0f * h_eff) * beta));
#else
                    float prob_plus = 1.0f / (1.0f + exp(2.0f * beta * h_eff));
#endif
                    uint rand_val = xoshiro128starstar(rng_state);
                    float rand_normalized = float(rand_val) / 4294967295.0f;
                    new_spin = (rand_normalized < prob_plus) ? 1 : -1;
#endif
                    shared_state[var] = new_spin;
                }

                // Every update of this class lands before the next class reads.
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Persist the spins and the streams for the next chunk. The barrier that
    // closed the last colour class already ordered the spins.
    device uint* dst_rng = &persistent_rng[(sample_id * group_size + thread_in_group) * 4];
    dst_rng[0] = rng_state.s0;
    dst_rng[1] = rng_state.s1;
    dst_rng[2] = rng_state.s2;
    dst_rng[3] = rng_state.s3;

    device int8_t* dst_state = &persistent_state[sample_id * uint(n)];
    for (uint var = thread_in_group; var < n; var += group_size) {
        dst_state[var] = shared_state[var];
    }

    // Each thread packs whole bytes of the output: bit i of a byte is spin i,
    // set for -1.
    device int8_t* output = &final_samples[ulong(sample_id) * ulong(packed_size)];
    for (uint byte_idx = thread_in_group; byte_idx < packed_size; byte_idx += group_size) {
        uchar bits = 0;
        for (uint bit_idx = 0; bit_idx < 8; bit_idx++) {
            uint var = byte_idx * 8 + bit_idx;
            if (var < n && shared_state[var] < 0) {
                bits |= uchar(1 << bit_idx);
            }
        }
        output[byte_idx] = int8_t(bits);
    }
}
