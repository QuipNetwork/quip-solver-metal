// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// DENSE-MATRIX SIMULATED ANNEALING
// ==============================================================================
// The host appends this file to kernels/sa.metal, so it reuses that file's
// typedefs, `xorshift32`, and packed-spin helpers, and it always prepends
// `#define QUIP_SA_GLOBAL 1`: each read keeps its packed spins and
// `delta_energy` in the persistent buffers, interleaved by thread. Each
// update reads one row of an N * N coupling matrix instead of a CSR row, so
// the kernel holds a complete or near-complete model of any degree.
//
// Matrix element per build, row-major, one row per node:
// - QUIP_SA_MATRIX_BITS (with the default int8 build): two bit planes per 32
//   columns, the words (negative, nonzero) for columns 32b .. 32b + 31, so a
//   row is 2 * ceil(N / 32) contiguous `uint`. A unit model only.
// - QUIP_WIDE_I8: `int8_t` times the integer `coef_scale` (buffer 23).
// - QUIP_FLOAT: `float`.
// - QUIP_WIDE: milli `int`.
// - otherwise: `int8_t`.
// The diagonal is zero: the host routes a model with a self-loop or a
// repeated coupler to CSR storage.
//
// The reads equal the sparse kernel's for the same seed, bit for bit:
// - Each thread is one read with the sparse kernel's seed, `xorshift32`
//   stream, sequential spin order, and acceptance rule, and it keeps a
//   `delta_energy` per spin. A flip updates the delta of every neighbour
//   of the row, an O(N) walk that adds to each delta exactly the term the
//   sparse walk adds.
// - The initial field sums `h`, then every nonzero column in ascending
//   order. The CSR rows are sorted ascending, and a zero entry adds nothing,
//   so the Float32 sum takes the CSR walk's order of additions.
//   `the_dense_path_matches_the_sparse_path` (tests/sa_dense.rs) pins it
//   with a model whose fields round differently in another order.
// The host scores every read, so no energy is kept on the device.

#ifdef QUIP_SA_MATRIX_BITS
typedef uint matrix_elem_t;
#else
typedef stored_t matrix_elem_t;
#endif

// The elements a row of the matrix occupies.
inline uint matrix_row_elems(int n) {
#ifdef QUIP_SA_MATRIX_BITS
    return 2u * uint((n + 31) / 32);
#else
    return uint(n);
#endif
}

// The couplings of the eight columns [base, base + 8) of a row, as the
// coefficients the sparse walk reads, zero for an absent coupling. `base` is a
// multiple of 8, so the eight spins of the columns are one byte of the packed
// state. Columns past `n` read as zero.
inline void matrix_group(
    device const matrix_elem_t* row,
    int base,
    int n,
    int coef_scale,
    thread coef_t* coefficients
) {
#ifdef QUIP_SA_MATRIX_BITS
    // Bits [base % 32, base % 32 + 8) of the two words of the block.
    uint shift = uint(base) & 31u;
    uint block = uint(base) >> 5;
    uint present = (row[2u * block + 1u] >> shift) & 0xFFu;
    uint negative = (row[2u * block] >> shift) & 0xFFu;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        coefficients[k] = ((present >> uint(k)) & 1u)
            ? (((negative >> uint(k)) & 1u) ? coef_t(-1) : coef_t(1))
            : coef_t(0);
    }
#else
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        coefficients[k] = (base + k < n) ? COEF(row[base + k], coef_scale) : coef_t(0);
    }
#endif
}

// Delta energy of flipping `var`, from the matrix row of `var`.
inline delta_t matrix_flip_energy(
    int var,
    int n,
    device const int8_t* packed_state,
    uint stride,
    device const matrix_elem_t* row,
    device const stored_t* h_vals,
    int coef_scale
) {
    field_t energy = COEF(h_vals[var], coef_scale);
    for (int base = 0; base < n; base += 8) {
        coef_t coefficients[8];
        matrix_group(row, base, n, coef_scale, coefficients);
        int spin_bits = packed_state[uint(base >> 3) * stride];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (coefficients[k] != 0) {
                int8_t neighbor_spin = ((spin_bits >> k) & 1) ? -1 : 1;
                energy += field_t(neighbor_spin) * field_t(coefficients[k]);
            }
        }
    }
    int8_t var_spin = get_spin_packed(var, packed_state, stride);
    return (delta_t)(field_t(-2 * var_spin) * energy);
}

// Adds the effect of flipping `var`, whose spin was `var_spin`, to the delta
// of every neighbour. The loads of a group of eight columns issue together.
inline void matrix_update_neighbors(
    int n,
    int var_spin,
    device const int8_t* packed_state,
    uint stride,
    device delta_t* delta_energy,
    device const matrix_elem_t* row,
    int coef_scale
) {
    int multiplier = 4 * var_spin;
    for (int base = 0; base < n; base += 8) {
        coef_t coefficients[8];
        matrix_group(row, base, n, coef_scale, coefficients);
        int spin_bits = packed_state[uint(base >> 3) * stride];
        delta_t deltas[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            deltas[k] = (coefficients[k] != 0)
                ? delta_energy[uint(base + k) * stride]
                : delta_t(0);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (coefficients[k] != 0) {
                int8_t neighbor_spin = ((spin_bits >> k) & 1) ? -1 : 1;
#ifdef QUIP_FLOAT
                // 4*J may overflow even when the updated delta fits, so the
                // update rounds once, as the sparse kernel does.
                deltas[k] = fma(float(multiplier * neighbor_spin), coefficients[k], deltas[k]);
#else
                deltas[k] += (delta_t)(multiplier * coefficients[k] * neighbor_spin);
#endif
                delta_energy[uint(base + k) * stride] = deltas[k];
            }
        }
    }
}

// The buffer slots of `pure_simulated_annealing`, with the matrix at buffer 2
// and the CSR slots left unread.
kernel void sa_matrix_anneal(
    device const int* unused_row_ptr [[buffer(0)]],
    device const int* unused_col_ind [[buffer(1)]],
    device const matrix_elem_t* matrix [[buffer(2)]],      // [num_problems * N * row_elems]
    device const int* unused_row_off [[buffer(3)]],
    device const int* unused_col_off [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    constant int& num_threads [[buffer(12)]],
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    device const float* beta_schedule [[buffer(9)]],
    device const stored_t* h_vals [[buffer(15)]],          // [num_problems * N]

    device int8_t* final_samples [[buffer(10)]],           // [num_threads * (N+7)/8] - bit-packed
    device int* unused_energies [[buffer(11)]],            // the host scores every read

    constant int& beta_start [[buffer(16)]],
    constant int& beta_count [[buffer(17)]],
    device int8_t* persistent_state [[buffer(18)]],        // packed spins, interleaved by thread
    device delta_t* persistent_delta_energy [[buffer(19)]], // delta_energy, interleaved by thread
    constant int& sweep_start [[buffer(24)]],
    constant int& sweep_count [[buffer(25)]],
    device uint* persistent_rng [[buffer(20)]],            // [num_threads]
    device int* persistent_energy [[buffer(21)]],          // [num_threads], always 0

#ifdef QUIP_WIDE
    device const int* wide_rows [[buffer(22)]],            // [num_betas * WIDE_ROW] Metropolis thresholds
#endif
#ifdef QUIP_WIDE_I8
    constant int& coef_scale [[buffer(23)]],               // milli value of one stored unit
#endif

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
#ifndef QUIP_WIDE_I8
    const int coef_scale = 1;
#endif
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;
    if (thread_id >= num_threads) {
        return;
    }
    uint problem_id = thread_id / num_reads;

    // The matrix of one problem can pass 2^32 elements across a batch, so
    // its base is computed once in 64 bits; rows inside it index in 32.
    int n = N;
    uint row_elems = matrix_row_elems(n);
    device const matrix_elem_t* my_matrix =
        matrix + ulong(problem_id) * ulong(n) * ulong(row_elems);
    device const stored_t* my_h_vals = &h_vals[ulong(problem_id) * ulong(n)];

    int sweeps_per_beta_val = sweeps_per_beta;
    int packed_size = (n + 7) / 8;
    const uint stride = uint(num_threads);
    device int8_t* packed_state = &persistent_state[thread_id];
    device delta_t* delta_energy = &persistent_delta_energy[thread_id];

    uint rng_state;
    int sweep_end = min(sweep_start + sweep_count, sweeps_per_beta_val);
    if (beta_start == 0 && sweep_start == 0) {
        rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);
        // A zero xorshift32 state is a fixed point; see pure_simulated_annealing.
        if (rng_state == 0u) {
            rng_state = 1u;
        }
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            packed_state[uint(byte_idx) * stride] = 0;
        }
        for (int var = 0; var < n; var++) {
            uint rand_val = xorshift32(rng_state);
            int8_t spin = (rand_val & 1) ? -1 : 1;
            set_spin_packed(var, spin, packed_state, stride);
        }
        for (int var = 0; var < n; var++) {
            delta_energy[uint(var) * stride] = matrix_flip_energy(
                var, n, packed_state, stride,
                my_matrix + uint(var) * row_elems, my_h_vals, coef_scale);
        }
    } else {
        rng_state = persistent_rng[thread_id];
    }

    int chunk_end = min(beta_start + beta_count, num_betas);
    for (int beta_idx = beta_start; beta_idx < chunk_end; beta_idx++) {
#ifdef QUIP_WIDE
        device const int* rung = &wide_rows[beta_idx * WIDE_ROW];
#else
        float beta = beta_schedule[beta_idx];
        // log(1/2^32) is about -22.18, so a delta past 22.18/beta accepts
        // with probability below 2^-32.
        float threshold = 22.18f / beta;
#endif

        for (int sweep = sweep_start; sweep < sweep_end; sweep++) {
            for (int var = 0; var < n; var++) {
                delta_t de = delta_energy[uint(var) * stride];

#ifdef QUIP_WIDE
                bool flip_spin = de <= 0;
                if (!flip_spin) {
                    uint rand_val = xorshift32(rng_state);
                    flip_spin = de <= rung[rand_val & (WIDE_ROW - 1)];
                }
#else
                if (de >= threshold) continue;

                bool flip_spin = false;
                if (de <= 0) {
                    flip_spin = true;
                } else {
                    uint rand_val = xorshift32(rng_state);
                    float prob = exp(-float(de) * beta);
                    float rand_normalized = float(rand_val) / 4294967295.0f;  // 2^32 - 1
                    if (prob > rand_normalized) {
                        flip_spin = true;
                    }
                }
#endif

                if (flip_spin) {
                    int8_t var_spin = get_spin_packed(var, packed_state, stride);
                    matrix_update_neighbors(
                        n, var_spin, packed_state, stride, delta_energy,
                        my_matrix + uint(var) * row_elems, coef_scale);
                    flip_spin_packed(var, packed_state, stride);
                    delta_energy[uint(var) * stride] = -de;
                }
            }
        }
    }

    persistent_rng[thread_id] = rng_state;
    persistent_energy[thread_id] = 0;

    device int8_t* output = &final_samples[ulong(thread_id) * ulong(packed_size)];
    for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
        output[byte_idx] = packed_state[uint(byte_idx) * stride];
    }
}
