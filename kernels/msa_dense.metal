// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// DENSE-MATRIX MULTI-SPIN SIMULATED ANNEALING
// ==============================================================================
// The host appends this file to kernels/msa.metal, so it reuses that file's
// RNG, geometric draw, float lane uniform, `SplitSpins`, and coefficient
// typedefs. Each update reads one row of an N * N coupling matrix instead of
// a CSR row, so the kernel holds a complete or near-complete model of any
// degree. (The QUIP_MSA_DENSE build of msa.metal is a different thing: CSR
// rows past one chunk of neighbours.)
//
// Matrix element per build, row-major, one row per node:
// - QUIP_MSA_MATRIX_BITS: two bit planes per 32 columns, the words
//   (negative, nonzero) for columns 32b .. 32b + 31, so a row is
//   2 * ceil(N / 32) contiguous `uint`. A unit model only.
// - QUIP_WIDE_I8: `int8_t` times the integer `scale` (buffer 26).
// - QUIP_FLOAT: `float`.
// - otherwise: milli `int`.
// The diagonal is zero: the host routes a model with a self-loop or a
// repeated coupler to CSR storage.
//
// Update order and RNG streams equal the sparse kernels' (`msa_body`), so a
// model both paths hold returns the same reads:
// - One threadgroup per (problem, word), the sparse grid. Its 32 threads are
//   one SIMD group and lane r is replica r of the word.
// - The sparse kernel splits each colour class over `V` threads, the node at
//   class position k going to thread k % V, and every thread draws once per
//   node from its own xoshiro128** stream. Here `V` (buffer 30) is the
//   sparse dispatch width, and the V streams live in this threadgroup's
//   slice of `persistent_rng`. The SIMD group visits the nodes of a class in
//   position order; lane 0 advances stream k % V once per node and
//   broadcasts the draw. Same-colour nodes share no coupling, so visiting
//   them one after another reads the same fields as visiting them at once.
// - Each lane sums its local field in ascending column order: `h`, then
//   every column. The CSR rows are sorted ascending, and a zero entry adds
//   nothing, so the Float32 sum takes the CSR walk's order of additions.
//   `the_dense_path_matches_the_sparse_path` (tests/msa_dense.rs) pins it
//   with a model whose fields round differently in another order.
// - The acceptance rule is the one the sparse kernel applies to the model:
//   the unit kernel's geometric draw (QUIP_MSA_MATRIX_UNIT), or the threshold
//   rows of the weighted lane, or the float lane's per-lane uniform.
// Spin words [0, shared_spins) live in threadgroup memory and the rest in
// `persistent_state`, as `SplitSpins` keeps them. The host scores every read.

#ifdef QUIP_MSA_MATRIX_BITS
typedef uint matrix_elem_t;
#else
typedef weighted_coef_t matrix_elem_t;
#endif

// The words a row of the matrix occupies.
inline uint matrix_row_elems(int n) {
#ifdef QUIP_MSA_MATRIX_BITS
    return 2u * uint((n + 31) / 32);
#else
    return uint(n);
#endif
}

// Lane `lane`'s spins at the 32 columns [32b, 32b + 32), as one word: bit j
// is the spin of column 32b + j (1 meaning -1). Lane j loads the spin word
// of column 32b + j, one coalesced read for the SIMD group, and a butterfly
// of shuffles transposes the 32 x 32 bit block, so lane l ends with bit l of
// every word. Columns past `n` read as +1; their couplings are zero.
template <typename Spins>
inline uint matrix_lane_block(Spins state, int b, int n, uint lane) {
    int column = b * 32 + int(lane);
    uint word = (column < n) ? state.get(column) : 0u;
    // Stage s swaps bit c of row r with bit c ^ s of row r ^ s wherever bit
    // s of r and of c differ; the five stages together are the transpose.
    const uint masks[5] = {0x0000FFFFu, 0x00FF00FFu, 0x0F0F0F0Fu, 0x33333333u, 0x55555555u};
    #pragma unroll
    for (int stage = 0; stage < 5; ++stage) {
        uint s = 16u >> uint(stage);
        uint low = masks[stage];
        uint other = simd_shuffle_xor(word, ushort(s));
        word = ((lane & s) == 0u)
            ? ((word & low) | ((other & low) << s))
            : ((word & ~low) | ((other & ~low) >> s));
    }
    return word;
}

kernel void msa_matrix_anneal(
    device const int* unused_row_ptr [[buffer(0)]],
    device const int* unused_col_ind [[buffer(1)]],
    device const matrix_elem_t* matrix [[buffer(2)]],      // [num_problems * N * row_elems]
    device const int* unused_row_off [[buffer(3)]],
    device const int* unused_col_off [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    device const float* beta_schedule [[buffer(9)]],

    device int8_t* final_samples [[buffer(10)]],           // [num_problems * num_reads * packed_size]
    device int* unused_energies [[buffer(11)]],            // the host scores every read

    constant int& num_threadgroups [[buffer(12)]],         // num_problems * words
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

#ifdef QUIP_MSA_MATRIX_BITS
    device const int8_t* h_vals [[buffer(15)]],            // unit fields
#else
    device const weighted_coef_t* h_vals [[buffer(15)]],
#endif

    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],

    constant int& words [[buffer(19)]],
    constant int& num_colors [[buffer(20)]],

    constant int& beta_start [[buffer(21)]],
    constant int& beta_count [[buffer(22)]],
    device uint* persistent_state [[buffer(23)]],          // [num_threadgroups * N] spin words
    device uint* persistent_rng [[buffer(24)]],            // [num_threadgroups * V * 4]
#if !defined(QUIP_FLOAT) && !defined(QUIP_MSA_MATRIX_UNIT)
    device const int* wide_rows [[buffer(25)]],            // [num_betas * WIDE_ROW] milli thresholds
#endif
#ifdef QUIP_WIDE_I8
    constant int& coef_scale [[buffer(26)]],
#endif
    constant int& shared_spins [[buffer(27)]],
    constant int& sweep_start [[buffer(28)]],
    constant int& sweep_count [[buffer(29)]],
    constant int& virtual_threads [[buffer(30)]],          // V, the sparse dispatch width
    threadgroup uint* state [[threadgroup(0)]],            // [min(N, shared_spins)] spin words

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint tg = threadgroup_pos.x;
    if (tg >= uint(num_threadgroups)) {
        return;
    }
    int n = N;
    uint problem_id = tg / uint(words);
    uint w = tg - problem_id * uint(words);
    int vthreads = virtual_threads;
    int beta_end = min(beta_start + beta_count, num_betas);
    int sweep_end = min(sweep_start + sweep_count, sweeps_per_beta);
    bool fresh = beta_start == 0 && sweep_start == 0;
    bool last = beta_end >= num_betas && sweep_end >= sweeps_per_beta;
    device uint* my_state = &persistent_state[tg * uint(n)];
    device uint* my_rng = &persistent_rng[tg * uint(vthreads) * 4u];
    SplitSpins spins{state, my_state, shared_spins};

    // The matrix of one problem can pass 2^32 elements across a batch, so
    // its base is computed once in 64 bits; rows inside it index in 32.
    uint row_elems = matrix_row_elems(n);
    device const matrix_elem_t* my_matrix =
        matrix + ulong(problem_id) * ulong(n) * ulong(row_elems);
#ifdef QUIP_MSA_MATRIX_BITS
    device const int8_t* my_h = &h_vals[problem_id * uint(n)];
#else
    device const weighted_coef_t* my_h = &h_vals[problem_id * uint(n)];
#endif
#ifdef QUIP_WIDE_I8
    int scale = coef_scale;
#endif

    if (fresh) {
        // The sparse kernel's thread v seeds its stream and draws the words
        // of nodes v, v + V, ...; lanes emulate the V threads in turn.
        uint seed = (base_seed ? base_seed : 1u) ^ (tg * 2654435761u);
        for (int v = int(lane); v < vthreads; v += MSA_LANES) {
            RngState rng = seed_rng(seed ^ (uint(v) * 2246822519u));
            for (int var = v; var < n; var += vthreads) {
                spins.set(var, xoshiro128starstar(rng));
            }
            device uint* dst = &my_rng[uint(v) * 4u];
            dst[0] = rng.s0;
            dst[1] = rng.s1;
            dst[2] = rng.s2;
            dst[3] = rng.s3;
        }
    } else {
        // Threadgroup memory does not survive dispatches; spilled words and
        // the streams already live in device memory.
        for (int var = int(lane); var < spins.shared(n); var += MSA_LANES) {
            spins.set(var, my_state[var]);
        }
    }
    spins.barrier();

    for (int beta_idx = beta_start; beta_idx < beta_end; beta_idx++) {
        float beta = beta_schedule[beta_idx];
        float two_beta = 2.0f * beta;
#if !defined(QUIP_FLOAT) && !defined(QUIP_MSA_MATRIX_UNIT)
        device const int* rung_rows = &wide_rows[uint(beta_idx) * uint(WIDE_ROW)];
#endif
        for (int sweep = sweep_start; sweep < sweep_end; sweep++) {
            for (int color = 0; color < num_colors; color++) {
                int block_start = color_block_starts[color];
                int block_count = color_block_counts[color];
                for (int k = 0; k < block_count; ++k) {
                    int var = color_node_indices[block_start + k];
                    device const matrix_elem_t* row = my_matrix + uint(var) * row_elems;

                    // The draw of stream k % V, the sparse thread that owns
                    // this node. Only lane 0 touches the stream.
                    uint draw = 0u;
                    if (lane == 0u) {
                        device uint* src = &my_rng[uint(k % vthreads) * 4u];
                        RngState rng;
                        rng.s0 = src[0];
                        rng.s1 = src[1];
                        rng.s2 = src[2];
                        rng.s3 = src[3];
                        draw = xoshiro128starstar(rng);
                        src[0] = rng.s0;
                        src[1] = rng.s1;
                        src[2] = rng.s2;
                        src[3] = rng.s3;
                    }
                    draw = simd_broadcast(draw, ushort(0));

                    uint bi = spins.get(var);
                    uint mine = (bi >> lane) & 1u;
                    bool accept;
#ifdef QUIP_MSA_MATRIX_BITS
                    // Satisfied terms: a bond is satisfied where its sign,
                    // this spin, and the neighbour's spin give J s_i s_j < 0.
                    // A field is a bond to a spin pinned at +1.
                    int h = int(my_h[var]);
                    int satisfied = (h != 0) ? int(((h < 0) ? (mine ^ 1u) : mine)) : 0;
                    int d = int(h != 0);
                    uint mine_word = 0u - mine;
                    int blocks = (n + 31) / 32;
                    for (int b = 0; b < blocks; ++b) {
                        uint neg = row[2 * b];
                        uint nz = row[2 * b + 1];
                        uint sj = matrix_lane_block(spins, b, n, lane);
                        satisfied += int(popcount(nz & (neg ^ mine_word ^ sj)));
                        d += int(popcount(nz));
                    }
#ifdef QUIP_MSA_MATRIX_UNIT
                    // The unit kernel's rule: flip where L <= (d + M) / 2.
                    // Only a model of degree at most MSA_UNIT_MAX_DEG takes
                    // this build, so d <= 21 and the draw's cap of
                    // MSA_MAX_FIELD never binds, as in the unit kernel.
                    int m = geometric_draw(draw, two_beta);
                    int limit = (d + m) >> 1;
                    accept = limit >= d || satisfied <= limit;
#else
                    // The weighted lane's rule on milli fields: s_i * field
                    // is d - 2L units, so de = -2 s_i field = 2000 (2L - d).
                    int threshold = rung_rows[draw & uint(WIDE_ROW - 1)];
                    int de = 2000 * (2 * satisfied - d);
                    accept = de <= threshold;
#endif
#else
#ifdef QUIP_WIDE_I8
                    weighted_field_t field = weighted_field_t(my_h[var]) * scale;
#else
                    weighted_field_t field = weighted_field_t(my_h[var]);
#endif
                    int blocks = (n + 31) / 32;
                    for (int b = 0; b < blocks; ++b) {
                        // Lane q loads the coupling of column 32b + q, one
                        // coalesced read, and the block's spins arrive
                        // transposed; the sum then visits the columns in
                        // ascending order.
                        int column = b * 32 + int(lane);
                        weighted_field_t mine_j = (column < n)
                            ? weighted_field_t(row[column]) : weighted_field_t(0);
#ifdef QUIP_WIDE_I8
                        mine_j *= scale;
#endif
                        uint sj = matrix_lane_block(spins, b, n, lane);
                        #pragma unroll
                        for (int q = 0; q < 32; ++q) {
                            weighted_field_t j = simd_shuffle(mine_j, ushort(q));
                            // Spin bit 1 is -1. A missing coupling is zero.
                            field += ((sj >> uint(q)) & 1u) ? -j : j;
                        }
                    }
#ifdef QUIP_FLOAT
                    // The float lane's rule (msa.metal `flip_mask`).
                    float uphill = mine ? field : -field;
                    accept = uphill <= 0.0f;
                    if (!accept) {
                        accept = two_beta * uphill < -log(lane_uniform(draw, int(lane)));
                    }
#else
                    // The weighted lane's rule: de <= one threshold drawn
                    // from the rung's row.
                    int threshold = rung_rows[draw & uint(WIDE_ROW - 1)];
                    int de = mine ? 2 * field : -2 * field;
                    accept = de <= threshold;
#endif
#endif
                    uint mask = simd_or(uint(accept) << lane);
                    if (lane == 0u) {
                        spins.set(var, bi ^ mask);
                    }
                }
                spins.barrier();
            }
        }
    }

    // Persist the threadgroup words; spilled words and streams are already
    // in device memory.
    for (int var = int(lane); var < spins.shared(n); var += MSA_LANES) {
        my_state[var] = spins.get(var);
    }

    if (!last) {
        return;
    }
    // Pack lane r of this word as read w * 32 + r (bit 1 == spin -1, LSB
    // first per byte), as `msa_body` does.
    int packed_size = (n + 7) / 8;
    int total = MSA_LANES * packed_size;
    for (int idx = int(lane); idx < total; idx += MSA_LANES) {
        int r = idx / packed_size;
        int b = idx - r * packed_size;
        int read = int(w) * MSA_LANES + r;
        if (read >= num_reads) {
            continue;
        }
        uint byte = 0u;
        int base = b * 8;
        for (int bit = 0; bit < 8; ++bit) {
            int var = base + bit;
            if (var < n) {
                byte |= ((spins.get(var) >> uint(r)) & 1u) << uint(bit);
            }
        }
        uint out = (problem_id * uint(num_reads) + uint(read)) * uint(packed_size) + uint(b);
        final_samples[out] = as_type<int8_t>(uchar(byte));
    }
}
