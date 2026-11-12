#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// METAL SIMULATED ANNEALING - Based on the D-Wave Implementation
// ==============================================================================
// This kernel mimics D-Wave's cpu_sa.cpp implementation:
// 1. Delta energy array optimization (pre-compute, update incrementally)
// 2. xorshift32 RNG
// 3. Sequential variable ordering (spins 0..N-1)
// 4. Metropolis criterion with threshold optimization (skip if delta_E > 22.18/beta)
// 5. Efficient neighbor update after each flip

typedef unsigned int uint;

// Coefficient and delta types. The wide build prepends `#define QUIP_WIDE 1`:
// `int` milli coefficients, an `int` delta, and integer acceptance against a
// host-built threshold row, so no floating point decides a flip. The float
// build prepends `#define QUIP_FLOAT 1`: `float` coefficients and delta, and
// the same `float` Metropolis test as the int8 build. Only the local int8
// build keeps a device energy; the host scores every SA read.
#if defined(QUIP_WIDE)
typedef int coef_t;
typedef int delta_t;
typedef int field_t;
#define WIDE_ROW 2048
#elif defined(QUIP_FLOAT)
typedef float coef_t;
typedef float delta_t;
typedef float field_t;
#else
typedef int8_t coef_t;
typedef int8_t delta_t;
typedef int field_t;
// The host scores every SA read, so the global build, whose models are past
// the thread-private frame, skips the device energy and its O(edges) setup.
#ifndef QUIP_SA_GLOBAL
#define QUIP_DEVICE_ENERGY 1
#endif
#endif

// Storage type of the `h` and `J` buffers. The QUIP_WIDE_I8 build (always with
// QUIP_WIDE) holds each as `int8_t` and multiplies by the model's integer
// scale, passed in buffer 23, as it reads, so every field and delta stays
// milli `int` and the reads equal the plain wide build's.
#if defined(QUIP_WIDE_I8)
typedef int8_t stored_t;
#define COEF(v, s) (int(v) * (s))
#else
typedef coef_t stored_t;
#define COEF(v, s) (v)
#endif

// Address space of each read's packed spins and `delta_energy`. The local
// build keeps them in thread-private arrays of a fixed frame. The
// QUIP_SA_GLOBAL build keeps them in the persistent buffers, interleaved by
// thread: entry `i` of thread `t` is at `i * num_threads + t`, so the threads
// of a SIMD group, which visit the same spin together, touch adjacent bytes.
// Every helper takes that stride; the local build passes 1.
#if defined(QUIP_SA_GLOBAL)
#define SA_STATE device
#else
#define SA_STATE thread
#endif

inline uint xorshift32(thread uint &state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Bit-packing helpers for per-read state
// Pack 8 spins into 1 byte: bit i stores spin i (0=+1, 1=-1)
inline int8_t get_spin_packed(int var, SA_STATE const int8_t* packed_state, uint stride) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    int bit = (packed_state[uint(byte_idx) * stride] >> bit_idx) & 1;
    return bit ? -1 : 1;  // 0 -> +1, 1 -> -1
}

inline void set_spin_packed(int var, int8_t spin, SA_STATE int8_t* packed_state, uint stride) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    int8_t bit = (spin < 0) ? 1 : 0;  // -1 -> 1, +1 -> 0
    int8_t mask = 1 << bit_idx;

    if (bit) {
        packed_state[uint(byte_idx) * stride] |= mask;   // Set bit
    } else {
        packed_state[uint(byte_idx) * stride] &= ~mask;  // Clear bit
    }
}

inline void flip_spin_packed(int var, SA_STATE int8_t* packed_state, uint stride) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    packed_state[uint(byte_idx) * stride] ^= (1 << bit_idx);  // Toggle bit
}

// Compute delta energy for flipping a single variable
// using the bit-packed per-read state
inline delta_t get_flip_energy(
    int var,
    SA_STATE const int8_t* packed_state,
    uint stride,
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    device const stored_t* csr_J_vals,
    device const stored_t* h_vals,
    int coef_scale
) {
    int start = csr_row_ptr[var];
    int end = csr_row_ptr[var + 1];

    // The field starts at h, then adds the row's couplings in CSR order. The
    // host bounds the float build's rounded field for any summation order
    // (sampler::check_float_schedule in src/), so 2 * field stays finite.
    field_t energy = COEF(h_vals[var], coef_scale);

    // Aggressive unrolling for typical Zephyr degrees (18-20)
    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = csr_col_ind[p];
        coef_t Jij = COEF(csr_J_vals[p], coef_scale);
        int8_t neighbor_spin = get_spin_packed(neighbor, packed_state, stride);
        energy += field_t(neighbor_spin) * field_t(Jij);
    }

    // Delta energy = -2 * state[var] * energy. In the int8 build the host
    // admits a local field |h| + sum |J| of at most 63 units
    // (representation::classify in src/), so |delta_E| = 2 * |energy| <= 126,
    // inside int8_t. The wide build holds the delta in int, the float build
    // in float.
    int8_t var_spin = get_spin_packed(var, packed_state, stride);
    return (delta_t)(field_t(-2 * var_spin) * energy);
}

// Pure SA kernel with delta energy optimization
// Batched multi-problem evaluation for reduced kernel overhead
// Each problem gets its own CSR structure, threads process different problems in parallel
//
// Supports chunked dispatch for GPU yielding: when beta_start > 0,
// state is loaded from persistent buffers instead of random init.
// This lets the Python side break the full beta schedule into small
// chunks (~10 betas each) with duty-cycle sleeps between dispatches,
// keeping the GPU responsive on Apple Silicon's shared architecture.
kernel void pure_simulated_annealing(
    device const int* csr_row_ptr [[buffer(0)]],          // Concatenated [N+1] for all problems
    device const int* csr_col_ind [[buffer(1)]],          // Concatenated [nnz] for all problems
    device const stored_t* csr_J_vals [[buffer(2)]],        // Concatenated [nnz] for all problems
    device const int* row_ptr_offsets [[buffer(3)]],      // [num_problems+1] offset into csr_row_ptr
    device const int* col_ind_offsets [[buffer(4)]],      // [num_problems+1] offset into csr_col_ind

    constant int& N [[buffer(5)]],                         // number of spins (same for all problems)
    constant int& num_betas [[buffer(6)]],                 // number of beta values in full schedule
    constant int& sweeps_per_beta [[buffer(7)]],           // sweeps per beta
    constant uint& base_seed [[buffer(8)]],                // RNG seed

    constant int& num_threads [[buffer(12)]],              // total number of threads
    constant int& num_problems [[buffer(13)]],             // number of batched problems
    constant int& num_reads [[buffer(14)]],                // reads per individual problem

    device const float* beta_schedule [[buffer(9)]],       // [num_betas]
    device const stored_t* csr_h_vals [[buffer(15)]],        // Concatenated [N] for all problems - h field values

    // Outputs only (no working memory needed - using thread-local)
    device int8_t* final_samples [[buffer(10)]],           // [num_threads * (N+7)/8] - bit-packed
    device int* final_energies [[buffer(11)]],             // [num_threads]

    // Chunked dispatch parameters for GPU yielding
    constant int& beta_start [[buffer(16)]],               // first beta index this chunk (0 = init)
    constant int& beta_count [[buffer(17)]],               // betas to process this chunk
    device int8_t* persistent_state [[buffer(18)]],        // [num_threads * packed_size] persisted packed state
    device delta_t* persistent_delta_energy [[buffer(19)]], // [num_threads * N] persisted delta_energy
    constant int& sweep_start [[buffer(24)]],              // first sweep of each rung this chunk
    constant int& sweep_count [[buffer(25)]],              // sweeps of each rung this chunk
    device uint* persistent_rng [[buffer(20)]],            // [num_threads] persisted RNG state
    device int* persistent_energy [[buffer(21)]],          // [num_threads] persisted current_energy

#ifdef QUIP_WIDE
    device const int* wide_rows [[buffer(22)]],            // [num_betas * WIDE_ROW] Metropolis thresholds
#endif
#ifdef QUIP_WIDE_I8
    constant int& coef_scale [[buffer(23)]],               // milli value of one stored unit
#endif

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
#ifndef QUIP_WIDE_I8
    const int coef_scale = 1;
#endif
    // Compute global thread ID
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;

    // Early exit if thread_id is out of bounds
    if (thread_id >= num_threads) {
        return;
    }

    // Determine which problem this thread is working on
    uint problem_id = thread_id / num_reads;

    // Get CSR offsets for this specific problem
    int row_ptr_start = row_ptr_offsets[problem_id];
    int col_ind_start = col_ind_offsets[problem_id];

    // Point to this problem's CSR data
    device const int* my_csr_row_ptr = &csr_row_ptr[row_ptr_start];
    device const int* my_csr_col_ind = &csr_col_ind[col_ind_start];
    device const stored_t* my_csr_J_vals = &csr_J_vals[col_ind_start];
    device const stored_t* my_h_vals = &csr_h_vals[problem_id * N];  // h values for this problem

    int n = N;
    int sweeps_per_beta_val = sweeps_per_beta;
    int packed_size = (n + 7) / 8;  // bytes needed for bit-packed state

    // Per-read state: bit-packed spins, (N+7)/8 bytes, and delta_energy, N
    // bytes in the int8 build and 4*N bytes in the wide and float builds.
#ifdef QUIP_SA_GLOBAL
    // Interleaved in the persistent buffers, which every chunk updates in
    // place. The host bounds num_threads * N to 32-bit indexing.
    const uint stride = uint(num_threads);
    device int8_t* packed_state = &persistent_state[thread_id];
    device delta_t* delta_energy = &persistent_delta_energy[thread_id];
#else
    // Thread-local frame, copied from and to the persistent buffers.
    const uint stride = 1;
    thread int8_t packed_state[4593 / 8 + 1];  // Bit-packed state
    thread delta_t delta_energy[4593];           // Delta energy
#endif

    uint rng_state;
    int current_energy;

    // A window [sweep_start, sweep_end) of every rung in this chunk. Only the
    // first window of the first rung initializes.
    int sweep_end = min(sweep_start + sweep_count, sweeps_per_beta_val);
    if (beta_start == 0 && sweep_start == 0) {
        // ── First chunk: random initialization ──────────────
        rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

        // A zero xorshift32 state is a fixed point (xorshift(0) == 0), which
        // freezes this thread's whole RNG stream at a constant. The mixing
        // above yields 0 whenever base_seed, the host's kernel_seed of the
        // public seed, equals thread_id * 12345u (e.g. base_seed 12345 at
        // thread_id 1). Substitute the nonzero state 1; every nonzero mixed
        // value is preserved unchanged.
        if (rng_state == 0u) {
            rng_state = 1u;
        }

        // Generate random initial state (bit-packed)
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            packed_state[uint(byte_idx) * stride] = 0;
        }
        for (int var = 0; var < n; var++) {
            uint rand_val = xorshift32(rng_state);
            int8_t spin = (rand_val & 1) ? -1 : 1;
            set_spin_packed(var, spin, packed_state, stride);
        }

        // Build initial delta_energy array
        for (int var = 0; var < n; var++) {
            delta_energy[uint(var) * stride] = get_flip_energy(var, packed_state, stride, my_csr_row_ptr, my_csr_col_ind, my_csr_J_vals, my_h_vals, coef_scale);
        }

        // Compute initial energy
        current_energy = 0;
#ifdef QUIP_DEVICE_ENERGY
        for (int i = 0; i < n; i++) {
            int8_t spin_i = get_spin_packed(i, packed_state, stride);
            current_energy += my_h_vals[i] * spin_i;
            int start = my_csr_row_ptr[i];
            int end = my_csr_row_ptr[i + 1];
            for (int p = start; p < end; ++p) {
                int j = my_csr_col_ind[p];
                if (j > i) {
                    coef_t Jij = my_csr_J_vals[p];
                    int8_t spin_j = get_spin_packed(j, packed_state, stride);
                    current_energy += Jij * spin_i * spin_j;
                }
            }
        }
#endif
    } else {
        // ── Continuation chunk: load from persistent buffers ──
#ifndef QUIP_SA_GLOBAL
        device const int8_t* src_state = &persistent_state[thread_id * packed_size];
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            packed_state[byte_idx] = src_state[byte_idx];
        }

        device const delta_t* src_de = &persistent_delta_energy[thread_id * n];
        for (int var = 0; var < n; var++) {
            delta_energy[var] = src_de[var];
        }
#endif

        rng_state = persistent_rng[thread_id];
        current_energy = persistent_energy[thread_id];
    }

    // Perform sweeps for this chunk's beta range
    int chunk_end = min(beta_start + beta_count, num_betas);
    for (int beta_idx = beta_start; beta_idx < chunk_end; beta_idx++) {
#ifdef QUIP_WIDE
        device const int* row = &wide_rows[beta_idx * WIDE_ROW];
#else
        float beta = beta_schedule[beta_idx];

        // D-Wave optimization: threshold to skip impossible flips
        // log(1/2^32) ≈ -22.18, so if delta_E > 22.18/beta, probability is < 2^-32
        float threshold = 22.18f / beta;
#endif

        for (int sweep = sweep_start; sweep < sweep_end; sweep++) {
            // Sequential variable ordering (matching D-Wave default)
            for (int var = 0; var < n; var++) {
                delta_t de = delta_energy[uint(var) * stride];

#ifdef QUIP_WIDE
                bool flip_spin = de <= 0;
                if (!flip_spin) {
                    uint rand_val = xorshift32(rng_state);
                    flip_spin = de <= row[rand_val & (WIDE_ROW - 1)];
                }
#else
                // Skip if delta energy too large (D-Wave optimization)
                if (de >= threshold) continue;

                bool flip_spin = false;

                // Metropolis-Hastings acceptance rule
                if (de <= 0) {
                    // Always accept energy-lowering flips
                    flip_spin = true;
                } else {
                    // Get random number
                    uint rand_val = xorshift32(rng_state);

                    // Accept with probability exp(-delta_energy * beta)
                    float prob = exp(-float(de) * beta);
                    float rand_normalized = float(rand_val) / 4294967295.0f;  // 2^32 - 1

                    if (prob > rand_normalized) {
                        flip_spin = true;
                    }
                }
#endif

                if (flip_spin) {
#ifdef QUIP_DEVICE_ENERGY
                    current_energy += de;
#endif

                    // Get current spin value
                    int8_t var_spin = get_spin_packed(var, packed_state, stride);

                    // Update delta energies of all neighbors
#ifdef QUIP_DEVICE_ENERGY
                    int8_t multiplier = 4 * var_spin;
#else
                    int multiplier = 4 * var_spin;
#endif
                    int start = my_csr_row_ptr[var];
                    int end = my_csr_row_ptr[var + 1];

                    // Aggressive unrolling for typical Zephyr degrees (18-20)
                    #pragma unroll 20
                    for (int p = start; p < end; ++p) {
                        int neighbor = my_csr_col_ind[p];
                        coef_t Jij = COEF(my_csr_J_vals[p], coef_scale);
                        int8_t neighbor_spin = get_spin_packed(neighbor, packed_state, stride);
                        SA_STATE delta_t& neighbor_de = delta_energy[uint(neighbor) * stride];
#ifdef QUIP_FLOAT
                        // 4*J may overflow even when the updated delta fits,
                        // so the update rounds once, as CUDA's sa.cu does.
                        neighbor_de = fma(float(multiplier * neighbor_spin), Jij, neighbor_de);
#else
                        neighbor_de += (delta_t)(multiplier * Jij * neighbor_spin);
#endif
                    }

                    // Flip the spin and negate its delta energy
                    flip_spin_packed(var, packed_state, stride);
                    delta_energy[uint(var) * stride] = -de;
                }
            }
        }
    }

    // Persist state for next chunk (always write — the Python side
    // decides whether there will be a next chunk or not). The global build
    // already holds spins and delta_energy in the persistent buffers.
    {
#ifndef QUIP_SA_GLOBAL
        device int8_t* dst_state = &persistent_state[thread_id * packed_size];
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            dst_state[byte_idx] = packed_state[byte_idx];
        }

        device delta_t* dst_de = &persistent_delta_energy[thread_id * n];
        for (int var = 0; var < n; var++) {
            dst_de[var] = delta_energy[var];
        }
#endif

        persistent_rng[thread_id] = rng_state;
        persistent_energy[thread_id] = current_energy;
    }

    // Write final output (always — the Python side reads only after
    // the last chunk, but writing every time is cheaper than branching)
    device int8_t* output = &final_samples[thread_id * packed_size];
    for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
        output[byte_idx] = packed_state[uint(byte_idx) * stride];
    }

    final_energies[thread_id] = current_energy;
}
