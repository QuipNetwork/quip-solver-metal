// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// METAL MULTI-SPIN CODED SIMULATED ANNEALING
// ==============================================================================
// Port of quip-solver-cuda's kernels/msc.cu, itself a port of quip-solver-cpu's
// sa_msc.rs (Isakov, Zintchenko, Ronnow, Troyer 2015). 32 replicas share the
// bits of one 32-bit word per spin: bit r of state[i] is spin i of replica r,
// 0 meaning +1 and 1 meaning -1 (the same convention as the packed output of
// every kernel in this crate).
//
// Differences from msc.cu, all forced by Apple GPUs:
// - 32-bit words, not 64-bit. Threadgroup memory is capped at 32 KB per
//   threadgroup with no opt-in, so N * 8 bytes does not fit Advantage2's
//   4577 spins, and Apple ALUs are 32-bit (64-bit integer ops are emulated).
// - One threadgroup per (problem, word). A job of R reads dispatches R / 32
//   independent threadgroups per problem; there is no `words` loop in-kernel.
// - The host dispatches batches or slot steps and chunks the beta ladder
//   across command buffers (macOS GPU watchdog). Both entries resume from
//   persistent spin and RNG buffers.
// - Geometric thresholds are drawn inline from a 32-bit RNG per word update.
//
// The host checks CSR degree <= MSA_MAX_DEG. The unit entries read each J and
// h by sign, and the host sends them unit models only (J and h in {-1, 0, 1}).
// `msa_anneal_weighted` reads milli `int` coefficients and accepts each flip
// against a host-built integer threshold row (`wide_tables::metropolis_rows`),
// so it holds any model whose local fields fit `WIDE_MAX_FIELD_MILLI`; the
// host scores its energies.
// The batch entry writes each read's energy on the last chunk;
// slot steps write it at flagged checkpoints, in milli units to final_energies.
// The value equals energy_milli whenever every
// coefficient is a whole number in int8 range; the host checks that and
// rescores otherwise.

#define MSA_LANES      32
#define MSA_PLANES     6
#define MSA_MAX_COUNT  63
#define MSA_MAX_FIELD  63
#define MSA_MAX_DEG    20
// Threshold entries per rung for the weighted entry. Must equal
// `wide_tables::WIDE_ROW`.
#define WIDE_ROW       2048

typedef unsigned int uint;
typedef unsigned char uchar;

// ==============================================================================
// RNG - xoshiro128** (same generator and seeding as kernels/gibbs.metal)
// ==============================================================================

struct RngState {
    uint s0, s1, s2, s3;
};

inline uint rotl(uint x, int k) {
    return (x << k) | (x >> (32 - k));
}

inline uint xoshiro128starstar(thread RngState &state) {
    uint result = rotl(state.s1 * 5, 7) * 9;
    uint t = state.s1 << 9;
    state.s2 ^= state.s0;
    state.s3 ^= state.s1;
    state.s1 ^= state.s2;
    state.s0 ^= state.s3;
    state.s2 ^= t;
    state.s3 = rotl(state.s3, 11);
    return result;
}

inline uint splitmix32(thread uint &z) {
    z += 0x9e3779b9u;
    uint r = z;
    r = (r ^ (r >> 16)) * 0x85ebca6bu;
    r = (r ^ (r >> 13)) * 0xc2b2ae35u;
    return r ^ (r >> 16);
}

inline RngState seed_rng(uint seed) {
    uint z = seed;
    RngState state;
    state.s0 = splitmix32(z);
    state.s1 = splitmix32(z);
    state.s2 = splitmix32(z);
    state.s3 = splitmix32(z);
    return state;
}

// One geometric draw M with P(M >= m) = exp(-2 beta m), capped at
// MSA_MAX_FIELD. Same distribution as the old per-rung threshold row
// (M = max m with u < floor(exp(-2 beta m) 2^32)), drawn per word update so
// no row is rebuilt at each rung.
inline int geometric_draw(uint u, float two_beta) {
    if (two_beta <= 0.0f) {
        return MSA_MAX_FIELD;
    }
    float x = min((float(u) + 1.0f) * 2.3283064365386963e-10f, 1.0f);
    float m = floor(-log(x) / two_beta);
    return int(clamp(m, 0.0f, float(MSA_MAX_FIELD)));
}

// ==============================================================================
// Bit-sliced arithmetic (32 lanes)
// ==============================================================================

// Carry-save adder: (h, l) = a + b + c per lane.
inline void csa(thread uint &h, thread uint &l, uint a, uint b, uint c) {
    uint u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}

// Per-lane popcount of 21 one-bit inputs (missing inputs are zero words)
// into planes[0..5] = ones, twos, fours, eights, sixteens, 0. Harley-Seal
// tree: about 100 ops instead of 21 x 18 for a ripple add.
inline void popcount21(thread const uint* x, thread uint* planes) {
    uint ones = 0, twos = 0, fours = 0, eights = 0;
    uint tA, tB, fA, fB, eA, eB, sA, sB;
    csa(tA, ones, ones, x[0], x[1]);   csa(tB, ones, ones, x[2], x[3]);   csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[4], x[5]);   csa(tB, ones, ones, x[6], x[7]);   csa(fB, twos, twos, tA, tB);
    csa(eA, fours, fours, fA, fB);
    csa(tA, ones, ones, x[8], x[9]);   csa(tB, ones, ones, x[10], x[11]); csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[12], x[13]); csa(tB, ones, ones, x[14], x[15]); csa(fB, twos, twos, tA, tB);
    csa(eB, fours, fours, fA, fB);
    csa(sA, eights, eights, eA, eB);
    csa(tA, ones, ones, x[16], x[17]); csa(tB, ones, ones, x[18], x[19]); csa(fA, twos, twos, tA, tB);
    tA = ones & x[20]; ones ^= x[20];
    fB = twos & tA;    twos ^= tA;
    csa(eA, fours, fours, fA, fB);
    sB = eights & eA;  eights ^= eA;
    planes[0] = ones;
    planes[1] = twos;
    planes[2] = fours;
    planes[3] = eights;
    planes[4] = sA | sB;
    planes[5] = 0u;
}

// Lanes whose 6-bit counter is <= limit (bit-serial compare, LSB first).
// Branchless: the constant changes on every word update.
inline uint le_constant(thread const uint* planes, int limit) {
    int bound = limit + 1;
    if (bound > MSA_MAX_COUNT) return 0xFFFFFFFFu;
    uint ge = 0xFFFFFFFFu;
    for (int k = 0; k < MSA_PLANES; ++k) {
        uint set = 0u - uint((bound >> k) & 1);
        uint p = planes[k];
        uint both = p & ge;
        uint either = p ^ ge;
        ge = both | (either & ~set);
    }
    return ~ge;
}

// ==============================================================================
// Coupling sources
// ==============================================================================
// The sweep and the energy read couplings through one of these. `bond(p)`
// gives the lanes where CSR slot p's coupling is negative and whether it is
// nonzero; `coupling(p, r)` gives lane r's coupling value.

struct Bond {
    uint neg;
    bool nonzero;
};

// One model per word: every lane shares an int8 coupling per slot.
struct ModelCouplings {
    device const int8_t* j;
    device const int8_t* h;
    inline Bond bond(int p) const {
        int J = int(j[p]);
        return Bond{(J < 0) ? 0xFFFFFFFFu : 0u, J != 0};
    }
    inline int field(int var) const { return int(h[var]); }
    inline int coupling(int p, int r) const { return int(j[p]); }
};

// Eight models per word, couplings +-1 and no fields. Lane r belongs to
// model r % 8, and bit g of masks[p] is set where model g's coupling at
// slot p is -1, so the byte repeated four times is the lane mask.
struct ScreenCouplings {
    device const uchar* masks;
    inline Bond bond(int p) const {
        return Bond{uint(masks[p]) * 0x01010101u, true};
    }
    inline int field(int var) const { return 0; }
    inline int coupling(int p, int r) const {
        return ((uint(masks[p]) >> uint(r & 7)) & 1u) ? -1 : 1;
    }
};

// One model per word with milli `int` couplings and fields, plus the
// batch's Metropolis threshold rows: one row of WIDE_ROW milli thresholds
// per rung, accepting an uphill change `de` when `de <= entry`.
struct WeightedCouplings {
    device const int* j;
    device const int* h;
    device const int* rows;
};

// Per-lane energy in milli of the spin words in `state` for the nodes this
// thread owns (var = tid, tid + gsz, ...). h*spin*1000 per node plus one
// directed CSR half-edge J*si*sj*500 per slot; a self-loop is stored once,
// so it uses 1000.
template <typename Couplings>
inline void lane_energies(
    thread int* ener,
    threadgroup const uint* state,
    device const int* row_ptr,
    device const int* col_ind,
    Couplings couplings,
    int n, uint tid, uint gsz
) {
    for (int r = 0; r < MSA_LANES; ++r) {
        ener[r] = 0;
    }
    for (uint var = tid; var < uint(n); var += gsz) {
        int bi = int(state[var]);
        int pstart = row_ptr[var];
        int pend = row_ptr[var + 1];
        int h = couplings.field(int(var));
        for (int r = 0; r < MSA_LANES; ++r) {
            int spin = ((bi >> r) & 1) ? -1 : 1;
            ener[r] += h * spin * 1000;
        }
        for (int q = 0; q < MSA_MAX_DEG; ++q) {
            int p = pstart + q;
            if (p < pend && couplings.bond(p).nonzero) {
                int nb = col_ind[p];
                int nbword = int(state[nb]);
                int coeff = (nb == int(var)) ? 1000 : 500;
                for (int r = 0; r < MSA_LANES; ++r) {
                    int spin = ((bi >> r) & 1) ? -1 : 1;
                    int sj = ((nbword >> r) & 1) ? -1 : 1;
                    ener[r] += couplings.coupling(p, r) * spin * sj * coeff;
                }
            }
        }
    }
}

// The screen's couplings are all +-1 with no fields, so a lane's energy is
// 500 * (slots - 2 * satisfied) over this thread's directed CSR slots.
// Count satisfied bonds bit-sliced: popcount21 gives each node's per-lane
// count, and a 16-plane ripple add sums them, about 80 operations a node
// instead of a loop over 32 lanes per slot.
inline void lane_energies(
    thread int* ener,
    threadgroup const uint* state,
    device const int* row_ptr,
    device const int* col_ind,
    ScreenCouplings couplings,
    int n, uint tid, uint gsz
) {
    uint acc[16];
    for (int k = 0; k < 16; ++k) {
        acc[k] = 0u;
    }
    int slots = 0;
    for (uint var = tid; var < uint(n); var += gsz) {
        uint bi = state[var];
        int pstart = row_ptr[var];
        int pend = row_ptr[var + 1];
        uint x[MSA_MAX_DEG + 1];
        for (int q = 0; q < MSA_MAX_DEG; ++q) {
            int p = pstart + q;
            x[q] = (p < pend) ? (couplings.bond(p).neg ^ bi ^ state[col_ind[p]]) : 0u;
        }
        x[MSA_MAX_DEG] = 0u;
        slots += pend - pstart;
        uint planes[MSA_PLANES];
        popcount21(x, planes);
        uint carry = 0u;
        for (int k = 0; k < 16; ++k) {
            uint b = (k < MSA_PLANES) ? planes[k] : 0u;
            uint a = acc[k];
            acc[k] = a ^ b ^ carry;
            carry = (a & b) | (carry & (a ^ b));
        }
    }
    for (int r = 0; r < MSA_LANES; ++r) {
        int satisfied = 0;
        for (int k = 0; k < 16; ++k) {
            satisfied |= int((acc[k] >> uint(r)) & 1u) << k;
        }
        ener[r] = 500 * (slots - 2 * satisfied);
    }
}

// Weighted energies can pass `int` range (fields reach WIDE_MAX_FIELD_MILLI
// per node), so the host scores every weighted read and never reads these.
inline void lane_energies(
    thread int* ener,
    threadgroup const uint* state,
    device const int* row_ptr,
    device const int* col_ind,
    WeightedCouplings couplings,
    int n, uint tid, uint gsz
) {
    for (int r = 0; r < MSA_LANES; ++r) {
        ener[r] = 0;
    }
}

// Lanes of `var`'s word that flip this update. The unit overload counts
// satisfied bonds bit-sliced and compares the count with a geometric draw.
template <typename Couplings>
inline uint flip_mask(
    Couplings couplings,
    device const int* my_csr_row_ptr,
    device const int* my_csr_col_ind,
    threadgroup const uint* state,
    int var, int beta_idx, float two_beta,
    thread RngState& rng
) {
    int pstart = my_csr_row_ptr[var];
    int pend = my_csr_row_ptr[var + 1];
    int h = couplings.field(var);

    // Prefetch every neighbour index and coupling before
    // touching threadgroup memory. Slots past the degree read
    // node 0 with a zero coupling and contribute nothing.
    int nb[MSA_MAX_DEG];
    uint neg[MSA_MAX_DEG];
    bool nz[MSA_MAX_DEG];
    #pragma unroll
    for (int q = 0; q < MSA_MAX_DEG; ++q) {
        int p = pstart + q;
        bool ok = p < pend;
        Bond bond = ok ? couplings.bond(p) : Bond{0u, false};
        nb[q] = ok ? my_csr_col_ind[p] : 0;
        neg[q] = bond.neg;
        nz[q] = bond.nonzero;
    }

    uint bi = state[var];
    uint x[MSA_MAX_DEG + 1];
    int d = 0;
    #pragma unroll
    for (int q = 0; q < MSA_MAX_DEG; ++q) {
        uint sj = state[nb[q]];
        // Set on the replicas where bond q is satisfied.
        uint l = neg[q] ^ bi ^ sj;
        x[q] = nz[q] ? l : 0u;
        d += int(nz[q]);
    }
    // A field is one more bond to a spin pinned at +1.
    x[MSA_MAX_DEG] = (h != 0) ? ((h < 0) ? ~bi : bi) : 0u;
    d += (h != 0);

    uint planes[MSA_PLANES];
    popcount21(x, planes);

    // Metropolis: flip where L <= (d + M) / 2.
    int m = geometric_draw(xoshiro128starstar(rng), two_beta);
    int limit = (d + m) >> 1;
    return (limit >= d) ? 0xFFFFFFFFu : le_constant(planes, limit);
}

// The weighted overload sums each lane's local field in milli and flips where
// the energy change `de = -2 s field` is at most one threshold drawn from the
// rung's row, shared by the word's 32 lanes as the unit draw is. Every entry
// is non-negative, so a downhill or level move always flips. The host bounds
// each field by WIDE_MAX_FIELD_MILLI, so `2 * field` stays inside `int`.
inline uint flip_mask(
    WeightedCouplings couplings,
    device const int* my_csr_row_ptr,
    device const int* my_csr_col_ind,
    threadgroup const uint* state,
    int var, int beta_idx, float two_beta,
    thread RngState& rng
) {
    int pstart = my_csr_row_ptr[var];
    int pend = my_csr_row_ptr[var + 1];
    int h = couplings.h[var];

    int jj[MSA_MAX_DEG];
    uint sw[MSA_MAX_DEG];
    #pragma unroll
    for (int q = 0; q < MSA_MAX_DEG; ++q) {
        int p = pstart + q;
        bool ok = p < pend;
        jj[q] = ok ? couplings.j[p] : 0;
        sw[q] = state[ok ? my_csr_col_ind[p] : 0];
    }
    uint bi = state[var];

    uint draw = xoshiro128starstar(rng) & uint(WIDE_ROW - 1);
    int threshold = couplings.rows[uint(beta_idx) * uint(WIDE_ROW) + draw];

    uint accept = 0u;
    for (int lane = 0; lane < MSA_LANES; ++lane) {
        int field = h;
        #pragma unroll
        for (int q = 0; q < MSA_MAX_DEG; ++q) {
            // Spin bit 1 is -1. Missing neighbours have J = 0.
            field += ((sw[q] >> uint(lane)) & 1u) ? -jj[q] : jj[q];
        }
        int de = ((bi >> uint(lane)) & 1u) ? 2 * field : -2 * field;
        accept |= uint(de <= threshold) << uint(lane);
    }
    return accept;
}

// ==============================================================================
// Kernel
// ==============================================================================
// One threadgroup per (problem, word): threadgroup_position_in_grid.x =
// problem_id * words + word. Threads split each colour class; a barrier
// closes every class so no thread reads a neighbour mid-update. Buffer
// indices 0..15 match every other kernel in this crate, 16..18 and 20 match
// the Gibbs colour-block bindings, 19 carries `words`, and 21..24 match the
// Gibbs chunk bindings.

template <typename Couplings>
static inline void msa_body(
    device const int* my_csr_row_ptr,
    device const int* my_csr_col_ind,
    Couplings couplings,
    device const float* beta_schedule,
    device const uint* src_state,
    device uint* dst_state,
    device const uint* src_rng,
    device uint* dst_rng,
    device int* final_energies,
    device int8_t* final_samples,
    device const int* color_block_starts,
    device const int* color_block_counts,
    device const int* color_node_indices,
    int N, int num_reads, int num_colors, int sweeps_per_beta,
    int beta_start, int beta_end, uint init_seed, bool fresh, bool persist,
    bool write_output, bool write_samples,
    uint w, uint tid, uint gsz,
#ifdef QUIP_MSA_DIAGNOSTICS
    device ulong* diag_accept_counts,
    device int* diag_energy_partials,
#endif
    threadgroup uint* state,
    threadgroup atomic_int* lane_total
) {
#ifdef QUIP_MSA_DIAGNOSTICS
    ulong accept_count = 0;                      // per-thread flips across this chunk's sweeps
#endif

    if (tid < uint(MSA_LANES)) {
        atomic_store_explicit(&lane_total[tid], 0, memory_order_relaxed);
    }

    int n = N;
    int packed_size = (n + 7) / 8;

    RngState rng;
    if (fresh) {
        // A new anneal: seed per (threadgroup, thread) and draw random words.
        rng = seed_rng(init_seed);
        for (uint var = tid; var < uint(n); var += gsz) {
            state[var] = xoshiro128starstar(rng);
        }
    } else {
        // Continuation chunk: threadgroup memory does not survive dispatches,
        // so rebuild the words and the RNG stream from device memory.
        rng.s0 = src_rng[0];
        rng.s1 = src_rng[1];
        rng.s2 = src_rng[2];
        rng.s3 = src_rng[3];
        for (uint var = tid; var < uint(n); var += gsz) {
            state[var] = src_state[var];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int beta_idx = beta_start; beta_idx < beta_end; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        float two_beta = 2.0f * beta;

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
            for (int color = 0; color < num_colors; color++) {
                int block_start = color_block_starts[color];
                int block_count = color_block_counts[color];

                for (uint k = tid; k < uint(block_count); k += gsz) {
                    int var = color_node_indices[block_start + int(k)];
                    uint accept = flip_mask(couplings, my_csr_row_ptr, my_csr_col_ind,
                                            state, var, beta_idx, two_beta, rng);
                    state[var] ^= accept;
#ifdef QUIP_MSA_DIAGNOSTICS
                    accept_count += ulong(popcount(accept));
#endif
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Persist for the next chunk (the host decides whether one follows).
    // The barrier closing the last colour class already synchronised
    // `state`, and each thread writes a disjoint stride.
    if (persist) {
        dst_rng[0] = rng.s0;
        dst_rng[1] = rng.s1;
        dst_rng[2] = rng.s2;
        dst_rng[3] = rng.s3;
        for (uint var = tid; var < uint(n); var += gsz) {
            dst_state[var] = state[var];
        }
    }

    // Energies. The diagnostic build keeps its per-(thread, lane) partials on
    // every chunk; the production build reduces them only when requested.
#ifdef QUIP_MSA_DIAGNOSTICS
    bool need_energy = true;
#else
    bool need_energy = write_output;
#endif
    if (need_energy) {
        int ener[MSA_LANES];
        lane_energies(ener, state, my_csr_row_ptr, my_csr_col_ind, couplings,
                      n, tid, gsz);
#ifdef QUIP_MSA_DIAGNOSTICS
        device int* dst = diag_energy_partials;
        for (int r = 0; r < MSA_LANES; ++r) {
            dst[r] = ener[r];
        }
#endif
        if (write_output) {
            for (int r = 0; r < MSA_LANES; ++r) {
                int s = simd_sum(ener[r]);
                if (simd_is_first()) {
                    atomic_fetch_add_explicit(&lane_total[r], s, memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (write_output && tid < uint(MSA_LANES)) {
        int read = int(w) * MSA_LANES + int(tid);
        if (read < num_reads) {
            final_energies[read] =
                atomic_load_explicit(&lane_total[tid], memory_order_relaxed);
        }
    }

#ifdef QUIP_MSA_DIAGNOSTICS
    // Per-thread accepted-flip counter for this chunk (overwritten per chunk;
    // no atomics, one disjoint write per thread).
    diag_accept_counts[0] = accept_count;
#endif

    // Pack lane r of this word as read w * 32 + r (bit 1 == spin -1, LSB
    // first per byte). (lane, byte) pairs are spread over the threadgroup;
    // each pair owns one output byte, so there are no write races.
    if (!write_samples) {
        return;
    }
    int total = MSA_LANES * packed_size;
    for (int idx = int(tid); idx < total; idx += int(gsz)) {
        int lane = idx / packed_size;
        int b = idx - lane * packed_size;
        int read = int(w) * MSA_LANES + lane;
        if (read >= num_reads) {
            continue;
        }
        uint byte = 0u;
        int base = b * 8;
        for (int bit = 0; bit < 8; ++bit) {
            int var = base + bit;
            if (var < n) {
                byte |= ((state[var] >> uint(lane)) & 1u) << uint(bit);
            }
        }
        uint out = uint(read) * uint(packed_size) + uint(b);
        final_samples[out] = as_type<int8_t>(uchar(byte));
    }
}

#ifdef QUIP_MSA_DIAGNOSTICS
kernel void msa_anneal_diag(
#else
kernel void msa_anneal(
#endif
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device const int8_t* csr_J_vals [[buffer(2)]],
    device const int* row_ptr_offsets [[buffer(3)]], // unused by msa: the CSR structure is shared
    device const int* col_ind_offsets [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    device const float* beta_schedule [[buffer(9)]],

    device int8_t* final_samples [[buffer(10)]],           // [num_problems * num_reads * packed_size]
    device int* final_energies [[buffer(11)]],             // [num_problems * num_reads] milli, last chunk only

    constant int& num_threadgroups [[buffer(12)]],         // num_problems * words
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    device const int8_t* csr_h_vals [[buffer(15)]],

    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],

    constant int& words [[buffer(19)]],                    // words per problem = num_reads / 32
    constant int& num_colors [[buffer(20)]],

    constant int& beta_start [[buffer(21)]],               // first beta index this chunk (0 = init)
    constant int& beta_count [[buffer(22)]],               // betas to process this chunk
    device uint* persistent_state [[buffer(23)]],          // [num_threadgroups * N] spin words
    device uint* persistent_rng [[buffer(24)]],            // [num_threadgroups * group_size * 4]
#ifdef QUIP_MSA_DIAGNOSTICS
    device ulong* diag_accept_counts [[buffer(25)]],       // [num_threadgroups * group_size] per-thread flips
    device int* diag_energy_partials [[buffer(26)]],       // [num_threadgroups * group_size * 32] per-(thread, lane)
#endif
    threadgroup uint* state [[threadgroup(0)]],            // [N] spin words, sized by the host

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup atomic_int lane_total[MSA_LANES];
    uint tg = threadgroup_pos.x;
    if (tg >= uint(num_threadgroups)) {
        return;
    }
    uint tid = thread_pos_in_group.x;
    uint gsz = threads_per_group.x;
    uint problem_id = tg / uint(words);
    uint w = tg - problem_id * uint(words);
    int beta_end = min(beta_start + beta_count, num_betas);
    uint init_seed = (base_seed ? base_seed : 1u) ^ (tg * 2654435761u) ^ (tid * 2246822519u);
    ModelCouplings couplings{&csr_J_vals[col_ind_offsets[problem_id]],
                             &csr_h_vals[problem_id * uint(N)]};
    msa_body(csr_row_ptr, csr_col_ind, couplings, beta_schedule,
             &persistent_state[tg * uint(N)], &persistent_state[tg * uint(N)],
             &persistent_rng[(tg * gsz + tid) * 4], &persistent_rng[(tg * gsz + tid) * 4],
             &final_energies[problem_id * uint(num_reads)],
             &final_samples[problem_id * uint(num_reads) * uint((N + 7) / 8)],
             color_block_starts, color_block_counts, color_node_indices,
             N, num_reads, num_colors, sweeps_per_beta, beta_start, beta_end, init_seed,
             beta_start == 0, true, beta_end >= num_betas, true, w, tid, gsz,
#ifdef QUIP_MSA_DIAGNOSTICS
             &diag_accept_counts[tg * gsz + tid],
             &diag_energy_partials[(tg * gsz + tid) * MSA_LANES],
#endif
             state, lane_total);
}

#ifndef QUIP_MSA_DIAGNOSTICS
// `msa_anneal` for weighted models: buffers 2 and 15 hold milli `int`
// coefficients and buffer 25 the Metropolis rows, [num_betas * WIDE_ROW].
// Energies are left to the host.
kernel void msa_anneal_weighted(
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device const int* csr_J_vals [[buffer(2)]],
    device const int* row_ptr_offsets [[buffer(3)]], // unused: the CSR structure is shared
    device const int* col_ind_offsets [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    device const float* beta_schedule [[buffer(9)]],

    device int8_t* final_samples [[buffer(10)]],
    device int* final_energies [[buffer(11)]],

    constant int& num_threadgroups [[buffer(12)]],
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    device const int* csr_h_vals [[buffer(15)]],

    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],

    constant int& words [[buffer(19)]],
    constant int& num_colors [[buffer(20)]],

    constant int& beta_start [[buffer(21)]],
    constant int& beta_count [[buffer(22)]],
    device uint* persistent_state [[buffer(23)]],
    device uint* persistent_rng [[buffer(24)]],
    device const int* wide_rows [[buffer(25)]],
    threadgroup uint* state [[threadgroup(0)]],

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup atomic_int lane_total[MSA_LANES];
    uint tg = threadgroup_pos.x;
    if (tg >= uint(num_threadgroups)) {
        return;
    }
    uint tid = thread_pos_in_group.x;
    uint gsz = threads_per_group.x;
    uint problem_id = tg / uint(words);
    uint w = tg - problem_id * uint(words);
    int beta_end = min(beta_start + beta_count, num_betas);
    uint init_seed = (base_seed ? base_seed : 1u) ^ (tg * 2654435761u) ^ (tid * 2246822519u);
    WeightedCouplings couplings{&csr_J_vals[col_ind_offsets[problem_id]],
                                &csr_h_vals[problem_id * uint(N)], wide_rows};
    msa_body(csr_row_ptr, csr_col_ind, couplings, beta_schedule,
             &persistent_state[tg * uint(N)], &persistent_state[tg * uint(N)],
             &persistent_rng[(tg * gsz + tid) * 4], &persistent_rng[(tg * gsz + tid) * 4],
             &final_energies[problem_id * uint(num_reads)],
             &final_samples[problem_id * uint(num_reads) * uint((N + 7) / 8)],
             color_block_starts, color_block_counts, color_node_indices,
             N, num_reads, num_colors, sweeps_per_beta, beta_start, beta_end, init_seed,
             beta_start == 0, true, beta_end >= num_betas, true, w, tid, gsz,
             state, lane_total);
}

struct SlotStep {
    uint slot;        // storage index for h, J, schedule, state, rng, outputs
    int  beta_start;  // 0 = initialise spins and RNG from seed
    int  beta_count;  // rungs this step
    int  num_betas;   // this slot's schedule length
    uint seed;        // this leg's seed folded to 32 bits, never 0
    uint flags;       // bit 0: write energies and packed samples
                      // bit 2: start a fresh anneal at beta_start
};

kernel void msa_anneal_slots(
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device const int8_t* csr_J_vals [[buffer(2)]],
    device const int* row_ptr_offsets [[buffer(3)]], // unused by msa: the CSR structure is shared
    constant int& j_stride [[buffer(4)]],

    constant int& N [[buffer(5)]],
    constant int& sched_stride [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    device const float* beta_schedule [[buffer(9)]],

    device int8_t* final_samples [[buffer(10)]],           // [num_slots * num_reads * packed_size]
    device int* final_energies [[buffer(11)]],             // [num_slots * num_reads] milli, output flag only

    constant int& num_threadgroups [[buffer(12)]],         // num_steps * words
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    device const int8_t* csr_h_vals [[buffer(15)]],

    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],

    constant int& words [[buffer(19)]],                    // words per slot = num_reads / 32
    constant int& num_colors [[buffer(20)]],

    constant int& beta_start [[buffer(21)]],               // unused: supplied by the step
    constant int& beta_count [[buffer(22)]],               // unused: supplied by the step
    device uint* persistent_state [[buffer(23)]],          // [num_slots * words * N] spin words
    device uint* persistent_rng [[buffer(24)]],            // [num_slots * words * group_size * 4]
    device const SlotStep* steps [[buffer(25)]],
    threadgroup uint* state [[threadgroup(0)]],            // [N] spin words, sized by the host

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup atomic_int lane_total[MSA_LANES];
    uint tg = threadgroup_pos.x;
    if (tg >= uint(num_threadgroups)) {
        return;
    }
    uint tid = thread_pos_in_group.x;
    uint gsz = threads_per_group.x;
    SlotStep st = steps[tg / uint(words)];
    uint w = tg % uint(words);
    uint g = st.slot * uint(words) + w;
    int beta_end = min(st.beta_start + st.beta_count, st.num_betas);
    uint init_seed = st.seed ^ (w * 2654435761u) ^ (tid * 2246822519u);
    bool write_output = (st.flags & 1u) != 0;
    bool fresh = st.beta_start == 0 || (st.flags & 4u) != 0;
    ModelCouplings couplings{&csr_J_vals[st.slot * uint(j_stride)], &csr_h_vals[st.slot * uint(N)]};
    msa_body(csr_row_ptr, csr_col_ind, couplings, &beta_schedule[st.slot * uint(sched_stride)],
             &persistent_state[g * uint(N)], &persistent_state[g * uint(N)],
             &persistent_rng[(g * gsz + tid) * 4], &persistent_rng[(g * gsz + tid) * 4],
             &final_energies[st.slot * uint(num_reads)],
             &final_samples[st.slot * uint(num_reads) * uint((N + 7) / 8)],
             color_block_starts, color_block_counts, color_node_indices,
             N, num_reads, num_colors, 1, st.beta_start, beta_end, init_seed, fresh, true,
             write_output, write_output, w, tid, gsz, state, lane_total);
}

// ==============================================================================
// Nonce-parallel screen
// ==============================================================================
// One threadgroup anneals one word of eight models with four reads each:
// lane r is read r / 8 of model r % 8. Every word runs a complete anneal in
// one dispatch and nothing persists. The prologue builds the word's mask
// bytes in device memory; threadgroup memory cannot hold them beside the
// spin words. A model's row is either its edge sign bits (bit e set where
// edge e has coupling -1) or, in draw mode, its ChaCha8 key: the lease draw
// of quip-protocol's chacha8, where keystream word first_word + e picks
// allowed_j[word % allowed_len] for edge e.

// The draw mode of a dispatch. mode 0 reads sign bits.
struct ScreenDraw {
    uint mode;         // 1: rows are ChaCha8 keys
    uint first_word;   // keystream word of edge 0
    uint allowed_len;  // allowed coupling values, 1..32
    uint neg_table;    // bit k set where allowed_j[k] < 0
};

// One ChaCha8 block (four double rounds) at block `counter`, stream 0.
inline void chacha8_block(device const uint* key, uint counter, thread uint* out) {
    uint s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
                  key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
                  counter, 0u, 0u, 0u};
    uint x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = s[i];
    }
#define QR(a, b, c, d)                                   \
    x[a] += x[b]; x[d] ^= x[a]; x[d] = rotate(x[d], 16u); \
    x[c] += x[d]; x[b] ^= x[c]; x[b] = rotate(x[b], 12u); \
    x[a] += x[b]; x[d] ^= x[a]; x[d] = rotate(x[d], 8u);  \
    x[c] += x[d]; x[b] ^= x[c]; x[b] = rotate(x[b], 7u);
    for (int round = 0; round < 4; ++round) {
        QR(0, 4, 8, 12) QR(1, 5, 9, 13) QR(2, 6, 10, 14) QR(3, 7, 11, 15)
        QR(0, 5, 10, 15) QR(1, 6, 11, 12) QR(2, 7, 8, 13) QR(3, 4, 9, 14)
    }
#undef QR
    for (int i = 0; i < 16; ++i) {
        out[i] = x[i] + s[i];
    }
}

struct ScreenWord {
    uint sched;      // offset of this word's betas in beta_schedule
    int  num_betas;  // sweeps, one per beta
    uint seed;       // never 0
    uint pad;
};

kernel void msa_screen(
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device uchar* masks [[buffer(2)]],                     // [words * nnz]
    device const uint* rows [[buffer(3)]],                 // [words * 8 * row_words]
    device const uint2* edge_pos [[buffer(4)]],            // [num_edges] CSR slots (ij, ji)
    constant int& N [[buffer(5)]],
    constant int& nnz [[buffer(6)]],
    constant int& num_edges [[buffer(7)]],
    constant int& row_words [[buffer(8)]],                 // edge_words, or 8 in draw mode
    device const float* beta_schedule [[buffer(9)]],
    device int8_t* final_samples [[buffer(10)]],           // [words * 32 * packed_size]
    device int* final_energies [[buffer(11)]],             // [words * 32] milli
    constant int& num_threadgroups [[buffer(12)]],         // words this dispatch
    constant ScreenDraw& draw [[buffer(13)]],
    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],
    constant int& num_colors [[buffer(20)]],
    device const ScreenWord* words [[buffer(25)]],
    threadgroup uint* state [[threadgroup(0)]],            // [N] spin words, sized by the host

    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup atomic_int lane_total[MSA_LANES];
    uint tg = threadgroup_pos.x;
    if (tg >= uint(num_threadgroups)) {
        return;
    }
    uint tid = thread_pos_in_group.x;
    uint gsz = threads_per_group.x;
    ScreenWord word = words[tg];
    device uchar* my_masks = &masks[tg * uint(nnz)];
    device const uint* my_rows = &rows[tg * 8u * uint(row_words)];
    if (draw.mode == 1u) {
        // One thread owns a keystream block for all eight models, so it owns
        // every mask byte of the block's edges.
        uint first = draw.first_word;
        uint end = first + uint(num_edges);
        for (uint b = first / 16u + tid; b * 16u < end; b += gsz) {
            uint bytes[16];
            for (int i = 0; i < 16; ++i) {
                bytes[i] = 0u;
            }
            for (uint g = 0; g < 8u; ++g) {
                uint block[16];
                chacha8_block(&my_rows[g * uint(row_words)], b, block);
                for (int i = 0; i < 16; ++i) {
                    bytes[i] |= ((draw.neg_table >> (block[i] % draw.allowed_len)) & 1u) << g;
                }
            }
            for (uint i = 0; i < 16u; ++i) {
                uint w = b * 16u + i;
                if (w >= first && w < end) {
                    uint2 pos = edge_pos[w - first];
                    my_masks[pos.x] = uchar(bytes[i]);
                    my_masks[pos.y] = uchar(bytes[i]);
                }
            }
        }
    } else {
        for (uint e = tid; e < uint(num_edges); e += gsz) {
            uint shift = e & 31u;
            uint at = e >> 5;
            uint byte = 0u;
            for (uint g = 0; g < 8u; ++g) {
                byte |= ((my_rows[g * uint(row_words) + at] >> shift) & 1u) << g;
            }
            uint2 pos = edge_pos[e];
            my_masks[pos.x] = uchar(byte);
            my_masks[pos.y] = uchar(byte);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    ScreenCouplings couplings{my_masks};
    uint init_seed = word.seed ^ (tid * 2246822519u);
    msa_body(csr_row_ptr, csr_col_ind, couplings, &beta_schedule[word.sched],
             nullptr, nullptr, nullptr, nullptr,
             &final_energies[tg * uint(MSA_LANES)],
             &final_samples[tg * uint(MSA_LANES) * uint((N + 7) / 8)],
             color_block_starts, color_block_counts, color_node_indices,
             N, MSA_LANES, num_colors, 1, 0, word.num_betas, init_seed, true, false,
             true, true, 0, tid, gsz, state, lane_total);
}
#endif
