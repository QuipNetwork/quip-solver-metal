// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// Otsuka heuristic cluster optimizer for prepared zero-field models, ported
// from the CUDA kernels/cluster.cu (gpucluster.cu:63-257). Parallel component
// decisions are a heuristic. Union-find:86-97; colored moves:113-136;
// cluster/residual:160-252; final descent:150-158; per-anneal state:145-148.
//
// Metal mapping:
// - One threadgroup of CLUSTER_THREADS threads runs each anneal.
// - `parent`, `acc`, and the spins live in device buffers, one region of
//   `n` entries per anneal, so the node count is not bound by the 32 KB of
//   threadgroup memory. Barriers order device memory (`mem_device`).
// - `atomicCAS` is `atomic_compare_exchange_weak_explicit` in a retry loop;
//   `atomicAdd(&p[x], 0)` is `atomic_load_explicit`.
// - The generator is the two-word 32-bit xorshift (Marsaglia 2003, triple
//   10, 13, 10, period 2^64 - 1), not the 64-bit xorshift of the CUDA kernel.
// - LANES is 1, so no SIMD shuffle reduces a field.
// - A final pass is one greedy sweep; the host repeats it, one command
//   buffer each, until no spin flips.
// The host compiles this library with fast math off, so `exp` and `pow`
// round as the host reference expects.
#include <metal_stdlib>
using namespace metal;

#define CLUSTER_THREADS 512

// The layout of `ClusterArgs` in src/cluster/device.rs.
struct Params {
    int n, tiles, pairs, colors;
    int sweeps, single_every, rounds;
    float beta_min, beta_knee, knee_frac, beta_max, tile_w, dimer_w;
};

// The window of one command buffer, `Window` in src/cluster/device.rs.
struct Window {
    int sweep0, nsweeps, final_pass;
};

struct Rng {
    uint x, y;
};

inline uint next(thread Rng& r) {
    uint t = r.x ^ (r.x << 10);
    r.x = r.y;
    r.y = (r.y ^ (r.y >> 10)) ^ (t ^ (t >> 13));
    return r.y;
}

inline float urand(thread Rng& r) {
    return float(next(r) >> 8) * (1.0f / 16777216.0f);
}

inline uint mix32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}

inline int find_root(device atomic_int* p, int x) {
    while (true) {
        int y = atomic_load_explicit(&p[x], memory_order_relaxed);
        if (y == x) return x;
        x = y;
    }
}

inline void unite(device atomic_int* p, int a, int b) {
    while (true) {
        a = find_root(p, a);
        b = find_root(p, b);
        if (a == b) return;
        if (a < b) { int t = a; a = b; b = t; }
        int expected = a;
        if (atomic_compare_exchange_weak_explicit(&p[a], &expected, b,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

inline float beta_at(constant Params& q, int sweep) {
    float f = q.sweeps > 1 ? float(sweep) / float(q.sweeps - 1) : 1.0f;
    if (q.beta_knee > 0.0f && q.knee_frac > 0.0f && q.knee_frac < 1.0f) {
        if (f < q.knee_frac) return q.beta_min * pow(q.beta_knee / q.beta_min, f / q.knee_frac);
        return q.beta_knee * pow(q.beta_max / q.beta_knee, (f - q.knee_frac) / (1.0f - q.knee_frac));
    }
    return q.beta_min * pow(q.beta_max / q.beta_min, f);
}

inline bool selected(int root, uint salt, int rounds, int round) {
    return mix32(uint(root) ^ salt) % uint(rounds) == uint(round);
}

// One colored single-spin sweep. Nodes of one color are not adjacent, so
// their moves commute. `changed` is set when a greedy move flips a spin.
inline void metropolis(constant Params& q, device const int* color_off, device const int* h,
                       device const int* all_deg, device const int* all_ell, device char* s,
                       float beta, thread Rng& rng, bool greedy, device atomic_int* changed,
                       uint tid) {
    for (int c = 0; c < q.colors; ++c) {
        int end = color_off[c + 1];
        for (int base = color_off[c]; base < end; base += CLUSTER_THREADS) {
            int u = base + int(tid);
            if (u < end) {
                int field = h[u];
                for (int e = 0, d = all_deg[u]; e < d; ++e) {
                    uint w = uint(all_ell[uint(e) * uint(q.n) + uint(u)]);
                    field += int(short(w & 0xffffu)) * int(s[w >> 16]);
                }
                int delta = -2 * int(s[u]) * field;  // milli
                bool accept = greedy
                    ? (delta < 0)
                    : (delta <= 0 || urand(rng) < exp(-beta * float(delta) * 0.001f));
                if (accept) {
                    s[u] = -s[u];
                    if (greedy) atomic_store_explicit(changed, 1, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void cluster_anneal(
    device const int4* tiles      [[buffer(0)]],
    device const char4* gauges    [[buffer(1)]],
    device const int2* pairs      [[buffer(2)]],
    device const int* weak_deg    [[buffer(3)]],
    device const int* weak_ell    [[buffer(4)]],
    device const int* all_deg     [[buffer(5)]],
    device const int* all_ell     [[buffer(6)]],
    device const int* color_off   [[buffer(7)]],
    device const int* h           [[buffer(8)]],
    device const int* tile_w      [[buffer(9)]],
    device const int* link_w      [[buffer(10)]],
    constant Params& q            [[buffer(11)]],
    constant Window& win          [[buffer(12)]],
    device char* gspins           [[buffer(13)]],
    device uint2* grng            [[buffer(14)]],
    device atomic_int* gparent    [[buffer(15)]],
    device atomic_int* gacc       [[buffer(16)]],
    device atomic_int* changed    [[buffer(17)]],
    uint anneal                   [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]]) {
    int n = q.n;
    // The host bounds anneals * n and anneals * CLUSTER_THREADS to uint.
    uint region = anneal * uint(n);
    device char* s = gspins + region;
    device atomic_int* parent = gparent + region;
    device atomic_int* acc = gacc + region;
    uint slot = anneal * CLUSTER_THREADS + tid;
    Rng rng = {grng[slot].x, grng[slot].y};

    if (win.final_pass) {
        metropolis(q, color_off, h, all_deg, all_ell, s, 0.0f, rng, true, changed, tid);
    } else {
        for (int sweep = win.sweep0; sweep < win.sweep0 + win.nsweeps; ++sweep) {
            float beta = beta_at(q, sweep);
            float bt = beta * q.tile_w * 0.001f;
            float z = exp(-2.0f * bt);
            float w0 = z * z * z * z;
            float zero_ice = w0, single_ice = w0 + 4.0f * (z - w0) / 3.0f;
            float zero_defect = exp(-6.0f * bt);
            float dimer_link = 1.0f - exp(-2.0f * beta * q.dimer_w * 0.001f);

            for (int i = int(tid); i < n; i += CLUSTER_THREADS) {
                atomic_store_explicit(&parent[i], i, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_device);
            for (int t = int(tid); t < q.tiles; t += CLUSTER_THREADS) {
                int4 nd = tiles[t];
                char4 ga = gauges[t];
                int node[4] = {nd.x, nd.y, nd.z, nd.w};
                int gg[4] = {ga.x, ga.y, ga.z, ga.w};
                int pos[4], neg[4], np = 0, nn = 0;
                for (int k = 0; k < 4; ++k) {
                    if (int(s[node[k]]) * gg[k] > 0) pos[np++] = node[k]; else neg[nn++] = node[k];
                }
                if (np == 0 || np == 4) continue;
                float r = urand(rng);
                float tz0 = zero_ice, tsi = single_ice, tzd = zero_defect;
                if (float(tile_w[t]) != q.tile_w) {
                    float tbt = beta * float(tile_w[t]) * 0.001f;
                    float tz = exp(-2.0f * tbt);
                    tz0 = tz * tz * tz * tz;
                    tsi = tz0 + 4.0f * (tz - tz0) / 3.0f;
                    tzd = exp(-6.0f * tbt);
                }
                if (np == 2) {
                    if (r < tz0) continue;
                    if (r < tsi) {
                        uint b = next(rng);
                        unite(parent, pos[b & 1], neg[(b >> 1) & 1]);
                    } else {
                        int ch = int(next(rng) & 1);
                        unite(parent, pos[0], neg[ch]);
                        unite(parent, pos[1], neg[1 - ch]);
                    }
                } else if (r >= tzd) {
                    uint b = next(rng);
                    unite(parent, pos[(b & 0xffffu) % uint(np)], neg[(b >> 16) % uint(nn)]);
                }
            }
            for (int p = int(tid); p < q.pairs; p += CLUSTER_THREADS) {
                int2 pr = pairs[p];
                int lw = link_w[p];
                float bond = (abs(lw) == int(q.dimer_w))
                    ? dimer_link
                    : 1.0f - exp(-2.0f * beta * fabs(float(lw)) * 0.001f);
                bool satisfied = lw > 0 ? s[pr.x] == s[pr.y] : s[pr.x] != s[pr.y];
                if (satisfied && urand(rng) < bond) unite(parent, pr.x, pr.y);
            }
            threadgroup_barrier(mem_flags::mem_device);
            for (int i = int(tid); i < n; i += CLUSTER_THREADS) {
                atomic_store_explicit(&parent[i], find_root(parent, i), memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_device);
            uint salt = mix32(uint(sweep) * 0x9e3779b9u + anneal * 0x85ebca6bu);
            for (int round = 0; round < q.rounds; ++round) {
                for (int i = int(tid); i < n; i += CLUSTER_THREADS) {
                    atomic_store_explicit(&acc[i], 0, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_device);
                for (int u = int(tid); u < n; u += CLUSTER_THREADS) {
                    int lu = atomic_load_explicit(&parent[u], memory_order_relaxed);
                    if (!selected(lu, salt, q.rounds, round)) continue;
                    int a = h[u];
                    for (int e = 0, d = weak_deg[u]; e < d; ++e) {
                        uint w = uint(weak_ell[uint(e) * uint(n) + uint(u)]);
                        int v = int(w >> 16);
                        if (atomic_load_explicit(&parent[v], memory_order_relaxed) != lu) {
                            a += int(short(w & 0xffffu)) * int(s[v]);
                        }
                    }
                    if (a) atomic_fetch_add_explicit(&acc[lu], int(s[u]) * a, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_device);
                // Decide on roots; acc[root] = 1 marks a flip.
                for (int r = int(tid); r < n; r += CLUSTER_THREADS) {
                    if (atomic_load_explicit(&parent[r], memory_order_relaxed) != r) continue;
                    if (!selected(r, salt, q.rounds, round)) continue;
                    float delta = -2.0f * float(atomic_load_explicit(&acc[r], memory_order_relaxed));
                    int flip = (urand(rng) < 1.0f / (1.0f + exp(beta * delta * 0.001f))) ? 1 : 0;
                    atomic_store_explicit(&acc[r], flip, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_device);
                for (int u = int(tid); u < n; u += CLUSTER_THREADS) {
                    int lu = atomic_load_explicit(&parent[u], memory_order_relaxed);
                    if (selected(lu, salt, q.rounds, round)
                        && atomic_load_explicit(&acc[lu], memory_order_relaxed) == 1) {
                        s[u] = -s[u];
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
            if (q.single_every > 0 && (sweep + 1) % q.single_every == 0) {
                metropolis(q, color_off, h, all_deg, all_ell, s, beta, rng, false, changed, tid);
            }
        }
    }
    grng[slot] = uint2(rng.x, rng.y);
}
