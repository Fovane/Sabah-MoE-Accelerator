// ---------------------------------------------------------------------------
// Sabah runtime - CUDA side.
//
// This file is the part of Sabah that is allowed to touch the model's
// arithmetic, and therefore the part that must not change it. If the router
// selects expert E, these kernels execute expert E's real quantized weights.
// There is no approximation, no substitution and no predicted stand-in.
//
// What it provides:
//   * device allocation for expert slots and activations
//   * pinned host staging + an async H2D copy stream with events
//   * dequantize-and-matvec kernels for the quant types the Flash-Next
//     expert bank actually uses: Q4_K, Q5_K (gate/up) and Q5_1, Q8_0 (down)
//   * a standalone dequantizer, so the kernels above can be checked against
//     gguf-py's reference implementation instead of being trusted
//
// Quant layouts follow ggml-common.h exactly; the block geometries are
// asserted against the model at load time on the Python side.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>

#if defined(_WIN32)
  #define SABAH_API __declspec(dllexport)
#else
  #define SABAH_API __attribute__((visibility("default")))
#endif

// ---- quant type codes (ggml enum values, so traces stay comparable) --------
#define SB_Q5_1  7
#define SB_Q8_0  8
#define SB_Q4_K 12
#define SB_Q5_K 13

// Test-only sentinel builds (never defined for a shipped library):
//   1  slot 0 of every call executes the neighbouring expert
//   2  expert slot i reads input row (i+1) mod n_used   (the RC3 bug class)
//   3  token t reads input row of token (t+1) mod n_tokens  (cross-sequence)
#ifndef SABAH_SENTINEL
#define SABAH_SENTINEL 0
#endif
#ifdef SABAH_SENTINEL_WRONG_EXPERT
#undef SABAH_SENTINEL
#define SABAH_SENTINEL 1
#endif

static char g_err[512] = {0};
static cudaStream_t g_copy    = 0;
static cudaStream_t g_compute = 0;

// The Python block executor and the llama.cpp adapter share this native
// residency core.  Entries are keyed by the logical source tensor pointer and
// expert id; the source tensor pointer is stable for the lifetime of a loaded
// GGUF model.  A single global byte budget is used so gate/up/down tensors
// compete for the same hot tier instead of creating three independent caches.
struct sb_mmid_entry {
    const void * source_key;
    int expert;
    void * device;
    size_t bytes;
    uint64_t touch;
    uint64_t epoch;     // last MUL_MAT_ID call that referenced this slot
};

static std::mutex g_mmid_mutex;
static std::vector<sb_mmid_entry> g_mmid_entries;
static size_t g_mmid_capacity = 0;
static size_t g_mmid_bytes = 0;
static uint64_t g_mmid_tick = 0;
static uint64_t g_mmid_hits = 0;
static uint64_t g_mmid_misses = 0;
static uint64_t g_mmid_evictions = 0;
static uint64_t g_mmid_bytes_fetched = 0;
// Diagnostics. `epoch` identifies the MUL_MAT_ID call in progress: a slot whose
// pointer has already been handed to that call must not be evicted before the
// call's kernel has run, whatever the capacity says.
static uint64_t g_mmid_epoch = 0;
static uint64_t g_mmid_calls = 0;
static uint64_t g_mmid_tokens = 0;
static uint64_t g_mmid_overflow = 0;       // allocations that exceeded capacity
static uint64_t g_mmid_verify_ok = 0;
static uint64_t g_mmid_verify_fail = 0;
static bool     g_mmid_verify = false;     // SABAH_LLAMA_VERIFY_BYTES=1 | fetch
static bool     g_mmid_verify_all_hits = false;
static uint64_t g_mmid_verify_hit_every = 64; // "fetch" mode: every Nth hit, by counter
static uint64_t g_mmid_empty_calls = 0;    // MUL_MAT_ID with zero token rows (no-op)
// in-runtime float64 self-check (SABAH_LLAMA_SELFCHECK=K samples per call)
static int      g_sc_k = 0;
static double   g_sc_tol = 2e-6;
static uint64_t g_sc_ok = 0;
static uint64_t g_sc_fail = 0;
static double   g_sc_max = 0.0;
static uint64_t g_sc_slot[16] = {0};
static uint64_t g_sc_multi_token = 0;      // samples taken from calls with >1 token row
static std::vector<const void *> g_sc_tensors;   // distinct expert tensors checked

static bool ck(cudaError_t e, const char * what);

static size_t sb_mmid_capacity_bytes() {
    const char * env = std::getenv("SABAH_LLAMA_HOT_BYTES");
    if (env && *env) {
        char * end = nullptr;
        const unsigned long long value = std::strtoull(env, &end, 10);
        if (end != env && value > 0) {
            return (size_t) value;
        }
    }
    return (size_t) 2ull * 1024ull * 1024ull * 1024ull;
}

static void sb_mmid_reset() {
    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    for (const auto & entry : g_mmid_entries) {
        if (entry.device) {
            cudaFree(entry.device);
        }
    }
    g_mmid_entries.clear();
    g_mmid_capacity = sb_mmid_capacity_bytes();
    g_mmid_bytes = 0;
    g_mmid_tick = 0;
    g_mmid_hits = 0;
    g_mmid_misses = 0;
    g_mmid_evictions = 0;
    g_mmid_bytes_fetched = 0;
    g_mmid_epoch = 0;
    g_mmid_calls = 0;
    g_mmid_tokens = 0;
    g_mmid_overflow = 0;
    g_mmid_verify_ok = 0;
    g_mmid_verify_fail = 0;
    // "1": every fetch AND every hit (small tests only: a long prefill would
    //      copy terabytes back); "fetch": every fetch (miss) plus every 64th
    //      hit, chosen by the hit counter, never by an outcome.
    const char * v = std::getenv("SABAH_LLAMA_VERIFY_BYTES");
    g_mmid_verify = v && (std::strcmp(v, "1") == 0 || std::strcmp(v, "fetch") == 0);
    g_mmid_verify_all_hits = v && std::strcmp(v, "1") == 0;
    g_mmid_empty_calls = 0;
    const char * k = std::getenv("SABAH_LLAMA_SELFCHECK");
    g_sc_k = k ? std::atoi(k) : 0;
    const char * tol = std::getenv("SABAH_LLAMA_SELFCHECK_TOL");
    g_sc_tol = tol ? std::atof(tol) : 2e-6;
    g_sc_ok = g_sc_fail = g_sc_multi_token = 0;
    g_sc_max = 0.0;
    for (auto & c : g_sc_slot) c = 0;
    g_sc_tensors.clear();
}

static void sb_mmid_evict_until(size_t needed, cudaStream_t compute_stream) {
    // A slot can still be read by a previously submitted compute kernel.  The
    // conservative synchronization is only taken on an actual eviction and
    // preserves the exact-output invariant.
    //
    // Slots already handed to the call in progress are never victims: their
    // device pointers sit in that call's pointer table and the kernel has not
    // run yet. If nothing else can be evicted the tier temporarily exceeds its
    // budget instead of producing a dangling pointer.
    while (g_mmid_bytes + needed > g_mmid_capacity) {
        auto victim = g_mmid_entries.end();
        for (auto it = g_mmid_entries.begin(); it != g_mmid_entries.end(); ++it) {
            if (it->epoch == g_mmid_epoch) continue;
            if (victim == g_mmid_entries.end() || it->touch < victim->touch) victim = it;
        }
        if (victim == g_mmid_entries.end()) {
            ++g_mmid_overflow;
            return;
        }
        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(g_copy);
        cudaFree(victim->device);
        g_mmid_bytes -= victim->bytes;
        g_mmid_entries.erase(victim);
        ++g_mmid_evictions;
    }
}

// Byte-exact check of a resident slot against the original GGUF range.
static bool sb_mmid_verify(const void * device, const void * source, size_t bytes) {
    std::vector<uint8_t> back(bytes);
    if (cudaStreamSynchronize(g_copy) != cudaSuccess ||
        cudaMemcpy(back.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        ++g_mmid_verify_fail;
        return false;
    }
    if (std::memcmp(back.data(), source, bytes) != 0) {
        ++g_mmid_verify_fail;
        return false;
    }
    ++g_mmid_verify_ok;
    return true;
}

static void * sb_mmid_resident(const void * source_key, int expert,
                               const void * source, size_t bytes,
                               cudaStream_t compute_stream) {
    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    if (g_mmid_capacity == 0) {
        g_mmid_capacity = sb_mmid_capacity_bytes();
    }

    for (auto & entry : g_mmid_entries) {
        if (entry.source_key == source_key && entry.expert == expert && entry.bytes == bytes) {
            entry.touch = ++g_mmid_tick;
            entry.epoch = g_mmid_epoch;
            ++g_mmid_hits;
            if (g_mmid_verify && (g_mmid_verify_all_hits || g_mmid_hits % g_mmid_verify_hit_every == 0) &&
                !sb_mmid_verify(entry.device, source, bytes)) {
                snprintf(g_err, sizeof(g_err), "Sabah resident expert %d differs from its GGUF bytes (hit)", expert);
                return nullptr;
            }
            return entry.device;
        }
    }

    ++g_mmid_misses;
    sb_mmid_evict_until(bytes, compute_stream);

    void * device = nullptr;
    if (!ck(cudaMalloc(&device, bytes), "Sabah expert slot allocation")) {
        return nullptr;
    }
    if (!ck(cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, g_copy),
            "Sabah expert H2D")) {
        cudaFree(device);
        return nullptr;
    }

    g_mmid_entries.push_back({ source_key, expert, device, bytes, ++g_mmid_tick, g_mmid_epoch });
    g_mmid_bytes += bytes;
    g_mmid_bytes_fetched += bytes;
    if (g_mmid_verify && !sb_mmid_verify(device, source, bytes)) {
        snprintf(g_err, sizeof(g_err), "Sabah fetched expert %d differs from its GGUF bytes (miss)", expert);
        return nullptr;
    }
    return device;
}

static bool ck(cudaError_t e, const char * what) {
    if (e == cudaSuccess) return true;
    snprintf(g_err, sizeof(g_err), "%s: %s", what, cudaGetErrorString(e));
    return false;
}

// ---------------------------------------------------------------------------
// block geometry, kept in one place so host and device agree
// ---------------------------------------------------------------------------
__host__ __device__ static inline int sb_block_bytes(int qt) {
    switch (qt) {
        case SB_Q4_K: return 144;
        case SB_Q5_K: return 176;
        case SB_Q5_1: return  24;
        case SB_Q8_0: return  34;
    }
    return 0;
}
__host__ __device__ static inline int sb_block_elems(int qt) {
    switch (qt) {
        case SB_Q4_K: return 256;
        case SB_Q5_K: return 256;
        case SB_Q5_1: return  32;
        case SB_Q8_0: return  32;
    }
    return 0;
}

// row byte stride for a row of n elements
__host__ __device__ static inline size_t sb_row_bytes(int qt, int n) {
    const int be = sb_block_elems(qt);
    if (be == 0) return 0;
    return (size_t) (n / be) * sb_block_bytes(qt);
}

// ---------------------------------------------------------------------------
// Host reference dequantizer for the self-check. A direct scalar port of
// ggml's dequantize_row_{q4_K,q5_K,q5_1,q8_0}; it deliberately shares no code
// with the device sub32_* path it is used to check.
// ---------------------------------------------------------------------------
static float sb_host_f16(const uint8_t * p) {
    const uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                                   // subnormal
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static void sb_host_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Dequantize n elements (a whole number of blocks) of type qt into y.
static bool sb_host_dequant(const uint8_t * x, float * y, int qt, int n) {
    switch (qt) {
    case SB_Q8_0:
        for (int b = 0; b < n / 32; ++b, x += 34) {
            const float d = sb_host_f16(x);
            for (int j = 0; j < 32; ++j) *y++ = (float) (int8_t) x[2 + j] * d;
        }
        return true;
    case SB_Q5_1:
        for (int b = 0; b < n / 32; ++b, x += 24) {
            const float d = sb_host_f16(x), m = sb_host_f16(x + 2);
            uint32_t qh;
            std::memcpy(&qh, x + 4, 4);
            const uint8_t * qs = x + 8;
            for (int j = 0; j < 16; ++j) {
                const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
                const uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
                y[j]      = (float) ((qs[j] & 0x0F) | xh0) * d + m;
                y[j + 16] = (float) ((qs[j] >> 4)   | xh1) * d + m;
            }
            y += 32;
        }
        return true;
    case SB_Q4_K:
        for (int b = 0; b < n / 256; ++b, x += 144) {
            const float d = sb_host_f16(x), dmin = sb_host_f16(x + 2);
            const uint8_t * sc = x + 4;
            const uint8_t * q = x + 16;
            int is = 0;
            uint8_t s6, m6;
            for (int j = 0; j < 256; j += 64) {
                sb_host_scale_min_k4(is + 0, sc, &s6, &m6);
                const float d1 = d * s6, m1 = dmin * m6;
                sb_host_scale_min_k4(is + 1, sc, &s6, &m6);
                const float d2 = d * s6, m2 = dmin * m6;
                for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32; is += 2;
            }
        }
        return true;
    case SB_Q5_K:
        for (int b = 0; b < n / 256; ++b, x += 176) {
            const float d = sb_host_f16(x), dmin = sb_host_f16(x + 2);
            const uint8_t * sc = x + 4;
            const uint8_t * qh = x + 16;
            const uint8_t * ql = x + 48;
            int is = 0;
            uint8_t s6, m6, u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                sb_host_scale_min_k4(is + 0, sc, &s6, &m6);
                const float d1 = d * s6, m1 = dmin * m6;
                sb_host_scale_min_k4(is + 1, sc, &s6, &m6);
                const float d2 = d * s6, m2 = dmin * m6;
                for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
        return true;
    }
    return false;
}

static uint64_t sb_mix64(uint64_t z) {          // splitmix64 finaliser
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// K-quant 6-bit packed scale/min extraction (ggml get_scale_min_k4)
__device__ static inline void get_scale_min_k4(int j, const uint8_t * q,
                                               uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

__device__ static inline float h2f(const uint8_t * p) {
    __half h;
    memcpy(&h, p, sizeof(__half));
    return __half2float(h);
}

// ---------------------------------------------------------------------------
// sub-block dot products.
//
// Every supported type is processed in units of 32 elements, so a row of any
// type is a flat loop over n/32 sub-blocks. Scales are read once per
// sub-block, never per element.
//
//   row : pointer to the start of the quantized row
//   s   : sub-block index within the row
//   x   : the 32 activations aligned with that sub-block
// ---------------------------------------------------------------------------

__device__ static inline float sub32_q4_K(const uint8_t * row, int s, const float * x) {
    const int i = s >> 3;            // super-block (256 elems)
    const int j = s & 7;             // sub-block within it
    const uint8_t * b  = row + (size_t) i * 144;
    const float d      = h2f(b);
    const float dmin   = h2f(b + 2);
    const uint8_t * sc = b + 4;
    const uint8_t * qs = b + 16;

    uint8_t s6, m6;
    get_scale_min_k4(j, sc, s6, m6);

    const uint8_t * q = qs + (j >> 1) * 32;
    const bool hi = (j & 1) != 0;

    float sxq = 0.f, sx = 0.f;
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
        const float nib = (float) (hi ? (q[l] >> 4) : (q[l] & 0xF));
        sxq += x[l] * nib;
        sx  += x[l];
    }
    return d * (float) s6 * sxq - dmin * (float) m6 * sx;
}

__device__ static inline float sub32_q5_K(const uint8_t * row, int s, const float * x) {
    const int i = s >> 3;
    const int j = s & 7;
    const uint8_t * b  = row + (size_t) i * 176;
    const float d      = h2f(b);
    const float dmin   = h2f(b + 2);
    const uint8_t * sc = b + 4;
    const uint8_t * qh = b + 16;     // 32 bytes, one bit per element per pass
    const uint8_t * ql = b + 48;     // 128 bytes

    uint8_t s6, m6;
    get_scale_min_k4(j, sc, s6, m6);

    const int k = j >> 1;            // which 64-element pass
    const uint8_t * q = ql + k * 32;
    const bool hi = (j & 1) != 0;
    const uint8_t umask = (uint8_t) ((hi ? 2 : 1) << (2 * k));

    float sxq = 0.f, sx = 0.f;
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
        float nib = (float) (hi ? (q[l] >> 4) : (q[l] & 0xF));
        if (qh[l] & umask) nib += 16.f;
        sxq += x[l] * nib;
        sx  += x[l];
    }
    return d * (float) s6 * sxq - dmin * (float) m6 * sx;
}

__device__ static inline float sub32_q5_1(const uint8_t * row, int s, const float * x) {
    const uint8_t * b = row + (size_t) s * 24;
    const float d = h2f(b);
    const float m = h2f(b + 2);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    const uint8_t * qs = b + 8;

    float sxq = 0.f, sx = 0.f;
    #pragma unroll
    for (int l = 0; l < 16; ++l) {
        const uint8_t xh0 = (uint8_t) (((qh >> l) << 4) & 0x10);
        const uint8_t xh1 = (uint8_t) ((qh >> (l + 12)) & 0x10);
        const float q0 = (float) ((qs[l] & 0xF) | xh0);
        const float q1 = (float) ((qs[l] >>  4) | xh1);
        sxq += x[l] * q0 + x[l + 16] * q1;
        sx  += x[l] + x[l + 16];
    }
    return d * sxq + m * sx;
}

__device__ static inline float sub32_q8_0(const uint8_t * row, int s, const float * x) {
    const uint8_t * b = row + (size_t) s * 34;
    const float d = h2f(b);
    const int8_t * qs = (const int8_t *) (b + 2);
    float sxq = 0.f;
    #pragma unroll
    for (int l = 0; l < 32; ++l) sxq += x[l] * (float) qs[l];
    return d * sxq;
}

__device__ static inline float sub32(int qt, const uint8_t * row, int s, const float * x) {
    switch (qt) {
        case SB_Q4_K: return sub32_q4_K(row, s, x);
        case SB_Q5_K: return sub32_q5_K(row, s, x);
        case SB_Q5_1: return sub32_q5_1(row, s, x);
        case SB_Q8_0: return sub32_q8_0(row, s, x);
    }
    return 0.f;
}

// ---------------------------------------------------------------------------
// block-wide reduction (NT threads, NT a multiple of 32)
// ---------------------------------------------------------------------------
#define NT 128
#define NWARP (NT / 32)

__device__ static inline float block_sum(float v, float * sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = 0.f;
        #pragma unroll
        for (int i = 0; i < NWARP; ++i) r += sh[i];
        sh[NWARP] = r;
    }
    __syncthreads();
    return sh[NWARP];
}

// ---------------------------------------------------------------------------
// gate/up kernel:  h[e][j] = silu(<Wgate[e][j], x>) * <Wup[e][j], x>
//   grid  = (ff, n_used)
//   block = NT threads reducing over d_model
// ---------------------------------------------------------------------------
__global__ void k_gate_up(const float * __restrict__ x,
                          float * __restrict__ h,
                          const void * const * __restrict__ gate,
                          const void * const * __restrict__ up,
                          int d_model, int ff, int qt_gate, int qt_up) {
    __shared__ float sh[NWARP + 1];
    const int j = blockIdx.x;                 // output row
    const int e = blockIdx.y;                 // which selected expert

    const size_t rbg = sb_row_bytes(qt_gate, d_model);
    const size_t rbu = sb_row_bytes(qt_up,   d_model);
    const uint8_t * rg = (const uint8_t *) gate[e] + (size_t) j * rbg;
    const uint8_t * ru = (const uint8_t *) up[e]   + (size_t) j * rbu;

    const int nsub = d_model / 32;
    float ag = 0.f, au = 0.f;
    for (int s = threadIdx.x; s < nsub; s += NT) {
        const float * xs = x + s * 32;
        ag += sub32(qt_gate, rg, s, xs);
        au += sub32(qt_up,   ru, s, xs);
    }
    const float g = block_sum(ag, sh);
    __syncthreads();
    const float u = block_sum(au, sh);

    if (threadIdx.x == 0) {
        const float silu = g / (1.f + __expf(-g));
        h[(size_t) e * ff + j] = silu * u;
    }
}

// ---------------------------------------------------------------------------
// down kernel:  out[i] = sum_e  w[e] * <Wdown[e][i], h[e]>
//   grid  = (d_model)
//   block = NT threads
// One block owns one output element across all experts, so the weighted sum
// needs no atomics and is summed in a deterministic expert order.
// ---------------------------------------------------------------------------
__global__ void k_down(const float * __restrict__ h,
                       float * __restrict__ out,
                       const void * const * __restrict__ down,
                       const float * __restrict__ w,
                       int d_model, int ff, int n_used, int qt_down,
                       int accumulate) {
    __shared__ float sh[NWARP + 1];
    const int i = blockIdx.x;
    const size_t rb = sb_row_bytes(qt_down, ff);
    const int nsub = ff / 32;

    float total = 0.f;
    for (int e = 0; e < n_used; ++e) {
        const uint8_t * rd = (const uint8_t *) down[e] + (size_t) i * rb;
        const float * he = h + (size_t) e * ff;
        float a = 0.f;
        for (int s = threadIdx.x; s < nsub; s += NT) a += sub32(qt_down, rd, s, he + s * 32);
        const float v = block_sum(a, sh);
        __syncthreads();
        total += w[e] * v;
    }
    if (threadIdx.x == 0) out[i] = accumulate ? out[i] + total : total;
}

// Generic GGML MUL_MAT_ID path used by the optional llama.cpp integration.
// The host graph keeps the authoritative ids and this kernel receives one
// resident device pointer per (token, selected-expert) output slice.  No id is
// rewritten, filtered or substituted.
__global__ void k_mul_mat_id(const void * const * __restrict__ weights,
                             const float * __restrict__ x,
                             float * __restrict__ dst,
                             int rows, int k, int n_ids, int n_tokens,
                             size_t row_stride,
                             size_t x_id_stride, int x_rows, size_t x_token_stride,
                             size_t dst_id_stride, size_t dst_token_stride,
                             int qt) {
    __shared__ float sh[NWARP + 1];

    const int row = (int) blockIdx.x;
    const int id_i = (int) blockIdx.y;
    const int token = (int) blockIdx.z;
    if (row >= rows || id_i >= n_ids || token >= n_tokens) {
        return;
    }

    const void * expert = weights[(size_t) token * n_ids + id_i];
    const uint8_t * row_ptr = (const uint8_t *) expert + (size_t) row * row_stride;
    // ggml MUL_MAT_ID: selected slot id_i reads input row (id_i % ne11) of
    // token `token`. gate/up broadcast one hidden state (ne11 == 1); the down
    // projection has one SwiGLU activation PER slot (ne11 == n_used).
#if SABAH_SENTINEL == 2
    const int x_row = x_rows > 1 ? (id_i + 1) % x_rows : 0;
#else
    const int x_row = id_i % x_rows;
#endif
#if SABAH_SENTINEL == 3
    const int x_tok = (token + 1) % n_tokens;
#else
    const int x_tok = token;
#endif
    const float * x_ptr = (const float *) ((const uint8_t *) x
                        + (size_t) x_row * x_id_stride
                        + (size_t) x_tok * x_token_stride);

    const int nsub = k / 32;
    float partial = 0.0f;
    for (int s = threadIdx.x; s < nsub; s += NT) {
        partial += sub32(qt, row_ptr, s, x_ptr + (size_t) s * 32);
    }
    const float value = block_sum(partial, sh);
    if (threadIdx.x == 0) {
        uint8_t * out_ptr = (uint8_t *) dst + (size_t) token * dst_token_stride
                          + (size_t) id_i * dst_id_stride
                          + (size_t) row * sizeof(float);
        *(float *) out_ptr = value;
    }
}

// ---------------------------------------------------------------------------
// standalone dequantizer - used only to prove the kernels above read the
// formats correctly, by diffing against gguf-py.
//
// It recovers element k of a sub-block by dotting that sub-block with the k-th
// unit vector, so it exercises the SAME code path the matvec kernels use. A
// separate dequant implementation would prove nothing about them.
// ---------------------------------------------------------------------------
__global__ void k_dequant(const uint8_t * __restrict__ src,
                          float * __restrict__ dst,
                          int qt, int n_rows, int row_elems) {
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const uint8_t * row = src + (size_t) r * sb_row_bytes(qt, row_elems);
    float * o = dst + (size_t) r * row_elems;
    const int nsub = row_elems / 32;
    for (int s = threadIdx.x; s < nsub; s += blockDim.x) {
        float unit[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) unit[k] = 0.f;
        for (int k = 0; k < 32; ++k) {
            unit[k] = 1.f;
            o[s * 32 + k] = sub32(qt, row, s, unit);
            unit[k] = 0.f;
        }
    }
}

// ===========================================================================
// C API
// ===========================================================================
extern "C" {

SABAH_API const char * sabah_rt_last_error(void) { return g_err; }

static int sb_snapshot_json(char * buf, size_t n);

static void sb_report_diag() {
    if (!std::getenv("SABAH_LLAMA_TRACE")) return;
    std::fprintf(stderr,
                 "SABAH_DIAG calls=%llu tokens=%llu lookups=%llu overflow=%llu verify_ok=%llu verify_fail=%llu\n",
                 (unsigned long long) g_mmid_calls, (unsigned long long) g_mmid_tokens,
                 (unsigned long long) (g_mmid_hits + g_mmid_misses),
                 (unsigned long long) g_mmid_overflow,
                 (unsigned long long) g_mmid_verify_ok, (unsigned long long) g_mmid_verify_fail);
    char buf[1024];
    sb_snapshot_json(buf, sizeof(buf));
    std::fprintf(stderr, "SABAH_SNAPSHOT %s\n", buf);
}

SABAH_API int sabah_rt_init(int device) {
    if (!ck(cudaSetDevice(device), "cudaSetDevice")) return -1;
    if (!ck(cudaStreamCreate(&g_copy),    "create copy stream"))    return -1;
    if (!ck(cudaStreamCreate(&g_compute), "create compute stream")) return -1;
    sb_mmid_reset();
    static bool registered = false;
    if (!registered) { std::atexit(sb_report_diag); registered = true; }
    return 0;
}

SABAH_API int sabah_rt_shutdown(void) {
    sb_mmid_reset();
    if (g_copy)    { cudaStreamDestroy(g_copy);    g_copy = 0; }
    if (g_compute) { cudaStreamDestroy(g_compute); g_compute = 0; }
    return 0;
}

SABAH_API int sabah_rt_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

SABAH_API void * sabah_dev_alloc(size_t bytes) {
    void * p = 0;
    if (!ck(cudaMalloc(&p, bytes), "cudaMalloc")) return 0;
    return p;
}
SABAH_API void sabah_dev_free(void * p) { if (p) cudaFree(p); }

SABAH_API void * sabah_host_alloc(size_t bytes) {
    void * p = 0;
    if (!ck(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), "cudaHostAlloc")) return 0;
    return p;
}
SABAH_API void sabah_host_free(void * p) { if (p) cudaFreeHost(p); }

SABAH_API int sabah_dev_mem(size_t * freeb, size_t * totalb) {
    return ck(cudaMemGetInfo(freeb, totalb), "cudaMemGetInfo") ? 0 : -1;
}

SABAH_API int sabah_memcpy_h2d(void * dst, const void * src, size_t n) {
    return ck(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice), "memcpy h2d") ? 0 : -1;
}
SABAH_API int sabah_memcpy_d2h(void * dst, const void * src, size_t n) {
    return ck(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost), "memcpy d2h") ? 0 : -1;
}
SABAH_API int sabah_memset_d(void * dst, int v, size_t n) {
    return ck(cudaMemset(dst, v, n), "memset") ? 0 : -1;
}

// ---- async transfer on the copy stream ------------------------------------
SABAH_API int sabah_h2d_async(void * dst, const void * src, size_t n) {
    return ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, g_copy),
              "memcpyAsync h2d") ? 0 : -1;
}

SABAH_API void * sabah_event_create(void) {
    cudaEvent_t ev;
    if (!ck(cudaEventCreate(&ev), "eventCreate")) return 0;
    return (void *) ev;
}
SABAH_API void sabah_event_destroy(void * ev) { if (ev) cudaEventDestroy((cudaEvent_t) ev); }
SABAH_API int  sabah_event_record_copy(void * ev) {
    return ck(cudaEventRecord((cudaEvent_t) ev, g_copy), "record copy") ? 0 : -1;
}
SABAH_API int  sabah_event_record_compute(void * ev) {
    return ck(cudaEventRecord((cudaEvent_t) ev, g_compute), "record compute") ? 0 : -1;
}
SABAH_API int  sabah_compute_wait_event(void * ev) {
    return ck(cudaStreamWaitEvent(g_compute, (cudaEvent_t) ev, 0), "streamWaitEvent") ? 0 : -1;
}
SABAH_API float sabah_event_elapsed_ms(void * a, void * b) {
    float ms = -1.f;
    cudaEventSynchronize((cudaEvent_t) b);
    if (!ck(cudaEventElapsedTime(&ms, (cudaEvent_t) a, (cudaEvent_t) b), "elapsed")) return -1.f;
    return ms;
}
SABAH_API int sabah_event_sync(void * ev) {
    return ck(cudaEventSynchronize((cudaEvent_t) ev), "eventSynchronize") ? 0 : -1;
}
SABAH_API int sabah_event_query(void * ev) {
    cudaError_t e = cudaEventQuery((cudaEvent_t) ev);
    if (e == cudaSuccess)       return 1;   // complete
    if (e == cudaErrorNotReady) return 0;   // still in flight
    ck(e, "eventQuery");
    return -1;
}
SABAH_API int sabah_sync_copy(void)    { return ck(cudaStreamSynchronize(g_copy),    "sync copy")    ? 0 : -1; }
SABAH_API int sabah_sync_compute(void) { return ck(cudaStreamSynchronize(g_compute), "sync compute") ? 0 : -1; }
SABAH_API int sabah_sync_all(void)     { return ck(cudaDeviceSynchronize(), "sync device") ? 0 : -1; }

// ---- execution ------------------------------------------------------------
//
// d_ptrs is a device-resident array laid out as
//   [ gate_0..gate_{n-1}, up_0..up_{n-1}, down_0..down_{n-1} ]
// and d_w the routing weights. Both live on the device so a block costs no
// host round-trip.
//
SABAH_API int sabah_moe_block(const float * d_x, float * d_out, float * d_h,
                              const void * const * d_ptrs, const float * d_w,
                              int d_model, int ff, int n_used,
                              int qt_gate, int qt_up, int qt_down,
                              int accumulate) {
    if ((d_model % 32) || (ff % 32)) {
        snprintf(g_err, sizeof(g_err), "d_model and ff must be multiples of 32");
        return -1;
    }
    dim3 g1((unsigned) ff, (unsigned) n_used);
    k_gate_up<<<g1, NT, 0, g_compute>>>(d_x, d_h, d_ptrs, d_ptrs + n_used,
                                        d_model, ff, qt_gate, qt_up);
    dim3 g2((unsigned) d_model);
    k_down<<<g2, NT, 0, g_compute>>>(d_h, d_out, d_ptrs + 2 * n_used, d_w,
                                     d_model, ff, n_used, qt_down, accumulate);
    return ck(cudaGetLastError(), "moe_block launch") ? 0 : -1;
}

// Exact host-backed GGML MUL_MAT_ID adapter.  `source` is the original
// read-only GGUF tensor, not a compacted or rewritten expert bank.  `stream`
// is llama.cpp's active CUDA compute stream; the copy stream is owned by
// Sabah and joined with an event before the kernel launch.
// Live counters for processes that are never shut down cleanly (servers).
// With SABAH_LLAMA_STATUS_FILE set, the counters are rewritten at most every
// 250 ms, so an operator can see that Sabah is really executing expert ops.
// Counters as one JSON object. Every field is read under the residency lock,
// so a snapshot taken after a call returns is exact, not sampled.
static int sb_snapshot_json(char * buf, size_t n) {
    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    char slots[256];
    int o = 0;
    for (int i = 0; i < 16 && o < (int) sizeof(slots) - 24; ++i) {
        o += snprintf(slots + o, sizeof(slots) - o, "%s%llu", i ? "," : "",
                      (unsigned long long) g_sc_slot[i]);
    }
    return snprintf(buf, n,
        "{\"calls\":%llu,\"empty_calls\":%llu,\"tokens\":%llu,\"lookups\":%llu,\"hits\":%llu,"
        "\"misses\":%llu,\"evictions\":%llu,\"bytes_fetched\":%llu,\"resident_bytes\":%llu,"
        "\"overflow\":%llu,\"verify_ok\":%llu,\"verify_fail\":%llu,"
        "\"selfcheck_k\":%d,\"selfcheck_ok\":%llu,\"selfcheck_fail\":%llu,\"selfcheck_max_rel_l2\":%.6e,"
        "\"selfcheck_multi_token\":%llu,\"selfcheck_tensors\":%llu,\"selfcheck_slots\":[%s],"
        "\"sentinel\":%d}",
        (unsigned long long) g_mmid_calls, (unsigned long long) g_mmid_empty_calls,
        (unsigned long long) g_mmid_tokens,
        (unsigned long long) (g_mmid_hits + g_mmid_misses), (unsigned long long) g_mmid_hits,
        (unsigned long long) g_mmid_misses, (unsigned long long) g_mmid_evictions,
        (unsigned long long) g_mmid_bytes_fetched, (unsigned long long) g_mmid_bytes,
        (unsigned long long) g_mmid_overflow, (unsigned long long) g_mmid_verify_ok,
        (unsigned long long) g_mmid_verify_fail, g_sc_k, (unsigned long long) g_sc_ok,
        (unsigned long long) g_sc_fail, g_sc_max, (unsigned long long) g_sc_multi_token,
        (unsigned long long) g_sc_tensors.size(), slots, SABAH_SENTINEL);
}

// With SABAH_LLAMA_STATUS_FILE set, the counters are rewritten after EVERY
// call, before the call returns, so a snapshot read after a request completes
// is exact (evidence grade). Cost: one small file write per MUL_MAT_ID.
static void sb_status_write(bool) {
    static const char * path = std::getenv("SABAH_LLAMA_STATUS_FILE");
    if (!path || !*path) return;
    char buf[1024];
    sb_snapshot_json(buf, sizeof(buf));
    FILE * f = std::fopen(path, "wb");
    if (!f) return;
    std::fputs(buf, f);
    std::fputc('\n', f);
    std::fclose(f);
}

// Float64 recomputation of K sampled (token, slot) outputs of the call that
// just ran, from the ORIGINAL host GGUF bytes and the exact device input row.
// It checks the mapping the kernel must honour: output (t, i) must equal
// W[ids(t,i)] . x(t, i mod x_rows). Samples are chosen by a hash of the call
// counter only, never by observed error.
static void sb_selfcheck(const void * source, size_t expert_stride, size_t row_stride,
                         const void * d_x, size_t x_id_stride, int x_rows, size_t x_token_stride,
                         const std::vector<int32_t> & ids, int n_ids, int n_tokens,
                         const float * d_dst, size_t dst_id_stride, size_t dst_token_stride,
                         int rows, int k, int qt, cudaStream_t stream, uint64_t call_no) {
    if (g_sc_k <= 0) return;
    cudaStreamSynchronize(stream);
    std::vector<float> x(k), out(rows), w(k);
    const int n_pairs = n_ids * n_tokens;
    const int samples = g_sc_k < n_pairs ? g_sc_k : n_pairs;
    const uint64_t h0 = sb_mix64(call_no * 0x100000001B3ull + 0x5AB4ull);
    for (int s = 0; s < samples; ++s) {
        const uint64_t h = sb_mix64(h0 + (uint64_t) s);
        const int t = (int) (h % (uint64_t) n_tokens);
        const int slot = (int) ((sb_mix64(h) + (uint64_t) s) % (uint64_t) n_ids);
        const int expert = ids[(size_t) t * n_ids + slot];
        const uint8_t * xp = (const uint8_t *) d_x + (size_t) (slot % x_rows) * x_id_stride
                           + (size_t) t * x_token_stride;
        const uint8_t * op = (const uint8_t *) d_dst + (size_t) t * dst_token_stride
                           + (size_t) slot * dst_id_stride;
        if (cudaMemcpy(x.data(), xp, (size_t) k * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(out.data(), op, (size_t) rows * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::lock_guard<std::mutex> lock(g_mmid_mutex);
            ++g_sc_fail;
            continue;
        }
        const uint8_t * wexp = (const uint8_t *) source + (size_t) expert * expert_stride;
        double num = 0.0, den = 0.0;
        for (int r = 0; r < rows; ++r) {
            sb_host_dequant(wexp + (size_t) r * row_stride, w.data(), qt, k);
            double acc = 0.0;
            for (int c = 0; c < k; ++c) acc += (double) w[c] * (double) x[c];
            const double e = (double) out[r] - acc;
            num += e * e;
            den += acc * acc;
        }
        const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
        std::lock_guard<std::mutex> lock(g_mmid_mutex);
        if (rel <= g_sc_tol) {
            ++g_sc_ok;
        } else {
            if (g_sc_fail == 0) {
                std::fprintf(stderr, "SABAH_SELFCHECK_FAIL call=%llu token=%d/%d slot=%d expert=%d rel_l2=%.3e\n",
                             (unsigned long long) call_no, t, n_tokens, slot, expert, rel);
            }
            ++g_sc_fail;
        }
        if (rel > g_sc_max) g_sc_max = rel;
        if (slot < 16) ++g_sc_slot[slot];
        if (n_tokens > 1) ++g_sc_multi_token;
        if (std::find(g_sc_tensors.begin(), g_sc_tensors.end(), source) == g_sc_tensors.end()) {
            g_sc_tensors.push_back(source);
        }
    }
}

SABAH_API int sabah_rt_mul_mat_id_v2(
        const void * source,
        size_t expert_stride,
        size_t row_stride,
        const void * d_x,
        size_t x_id_stride,
        int x_rows,
        size_t x_token_stride,
        const void * d_ids,
        size_t ids_id_stride,
        size_t ids_token_stride,
        int n_ids,
        int n_tokens,
        float * d_dst,
        size_t dst_id_stride,
        size_t dst_token_stride,
        int rows,
        int k,
        int n_experts,
        int qt,
        void * stream_ptr) {
    if (!source || !d_x || !d_ids || !d_dst || !g_copy) {
        snprintf(g_err, sizeof(g_err), "invalid Sabah MUL_MAT_ID arguments");
        return -1;
    }
    if (n_tokens == 0) {
        // A graph may legitimately route zero token rows through an expert op
        // (e.g. the last block of an ubatch that produces no outputs). There
        // is nothing to compute; count it and return.
        std::lock_guard<std::mutex> lock(g_mmid_mutex);
        ++g_mmid_empty_calls;
        return 0;
    }
    if (k <= 0 || rows <= 0 || n_ids <= 0 || n_tokens < 0 || n_experts <= 0 || (k % 32) != 0) {
        snprintf(g_err, sizeof(g_err), "invalid Sabah MUL_MAT_ID geometry");
        return -1;
    }
    if (x_rows <= 0 || (x_rows != 1 && x_rows != n_ids)) {
        snprintf(g_err, sizeof(g_err),
                 "Sabah MUL_MAT_ID input rows %d must be 1 (broadcast) or n_ids %d", x_rows, n_ids);
        return -1;
    }
    uint64_t call_no;
    {
        std::lock_guard<std::mutex> lock(g_mmid_mutex);
        ++g_mmid_epoch;
        call_no = ++g_mmid_calls;
        g_mmid_tokens += (uint64_t) n_tokens;
    }
    if (sb_block_bytes(qt) == 0 || row_stride < sb_row_bytes(qt, k)) {
        snprintf(g_err, sizeof(g_err), "unsupported Sabah MUL_MAT_ID quantized row");
        return -1;
    }

    cudaStream_t compute_stream = stream_ptr ? (cudaStream_t) stream_ptr : g_compute;

    const size_t ids_bytes = (size_t) (n_ids - 1) * ids_id_stride
                           + (size_t) (n_tokens - 1) * ids_token_stride
                           + sizeof(int32_t);
    std::vector<uint8_t> ids_raw(ids_bytes);
    if (!ck(cudaMemcpyAsync(ids_raw.data(), d_ids, ids_bytes,
                            cudaMemcpyDeviceToHost, compute_stream),
            "Sabah router id download")) {
        return -1;
    }
    if (!ck(cudaStreamSynchronize(compute_stream), "Sabah router id synchronize")) {
        return -1;
    }

    std::vector<int32_t> ids((size_t) n_ids * n_tokens);
    std::vector<const void *> ptrs(ids.size());
    for (int token = 0; token < n_tokens; ++token) {
        for (int id_i = 0; id_i < n_ids; ++id_i) {
            const size_t raw_offset = (size_t) token * ids_token_stride
                                    + (size_t) id_i * ids_id_stride;
            const int expert = *(const int32_t *) (ids_raw.data() + raw_offset);
            ids[(size_t) token * n_ids + id_i] = expert;
            const size_t i = (size_t) token * n_ids + id_i;
            if (expert < 0 || expert >= n_experts) {
                snprintf(g_err, sizeof(g_err), "Sabah expert id out of range");
                return -1;
            }
#if SABAH_SENTINEL == 1
            // TEST-ONLY BUILD. Never defined for a shipped library: it exists
            // so the correctness gate can prove it rejects a wrong expert in
            // the real graph. Slot 0 of every call runs the neighbouring expert.
            if (id_i == 0) {
                ids[(size_t) token * n_ids + id_i] = expert;
                void * wrong = sb_mmid_resident(
                    source, (expert + 1) % n_experts,
                    (const uint8_t *) source + (size_t) ((expert + 1) % n_experts) * expert_stride,
                    expert_stride, compute_stream);
                if (!wrong) return -1;
                ptrs[(size_t) token * n_ids + id_i] = wrong;
                continue;
            }
#endif
            void * resident = sb_mmid_resident(
                source, expert,
                (const uint8_t *) source + (size_t) expert * expert_stride,
                expert_stride, compute_stream);
            if (!resident) {
                return -1;
            }
            ptrs[i] = resident;
        }
    }

    void ** d_ptrs = nullptr;
    if (!ck(cudaMalloc(&d_ptrs, ptrs.size() * sizeof(void *)),
            "Sabah pointer table allocation")) {
        return -1;
    }
    if (!ck(cudaMemcpyAsync(d_ptrs, ptrs.data(), ptrs.size() * sizeof(void *),
                            cudaMemcpyHostToDevice, g_copy),
            "Sabah pointer table upload")) {
        cudaFree(d_ptrs);
        return -1;
    }

    cudaEvent_t ready = nullptr;
    if (!ck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
            "Sabah copy event")) {
        cudaFree(d_ptrs);
        return -1;
    }
    if (!ck(cudaEventRecord(ready, g_copy), "Sabah copy event record") ||
        !ck(cudaStreamWaitEvent(compute_stream, ready, 0), "Sabah compute wait")) {
        cudaEventDestroy(ready);
        cudaFree(d_ptrs);
        return -1;
    }

    dim3 grid((unsigned) rows, (unsigned) n_ids, (unsigned) n_tokens);
    k_mul_mat_id<<<grid, NT, 0, compute_stream>>>(
        (const void * const *) d_ptrs,
        (const float *) d_x,
        d_dst,
        rows, k, n_ids, n_tokens,
        row_stride, x_id_stride, x_rows, x_token_stride,
        dst_id_stride, dst_token_stride, qt);
    if (!ck(cudaGetLastError(), "Sabah MUL_MAT_ID launch")) {
        cudaEventDestroy(ready);
        cudaFree(d_ptrs);
        return -1;
    }

    // The event is no longer needed after the wait has been enqueued. CUDA
    // permits deferred destruction while dependent work is in flight.
    cudaEventDestroy(ready);
    cudaFree(d_ptrs);
    sb_selfcheck(source, expert_stride, row_stride, d_x, x_id_stride, x_rows, x_token_stride,
                 ids, n_ids, n_tokens, d_dst, dst_id_stride, dst_token_stride,
                 rows, k, qt, compute_stream, call_no);
    sb_status_write(false);
    return 0;
}

SABAH_API int sabah_rt_snapshot(char * buf, size_t n) {
    return sb_snapshot_json(buf, n);
}

SABAH_API const char * sabah_rt_build_info(void) {
    static char info[128];
    snprintf(info, sizeof(info), "sabah_rt abi=v2 sentinel=%d", SABAH_SENTINEL);
    return info;
}

SABAH_API int sabah_rt_host_dequant(const void * src, float * dst, int qt, int n) {
    return sb_host_dequant((const uint8_t *) src, dst, qt, n) ? 0 : -1;
}

SABAH_API int sabah_rt_get_diag(
        unsigned long long * calls,
        unsigned long long * tokens,
        unsigned long long * overflow,
        unsigned long long * verify_ok,
        unsigned long long * verify_fail) {
    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    if (calls)       *calls = g_mmid_calls;
    if (tokens)      *tokens = g_mmid_tokens;
    if (overflow)    *overflow = g_mmid_overflow;
    if (verify_ok)   *verify_ok = g_mmid_verify_ok;
    if (verify_fail) *verify_fail = g_mmid_verify_fail;
    return 0;
}

SABAH_API int sabah_rt_get_metrics(
        unsigned long long * hits,
        unsigned long long * misses,
        unsigned long long * evictions,
        unsigned long long * bytes_fetched,
        unsigned long long * resident_bytes) {
    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    if (hits)           *hits = g_mmid_hits;
    if (misses)         *misses = g_mmid_misses;
    if (evictions)      *evictions = g_mmid_evictions;
    if (bytes_fetched)  *bytes_fetched = g_mmid_bytes_fetched;
    if (resident_bytes) *resident_bytes = g_mmid_bytes;
    return 0;
}

SABAH_API int sabah_dequant(const void * d_src, float * d_dst,
                            int qt, int n_rows, int row_elems) {
    k_dequant<<<n_rows, 64, 0, g_compute>>>((const uint8_t *) d_src, d_dst,
                                            qt, n_rows, row_elems);
    if (!ck(cudaGetLastError(), "dequant launch")) return -1;
    return ck(cudaStreamSynchronize(g_compute), "dequant sync") ? 0 : -1;
}

SABAH_API size_t sabah_row_bytes(int qt, int n) { return sb_row_bytes(qt, n); }

} // extern "C"
