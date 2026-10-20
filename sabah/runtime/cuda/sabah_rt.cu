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
#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  #include <psapi.h>
  #pragma comment(lib, "psapi.lib")
#else
  #include <fcntl.h>
  #include <unistd.h>
  #include <sys/stat.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <map>
#include <vector>
#include <condition_variable>
#include <deque>
#include <thread>
#include <utility>

// The IQ-family expert quants need ggml's grid/sign lookup tables. They are
// extracted verbatim from ggml-common.h by tools/gen_iq_tables.py; plain
// `static const`, so both the host reference dequantizer and the device
// sub-block kernels can read the same arrays (exactly as ggml does).
#include "sabah_iq_tables.h"

#if defined(_WIN32)
  #define SABAH_API __declspec(dllexport)
#else
  #define SABAH_API __attribute__((visibility("default")))
#endif

// ---- quant type codes (ggml enum values, so traces stay comparable) --------
// The four the UD-Q4_K_XL artifact uses...
#define SB_Q5_1      7
#define SB_Q8_0      8
#define SB_Q4_K     12
#define SB_Q5_K     13
// ...and the six the GSQ-RCO IQ3_S artifact uses.
#define SB_IQ3_XXS  18
#define SB_IQ4_NL   20
#define SB_IQ3_S    21
#define SB_IQ2_S    22
#define SB_IQ4_XS   23
#define SB_Q2_0     42

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
    bool slab;          // true: came from the hot-tier slab; false: over-budget
    bool filled;        // true: the expert bytes are already in `device`
};

// Per-phase cost of a MUL_MAT_ID call, in milliseconds, accumulated over the run.
// Reported via SABAH_LLAMA_TRACE so tuning is driven by measurement rather than
// by which part of the path looks expensive.
static double g_t_slot  = 0;   // residency lookup + allocation + eviction
static double g_t_fetch = 0;   // resolve + disk reads + host-to-device copies
static double g_t_wait  = 0;   // of which: blocked on the reads
static double g_t_ids   = 0;   // router id download and its stream synchronise
static double g_t_ptr   = 0;   // pointer table upload and copy/compute hand-off
static double g_t_prime = 0;   // residency snapshot: one table sweep per tensor

// Optional CUDA-event probe (SABAH_LLAMA_EVENTS=1). Attributes a slowdown to the
// kernels themselves (clock/width) or to the gaps around them (host submission,
// queue depth) without nsys, which cannot intercept this DLL. Every k_mul_mat_id
// launch is bracketed by an event pair; the pairs are read at the next router-id
// drain, which already guarantees they have completed. Reported per token.
struct sb_ev_pair {
    cudaEvent_t a, b;
    double host_ms;
    int nt;
    int rows;
};
static bool                 g_ev_enabled = false;
static bool                 g_ev_checked = false;
static double               g_ev_kernel_ms = 0;   // device time inside the kernels (decode rows only)
static double               g_ev_launch_ms = 0;   // host time to submit each launch (decode rows only)
static uint64_t             g_ev_calls = 0;
static double               g_ev_kernel_ms_all = 0; // every call, for reference
static double               g_ev_launch_ms_all = 0;
static uint64_t             g_ev_calls_all = 0;
static double               g_ev_kernel_ms_gu = 0;  // decode gate/up calls (rows < 1024)
static uint64_t             g_ev_calls_gu = 0;
static double               g_ev_kernel_ms_dn = 0;  // decode down calls (rows >= 1024)
static uint64_t             g_ev_calls_dn = 0;
static std::vector<sb_ev_pair> g_ev_pending;
static std::vector<sb_ev_pair> g_ev_pool;

static bool sb_ev_on() {
    if (!g_ev_checked) {
        g_ev_checked = true;
        const char * v = std::getenv("SABAH_LLAMA_EVENTS");
        g_ev_enabled = v != nullptr && v[0] != 0 && v[0] != '0';
    }
    return g_ev_enabled;
}

struct sb_phase_timer {
    std::chrono::steady_clock::time_point t0;
    double * acc;
    explicit sb_phase_timer(double * a) : t0(std::chrono::steady_clock::now()), acc(a) {}
    ~sb_phase_timer() {
        *acc += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
    }
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
// Slots reserved on behalf of the tensor that follows this call in the same
// layer, so that a layer's three tensors share one disk wave (merged fetch).
static uint64_t g_mp_reserved = 0;
// Diagnostics. `epoch` identifies the MUL_MAT_ID call in progress: a slot whose
// pointer has already been handed to that call must not be evicted before the
// call's kernel has run, whatever the capacity says.
static uint64_t g_mmid_epoch = 0;
// The router ids of one layer's three expert tensors are the same tensor, so the
// downloads after the first of a group are redundant. Reuse is allowed only
// while the ids tensor is unchanged and the group stays this short; that bound
// is also the width of the eviction-protection window above.
static constexpr uint64_t SB_ID_GROUP_MAX = 4;
static uint64_t g_mmid_calls = 0;
static uint64_t g_mmid_tokens = 0;
static uint64_t g_id_reuse = 0;       // calls that reused the previous call's ids
static uint64_t g_id_group_max = 0;   // longest reuse streak observed
static bool     g_id_check = false;   // SABAH_LLAMA_ID_CHECK=1: prove the reuse
static uint64_t g_id_check_ok = 0, g_id_check_fail = 0;
static uint64_t g_mmid_overflow = 0;   // allocations that exceeded capacity
static uint64_t g_mmid_verify_ok = 0;
static uint64_t g_mmid_verify_fail = 0;
static bool     g_mmid_verify = false;     // SABAH_LLAMA_VERIFY_BYTES=1 | fetch
static bool     g_mmid_verify_all_hits = false;
static uint64_t g_mmid_verify_hit_every = 64; // "fetch" mode: every Nth hit, by counter
static uint64_t g_mmid_verify_budget = UINT64_MAX; // how many checks may touch the mapped oracle
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

// Read a byte count written as a plain integer, in scientific notation, or with
// a binary multiplier suffix ("9800000000", "9.8e9", "9.8G", "9800MiB").
//
// This used to be a plain strtoull(..., 10), and strtoull stops at the first
// character it does not understand. The documented value "9.8e9" therefore
// parsed as 9 - nine bytes - and because 9 bytes is a valid (if useless) answer
// nothing complained. The hot tier then failed to hold even one expert, every
// miss fell through to a fresh cudaMalloc, and decode ran 4x slower than the
// documented figure with no error anywhere. A value that cannot be read whole
// is now refused, and a tier too small to hold more than a few experts is said
// out loud.
static bool sb_parse_size(const char * text, size_t & out) {
    if (text == nullptr || *text == '\0') return false;
    char * end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || !(value > 0)) return false;

    double multiplier = 1.0;
    if (*end != '\0') {
        switch (*end) {
            case 'k': case 'K': multiplier = 1024.0;                       break;
            case 'm': case 'M': multiplier = 1024.0 * 1024.0;              break;
            case 'g': case 'G': multiplier = 1024.0 * 1024.0 * 1024.0;     break;
            case 't': case 'T': multiplier = 1024.0 * 1024.0 * 1024.0 * 1024.0; break;
            default: return false;
        }
        const char * rest = end + 1;
        if (*rest == 'i' || *rest == 'I') ++rest;
        if (*rest == 'b' || *rest == 'B') ++rest;
        if (*rest != '\0') return false;
    }

    const double bytes = value * multiplier;
    if (!(bytes > 0) || bytes > 1.0e15) return false;
    out = (size_t) bytes;
    return true;
}

// Below this a hot tier holds a handful of experts, which is never what anyone
// meant: it is the shape of a value that did not parse the way it was written.
static const size_t SB_TIER_SUSPICIOUSLY_SMALL = (size_t) 256 * 1024 * 1024;

static size_t sb_mmid_capacity_bytes() {
    const char * env = std::getenv("SABAH_LLAMA_HOT_BYTES");
    if (env && *env) {
        size_t value = 0;
        if (sb_parse_size(env, value)) {
            if (value < SB_TIER_SUSPICIOUSLY_SMALL) {
                std::fprintf(stderr,
                             "SABAH_LLAMA: SABAH_LLAMA_HOT_BYTES=%s is only %llu bytes. "
                             "A tier that small holds almost nothing, so every expert miss "
                             "ends up as a fresh cudaMalloc and decode gets much slower. "
                             "Did you mean %s000000000?\n",
                             env, (unsigned long long) value, env);
                std::fflush(stderr);
            }
            return value;
        }
        std::fprintf(stderr,
                     "SABAH_LLAMA: cannot read SABAH_LLAMA_HOT_BYTES=%s. Use a plain byte "
                     "count or a binary suffix (9.8G, 9800M); using the default 2 GiB.\n",
                     env);
        std::fflush(stderr);
    }
    return (size_t) 2ull * 1024ull * 1024ull * 1024ull;
}

// ---------------- Device slab allocator -------------------------------------
//
// One cudaMalloc for the whole hot tier, subdivided on demand. The point is to
// remove cudaMalloc/cudaFree from the per-miss path: a miss turns into a
// best-fit search over a free list and a pointer offset, not an allocator call
// (and, more importantly, no free/malloc churn that can serialise the device).
//
// Every allocation is a sub-range [base+offset, base+offset+bytes). `free`
// returns the range and coalesces it with any adjacent free range, so the pool
// does not fragment under the miss/evict steady state.
static void * g_slab_base = nullptr;
static size_t g_slab_size = 0;
static std::vector<std::pair<size_t, size_t>> g_slab_free;   // (offset, bytes)
static std::map<void *, size_t> g_slab_alloc;                // ptr -> bytes
static std::mutex g_slab_mutex;

static void sb_slab_destroy() {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    if (g_slab_base) {
        cudaFree(g_slab_base);
        g_slab_base = nullptr;
    }
    g_slab_size = 0;
    g_slab_free.clear();
    g_slab_alloc.clear();
}

static void sb_slab_init(size_t capacity_bytes) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    if (g_slab_base) return;
    void * p = nullptr;
    if (!ck(cudaMalloc(&p, capacity_bytes), "sabah slab alloc")) return;
    g_slab_base = p;
    g_slab_size = capacity_bytes;
    g_slab_free.clear();
    g_slab_free.push_back({ 0, capacity_bytes });
    g_slab_alloc.clear();
}

static void * sb_slab_alloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    if (!g_slab_base) {
        // lazy init without re-locking
        size_t cap = sb_mmid_capacity_bytes();
        void * p = nullptr;
        if (!ck(cudaMalloc(&p, cap), "sabah slab alloc")) return nullptr;
        g_slab_base = p;
        g_slab_size = cap;
        g_slab_free.clear();
        g_slab_free.push_back({ 0, cap });
        g_slab_alloc.clear();
    }
    // best fit: smallest free range that still holds the request
    size_t best_i = g_slab_free.size(), best_sz = 0;
    for (size_t i = 0; i < g_slab_free.size(); ++i) {
        const size_t sz = g_slab_free[i].second;
        if (sz >= bytes && (best_i == g_slab_free.size() || sz < best_sz)) {
            best_i = i; best_sz = sz;
        }
    }
    if (best_i == g_slab_free.size()) return nullptr;   // pool exhausted
    const size_t off = g_slab_free[best_i].first;
    if (best_sz == bytes) {
        g_slab_free.erase(g_slab_free.begin() + (long) best_i);
    } else {
        g_slab_free[best_i] = { off + bytes, best_sz - bytes };
    }
    void * ptr = (char *) g_slab_base + off;
    g_slab_alloc[ptr] = bytes;
    return ptr;
}

static void sb_slab_free(void * ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    if (!g_slab_base) return;
    auto it = g_slab_alloc.find(ptr);
    if (it == g_slab_alloc.end()) return;         // not a slab pointer
    const size_t off = (size_t) ((char *) ptr - (char *) g_slab_base);
    const size_t sz = it->second;
    g_slab_alloc.erase(it);
    g_slab_free.push_back({ off, sz });
    std::sort(g_slab_free.begin(), g_slab_free.end());
    std::vector<std::pair<size_t, size_t>> merged;
    for (const auto & r : g_slab_free) {
        if (!merged.empty() && merged.back().first + merged.back().second == r.first) {
            merged.back().second += r.second;
        } else {
            merged.push_back(r);
        }
    }
    g_slab_free.swap(merged);
}

static void sb_mmid_reset() {
    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    // Entries are slab sub-ranges (or, if the slab never initialised, plain
    // cudaMalloc'd blocks). Return each to whichever allocator produced it.
    for (const auto & entry : g_mmid_entries) {
        if (!entry.device) continue;
        if (entry.slab) {
            if (g_slab_base) sb_slab_free(entry.device);
        } else {
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
    g_mp_reserved = 0;
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
    g_mmid_verify_budget = UINT64_MAX;
    if (const char * lim = std::getenv("SABAH_LLAMA_VERIFY_LIMIT")) {
        g_mmid_verify_budget = (uint64_t) std::strtoull(lim, nullptr, 10);
    }
    g_mmid_empty_calls = 0;
    g_id_reuse = 0;
    g_id_group_max = 0;
    g_id_check_ok = 0;
    g_id_check_fail = 0;
    {
        const char * ic = std::getenv("SABAH_LLAMA_ID_CHECK");
        g_id_check = ic && (std::strcmp(ic, "1") == 0 || std::strcmp(ic, "on") == 0);
    }
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
    // No stream synchronisation belongs here, and saying why matters because the
    // obvious "drain before reuse" is what this used to do - twice per eviction,
    // which at roughly seven hundred evictions per token cost more than the
    // expert reads themselves.
    //
    // Every entry that can be evicted here was created by an EARLIER MUL_MAT_ID
    // call. Each call begins by downloading the router ids on the compute stream
    // and synchronising that stream, so by the time this runs:
    //   * the previous calls' kernels have completed, and
    //   * each of those kernels had already waited on the copy-stream event that
    //     ordered its own host-to-device expert transfers,
    // so both the kernel and the transfer that touched a victim are finished.
    // Entries created by the call in progress are never victims: their pointers
    // are already in that call's pointer table and its kernel has not run yet.
    // If nothing else can be evicted the tier temporarily exceeds its budget
    // instead of producing a dangling pointer.
    // The router-id reuse keeps up to SB_ID_GROUP_MAX calls of one layer from
    // draining the compute stream, so kernels from the last epochs can still be
    // reading their slots. Their entries are held back as victims for as many
    // epochs as that window is wide; older ones are provably finished.
    (void) compute_stream;
    while (g_mmid_bytes + needed > g_mmid_capacity) {
        auto victim = g_mmid_entries.end();
        for (auto it = g_mmid_entries.begin(); it != g_mmid_entries.end(); ++it) {
            if (it->epoch + SB_ID_GROUP_MAX > g_mmid_epoch) continue;
            if (victim == g_mmid_entries.end() || it->touch < victim->touch) victim = it;
        }
        if (victim == g_mmid_entries.end()) {
            ++g_mmid_overflow;
            return;
        }
        if (victim->slab) {
            if (g_slab_base) sb_slab_free(victim->device);
        } else {
            cudaFree(victim->device);
        }
        g_mmid_bytes -= victim->bytes;
        // Order in the table is irrelevant: recency is `touch`. Swapping the last
        // element in avoids shifting the tens of thousands of entries behind the
        // victim, which cost more than the disk read it made room for.
        *victim = g_mmid_entries.back();
        g_mmid_entries.pop_back();
        ++g_mmid_evictions;
    }
}

// Byte-exact check of a resident slot against the original GGUF range. The
// oracle is the mapped file, so every check pulls those pages into RAM - the
// budget bounds that, otherwise a long prefill would fill the machine with the
// pages the verifier itself asked for.
//
// A verification failure also says *where* the served bytes came from, because a
// bare "differs" is not actionable. The origin of every pending fetch is kept
// while verification is on, so the mismatch line can name the file and offset.
struct sb_fetch_origin {
    const void * dst;
    size_t       file;
    uint64_t     off;
    size_t       bytes;
};
static std::mutex                   g_origin_mutex;
static std::vector<sb_fetch_origin> g_origin;
// The last object a verification rejected, so the error message the host sees can
// name the file and offset instead of only "differs".
static size_t                       g_verify_last_file = (size_t) -1;
static uint64_t                     g_verify_last_off  = 0;

// Defined after the disk-file table; used only to name a file in a diagnostic.
static const char * sb_disk_file_path(size_t i, uint64_t * size_out);

static bool sb_mmid_verify(const void * device, const void * source, size_t bytes) {
    if (g_mmid_verify_budget == 0) return true;   // budget spent: assert nothing
    if (g_mmid_verify_budget != UINT64_MAX) --g_mmid_verify_budget;
    std::vector<uint8_t> back(bytes);
    if (cudaStreamSynchronize(g_copy) != cudaSuccess ||
        cudaMemcpy(back.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        ++g_mmid_verify_fail;
        return false;
    }
    if (std::memcmp(back.data(), source, bytes) != 0) {
        ++g_mmid_verify_fail;
        // How they differ is the whole diagnosis: a mismatch at byte ~0 with the
        // rest matching is a placement error, a contiguous run at the end is a
        // short read (the staging buffer's stale tail), and mismatches scattered
        // through the whole object is the wrong object.
        const uint8_t * a = back.data();
        const uint8_t * b = (const uint8_t *) source;
        size_t nmis = 0, firsts[8], nf = 0, runs = 0, prev = (size_t) -1;
        for (size_t i = 0; i < bytes; ++i) {
            if (a[i] != b[i]) {
                ++nmis;
                if (prev == (size_t) -1 || i != prev + 1) ++runs;
                prev = i;
                if (nf < 8) firsts[nf++] = i;
            }
        }
        char pos[8 * 12 + 1];
        size_t po = 0;
        for (size_t i = 0; i < nf; ++i) {
            po += (size_t) snprintf(pos + po, sizeof(pos) - po, "%s%llu",
                                    i ? "," : "", (unsigned long long) firsts[i]);
        }
        const char * where = "unresolved";
        uint64_t fsize = 0;
        g_verify_last_file = (size_t) -1;
        g_verify_last_off  = 0;
        {
            std::lock_guard<std::mutex> lock(g_origin_mutex);
            for (const auto & o : g_origin) {
                if (o.dst == device) {
                    g_verify_last_file = o.file;
                    g_verify_last_off  = o.off;
                    std::fprintf(stderr,
                                 "SABAH_VERIFY_MISMATCH bytes=%llu differing=%llu runs=%llu first=%s\n"
                                 "  read %llu bytes at offset %llu of file #%llu (%s, size %llu),"
                                 " source=%p\n",
                                 (unsigned long long) bytes, (unsigned long long) nmis,
                                 (unsigned long long) runs, pos,
                                 (unsigned long long) o.bytes, (unsigned long long) o.off,
                                 (unsigned long long) o.file, sb_disk_file_path(o.file, &fsize),
                                 (unsigned long long) fsize, source);
                    std::fflush(stderr);
                    return false;
                }
            }
        }
        std::fprintf(stderr,
                     "SABAH_VERIFY_MISMATCH bytes=%llu differing=%llu runs=%llu first=%s (%s)\n",
                     (unsigned long long) bytes, (unsigned long long) nmis,
                     (unsigned long long) runs, pos, where);
        std::fflush(stderr);
        return false;
    }
    ++g_mmid_verify_ok;
    return true;
}

// A pending expert fetch: the slot is reserved but its bytes are not in it yet.
struct sb_fetch {
    const void * source_key;
    int          expert;
    const void * source;   // mapped bytes: fallback route and verification oracle
    size_t       bytes;
    void *       dst;      // device slot
};

// Residency memo for one call. Every lookup of a call asks the same table for
// the same tensor, so one pass answers all of them; testing each expert with its
// own scan made `slot` the second largest cost after the disk. The pass also
// stamps recency, so a memo hit needs no entry access at all - which keeps it
// correct even when an eviction shifts the table underneath it.
//
// One snapshot PER TENSOR the call looks at, not one per call. A call may also
// reserve the experts of the tensor that follows it in the same layer (see the
// merged fetch in sabah_rt_mul_mat_id_v2), and that is a different tensor with
// its own residency and its own expert geometry.
struct sb_memo_hit { int expert; void * device; };

struct sb_memo {
    const void *             source = nullptr;
    size_t                   bytes  = 0;
    bool                     valid  = false;   // scan complete: absent means "not resident"
    std::vector<sb_memo_hit> hits;
};

static constexpr size_t SB_MEMO_MAX = 3;
static thread_local sb_memo g_memos[SB_MEMO_MAX];
static thread_local size_t   g_memos_n = 0;   // number of live snapshots this call
static thread_local std::vector<int> g_memo_ids;

static sb_memo * sb_memo_find(const void * source_key, size_t bytes) {
    for (size_t i = 0; i < g_memos_n; ++i) {
        if (g_memos[i].source == source_key && g_memos[i].bytes == bytes) return &g_memos[i];
    }
    return nullptr;
}

// `count_hits` is false when the snapshot is taken for a tensor that this call
// did not itself ask for: those experts are counted where they are actually
// used, by the call that asks for them, so the tier's hit and miss totals stay
// what they always were - only the wave they arrive in changes.
static void sb_mmid_prime(const void * source_key, size_t bytes,
                          const uint8_t * ids_raw, size_t id_stride, size_t token_stride,
                          int n_ids, int n_tokens, bool count_hits) {
    sb_phase_timer _t(&g_t_prime);
    sb_memo * memo = sb_memo_find(source_key, bytes);
    if (memo == nullptr) {
        if (g_memos_n >= SB_MEMO_MAX) return;
        memo = &g_memos[g_memos_n++];
        memo->source = source_key;
        memo->bytes  = bytes;
    }
    memo->valid = false;
    memo->hits.clear();
    if (n_ids <= 0 || n_tokens <= 0) return;

    g_memo_ids.clear();
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_ids; ++i) {
            const int e = *(const int32_t *) (ids_raw + (size_t) t * token_stride
                                                       + (size_t) i * id_stride);
            bool seen = false;
            for (int v : g_memo_ids) { if (v == e) { seen = true; break; } }
            if (!seen) g_memo_ids.push_back(e);
        }
    }

    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    for (auto & entry : g_mmid_entries) {
        if (entry.source_key != source_key || entry.bytes != bytes || !entry.filled) continue;
        for (int v : g_memo_ids) {
            if (v != entry.expert) continue;
            entry.touch = ++g_mmid_tick;
            entry.epoch = g_mmid_epoch;
            if (count_hits) ++g_mmid_hits;
            if (count_hits && g_mmid_verify &&
                (g_mmid_verify_all_hits || g_mmid_hits % g_mmid_verify_hit_every == 0)) {
                const uint8_t * src = (const uint8_t *) source_key;
                src += (size_t) entry.expert * bytes;
                if (!sb_mmid_verify(entry.device, src, bytes)) {
                    memo->valid = false;
                    return;
                }
            }
            memo->hits.push_back({ entry.expert, entry.device });
            break;
        }
    }
    memo->valid = true;
}

// Turn one non-resident expert into a reserved device slot: make room, take the
// space from the slab, and record the fetch that will fill it. The caller holds
// g_mmid_mutex (eviction looks at the table it is working on, and does not lock
// it itself).
static void * sb_mmid_reserve_locked(const void * source_key, int expert,
                                     const void * source, size_t bytes,
                                     std::vector<sb_fetch> & miss,
                                     cudaStream_t compute_stream) {
    ++g_mmid_misses;
    sb_mmid_evict_until(bytes, compute_stream);

    void * device = sb_slab_alloc(bytes);
    bool from_slab = device != nullptr;
    if (!device) {
        // The pool is smaller than this working set and nothing else can be
        // evicted. Documented behaviour is to EXCEED the budget for this fetch
        // (a plain device allocation) rather than fail the call: the slot is
        // still freed normally when it is later evicted, and the routed result
        // stays exact. Overflow counts these over-budget allocations.
        ++g_mmid_overflow;
        if (!ck(cudaMalloc(&device, bytes), "Sabah over-budget expert")) return nullptr;
    }

    g_mmid_entries.push_back({ source_key, expert, device, bytes, ++g_mmid_tick,
                               g_mmid_epoch, from_slab, false });
    g_mmid_bytes += bytes;
    g_mmid_bytes_fetched += bytes;
    miss.push_back({ source_key, expert, source, bytes, device });
    return device;
}

// Reserve (or find) the device slot for one expert. This is pure bookkeeping -
// no I/O is issued here - so a whole MUL_MAT_ID call can be scanned first and
// then fetched with a bounded number of concurrent unbuffered reads.
static void * sb_mmid_slot(const void * source_key, int expert,
                           const void * source, size_t bytes,
                           std::vector<sb_fetch> & miss,
                           cudaStream_t compute_stream) {
    sb_memo * memo = sb_memo_find(source_key, bytes);
    if (memo != nullptr && memo->valid) {
        // The snapshot is complete, so it answers the question outright: a
        // resident expert comes back straight from it, and an expert that is
        // not in it is not resident at all, which saves the table scan that the
        // miss path used to repeat for every single lookup.
        for (const auto & h : memo->hits) {
            if (h.expert == expert) return h.device;
        }
        sb_phase_timer _t(&g_t_slot);
        std::lock_guard<std::mutex> lock(g_mmid_mutex);
        if (g_mmid_capacity == 0) {
            g_mmid_capacity = sb_mmid_capacity_bytes();
        }
        void * device = sb_mmid_reserve_locked(source_key, expert, source, bytes,
                                               miss, compute_stream);
        if (device != nullptr) memo->hits.push_back({ expert, device });
        return device;
    }

    sb_phase_timer _t(&g_t_slot);
    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    if (g_mmid_capacity == 0) {
        g_mmid_capacity = sb_mmid_capacity_bytes();
    }

    for (auto & entry : g_mmid_entries) {
        // `filled` is what makes a slot a hit. An entry created earlier in this
        // same call is already reserved for us, so it is returned as well.
        if (entry.source_key == source_key && entry.expert == expert && entry.bytes == bytes &&
            (entry.filled || entry.epoch == g_mmid_epoch)) {
            entry.touch = ++g_mmid_tick;
            entry.epoch = g_mmid_epoch;
            ++g_mmid_hits;
            if (entry.filled && g_mmid_verify &&
                (g_mmid_verify_all_hits || g_mmid_hits % g_mmid_verify_hit_every == 0) &&
                !sb_mmid_verify(entry.device, source, bytes)) {
                snprintf(g_err, sizeof(g_err),
                         "Sabah resident expert %d differs from its GGUF bytes (hit): the slot was "
                         "delivered earlier in this run (see SABAH_VERIFY_MISMATCH; a stale OS page "
                         "cache for the model file is the usual cause, not the read)", expert);
                return nullptr;
            }
            return entry.device;
        }
    }

    return sb_mmid_reserve_locked(source_key, expert, source, bytes, miss, compute_stream);
}

static bool ck(cudaError_t e, const char * what) {
    if (e == cudaSuccess) return true;
    snprintf(g_err, sizeof(g_err), "%s: %s", what, cudaGetErrorString(e));
    return false;
}

// ===========================================================================
// Direct expert streaming:  file -> pinned staging -> VRAM slot
// ===========================================================================
//
// The expert bank is 50.29 GB and this machine has 16 GB of RAM. Reading those
// bytes through llama.cpp's mmap would push every fetched entity through the
// Windows file cache, so the box would spend its RAM caching expert bytes that
// a multi-gigabyte VRAM tier is going to ask for again only rarely - and that
// is exactly what makes it thrash.
//
// Instead the bytes are read with FILE_FLAG_NO_BUFFERING straight from the GGUF
// file into a pinned, sector-aligned staging block and copied on to their device
// slot. The resident host footprint is a fixed few tens of megabytes no matter
// how large the model is, and the OS file cache is never touched.
//
// The runtime is a drop-in adapter, so it is not told which file backs a given
// host pointer. Windows has no inverse-mapping API, but a mapped view is a
// single MEM_MAPPED allocation: VirtualQuery reports its base and
// GetMappedFileName names the file. llama.cpp maps whole files from offset 0,
// which the "GGUF" magic at the view base confirms before anything is trusted.
// ===========================================================================

static bool sb_direct_io_enabled() {
    const char * v = std::getenv("SABAH_LLAMA_DIRECT_IO");
    return !(v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0));
}

static size_t sb_io_depth() {
    const char * v = std::getenv("SABAH_LLAMA_IO_DEPTH");
    long n = v ? std::atol(v) : 8;
    if (n < 1)  n = 1;
    if (n > 32) n = 32;
    return (size_t) n;
}

// Unbuffered I/O requires the offset, the length and the buffer to share an
// alignment that is a multiple of the volume sector size. 4096 satisfies both
// 512e and 4Kn volumes.
static const size_t SB_IO_ALIGN = 4096;

// Read lanes: synchronous readers, one file object each. The count is a thread
// count - see the read pool for the measurement that forced it.
static const int SB_IO_LANES = 8;

struct sb_disk_file {
#if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE;   // overlapped: the one-off and prefetch reads
    // One SYNCHRONOUS handle per read lane. See the read pool below for why the
    // lanes are threads with their own file object rather than requests on one.
    HANDLE lane[SB_IO_LANES];
    int    lanes = 0;
#else
    int    fd = -1;
#endif
    uint64_t    size = 0;
    std::string path;
};

static std::mutex                    g_disk_mutex;
static std::vector<sb_disk_file>     g_disk_files;
static std::map<std::string, size_t> g_disk_paths;

// Forward-declared beside sb_mmid_verify, which needs it in a diagnostic.
static const char * sb_disk_file_path(size_t i, uint64_t * size_out) {
    if (size_out) *size_out = 0;
    if (i >= g_disk_files.size()) return "file out of range";
    if (size_out) *size_out = g_disk_files[i].size;
    return g_disk_files[i].path.c_str();
}

#if defined(_WIN32)
// "\Device\HarddiskVolume3\Strata-data\x.gguf" -> "C:\Strata-data\x.gguf"
static bool sb_device_path_to_dos(const std::string & dev, std::string & out) {
    char drives[1024];
    const DWORD n = GetLogicalDriveStringsA((DWORD) sizeof(drives) - 1, drives);
    if (n == 0 || n >= sizeof(drives)) return false;
    for (const char * d = drives; d < drives + n && *d; d += std::strlen(d) + 1) {
        char target[1024];
        char drive[4];
        snprintf(drive, sizeof(drive), "%c:", d[0]);   // QueryDosDevice wants "C:"
        if (!QueryDosDeviceA(drive, target, sizeof(target))) continue;
        const size_t tl = std::strlen(target);
        if (dev.size() >= tl && dev.compare(0, tl, target) == 0) {
            out = d;                                        // "C:\"
            if (!out.empty() && out.back() == '\\') out.pop_back();
            out += dev.substr(tl);
            return true;
        }
    }
    return false;
}
#endif

// Must be called with g_disk_mutex held. Returns the file index or (size_t)-1.
static size_t sb_disk_open_locked(const std::string & path) {
    auto it = g_disk_paths.find(path);
    if (it != g_disk_paths.end()) return it->second;

    sb_disk_file f;
    f.path = path;
#if defined(_WIN32)
    f.h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                      nullptr, OPEN_EXISTING,
                      FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (f.h == INVALID_HANDLE_VALUE) return (size_t) -1;
    for (auto & l : f.lane) l = INVALID_HANDLE_VALUE;
    f.lanes = 0;
    // The read lanes, opened here so that a shard registered later still has
    // them: synchronous handle (no FILE_FLAG_OVERLAPPED), one per lane.
    for (int i = 0; i < SB_IO_LANES; ++i) {
        HANDLE lane = CreateFileA(path.c_str(), GENERIC_READ,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
        if (lane == INVALID_HANDLE_VALUE) break;
        f.lane[f.lanes++] = lane;
    }
    LARGE_INTEGER li;
    if (!GetFileSizeEx(f.h, &li)) { CloseHandle(f.h); return (size_t) -1; }
    f.size = (uint64_t) li.QuadPart;
#else
    f.fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (f.fd < 0) return (size_t) -1;
    struct stat st;
    if (fstat(f.fd, &st) != 0) { ::close(f.fd); return (size_t) -1; }
    f.size = (uint64_t) st.st_size;
#endif
    g_disk_paths[path] = g_disk_files.size();
    g_disk_files.push_back(f);
    return g_disk_files.size() - 1;
}

// A host pointer that lives in a memory-mapped file -> which file and where.
struct sb_host_map {
    const uint8_t * base;
    uint64_t        offset;   // file offset of `base`
    uint64_t        size;     // covered range, so two files cannot be confused
    size_t          file;
};

static std::vector<sb_host_map> g_host_maps;

// Resolve a mapped host pointer to (file, byte offset in that file).
// `trace`, when given, receives the reason for failure - the runtime must be
// able to explain why it fell back to the mapped bytes.
static bool sb_resolve(const void * source, size_t & file_idx, uint64_t & file_off,
                       std::string * trace = nullptr) {
#if defined(_WIN32)
    const uint8_t * p = (const uint8_t *) source;

    {
        std::lock_guard<std::mutex> lock(g_disk_mutex);
        for (const auto & m : g_host_maps) {
            if (p >= m.base && p < m.base + m.size) {
                file_idx = m.file;
                file_off = (uint64_t) (p - m.base) + m.offset;
                return true;
            }
        }
    }

    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(source, &mbi, sizeof(mbi)) == 0) {
        if (trace) *trace = "VirtualQuery failed";
        return false;
    }

    // The state of the region the pointer sits in is NOT what identifies the
    // mapping. llama.cpp unmaps fragments of a view once their bytes have been
    // copied elsewhere, so a tensor can legitimately point into a MEM_RESERVE
    // hole inside a MEM_MAPPED allocation. What identifies the mapping is the
    // allocation base, which is the view base, which carries the GGUF magic
    // because llama.cpp maps whole files from offset 0.
    const uint8_t * base = (const uint8_t *) mbi.AllocationBase;
    if (base == nullptr || base > p) {
        if (trace) *trace = "allocation base is not below the pointer";
        return false;
    }
    if (std::memcmp(base, "GGUF", 4) != 0) {
        char b[192];
        snprintf(b, sizeof(b),
                 "no GGUF magic at allocation base (Type=%lu State=%lu first bytes %02x %02x %02x %02x)",
                 (unsigned long) mbi.Type, (unsigned long) mbi.State,
                 base[0], base[1], base[2], base[3]);
        if (trace) *trace = b;
        return false;
    }

    char devname[4096];
    if (GetMappedFileNameA(GetCurrentProcess(), (LPVOID) base, devname,
                           (DWORD) sizeof(devname)) == 0) {
        if (trace) {
            char b[160];
            snprintf(b, sizeof(b), "GetMappedFileName failed (err %lu)", (unsigned long) GetLastError());
            *trace = b;
        }
        return false;
    }
    std::string path;
    if (!sb_device_path_to_dos(devname, path)) {
        if (trace) *trace = std::string("cannot map device path to a drive: ") + devname;
        return false;
    }

    std::lock_guard<std::mutex> lock(g_disk_mutex);
    const size_t fi = sb_disk_open_locked(path);
    if (fi == (size_t) -1) {
        if (trace) {
            char b[160];
            snprintf(b, sizeof(b), "cannot open unbuffered: %s (err %lu)",
                     path.c_str(), (unsigned long) GetLastError());
            *trace = b;
        }
        return false;
    }
    g_host_maps.push_back({ base, 0, g_disk_files[fi].size, fi });
    file_idx = fi;
    file_off = (uint64_t) (p - base);
    return true;
#else
    (void) source; (void) file_idx; (void) file_off;
    if (trace) *trace = "direct I/O is Windows only in this build";
    return false;
#endif
}

// --------------------------- pinned staging pool ---------------------------

struct sb_io_slot {
    void * staging = nullptr;
    size_t cap     = 0;
#if defined(_WIN32)
    HANDLE ev      = nullptr;
#endif
};

static std::vector<sb_io_slot> g_io_slots;
static std::mutex              g_io_slots_mutex;

// One event per staging parity, recorded on the copy stream after a wave's
// host-to-device transfers so the following reuse of those buffers can wait for
// them. Two entries are enough for double buffering.
static cudaEvent_t g_io_wave_ev[2] = { nullptr, nullptr };

static sb_io_slot * sb_io_slot_at(size_t i, size_t need) {
    std::lock_guard<std::mutex> lock(g_io_slots_mutex);
    while (g_io_slots.size() <= i) {
        sb_io_slot s;
#if defined(_WIN32)
        s.ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        if (s.ev == nullptr) return nullptr;
#endif
        g_io_slots.push_back(s);
    }
    sb_io_slot & s = g_io_slots[i];
    if (s.cap < need) {
        if (s.staging) { cudaFreeHost(s.staging); s.staging = nullptr; }
        void * p = nullptr;
        if (!ck(cudaHostAlloc(&p, need, cudaHostAllocDefault), "sabah io staging")) return nullptr;
        s.staging = p;
        s.cap = need;
    }
    return &s;
}

static void sb_io_pool_destroy() {
    std::lock_guard<std::mutex> lock(g_io_slots_mutex);
    for (auto & s : g_io_slots) {
        if (s.staging) { cudaFreeHost(s.staging); s.staging = nullptr; }
#if defined(_WIN32)
        if (s.ev) { CloseHandle(s.ev); s.ev = nullptr; }
#endif
    }
    g_io_slots.clear();
    for (auto & e : g_io_wave_ev) {
        if (e) { cudaEventDestroy(e); e = nullptr; }
    }
}

// --------------------------- one unbuffered read ---------------------------

// How each ReadFile actually returned. This is the difference between an
// overlapped path and a serialised one: a request the driver completes inline
// never overlaps with the next one, so counting them is the only way to tell
// whether issuing several reads before waiting means anything at all.
static uint64_t g_io_sync_reads    = 0;   // ReadFile returned TRUE: done already
static uint64_t g_io_pending_reads = 0;   // ReadFile returned ERROR_IO_PENDING

struct sb_io_req {
    size_t   file   = 0;
    uint64_t off    = 0;       // aligned
    size_t   len    = 0;       // aligned
    void *   staging = nullptr;
    size_t   delta  = 0;       // expert bytes start here inside `staging`
    size_t   bytes  = 0;       // expert bytes requested
    void *   dst    = nullptr; // device slot
    size_t   got    = 0;
    bool     issued = false;
    bool     ok     = false;
#if defined(_WIN32)
    HANDLE   ev     = nullptr;
    OVERLAPPED ov;
#endif
};

static bool sb_io_issue(sb_io_req & r) {
#if defined(_WIN32)
    std::memset(&r.ov, 0, sizeof(r.ov));
    r.ov.Offset     = (DWORD) (r.off & 0xffffffffull);
    r.ov.OffsetHigh = (DWORD) (r.off >> 32);
    r.ov.hEvent     = r.ev;
    if (r.ev) ResetEvent(r.ev);
    DWORD got = 0;
    HANDLE h = g_disk_files[r.file].h;
    if (ReadFile(h, r.staging, (DWORD) r.len, &got, &r.ov)) {
        r.issued = false; r.ok = true; r.got = got;
        ++g_io_sync_reads;
        return true;
    }
    const DWORD e = GetLastError();
    if (e == ERROR_IO_PENDING) { r.issued = true; r.ok = true; ++g_io_pending_reads; return true; }
    // A read that runs past EOF yields a short (or empty) transfer rather than
    // damage: the caller checks that the expert's bytes were really delivered.
    if (e == ERROR_HANDLE_EOF) { r.issued = false; r.ok = true; r.got = 0; return true; }
    snprintf(g_err, sizeof(g_err), "ReadFile failed (err %lu, off %llu, len %llu)",
             (unsigned long) e, (unsigned long long) r.off, (unsigned long long) r.len);
    r.issued = false; r.ok = false; r.got = 0;
    return false;
#else
    (void) r;
    return false;
#endif
}

static bool sb_io_wait(sb_io_req & r) {
#if defined(_WIN32)
    if (!r.issued) return r.ok;
    DWORD got = 0;
    if (!GetOverlappedResult(g_disk_files[r.file].h, &r.ov, &got, TRUE)) {
        const DWORD e = GetLastError();
        r.issued = false;
        r.got = got;
        // A short tail read is fine; anything else is not.
        r.ok = (e == ERROR_HANDLE_EOF);
        return r.ok;
    }
    r.issued = false;
    r.got = got;
    r.ok = true;
    return true;
#else
    (void) r;
    return false;
#endif
}

// Numbers of bytes actually resident on the host because of this module.
static uint64_t g_io_reads      = 0;   // unbuffered reads issued
static uint64_t g_io_read_bytes = 0;   // bytes delivered by those reads
static uint64_t g_io_waves      = 0;   // read waves issued (one latency each)
static uint64_t g_io_fallbacks  = 0;   // times the mapped path had to be used
static size_t   g_io_host_bytes = 0;   // pinned staging currently held

// --------------------------- read lanes (the pool) ---------------------------
//
// A wave of expert reads, issued from a small pool of threads that read
// SYNCHRONOUSLY through one file object each.
//
// Why it looks like this. The drive's own numbers (tools/disk_probe.py, and the
// C++ twin io_bench.cu, 676 KB reads - the expert stride - over this shard) are:
//
//     one reader                        ~1.2 GB/s
//     four readers in flight            ~1.8 GB/s
//     eight readers in flight           ~1.86 GB/s
//
// and both benchmarks read synchronously, from one handle per reader thread.
// This path used to issue the same requests as OVERLAPPED reads from the calling
// thread, and measured 1.37-1.42 GB/s at every depth from 1 to 32: the requests
// really were all in flight (SABAH_DISK sync=0 pending=25000) and the drive
// still served them one at a time. Handing the wave to threads that block in
// ReadFile is what makes the overlap real. SABAH_IO reports it: the pool runs at
// overlap 3.8-4.3 (the sum of the individual read calls against the wall time
// the caller spent waiting), and SABAH_LLAMA_IO_WORKERS=1 collapses it to 1.00.
//
// The honest result of the A/B, though: the wall time does not change. Whole-run
// unbuffered time for 18.97 GB was 13713.9 / 13663.7 / 13734.0 ms for 1, 2 and 8
// lanes, and the decode median 205.8 / 203.1 / 204.5 ms. The drive serves this
// access pattern at ~1.4 GB/s whether the requests overlap or not, so the lanes
// buy nothing today - they are kept because they are the right shape for a wave
// that is actually deep (see the merged-fetch note below for what that means).
//
// The staging buffers and the host-to-device copies stay on the calling thread:
// the pool only moves disk bytes into pinned host memory, where the caller
// already had a wave-sized barrier.
struct sb_io_wave {
    sb_io_req * reqs    = nullptr;
    size_t      n       = 0;
    int         workers = 0;
    int         done    = 0;
    bool        ok      = true;
};

static std::mutex               g_wave_mutex;
static std::condition_variable  g_wave_start_cv;
static std::condition_variable  g_wave_done_cv;
static sb_io_wave *             g_wave      = nullptr;
static uint64_t                 g_wave_gen  = 0;
static int                      g_wave_workers = 0;
static bool                     g_wave_stop = false;
static bool                     g_wave_started = false;
// The pool's threads are held behind a pointer that is deliberately never freed.
// A std::thread that is still joinable when its destructor runs calls
// std::terminate -> abort, and no host calls sabah_rt_shutdown, so the vector's
// own destructor *would* run at process exit and the run would end in a hard
// fast-fail (STATUS_STACK_BUFFER_OVERRUN, exit 0xC0000409) instead of exiting 0.
// Nothing needs to run at exit either: the workers are parked on
// g_wave_start_cv and the process reclaims them. A host that wants the pool gone
// before it unloads this DLL calls sabah_rt_shutdown, which joins them.
static std::vector<std::thread> * g_wave_threads = nullptr;
static uint64_t                 g_io_lane_reads = 0;
// Sum of the individual read durations, per lane. If the lanes really overlap
// this is a large multiple of the time the caller spends waiting for a wave.
static uint64_t                 g_io_read_ns[SB_IO_LANES] = { 0 };

static HANDLE sb_io_lane(size_t file, int lane_index) {
#if defined(_WIN32)
    const sb_disk_file & f = g_disk_files[file];
    // Never hand one file object to two workers: these handles are synchronous,
    // so a shared one means a shared file pointer and two threads racing on
    // SetFilePointerEx. An out-of-range lane fails the read, which sends the
    // batch to the mapped path, which is slow but never wrong.
    if (lane_index < 0 || lane_index >= f.lanes) return INVALID_HANDLE_VALUE;
    return f.lane[lane_index];
#else
    (void) file; (void) lane_index;
    return nullptr;
#endif
}

// One synchronous unbuffered read on a lane. No OVERLAPPED structure and no
// FILE_FLAG_OVERLAPPED on these handles: the read blocks, and the concurrency
// comes from the other lanes running at the same time.
static bool sb_io_read_lane(sb_io_req & r, int lane_index) {
#if defined(_WIN32)
    r.ok  = false;
    r.got = 0;
    HANDLE h = sb_io_lane(r.file, lane_index);
    if (h == INVALID_HANDLE_VALUE || h == nullptr) return false;
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG) r.off;
    if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN)) return false;
    DWORD got = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const BOOL rc = ReadFile(h, r.staging, (DWORD) r.len, &got, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    g_io_read_ns[lane_index % SB_IO_LANES] +=
        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    if (!rc) return false;
    r.got = got;
    r.ok  = true;
    ++g_io_lane_reads;
    return true;
#else
    (void) r; (void) lane_index;
    return false;
#endif
}

static void sb_io_wave_worker(int id) {
    uint64_t seen = 0;
    for (;;) {
        sb_io_wave * w = nullptr;
        {
            std::unique_lock<std::mutex> lock(g_wave_mutex);
            g_wave_start_cv.wait(lock, [&] { return g_wave_stop || g_wave_gen != seen; });
            if (g_wave_stop) return;
            w = g_wave;
            seen = g_wave_gen;
        }
        bool ok = true;
        for (size_t i = (size_t) id; i < w->n; i += (size_t) w->workers) {
            if (!sb_io_read_lane(w->reqs[i], id)) ok = false;
        }
        // The increment and the wake-up share one critical section: the caller
        // must not be able to see a finished wave and free it while a worker is
        // still on its way to the lock.
        std::lock_guard<std::mutex> lock(g_wave_mutex);
        if (!ok) w->ok = false;
        if (++w->done == w->workers) g_wave_done_cv.notify_all();
    }
}

// Start the pool. Returns false if it could not be started, in which case the
// caller falls back to reading the wave on its own thread. Cheap after the first
// call: one uncontended lock, then a load of g_wave_workers.
static bool sb_io_pool_start() {
    {
        std::lock_guard<std::mutex> lock(g_wave_mutex);
        if (g_wave_workers > 0 || g_wave_started) return g_wave_workers > 0;
    }
    size_t want = sb_io_depth();
    if (const char * w = std::getenv("SABAH_LLAMA_IO_WORKERS")) {
        long v = strtol(w, nullptr, 10);
        if (v >= 1) want = (size_t) v;
    }
    if (want > (size_t) SB_IO_LANES) want = (size_t) SB_IO_LANES;
    if (want < 1) want = 1;
    // Never more readers than the shards actually got file objects for.
    {
        std::lock_guard<std::mutex> lock(g_disk_mutex);
        for (const auto & f : g_disk_files) {
            if (f.lanes > 0 && want > (size_t) f.lanes) want = (size_t) f.lanes;
        }
    }
    std::lock_guard<std::mutex> lock(g_wave_mutex);
    if (g_wave_workers > 0 || g_wave_started) return g_wave_workers > 0;
    if (!g_wave_threads) g_wave_threads = new std::vector<std::thread>();
    g_wave_workers = (int) want;
    g_wave_started = true;
    g_wave_stop    = false;
    // Starting the threads here, under the lock, is safe: they go straight to
    // g_wave_start_cv and this thread never waits for them.
    g_wave_threads->reserve((size_t) g_wave_workers);
    for (int i = 0; i < g_wave_workers; ++i) {
        g_wave_threads->emplace_back(sb_io_wave_worker, i);
    }
    return true;
}

// Stop the pool and join its workers. Called by sabah_rt_shutdown, before the
// staging the workers write into is freed and before a host may unload this DLL;
// the pool can be started again afterwards. Process exit deliberately does not
// do this - see g_wave_threads - so the only caller is a host that asked for it.
static void sb_io_pool_stop() {
    std::vector<std::thread> * threads = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_wave_mutex);
        threads = g_wave_threads;
        g_wave_started = false;
        g_wave_workers = 0;
        if (threads) g_wave_stop = true;
    }
    if (!threads) return;
    g_wave_start_cv.notify_all();
    for (auto & t : *threads) {
        if (t.joinable()) t.join();
    }
    std::lock_guard<std::mutex> lock(g_wave_mutex);
    threads->clear();
    g_wave_stop = false;
}

// Read a whole wave. On return every request either has `got` set or `ok`
// false; the caller checks the delivered length before copying.
static void sb_io_read_wave(sb_io_req * reqs, size_t n) {
    if (n == 0) return;
    // A wave of one has nothing to overlap, and handing it to the pool would add
    // a condition-variable round trip to a cost that is pure device latency. Most
    // calls in this workload are exactly that, so the single-request case reads
    // inline.
    if (n == 1 || !sb_io_pool_start() || g_wave_workers <= 1) {
        for (size_t i = 0; i < n; ++i) {
            if (!sb_io_read_lane(reqs[i], 0)) reqs[i].ok = false;
        }
        return;
    }
    sb_io_wave w;
    w.reqs    = reqs;
    w.n       = n;
    w.workers = g_wave_workers;
    {
        std::lock_guard<std::mutex> lock(g_wave_mutex);
        g_wave = &w;
        ++g_wave_gen;
    }
    g_wave_start_cv.notify_all();
    {
        std::unique_lock<std::mutex> lock(g_wave_mutex);
        g_wave_done_cv.wait(lock, [&] { return w.done >= w.workers; });
        g_wave = nullptr;
    }
}

// --------------------------- batched fetch ---------------------------

// Reads every pending expert with at most `SABAH_LLAMA_IO_DEPTH` reads in flight.
// A whole MUL_MAT_ID call is scanned before any of this runs, so the misses of
// one layer arrive here together.
//
// Depth, though, is not what this drive is short of: measured, it serves this
// access pattern at ~1.4 GB/s for every depth from 1 to 32, for 1 to 8 lanes,
// and for merged or unmerged waves, while a benchmark that keeps four to eight
// readers continuously busy reaches 1.8-1.86 GB/s. The difference is the idle
// time between a layer's burst and the next, which a demand pager cannot remove.
//
// Returns false if the direct path cannot be used at all, which tells the caller
// to complete the slots from the mapped bytes instead.
static bool sb_fetch_direct(std::vector<sb_fetch> & miss) {
    if (!sb_direct_io_enabled()) {
        snprintf(g_err, sizeof(g_err), "direct I/O disabled");
        return false;
    }
    if (miss.empty()) return true;
    sb_phase_timer _tf(&g_t_fetch);

    // Resolve every source first. One failure sends the whole batch down the
    // fallback so a slot is never a mix of bytes from two different routes.
    std::vector<size_t>   file(miss.size(), (size_t) -1);
    std::vector<uint64_t> off(miss.size(), 0);
    for (size_t i = 0; i < miss.size(); ++i) {
        std::string why;
        if (!sb_resolve(miss[i].source, file[i], off[i], &why)) {
            snprintf(g_err, sizeof(g_err), "%s", why.empty() ? "unresolved source" : why.c_str());
            return false;
        }
    }
    if (g_mmid_verify) {
        // Diagnostics only: recording this costs a mutex and a store per miss, and
        // it exists so a rejected object can name the file and offset it came from.
        // Verification is a small-test switch, so the hot path never pays for it.
        std::lock_guard<std::mutex> lock(g_origin_mutex);
        g_origin.clear();
        for (size_t i = 0; i < miss.size(); ++i) {
            g_origin.push_back({miss[i].dst, file[i], off[i], miss[i].bytes});
        }
    }

    const size_t depth = sb_io_depth();
    size_t wave_no = 0;
    for (size_t base = 0; base < miss.size(); base += depth, ++wave_no) {
        const size_t wave = std::min(depth, miss.size() - base);
        ++g_io_waves;
        // Staging is double buffered. A wave's host-to-device copies are still in
        // flight on the copy stream when the next wave starts issuing reads, so
        // the reads must not write the buffers the copies are reading. Wave w
        // therefore uses the buffers wave w-2 used, and waits for that wave's
        // copies to land first. This was the source of both a data race and of a
        // hard serialisation between reading and copying.
        const size_t parity = wave_no % 2;
        const size_t slot0  = parity * depth;
        if (g_io_wave_ev[parity] != nullptr &&
            !ck(cudaEventSynchronize(g_io_wave_ev[parity]), "Sabah staging reuse wait")) {
            return false;
        }
        std::vector<sb_io_req> reqs(wave);

        for (size_t i = 0; i < wave; ++i) {
            const size_t j = base + i;
            const sb_fetch & f = miss[j];
            const uint64_t a_off = off[j] & ~(uint64_t) (SB_IO_ALIGN - 1);
            const size_t   delta = (size_t) (off[j] - a_off);
            const size_t   len   = ((delta + f.bytes + SB_IO_ALIGN - 1) / SB_IO_ALIGN) * SB_IO_ALIGN;
            sb_io_slot * s = sb_io_slot_at(slot0 + i, len);
            if (s == nullptr) return false;
            sb_io_req & r = reqs[i];
            r.file    = file[j];
            r.off     = a_off;
            r.len     = len;
            r.staging = s->staging;
            r.delta   = delta;
            r.bytes   = f.bytes;
            r.dst     = f.dst;
#if defined(_WIN32)
            r.ev      = s->ev;
#endif
        }

        // All of the wave's reads at once, on the lanes. Waiting for them is
        // the disk cost, so it is what `wait` accounts for.
        {
            sb_phase_timer _tw(&g_t_wait);
            sb_io_read_wave(reqs.data(), wave);
        }

        for (size_t i = 0; i < wave; ++i) {
            const size_t j = base + i;
            sb_io_req & r = reqs[i];
            if (!r.ok) {
                snprintf(g_err, sizeof(g_err), "unbuffered read of %llu bytes failed",
                         (unsigned long long) r.len);
                return false;
            }
            if (r.got < r.delta + r.bytes) {
                snprintf(g_err, sizeof(g_err),
                         "unbuffered read delivered %llu of %llu bytes",
                         (unsigned long long) r.got,
                         (unsigned long long) (r.delta + r.bytes));
                return false;
            }
            if (!ck(cudaMemcpyAsync(miss[j].dst, (const uint8_t *) r.staging + r.delta,
                                    miss[j].bytes, cudaMemcpyHostToDevice, g_copy),
                    "Sabah expert H2D")) {
                return false;
            }
            ++g_io_reads;
            g_io_read_bytes += miss[j].bytes;
        }
        if (g_io_wave_ev[parity] == nullptr &&
            !ck(cudaEventCreateWithFlags(&g_io_wave_ev[parity], cudaEventDisableTiming),
                "Sabah staging event")) {
            return false;
        }
        if (!ck(cudaEventRecord(g_io_wave_ev[parity], g_copy), "Sabah staging event record")) {
            return false;
        }
    }
    return true;
}

// ===========================================================================
// Exact cross-call expert prefetch
// ===========================================================================
//
// A forward pass issues a fixed sequence of MUL_MAT_ID calls, and the three
// expert tensors of one mixture-of-experts layer all consume the SAME router
// ids. So as soon as a call's ids are in hand, the experts the FOLLOWING calls
// of that layer will ask for are already known - only their bytes are missing.
//
// A single call has roughly three reads in flight, and the disk sits idle
// through the router download and the kernel that follow it. Issuing the
// followers' reads during that window both deepens the queue and stops the
// device idling, without changing which experts execute: the demand path still
// chooses residency from the true ids, so a wrong prediction costs bytes and
// nothing else.
//
// Disable with SABAH_LLAMA_PREFETCH=0.

struct sb_pf_desc {
    const void * source        = nullptr;
    size_t       expert_stride = 0;
    int          n_experts     = 0;
};

struct sb_pf_job {
    size_t       slot       = 0;
    const void * source_key = nullptr;
    const void * source     = nullptr;   // mapped bytes: fallback source
    int          expert     = 0;
    size_t       bytes      = 0;
};

enum sb_pf_state { SB_PF_FREE = 0, SB_PF_FILLING, SB_PF_READY, SB_PF_TAKEN, SB_PF_FAIL };

// A readahead buffer. The prefetcher reads into pinned staging and LEAVES the
// bytes there; the demand path copies them straight on to their device slot and
// never touches the disk for them. Crucially it reserves no tier space: that is
// what made the first shape cost more than it gained, because the demand path
// then found its own slot reserved-but-unfilled and read the same bytes again.
struct sb_pf_slot {
    void *       staging      = nullptr;
    size_t       cap          = 0;
    size_t       delta        = 0;   // expert bytes begin here inside `staging`
    int          state        = SB_PF_FREE;
    bool         free_pending = false;
    const void * source_key   = nullptr;
    int          expert       = 0;
    size_t       bytes        = 0;
    uint64_t     seq          = 0;
#if defined(_WIN32)
    HANDLE       ev           = nullptr;   // overlapped read event
#endif
    cudaEvent_t  free_ev      = nullptr;   // copy-out finished
};

static constexpr int SB_PF_DEPTH      = 8;   // readahead buffers / reads in flight
static constexpr int SB_PF_MAX_QUEUE  = 64;
static constexpr int SB_PF_MAX_TOKENS = 4;   // only worth it when a call has few ids

static bool         g_pf_enabled = false;
static int          g_pf_dev     = 0;
static cudaStream_t g_pf_stream  = nullptr;

static std::mutex              g_pf_mutex;
static std::condition_variable g_pf_cv;
// A readahead buffer changed state. Kept separate from g_pf_cv so a notify for
// one purpose can never wake the waiter of the other and leave the real waiter
// asleep: the worker waits for work, the demand path waits for one buffer.
static std::condition_variable g_pf_done_cv;
static std::deque<sb_pf_job>   g_pf_queue;
static bool                    g_pf_stop  = false;
static std::thread             g_pf_thread;
static std::once_flag          g_pf_once;

static std::vector<sb_pf_slot> g_pf_slots;
static uint64_t                g_pf_seq = 0;

static uint64_t g_pf_reads = 0, g_pf_bytes = 0, g_pf_drops = 0;
static uint64_t g_pf_used = 0, g_pf_used_bytes = 0, g_pf_stale = 0;
static uint64_t g_pf_waits = 0, g_pf_wait_timeouts = 0;
static double   g_pf_wait_ms = 0.0;

// The call sequence, learned as a chain. Indexing it by the router ids tensor
// was wrong: the graph allocator hands every layer the same address for that
// buffer, so the chain collapsed into a single node claiming all 144 tensors and
// the prefetcher pulled the whole bank. The expert tensors themselves are
// distinct, so the chain is keyed by those, and whether the following call
// reuses the ids is learned by comparing the ids this call downloaded with the
// ones the previous call did.
struct sb_pf_link {
    const void * source       = nullptr;
    const void * next_source  = nullptr;
    size_t       next_stride  = 0;
    int          next_experts = 0;
    bool         shares_ids   = false;
};

static std::mutex              g_pf_chain_mutex;
static std::vector<sb_pf_link> g_pf_links;
static const void *            g_pf_prev_source = nullptr;
static std::vector<int32_t>    g_pf_prev_ids;

// Make a readahead buffer usable: room for `need` bytes, and nobody still
// copying out of it.
static bool sb_pf_prepare(sb_pf_slot & s, size_t need) {
    if (s.free_pending) {
        if (s.free_ev != nullptr) cudaEventSynchronize(s.free_ev);
        s.free_pending = false;
    }
    if (s.cap < need) {
        if (s.staging) { cudaFreeHost(s.staging); s.staging = nullptr; }
        void * p = nullptr;
        if (!ck(cudaHostAlloc(&p, need, cudaHostAllocDefault), "sabah prefetch staging")) return false;
        s.staging = p;
        s.cap = need;
    }
    return true;
}

// Hand the demand path the bytes of an expert the prefetcher has already read,
// or is still reading. Waiting for the read that is already in flight is the
// point.
//
// An earlier revision returned false the moment a buffer was not READY yet, so
// the demand path issued its own read of the very same bytes. Every readahead
// that had not finished by then was read TWICE; with the disk as the binding
// constraint that turned the readahead into extra load rather than lead time,
// which is exactly what the earlier measurement saw (the drive's throughput
// went up, the token did not). The staged read is on the critical path either
// way - the difference is that it was issued one call earlier, while the
// previous call's kernel was running, and that it keeps the queue deep.
//
// Returns false only when nothing is staged or coming for this expert, or when
// the wait gave up. The caller then reads it from the disk as usual.
#define SB_PF_WAIT_LIMIT_MS 50
static bool sb_pf_take(const sb_fetch & f) {
    size_t idx = g_pf_slots.size();
    bool   waited = false;
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lock(g_pf_mutex);
        for (;;) {
            size_t found = g_pf_slots.size();
            for (size_t i = 0; i < g_pf_slots.size(); ++i) {
                sb_pf_slot & s = g_pf_slots[i];
                if (s.source_key != f.source_key || s.expert != f.expert ||
                    s.bytes != f.bytes) {
                    continue;
                }
                if (s.state == SB_PF_FAIL) {          // the read itself failed
                    s.state = SB_PF_FREE;
                    return false;
                }
                if (s.state == SB_PF_READY || s.state == SB_PF_FILLING) {
                    found = i;
                    break;
                }
                // SB_PF_TAKEN: already leaving. Its identity can only survive
                // until the copy-out finishes, so this cannot be our expert's
                // last chance - treat it as absent.
            }
            if (found == g_pf_slots.size()) return false;   // nothing staged at all
            if (g_pf_slots[found].state == SB_PF_READY) {
                g_pf_slots[found].state        = SB_PF_TAKEN;
                g_pf_slots[found].free_pending = true;
                idx = found;
                break;
            }
            // Still filling: wait for the worker instead of reading it twice.
            waited = true;
            if (g_pf_done_cv.wait_for(lock, std::chrono::milliseconds(SB_PF_WAIT_LIMIT_MS))
                    == std::cv_status::timeout) {
                ++g_pf_wait_timeouts;
                return false;
            }
        }
    }
    if (waited) {
        ++g_pf_waits;
        g_pf_wait_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
    }

    sb_pf_slot & s = g_pf_slots[idx];
    if (!ck(cudaMemcpyAsync(f.dst, (const uint8_t *) s.staging + s.delta, f.bytes,
                            cudaMemcpyHostToDevice, g_copy), "Sabah prefetched expert H2D")) {
        std::lock_guard<std::mutex> lock(g_pf_mutex);
        s.state = SB_PF_FREE;
        s.free_pending = false;
        return false;
    }
    if (s.free_ev == nullptr &&
        !ck(cudaEventCreateWithFlags(&s.free_ev, cudaEventDisableTiming),
            "Sabah prefetch free event")) {
        return false;
    }
    if (!ck(cudaEventRecord(s.free_ev, g_copy), "Sabah prefetch free record")) return false;
    {
        std::lock_guard<std::mutex> lock(g_pf_mutex);
        s.state = SB_PF_FREE;   // reusable once the worker has waited on free_ev
    }
    ++g_pf_used;
    g_pf_used_bytes += f.bytes;
    return true;
}

// Read a batch of experts into their readahead buffers. Nothing is copied to a
// device slot here: the bytes stay in pinned staging until a demand call asks
// for them, which is what keeps the tier free of reservations nobody wanted.
static void sb_pf_run_batch(std::vector<sb_pf_job> & batch) {
    const size_t n = batch.size();
    std::vector<sb_io_req> reqs(n);
    std::vector<char>      issued(n, 0);

    for (size_t i = 0; i < n; ++i) {
        sb_pf_job & j = batch[i];
        sb_pf_slot & s = g_pf_slots[j.slot];
        size_t   file = 0;
        uint64_t off  = 0;
        if (!sb_resolve(j.source, file, off)) continue;
        const uint64_t a_off = off & ~(uint64_t) (SB_IO_ALIGN - 1);
        const size_t   delta = (size_t) (off - a_off);
        const size_t   len   = ((delta + j.bytes + SB_IO_ALIGN - 1) / SB_IO_ALIGN) * SB_IO_ALIGN;
        if (!sb_pf_prepare(s, len)) continue;
        s.delta = delta;
        sb_io_req & r = reqs[i];
        r.file    = file;
        r.off     = a_off;
        r.len     = len;
        r.staging = s.staging;
        r.delta   = delta;
        r.bytes   = j.bytes;
        r.dst     = nullptr;
#if defined(_WIN32)
        r.ev      = s.ev;
#endif
        if (!sb_io_issue(r)) continue;
        issued[i] = 1;
    }

    std::vector<char> ok(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (!issued[i]) continue;
        ok[i] = sb_io_wait(reqs[i]) && reqs[i].got >= reqs[i].delta + reqs[i].bytes;
    }

    {
        std::lock_guard<std::mutex> lock(g_pf_mutex);
        for (size_t i = 0; i < n; ++i) {
            sb_pf_slot & s = g_pf_slots[batch[i].slot];
            if (!ok[i]) { s.state = SB_PF_FAIL; continue; }
            s.state = SB_PF_READY;
            ++g_pf_reads;
            g_pf_bytes += batch[i].bytes;
        }
    }
    // A demand call may be parked on any of these buffers.
    g_pf_done_cv.notify_all();
}

static void sb_pf_worker() {
    cudaSetDevice(g_pf_dev);
    std::vector<sb_pf_job> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(g_pf_mutex);
            g_pf_cv.wait(lock, [] { return g_pf_stop || !g_pf_queue.empty(); });
            if (g_pf_queue.empty()) {
                if (g_pf_stop) break;
                continue;
            }
            const size_t take = std::min(g_pf_queue.size(), (size_t) SB_PF_DEPTH);
            batch.assign(g_pf_queue.begin(), g_pf_queue.begin() + (std::ptrdiff_t) take);
            g_pf_queue.erase(g_pf_queue.begin(), g_pf_queue.begin() + (std::ptrdiff_t) take);
        }
        sb_pf_run_batch(batch);
    }
}

// Queue one expert of one follower tensor for readahead.
static void sb_pf_offer(const sb_pf_desc & d, int expert) {
    if (expert < 0 || expert >= d.n_experts || d.expert_stride == 0) return;
    std::lock_guard<std::mutex> lock(g_pf_mutex);
    if (g_pf_queue.size() >= (size_t) SB_PF_MAX_QUEUE) return;

    for (const auto & s : g_pf_slots) {
        if ((s.state == SB_PF_READY || s.state == SB_PF_FILLING) &&
            s.source_key == d.source && s.expert == expert && s.bytes == d.expert_stride) {
            return;   // already staged, or already on its way
        }
    }

    size_t idx = g_pf_slots.size();
    for (size_t i = 0; i < g_pf_slots.size(); ++i) {
        // FAIL is reclaimable too: the previous read of that buffer did not
        // complete, and nobody is waiting on it any more.
        if (g_pf_slots[i].state == SB_PF_FREE || g_pf_slots[i].state == SB_PF_FAIL) {
            idx = i;
            break;
        }
    }
    if (idx == g_pf_slots.size()) {
        // Every buffer is busy; take the oldest readahead nobody claimed.
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < g_pf_slots.size(); ++i) {
            if (g_pf_slots[i].state == SB_PF_READY && g_pf_slots[i].seq < oldest) {
                oldest = g_pf_slots[i].seq;
                idx = i;
            }
        }
        if (idx == g_pf_slots.size()) { ++g_pf_drops; return; }
        ++g_pf_stale;
    }

    sb_pf_slot & s = g_pf_slots[idx];
    s.state      = SB_PF_FILLING;
    s.source_key = d.source;
    s.expert     = expert;
    s.bytes      = d.expert_stride;
    s.seq        = ++g_pf_seq;

    sb_pf_job j;
    j.slot       = idx;
    j.source_key = d.source;
    j.source     = (const uint8_t *) d.source + (size_t) expert * d.expert_stride;
    j.expert     = expert;
    j.bytes      = d.expert_stride;
    g_pf_queue.push_back(j);
    g_pf_cv.notify_one();
}

// Queue the experts of one tensor that the tier does not already hold. The
// residency test scans the entry table once and checks membership in the expert
// list, rather than testing each expert with its own scan.
static void sb_pf_plan(const sb_pf_desc & d, const std::vector<int> & experts) {
    if (d.source == nullptr || d.expert_stride == 0 || experts.empty()) return;

    std::vector<char> resident(experts.size(), 0);
    {
        std::lock_guard<std::mutex> lock(g_mmid_mutex);
        for (const auto & e : g_mmid_entries) {
            if (e.source_key != d.source || e.bytes != d.expert_stride) continue;
            for (size_t k = 0; k < experts.size(); ++k) {
                if (experts[k] == e.expert) { resident[k] = 1; break; }
            }
        }
    }
    for (size_t k = 0; k < experts.size(); ++k) {
        if (!resident[k]) sb_pf_offer(d, experts[k]);
    }
}

// Learn the call sequence and hand back the tensors that will follow `source`.
//
// The graph is static, so every forward pass issues the same MUL_MAT_ID calls in
// the same order: layer L always runs gate, then up, then down, and all three of
// them are handed THE SAME router ids, so they ask for the same experts. That
// makes a call's followers knowable one call in advance, and it is the only
// cross-call knowledge this workload has - a later layer's ids depend on this
// layer's experts, so nothing further ahead can be predicted without changing
// the routing.
//
// LEARNING IS UNCONDITIONAL; only what is done with it is optional. An earlier
// revision returned early unless the readahead switch was on, so the chain was
// never learned at all in the default configuration.
//
// Returns how many followers were written to `out` (at most 2).
static int sb_chain_note(const void * source, size_t expert_stride, int n_experts,
                         int n_ids, int n_tokens, size_t ids_id_stride,
                         const uint8_t * ids_raw, sb_pf_desc out[2]) {
    if (n_ids <= 0 || n_tokens <= 0 || n_experts <= 0) return 0;

    // The first token row stands in for the call. Within a layer the three calls
    // agree on every row; a comparison is only ever used to decide whether a
    // follower shares this call's routing.
    std::vector<int32_t> experts((size_t) n_ids);
    for (int i = 0; i < n_ids; ++i) {
        experts[(size_t) i] = *(const int32_t *) (ids_raw + (size_t) i * ids_id_stride);
    }

    int n_out = 0;
    {
        std::lock_guard<std::mutex> lock(g_pf_chain_mutex);

        // Close the previous call's edge. This happens before this call's own
        // edge is read, so a prediction always rests on what the previous token
        // observed rather than on a fact learned mid-flight.
        if (g_pf_prev_source != nullptr && g_pf_prev_source != source) {
            size_t at = g_pf_links.size();
            for (size_t i = 0; i < g_pf_links.size(); ++i) {
                if (g_pf_links[i].source == g_pf_prev_source) { at = i; break; }
            }
            if (at == g_pf_links.size()) {
                sb_pf_link l;
                l.source = g_pf_prev_source;
                g_pf_links.push_back(l);
                at = g_pf_links.size() - 1;
            }
            g_pf_links[at].next_source  = source;
            g_pf_links[at].next_stride  = expert_stride;
            g_pf_links[at].next_experts = n_experts;
            g_pf_links[at].shares_ids   = (g_pf_prev_ids == experts);
        }

        for (const auto & l : g_pf_links) {
            if (l.source != source || !l.shares_ids || l.next_source == nullptr) continue;
            out[n_out].source        = l.next_source;
            out[n_out].expert_stride = l.next_stride;
            out[n_out].n_experts     = l.next_experts;
            ++n_out;
            for (const auto & m : g_pf_links) {
                if (m.source == out[0].source && m.shares_ids && m.next_source != nullptr) {
                    out[n_out].source        = m.next_source;
                    out[n_out].expert_stride = m.next_stride;
                    out[n_out].n_experts     = m.next_experts;
                    ++n_out;
                    break;
                }
            }
            break;
        }

        g_pf_prev_source = source;
        g_pf_prev_ids.swap(experts);
    }
    return n_out;
}

// --------------------------- merged layer fetch ---------------------------
//
// One wave per layer instead of up to three.
//
// The idea was that each of a layer's three expert tensors pays a wave latency
// of its own - decode measures 99.9 waves of 262 reads of ~676 KB - and that if
// the layer's three tensors shared one wave the latency would be paid once
// (48 x 0.45 ms instead of 99.9 x 0.45 ms). The wave count really does halve;
// the time does not move, because this drive does not charge per wave. See the
// measurement at the bottom of this section before enabling it.
//
// The three expert tensors of a layer all read the same router ids, so the
// followers' misses are already known when the first of them runs. Reserving
// their slots at that point puts every expert the layer will need into the ONE
// wave the first call already issues. The followers' own calls then find their
// slots filled and issue no wave at all, so the layer pays the latency once.
//
// Nothing about routing changes: the followers' entries are keyed by the very
// expert indices the true ids name, read from the very file ranges the demand
// path would have read, and the demand path still asks for residency from the
// true ids on every call. A prediction that turned out wrong would cost the
// bytes it read and a tier slot, and nothing else.
//
// Supersedes the staging readahead above, which read the same experts a second
// time when its prediction did not pan out in time.
//
// MEASURED, and the reason this defaults to off. It does what it says to the
// wave count - decode goes 99.9 waves/token to 49.5, the prefill's 5664 to
// 4088, with the same 262.4 reads and the same 166.6 MB - and the disk time does
// not move at all: the whole run reads 18.97 GB in 13.7 s (1.385 GB/s) with the
// merge, 13.70 s without it. The wave latency this was supposed to halve is not
// something the drive charges: it serves these reads at ~1.4 GB/s whatever the
// wave size, so there is nothing here to save, and concentrating the layer's
// wait into one call instead of three leaves the GPU idle for a longer stretch
// at a time, which the next group's id synchronise then exposes. Net decode:
// ~205 ms/token without the merge, ~218 ms with it, reproducible across runs.
//
// It stays because the mechanism is sound and it is the thing that would pay
// off if a wave ever became deep enough for depth to matter (batch
// verification, where one layer's demand is ~4x a decode layer's).
// Force it on with SABAH_LLAMA_LAYER_MERGE=1.
static bool          g_layer_merge = false;
static std::once_flag g_layer_merge_once;

static bool sb_layer_merge_enabled() {
    std::call_once(g_layer_merge_once, []() {
        const char * v = std::getenv("SABAH_LLAMA_LAYER_MERGE");
        if (v == nullptr) return;
        if (std::strcmp(v, "1") == 0 || std::strcmp(v, "on") == 0) g_layer_merge = true;
        if (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0) g_layer_merge = false;
    });
    return g_layer_merge;
}

static void sb_pf_quit() {
    if (!g_pf_enabled) return;
    {
        std::lock_guard<std::mutex> lock(g_pf_mutex);
        g_pf_stop = true;
    }
    g_pf_cv.notify_all();
    if (g_pf_thread.joinable()) g_pf_thread.join();
    g_pf_enabled = false;
    for (auto & s : g_pf_slots) {
        if (s.staging) { cudaFreeHost(s.staging); s.staging = nullptr; }
        if (s.free_ev) { cudaEventDestroy(s.free_ev); s.free_ev = nullptr; }
#if defined(_WIN32)
        if (s.ev) { CloseHandle(s.ev); s.ev = nullptr; }
#endif
    }
    g_pf_slots.clear();
    if (g_pf_stream) { cudaStreamDestroy(g_pf_stream); g_pf_stream = nullptr; }
}

static void sb_pf_start() {
    std::call_once(g_pf_once, []() {
        // Opt-in while it is still being measured against the plain path; what
        // decides whether it stays is the SABAH_PREFETCH line, not intent.
        const char * v = std::getenv("SABAH_LLAMA_PREFETCH");
        if (v == nullptr) return;
        if (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0) return;
        if (cudaGetDevice(&g_pf_dev) != cudaSuccess) return;
        if (cudaStreamCreateWithFlags(&g_pf_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_pf_stream = nullptr;
            return;
        }
        // Fixed size, created once: a slot must never move, because the demand
        // path holds references to these buffers.
        g_pf_slots.resize(SB_PF_DEPTH);
        for (auto & s : g_pf_slots) {
#if defined(_WIN32)
            s.ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
            if (s.ev == nullptr) {
                g_pf_slots.clear();
                cudaStreamDestroy(g_pf_stream);
                g_pf_stream = nullptr;
                return;
            }
#endif
        }
        g_pf_enabled = true;
        std::atexit(sb_pf_quit);
        g_pf_thread = std::thread(sb_pf_worker);
    });
}

// Completes every reserved slot and marks it filled.
static bool sb_mmid_fill(std::vector<sb_fetch> & miss) {
    if (miss.empty()) return true;

    // Satisfy what readahead already holds before scheduling any disk work. The
    // staged entries stay in `miss` on purpose: they are still reserved slots
    // that have to be marked filled, or the next token would fetch them again.
    std::vector<sb_fetch> disk;
    disk.reserve(miss.size());
    if (g_pf_enabled) {
        for (const sb_fetch & f : miss) {
            if (!sb_pf_take(f)) disk.push_back(f);
        }
    } else {
        disk = miss;
    }

    if (!disk.empty() && !sb_fetch_direct(disk)) {
        ++g_io_fallbacks;
        {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "SABAH_LLAMA: unbuffered expert read unavailable: %s\n",
                             g_err[0] ? g_err : "no reason reported");
                std::fflush(stderr);
            }
        }
        // Diagnostic: stop before a single expert byte is read through the
        // mapping, so the disk path can be debugged without the box thrashing.
        const char * act = std::getenv("SABAH_LLAMA_RESOLVE_TRACE");
        if (act && std::strcmp(act, "exit") == 0) {
            std::fprintf(stderr, "SABAH_LLAMA_RESOLVE_TRACE=exit: stopping before any mapped read\n");
            std::fflush(stderr);
            std::_Exit(0);
        }
        g_err[0] = 0;
        for (const sb_fetch & f : disk) {
            if (!ck(cudaMemcpyAsync(f.dst, f.source, f.bytes,
                                    cudaMemcpyHostToDevice, g_copy),
                    "Sabah expert H2D")) {
                return false;
            }
        }
    }

    std::lock_guard<std::mutex> lock(g_mmid_mutex);
    for (const auto & f : miss) {
        for (auto & e : g_mmid_entries) {
            if (e.source_key == f.source_key && e.expert == f.expert && e.bytes == f.bytes) {
                e.filled = true;
                break;
            }
        }
        if (g_mmid_verify && !sb_mmid_verify(f.dst, f.source, f.bytes)) {
            snprintf(g_err, sizeof(g_err),
                     "Sabah fetched expert %d differs from its GGUF bytes (miss): %llu bytes at "
                     "offset %llu of %s (see SABAH_VERIFY_MISMATCH; a stale OS page cache for "
                     "that file is the usual cause, not the read)",
                     f.expert, (unsigned long long) f.bytes,
                     (unsigned long long) g_verify_last_off,
                     sb_disk_file_path(g_verify_last_file, nullptr));
            return false;
        }
    }
    return true;
}

// Single-expert convenience, used by the sentinel test path.
static void * sb_mmid_resident(const void * source_key, int expert,
                               const void * source, size_t bytes,
                               cudaStream_t compute_stream) {
    std::vector<sb_fetch> miss;
    void * p = sb_mmid_slot(source_key, expert, source, bytes, miss, compute_stream);
    if (!p) return nullptr;
    if (!sb_mmid_fill(miss)) return nullptr;
    return p;
}

// Pointer table and hand-off event, reused across calls. cudaMalloc and
// cudaEventCreate cost tens of microseconds each and this path runs 144 times
// per token, so neither may appear in the steady state.
static void **     g_ptr_table    = nullptr;
static size_t      g_ptr_table_cap = 0;
static cudaEvent_t g_ready_ev      = nullptr;

static void sb_scratch_destroy() {
    if (g_ptr_table) { cudaFree(g_ptr_table); g_ptr_table = nullptr; g_ptr_table_cap = 0; }
    if (g_ready_ev)  { cudaEventDestroy(g_ready_ev); g_ready_ev = nullptr; }
}

static bool sb_ptr_table(size_t bytes, void *** out) {
    if (bytes > g_ptr_table_cap) {
        if (g_ptr_table) { cudaFree(g_ptr_table); g_ptr_table = nullptr; g_ptr_table_cap = 0; }
        size_t cap = bytes < (1u << 16) ? (1u << 16) : bytes;
        if (!ck(cudaMalloc(&g_ptr_table, cap), "Sabah pointer table allocation")) return false;
        g_ptr_table_cap = cap;
    }
    *out = g_ptr_table;
    return true;
}

// ---------------------------------------------------------------------------
// block geometry, kept in one place so host and device agree
// ---------------------------------------------------------------------------
__host__ __device__ static inline int sb_block_bytes(int qt) {
    switch (qt) {
        case SB_Q4_K:    return 144;
        case SB_Q5_K:    return 176;
        case SB_Q5_1:    return  24;
        case SB_Q8_0:    return  34;
        case SB_IQ3_XXS: return  98;   // d + 3*QK_K/8
        case SB_IQ4_NL:  return  18;   // d + QK4_NL/2
        case SB_IQ3_S:   return 110;   // d + 13*(QK_K/32) + IQ3S_N_SCALE
        case SB_IQ2_S:   return  82;   // d + QK_K/4 + QK_K/16
        case SB_IQ4_XS:  return 136;   // d + scales_h + QK_K/64 + QK_K/2
        case SB_Q2_0:    return  18;   // d + QK2_0/4
    }
    return 0;
}
__host__ __device__ static inline int sb_block_elems(int qt) {
    switch (qt) {
        case SB_Q4_K:    return 256;
        case SB_Q5_K:    return 256;
        case SB_Q5_1:    return  32;
        case SB_Q8_0:    return  32;
        case SB_IQ3_XXS: return 256;
        case SB_IQ4_NL:  return  32;
        case SB_IQ3_S:   return 256;
        case SB_IQ2_S:   return 256;
        case SB_IQ4_XS:  return 256;
        case SB_Q2_0:    return  64;
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
    case SB_IQ2_S:
        // d | qs[64] (32 grid bytes, then 32 sign bytes) | qh[8] | scales[8]
        for (int b = 0; b < n / 256; ++b, x += 82) {
            const float d = sb_host_f16(x);
            const uint8_t * qs    = x + 2;
            const uint8_t * qh    = x + 66;
            const uint8_t * signs = qs + 32;
            const uint8_t * sc    = x + 74;
            int qo = 0, so = 0;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db0 = d * (0.5f + (sc[ib32] & 0xf)) * 0.25f;
                const float db1 = d * (0.5f + (sc[ib32] >>  4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const float dl = (l < 2) ? db0 : db1;
                    const uint32_t gi = (uint32_t) qs[qo + l]
                                      | (((uint32_t) qh[ib32] << (8 - 2 * l)) & 0x300u);
                    const uint64_t g  = iq2s_grid[gi];
                    const uint8_t  sg = signs[so + l];
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t gv = (uint8_t) (g >> (8 * j));
                        *y++ = dl * gv * ((sg & kmask_iq2xs[j]) ? -1.f : 1.f);
                    }
                }
                qo += 4; so += 4;
            }
        }
        return true;
    case SB_IQ3_XXS:
        // d | qs[96]: 64 grid bytes, then 32 scale/sign bytes (4 per ib32)
        for (int b = 0; b < n / 256; ++b, x += 98) {
            const float d = sb_host_f16(x);
            const uint8_t * qs = x + 2;
            const uint8_t * ss = qs + 64;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t aux32;
                std::memcpy(&aux32, ss + 4 * ib32, 4);
                const float db = d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t  sg = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                    const uint32_t g1 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 0]];
                    const uint32_t g2 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db * (float) ((g1 >> (8 * j)) & 0xff)
                                 * ((sg & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                        y[j + 4] = db * (float) ((g2 >> (8 * j)) & 0xff)
                                 * ((sg & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                    }
                    y += 8;
                }
            }
        }
        return true;
    case SB_IQ3_S:
        // d | qs[64] | qh[8] | signs[32] | scales[4]
        for (int b = 0; b < n / 256; ++b, x += 110) {
            const float d = sb_host_f16(x);
            const uint8_t * qs    = x + 2;
            const uint8_t * qh    = x + 66;
            const uint8_t * signs = x + 74;
            const uint8_t * sc    = x + 106;
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                const float db1 = d * (1 + 2 * (float) (sc[ib32 / 2] & 0xf));
                const float db2 = d * (1 + 2 * (float) (sc[ib32 / 2] >>  4));
                for (int h = 0; h < 2; ++h) {
                    const float db = h ? db2 : db1;
                    for (int l = 0; l < 4; ++l) {
                        const uint32_t i1 = (uint32_t) qs[2 * l + 0]
                                          | (((uint32_t) qh[h] << (8 - 2 * l)) & 256u);
                        const uint32_t i2 = (uint32_t) qs[2 * l + 1]
                                          | (((uint32_t) qh[h] << (7 - 2 * l)) & 256u);
                        const uint32_t g1 = iq3s_grid[i1];
                        const uint32_t g2 = iq3s_grid[i2];
                        const uint8_t  sg = signs[l];
                        for (int j = 0; j < 4; ++j) {
                            y[j + 0] = db * (float) ((g1 >> (8 * j)) & 0xff)
                                     * ((sg & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                            y[j + 4] = db * (float) ((g2 >> (8 * j)) & 0xff)
                                     * ((sg & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                        }
                        y += 8;
                    }
                    qs += 8; signs += 4;
                }
                qh += 2;
            }
        }
        return true;
    case SB_IQ4_NL:
        for (int b = 0; b < n / 32; ++b, x += 18) {
            const float d = sb_host_f16(x);
            const uint8_t * qs = x + 2;
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = d * (float) kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = d * (float) kvalues_iq4nl[qs[j] >>  4];
            }
            y += 32;
        }
        return true;
    case SB_IQ4_XS:
        // d | scales_h | scales_l[4] | qs[128]
        for (int b = 0; b < n / 256; ++b, x += 136) {
            const float d = sb_host_f16(x);
            const uint16_t scales_h = (uint16_t) (x[2] | (x[3] << 8));
            const uint8_t * scales_l = x + 4;
            const uint8_t * qs = x + 8;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                             | (((scales_h >> (2 * ib)) & 3) << 4);
                const float dl = d * (float) (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    y[j +  0] = dl * (float) kvalues_iq4nl[qs[j] & 0xf];
                    y[j + 16] = dl * (float) kvalues_iq4nl[qs[j] >>  4];
                }
                y += 32;
                qs += 16;
            }
        }
        return true;
    case SB_Q2_0:
        for (int b = 0; b < n / 64; ++b, x += 18) {
            const float d = sb_host_f16(x);
            const uint8_t * qs = x + 2;
            for (int j = 0; j < 64; ++j) {
                const uint8_t q = (qs[j >> 2] >> ((j & 3) * 2)) & 3;
                *y++ = ((float) q - 1.0f) * d;
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

__device__ static inline float sub32_q2_0(const uint8_t * row, int s, const float * x) {
    // 64 elements per block, so one block covers two sub-blocks.
    const uint8_t * b = row + (size_t) (s >> 1) * 18;
    const uint8_t * qs = b + 2 + (s & 1) * 8;
    float sxq = 0.f;
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
        const uint8_t q = (qs[l >> 2] >> ((l & 3) * 2)) & 3;
        sxq += x[l] * ((float) q - 1.0f);
    }
    return h2f(b) * sxq;
}

// The IQ-family super-blocks are 256 elements = eight "ib32" groups, so sub-block
// s is decoded on its own; the index arithmetic below mirrors sb_host_dequant and
// is what the kernel test (unit-vector dot) diffs against ggml/gguf-py.

__device__ static inline float sub32_iq2_s(const uint8_t * row, int s, const float * x) {
    const uint8_t * b = row + (size_t) (s >> 3) * 82;
    const int j = s & 7;
    const float d = h2f(b);
    const uint8_t * qs    = b + 2;
    const uint8_t * qh    = b + 66;
    const uint8_t * signs = qs + 32;
    const uint8_t sc = b[74 + j];
    const float db0 = d * (0.5f + (sc & 0xf)) * 0.25f;
    const float db1 = d * (0.5f + (sc >>  4)) * 0.25f;
    float sxq = 0.f;
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const float dl = (l < 2) ? db0 : db1;
        const uint32_t gi = (uint32_t) qs[j * 4 + l]
                          | (((uint32_t) qh[j] << (8 - 2 * l)) & 0x300u);
        const uint64_t g  = iq2s_grid_dev[gi];
        const uint8_t  sg = signs[j * 4 + l];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const float gv  = (float) (uint8_t) (g >> (8 * m));
            const float sgn = (sg & kmask_iq2xs_dev[m]) ? -1.f : 1.f;
            sxq += (dl * gv * sgn) * x[l * 8 + m];
        }
    }
    return sxq;
}

__device__ static inline float sub32_iq3_xxs(const uint8_t * row, int s, const float * x) {
    const uint8_t * b = row + (size_t) (s >> 3) * 98;
    const int j = s & 7;
    const float d = h2f(b);
    const uint8_t * qs = b + 2;
    const uint8_t * ss = qs + 64;
    uint32_t aux32;
    memcpy(&aux32, ss + 4 * j, 4);
    const float db = d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
    float sxq = 0.f;
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint8_t  sg = ksigns_iq2xs_dev[(aux32 >> (7 * l)) & 127];
        const uint32_t g1 = iq3xxs_grid_dev[qs[8 * j + 2 * l + 0]];
        const uint32_t g2 = iq3xxs_grid_dev[qs[8 * j + 2 * l + 1]];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float s1 = (sg & kmask_iq2xs_dev[m + 0]) ? -1.f : 1.f;
            const float s2 = (sg & kmask_iq2xs_dev[m + 4]) ? -1.f : 1.f;
            sxq += (db * (float) ((g1 >> (8 * m)) & 0xff) * s1) * x[l * 8 + m + 0];
            sxq += (db * (float) ((g2 >> (8 * m)) & 0xff) * s2) * x[l * 8 + m + 4];
        }
    }
    return sxq;
}

__device__ static inline float sub32_iq3_s(const uint8_t * row, int s, const float * x) {
    const uint8_t * b = row + (size_t) (s >> 3) * 110;
    const int j = s & 7;
    const float d = h2f(b);
    const uint8_t * qs    = b + 2;
    const uint8_t * qh    = b + 66;
    const uint8_t * signs = b + 74;
    const uint8_t * sc    = b + 106;
    const int p = j >> 1, h = j & 1;
    const float db = d * (1 + 2 * (float) ((sc[p] >> (4 * h)) & 0xf));
    float sxq = 0.f;
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t i1 = (uint32_t) qs[j * 8 + 2 * l + 0]
                          | (((uint32_t) qh[j] << (8 - 2 * l)) & 256u);
        const uint32_t i2 = (uint32_t) qs[j * 8 + 2 * l + 1]
                          | (((uint32_t) qh[j] << (7 - 2 * l)) & 256u);
        const uint32_t g1 = iq3s_grid_dev[i1];
        const uint32_t g2 = iq3s_grid_dev[i2];
        const uint8_t  sg = signs[j * 4 + l];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float s1 = (sg & kmask_iq2xs_dev[m + 0]) ? -1.f : 1.f;
            const float s2 = (sg & kmask_iq2xs_dev[m + 4]) ? -1.f : 1.f;
            sxq += (db * (float) ((g1 >> (8 * m)) & 0xff) * s1) * x[l * 8 + m + 0];
            sxq += (db * (float) ((g2 >> (8 * m)) & 0xff) * s2) * x[l * 8 + m + 4];
        }
    }
    return sxq;
}

__device__ static inline float sub32_iq4_nl(const uint8_t * row, int s, const float * x) {
    const uint8_t * b = row + (size_t) s * 18;
    const float d = h2f(b);
    const uint8_t * qs = b + 2;
    float sxq = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        sxq += (d * (float) kvalues_iq4nl_dev[qs[j] & 0xf]) * x[j +  0];
        sxq += (d * (float) kvalues_iq4nl_dev[qs[j] >>  4]) * x[j + 16];
    }
    return sxq;
}

__device__ static inline float sub32_iq4_xs(const uint8_t * row, int s, const float * x) {
    const uint8_t * b = row + (size_t) (s >> 3) * 136;
    const int j = s & 7;
    const float d = h2f(b);
    const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
    const uint8_t * scales_l = b + 4;
    const uint8_t * qs = b + 8;
    const int ls = ((scales_l[j / 2] >> (4 * (j % 2))) & 0xf)
                 | (((scales_h >> (2 * j)) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    float sxq = 0.f;
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        sxq += (dl * (float) kvalues_iq4nl_dev[qs[j * 16 + m] & 0xf]) * x[m +  0];
        sxq += (dl * (float) kvalues_iq4nl_dev[qs[j * 16 + m] >>  4]) * x[m + 16];
    }
    return sxq;
}

__device__ static inline float sub32(int qt, const uint8_t * row, int s, const float * x) {
    switch (qt) {
        case SB_Q4_K:    return sub32_q4_K(row, s, x);
        case SB_Q5_K:    return sub32_q5_K(row, s, x);
        case SB_Q5_1:    return sub32_q5_1(row, s, x);
        case SB_Q8_0:    return sub32_q8_0(row, s, x);
        case SB_IQ3_XXS: return sub32_iq3_xxs(row, s, x);
        case SB_IQ4_NL:  return sub32_iq4_nl(row, s, x);
        case SB_IQ3_S:   return sub32_iq3_s(row, s, x);
        case SB_IQ2_S:   return sub32_iq2_s(row, s, x);
        case SB_IQ4_XS:  return sub32_iq4_xs(row, s, x);
        case SB_Q2_0:    return sub32_q2_0(row, s, x);
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

// Safe diagnostic: reports what the runtime would conclude about a host pointer
// that the model's tensors live in, without loading anything or touching CUDA.
// Lets the disk path be validated against a real mmap in milliseconds.
SABAH_API int sabah_rt_resolve_probe(void * ptr, char * buf, size_t n) {
    if (!buf || n == 0) return -1;
    buf[0] = 0;
    size_t   fi  = (size_t) -1;
    uint64_t off = 0;
    std::string why;
    const bool ok = sb_resolve(ptr, fi, off, &why);
    if (ok) {
        const sb_disk_file & f = g_disk_files[fi];
        snprintf(buf, n,
                 "resolved=1 file=%s offset=%llu file_size=%llu align=%llu direct_io=%d depth=%llu",
                 f.path.c_str(), (unsigned long long) off, (unsigned long long) f.size,
                 (unsigned long long) SB_IO_ALIGN,
                 sb_direct_io_enabled() ? 1 : 0, (unsigned long long) sb_io_depth());
    } else {
        snprintf(buf, n, "resolved=0 reason=%s", why.empty() ? "unknown" : why.c_str());
    }
    return ok ? 0 : -1;
}

// Reads `bytes` at `off` with an unbuffered read and reports the first 8 bytes,
// so the direct path can be proven to deliver the same bytes as the mapping.
SABAH_API int sabah_rt_read_probe(uint64_t off, size_t bytes, char * buf, size_t n) {
    if (!buf || n == 0) return -1;
    buf[0] = 0;
    if (g_disk_files.empty()) { snprintf(buf, n, "no file registered"); return -1; }
    const size_t   len = ((bytes + SB_IO_ALIGN - 1) / SB_IO_ALIGN) * SB_IO_ALIGN;
    sb_io_slot * s = sb_io_slot_at(0, len);
    if (!s) { snprintf(buf, n, "staging allocation failed"); return -1; }
    sb_io_req r;
    r.file = 0; r.off = off & ~(uint64_t) (SB_IO_ALIGN - 1);
    r.len = len; r.staging = s->staging; r.delta = (size_t) (off - r.off);
    r.bytes = bytes; r.dst = nullptr;
#if defined(_WIN32)
    r.ev = s->ev;
#endif
    if (!sb_io_issue(r) || !sb_io_wait(r) || r.got < r.delta + bytes) {
        snprintf(buf, n, "read failed (got %llu of %llu)", (unsigned long long) r.got,
                 (unsigned long long) (r.delta + bytes));
        return -1;
    }
    const uint8_t * q = (const uint8_t *) s->staging + r.delta;
    char hex[64];
    for (int i = 0; i < 8; ++i) snprintf(hex + i * 3, 4, "%02x ", q[i]);
    snprintf(buf, n, "read_ok bytes=%llu first8=%s", (unsigned long long) r.got, hex);
    return 0;
}

static int sb_snapshot_json(char * buf, size_t n);

static void sb_report_diag() {
    if (!std::getenv("SABAH_LLAMA_TRACE")) return;
    std::fprintf(stderr,
                 "SABAH_DIAG calls=%llu tokens=%llu lookups=%llu overflow=%llu verify_ok=%llu verify_fail=%llu\n",
                 (unsigned long long) g_mmid_calls, (unsigned long long) g_mmid_tokens,
                 (unsigned long long) (g_mmid_hits + g_mmid_misses),
                 (unsigned long long) g_mmid_overflow,
                 (unsigned long long) g_mmid_verify_ok, (unsigned long long) g_mmid_verify_fail);
    std::fprintf(stderr,
                 "SABAH_DISK unbuffered_reads=%llu bytes=%llu waves=%llu mapped_fallbacks=%llu staging_bytes=%llu direct_io=%d depth=%llu lanes=%d lane_reads=%llu sync=%llu pending=%llu\n",
                 (unsigned long long) g_io_reads,
                 (unsigned long long) g_io_read_bytes,
                 (unsigned long long) g_io_waves,
                 (unsigned long long) g_io_fallbacks,
                 (unsigned long long) g_io_host_bytes,
                 sb_direct_io_enabled() ? 1 : 0,
                 (unsigned long long) sb_io_depth(),
                 g_wave_workers,
                 (unsigned long long) g_io_lane_reads,
                 (unsigned long long) g_io_sync_reads,
                 (unsigned long long) g_io_pending_reads);
    {
        // Sum of the individual read calls against the wall time the caller
        // spent inside the waves: ~1.0 means the lanes were served one at a
        // time, a larger number means they really overlapped.
        unsigned long long read_ns = 0;
        for (int i = 0; i < SB_IO_LANES; ++i) read_ns += g_io_read_ns[i];
        std::fprintf(stderr, "SABAH_IO read_ms=%.1f wave_ms=%.1f overlap=%.2f reads_per_wave=%.2f\n",
                     (double) read_ns / 1e6, g_t_wait,
                     g_t_wait > 0.0 ? ((double) read_ns / 1e6) / g_t_wait : 0.0,
                     g_io_waves > 0 ? (double) g_io_reads / (double) g_io_waves : 0.0);
    }
    {
        const double calls = g_mmid_calls > 0 ? (double) g_mmid_calls : 1.0;        std::fprintf(stderr,
                     "SABAH_PHASE calls=%llu ids=%.1fms fetch=%.1fms wait=%.1fms slot=%.1fms ptr=%.1fms prime=%.1fms "
                     "| per_call us: ids=%.0f fetch=%.0f wait=%.0f slot=%.0f ptr=%.0f prime=%.0f | accounting=%.1fms\n",
                     (unsigned long long) g_mmid_calls,
                     g_t_ids, g_t_fetch, g_t_wait, g_t_slot, g_t_ptr, g_t_prime,
                     g_t_ids / calls * 1000.0, g_t_fetch / calls * 1000.0,
                     g_t_wait / calls * 1000.0, g_t_slot / calls * 1000.0,
                     g_t_ptr / calls * 1000.0, g_t_prime / calls * 1000.0,
                     g_t_ids + g_t_fetch + g_t_slot + g_t_ptr + g_t_prime);
    }
    if (sb_ev_on()) {
        const double dcalls = g_ev_calls > 0 ? (double) g_ev_calls : 1.0;
        const double dtok = dcalls / 144.0;
        std::fprintf(stderr,
                     "SABAH_EV decode_calls=%llu kernel_ms=%.1f launch_ms=%.1f "
                     "| per_decode_token ms: kernel=%.2f launch=%.2f "
                     "| per_call us: kernel=%.1f launch=%.1f "
                     "| gu calls=%llu kernel_ms=%.1f per_token=%.2f per_call_us=%.1f "
                     "| dn calls=%llu kernel_ms=%.1f per_token=%.2f per_call_us=%.1f "
                     "| all_calls=%llu kernel_ms=%.1f launch_ms=%.1f\n",
                     (unsigned long long) g_ev_calls, g_ev_kernel_ms, g_ev_launch_ms,
                     g_ev_kernel_ms / dtok, g_ev_launch_ms / dtok,
                     g_ev_kernel_ms / dcalls * 1000.0, g_ev_launch_ms / dcalls * 1000.0,
                     (unsigned long long) g_ev_calls_gu, g_ev_kernel_ms_gu,
                     g_ev_kernel_ms_gu / dtok,
                     g_ev_calls_gu ? g_ev_kernel_ms_gu / g_ev_calls_gu * 1000.0 : 0.0,
                     (unsigned long long) g_ev_calls_dn, g_ev_kernel_ms_dn,
                     g_ev_kernel_ms_dn / dtok,
                     g_ev_calls_dn ? g_ev_kernel_ms_dn / g_ev_calls_dn * 1000.0 : 0.0,
                     (unsigned long long) g_ev_calls_all, g_ev_kernel_ms_all, g_ev_launch_ms_all);
    }
    std::fprintf(stderr,
                 "SABAH_IDS reused=%llu group_max=%llu check=%d check_ok=%llu check_fail=%llu\n",
                 (unsigned long long) g_id_reuse, (unsigned long long) g_id_group_max,
                 g_id_check ? 1 : 0,
                 (unsigned long long) g_id_check_ok, (unsigned long long) g_id_check_fail);
    std::fprintf(stderr,
                 "SABAH_MERGE enabled=%d reserved=%llu\n",
                 sb_layer_merge_enabled() ? 1 : 0,
                 (unsigned long long) g_mp_reserved);
    std::fprintf(stderr,
                 "SABAH_PREFETCH enabled=%d reads=%llu read_bytes=%llu used=%llu used_bytes=%llu dropped=%llu stale=%llu queued=%llu waits=%llu wait_ms=%.1f wait_timeouts=%llu\n",
                 g_pf_enabled ? 1 : 0,
                 (unsigned long long) g_pf_reads,
                 (unsigned long long) g_pf_bytes,
                 (unsigned long long) g_pf_used,
                 (unsigned long long) g_pf_used_bytes,
                 (unsigned long long) g_pf_drops,
                 (unsigned long long) g_pf_stale,
                 (unsigned long long) g_pf_queue.size(),
                 (unsigned long long) g_pf_waits,
                 g_pf_wait_ms,
                 (unsigned long long) g_pf_wait_timeouts);
    char buf[1024];
    sb_snapshot_json(buf, sizeof(buf));
    std::fprintf(stderr, "SABAH_SNAPSHOT %s\n", buf);
}

SABAH_API int sabah_rt_init(int device) {
    if (!ck(cudaSetDevice(device), "cudaSetDevice")) return -1;
    if (!ck(cudaStreamCreate(&g_copy),    "create copy stream"))    return -1;
    if (!ck(cudaStreamCreate(&g_compute), "create compute stream")) return -1;
    sb_mmid_reset();
    sb_slab_init(sb_mmid_capacity_bytes());
    static bool registered = false;
    if (!registered) { std::atexit(sb_report_diag); registered = true; }
    return 0;
}

// --- Pinned host memory pool globals (must precede sb_pinned_pool_destroy) ---
struct PinnedBlock {
    void * ptr = nullptr;
    size_t bytes = 0;
    int refcount = 0;
};

static std::mutex g_pinned_mutex;
static std::vector<PinnedBlock> g_pinned_pool;


static void sb_pinned_pool_destroy() {
    std::lock_guard<std::mutex> lock(g_pinned_mutex);
    for (auto & blk : g_pinned_pool) {
        if (blk.ptr) cudaFreeHost(blk.ptr);
    }
    g_pinned_pool.clear();
}


SABAH_API int sabah_rt_shutdown(void) {
    sb_pf_quit();          // stop reserving and copying before the tier is freed
    sb_io_pool_stop();     // and stop the wave readers before staging is freed
    sb_mmid_reset();
    sb_slab_destroy();
    sb_io_pool_destroy();
    sb_scratch_destroy();
    sb_pinned_pool_destroy();
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

// --- Pinned host memory pool with reference counting -------------------------
// Avoids "resource already mapped" by reusing allocations and only freeing
// when the last reference is released.
SABAH_API void * sabah_host_alloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pinned_mutex);
    // Find a free block of sufficient size
    for (auto & blk : g_pinned_pool) {
        if (blk.refcount == 0 && blk.bytes >= bytes) {
            blk.refcount = 1;
            return blk.ptr;
        }
    }
    // Allocate new
    void * p = 0;
    if (!ck(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), "cudaHostAlloc")) return 0;
    g_pinned_pool.push_back({ p, bytes, 1 });
    return p;
}

SABAH_API void sabah_host_free(void * p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(g_pinned_mutex);
    for (auto & blk : g_pinned_pool) {
        if (blk.ptr == p) {
            if (blk.refcount > 0) blk.refcount--;
            // Note: we keep the allocation in the pool for reuse.
            // Actual cudaFreeHost happens at shutdown.
            return;
        }
    }
    // Not from pool — fall back to direct free (should not happen in normal use)
    cudaFreeHost(p);
}

// Register already-allocated host memory as pinned (for async DMA)
// The pointer must be aligned to 256 bytes and size multiple of 256 for best perf.
// We use cudaHostRegisterDefault which allows the memory to be used with any stream.
SABAH_API int sabah_host_register(void * ptr, size_t bytes) {
    return ck(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault),
              "cudaHostRegister") ? 0 : -1;
}

SABAH_API int sabah_host_unregister(void * ptr) {
    return ck(cudaHostUnregister(ptr), "cudaHostUnregister") ? 0 : -1;
}

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
// The mirror of the above: make the COPY stream wait for a compute-stream
// event. Needed when a VRAM slot is about to be overwritten and a kernel that
// still reads it has been submitted but not necessarily executed. This is the
// precise alternative to draining the whole compute stream on every eviction.
SABAH_API int  sabah_copy_wait_event(void * ev) {
    return ck(cudaStreamWaitEvent(g_copy, (cudaEvent_t) ev, 0), "copyStreamWaitEvent") ? 0 : -1;
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
                         const uint8_t * ids_raw, size_t ids_id_stride, size_t ids_token_stride,
                         int n_ids, int n_tokens,
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
        const int expert = *(const int32_t *) (ids_raw
                             + (size_t) t * ids_token_stride
                             + (size_t) slot * ids_id_stride);
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
        void * stream_ptr,
        const void * ids_key) {
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

    const auto t_ids0 = std::chrono::steady_clock::now();
    const size_t ids_bytes = (size_t) (n_ids - 1) * ids_id_stride
                           + (size_t) (n_tokens - 1) * ids_token_stride
                           + sizeof(int32_t);
    static thread_local std::vector<uint8_t> ids_raw;
    static thread_local std::vector<uint8_t> ids_fresh;
    if (ids_raw.size() < ids_bytes) ids_raw.resize(ids_bytes);
    if (ids_fresh.size() < ids_bytes) ids_fresh.resize(ids_bytes);

    // The three expert tensors of one MoE layer read the SAME router tensor, so
    // the ids downloaded for the first of them are still the ids of the second
    // and third. Noticing that (same tensor as the call before, short group)
    // turns two of three stream drains per layer into nothing, which lets the
    // GPU run a layer's kernels back to back instead of one at a time.
    static const void * last_ids_key  = nullptr;
    static size_t       last_ids_bytes = 0;
    static uint64_t     group_len      = 0;
    const bool adjacent = ids_key != nullptr && ids_key == last_ids_key &&
                          ids_bytes == last_ids_bytes;
    const bool reuse = adjacent && !g_id_check && group_len < SB_ID_GROUP_MAX;
    group_len = reuse ? group_len + 1 : 1;
    if (reuse) {
        ++g_id_reuse;
        if (group_len > g_id_group_max) g_id_group_max = group_len;
    }

    if (!reuse) {
        // First call of a group: the routing is not known yet, and the host MUST
        // wait for the router to produce it. There is no way to overlap this with
        // the same call's own work, because that work depends on the answer.
        // (Overlapping it across tokens is the job of the prefetcher, which
        // predicts ids instead of waiting for them.)
        uint8_t * into = (g_id_check && adjacent) ? ids_fresh.data() : ids_raw.data();
        if (!ck(cudaMemcpyAsync(into, d_ids, ids_bytes,
                                cudaMemcpyDeviceToHost, compute_stream),
                "Sabah router id download")) {
            return -1;
        }
        if (!ck(cudaStreamSynchronize(compute_stream),
                "Sabah router id synchronize")) {
            return -1;
        }
        if (g_id_check && adjacent) {
            // Evidence that the reuse above is byte for byte the same routing.
            if (std::memcmp(ids_raw.data(), ids_fresh.data(), ids_bytes) == 0) {
                ++g_id_check_ok;
            } else {
                ++g_id_check_fail;
            }
        }
        // The synchronise above drained the compute stream, so every event pair
        // recorded since the last drain has completed and can be read now.
        if (sb_ev_on() && !g_ev_pending.empty()) {
            for (const auto & p : g_ev_pending) {
                float ms = 0.f;
                if (cudaEventElapsedTime(&ms, p.a, p.b) == cudaSuccess) {
                    g_ev_kernel_ms_all += (double) ms;
                    if (p.nt == 1) g_ev_kernel_ms += (double) ms;
                }
                g_ev_launch_ms_all += p.host_ms;
                ++g_ev_calls_all;
                if (p.nt == 1) {
                    g_ev_launch_ms += p.host_ms;
                    ++g_ev_calls;
                    if (p.rows < 1024) {
                        g_ev_kernel_ms_gu += (double) ms;
                        ++g_ev_calls_gu;
                    } else {
                        g_ev_kernel_ms_dn += (double) ms;
                        ++g_ev_calls_dn;
                    }
                }
            }
            g_ev_pending.clear();
        }
    }
    last_ids_key   = ids_key;
    last_ids_bytes = ids_bytes;
    g_t_ids += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t_ids0).count();

    // While these ids are in hand, the experts the following calls of this same
    // layer will need are already known - only their bytes are missing.
    g_memos_n = 0;
    sb_pf_start();
    sb_pf_desc followers[2];
    const int n_followers = sb_chain_note(source, expert_stride, n_experts, n_ids, n_tokens,
                                         ids_id_stride, ids_raw.data(), followers);
    // The staging readahead is superseded by the merged fetch below whenever that
    // is active: the merged fetch fills the followers' slots from this call's own
    // wave, so there is nothing left for a readahead to guess at, and running both
    // would read the same experts twice.
    if (g_pf_enabled && !sb_layer_merge_enabled() && n_tokens <= SB_PF_MAX_TOKENS) {
        for (int k = 0; k < n_followers; ++k) {
            sb_pf_plan(followers[k], g_pf_prev_ids);
        }
    }
    // A decode call asks about a handful of experts out of tens of thousands of
    // entries. A prefill call asks about nearly all of them, and priming it would
    // cost more than the scans it replaces, so it keeps the per-lookup path.
    const bool small_call = (size_t) n_ids * (size_t) n_tokens <= 64;
    if (small_call) {
        sb_mmid_prime(source, expert_stride, ids_raw.data(),
                      ids_id_stride, ids_token_stride, n_ids, n_tokens, true);
    }

    static thread_local std::vector<const void *> ptrs;
    static thread_local std::vector<sb_fetch>     miss;
    const size_t n_slots = (size_t) n_ids * n_tokens;
    if (ptrs.size() < n_slots) ptrs.resize(n_slots);
    miss.clear();

    // Pass 1: reserve every slot this call needs. Nothing is read from disk
    // yet, so by the time the fetches are issued the whole call's miss set is
    // known and can be fetched with several reads in flight at once.
    for (int token = 0; token < n_tokens; ++token) {
        for (int id_i = 0; id_i < n_ids; ++id_i) {
            const size_t raw_offset = (size_t) token * ids_token_stride
                                    + (size_t) id_i * ids_id_stride;
            const int expert = *(const int32_t *) (ids_raw.data() + raw_offset);
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
                void * wrong = sb_mmid_slot(
                    source, (expert + 1) % n_experts,
                    (const uint8_t *) source + (size_t) ((expert + 1) % n_experts) * expert_stride,
                    expert_stride, miss, compute_stream);
                if (!wrong) return -1;
                ptrs[i] = wrong;
                continue;
            }
#endif
            void * resident = sb_mmid_slot(
                source, expert,
                (const uint8_t *) source + (size_t) expert * expert_stride,
                expert_stride, miss, compute_stream);
            if (!resident) {
                return -1;
            }
            ptrs[i] = resident;
        }
    }

    // The followers of this call - the other two expert tensors of this same
    // layer, which read these very ids - have their slots reserved here, on the
    // first call of the group. Their misses then ride in the one wave below
    // instead of each paying a fresh read latency of its own. See the "merged
    // layer fetch" note above for why that is the whole of the win.
    if (small_call && !reuse && n_followers > 0 && sb_layer_merge_enabled()) {
        for (int k = 0; k < n_followers; ++k) {
            const sb_pf_desc & fo = followers[k];
            if (fo.source == nullptr || fo.expert_stride == 0 || fo.n_experts <= 0) continue;
            sb_mmid_prime(fo.source, fo.expert_stride, ids_raw.data(), ids_id_stride,
                          ids_token_stride, n_ids, n_tokens, false);
            for (int token = 0; token < n_tokens; ++token) {
                for (int id_i = 0; id_i < n_ids; ++id_i) {
                    const size_t raw_offset = (size_t) token * ids_token_stride
                                            + (size_t) id_i * ids_id_stride;
                    const int expert = *(const int32_t *) (ids_raw.data() + raw_offset);
                    if (expert < 0 || expert >= fo.n_experts) continue;
                    void * slot = sb_mmid_slot(
                        fo.source, expert,
                        (const uint8_t *) fo.source + (size_t) expert * fo.expert_stride,
                        fo.expert_stride, miss, compute_stream);
                    if (!slot) return -1;
                    ++g_mp_reserved;
                }
            }
        }
    }

    // Pass 2: one bounded-depth unbuffered read wave set fills every miss.
    if (!sb_mmid_fill(miss)) {
        return -1;
    }

    const auto t_ptr0 = std::chrono::steady_clock::now();
    void ** d_ptrs = nullptr;
    if (!sb_ptr_table(n_slots * sizeof(void *), &d_ptrs)) {
        return -1;
    }
    // The table lives in ONE device buffer that every call rewrites, so this
    // copy is queued on the compute stream: that orders it after the previous
    // call's kernel, which is still reading the same buffer. (Per-call stream
    // drains used to provide that ordering implicitly; the router-id reuse
    // removed two of three of them.) It does not need to wait for the expert
    // transfers, because it carries addresses, not data.
    if (!ck(cudaMemcpyAsync(d_ptrs, ptrs.data(), n_slots * sizeof(void *),
                            cudaMemcpyHostToDevice, compute_stream),
            "Sabah pointer table upload")) {
        return -1;
    }

    if (g_ready_ev == nullptr &&
        !ck(cudaEventCreateWithFlags(&g_ready_ev, cudaEventDisableTiming),
            "Sabah copy event")) {
        return -1;
    }
    if (!ck(cudaEventRecord(g_ready_ev, g_copy), "Sabah copy event record") ||
        !ck(cudaStreamWaitEvent(compute_stream, g_ready_ev, 0), "Sabah compute wait")) {
        return -1;
    }
    g_t_ptr += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t_ptr0).count();

    dim3 grid((unsigned) rows, (unsigned) n_ids, (unsigned) n_tokens);
    const bool ev_on = sb_ev_on();
    sb_ev_pair ev{};
    auto ev_t0 = std::chrono::steady_clock::now();
    if (ev_on) {
        while (g_ev_pool.size() <= g_ev_pending.size()) {
            cudaEvent_t a = nullptr, b = nullptr;
            if (!ck(cudaEventCreateWithFlags(&a, cudaEventDefault), "Sabah event create") ||
                !ck(cudaEventCreateWithFlags(&b, cudaEventDefault), "Sabah event create")) {
                return -1;
            }
            g_ev_pool.push_back({a, b, 0.0});
        }
        ev = g_ev_pool[g_ev_pending.size()];
        if (!ck(cudaEventRecord(ev.a, compute_stream), "Sabah event record")) {
            return -1;
        }
    }
    k_mul_mat_id<<<grid, NT, 0, compute_stream>>>(
        (const void * const *) d_ptrs,
        (const float *) d_x,
        d_dst,
        rows, k, n_ids, n_tokens,
        row_stride, x_id_stride, x_rows, x_token_stride,
        dst_id_stride, dst_token_stride, qt);    if (!ck(cudaGetLastError(), "Sabah MUL_MAT_ID launch")) {
        return -1;
    }
    if (ev_on) {
        if (!ck(cudaEventRecord(ev.b, compute_stream), "Sabah event record")) {
            return -1;
        }
        ev.host_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - ev_t0).count();
        ev.nt = n_tokens;
        ev.rows = rows;
        g_ev_pending.push_back(ev);
    }

    sb_selfcheck(source, expert_stride, row_stride, d_x, x_id_stride, x_rows, x_token_stride,
                 ids_raw.data(), ids_id_stride, ids_token_stride, n_ids, n_tokens,
                 d_dst, dst_id_stride, dst_token_stride,
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

SABAH_API int sabah_slab_init(size_t capacity_bytes) {
    sb_slab_init(capacity_bytes);
    return g_slab_base ? 0 : -1;
}
SABAH_API void * sabah_slab_alloc(size_t bytes) {
    return sb_slab_alloc(bytes);
}
SABAH_API void sabah_slab_free(void *ptr) {
    sb_slab_free(ptr);
}
SABAH_API void sabah_slab_destroy(void) {
    sb_slab_destroy();
}

} // extern "C"
