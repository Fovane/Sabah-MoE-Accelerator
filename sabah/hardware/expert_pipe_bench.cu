// SABAH v4 RUNTIME / M0 : does the expert H2D transfer actually overlap compute?
//
// Q7 modelled the realistic regime as intra-block overlap:
//     t(block) = max( fetch_misses(block) , compute_experts(block) )
// and produced ~46.5 tok/s for a 3x12 GB configuration. That model assumes a
// copy engine can stream expert weights on one stream while the SM array runs
// the expert matmul on another. This harness measures whether that is true on
// real hardware, and what pinned H2D bandwidth is actually achieved at the
// granularity the design uses (one expert = 3,072,000 contiguous bytes).
//
// Measured here:
//   A. pinned vs pageable H2D bandwidth at expert granularity
//   B. H2D bandwidth vs chunk size (is 3 MB big enough to saturate the link?)
//   C. overlap efficiency: copy on stream 1 while a kernel runs on stream 0
//   D. a block-pipeline replay: per block, fetch `miss` experts in `chunks`
//      stages, overlapping stage i's copy with stage i-1's compute, and report
//      the GPU's wait-for-expert time -- the success metric for the runtime.
//
// Build: nvcc -O3 -o expert_pipe_bench expert_pipe_bench.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

// Expert geometry from MODEL_LAYOUT_REPORT.md
static const size_t EXPERT_BYTES = 3072000;     // gate+up (Q4_K) + down (Q5_1)

// A kernel whose duration scales with `iters`, used to stand in for the expert
// matmul so the overlap can be measured against a controllable compute time.
__global__ void burn(float * out, int iters, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = out[i], b = 1.000001f;
    for (int k = 0; k < iters; ++k) a = fmaf(a, b, 1e-7f);
    out[i] = a;
}

static double ms_of(cudaEvent_t a, cudaEvent_t b) {
    float f; CK(cudaEventElapsedTime(&f, a, b)); return (double) f;
}

int main(int argc, char ** argv) {
    int n_experts = 256;        // host-side expert bank (x 3.072 MB)
    int miss      = 4;          // experts fetched per block
    int layers    = 48;
    int chunks    = 4;          // pipeline stages within a block
    int burn_us   = 30;         // target compute time per block, microseconds
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--experts") && i + 1 < argc) n_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--miss") && i + 1 < argc) miss = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunks") && i + 1 < argc) chunks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--burn-us") && i + 1 < argc) burn_us = atoi(argv[++i]);
    }

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("========================================================================\n");
    printf("SABAH v4 RUNTIME / M0 : EXPERT H2D OVERLAP ON REAL HARDWARE\n");
    printf("========================================================================\n");
    printf("device: %s | copy engines: %d | concurrent kernels: %d\n",
           prop.name, prop.asyncEngineCount, prop.concurrentKernels);
    printf("expert = %zu bytes; host bank = %d experts = %.3f GB\n",
           EXPERT_BYTES, n_experts, n_experts * (double) EXPERT_BYTES / 1e9);

    const size_t bank = (size_t) n_experts * EXPERT_BYTES;
    unsigned char * h_pinned = nullptr, * h_paged = nullptr;
    CK(cudaHostAlloc((void **) &h_pinned, bank, cudaHostAllocDefault));
    h_paged = (unsigned char *) malloc(bank);
    if (!h_paged) { fprintf(stderr, "host alloc failed\n"); return 1; }
    memset(h_pinned, 1, bank);
    memset(h_paged, 1, bank);

    const size_t stage = (size_t) miss * EXPERT_BYTES;
    unsigned char * d_buf = nullptr;
    CK(cudaMalloc((void **) &d_buf, stage * 2));       // double buffer

    cudaStream_t s_copy, s_comp;
    CK(cudaStreamCreate(&s_copy));
    CK(cudaStreamCreate(&s_comp));
    cudaEvent_t e0, e1, ec0, ec1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventCreate(&ec0)); CK(cudaEventCreate(&ec1));

    // ---------------- A. pinned vs pageable ----------------------------
    printf("\n--- A. H2D bandwidth, one expert at a time (%zu B each) ---\n", EXPERT_BYTES);
    for (int mode = 0; mode < 2; ++mode) {
        unsigned char * src = mode ? h_paged : h_pinned;
        const int reps = 200;
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int r = 0; r < reps; ++r) {
            size_t off = ((size_t) (r % n_experts)) * EXPERT_BYTES;
            CK(cudaMemcpyAsync(d_buf, src + off, EXPERT_BYTES,
                               cudaMemcpyHostToDevice, s_copy));
        }
        CK(cudaEventRecord(e1, s_copy));
        CK(cudaEventSynchronize(e1));
        double ms = ms_of(e0, e1);
        printf("  %-9s %8.3f ms for %d experts -> %7.2f GB/s\n",
               mode ? "pageable" : "PINNED", ms, reps,
               reps * (double) EXPERT_BYTES / (ms * 1e-3) / 1e9);
    }

    // ---------------- B. bandwidth vs chunk size -----------------------
    printf("\n--- B. H2D bandwidth vs transfer granularity (pinned) ---\n");
    printf("  %-14s %12s %12s\n", "chunk", "ms", "GB/s");
    size_t sizes[] = {256 * 1024, 1024 * 1024, EXPERT_BYTES, 4 * EXPERT_BYTES,
                      16 * EXPERT_BYTES, 64 * EXPERT_BYTES};
    for (size_t sz : sizes) {
        if (sz > bank) continue;
        size_t total = 0; const int reps = std::max<size_t>(4, (size_t)(1.5e9 / sz));
        CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
        for (int r = 0; r < (int) reps; ++r) {
            size_t off = ((size_t) r * sz) % (bank - sz);
            CK(cudaMemcpyAsync(d_buf, h_pinned + off, std::min(sz, stage * 2),
                               cudaMemcpyHostToDevice, s_copy));
            total += std::min(sz, stage * 2);
        }
        CK(cudaEventRecord(e1, s_copy)); CK(cudaEventSynchronize(e1));
        double ms = ms_of(e0, e1);
        printf("  %-14.2f MB %12.3f %12.2f\n", sz / 1e6, ms, total / (ms * 1e-3) / 1e9);
    }

    // ---------------- C. overlap efficiency ----------------------------
    printf("\n--- C. OVERLAP: copy on one stream, kernel on another ---\n");
    const int N = 1 << 20;
    float * d_out; CK(cudaMalloc((void **) &d_out, N * sizeof(float)));
    CK(cudaMemset(d_out, 0, N * sizeof(float)));

    // calibrate the burn kernel to the requested duration
    int iters = 1000;
    for (int tune = 0; tune < 12; ++tune) {
        CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
        burn<<<(N + 255) / 256, 256, 0, s_comp>>>(d_out, iters, N);
        CK(cudaEventRecord(e1, s_comp)); CK(cudaEventSynchronize(e1));
        double us = ms_of(e0, e1) * 1e3;
        if (us > burn_us * 0.9 && us < burn_us * 1.1) break;
        iters = std::max(1, (int) (iters * (burn_us / std::max(us, 1.0))));
    }
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
    burn<<<(N + 255) / 256, 256, 0, s_comp>>>(d_out, iters, N);
    CK(cudaEventRecord(e1, s_comp)); CK(cudaEventSynchronize(e1));
    double t_comp = ms_of(e0, e1);

    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
    CK(cudaMemcpyAsync(d_buf, h_pinned, stage, cudaMemcpyHostToDevice, s_copy));
    CK(cudaEventRecord(e1, s_copy)); CK(cudaEventSynchronize(e1));
    double t_copy = ms_of(e0, e1);

    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
    burn<<<(N + 255) / 256, 256, 0, s_comp>>>(d_out, iters, N);
    CK(cudaMemcpyAsync(d_buf, h_pinned, stage, cudaMemcpyHostToDevice, s_copy));
    CK(cudaEventRecord(ec1, s_comp));
    CK(cudaEventRecord(e1, s_copy));
    CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(ec1));
    CK(cudaDeviceSynchronize());
    double t_both;
    { cudaEvent_t te; CK(cudaEventCreate(&te)); CK(cudaEventRecord(te));
      CK(cudaEventSynchronize(te));
      t_both = std::max(ms_of(e0, e1), ms_of(e0, ec1)); CK(cudaEventDestroy(te)); }

    printf("  compute alone      : %8.3f ms\n", t_comp);
    printf("  copy alone (%.1f MB): %8.3f ms  (%.2f GB/s)\n",
           stage / 1e6, t_copy, stage / (t_copy * 1e-3) / 1e9);
    printf("  both concurrently  : %8.3f ms\n", t_both);
    printf("  serial would be    : %8.3f ms | perfect overlap would be %8.3f ms\n",
           t_comp + t_copy, std::max(t_comp, t_copy));
    double eff = (t_comp + t_copy - t_both) / std::min(t_comp, t_copy);
    printf("  OVERLAP EFFICIENCY : %6.1f%%  (100%% = the shorter one is free)\n",
           100.0 * std::max(0.0, std::min(1.0, eff)));

    // ---------------- D. block-pipeline replay -------------------------
    printf("\n--- D. BLOCK PIPELINE: %d blocks, %d missing experts each, %d stages ---\n",
           layers, miss, chunks);
    const size_t per_stage = (stage + chunks - 1) / chunks;
    std::vector<cudaEvent_t> evs(chunks);
    for (auto & ev : evs) CK(cudaEventCreate(&ev));
    CK(cudaDeviceSynchronize());
    cudaEvent_t g0, g1; CK(cudaEventCreate(&g0)); CK(cudaEventCreate(&g1));
    CK(cudaEventRecord(g0));
    for (int L = 0; L < layers; ++L) {
        for (int c = 0; c < chunks; ++c) {
            size_t off = ((size_t) (L * chunks + c) % (n_experts - 4)) * EXPERT_BYTES;
            size_t sz = std::min(per_stage, stage - c * per_stage);
            CK(cudaMemcpyAsync(d_buf + (c % 2) * stage, h_pinned + off, sz,
                               cudaMemcpyHostToDevice, s_copy));
            CK(cudaEventRecord(evs[c], s_copy));
            CK(cudaStreamWaitEvent(s_comp, evs[c], 0));
            burn<<<(N + 255) / 256, 256, 0, s_comp>>>(d_out, iters / chunks, N);
        }
    }
    CK(cudaEventRecord(g1, s_comp));
    CK(cudaEventSynchronize(g1));
    double t_pipe = ms_of(g0, g1);
    double t_fetch_only = layers * t_copy;
    double t_comp_only = layers * t_comp;
    printf("  pipelined total    : %8.3f ms for %d blocks (%.3f ms/block)\n",
           t_pipe, layers, t_pipe / layers);
    printf("  fetch alone        : %8.3f ms | compute alone %8.3f ms\n",
           t_fetch_only, t_comp_only);
    printf("  serial bound       : %8.3f ms | overlap bound %8.3f ms\n",
           t_fetch_only + t_comp_only, std::max(t_fetch_only, t_comp_only));
    double waited = t_pipe - t_comp_only;
    printf("  GPU WAIT-FOR-EXPERT: %8.3f ms (%.1f%% of the pipeline)\n",
           std::max(0.0, waited), 100.0 * std::max(0.0, waited) / t_pipe);
    printf("\n  Success metric is the wait, not the transfer: %.3f ms of copy\n", t_fetch_only);
    printf("  produced %.3f ms of stall, so %.1f%% of the transfer was hidden.\n",
           std::max(0.0, waited),
           100.0 * (1.0 - std::max(0.0, waited) / std::max(t_fetch_only, 1e-9)));

    CK(cudaFreeHost(h_pinned)); free(h_paged);
    CK(cudaFree(d_buf)); CK(cudaFree(d_out));
    return 0;
}
