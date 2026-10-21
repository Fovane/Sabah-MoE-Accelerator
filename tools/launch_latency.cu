// Phase 0d: host-side submission / synchronisation cost probe.
//
// Motivation: the native steady-state token is ~58.6 ms while the GPU is busy
// only ~43.4 ms (nsys). The missing ~15 ms/token is host time. This probe
// isolates the fixed per-call costs so the gap can be attributed to:
//   * kernel launch submission (empty kernel, N launches, one sync)
//   * launch + immediate sync (what a serial dependency costs)
//   * small D2H + sync (the per-layer router-id drain in sabah_rt.cu)
//   * sync policy (cudaDeviceScheduleSpin / BlockingSync / Auto)
//   * thread priority of the submitting thread
//
// Build: nvcc -O3 -arch=sm_120 -o launch_latency.exe launch_latency.cu
// Run:   launch_latency.exe --sched=auto --prio=normal
//
// Nothing here touches the model or the runtime; it is a pure timing probe.

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <windows.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(2); } } while (0)

__global__ void k_empty() {}

// ~30 us of work on this GPU: represents one expert-tensor kernel.
__global__ void k_busy(float * sink, int iters) {
    float a = threadIdx.x * 1e-6f;
    for (int i = 0; i < iters; ++i) a = fmaf(a, 1.000001f, 1e-7f);
    if (a == 1234.5f) sink[0] = a;
}

static double now_ms() {
    using clk = std::chrono::steady_clock;
    static const clk::time_point t0 = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Stats { double mean, p50, p90, p99, min, max; };

static Stats summarise(std::vector<double> v) {
    Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    auto q = [&](double p) { return v[std::min(v.size() - 1, (size_t) (p * v.size()))]; };
    s.mean = sum / v.size(); s.p50 = q(0.5); s.p90 = q(0.9); s.p99 = q(0.99);
    s.min = v.front(); s.max = v.back();
    return s;
}

static void report(const char * name, const Stats & s) {
    std::printf("%-46s mean %8.2f  p50 %8.2f  p90 %8.2f  p99 %8.2f  min %8.2f  max %8.2f  (us)\n",
                name, s.mean * 1000, s.p50 * 1000, s.p90 * 1000, s.p99 * 1000, s.min * 1000, s.max * 1000);
}

// ---------------------------------------------------------------------------
// Token probe: replay one decode token's launch shape and compare eager
// submission with a captured graph, without building llama.cpp.
//
// Measured workload being replayed (results/phase0/REPORT.md):
//   48 layers, 78 launches per layer, 3 Sabah calls per layer
//   (gate/up 80 us, down 73 us), dense 12.6 ms/token = 262 us/layer,
//   one 64 B D2H plus a stream sync per Sabah call (the router-id drain).
//
// Variants:
//   A        eager: 78 launches per layer, drain at each Sabah boundary
//   B_split  one graph per dense block (4 per layer), same drains
//   B_ideal  one graph per layer with all 78 nodes and one drain per layer
//
// A spin kernel is calibrated to each class duration so the GPU busy time per
// token matches the measured workload; the difference between wall time and
// that busy time is the idle gap the host creates.
// ---------------------------------------------------------------------------

// Memory-bound, so the emulated classes have the same power profile as the real
// ones (which stream DRAM at ~330 GB/s): a compute-bound spin hits the 180 W
// limit and throttles the SM clock mid-run, which would make the comparison a
// measurement of the cooler instead of the launch path.
// Each pass reads gridDim*blockDim floats = 64 KiB from a 64 MiB buffer.
#define SPIN_BUF_FLOATS (16u * 1024u * 1024u)
__global__ void k_spin(const float * __restrict__ buf, int iters, float * sink) {
    const unsigned stride = gridDim.x * blockDim.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f;
    for (int k = 0; k < iters; ++k, i = (i + stride) & (SPIN_BUF_FLOATS - 1))
        a += buf[i];
    if (a == 1234.5f) sink[0] = a;
}

static float * g_buf = nullptr;   // 64 MiB, allocated in main

static int calibrate_iters(float target_us, float * d_sink) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    const int batch = 20;
    auto measure = [&](int iters) {
        k_spin<<<128, 128>>>(g_buf, iters, d_sink);      // warm
        CK(cudaStreamSynchronize(0));
        CK(cudaEventRecord(a));
        for (int i = 0; i < batch; ++i) k_spin<<<128, 128>>>(g_buf, iters, d_sink);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        return ms * 1000.0f / batch;
    };
    int lo = 1, hi = 1;
    while (measure(hi) < target_us && hi < (1 << 28)) { lo = hi; hi *= 2; }
    for (int it = 0; it < 24; ++it) {
        int mid = lo + (hi - lo) / 2;
        if (mid == lo) break;
        if (measure(mid) < target_us) lo = mid; else hi = mid;
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return hi;
}

// Kernel mix per layer, from the nsys warm replay (REPORT.md section 2):
//   3,756 kernels/token / 48 layers = 78 launches per layer, of which
//     3   Sabah expert calls  (80, 80, 73 us)
//     15  dense matmuls       (12.6 ms/token = 262 us/layer, ~17.5 us each)
//     60  elementwise/router  (2.5 ms/token = 52 us/layer, ~0.9 us each)
struct TokenPlan {
    int n_dense = 15;
    int n_elem  = 60;
    int sabah_us[3] = {80, 80, 73};
    int layers = 48;
};

static void token_report(const char * name, std::vector<double> wall,
                         const std::vector<double> & submit, double busy_ms) {
    std::sort(wall.begin(), wall.end());
    if (wall.empty()) return;
    const double med = wall[wall.size() / 2];
    double sub = 0;
    for (double x : submit) sub += x;
    sub /= submit.empty() ? 1.0 : (double) submit.size();
    std::printf("%-24s wall p50 %7.3f ms  min %7.3f  max %7.3f  |  "
                "submit p50 %6.3f ms  |  gpu busy %6.3f ms  |  idle %6.3f ms  (%.1f%%)\n",
                name, med, wall.front(), wall.back(), sub, busy_ms,
                med - busy_ms, med > 0 ? 100.0 * (med - busy_ms) / med : 0.0);
}

static void split4(int n, int out[4]) {
    for (int b = 0; b < 4; ++b) out[b] = n * (b + 1) / 4 - n * b / 4;
}

static double busy_per_token(const TokenPlan & plan, int dense_iters, int elem_iters,
                             const int sabah_iters[3], float * d_sink);

static void token_probe(float * d_sink, int * d_ids, TokenPlan & plan) {
    int ids_host[16] = {0};
    const int dense_iters = calibrate_iters(17.5f, d_sink);
    const int elem_iters  = calibrate_iters(0.9f, d_sink);
    int sabah_iters[3];
    for (int i = 0; i < 3; ++i) sabah_iters[i] = calibrate_iters((float) plan.sabah_us[i], d_sink);
    int db[4], eb[4];
    split4(plan.n_dense, db);
    split4(plan.n_elem, eb);

    // The legacy default stream cannot be captured; capture on a private
    // non-blocking stream and launch the resulting graphs anywhere.
    cudaStream_t cap = nullptr;
    CK(cudaStreamCreateWithFlags(&cap, cudaStreamNonBlocking));

    // Achieved duration of each class, measured on this run (not assumed).
    auto class_us = [&](int grid, int block, int iters, int n) {
        for (int i = 0; i < 20; ++i) k_spin<<<grid, block>>>(g_buf, iters, d_sink);
        CK(cudaStreamSynchronize(0));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < n; ++i) k_spin<<<grid, block>>>(g_buf, iters, d_sink);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        return (double) ms * 1000.0 / n;
    };
    const double dense_us = class_us(128, 128, dense_iters, 200);
    const double elem_us  = class_us(64, 32, elem_iters, 2000);
    double sabah_us[3];
    for (int i = 0; i < 3; ++i) sabah_us[i] = class_us(128, 128, sabah_iters[i], 200);
    const double busy_ms = plan.layers * (plan.n_dense * dense_us + plan.n_elem * elem_us
                                          + sabah_us[0] + sabah_us[1] + sabah_us[2]) / 1000.0;
    std::printf("class           us/launch  launches/token   ms/token\n");
    std::printf("dense matmul   %9.2f  %14d  %9.2f\n", dense_us, plan.n_dense * plan.layers,
                plan.layers * plan.n_dense * dense_us / 1000.0);
    std::printf("elementwise    %9.2f  %14d  %9.2f\n", elem_us, plan.n_elem * plan.layers,
                plan.layers * plan.n_elem * elem_us / 1000.0);
    std::printf("sabah          %9.2f  %14d  %9.2f\n",
                (sabah_us[0] + sabah_us[1] + sabah_us[2]) / 3.0, 3 * plan.layers,
                plan.layers * (sabah_us[0] + sabah_us[1] + sabah_us[2]) / 1000.0);
    std::printf("total GPU busy %36s  %9.2f\n", "", busy_ms);

    std::printf("\ntoken probe: %d layers, %d dense (17.5 us) + %d elementwise (~0.9 us)"
                " + 3 sabah (80/80/73 us) per layer = %d launches, busy %.2f ms/token\n",
                plan.layers, plan.n_dense, plan.n_elem, plan.n_dense + plan.n_elem + 3, busy_ms);

    const int N = 30;

    // ---- A: eager, exactly what the runtime does today ----------------------
    {
        std::vector<double> wall, submit;
        for (int t = 0; t < N; ++t) {
            const double t0 = now_ms();
            double s = 0;
            for (int L = 0; L < plan.layers; ++L) {
                for (int b = 0; b < 4; ++b) {
                    const double s0 = now_ms();
                    for (int k = 0; k < db[b]; ++k) k_spin<<<128, 128>>>(g_buf, dense_iters, d_sink);
                    for (int k = 0; k < eb[b]; ++k) k_spin<<<64, 32>>>(g_buf, elem_iters, d_sink);
                    s += now_ms() - s0;
                    if (b < 3) {
                        k_spin<<<128, 128>>>(g_buf, sabah_iters[b], d_sink);
                        CK(cudaMemcpyAsync(ids_host, d_ids, 16 * sizeof(int),
                                           cudaMemcpyDeviceToHost, 0));
                        CK(cudaStreamSynchronize(0));
                    }
                }
            }
            CK(cudaStreamSynchronize(0));   // the token ends only when its last block does
            wall.push_back(now_ms() - t0);
            submit.push_back(s);
        }
        token_report("A eager", wall, submit, busy_ms);
    }

    // ---- A1: eager, but a single drain per layer ---------------------------
    // The spec's structure (one D2H + sync per layer). The three-drain variant
    // above is what the runtime actually does today (the Sabah bridge drains
    // between expert tensors); this one is the lower bound for eager.
    {
        std::vector<double> wall, submit;
        for (int t = 0; t < N; ++t) {
            const double t0 = now_ms();
            double s = 0;
            for (int L = 0; L < plan.layers; ++L) {
                const double s0 = now_ms();
                for (int b = 0; b < 4; ++b) {
                    for (int k = 0; k < db[b]; ++k) k_spin<<<128, 128>>>(g_buf, dense_iters, d_sink);
                    for (int k = 0; k < eb[b]; ++k) k_spin<<<64, 32>>>(g_buf, elem_iters, d_sink);
                    if (b < 3) k_spin<<<128, 128>>>(g_buf, sabah_iters[b], d_sink);
                }
                s += now_ms() - s0;
                CK(cudaMemcpyAsync(ids_host, d_ids, 16 * sizeof(int),
                                   cudaMemcpyDeviceToHost, 0));
                CK(cudaStreamSynchronize(0));
            }
            wall.push_back(now_ms() - t0);
            submit.push_back(s);
        }
        token_report("A1 eager, 1 drain/layer", wall, submit, busy_ms);
    }

    // ---- B_split: one graph per dense block, captured once ------------------
    // 4 graph launches per layer instead of ~75 individual ones; the Sabah call
    // and its D2H drain stay eager, which is the only part that must stay on
    // the host.
    {
        cudaGraph_t g[4] = {};
        cudaGraphExec_t ge[4] = {};
        for (int b = 0; b < 4; ++b) {
            CK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeRelaxed));
            for (int k = 0; k < db[b]; ++k) k_spin<<<128, 128, 0, cap>>>(g_buf, dense_iters, d_sink);
            for (int k = 0; k < eb[b]; ++k) k_spin<<<64, 32, 0, cap>>>(g_buf, elem_iters, d_sink);
            CK(cudaStreamEndCapture(cap, &g[b]));
            CK(cudaGraphInstantiate(&ge[b], g[b], 0));
        }
        std::vector<double> wall, submit;
        for (int t = 0; t < N; ++t) {
            const double t0 = now_ms();
            double s = 0;
            for (int L = 0; L < plan.layers; ++L) {
                for (int b = 0; b < 4; ++b) {
                    const double s0 = now_ms();
                    CK(cudaGraphLaunch(ge[b], 0));
                    s += now_ms() - s0;
                    if (b < 3) {
                        k_spin<<<128, 128>>>(g_buf, sabah_iters[b], d_sink);
                        CK(cudaMemcpyAsync(ids_host, d_ids, 16 * sizeof(int),
                                           cudaMemcpyDeviceToHost, 0));
                        CK(cudaStreamSynchronize(0));
                    }
                }
            }
            CK(cudaStreamSynchronize(0));
            wall.push_back(now_ms() - t0);
            submit.push_back(s);
        }
        token_report("B_split (4 graphs/layer)", wall, submit, busy_ms);
        for (int b = 0; b < 4; ++b) { CK(cudaGraphExecDestroy(ge[b])); CK(cudaGraphDestroy(g[b])); }
    }

    // ---- B_ideal: one graph per layer, one drain per layer -----------------
    // The upper bound: every launch in the layer captured, only the router-id
    // read left on the host.
    {
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        CK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeRelaxed));
        for (int b = 0; b < 4; ++b) {
            for (int k = 0; k < db[b]; ++k) k_spin<<<128, 128, 0, cap>>>(g_buf, dense_iters, d_sink);
            for (int k = 0; k < eb[b]; ++k) k_spin<<<64, 32, 0, cap>>>(g_buf, elem_iters, d_sink);
            if (b < 3) k_spin<<<128, 128, 0, cap>>>(g_buf, sabah_iters[b], d_sink);
        }
        CK(cudaStreamEndCapture(cap, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));

        std::vector<double> wall, submit;
        for (int t = 0; t < N; ++t) {
            const double t0 = now_ms();
            double s = 0;
            for (int L = 0; L < plan.layers; ++L) {
                const double s0 = now_ms();
                CK(cudaGraphLaunch(ge, 0));
                s += now_ms() - s0;
                CK(cudaMemcpyAsync(ids_host, d_ids, 16 * sizeof(int),
                                   cudaMemcpyDeviceToHost, 0));
                CK(cudaStreamSynchronize(0));
            }
            CK(cudaStreamSynchronize(0));
            wall.push_back(now_ms() - t0);
            submit.push_back(s);
        }
        token_report("B_ideal (1 graph/layer)", wall, submit, busy_ms);

        // The per-token cost of changing launch parameters inside an
        // instantiated graph (pointers move every token).
        size_t nn = 0;
        CK(cudaGraphGetNodes(g, nullptr, &nn));
        std::vector<cudaGraphNode_t> nodes(nn);
        CK(cudaGraphGetNodes(g, nodes.data(), &nn));
        cudaGraphNode_t knode = nullptr;
        for (auto n : nodes) {
            cudaGraphNodeType ty;
            CK(cudaGraphNodeGetType(n, &ty));
            if (ty == cudaGraphNodeTypeKernel) { knode = n; break; }
        }
        if (knode) {
            cudaKernelNodeParams kp{};
            CK(cudaGraphKernelNodeGetParams(knode, &kp));
            float * alt = nullptr;
            CK(cudaMalloc(&alt, 16));
            void * args[3] = {&alt, (void *) &dense_iters, (void *) &d_sink};
            kp.kernelParams = args;
            std::vector<double> upd;
            for (int i = 0; i < 2000; ++i) {
                const double t0 = now_ms();
                CK(cudaGraphExecKernelNodeSetParams(ge, knode, &kp));
                upd.push_back(now_ms() - t0);
            }
            Stats us = summarise(upd);
            std::printf("%-24s p50 %7.3f us  mean %7.3f us  (graph has %zu nodes)\n",
                        "SetParams 1 node", us.p50 * 1000, us.mean * 1000, nn);

            // Bulk update path: clone the graph, change every kernel node's
            // pointer in the clone, then time cudaGraphExecUpdate.
            cudaGraph_t g2 = nullptr;
            CK(cudaGraphClone(&g2, g));
            size_t nn2 = 0;
            CK(cudaGraphGetNodes(g2, nullptr, &nn2));
            std::vector<cudaGraphNode_t> nodes2(nn2);
            CK(cudaGraphGetNodes(g2, nodes2.data(), &nn2));
            for (auto n : nodes2) {
                cudaGraphNodeType ty;
                CK(cudaGraphNodeGetType(n, &ty));
                if (ty != cudaGraphNodeTypeKernel) continue;
                cudaKernelNodeParams kp2{};
                CK(cudaGraphKernelNodeGetParams(n, &kp2));
                void * a2[3] = {&alt, (void *) &dense_iters, (void *) &d_sink};
                kp2.kernelParams = a2;
                CK(cudaGraphKernelNodeSetParams(n, &kp2));
            }
            std::vector<double> upd2;
            for (int i = 0; i < 300; ++i) {
                const double t0 = now_ms();
                cudaGraphExecUpdateResultInfo ri{};
                CK(cudaGraphExecUpdate(ge, g2, &ri));
                upd2.push_back(now_ms() - t0);
            }
            Stats us2 = summarise(upd2);
            std::printf("%-24s p50 %7.3f us  mean %7.3f us  (%zu nodes, all updated)\n",
                        "ExecUpdate whole graph", us2.p50 * 1000, us2.mean * 1000, nn2);
            CK(cudaGraphDestroy(g2));
            CK(cudaFree(alt));
        }
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }

    CK(cudaStreamDestroy(cap));
}

static double busy_per_token(const TokenPlan & plan, int dense_iters, int elem_iters,
                             const int sabah_iters[3], float * d_sink) {
    int db[4], eb[4];
    split4(plan.n_dense, db);
    split4(plan.n_elem, eb);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int L = 0; L < plan.layers; ++L) {
        for (int bb = 0; bb < 4; ++bb) {
            for (int k = 0; k < db[bb]; ++k) k_spin<<<128, 128>>>(g_buf, dense_iters, d_sink);
            for (int k = 0; k < eb[bb]; ++k) k_spin<<<64, 32>>>(g_buf, elem_iters, d_sink);
            if (bb < 3) k_spin<<<128, 128>>>(g_buf, sabah_iters[bb], d_sink);
        }
    }
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return (double) ms;
}

int main(int argc, char ** argv) {
    const char * sched = "auto";
    const char * prio  = "normal";
    bool token_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--sched=", 8)) sched = argv[i] + 8;
        else if (!std::strncmp(argv[i], "--prio=", 7)) prio = argv[i] + 7;
        else if (!std::strcmp(argv[i], "--token-probe")) token_mode = true;
    }

    unsigned flags = cudaDeviceScheduleAuto;
    if (!std::strcmp(sched, "spin")) flags = cudaDeviceScheduleSpin;
    else if (!std::strcmp(sched, "blocking")) flags = cudaDeviceScheduleBlockingSync;
    else if (!std::strcmp(sched, "yield")) flags = cudaDeviceScheduleYield;
    CK(cudaSetDeviceFlags(flags));

    HANDLE th = GetCurrentThread();
    if (!std::strcmp(prio, "highest")) SetThreadPriority(th, THREAD_PRIORITY_HIGHEST);
    else if (!std::strcmp(prio, "timecritical")) SetThreadPriority(th, THREAD_PRIORITY_TIME_CRITICAL);
    else SetThreadPriority(th, THREAD_PRIORITY_NORMAL);

    CK(cudaFree(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));

    float * d_sink = nullptr;
    int * d_ids = nullptr;
    CK(cudaMalloc(&d_sink, 16));
    CK(cudaMalloc(&g_buf, SPIN_BUF_FLOATS * sizeof(float)));
    CK(cudaMemset(g_buf, 0, SPIN_BUF_FLOATS * sizeof(float)));
    CK(cudaMalloc(&d_ids, 64));
    int ids[16] = {0};
    CK(cudaMemcpy(d_ids, ids, sizeof(ids), cudaMemcpyHostToDevice));

    // Find the busy-kernel iteration count that lands near 30 us.
    int busy_iters = 4096;
    {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (int trial = 0; trial < 32; ++trial) {
            CK(cudaEventRecord(a));
            k_busy<<<128, 128>>>(d_sink, busy_iters);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
            if (ms > 0.030f) break;
            busy_iters *= 2;
        }
    }

    // Clock ramp: hold the GPU busy for ~1.5 s so measurements are at P0.
    {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < 200; ++i) k_busy<<<1024, 256>>>(d_sink, 8192);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    }

    std::printf("device=%s sm_%d%d  sched=%s prio=%s  busy_kernel_iters=%d  busy_kernel_us=",
                prop.name, prop.major, prop.minor, sched, prio, busy_iters);
    {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a)); k_busy<<<128, 128>>>(d_sink, busy_iters); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        std::printf("%.1f\n", ms * 1000.0);
    }

    if (token_mode) {
        TokenPlan plan;
        token_probe(d_sink, d_ids, plan);
        CK(cudaDeviceSynchronize());
        return 0;
    }

    const int N = 4000;
    std::vector<double> v; v.reserve(N);

    // 1. Empty kernel: launch then sync, one at a time. Fixed round-trip cost.
    for (int i = 0; i < 200; ++i) { k_empty<<<1, 32>>>(); CK(cudaStreamSynchronize(0)); }
    v.clear();
    for (int i = 0; i < N; ++i) { double t = now_ms(); k_empty<<<1, 32>>>(); CK(cudaStreamSynchronize(0)); v.push_back(now_ms() - t); }
    report("empty launch + sync (1 at a time)", summarise(v));

    // 2. Submission throughput: queue 16, sync once.
    v.clear();
    for (int i = 0; i < N / 16; ++i) {
        double t = now_ms();
        for (int j = 0; j < 16; ++j) k_empty<<<1, 32>>>();
        CK(cudaStreamSynchronize(0));
        v.push_back((now_ms() - t) / 16.0);
    }
    report("empty launch, batched 16 per sync (/launch)", summarise(v));

    // 3. Launch-only cost (no sync): 64 launches then one sync, GPU idle.
    v.clear();
    for (int i = 0; i < N / 64; ++i) {
        CK(cudaStreamSynchronize(0));
        double t = now_ms();
        for (int j = 0; j < 64; ++j) k_empty<<<1, 32>>>();
        double submit = now_ms() - t;
        CK(cudaStreamSynchronize(0));
        v.push_back(submit / 64.0);
    }
    report("empty launch submission only (/launch)", summarise(v));

    // 4. 40-byte D2H + sync (the sabah router id drain, no GPU work in front).
    v.clear();
    for (int i = 0; i < N; ++i) {
        double t = now_ms();
        CK(cudaMemcpy(ids, d_ids, 16 * sizeof(int), cudaMemcpyDeviceToHost));
        v.push_back(now_ms() - t);
    }
    report("D2H 64B blocking copy (includes sync)", summarise(v));

    // 5. Busy kernel (~30 us) + sync: a serial dependent step.
    v.clear();
    for (int i = 0; i < N; ++i) {
        double t = now_ms();
        k_busy<<<128, 128>>>(d_sink, busy_iters);
        CK(cudaStreamSynchronize(0));
        v.push_back(now_ms() - t);
    }
    report("busy30 launch + sync (wall)", summarise(v));

    // 6. The real shape: K=48 layers of (kernel, D2H 64B, sync) per token.
    //    K independent steps, each of which must drain before the host proceeds.
    v.clear();
    for (int i = 0; i < 200; ++i) {
        double t = now_ms();
        for (int j = 0; j < 48; ++j) {
            k_busy<<<128, 128>>>(d_sink, busy_iters);
            CK(cudaMemcpyAsync(ids, d_ids, 16 * sizeof(int), cudaMemcpyDeviceToHost, 0));
            CK(cudaStreamSynchronize(0));
        }
        v.push_back(now_ms() - t);
    }
    report("48x (busy30 + D2H + sync) per token", summarise(v));

    // 7. Same, but empty kernels: pure per-layer dependency overhead.
    v.clear();
    for (int i = 0; i < 200; ++i) {
        double t = now_ms();
        for (int j = 0; j < 48; ++j) {
            k_empty<<<1, 32>>>();
            CK(cudaMemcpyAsync(ids, d_ids, 16 * sizeof(int), cudaMemcpyDeviceToHost, 0));
            CK(cudaStreamSynchronize(0));
        }
        v.push_back(now_ms() - t);
    }
    report("48x (empty + D2H + sync) per token", summarise(v));

    // 8. Launch cost with a deep queue (GPU already busy): amortised submission.
    v.clear();
    for (int i = 0; i < 64; ++i) {
        // Fill the pipe with ~2 ms of work, then time submission of 256 empties.
        for (int j = 0; j < 8; ++j) k_busy<<<1024, 256>>>(d_sink, 8192);
        double t = now_ms();
        for (int j = 0; j < 256; ++j) k_empty<<<1, 32>>>();
        double submit = now_ms() - t;
        CK(cudaStreamSynchronize(0));
        v.push_back(submit / 256.0);
    }
    report("empty launch submission, GPU busy (/launch)", summarise(v));

    CK(cudaFree(d_sink)); CK(cudaFree(d_ids));
    return 0;
}
