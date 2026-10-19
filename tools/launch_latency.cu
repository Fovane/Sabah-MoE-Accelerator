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

int main(int argc, char ** argv) {
    const char * sched = "auto";
    const char * prio  = "normal";
    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--sched=", 8)) sched = argv[i] + 8;
        else if (!std::strncmp(argv[i], "--prio=", 7)) prio = argv[i] + 7;
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
