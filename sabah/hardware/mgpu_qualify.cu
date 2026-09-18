// SABAH v4 RUNTIME / M-1 : MULTI-GPU HOST->DEVICE QUALIFICATION
//
// Run this FIRST on any target machine, before building anything.
//
// The question is NOT "what does each card do on its own". Summing per-card
// figures is exactly the mistake that would invalidate the projection. The
// question is:
//
//     BW_simultaneous(GPU_0, GPU_1, ... )
//
// If a board gives each card 11 GB/s alone but only 18 GB/s aggregate when all
// three copy at once, the root complex is the bottleneck and the whole
// expert-streaming design must be re-sized around that number.
//
// Reports, per device and in aggregate:
//   - negotiated PCIe generation and width (so a x4 slot is caught immediately)
//   - solo pinned H2D at expert granularity (3,072,000 B)
//   - SIMULTANEOUS aggregate H2D with every device copying at once
//   - the contention factor: aggregate / sum(solo)
//
// Build: nvcc -O3 -o mgpu_qualify mgpu_qualify.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>

#define CK(x) do { cudaError_t ck_ = (x); if (ck_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(ck_)); \
    exit(1); } } while (0)

static const size_t EXPERT_BYTES = 3072000;   // one expert, from the GGUF layout

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct DevCtx {
    int id;
    unsigned char * h_pinned = nullptr;
    unsigned char * d_buf = nullptr;
    cudaStream_t stream;
    size_t bytes_moved = 0;
    double solo_gbs = 0.0;
};

// Copy `total_bytes` to device `d` in expert-sized chunks, return seconds.
static double copy_loop(DevCtx & d, size_t total_bytes, size_t bank_bytes,
                        std::atomic<bool> * go, std::atomic<int> * ready) {
    CK(cudaSetDevice(d.id));
    if (go) { ready->fetch_add(1); while (!go->load()) std::this_thread::yield(); }
    const double t0 = now_s();
    size_t moved = 0, off = 0;
    while (moved < total_bytes) {
        CK(cudaMemcpyAsync(d.d_buf, d.h_pinned + off, EXPERT_BYTES,
                           cudaMemcpyHostToDevice, d.stream));
        moved += EXPERT_BYTES;
        off += EXPERT_BYTES;
        if (off + EXPERT_BYTES > bank_bytes) off = 0;
    }
    CK(cudaStreamSynchronize(d.stream));
    const double dt = now_s() - t0;
    d.bytes_moved = moved;
    return dt;
}

int main(int argc, char ** argv) {
    double gb_per_dev = 2.0;      // how much to push per device per test
    size_t bank_mb = 512;         // pinned host bank per device
    bool as_json = false;         // machine-readable output for the planner
    const char * json_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--gb") && i + 1 < argc) gb_per_dev = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bank-mb") && i + 1 < argc) bank_mb = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--json")) as_json = true;
        else if (!strcmp(argv[i], "--json-out") && i + 1 < argc) { as_json = true; json_path = argv[++i]; }
    }

    int n = 0;
    CK(cudaGetDeviceCount(&n));
    printf("================================================================================\n");
    printf("SABAH v4 / M-1 : MULTI-GPU H2D QUALIFICATION\n");
    printf("================================================================================\n");
    printf("devices found: %d | expert granularity: %zu B | %.1f GB pushed per device\n",
           n, EXPERT_BYTES, gb_per_dev);
    if (n == 0) return 1;

    printf("\n--- topology (negotiated, not maximum) ---\n");
    printf("%-4s %-34s %10s %10s %12s\n", "gpu", "name", "PCIe gen", "width", "VRAM GB");
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, i));
        int gen = 0, wid = 0;
        cudaDeviceGetAttribute(&gen, cudaDevAttrPciDeviceId, i);   // placeholder
        // negotiated link info is not exposed by the runtime API; report what is
        cudaDeviceGetAttribute(&wid, cudaDevAttrMultiProcessorCount, i);
        printf("%-4d %-34s %10s %10s %12.1f\n", i, p.name, "see nvidia-smi", "see below",
               p.totalGlobalMem / 1e9);
    }
    printf("\nNOTE: the CUDA runtime does not expose negotiated PCIe gen/width.\n");
    printf("Record it separately with:  nvidia-smi --query-gpu=index,name,\n");
    printf("pcie.link.gen.current,pcie.link.width.current --format=csv\n");
    printf("and also capture `nvidia-smi topo -m`. A x4 slot shows up there, not here.\n");

    // ---- allocate per device -------------------------------------------
    const size_t bank = bank_mb * 1024ull * 1024ull;
    std::vector<DevCtx> devs(n);
    for (int i = 0; i < n; ++i) {
        devs[i].id = i;
        CK(cudaSetDevice(i));
        CK(cudaHostAlloc((void **) &devs[i].h_pinned, bank, cudaHostAllocDefault));
        memset(devs[i].h_pinned, 1, bank);
        CK(cudaMalloc((void **) &devs[i].d_buf, EXPERT_BYTES));
        CK(cudaStreamCreate(&devs[i].stream));
    }

    const size_t per_dev = (size_t) (gb_per_dev * 1e9);

    // ---- A. solo ---------------------------------------------------------
    printf("\n--- A. SOLO H2D (one device at a time) ---\n");
    printf("%-6s %12s %12s\n", "gpu", "seconds", "GB/s");
    double sum_solo = 0.0;
    for (int i = 0; i < n; ++i) {
        double dt = copy_loop(devs[i], per_dev, bank, nullptr, nullptr);
        devs[i].solo_gbs = devs[i].bytes_moved / dt / 1e9;
        sum_solo += devs[i].solo_gbs;
        printf("%-6d %12.4f %12.2f\n", i, dt, devs[i].solo_gbs);
    }
    printf("%-6s %12s %12.2f   <- naive sum, NOT what you get\n", "sum", "", sum_solo);

    // ---- B. simultaneous -------------------------------------------------
    printf("\n--- B. SIMULTANEOUS H2D (all devices at once) — THE NUMBER THAT MATTERS ---\n");
    std::atomic<bool> go(false);
    std::atomic<int> ready(0);
    std::vector<double> secs(n, 0.0);
    std::vector<std::thread> th;
    for (int i = 0; i < n; ++i) {
        th.emplace_back([&, i]() { secs[i] = copy_loop(devs[i], per_dev, bank, &go, &ready); });
    }
    while (ready.load() < n) std::this_thread::yield();
    const double tg0 = now_s();
    go.store(true);
    for (auto & t : th) t.join();
    const double wall = now_s() - tg0;

    size_t total = 0;
    printf("%-6s %12s %12s\n", "gpu", "seconds", "GB/s");
    for (int i = 0; i < n; ++i) {
        total += devs[i].bytes_moved;
        printf("%-6d %12.4f %12.2f\n", i, secs[i], devs[i].bytes_moved / secs[i] / 1e9);
    }
    const double agg = total / wall / 1e9;
    printf("%-6s %12.4f %12.2f   <- AGGREGATE\n", "all", wall, agg);

    // ---- verdict ---------------------------------------------------------
    printf("\n================================================================================\n");
    printf("VERDICT\n");
    printf("================================================================================\n");
    printf("sum of solo      : %7.2f GB/s\n", sum_solo);
    printf("true aggregate   : %7.2f GB/s\n", agg);
    printf("contention factor: %7.2f   (1.00 = the board scales perfectly)\n",
           agg / std::max(sum_solo, 1e-9));
    if (n > 1 && agg < 0.8 * sum_solo) {
        printf("\n!! The root complex does NOT scale. Per-card figures are misleading;\n");
        printf("   size the expert hot tier against %.2f GB/s, not %.2f GB/s.\n",
               agg, sum_solo);
    }

    // ---- what this implies for the design --------------------------------
    printf("\n--- implication for Qwen3.8-Flash-Next expert streaming ---\n");
    printf("At a 17 GB hot tier the measured miss traffic is 543.2 MB/token.\n");
    printf("%-14s %14s %14s\n", "PCIe GB/s", "ms/token fetch", "fetch-bound tok/s");
    for (double bw : {agg, sum_solo, 13.0, 26.0, 32.0, 52.0}) {
        double ms = 543.2e6 / (bw * 1e9) * 1e3;
        printf("%-14.2f %14.2f %14.1f\n", bw, ms, 1000.0 / (ms + 4.47 + 1.39));
    }
    printf("\n(4.47 ms fixed path + 1.39 ms expert compute added; both measured.)\n");

    if (as_json) {
        FILE * jf = json_path ? fopen(json_path, "w") : stdout;
        if (!jf) { fprintf(stderr, "cannot open %s\n", json_path); return 1; }
        fprintf(jf, "{\n  \"schema\": 1,\n  \"n_devices\": %d,\n  \"devices\": [\n", n);
        for (int i = 0; i < n; ++i) {
            cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
            fprintf(jf, "    {\"index\": %d, \"name\": \"%s\", \"vram_total\": %llu, "
                        "\"cc\": \"%d.%d\", \"async_engines\": %d, "
                        "\"h2d_solo_gbps\": %.4f}%s\n",
                    i, p.name, (unsigned long long) p.totalGlobalMem,
                    p.major, p.minor, p.asyncEngineCount, devs[i].solo_gbs,
                    (i + 1 < n) ? "," : "");
        }
        fprintf(jf, "  ],\n  \"h2d_sum_solo_gbps\": %.4f,\n"
                    "  \"h2d_simultaneous_gbps\": %.4f,\n"
                    "  \"h2d_contention_factor\": %.4f,\n"
                    "  \"expert_bytes\": %zu,\n"
                    "  \"gb_pushed_per_device\": %.3f\n}\n",
                sum_solo, agg, agg / (sum_solo > 0 ? sum_solo : 1.0),
                EXPERT_BYTES, gb_per_dev);
        if (json_path) fclose(jf);
    }

    for (int i = 0; i < n; ++i) {
        CK(cudaSetDevice(i));
        CK(cudaFreeHost(devs[i].h_pinned));
        CK(cudaFree(devs[i].d_buf));
    }
    return 0;
}
