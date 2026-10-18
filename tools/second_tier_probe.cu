// SABAH / lever 5: is a second GPU tier worth it?
//
// The idea: the GTX 1080's VRAM as an L2 in front of the disk. On a tier miss the
// runtime would look in the 1080 instead of reading 635 KB from the NVMe, and on
// an eviction it would park the victim there instead of dropping it.
//
// What decides that is the cost of the path itself, not the cache policy:
//
//   hit  = VRAM(1080) -> pinned host -> VRAM(5060)      vs  ~0.45 ms from disk
//   fill = VRAM(5060) -> pinned host -> VRAM(1080)      (paid on every eviction)
//
// This measures both, per expert-sized chunk, alone and with several in flight,
// plus whether peer access exists at all (consumer parts normally do not have
// it). Build:
//
//   nvcc -O3 -o second_tier_probe.exe second_tier_probe.cu
//
// and read the result next to docs/SECOND_TIER.md.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); \
    std::exit(1); } } while (0)

static const size_t EXPERT_BYTES = 635 * 1024;   // one expert tensor, this model
static const int    REPS = 200;

struct dev {
    int id = -1;
    std::string name;
    void * buf = nullptr;
    cudaStream_t stream;
};

// Times one direction of the path on that device's own stream (events must not be
// compared across devices). The caller adds the two halves.
static double bench_direction(dev & d, void * pinned, int chunks, bool to_host) {
    CK(cudaSetDevice(d.id));          // the event and stream below belong to d
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    auto issue = [&](int c) {
        if (to_host) {
            CK(cudaMemcpyAsync((char *) pinned + c * EXPERT_BYTES, d.buf, EXPERT_BYTES,
                               cudaMemcpyDeviceToHost, d.stream));
        } else {
            CK(cudaMemcpyAsync((char *) d.buf + c * EXPERT_BYTES, (char *) pinned + c * EXPERT_BYTES,
                               EXPERT_BYTES, cudaMemcpyHostToDevice, d.stream));
        }
    };
    for (int w = 0; w < 3; ++w) {
        for (int c = 0; c < chunks; ++c) issue(c);
    }
    CK(cudaStreamSynchronize(d.stream));
    CK(cudaEventRecord(a, d.stream));
    for (int r = 0; r < REPS; ++r) {
        for (int c = 0; c < chunks; ++c) issue(c);
    }
    CK(cudaEventRecord(b, d.stream));
    CK(cudaStreamSynchronize(d.stream));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return ms / (double) (REPS * chunks);         // ms per chunk, one hop
}

static double bench_peer(dev & from, dev & to, int chunks) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaSetDevice(to.id));
    for (int w = 0; w < 3; ++w) {
        for (int c = 0; c < chunks; ++c) {
            CK(cudaMemcpyPeerAsync((char *) to.buf + c * EXPERT_BYTES, to.id, from.buf, from.id,
                                   EXPERT_BYTES, to.stream));
        }
        CK(cudaStreamSynchronize(to.stream));
    }
    CK(cudaEventRecord(a, to.stream));
    for (int r = 0; r < REPS; ++r) {
        for (int c = 0; c < chunks; ++c) {
            CK(cudaMemcpyPeerAsync((char *) to.buf + c * EXPERT_BYTES, to.id, from.buf, from.id,
                                   EXPERT_BYTES, to.stream));
        }
    }
    CK(cudaEventRecord(b, to.stream));
    CK(cudaStreamSynchronize(to.stream));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return ms / (double) (REPS * chunks);
}

int main(int argc, char ** argv) {
    int chunks = argc > 1 ? std::atoi(argv[1]) : 8;
    if (chunks < 1) chunks = 1;

    int n = 0;
    CK(cudaGetDeviceCount(&n));
    std::vector<dev> d(n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, i));
        d[i].id = i;
        d[i].name = p.name;
        CK(cudaSetDevice(i));
        CK(cudaMalloc(&d[i].buf, EXPERT_BYTES * chunks));
        CK(cudaStreamCreate(&d[i].stream));
    }
    void * pinned = nullptr;
    CK(cudaSetDevice(d[0].id));
    CK(cudaHostAlloc(&pinned, EXPERT_BYTES * chunks, cudaHostAllocDefault));

    std::printf("devices: %d | chunk: %zu B | chunks in flight per wave: %d | waves: %d\n",
                n, EXPERT_BYTES, chunks, REPS);
    for (int i = 0; i < n; ++i) {
        std::printf("  %d: %s\n", i, d[i].name.c_str());
    }
    std::printf("\npeer access:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int can = 0;
            CK(cudaDeviceCanAccessPeer(&can, i, j));
            std::printf("  %d -> %d: %s\n", i, j, can ? "yes" : "no");
        }
    }

    // i == j is the host-RAM-tier path (evict out of the serving GPU, read back
    // into it), i != j is the other-GPU-as-L2 path.  Both are candidate second
    // tiers, so both are measured.
    std::printf("\n--- via pinned host, one chunk at a time ---\n");
    if (n > 1) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double out = bench_direction(d[j], pinned, 1, true);   // from -> host
                const double in  = bench_direction(d[i], pinned, 1, false);  // host -> to
                std::printf("  %d -> host -> %d: %.3f ms/chunk (%.3f out + %.3f in)  %.2f GB/s round trip%s\n",
                            j, i, out + in, out, in, EXPERT_BYTES / ((out + in) * 1e6),
                            i == j ? "  [host-RAM tier]" : "");
            }
        }
        std::printf("\n--- via pinned host, %d chunks in flight ---\n", chunks);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double out = bench_direction(d[j], pinned, chunks, true);
                const double in  = bench_direction(d[i], pinned, chunks, false);
                std::printf("  %d -> host -> %d: %.3f ms/chunk (%.3f out + %.3f in)  %.2f GB/s round trip%s\n",
                            j, i, out + in, out, in, EXPERT_BYTES / ((out + in) * 1e6),
                            i == j ? "  [host-RAM tier]" : "");
            }
        }
        std::printf("\n--- device to device ---\n");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                int can = 0;
                CK(cudaDeviceCanAccessPeer(&can, i, j));
                if (!can) {
                    std::printf("  %d -> %d: no peer access (the copy falls back to the host path)\n",
                                i, j);
                    continue;
                }
                const double ms = bench_peer(d[i], d[j], chunks);
                std::printf("  %d -> %d: %.3f ms/chunk (%.2f GB/s)\n",
                            i, j, ms, EXPERT_BYTES / (ms * 1e6));
            }
        }
    }

    std::printf("\nfor comparison, this rig's disk serves a fresh 635 KB read in ~0.45 ms (1.4 GB/s).\n");
    std::printf("A second tier only pays if (hit cost) + (park cost on eviction) < (disk cost).\n");
    std::printf("Per decode token the runtime misses 247 experts (157 MB, 121.8 ms) and evicts 239;\n");
    std::printf("its no-disk floor is 67.3 ms, so the most a better cache can win is bounded by that.\n");

    CK(cudaFreeHost(pinned));
    for (auto & dv : d) {
        CK(cudaSetDevice(dv.id));
        CK(cudaFree(dv.buf));
        CK(cudaStreamDestroy(dv.stream));
    }
    return 0;
}
