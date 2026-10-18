// io_bench - what actually governs the scattered unbuffered read rate?
//
// `tools/disk_probe.py` answers "how fast can this drive read 676 KB blocks"
// with a thread per handle and pageable buffers. The runtime asks it a narrower
// question - reads into CUDA pinned staging, issued as one wave per expert
// tensor - and measures ~1.4 GB/s, at every IO depth from 1 to 32, with one
// file object or eight, merged waves or not.
//
// This benchmark varies one factor at a time to find out which of the
// differences matters:
//
//   * the offsets: fresh every run, or the same list re-read
//   * the buffer: pageable (VirtualAlloc) or cudaHostAlloc pinned
//   * the concurrency: N threads, one synchronous handle each
//   * the open flags: NO_BUFFERING, with or without RANDOM_ACCESS
//   * the locality: offsets over the whole file, or clustered inside a moving
//     window of one tensor's size (a layer's ten experts all live in one
//     ~600 MB region), optionally clustered per wave
//   * the copies: a cudaMemcpyAsync behind every read, two staging parities
//     deep, the way the runtime's staging works
//
// Measured on the reference rig (RTX 5060 Ti, MSI M371 NVMe, 676 KB reads, 800
// requests, 2026-10). Plain and pinned buffers behave identically per read - the
// allocation itself costs ~70 ms of process startup, which is the whole of the
// MB/s difference at one thread - so only the pinned column is shown:
//
//   readers                 1      2      4      8     16
//   fresh offsets         1047   1136   1127   1196   1220   MB/s   (pass A)
//   fresh offsets         1152   1293   1372   1457   1458   MB/s   (pass B)
//   the same list re-read 1148   1617   1814   1809   1797   MB/s
//
// Rows two and three are back-to-back runs, the only difference being whether
// the offsets repeat. **This drive serves a repeated read ~1.5x faster than a
// fresh one**, which is where the 2.4-2.5 GB/s that the read-path plan was
// built on came from: `disk_probe.py` used to draw one offset list and re-read
// it for every queue depth it reported.
//
// Nothing else moves the number: RANDOM_ACCESS or not; a 600 MB window,
// per-request or per-wave; and the host-to-device copies behind each read cost
// nothing measurable (1826 vs 1852 MB/s at eight readers).
//
// So the drive's fresh-offset curve is nearly flat: one reader already reaches
// ~1.05-1.29 GB/s and sixteen add ~15%. The runtime's demand path reads at
// 1.38-1.46 GB/s over a whole run and a decode layer asks for one to six reads
// before it must wait for routing, so the demand path sits at the top of this
// curve, not short of it. See docs/DECODE_BUDGET.md, "Lever 2, closed".
//
// Build (from a VS developer prompt, CUDA in PATH):
//   nvcc -O3 -o io_bench.exe tools/io_bench.cu
// Run:
//   io_bench.exe <file.gguf> 676 800 4 pinned ra 600 copy wave 1234
//                     size_kb  ^  ^      ^    ^   ^    ^    ^ seed
//                         count  |      |    flags: ra|seq|none
//                                threads     window_mb, 0 = whole file
//                                            copy = cudaMemcpyAsync per read
//                                                   wave = cluster per wave
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>

static std::vector<uint64_t> g_offs;
static DWORD                 g_flags = FILE_FLAG_NO_BUFFERING;
static bool                  g_copy  = false;   // run a device copy behind every read

static void run_case(const char * path, const char * region, size_t size, int threads,
                     std::vector<uint64_t> & offs, bool pinned) {
    const size_t n = offs.size();
    std::atomic<size_t>     next{0};
    std::atomic<size_t>     bytes{0};
    std::atomic<long long>  us{0};

    auto worker = [&]() {
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_EXISTING, g_flags, nullptr);
        if (h == INVALID_HANDLE_VALUE) { printf("  open failed %lu\n", GetLastError()); return; }
        // Two buffers, alternating, exactly like the runtime's staging parities,
        // so a read never writes the buffer a copy is still reading.
        void * buf[2] = { nullptr, nullptr };
        void * dev[2] = { nullptr, nullptr };
        cudaStream_t st = nullptr;
        cudaEvent_t  ev[2] = { nullptr, nullptr };
        for (int k = 0; k < 2; ++k) {
            if (pinned) {
                if (cudaHostAlloc(&buf[k], size + 4096, cudaHostAllocDefault) != cudaSuccess) {
                    printf("  cudaHostAlloc failed\n"); return;
                }
            } else {
                buf[k] = VirtualAlloc(nullptr, size + 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                if (buf[k] == nullptr) { printf("  VirtualAlloc failed\n"); return; }
            }
            if (g_copy) {
                if (cudaMalloc(&dev[k], size + 4096) != cudaSuccess) { printf("  cudaMalloc failed\n"); return; }
            }
        }
        if (g_copy) {
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming);
        }
        size_t    local    = 0;
        long long local_us = 0;
        int       slot     = 0;
        for (;;) {
            size_t i = next.fetch_add(1);
            if (i >= n) break;
            if (g_copy) {
                // The runtime lets a wave's host-to-device copies run while the
                // next wave reads, and only waits for the copies one parity back.
                cudaEventSynchronize(ev[slot]);
            }
            LARGE_INTEGER li; li.QuadPart = (LONGLONG) offs[i];
            SetFilePointerEx(h, li, nullptr, FILE_BEGIN);
            DWORD got = 0;
            auto t0 = std::chrono::steady_clock::now();
            ReadFile(h, buf[slot], (DWORD) size, &got, nullptr);
            local_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0).count();
            local += got;
            if (g_copy) {
                cudaMemcpyAsync(dev[slot], buf[slot], size, cudaMemcpyHostToDevice, st);
                cudaEventRecord(ev[slot], st);
                slot ^= 1;
            }
        }
        if (g_copy) { cudaStreamSynchronize(st); cudaStreamDestroy(st); }
        for (int k = 0; k < 2; ++k) {
            if (g_copy) cudaFree(dev[k]);
            if (pinned) cudaFreeHost(buf[k]); else VirtualFree(buf[k], 0, MEM_RELEASE);
        }
        CloseHandle(h);
        bytes += local;
        us += local_us;
    };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int i = 0; i < threads; ++i) ts.emplace_back(worker);
    for (auto & t : ts) t.join();
    double wall_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
    double mbs = (double) bytes.load() / 1e6 / (wall_ms / 1000.0);
    printf("  %-7s threads=%-2d  %6zu reqs  %8.1f MB/s  %7.2f ms/read  (sum of read calls %7.2f ms/read)\n",
           region, threads, n, mbs, wall_ms / (double) n,
           (double) us.load() / 1e3 / (double) n);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        printf("usage: io_bench <file> <size_kb> <count> <threads> <plain|pinned>"
               " [flags] [window_mb] [copy] [wave]\n");
        return 2;
    }
    const char * path    = argv[1];
    size_t       size    = (size_t) atoi(argv[2]) * 1024;
    size_t       count   = (size_t) atoi(argv[3]);
    int          threads = atoi(argv[4]);
    bool         pinned  = strcmp(argv[5], "pinned") == 0;
    g_flags = FILE_FLAG_NO_BUFFERING;
    if (argc >= 7) {
        if (strcmp(argv[6], "ra") == 0)   g_flags |= FILE_FLAG_RANDOM_ACCESS;
        if (strcmp(argv[6], "seq") == 0)  g_flags |= FILE_FLAG_SEQUENTIAL_SCAN;
    }

    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    LARGE_INTEGER li;
    GetFileSizeEx(h, &li);
    CloseHandle(h);
    uint64_t fsize = (uint64_t) li.QuadPart;

    // window_mb > 0 clusters every request inside a moving window of that size:
    // the runtime's reads are the ten selected experts of ONE layer, and they all
    // live inside a single ~600 MB tensor region.
    uint64_t window = 0;
    if (argc >= 8) {
        window = (uint64_t) atoll(argv[7]) * 1024ull * 1024ull;
        if (window > fsize / 2) window = 0;   // meaningless, use the whole file
    }
    const uint64_t span = window ? window : fsize;

    // `wave` mimics the runtime's *within-wave* locality: successive reads are
    // not independent draws, they are the experts of one tensor, so a group of
    // `threads` consecutive requests shares one window and the next group draws
    // a new one. Without it every request picks its own region, which is a much
    // friendlier pattern than the runtime ever sees.
    bool wave = (argc >= 10 && strcmp(argv[9], "wave") == 0);
    g_copy    = (argc >= 9  && strcmp(argv[8], "copy") == 0);
    // A fresh seed each run is worth up to 2x on this drive: see the header.
    unsigned seed = (argc >= 11) ? (unsigned) atoi(argv[10]) : 1234u;

    srand(seed);
    g_offs.clear();
    uint64_t base = 0;
    for (size_t i = 0; i < count; ++i) {
        if (window && (!wave || (i % (size_t) threads) == 0)) {
            base = ((uint64_t) rand() * 2654435761ull) % (fsize - window);
        }
        uint64_t o = window ? base + (((uint64_t) rand() * 40503ull) % (span - size))
                            : ((uint64_t) rand() * 2654435761ull) % (fsize - size);
        g_offs.push_back(o - (o % 4096));
    }
    printf("file %.2f GB  size %zu KB  count %zu  window %s  %s%s\n",
           (double) fsize / 1e9, size / 1024, count, window ? argv[7] : "whole-file",
           wave ? "per-wave " : "", g_copy ? "with-h2d-copies" : "");
    run_case(path, pinned ? "pinned" : "plain", size, threads, g_offs, pinned);
    return 0;
}
