// Bitwise verifier for the bit-exact tiling of k_mul_mat_id.
//
// Baseline reduction (sabah_rt.cu):
//   each of 128 threads t holds  partial_t = (t < nsub) ? p[t] : 0.0f   (s += NT=128, nsub<=80)
//   block_sum: warp shfl_down 16,8,4,2,1 ; sh[warp]=lane0 ; r = 0; r += sh[0..3]
//
// Proposed bit-exact tile: one physical warp per row, four rows per 128-thread
// block, four sequential virtual-warp passes with the SAME shfl tree and the
// SAME serial warp-sum order. Only lane 0's value is broadcast out.
//
// This test feeds identical partials to both and compares the raw 32-bit
// patterns, including edge values.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

#define NT 128
#define NWARP (NT / 32)

__device__ static inline float block_sum_ref(float v, float * sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
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

// Baseline: 1 row per block of 128 threads.
__global__ void ref_row(const float * p, float * out, int nsub) {
    __shared__ float sh[NWARP + 1];
    const float * prow = p + (size_t) blockIdx.x * 128;
    float v = 0.f;
    for (int s = threadIdx.x; s < nsub; s += NT) v += prow[s];
    const float r = block_sum_ref(v, sh);
    if (threadIdx.x == 0) out[blockIdx.x] = r;
}

// Bit-exact tile: 4 rows per block, 1 warp per row, 4 sequential virtual warps.
__global__ void exact_row(const float * p, float * out, int rows, int nsub) {
    const int lane = threadIdx.x & 31;
    const int r = threadIdx.x >> 5;            // row within the 4-row tile
    const int row = (int) blockIdx.x * 4 + r;
    if (row >= rows) return;
    const float * prow = p + (size_t) row * 128;

    float total = 0.f;
    for (int w = 0; w < 4; ++w) {
        if (w * 32 >= nsub) break;              // warp-uniform, same as the kernel
        const int v = w * 32 + lane;
        float part = (v < nsub) ? prow[v] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        // exact serial order: r = 0 + w0 + w1 + w2 + w3
        total += __shfl_sync(0xffffffffu, part, 0);
    }
    if (lane == 0) out[row] = total;
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

int main() {
    const int rows = 20000;
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::normal_distribution<float> nor(0.f, 1.f);

    std::vector<float> h(rows * 128);
    // edge values sprinkled in
    const float edges[] = {0.0f, -0.0f, 1e-40f, -1e-40f, 1e38f, -1e38f,
                           INFINITY, -INFINITY, NAN, -NAN, 3.4028235e38f};
    const int nedges = sizeof(edges) / sizeof(edges[0]);

    for (int r = 0; r < rows; ++r) {
        int mode = r % 5;
        for (int v = 0; v < 128; ++v) {
            float x;
            if (mode == 0) x = uni(rng);
            else if (mode == 1) x = nor(rng);
            else if (mode == 2) x = (v % 7 == 0) ? edges[(r + v) % nedges] : uni(rng);
            else if (mode == 3) x = (v < 20) ? nor(rng) : 0.f;   // down-like
            else x = (v < 80) ? nor(rng) : 0.f;                  // gate/up-like
            h[(size_t) r * 128 + v] = x;
        }
    }

    float *dp, *dref, *dex;
    cudaMalloc(&dp, h.size() * sizeof(float));
    cudaMalloc(&dref, rows * sizeof(float));
    cudaMalloc(&dex, rows * sizeof(float));

    int fails = 0, fails20 = 0, fails80 = 0;
    for (int nsub : {1, 2, 19, 20, 21, 31, 32, 33, 63, 64, 65, 79, 80}) {
        cudaMemcpy(dp, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
        ref_row<<<rows, NT, (NWARP + 1) * sizeof(float)>>>(dp, dref, nsub);
        exact_row<<<(rows + 3) / 4, NT>>>(dp, dex, rows, nsub);
        std::vector<float> hr(rows), hx(rows);
        cudaMemcpy(hr.data(), dref, rows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hx.data(), dex, rows * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0, first = -1;
        for (int r = 0; r < rows; ++r) {
            if (bits(hr[r]) != bits(hx[r])) {
                ++bad; if (first < 0) first = r;
            }
        }
        fails += bad;
        if (nsub == 20) fails20 = bad;
        if (nsub == 80) fails80 = bad;
        printf("nsub=%3d rows=%d  bitwise mismatches=%d", nsub, rows, bad);
        if (bad) printf("  first row=%d ref=%.9g(0x%08x) exact=%.9g(0x%08x)",
                        first, hr[first], bits(hr[first]), hx[first], bits(hx[first]));
        printf("\n");
    }
    printf("TOTAL mismatches=%d  (nsub20=%d nsub80=%d)\n", fails, fails20, fails80);
    return fails == 0 ? 0 : 1;
}
