/* Minimal sustained-FMA burn, used to observe the card's P-state behaviour
 * without the 83 GB model in the loop.
 *
 *   gpu_burn.exe <seconds>
 *
 * Prints the number of kernel launches and the wall time. Clocks are sampled
 * externally (tools/clock_trace.py).
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void burn(float * out, int iters) {
    float a0 = threadIdx.x * 1e-6f + 1.0f;
    float a1 = a0 + 0.5f, a2 = a0 + 0.25f, a3 = a0 + 0.125f;
    const float m = 1.0000001f, c = 1e-7f;
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, m, c);
        a1 = fmaf(a1, m, c);
        a2 = fmaf(a2, m, c);
        a3 = fmaf(a3, m, c);
    }
    if (a0 == 12345.0f) {
        out[threadIdx.x] = a0 + a1 + a2 + a3;
    }
}

int main(int argc, char ** argv) {
    const int seconds = argc > 1 ? atoi(argv[1]) : 20;
    cudaSetDevice(0);
    float * out = nullptr;
    cudaMalloc(&out, 4096 * sizeof(float));
    const int iters = 200000;
    const dim3 grid(1024), block(256);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int launches = 0;
    cudaError_t err = cudaSuccess;
    for (double acc = 0.0; acc < seconds; ) {
        burn<<<grid, block>>>(out, iters);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { break; }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        acc = ms / 1000.0;
        ++launches;
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("BURN launches=%d wall_ms=%.1f iters_per_launch=%d err=%s\n",
           launches, ms, iters, cudaGetErrorString(err));
    cudaFree(out);
    return err == cudaSuccess ? 0 : 1;
}
