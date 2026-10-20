#include <cuda_runtime.h>
#include <cstdio>
__global__ void busy(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float s = 0; for (int k = 0; k < 2000; ++k) s += a[i] * 1.0001f; a[i] = s; }
}
__global__ void scale(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= 1.5f;
}
int main() {
    const int n = 1 << 20;
    float *d; cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    for (int it = 0; it < 40; ++it) {
        busy<<<(n + 255) / 256, 256>>>(d, n);
        scale<<<(n + 255) / 256, 256>>>(d, n);
    }
    cudaDeviceSynchronize();
    printf("done %d\n", (int) cudaGetErrorString(cudaGetLastError()));
    return 0;
}
