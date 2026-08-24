#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // 1) Map this thread to a unique global index (one thread per element).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2) Guard: extra threads beyond N must not touch memory.
    if (idx >= N) return;

    // 3) Branch‑free ReLU: max(input, 0) via a select (no warp divergence).
    output[idx] = fmaxf(input[idx], 0.0f);
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
