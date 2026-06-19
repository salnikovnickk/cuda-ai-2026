#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

struct Context {

    float* vidA = nullptr;
    float* vidB = nullptr;
    float* vidC = nullptr;

    size_t allocated_bytes = 0;

    void EnsureCapacity(size_t bytes) {
        if (bytes <= allocated_bytes)
            return;

        if (vidA) cudaFree(vidA);
        if (vidB) cudaFree(vidB);
        if (vidC) cudaFree(vidC);

        cudaMalloc(&vidA, bytes);
        cudaMalloc(&vidB, bytes);
        cudaMalloc(&vidC, bytes);

        allocated_bytes = bytes;
    }

    ~Context() {
        if (vidA) cudaFree(vidA);
        if (vidB) cudaFree(vidB);
        if (vidC) cudaFree(vidC);
    }
};

Context ctx;

__global__ void NaiveGemmKernel(const float* A,
                                const float* B,
                                float* C,
                                int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;

#pragma unroll
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    ctx.EnsureCapacity(bytes);

    cudaMemcpy(ctx.vidA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.vidB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    NaiveGemmKernel<<<grid, block>>>(ctx.vidA, ctx.vidB, ctx.vidC, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), ctx.vidC, bytes, cudaMemcpyDeviceToHost);

    return c;
}