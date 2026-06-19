#include "block_gemm_cuda.h"

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

static Context ctx;

constexpr int TILE = 16;

__global__ void BlockGemmKernel(const float* A,
                                const float* B,
                                float* C,
                                int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int m = 0; m < n; m += TILE) {

        As[ty][tx] = A[row * n + (m + tx)];
        Bs[ty][tx] = B[(m + ty) * n + col];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    ctx.EnsureCapacity(bytes);

    cudaMemcpy(ctx.vidA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.vidB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(n / TILE, n / TILE);

    BlockGemmKernel<<<grid, block>>>(ctx.vidA, ctx.vidB, ctx.vidC, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), ctx.vidC, bytes, cudaMemcpyDeviceToHost);

    return c;
}