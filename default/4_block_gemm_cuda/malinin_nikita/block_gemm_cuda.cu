#include "block_gemm_cuda.h"
#include <cuda/cmath>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_DOUBLE BLOCK_SIZE * BLOCK_SIZE

__global__ void BlockGemmCUDAImpl(float *a, float *b, float *c, int n)
{
    int t_x = threadIdx.y;
    int t_y = threadIdx.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float blockA[BLOCK_SIZE_DOUBLE];
    __shared__ float blockB[BLOCK_SIZE_DOUBLE];

    float sum = 0;

    for (int i = 0; i < gridDim.x; ++i)
    {
        blockA[t_x * BLOCK_SIZE + t_y] = a[row * n + i * BLOCK_SIZE + t_y];
        blockB[t_x * BLOCK_SIZE + t_y] = b[(i * BLOCK_SIZE + t_x) * n + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += blockA[t_x * BLOCK_SIZE + k] * blockB[k * BLOCK_SIZE + t_y];
        }
        __syncthreads();
    }
    c[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = a.size();
    int bytesSize = n * n * sizeof(float);

    float* gpuBufferA = nullptr;
    float* gpuBufferB = nullptr;
    float* gpuBufferC = nullptr;

    cudaMalloc(&gpuBufferA, bytesSize);
    cudaMalloc(&gpuBufferB, bytesSize);
    cudaMalloc(&gpuBufferC, bytesSize);

    cudaMemcpy(gpuBufferA, a.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuBufferB, b.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemset(gpuBufferC, 0, bytesSize);

    dim3 threadsDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksDim(n / BLOCK_SIZE, n / BLOCK_SIZE);

    BlockGemmCUDAImpl<<<blocksDim, threadsDim>>>(gpuBufferA, gpuBufferB, gpuBufferC, n);

    std::vector<float> c(size);
    float* cData = c.data();

    cudaDeviceSynchronize();
    cudaMemcpy(cData, gpuBufferC, bytesSize, cudaMemcpyDeviceToHost);
    cudaFree(gpuBufferA);
    cudaFree(gpuBufferB);
    cudaFree(gpuBufferC);

    return c;
}
