#include "softmax_cuda.h"

#include <thread>


__global__ void getRowMaxes(float* mat, size_t m, size_t n, float* rowMaxes) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m) {
        return;
    }

    float max = -INFINITY;
    float x = 0.f;
    for (size_t j = 0; j < n; ++j) {
        x = mat[i * n + j];
        if (x > max) {
            max = x;
        }
    }
    rowMaxes[i] = max;
}

__global__ void exp(float* mat, size_t m, size_t n, float* rowMaxes) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= n) {
        return;
    }

    mat[i * n + j] = std::exp(mat[i * n + j] - rowMaxes[i]);
}

__global__ void getRowSums(float* mat, size_t m, size_t n, float* rowSums) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m) {
        return;
    }

    float sum = 0.f;
    for (size_t j = 0; j < n; ++j) {
        sum += mat[i * n + j];
    }
    rowSums[i] = sum;
}

__global__ void inv(float* x, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    x[i] = 1.f / x[i];
}

__global__ void mul(float* mat, size_t m, size_t n, float* rowMuls) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= n) {
        return;
    }

    mat[i * n + j] *= rowMuls[i];
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& mat, int m) {
    std::vector<float> out;
    std::thread t([&](){ out.resize(mat.size()); });

    const size_t n = mat.size() / m;
    const size_t matNumBytes = mat.size() * sizeof(float);
    const size_t colNumBytes = m * sizeof(float);
    
    float *matGpu, *colGpu;
    cudaMalloc(&matGpu, matNumBytes);
    cudaMemcpy(matGpu, mat.data(), matNumBytes, cudaMemcpyHostToDevice);
    cudaMalloc(&colGpu, colNumBytes);

    const int blockSize1d = 256;
    const int numBlocks1d = (m + blockSize1d - 1) / blockSize1d;
    getRowMaxes<<<numBlocks1d, blockSize1d>>>(matGpu, m, n, colGpu);

    dim3 blockSize2d(16, 16);
    dim3 numBlocks2d((n + blockSize2d.x - 1) / blockSize2d.x, (m + blockSize2d.y - 1) / blockSize2d.y);
    exp<<<numBlocks2d, blockSize2d>>>(matGpu, m, n, colGpu);

    getRowSums<<<numBlocks1d, blockSize1d>>>(matGpu, m, n, colGpu);

    inv<<<numBlocks1d, blockSize1d>>>(colGpu, m);

    mul<<<numBlocks2d, blockSize2d>>>(matGpu, m, n, colGpu);

    t.join();
    cudaMemcpy(out.data(), matGpu, matNumBytes, cudaMemcpyDeviceToHost);

    cudaFree(matGpu);
    cudaFree(colGpu);

    return out;
}
