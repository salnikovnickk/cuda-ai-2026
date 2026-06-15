#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

#include "gelu_cuda.h"

__global__ void GeluKernel(float* input, float* output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float el = input[idx];
        output[idx] = 0.5f * el * (1 + tanhf(0.79788456f * el * (1.f + 0.044715f * el * el)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int count = input.size();
    std::vector<float> output(count);

    const float* hostInput = input.data();
    float* hostOutput = output.data();

    float* gpuInput = nullptr;
    float* gpuOutput = nullptr;
    cudaMalloc(&gpuInput, count * sizeof(float));
    cudaMalloc(&gpuOutput, count * sizeof(float));

    cudaMemcpy(gpuInput, hostInput, count * sizeof(float), cudaMemcpyHostToDevice);

    const int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;

    GeluKernel <<< num_blocks, block_size >>> (gpuInput, gpuOutput, count);

    cudaMemcpy(hostOutput, gpuOutput, count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    
    return output;
}
