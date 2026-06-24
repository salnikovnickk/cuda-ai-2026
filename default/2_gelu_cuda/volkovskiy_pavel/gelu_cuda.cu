#include <cuda/cmath>
#include <cuda_runtime.h>
#include "gelu_cuda.h"

__global__ void gelu_kernel(const float* in, float* out, size_t size)
{
    constexpr float C1 = 1.59576912f;   // 2.0f * std::sqrt(2.0f / M_PI);
    constexpr float C2 = 0.0713548138f; // C1 * 0.044715f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = in[i];
        // float res = 0.5f * x * (1.0f + std::tanh(sqrt(2.0f/M_PI)*(x + 0.044715f * x * x* x)));
        const float exp2x = std::exp((C1 + C2 * x * x) * x);
        const float res = x * ((exp2x) / (exp2x + 1));
        out[i] = res;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    static int minGridSize = 0;
    static int maxBlockSize = 0;
    static bool blockSizeComputed = false;
    if (!blockSizeComputed) {
        blockSizeComputed = true;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, gelu_kernel);
    }

    std::vector<float> result(input.size());

    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    const size_t memsize = input.size() * sizeof(float);
    float *in = nullptr;
    float *out = nullptr;

    cudaMalloc((void**)&in, memsize);
    cudaMalloc((void**)&out, memsize);

    cudaMemcpy(in, input.data(), memsize, cudaMemcpyHostToDevice);

    // std::size_t blockSize = 256;
    std::size_t blockSize = maxBlockSize;
    std::size_t numBlocks = cuda::ceil_div(input.size(), blockSize); // (input.size()  + blockSize - 1) / blockSize;

    gelu_kernel<<<numBlocks, blockSize>>>(in, out, input.size());

    // cudaDeviceSynchronize();
    cudaMemcpy(result.data(), out, memsize, cudaMemcpyDeviceToHost); // waits gelu_kernel

    cudaFree(in);
    cudaFree(out);

    return result;
}
