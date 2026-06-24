#include "naive_gemm_cuda.h"

__global__ void NaiveGemmKernel(float* a, float* b, float* res, int n)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n && j < n)
    {
        float sum = 0;
        for(int k = 0; k < n; k++)
            sum += a[i * n + k] * b[k * n + j];
        res[i * n + j] = sum;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const size_t size = n * n * sizeof(float);
    std::vector<float> resHost(n * n, 0);

    float* aDev = nullptr;
    cudaMalloc(&aDev, size);

    float* bDev = nullptr;
    cudaMalloc(&bDev, size);

    float* resDev = nullptr;
    cudaMalloc(&resDev, size);

    cudaMemcpy(aDev, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(bDev, b.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blockCount((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    NaiveGemmKernel<<<blockCount, threadsPerBlock>>>(aDev, bDev, resDev, n);

    cudaMemcpy(resHost.data(), resDev, size, cudaMemcpyDeviceToHost);

    cudaFree(aDev);
    cudaFree(bDev);
    cudaFree(resDev);
    
    return resHost;
}