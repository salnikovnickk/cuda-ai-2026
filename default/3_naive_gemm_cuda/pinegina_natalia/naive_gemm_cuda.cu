#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include "naive_gemm_cuda.h"

__global__ void multKernel(const float * __restrict__ A,
                             const float * __restrict__ B,
                             float * __restrict__ C,
                             int N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        float res = 0.;
        for(int k = 0; k < N; k++)
        {
            res += A[row*N+k]*B[k*N+col];
        }
        C[row*N+col] = res;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const int tileSize = 16;
    const size_t bytes = n * n * sizeof(float);

    const float* in_a = a.data();
    const float* in_b = b.data();

    std::vector<float> output(bytes);
    float *out_c = output.data();

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, in_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, in_b, bytes, cudaMemcpyHostToDevice);

    int size = (n+tileSize-1)/tileSize;

    dim3 dimBlock(tileSize, tileSize);
    dim3 dimGrid(size, size);

    multKernel <<<dimGrid, dimBlock>>> (d_a, d_b, d_c, n);

    cudaMemcpy(out_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return output;
}

// int main(int argc, char *argv[])
// {
//     int size = 4096;

//     std::vector<float> resMult;
//     std::vector<float> in_a(size*size, 0), in_b(size*size, 0);

//     srand(1234);
//     for (int i = 0; i < size * size; ++i) {
//         in_a[i] = static_cast<float>(rand()) / RAND_MAX;
//         in_b[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     resMult = NaiveGemmCUDA(in_a, in_b, size);

//     // Performance Measuring
//     std::vector<double> time_list;
//     for (int i = 0; i < 20; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         resMult = NaiveGemmCUDA(in_a, in_b, size);
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> duration = end - start;
//         time_list.push_back(duration.count());
//     }
//     double time = *std::min_element(time_list.begin(), time_list.end());

//     std::cout << time << std::endl;

//     std::vector<float> resCPUMult(size*size, 0);
//     int N = size;
//     for (int i = 0; i < N; ++i)
//     {
//         for (int j = 0; j < N; ++j)
//         {
//             float res = 0.0f;
//             for (int k = 0; k < N; ++k)
//             {
//                 res += in_a[i * N + k] * in_b[k * N + j];
//             }
//             resCPUMult[i * N + j] = res;
//         }
//     }

//     double maxErr = 0.0;
//     for (int i = 0; i < N * N; ++i)
//     {
//         double err = fabs(resMult[i] - resCPUMult[i]);
//         if (err > maxErr) maxErr = err;
//     }
//     printf("Maximum absolute error: %e\n", maxErr);

//     return 0;
// }
