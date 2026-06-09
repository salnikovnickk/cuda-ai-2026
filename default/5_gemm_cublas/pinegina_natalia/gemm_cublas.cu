#include "gemm_cublas.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <cublas_v2.h>
#include <algorithm>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    const float alpha = 1.0f, beta = 0.0f;
    const size_t bytes = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    std::vector<float> output(n*n);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, output.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n, &alpha, d_B, n, d_A, n, &beta, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return output;
}

// int main(int argc, char *argv[])
// {
//     int size = 512;

//     std::vector<float> resMult;
//     std::vector<float> in_a(size*size, 0), in_b(size*size, 0);

//     srand(1234);
//     for (int i = 0; i < size * size; ++i) {
//         in_a[i] = static_cast<float>(rand()) / RAND_MAX;
//         in_b[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     resMult = GemmCUBLAS(in_a, in_b, size);

//     // Performance Measuring
//     std::vector<double> time_list;
//     for (int i = 0; i < 20; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         resMult = GemmCUBLAS(in_a, in_b, size);
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
