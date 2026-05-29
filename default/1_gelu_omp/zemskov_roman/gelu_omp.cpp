#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <immintrin.h>

#pragma GCC optimize("O3")
#pragma omp
#pragma simd
#define CASE 5

// #ifdef __x86_64__

#pragma GCC target("avx2,fma")

inline __m256 exp256_ps(__m256 x) {
    // From "sse_mathfun.h", by Julien Pommier   http://gruntthepeon.free.fr/ssemath/
    __m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
    __m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

    __m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
    __m256   cephes_exp_C1 = _mm256_set1_ps(0.693359375);
    __m256   cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

    __m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
    __m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
    __m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
    __m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
    __m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
    __m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
    __m256   tmp           = _mm256_setzero_ps(), fx;
    __m256i  imm0;
    __m256   one           = _mm256_set1_ps(1.0f);

    x     = _mm256_min_ps(x, exp_hi);
    x     = _mm256_max_ps(x, exp_lo);

            fx    = _mm256_mul_ps(x, cephes_LOG2EF);
            fx    = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
            tmp   = _mm256_floor_ps(fx);
    __m256  mask  = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
            mask  = _mm256_and_ps(mask, one);
            fx    = _mm256_sub_ps(tmp, mask);
            tmp   = _mm256_mul_ps(fx, cephes_exp_C1);
    __m256  z     = _mm256_mul_ps(fx, cephes_exp_C2);
            x     = _mm256_sub_ps(x, tmp);
            x     = _mm256_sub_ps(x, z);
            z     = _mm256_mul_ps(x,x);

    __m256  y     = cephes_exp_p0;
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p1);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p2);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p3);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p4);
            y     = _mm256_mul_ps(y, x);
            y     = _mm256_add_ps(y, cephes_exp_p5);
            y     = _mm256_mul_ps(y, z);
            y     = _mm256_add_ps(y, x);
            y     = _mm256_add_ps(y, one);

    /* build 2^n */
            imm0  = _mm256_cvttps_epi32(fx);
            imm0  = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
            imm0  = _mm256_slli_epi32(imm0, 23);
    __m256  pow2n = _mm256_castsi256_ps(imm0);
            y     = _mm256_mul_ps(y, pow2n);
            return y;
}


void GeluOMP(const std::vector<float>& input, std::vector<float>& output) {
    const size_t size = input.size();
    // output.resize(size);
#if CASE == 0
    // Reference implementation
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 0.5f * input[i] * (1.f + tanh(sqrt(2 / M_PI) * (input[i] + 0.044715f * pow(input[i], 3))));
    }
#elif CASE == 1
    // Custom tanh function
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 0.5f * input[i] * (1.f + my_tanh(sqrt(2.f / M_PI) * (input[i] + 0.044715f * pow(input[i], 3))));
    }
#elif CASE == 2
    // OpenMP loop vectorization
    const float coeff1 = 0.5f;
    const float coeff2 = std::sqrt(2.0f / M_PI);
    const float coeff3 = 0.044715f;

    #pragma omp simd
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = coeff2 * (x + coeff3 * x3);
        output[i] = coeff1 * x * (1.0f + my_tanh(inner));
    }
#elif CASE == 3
    // OpenMP loop unrolling
    const float coeff1 = 0.5f;
    const float coeff2 = std::sqrt(2.0f / M_PI);
    const float coeff3 = 0.044715f;

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = coeff2 * (x + coeff3 * x3);
        output[i] = coeff1 * x * (1.0f + my_tanh(inner));
    }
#elif CASE == 4
    // OpenMP loop unrolling - using multiple threads
    const float coeff1 = 0.5f;
    const float coeff2 = std::sqrt(2.0f / M_PI);
    const float coeff3 = 0.044715f;

    #pragma omp parallel num_threads(8)
    #pragma omp for
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = coeff2 * (x + coeff3 * x3);
        output[i] = coeff1 * x * (1.0f + my_tanh(inner));
    }

#elif CASE == 5
    // with AVX2 intrinsics - build with: g++ -O3 -mavx2 -mfma 
    constexpr size_t VecSize = 8u; // Avx2 vector size
    constexpr size_t Nodes = 8u; // Number of elements processed per iteration
    
    constexpr size_t NGroups = Nodes * VecSize; // Number of elements processed per iteration
    const size_t GroupsSize = input.size()/NGroups; // Number of elements processed per iteration
    
    size_t n;
    size_t i;
    size_t groupInx;

    const float* xp = input.data();
    float * yp = output.data();

    constexpr float c1 = 0.044715f;
    const float c2 = 2.0f * std::sqrt(2.0f / float(M_PI));
    
    const __m256 coeff1 = _mm256_set1_ps(c1);
    const __m256 coeff2 = _mm256_set1_ps(c2);

    const __m256 one = _mm256_set1_ps(1.0f);
    
    #pragma omp parallel for
    for (n = 0; n < NGroups; ++n) {
        groupInx = n*GroupsSize;
        size_t nextIdx = (n+1)*GroupsSize;
        // 
        for (i = groupInx; i + VecSize <= nextIdx; i+=VecSize){
            __m256 x = _mm256_loadu_ps(xp + i);
            __m256 k = _mm256_fmadd_ps(coeff1, _mm256_mul_ps(x, x), one);
            __m256 arg = _mm256_mul_ps(_mm256_mul_ps(coeff2, k), x);
            __m256 expRes = exp256_ps(arg);

            __m256 res = _mm256_mul_ps(x, _mm256_sub_ps(one, _mm256_div_ps(one, _mm256_add_ps(expRes, one))));
            _mm256_storeu_ps(yp + i, res);
        }
    }

    #pragma omp parallel for
    for (size_t j = groupInx; j < input.size(); j++) {
        float x = input[j];
        float x2 = x * x;
        float inner = c2 * x * (1.f + c1 * x2);
        output[j] =  x * (1.0f - 1.0f / (exp(inner) + 1.f));
    }

#endif
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    GeluOMP(input, output);
    return output;
}

std::vector<float> RefGelu(const std::vector<float>& x) {
    std::vector<float> output(x.size());
    // Reference implementation
    for (size_t i = 0; i < x.size(); ++i) {
        output[i] = 0.5f * x[i] * (1.f + tanh(sqrt(2.f / M_PI) * (x[i] + 0.044715f * pow(x[i], 3))));
    }
    return output;
}

// int main() {
//     // ...
//     const size_t size = 134217728;
//     std::vector<float> input(size);
//     float left = -20.0;
//     float right = -left;
//     float step = (right - left) / size;
//     for (int i = 0; i < size; ++i){
//         input[i] = left + step * static_cast<float>(i);
//     }

//     // Warming-up
//     auto ref = RefGelu(input);
//     auto target = GeluOMP(input);
//     std::vector<double> diffVec(size);
//     double diffMax = 0.0;
//     double sumDiff = 0.0;
//     double stdDiff = 0.0;
//     for (int i = 0; i < size; ++i){
//         diffVec[i] = abs(ref[i] - target[i]);
//         if (diffMax < diffVec[i]) {
//             diffMax = diffVec[i];
//         }
//         sumDiff += diffVec[i];
//         stdDiff += diffVec[i]*diffVec[i];
//     }

//     std::cout << "Max difference: " << diffMax << std::endl;
//     std::cout << "Sum difference: " << sumDiff/size << std::endl;
//     std::cout << "Std difference: " << sqrt(stdDiff/size) << std::endl;

//     // Performance Measuring
//     std::vector<double> time_list;
//     for (int i = 0; i < 10; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         // GeluOMP(input, target);
//         GeluOMP(input);
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> duration = end - start;
//         time_list.push_back(duration.count());
//     }
//     double time = *std::min_element(time_list.begin(), time_list.end());

//     std::cout << "Time: " << time << "s" << std::endl;
// }