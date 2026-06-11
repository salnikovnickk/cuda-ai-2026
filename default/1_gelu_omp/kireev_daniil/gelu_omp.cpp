#include "gelu_omp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>


#include <omp.h>
#include <immintrin.h>


inline float tanh(const float x) {
    if (abs(x) < 40.f) {
        const float exp2X = std::exp(2*x);
        return (exp2X - 1) / (exp2X + 1);
    }
    return std::tanh(x);
}

inline constexpr float SQRT_RESULT = 2 * sqrt(2.0f / M_PI);
inline constexpr size_t THREAD_NUM = 8;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t isize = input.size();
    std::vector<float> ret(isize);
    const size_t vec_size = isize - isize % THREAD_NUM;

    #pragma omp parallel for
    for (size_t t = 0; t < vec_size; t += THREAD_NUM) {
        ret[t] = input[t] * input[t] * input[t];
        ret[t + 1] = input[t + 1] * input[t + 1] * input[t + 1];
        ret[t + 2] = input[t + 2] * input[t + 2] * input[t + 2];
        ret[t + 3] = input[t + 3] * input[t + 3] * input[t + 3];
        ret[t + 4] = input[t + 4] * input[t + 4] * input[t + 4];
        ret[t + 5] = input[t + 5] * input[t + 5] * input[t + 5];
        ret[t + 6] = input[t + 6] * input[t + 6] * input[t + 6];
        ret[t + 7] = input[t + 7] * input[t + 7] * input[t + 7];
        ret[t] *= 0.044715f;
        ret[t + 1] *= 0.044715f;
        ret[t + 2] *= 0.044715f;
        ret[t + 3] *= 0.044715f;
        ret[t + 4] *= 0.044715f;
        ret[t + 5] *= 0.044715f;
        ret[t + 6] *= 0.044715f;
        ret[t + 7] *= 0.044715f;
        ret[t] += input[t];
        ret[t + 1] += input[t + 1];
        ret[t + 2] += input[t + 2];
        ret[t + 3] += input[t + 3];
        ret[t + 4] += input[t + 4];
        ret[t + 5] += input[t + 5];
        ret[t + 6] += input[t + 6];
        ret[t + 7] += input[t + 7];
        ret[t] *= SQRT_RESULT;
        ret[t + 1] *= SQRT_RESULT;
        ret[t + 2] *= SQRT_RESULT;
        ret[t + 3] *= SQRT_RESULT;
        ret[t + 4] *= SQRT_RESULT;
        ret[t + 5] *= SQRT_RESULT;
        ret[t + 6] *= SQRT_RESULT;
        ret[t + 7] *= SQRT_RESULT;
        ret[t] = tanh(ret[t]);
        ret[t + 1] = tanh(ret[t + 1]);
        ret[t + 2] = tanh(ret[t + 2]);
        ret[t + 3] = tanh(ret[t + 3]);
        ret[t + 4] = tanh(ret[t + 4]);
        ret[t + 5] = tanh(ret[t + 5]);
        ret[t + 6] = tanh(ret[t + 6]);
        ret[t + 7] = tanh(ret[t + 7]);
        ret[t] += 1.f;
        ret[t + 1] += 1.f;
        ret[t + 2] += 1.f;
        ret[t + 3] += 1.f;
        ret[t + 4] += 1.f;
        ret[t + 5] += 1.f;
        ret[t + 6] += 1.f;
        ret[t + 7] += 1.f;
        ret[t] *= 0.5f * input[t];
        ret[t + 1] *= 0.5f * input[t + 1];
        ret[t + 2] *= 0.5f * input[t + 2];
        ret[t + 3] *= 0.5f * input[t + 3];
        ret[t + 4] *= 0.5f * input[t + 4];
        ret[t + 5] *= 0.5f * input[t + 5];
        ret[t + 6] *= 0.5f * input[t + 6];
        ret[t + 7] *= 0.5f * input[t + 7];
    }

    #pragma omp parallel for
    for (size_t idx = vec_size; idx < isize; ++idx) {
        ret[idx] = input[idx] * input[idx] * input[idx];
        ret[idx] *= 0.044715f;
        ret[idx] += input[idx];
        ret[idx] *= SQRT_RESULT;
        ret[idx] = tanh(ret[idx]);
        ret[idx] += 1.f;
        ret[idx] *= 0.5f * input[idx];
    }

    return ret;
}
