#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input);

void GeluOMP(const std::vector<float>& input, std::vector<float>& output);

inline float my_tanh(float x) {
    float e_x2 = exp(2*x);
    return 1 - 2/(e_x2+1);
}

#endif // __GELU_OMP_H