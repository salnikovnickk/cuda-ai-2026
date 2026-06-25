#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H
