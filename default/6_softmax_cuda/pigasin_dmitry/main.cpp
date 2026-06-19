#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "softmax_cuda.h"
#include <algorithm>


std::vector<float> SoftmaxRef(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    size_t col_count = input.size() / row_count;
    std::vector<float> output(input.size());

    for (int i = 0; i < row_count; ++i) {
        const float* row_start = input.data() + i * col_count;
        const float* row_end = row_start + col_count;
        float* out_start = output.data() + i * col_count;

        // 1. Ищем максимальный элемент в строке для численной стабильности
        float max_val = *std::max_element(row_start, row_end);

        // 2. Считаем сумму экспонент и сами экспоненты
        float sum_exp = 0.0f;
        for (size_t j = 0; j < col_count; ++j) {
            float exp_val = std::exp(row_start[j] - max_val);
            out_start[j] = exp_val;
            sum_exp += exp_val;
        }

        // 3. Нормализуем значения для получения распределения вероятностей
        float inv_sum_exp = 1.0f / sum_exp;
        for (size_t j = 0; j < col_count; ++j) {
            out_start[j] *= inv_sum_exp;
        }
    }

    return output;
}

// Функция для генерации случайной матрицы
std::vector<float> GenerateRandomMatrix(int rows, int cols) {
    std::vector<float> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

// Функция для сравнения точности двух матриц
float ComputeMaxAbsoluteError(const std::vector<float>& ref, const std::vector<float>& test) {
    float max_error = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float error = std::abs(ref[i] - test[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

int main() {
    const int m = 8192; 
    const int n = 16384; 
    std::cout << "Матрицы размером: " << m << "x" << n << "\n\n";

    auto a = GenerateRandomMatrix(m, n);
    auto b = GenerateRandomMatrix(m, n);

    // 1. Тестирование SoftmaxRef
    std::cout << "Запуск SoftmaxRef (CPU)..." << std::endl;
    auto start_ref = std::chrono::high_resolution_clock::now();
    
    auto res_ref = SoftmaxRef(a, m);
    
    auto end_ref = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ref = end_ref - start_ref;
    std::cout << "Время CPU: " << duration_ref.count() << " мс\n\n";

    // 2. Тестирование SoftmaxCUDA
    std::cout << "Запуск SoftmaxCUDA (GPU)..." << std::endl;
    SoftmaxCUDA(a, m);
    SoftmaxCUDA(a, m);
    SoftmaxCUDA(a, m);

    // Первая итерация часто включает инициализацию контекста CUDA, делаем "разогрев" при необходимости
    auto start_cuda = std::chrono::high_resolution_clock::now();
    
    auto res_cuda = SoftmaxCUDA(a, m);
    
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cuda = end_cuda - start_cuda;
    std::cout << "Время CUDA: " << duration_cuda.count() << " мс\n\n";

    // 3. Проверка точности
    float max_error = ComputeMaxAbsoluteError(res_ref, res_cuda);
    std::cout << "=== Результаты сравнения ===" << std::endl;
    std::cout << "Максимальная абсолютная ошибка: " << max_error << std::endl;
    
    // Допустимый порог ошибки из-за погрешности float (около 1e-4 или 1e-5 в зависимости от N)
    const float epsilon = 1e-4f;
    if (max_error < epsilon) {
        std::cout << "СТАТУС: УСПЕШНО (Результаты совпадают)" << std::endl;
    } else {
        std::cout << "СТАТУС: ОШИБКА (Слишком большая разница в вычислениях!)" << std::endl;
    }

    // Ускорение
    double speedup = duration_ref.count() / duration_cuda.count();
    std::cout << "Ускорение CUDA относительно CPU: " << speedup << " x" << std::endl;

    return 0;
}
