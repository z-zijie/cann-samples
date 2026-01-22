#include "gelu_cpu.h"
#include <cmath>
#include <algorithm>
#include <iostream>


void gelu_cpu(const std::vector<float>& input, std::vector<float>& output)
{
    const float TANH_APPROX_FACTOR = 1 / 0.044715;
    const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
    if (output.size() != input.size()) {
        output.resize(input.size());
    }

    for (std::size_t i = 0; i < input.size(); ++i)
    {
        float x = input[i];
        float x_cube = x * x * x;
        output[i] = x / (1.0f + std::exp((x * TANH_APPROX_FACTOR + x_cube) * NEG_SQRT_EIGHT_OVER_PI));
    }
}
