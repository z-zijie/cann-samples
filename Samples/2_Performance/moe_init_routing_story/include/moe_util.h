/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_util.h
 * \brief
 */

template <typename T>
void printData(std::vector<T>& res) {
    for (auto& elem : res) {
        std::cout << static_cast<float>(elem) << "\t";
    }
    std::cout << std::endl;
}

int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(4)));
}

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b = 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};

int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + 32 - 1) / 32 * 32 / bytes;
}