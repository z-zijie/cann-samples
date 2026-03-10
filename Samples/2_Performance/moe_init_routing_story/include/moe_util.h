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
void getDataFromBin(const std::string &filename, std::vector<T> &data)
{
    // 以二进制模式打开文件
    std::ifstream file(filename, std::ios::binary);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 清空原有的数据
    data.clear();

    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件是否为空
    if (file_size == 0) {
        std::cerr << "Warning: File is empty" << std::endl;
        file.close();
        return;
    }

    // 计算元素数量
    size_t num_elements = file_size / sizeof(T);
    size_t remainder = file_size % sizeof(T);

    // 检查文件大小是否为元素大小的整数倍
    if (remainder != 0) {
        std::cerr << "Warning: File size (" << file_size << " bytes) is not a multiple of element size (" << sizeof(T)
                  << " bytes)" << std::endl;
        std::cerr << "Ignoring last " << remainder << " bytes of incomplete data" << std::endl;
    }

    if (num_elements > 0) {
        // 预先分配空间
        data.resize(num_elements);

        // 读取数据
        file.read(reinterpret_cast<char *>(data.data()), num_elements * sizeof(T));

        // 检查实际读取的字节数
        std::streamsize bytes_read = file.gcount();
        if (bytes_read != static_cast<std::streamsize>(num_elements * sizeof(T))) {
            std::cerr << "Warning: Actual bytes read (" << bytes_read << ") does not match expected ("
                      << num_elements * sizeof(T) << ")" << std::endl;

            // 调整vector大小以匹配实际读取的数据
            size_t actual_elements = bytes_read / sizeof(T);
            data.resize(actual_elements);
        }
    }

    file.close();
}

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