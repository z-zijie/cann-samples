/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <algorithm>

// misc
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

// aclshmem_host
#include "shmem.h"
// utils
#include "utils.h"
#include "select_helper.h"

static uint32_t gNpuNum = 16U;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024UL;

using namespace AscendC;

struct DispatchTilingData {
    uint32_t batchSize = 0U;
    uint32_t hiddenSize = 0U;
    uint32_t topK = 0U;
    uint32_t chipId = 0U;
    uint32_t worldSize = 0U;

    uint32_t quantMode = 0U;
};

constexpr uint32_t BLOCK_NUM = 8U;
constexpr int32_t BLOCK_SIZE_16 = 16U;

template<int I, bool B>
class DispatchGMMClass {
public:
    __aicore__ inline DispatchGMMClass(){}

    __aicore__ inline void Run(
        GM_ADDR x, GM_ADDR expertIds,
        GM_ADDR expandX,
        GM_ADDR workSpace, DispatchTilingData dispatchTilingData)
    {
        __gm__ uint32_t *xUint32 = (__gm__ uint32_t *)(x);
        xUint32[0] = dispatchTilingData.batchSize;
        xUint32[1] = dispatchTilingData.topK;
        xUint32[2] = dispatchTilingData.hiddenSize;
        xUint32[3] = dispatchTilingData.chipId;
        xUint32[4] = dispatchTilingData.worldSize;
        xUint32[5] = dispatchTilingData.quantMode;
        xUint32[6] = I;
        if constexpr (B) {
            xUint32[7] = 4396U;
        } else {
            xUint32[7] = 6324U;
        }
    }
};

extern "C" __global__ __aicore__ void DispatchGMM(
    int64_t tilingKey,
    GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR expandX,
    GM_ADDR workSpace, DispatchTilingData dispatchTilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (tilingKey == 9527) {
        DispatchGMMClass<3, true> op;
        op.Run(
            x, expertIds,
            expandX,
            workSpace, dispatchTilingData
        );
    } else {
        DispatchGMMClass<10, false> op;
        op.Run(
            x, expertIds,
            expandX,
            workSpace, dispatchTilingData
        );
    }
}

aclshmemx_uniqueid_t default_flag_uid;

int main(int argc, char **argv)
{
    auto InitData = [](uint8_t **devicePtr, uint8_t **hostPtr, size_t tensorBytes, std::string path = "") {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**> (devicePtr), tensorBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(hostPtr), tensorBytes));
        ACL_CHECK(aclrtMemcpy(*devicePtr, tensorBytes, *hostPtr, tensorBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    };

    int status = ACLSHMEM_SUCCESS;
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("n_rank=%d, rank_id=%d, ipport=%s", n_ranks, rank_id, ipport);

    // Acl init
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rank_id % gNpuNum;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // shmem init
    uint64_t local_mem_size = gNpuMallocSpace;
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, ipport, default_flag_uid, &attributes);
    ACL_CHECK_WITH_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes),
        ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    uint32_t bs = 32U;
    uint32_t h = 7168U;
    uint32_t k = 8U;
    uint32_t quantMode = 0U;

    uint32_t A = bs * n_ranks;

    // allocate memory in device for input, output, opretor execution
    size_t xSize = static_cast<size_t>(bs) * static_cast<size_t>(h) * sizeof(float16_t);
    uint8_t *xDevice;
    uint8_t *xHost;
    InitData(&xDevice, &xHost, xSize);
    
    size_t expertIdsSize = static_cast<size_t>(bs) * static_cast<size_t>(k) * sizeof(int32_t);
    uint8_t *expertIdsDevice;
    uint8_t *expertIdsHost;
    InitData(&expertIdsDevice, &expertIdsHost, expertIdsSize);

    size_t expandXSize = static_cast<size_t>(A) * static_cast<size_t>(h) * sizeof(float16_t);
    uint8_t *expandXDevice;
    uint8_t *expandXHost;
    InitData(&expandXDevice, &expandXHost, expandXSize);

    size_t workspaceSize = 16UL * 1024UL * 1024UL;
    uint8_t *workspace;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&workspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    DispatchTilingData dispatchTilingData;
    dispatchTilingData.batchSize = bs;
    dispatchTilingData.hiddenSize = h;
    dispatchTilingData.topK = k;
    dispatchTilingData.chipId = rank_id;
    dispatchTilingData.worldSize = n_ranks;
    dispatchTilingData.quantMode = quantMode;

    int64_t tilingKey = 9527;

    // execute the operator
    for (int i = 0; i < 1; ++i) {
        DispatchGMM<<<BLOCK_NUM, nullptr, stream>>>(
            tilingKey,
            xDevice, expertIdsDevice,
            expandXDevice,
            workspace, dispatchTilingData
        );
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(xHost, xSize, xDevice, xSize, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int i = 0; i < 8; i++) {
        std::cout << "dispatch tiling info: " << i << ", " << xHost[i] << std::endl;
    }
    INFO_LOG("test finished");

    // clean up
    ACL_CHECK(aclrtFreeHost(xHost));
    ACL_CHECK(aclrtFree(xDevice));

    ACL_CHECK(aclrtFreeHost(expertIdsHost));
    ACL_CHECK(aclrtFree(expertIdsDevice));

    ACL_CHECK(aclrtFreeHost(expandXHost));
    ACL_CHECK(aclrtFree(expandXDevice));

    ACL_CHECK(aclrtFree(workspace));

    ACL_CHECK_WITH_RET(aclshmem_finalize(), ERROR_LOG("aclshmem_finalize failed"), return -1);

    ACL_CHECK_WITH_RET(aclrtDestroyStream(stream), ERROR_LOG("aclrtDestroyStream failed"), return -1);
    ACL_CHECK_WITH_RET(aclrtResetDevice(deviceId), ERROR_LOG("aclrtResetDevice failed"), return -1);
    ACL_CHECK_WITH_RET(aclFinalize(), ERROR_LOG("aclFinalize failed"), return -1);

    INFO_LOG("[SUCCESS] demo run success in relative_pe_id %d", rank_id);
    return 0;
}
