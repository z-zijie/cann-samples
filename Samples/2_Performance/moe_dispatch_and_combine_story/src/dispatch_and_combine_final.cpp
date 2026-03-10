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
 * \file quant_matmul_mxfp4.cpp
 * \brief
 */
#include <iostream>
#include <cstdlib>
#include <memory>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"

#include "shmem.h"

#include "utils.h"
#include "../include/moe_distribute_dispatch_v2_full_mesh.h"

static uint32_t gNpuNum = 2U;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024UL;

extern "C" __global__ __aicore__ void MoeDistributeDispatchKernel(
    GM_ADDR shmemSpace, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR workspaceGM, MoeDistributeDispatchV2TilingData tilingData)
{
    // 待补齐kernel实现
    #ifndef REDUCED
        TPipe pipe;
        MoeDistributeDispatchV2FullMesh<float16_t, float16_t, UNQUANT, false, false> op;
        op.Init(x, expertIds, nullptr, nullptr, nullptr, nullptr, 
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, nullptr,
            workspaceGM, nullptr, &tilingData, &pipe);
        op.Process();
        return;
    #else
        TPipe pipe;
        MoeDistributeDispatchV2FullMesh op;
        op.Init(x, expertIds,
            expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
            workspaceGM, &tilingData, &pipe);

        op.Process();
        return;
    #endif
}

void SetDispatchTilingData(MoeDistributeDispatchV2TilingData& dispatchTilingData, int bs, int epRankId)
{
    #ifndef REDUCED
        // 原始tiling
        dispatchTilingData.moeDistributeDispatchV2Info.bs = bs;
        dispatchTilingData.moeDistributeDispatchV2Info.h = 7168;
        dispatchTilingData.moeDistributeDispatchV2Info.epWorldSize = 2;
        dispatchTilingData.moeDistributeDispatchV2Info.epRankId = epRankId;
        dispatchTilingData.moeDistributeDispatchV2Info.hasElasticInfo = false;
        dispatchTilingData.moeDistributeDispatchV2Info.isPerformance = false;
        dispatchTilingData.moeDistributeDispatchV2Info.globalBs = 
            dispatchTilingData.moeDistributeDispatchV2Info.bs
            * dispatchTilingData.moeDistributeDispatchV2Info.epWorldSize;
        dispatchTilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = 0;
        dispatchTilingData.moeDistributeDispatchV2Info.moeExpertNum = 8;
        dispatchTilingData.moeDistributeDispatchV2Info.sharedExpertNum = 0;
        dispatchTilingData.moeDistributeDispatchV2Info.expertTokenNumsType = 1;
        dispatchTilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum = 0;
        dispatchTilingData.moeDistributeDispatchV2Info.isTokenMask = false;
        dispatchTilingData.moeDistributeDispatchV2Info.isExpertMask = false;
        dispatchTilingData.moeDistributeDispatchV2Info.k = 8;
        dispatchTilingData.moeDistributeDispatchV2Info.aivNum = 48;
        dispatchTilingData.moeDistributeDispatchV2Info.scalesCol = 0
        dispatchTilingData.moeDistributeDispatchV2Info.scalesTypeSize = 0;
        dispatchTilingData.moeDistributeDispatchV2Info.scalesCount =0;
        dispatchTilingData.moeDistributeDispatchV2Info.totalWinSizeEp = 0;
    #else
        // 简化tiling
        dispatchTilingData.moeDistributeDispatchV2Info.epWorldSize = 2U;                // epWorldSize
        dispatchTilingData.moeDistributeDispatchV2Info.epRankId = epRankId;                   // epRankId
        dispatchTilingData.moeDistributeDispatchV2Info.moeExpertNum = 8;               // moe expert number
        dispatchTilingData.moeDistributeDispatchV2Info.bs = bs;                         // bs
        dispatchTilingData.moeDistributeDispatchV2Info.k = 8;                          // k
        dispatchTilingData.moeDistributeDispatchV2Info.h = 7168;                          // h
        dispatchTilingData.moeDistributeDispatchV2Info.globalBs = 
            bs
            * dispatchTilingData.moeDistributeDispatchV2Info.epWorldSize;                   // globalBs = BS * worldSize
        dispatchTilingData.moeDistributeDispatchV2Info.aivNum = 48;                     // aivNum
        uint32_t expertTokenNumsType = 1;        // expert token nums type, support 0: cumsum mode, 1: count mode
    #endif
}

void InitData(uint8_t **hostPtr, uint8_t **devicePtr, size_t aSize, std::string path = "")
{
    std::cout << path << std::endl;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**> (devicePtr), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(hostPtr), aSize));
    if (path.length() == 0) {
        return;
    }
    ReadFile(path, *hostPtr, aSize);
    ACL_CHECK(aclrtMemcpy(*devicePtr, aSize, *hostPtr, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

void FinalizeData(uint8_t **hostPtr, uint8_t **devicePtr, size_t aSize = 0, std::string path = "")
{
    std::cout << path << std::endl;
    if (path.length() > 0 && aSize > 0) {
        ACL_CHECK(aclrtMemcpy(*hostPtr, aSize, *devicePtr, aSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(path, hostPtr, aSize);
    }
    ACL_CHECK(aclrtFreeHost(hostPtr));
    ACL_CHECK(aclrtFreeHost(devicePtr));
}

std::string GetInputFilePath(std::string tensorName, int rankId)
{
    std::string rankIdStr = std::to_string(rankId);
    return "./input/chip_" + rankIdStr + "/" + tensorName + "_" + rankIdStr + ".bin";
}

std::string GetOuputFilePath(std::string tensorName, int rankId)
{
    std::string rankIdStr = std::to_string(rankId);
    return "./output/chip_" + rankIdStr + "/" + tensorName + "_" + rankIdStr + ".bin";
}

aclshmemx_uniqueid_t flagUid;

int main(int argc, char* argv[])
{
    int status = ACLSHMEM_SUCCESS;
    int rankNum = atoi(argv[1]);
    int rankId = atoi(argv[2]);

    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s", n_ranks, rank_id, ipport);

    // Acl && Shmem init
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = atoi(argv[4]) + rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    uint64_t localMemSize = gNpu;
    aclshmemx_init_attr_t attributes;
    TestSetAttr(rankId, rankNum, localMemSize, ipport.c_str(), flagUid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // Acl init
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId % gNpuNum;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // shmem init
    uint64_t local_mem_size = gNpuMallocSpace;
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, ipport, default_flag_uid, &attributes);
    ACL_CHECK_WITH_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes),
        ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    int32_t aclshmemSize = gNpuMallocSpace;
    void *symmPtr = aclshmem_malloc(aclshmemSize);
    uint8_t *shmemSpace = (uint8_t *) symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    MoeDistributeDispatchV2TilingData dispatchTilingData;

    SetDispatchTilingData(dispatchTilingData, 8, 0);
    size_t localExpertNum = dispatchTilingData.moeDistributeDispatchV2Info.moeExpertNum
        / dispatchTilingData.moeDistributeDispatchV2Info.epWorldSize;
    size_t maxReceivedTokenNum = dispatchTilingData.moeDistributeDispatchV2Info.golbalBs
        * std::min(dispatchTilingData.moeDistributeDispatchV2Info.k, localExpertNum);

    uint8_t *xHost;
    uint8_t *xDevice;
    size_t xSize = dispatchTilingData.moeDistributeDispatchV2Info.bs * dispatchTilingData.moeDistributeDispatchV2Info.h * sizeof(float16_t);
    InitData(xHost, xDevice, xSize, GetInputFilePath("x", rankId));

    uint8_t *expertIdsHost;
    uint8_t *expertIdsDevice;
    size_t expertIdsSize = dispatchTilingData.moeDistributeDispatchV2Info.bs * dispatchTilingData.moeDistributeDispatchV2Info.k * sizeof(int32_t);
    InitData(expertIdsHost, expertIdsDevice, expertIdsSize, GetInputFilePath("expert_ids", rankId));

    uint8_t *expandXHost;
    uint8_t *expandXDevice;
    size_t expandXSize = maxReceivedTokenNum * dispatchTilingData.moeDistributeDispatchV2Info.h * sizeof(fp8_e5m2_t);
    InitData(expandXHost, expandXDevice, expandXSize);

    uint8_t *dynamicScalesHost;
    uint8_t *dynamicScalesDevice;
    size_t dynamicScalesSize = maxReceivedTokenNum * (dispatchTilingData.moeDistributeDispatchV2Info.h / 32) * sizeof(fp8_e8m0_t);
    InitData(dynamicScalesHost, dynamicScalesDevice, dynamicScalesSize);

    uint8_t *dynamicScalesHost;
    uint8_t *dynamicScalesDevice;
    size_t dynamicScalesSize = maxReceivedTokenNum * (dispatchTilingData.moeDistributeDispatchV2Info.h / 32) * sizeof(fp8_e8m0_t);
    InitData(dynamicScalesHost, dynamicScalesDevice, dynamicScalesSize);

    uint8_t *tokenSrcInfoHost;
    uint8_t *tokenSrcInfoDevice;
    size_t tokenSrcInfoSize = maxReceivedTokenNum * 128 * sizeof(int32_t);
    InitData(tokenSrcInfoHost, tokenSrcInfoDevice, tokenSrcInfoSize);

    uint8_t *expertTokenNumsHost;
    uint8_t *expertTokenNumsDevice;
    size_t expertTokenNumsSize = localExpertNum * sizeof(int64_t);
    InitData(expertTokenNumsHost, expertTokenNumsDevice, expertTokenNumsSize);

    uint8_t *sendCountsHost;
    uint8_t *sendCountsDevice;
    size_t sendCountSize = localExpertNum * dispatchTilingData.moeDistributeDispatchV2Info.epWorldSize * sizeof(int32_t);
    InitData(sendCountsHost, sendCountsDevice, sendCountSize);

    uint8_t *workspaceGM;
    size_t workspaceSize = 16 * 1024 * 1024;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&workspaceGM), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // 待补齐相应参数生成和传递
    for (int i = 0; i < 1; ++i) {
        MoeDistributeDispatchKernel<<<BLOCK_NUM, nullptr, stream>>>(
            shmemSpace, xDevice, expertIdsDevice,
            expandXDevice, dynamicScalesDevice, tokenSrcInfoDevice, expertTokenNumsDevice, sendCountsDevice,  
            workspaceGM, dispatchTilingData);
        MoeDistributeCombineKernel<<<BLOCK_NUM, nullptr, stream>>>(shmemSpace, combineTilingData);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    aclshmem_free(symmPtr);

    uint8_t *xHost;
    uint8_t *xDevice;
    FinalizeData(xHost, xDevice);

    uint8_t *expertIdsHost;
    uint8_t *expertIdsDevice;
    FinalizeData(expertIdsHost, expertIdsDevice);

    uint8_t *expandXHost;
    uint8_t *expandXDevice;
    FinalizeData(expandXHost, expandXDevice, expandXSize, GetOuputFilePath("expand_x", rankId));

    uint8_t *dynamicScalesHost;
    uint8_t *dynamicScalesDevice;
    FinalizeData(
        dynamicScalesHost, dynamicScalesDevice, dynamicScalesSize, GetOuputFilePath("dynamic_scales", rankId));

    uint8_t *tokenSrcInfoHost;
    uint8_t *tokenSrcInfoDevice;
    FinalizeData(
        tokenSrcInfoHost, tokenSrcInfoDevice, tokenSrcInfoSize, GetOuputFilePath("assist_info_for_combine", rankId));

    uint8_t *expertTokenNumsHost;
    uint8_t *expertTokenNumsDevice;
    InitData(expertTokenNumsHost, expertTokenNumsDevice, expertTokenNumsSize);
    FinalizeData(
        expertTokenNumsHost, expertTokenNumsDevice, expertTokenNumsSize, GetOuputFilePath("expert_token_nums", rankId));

    uint8_t *sendCountsHost;
    uint8_t *sendCountsDevice;
    FinalizeData(sendCountsHost, sendCountsDevice, sendCountSize, GetOuputFilePath("ep_recv_count", rankId));

    aclshmem_free(symmPtr);
    status = aclrtDestroyStream(stream);
    status = aclshmem_finalize();
    status = aclrtResetDevice(deviceId);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in relative_pe_id " << rankId << std::endl;
    return 0;
}
