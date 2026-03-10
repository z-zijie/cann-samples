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
#include "moe_distribute_dispatch.h"
#include "moe_distribute_combine.h"

inline int32_t TestSetAttr(int32_t myPe, int32_t nPes, uint64_t localMemSize, const char *ipPort, aclshmemx_uniqueid_t flagUid,
                       aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ipPort != nullptr) {
        ip_len = std::min(strlen(ipPort), static_cast<size_t>(64-1);
        std::copy_n(ipPort, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return 1;
        }
    }
    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = myPe;
    attributes->n_pes = nPes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = localMemSize;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&flagUid);
    return 0;
}

/**
    描述当前sample 实现与transformer仓下实现，参数做如下调整
    GM_ADDR shmemSpace,         shmem 申请的单卡内存空间，新增
    GM_ADDR x,                  token输入
    GM_ADDR expertIds,          专家ID
    GM_ADDR scales,             不需要
    GM_ADDR xActiveMask,        不需要
    GM_ADDR expertScales,       不需要
    GM_ADDR elasticInfo,        不需要
    GM_ADDR expandXOut,         输出
    GM_ADDR dynamicScalesOut,   不需要
    GM_ADDR assistInfoOut,      不需要
    GM_ADDR expertTokenNumsOut, 不需要
    GM_ADDR epSendCountsOut,    不需要
    GM_ADDR tpSendCountsOut,    不需要
    GM_ADDR expandScalesOut,    不需要
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM
*/

extern "C" __global__ __aicore__ void MoeDistributeDispatchKernel(
    GM_ADDR shmemSpace, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR workspaceGM, MoeDistributeDispatchV2TilingData tilingData)
{
    // 待补齐kernel实现
    #ifndef REDUCED
        TPipe pipe;
        MoeDistributeDispatchV2FullMesh<float16_t, fp8_e5m2_t, MX_QUANT, false, false> op;
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

/** 
    描述当前sample 实现与transformer仓下实现，参数做如下调整：
    GM_ADDR shmemSpace,             shmem 申请的单卡内存空间
    GM_ADDR expandX,                token输入
    GM_ADDR expertIds,              专家ID
    GM_ADDR assistInfoForCombine,   不需要
    GM_ADDR epSendCount,            不需要
    GM_ADDR scales,                 不需要
    GM_ADDR tpSendCount,            不需要
    GM_ADDR xActiveMask,            不需要
    GM_ADDR activationScale,        不需要
    GM_ADDR weightScale,            不需要
    GM_ADDR groupList,              不需要
    GM_ADDR expandScales,           不需要
    GM_ADDR sharedExpertX,          不需要
    GM_ADDR elasticInfo,            不需要
    GM_ADDR oriX,                   不需要
    GM_ADDR constExpertAlpha1,      不需要
    GM_ADDR constExpertAlpha2,      不需要
    GM_ADDR constExpertV,           不需要
    GM_ADDR XOut,                   输出
    GM_ADDR workspaceGM,            需要考虑下？
    GM_ADDR tilingGM                需要提前申请下
*/


__global__ __aicore__ void MoeDistributeCombineKernel(
    GM_ADDR shmemSpace, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 待补齐kernel实现

}

void SetDispatchTilingData(MoeDistributeDispatchTilingData& dispatchTilingData, int epRankId, int bs)
{
    #ifndef REDUCED
        // 原始tiling
        dispatchTilingData.bs = bs;
        dispatchTilingData.h = 7168;
        dispatchTilingData.epWorldSize = 2;
        dispatchTilingData.epRankId = epRankId;
        dispatchTilingData.hasElasticInfo = false;
        dispatchTilingData.isPerformance = false;
        dispatchTilingData.globalBs = dispatchTilingData.bs * dispatchTilingData.epWorldSize;
        dispatchTilingData.sharedExpertRankNum = 0;
        dispatchTilingData.moeExpertNum = 8;
        dispatchTilingData.sharedExpertNum = 0;
        dispatchTilingData.expertTokenNumsType = 1;
        dispatchTilingData.zeroComputeExpertNum = 0;
        dispatchTilingData.isTokenMask = false;
        dispatchTilingData.isExpertMask = false;
        dispatchTilingData.k = 8;
        dispatchTilingData.aivNum = 72;
        dispatchTilingData.scalesCol = 0
        dispatchTilingData.scalesTypeSize = 0;
        dispatchTilingData.scalesCount =0;
    #else
        // 简化tiling
        dispatchTilingData.epWorldSize = 2U;                // epWorldSize
        dispatchTilingData.epRankId = epRankId;                   // epRankId
        dispatchTilingData.moeExpertNum = 8;               // moe expert number
        dispatchTilingData.bs = bs;                         // bs
        dispatchTilingData.k = 8;                          // k
        dispatchTilingData.h = 7168;                          // h
        dispatchTilingData.globalBs = bs * dispatchTilingData.epWorldSize;                   // globalBs = BS * worldSize
        dispatchTilingData.aivNum = 72;                     // aivNum
        uint32_t expertTokenNumsType = 1;        // expert token nums type, support 0: cumsum mode, 1: count mode
    #endif
}

void SetCombineTilingData(MoeDistributeCombineTilingData& combineTilingData)
{
    // 待补齐tilingData数据填写
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

int main(int argc, char* argv[])
{
    int status = ACLSHMEM_SUCCESS;
    int rankNum = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    std::string ipport = argv[3];

    // Acl && Shmem init
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = atoi(argv[4]) + rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    aclshmemx_uniqueid_t flagUid;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    TestSetAttr(rankId, rankNum, localMemSize, ipport.c_str(), flagUid, &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    int32_t aclshmem_size = (504 * 1024 * 1024) * sizeof(__fp16);
    void *symmPtr = aclshmem_malloc(aclshmem_size);
    uint8_t *symmetricPtr = (uint8_t *) symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    MoeDistributeDispatchV2TilingData dispatchTilingData;
    MoeDistributeCombineTilingData combineTilingData;

    SetDispatchTilingData(dispatchTilingData, 8, 0);
    size_t localExpertNum = dispatchTilingData.moeExpertNum / dispatchTilingData.epWorldSize;
    size_t maxReceivedTokenNum = dispatchTilingData.golbalBs * std::min(dispatchTilingData.k, localExpertNum);

    uint8_t *xHost;
    uint8_t *xDevice;
    size_t xSize = dispatchTilingData.bs * dispatchTilingData.h * sizeof(float16_t);
    InitData(xHost, xDevice, xSize, "x_" + std::to_string(0) + ".bin");

    uint8_t *expertIdsHost;
    uint8_t *expertIdsDevice;
    size_t expertIdsSize = dispatchTilingData.bs * dispatchTilingData.k * sizeof(int32_t);
    InitData(expertIdsHost, expertIdsDevice, expertIdsSize, "expert_ids_" + std::to_string(0) + ".bin");

    uint8_t *expandXHost;
    uint8_t *expandXDevice;
    size_t expandXSize = maxReceivedTokenNum * dispatchTilingData.h * sizeof(fp8_e5m2_t);
    InitData(expandXHost, expandXDevice, expandXSize);

    uint8_t *dynamicScalesHost;
    uint8_t *dynamicScalesDevice;
    size_t dynamicScalesSize = maxReceivedTokenNum * (dispatchTilingData.h / 32) * sizeof(fp8_e8m0_t);
    InitData(dynamicScalesHost, dynamicScalesDevice, dynamicScalesSize);

    uint8_t *dynamicScalesHost;
    uint8_t *dynamicScalesDevice;
    size_t dynamicScalesSize = maxReceivedTokenNum * (dispatchTilingData.h / 32) * sizeof(fp8_e8m0_t);
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
    size_t sendCountSize = localExpertNum * dispatchTilingData.epWorldSize * sizeof(int32_t);
    InitData(sendCountsHost, sendCountsDevice, sendCountSize);

    uint8_t *workspaceGM;
    size_t workspaceSize = 16 * 1024 * 1024;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&workspaceGM), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // 待补齐相应参数生成和传递
    for (int i = 0; i < 1; ++i) {
        MoeDistributeDispatchKernel<<<BLOCK_NUM, nullptr, stream>>>(
            symmetricPtr, xDevice, expertIdsDevice,
            expandXDevice, dynamicScalesDevice, tokenSrcInfoDevice, expertTokenNumsDevice, sendCountsDevice,  
            workspaceGM, dispatchTilingData);
        MoeDistributeCombineKernel<<<BLOCK_NUM, nullptr, stream>>>(symmetricPtr, combineTilingData);
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
    FinalizeData(expandXHost, expandXDevice, expandXSize, "expand_x_" + std::to_string(0) + ".bin");

    uint8_t *dynamicScalesHost;
    uint8_t *dynamicScalesDevice;
    size_t dynamicScalesSize = maxReceivedTokenNum * (dispatchTilingData.h / 32) * sizeof(fp8_e8m0_t);
    FinalizeData(
        dynamicScalesHost, dynamicScalesDevice, dynamicScalesSize, "dynamic_scales_" + std::to_string(0) + ".bin");

    uint8_t *tokenSrcInfoHost;
    uint8_t *tokenSrcInfoDevice;
    size_t tokenSrcInfoSize = maxReceivedTokenNum * 128 * sizeof(int32_t);
    FinalizeData(
        tokenSrcInfoHost, tokenSrcInfoDevice, tokenSrcInfoSize, "token_src_info_" + std::to_string(0) + ".bin");

    uint8_t *expertTokenNumsHost;
    uint8_t *expertTokenNumsDevice;
    size_t expertTokenNumsSize = localExpertNum * sizeof(int64_t);
    InitData(expertTokenNumsHost, expertTokenNumsDevice, expertTokenNumsSize);
    FinalizeData(
        expertTokenNumsHost, expertTokenNumsDevice, expertTokenNumsSize, "expert_token_nums_" + std::to_string(0) + ".bin");

    uint8_t *sendCountsHost;
    uint8_t *sendCountsDevice;
    size_t sendCountSize = localExpertNum * dispatchTilingData.epWorldSize * sizeof(int32_t);
    FinalizeData(sendCountsHost, sendCountsDevice, sendCountSize, "recv_count_" + std::to_string(0) + ".bin");

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
