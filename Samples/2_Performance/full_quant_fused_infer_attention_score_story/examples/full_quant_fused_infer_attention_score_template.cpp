/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include <iostream>
#include <cstdlib>
#include <memory>

#define FIA_ENABLE_MLA
#include "common_utils.h"
#include "io_utils.h"
#include "flash_attention_score_tiling_regbase.h"
#include "fia_entry.h"

template<uint8_t inOutLayoutType, bool hasAttenMask>
    __global__ __aicore__ void FiaKernelFullQuant(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attenMask, GM_ADDR keyAntiquantScale,
        GM_ADDR valueAntiquantScale, GM_ADDR dequantScaleQuery, GM_ADDR attentionOut,
        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FlashAttentionEntry<inOutLayoutType, hasAttenMask>(
        query, key, value,
        attenMask, keyAntiquantScale, valueAntiquantScale, dequantScaleQuery,  attentionOut,
        workspace, tiling);
    return;
}

void SetTilingData(optiling::FlashAttentionScoreSimplifiedTilingData& tilingData)
{
    tilingData.inputParamsRegbase.bSize = 1;
    tilingData.inputParamsRegbase.t1Size = 0;
    tilingData.inputParamsRegbase.t2Size = 0;
    tilingData.inputParamsRegbase.n2Size = 1;
    tilingData.inputParamsRegbase.gSize = 1;
    tilingData.inputParamsRegbase.s1Size = 8192;
    tilingData.inputParamsRegbase.s2Size = 8192;
    tilingData.inputParamsRegbase.alignedS2 = 0;
    tilingData.inputParamsRegbase.dSize = 128;
    tilingData.inputParamsRegbase.dSizeV = 128;
    tilingData.inputParamsRegbase.dSizeRope = 64;
    tilingData.inputParamsRegbase.keepProb = 0.000000;
    tilingData.inputParamsRegbase.scaleValue = 0.088388;
    tilingData.inputParamsRegbase.preTokens = 2147483647;
    tilingData.inputParamsRegbase.nextTokens = 2147483647;
    tilingData.inputParamsRegbase.pseS1Size = 0;
    tilingData.inputParamsRegbase.pseS2Size = 0;
    tilingData.inputParamsRegbase.pseBSize = 0;
    tilingData.inputParamsRegbase.bandIndex = 0;
    tilingData.inputParamsRegbase.layoutType = 3;
    tilingData.inputParamsRegbase.pseShapeType = 0;
    tilingData.inputParamsRegbase.attenMaskShapeType = 0;
    tilingData.inputParamsRegbase.attenMaskDataType = 1;
    tilingData.inputParamsRegbase.attenMaskCompressMode = 0;
    tilingData.inputParamsRegbase.implMode = 0;
    tilingData.inputParamsRegbase.sparseType = 0;
    tilingData.inputParamsRegbase.needDropMaskOp = 0;
    tilingData.inputParamsRegbase.dropMaskOuter = 0;
    tilingData.inputParamsRegbase.pseEncodeType = 0;
    tilingData.inputParamsRegbase.remain = 0;
    tilingData.inputParamsRegbase.attenMaskS2Size = 0;
    tilingData.inputParamsRegbase.pseType = 0;
    tilingData.inputParamsRegbase.rsv1 = 0;
    tilingData.inputParamsRegbase.qStartIdx = 0;
    tilingData.inputParamsRegbase.kvStartIdx = 0;
    tilingData.inputParamsRegbase.s1SparseValidSize = 0;
    tilingData.inputParamsRegbase.s2SparseValidSize = 0;
    tilingData.inputParamsRegbase.seed = 0;
    tilingData.inputParamsRegbase.offset = 0;
    tilingData.inputParamsRegbase.keepProbUint8 = 0;
    tilingData.inputParamsRegbase.pseAlibiBaseS1 = 0;
    tilingData.inputParamsRegbase.pseAlibiBaseS2 = 0;
    tilingData.inputParamsRegbase.deqScaleFlag = 1;
    tilingData.inputParamsRegbase.deqScale2Flag = 1;
    tilingData.inputParamsRegbase.isActualSeqLengthsNull = 1;
    tilingData.inputParamsRegbase.isActualSeqLengthsKVNull = 1;
    tilingData.inputParamsRegbase.actualSeqLengthsSize = 0;
    tilingData.inputParamsRegbase.actualSeqLengthsKVSize = 0;
    tilingData.inputParamsRegbase.isKvContinuous = 1;
    tilingData.inputParamsRegbase.fromFused = 1;
    tilingData.inputParamsRegbase.isBSNDOut = 0;
    tilingData.inputParamsRegbase.transposeLayout = 0;
    tilingData.inputParamsRegbase.isGqa = 0;
    tilingData.inputParamsRegbase.isSoftMaxLseEnable = 0;
    tilingData.inputParamsRegbase.isActualSharedPrefixLenNull = 1;
    tilingData.inputParamsRegbase.isQHasLeftPadding = 0;
    tilingData.inputParamsRegbase.isKVHasLeftPadding = 0;
    tilingData.inputParamsRegbase.ropeHeadSize = 0;
    tilingData.inputParamsRegbase.prefixSeqInnerSize = 0;
    tilingData.inputParamsRegbase.headNumRatio = 1;
    tilingData.inputParamsRegbase.blockSize = 128;
    tilingData.inputParamsRegbase.blockTableDim2 = 1;
    tilingData.inputParamsRegbase.paBlockNumSum = 1;
    tilingData.inputParamsRegbase.attenMaskS1Size = 0;
    tilingData.inputParamsRegbase.kvSplitPart = 32574;
    tilingData.inputParamsRegbase.accumOutSize = 2921063272;
    tilingData.inputParamsRegbase.logSumExpSize = 32574;
    tilingData.inputParamsRegbase.paLayoutType = 0;
    tilingData.inputParamsRegbase.isRowInvalid = 0;
    tilingData.inputParamsRegbase.isPostQuantPerChnl = 0;
    tilingData.inputParamsRegbase.isPostQuantBF16 = 0;
    tilingData.inputParamsRegbase.antiquantPerTensorFlag = 0;
    tilingData.inputParamsRegbase.antiquantPerHeadFlag = 0;
    tilingData.inputParamsRegbase.antiquantParaSeqSize = 1;
    tilingData.multiCoreParamsRegbase.coreNum = 32;
    tilingData.multiCoreParamsRegbase.totalSize = 64;
    tilingData.multiCoreParamsRegbase.s1OuterSize = 64;
    tilingData.multiCoreParamsRegbase.splitFactorSize = 2;
    tilingData.multiCoreParamsRegbase.splitFactorTailSize = 2;
    tilingData.multiCoreParamsRegbase.bnStartIdx[0] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[1] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[2] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[3] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[4] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[5] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[6] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[7] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[8] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[9] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[10] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[11] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[12] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[13] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[14] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[15] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[16] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[17] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[18] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[19] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[20] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[21] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[22] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[23] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[24] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[25] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[26] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[27] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[28] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[29] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[30] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[31] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[32] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[33] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[34] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[35] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[36] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[37] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[38] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[39] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[40] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[41] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[42] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[43] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[44] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[45] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[46] = 0;
    tilingData.multiCoreParamsRegbase.bnStartIdx[47] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[0] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[1] = 2;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[2] = 4;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[3] = 6;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[4] = 8;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[5] = 10;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[6] = 12;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[7] = 14;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[8] = 16;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[9] = 18;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[10] = 20;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[11] = 22;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[12] = 24;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[13] = 26;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[14] = 28;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[15] = 30;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[16] = 32;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[17] = 34;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[18] = 36;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[19] = 38;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[20] = 40;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[21] = 42;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[22] = 44;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[23] = 46;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[24] = 48;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[25] = 50;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[26] = 52;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[27] = 54;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[28] = 56;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[29] = 58;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[30] = 60;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[31] = 62;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[32] = 64;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[33] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[34] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[35] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[36] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[37] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[38] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[39] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[40] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[41] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[42] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[43] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[44] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[45] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[46] = 0;
    tilingData.multiCoreParamsRegbase.sparseStartIdx[47] = 0;
    tilingData.multiCoreParamsRegbase.firstFullLoadS1OuterIdx = 0;
    tilingData.multiCoreParamsRegbase.splitCoreMode = 0;
    tilingData.multiCoreParamsRegbase.reserve[0] = 0;
    tilingData.multiCoreParamsRegbase.reserve[1] = 0;
    tilingData.multiCoreParamsRegbase.reserve[2] = 0;
    tilingData.dropmaskParamsRegbase.multiCoreFactorSize = 0;
    tilingData.dropmaskParamsRegbase.baseUbCalSize = 0;
    tilingData.dropmaskParamsRegbase.multiCoreTotalSize = 139907189245643;
    tilingData.dropmaskParamsRegbase.shapeTotalSize = 0;
    tilingData.dropmaskParamsRegbase.dropMaskAddrOffset = -1;
    tilingData.initOutputParams.singleCoreSize = 0;
    tilingData.initOutputParams.rsvd[0] = 0;
    tilingData.initOutputParams.rsvd[1] = 0;
    tilingData.initOutputParams.totalOutputSize = 0;
    tilingData.initOutputParams.totalSoftMaxLseOutputSize = 0;
}

int main(int argc, char* argv[])
{
    std::cerr << "Start main" << std::endl;
    // -------------------------------------------------------------------------
    // 1. Parse the problem shape.
    // -------------------------------------------------------------------------
    uint32_t batchSize = 1;
    uint32_t numHeadsQ = 1;
    uint32_t numHEadsKV = 1;
    uint64_t seqLengthsQ = 8192;
    uint64_t seqLengthsKV = 8192;
    uint32_t headDim = 128;

    // -------------------------------------------------------------------------
    // 2. Initialize the ACL runtime and create a stream.
    //
    // The sample assumes one device and one stream for clarity. More advanced
    // applications may use multiple streams or pre-created runtime contexts.
    // -------------------------------------------------------------------------
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    uintew_t deviceCount;
    CHECK_COND(aclrtGetDeviceCount(&deviceCount) == ACL_SUCCESS, "Failed to get ACLRT devices.");
    CHECK_COND(deviceCount > 0U, "No ACLRT devices found.");
    CHECK_COND(aclInit(nullptr) == ACL_SUCCESS, "aclInit failed.");
    CHECK_COND(aclrtSetDevice(deviceId) == ACL_SUCCESS, "aclrtSetDevice failed.");
    CHECK_COND(aclrtCreateStream(&stream) == ACL_SUCCESS, "aclrtCreateStream failed.");

    // -------------------------------------------------------------------------
    // 3. Declare host-side buffers.
    //
    // Host buffers hold the packed input tensors loaded from disk and receive
    // the output tensor copied back from device memory.
    // -------------------------------------------------------------------------
    uint8_t* queryHost = nullptr;
    uint8_t* keyHost = nullptr;
    uint8_t* valueHost = nullptr;
    uint8_t* keyAntiquantScaleHost = nullptr;
    uint8_t* valueAntiquantScaleHost = nullptr;
    uint8_t* dequantScaleQueryHost = nullptr;
    uint8_t* outputHost = nullptr;

    // -------------------------------------------------------------------------
    // 4. Declare device-side buffers.
    //
    // Device buffers mirror the host buffers.
    // `GM_ADDR` is the generic "global memory address" type used by the kernel
    // launch interface in Ascend C samples.
    // -------------------------------------------------------------------------
    GM_ADDR queryDevice = nullptr;
    GM_ADDR keyDevice = nullptr;
    GM_ADDR valueDevice = nullptr;
    GM_ADDR keyAntiquantScaleDevice = nullptr;
    GM_ADDR valueAntiquantScaleDevice = nullptr;
    GM_ADDR dequantScaleQueryDevice = nullptr;
    GM_ADDR outputDevice = nullptr;
    GM_ADDR workspace = nullptr;
    GM_ADDR tilingDataDevice = nullptr;

    // -------------------------------------------------------------------------
    // 4. Compute tensor sizes in bytes.
    // -------------------------------------------------------------------------
    //
    // query:
    //   float8_e4m3fn input tensor of shape [batchSize, numHeadsQ, seqLengthsQ, headDim].
    //
    // key and value:
    //   float8_e4m3fn input tensor of shape [batchSize, numHeadsKV, seqLengthsKV, headDim].
    //
    // dequantScaleQuery:
    //   float32 input tensor shape [batchSize, numHeadsQ, seqLengthsQ / 128, 1].
    //
    // keyAntiquantScale and valueAntiquantScale:
    //   float32 input tensor shape [batchSize, numHeadsKV, seqLengthsKV / 256, 1].
    size_t querySize = (batchSize * numHeadsQ * seqLengthsQ * headDim) * sizeof(uint8_t);
    size_t keySize = (batchSize * numHeadsKV * seqLengthsKV * headDim) * sizeof(uint8_t);
    size_t valueSize = (batchSize * numHeadsKV * seqLengthsKV * headDim) * sizeof(uint8_t);
    size_t queryQuantScaleSize = (batchSize * numHeadsQ * (seqLengthsQ / 128) * 1) * sizeof(float);
    size_t keyAntiquantScaleSize = (batchSize * numHeadsKV * (seqLengthsKV / 128) * 1) * sizeof(float);
    size_t valueAntiquantScaleSize = (batchSize * numHeadsKV * (seqLengthsKV / 128) * 1) * sizeof(float);
    size_t outputSize = (batchSize * numHeadsQ * seqLengthsQ * headDim) * sizeof(uint16_t);
    size_t tilingDataSize = sizeof(optiling::FlashAttentionScoreSimplifiedTilingData);

    // -------------------------------------------------------------------------
    // 4. Materialize the default tiling configuration for this problem shape.
    // -------------------------------------------------------------------------
    //
    // tilingData的作用：
    optiling::FlashAttentionScoreSimplifiedTilingData tilingData;
    SetTilingData(tilingData);

    // -------------------------------------------------------------------------
    // 5. Allocate pinned host memory.
    //
    // Pinned buffers are used because they are the typical choice for explicit
    // async H2D / D2H copies in standalone performance samples.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtMallocHost((void**)&queryHost, querySize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostQ(queryHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&keyHost, keySize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostK(keyHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&valueHost, valueSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostV(valueHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&keyAntiquantScaleHost, keyAntiquantScaleSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostKScale(keyAntiquantScaleHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&valueAntiquantScaleHost, valueAntiquantScaleSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostVScale(valueAntiquantScaleHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&dequantScaleQueryHost, queryQuantScaleSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostQScale(dequantScaleQueryHost, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&outputHost, outputSize) == ACL_SUCCESS, "aclrtMallocHost failed.");
    std::unique_ptr<void, aclError (*)(void*)> HostO(outputHost, aclrtFreeHost);

    // Load pre-generated test tensors from disk.
    //
    // The sample keeps input generation out of the main executable so that
    // compute code stays easy to follow and data can be reproduced offline.
    ReadFile("./input/input_0.bin", querySize, queryHost, querySize);
    ReadFile("./input/input_1.bin", keySize, keyHost, keySize);
    ReadFile("./input/input_2.bin", valueSize, valueHost, valueSize);
    ReadFile("./input/input_15.bin", keyAntiquantScaleSize, keyAntiquantScaleHost, keyAntiquantScaleSize);
    ReadFile("./input/input_17.bin", valueAntiquantScaleSize, valueAntiquantScaleHost, valueAntiquantScaleSize);
    ReadFile("./input/input_27.bin", queryQuantScaleSize, dequantScaleQueryHost, queryQuantScaleSize);

    // -------------------------------------------------------------------------
    // 6. Allocate global memory on device.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtMalloc((void**)&queryDevice, querySize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceQ(queryDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&keyDevice, keySize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceK(keyDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&valueDevice, valueSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceV(valueDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&keyAntiquantScaleDevice, keyAntiquantScaleSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceKScale(keyAntiquantScaleDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&valueAntiquantScaleDevice, valueAntiquantScaleSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceVScale(valueAntiquantScaleDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&dequantScaleQueryDevice, queryQuantScaleSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceQScale(dequantScaleQueryDevice, aclrtFree);

    CHECK_COND(aclrtMalloc((void**)&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceO(outputDevice, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceWS(workspace, aclrtFree);
    CHECK_COND(aclrtMalloc((void**)&tilingDataDevice, tilingDataSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "aclrtMalloc failed.");
    std::unique_ptr<void, aclError (*)(void*)> DeviceTD(tilingDataDevice, aclrtFree);

    // -------------------------------------------------------------------------
    // 7. Copy host inputs to device memory.
    //
    // These copies are queued on the same stream that will later launch the
    // kernel, which preserves execution order without extra synchronization.
    // -------------------------------------------------------------------------
    CHECK_COND(
        aclrtMemcpyAsync(queryDevice, querySize, queryHost, querySize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(keyDevice, keySize, keyHost, keySize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(valueDevice, valueSize, valueHost, valueSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(keyAntiquantScaleDevice, keyAntiquantScaleSize, keyAntiquantScaleHost, keyAntiquantScaleSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(valueAntiquantScaleDevice, valueAntiquantScaleSize, valueAntiquantScaleHost, valueAntiquantScaleSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(dequantScaleQueryDevice, queryQuantScaleSize, dequantScaleQueryHost, queryQuantScaleSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");
    CHECK_COND(
        aclrtMemcpyAsync(tilingDataDevice, tilingDataSize, &tilingData, tilingDataSize, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // Query the platform object to learn how many AIC cores are available.
    //
    // This sample launches one block per AIC core and lets the scheduler assign
    // multiple tiles to each block when the problem is larger than the machine.
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    CHECK_COND(ascendcPlatform != nullptr, "Get ascendcPlatform failed.");
    uint32_t blockDimToBeSet = ascendcPlatform->CalcTschBlockDim(ascendcPlatform->GetCoreNumAiv(),
                    ascendcPlatform->GetCoreNumAic(), ascendcPlatform->GetCoreNumAiv());

    // -------------------------------------------------------------------------
    // 8. Launch the kernel.
    //
    // The kernel itself is small because most of the interesting logic is
    // encoded in the template stack and the runtime tiling parameters.
    // -------------------------------------------------------------------------
    FiaKernelFullQuant<<<blockDimToBeSet, nullptr, stream>>>(
        queryDevice, keyDevice, valueDevice,
        keyAntiquantScaleDevice,valueAntiquantScaleDevice, dequantScaleQueryDevice,
        outputDevice,
        workspace,
        tilingDataDevice
    )
    
    // Queue the output copy after the kernel launch on the same stream.
    CHECK_COND(
        aclrtMemcpyAsync(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
        "aclrtMemcpyAsync failed.");

    // -------------------------------------------------------------------------
    // 9. Synchronize, dump the output, and tear everything down.
    // -------------------------------------------------------------------------
    CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "aclrtSynchronizeStream failed.");
    WriteFile("./output/npu_out.bin", outputHost, outputSize);

    // `unique_ptr` takes care of freeing host/device buffers.
    // The runtime objects still need explicit destruction/finalization.
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

