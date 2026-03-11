/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"
//#include "op_host/fia_tiling.h"
#include "op_host/prompt_flash_attention_tiling_v2.h"
#include "op_host/prompt_flash_attention_tiling_compile_info.h"


#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#define FIA_ENABLE_MLA
#include "op_kernel/fia_entry.h"
namespace ascend_ops {
//namespace optiling {
int32_t g_headsNumber;
int32_t g_headsNumberKV;
int32_t g_sparseMode;
int64_t g_preToken;
int64_t g_nextToken;
size_t g_workspaceSize;
float g_scaleValue;
char g_layout[10];
int32_t g_numKeyValueHeads;
int64_t g_keyAntiquantMode;
int64_t g_valueAntiquantMode;
int64_t g_queryQuantMode;

template<uint8_t inOutLayoutType, bool hasAttenMask>
__global__ __aicore__ void FiaKernelFullQuant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attenMask, GM_ADDR keyAntiquantScale, GM_ADDR valueAntiquantScale, GM_ADDR dequantScaleQuery,
    GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FlashAttentionEntry<inOutLayoutType, hasAttenMask>(
        query, key, value,
        attenMask, keyAntiquantScale, valueAntiquantScale, dequantScaleQuery,  attentionOut,
        workspace, tiling);
    return;
}

//对输入进行校验拦截，不支持场景一律拦截报错
bool CheckInput(ContextParamsForPFATiling& contextKeyParams, const at::Tensor& q, const at::Tensor& k,
                                const at::Tensor& v, const at::Tensor& mask, const at::Tensor keyAntiquantScale,
                                const at::Tensor& valueAntiquantScale, const at::Tensor& dequantScaleQuery,
                                int64_t numHeads, int64_t numKeyValueHeads, double scaleValue, int64_t preToken,
                                int64_t nextToken, const std::string& inputLayout, int64_t sparseMode,
                                int64_t queryQuantMode, int64_t keyAntiquantMode, int64_t valueAntiquantMode)
{
    //q、k、v、mask不允许传None
    if(!q.defined()){
        printf("Error: q can't be None.\n");
        return false; 
    }

    if(!k.defined()){
        printf("Error: k can't be None.\n");
        return false; 
    }

    if(!v.defined()){
        printf("Error: v can't be None.\n");
        return false; 
    }

    // if(!mask.defined()){
    //     printf("Error: mask can't be None.\n");
    //     return false; 
    // }

    //空Tensor检查
    if(q.numel() == 0){
        printf("Error: querry empty tensor is not supported.\n");
        return false;
    }
    if(k.numel() == 0){
        printf("Error: key empty tensor is not supported.\n");
        return false;
    }
    if(v.numel() == 0){
        printf("Error: value empty tensor is not supported.\n");
        return false;
    }

    // if(q.scalar_type() != at::kBFloat16){
    //     printf("Error: querry dtype only support bfloat16.\n");
    //     return false;
    // }

    // if(k.scalar_type() != at::kBFloat16){
    //     printf("Error: key dtype only support bfloat16.\n");
    //     return false;
    // }

    // if(v.scalar_type() != at::kBFloat16){
    //     printf("Error: value dtype only support bfloat16.\n");
    //     return false;
    // }

    // if(sparseMode != 1){
    //     printf("Error: sparseMode only support 1.\n");
    //     return false;
    // }

    if(inputLayout != "BNSD"){
        printf("Error: inputLayout only support BNSD.\n");
        return false;
    }

    //GQA场景Qn与KVn检查
    if(numHeads % numKeyValueHeads != 0){
        printf("Error: querry N must be divided evenly by kv n.\n");
        return false;
    }

    //QKV 的S 128对齐检查
    if(q.size(2) % 128 != 0){
        printf("Error: the value of querry S only support 128-aligned.\n");
        return false;
    }

    if(k.size(2) % 128 != 0){
        printf("Error: the value of key S only support 128-aligned.\n");
        return false;
    }
    if(v.size(2) % 128 != 0){
        printf("Error: the value of value S only support 128-aligned.\n");
        return false;
    }
    //QKV 的D检查
    if(q.size(3) != 128){
        printf("Error: the value of querry D only support 128.\n");
        return false;
    }
    if(k.size(3) != 128){
        printf("Error: the value of key D only support 128.\n");
        return false;
    }
    if(v.size(3) != 128){
        printf("Error: the value of value D only support 128.\n");
        return false;
    }
    return true;

}

uint32_t FullquantGetTransposeLayout(const std::string &layout) 
{
    const std::map<std::string, uint32_t> transposeLayoutMp = {
        {"BNSD_BSND", 1},
        {"BSND_BNSD", 2},
        {"BSH_BNSD", 3},
        {"BNSD_NBSD", 4},
        {"BSND_NBSD", 5},
        {"BSH_NBSD", 6},
        {"NTD_TND", 7},
        {"TND_NTD", 8}
    };
    if (transposeLayoutMp.find(layout) != transposeLayoutMp.end()) {
        return transposeLayoutMp.at(layout);
    }
    return 0;
}

bool ConvertContextToParams(ContextParamsForPFATiling& contextKeyParams, const at::Tensor& q, const at::Tensor& k,
                                const at::Tensor& v, const at::Tensor& mask, const at::Tensor &keyAntiquantScale,
                                const at::Tensor &valueAntiquantScale, const at::Tensor &dequantScaleQuery, int64_t numHeads,
                                int64_t numKeyValueHeads, double scaleValue, int64_t preToken, int64_t nextToken,
                                const std::string& inputLayout, int64_t sparseMode, int64_t queryQuantMode,
                                int64_t keyAntiquantMode, int64_t valueAntiquantMode)
{
    if(!CheckInput(contextKeyParams, q, k, v, mask, keyAntiquantScale, valueAntiquantScale, dequantScaleQuery, 
                   numHeads, numKeyValueHeads,scaleValue,preToken,nextToken,inputLayout,sparseMode, queryQuantMode,
                   keyAntiquantMode, valueAntiquantMode
                   )){
        return false;
    }
    contextKeyParams.isKvContinuous = 1;
    contextKeyParams.emptyTensor = 0U;
    contextKeyParams.fromTilingSink = 0U;
    contextKeyParams.pseShift = nullptr;
    contextKeyParams.maxKVs = 0;
    contextKeyParams.attentionMask = nullptr;
    contextKeyParams.actualSequenceLengthQ = nullptr;
    contextKeyParams.actualSequenceLengthKV = nullptr;
    contextKeyParams.antiquantScale = nullptr;
    contextKeyParams.antiquantOffset = nullptr;
    contextKeyParams.inputDataType = at::ScalarType::Float8_e4m3fn;
    contextKeyParams.kDataType = at::ScalarType::Float8_e4m3fn;
    contextKeyParams.vDataType = at::ScalarType::Float8_e4m3fn;
    contextKeyParams.blockTable = nullptr;
    contextKeyParams.keySharedPrefix = nullptr;
    contextKeyParams.valueSharedPrefix = nullptr;
    contextKeyParams.actualSharedPrefixLen = nullptr;
    contextKeyParams.pseShiftDataType = at::ScalarType::Float;
    contextKeyParams.maskDataType = at::ScalarType::Bool;
    contextKeyParams.outputDataType = at::ScalarType::BFloat16;
    contextKeyParams.queryInputShape = q.sizes();
    contextKeyParams.keyInputShape = k.sizes();
    contextKeyParams.valueInputShape = v.sizes();
    contextKeyParams.pseShiftShape = {};
    contextKeyParams.attentionMaskShape = mask.sizes();
    contextKeyParams.deqScale1Shape = {};
    contextKeyParams.scale1Shape = {};
    contextKeyParams.deqScale2Shape = {};
    contextKeyParams.scale2Shape = {};
    contextKeyParams.offset2Shape = {};
    contextKeyParams.antiquantScaleShape = {};
    contextKeyParams.antiquantOffsetShape = {};
    contextKeyParams.outputShape = contextKeyParams.queryInputShape;
    contextKeyParams.innerPrecisePtr = nullptr;
    g_headsNumber = numHeads;
    contextKeyParams.headsNumber = &g_headsNumber;
    g_sparseMode = sparseMode;
    contextKeyParams.sparseMode = &g_sparseMode;
    g_preToken = preToken;
    contextKeyParams.preToken = &g_preToken;
    g_nextToken = nextToken;
    contextKeyParams.nextToken = &g_nextToken;
    g_scaleValue = scaleValue;
    contextKeyParams.scaleValue = &g_scaleValue;
    strcpy(g_layout, inputLayout.c_str());
    contextKeyParams.layout = g_layout;
    g_headsNumberKV = numKeyValueHeads;
    contextKeyParams.numKeyValueHeads = &g_headsNumberKV;
    g_workspaceSize = 200 * 2048 * 1024;
    contextKeyParams.workspaceSize = &g_workspaceSize;
    contextKeyParams.isBSNDOut = 0;
    contextKeyParams.transposeLayout = FullquantGetTransposeLayout(string(contextKeyParams.layout));
    contextKeyParams.fromFused = 71;
    contextKeyParams.deqScaleType = at::ScalarType::Float;
    contextKeyParams.deqScale2Type = at::ScalarType::Float;
    contextKeyParams.quantScale2Type = at::ScalarType::Float;
    contextKeyParams.quantOffset2Type = at::ScalarType::Float;
    contextKeyParams.maxKVs = k.size(3);
    g_queryQuantMode = queryQuantMode;
    contextKeyParams.queryQuantMode = &g_queryQuantMode;
    g_keyAntiquantMode = keyAntiquantMode;
    contextKeyParams.keyAntiquantMode = &g_keyAntiquantMode;
    g_valueAntiquantMode = valueAntiquantMode;
    contextKeyParams.valueAntiquantMode = &g_valueAntiquantMode;
    contextKeyParams.dequantScaleQueryShape = dequantScaleQuery.sizes();
    contextKeyParams.KeyAntiquantScaleShape = keyAntiquantScale.sizes();
    contextKeyParams.valueAntiquantScaleShape = valueAntiquantScale.sizes();
    contextKeyParams.dequantScaleQueryType = at::ScalarType::Float;
    contextKeyParams.KeyAntiquantScaleType = at::ScalarType::Float;
    contextKeyParams.valueAntiquantScaleType = at::ScalarType::Float;
    contextKeyParams.keyAntiquantScale = &keyAntiquantScale;
    contextKeyParams.valueAntiquantScale = &valueAntiquantScale;
    contextKeyParams.dequantScaleQuery = &dequantScaleQuery;
    return true;
}

at::Tensor FiaNpuFullQuant( const torch::Tensor& q,                    // 查询张量 [B, Lq, H, D]
                    const torch::Tensor& k,                    // 键张量 [B, Lk, H_kv, D]
                    const torch::Tensor& v,                    // 值张量 [B, Lk, H_kv, D]
                    const torch::Tensor& mask,
                    const torch::Tensor&keyAntiquantScale,
                    const torch::Tensor&valueAntiquantScale,
                    const torch::Tensor&dequantScaleQuery,                 // 注意力掩码 [B, Lq, Lk]
                    int64_t numHeads,                       // 总注意力头数
                    int64_t numKeyValueHeads,               // KV头数（GQA/MQA用）
                    double scaleValue,                      // 注意力缩放系数
                    const std::string& inputLayout,         // 输入布局
                    int64_t sparseMode,                      // 稀疏模式
                    int64_t queryQuantMode,
                    int64_t keyAntiquantMode,
                    int64_t valueAntiquantMode)
{
    // 获取npu信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    // stream
    int devidx = q.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream(false);
    // 计算输出shape
    at::Tensor output = torch::zeros(q.sizes(), q.options());
    // workspace
    int64_t workspaceSize = 200 * 2048 * 1024; // default value
    auto workspaceTensor = 
        at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(q.options().device()));
    // tilingData
    ContextParamsForPFATiling contextFullQuantParamsForTiling;
    PromptFlashAttentionCompileInfo tempCompileInfoPtr;
    tempCompileInfoPtr.aivNum = ascendcPlatform->GetCoreNumAiv();
    tempCompileInfoPtr.aicNum = ascendcPlatform->GetCoreNumAic();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, tempCompileInfoPtr.ubSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, tempCompileInfoPtr.l1Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, tempCompileInfoPtr.l0CSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, tempCompileInfoPtr.l0ASize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
    tempCompileInfoPtr.socShortName = ascendcPlatform->GetSocVersion();
    tempCompileInfoPtr.defaultSysWorkspaceSize = 0;
    contextFullQuantParamsForTiling.compileInfoPtr = &tempCompileInfoPtr;

    if(!ConvertContextToParams(contextFullQuantParamsForTiling, q, k, v, mask, keyAntiquantScale, valueAntiquantScale,
                            dequantScaleQuery, numHeads, numKeyValueHeads,
                            scaleValue, 65535, 65535, inputLayout, sparseMode, queryQuantMode, keyAntiquantMode,
                            valueAntiquantMode)){
        return output;
    }
    optiling::v2::PromptFlashAttentionTilingV2 tiling;
    auto ret = tiling.DoOpTiling(contextFullQuantParamsForTiling);

    uint32_t blockDimToBeSet = ascendcPlatform->CalcTschBlockDim(ascendcPlatform->GetCoreNumAiv(),
                    ascendcPlatform->GetCoreNumAic(), ascendcPlatform->GetCoreNumAiv());

    optiling::FlashAttentionScoreSimplifiedTilingData tilingData = tiling.faTilingAdapter;
    
    uint16_t tilingFileSize = sizeof(optiling::FlashAttentionScoreSimplifiedTilingData);
    uint8_t *tilingDevice;
    aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingDevice, tilingFileSize, &tilingData, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    constexpr uint8_t inOutLayoutType = 0; // 输入为BSND
    constexpr bool hasAttenMask = false; // 使能mask

    __gm__ uint8_t* q_inputPtr = (__gm__ uint8_t*)q.data_ptr();
    __gm__ uint8_t* k_inputPtr = (__gm__ uint8_t*)k.data_ptr();
    __gm__ uint8_t* v_inputPtr = (__gm__ uint8_t*)v.data_ptr();
    __gm__ uint8_t* mask_inputPtr = nullptr;
    __gm__ uint8_t* keyAntiquantScale_inputPtr = (__gm__ uint8_t*)keyAntiquantScale.data_ptr();
    __gm__ uint8_t* valueAntiquantScale_inputPtr = (__gm__ uint8_t*)valueAntiquantScale.data_ptr();
    __gm__ uint8_t* dequantScaleQuery_inputPtr = (__gm__ uint8_t*)dequantScaleQuery.data_ptr();
    __gm__ uint8_t* outputPtr = (__gm__ uint8_t*)output.data_ptr();
    __gm__ uint8_t* workspacePtr = (__gm__ uint8_t*)workspaceTensor.data_ptr();

    auto aclCal = [=]() -> int {
        FiaKernelFullQuant<inOutLayoutType, hasAttenMask><<<blockDimToBeSet, nullptr, aclstream>>>(
            q_inputPtr,
            k_inputPtr,
            v_inputPtr,
            mask_inputPtr,
            keyAntiquantScale_inputPtr,
            valueAntiquantScale_inputPtr,
            dequantScaleQuery_inputPtr,
            outputPtr,
            workspacePtr,
            (GM_ADDR)(tilingDevice)
            );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("FIA", aclCal);
    
    // auto aclCal = [=]() -> int {
    //     FiaKernel<inOutLayoutType, hasAttenMask><<<blockDimToBeSet, nullptr, aclstream>>>(
    //         (GM_ADDR)(q.storage().data()),
    //         (GM_ADDR)(k.storage().data()),
    //         (GM_ADDR)(v.storage().data()),
    //         (GM_ADDR)(mask.storage().data()),
    //         (GM_ADDR)(output.storage().data()),
    //         (GM_ADDR)(workspaceTensor.storage().data()),
    //         (GM_ADDR)(tilingDevice)
    //         );
    //     return 0;
    // };
    // at_npu::native::OpCommand::RunOpApiV2("FIA", aclCal);

    aclrtFree(tilingDevice);

    return output;
}

torch::Tensor FiaMetaFullQuant( const torch::Tensor& q,                // 查询张量 [B, Lq, H, D]
                        const torch::Tensor& k,                // 键张量 [B, Lk, H_kv, D]
                        const torch::Tensor& v,                // 值张量 [B, Lk, H_kv, D]
                        const torch::Tensor& mask,             // 注意力掩码 [B, Lq, Lk] 
                        const torch::Tensor& keyAntiquantScale,
                        const torch::Tensor& valueAntiquantScale,
                        const torch::Tensor& dequantScaleQuery,
                        int64_t numHeads,                       // 总注意力头数
                        int64_t numKeyValueHeads,               // KV头数（GQA/MQA用）
                        double scaleValue,                      // 注意力缩放系数
                        const std::string& inputLayout,         // 输入布局
                        int64_t sparseMode,                      // 稀疏模式
                        int64_t queryQuantMode,
                        int64_t keyAntiquantMode,
                        int64_t valueAntiquantMode)
{
    TORCH_CHECK(q.defined(), "Input tensor must be defined");
    return q;
}

// Register Ascend implementations for RopeMatrix
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("full_quant_fused_infer_attention_score", FiaNpuFullQuant);
}

// Register Meta Function for RopeMatrix
TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("full_quant_fused_infer_attention_score", TORCH_FN(FiaMetaFullQuant));
}

//} // namespace RopeMatrix
} // namespace ascend_ops
