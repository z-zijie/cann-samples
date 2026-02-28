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
#include "op_host/fia_tiling.h"

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#define FIA_ENABLE_MLA
#include "op_kernel/fia_entry.h"
namespace ascend_ops {
namespace FIA {

template<uint8_t inOutLayoutType, bool hasAttenMask>
__global__ __aicore__ void FiaKernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attenMask,
    GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FlashAttentionEntry<inOutLayoutType, hasAttenMask>(
        query, key, value,
        attenMask,  attentionOut,
        workspace, tiling);
    return;
}

//对输入进行校验拦截，不支持场景一律拦截报错
bool CheckInput(ContextParamsForTiling& contextKeyParams, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const at::Tensor& mask, int64_t numHeads, int64_t numKeyValueHeads,
                                double scaleValue, int64_t preToken, int64_t nextToken, const std::string& inputLayout, int64_t sparseMode){
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

    if(!mask.defined()){
        printf("Error: mask can't be None.\n");
        return false; 
    }

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

    if(q.scalar_type() != at::kBFloat16){
        printf("Error: querry dtype only support bfloat16.\n");
        return false;
    }

    if(k.scalar_type() != at::kBFloat16){
        printf("Error: key dtype only support bfloat16.\n");
        return false;
    }

    if(v.scalar_type() != at::kBFloat16){
        printf("Error: value dtype only support bfloat16.\n");
        return false;
    }

    if(sparseMode != 1){
        printf("Error: sparseMode only support 1.\n");
        return false;
    }

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

bool ConvertContextToParams(ContextParamsForTiling& contextKeyParams, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const at::Tensor& mask, int64_t numHeads, int64_t numKeyValueHeads,
                                double scaleValue, int64_t preToken, int64_t nextToken, const std::string& inputLayout, int64_t sparseMode){
    if(!CheckInput(contextKeyParams, q, k, v, mask, numHeads, numKeyValueHeads,scaleValue,preToken,nextToken,inputLayout,sparseMode)){
        return false;
    }
    contextKeyParams.isKvContinuous = 1;
    contextKeyParams.maxKVs = 0;
    contextKeyParams.attentionMask = &mask;
    contextKeyParams.attentionMaskShape = mask.sizes();
    contextKeyParams.maskDataType = mask.scalar_type();
    contextKeyParams.inputDataType = q.scalar_type();
    contextKeyParams.kDataType = k.scalar_type();
    contextKeyParams.vDataType = v.scalar_type();
    contextKeyParams.outputDataType = contextKeyParams.inputDataType;
    contextKeyParams.queryInputShape = q.sizes();
    contextKeyParams.keyInputShape = k.sizes();
    contextKeyParams.valueInputShape = v.sizes();
    contextKeyParams.outputShape = contextKeyParams.queryInputShape;
    contextKeyParams.headsNumber = numHeads;
    contextKeyParams.sparseMode = sparseMode;
    contextKeyParams.preToken = preToken;
    contextKeyParams.nextToken = nextToken;
    contextKeyParams.scaleValue = scaleValue;
    contextKeyParams.layout = inputLayout;
    contextKeyParams.numKeyValueHeads = numKeyValueHeads;
    contextKeyParams.workspaceSize = 200 * 2048 * 1024;
    contextKeyParams.isBSNDOut = 0;
    contextKeyParams.maxKVs = k.size(3); 
 
    return true;
}

at::Tensor FiaNpu( const at::Tensor& q,                    // 查询张量 [B, Lq, H, D]
                    const at::Tensor& k,                    // 键张量 [B, Lk, H_kv, D]
                    const at::Tensor& v,                    // 值张量 [B, Lk, H_kv, D]
                    const at::Tensor& mask,                 // 注意力掩码 [B, Lq, Lk]
                    int64_t numHeads,                       // 总注意力头数
                    int64_t numKeyValueHeads,               // KV头数（GQA/MQA用）
                    double scaleValue,                      // 注意力缩放系数
                    const std::string& inputLayout,         // 输入布局
                    int64_t sparseMode                      // 稀疏模式
){
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
    ContextParamsForTiling contextParamsForTiling;
    FiaCompileInfo tempCompileInfoPtr;
    tempCompileInfoPtr.aivNum = ascendcPlatform->GetCoreNumAiv();
    tempCompileInfoPtr.aicNum = ascendcPlatform->GetCoreNumAic();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, tempCompileInfoPtr.ubSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, tempCompileInfoPtr.l1Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, tempCompileInfoPtr.l0CSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, tempCompileInfoPtr.l0ASize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
    tempCompileInfoPtr.socShortName = ascendcPlatform->GetSocVersion();
    tempCompileInfoPtr.defaultSysWorkspaceSize = 0;
    contextParamsForTiling.compileInfoPtr = &tempCompileInfoPtr;

    if(!ConvertContextToParams(contextParamsForTiling, q, k, v, mask, numHeads, numKeyValueHeads,
                            scaleValue, 65535, 65535, inputLayout, sparseMode)){
        return output;
    }
    FiaTiling tiling;
    auto ret = tiling.DoTiling(contextParamsForTiling);

    uint32_t blockDimToBeSet = ascendcPlatform->CalcTschBlockDim(ascendcPlatform->GetCoreNumAiv(),
                    ascendcPlatform->GetCoreNumAic(), ascendcPlatform->GetCoreNumAiv());

    optiling::FlashAttentionScoreSimplifiedTilingData tilingData = tiling.faTilingAdapter;
    
    uint16_t tilingFileSize = sizeof(optiling::FlashAttentionScoreSimplifiedTilingData);
    uint8_t *tilingDevice;
    aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingDevice, tilingFileSize, &tilingData, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    constexpr uint8_t inOutLayoutType = 0; // 输入为BSND
    constexpr bool hasAttenMask = true; // 使能mask
    
    auto aclCal = [=]() -> int {
        FiaKernel<inOutLayoutType, hasAttenMask><<<blockDimToBeSet, nullptr, aclstream>>>(
            (GM_ADDR)(q.storage().data()),
            (GM_ADDR)(k.storage().data()),
            (GM_ADDR)(v.storage().data()),
            (GM_ADDR)(mask.storage().data()),
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            (GM_ADDR)(tilingDevice)
            );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("FIA", aclCal);

    aclrtFree(tilingDevice);

    return output;
}

torch::Tensor FiaMeta( const torch::Tensor& q,                // 查询张量 [B, Lq, H, D]
                        const torch::Tensor& k,                // 键张量 [B, Lk, H_kv, D]
                        const torch::Tensor& v,                // 值张量 [B, Lk, H_kv, D]
                        const torch::Tensor& mask,             // 注意力掩码 [B, Lq, Lk] 
                        int64_t numHeads,                       // 总注意力头数
                        int64_t numKeyValueHeads,               // KV头数（GQA/MQA用）
                        double scaleValue,                      // 注意力缩放系数
                        const std::string& inputLayout,         // 输入布局
                        int64_t sparseMode                      // 稀疏模式
                    )
{
    TORCH_CHECK(q.defined(), "Input tensor must be defined");
    return q;
}

// Register Ascend implementations for RopeMatrix
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("fused_infer_attention_score", FiaNpu);
}

// Register Meta Function for RopeMatrix
TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("fused_infer_attention_score", TORCH_FN(FiaMeta));
}

} // namespace RopeMatrix
} // namespace ascend_ops
