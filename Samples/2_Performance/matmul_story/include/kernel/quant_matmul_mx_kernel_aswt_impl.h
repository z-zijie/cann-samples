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
 * \file quant_matmul_mx_kernel_aswt_impl.h
 * \brief
 */

#ifndef QUANT_MATMUL_MX_KERNEL_ASWT_IMPL_H
#define QUANT_MATMUL_MX_KERNEL_ASWT_IMPL_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "kernel_utils/common_utils.h"
#include "kernel_utils/layout_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "../block/block_scheduler_mx.h"
#include "../block/block_mmad_mx.h"
#include "../utils/coord_utils.h"
#include "../utils/quant_matmul_constant.h"
#include "include/experimental/tensor_api/tensor.h"

namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;
using namespace AscendC::Te;

template<typename T>
struct MyMakeNDLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeNDLayout<T>(row, column);
}
};

template<typename T>
struct MyMakeDNLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeDNLayout<T>(row, column);
}
};

template<typename T>
struct MyMakeScaleANDLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeScaleANDLayout<T>(row, column);
}
};

template<typename T>
struct MyMakeScaleADNLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeScaleADNLayout<T>(row, column);
}
};

template<typename T>
struct MyMakeScaleBNDLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeScaleBNDLayout<T>(row, column);
}
};

template<typename T>
struct MyMakeScaleBDNLayout {
__aicore__ inline static decltype(auto) Execute(size_t row, size_t column)
{
    return AscendC::Te::MakeScaleBDNLayout<T>(row, column);
}
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantMatmulMxKernelAswtImpl {
public:
    __aicore__ inline QuantMatmulMxKernelAswtImpl()
    {}
    __aicore__ inline ~QuantMatmulMxKernelAswtImpl()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x1, x2, x1Scale, x2Scale, bias, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // using CoordClass = Coordinate<transA, transB, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA = AscendC::Std::conditional_t<transA, MyMakeDNLayout<AType>, MyMakeNDLayout<AType>>;
    using MakeLayoutB = AscendC::Std::conditional_t<transB, MyMakeDNLayout<BType>, MyMakeNDLayout<BType>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<transA, MyMakeScaleADNLayout<fp8_e8m0_t>, MyMakeScaleANDLayout<fp8_e8m0_t>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<transB, MyMakeScaleBDNLayout<fp8_e8m0_t>, MyMakeScaleBNDLayout<fp8_e8m0_t>>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void operator()(const Params& params);

private:
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    bool isBias_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelAswtImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::operator()(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }

    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);
    Process(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelAswtImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{

}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelAswtImpl<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    auto layoutA = MakeLayoutA::Execute(params.problemShape.m, params.problemShape.k);
    auto layoutScaleA =
        MakeLayoutScaleA::Execute(params.problemShape.m, CeilDiv(params.problemShape.k, 64) * 2);
    auto layoutB = MakeLayoutB::Execute(params.problemShape.k, params.problemShape.n);
    auto layoutScaleB =
        MakeLayoutScaleB::Execute(CeilDiv(params.problemShape.k, 64) * 2, params.problemShape.n);
    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr((__gm__ AType*)params.mmadParams.aGmAddr), layoutA);
    auto gmScaleA = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr)),
        layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr((__gm__ BType*)params.mmadParams.bGmAddr), layoutB);
    auto gmScaleB = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr)), layoutScaleB);
    auto layoutBias = AscendC::Te::MakeNDLayout<BiasType>(1L, params.problemShape.n);
    auto gmBias = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr)), layoutBias);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(params.problemShape.m, params.problemShape.n);
    auto gmC = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr)), layoutC);

    BlockCoord blockIdx;
    // // 每个核依次处理 block
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.GetBlockShape(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }

        auto gmBlockA =
            gmA(AscendC::Te::MakeCoord(Get<IDX_M_TILEIDX>(blockIdx) * params.qbmmParams.baseM, 0L),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockB =
            gmB(AscendC::Te::MakeCoord(0L, Get<IDX_N_TILEIDX>(blockIdx) * params.qbmmParams.baseN),
                AscendC::Te::MakeShape(params.problemShape.k, Get<MNK_N>(singleShape)));
        auto gmBlockScaleA = gmScaleA(
            AscendC::Te::MakeCoord(Get<IDX_M_TILEIDX>(blockIdx) * params.qbmmParams.baseM, 0L),
            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), CeilDiv(params.problemShape.k, 64) * 2));
        auto gmBlockScaleB = gmScaleB(
            AscendC::Te::MakeCoord(0, Get<IDX_N_TILEIDX>(blockIdx) * params.qbmmParams.baseN),
            AscendC::Te::MakeShape(CeilDiv(params.problemShape.k, 64) * 2, Get<MNK_N>(singleShape)));
        auto gmBlockBias = gmBias(
            AscendC::Te::MakeCoord(0L, Get<IDX_N_TILEIDX>(blockIdx) * params.qbmmParams.baseN),
            AscendC::Te::MakeShape(1L, Get<MNK_N>(singleShape)));
        auto gmBlockC =
            gmC(AscendC::Te::MakeCoord(
                    Get<IDX_M_TILEIDX>(blockIdx) * params.qbmmParams.baseM,
                    Get<IDX_N_TILEIDX>(blockIdx) * params.qbmmParams.baseN),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
    }
}

} // namespace Kernel

#endif