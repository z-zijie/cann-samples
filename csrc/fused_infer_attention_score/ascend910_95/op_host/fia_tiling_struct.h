/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef FIA_TILING_STRUCT_H_
#define FIA_TILING_STRUCT_H_
#include "fia_tiling_compile_info.h"
#include <ATen/Operators.h>
using namespace std;
enum class InputLayout {
    SH,
    BSH,
    BNSD,
    NSD,
    BSND,
    BNSD_BSND,
    TND,
    NTD_TND,
    NZ,
    BBH,
    BNBD,
    NONE,
};

enum class FiaSparseEnum : uint8_t {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

enum class SplitCoreMode {
    SPLIT_NBS_VECTOR = 0,
    SPLIT_NBS_CUBE,
    SPLIT_ONEN_VECTOR,
    SPLIT_ONEN_CUBE,
    BALANCE_VECTOR,
    BALANCE_CUBE,
};

enum class LayoutType : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum FiaAttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE,
    BAND_LEFT_UP_CAUSAL_MODE
};

struct ContextParamsForTiling {
    
    const at::Tensor *attentionMask = nullptr;
    const at::Tensor *actualSequenceLengthQ = nullptr;
    const at::Tensor *actualSequenceLengthKV = nullptr;

    at::ScalarType inputDataType = at::ScalarType::Float;
    at::ScalarType kDataType = at::ScalarType::Float;
    at::ScalarType vDataType = at::ScalarType::Float;
 
    at::ScalarType maskDataType = at::ScalarType::Bool;
  
    at::ScalarType outputDataType = at::ScalarType::Float;

    at::IntArrayRef queryInputShape;
    at::IntArrayRef keyInputShape;
    at::IntArrayRef valueInputShape;
    at::IntArrayRef attentionMaskShape;
    at::IntArrayRef outputShape;


    int32_t headsNumber;
    int32_t sparseMode;
    int64_t preToken;
    int64_t nextToken;
    double scaleValue;

    string layout;
    int32_t numKeyValueHeads;
    size_t workspaceSize;

    const FiaDemoCompileInfo *compileInfoPtr = nullptr;

    uint32_t isKvContinuous = 0;
    std::vector<const at::IntArrayRef *> kTensorList = {nullptr};
    std::vector<const at::IntArrayRef *> vTensorList = {nullptr};
    uint32_t maxKVs = 0;
    uint32_t isBSNDOut = 0;
};

#endif
