/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "fia_tiling.h"

//打印调试使用
#include <torch/torch.h>
#include <iostream>

constexpr uint32_t BYTE_BLOCK = 32; // The block size of datacopy, which moves data at the block granularity.
constexpr uint32_t MASKDIM_2 = 2;
constexpr uint32_t MASKDIM_3 = 3;
constexpr uint32_t MASKDIM_4 = 4;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;

// The current requirement is a multiple of 128, and to prevent cross block handling, the mm base is also set to 128.


constexpr uint32_t SOUTER_FACTOR_SUB = 32;
constexpr uint32_t SOUTER_FACTOR_DEFAULT = 64;
constexpr uint32_t SINNER_FACTOR_SUB = 64;
constexpr uint32_t SINNER_FACTOR_DEFAULT = 128;
constexpr uint32_t SINNER_FACTOR_DOUBLE = 256;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t CV_RATIO = 2U; // Vector : Cube
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;
static const int64_t GM_ALIGN = 512;

constexpr uint32_t LOOP_BEGIN_NUM = 0;

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

uint8_t FiaDemoTiling::SetPlatMemoryInfo(ContextParamsForTiling& contextKeyParams) {
    // In subsequent version, contextKeyParams will be written as a member variable of the class.
    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    contextKeyParamsPtr = &contextKeyParams;
    coreNum = compileInfoPtr->aivNum;
    aivNum = compileInfoPtr->aivNum;
    aicNum = compileInfoPtr->aicNum;
    return 1;
}

uint8_t FiaDemoTiling::SetAttributeInfo(ContextParamsForTiling&contextKeyParams){
    actSeqLenDims=0;
    actSeqLenKVDims = 0;
    enableActSeqLen = 0;
    enableActSeqLenKV = 0;
    // mask check
    const at::IntArrayRef attenMaskShape = contextKeyParams.attentionMaskShape;
    enableMask = !attenMaskShape.empty();

    // sparsemode check
    const int32_t sparseMode = contextKeyParams.sparseMode;
    isDefaultSparseMode = (sparseMode == SPARSE_MODE_NO_MASK);
    enableTensorList = false;
    return 1;
}

bool FiaDemoTiling::SetInputLayout(string& layout) {
    if (layout == "" || layout == "BSH") {
        inputLayout = InputLayout::BSH;
    } else if (layout == "TND") {
        inputLayout = InputLayout::TND;
    } else if (layout == "BSND") {
        inputLayout = InputLayout::BSND;
    } else if (layout == "BNSD" || layout == "BNSD_BSND") { // Reuse BNSD process for BNSD_BSND
        inputLayout = InputLayout::BNSD;
    } else {
        return false;
    }

    return true;
}
int64_t GetMaxSeq(const at::Tensor* actualSeqLength) {
    int64_t max = actualSeqLength[0].item<int64_t>();
    for (int i = 1; i < actualSeqLength->numel(); ++i) {
        max = std::max(max, actualSeqLength[i].item<int64_t>() - actualSeqLength[i-1].item<int64_t>());
    }

    return max;
}

bool FiaDemoTiling::SetShape(FIAShapeInfo& shapeInfo, ContextParamsForTiling& contextKeyParams, const std::string inputName) {
    const at::IntArrayRef shape = (inputName == "querry") ? contextKeyParams.queryInputShape :
                                  (inputName == "key")    ? contextKeyParams.keyInputShape :
                                                            contextKeyParams.valueInputShape;

    if ((inputLayout == InputLayout::BNSD)){
        shapeInfo.b = shape[0];
        shapeInfo.n = shape[1];
        shapeInfo.s = shape[2]; // 2 for Sequence length
        shapeInfo.d = shape[3]; // 3 for D dim
        shapeInfo.h = shapeInfo.n * shapeInfo.d;
    } else if ((inputLayout == InputLayout::BSH)){
        shapeInfo.b = shape[0];
        shapeInfo.s = shape[1];
        shapeInfo.h = shape[2];
        if (inputName == "query") {
            shapeInfo.n = static_cast<int64_t>(contextKeyParams.headsNumber);
        } else {
            shapeInfo.n = static_cast<int64_t>(contextKeyParams.numKeyValueHeads);
            shapeInfo.n = shapeInfo.n > 0 ? shapeInfo.n : static_cast<int64_t>(contextKeyParams.headsNumber);
        }
        shapeInfo.d = shapeInfo.n > 0 ? shapeInfo.h / shapeInfo.n : 0;
    }else if ((inputLayout == InputLayout::BSND)) {
        shapeInfo.b = shape[0];
        shapeInfo.s = shape[1];
        shapeInfo.n = shape[2]; // 2 for head dim
        shapeInfo.d = shape[3]; // 3 for D dim
        shapeInfo.h = shapeInfo.n * shapeInfo.d;
    }  else {
        return false;
    }
    return true;
}

bool FiaDemoTiling::SetHeadNumRatio(ContextParamsForTiling& contextKeyParams) {
    const int32_t nQ = contextKeyParams.headsNumber;
    const int32_t nKV = contextKeyParams.numKeyValueHeads;
    if(enableIFA){
        gSize = nQ / nKV;
    }

    if (nKV == 0) { // Detected that nKV is the default value, which means that the customer did not pass in.
        faTilingAdapter.inputParamsRegbase.set_headNumRatio(1);
        m_gOfMla = 1;
        return true;
    }

    if (enableIFA) {
        faTilingAdapter.inputParamsRegbase.set_headNumRatio(1);
        m_gOfMla = gSize;
        return true;
    } else {
         m_gOfMla = 1;
    }

    faTilingAdapter.inputParamsRegbase.set_headNumRatio(nQ / nKV);
    return true;
}

bool FiaDemoTiling::SetMaskTypeAndShape(ContextParamsForTiling& contextKeyParams) {
    const at::IntArrayRef attenMaskShape = contextKeyParams.attentionMaskShape;

    int64_t maskKVsSize = 2048; // 2048 : default the last frist dim.
    int64_t maskQsSize = 2048; // 2048 : default the last second dim.
    // 1: last frist dim, 2: last second dim
    maskKVsSize = attenMaskShape[attenMaskShape.size() - 1];
    maskQsSize = attenMaskShape[attenMaskShape.size() - 2]; // 2 for Q dim index
    
    if (enableIFA) {
        maskQsSize = 1;
    }

    faTilingAdapter.inputParamsRegbase.set_attenMaskS2Size(maskKVsSize);
    faTilingAdapter.inputParamsRegbase.set_attenMaskS1Size(maskQsSize);
    return true;
}

void FiaDemoTiling::SetSparseModeData(ContextParamsForTiling& contextKeyParams,
    const at::IntArrayRef attenMaskShape, 
    const int32_t& sparseMode, const int64_t& preTokens, const int64_t& nextTokens) {


    if (preTokens > SPARSE_MODE_INT_MAX) {
        sparsePreTokens = SPARSE_MODE_INT_MAX;
    } else if (preTokens < -(SPARSE_MODE_INT_MAX)) {
        sparsePreTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparsePreTokens = preTokens;
    }

    if (nextTokens > SPARSE_MODE_INT_MAX) {
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    } else if (nextTokens < -(SPARSE_MODE_INT_MAX)) {
        sparseNextTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparseNextTokens = nextTokens;
    }
    if ((!attenMaskShape.empty()) && (sparseMode != -1)) {
        if (sparseMode == SPARSE_MODE_LEFT_UP) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = 0;
        // Right down tokens are calculated on the kernel side.
        } else if (sparseMode == SPARSE_MODE_RIGHT_DOWN) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
        } else if (sparseMode == SPARSE_MODE_ALL_MASK) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
        } 
        if (enableIFA) {
            sparseModeVal = SPARSE_MODE_NO_MASK;
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
        }
        sparseModeVal = sparseMode;
        
    }
}

void FiaDemoTiling::SetSparseType(uint32_t qS)
{
    if (sparseModeVal == SPARSE_MODE_NO_MASK) {
        if (sparsePreTokens >= qS && sparseNextTokens == 0) {
            sparseType = static_cast<uint8_t>(FiaSparseEnum::CAUSAL);
        } else if (sparsePreTokens >= qS  && sparseNextTokens >= S2) {
            sparseType = static_cast<uint8_t>(FiaSparseEnum::ALL);
        } else {
            sparseType = static_cast<uint8_t>(FiaSparseEnum::BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_ALL_MASK) {
        sparseType = static_cast<uint8_t>(FiaSparseEnum::ALL);
    } else if (sparseModeVal == SPARSE_MODE_LEFT_UP) {
        sparseType = static_cast<uint8_t>(FiaSparseEnum::CAUSAL);
    } else if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
        if (qS == S2) {
            sparseType = static_cast<uint8_t>(FiaSparseEnum::CAUSAL);
        } else {
            sparseType = static_cast<uint8_t>(FiaSparseEnum::BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_BAND) {
        sparseType = static_cast<uint8_t>(FiaSparseEnum::BAND);
    }
}

bool FiaDemoTiling::SetSparseMode(ContextParamsForTiling& contextKeyParams,uint32_t qS) {
    const int32_t sparseMode = contextKeyParams.sparseMode;
    const int64_t nextTokens = contextKeyParams.nextToken;
    const int64_t preTokens = contextKeyParams.preToken;

    const at::IntArrayRef attenMaskShape = contextKeyParams.attentionMaskShape;
    SetSparseModeData(contextKeyParams, attenMaskShape, sparseMode, preTokens, nextTokens);

    SetSparseType(qS);
    return true;
}

uint8_t FiaDemoTiling::CheckSingleAttribute(ContextParamsForTiling& contextKeyParams, FIAShapeInfo& queryShapeInfo,
    FIAShapeInfo& keyShapeInfo, FIAShapeInfo& valueShapeInfo) {
    SetInputLayout(contextKeyParams.layout);
    SetShape(queryShapeInfo, contextKeyParams, "querry");
    if(queryShapeInfo.s == 1) enableIFA = true;
    SetShape(keyShapeInfo, contextKeyParams, "key");
    SetShape(valueShapeInfo, contextKeyParams, "value");

    SetHeadNumRatio(contextKeyParams);


    if (enableTensorList) {
        S2 = contextKeyParams.maxKVs;
    } else {
        S2 = keyShapeInfo.s;
    }
    
    // mask check
    if (enableMask) {
        SetMaskTypeAndShape(contextKeyParams);
    } 

    SetSparseMode(contextKeyParams, queryShapeInfo.s);

    return 1;
}

void FiaDemoTiling::SetTilingDataAttribute(ContextParamsForTiling& contextKeyParams) {
    faTilingAdapter.inputParamsRegbase.set_preTokens(sparsePreTokens);
    faTilingAdapter.inputParamsRegbase.set_nextTokens(sparseNextTokens);

    faTilingAdapter.inputParamsRegbase.set_isActualSeqLengthsNull(static_cast<uint32_t>(!enableActSeqLen));
    faTilingAdapter.inputParamsRegbase.set_isActualSeqLengthsKVNull(static_cast<uint32_t>(!enableActSeqLenKV));
    faTilingAdapter.inputParamsRegbase.set_actualSeqLengthsSize(actSeqLenDims);
    faTilingAdapter.inputParamsRegbase.set_actualSeqLengthsKVSize(actSeqLenKVDims);

    faTilingAdapter.inputParamsRegbase.set_scaleValue(contextKeyParams.scaleValue);

    faTilingAdapter.inputParamsRegbase.set_isKvContinuous(contextKeyParams.isKvContinuous);

    faTilingAdapter.inputParamsRegbase.set_fromFused(1);
    faTilingAdapter.inputParamsRegbase.set_isBSNDOut(contextKeyParams.isBSNDOut);


    uint32_t originHeadSize = faTilingAdapter.inputParamsRegbase.get_dSize();

    uint32_t blockElementCnt = BYTE_BLOCK / dataTypeSize;
    if (originHeadSize % blockElementCnt != 0) { // Determine if D is aligned with 32B, using fp16 type with 16 elements.
        isDNoTail = false;
    } 
}

void FiaDemoTiling::SetTilingData(ContextParamsForTiling& contextKeyParams, FIAShapeInfo& queryShapeInfo, FIAShapeInfo& valueShapeInfo) {
    //  IFA flag
    faTilingAdapter.inputParamsRegbase.set_isGqa(enableIFA);


    faTilingAdapter.inputParamsRegbase.set_dSize(queryShapeInfo.d);
    faTilingAdapter.inputParamsRegbase.set_dSizeV(valueShapeInfo.d);
    faTilingAdapter.inputParamsRegbase.set_s2Size(S2);
    faTilingAdapter.inputParamsRegbase.set_s1Size(queryShapeInfo.s);
    m_qn = queryShapeInfo.n;
    faTilingAdapter.inputParamsRegbase.set_bSize(queryShapeInfo.b);

    SetTilingDataAttribute(contextKeyParams);
}



void FiaDemoTiling::InferSplitCoreMode() {
    splitCoreMode = SplitCoreMode::SPLIT_NBS_CUBE;
}

bool FiaDemoTiling::AdjustCVTilingCVDiff(const ContextParamsForTiling& contextKeyParams,
    uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor, const FIAShapeInfo& queryShapeInfo,
    const FIAShapeInfo& valueShapeInfo) {
    uint32_t minFactor = SOUTER_FACTOR_DEFAULT;
    uint32_t rectangleFactor = SINNER_FACTOR_DEFAULT;
    softmaxSOuterFactor = SOUTER_FACTOR_DEFAULT;
    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    if (faTilingAdapter.inputParamsRegbase.get_dSizeV() <= 128) { 
        bool checkDtype = contextKeyParams.inputDataType == at::kHalf || contextKeyParams.inputDataType == at::kBFloat16;
        bool checkQueryAndValueS = queryShapeInfo.s <= SOUTER_FACTOR_DEFAULT && S2 > SINNER_FACTOR_DEFAULT;
    
        int32_t preTokens = faTilingAdapter.inputParamsRegbase.get_preTokens();
        int32_t nextTokens = faTilingAdapter.inputParamsRegbase.get_nextTokens();
        
        if (sparseModeVal == SPARSE_MODE_NO_MASK) {
            preTokens = (preTokens > 0) ? 0 : preTokens;
        } else if (sparseModeVal == SPARSE_MODE_BAND) {
            nextTokens = (nextTokens > 0) ? 0 : nextTokens;
        }
        // actual calculation area is smaller than SINNER_FACTOR_DEFAULT in SPARSE_MODE_LEFT_UP mode
        bool checkSparseMode = ((sparseModeVal == SPARSE_MODE_ALL_MASK) || (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) || \
            (((sparseModeVal == SPARSE_MODE_NO_MASK) || (sparseModeVal == SPARSE_MODE_BAND)) && \
            (preTokens + nextTokens > static_cast<int32_t>(SINNER_FACTOR_DEFAULT))));
        if (checkDtype && checkQueryAndValueS && checkSparseMode) {
            minFactor = SOUTER_FACTOR_SUB;
            rectangleFactor = SINNER_FACTOR_DOUBLE;
            softmaxSOuterFactor = SOUTER_FACTOR_SUB;
        } 
    } else if (faTilingAdapter.inputParamsRegbase.get_dSizeV() > 128 && !enableIFA) { // 128 : D size
        minFactor = SOUTER_FACTOR_DEFAULT;
        rectangleFactor = SINNER_FACTOR_DEFAULT;
        softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    } else if ((enableIFA && faTilingAdapter.inputParamsRegbase.get_dSizeV() > 128)) { // IFA VD > 128
        minFactor = SOUTER_FACTOR_DEFAULT;
        rectangleFactor = SINNER_FACTOR_DEFAULT;
        softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    }
    sOuterFactor = minFactor;
    sInnerFactor = rectangleFactor;

    return true;
}

uint8_t FiaDemoTiling::AdjustTilingData(ContextParamsForTiling& contextKeyParams, const FIAShapeInfo& queryShapeInfo,
    const FIAShapeInfo& valueShapeInfo) {
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    uint32_t softmaxSInnerFactor = 0;
    uint32_t softmaxSOuterFactor = 0;
    // Currently, there will be no D splitting scenario, and split D = 0 is default when splitting.
    auto ret = AdjustCVTilingCVDiff(contextKeyParams, sOuterFactor, sInnerFactor, softmaxSOuterFactor, queryShapeInfo, valueShapeInfo);
    softmaxSInnerFactor = sInnerFactor;
    
    m_sOuterFactor = sOuterFactor;
    m_sInnerFactor = sInnerFactor;

    sOuterFactorTiling = sOuterFactor;
    softmaxSInnerFactorTiling = softmaxSInnerFactor;
    softmaxSOuterFactorTiling = softmaxSOuterFactor;
    return 1;
}


uint8_t FiaDemoTiling::SetQKVStartIdx(ContextParamsForTiling& contextKeyParams) {
    auto &inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_qStartIdx(0);
    inputParams.set_kvStartIdx(0);
    return 1;
}

void FiaDemoTiling::SetLayoutType()
{
    static std::map<InputLayout, LayoutType> layoutStrToLayoutTypeMap = {
        {InputLayout::BSH, LayoutType::LAYOUT_BSH},
        {InputLayout::TND, LayoutType::LAYOUT_TND},
        {InputLayout::BSND, LayoutType::LAYOUT_BSND},
        {InputLayout::BNSD, LayoutType::LAYOUT_BNSD},
    };
    auto itr = layoutStrToLayoutTypeMap.find(inputLayout);
    if (itr == layoutStrToLayoutTypeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(0));
    } else {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(itr->second));
    }
}

void FiaDemoTiling::SetAttenMaskCompressMode()
{
    static std::map<uint32_t, uint8_t> sparseToCompressModeMap = {
        {SPARSE_MODE_NO_MASK, FiaAttenMaskCompressMode::NO_COMPRESS_MODE},
        {SPARSE_MODE_ALL_MASK, FiaAttenMaskCompressMode::NO_COMPRESS_MODE},
        {SPARSE_MODE_LEFT_UP, FiaAttenMaskCompressMode::LEFT_UP_CAUSAL_MODE},
        {SPARSE_MODE_RIGHT_DOWN, FiaAttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE},
        {SPARSE_MODE_BAND, FiaAttenMaskCompressMode::BAND_MODE}
    };
    auto itr = sparseToCompressModeMap.find(sparseModeVal);
    if (itr == sparseToCompressModeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(0);
    } else {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(itr->second);
    }
}
void FiaDemoTiling::FlashAttentionInitOutputSplit(int64_t totalSize) {
    // Upward rounding, coreNum has been verified to be non-zero when obtained.
    uint32_t singleCoreSize = (totalSize + coreNum - 1) / (coreNum);

    faTilingAdapter.initOutputParams.set_singleCoreSize(singleCoreSize);
    faTilingAdapter.initOutputParams.set_totalOutputSize(totalSize);
}


void FiaDemoTiling::GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) {

    if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
    } else if (sparseModeVal == SPARSE_MODE_BAND) {
        preTokensLeftUp = faTilingAdapter.inputParamsRegbase.get_preTokens() - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = faTilingAdapter.inputParamsRegbase.get_nextTokens() + actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = faTilingAdapter.inputParamsRegbase.get_preTokens();
        nextTokensLeftUp = faTilingAdapter.inputParamsRegbase.get_nextTokens();
    }
}

void FiaDemoTiling::FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
    int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const {
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
        (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

int64_t FiaDemoTiling::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
        int64_t sInner, int64_t sOuter, int64_t token) {
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength- blockToken) / smallSize - tolerance;
    blockNums += (innerCutBlockNums > 0) ? (innerCutBlockNums % tolerance + innerCutBlockNums) *
        (innerCutBlockNums / tolerance + 1) / 2 : 0; // 2: The denominator of the arithmetic sequence summation formula
    blockNums -= (innerCutBlockLeftNums > 0) ? (innerCutBlockLeftNums % tolerance + innerCutBlockLeftNums) *
        (innerCutBlockLeftNums / tolerance + 1) / 2 : 0; // 2: The denominator of the arithmetic sequence summation formula
    blockNums -= (innerCutBlockDownNums > 0) ? (innerCutBlockDownNums % tolerance + innerCutBlockDownNums) *
        (innerCutBlockDownNums / tolerance + 1) / 2 : 0; // 2: The denominator of the arithmetic sequence summation formula
    return blockNums;
}

int64_t FiaDemoTiling::GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
    uint32_t sOuterSize, uint32_t sInnerSize, int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed) {
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums = (actualSeqLengthKV + static_cast<int64_t>(sInnerSize) - 1) /
            static_cast<int64_t>(sInnerSize);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize);
        int64_t outerBlockNums = (actualSeqLength + static_cast<int64_t>(sOuterSize) - 1) /
            static_cast<int64_t>(sOuterSize);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // Must meet this condition : pretoken + nexttoken > 0
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
            static_cast<int64_t>(sOuterSize), nextTokensLeftUp);
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
            static_cast<int64_t>(sOuterSize), blockSeqLengthKV - blockSeqLength + preTokensLeftUp);
        return toCalcBlockNums;
    }
}


int64_t FiaDemoTiling::GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) {
    int64_t sInnerBlockNums = 0;
 
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ? innerBlockNums :
            (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }
 
    return sInnerBlockNums;
}

void FiaDemoTiling::ComputeSplitNBSeq(uint32_t batchSize,
    const size_t tilingElementArrayLen, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
    uint32_t sOuterSize, uint32_t sInnerSize, double coreWightTarget, uint32_t& curCore) {
    std::vector<uint32_t> coreSposEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSposStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidStart(tilingElementArrayLen, 0U);
    std::vector<int64_t> sparseStartIdx(tilingElementArrayLen, 0L);
    std::vector<uint32_t> bnStartIdx(tilingElementArrayLen, 0U);
    std::vector<int64_t> gS1StartIdx(tilingElementArrayLen, 0L);
    // Temporary algorithm to be optimized
    int64_t curWight = 0;
    curCore = 0;
    uint32_t tmpCoreNidEnd = 0; // actual seq为0时不分配核
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        for (uint32_t headNum = 0; headNum < m_qn; headNum++) {
            // 针对行无效情况修正actualseqlen
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx] + actualSharedPrefixLen,
                preTokensLeftUp, nextTokensLeftUp);
            int64_t actualSeqLength = actualSeqLengths[sIdx];
            int64_t actualSeqLengthKV = actualSeqLengthsKV[sIdx];
            FixParamWithRowInvalid(actualSeqLength, actualSeqLengthKV + actualSharedPrefixLen,
                preTokensLeftUp, nextTokensLeftUp);

            int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
            int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize +
                (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(coreWightTarget * double(curCore + 1)) - curWight;
                int64_t sInnerIndexStart = -(preTokensLeftUp > 0 ? (preTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                    static_cast<int64_t>(sInnerSize) : preTokensLeftUp / static_cast<int64_t>(sInnerSize));
                int64_t sInnerIndexEnd = nextTokensLeftUp > 0 ? (nextTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                    static_cast<int64_t>(sInnerSize) : nextTokensLeftUp / static_cast<int64_t>(sInnerSize);
                
                // The number of innerBlock blocks in each outBlock row represents the calculation amount of each outBlock row.
                int64_t sInnerBlockNums = GetSInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
                if (sInnerBlockNums - dif > dif && !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {

                    coreNidEnd[curCore] = tmpCoreNidEnd;
                    coreSidEnd[curCore] = tmpCoreSidEnd;
                    coreSposEnd[curCore] = tmpCoreSposEnd;
                    curCore += 1;
                    coreNidStart[curCore] = headNum;
                    coreSidStart[curCore] = sIdx;
                    coreSposStart[curCore] = sOuterIndex;

                    bnStartIdx[curCore] = sIdx * m_qn + headNum;
                    gS1StartIdx[curCore] = sOuterIndex;
                }
                tmpCoreNidEnd = headNum + 1;
                tmpCoreSidEnd = sIdx + 1;
                tmpCoreSposEnd = sOuterIndex + 1;

                curWight += sInnerBlockNums;
                preTokensLeftUp -= sOuterSize;
                nextTokensLeftUp += sOuterSize;
            }
        }
    }
    coreNidEnd[curCore] = tmpCoreNidEnd;
    coreSidEnd[curCore] = tmpCoreSidEnd;
    coreSposEnd[curCore] = tmpCoreSposEnd;
    bnStartIdx[curCore + 1] = batchSize * m_qn;
    gS1StartIdx[curCore + 1] = tmpCoreSposEnd;

    faTilingAdapter.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    faTilingAdapter.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}

void FiaDemoTiling::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum)
{
    faTilingAdapter.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_totalSize(totalSize);
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorTailSize(CalcTailSize(totalSize, faTilingAdapter.multiCoreParamsRegbase.get_splitFactorSize()));
}

 bool FiaDemoTiling::CheckMaskShape(ContextParamsForTiling& contextKeyParams, const int32_t* sparseMode,
    int64_t& attenMaskBatch, int64_t& attenMaskS1, int64_t& attenMaskS2, bool& checkMask, const uint32_t sQ,
    const uint32_t sK, const uint32_t batchSize, std::string& strMaskShape) {
    const at::IntArrayRef attenMaskShape = contextKeyParams.attentionMaskShape;
    size_t attenMaskDim = attenMaskShape.size();
    int64_t attenMaskN = 1U;
    if (attenMaskDim == MASKDIM_2) {
        if (enableIFA) {
            attenMaskBatch = attenMaskShape[0];
            attenMaskS1 = 1;
            attenMaskS2 = attenMaskShape[1];
            strMaskShape = std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskS2);
        } else {
            attenMaskS1 = attenMaskShape[0];
            attenMaskS2 = attenMaskShape[1];
            strMaskShape = std::to_string(attenMaskS1) + ", " + std::to_string(attenMaskS2);
        }
    } else if (attenMaskDim == MASKDIM_3) {
        attenMaskBatch = attenMaskShape[0];
        attenMaskS1 = attenMaskShape[1];
        attenMaskS2 = attenMaskShape[2]; // 2: When the dim is 3, the second dimension is S2.
        strMaskShape = std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskS1) + ", " + 
            std::to_string(attenMaskS2);
    } else if (attenMaskDim == MASKDIM_4) {
        attenMaskBatch = attenMaskShape[0];
        attenMaskN = attenMaskShape[1];
        attenMaskS1 = attenMaskShape[2]; // 2: When the dim is 4, the second dimension is S1.
        attenMaskS2 = attenMaskShape[3]; // 3: When the dim is 4, the third dimension is S2.
        strMaskShape = std::to_string(attenMaskBatch) + ", " + std::to_string(attenMaskN) + ", " + 
            std::to_string(attenMaskS1) + ", " + std::to_string(attenMaskS2);
    } else {
        
        return false;
    }

    if (enableIFA) {
        checkMask = (attenMaskBatch == batchSize) && (attenMaskS1 == 1) && (attenMaskS2 >= S2);
    } else if (isDefaultSparseMode || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
        checkMask = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) && (attenMaskBatch == 1 || attenMaskBatch == batchSize);
    } else if ((sparseMode != nullptr) && ((*sparseMode == SPARSE_MODE_LEFT_UP) ||
        (*sparseMode == SPARSE_MODE_RIGHT_DOWN) || (*sparseMode == SPARSE_MODE_BAND))) {
        checkMask = (attenMaskBatch == 1) && (attenMaskN == 1) &&
            (attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE) && (attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE);
    }
    return true;
}

bool FiaDemoTiling::CheckMaskShapeCrossSparse(ContextParamsForTiling& contextKeyParams,
    const int32_t* sparseMode, uint32_t sQ, const uint32_t sK, const uint32_t batchSize) {
    if (!enableMask) {
        return true;
    }
    if (enableIFA ) {
        sQ /= gSize; // 合轴场景使用原始的seq长度校验
    }
    int64_t attenMaskBatch = 1;
    int64_t attenMaskS1 = 0;
    int64_t attenMaskS2 = 0;
    bool checkMask = 0;
    std::string strMaskShape;
    if (!CheckMaskShape(contextKeyParams, sparseMode, attenMaskBatch, attenMaskS1, attenMaskS2, checkMask, sQ, sK, batchSize, strMaskShape)) {
        return false;
    }

    attenMaskShapeType = attenMaskBatch > 1 ? 1 : 2; // 1 for multi-batch and 2 for 1 batch, same as fa
    return true;
}

uint8_t FiaDemoTiling::CheckCrossoverAttribute(ContextParamsForTiling& contextKeyParams,
    FIAShapeInfo& queryShapeInfo) {
    const int32_t* sparseMode = &contextKeyParams.sparseMode;
    CheckMaskShapeCrossSparse(contextKeyParams, sparseMode, queryShapeInfo.s, S2 + actualSharedPrefixLen, queryShapeInfo.b);

    return 1;
}

bool FiaDemoTiling::ParseActualSeqLengths(ContextParamsForTiling& contextKeyParams,
    FIAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV) {
    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    const at::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const at::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;

    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        
        if (!enableActSeqLen) {
            actualSeqLengths[i] = queryShapeInfo.s;
        } 
        middleActualSeqLengths += actualSeqLengths[i];
        if (!enableActSeqLenKV) {       // The user did not input act_seq_kv
            if (!enableTensorList) {
                actualSeqLengthsKV[i] = S2;
            } 
        }
        maxActualseqKV = std::max(maxActualseqKV, actualSeqLengthsKV[i]);
    }

    return true;
}

bool FiaDemoTiling::CheckMultiFeatureCrossover(ContextParamsForTiling& contextKeyParams,
    FIAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV) {
    if (!ParseActualSeqLengths(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV)) {
        return false;
    }
    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    int64_t preTokensPerbatch = 0;
    int64_t nextTokensPerbatch = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
            preTokensPerbatch = SPARSE_MODE_INT_MAX;
            
            nextTokensPerbatch = actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
            
        } else if (sparseModeVal == SPARSE_MODE_BAND) {
            preTokensPerbatch = sparsePreTokens - actualSeqLengthsKV[i] - actualSharedPrefixLen + actualSeqLengths[i];
            nextTokensPerbatch = sparseNextTokens + actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
        } else {
            preTokensPerbatch = sparsePreTokens;
            nextTokensPerbatch = sparseNextTokens;
        }
        if ((nextTokensPerbatch < 0) ||
            (actualSeqLengths[i] > (actualSeqLengthsKV[i] + actualSharedPrefixLen + preTokensPerbatch))) {
            needInit = 1;
        }
    }

    return true;
}

void FiaDemoTiling::FlashAttentionSplitNBSeq(std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed) {

    uint32_t curCoreNum = coreNum;
    uint32_t batchSize = faTilingAdapter.inputParamsRegbase.get_bSize();
    uint32_t sOuterSize = m_sOuterFactor;
    uint32_t sInnerSize = m_sInnerFactor;

    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) { // From the perspective of cube
        sOuterSize = sOuterSize * CV_RATIO;
        curCoreNum = curCoreNum / CV_RATIO;
    }

    int64_t totalBlockNumsOneHead = 0; // The calculation amount of all sequences for a single head

    std::vector<uint32_t> sInnerLoopTimes(batchSize);
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx] + actualSharedPrefixLen,
            preTokensLeftUp, nextTokensLeftUp);

        // 计算各sparse mode情况下，减去行无效后真实的actseqlen
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], preTokensLeftUp, nextTokensLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        sInnerLoopTimes[sIdx] = (actualSeqLengthsKV[sIdx] + sInnerSize - 1) / sInnerSize +
            (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], sOuterSize,
            sInnerSize, preTokensLeftUp, nextTokensLeftUp, isAttenMaskUsed);
    }

    // Amount of computation per core
    double coreWightTarget = (double(totalBlockNumsOneHead * m_qn) / double(curCoreNum));

    int64_t s1OuterSize = (faTilingAdapter.inputParamsRegbase.get_s1Size() + sOuterSize - 1) / sOuterSize;
    faTilingAdapter.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    // The tiling structure element needs to have a length greater than or equal to the length specified
    // by TILING_DATA_FIELD_DEF_ARR. If the tiling structure definition specifies a length of 64,
    // the vector definition needs to compare its size with coreNum and take the larger value
    const size_t tilingElementArrayLen = (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    uint32_t curIndx = 0;
    ComputeSplitNBSeq(batchSize, tilingElementArrayLen, actualSeqLengths, actualSeqLengthsKV, sOuterSize,
        sInnerSize, coreWightTarget, curIndx);

    uint32_t actualCoreNums = (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) ? (curIndx + 1) * CV_RATIO : curIndx + 1;
    int64_t sinnerBlocknum = (faTilingAdapter.inputParamsRegbase.get_s2Size() + sInnerSize - 1) / sInnerSize;
    SetMultiCoreParamsRegbase((totalBlockNumsOneHead / sinnerBlocknum) * m_qn, static_cast<int64_t>((curIndx + 1)));
}

uint8_t FiaDemoTiling::ComputeTilingData(ContextParamsForTiling& contextKeyParams,
    std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV) {
    // Compute tiling data.
    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE ) {
        bool isAttenMaskUsed = (!contextKeyParams.attentionMaskShape.empty());
        FlashAttentionSplitNBSeq(actualSeqLengths, actualSeqLengthsKV, isAttenMaskUsed);
    }

    if (needInit == 1) {
        int64_t shapeSize = 1;
        for(int64_t dim: contextKeyParams.outputShape){
            shapeSize *= dim;
        }
        FlashAttentionInitOutputSplit(shapeSize);
    }


    if (enableIFA){
        uint32_t sOuterSize = m_sOuterFactor;
        uint64_t batchSize = faTilingAdapter.inputParamsRegbase.get_bSize();

        uint64_t headNumKVSize = m_qn;
        uint64_t bng = batchSize * headNumKVSize * (gSize + sOuterSize - 1) / sOuterSize;
    }
    return 1;
}


void FiaDemoTiling::TilingDataconvert() {
    SetLayoutType();
    auto &inputParams = faTilingAdapter.inputParamsRegbase;

    // 将GS1合轴与不合轴场景下，有不同含义的n2Size、gSize与s1Size参数，转化为各自实际的值
    if (enableIFA) {
        inputParams.set_n2Size(m_qn);
        inputParams.set_gSize(m_gOfMla);
        inputParams.set_s1Size(inputParams.get_s1Size() / m_gOfMla);
    } else {
        inputParams.set_n2Size(m_qn / inputParams.get_headNumRatio());
        inputParams.set_gSize(inputParams.get_headNumRatio());
        inputParams.set_s1Size(inputParams.get_s1Size());
    }
    // inputParams.set_bandIndex(0); // 训练代码中在TND场景生效，用于计算s2方向循环的起始位置
    inputParams.set_attenMaskShapeType(attenMaskShapeType);
    inputParams.set_attenMaskDataType(1); // 默认值
    SetAttenMaskCompressMode();
    inputParams.set_implMode(0);
    inputParams.set_sparseType(sparseType);
}


uint8_t FiaDemoTiling::RunBigKernelTilingWithParams(ContextParamsForTiling& contextKeyParams)
{
    SetPlatMemoryInfo(contextKeyParams);
    SetAttributeInfo(contextKeyParams);
    FIAShapeInfo queryShapeInfo;
    FIAShapeInfo keyShapeInfo;
    FIAShapeInfo valueShapeInfo;
    
    CheckSingleAttribute(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo);

    if (enableIFA) {
        queryShapeInfo.n = queryShapeInfo.n / gSize;
        queryShapeInfo.s = queryShapeInfo.s * gSize;
    }

    std::vector<int64_t> actualSeqLengths(queryShapeInfo.b);
    std::vector<int64_t> actualSeqLengthsKV(keyShapeInfo.b);

    //mask
    CheckCrossoverAttribute(contextKeyParams, queryShapeInfo);

    CheckMultiFeatureCrossover(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV);

    SetTilingData(contextKeyParams, queryShapeInfo, valueShapeInfo);

    InferSplitCoreMode();

    AdjustTilingData(contextKeyParams, queryShapeInfo, valueShapeInfo);

    ComputeTilingData(contextKeyParams, actualSeqLengths, actualSeqLengthsKV);

    TilingDataconvert();

    SetQKVStartIdx(contextKeyParams);

    return 1;
}