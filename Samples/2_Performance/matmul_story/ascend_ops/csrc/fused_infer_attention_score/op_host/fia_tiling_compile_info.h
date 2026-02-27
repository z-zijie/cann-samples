#ifndef FIA_TILING_COMPILE_INFO_H
#define FIA_TILING_COMPILE_INFO_H
#include "tiling/platform/platform_ascendc.h"

struct FiaCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
    platform_ascendc::SocVersion socShortName;
};

#endif
