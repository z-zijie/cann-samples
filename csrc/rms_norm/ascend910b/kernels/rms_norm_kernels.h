#pragma once
#include "kernel_operator.h"

namespace x {
namespace RmsNorm {

// Declaration of the RMSNorm kernels
__global__ __aicore__ void rms_norm_full_load_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd);
__global__ __aicore__ void rms_norm_multi_core_reduce_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace);

} // namespace RmsNorm
} // namespace x
