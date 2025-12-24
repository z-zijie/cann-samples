#pragma once
#include "kernel_operator.h"

namespace x {
namespace RmsNorm {

// Implemention of the RMSNorm kernels
__global__ __aicore__ void rms_norm_full_load_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd)
{
    // TODO: Implement the full load kernel logic
}

} // namespace RmsNorm
} // namespace x
