#pragma once
#include "kernel_operator.h"

namespace x {
namespace RmsNorm {

// Implemention of the RMSNorm kernels
__global__ __aicore__ void rms_norm_multi_core_reduce_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace)
{
    // TODO: Implement the multi core reduce kernel logic
}

} // namespace RmsNorm
} // namespace x
