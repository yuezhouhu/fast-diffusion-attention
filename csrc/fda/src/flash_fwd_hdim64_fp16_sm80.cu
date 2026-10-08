// Copyright (c) 2024, Tri Dao.
// Copyright (c) 2026, fast-diffusion-attention authors.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream) {
#ifdef FDA_TUNING
    // Tuning builds: FDA_TUNE_FWD=<i> runs candidate i of tuning.h instead of the dispatch
    if (run_mha_fwd_tuning<64, cutlass::half_t>(params, stream)) { return; }
#endif
    run_mha_fwd_hdim64<cutlass::half_t>(params, stream);
}

} // namespace FLASH_NAMESPACE
