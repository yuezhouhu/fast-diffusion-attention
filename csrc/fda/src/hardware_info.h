/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include <cstdlib>
#include <cstdio>
#include <tuple>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status_));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


inline int get_current_device() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    return device;
}

// Testing aid (not in FlashAttention-2): FDA_EMULATE_ARCH=86 or 89 makes the dispatch select the kernels of an sm86 /
// sm89 GPU (compute capability 8.x, 99 KB of shared memory per thread block), so that their numerics can be tested
// on an sm80 GPU. The selected kernels must fit the actual GPU, which is the case on sm80 (more shared memory).
inline int get_emulated_arch() {
    static const int arch = [] {
        const char *s = std::getenv("FDA_EMULATE_ARCH");
        return s == nullptr ? 0 : std::atoi(s);
    }();
    return arch;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    if (get_emulated_arch() > 0) { return {get_emulated_arch() / 10, get_emulated_arch() % 10}; }
    int capability_major, capability_minor;
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));
    return {capability_major, capability_minor};
}

// Shared memory per thread block (opt-in) and per SM, as used by the dispatch of the kernels.
inline int get_max_smem_per_block(int device) {
    const int arch = get_emulated_arch();
    if (arch == 86 || arch == 89) { return 99 * 1024; }
    int max_smem_per_block;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return max_smem_per_block;
}

inline int get_max_smem_per_sm(int device) {
    const int arch = get_emulated_arch();
    if (arch == 86 || arch == 89) { return 100 * 1024; }
    int max_smem_per_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    return max_smem_per_sm;
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}

