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


// Integer value of an environment variable (0 if it is not a number). Parsed by hand: std::atoi / strtol compiled
// with glibc >= 2.38 headers reference __isoc23_strtol, which older systems (e.g. glibc 2.35) do not have.
inline int parse_env_int(const char *s) {
    if (s == nullptr) { return 0; }
    bool negative = false;
    if (*s == '-' || *s == '+') { negative = *s == '-'; ++s; }
    int value = 0;
    for (; *s >= '0' && *s <= '9'; ++s) { value = value * 10 + (*s - '0'); }
    return negative ? -value : value;
}

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
        return parse_env_int(s);
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

inline int get_l2_size(int device) {
    int l2_size;
    CHECK_CUDA(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device));
    return l2_size;
}

// Number of dQ accumulation buffers of the deterministic backward (= thread blocks per (batch, head), each processing
// a fixed sequence of key tiles): enough thread blocks for about `waves` waves, at most one per key tile.
// FlashAttention-2 uses ceil(num_sm / (b h)) (1 wave), but its thread blocks cover one sequence, ours the two halves
// of a sample. FDA_DET_WAVES overrides the number of waves (experiments).
inline int get_deterministic_splits(const int num_sm, const int num_bh, const int num_key_tiles) {
    const char *s = std::getenv("FDA_DET_WAVES");
    const int waves = s != nullptr ? std::max(1, parse_env_int(s)) : 2;
    return std::max(1, std::min((waves * num_sm + num_bh - 1) / num_bh, num_key_tiles));
}

// Number of (batch, head) pairs per group in the order of the thread blocks (tile_of_block in diffusion.h): about
// `waves` waves of thread blocks per group, i.e. long enough to schedule the expensive tiles of the group first and
// absorb the tail, short enough to keep the working set of the concurrent thread blocks in L2; with bytes_per_bh > 0,
// the data shared by the thread blocks of a group (bytes_per_bh per (batch, head)) must also fit in twice the L2.
// The group size is a divisor of the number of (batch, head) pairs: all the groups are equal (a last, smaller group
// would be a tail of expensive tiles). FDA_SCHED_HEADS_FWD / FDA_SCHED_HEADS_BWD override it (experiments).
inline int get_sched_heads(const char *env_name, const int num_tiles, const int num_bh, const int num_sm,
                           const int waves = 4, const int64_t bytes_per_bh = 0) {
    const char *s = std::getenv(env_name);
    if (s != nullptr) {
        const int heads = parse_env_int(s);
        return heads < 1 ? 1 : (heads > num_bh ? num_bh : heads);
    }
    const float target = float(waves * num_sm) / float(num_tiles);
    const int64_t cap = bytes_per_bh <= 0 ? int64_t(num_bh)
        : std::max<int64_t>(1, 2 * int64_t(get_l2_size(get_current_device())) / bytes_per_bh);
    int best = 1;
    float best_dist = 1e30f;
    for (int g = 1; g <= num_bh; ++g) {
        if (g > cap) { break; }
        if (num_bh % g != 0) { continue; }
        // Distance in log scale
        const float r = g > target ? float(g) / target : target / float(g);
        if (r < best_dist) { best_dist = r; best = g; }
    }
    return best;
}

