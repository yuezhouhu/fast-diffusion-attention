/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "philox.cuh"
#include "utils.h"

namespace FLASH_NAMESPACE {

struct Dropout {

    const unsigned long long seed, offset;
    const uint8_t p_dropout_in_uint8_t;

    __forceinline__ __device__ Dropout(const unsigned long long seed, const unsigned long long offset,
                              const uint8_t p_dropout_in_uint8_t,
                              const int bid, const int hid, const int tid, const int nheads)
            : seed(seed)
            , offset(offset + (bid * nheads + hid) * 32 + tid % 32)
            , p_dropout_in_uint8_t(p_dropout_in_uint8_t) {
    }

    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout(Tensor<Engine, Layout> &tensor_,
                                         int block_row_start, int block_col_start, int block_row_stride) {
        // convert shape from (4, MMA_M, MMA_N) to (8, MMA_M, MMA_N / 2)
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_dropout(tensor_.layout()));
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0));
        };
        static_assert(decltype(size<2>(tensor))::value % 2 == 0);
        const uint16_t p_dropout_8bit_in_uint16_t = uint16_t(p_dropout_in_uint8_t);
        const uint32_t p_dropout_8bit_in_uint32_t = (uint32_t(p_dropout_8bit_in_uint16_t) << 16) | uint32_t(p_dropout_8bit_in_uint16_t);
        // if (cute::thread0()) { printf("threshold2 = 0x%x\n", p_dropout_8bit_in_uint32_t); }
        #pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            uint2 rowcol = make_uint2(block_row_start, block_col_start);
            #pragma unroll
            for (int n = 0; n < size<2>(tensor) / 2; ++n, ++rowcol.y) {
                // if (cute::thread(32, 0)) { printf("m = %d, n = %d, row = %d, col = %d\n", m, n, int(rowcol.x), int(rowcol.y));}
                uint4 random_uint4 = FLASH_NAMESPACE::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
                // if (cute::thread0()) { printf("philox = %u, %d, %d, %d\n", random_uint4.x, random_uint4.y, random_uint4.z, random_uint4.w);}
                uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                // Special implementation for 16-bit types: we duplicate the threshold to the
                // low and high 16 bits of a 32-bit value, then use the f16x2 comparison instruction
                // to get a mask. The low 16 bits of the mask will be either 0xffff or 0x0000,
                // and the high 16 bits will be either 0xffff or 0x0000, depending on whether
                // the random value is less than the threshold.
                // We then do a bit-wise AND between the mask and the original value (in 32-bit).
                // We're exploiting the fact that floating point comparison is equivalent to integer
                // comparison, since we're comparing unsigned integers whose top 8-bits are zero.
                if (!encode_dropout_in_sign_bit
                    && (std::is_same<T, cutlass::half_t>::value || std::is_same<T, cutlass::bfloat16_t>::value)) {
                    uint16_t rnd_16[16];
                    #pragma unroll
                    for (int i = 0; i < 16; i++) { rnd_16[i] = uint16_t(rnd_8[i]); }
                    uint32_t (&rnd_32)[8] = reinterpret_cast<uint32_t (&)[8]>(rnd_16);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                        // if (cute::thread0()) { printf("random = 0x%x, 0x%x, 0x%x, 0x%x\n", rnd_32[j * 4 + 0], rnd_32[j * 4 + 1], rnd_32[j * 4 + 2], rnd_32[j * 4 + 3]); }
                        // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            uint32_t mask;
                            asm volatile("set.le.u32.f16x2 %0, %1, %2;\n" : "=r"(mask) : "r"(rnd_32[j * 4 + i]), "r"(p_dropout_8bit_in_uint32_t));
                            tensor_uint32(i) &= mask;
                        }
                        // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            tensor(i, m, n * 2 + j) = encode_dropout(rnd_8[j * 8 + i] <= p_dropout_in_uint8_t, tensor(i, m, n * 2 + j));
                        }
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                        // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                    }
                }
                // // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                // //     printf("n = %d, ph  Philox: %u, %u, %u, %u\n", n, rnd_8.x, rnd_8.y, rnd_8.z, rnd_8.w);
                // // }
            }
        }
    }

    // Dropout on a 16-column sub-tile of the scores (MMA_N = 2): half `half` (0: columns 0-15, 1: columns 16-31) of
    // the 16 x 32 blocks block_col_start of the random numbers. The same random numbers for the same elements as
    // apply_dropout on the whole 32 columns (each thread uses bytes [8 * half, 8 * half + 8) of its Philox output).
    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout_half(Tensor<Engine, Layout> &tensor_, int block_row_start,
                                                       const int block_col_start, const int half,
                                                       const int block_row_stride) {
        // convert shape from (4, MMA_M, 2) to (8, MMA_M, 1)
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_dropout(tensor_.layout()));
        using T = typename Engine::value_type;
        static_assert(decltype(size<2>(tensor))::value == 1);
        const uint16_t p_dropout_8bit_in_uint16_t = uint16_t(p_dropout_in_uint8_t);
        const uint32_t p_dropout_8bit_in_uint32_t = (uint32_t(p_dropout_8bit_in_uint16_t) << 16) | uint32_t(p_dropout_8bit_in_uint16_t);
        #pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            uint2 rowcol = make_uint2(block_row_start, block_col_start);
            uint4 random_uint4 = FLASH_NAMESPACE::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
            // The 8 bytes of this half (selects, no dynamic indexing of registers)
            const uint32_t lo = half ? random_uint4.z : random_uint4.x;
            const uint32_t hi = half ? random_uint4.w : random_uint4.y;
            uint8_t rnd_8[8];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { rnd_8[i] = uint8_t(lo >> (8 * i)); rnd_8[4 + i] = uint8_t(hi >> (8 * i)); }
            if (!encode_dropout_in_sign_bit
                && (std::is_same<T, cutlass::half_t>::value || std::is_same<T, cutlass::bfloat16_t>::value)) {
                uint16_t rnd_16[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) { rnd_16[i] = uint16_t(rnd_8[i]); }
                uint32_t (&rnd_32)[4] = reinterpret_cast<uint32_t (&)[4]>(rnd_16);
                Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, 0));
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    uint32_t mask;
                    asm volatile("set.le.u32.f16x2 %0, %1, %2;\n" : "=r"(mask) : "r"(rnd_32[i]), "r"(p_dropout_8bit_in_uint32_t));
                    tensor_uint32(i) &= mask;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const bool keep = rnd_8[i] <= p_dropout_in_uint8_t;
                    tensor(i, m, 0) = keep ? tensor(i, m, 0) : (encode_dropout_in_sign_bit ? -tensor(i, m, 0) : T(0));
                }
            }
        }
    }

};

} // namespace FLASH_NAMESPACE
