/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Backward preprocessing: dPsum = rowsum(dO * O), LSE in log2 units, zero dQaccum, for one query tile of one half
 * of a sample (padded layout of the accumulators, see DiffusionSeqlenInfo::offset_padded).
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "diffusion.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementAccum, class ArchTag_, bool Clear_dQaccum, bool Varlen>
class FlashAttnBwdPreprocess {

public:

    // Type Aliases
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;

    static_assert(std::is_same_v<Element, cutlass::half_t> && ArchTag::kMinComputeCapability >= 75 ||
                  std::is_same_v<Element, cutlass::bfloat16_t> && ArchTag::kMinComputeCapability >= 80);

    static constexpr uint32_t MaxThreadsPerBlock = 256;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
    static constexpr int SharedStorageSize = 0;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<1>(TileShape_MK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBlockM = get<0>(TileShape_MK{});
    static constexpr int kHeadDim = get<1>(TileShape_MK{});
    // We want kBlockKGmem to be a power of 2 so that when we do the summing,
    // it's just between threads in the same warp
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static_assert((kBlockM * kHeadDim / kGmemElemsPerLoadAccum) % MaxThreadsPerBlock == 0, "MaxThreadsPerBlock must divide kBlockM * kHeadDim / kGmemElemsPerLoadAccum");
    using GmemLayoutAtomAccum = Layout<Shape<Int<MaxThreadsPerBlock>>>;
    using GmemTiledCopyAccum = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomAccum{},
                        Layout<Shape<Int<kGmemElemsPerLoadAccum>>>{}));  // Val layout, 4 vals per store

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;
    using ShapedPsum = cute::Shape<int32_t, int32_t>;  // (total_q_padded, head)
    using StridedPsum = cute::Stride<_1, int64_t>;
    using ShapedQaccum = cute::Shape<int32_t, int32_t>;  // (total_q_padded * d, head)
    using StridedQaccum = cute::Stride<_1, int64_t>;

    // Device side arguments
    struct Arguments {
        Element const* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        Element const* ptr_dO;
        StrideO const stride_dO;
        float* ptr_dPsum;
        ShapedPsum const shape_dPsum;
        StridedPsum const stride_dPsum;
        float const* ptr_LSE;
        StrideLSE const stride_LSE;
        float *ptr_LSE_log2;
        StridedPsum const stride_LSE_log2;
        ElementAccum* ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        int const seqlen;  // length of a sample (2n) if not varlen
        int const* cu_seqlens = nullptr;
        bool const skip_dqaccum_noisy = false;  // dQaccum of the noisy half is initialized by FlashAttnBwdNoisyDiag
    };

    // Kernel entry point API
    using Params = Arguments;

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        return args;
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, [[maybe_unused]] char* smem_buf) {

        static constexpr int kBlockM = get<0>(TileShape_MK{});

        int const thread_idx = threadIdx.x;
        int const m_block = blockIdx.x >> 1;
        int const half = blockIdx.x & 1;
        int const bidh = blockIdx.y;
        int const bidb = blockIdx.z;

        flash::DiffusionSeqlenInfo<Varlen> seqlen_info(bidb, params.seqlen, params.cu_seqlens);
        bool const is_varlen = Varlen && params.cu_seqlens;
        int const seqlen_o = seqlen_info.n;
        if (m_block * kBlockM >= seqlen_o) { return; }
        int const offset = seqlen_info.offset_half(half);
        int const offset_padded = seqlen_info.template offset_padded<kBlockM>(bidb, half);

        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gO = local_tile(cute::domain_offset(make_coord(offset, _0{}), mO), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)
        Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_O, params.stride_dO)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdO = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdO), TileShape_MK{}, make_coord(m_block, _0{}));  // (M, K)

        auto shape_LSE = select<0, 2, 3>(params.shape_O);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), shape_LSE, params.stride_LSE)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gLSE = local_tile(cute::domain_offset(make_coord(offset), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block));
        static_assert(kBlockM <= MaxThreadsPerBlock);
        float lse = thread_idx < seqlen_o - m_block * kBlockM && thread_idx < kBlockM ? gLSE(thread_idx) : INFINITY;

        GmemTiledCopy gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

        Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
        Tensor tOgdO = gmem_thr_copy_O.partition_S(gdO);
        // Construct identity layout for gO
        Tensor cO = cute::make_identity_tensor(TileShape_MK{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }

        // (8, kBlockM / 32, kHeadDim / 64) or (8, kBlockM / 16, kHeadDim / 128)
        Tensor tOrO = make_fragment_like(tOgO);
        Tensor tOrdO = make_fragment_like(tOgdO);
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
            gmem_tiled_copy_O, tOgO, tOrO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
            gmem_tiled_copy_O, tOgdO, tOrdO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );

        // Reshape from e.g. (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, (8, kHeadDim / 64))
        Layout l = make_layout(get<1>(tOrO.layout()), make_layout(get<0>(tOrO.layout()), get<2>(tOrO.layout())));
        Tensor tOrO_l = make_tensor(tOrO.data(), l);
        Tensor o_fp32 = make_tensor_like<float>(tOrO_l);
        flash::convert_type_out(tOrO_l, o_fp32);
        Tensor tOrdO_l = make_tensor(tOrdO.data(), l);
        Tensor do_fp32 = make_tensor_like<float>(tOrdO_l);
        flash::convert_type_out(tOrdO_l, do_fp32);
        // Sum across the last dimension
        Tensor dP_sum = make_tensor<float>(make_shape(size<0>(o_fp32)));
        #pragma unroll
        for (int mi = 0; mi < size<0>(o_fp32); ++mi) {
            float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
            #pragma unroll
            for (int ni = 1; ni < size<1>(o_fp32); ni++) {
                dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
            }
            flash::SumOp<float> sum_op;
            dP_sum(mi) = flash::Allreduce<kGmemThreadsPerRow>::run(dP_sum_cur, sum_op);
        }

        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_dPsum, params.stride_dPsum)(_, bidh);
        Tensor gdPsum = local_tile(cute::domain_offset(make_coord(offset_padded), mdPsum), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (get<1>(tOcO(_0{}, _0{}, _0{})) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(dP_sum); ++mi) {
                int const row = get<0>(tOcO(_0{}, mi, _0{}));
                gdPsum(row) = row < seqlen_o - m_block * kBlockM ? dP_sum(mi) : 0;
            }
        }

        // Rows beyond the end of the half get LSE = +inf, so that P = 0 for them in the backward kernel.
        Tensor mLSElog2 = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_dPsum, params.stride_LSE_log2)(_, bidh);
        Tensor gLSElog2 = local_tile(cute::domain_offset(make_coord(offset_padded), mLSElog2), Shape<Int<kBlockM>>{}, make_coord(m_block));
        if (thread_idx < kBlockM) {
            gLSElog2(thread_idx) = lse == -INFINITY ? 0.f : lse * float(M_LOG2E);
        }

        if (Clear_dQaccum && !(params.skip_dqaccum_noisy && half == 0)) {
            Tensor mdQaccum = make_tensor(make_gmem_ptr(params.ptr_dQaccum), params.shape_dQaccum, params.stride_dQaccum)(_, bidh);
            Tensor gdQaccum = local_tile(cute::domain_offset(make_coord(offset_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(m_block));
            GmemTiledCopyAccum gmem_tiled_copy_dQaccum;
            auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
            Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
            Tensor zero = make_fragment_like(tdQgdQaccum);
            clear(zero);
            cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, zero, tdQgdQaccum);
        }

    }

};

} // namespace flash
