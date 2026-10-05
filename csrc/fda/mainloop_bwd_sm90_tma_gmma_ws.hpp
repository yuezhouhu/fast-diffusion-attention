/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Backward mainloop of the diffusion attention (adapted from the FlashAttention-3 Hopper mainloop).
 * A CTA computes dK and dV of one key tile (of the noisy or the clean half of a sample). It visits the query tiles
 * of DiffusionBwdTiles: for a clean key tile, the clean query tiles and the noisy query tiles of the later diffusion
 * blocks; for a noisy key tile, the noisy query tiles of the same diffusion blocks. dQ is accumulated in fp32 in
 * dQaccum (padded layout, see DiffusionSeqlenInfo::offset_padded).
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "diffusion.h"
#include "softmax.h"
#include "utils.h"
#include "copy_sm90_bulk_reduce.hpp"

namespace flash {

using namespace cute;

template <int Stages, int Stages_dO, int Stages_dS, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Has_softcap_, bool Varlen_, bool Deterministic,
        bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
        int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
        bool Mma_dP_is_RS=false>
struct CollectiveMainloopBwdSm90 {

    static constexpr int kStages = Stages;
    static constexpr int kStages_dO = Stages_dO;
    static constexpr int kStages_dS = Stages_dS;
    static_assert(kStages >= kStages_dO);
    static_assert(Stages_dS == 1 || Stages_dS == kStages);
    static_assert(!Mma_dP_is_RS || SdP_swapAB_);  // If Mma_dP_is_RS, we need SdP_SwapAB
    using ClusterShape = Shape<_1, _1, _1>;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;

    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using SeqlenInfo_t = flash::DiffusionSeqlenInfo<Varlen>;
    using Tiles_t = flash::DiffusionBwdTiles<kBlockM, kBlockN>;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp * 2;

    static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);
    static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarpGroups && SdP_swapAB && !dKV_swapAB;
    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarpGroups && AtomLayoutMdQ == NumMmaWarpGroups && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
    // static constexpr GMMA::Major PdS_Major = GMMA::Major::MN;
    static constexpr GMMA::Major PdSt_Major = PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<NumMmaWarpGroups / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TiledMmadPRS = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarpGroups / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Mma_dKV_is_RS,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, !dKV_swapAB ? PdSt_Major : GMMA::Major::MN, !dKV_swapAB ? GMMA::Major::MN : PdSt_Major>())
        >{},
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<NumMmaWarpGroups / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Mma_dQ_is_RS,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, !dQ_swapAB ? PdS_Major : GMMA::Major::MN, !dQ_swapAB ? GMMA::Major::MN : PdS_Major>())
        >{},
        AtomLayoutdQ{}));

    // We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
    // Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
    // Since this is GMMA::Major::K, the M dimension (kBlockM) doesn't matter for the layout, only the K dimension
    // changes the layout.
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>>()); // for dKV_Mma
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages_dO>{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomPdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<PdS_Major, Element,
                                       Int<kBlockM / AtomLayoutMSdP>,
                                       Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
        std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    // Need stride to be multiple of 32, otherwise we get error (misaligned address) when doing TMA if e.g. kBlockM=80
    // We set stride to be multiple of 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
    // it's still a valid smem address.
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
    using SmemLayoutLSEMma = std::conditional_t<
        SdP_swapAB,
        cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>,
        cute::Layout<cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>, cute::Stride<_1, _0, Int<cute::round_up(kBlockM, 64)>>>
    >;

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages_dO>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    using SmemLayoutPdSt =
        decltype(cute::composition(SmemLayoutPdS{},
                                   make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

    // Thread layout, 256 or 384 threads per row
    // We split into NumMmaWarpGroups so that we can do Bulk reduce add for each WG separately.
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroups>>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim / NumMmaWarpGroups>, Int<NumMmaWarpGroups>>>;

    static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;
    // If !SdP_swapAB, the accum registers hold P / dS, otherwise they hold Pt / dSt.
    // If PdS_major is MN, then we need to "transpose" the write.
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<(!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U32x4_STSM_N, cute::SM90_U32x2_STSM_N>,
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>
        >,
        Element
    >;

    using GmemTiledCopyQdO = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t>;  // (total_q_padded, head)
    using StrideLSE = cute::Stride<_1, int64_t>;  // (total_q_padded, head)
    using ShapedQaccum = cute::Shape<int32_t, int32_t>;  // (total_q_padded * d, head)
    using StridedQaccum = cute::Stride<_1, int64_t>;

    using TMA_QdO = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}),
        TileShape_MNK{},
        ClusterShape{})); // mcast along N mode for this M load, if any

    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{},
        TileShape_MNK{},
        ClusterShape{})); // no mcast for KV

    using TMA_V = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{},
        TileShape_MNK{},
        ClusterShape{})); // no mcast for KV

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);

    // These are tuned for speed.
    // They don't affect correctness.
    // Separate loops for the masked and the fully visible query tiles duplicate the code of a step, which costs
    // registers (spills with hdim 128). With a single loop, the mask is applied under a (uniform) branch.
    static constexpr bool SeparateMaskingIterations = false;
    // For hdim64, it's faster to shuffle the LSE and dPsum values among the 8 threads that share the same rows.
    static constexpr bool ShuffleLSE = SdP_swapAB && kHeadDim <= 64;
    static constexpr bool ShuffledPsum = SdP_swapAB && kHeadDim <= 64;
    // If we have extra registers, we can keep V in registers to reduce smem traffic
    static constexpr bool dQacc_use_TMA = kHeadDim < 256;
    // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x 128 on 2 WGs) so that we can
    // do atomic add on one half before doing the other half of the MMA, to reduce register pressure.
    static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && !dQacc_use_TMA && dQ_swapAB && AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
    static_assert(!(Deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");

    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    // Without this SmemAlignment, with hdim 256 we get "misaligned address" error in TMA
    static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
    static constexpr size_t SmemAlignmentV = !Mma_dP_is_RS ? SmemAlignmentQKVdO : cutlass::detail::alignment_for_swizzle(SmemLayoutV{});
    static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");

    // TODO: do we have to worry that smem_dk and smem_dv in the epilogue don't line up w smem_k and smem_v due to alignment?
    using SmemdQacc_t = std::conditional_t<!dQacc_use_TMA, cute::array<ElementAccum, 0>, cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccum>>>;
    using SmemP_t = std::conditional_t<Mma_dKV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP>>;
    struct TensorStorage : cute::aligned_struct<cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
        SmemdQacc_t smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
        SmemP_t smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    };

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* const ptr_V;
        StrideQKV const stride_V;
        Element const* const ptr_dO;
        StrideQKV const stride_dO;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        float const softcap_val;
        int const block_size;
        int* const dq_semaphore;
        int const* const cu_seqlens = nullptr;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum stride_dQaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_QdO tma_load_Q, tma_load_dO;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale, softmax_scale_log2;
        float const softcap_val;
        cutlass::FastDivmod bs_divmod;
        int* const dq_semaphore;
        int const* const cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_QdO tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mQ,
            SmemLayoutQ{}(_, _, _0{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_Q, args.stride_dO);
        TMA_QdO tma_load_dO = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mdO,
            SmemLayoutdO{}(_, _, _0{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for KV
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_K, args.stride_V);
        TMA_V tma_load_V = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for KV
        if constexpr (Deterministic) { assert(args.dq_semaphore != nullptr); }
        assert(args.block_size >= 1);
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        // In the backward, we need to multiply by
        // (1 - tanh^2) * softmax_scale / softcap_val * softcap_val = (1 - tanh^2) * softmax_scale.
        // Instead we multiply by (1 - tanh^2) and multiply dK and dV by params.softmax_scale
        // (the original softmax_scale) at the end.
        return {args.shape_Q, args.shape_K,
                args.ptr_dQaccum, args.shape_dQaccum, args.stride_dQaccum,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_dO, tma_load_K, tma_load_V,
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                cutlass::FastDivmod(args.block_size),
                args.dq_semaphore,
                args.cu_seqlens};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    static SeqlenInfo_t get_seqlen_info(Params const& params, int const bidb) {
        return SeqlenInfo_t{bidb, get<0>(params.shape_Q), params.cu_seqlens};
    }

    // Deterministic backward: rank of the key tile n_block (of half k_noisy) among the key tiles that accumulate
    // into the query tile m_block (of half q_noisy), in the order in which DiffusionBwdScheduler launches them
    // (SPT: noisy key tiles first in increasing order, then clean key tiles in decreasing order).
    CUTLASS_DEVICE
    static int dq_rank(DiffusionGeometry const& g, bool const q_noisy, bool const k_noisy, int const n_block, int const m_block) {
        // Local copies: binding static constexpr members to the references of ceil_div would ODR-use them
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const i0 = m_block * kBlockM;
        int const i_last = cute::min(i0 + kBlockM, g.n) - 1;
        int const blk_first = g.block_start(i0), blk_last = g.block_start(i_last);
        if (!q_noisy) {
            int const num_clean = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockN);
            return num_clean - 1 - n_block;
        } else {
            int const num_clean = cute::ceil_div(blk_last, kBlockN);
            int const noisy_lo = blk_first / kBlockN;
            int const num_noisy = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockN) - noisy_lo;
            return k_noisy ? n_block - noisy_lo : num_noisy + num_clean - 1 - n_block;
        }
    }

    // block_coord: (n_block, bidh, bidb, half of the key tile: 0 noisy, 1 clean)
    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipeline pipeline_q,
         MainloopPipeline_dO pipeline_do,
         PipelineState& smem_pipe_write,
         PipelineState_dO& smem_pipe_write_do,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
         ) {

        auto [n_block, bidh, bidb, k_half] = block_coord;
        bool const k_noisy = k_half == 0;
        SeqlenInfo_t seqlen_info = get_seqlen_info(params, bidb);
        DiffusionGeometry const geom(params.bs_divmod, seqlen_info.n);
        Tiles_t const tiles(geom, k_noisy, n_block);
        int const m_block_max = tiles.m_block_max();
        // It's possible to have m_block_max <= 0. Loading Q, K can cause illegal memory access.
        if (m_block_max <= 0) {
            scheduler_prefetch();
            return;
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        bool const is_varlen = Varlen && params.cu_seqlens;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen ? bidb : 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen ? bidb : 0);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh);

        // Query tiles of the noisy and the clean half
        Tensor gQ_noisy = local_tile(domain_offset(make_coord(seqlen_info.offset_half(0), _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gQ_clean = local_tile(domain_offset(make_coord(seqlen_info.offset_half(1), _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO_noisy = local_tile(domain_offset(make_coord(seqlen_info.offset_half(0), _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO_clean = local_tile(domain_offset(make_coord(seqlen_info.offset_half(1), _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_half(k_half), _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_half(k_half), _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gLSE_noisy = local_tile(domain_offset(make_coord(seqlen_info.template offset_padded<kBlockM>(bidb, 0)), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gLSE_clean = local_tile(domain_offset(make_coord(seqlen_info.template offset_padded<kBlockM>(bidb, 1)), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum_noisy = local_tile(domain_offset(make_coord(seqlen_info.template offset_padded<kBlockM>(bidb, 0)), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum_clean = local_tile(domain_offset(make_coord(seqlen_info.template offset_padded<kBlockM>(bidb, 1)), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        auto block_tma_dO = params.tma_load_dO.get_slice(_0{});
        Tensor tQgQ_noisy = group_modes<0, 3>(block_tma_Q.partition_S(gQ_noisy));
        Tensor tQgQ_clean = group_modes<0, 3>(block_tma_Q.partition_S(gQ_clean));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        Tensor tdOgdO_noisy = group_modes<0, 3>(block_tma_dO.partition_S(gdO_noisy));
        Tensor tdOgdO_clean = group_modes<0, 3>(block_tma_dO.partition_S(gdO_clean));
        Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};

        auto load_Q_LSE = [&](int const u, auto const& smem_pipe_write) {
            pipeline_q.producer_acquire(smem_pipe_write);
            int const m = tiles.tile_idx(u);
            auto tma_copy = params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), 0 /*mcast_mask*/, TMA::CacheHintSm90::EVICT_LAST);
            auto lse_copy = bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write));
            if (tiles.q_is_noisy(u, k_noisy)) {
                copy(tma_copy, tQgQ_noisy(_, m), tQsQ(_, smem_pipe_write.index()));
                copy(lse_copy, gLSE_noisy(_, m), sLSE(_, smem_pipe_write.index()));
            } else {
                copy(tma_copy, tQgQ_clean(_, m), tQsQ(_, smem_pipe_write.index()));
                copy(lse_copy, gLSE_clean(_, m), sLSE(_, smem_pipe_write.index()));
            }
        };

        auto load_dO_dPsum = [&](int const u, auto const& smem_pipe_write_do_cur) {
            pipeline_do.producer_acquire(smem_pipe_write_do_cur);
            int const m = tiles.tile_idx(u);
            auto tma_copy = params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), 0 /*mcast_mask*/, TMA::CacheHintSm90::EVICT_LAST);
            auto dpsum_copy = bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur));
            if (tiles.q_is_noisy(u, k_noisy)) {
                copy(tma_copy, tdOgdO_noisy(_, m), tdOsdO(_, smem_pipe_write_do_cur.index()));
                copy(dpsum_copy, gdPsum_noisy(_, m), sdPsum(_, smem_pipe_write_do_cur.index()));
            } else {
                copy(tma_copy, tdOgdO_clean(_, m), tdOsdO(_, smem_pipe_write_do_cur.index()));
                copy(dpsum_copy, gdPsum_clean(_, m), sdPsum(_, smem_pipe_write_do_cur.index()));
            }
        };

        int u = 0;

        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            load_Q_LSE(u, smem_pipe_write);
        }

        if (lane_predicate) {
            // Copy K tile and V tile from GMEM to SMEM.
            shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);

            #pragma unroll (kHeadDim < 256 ? 2 : 1)
            for (; u < m_block_max - 1; ++u) {
                // If Q and dO have the same number of stages, we can use the same pipeline state variable
                // to reduce registers
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                load_dO_dPsum(u, smem_pipe_write_do_cur);
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                ++smem_pipe_write;
                load_Q_LSE(u + 1, smem_pipe_write);
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
            load_dO_dPsum(u, smem_pipe_write_do_cur);
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            ++smem_pipe_write;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write) {
        static_assert(Q_dO_same_stages, "Q and dO must have the same number of stages");
        // Need to copy since pipeline_q.producer_tail(smem_pipe_write) will increment smem_pipe_write
        PipelineState smem_pipe_write_do = smem_pipe_write;
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do) {
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_dq(Params const& params,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
             ) {
        if constexpr (!dQacc_use_TMA) { return; }

        auto [n_block, bidh, bidb, k_half] = block_coord;
        bool const k_noisy = k_half == 0;
        SeqlenInfo_t seqlen_info = get_seqlen_info(params, bidb);
        DiffusionGeometry const geom(params.bs_divmod, seqlen_info.n);
        Tiles_t const tiles(geom, k_noisy, n_block);
        int const m_block_max = tiles.m_block_max();
        // It's possible to have m_block_max <= 0. Exit early
        if (m_block_max <= 0) { return; }

        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        static constexpr int dQ_TMA_num_bytes = CUTE_STATIC_V(size<0>(sdQ)) * sizeof(ElementAccum);

        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh);
        int const offset_padded_noisy = seqlen_info.template offset_padded<kBlockM>(bidb, 0);
        int const offset_padded_clean = seqlen_info.template offset_padded<kBlockM>(bidb, 1);
        Tensor gdQaccum_noisy_ = local_tile(domain_offset(make_coord(offset_padded_noisy * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum_clean_ = local_tile(domain_offset(make_coord(offset_padded_clean * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum_noisy = cute::flat_divide(gdQaccum_noisy_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)
        Tensor gdQaccum_clean = cute::flat_divide(gdQaccum_clean_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)

        int const num_head = get<2>(params.shape_Q);
        int *lock_ptr = !Deterministic ? nullptr : params.dq_semaphore + bidh;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
        bool const lane_predicate = cute::elect_one_sync();
        #pragma unroll 2
        for (int u = 0; u < m_block_max; ++u) {
            bool const q_noisy = tiles.q_is_noisy(u, k_noisy);
            int const m_block = tiles.tile_idx(u);
            int const tile_id = (q_noisy ? offset_padded_noisy : offset_padded_clean) / kBlockM + m_block;
            if constexpr (Deterministic) {
                Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, tile_id * num_head,
                                 dq_rank(geom, q_noisy, k_noisy, n_block, m_block));
            }
            #pragma unroll
            for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) + warpgroup_idx /*id*/);  // sdQ full, to be written to gmem
                if (lane_predicate) {
                    float* gdQ_ptr = q_noisy ? raw_pointer_cast(gdQaccum_noisy(_, warpgroup_idx, m_block).data())
                                             : raw_pointer_cast(gdQaccum_clean(_, warpgroup_idx, m_block).data());
                    SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdQ(_, warpgroup_idx).data()), gdQ_ptr, dQ_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                    tma_store_arrive();
                }
            }
            // Note, the for_each() function is required here to ensure `warpgroup_idx` is of type Int<x>.
            for_each(make_int_sequence<NumMmaWarpGroups>{}, [&] (auto warpgroup_idx) {
                if (lane_predicate) { tma_store_wait<NumMmaWarpGroups - 1 - CUTE_STATIC_V(warpgroup_idx)>(); }
                cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) + warpgroup_idx /*id*/);  // sdQ empty, ready to be written to
            });
            if constexpr (Deterministic) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, tile_id * num_head);
            }
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if constexpr (dQacc_use_TMA) {
            if (warp_idx_in_warpgroup == 0) {
                cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);  // sdQ empty, ready to be written to
            }
        }
    }

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipeline pipeline_q,
        MainloopPipeline_dO pipeline_do,
        PipelineState& smem_pipe_read,
        PipelineState_dO& smem_pipe_read_do,
        FrgTensordKV& tdKrdK,
        FrgTensordKV& tdVrdV,
        int thread_idx,
        int &work_idx,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

        int n_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        bool const k_noisy = get<3>(block_coord) == 0;
        SeqlenInfo_t seqlen_info = get_seqlen_info(params, bidb);
        DiffusionGeometry const geom(params.bs_divmod, seqlen_info.n);
        Tiles_t const tiles(geom, k_noisy, n_block);
        int const m_block_max = tiles.m_block_max();
        // It's possible to have m_block_max <= 0. Exit early
        if (m_block_max <= 0) { return false; }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKt{});
        Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdS{});
        Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
        Tensor sPt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdSt{});
        Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});
        Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdSt{});
        Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        static_assert(stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
                      stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
                      size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                      size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        Layout warp_group_thread_layout_dq = make_layout(make_shape(Int<NumMmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaSdP tiled_mma_SdP;
        using TiledMmadP = std::conditional_t<!Mma_dP_is_RS, TiledMmaSdP, TiledMmadPRS>;
        TiledMmadP tiled_mma_dP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ tiled_mma_dQ;

        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout_dq(warp_group_idx));

        auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
        auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);

        R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);

        // Allocate "fragments/descriptors"
        // We have to use the templated mma_partition_fragment_AB instead of cute::conditional_return or lambda,
        // because some partition_fragment_A/B don't compile.
        // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
        Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
        Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
        Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
        Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
        Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
        Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
        Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

        Tensor tPsP = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_pi, sPt_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // thread_mma_SdP.partition_C(sLSEMma) has shape ((2, 2, V), MMA_M, MMA_N, PIPE), we only take the col indices
        // or row indices, depending on whether SdP_swapAB.
        Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),  // (2, MMA_M, PIPE)
            group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _)));  // (2, V, MMA_N, PIPE)
        Tensor tLSEsdPsum = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),
            group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _, _)));
        // If we want to split the stats among the 8 threads that share the same rows.
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int bidh = get<1>(block_coord);

        // For the case where we do atomicAdd directly to gdQaccum instead of using TMA
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh);
        Tensor gdQaccum_noisy_ = local_tile(domain_offset(make_coord(seqlen_info.template offset_padded<kBlockM>(bidb, 0) * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum_clean_ = local_tile(domain_offset(make_coord(seqlen_info.template offset_padded<kBlockM>(bidb, 1) * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum_noisy = cute::flat_divide(gdQaccum_noisy_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)
        Tensor gdQaccum_clean = cute::flat_divide(gdQaccum_clean_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)
        // We can reuse r2s_thr_copy_dQaccum for this partitioning
        Tensor tdQgdQaccum_noisy = r2s_thr_copy_dQaccum.partition_D(gdQaccum_noisy);
        Tensor tdQgdQaccum_clean = r2s_thr_copy_dQaccum.partition_D(gdQaccum_clean);

        flash::DiffusionMask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask(thread_idx, geom);

        clear(tdKrdK);
        clear(tdVrdV);
        // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_KV.wait(work_idx % 2); }

        if constexpr (Mma_dP_is_RS) {
            using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
            auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
            auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
            Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
            Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sV));
            cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
        }

        auto bwd_step = [&](int u, auto mask_fn) {
            bool const q_noisy = tiles.q_is_noisy(u, k_noisy);
            int const m_block = tiles.tile_idx(u);
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            consumer_wait(pipeline_q, smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
            Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(make_fragment_like(tLSEsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffleLSE) {
                cute::copy(tLSEsLSE(_, smem_pipe_read.index()), tLSErLSE);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    // It's ok to read OOB, since we made sure sLSE is large enough and we won't use the OOB values
                    tLSErLSE(i) = tLSEsLSE((thread_idx % 32) / 4 + i * 8, smem_pipe_read.index());
                }
            }
            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
            warpgroup_wait<1>();
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }

            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            // dtanh needs to happen before masking, otherwise we get 1 - (-inf)^2 = NaN in the dtanh
            auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();
            mask_fn(tSrS, q_noisy, m_block);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float const lse_scaled = [&] {
                    if constexpr (!ShuffleLSE) return tLSErLSE(mi);
                    else return __shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    scores(mi, ni) = exp2f(scores(mi, ni) * params.softmax_scale_log2 - lse_scaled);
                }
            }

            Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(make_fragment_like(tLSEsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffledPsum) {
                cute::copy(tLSEsdPsum(_, smem_pipe_read_do_cur.index()), tLSErdPsum);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    tLSErdPsum(i) = tLSEsdPsum((thread_idx % 32) / 4 + i * 8, smem_pipe_read_do_cur.index());
                }
            }

            warpgroup_wait<0>();
            // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                float const dP_sum_cur = [&] {
                    if constexpr (!ShuffledPsum) return tLSErdPsum(mi);
                    else return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum_cur);
                    if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
                }
            }

            // Convert scores from fp32 to fp16/bf16
            Tensor rP = make_tensor_like<Element>(tSrS);
            flash::convert_type_out(tSrS, rP);
            if constexpr (!Mma_dKV_is_RS) {
                // Need to sync to make sure P has already been used in the previous iteration before writing new values
                if constexpr (kStages_dS == 1) {
                    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
                }
                Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);     // ((Atom,AtomNum), MMA_N, MMA_N)
                cute::copy(smem_tiled_copy_PdS, tPaP, tPsP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));
            }
            Tensor rdS = make_tensor_like<Element>(tdPrdP);
            flash::convert_type_out(tdPrdP, rdS);
            // If there's double buffering on dS, we don't need to sync here.
            // Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
            // But because both WGs have to sync at the end of the loop and double buffering,
            // this race condition is not possible.
            // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
            // (2) dS is already read by the Mma in the previous iteration in case of Mma_dKV_is_RS.
            if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
            }
            // For hdim 64, It's faster to write to smem_dS first before the dV gemm
            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));

            if constexpr (!Slice_dQKV_Mma) {
                // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
                if constexpr (Mma_dKV_is_RS) {
                    Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                } else {
                    Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
                    Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                }
                // SMEM fence to make sure sdS is written before it's read by WGMMA
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                if constexpr (Mma_dKV_is_RS) {
                    Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                } else {
                    Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
                    Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                }
                if constexpr (dQacc_use_TMA) {
                    int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
                    cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) + warp_group_idx /*id*/);  // sdQ full, to be written to gmem
                    Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
                    cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::arrive(cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) + warp_group_idx /*id*/);  // sdQ full, to be written to gmem
                } else {
                    // We can reuse r2s_thr_copy_dQaccum for this partitioning
                    Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
                    Tensor tdQgdQaccum_atomic = recast<float4>(q_noisy ? tdQgdQaccum_noisy(_, _, _, m_block) : tdQgdQaccum_clean(_, _, _, m_block));
                    static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQaccum_atomic)));
                    #pragma unroll
                    for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
                }

            } else {  // Slice_dQKV_Mma

                static_assert(!(Slice_dQKV_Mma && Mma_dKV_is_RS));
                Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
                Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);

                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
                Tensor tdQgdQaccum_atomic = recast<float4>(q_noisy ? tdQgdQaccum_noisy(_, _, _, m_block) : tdQgdQaccum_clean(_, _, _, m_block));
                #pragma unroll
                for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
                Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                #pragma unroll
                for (int i = size(tdQrdQ_atomic) / 2;  i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
            }

            warpgroup_wait<0>();
            pipeline_q.consumer_release(smem_pipe_read);   // release Q
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
        };

        auto mask_fn = [&](auto& tSrS, bool const q_noisy, int const m_block) {
            mask.apply(tSrS, q_noisy, m_block, k_noisy, n_block);
        };
        auto no_mask_fn = [](auto& tSrS, bool const q_noisy, int const m_block) { };
        int u = 0;
        if constexpr (SeparateMaskingIterations) {
            // Query tiles crossing an interval boundary first (masking), then the fully visible ones
            int const n_mask = tiles.n_mask();
            CUTLASS_PRAGMA_NO_UNROLL
            for (; u < n_mask; ++u) { bwd_step(u, mask_fn); }
            CUTLASS_PRAGMA_NO_UNROLL
            for (; u < m_block_max; ++u) { bwd_step(u, no_mask_fn); }
        } else {
            int const n_mask = tiles.n_mask();
            auto mask_cond_fn = [&](auto& tSrS, bool const q_noisy, int const m_block) {
                if (u < n_mask) { mask.apply(tSrS, q_noisy, m_block, k_noisy, n_block); }
            };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; u < m_block_max; ++u) { bwd_step(u, mask_cond_fn); }
        }

        // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(tdVrdV); }
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= params.softmax_scale; }

        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
        ++work_idx;
        return true;
    }

};

} // namespace flash
