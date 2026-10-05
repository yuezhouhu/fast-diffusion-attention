/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Forward mainloop of the diffusion attention (adapted from the FlashAttention-3 Hopper mainloop):
 * TMA loads, warp-specialized producer / consumers, WGMMA, intra-warpgroup overlap of softmax and GEMMs.
 * A query tile of one half of a sample visits the key tiles of DiffusionFwdTiles: the clean key tiles and, for
 * noisy queries, the noisy key tiles of the same diffusion blocks.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "diffusion.h"
#include "utils.h"
#include "sm90_pipeline_no_cluster.hpp"

namespace flash {

using namespace cute;

template <int Stages, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Has_softcap_, bool Varlen_, bool MmaPV_is_RS, bool IntraWGOverlap>
struct CollectiveMainloopFwdSm90 {

    static constexpr int kStages = Stages;
    using ClusterShape = Shape<_1, _1, _1>;
    using TileShape_MNK = TileShape_MNK_;
    static constexpr int kHeadDimV = get<2>(TileShape_MNK{});
    using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    // V is row-major in gmem. FP8 WGMMA needs it K-major in smem, so the producer warpgroup transposes it.
    static constexpr bool Transpose_V = Is_FP8;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr cute::GMMA::Major TmaMajorV = GMMA::Major::MN;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using SeqlenInfo_t = flash::DiffusionSeqlenInfo<Varlen>;
    using Tiles_t = flash::DiffusionFwdTiles<kBlockM, kBlockN>;

    // Register bandwidth is actually a bottleneck so we don't want Q to be in registers.
    static constexpr bool MmaQK_is_RS = false;
    // We can have MmaPV with P in smem in rmem to reduce register pressure at the cost of more smem.
    static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8");

    using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutQK{}));
    using AtomLayoutPV = AtomLayoutQK;
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !MmaPV_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())
        >{},
        AtomLayoutPV{}));

    static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
    static constexpr int NumMmaThreads = size(TiledMmaPV{});
    static constexpr int NumProducerThreads = !Transpose_V ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaThreadsQK % cutlass::NumThreadsPerWarpGroup == 0);
    static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
    static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
                                      Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(Int<kHeadDimV>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    using SmemLayoutAtomVtMma = decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, Element,
                                         Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
    using SmemLayoutVtMma = decltype(tile_to_shape(
        SmemLayoutAtomVtMma{},
        make_shape(Int<kHeadDimV>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
        std::conditional_t<MmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));

    using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

    // Use LDSM.T and STSM to transpose V in the case of FP8 and V being row-major.
    // For FP16/BF16 we don't do any transposing.
    static_assert(!Transpose_V || (kHeadDimV % 32 == 0 && kBlockN % 32 == 0));
    static constexpr bool kHeadDimV_multiple_64 = kHeadDimV % 64 == 0;
    // Either kHeadDimV is a multiple of 64 (in which case we use a block size of 64 x 32 for the transpose),
    // or we need kBlockN to be a multiple of 64 (in which case we use a block size of 32 x 64 for the transpose).
    static_assert(!Transpose_V || (kHeadDimV_multiple_64 || kBlockN % 64 == 0));
    using LDSM_thread_shape  = std::conditional_t<kHeadDimV_multiple_64, Shape<_32, _4, _1, _1>, Shape<_16, _4, _1, _2>>;
    using LDSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _0, _0>, Stride<_4, _1, _0, _64>>;
    using LDSM_value_shape = Shape<_2, _2, _1, _4>;
    using LDSM_value_stride = Stride<_1, _2, _16, _4>;
    using LDSM_divide_shape = std::conditional_t<kHeadDimV_multiple_64, Shape<_64, _8>, Shape<_32, _8>>;
    using S2RTiledCopyVt = decltype(make_tiled_copy(
        Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<LDSM_thread_shape, LDSM_thread_stride>{},
        Layout<LDSM_value_shape, LDSM_value_stride>{}));

    using STSM_thread_shape  = std::conditional_t<kHeadDimV_multiple_64, Shape<_8, _4, _4, _1>, Shape<_8, _4, _2, _2>>;
    using STSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _32, _0>, Stride<_4, _1, _32, _64>>;
    using STSM_value_shape = Shape<_1, _4, _2, _2>;
    using STSM_value_stride = Stride<_0, _1, _4, _8>;
    using STSM_divide_shape = Shape<_8, _16>;
    // These will not permute the columns of V (the kHeadDimV dimension) but incur bank conflicts
    // so a little slower (e.g. 1150 TFLOPS for hdim 256 instead of 1200 TFLOPS).
    // Instead we will permute the cols of V, and un-permute the cols of O in the epilogue.
    using R2STiledCopyV = decltype(make_tiled_copy(
        Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<STSM_thread_shape, STSM_thread_stride>{},
        Layout<STSM_value_shape, STSM_value_stride>{}));

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = StrideQK;
    using StrideDescale = cute::Stride<int64_t, int64_t>;

    using TMA_Q = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        SmemLayoutQ{},
        TileShape_MNK{},
        ClusterShape{}));

    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        take<0, 2>(SmemLayoutK{}),
        TileShape_MNK{},
        ClusterShape{}));

    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})),
        take<0, 2>(SmemLayoutVt{}),
        select<1, 2>(TileShape_MNK_PV{}),
        size<0>(ClusterShape{})));

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutVt{})) * cutlass::sizeof_bits_v<Element> / 8);
    static_assert(TmaTransactionBytesK == TmaTransactionBytesV);

    using PipelineTmaAsync = typename cutlass::PipelineTmaAsyncNoCluster<kStages>;
    using MainloopPipelineK = PipelineTmaAsync;
    using MainloopPipelineV = std::conditional_t<!Transpose_V, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineVt = PipelineTmaAsync;
    using PipelineState = cutlass::PipelineState<kStages>;

    static constexpr size_t SmemAlignmentQ = 128;
    static constexpr size_t SmemAlignmentK = 128;
    static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");
    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
    static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");

    using SmemP_t = std::conditional_t<MmaPV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
    // Sometimes even with SmemP_t = cute::array<Element, 0>, putting it in the TensorStorage struct causes
    // smem size to go from 227KB to 228KB and we get "invalid argument".

    struct TensorStorageWithoutPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    };

    struct TensorStorageWithPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemP_t smem_p;
    };

    using TensorStorageNoTranspose = std::conditional_t<MmaPV_is_RS, TensorStorageWithoutPNoTranspose, TensorStorageWithPNoTranspose>;

    static constexpr size_t SmemAlignmentVt = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static constexpr size_t SmemAlignmentV = cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma{});
    static_assert(SmemAlignmentVt >= 128 and SmemAlignmentV >= 128, "Require at least 128B alignment");
    struct TensorStorageTransposeV : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentV), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVtMma>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt> smem_vt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    };

    using TensorStorage = std::conditional_t<!Transpose_V, TensorStorageNoTranspose, TensorStorageTransposeV>;

    // These are tuned for speed. They don't affect correctness.
    static constexpr bool UseSchedulerBarrier = (IntraWGOverlap
        ? (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : kHeadDim >= 128)
        : NumMmaWarpGroups == 2);
    static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && IntraWGOverlap;

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element const* const ptr_V;
        StrideV const stride_V;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        float const softcap_val;
        int const block_size;
        int const* const cu_seqlens = nullptr;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        float const softmax_scale_log2;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        float const softcap_val;
        cutlass::FastDivmod bs_divmod;
        int const* const cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQ{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            take<0, 2>(SmemLayoutK{}),
            TileShape_MNK{},
            ClusterShape{});
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V),
                                make_shape(get<1>(args.shape_K), get<0>(args.shape_K), get<2>(args.shape_K), get<3>(args.shape_K)),
                                select<1, 0, 2, 3>(args.stride_V));
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            take<0, 2>(SmemLayoutVt{}),
            select<1, 2>(TileShape_MNK_PV{}),
            size<0>(ClusterShape{}));
        assert(args.block_size >= 1);
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        return {args.shape_Q, args.shape_K,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_K, tma_load_V,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                cutlass::FastDivmod(args.block_size),
                args.cu_seqlens};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }

    // block_coord: (m_block, bidh, bidb, half), half = 0 for the noisy half, 1 for the clean half.
    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         MainloopPipelineVt pipeline_vt,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int &work_idx
         ) {

        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        bool const q_noisy = get<3>(block_coord) == 0;
        DiffusionGeometry const geom(params.bs_divmod, seqlen_info.n);
        Tiles_t const tiles(geom, q_noisy, m_block);
        int const n_block_max = tiles.n_block_max();
        // It's possible to have n_block_max <= 0 (empty tile). Loading K can cause illegal memory access.
        if (n_block_max <= 0) {
            scheduler_prefetch();
            return;
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        // as_position_independent_swizzle_tensor makes address calculation easier when we do LDSM & STSM to transpose.
        // But it requires smem_vt and smem_v to be aligned to e.g 512 bytes.
        Tensor sVt = [&] {
            if constexpr (!Transpose_V) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{}));
            }
        }();
        // Only used if Transpose_V
        Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{}));

        int const thread_idx = threadIdx.x % NumProducerThreads;
        int const bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        bool const is_varlen = Varlen && params.cu_seqlens;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor mK_TMA = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen ? bidb : 0);
        auto shape_V = make_shape(get<1>(params.shape_K), get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
        Tensor mVt_TMA = params.tma_load_V.get_tma_tensor(shape_V)(_, _, bidh_kv, !is_varlen ? bidb : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_half(q_noisy ? 0 : 1), _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        // Key tiles of the noisy and of the clean half
        Tensor gK_noisy = local_tile(domain_offset(make_coord(seqlen_info.offset_half(0), _0{}), mK_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gK_clean = local_tile(domain_offset(make_coord(seqlen_info.offset_half(1), _0{}), mK_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVt_noisy = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_half(0)), mVt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _));  // (K, N, _)
        Tensor gVt_clean = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_half(1)), mVt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _));  // (K, N, _)

        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));  // (TMA)
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));  // (TMA)
        if (thread_idx == 0) { prefetch(params.tma_load_Q, tQgQ); }
        auto block_tma_K = params.tma_load_K.get_slice(_0{});
        Tensor tKgK_noisy = group_modes<0, 3>(block_tma_K.partition_S(gK_noisy));  // (TMA, k)
        Tensor tKgK_clean = group_modes<0, 3>(block_tma_K.partition_S(gK_clean));  // (TMA, k)
        Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));  // (TMA, PIPE)
        auto block_tma_V = params.tma_load_V.get_slice(_0{});
        Tensor tVgVt_noisy = group_modes<0, 3>(block_tma_V.partition_S(gVt_noisy));  // (TMA, k)
        Tensor tVgVt_clean = group_modes<0, 3>(block_tma_V.partition_S(gVt_clean));  // (TMA, k)
        Tensor tVsVt = group_modes<0, 3>(block_tma_V.partition_D(sVt));  // (TMA, PIPE)

        // Set up for transposing V, only used if Transpose_V
        S2RTiledCopyVt s2r_tiled_copy_vt;
        R2STiledCopyV r2s_tiled_copy_v;
        auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
        auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
        // flat_divide(sVt, LDSM_divide_shape{}):  (64, 8, kHeadDim / 64, kBlockN / 8, kStages)
        Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));  // ((16, 1), 1, 1, kHeadDim / 64, kBlockN / 32, kStages)
        // flat_divide(sV, STSM_divide_shape{}):  (8, 16, kHeadDim / 8, (4, kBlockN / 64), kStages)
        Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));  // ((16, 1), 1, 1, kHeadDim / 64, (2, kBlockN / 64), kStages)
        CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
        // Faster to have 2 LDSM.T, byte permute, STSM for better ILP
        static constexpr int Transpose_ILP = (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
        Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_), Shape<Underscore, Int<Transpose_ILP>>{});  // ((16, 1), (2, kHeadDim / 64 * kBlockN / 32 / 2), kStages)
        Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_), Shape<Underscore, Int<Transpose_ILP>>{});  // ((16, 1), (2, kHeadDim / 64 * kBlockN / 32 / 2), kStages)
        auto transpose_V = [&](int stage) {
            if constexpr (Transpose_V) {
                #pragma unroll
                for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
                    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
                    static_assert(size<0>(tTransrV) == 16);
                    Tensor tTransrV_64 = recast<uint2>(tTransrV);
                    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage), tTransrV);
                    #pragma unroll
                    for (int j = 0; j < size(tTransrV_64); ++j) {
                        uint32_t upper = tTransrV_64[j].x;
                        uint32_t lower = tTransrV_64[j].y;
                        tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
                        tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
                    }
                    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
                }
            }
        };

        auto load_K = [&] (int const v, auto const& smem_pipe_write) {
            pipeline_k.producer_acquire(smem_pipe_write);
            int const idx = tiles.tile_idx(v);
            auto tma_copy = params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), 0 /*mcast_mask*/, TMA::CacheHintSm90::EVICT_LAST);
            if (tiles.is_noisy(v)) {
                copy(tma_copy, tKgK_noisy(_, idx), tKsK(_, smem_pipe_write.index()));
            } else {
                copy(tma_copy, tKgK_clean(_, idx), tKsK(_, smem_pipe_write.index()));
            }
        };

        auto load_V = [&] (int const v, auto const& smem_pipe_write) {
            auto pipeline_v_load = cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
            pipeline_v_load.producer_acquire(smem_pipe_write);
            int const idx = tiles.tile_idx(v);
            auto tma_copy = params.tma_load_V.with(*pipeline_v_load.producer_get_barrier(smem_pipe_write), 0 /*mcast_mask*/, TMA::CacheHintSm90::EVICT_LAST);
            if (tiles.is_noisy(v)) {
                copy(tma_copy, tVgVt_noisy(_, idx), tVsVt(_, smem_pipe_write.index()));
            } else {
                copy(tma_copy, tVgVt_clean(_, idx), tVsVt(_, smem_pipe_write.index()));
            }
        };

        auto copy_Vt_to_V = [&] (auto const& smem_pipe_write) {
            // Instead of maintaining smem_pipe_read as a separate variable, we can just use smem_pipe_write,
            // and exploit the invariance that smem_pipe_write.phase() == smem_pipe_read.phase() ^ 1.
            // This saves 1 or 2 registers.
            PipelineState smem_pipe_read{smem_pipe_write.index(), smem_pipe_write.phase() ^ 1, smem_pipe_write.count()};
            pipeline_vt.consumer_wait(smem_pipe_read);
            pipeline_v.producer_acquire(smem_pipe_write);
            transpose_V(smem_pipe_write.index());
            // SMEM fence to make sure V is transposed before math
            cutlass::arch::fence_view_async_shared();
            pipeline_v.producer_commit(smem_pipe_write);
            // Very important: PipelineTmaAsync::consumer_release assumes that the warpgroup is synchronized
            // before calling. Without this we get race conditions.
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, cutlass::arch::ReservedNamedBarriers::TransposeBarrier /*id*/);
            pipeline_vt.consumer_release(smem_pipe_read);
        };

        int v = n_block_max - 1;

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // If this is true, we're guaranteed that only the first warp will execute this function
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync();

        if (should_load_KV) {
            if constexpr (Transpose_V) { load_V(v, smem_pipe_write); }
            load_K(v, smem_pipe_write);
        }

        // Wait for the MMA warpgroups to signal that smem_q is ready
        if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        }

        if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
            shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(params.tma_load_Q.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q), 0 /*mcast_mask*/, TMA::CacheHintSm90::EVICT_FIRST),
                tQgQ, tQsQ);
        }

        // Wait for the MMA WGs to signal that smem_v are ready and V can be copied from gmem
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);

        if constexpr (!Transpose_V && !IntraWGOverlap) {
            if (should_load_KV) { load_V(v, smem_pipe_write); }
        }
        int v_prev = v;
        --v;
        #pragma unroll (!Transpose_V ? 2 : 1)
        for (; v >= 0; --v) {
            PipelineState smem_pipe_write_v = smem_pipe_write; // copy the state, write_v is always 1 step behind
            ++smem_pipe_write;
            if (should_load_KV) {
                if constexpr (Transpose_V) { load_V(v, smem_pipe_write); }
                load_K(v, smem_pipe_write);
                if constexpr (!Transpose_V) {
                    if constexpr (IntraWGOverlap) {
                        load_V(v_prev, smem_pipe_write_v);
                    } else {
                        load_V(v, smem_pipe_write);
                    }
                }
            }
            v_prev = v;
            if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write_v); }
        }
        scheduler_prefetch();
        if constexpr (!Transpose_V && IntraWGOverlap) {
            if (should_load_KV) { load_V(v_prev, smem_pipe_write); }
        }
        if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
        ++smem_pipe_write;
        // At the end, all threads have the correct smem_pipe_write.
        ++work_idx;
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, MainloopPipelineVt pipeline_vt,
              PipelineState& smem_pipe_write, SharedStorage &shared_storage, int const work_idx) {
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            *  Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            *  then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
            if constexpr (Transpose_V) { pipeline_vt.producer_tail(smem_pipe_write); }
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (UseSchedulerBarrier) {
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            int const cur_WG = flash::canonical_warp_group_idx_nosync() - 1;
            int const next_WG = NumMmaWarpGroups == 2
                ? 1 - cur_WG
                : (cur_WG < NumMmaWarpGroups - 1 ? cur_WG + 1 : 0);
            cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_WG /*id*/);
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        int warp_group_idx = flash::canonical_warp_group_idx_nosync();
        // Tell producers that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        if constexpr (UseSchedulerBarrier) {
            // We have NamedBarrier for up to 3 WGs
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            // WG1 needs the very first signal to start
            if (warp_group_idx == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v,
        PipelineState& smem_pipe_read,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int const thread_idx,
        int &work_idx,
        SeqlenInfo_t const& seqlen_info,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        // can't use auto [m_block, ...] = block_coord since structured binding cannot be captured in lambda
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        bool const q_noisy = get<3>(block_coord) == 0;
        int const bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
        DiffusionGeometry const geom(params.bs_divmod, seqlen_info.n);
        Tiles_t const tiles(geom, q_noisy, m_block);
        int const n_block_max = tiles.n_block_max();
        // It's possible to have n_block_max <= 0. We don't want to load Q or change any barrier
        if (n_block_max <= 0) { return false; }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
        Tensor sP = [&] {
            if constexpr (MmaPV_is_RS) {
                // We might not have smem_p if !MmaPV_is_RS, just use smem_q as a placeholder since we don't use it
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutP{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
            }
        }();

        static_assert(stride<0>(typename TiledMmaQK::ALayout{}) == 0 and
                    stride<0>(typename TiledMmaQK::BLayout{}) == 0 and
                    size<0>(typename TiledMmaQK::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                    size<0>(typename TiledMmaQK::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
            "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));

        auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
        auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors"
        Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
        Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
        Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
        Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
        Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int v = n_block_max - 1;

        flash::DiffusionMask<kBlockM, kBlockN, TiledMmaQK> mask(thread_idx, geom);
        auto mask_fn = [&](auto& tSrS, int const v) {
            mask.apply(tSrS, q_noisy, m_block, tiles.is_noisy(v), tiles.tile_idx(v));
        };

        float softcap_val = params.softcap_val;
        if constexpr (Has_softcap && Is_FP8) {
            float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
            float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
            softcap_val *= q_descale * k_descale;
        }
        // Softcapping needs to happen before masking since if we apply after masking, softcapping
        // can turn -inf to e.g. -50.0, which can affect the attention softmax.
        auto scoremod_premask_fn = [&](auto& tSrS) {
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
        };

        auto write_P_to_smem = [&](auto& tOrP) {
            cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
        };

        auto arrive_on_P_write_barrier = [&] {
            cutlass::arch::fence_view_async_shared();
            __syncwarp();  // Only need syncwarp since each warp is using its own P values for MmaPV
        };

        auto &barrier_Q = shared_storage.pipelines.barrier_Q;
        barrier_Q.wait(work_idx % 2);

        if constexpr (IntraWGOverlap) {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
            warpgroup_wait<0>();
            pipeline_k.consumer_release(smem_pipe_read);
            scoremod_premask_fn(tSrS);
            mask_fn(tSrS, v);

            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
            if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
            if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
            --v;

            // Need to initialize tOrO in the case of RescaleOBeforeGemm where we will scale tOrO even in the 1st iter
            clear(tOrO);

            // Each step does gemm0 for iter v, gemm1 for iter v + 1, and softmax for iter v.
            auto fwd_step = [&](int const v, auto mask_fn, auto check_inf_type) {
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                PipelineState smem_pipe_read_v(smem_pipe_read.index(), smem_pipe_read.phase(), smem_pipe_read.count());
                ++smem_pipe_read;
                Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
                if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_k, smem_pipe_read); }
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
                if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_v, smem_pipe_read_v); }
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
                warp_scheduler_barrier_arrive();
                warpgroup_wait<1>();
                pipeline_k.consumer_release(smem_pipe_read);  // release K
                scoremod_premask_fn(tSrS);
                mask_fn(tSrS, v);
                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);
                warpgroup_wait<0>();
                pipeline_v.consumer_release(smem_pipe_read_v);  // release V
                if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
                convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
                if constexpr (!RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
                if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
            };

            // Key tiles crossing an interval boundary: masking
            #pragma unroll 1
            for (; v >= tiles.n_block_mask_min; --v) {
                fwd_step(v, mask_fn, cute::true_type{} /*check_inf*/);
            }
            // Key tiles fully visible
            auto no_mask_fn = [](auto& tSrS, int v) { };
            #pragma unroll 1
            for (; v >= 0; --v) {
                fwd_step(v, no_mask_fn, cute::false_type{} /*check_inf*/);
            }
            // Tell producers that smem_q is ready
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
            if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
            consumer_wait(pipeline_v, smem_pipe_read);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
            float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
            cute::copy(softmax.finalize(v_descale), scores_scale);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang
            softmax.rescale_o(tOrO, scores_scale);
            if constexpr (Is_FP8) { flash::permute_output_fp8(tOrO); }
            ++smem_pipe_read;

        } else {  // No intra-WG overlap

            warp_scheduler_barrier_sync();

            auto fwd_step = [&](int const v, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
                static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                if constexpr (!Is_first_iter) { ++smem_pipe_read; }
                Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warp_scheduler_barrier_arrive();
                warpgroup_wait<0>();
                pipeline_k.consumer_release(smem_pipe_read);  // release K
                scoremod_premask_fn(tSrS);
                mask_fn(tSrS, v);
                Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
                softmax.template online_softmax</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
                if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
                Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
                Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
                convert_type_out(tOrP_acc, tOrP);
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
                if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
                if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
                consumer_wait(pipeline_v, smem_pipe_read);
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/Is_first_iter, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                warpgroup_wait<0>();
                pipeline_v.consumer_release(smem_pipe_read);  // release V
            };

            fwd_step(v, mask_fn, cute::true_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
            --v;
            #pragma unroll 1
            for (; v >= tiles.n_block_mask_min; --v) {
                fwd_step(v, mask_fn, cute::false_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
            }
            auto no_mask_fn = [](auto& tSrS, int v) { };
            #pragma unroll 1
            for (; v >= 0; --v) {
                fwd_step(v, no_mask_fn, cute::false_type{} /*is_first_iter*/, cute::false_type{} /*check_inf*/);
            }
            warp_scheduler_barrier_arrive();
            // Tell producers that smem_q is ready
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
            float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
            Tensor scores_scale = softmax.finalize(v_descale);
            softmax.rescale_o(tOrO, scores_scale);
            if constexpr (Is_FP8) { flash::permute_output_fp8(tOrO); }
            ++smem_pipe_read;
        }
        ++work_idx;
        return true;
    }

};

} // namespace flash
