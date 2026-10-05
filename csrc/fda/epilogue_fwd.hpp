/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>  // For FastDivMod
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"

#include "diffusion.h"
#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

// Writes the output tile of one half of a sample. The rows of a tile beyond the end of the half belong to the
// other half (or to the next sample), so the stores are predicated (no TMA store).
template <class TileShape_MNK_PV_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool FP8PermuteCol=false>
struct CollectiveEpilogueFwd {

    using TileShape_MNK_PV = TileShape_MNK_PV_;
    using ClusterShape = Shape<_1, _1, _1>;
    using Element = Element_;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    using SeqlenInfo_t = flash::DiffusionSeqlenInfo<Varlen>;

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(sizeof(Element) <= 2);

    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    static constexpr int kHeadDimV = get<1>(TileShape_MNK_PV{});

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimV % kGmemElemsPerStore == 0, "Headdim must be a multiple of kGmemElemsPerStore");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). We want each thread to have 4 elements
    // in the M direction and 2 elements in the K direction.
    static constexpr int kBytePerRow = kHeadDimV * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumEpilogueThreads / kGmemThreadsPerRow");
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;

    using EpilogueTile_MN = decltype(select<0, 1>(TileShape_MNK_PV{}));
    // cute::SM90_U32x4_STSM_N if Element size is 2 bytes (fp16, bf16)
    using CopyOpR2S = decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<StrideO, Element, EpilogueTile_MN>());
    using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
    };

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        int const* cu_seqlens = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE;
        int const* cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.ptr_O, args.shape_O, args.stride_O,
                args.ptr_LSE, select<0, 2, 3>(args.shape_O), args.stride_LSE,
                args.cu_seqlens};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) { }

    template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO& tOrO,
          FrgTensorLSE const& lse,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord,
          SeqlenInfo_t const& seqlen_info
          ) {

        auto [m_block, bidh, bidb, half] = block_coord;
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});

        static constexpr bool NeedFP8Permute = FP8PermuteCol && (sizeof(Element) == 2 || sizeof(Element) == 4);
        Tensor tOrO_out = make_tensor_like<Element>(tOrO);
        flash::convert_type_out(tOrO, tOrO_out);
        if constexpr (NeedFP8Permute) { flash::permute_output_fp8_Vcolmajor(tOrO_out); }

        // Make sure all WGs have finished reading V
        // Technically we don't need this if we're not using smem, but the mainloop makes the assumption that
        // all epilogue threads sync at least once during the epilogue (so that we can start loading Q with
        // cp.async if we need).
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        // Step 1: Write O from rmem -> smem
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        bool const is_varlen = Varlen && params.cu_seqlens;
        int const offset_o = seqlen_info.offset_half(half);
        int const seqlen_o = seqlen_info.n;

        // Step 2: Write LSE from rmem -> gmem
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        // (MMA,MMA_M,MMA_K)
        Tensor taccOcO = thread_mma.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
        static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M

        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE + offset_o * get<0>(params.stride_LSE)),
                                  params.shape_LSE, params.stride_LSE)(_, bidh, !is_varlen ? bidb : 0);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
            if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) { mLSE(row) = lse(mi); }
        }

        // Step 3: Write O from smem -> gmem
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O, params.stride_O)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));  // (M, K)
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOrO_gmem = make_fragment_like(tOsO);
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO_gmem);
        cutlass::arch::fence_view_async_shared(); // ensure smem reads are done before next TMA to smem_v
        shared_storage.pipelines.barrier_O.arrive();
        // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO_gmem, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
    }

    CUTLASS_DEVICE void
    store_tail() {
    }

    // Write 0 to output and -inf to LSE
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord,
         SeqlenInfo_t const& seqlen_info
         ) {
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
        auto [m_block, bidh, bidb, half] = block_coord;
        bool const is_varlen = Varlen && params.cu_seqlens;
        int const offset_o = seqlen_info.offset_half(half);
        int const seqlen_o = seqlen_info.n;
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE + offset_o * get<0>(params.stride_LSE)),
                                  params.shape_LSE, params.stride_LSE)(_, bidh, !is_varlen ? bidb : 0);

        static_assert(kBlockM <= NumEpilogueThreads);
        if (thread_idx < kBlockM) {
            const int row = m_block * kBlockM + thread_idx;
            if (row < seqlen_o) { mLSE(row) = -INFINITY; }
        }

        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O, params.stride_O)(_, _, bidh, !is_varlen ? bidb : 0);
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
        Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOgO);
        cute::clear(tOrO);
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
    }

};

} // namespace flash
