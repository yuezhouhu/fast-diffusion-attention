/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "diffusion.h"
#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

// Writes dK and dV of a key tile of one half of a sample. Full tiles are written with TMA stores. The rows of the
// last tile of a half beyond the end of the half belong to the other half (or to the next sample), so that tile is
// written with predicated stores.
template <class TileShape_MNK_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool dKV_swapAB_, int AtomLayoutKdKV=1>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    using SeqlenInfo_t = flash::DiffusionSeqlenInfo<Varlen>;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using GmemTiledCopydKVTMA = cute::SM90_TMA_STORE;
    // Smem layout suitable for TMA stores (and for the predicated stores of the last tile of a half)
    using SmemLayoutAtomdKV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       decltype(cute::get<1>(TileShape_MNK{})), Int<CUTE_STATIC_V(cute::get<2>(TileShape_MNK{})) / AtomLayoutKdKV>>());
    using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVt =
        decltype(cute::composition(SmemLayoutdKV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;

    static constexpr size_t SmemAlignmentdKV = cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{});
    static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

    struct TensorStorage : cute::aligned_struct<SmemAlignmentdKV> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dk;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_k, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

    using TMA_dKV = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdKV{},
        select<1, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for dKV

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        StridedKV const stride_dV;
        int const num_heads_q;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        StridedKV const stride_dV;
        TMA_dKV tma_store_dK, tma_store_dV;
        int const* cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
        Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dK, args.stride_dV);
        TMA_dKV tma_store_dK = make_tma_copy(GmemTiledCopydKVTMA{}, mdK, SmemLayoutdKV{}, select<1, 2>(TileShape_MNK{}), _1{}); // no mcast for dKV
        TMA_dKV tma_store_dV = make_tma_copy(GmemTiledCopydKVTMA{}, mdV, SmemLayoutdKV{}, select<1, 2>(TileShape_MNK{}), _1{}); // no mcast for dKV
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.cu_seqlens};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
    }

    template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO const& tdKrdK,
          FrgTensorO const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb, k_half] = block_coord;
        Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKV{}));
        Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKV{}));
        Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKVt{}));
        Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKVt{}));
        auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);

        Tensor tdVrdV_out = make_tensor_like<Element>(tdVrdV);
        flash::convert_type_out(tdVrdV, tdVrdV_out);
        Tensor tdKrdK_out = make_tensor_like<Element>(tdKrdK);
        flash::convert_type_out(tdKrdK, tdKrdK_out);
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);

        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        SeqlenInfo_t seqlen_info{bidb, size<0>(params.shape_dK), params.cu_seqlens};
        bool const is_varlen = Varlen && params.cu_seqlens;
        int const offset = seqlen_info.offset_half(k_half);
        int const seqlen = seqlen_info.n;
        if ((n_block + 1) * kBlockN <= seqlen) {  // Full tile: TMA store
            cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dK)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdK = local_tile(domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor gdV = local_tile(domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
            auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
            Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
            Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
            Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (cute::elect_one_sync()) {
                    cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                    cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                    tma_store_arrive();
                }
            }
            tma_store_wait<0>();
            return;
        }
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dK, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
        Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKVsdV = gmem_thr_copy_dKV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
        Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdKVsdK = gmem_thr_copy_dKV.partition_S(sdK); // (TMA, TMA_M, TMA_K)
        Tensor tdKVrdV = make_fragment_like(tdKVgdV);
        Tensor tdKVrdK = make_fragment_like(tdKVgdK);
        Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        // Need to check OOB when reading from smem if kBlockN isn't evenly tiled
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(size<0>(GmemLayoutAtom{})) == 0;
        flash::copy</*Is_even_MN=*/EvenN, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
            gmem_tiled_copy_dKV, tdKVsdV, tdKVrdV, tdKVcdKV, tdKVpdKV, kBlockN);
        flash::copy</*Is_even_MN=*/EvenN, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false>(
            gmem_tiled_copy_dKV, tdKVsdK, tdKVrdK, tdKVcdKV, tdKVpdKV, kBlockN);
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdV, tdKVgdV, tdKVcdKV, tdKVpdKV, std::min(seqlen - n_block * kBlockN, kBlockN)
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdK, tdKVgdK, tdKVcdKV, tdKVpdKV, std::min(seqlen - n_block * kBlockN, kBlockN)
        );
    }

    CUTLASS_DEVICE void
    store_tail() {
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        auto [n_block, bidh, bidb, k_half] = block_coord;
        SeqlenInfo_t seqlen_info{bidb, size<0>(params.shape_dK), params.cu_seqlens};
        bool const is_varlen = Varlen && params.cu_seqlens;
        int const offset = seqlen_info.offset_half(k_half);
        int const seqlen = seqlen_info.n;
        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dK, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
        Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKVrdKV = make_fragment_like(tdKVgdK);
        clear(tdKVrdKV);
        // Construct identity layout for gdKV
        Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdK, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdV, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
    }

};

// GQA: dK and dV of the query heads of a group are accumulated in fp32 in dKaccum / dVaccum (padded layout,
// every half of every sample starts at a multiple of kBlockN), and converted by a postprocessing kernel.
template <class TileShape_MNK_, class ElementAccum, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool Deterministic>
struct CollectiveEpilogueBwdGQA {

    using TileShape_MNK = TileShape_MNK_;
    using Element = ElementAccum;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    using SeqlenInfo_t = flash::DiffusionSeqlenInfo<Varlen>;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static_assert(NumEpilogueThreads % cutlass::NumThreadsPerWarp == 0, "NumEpilogueThreads must be a multiple of NumThreadsPerWarp");
    static constexpr int NumWarpGroups = NumEpilogueThreads / cutlass::NumThreadsPerWarpGroup;
    // Thread layout, 256 or 384 threads per row
    // We split into NumWarpGroups so that we can use the same postprocessing kernel as dQ
    using R2SLayoutAtomdKVaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumWarpGroups>>>;
    using R2STiledCopydKVaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdKVaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    // For Sm80
    using SmemLayoutdKVaccum = Layout<Shape<Int<kBlockN * kHeadDim / NumWarpGroups>, Int<NumWarpGroups>>>;
    using SmemLayoutdKVaccumFlat = Layout<Shape<Int<kBlockN * kHeadDim>>>;

    static constexpr int SmemAlignment = kHeadDim % 64 == 0 ? 1024 : (kHeadDim % 32 == 0 ? 512 : 256);
    struct TensorStorage : cute::aligned_struct<SmemAlignment> {
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdKVaccum>, SmemAlignment> smem_dkv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t>;  // (total_k_padded * d, head)
    using StridedKV = cute::Stride<_1, int64_t>;

    // Host side kernel arguments
    struct Arguments {
        ElementAccum* ptr_dKaccum;
        ShapedKV const shape_dKaccum;
        StridedKV const stride_dKaccum;
        ElementAccum* ptr_dVaccum;
        StridedKV const stride_dVaccum;
        int const num_heads_q;
        int const seqlen;  // length of a sample if not varlen (2n)
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens;
    };

    // Device side kernel params
    struct Params {
        ElementAccum* ptr_dKaccum;
        ShapedKV const shape_dKaccum;
        StridedKV const stride_dKaccum;
        ElementAccum* ptr_dVaccum;
        StridedKV const stride_dVaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        int const seqlen;
        int* dk_semaphore;
        int* dv_semaphore;
        int const* cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Deterministic) {
            assert(args.dk_semaphore != nullptr);
            assert(args.dv_semaphore != nullptr);
        }
        return {args.ptr_dKaccum, args.shape_dKaccum, args.stride_dKaccum, args.ptr_dVaccum, args.stride_dVaccum,
                cutlass::FastDivmod(cute::ceil_div(args.num_heads_q, get<1>(args.shape_dKaccum))),
                args.seqlen, args.dk_semaphore, args.dv_semaphore, args.cu_seqlens};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
    }

    template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO const& tdKrdK,
          FrgTensorO const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb, k_half] = block_coord;
        int bidh_idx_in_group;
        int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);
        Tensor sdKV = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dkv.data()), SmemLayoutdKVaccum{});
        Tensor sdKV_flat = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dkv.data()), SmemLayoutdKVaccumFlat{});
        static constexpr int dKV_TMA_num_bytes = CUTE_STATIC_V(size(sdKV_flat)) * sizeof(ElementAccum);

        SeqlenInfo_t seqlen_info{bidb, params.seqlen, params.cu_seqlens};
        int const offset_padded = seqlen_info.template offset_padded<kBlockN>(bidb, k_half);
        Tensor mdKaccum = make_tensor(make_gmem_ptr(params.ptr_dKaccum), params.shape_dKaccum, params.stride_dKaccum)(_, bidh_kv);
        Tensor mdVaccum = make_tensor(make_gmem_ptr(params.ptr_dVaccum), params.shape_dKaccum, params.stride_dVaccum)(_, bidh_kv);
        Tensor gdKaccum = local_tile(domain_offset(make_coord(offset_padded * kHeadDim), mdKaccum), Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block));  // (M * K)
        Tensor gdVaccum = local_tile(domain_offset(make_coord(offset_padded * kHeadDim), mdVaccum), Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block));  // (M * K)

        R2STiledCopydKVaccum r2s_tiled_copy_dKVaccum;
        auto r2s_thr_copy_dKVaccum = r2s_tiled_copy_dKVaccum.get_thread_slice(thread_idx);
        Tensor tdKVsdKVaccum = r2s_thr_copy_dKVaccum.partition_D(sdKV);

        // Make sure all WGs have finished reading K and V, otherwise we get racy dQ
        // because smem_q could be changed.
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        Tensor taccdKVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV); // ((Atom,AtomNum), MMA_M, MMA_N)
        cute::copy(r2s_tiled_copy_dKVaccum, taccdKVrdV, tdKVsdKVaccum);

        int const num_head_kv = get<1>(params.shape_dKaccum);
        int const tile_id = offset_padded / kBlockN + n_block;
        int *lock_ptr = !Deterministic ? nullptr : params.dv_semaphore + bidh_kv;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;

        if constexpr (Deterministic) {
            Barrier::wait_eq(lock_ptr, thread_idx, tile_id * num_head_kv, bidh_idx_in_group);
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (thread_idx == 0) {
            SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdKV_flat.data()), raw_pointer_cast(gdVaccum.data()), dKV_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
            tma_store_arrive();
            tma_store_wait<0>();
        }
        if constexpr (Deterministic) {
            Barrier::arrive_inc(lock_ptr, thread_idx, tile_id * num_head_kv);
        }

        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        Tensor taccdKVrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK); // ((Atom,AtomNum), MMA_M, MMA_N)
        cute::copy(r2s_tiled_copy_dKVaccum, taccdKVrdK, tdKVsdKVaccum);
        lock_ptr = !Deterministic ? nullptr : params.dk_semaphore + bidh_kv;

        if constexpr (Deterministic) {
            Barrier::wait_eq(lock_ptr, thread_idx, tile_id * num_head_kv, bidh_idx_in_group);
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (thread_idx == 0) {
            SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdKV_flat.data()), raw_pointer_cast(gdKaccum.data()), dKV_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
            tma_store_arrive();
            tma_store_wait<0>();
        }
        if constexpr (Deterministic) {
            Barrier::arrive_inc(lock_ptr, thread_idx, tile_id * num_head_kv);
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
         ) {
        // Don't need to do anything since dKaccum and dVaccum are already zero-initialized
        // (the semaphores of the deterministic mode are not needed either: a key tile without query tile is
        // empty for all the query heads of a group).
    }

};

} // namespace flash
