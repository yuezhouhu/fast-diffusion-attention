/******************************************************************************
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Backward pass of the noisy-noisy part of the diffusion attention (the block-diagonal part of the mask), for block
 * sizes dividing 64.
 *
 * A noisy key is only seen by the noisy queries of its own diffusion block. When the block size divides 64, the
 * diagonal blocks are contained in the 64-row tiles of the noisy half, so this part decomposes into independent
 * (kBlockM x kBlockM) diagonal tiles. Processing them in the main backward kernel would make work items with a single
 * query tile (a key tile load and a dK / dV store for very little compute), so they are computed by this kernel, which
 * runs after the preprocessing (which computes dPsum and LSE in log2 units of all the rows) and before the main kernel.
 * For a noisy tile of a sample and a KV head, looping over the query heads of the group (with the loads of Q and dO of
 * the next head in flight while computing the current one):
 *   - S = Q K^T, P = exp(S - LSE), dP = dO V^T, dS = P * (dP - dPsum) on the block-diagonal tile,
 *   - dQaccum = dS K: the initial value of dQaccum of the rows, to which the main kernel then adds the contributions
 *     of the clean keys. It is written in the layout of dQaccum of the main kernel (that of its dQ MMA fragments),
 *   - dK += dS^T Q, dV += P^T dO.
 * dK and dV of the noisy keys are complete at the end (summed over the query heads of the group) and written directly.
 *
 * Same numerics as the main kernel: FP32 accumulation, P and dS rounded to the input type before the second GEMMs.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "diffusion.h"
#include "utils.h"

namespace flash {

using namespace cute;

// MainTiledMmadQ, NumMmaThreadsMain, dQ_swapAB_main: the dQ MMA of the main backward kernel, which defines the layout
// of a (kBlockM x kHeadDim) tile of dQaccum.
template <int kHeadDim, int kBlockM, class Element, bool Varlen, bool Has_softcap,
          class MainTiledMmadQ, int NumMmaThreadsMain, bool dQ_swapAB_main>
class FlashAttnBwdNoisyDiag {

public:

    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    static_assert(kBlockM == 64 || kBlockM == 128);
    // The keys of the diagonal tile are the rows of the query tile
    static constexpr int kBlockN = kBlockM;
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = NumMmaThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    // kBlockM == 64:  the warpgroups split the keys (S, dP) and the head dimension (dK, dV, dQ).
    // kBlockM == 128: the warpgroups split the queries (S, dP, dQ) and the keys (dK, dV).
    static constexpr bool SplitRows = kBlockM == 128;

    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<_64, Int<!SplitRows ? kBlockN / 2 : kBlockN>, Int<kHeadDim>>>(),
        std::conditional_t<!SplitRows, Layout<Shape<_1, _2, _1>>, Layout<Shape<_2, _1, _1>>>{}));
    // dK = dS^T Q, dV = P^T dO: (kBlockN x kHeadDim) with K = kBlockM. A = P^T, B = dO^T are MN-major in smem.
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<_64, Int<!SplitRows ? kHeadDim / 2 : kHeadDim>, Int<kBlockM>>,
                                   GMMA::Major::MN, GMMA::Major::MN>(),
        std::conditional_t<!SplitRows, Layout<Shape<_1, _2, _1>>, Layout<Shape<_2, _1, _1>>>{}));
    // dQ = dS K: (kBlockM x kHeadDim) with K = kBlockN. A = dS is K-major, B = K^T is MN-major in smem.
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<_64, Int<!SplitRows ? kHeadDim / 2 : kHeadDim>, Int<kBlockN>>,
                                   GMMA::Major::K, GMMA::Major::MN>(),
        std::conditional_t<!SplitRows, Layout<Shape<_1, _2, _1>>, Layout<Shape<_2, _1, _1>>>{}));

    // Q and dO: K-major for S / dP, read as MN-major (transposed views) for dK / dV
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<!SplitRows ? kHeadDim / 2 : kHeadDim>>());
    using SmemLayoutQdO = decltype(tile_to_shape(SmemLayoutAtomQdO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutQdOt = decltype(cute::composition(SmemLayoutQdO{},
                                    make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, make_stride(Int<kBlockM>{}, _1{}))));
    // K: K-major for S, read as MN-major (transposed view) for dQ
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<!SplitRows ? kHeadDim / 2 : kHeadDim>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutKt = decltype(cute::composition(SmemLayoutK{},
                                  make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, make_stride(Int<kBlockN>{}, _1{}))));
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    // P and dS: K-major (keys contiguous) for dQ, read as MN-major (transposed views) for dK / dV
    using SmemLayoutAtomPdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<64>, Int<!SplitRows ? kBlockN / 2 : kBlockN>>());
    using SmemLayoutPdS = decltype(tile_to_shape(SmemLayoutAtomPdS{}, Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using SmemLayoutPdSt = decltype(cute::composition(SmemLayoutPdS{},
                                    make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, make_stride(Int<kBlockM>{}, _1{}))));
    static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;
    using SmemCopyAtomPdS = Copy_Atom<std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U32x4_STSM_N, cute::SM90_U32x2_STSM_N>, Element>;
    // dK, dV staging for the output (reuses smem_q / smem_do)
    using SmemCopyAtomdKV = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

    // Gmem loads / stores: 128-bit per thread, each row handled by kGmemThreadsPerRow consecutive threads
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0);
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load
    using GmemTiledCopyAsync = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    // dQaccum tile of the main kernel: (kBlockM * kHeadDim / NumMmaWarpGroupsMain, NumMmaWarpGroupsMain) floats
    static constexpr int NumMmaWarpGroupsMain = NumMmaThreadsMain / cutlass::NumThreadsPerWarpGroup;
    using R2STiledCopydQaccumMain = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                                             Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroupsMain>>>{},
                                             Layout<Shape<_4>>{}));

    static constexpr int kStages = 2;  // Q, dO, LSE, dPsum of the current and of the next query head
    struct SharedStorage : cute::aligned_struct<1024> {
        // stage s: smem_qdo[2 s] = Q, smem_qdo[2 s + 1] = dO. dQ (FP32) is written in the stage being processed once
        // the GEMMs reading it are done.
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQdO>, 1024> smem_qdo[2 * kStages];
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 1024> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, 1024> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, 1024> smem_ds;
        cute::array_aligned<ElementAccum, kBlockM * kStages, 128> smem_lse;
        cute::array_aligned<ElementAccum, kBlockM * kStages, 128> smem_dpsum;
    };
    static_assert(sizeof(ElementAccum) * kBlockM * kHeadDim <= 2 * sizeof(Element) * cute::cosize_v<SmemLayoutQdO>);

    // dQ in smem: row-major with the columns XOR-swizzled by the row, so that 4 consecutive rows of an MMA fragment
    // (8 consecutive floats each) cover the 32 banks. The XOR only changes bits 3-4 of the column: the swizzled
    // column stays in the same 32-column chunk, hence in the row (kHeadDim is a multiple of 32).
    static_assert(kHeadDim % 32 == 0);
    CUTLASS_DEVICE static int sdq_index(int const row, int const col) {
        return row * kHeadDim + (col ^ ((row & 3) << 3));
    }

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;
    using ShapePadded = cute::Shape<int32_t, int32_t>;  // (total_padded (* d), head)
    using StridePadded = cute::Stride<_1, int64_t>;

    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* ptr_V;
        StrideQKV const stride_V;
        Element const* ptr_O;
        StrideQKV const stride_O;
        Element const* ptr_dO;
        StrideQKV const stride_dO;
        Element* ptr_dK;
        StrideQKV const stride_dK;
        Element* ptr_dV;
        StrideQKV const stride_dV;
        float const* ptr_LSE;
        StrideLSE const stride_LSE;
        float* ptr_LSE_log2;
        float* ptr_dPsum;
        ShapePadded const shape_dPsum;     // (total_q_padded, head)
        StridePadded const stride_dPsum;
        ElementAccum* ptr_dQaccum;
        ShapePadded const shape_dQaccum;   // (total_q_padded * d_rounded, head)
        StridePadded const stride_dQaccum;
        float const softmax_scale;
        float const softcap_val;
        int const block_size;
        int const seqlen;  // length of a sample (2n) if not varlen
        int const* cu_seqlens = nullptr;
    };

    struct Params {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* ptr_V;
        StrideQKV const stride_V;
        Element const* ptr_O;
        StrideQKV const stride_O;
        Element const* ptr_dO;
        StrideQKV const stride_dO;
        Element* ptr_dK;
        StrideQKV const stride_dK;
        Element* ptr_dV;
        StrideQKV const stride_dV;
        float const* ptr_LSE;
        StrideLSE const stride_LSE;
        float* ptr_LSE_log2;
        float* ptr_dPsum;
        ShapePadded const shape_dPsum;
        StridePadded const stride_dPsum;
        ElementAccum* ptr_dQaccum;
        ShapePadded const shape_dQaccum;
        StridePadded const stride_dQaccum;
        float const softmax_scale, softmax_scale_log2, softcap_val;
        cutlass::FastDivmod qhead_per_khead_divmod;
        cutlass::FastDivmod bs_divmod;
        int const seqlen;
        int const* cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        // Softcapping: as in the main kernel (see CollectiveMainloopBwdSm90::to_underlying_arguments)
        return {args.ptr_Q, args.shape_Q, args.stride_Q, args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.stride_V,
                args.ptr_O, args.stride_O, args.ptr_dO, args.stride_dO, args.ptr_dK, args.stride_dK, args.ptr_dV, args.stride_dV,
                args.ptr_LSE, args.stride_LSE, args.ptr_LSE_log2, args.ptr_dPsum, args.shape_dPsum, args.stride_dPsum,
                args.ptr_dQaccum, args.shape_dQaccum, args.stride_dQaccum,
                args.softmax_scale,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                cutlass::FastDivmod(args.block_size),
                args.seqlen, args.cu_seqlens};
    }

    // Load (cp.async) a (kBlockM x kHeadDim) tile of rows [0, max_rows) of gmem tensor g into smem tensor s, zeros
    // elsewhere (the zeros matter: garbage could be inf / nan, and 0 * nan = nan in the GEMMs).
    template <typename TensorG, typename TensorS>
    CUTLASS_DEVICE static void
    load_tile_async(TensorG const& g, TensorS& s, int const max_rows, int const headdim, int const thread_idx) {
        GmemTiledCopyAsync gmem_tiled_copy;
        auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(thread_idx);
        Tensor tXgX = gmem_thr_copy.partition_S(g);
        Tensor tXsX = gmem_thr_copy.partition_D(s);
        Tensor cX = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tXcX = gmem_thr_copy.partition_D(cX);
        Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXgX)));
        #pragma unroll
        for (int k = 0; k < size(tXpX); ++k) { tXpX(k) = get<1>(tXcX(_0{}, _0{}, k)) < headdim; }
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clear_OOB_K=*/true>(
            gmem_tiled_copy, tXgX, tXsX, tXcX, tXpX, max_rows);
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const thread_idx = threadIdx.x;
        int const m_block = blockIdx.x;
        int const bidh_kv = blockIdx.y;
        int const bidb = blockIdx.z;

        flash::DiffusionSeqlenInfo<Varlen> seqlen_info(bidb, params.seqlen, params.cu_seqlens);
        bool const is_varlen = Varlen && params.cu_seqlens;
        int const n = seqlen_info.n;
        if (m_block * kBlockM >= n) { return; }
        int const offset = seqlen_info.offset_half(0);  // the noisy half
        int const offset_padded = seqlen_info.template offset_padded<kBlockM>(bidb, 0);
        int const max_rows = n - m_block * kBlockM;  // number of valid rows of the tile
        int const headdim = get<1>(params.shape_Q);
        int const qhead_per_khead = params.qhead_per_khead_divmod.divisor;

        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutKt{});
        Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPdS{});
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdS{});
        Tensor sPt = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPdSt{});
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdSt{});

        auto batch_idx = !is_varlen ? bidb : 0;
        Tensor mLSElog2_all = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_dPsum, params.stride_dPsum);
        Tensor mdPsum_all = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_dPsum, params.stride_dPsum);
        Tensor mdQaccum_all = make_tensor(make_gmem_ptr(params.ptr_dQaccum), params.shape_dQaccum, params.stride_dQaccum);

        // Loads (cp.async) of Q, dO, LSE_log2, dPsum of query head h into stage s
        auto load_head = [&](int const h, int const stage) {
            int const bidh = bidh_kv * qhead_per_khead + h;
            Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q), params.shape_Q, params.stride_Q)(_, _, bidh, batch_idx);
            Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_Q, params.stride_dO)(_, _, bidh, batch_idx);
            Tensor gQ = local_tile(domain_offset(make_coord(offset, _0{}), mQ), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
            Tensor gdO = local_tile(domain_offset(make_coord(offset, _0{}), mdO), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
            Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_qdo[2 * stage].data()), SmemLayoutQdO{}));
            Tensor sdO_pi = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_qdo[2 * stage + 1].data()), SmemLayoutQdO{}));
            load_tile_async(gQ, sQ_pi, max_rows, headdim, thread_idx);
            load_tile_async(gdO, sdO_pi, max_rows, headdim, thread_idx);
            // LSE_log2 and dPsum (written by the preprocessing): kBlockM floats each, 16 bytes per thread
            static constexpr int kChunks = kBlockM * int(sizeof(ElementAccum)) / 16;
            static_assert(2 * kChunks <= NumMmaThreads);
            if (thread_idx < 2 * kChunks) {
                bool const is_lse = thread_idx < kChunks;
                int const chunk = is_lse ? thread_idx : thread_idx - kChunks;
                ElementAccum const* src = (is_lse ? &mLSElog2_all(0, bidh) : &mdPsum_all(0, bidh)) + offset_padded + m_block * kBlockM + chunk * 4;
                ElementAccum* dst = (is_lse ? shared_storage.smem_lse.data() : shared_storage.smem_dpsum.data()) + stage * kBlockM + chunk * 4;
                cute::copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementAccum>{},
                           make_tensor(make_gmem_ptr(src), Shape<_4>{}), make_tensor(make_smem_ptr(dst), Shape<_4>{}));
            }
        };

        // K and V of the noisy keys of the tile, and the first query head
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, batch_idx);
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_K, params.stride_V)(_, _, bidh_kv, batch_idx);
        Tensor gK = local_tile(domain_offset(make_coord(offset, _0{}), mK), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
        Tensor gV = local_tile(domain_offset(make_coord(offset, _0{}), mV), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
        Tensor sK_pi = cute::as_position_independent_swizzle_tensor(sK);
        Tensor sV_pi = cute::as_position_independent_swizzle_tensor(sV);
        load_tile_async(gK, sK_pi, max_rows, headdim, thread_idx);
        load_tile_async(gV, sV_pi, max_rows, headdim, thread_idx);
        load_head(0, 0);
        cute::cp_async_fence();

        int const warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        Layout warp_group_thread_layout = make_layout(make_shape(Int<NumMmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        TiledMmaSdP tiled_mma_SdP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ tiled_mma_dQ;
        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto thread_mma_dQ = tiled_mma_dQ.get_thread_slice(thread_idx);

        Tensor tSrK = wg_mma_SdP.partition_fragment_B(sK);
        Tensor tdPrV = wg_mma_SdP.partition_fragment_B(sV);
        Tensor tdVrP = wg_mma_dKV.partition_fragment_A(sPt);
        Tensor tdKrdS = wg_mma_dKV.partition_fragment_A(sdSt);
        Tensor tdQrdS = wg_mma_dQ.partition_fragment_A(sdS);
        Tensor tdQrK = wg_mma_dQ.partition_fragment_B(sKt);

        auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
        auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);
        Tensor tPsP = smem_thr_copy_PdS.partition_D(cute::as_position_independent_swizzle_tensor(sP));
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(cute::as_position_independent_swizzle_tensor(sdS));

        // (row, col) coordinates of the S fragment: rows are queries, cols are keys (relative to the tile)
        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma_SdP.partition_C(cS);
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
        // (row, col) coordinates of the dQ fragment
        Tensor cdQ = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tdQcdQ = thread_mma_dQ.partition_C(cdQ);

        Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        clear(tdKrdK);
        clear(tdVrdV);

        #pragma unroll 1
        for (int h = 0; h < qhead_per_khead; ++h) {
            int const bidh = bidh_kv * qhead_per_khead + h;
            int const stage = h % kStages;
            // Prefetch the next query head into the other stage (free: the previous iteration ended with a sync)
            if (h + 1 < qhead_per_khead) {
                load_head(h + 1, (h + 1) % kStages);
                cute::cp_async_fence();
                cute::cp_async_wait<1>();
            } else {
                cute::cp_async_wait<0>();
            }
            cutlass::arch::fence_view_async_shared();  // cp.async writes are read by the GMMAs
            __syncthreads();

            Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_qdo[2 * stage].data()), SmemLayoutQdO{});
            Tensor sdO = make_tensor(make_smem_ptr(shared_storage.smem_qdo[2 * stage + 1].data()), SmemLayoutQdO{});
            Tensor sQt = make_tensor(make_smem_ptr(shared_storage.smem_qdo[2 * stage].data()), SmemLayoutQdOt{});
            Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.smem_qdo[2 * stage + 1].data()), SmemLayoutQdOt{});
            Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse.data() + stage * kBlockM), Shape<Int<kBlockM>>{});
            Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.smem_dpsum.data() + stage * kBlockM), Shape<Int<kBlockM>>{});
            Tensor tSrQ = wg_mma_SdP.partition_fragment_A(sQ);
            Tensor tdPrdO = wg_mma_SdP.partition_fragment_A(sdO);
            Tensor tdVrdO = wg_mma_dKV.partition_fragment_B(sdOt);
            Tensor tdKrQ = wg_mma_dKV.partition_fragment_B(sQt);

            // S = Q K^T, dP = dO V^T
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, Shape<Int<kBlockM>, Int<kBlockN>>{});
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrQ, tSrK, tSrS);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tdPrdO, tdPrV, tdPrdP);
            warpgroup_wait<1>();
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
            auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();
            // Block-diagonal mask: the tile starts at a multiple of the block size. Rows beyond the end of the half
            // have LSE = +inf, so P = 0 there.
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                int const row = get<0>(tScS_rowcol(mi, _0{}));
                int const row_blk = params.bs_divmod.div(row);
                float const lse_scaled = sLSE(row);
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    int const col = get<1>(tScS_rowcol(mi, ni));
                    bool const visible = params.bs_divmod.div(col) == row_blk && col < max_rows;
                    scores(mi, ni) = visible ? exp2f(scores(mi, ni) * params.softmax_scale_log2 - lse_scaled) : 0.f;
                }
            }
            warpgroup_wait<0>();
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                int const row = get<0>(tScS_rowcol(mi, _0{}));
                float const dP_sum_cur = sdPsum(row);
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum_cur);
                    if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
                }
            }
            Tensor rP = make_tensor_like<Element>(tSrS);
            flash::convert_type_out(tSrS, rP);
            Tensor rdS = make_tensor_like<Element>(tdPrdP);
            flash::convert_type_out(tdPrdP, rdS);
            cute::copy(smem_tiled_copy_PdS, smem_thr_copy_PdS.retile_S(rP), tPsP);
            cute::copy(smem_tiled_copy_PdS, smem_thr_copy_PdS.retile_S(rdS), tdSsdS);
            cutlass::arch::fence_view_async_shared();
            __syncthreads();

            // dV += P^T dO, dK += dS^T Q, dQ = dS K
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO, tdVrdV);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, tdKrdS, tdKrQ, tdKrdK);
            Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, Shape<Int<kBlockM>, Int<kHeadDim>>{});
            flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma_dQ, tdQrdS, tdQrK, tdQrdQ);
            warpgroup_wait<0>();
            // dQ is written to smem in the stage of Q and dO: wait for all the GEMMs reading them
            __syncthreads();
            ElementAccum* const sdQ = reinterpret_cast<ElementAccum*>(shared_storage.smem_qdo[2 * stage].data());
            #pragma unroll
            for (int i = 0; i < size(tdQrdQ); ++i) { sdQ[sdq_index(get<0>(tdQcdQ(i)), get<1>(tdQcdQ(i)))] = tdQrdQ(i); }
            __syncthreads();

            // Write dQ to dQaccum in the layout of the main kernel: its dQ MMA thread vt holds the dQ values at the
            // coordinates of its fragment, and stores them with its R2S copy
            {
                Tensor mdQaccum = mdQaccum_all(_, bidh);
                Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(m_block));  // (M * K)
                Tensor gdQaccum_wg = cute::flat_divide(gdQaccum, Int<kBlockM * kHeadDim / NumMmaWarpGroupsMain>{});  // (M * K / WG, WG)
                MainTiledMmadQ main_tiled_mma_dQ;
                R2STiledCopydQaccumMain r2s_tiled_copy_dQaccum;
                Tensor cdQ_main = cute::make_identity_tensor(Shape<Int<!dQ_swapAB_main ? kBlockM : kHeadDim>, Int<!dQ_swapAB_main ? kHeadDim : kBlockM>>{});
                #pragma unroll 1
                for (int vt = thread_idx; vt < NumMmaThreadsMain; vt += NumMmaThreads) {
                    Tensor tdQcdQ_main = main_tiled_mma_dQ.get_thread_slice(vt).partition_C(cdQ_main);
                    // Same fragment as the main kernel's dQ accumulator (make_fragment_like of the identity partition
                    // does not compile when it has a mode of size 1)
                    Tensor tdQrdQ_main = partition_fragment_C(main_tiled_mma_dQ, cdQ_main.shape());
                    static_assert(CUTE_STATIC_V(size(tdQrdQ_main)) == CUTE_STATIC_V(size(tdQcdQ_main)));
                    #pragma unroll
                    for (int i = 0; i < size(tdQrdQ_main); ++i) {
                        int const row = get<!dQ_swapAB_main ? 0 : 1>(tdQcdQ_main(i));
                        int const col = get<!dQ_swapAB_main ? 1 : 0>(tdQcdQ_main(i));
                        tdQrdQ_main(i) = sdQ[sdq_index(row, col)];
                    }
                    auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(vt);
                    Tensor tdQrdQ_store = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ_main));
                    Tensor tdQgdQ_store = recast<float4>(r2s_thr_copy_dQaccum.partition_D(gdQaccum_wg));
                    static_assert(CUTE_STATIC_V(size(tdQrdQ_store)) == CUTE_STATIC_V(size(tdQgdQ_store)));
                    #pragma unroll
                    for (int i = 0; i < size(tdQrdQ_store); ++i) { tdQgdQ_store(i) = tdQrdQ_store(i); }
                }
            }
            // The next iteration overwrites this stage (prefetch of query head h + 2)
            __syncthreads();
        }

        // dK, dV of the noisy keys: complete (all the query heads of the group). Stage them in smem (stage 0).
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= params.softmax_scale; }
        {
            auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma_dKV);
            auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);
            Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_qdo[0].data()), SmemLayoutQdO{}));
            Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_qdo[1].data()), SmemLayoutQdO{}));
            Tensor rdK = make_tensor_like<Element>(tdKrdK);
            flash::convert_type_out(tdKrdK, rdK);
            Tensor rdV = make_tensor_like<Element>(tdVrdV);
            flash::convert_type_out(tdVrdV, rdV);
            cute::copy(smem_tiled_copy_dKV, smem_thr_copy_dKV.retile_S(rdK), smem_thr_copy_dKV.partition_D(sdK));
            cute::copy(smem_tiled_copy_dKV, smem_thr_copy_dKV.retile_S(rdV), smem_thr_copy_dKV.partition_D(sdV));
            __syncthreads();
            Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_K, params.stride_dK)(_, _, bidh_kv, batch_idx);
            Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_K, params.stride_dV)(_, _, bidh_kv, batch_idx);
            Tensor gdK = local_tile(domain_offset(make_coord(offset, _0{}), mdK), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
            Tensor gdV = local_tile(domain_offset(make_coord(offset, _0{}), mdV), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
            GmemTiledCopy gmem_tiled_copy;
            auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(thread_idx);
            Tensor cX = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
            Tensor tXcX = gmem_thr_copy.partition_D(cX);
            Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXcX)));
            #pragma unroll
            for (int k = 0; k < size(tXpX); ++k) { tXpX(k) = get<1>(tXcX(_0{}, _0{}, k)) < headdim; }
            Tensor tXsdK = gmem_thr_copy.partition_S(sdK);
            Tensor tXsdV = gmem_thr_copy.partition_S(sdV);
            Tensor tXrdK = make_fragment_like(tXsdK);
            Tensor tXrdV = make_fragment_like(tXsdV);
            cute::copy(tXsdK, tXrdK);
            cute::copy(tXsdV, tXrdV);
            Tensor tXgdK = gmem_thr_copy.partition_D(gdK);
            Tensor tXgdV = gmem_thr_copy.partition_D(gdV);
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy, tXrdK, tXgdK, tXcX, tXpX, max_rows);
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy, tXrdV, tXgdV, tXcX, tXpX, max_rows);
        }
    }

};

} // namespace flash
