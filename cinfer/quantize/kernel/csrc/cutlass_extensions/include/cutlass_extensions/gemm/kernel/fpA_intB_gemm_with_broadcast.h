/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or
   support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,      ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_, ///! Epilogue
          typename ThreadblockSwizzle_, ///! Threadblock swizzling function
          typename KernelArch ///! The Architecture this kernel is compiled for.
                              /// Used since SIMT kernels lose top-level arch.
                              //////
          >
struct GemmFpAIntBWithBroadcast {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Element;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Mma::LayoutC;
  using ElementScale = ElementC;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformA;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC =
      Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  static constexpr int kInterleave =
      Mma::IteratorB::Shape::kRow / Mma::Shape::kK;

  /// Parameters structure
  struct Arguments {
    GemmUniversalMode mode = GemmUniversalMode::kGemm;

    cutlass::gemm::GemmCoord problem_size;
    int batch_count;
    typename EpilogueOutputOp::Params epilogue;

    void const *ptr_A;
    void const *ptr_B;
    void const *ptr_scales;
    void const *ptr_C;
    void *ptr_D;

    void const *ptr_Vector;
    void const *ptr_Tensor;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;
    int64_t batch_stride_Vector;
    int64_t batch_stride_Tensor;

    int lda, ldb, ldc, ldd, ldr, ldt;

    typename EpilogueOutputOp::Params output_op;

    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    CUTLASS_HOST_DEVICE
    Arguments() {}

    CUTLASS_HOST_DEVICE
    Arguments(cutlass::gemm::GemmCoord const &problem_size, int batch_count,
              typename EpilogueOutputOp::Params epilogue, void const *ptr_A,
              void const *ptr_B, void const *ptr_scales, void const *ptr_C,
              void *ptr_D, const void *ptr_Vector, const void *ptr_Tensor,
              int64_t batch_stride_A, int64_t batch_stride_B,
              int64_t batch_stride_C, int64_t batch_stride_D,
              int64_t batch_stride_Vector, int64_t batch_stride_Tensor,
	      int lda, int ldb, int ldc, int ldd, int ldr, int ldt,
              typename EpilogueOutputOp::Params output_op =
                  typename EpilogueOutputOp::Params())
        : problem_size(problem_size), batch_count(batch_count),
          epilogue(epilogue), ptr_A(ptr_A), ptr_B(ptr_B),
          ptr_scales(ptr_scales), ptr_C(ptr_C), ptr_D(ptr_D),
          ptr_Vector(ptr_Vector), ptr_Tensor(ptr_Tensor),
          batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B),
          batch_stride_C(batch_stride_C), batch_stride_D(batch_stride_D),
          batch_stride_Vector(batch_stride_Vector),
          batch_stride_Tensor(batch_stride_Tensor), lda(lda), ldb(ldb),
          ldc(ldc), ldd(ldd), ldr(ldr), ldt(ldt), output_op(output_op),
          gather_A_indices(nullptr), gather_B_indices(nullptr),
          scatter_D_indices(nullptr) {}
  };

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorScale::Params params_scale;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::TensorTileIterator::Params params_Tensor;

    typename EpilogueOutputOp::Params output_op;

    // GemmUniversalMode mode; todo
    int batch_count;
    int gemm_k_size;
    void *ptr_A;
    void *ptr_B;
    void *ptr_C;
    void *ptr_scales;
    void *ptr_D;

    void *ptr_Vector;
    typename LayoutC::Stride::Index ldr;

    void *ptr_Tensor;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;
    int64_t batch_stride_Vector;
    int64_t batch_stride_Tensor;

    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : swizzle_log_tile(0), gemm_k_size(0) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args,
           cutlass::gemm::GemmCoord const &grid_tiled_shape,
           const int gemm_k_size, void *workspace = nullptr)
        : problem_size(args.problem_size), grid_tiled_shape(grid_tiled_shape),
          swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
          params_A(args.lda), params_B(args.ldb), params_C(args.ldc),
          params_D(args.ldd), params_Tensor(args.ldt), output_op(args.epilogue),
          batch_count(args.batch_count), gemm_k_size(gemm_k_size),
          ptr_A(const_cast<void *>(args.ptr_A)),
          ptr_B(const_cast<void *>(args.ptr_B)),
          ptr_scales(const_cast<void *>(args.ptr_scales)),
          ptr_C(const_cast<void *>(args.ptr_C)), ptr_D(args.ptr_D),
          ptr_Vector(const_cast<void *>(args.ptr_Vector)), ldr(args.ldr),
          ptr_Tensor(const_cast<void *>(args.ptr_Tensor)), batch_stride_A(args.batch_stride_A),
          batch_stride_B(args.batch_stride_B),
          batch_stride_C(args.batch_stride_C),
          batch_stride_D(args.batch_stride_D),
          batch_stride_Vector(args.batch_stride_Vector),
          batch_stride_Tensor(args.batch_stride_Tensor),
          gather_A_indices(args.gather_A_indices),
          gather_B_indices(args.gather_B_indices),
          scatter_D_indices(args.scatter_D_indices) {}
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  GemmFpAIntBWithBroadcast() {}

  CUTLASS_HOST_DEVICE
  static Status can_implement(Arguments const &args) {
    // todo
    return Status::kSuccess;
  }

  static size_t
  get_extra_workspace_size(Arguments const &args,
                           cutlass::gemm::GemmCoord const &grid_tiled_shape) {

    return 0;
  }

  // The dummy template parameter is not used and exists so that we can compile
  // this code using a standard earlier than C++17. Prior to C++17, fully
  // specialized templates HAD to exists in a namespace
  template <bool B, typename dummy = void> struct KernelRunner {
    CUTLASS_DEVICE
    static void run_kernel(Params const &params,
                           SharedStorage &shared_storage) {
      CUTLASS_NOT_IMPLEMENTED();
    }
  };

  template <typename dummy> struct KernelRunner<true, dummy> {
    CUTLASS_DEVICE
    static void run_kernel(Params const &params,
                           SharedStorage &shared_storage) {
      using LayoutB = typename Mma::IteratorB::Layout;
      static_assert(
          platform::is_same<LayoutB, layout::RowMajor>::value &&
                  kInterleave == 1 ||
              platform::is_same<LayoutB, layout::ColumnMajor>::value &&
                  kInterleave >= 1,
          "B must be row major/col major OR col major  interleaved.");

      // Compute threadblock location
      ThreadblockSwizzle threadblock_swizzle;

      cutlass::gemm::GemmCoord threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

      // Early exit if CTA is out of range
      if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
          params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

        return;
      }

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
          threadblock_tile_offset.m() * Mma::Shape::kM,
          threadblock_tile_offset.k() * params.gemm_k_size,
      };

      cutlass::MatrixCoord tb_offset_B{
          threadblock_tile_offset.k() * params.gemm_k_size * kInterleave,
          threadblock_tile_offset.n() * Mma::Shape::kN / kInterleave};

      cutlass::MatrixCoord tb_offset_scale{0, threadblock_tile_offset.n() *
                                                  Mma::Shape::kN};

      // Problem size is a function of threadblock index in the K dimension
      int problem_size_k =
          min(params.problem_size.k(),
              (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations =
          (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) /
          Mma::Shape::kK;

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
          params.params_A, static_cast<ElementA *>(params.ptr_A),
          {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_A,
          params.gather_A_indices);

      typename Mma::IteratorB iterator_B(
          params.params_B, static_cast<ElementB *>(params.ptr_B),
          {problem_size_k * kInterleave, params.problem_size.n() / kInterleave},
          thread_idx, tb_offset_B, params.gather_B_indices);

      typename Mma::IteratorScale iterator_scale(
          params.params_scale, static_cast<ElementScale *>(params.ptr_scales),
          {1, params.problem_size.n()}, thread_idx, tb_offset_scale);

      // Broadcast the warp_id computed by lane 0 to ensure dependent code is
      // compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
      int lane_idx = threadIdx.x % 32;

      //
      // Main loop
      //
      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      if (gemm_k_iterations > 0) {
        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
            iterator_scale, accumulators);
      }

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      //
      // Masked tile iterators constructed from members
      //

      threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

      // assume identity swizzle
      MatrixCoord threadblock_offset(
          threadblock_tile_offset.m() * Mma::Shape::kM,
          threadblock_tile_offset.n() * Mma::Shape::kN);

      int block_idx = threadblock_tile_offset.m() +
                      threadblock_tile_offset.n() * params.grid_tiled_shape.m();

      ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
      ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(
          params.params_C, ptr_C, params.problem_size.mn(),
          thread_idx, threadblock_offset, params.scatter_D_indices);

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
          params.params_D, ptr_D, params.problem_size.mn(),
          thread_idx, threadblock_offset, params.scatter_D_indices);

      typename Epilogue::ElementTensor *ptr_Tensor =
          static_cast<typename Epilogue::ElementTensor *>(params.ptr_Tensor);

      // Define the reduction output pointer and move to the appropriate place
      typename Epilogue::ElementVector *ptr_Vector =
          static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

      typename Epilogue::TensorTileIterator tensor_iterator(
          params.params_Tensor,
          // Only the final block outputs Tensor
          ptr_Tensor, params.problem_size.mn(), thread_idx, threadblock_offset);

      Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                        lane_idx);

      if (ptr_Vector) {
        ptr_Vector += threadblock_offset.column() +
                      threadblock_tile_offset.m() * params.ldr;
      }

      epilogue(output_op, ptr_Vector, iterator_D, accumulators, iterator_C,
               tensor_iterator, params.problem_size.mn(), threadblock_offset);
    }
  };

  /*
      To improve compilation speed, we do not compile the device operator if the
     CUDA_ARCH does not correspond to the ArchTag of the cutlass kernel
     operator.
    */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm70>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm75>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 900)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm80>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
