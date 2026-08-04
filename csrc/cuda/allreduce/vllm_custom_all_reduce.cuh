// SPDX-FileCopyrightText: 2025 vLLM Team
// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file has adaption of open-source code from the following sources:
 * - The kernel is originally from vLLM
 *   (https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh),
 *   licensed under Apache 2.0.
 */
#pragma once

#include "common.h"

#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace chitu {

constexpr int kMaxBlocks = 36;
// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;
struct Signal {
    alignas(128) FlagType self_counter[kMaxBlocks][8];
    // Two sets of peer counters are needed for two syncs. The reason is that
    // it's possible for peer GPU block to arrive at the second sync point while
    // the current GPU block haven't passed the first sync point. Thus, peer GPU
    // may write counter+1 while current GPU is busy waiting for counter. We use
    // alternating counter array to avoid this possibility.
    alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
    // Device-side row counts exchanged by the variable-length collectives.
    // Two epoch slots prevent a faster rank from overwriting a count that a
    // slower rank is still consuming. Counts are per block because the peer
    // barrier deliberately has no cross-block synchronization.
    alignas(128) FlagType token_count[2][kMaxBlocks][8];
    // Used only on a fatal global-capacity overflow. The final local block
    // traps after every block has completed the peer tail barrier, so no rank
    // is left spinning in the collective when the device error is raised.
    alignas(128) FlagType varlen_overflow_blocks;
#endif
};

struct __align__(16) RankData {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const void *ptrs[8];
#else
    const void *__restrict__ ptrs[8];
#endif
};

struct __align__(16) RankSignals {
    Signal *signals[8];
};

// like std::array, but aligned
template <typename T, int sz> struct __align__(alignof(T) * sz) array_t {
    T data[sz];
    using type = T;
    static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T> struct packed_t {
    // the (P)acked type for load/store
    using P = array_t<T, 16 / sizeof(T)>;
    // the (A)ccumulator type for reduction
    using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T> DINLINE T downcast_s(float val);
template <> DINLINE half downcast_s(float val) { return __float2half(val); }

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half &assign_add(half &a, half b) {
    a = __hadd(a, b);
    return a;
}
DINLINE float &assign_add(float &a, float b) { return a += b; }

#if defined(__HIP_PLATFORM_AMD__) || (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <> DINLINE nv_bfloat16 downcast_s(float val) {
    return __float2bfloat16(val);
}
DINLINE nv_bfloat16 &assign_add(nv_bfloat16 &a, nv_bfloat16 b) {
    a = __hadd(a, b);
    return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N> &packed_assign_add(array_t<T, N> &a, array_t<T, N> b) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        assign_add(a.data[i], b.data[i]);
    }
    return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
    if constexpr (std::is_same<T, float>::value) {
        return val;
    } else {
        array_t<float, N> out;
#pragma unroll
        for (int i = 0; i < N; i++) {
            out.data[i] = upcast_s(val.data[i]);
        }
        return out;
    }
}

template <typename O> DINLINE O downcast(array_t<float, O::size> val) {
    if constexpr (std::is_same<typename O::type, float>::value) {
        return val;
    } else {
        O out;
#pragma unroll
        for (int i = 0; i < O::size; i++) {
            out.data[i] = downcast_s<typename O::type>(val.data[i]);
        }
        return out;
    }
}

static DINLINE void st_flag_release(FlagType *flag_addr, FlagType flag) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    __hip_atomic_store(flag_addr, flag, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
                 "l"(flag_addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
                 "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType *flag_addr) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __hip_atomic_load(flag_addr, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM);
#else
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(flag_addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
                 : "=r"(flag)
                 : "l"(flag_addr));
#endif
    return flag;
#endif
}

static DINLINE void st_flag_volatile(FlagType *flag_addr, FlagType flag) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    __hip_atomic_store(flag_addr, flag, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag),
                 "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_volatile(FlagType *flag_addr) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __hip_atomic_load(flag_addr, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_SYSTEM);
#else
    FlagType flag;
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(flag_addr));
    return flag;
#endif
}

// is_start: whether this is the very first synchronization barrier.
// need_fence: whether a memory fence is needed. If true, a release-acquire
// semantic is used to enforce memory access order before and after this
// barrier.
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals &sg, Signal *self_sg,
                               int rank) {
    if constexpr (!is_start)
        __syncthreads();
    static_assert(
        !(is_start && need_fence)); // Start barrier shouldn't need fence.
    if (threadIdx.x < ngpus) {
        // Increment the counter. Technically we only need one counter, but we
        // use multiple per block to eliminate the need to share the counter via
        // smem.
        auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
        // Write the expected counter value to peer and wait for correct value
        // from peer.
        auto peer_counter_ptr =
            &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
        auto self_counter_ptr =
            &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
        if constexpr (need_fence) {
            st_flag_release(peer_counter_ptr, val);
            while (ld_flag_acquire(self_counter_ptr) != val)
                ;
        } else {
            st_flag_volatile(peer_counter_ptr, val);
            while (ld_flag_volatile(self_counter_ptr) != val)
                ;
        }
    }
    if constexpr (is_start || need_fence)
        __syncthreads();
}

#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
// Publish each rank's logical row count together with the input payload. All
// ranks launch the same fixed grid, including ranks whose count is zero.
template <int ngpus>
DINLINE void multi_gpu_barrier_exchange_count(const RankSignals &sg,
                                               Signal *self_sg, int rank,
                                               int local_count,
                                               int *shared_counts) {
    if (threadIdx.x < ngpus) {
        const auto val =
            self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
        const int peer = threadIdx.x;
        const int slot = val % 2;
        auto peer_count_ptr =
            &sg.signals[peer]->token_count[slot][blockIdx.x][rank];
        auto peer_counter_ptr =
            &sg.signals[peer]->peer_counter[slot][blockIdx.x][rank];
        auto self_counter_ptr =
            &self_sg->peer_counter[slot][blockIdx.x][peer];

        st_flag_volatile(peer_count_ptr, static_cast<FlagType>(local_count));
        st_flag_release(peer_counter_ptr, val);
        while (ld_flag_acquire(self_counter_ptr) != val)
            ;
        shared_counts[peer] = static_cast<int>(ld_flag_volatile(
            &self_sg->token_count[slot][blockIdx.x][peer]));
    }
    __syncthreads();
}

DINLINE void trap_varlen_overflow_after_all_blocks(Signal *self_sg) {
    if (threadIdx.x != 0)
        return;
    const auto arrived = atomicAdd(&self_sg->varlen_overflow_blocks, 1u);
    if (arrived + 1 == gridDim.x)
        __builtin_trap();
}
#endif

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P *ptrs[], int idx) {
    A tmp = upcast(ptrs[0][idx]);
#pragma unroll
    for (int i = 1; i < ngpus; i++) {
        packed_assign_add(tmp, upcast(ptrs[i][idx]));
    }
    return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData *_dp, RankSignals sg, Signal *self_sg,
                               T *__restrict__ result, int rank, int size) {
    using P = typename packed_t<T>::P;
    using A = typename packed_t<T>::A;
    // note: we don't reorder the address so the accumulation order is the same
    // for all ranks, ensuring bitwise identical results
    auto dp = *_dp;
    multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
    // do the actual reduction
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
         idx += gridDim.x * blockDim.x) {
        ((P *)result)[idx] =
            packed_reduce<P, ngpus, A>((const P **)&dp.ptrs[0], idx);
    }
    multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
}

template <typename P> DINLINE P *get_tmp_buf(Signal *sg) {
    return (P *)(((Signal *)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData *_dp, RankSignals sg, Signal *self_sg,
                               T *__restrict__ result, int rank, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    using P = typename packed_t<T>::P;
    using A = typename packed_t<T>::A;
    int part = size / ngpus;
    int start = rank * part;
    int end = rank == ngpus - 1 ? size : start + part;
    int largest_part = part + size % ngpus;
    const P *ptrs[ngpus];
    P *tmps[ngpus];
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
        int target = (rank + i) % ngpus;
        ptrs[i] = (const P *)_dp->ptrs[target];
        tmps[i] = get_tmp_buf<P>(sg.signals[target]);
    }
    auto tmp_out = tmps[0];
    multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
    // stage 1: reduce scatter
    for (int idx = start + tid; idx < end; idx += stride) {
        tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
    }
    multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);

    // stage 2: allgather. Note: it's important to match the tid between
    // the two stages, because visibility across devices is only guaranteed
    // between threads that have the same tid. If thread i computes the sum of
    // start + i in the first stage, then thread i also gathers start + i from
    // all ranks.
    for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
        for (int i = 0; i < ngpus; i++) {
            int gather_from_rank = ((rank + i) % ngpus);
            if (gather_from_rank == ngpus - 1 || idx < part) {
                int dst_idx = gather_from_rank * part + idx;
                ((P *)result)[dst_idx] = tmps[i][idx];
            }
        }
    }
}

#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
// Direct-pull compact variable-length all-gather. Each rank exposes a static
// [global_capacity, row_elems] staging allocation, but peers only copy the
// rows named by the exchanged device counts. The unused output tail is zeroed
// locally. A rank may contribute the entire global capacity; only the sum of
// all rank counts is globally bounded.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_varlen_all_gather(RankData *_dp, RankSignals sg,
                                   Signal *self_sg,
                                   const int *__restrict__ local_count_ptr,
                                   T *__restrict__ output, int rank,
                                   int global_capacity, int row_elems) {
    using P = typename packed_t<T>::P;
    constexpr int pack_elems = P::size;
    __shared__ int counts[ngpus];
    __shared__ int offsets[ngpus + 1];
    __shared__ int invalid_count;

    multi_gpu_barrier_exchange_count<ngpus>(sg, self_sg, rank,
                                            *local_count_ptr, counts);

    if (threadIdx.x == 0) {
        offsets[0] = 0;
        invalid_count = 0;
#pragma unroll
        for (int peer = 0; peer < ngpus; ++peer) {
            int n = counts[peer];
            invalid_count |= n < 0 || n > global_capacity;
            n = n < 0 ? 0 : n;
            n = n > global_capacity ? global_capacity : n;
            counts[peer] = n;
            offsets[peer + 1] = offsets[peer] + n;
        }
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int row_packs = row_elems / pack_elems;
    const int capacity_packs = global_capacity * row_packs;
    const bool overflow = invalid_count || offsets[ngpus] > global_capacity;
    if (!overflow) {
        const auto dp = *_dp;
#pragma unroll
        for (int peer = 0; peer < ngpus; ++peer) {
            const P *src = reinterpret_cast<const P *>(dp.ptrs[peer]);
            P *dst = reinterpret_cast<P *>(output) + offsets[peer] * row_packs;
            const int valid_packs = counts[peer] * row_packs;
            for (int idx = tid; idx < valid_packs; idx += stride)
                dst[idx] = src[idx];
        }
        P zero{};
        const int valid_packs = offsets[ngpus] * row_packs;
        for (int idx = valid_packs + tid; idx < capacity_packs; idx += stride)
            reinterpret_cast<P *>(output)[idx] = zero;
    }

    // Do not let a fast rank reuse its staging input while a peer is reading.
    multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);
    if (overflow)
        trap_varlen_overflow_after_all_blocks(self_sg);
}

// Owner-pull compact variable-length reduce-scatter. Rank r reduces its real
// compact segment from every ETP rank and zeros the unused output capacity.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_varlen_reduce_scatter(RankData *_dp, RankSignals sg,
                                       Signal *self_sg,
                                       const int *__restrict__ local_count_ptr,
                                       T *__restrict__ output, int rank,
                                       int global_capacity, int row_elems) {
    using P = typename packed_t<T>::P;
    using A = typename packed_t<T>::A;
    constexpr int pack_elems = P::size;
    __shared__ int counts[ngpus];
    __shared__ int offsets[ngpus + 1];
    __shared__ int invalid_count;

    multi_gpu_barrier_exchange_count<ngpus>(sg, self_sg, rank,
                                            *local_count_ptr, counts);

    if (threadIdx.x == 0) {
        offsets[0] = 0;
        invalid_count = 0;
#pragma unroll
        for (int peer = 0; peer < ngpus; ++peer) {
            int n = counts[peer];
            invalid_count |= n < 0 || n > global_capacity;
            n = n < 0 ? 0 : n;
            n = n > global_capacity ? global_capacity : n;
            counts[peer] = n;
            offsets[peer + 1] = offsets[peer] + n;
        }
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int row_packs = row_elems / pack_elems;
    const int output_capacity_packs = global_capacity * row_packs;
    const bool overflow = invalid_count || offsets[ngpus] > global_capacity;
    if (!overflow) {
        const auto dp = *_dp;
        const int segment_start = offsets[rank] * row_packs;
        const int valid_packs = counts[rank] * row_packs;
        const P *ptrs[ngpus];
#pragma unroll
        for (int peer = 0; peer < ngpus; ++peer)
            ptrs[peer] =
                reinterpret_cast<const P *>(dp.ptrs[peer]) + segment_start;

        for (int idx = tid; idx < valid_packs; idx += stride) {
            reinterpret_cast<P *>(output)[idx] =
                packed_reduce<P, ngpus, A>(ptrs, idx);
        }
        P zero{};
        for (int idx = valid_packs + tid; idx < output_capacity_packs;
             idx += stride)
            reinterpret_cast<P *>(output)[idx] = zero;
    }

    multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);
    if (overflow)
        trap_varlen_overflow_after_all_blocks(self_sg);
}
#endif

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
  public:
    int rank_;
    int world_size_;
    bool full_nvlink_;

    RankSignals sg_;
    // Stores an map from a pointer to its peer pointters from all ranks.
    std::unordered_map<void *, RankData *> buffers_;
    Signal *self_sg_;

    // Stores rank data from all ranks. This is mainly for cuda graph purposes.
    // For cuda graph to work, all kernel arguments must be fixed during graph
    // capture time. However, the peer pointers are not known during graph
    // capture time. Therefore, during capture, we increment the rank data
    // pointer and use that as the argument to the kernel. The kernel arguments
    // are stored in graph_unreg_buffers_. The actual peer pointers will be
    // filled in at the memory pointed to by the pointers in
    // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
    //
    // The overall process looks like this:
    // 1. Graph capture.
    // 2. Each rank obtains the IPC handles for each addresses used during cuda
    // graph capture using get_graph_buffer_ipc_meta.
    // 3. (In Python) all gather the IPC handles.
    // 4. Obtain the peer pointers by opening the IPC handles, and store them in
    // the rank data array at corresponding positions.
    RankData *d_rank_data_base_, *d_rank_data_end_;
    std::vector<void *> graph_unreg_buffers_;
    // a map from IPC handles to opened IPC pointers
    std::map<IPC_KEY, char *> ipc_handles_;

    /**
     * Signals are an array of ipc-enabled buffers from all ranks.
     * For each of the buffer, the layout is as follows:
     * | -- sizeof(Signal) -- | ------ a few MB ----- |
     * The first section is for allreduce synchronization, and the second
     * section is for storing the intermediate results required by some
     * allreduce algos.
     *
     * Note: this class does not own any device memory. Any required buffers
     * are passed in from the constructor.
     */
    CustomAllreduce(Signal **signals, void *rank_data, size_t rank_data_sz,
                    int rank, int world_size, bool full_nvlink = true)
        : rank_(rank), world_size_(world_size), full_nvlink_(full_nvlink),
          self_sg_(signals[rank]),
          d_rank_data_base_(reinterpret_cast<RankData *>(rank_data)),
          d_rank_data_end_(d_rank_data_base_ +
                           rank_data_sz / sizeof(RankData)) {
        for (int i = 0; i < world_size_; i++) {
            sg_.signals[i] = signals[i];
        }
    }

    char *open_ipc_handle(const void *ipc_handle) {
        auto [it, new_handle] =
            ipc_handles_.insert({*((IPC_KEY *)ipc_handle), nullptr});
        if (new_handle) {
            char *ipc_ptr;
            CUDACHECK(cudaIpcOpenMemHandle(
                (void **)&ipc_ptr, *((const cudaIpcMemHandle_t *)ipc_handle),
                chituIpcMemLazyEnablePeerAccess));
            it->second = ipc_ptr;
        }
        return it->second;
    }

    std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
        auto num_buffers = graph_unreg_buffers_.size();
        auto handle_sz = sizeof(cudaIpcMemHandle_t);
        std::string handles(handle_sz * num_buffers, static_cast<char>(0));
        std::vector<int64_t> offsets(num_buffers);
        for (int i = 0; i < num_buffers; i++) {
            auto ptr = graph_unreg_buffers_[i];
            void *base_ptr;
            // note: must share the base address of each allocation, or we get
            // wrong address
            if (cuPointerGetAttribute(&base_ptr,
                                      CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                      (CUdeviceptr)ptr) != CUDA_SUCCESS)
                throw std::runtime_error("failed to get pointer attr");
            CUDACHECK(cudaIpcGetMemHandle(
                (cudaIpcMemHandle_t *)&handles[i * handle_sz], base_ptr));
            offsets[i] = ((char *)ptr) - ((char *)base_ptr);
        }
        return std::make_pair(handles, offsets);
    }

    void check_rank_data_capacity(size_t num = 1) {
        if (d_rank_data_base_ + num > d_rank_data_end_)
            throw std::runtime_error(
                "Rank data buffer is overflowed by " +
                std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
    }

    /**
     * Register already-shared IPC pointers.
     */
    void register_buffer(void **ptrs) {
        check_rank_data_capacity();
        RankData data;
        for (int i = 0; i < world_size_; i++) {
            data.ptrs[i] = ptrs[i];
        }
        auto d_data = d_rank_data_base_++;
        CUDACHECK(cudaMemcpy(d_data, &data, sizeof(RankData),
                             cudaMemcpyHostToDevice));
        buffers_[ptrs[rank_]] = d_data;
    }

    // Note: when registering graph buffers, we intentionally choose to not
    // deduplicate the addresses. That means if the allocator reuses some
    // addresses, they will be registered again. This is to account for the
    // remote possibility of different allocation patterns between ranks. For
    // example, rank 1 may get the same input address for the second allreduce,
    // but rank 2 got a different address. IPC handles have internal reference
    // counting mechanism so overhead should be small.
    void
    register_graph_buffers(const std::vector<std::string> &handles,
                           const std::vector<std::vector<int64_t>> &offsets) {
        auto num_buffers = graph_unreg_buffers_.size();
        check_rank_data_capacity(num_buffers);
        std::vector<RankData> rank_data(num_buffers);
        for (int i = 0; i < num_buffers; i++) {
            auto self_ptr = graph_unreg_buffers_[i];
            auto &rd = rank_data[i];
            for (int j = 0; j < world_size_; j++) {
                if (j != rank_) {
                    char *handle = open_ipc_handle(
                        &handles[j][i * sizeof(cudaIpcMemHandle_t)]);
                    handle += offsets[j][i];
                    rd.ptrs[j] = handle;
                } else {
                    rd.ptrs[j] = self_ptr;
                }
            }
        }
        CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                             sizeof(RankData) * num_buffers,
                             cudaMemcpyHostToDevice));
        d_rank_data_base_ += num_buffers;
        graph_unreg_buffers_.clear();
    }

#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
    RankData *registered_rank_data(void *input) {
        auto it = buffers_.find(input);
        if (it == buffers_.end())
            throw std::runtime_error(
                "variable-length collective input address " +
                std::to_string(reinterpret_cast<uint64_t>(input)) +
                " is not IPC-registered");
        return it->second;
    }

    template <typename T>
    void varlen_all_gather(cudaStream_t stream, T *input,
                           const int *local_count, T *output,
                           int global_capacity, int row_elems) {
        const int pack_elems = packed_t<T>::P::size;
        if (world_size_ != 8)
            throw std::runtime_error(
                "variable-length collectives currently require world_size=8");
        if (global_capacity <= 0 || row_elems <= 0)
            throw std::runtime_error(
                "global_capacity and row_elems must both be positive");
        if (row_elems % pack_elems != 0)
            throw std::runtime_error(
                "row length must be a multiple of the 16-byte pack width");

        constexpr int threads = 256;
        constexpr int blocks = 8;
        auto ptrs = registered_rank_data(input);
        cross_device_varlen_all_gather<T, 8><<<blocks, threads, 0, stream>>>(
            ptrs, sg_, self_sg_, local_count, output, rank_, global_capacity,
            row_elems);
    }

    template <typename T>
    void varlen_reduce_scatter(cudaStream_t stream, T *input,
                               const int *local_count, T *output,
                               int global_capacity, int row_elems) {
        const int pack_elems = packed_t<T>::P::size;
        if (world_size_ != 8)
            throw std::runtime_error(
                "variable-length collectives currently require world_size=8");
        if (global_capacity <= 0 || row_elems <= 0)
            throw std::runtime_error(
                "global_capacity and row_elems must both be positive");
        if (row_elems % pack_elems != 0)
            throw std::runtime_error(
                "row length must be a multiple of the 16-byte pack width");

        constexpr int threads = 256;
        constexpr int blocks = 8;
        auto ptrs = registered_rank_data(input);
        cross_device_varlen_reduce_scatter<T, 8>
            <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, local_count,
                                             output, rank_, global_capacity,
                                             row_elems);
    }
#endif

    /**
     * Performs allreduce, assuming input has already been registered.
     *
     * Block and grid default configs are results after careful grid search.
     * Using 36 blocks give the best or close to the best runtime on the devices
     * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
     * only take a small amount of SMs. Not quite sure the underlying reason,
     * but my guess is that too many SMs will cause contention on NVLink bus.
     */
    template <typename T>
    void allreduce(cudaStream_t stream, T *input, T *output, int size,
                   int threads = 512, int block_limit = 36) {
        auto d = packed_t<T>::P::size;
        if (size % d != 0)
            throw std::runtime_error("custom allreduce currently requires "
                                     "input length to be multiple "
                                     "of " +
                                     std::to_string(d));
        if (block_limit > kMaxBlocks)
            throw std::runtime_error("max supported block limit is " +
                                     std::to_string(kMaxBlocks) + ". Got " +
                                     std::to_string(block_limit));

        RankData *ptrs;
        cudaStreamCaptureStatus status;
        CUDACHECK(cudaStreamIsCapturing(stream, &status));
        auto registered_it = buffers_.find(input);
#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
        // Hygon graph capture stages input into a fixed uncached IPC buffer.
        // Prefer that pre-registered buffer; an unregistered address still
        // follows the generic graph-registration path.
        const bool use_graph_rank_data =
            status == cudaStreamCaptureStatusActive &&
            registered_it == buffers_.end();
#else
        // Preserve the upstream CUDA capture-first semantics. In particular,
        // never deduplicate graph inputs by virtual address across captures.
        const bool use_graph_rank_data =
            status == cudaStreamCaptureStatusActive;
#endif
        if (use_graph_rank_data) {
            ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
            graph_unreg_buffers_.push_back(input);
        } else {
            if (registered_it == buffers_.end())
                throw std::runtime_error(
                    "buffer address " +
                    std::to_string(reinterpret_cast<uint64_t>(input)) +
                    " is not registered!");
            ptrs = registered_it->second;
        }

        size /= d;
        auto bytes = size * sizeof(typename packed_t<T>::P);
        int blocks = std::min(block_limit, (size + threads - 1) / threads);
#define KL(ngpus, name)                                                        \
    name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_,        \
                                                   output, rank_, size);
        // TODO(hanzhi713): Threshold is different for A100 and H100.
        // Add per device threshold.
#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
        // TP8 measurements on Hygon HSW put the crossover between 126 KiB and
        // 140 KiB. Use 128 KiB as the dispatch boundary, and keep the existing
        // two-stage behavior for other, unmeasured world sizes.
        if (!full_nvlink_)
            throw std::runtime_error(
                "Hygon custom allreduce requires a fully connected "
                "peer topology");
        constexpr size_t kHygonTp8OneStageMaxBytes = 128 * 1024;
#define REDUCE_CASE(ngpus)                                                     \
    case ngpus: {                                                              \
        if (world_size_ == 8 && bytes <= kHygonTp8OneStageMaxBytes) {          \
            KL(ngpus, cross_device_reduce_1stage);                             \
        } else {                                                               \
            KL(ngpus, cross_device_reduce_2stage);                             \
        }                                                                      \
        break;                                                                 \
    }
#else
#define REDUCE_CASE(ngpus)                                                     \
    case ngpus: {                                                              \
        if (world_size_ == 2) {                                                \
            KL(ngpus, cross_device_reduce_1stage);                             \
        } else if (full_nvlink_) {                                             \
            if ((world_size_ <= 4 && bytes < 512 * 1024) ||                    \
                (world_size_ <= 8 && bytes < 256 * 1024)) {                    \
                KL(ngpus, cross_device_reduce_1stage);                         \
            } else {                                                           \
                KL(ngpus, cross_device_reduce_2stage);                         \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
#endif

        switch (world_size_) {
            REDUCE_CASE(2)
            REDUCE_CASE(4)
            REDUCE_CASE(6)
            REDUCE_CASE(8)
        default:
            throw std::runtime_error("custom allreduce only supports num gpus "
                                     "in (2,4,6,8). Actual num "
                                     "gpus = " +
                                     std::to_string(world_size_));
        }
#undef REDUCE_CASE
#undef KL
    }

    ~CustomAllreduce() {
        for (auto [_, ptr] : ipc_handles_) {
            CUDACHECK(cudaIpcCloseMemHandle(ptr));
        }
    }
};
/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and add
 a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 half *, int, int, int);
*/
} // namespace chitu
