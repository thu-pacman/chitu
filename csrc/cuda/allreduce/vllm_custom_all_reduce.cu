// SPDX-FileCopyrightText: 2025 vLLM Team
// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file has adaption of open-source code from the following sources:
 * - The kernel is originally from vLLM
 *   (https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cu),
 *   licensed under Apache 2.0.
 */
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "vllm_custom_all_reduce.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void *) == sizeof(fptr_t));

namespace chitu {

fptr_t init_custom_ar(const std::vector<fptr_t> &fake_ipc_ptrs,
                      torch::Tensor &rank_data, int64_t rank,
                      bool full_nvlink) {
    int world_size = fake_ipc_ptrs.size();
    if (world_size > 8)
        throw std::invalid_argument("world size > 8 is not supported");
    if (world_size % 2 != 0)
        throw std::invalid_argument("Odd num gpus is not supported for now");
    if (rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");

    chitu::Signal *ipc_ptrs[8];
    for (int i = 0; i < world_size; i++) {
        ipc_ptrs[i] = reinterpret_cast<chitu::Signal *>(fake_ipc_ptrs[i]);
    }
    return (fptr_t) new chitu::CustomAllreduce(ipc_ptrs, rank_data.data_ptr(),
                                               rank_data.numel(), rank,
                                               world_size, full_nvlink);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::Tensor &t) {
    return t.is_contiguous() ||
           (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
            t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
    auto fa = reinterpret_cast<chitu::CustomAllreduce *>(_fa);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
    TORCH_CHECK_EQ(inp.numel(), out.numel());
    TORCH_CHECK(_is_weak_contiguous(out));
    TORCH_CHECK(_is_weak_contiguous(inp));
    auto input_size = inp.numel() * inp.element_size();
    auto reg_buffer = reinterpret_cast<void *>(_reg_buffer);
    if (reg_buffer) {
        TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
        AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                      cudaMemcpyDeviceToDevice, stream));
    } else {
        TORCH_CHECK(inp.data_ptr() != out.data_ptr(),
                    "custom allreduce graph-registered mode requires distinct "
                    "input and output tensors");
        reg_buffer = inp.data_ptr();
    }
    switch (out.scalar_type()) {
    case at::ScalarType::Float: {
        fa->allreduce<float>(stream, reinterpret_cast<float *>(reg_buffer),
                             reinterpret_cast<float *>(out.data_ptr()),
                             out.numel());
        break;
    }
    case at::ScalarType::Half: {
        fa->allreduce<half>(stream, reinterpret_cast<half *>(reg_buffer),
                            reinterpret_cast<half *>(out.data_ptr()),
                            out.numel());
        break;
    }
#if defined(__HIP_PLATFORM_AMD__) ||                                           \
    (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
        fa->allreduce<nv_bfloat16>(
            stream, reinterpret_cast<nv_bfloat16 *>(reg_buffer),
            reinterpret_cast<nv_bfloat16 *>(out.data_ptr()), out.numel());
        break;
    }
#endif
    default:
        throw std::runtime_error(
            "custom allreduce only supports float32, float16 and bfloat16");
    }
}

#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
int64_t hygon_varlen_collective_abi_version() { return 2; }

namespace {

void check_varlen_common(chitu::CustomAllreduce *fa, torch::Tensor &inp,
                         torch::Tensor &out, torch::Tensor &local_count) {
    TORCH_CHECK(fa != nullptr, "custom collective handle is null");
    TORCH_CHECK(fa->world_size_ == 8,
                "variable-length collectives currently require 8 ranks");
    TORCH_CHECK(inp.dim() == 2, "input must be a 2-D tensor");
    TORCH_CHECK(out.dim() == 2, "output must be a 2-D tensor");
    TORCH_CHECK(inp.scalar_type() == at::ScalarType::BFloat16,
                "variable-length collectives support BF16 only");
    TORCH_CHECK_EQ(out.scalar_type(), inp.scalar_type());
    TORCH_CHECK(
        inp.data_ptr() != out.data_ptr(),
        "variable-length collectives require distinct input and output");
    TORCH_CHECK(inp.is_cuda() && out.is_cuda() && local_count.is_cuda(),
                "input, output and local_count must all be GPU tensors");
    TORCH_CHECK_EQ(inp.device(), out.device());
    TORCH_CHECK_EQ(inp.device(), local_count.device());
    TORCH_CHECK(inp.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(local_count.is_contiguous());
    TORCH_CHECK(local_count.scalar_type() == at::ScalarType::Int,
                "local_count must be a device int32 scalar");
    TORCH_CHECK(local_count.numel() == 1,
                "local_count must contain exactly one int32 value");
    TORCH_CHECK_EQ(inp.size(0), out.size(0));
    TORCH_CHECK_GT(inp.size(0), 0);
    TORCH_CHECK_EQ(inp.size(1), out.size(1));
    TORCH_CHECK_GT(inp.size(1), 0);
    TORCH_CHECK_LE(inp.size(0), std::numeric_limits<int>::max());
    TORCH_CHECK_LE(inp.size(1), std::numeric_limits<int>::max());
    TORCH_CHECK_LE(inp.numel(), std::numeric_limits<int>::max());
    TORCH_CHECK_LE(out.numel(), std::numeric_limits<int>::max());
    TORCH_CHECK((inp.size(1) * inp.element_size()) % 16 == 0,
                "each row must be an integer number of 16-byte packs");
}

void *prepare_varlen_input(torch::Tensor &inp, fptr_t fake_reg_buffer,
                           int64_t reg_buffer_sz_bytes, cudaStream_t stream) {
    void *reg_buffer = reinterpret_cast<void *>(fake_reg_buffer);
    TORCH_CHECK(reg_buffer != nullptr,
                "variable-length collectives require the registered staging "
                "buffer");

    const int64_t input_size = inp.numel() * inp.element_size();
    TORCH_CHECK(reg_buffer_sz_bytes >= input_size,
                "registered staging buffer is too small");
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
    return reg_buffer;
}

} // namespace

void varlen_all_gather(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out,
                       torch::Tensor &local_count, fptr_t _reg_buffer,
                       int64_t reg_buffer_sz_bytes) {
    auto fa = reinterpret_cast<chitu::CustomAllreduce *>(_fa);
    TORCH_CHECK(out.dim() == 2 && inp.dim() == 2,
                "all-gather input and output must be 2-D tensors");
    TORCH_CHECK_EQ(out.size(0), inp.size(0));
    check_varlen_common(fa, inp, out, local_count);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    void *input =
        prepare_varlen_input(inp, _reg_buffer, reg_buffer_sz_bytes, stream);
    fa->varlen_all_gather<nv_bfloat16>(
        stream, reinterpret_cast<nv_bfloat16 *>(input),
        reinterpret_cast<const int *>(local_count.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(out.data_ptr()),
        static_cast<int>(inp.size(0)), static_cast<int>(inp.size(1)));
    AT_CUDA_CHECK(cudaGetLastError());
}

void varlen_reduce_scatter(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out,
                           torch::Tensor &local_count, fptr_t _reg_buffer,
                           int64_t reg_buffer_sz_bytes) {
    auto fa = reinterpret_cast<chitu::CustomAllreduce *>(_fa);
    TORCH_CHECK(out.dim() == 2 && inp.dim() == 2,
                "reduce-scatter input and output must be 2-D tensors");
    TORCH_CHECK_EQ(inp.size(0), out.size(0));
    check_varlen_common(fa, inp, out, local_count);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    void *input =
        prepare_varlen_input(inp, _reg_buffer, reg_buffer_sz_bytes, stream);
    fa->varlen_reduce_scatter<nv_bfloat16>(
        stream, reinterpret_cast<nv_bfloat16 *>(input),
        reinterpret_cast<const int *>(local_count.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(out.data_ptr()),
        static_cast<int>(out.size(0)), static_cast<int>(out.size(1)));
    AT_CUDA_CHECK(cudaGetLastError());
}
#endif

void dispose(fptr_t _fa) {
    delete reinterpret_cast<chitu::CustomAllreduce *>(_fa);
}

int64_t meta_size() { return sizeof(chitu::Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t> &fake_ipc_ptrs) {
    auto fa = reinterpret_cast<chitu::CustomAllreduce *>(_fa);
    TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
    void *ipc_ptrs[8];
    for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
        ipc_ptrs[i] = reinterpret_cast<void *>(fake_ipc_ptrs[i]);
    }
    fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
    auto fa = reinterpret_cast<chitu::CustomAllreduce *>(_fa);
    auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
    std::vector<int64_t> bytes(handle.begin(), handle.end());
    return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>> &handles,
                            const std::vector<std::vector<int64_t>> &offsets) {
    auto fa = reinterpret_cast<chitu::CustomAllreduce *>(_fa);
    std::vector<std::string> bytes;
    bytes.reserve(handles.size());
    for (int i = 0; i < handles.size(); i++) {
        bytes.emplace_back(handles[i].begin(), handles[i].end());
    }
    bytes.reserve(handles.size());
    fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::Tensor>
allocate_shared_buffer_and_handle(int64_t size) {
    auto device_index = c10::cuda::current_device();
    at::DeviceGuard device_guard(
        at::Device(at::DeviceType::CUDA, device_index));
    void *buffer;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

// Allocate buffer
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // data buffers need to be "uncached" for signal on MI200
    AT_CUDA_CHECK(
        hipExtMallocWithFlags((void **)&buffer, size, hipDeviceMallocUncached));
#else
    AT_CUDA_CHECK(cudaMalloc((void **)&buffer, size));
#endif
    AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

    // Create IPC memhandle for the allocated buffer.
    // Will use it in open_mem_handle.
    auto options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto handle = torch::empty(
        {static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
    AT_CUDA_CHECK(
        cudaIpcGetMemHandle((cudaIpcMemHandle_t *)handle.data_ptr(), buffer));

    return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::Tensor &mem_handle) {
    void *ipc_ptr;
    AT_CUDA_CHECK(cudaIpcOpenMemHandle(
        (void **)&ipc_ptr, *((const cudaIpcMemHandle_t *)mem_handle.data_ptr()),
        chituIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
    AT_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(buffer)));
}

} // namespace chitu
