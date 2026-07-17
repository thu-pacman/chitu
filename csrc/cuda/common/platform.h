/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Platform detection and abstraction for CUDA vs HIP (ROCm/Hygon)

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// HIP platform (ROCm/Hygon)
// ---- _chitu_hygon_platform_host_safe_marker_ ----
// hip_bf16.h / hip_fp16.h transitively include device_library_decls.h, which
// uses _Float16 and __builtin_amdgcn_* — these require hipcc's device-mode
// predefines and are NOT host-safe. Include them only under device compilation
// (hipcc) so plain g++ host TUs (e.g. binding.cpp) keep building.
#include <hip/hip_runtime.h>
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// ---- _chitu_hygon_platform_v2_alias_guard_ ----
// These aliases reference __hip_bfloat16/__hip_bfloat162, which are only
// declared by hip_bf16.h above. Keep them under the same device-only guard so
// host TUs (g++ compiling binding.cpp) don't fail to resolve the target type.
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
using nv_bfloat16_2 = __hip_bfloat162;
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

// Note: __float2bfloat16 and __bfloat162float are already defined in hip_bf16.h
// They accept __hip_bfloat16 which matches our type alias

// CUDA compatibility aliases for types
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaIpcMemHandle_t = hipIpcMemHandle_t;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using CUdeviceptr = void *;

// CUDA enum value compatibility - use #define to avoid hipify transforming
// variable names
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed

// Use a project-local name because PyTorch's hipify rewrites identifiers in
// macro definitions and would create a self-referential HIP macro.
constexpr unsigned int chituIpcMemLazyEnablePeerAccess =
    hipIpcMemLazyEnablePeerAccess;

// CUDA driver API compatibility
#define CU_POINTER_ATTRIBUTE_RANGE_START_ADDR                                  \
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define CUDA_SUCCESS hipSuccess

// CUDA driver function compatibility
#define cuPointerGetAttribute hipPointerGetAttribute

// CUDA runtime API compatibility
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceReset hipDeviceReset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMalloc hipMalloc
#define cudaMemsetAsync hipMemsetAsync
#define cudaFree hipFree
#define cudaSetDevice hipSetDevice

#else
// CUDA platform (NVIDIA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr unsigned int chituIpcMemLazyEnablePeerAccess =
    cudaIpcMemLazyEnablePeerAccess;

#endif
