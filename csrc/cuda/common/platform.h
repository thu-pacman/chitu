/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Platform detection and abstraction for CUDA vs HIP (ROCm/Hygon)

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
// HIP platform (ROCm/Hygon)
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Provide nv_bfloat16 as an alias for source compatibility
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
using nv_bfloat16_2 = __hip_bfloat162;
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

// Note: __float2bfloat16 and __bfloat162float are already defined in hip_bf16.h
// They accept __hip_bfloat16 which matches our type alias

// CUDA compatibility aliases for types
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaIpcMemHandle_t = hipIpcMemHandle_t;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using CUdeviceptr = void *;

// CUDA enum value compatibility - use #define to avoid hipify transforming
// variable names
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive

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
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaSetDevice hipSetDevice

#else
// CUDA platform (NVIDIA)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#endif
