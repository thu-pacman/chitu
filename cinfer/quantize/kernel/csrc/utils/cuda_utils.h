/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "logger.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fastertransformer {
/* **************************** debug tools ********************************* */
template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + ("<unknown>") + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw std::runtime_error(std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void myAssert(bool result, const char* const file, int const line, std::string const& info = "")
{
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

#define FT_CHECK(val) myAssert(val, __FILE__, __LINE__)
#define FT_CHECK_WITH_INFO(val, info)                                                                                  \
    do {                                                                                                               \
        bool is_valid_val = (val);                                                                                     \
        if (!is_valid_val) {                                                                                           \
            fastertransformer::myAssert(is_valid_val, __FILE__, __LINE__, (info));                                     \
        }                                                                                                              \
    } while (0)

/* ***************************** common utils ****************************** */
inline int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

cudaError_t getSetDevice(int i_device, int* o_device = NULL);
/* ************************** end of common utils ************************** */
}  // namespace fastertransformer
