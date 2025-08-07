/*
 * SPDX-FileCopyrightText: 2025 kvcache-ai
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * This file has adaption of open-source code from the following sources:
 * - https://github.com/kvcache-ai/ktransformers, licensed under Apache 2.0.
 */

#ifndef CPUINFER_CONVERSION_H
#define CPUINFER_CONVERSION_H

#include "llama.cpp/ggml.h"
#include <memory.h>

inline void to_float(const void *input, float *output, int size,
                     ggml_type type) {
    if (type == ggml_type::GGML_TYPE_F32) {
        memcpy(output, input, size * sizeof(float));
    } else {
        ggml_internal_get_type_traits(type).to_float(input, output, size);
    }
}

inline void from_float(const float *input, void *output, int size,
                       ggml_type type) {
    if (type == ggml_type::GGML_TYPE_F32) {
        memcpy(output, input, size * sizeof(float));
    } else {
        ggml_internal_get_type_traits(type).from_float(input, output, size);
    }
}

#endif
