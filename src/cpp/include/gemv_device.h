#pragma once
#include "common.h"

#define BLOCK_SIZE 32

__global__ void matvec_bfloat16(const __nv_bfloat16 *__restrict__ matrix,
                                const __nv_bfloat16 *__restrict__ vector,
                                __nv_bfloat16 *__restrict__ result, int num_output, int num_input);