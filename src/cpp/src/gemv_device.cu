#ifdef CINFER_CUDA_GEMV

#include "gemv_device.h"

__global__ void matvec_bfloat16(const __nv_bfloat16 *__restrict__ matrix,
                                const __nv_bfloat16 *__restrict__ vector,
                                __nv_bfloat16 *__restrict__ result, int num_output, int num_input) {
    __shared__ __nv_bfloat16 shared_vector[BLOCK_SIZE];

    int output_id = blockIdx.x * blockDim.x + threadIdx.x;
    __nv_bfloat16 sum = __float2bfloat16(0.0f);
    #ifdef AVOID_ZERO
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    #endif

    int i = blockIdx.y;
    {
        int base = i * BLOCK_SIZE;
        shared_vector[threadIdx.x] = __ldg(&vector[base + threadIdx.x]);
        __syncthreads();
        if (output_id < num_output) {
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                #ifdef AVOID_ZERO
                if (shared_vector[j] == zero) continue;
                #endif
                int input_id = base + j;
                sum = __hadd(sum, __hmul((__ldg(&matrix[num_output * input_id + output_id])),
                                         shared_vector[j]));
            }
        }
    }
    if (output_id < num_output) {
        atomicAdd(result + output_id, sum);
    }
}

#endif // CINFER_CUDA_GEMV
