#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <cub/util_type.cuh>

#include "common.h"
#include "moe_kernel.h"

namespace chitu {

template <int TPB, size_t NUM_GROUPS, size_t TOPK_GROUPS, size_t TOPK,
          typename T>
__launch_bounds__(TPB) __global__
    void group_topk_kernel_gather_weights(const T *__restrict__ input_after_act,
                                          const T *__restrict__ original_scores,
                                          float route_scale, T *weights,
                                          int64_t *indices,
                                          const size_t group_dim) {
    using cub_kvp = cub::KeyValuePair<int64_t, T>;
    cub_kvp thread_kvp;
    cub::ArgMax arg_max;
    size_t cur_token_idx = blockIdx.x;
    T group_top2_sum[NUM_GROUPS];
    int64_t topk_group_indices[TOPK_GROUPS];
    int64_t topk_indices[TOPK];
    // 计算group_top2_sum
    for (size_t group_idx = 0; group_idx < NUM_GROUPS; group_idx++) {
        group_top2_sum[group_idx] = 0.f;
        int64_t prev_k = -1;
        for (int64_t k_idx = 0; k_idx < 2; k_idx++) {
            thread_kvp.key = 0;
            thread_kvp.value = -1.f;
            cub_kvp inp_kvp;
            for (int i = 0; i < group_dim; ++i) {
                size_t idx = cur_token_idx * NUM_GROUPS * group_dim +
                             group_idx * group_dim + i;
                inp_kvp.key = i;
                inp_kvp.value = input_after_act[idx];
                if (prev_k == i) {
                    inp_kvp = thread_kvp;
                }
                thread_kvp = arg_max(thread_kvp, inp_kvp);
                // prev_k = inp_kvp.key;
            }
            prev_k = thread_kvp.key;
            group_top2_sum[group_idx] += thread_kvp.value;
        }
    }

    for (size_t topk_group_idx = 0; topk_group_idx < TOPK_GROUPS;
         ++topk_group_idx) {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f;
        cub_kvp inp_kvp;
        for (size_t group_idx = 0; group_idx < NUM_GROUPS; group_idx++) {
            inp_kvp.key = group_idx;
            inp_kvp.value = group_top2_sum[group_idx];
            for (size_t prior_k = 0; prior_k < topk_group_idx; ++prior_k) {
                const int64_t prior_group_idx = topk_group_indices[prior_k];
                if (prior_group_idx == inp_kvp.key) {
                    inp_kvp = thread_kvp;
                }
            }
            thread_kvp = arg_max(thread_kvp, inp_kvp);
        }
        topk_group_indices[topk_group_idx] = thread_kvp.key;
    }

    size_t cur_token_start = cur_token_idx * NUM_GROUPS * group_dim;
    float weights_sum = 0.f;
    for (size_t topK_idx = 0; topK_idx < TOPK; ++topK_idx) {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f;
        cub_kvp inp_kvp;
        for (size_t topk_group_idx = 0; topk_group_idx < TOPK_GROUPS;
             ++topk_group_idx) {
            for (size_t i = 0; i < group_dim; ++i) {
                size_t group_idx =
                    topk_group_indices[topk_group_idx] * group_dim + i;
                size_t idx = cur_token_start + group_idx;
                inp_kvp.key = group_idx;
                inp_kvp.value = input_after_act[idx];
                for (size_t prior_k = 0; prior_k < topK_idx; ++prior_k) {
                    const int64_t prior_wining_expert = topk_indices[prior_k];
                    if (prior_wining_expert == inp_kvp.key) {
                        inp_kvp = thread_kvp;
                    }
                }
                thread_kvp = arg_max(thread_kvp, inp_kvp);
            }
        }
        topk_indices[topK_idx] = thread_kvp.key;
        auto &&weight = original_scores[cur_token_start + thread_kvp.key];
        weights[cur_token_idx * TOPK + topK_idx] = weight;
        weights_sum += weight;
        indices[cur_token_idx * TOPK + topK_idx] = thread_kvp.key;
    }
    for (size_t topK_idx = 0; topK_idx < TOPK; ++topK_idx) {
        weights[cur_token_idx * TOPK + topK_idx] /= weights_sum;
        weights[cur_token_idx * TOPK + topK_idx] *= route_scale;
    }
}

template <typename T>
void groupTopKLauncher(torch::Tensor &input_after_act,
                       torch::Tensor &original_scores, float route_scale,
                       torch::Tensor &weights, torch::Tensor &indices,
                       const size_t num_groups, cudaStream_t stream) {
    const int TPB = 1;
    const size_t NUM_GROUPS = num_groups;
    const int group_dim = input_after_act.size(-1);
    const int num_tokens = input_after_act.size(0);
    const int num_blocks = num_tokens;
    group_topk_kernel_gather_weights<TPB, 8, 4, 8, T>
        <<<num_blocks, TPB, 0, stream>>>(
            input_after_act.data_ptr<T>(), original_scores.data_ptr<T>(),
            route_scale, weights.data_ptr<T>(), indices.data_ptr<int64_t>(),
            group_dim);
}

void groupTopKIndices(torch::Tensor &input_after_act,
                      torch::Tensor &original_scores, float route_scale,
                      torch::Tensor &weights, torch::Tensor &indices,
                      const int num_groups) {
    ASSERTWITH(input_after_act.scalar_type() == original_scores.scalar_type(),
               "`groupTopKIndices` requires all floating point tensors to have "
               "the same dtype");
    ASSERTWITH(input_after_act.scalar_type() == weights.scalar_type(),
               "`groupTopKIndices` requires all floating point tensors to have "
               "the same dtype");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_after_act));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(
        input_after_act.scalar_type(), "groupTopKIndices", [&] {
            groupTopKLauncher<scalar_t>(input_after_act, original_scores,
                                        route_scale, weights, indices,
                                        num_groups, stream);
        });
}

} // namespace chitu
