# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from torch.utils.cpp_extension import CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))


def get_extensions():
    return [
        CUDAExtension(
            name="chitu_backend",
            sources=[
                os.path.join(this_dir, "cuda/binding.cpp"),
                os.path.join(this_dir, "cuda/moe/moe_align_kernel.cu"),
                os.path.join(this_dir, "cuda/moe/fused_shared_experts_kernel.cu"),
                os.path.join(this_dir, "cuda/moe/group_topk.cu"),
                os.path.join(this_dir, "cuda/moe/vllm_topk_softmax.cu"),
                os.path.join(this_dir, "cuda/rotary/rotary_pos_emb_llama.cu"),
                os.path.join(this_dir, "cuda/norm/rms_norm.cu"),
                os.path.join(this_dir, "cuda/frequency_penalty/frequency_penalty.cu"),
                os.path.join(this_dir, "cuda/response_append/response_append.cu"),
                os.path.join(this_dir, "cuda/weight_layout/weight_layout_change.cu"),
                os.path.join(this_dir, "cuda/dequant/dequant.cu"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17"],
            },
            include_dirs=[
                os.path.join(this_dir, "../third_party/spdlog/include"),
                os.path.join(this_dir, "cuda/common"),
            ],
        )
    ]


def get_extras_require():
    return {
        "cpu": [
            "numa",
            "cpuinfer @ file://localhost"
            + os.path.abspath(os.path.join(this_dir, "cpuinfer")),
        ],
    }
