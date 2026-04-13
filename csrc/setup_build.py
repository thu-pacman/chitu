# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch.utils.cpp_extension import CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))


def generate_files():
    files_path = [
        os.path.join(this_dir, "cuda/marlin/marlin_gemm/generate_kernel.py"),
        os.path.join(this_dir, "cuda/marlin/marlin_group_gemm/generate_kernel.py"),
    ]
    import subprocess

    for file in files_path:
        subprocess.run(["python", file])


def remove_files():
    files_path = [
        os.path.join(this_dir, "cuda/marlin/marlin_gemm/bf16_kernel.cu"),
        os.path.join(this_dir, "cuda/marlin/marlin_group_gemm/bf16_kernel_moe.cu"),
        os.path.join(this_dir, "cuda/marlin/marlin_gemm/fp16_kernel.cu"),
        os.path.join(this_dir, "cuda/marlin/marlin_group_gemm/fp16_kernel_moe.cu"),
    ]
    for file in files_path:
        if os.path.exists(file):
            os.remove(file)


def get_extensions():
    cxx_extra_args = []
    nvcc_extra_args = []
    extra_sources = []
    extra_include_dirs = []

    if torch.compiled_with_cxx11_abi():
        cxx_extra_args.append("-D_GLIBCXX_USE_CXX11_ABI=1")
        nvcc_extra_args.append("-D_GLIBCXX_USE_CXX11_ABI=1")
    else:
        cxx_extra_args.append("-D_GLIBCXX_USE_CXX11_ABI=0")
        nvcc_extra_args.append("-D_GLIBCXX_USE_CXX11_ABI=0")

    muxi_build = os.environ.get("CHITU_MUXI_BUILD", "0").strip()
    ascend_build = os.environ.get("CHITU_ASCEND_BUILD", "0").strip()
    torch_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()

    enable_nvfp4 = os.environ.get("ENABLE_NVFP4", "0") == "1"
    enable_marlin = False
    if muxi_build == "0":
        for arch in torch_cuda_arch_list.split():
            if arch.startswith("8.") or arch.startswith("9."):
                enable_marlin = True
    enable_custom_all_reduce = (muxi_build == "0") and (ascend_build == "0")

    if enable_nvfp4:
        cutlass_path = os.path.join(this_dir, "../third_party/cutlass")
        cxx_extra_args += ["-DENABLE_NVFP4"]
        nvcc_extra_args += ["-DENABLE_NVFP4"]
        extra_include_dirs += [
            os.path.join(cutlass_path, "include"),
            os.path.join(cutlass_path, "tools/util/include"),
        ]
        extra_sources += [
            os.path.join(this_dir, "cuda/hard_fp4/nvfp4_scaled_mm_kernels.cu"),
            os.path.join(this_dir, "cuda/hard_fp4/nvfp4_quant_kernels.cu"),
        ]

    if enable_marlin:
        cxx_extra_args += ["-DENABLE_MARLIN"]
        nvcc_extra_args += [
            "-DENABLE_MARLIN",
            "-static-global-template-stub=false",  # Backward compatibility for CUDA <13
        ]
        generate_files()
        extra_sources += [
            os.path.join(this_dir, "cuda/marlin/marlin_gemm/gptq_marlin.cu"),
            os.path.join(this_dir, "cuda/marlin/marlin_gemm/bf16_kernel.cu"),
            os.path.join(this_dir, "cuda/marlin/marlin_gemm/fp16_kernel.cu"),
            os.path.join(this_dir, "cuda/marlin/marlin_group_gemm/ops.cu"),
            os.path.join(this_dir, "cuda/marlin/marlin_group_gemm/bf16_kernel_moe.cu"),
            os.path.join(this_dir, "cuda/marlin/marlin_group_gemm/fp16_kernel_moe.cu"),
        ]

    if enable_custom_all_reduce:
        extra_sources += [
            os.path.join(this_dir, "cuda/allreduce/vllm_custom_all_reduce.cu"),
        ]

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
                os.path.join(this_dir, "cuda/topk/topk.cu"),
                os.path.join(this_dir, "cuda/norm/rms_norm.cu"),
                os.path.join(this_dir, "cuda/frequency_penalty/frequency_penalty.cu"),
                os.path.join(this_dir, "cuda/response_append/response_append.cu"),
                os.path.join(this_dir, "cuda/weight_layout/weight_layout_change.cu"),
                os.path.join(this_dir, "cuda/dequant/dequant.cu"),
                os.path.join(this_dir, "cuda/gemm/w4a8_per_group_gemm_cuda.cu"),
            ]
            + extra_sources,
            extra_compile_args={
                "cxx": ["-std=c++17"] + cxx_extra_args,
                "nvcc": ["-std=c++17"] + nvcc_extra_args,
            },
            extra_link_args=["-lcuda"],
            define_macros=[
                ("CHITU_MUXI_BUILD", os.environ.get("CHITU_MUXI_BUILD", "0")),
            ],
            include_dirs=[
                os.path.join(this_dir, "../third_party/spdlog/include"),
                os.path.join(this_dir, "cuda/common"),
                os.path.join(this_dir, "cuda/allreduce"),
            ]
            + extra_include_dirs,
        )
    ]


def get_extras_require():
    return {
        "cpu": [
            "cpuinfer @ file://localhost"
            + os.path.abspath(os.path.join(this_dir, "cpuinfer")),
        ],
    }
