# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import packaging.version
import torch

import csrc.setup_build as operators

setup_dir = os.path.dirname(os.path.abspath(__file__))

install_requires = [
    # Special notes on torch:
    # 1. Users should not expect the installing of chitu to automatically
    #    install torch, because different variants of torch should be used
    #    on different platforms.
    # 2. For the same reason, we don't set specific torch version here.
    # 3. In order to prevent `pip` from upgrading your platform-specifc
    #    torch back to the official version, please use `-c` on `pip`.
    "torch",
    "torchvision",
    "transformers[torch]>=4.49.0,<4.57.0",  # >=4.49.0 required by gptqmodel when quant enabled
    "safetensors",
    "fire",
    "tiktoken>=0.7.0",  # Required by glm4
    "blobfile",
    "faker",
    "hydra-core",
    "fastapi",
    "uvicorn[standard]",
    "tqdm",
    "einops",
    "typing-extensions",
    "pyzmq>=27.0.0",
    "msgpack",
    "plum-dispatch",
    "netifaces",
    "prometheus-client>=0.19.0",
]


extras_require = {
    "quant": [
        "optimum",
        "bitsandbytes",
        "autoawq-kernels==0.0.8",
        "autoawq[kernels]",
        "transformers>=4.49.0,<=4.56.2",  # >=4.49.0 required by gptqmodel 2.2.0, <=4.56.2 required by autoawq-kernels==0.0.8
        "gptqmodel>=2.2.0,<4.0.0",  # <4.0.0: gptqmodel 4.x requires torch>=2.7.1 and numpy>=2.2.6, incompatible with base image torch==2.7.0
        "tokenizers>=0.20.3",
    ],
    ##########################################################################
    # Our own kernels for various architectures
    "muxi_layout_kernels": [
        "muxi_layout_kernels @ file://localhost"
        + os.path.join(setup_dir, "third_party/muxi_layout_kernels"),
    ],
    "muxi_w8a8_kernels": [
        "tbsgemm @ file://localhost"
        + os.path.join(setup_dir, "third_party/muxi_w8a8_kernels/w8a8"),
    ],
    "ascend_kernels": [
        "cinfer_ascendc @ file://localhost"
        + os.path.join(setup_dir, "third_party/ascend-kernel"),
    ],
    "sugon_mixq4_kernels": [
        "sugon_mixq4_kernels @ file://localhost"
        + os.path.join(setup_dir, "third_party/sugon_mixq4_kernels"),
    ],
    ##########################################################################
    # Really third-party kernels
    "flash_attn": [
        (
            "flash-attn<2.8.0"
            if packaging.version.parse(torch.__version__)
            < packaging.version.parse("2.7.0")
            else "flash-attn"
        ),
        # Although `flash-attn` is available in PyPI, don't make it a required
        # dependency, because its installation runs forever on some platforms.
    ],
    # TODO: Upgrade to latest flashInfer version and resolve environment compatibility issues
    "flashinfer": [
        (
            "flashinfer-python<=0.2.5"
            if packaging.version.parse(torch.__version__)
            < packaging.version.parse("2.7.0")
            else "flashinfer-python<=0.2.7.post1,!=0.2.6"
            # !=0.2.6: https://github.com/flashinfer-ai/flashinfer/issues/1139
        ),
    ],
    "flash_linear_attention": [
        "flash-linear-attention",
    ],
    "flash_mla": [
        "flash_mla @ file://localhost"
        + os.path.join(setup_dir, "third_party/FlashMLA"),
    ],
    "deep_gemm": [
        "deep_gemm @ file://localhost"
        + os.path.join(setup_dir, "third_party/DeepGEMM"),
    ],
    "deep_ep": [
        "deep_ep @ file://localhost" + os.path.join(setup_dir, "third_party/DeepEP"),
    ],  # Please make sure `requirements-build-deep_ep-cu12.txt` is installed at BUILD TIME
    "hard_fp4_kernels": [
        "hard_fp4_kernels @ file://localhost"
        + os.path.join(setup_dir, "third_party/hard_fp4_kernels"),
    ],
    "scipy": ["scipy"],
    "fast_hadamard_transform": [
        "fast-hadamard-transform @ file://localhost"
        + os.path.join(setup_dir, "third_party/fast-hadamard-transform")
    ],
    **operators.get_extras_require(),
}
