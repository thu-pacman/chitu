# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import packaging.version
import torch

import csrc.setup_build as operators

setup_dir = os.path.dirname(os.path.abspath(__file__))

install_requires = [
    # Don't put `torch` here because it requires downloading from a specific source
    "transformers",
    "safetensors<0.6",  # 0.6 breaks on muxi
    "fire",
    "tiktoken>=0.7.0",  # Required by glm4
    "blobfile",
    "faker",
    "hydra-core",
    "fastapi",
    "uvicorn",
    "tqdm",
    "accelerate<1.10",  # 1.10 breaks on muxi
    "einops",
    "typing-extensions",
    "pyzmq>=27.0.0",
]


extras_require = {
    "quant": [
        "optimum",
        "bitsandbytes",
        "autoawq-kernels==0.0.8",
        "autoawq[kernels]",
        "gptqmodel>=2.2.0",
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
    ],  # export NVSHMEM_DIR=/path/to/installed/nvshmem
    **operators.get_extras_require(),
}
