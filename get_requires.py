import os

import csrc.setup_build as operators

setup_dir = os.path.dirname(os.path.abspath(__file__))

install_requires = [
    # Don't put `torch` here because it requires downloading from a specific source
    "transformers",
    "fire",
    "tiktoken>=0.7.0",  # Required by glm4
    "blobfile",
    "faker",
    "hydra-core",
    "fastapi",
    "uvicorn",
    "tqdm",
    "accelerate",
    "einops",
    "typing-extensions",
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
        "grouped_gemm @ file://localhost"
        + os.path.join(setup_dir, "third_party/ascend-kernel/grouped_gemm"),
    ],
    ##########################################################################
    # Really third-party kernels
    "flash_attn": [
        "flash-attn<2.8.0",
        # Although `flash-attn` is available in PyPI, don't make it a required
        # dependency, because its installation runs forever on some platforms.
    ],
    "flashinfer": [
        "flashinfer-python<=0.2.5",  # Later versions require a too-new torch
    ],
    "flash_mla": [
        "flash_mla @ file://localhost"
        + os.path.join(setup_dir, "third_party/FlashMLA"),
    ],
    "deep_gemm": [
        "deep_gemm @ file://localhost"
        + os.path.join(setup_dir, "third_party/DeepGEMM"),
    ],
    **operators.get_extras_require(),
}
