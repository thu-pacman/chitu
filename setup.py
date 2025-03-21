import os
import subprocess
import sys

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    # We use CUDAExtension instead of CMake for native sources, because many of the non-NVIDIA GPUs have
    # their custom CUDAExtension, but not their custom CMake support.
    CUDAExtension(
        name="chitu_backend",
        sources=[
            "./csrc/binding.cpp",
            "./csrc/moe_align_kernel.cu",
            "./csrc/fused_shared_experts_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-std=c++17"],
            "nvcc": ["-std=c++17"],
        },
        include_dirs=[os.path.join(setup_dir, "third_party/spdlog/include")],
    )
]


cython_unsafe_files = [
    "triton_kernels.py",
    "fused_moe.py",
    "triton_decode_attention.py",
]


def is_cython_unsafe(path):
    for unsafe_file in cython_unsafe_files:
        if str(path).endswith(unsafe_file):
            return True
    return False


def find_py_modules(directory):
    modules = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                if not is_cython_unsafe(os.path.join(root, file)):
                    module_name = os.path.splitext(os.path.join(root, file))[0].replace(
                        os.sep, "."
                    )
                    modules.append(module_name)
    return modules


def create_cython_extensions(directory):
    extensions = []
    for module in find_py_modules(directory):
        extension = Extension(module, [module.replace(".", os.sep) + ".py"])
        extensions.append(extension)
    return extensions


class SkipBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = [
            (pkg, mod, file) for (pkg, mod, file) in modules if is_cython_unsafe(file)
        ]
        return filtered_modules


my_build_py = build_py
if os.environ.get("CINFER_WITH_CYTHON", "0") != "0":
    ext_modules += cythonize(create_cython_extensions("chitu"))
    my_build_py = SkipBuildPy

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="chitu",
    version="0.1.0",
    install_requires=[
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
    ],
    extras_require={
        "quant": [
            "optimum",
            "bitsandbytes",
            "EETQ @ file://localhost" + os.path.join(setup_dir, "third_party/EETQ"),
            "awq_inference_engine @ file://localhost"
            + os.path.join(setup_dir, "third_party/llm-awq/awq/kernels"),
            "auto_gptq @ file://localhost"
            + os.path.join(setup_dir, "third_party/AutoGPTQ"),
            "w8a8gemm @ file://localhost"
            + os.path.join(setup_dir, "third_party/nv_w8a8_kernels/w8a8gemm"),
            "w8a8gemv @ file://localhost"
            + os.path.join(setup_dir, "third_party/nv_w8a8_kernels/w8a8gemv"),
        ],
        "muxi_layout_kernels": [
            "muxi_layout_kernels @ file://localhost"
            + os.path.join(setup_dir, "third_party/muxi_layout_kernels"),
        ],
        "muxi_w8a8_kernels": [
            "tbsgemm @ file://localhost"
            + os.path.join(setup_dir, "third_party/muxi_w8a8_kernels/w8a8"),
        ],
        "flash_attn": [
            "flash-attn",
            # Although `flash-attn` is available in PyPI, don't make it a required
            # dependency, because its installation runs forever on some platforms.
        ],
        "flashinfer": [
            "flashinfer-python",
        ],
        "flash_mla": [
            "flash_mla @ file://localhost"
            + os.path.join(setup_dir, "third_party/FlashMLA"),
        ],
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension, "build_py": my_build_py},
    package_data={"chitu": ["config/**/*.yaml"]},
)
