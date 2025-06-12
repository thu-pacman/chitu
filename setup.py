import os

import setuptools
from setuptools import Extension, setup, find_packages
from setuptools.command.build_py import build_py
import packaging.version
from Cython.Build import cythonize

try:
    import torch
except ImportError:
    raise RuntimeError(
        "torch is required to build chitu. Please install torch (with the correct CUDA version) before installing chitu.\n"
        "For example: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    )
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

assert packaging.version.parse(setuptools.__version__) >= packaging.version.parse(
    "62.3.0"
), "setuptools>=62.3.0 is required for `**` wildcard in package_data."

import csrc.setup_build as operators

setup_dir = os.path.dirname(os.path.abspath(__file__))


# We use CUDAExtension instead of CMake for native sources, because many of the non-NVIDIA GPUs have
# their custom CUDAExtension, but not their custom CMake support.

if os.environ.get("ASCEND_PLATFORM", "0") == "0":
    ext_modules = operators.get_extensions()
else:
    ext_modules = []


cython_unsafe_files = [
    "triton_kernels.py",  # Triton kernels inside
    "fused_moe.py",  # Triton kernels inside
    "triton_decode_attention.py",  # Triton kernels inside
    "triton_flash_attention.py",  # Triton kernels inside
    "__main__.py",  # Triton kernels inside
    "serve.py",  # Reason unkown. Test not passed for Cython. (FIXME)
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
if os.environ.get("CHITU_WITH_CYTHON", "0") != "0":
    ext_modules += cythonize(create_cython_extensions("chitu"))
    my_build_py = SkipBuildPy

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="chitu",
    version="0.3.5",
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
            "autoawq-kernels==0.0.8",
            "autoawq[kernels]",
            "gptqmodel>=2.2.0",
            "tokenizers>=0.20.3",
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
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension, "build_py": my_build_py},
    package_data={"chitu": ["config/**/*.yaml"]},
)
