import sys
import warnings
import os
import re
import ast
import glob
import subprocess

from pathlib import Path
from packaging.version import parse, Version
from setuptools import find_packages, setup
import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, CUDA_HOME

__version__ = "1.0.0-beta.0"

# ninja build does not work unless include_dirs are abs path
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )

_, bare_cuda_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_cuda_version >= Version("11.4"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6+PTX"
else:
    raise RuntimeError(
        "EETQ is only supported on CUDA 11.4 and above.  "
        "Your version of CUDA is: {}\n".format(bare_cuda_version)
    )

ext_modules = []

cutlass_sources = ["csrc/eetpy.cpp",
                   "csrc/cutlass_kernels/fpA_intB_gemm_wrapper.cu",
                   "csrc/cutlass_kernels/fpA_intB_gemm.cu",
                   "csrc/cutlass_kernels/cutlass_heuristic.cc",
                   "csrc/cutlass_kernels/cutlass_preprocessors.cc",
                   "csrc/utils/logger.cc",
                   "csrc/utils/cuda_utils.cc"
                   ]
custom_sources = ["csrc/embedding_kernels/pos_encoding_kernels.cu",
                  "csrc/layernorm_kernels/layernorm.cu"
                  ]
sources = cutlass_sources + custom_sources
for item in sources:
    sources[sources.index(item)] = os.path.join(current_dir, item)

include_paths = []
include_paths.append(cpp_extension.include_paths(cuda=True))    # cuda path
include_paths.append(os.path.join(current_dir, 'csrc'))
include_paths.append(os.path.join(current_dir, 'csrc/utils'))
include_paths.append(os.path.join(current_dir, 'csrc/cutlass/include'))
include_paths.append(os.path.join(current_dir, 'csrc/cutlass_kernels/include'))
include_paths.append(os.path.join(current_dir, 'csrc/cutlass_extensions/include'))


ext_modules.append(
    CUDAExtension(
        name="EETQ",
        sources=sources,
        include_dirs=include_paths,
        # libraries=['cublas', 'cudart', 'cudnn', 'curand', 'nvToolsExt'],
        extra_compile_args={
            "cxx": ['-g',
                    '-std=c++17',
                    # '-U NDEBUG',
                    '-O3',
                    '-fopenmp',
                    '-lgomp'],
            "nvcc": ['-O3',
                     '-std=c++17',
                     '-U__CUDA_NO_HALF_OPERATORS__',
                     '-U__CUDA_NO_HALF_CONVERSIONS__',
                     '-U__CUDA_NO_HALF2_OPERATORS__',
                     '-U__CUDA_NO_HALF2_CONVERSIONS__'],
        },
        define_macros=[('VERSION_INFO', __version__),
                       # ('_DEBUG_MODE_', None),
                       ]
    )
)

setup(
    name="EETQ",
    version=__version__,
    author="zhaosida, dingjingzhen",
    author_email="dingjingzhen@corp.netease.com, zhaosida@corp.netease.com",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
