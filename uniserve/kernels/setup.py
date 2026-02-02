from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 9.0"

if False:  # Debug
    extra_compile_args = {
        "nvcc": ["-G", "-g"],
        "cxx": ["-g", "-Og"],
    }
else:  # Release
    extra_compile_args = {
        "nvcc": [
            "-g",
            "--use_fast_math",
            "-lineinfo",
        ],
        "cxx": ["-g"],
    }

setup(
    name="uniserve_cuda",
    ext_modules=[
        CUDAExtension(
            "uniserve_cuda",
            [
                "operators.cpp",
                "add_cuda_kernel.cu",
                "ragged_nhwc_im2col_kernel.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
