from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="uniserve_cuda",
    ext_modules=[
        CUDAExtension(
            "uniserve_cuda",
            [
                "operators.cpp",
                # "lltm_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
