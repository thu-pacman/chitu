# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import glob

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
from get_requires import install_requires, extras_require


# We use CUDAExtension instead of CMake for native sources, because many of the non-NVIDIA GPUs have
# their custom CUDAExtension, but not their custom CMake support.


if (
    os.environ.get("CHITU_ASCEND_BUILD", "0") == "1"
    or os.environ.get("CHITU_HYGON_BUILD", "0") == "1"
):
    ext_modules = []
else:
    ext_modules = operators.get_extensions()

cython_unsafe_files = (
    glob.glob("chitu/ops/triton_ops/**/*.py", recursive=True)  # Triton kernels inside
    + glob.glob("chitu/moe/experts/*.py")  # Triton kernels inside
    + [
        "chitu/moe/batched_routed_activation.py",  # plum inside
        "chitu/muxi_utils.py",  # plum inside
        "chitu/native_layout.py",  # plum inside
        "__main__.py",
    ]
)


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
    version="0.5.0",
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension, "build_py": my_build_py},
    package_data={"chitu": ["config/**/*.yaml"]},
)
