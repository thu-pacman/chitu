# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import Extension, setup, find_packages

from pathlib import Path
import subprocess
import glob
import platform
import cpuinfo

setup_dir = os.path.dirname(os.path.abspath(__file__))


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CustomBuildExtension(BuildExtension):
    def build_extension(self, ext) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        info = cpuinfo.get_cpu_info()
        vendor = info.get("vendor_id_raw", "") or info.get("vendor_id", "")
        is_intel = "Intel" in vendor

        if ext.name == "llama.cpp":
            cmake_args = [
                "cmake",
                "-B",
                "build",
                "-D",
                "BUILD_SHARED_LIBS=ON",
                "-D",
                "LLAMA_NATIVE=ON",
                "-DCMAKE_CXX_COMPILER=g++",
            ]

            if is_intel:
                cmake_args += [
                    "-DLLAMA_AVX=ON",
                    "-DLLAMA_AVX2=ON",
                    "-DLLAMA_AVX512=ON",
                    "-DLLAMA_AVX512_BF16=ON",
                ]
            else:
                print("Non-Intel CPU detected; skipping AVX flags.")

            build_args = [
                "cmake",
                "--build",
                "build",
                "--config",
                "Release",
                "-j",
            ]

            try:
                subprocess.run(
                    cmake_args,
                    cwd=ext.sourcedir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    build_args,
                    cwd=ext.sourcedir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print("CMake stdout:\n", e.stdout)
                print("CMake stderr:\n", e.stderr)
                raise
        else:
            super().build_extension(ext)


llama_cpp_files = glob.glob("../../third_party/llamafile/*.cpp")
setup(
    name="cpuinfer",
    version="0.0.0",
    install_requires=["cpufeature"],
    cmdclass={"build_ext": CustomBuildExtension},
    ext_modules=[
        CMakeExtension("llama.cpp", "../../third_party/llama.cpp"),
        CUDAExtension(
            name="cpuinfer",
            sources=[
                "affinity.cpp",
                "bindings.cpp",
                "moe.cpp",
                "linear.cpp",
                "shared_mem_buffer.cpp",
            ]
            + llama_cpp_files,
            libraries=["ggml_static"],
            include_dirs=[
                os.path.join(setup_dir, "../../third_party/"),
            ],
            library_dirs=[
                os.path.join(setup_dir, "../../third_party/llama.cpp/build"),
            ],
            extra_compile_args=["-O3", "-march=native", "-DUSE_CUDA"],
        ),
    ],
)
