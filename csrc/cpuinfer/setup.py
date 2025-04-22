import os

from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import Extension, setup, find_packages

from pathlib import Path
import subprocess
import glob

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
        if ext.name == "llama.cpp":
            try:
                subprocess.run(
                    [
                        "cmake",
                        "-B",
                        "build",
                        "-D",
                        "BUILD_SHARED_LIBS=ON",
                        "-D",
                        "LLAMA_NATIVE=ON",
                    ],
                    cwd=ext.sourcedir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["cmake", "--build", "build", "--config", "Release", "-j"],
                    cwd=ext.sourcedir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                print("STDOUT:\n", e.stdout)
                print("STDERR:\n", e.stderr)
                raise e


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
                "ext_bindings.cpp",
                "moe.cpp",
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
        CUDAExtension(
            name="ktdequant",
            sources=[
                "custom_gguf/dequant.cu",
                "custom_gguf/binding.cpp",
            ],
            include_dirs=["custom_gguf/"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-Xcompiler",
                    "-fPIC",
                ],
            },
        ),
    ],
)
