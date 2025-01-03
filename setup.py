import os
import subprocess
import sys

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]
            elif (jobs := int(os.environ.get("CINFER_SETUP_JOBS", "0"))) > 0:
                print(f"Limit setup jobs to {jobs}")
                # CMake 3.12+ only.
                build_args += [f"-j{jobs}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


cython_unsafe_files = [
    "triton_kernels.py",
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


ext_modules = [CMakeExtension("cinfer_backend")]
my_build_py = build_py
if os.environ.get("CINFER_WITH_CYTHON", "0") != "0":
    ext_modules += cythonize(create_cython_extensions("cinfer"))
    my_build_py = SkipBuildPy

setup_dir = os.path.dirname(os.path.abspath(__file__))

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="CInfer",
    version="0.0.1",
    install_requires=[
        # Don't put `torch` here because it requires downloading from a specific source
        "transformers",
        "flash-attn",
        "fairscale",
        "fire",
        "tiktoken>=0.7.0",  # Required by glm4
        "blobfile",
        "faker",
        "hydra-core",
        "fastapi",
        "uvicorn",
        "tqdm",
        "accelerate",
    ],
    extras_require={
        "quant": [
            "bitsandbytes",
            "EETQ @ file://localhost" + os.path.join(setup_dir, "third_party/EETQ"),
            "awq_inference_engine @ file://localhost"
            + os.path.join(setup_dir, "third_party/llm-awq/awq/kernels"),
            "auto_gptq @ file://localhost"
            + os.path.join(setup_dir, "third_party/AutoGPTQ"),
        ],
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild, "build_py": my_build_py},
)
