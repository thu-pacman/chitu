#!/bin/bash

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

# Set CUDA architectures for H100 (9.0) and A100 (8.0)
# This prevents IndexError when PyTorch can't auto-detect architectures
export TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX"

# PyTorch's build system may read these environment variables
# Force builds to use GCC 11 toolchain (helps pybind11 + CUDA with GCC 12 incompat)
# export GCC11_BIN="/home/ppopp26_ae/miniconda3/envs/gcc11env/bin"

# Install dependencies
pip install torch==2.9.1 torchvision torchaudio==2.9.1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "numpy<2"

# Install torchperf
cd third_party/torchperf/
pip install . --no-build-isolation
cd ../..

# Install sfast (stable-fast)
cd third_party/uniserve-stable-fast/
pip install . --no-build-isolation
cd ../..

# Install uniserve kernels
cd "${PROJECT_ROOT}/uniserve/kernels/" || exit 1
python setup.py install
cd "${PROJECT_ROOT}" || exit 1

# flash-attention version 2.8.3
export FLASH_ATTN_CUDA_ARCHS="80;90"
git clone https://github.com/Dao-AILab/flash-attention.git
python setup.py install

# Install uniserve
pip install . --no-build-isolation