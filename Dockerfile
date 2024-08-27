FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ARG PYTHON_VERSION=3.10

RUN apt update -y \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update -y \
    && apt install -y git curl sudo \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip\
    && python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip \
    && python3 --version \
    && python3 -m pip --version

WORKDIR /workspace

COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U torch==2.1 --index-url https://download.pytorch.org/whl/cu121
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn

FROM base as build

WORKDIR /workspace/cinfer

COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_CUDA_ARCH_LIST=8.6 CINFER_WITH_CYTHON=1 pip install --no-build-isolation .[quant]

RUN rm -rf *
COPY example/ .