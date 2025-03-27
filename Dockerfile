FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel AS base

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

RUN apt update -y \
    && apt install -y git \
    && apt install -y gcc-10 g++-10

WORKDIR /workspace/chitu
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[flash_attn,flash_mla,flashinfer]
