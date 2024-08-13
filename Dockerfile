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

COPY requirements.txt requirements.txt
RUN echo "$(readlink -f requirements.txt)"
RUN pip install -U torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
