#####################################
# Base Image Stage
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel AS base

ARG torch_cuda_arch_list='12.0+PTX'
ARG optional_deps=''
ARG build_jobs=''
ARG enable_cython='true'
ARG enable_test='false'

RUN if [ "${enable_cython}" != "true" ] && [ "${enable_cython}" != "false" ]; then \
    echo "ARG enable_cython must either be 'true' or 'false'"; \
    exit 1; \
fi
RUN if [ "${enable_test}" != "true" ] && [ "${enable_test}" != "false" ]; then \
    echo "ARG enable_test must either be 'true' or 'false'"; \
    exit 1; \
fi

# Required for non-interactive apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ENV ENABLE_NVFP4=1

RUN apt update -y && apt install -y git gcc-11 g++-11 libnuma-dev build-essential cmake ninja-build

# NOTE: Always apt update before apt install to avoid out-dated docker cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# NOTE: Always apt update before apt install to avoid out-dated docker cache
# NOTE: Test dependencies include:
# - pytest is for test/pytest (for all platforms).
# - aiohttp is for service tests (for all platforms).
# - matplotlib is for benchmarks/op_bench (for platforms with triton).
RUN if [ "${enable_test}" = "true" ]; then \
    apt update -y && apt install -y expect vim tmux telnet htop lsof strace iputils-ping curl && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest aiohttp matplotlib; \
fi

# Always install build time dependencies. Some dependencies may fail to build
# if some build time dependencies are missing.
COPY ./requirements-build.txt /tmp/requirements-build.txt
COPY ./requirements-build-deep_ep-cu12.txt /tmp/requirements-build-deep_ep-cu12.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements-build.txt
RUN if [[ "${optional_deps}" == *"deep_ep"* ]]; then \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements-build-deep_ep-cu12.txt; \
fi


#####################################
# Dependency Resolver Stage
#
# The only purpose of this stage is to generate a requirements.txt file. This
# stage may trigger rebuild whenever there is any change in the source code,
# but this stage runs fast.
FROM base AS dependency_resolver

WORKDIR /workspace/chitu
COPY . .

RUN ./gen_tmp_requirements_txt.py "${optional_deps}" > /tmp/requirements.txt


#####################################
# Dependency Installer Stage
#
# This stage installs the dependencies listed in requirements.txt. Some of the
# dependencies may require compilation, so this stage may take a long time, but
# this stage only triggers rebuild when the requirements.txt file changes, or
# this source of the dependencies changes.
FROM base AS dependency_installer

WORKDIR /workspace/chitu
COPY --from=dependency_resolver /tmp/requirements.txt /tmp/requirements.txt

# Don't use `--mount=type=cache,target=/root/.cache/pip` here, because some dependencies
# compile at install time, and the compile results are environment dependent.
RUN --mount=type=bind,source=./third_party,target=./third_party,readwrite \
    --mount=type=bind,source=./csrc/cpuinfer,target=./csrc/cpuinfer,readwrite \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt && \
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#####################################
# Wheel build Stage
#
# This stage build wheel file of chitu.
FROM dependency_installer AS wheel_builder

WORKDIR /workspace/chitu
COPY . .

RUN cd /workspace/chitu/third_party/hard_fp4_kernels && python setup.py bdist_wheel
RUN cd /workspace/chitu && python setup.py bdist_wheel
# build wheel of chitu
RUN ./script/build_for_dist.sh "${enable_cython}"

# verify the wheel was created
RUN cp third_party/hard_fp4_kernels/dist/*.whl /tmp/
RUN cp dist/*.whl /tmp/
RUN ls -al /tmp/

RUN rm -rf /workspace/chitu/*

#####################################
# Build Stage
# 
# This stage builds chitu.
FROM dependency_installer AS build

COPY --from=wheel_builder /tmp/ /tmp/

# Don't use `--mount=type=cache,target=/root/.cache/pip` here, because some dependencies
# compile at install time, and the compile results are environment dependent.
RUN bash -c "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple /tmp/*.whl -c <(pip list --format freeze | grep -v -e 'pillow' -e 'fsspec' -e 'flash-mla' -e 'flash_mla')"

RUN pip install triton==3.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN rm -rf /tmp/*
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks

# These are optimization flags for NCCL, but according to our tests, they only make things
# worse, so we don't use them.
ENV NCCL_GRAPH_MIXING_SUPPORT=0
ENV NCCL_GRAPH_REGISTER=0
