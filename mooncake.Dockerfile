#####################################
# Base Image Stage
ARG BASE_IMAGE=mooncake-base:latest
FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-c"]

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ARG optional_deps='flash_attn,flash_mla,flashinfer'
ARG chitu_setup_jobs=''
ARG enable_cython='true'
ARG enable_test='false'

ENV CHITU_SETUP_JOBS=$chitu_setup_jobs
ENV MAX_JOBS=$CHITU_SETUP_JOBS

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

RUN apt update -y && apt install -y git gcc-10 g++-10 libnuma-dev libibverbs1 ibverbs-providers libibverbs-dev rdma-core

# NOTE: Always apt update before apt install to avoid out-dated docker cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U "pip<25.3" -i https://pypi.tuna.tsinghua.edu.cn/simple

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
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements-build.txt \
    -c <(pip list --format freeze | grep -v "setuptools")
RUN if [[ "${optional_deps}" == *"deep_ep"* ]]; then \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements-build-deep_ep-cu12.txt \
        -c <(pip list --format freeze | grep -v "setuptools"); \
fi

ENV FLASH_MLA_DISABLE_SM100=1


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
    pip install --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt \
        -c <(pip list --format freeze | grep -v -e "pillow" -e "fsspec" -e "numpy" -e "transformers" -e "pytest")

#####################################
# Wheel build Stage
#
# This stage build wheel file of chitu.
FROM dependency_installer AS wheel_builder

ARG enable_cython

WORKDIR /workspace/chitu
COPY . .

# build wheel of chitu
RUN ./script/build_for_dist.sh "${enable_cython}"

# verify the wheel was created
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
RUN bash -c "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple /tmp/*.whl -c <(pip list --format freeze | grep -v -e 'pillow' -e 'fsspec' -e 'flash-mla' -e 'flash_mla' -e 'numpy' -e 'transformers' -e 'pytest')"

RUN rm -rf /tmp/*
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks

# These are optimization flags for NCCL, but according to our tests, they only make things
# worse, so we don't use them.
ENV NCCL_GRAPH_MIXING_SUPPORT=0
ENV NCCL_GRAPH_REGISTER=0