#####################################
# Base Image Stage
ARG BASE_IMAGE=mooncake-base:latest
FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-c"]

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ARG optional_deps='flash_attn,flash_mla,flashinfer'
ARG build_jobs=''
ARG enable_editable_install='false'
ARG enable_cython='true'
ARG enable_test='false'

RUN if [ "${enable_editable_install}" != "true" ] && [ "${enable_editable_install}" != "false" ]; then \
    echo "ARG enable_editable_install must either be 'true' or 'false'"; \
    exit 1; \
fi
RUN if [ "${enable_cython}" != "true" ] && [ "${enable_cython}" != "false" ]; then \
    echo "ARG enable_cython must either be 'true' or 'false'"; \
    exit 1; \
fi
RUN if [ "{enable_cython}" = "true" ] && [ "${enable_editable_install}" = "true" ]; then \
    echo "Cython is not supported when installing in editable mode"; \
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

RUN apt update -y && apt install -y git gcc-10 g++-10 libnuma-dev

# NOTE: Always apt update before apt install to avoid out-dated docker cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# NOTE: Always apt update before apt install to avoid out-dated docker cache
RUN if [ "${enable_test}" = "true" ]; then \
    apt update -y && apt install -y expect && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest; \
fi

# Always install build time dependencies. Some dependencies may fail to build
# if some build time dependencies are missing.
COPY ./requirements-build.txt /tmp/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements-build.txt -c <(pip list --format freeze)


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
COPY ./third_party ./third_party
COPY ./csrc/cpuinfer ./csrc/cpuinfer

# Don't use `--mount=type=cache,target=/root/.cache/pip` here, because some dependencies
# compile at install time, and the compile results are environment dependent.
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt -c <(pip list --format freeze | grep -v "pillow" | grep -v "fsspec")

WORKDIR /workspace
RUN rm -rf /workspace/chitu


#####################################
# Build Stage
# 
# This stage builds chitu.
FROM dependency_installer AS build

WORKDIR /workspace/chitu
COPY . .

# Don't use `--mount=type=cache,target=/root/.cache/pip` here, because some dependencies
# compile at install time, and the compile results are environment dependent.
RUN bash script/install.sh "${optional_deps}" "${build_jobs}" "${enable_editable_install}" "${enable_cython}"

# These are optimization flags for NCCL, but according to our tests, they only make things
# worse, so we don't use them.
ENV NCCL_GRAPH_MIXING_SUPPORT=0
ENV NCCL_GRAPH_REGISTER=0


