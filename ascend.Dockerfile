# NOTE: CANN version is coupled with torch-npu version.
# See https://github.com/Ascend/pytorch/tags for the mapping.
FROM quay.io/ascend/cann:8.3.rc1.alpha001-910b-ubuntu22.04-py3.11 AS base

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

RUN apt update -y && apt install -y vim tmux telnet htop lsof strace iputils-ping curl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# NOTE: Always apt update before apt install to avoid out-dated docker cache
RUN if [ "${enable_test}" = "true" ]; then \
    apt update -y && apt install -y expect && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest; \
fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$(lscpu | grep x86)" ]; then \
        pip install -U torch==2.6.0+cpu -i https://download.pytorch.org/whl/cpu; \
    else \
        pip install -U torch==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/chitu
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks

# Currently, we require a development version of torch-npu to support aclgraph
ENV CHITU_ASCEND_BUILD=1

# The actual installing procedure requries a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
