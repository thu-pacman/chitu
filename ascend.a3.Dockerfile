# NOTE: A3 base image (openEuler). No extra torch or torch-npu installation required.
FROM quay.io/ascend/vllm-ascend:v0.10.0rc1-a3-openeuler AS base

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

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install test deps (expect + pytest) only when enabled
RUN if [ "${enable_test}" = "true" ]; then \
    dnf makecache && dnf install -y expect && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest; \
fi

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/chitu
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt

ENV CHITU_ASCEND_BUILD=1

# The actual installing procedure requires a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
