# NOTE: CANN version is coupled with torch-npu version.
# See https://github.com/Ascend/pytorch/tags for the mapping.
FROM quay.io/ascend/cann:8.2.rc1.alpha002-910b-ubuntu22.04-py3.10 AS base

ARG optional_deps=''
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

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# NOTE: Always apt update before apt install to avoid out-dated docker cache
RUN if [ "${enable_test}" = "true" ]; then \
    apt update -y && apt install -y expect && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest; \
fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$(lscpu | grep x86)" ]; then \
        pip install -U torch==2.5.1+cpu -i https://download.pytorch.org/whl/cpu; \
    else \
        pip install -U torch==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch-npu==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/chitu
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt

ENV ASCEND_PLATFORM=1

# The actual installing procedure requries a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
