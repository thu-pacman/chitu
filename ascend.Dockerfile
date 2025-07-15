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

WORKDIR /workspace/chitu
COPY . .

# Currently, we require a development version of torch-npu to support aclgraph:
# The .whl files are in our repo, so these lines should be after COPY.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$(lscpu | grep x86)" ]; then \
        pip install ./third_party/ascend/torch_npu-2.5.1.post1.dev20250529-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    else \
        pip install ./third_party/ascend/torch_npu-2.5.1.post1.dev20250702-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi
# To directly use the stable version of torch-npu, uncomment the following code:
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install torch-npu==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt

ENV CHITU_ASCEND_BUILD=1

# The actual installing procedure requries a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
