# NOTE: CANN version is coupled with torch-npu version.
# See https://github.com/Ascend/pytorch/tags for the mapping.
FROM quay.io/ascend/cann:8.3.rc1.alpha001-a3-ubuntu22.04-py3.11 AS base

ARG optional_deps=''
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

RUN pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# NOTE: Always apt update before apt install to avoid out-dated docker cache
# NOTE: Test dependencies include:
# - pytest is for test/pytest (for all platforms).
# - aiohttp is for service tests (for all platforms).
RUN if [ "${enable_test}" = "true" ]; then \
    apt update -y && apt install -y expect vim tmux telnet htop lsof strace iputils-ping && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest aiohttp; \
fi
RUN apt update -y && apt install -y curl

RUN if [ "$(lscpu | grep x86)" ]; then \
        pip install -U torch==2.6.0+cpu -i https://download.pytorch.org/whl/cpu; \
    else \
        pip install -U torch==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi
RUN pip install pyyaml setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/chitu
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks

# Download prometheus
RUN mkdir -p /workspace/prometheus && \
    # 根据框架下载prometheus
    case "$(uname -m)" in \
        x86_64|amd64) \
            URL="https://github.com/prometheus/prometheus/releases/download/v3.9.1/prometheus-3.9.1.linux-amd64.tar.gz" \
            ;; \
        aarch64|arm64) \
            URL="https://github.com/prometheus/prometheus/releases/download/v3.9.1/prometheus-3.9.1.linux-arm64.tar.gz" \
            ;; \
        *) \
            echo "不支持的架构: $(uname -m)" && exit 1 \
            ;; \
    esac && \
    echo "下载地址: $URL" && \
    curl -L --retry 3 --retry-delay 5 -o /workspace/prometheus.tar.gz "$URL" && \
    tar -xzvf /workspace/prometheus.tar.gz --strip-components=1 -C /workspace/prometheus && \
    rm -f /workspace/prometheus.tar.gz && \
    cp /workspace/prometheus/prometheus /usr/local/bin && \
    cp /workspace/prometheus/promtool /usr/local/bin/ && \
    rm -rf /workspace/prometheus && \
    prometheus --version

# Currently, we require a development version of torch-npu to support aclgraph
ENV CHITU_ASCEND_BUILD=1

# The actual installing procedure requries a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
