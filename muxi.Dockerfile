FROM maca-pytorch:3.0.0.4-torch2.6-py310-ubuntu24.04-amd64 AS base

SHELL ["/bin/bash", "-c"]

ARG optional_deps=''
ARG chitu_setup_jobs=''
ARG enable_cython='true'
ARG enable_test='false'

ENV CHITU_SETUP_JOBS=$chitu_setup_jobs
ENV MAX_JOBS=$CHITU_SETUP_JOBS

# The base image uses Conda as the Python environment. We need to activate it
# For `docker build` stage, the most straightforward way is to use `bash --login -c` as the shell
SHELL ["/bin/bash", "--login", "-c"]
# For `docker run` stage, we need an entrypoint
RUN echo "source /etc/profile; \"\$@\"" > /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

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

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN printf '%s\n' \
      "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" \
      "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" \
      "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" \
      "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" \
      > /etc/apt/sources.list; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates; \
    update-ca-certificates; \
    sed -i 's|http://mirrors.tuna.tsinghua.edu.cn|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# NOTE: Always apt update before apt install to avoid out-dated docker cache
# NOTE: g++-11 a downgrading of g++, which is required by compiling muxi_layout_kernels. This is
#       because mxcc can't compile C++20 when g++ is too new.
RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends g++-11 curl; \
    rm -rf /var/lib/apt/lists/*
RUN if [ "${enable_test}" = "true" ]; then \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends expect vim tmux telnet htop lsof strace iputils-ping; \
    rm -rf /var/lib/apt/lists/*; \
fi

# NOTE: Test dependencies include:
# - pytest is for test/pytest (for all platforms).
# - aiohttp is for service tests (for all platforms).
# - matplotlib is for benchmarks/op_bench (for platforms with triton).
RUN if [ "${enable_test}" = "true" ]; then \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest aiohttp matplotlib; \
fi

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

ENV CHITU_MUXI_BUILD=1

# The actual installing procedure requries a GPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
