# NOTE: CANN version is coupled with torch-npu version.
# See https://github.com/Ascend/pytorch/tags for the mapping.
FROM quay.io/ascend/cann:8.3.rc1.alpha001-a3-ubuntu22.04-py3.11 AS base

ARG optional_deps=''
ARG chitu_setup_jobs=''
ARG enable_cython='true'
ARG enable_test='false'
ARG pypi_mirror=''

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

# Upgrade pip and set mirror. The mirror should be set AFTER upgrading pip
RUN if [ "${pypi_mirror}" != "" ]; then \
    pip install -U "pip<25.3" -i "${pypi_mirror}"; \
else \
    pip install -U "pip<25.3"; \
fi
RUN if [ "${pypi_mirror}" != "" ]; then \
    pip config set global.index-url "${pypi_mirror}"; \
fi

# NOTE: Always apt update before apt install to avoid out-dated docker cache
# NOTE: Test dependencies include:
# - pytest is for test/pytest (for all platforms).
# - matplotlib is for op benchmarks in test/pytest, and benchmarks/visualize_response.py (for all platforms).
RUN if [ "${enable_test}" = "true" ]; then \
    apt update -y && apt install -y expect vim tmux telnet htop lsof strace iputils-ping && \
    pip install pytest matplotlib; \
fi
RUN apt update -y && apt install -y infiniband-diags curl

# Download prometheus
# NOTE: Always download from Github, because docker on our platform does not support mounting secret
RUN mkdir -p /workspace/prometheus && \
    case "$(uname -m)" in \
        x86_64|amd64) \
            GITHUB_URL="https://github.com/prometheus/prometheus/releases/download/v3.9.1/prometheus-3.9.1.linux-amd64.tar.gz" \
            ;; \
        aarch64|arm64) \
            GITHUB_URL="https://github.com/prometheus/prometheus/releases/download/v3.9.1/prometheus-3.9.1.linux-arm64.tar.gz" \
            ;; \
        *) \
            echo "Unsupport arch: $(uname -m)" && exit 1 \
            ;; \
    esac && \
    echo "Download prometheus from GitHub" && \
    curl -L --retry 3 --retry-delay 5 -o /workspace/prometheus.tar.gz "${GITHUB_URL}" && \
    tar -xzvf /workspace/prometheus.tar.gz --strip-components=1 -C /workspace/prometheus && \
    rm -rf /workspace/prometheus.tar.gz && \
    cp /workspace/prometheus/prometheus /usr/local/bin && \
    cp /workspace/prometheus/promtool /usr/local/bin/ && \
    rm -rf /workspace/prometheus && \
    prometheus --version

# Install Grafana
RUN mkdir -p /workspace/grafana && \
    case "$(uname -m)" in \
        x86_64|amd64) \
            CDN_URL="https://dl.grafana.com/grafana/release/12.4.1/grafana_12.4.1_22846628243_linux_amd64.tar.gz" \
            ;; \
        aarch64|arm64) \
            CDN_URL="https://dl.grafana.com/grafana/release/12.4.1/grafana_12.4.1_22846628243_linux_arm64.tar.gz" \
            ;; \
        *) \
            echo "Unsupported arch: $(uname -m)" && exit 1 \
            ;; \
    esac && \
    echo "Download Grafana from CDN" && \
    curl -L --retry 3 --retry-delay 5 -o /workspace/grafana.tar.gz "${CDN_URL}" && \
    tar -xzf /workspace/grafana.tar.gz --strip-components=1 -C /workspace/grafana && \
    rm -rf /workspace/grafana.tar.gz && \
    cp /workspace/grafana/bin/grafana-server /usr/local/bin/ && \
    cp /workspace/grafana/bin/grafana /usr/local/bin/ && \
    mkdir -p /usr/share/grafana && \
    cp -r /workspace/grafana/public /usr/share/grafana/public && \
    cp -r /workspace/grafana/conf /usr/share/grafana/conf && \
    rm -rf /workspace/grafana && \
    grafana-server -v

RUN if [ "$(lscpu | grep x86)" ]; then \
        pip install -U torch==2.6.0+cpu -i https://download.pytorch.org/whl/cpu; \
    else \
        pip install -U torch==2.6.0; \
    fi
RUN pip install pyyaml setuptools

WORKDIR /workspace/chitu
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks
COPY ./chitu/metrics/grafana ./grafana

# Currently, we require a development version of torch-npu to support aclgraph
ENV CHITU_ASCEND_BUILD=1

# The actual installing procedure requries a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
