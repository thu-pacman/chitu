FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.4.1-ubuntu22.04-dtk25.04-py3.10-fixpy AS base

SHELL ["/bin/bash", "-c"]

ARG optional_deps=''
ARG chitu_setup_jobs=''
ARG enable_editable_install='false'
ARG enable_cython='true'
ARG enable_test='false'
ARG pypi_mirror=''

ENV CHITU_SETUP_JOBS=$chitu_setup_jobs
ENV MAX_JOBS=$CHITU_SETUP_JOBS

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
ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_CACHE_DIR=1

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
RUN --mount=type=secret,id=tos_id \
    --mount=type=secret,id=tos_key \
    mkdir -p /workspace/prometheus && \
    case "$(uname -m)" in \
        x86_64|amd64) \
            GITHUB_URL="https://github.com/prometheus/prometheus/releases/download/v3.9.1/prometheus-3.9.1.linux-amd64.tar.gz" && \
            TOS_URL="tos://out-deliver/prometheus-3.9.1.linux-amd64.tar" && \
            TOOL_URL="https://tos-tools.tos-cn-beijing.volces.com/linux/tosutil" \
            ;; \
        aarch64|arm64) \
            GITHUB_URL="https://github.com/prometheus/prometheus/releases/download/v3.9.1/prometheus-3.9.1.linux-arm64.tar.gz" && \
            TOS_URL="tos://out-deliver/prometheus-3.9.1.linux-arm64.tar" && \
            TOOL_URL="https://m645b3e1bb36e-mrap.mrap.accesspoint.tos-global.volces.com/linux/arm64/tosutil" \
            ;; \
        *) \
            echo "Unsupport arch: $(uname -m)" && exit 1 \
            ;; \
    esac && \
    if [ -s /run/secrets/tos_id ] && [ -s /run/secrets/tos_key ]; then \
        echo "tos_id and tos_id exits, download prometheus from Tos" && \
        tos_id=$(cat /run/secrets/tos_id) && \
        tos_key=$(cat /run/secrets/tos_key) && \
        mkdir -p /workspace/prometheus && \
        mkdir -p /tmp && curl "${TOOL_URL}" --output /tmp/tosutil && chmod a+x /tmp/tosutil && \
        /tmp/tosutil cp -u -r -p=8 -j=8 -threshold=104857600 -k "${tos_key}" -i "${tos_id}" \
            -e tos-cn-beijing.volces.com -re out-deliver.tos-cn-beijing.volces.com "${TOS_URL}" /workspace && \
        tar -xf /workspace/prometheus-*.tar --strip-components=1 -C /workspace/prometheus && \
        rm -rf /workspace/prometheus-*.tar && \
        rm -rf /tmp/tosutil; \
    else \
        echo "Download prometheus from GitHub" && \
        curl -L --retry 3 --retry-delay 5 -o /workspace/prometheus.tar.gz "${GITHUB_URL}" && \
        tar -xzvf /workspace/prometheus.tar.gz --strip-components=1 -C /workspace/prometheus && \
        rm -rf /workspace/prometheus.tar.gz; \
    fi && \
    cp /workspace/prometheus/prometheus /usr/local/bin && \
    cp /workspace/prometheus/promtool /usr/local/bin/ && \
    rm -rf /workspace/prometheus && \
    prometheus --version

# Install Grafana
RUN --mount=type=secret,id=tos_id \
    --mount=type=secret,id=tos_key \
    mkdir -p /workspace/grafana && \
    case "$(uname -m)" in \
        x86_64|amd64) \
            CDN_URL="https://dl.grafana.com/grafana/release/12.4.1/grafana_12.4.1_22846628243_linux_amd64.tar.gz" && \
            TOS_URL="tos://out-deliver/grafana_12.4.1_22846628243_linux_amd64.tar.gz" && \
            TOOL_URL="https://tos-tools.tos-cn-beijing.volces.com/linux/tosutil" \
            ;; \
        aarch64|arm64) \
            CDN_URL="https://dl.grafana.com/grafana/release/12.4.1/grafana_12.4.1_22846628243_linux_arm64.tar.gz" && \
            TOS_URL="tos://out-deliver/grafana_12.4.1_22846628243_linux_arm64.tar.gz" && \
            TOOL_URL="https://m645b3e1bb36e-mrap.mrap.accesspoint.tos-global.volces.com/linux/arm64/tosutil" \
            ;; \
        *) \
            echo "Unsupported arch: $(uname -m)" && exit 1 \
            ;; \
    esac && \
    if [ -s /run/secrets/tos_id ] && [ -s /run/secrets/tos_key ]; then \
        echo "Download Grafana from TOS" && \
        tos_id=$(cat /run/secrets/tos_id) && \
        tos_key=$(cat /run/secrets/tos_key) && \
        mkdir -p /tmp && curl "${TOOL_URL}" --output /tmp/tosutil && chmod a+x /tmp/tosutil && \
        /tmp/tosutil cp -u -r -p=8 -j=8 -threshold=104857600 -k "${tos_key}" -i "${tos_id}" \
            -e tos-cn-beijing.volces.com -re out-deliver.tos-cn-beijing.volces.com "${TOS_URL}" /workspace && \
        tar -xzf /workspace/grafana_*.tar.gz --strip-components=1 -C /workspace/grafana && \
        rm -rf /workspace/grafana_*.tar.gz && \
        rm -rf /tmp/tosutil; \
    else \
        echo "Download Grafana from CDN" && \
        curl -L --retry 3 --retry-delay 5 -o /workspace/grafana.tar.gz "${CDN_URL}" && \
        tar -xzf /workspace/grafana.tar.gz --strip-components=1 -C /workspace/grafana && \
        rm -rf /workspace/grafana.tar.gz; \
    fi && \
    cp /workspace/grafana/bin/grafana-server /usr/local/bin/ && \
    cp /workspace/grafana/bin/grafana /usr/local/bin/ && \
    mkdir -p /usr/share/grafana && \
    cp -r /workspace/grafana/public /usr/share/grafana/public && \
    cp -r /workspace/grafana/conf /usr/share/grafana/conf && \
    rm -rf /workspace/grafana && \
    grafana-server -v

WORKDIR /workspace/chitu
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks
COPY ./chitu/metrics/grafana ./grafana

ENV CHITU_HYGON_BUILD=1
# The actual installing procedure requries a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
