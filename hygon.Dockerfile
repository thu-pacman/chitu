FROM harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10 AS base

SHELL ["/bin/bash", "-c"]

ARG optional_deps=''
ARG chitu_setup_jobs=''
ARG enable_editable_install='false'
ARG enable_cython='true'
ARG enable_test='false'
ARG pypi_mirror=''
ARG build_for_shca='false'

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
RUN if [ "${build_for_shca}" != "true" ] && [ "${build_for_shca}" != "false" ]; then \
    echo "ARG build_for_shca must either be 'true' or 'false'"; \
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
RUN apt update -y && apt install -y curl
RUN --mount=source=./third_party/hygon_wheels,destination=./third_party/hygon_wheels if [ "${build_for_shca}" = "true" ]; then \
    apt-get update -y; \
    apt remove -y rdmacm-utils ibacm perftest ibverbs-utils ucx libibverbs-dev libibmad-dev libibumad-dev librdmacm1 infiniband-diags opensm rdma-core libibmad5 libibumad3 ibverbs-providers libibverbs1 || true; \
    apt install -y libmosquitto1 && \
    dpkg -i ./third_party/hygon_wheels/shca-tools_2.500.4.B074-Ubuntu22.04_amd64.deb && \
    cp -r ./third_party/hygon_wheels/topo_lib /opt/topo_lib && \
    ln -s /opt/topo_lib/lib/librccl-net-shca.so.0.0.0 /opt/topo_lib/lib/librccl-net-shca.so && \
    ln -s /opt/topo_lib/lib/librccl-net-shca.so.0.0.0 /opt/topo_lib/lib/librccl-net-shca.so.0; \
else \
    apt update -y && apt install -y infiniband-diags; \
fi

RUN curl -L --retry 3 --retry-delay 5 -o /tmp/dtk_llvm.run https://download.sourcefind.cn:65024/file/4/dtk_llvm/dtk_llvm.run && \
    chmod +x /tmp/dtk_llvm.run && \
    /tmp/dtk_llvm.run && \
    rm -f /tmp/dtk_llvm.run
RUN --mount=source=./third_party/hygon_wheels,destination=./third_party/hygon_wheels pip install \
  ./third_party/hygon_wheels/aiter-0.1.2+das.opt1.dtk2604.torch290.2605071840.g1f8f50-cp310-cp310-linux_x86_64.whl \
  ./third_party/hygon_wheels/flash_mla-1.2.0+das.optphase1.d512.h64.dtk2604-cp310-cp310-linux_x86_64.whl \
  ./third_party/hygon_wheels/lmslim-0.3.1+das.opt4.dtk2604.torch290.2604281437.g61fdfe-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
  ./third_party/hygon_wheels/triton-3.5.1+das.opt1.dtk2604.torch290-cp310-cp310-manylinux_2_28_x86_64.whl \
  -c <(pip list --format freeze | grep -v -e "aiter" -e "flash_mla" -e "flash-mla" -e "lmslim" -e "triton")

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
ENV CHITU_HYGON_BUILD_FOR_SHCA=$build_for_shca
ENV HIP_GRAPH_ACCUMULATE_DISPATCH=1
ENV GPU_MAX_HW_QUEUES=3

# Update entrypoint, which calls the original entrypoint of the base image.
#
# If you want to change the base image, use
# `docker inspect --format='Entrypoint: {{.Config.Entrypoint}}' <image>`
# to check its entrypoint.
COPY ./script/entrypoint-hygon.sh /chitu-entrypoint.sh
ENTRYPOINT ["/chitu-entrypoint.sh", "/usr/local/bin/docker-entrypoint.sh"]

# The actual installing procedure requries a NPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
