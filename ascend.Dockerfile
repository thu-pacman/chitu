# NOTE: CANN version is coupled with torch-npu version.
# See https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md for the mapping.
FROM quay.io/ascend/cann:9.2.0-910b-ubuntu22.04-py3.11 AS base

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
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${pypi_mirror}" != "" ]; then \
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

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$(lscpu | grep x86)" ]; then \
        pip install -U torch==2.9.0+cpu -i https://download.pytorch.org/whl/cpu; \
    else \
        pip install -U torch==2.9.0; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml setuptools

# Build dependencies are needed by the wheel stage and by the local AscendC
# kernel package installed in the dependency installer.
COPY ./requirements-build.txt /tmp/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-build.txt

WORKDIR /workspace/chitu
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks
COPY ./chitu/metrics/grafana ./grafana

# Currently, we require a development version of torch-npu to support aclgraph
ENV CHITU_ASCEND_BUILD=1

# Update entrypoint, which calls the original entrypoint of the base image.
#
# If you want to change the base image, use
# `docker inspect --format='Entrypoint: {{.Config.Entrypoint}}' <image>`
# to check its entrypoint.
COPY ./script/entrypoint.sh /chitu-entrypoint.sh
ENTRYPOINT ["/chitu-entrypoint.sh", "/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh && exec \"$@\"", "--"]

#####################################
# Dependency resolver stage
#
# Generate the PyPI and local-package requirements files from get_requires.py,
# including the selected optional extras. This stage needs torch only for
# get_requires/csrc imports.
FROM base AS dependency_resolver
WORKDIR /workspace/chitu
COPY . .
RUN ./gen_tmp_requirements_txt.py "${optional_deps}" --kind pypi > /tmp/requirements-pypi.txt
RUN ./gen_tmp_requirements_txt.py "${optional_deps}" --kind local > /tmp/requirements-local.txt

#####################################
# Dependency installer stage
#
# Install torch_npu and the PyPI dependencies. Disable torch's backend
# auto-loading while installing, because torch_npu needs the host driver and
# we are not mounting it in this stage. The final image copies this Python
# environment and does not inherit this variable.
FROM base AS dependency_installer
WORKDIR /workspace/chitu
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0
COPY --from=dependency_resolver /tmp/requirements-pypi.txt /tmp/requirements-pypi.txt
RUN pip install torch-npu==2.9.0.post9
RUN if [ -n "${pypi_mirror}" ]; then \
        pip install --no-build-isolation -r /tmp/requirements-pypi.txt \
            -c <(pip list --format freeze | grep -v -e "pillow" -e "fsspec" -e "numpy" -e "transformers" -e "tokenizers" -e "xgrammar" -e "pytest" -e "build" -e "click") \
            -i "${pypi_mirror}" \
            --extra-index-url https://pypi.org/simple \
            --timeout 60 --retries 10; \
    else \
        pip install --no-build-isolation -r /tmp/requirements-pypi.txt \
            -c <(pip list --format freeze | grep -v -e "pillow" -e "fsspec" -e "numpy" -e "transformers" -e "tokenizers" -e "xgrammar" -e "pytest" -e "build" -e "click") \
            -i https://pypi.org/simple \
            --timeout 60 --retries 10; \
    fi

#####################################
# Local kernel stage
#
# Build and install the AscendC kernel package. The driver is mounted only
# here, because its setup.py imports torch_npu.
FROM dependency_installer AS kernel_builder
WORKDIR /workspace/chitu
COPY ./third_party/ascend-kernel ./third_party/ascend-kernel
COPY ./csrc/cpuinfer ./csrc/cpuinfer
COPY --from=dependency_resolver /tmp/requirements-local.txt /tmp/requirements-local.txt
RUN --mount=type=bind,from=ascend_driver,target=/usr/local/Ascend/driver/lib64 \
    --mount=type=bind,from=ascend_dcmi,target=/usr/local/dcmi \
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/inner:/usr/local/dcmi:${LD_LIBRARY_PATH}; \
    pip wheel --no-build-isolation --no-deps -w /tmp/kernel-wheels -r /tmp/requirements-local.txt

#####################################
# Wheel build stage
#
# Source is present only in the resolver/installer/wheel stages. The final
# image copies just the wheel, so no source tree ends up in the image.
FROM kernel_builder AS wheel_builder
ARG enable_cython
WORKDIR /workspace/chitu
COPY . .
RUN ./script/build_for_dist.sh "${enable_cython}"
RUN mkdir -p /tmp/wheels && cp dist/*.whl /tmp/wheels/
RUN rm -rf /workspace/chitu/*

#####################################
# Final image
# Start from dependency_installer, not base: the PyPI dependencies are
# installed there, and this stage must not copy the Python tree to get them.
FROM dependency_installer AS final
WORKDIR /workspace/chitu
# Install the local kernel wheel and the chitu wheel; both are prebuilt.
COPY --from=kernel_builder /tmp/kernel-wheels /tmp/kernel-wheels
# cinfer_ascendc's build step installs the AscendC vendor ops directly into
# site-packages, and the wheel does not contain them. Copy them explicitly.
COPY --from=kernel_builder /usr/local/python3.11.17/lib/python3.11/site-packages/vendors /usr/local/python3.11.17/lib/python3.11/site-packages/vendors
RUN --mount=from=wheel_builder,source=/tmp/wheels,target=/tmp/wheels <<'EOF'
set -e
pip install --no-index --no-deps /tmp/kernel-wheels/*.whl /tmp/wheels/*.whl
rm -rf /tmp/kernel-wheels
EOF
