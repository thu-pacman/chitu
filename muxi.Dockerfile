FROM cr.metax-tech.com/public-library/maca-pytorch:3.8.2.6-torch2.6-py310-ubuntu24.04-amd64 AS base

SHELL ["/bin/bash", "-c"]

ARG optional_deps=''
ARG chitu_setup_jobs=''
ARG enable_cython='true'
ARG enable_test='false'
ARG pypi_mirror=''

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
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends g++-11 infiniband-diags curl; \
    rm -rf /var/lib/apt/lists/*

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

RUN if [ "${enable_test}" = "true" ]; then \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends expect vim tmux telnet htop lsof strace iputils-ping; \
    rm -rf /var/lib/apt/lists/*; \
fi

# NOTE: Test dependencies include:
# - pytest is for test/pytest (for all platforms).
# - matplotlib is for op benchmarks in test/pytest, and benchmarks/visualize_response.py (for all platforms).
RUN if [ "${enable_test}" = "true" ]; then \
    pip install pytest matplotlib; \
fi

# Build-only dependencies (Cython, pybind11, ninja, ...) are needed by the
# local kernel packages and by the wheel stage. Install them in the base so
# they stay cached when a dependency or kernel layer below changes. Keep this
# at the end of the base stage so a change here does not rerun the heavy base
# setup above.
COPY ./requirements-build.txt /tmp/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-build.txt

WORKDIR /workspace/chitu
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks
COPY ./chitu/metrics/grafana ./grafana

ENV CHITU_MUXI_BUILD=1

# MetaX's cpp_extension already adds --offload-arch=xcore1000 in
# COMMON_MXCC_FLAGS. The generic CUDA _get_cuda_arch_flags path is CUDA-only and
# needs a visible card, so make it a no-op for MACA builds. Patching the torch
# module (rather than setting TORCH_CUDA_ARCH_LIST) keeps the native
# --offload-arch=xcore1000 target and applies to the optional kernel packages
# built below, not just to chitu itself.
RUN python <<'PY'
import pathlib
p = pathlib.Path("/opt/conda/lib/python3.10/site-packages/torch/utils/cpp_extension.py")
code = p.read_text()
marker = "# CHITU_MACA_ARCH_PATCH"
if marker not in code:
    code += (
        "\n# CHITU_MACA_ARCH_PATCH\n"
        "if __import__('os').environ.get('CHITU_MUXI_BUILD', '0').strip() == '1':\n"
        "    _get_cuda_arch_flags = lambda cflags=None: []\n"
    )
    p.write_text(code)
PY

# The MACA torch in the base image controls the compiled device targets with
# TORCH_EXTENSION_ENABLE_XC1500_COMPILE:
#   - unset (or empty): COMMON_MXCC_FLAGS only contains --offload-arch=xcore1000,
#     so the image only carries C500 kernels;
#   - set to a non-empty value: both xcore1000 and xcore1500 are compiled into
#     the same binaries, so the image also runs on C600.
# It is a build-time switch, not a per-device runtime choice, and the vendor
# code checks it for truthiness: the string "0" counts as set, so leave it
# unset for a C500-only image.

# Update entrypoint, which calls the original entrypoint of the base image.
#
# If you want to change the base image, use
# `docker inspect --format='Entrypoint: {{.Config.Entrypoint}}' <image>`
# to check its entrypoint.
COPY ./script/entrypoint.sh /chitu-entrypoint.sh
ENTRYPOINT ["/chitu-entrypoint.sh", "/bin/bash", "/entrypoint.sh"]

#####################################
# Dependency resolver stage
#
# Generate the PyPI and local-package requirements files from get_requires.py,
# including the selected optional extras. This stage needs torch only for
# get_requires/csrc imports. Splitting the result keeps a local kernel change
# from invalidating the PyPI dependency layer.
FROM base AS dependency_resolver
WORKDIR /workspace/chitu
COPY . .
RUN ./gen_tmp_requirements_txt.py "${optional_deps}" --kind pypi > /tmp/requirements-pypi.txt
RUN ./gen_tmp_requirements_txt.py "${optional_deps}" --kind local > /tmp/requirements-local.txt

#####################################
# Dependency installer stage
#
# Install only the PyPI dependencies, with the base image's torch pinned by
# `-c`. The local kernels are installed in the next stage, so changing a
# kernel does not download these packages again.
FROM base AS dependency_installer
WORKDIR /workspace/chitu
COPY --from=dependency_resolver /tmp/requirements-pypi.txt /tmp/requirements-pypi.txt
COPY ./script/pip-multi-indices.sh ./script/pip-multi-indices.sh
RUN ./script/pip-multi-indices.sh install --no-build-isolation -r /tmp/requirements-pypi.txt \
    -c <(pip list --format freeze | grep -v -e "pillow" -e "fsspec" -e "numpy" -e "transformers" -e "tokenizers" -e "xgrammar" -e "pytest" -e "build" -e "click") \
    -i "${pypi_mirror}" \
    -i https://pypi.org/simple \
    --timeout 60 --retries 10

#####################################
# Local kernel stage
#
# Build and install the local kernel packages from third_party and
# csrc/cpuinfer. This stage depends only on those sources, so changing one of
# them does not reinstall the PyPI dependencies from the stage above.
FROM dependency_installer AS kernel_builder
WORKDIR /workspace/chitu
COPY ./third_party/muxi_layout_kernels ./third_party/muxi_layout_kernels
COPY ./third_party/muxi_w8a8_kernels ./third_party/muxi_w8a8_kernels
COPY ./third_party/metax_soft_fp4_kernels ./third_party/metax_soft_fp4_kernels
COPY ./csrc/cpuinfer ./csrc/cpuinfer
COPY --from=dependency_resolver /tmp/requirements-local.txt /tmp/requirements-local.txt
# Build wheels for the local kernels instead of installing them, so the final
# image can install them without copying the whole conda environment.
RUN pip wheel --no-build-isolation --no-deps -w /tmp/kernel-wheels -r /tmp/requirements-local.txt

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
# installed there, and this stage must not copy the conda tree to get them.
FROM dependency_installer AS final
WORKDIR /workspace/chitu
# Install the local kernel wheels and the chitu wheel; both are prebuilt.
COPY --from=kernel_builder /tmp/kernel-wheels /tmp/kernel-wheels
RUN --mount=from=wheel_builder,source=/tmp/wheels,target=/tmp/wheels <<'EOF'
set -e
pip install --no-index --no-deps /tmp/kernel-wheels/*.whl /tmp/wheels/*.whl
rm -rf /tmp/kernel-wheels
EOF
