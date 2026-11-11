FROM harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10 AS base

SHELL ["/bin/bash", "-c"]

ARG optional_deps=''
ARG chitu_setup_jobs=''
ARG enable_editable_install='false'
ARG enable_cython='true'
ARG enable_test='false'
ARG pypi_mirror=''
ARG build_for_shca='false'
# Pin the GPU arch so compilation does not depend on runtime device detection
# (the `docker build` stage has no /dev/kfd device). PyTorch's ROCm build path
# reads `PYTORCH_ROCM_ARCH`, not `TORCH_CUDA_ARCH_LIST`; the latter is CUDA-only
# and rejects `gfx*` names. This also keeps the hipcc arch argument stable,
# which ccache relies on for cache hits.
ARG hygon_arch='gfx936'

ENV CHITU_SETUP_JOBS=$chitu_setup_jobs
ENV MAX_JOBS=$CHITU_SETUP_JOBS
ENV TORCH_CUDA_ARCH_LIST=$hygon_arch

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
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt update -y && apt install -y curl ccache

# Route hipcc / clang / gcc through ccache so the Cython/C++ kernel
# compilation in the second stage is cached across builds. ccache is installed
# above; the cache directory itself is mounted at `docker run` time
# (see two-stage-docker-build.sh).
RUN mkdir -p /usr/lib/ccache && \
    for c in hipcc clang clang++ gcc g++ cc c++; do \
        if command -v "$c" >/dev/null 2>&1; then \
            ln -sf /usr/bin/ccache "/usr/lib/ccache/$c"; \
        fi; \
    done
ENV PATH="/usr/lib/ccache:$PATH"
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_COMPILERCHECK=content
ENV CCACHE_MAXSIZE=20G
# Subprojects (deepgemm/flash_mla/...) are built in pip's per-build temp dirs.
# Don't hash the current working directory so the random temp dir name does not
# bust the cache.
ENV CCACHE_NOHASHDIR=true
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

# /opt/mpi in the `harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10` base image is broken, but
# the base image has another `/usr/mpi/gcc/openmpi-4.1.7a1` installed by `apt`. Use this instead.
RUN rm -rf /opt/mpi
ENV LD_LIBRARY_PATH="/usr/mpi/gcc/openmpi-4.1.7a1/lib:$LD_LIBRARY_PATH"
ENV PATH="/usr/mpi/gcc/openmpi-4.1.7a1/bin:$PATH"

# Install dtk_llvm, caching the .run installer in a BuildKit cache mount so it
# is only downloaded once across builds.
RUN --mount=type=cache,target=/var/cache/chitu \
    if [ ! -f /var/cache/chitu/dtk_llvm.run ]; then \
        curl -L --retry 3 --retry-delay 5 -o /var/cache/chitu/dtk_llvm.run https://download.sourcefind.cn:65024/file/4/dtk_llvm/dtk_llvm.run; \
    fi && \
    chmod +x /var/cache/chitu/dtk_llvm.run && \
    /var/cache/chitu/dtk_llvm.run

COPY ./script/pip-multi-indices.sh ./script/pip-multi-indices.sh
# NOTE: ./third_party/hygon_wheels/triton-3.5.1+das.opt1.dtk2604.torch290-cp310-cp310-manylinux_2_28_x86_64.whl is different from
#       `triton==3.5.1+das.opt1.dtk2604.torch290` from https://pypi.sourcefind.cn/release/dtk/, although they have the same name.
RUN --mount=source=./third_party/hygon_wheels,destination=./third_party/hygon_wheels \
    --mount=type=cache,target=/root/.cache/pip \
  ./script/pip-multi-indices.sh install \
    -i "${pypi_mirror}" \
    -i https://pypi.sourcefind.cn/release/dtk/ \
    -i https://pypi.sourcefind.cn/nightly/dtk/ \
    ./third_party/hygon_wheels/aiter-0.1.2+das.opt1.dtk2604.torch290.2605071840.g1f8f50-cp310-cp310-linux_x86_64.whl \
    ./third_party/hygon_wheels/triton-3.5.1+das.opt1.dtk2604.torch290-cp310-cp310-manylinux_2_28_x86_64.whl \
    "lightop==0.6.0+das.dtk2604.torch290.2605220936.g6bfaca" \
    "lmslim==0.3.1+das.opt4.dtk2604.torch290.2604281437.g61fdfe" \
    -c <(pip list --format freeze | grep -v -e "setuptools" -e "numpy" -e "triton" -e "aiter" -e "lightop" -e "lmslim")

# Build dependencies are needed by the wheel stage and by some of the local
# kernel packages installed in the dependency installer.
COPY ./requirements-build.txt /tmp/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-build.txt

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
# Effective ROCm arch pin. `TORCH_CUDA_ARCH_LIST` above is CUDA-only and is kept
# for any CUDA-style consumers; PyTorch's ROCm extension build reads
# `PYTORCH_ROCM_ARCH`, which is what makes `/dev/kfd` unnecessary here.
ENV PYTORCH_ROCM_ARCH=$hygon_arch
ENV HIP_GRAPH_ACCUMULATE_DISPATCH=1
ENV GPU_MAX_HW_QUEUES=3

# aiter uses `AITER_JIT_DIR` as jit directory if it is set, or uses its
# default directory. Although they are logically equivalent, the default-
# directory branch in aiter does not handle race during directory creation.
# Only the `AITER_JIT_DIR` branch is multi-process-safe. See aiter's
# `jit/core.py` for details.
ENV AITER_JIT_DIR=/tmp/chitu-aiter-jit

# Update entrypoint, which calls the original entrypoint of the base image.
#
# If you want to change the base image, use
# `docker inspect --format='Entrypoint: {{.Config.Entrypoint}}' <image>`
# to check its entrypoint.
COPY ./script/entrypoint-hygon.sh /chitu-entrypoint.sh
ENTRYPOINT ["/chitu-entrypoint.sh", "/usr/local/bin/docker-entrypoint.sh"]

#####################################
# Dependency resolver stage
#
# Generate the PyPI and local-package requirements files from get_requires.py,
# including the selected optional extras. This stage needs torch only for
# get_requires/csrc imports, hence the hyhal mount.
FROM base AS dependency_resolver
WORKDIR /workspace/chitu
COPY . .
RUN --mount=type=bind,from=hyhal,target=/opt/hyhal <<'EOF'
set -e
source /opt/hyhal/env.sh
./gen_tmp_requirements_txt.py "${optional_deps}" --kind pypi > /tmp/requirements-pypi.txt
./gen_tmp_requirements_txt.py "${optional_deps}" --kind local > /tmp/requirements-local.txt
EOF

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
# csrc/cpuinfer. The hyhal mount is only needed here (and in the wheel stage),
# because building the kernels imports the vendor torch, while installing the
# PyPI wheels above does not.
FROM dependency_installer AS kernel_builder
WORKDIR /workspace/chitu
COPY ./third_party/sugon_mixq4_kernels ./third_party/sugon_mixq4_kernels
COPY ./third_party/sugon_w4a8_kernels ./third_party/sugon_w4a8_kernels
COPY ./third_party/hygon_w4a16_kernel ./third_party/hygon_w4a16_kernel
COPY ./third_party/hygon_w8a8_kernel ./third_party/hygon_w8a8_kernel
COPY ./third_party/boltops-hygon ./third_party/boltops-hygon
COPY ./third_party/flashmla_hygon ./third_party/flashmla_hygon
COPY ./third_party/deepgemm_hygon ./third_party/deepgemm_hygon
COPY ./third_party/hygon_wheels ./third_party/hygon_wheels
COPY ./csrc/cpuinfer ./csrc/cpuinfer
COPY --from=dependency_resolver /tmp/requirements-local.txt /tmp/requirements-local.txt
# Build wheels for the local kernels instead of installing them. The final
# image only needs the wheels, so it does not have to copy the whole Python
# tree from this stage (that copy duplicated about 10 GB of the base image).
RUN --mount=type=bind,from=hyhal,target=/opt/hyhal <<'EOF'
set -e
source /opt/hyhal/env.sh
pip wheel --no-build-isolation --no-deps -w /tmp/kernel-wheels -r /tmp/requirements-local.txt
EOF

#####################################
# Wheel build stage
#
# Source is present only in the resolver/installer/wheel stages. The final
# image copies just the wheel, so no source tree ends up in the image.
FROM kernel_builder AS wheel_builder
ARG enable_cython
WORKDIR /workspace/chitu
COPY . .
# hyhal provides librocm_smi64.so.2, which the vendor torch loads at import
# time. The wheel build runs setup.py, so it needs the same mount as the
# dependency installer.
RUN --mount=type=bind,from=hyhal,target=/opt/hyhal <<'EOF'
set -e
source /opt/hyhal/env.sh
./script/build_for_dist.sh "${enable_cython}"
EOF
RUN mkdir -p /tmp/wheels && cp dist/*.whl /tmp/wheels/
RUN rm -rf /workspace/chitu/*

#####################################
# Final image
# Start from dependency_installer, not base: the PyPI dependencies are
# installed there, and this stage must not copy the Python tree to get them.
FROM dependency_installer AS final
WORKDIR /workspace/chitu
# Install the local kernel wheels and the chitu wheel. Both are already built,
# and `--no-index --no-deps` keeps this stage from downloading anything.
COPY --from=kernel_builder /tmp/kernel-wheels /tmp/kernel-wheels
RUN --mount=from=wheel_builder,source=/tmp/wheels,target=/tmp/wheels <<'EOF'
set -e
pip install --no-index --no-deps /tmp/kernel-wheels/*.whl /tmp/wheels/*.whl
rm -rf /tmp/kernel-wheels
EOF
