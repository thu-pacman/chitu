# `ARG` used in a `FROM` must be defined on the top, and may be redefined below
ARG base_image='pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel'
ARG is_at_least_blackwell='false'

#####################################
# Base Image Stage
FROM ${base_image} AS base

SHELL ["/bin/bash", "-c"]

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ARG is_at_least_blackwell='false'
ARG optional_deps='flash_attn,flash_mla,flashinfer'
ARG chitu_setup_jobs=''
ARG enable_cython='true'
ARG enable_test='false'
ARG pypi_mirror=''

COPY ./script/is_at_least_blackwell.sh /tmp/is_at_least_blackwell.sh
RUN if /tmp/is_at_least_blackwell.sh "${torch_cuda_arch_list}" && [ "${is_at_least_blackwell}" != "true" ]; then \
    echo "--build-arg is_at_least_blackwell must be 'true' when you have >=10.0 arch in --build-arg torch_cuda_arch_list"; \
    exit 1; \
fi
RUN if ! /tmp/is_at_least_blackwell.sh "${torch_cuda_arch_list}" && [ "${is_at_least_blackwell}" != "false" ]; then \
    echo "--build-arg is_at_least_blackwell must be 'false' when you don't have >=10.0 arch in --build-arg torch_cuda_arch_list"; \
    exit 1; \
fi
RUN if [ "${enable_cython}" != "true" ] && [ "${enable_cython}" != "false" ]; then \
    echo "ARG enable_cython must either be 'true' or 'false'"; \
    exit 1; \
fi
RUN if [ "${enable_test}" != "true" ] && [ "${enable_test}" != "false" ]; then \
    echo "ARG enable_test must either be 'true' or 'false'"; \
    exit 1; \
fi


#####################################
# Environment Setter Stages
#
# Because Dockerfile does not support setting ENV conditionally, we have to
# use separate stages.
FROM base AS env_is_at_least_blackwell_true
ENV FLASH_MLA_DISABLE_SM100=0
ENV ENABLE_NVFP4=1

FROM base AS env_is_at_least_blackwell_false
ENV FLASH_MLA_DISABLE_SM100=1
ENV ENABLE_NVFP4=0


#####################################
# Basic Dependencies Stage
FROM env_is_at_least_blackwell_${is_at_least_blackwell} AS basic_deps

ENV FLASH_ATTENTION_FORCE_BUILD="TRUE"
ENV FLASH_ATTENTION_OFFLINE_BUILD="TRUE"

ENV CHITU_SETUP_JOBS=$chitu_setup_jobs
ENV MAX_JOBS=$CHITU_SETUP_JOBS

# Required for non-interactive apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ENV TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list}"

RUN apt update -y && apt install -y \
    git gcc-11 g++-11 libnuma-dev build-essential cmake ninja-build \
    libibverbs1 ibverbs-providers libibverbs-dev rdma-core curl

# Backward compatibily of include path for software developed for CUDA 12
RUN if python3 -c "import torch; print(int(torch.version.cuda.split('.')[0]) >= 13)" | grep -q "True"; then \
    ln -s /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda; \
fi

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
    pip install pytest lark-oapi matplotlib; \
fi

# Always install build time dependencies. Some dependencies may fail to build
# if some build time dependencies are missing.
COPY ./requirements-build.txt /tmp/requirements-build.txt
COPY ./requirements-build-deep_ep-cu12.txt /tmp/requirements-build-deep_ep-cu12.txt
COPY ./requirements-build-deep_ep-cu13.txt /tmp/requirements-build-deep_ep-cu13.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-build.txt \
        -c <(pip list --format freeze | grep -v "setuptools")
RUN if [[ "${optional_deps}" == *"deep_ep"* ]]; then \
    if python3 -c "import torch; print(int(torch.version.cuda.split('.')[0]) == 13)" | grep -q "True"; then \
        pip install -r /tmp/requirements-build-deep_ep-cu13.txt \
            -c <(pip list --format freeze | grep -v "setuptools"); \
    elif python3 -c "import torch; print(int(torch.version.cuda.split('.')[0]) == 12)" | grep -q "True"; then \
        pip install -r /tmp/requirements-build-deep_ep-cu12.txt \
            -c <(pip list --format freeze | grep -v "setuptools"); \
    else \
        echo "Unsupported CUDA version"; \
        exit 1; \
    fi \
fi

# Triton's built-in assembler may be too old for blackwell. Use the system assembler
# as a workaround. See https://github.com/triton-lang/triton/issues/8539.
RUN if [ "${is_at_least_blackwell}" = "true" ]; then \
    ln -s --force `which ptxas` /opt/conda/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas; \
fi


#####################################
# Dependency Resolver Stage
#
# The only purpose of this stage is to generate a requirements.txt file. This
# stage may trigger rebuild whenever there is any change in the source code,
# but this stage runs fast.
FROM basic_deps AS dependency_resolver

WORKDIR /workspace/chitu
COPY . .

RUN ./gen_tmp_requirements_txt.py "${optional_deps}" > /tmp/requirements.txt


#####################################
# Dependency Installer Stage
#
# This stage installs the dependencies listed in requirements.txt. Some of the
# dependencies may require compilation, so this stage may take a long time, but
# this stage only triggers rebuild when the requirements.txt file changes, or
# this source of the dependencies changes.
FROM basic_deps AS dependency_installer

WORKDIR /workspace/chitu
COPY --from=dependency_resolver /tmp/requirements.txt /tmp/requirements.txt

COPY ./third_party ./third_party
COPY ./csrc/cpuinfer ./csrc/cpuinfer

# Don't use `--mount=type=cache,target=/root/.cache/pip` here, because some dependencies
# compile at install time, and the compile results are environment dependent.
#依赖 pytorch 的库应该一律都需要 --no-build-isolation。因为：
# 1. pytorch 是个构建时依赖。
# 2. pytorch 一般都要使用和具体卡以及其他基础软件（如 cuda）版本相关的版本。
# 3. 如果没有 --no-build-isolation ，pip 会在构建时用单独的环境重新下载所有构建时依赖，此时无法指定上述版本。
RUN pip install --no-build-isolation -r /tmp/requirements.txt \
        -c <(pip list --format freeze | grep -v -e "pillow" -e "fsspec" -e "numpy" -e "transformers" -e "pytest")

#####################################
# Wheel build Stage
#
# This stage build wheel file of chitu.
FROM dependency_installer AS wheel_builder

ARG enable_cython

WORKDIR /workspace/chitu
COPY . .

# build wheel of chitu
RUN ./script/build_for_dist.sh "${enable_cython}"

# verify the wheel was created
RUN cp dist/*.whl /tmp/
RUN ls -al /tmp/

RUN rm -rf /workspace/chitu/*

#####################################
# Build Stage
# 
FROM basic_deps AS build

WORKDIR /workspace/chitu
COPY --from=dependency_installer /opt/conda /opt/conda
COPY --from=wheel_builder /tmp/ /tmp/

# Don't use `--mount=type=cache,target=/root/.cache/pip` here, because some dependencies
# compile at install time, and the compile results are environment dependent.
RUN pip install /tmp/*.whl \
    -c <(pip list --format freeze | grep -v -e "pillow" -e "fsspec" -e "flash-mla" -e "flash_mla" -e "numpy" -e "transformers" -e "pytest" -e 'typing-extensions' -e 'typing_extensions')

RUN rm -rf /tmp/*
COPY ./test ./test
COPY ./script ./script
COPY ./benchmarks ./benchmarks
COPY ./chitu/metrics/grafana ./grafana

# These are optimization flags for NCCL, but according to our tests, they only make things
# worse, so we don't use them.
ENV NCCL_GRAPH_MIXING_SUPPORT=0
ENV NCCL_GRAPH_REGISTER=0
