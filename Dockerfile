FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS base

ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ARG optional_deps='flash_attn,flash_mla,flashinfer'
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

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

RUN apt update -y \
    && apt install -y git \
    && apt install -y gcc-10 g++-10

RUN if [ "${enable_test}" = "true" ]; then \
    apt install -y expect; \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest; \
fi

WORKDIR /workspace/chitu
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    bash script/install.sh "${optional_deps}" "${build_jobs}" "${enable_editable_install}" "${enable_cython}"
