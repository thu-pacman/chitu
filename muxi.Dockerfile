FROM mxc500-torch2.1-py310:mc2.29.0.7-ubuntu22.04-amd64 AS base

ARG optional_deps=''
ARG build_jobs=''
ARG enable_editable_install='false'
ARG enable_cython='true'

# The base image uses Conda as the Python environment. We need to activate it
# For `docker build` stage, the most straightforward way is to use `bash --login -c` as the shell
SHELL ["/bin/bash", "--login", "-c"]
# For `docker run` stage, we need an entrypoint
RUN echo "source /etc/profile; \"\$@\"" > /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

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

WORKDIR /workspace/chitu
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt

# The actual installing procedure requries a GPU device, which is not available in the `docker build` stage.
# We delay it to an additional `docker run` stage which runs `script/install.sh`.
