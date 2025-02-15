#!/usr/bin/bash

export CINFER_WITH_CYTHON=1

# Some of our packages have platform-specific versions, e.g. torch and flash_attn shipped by MUXI.
# So don't let `pip wheel` download them. Instead, call `pip wheel` with `--no-deps` and provide a
# list of packages to build.
pip3 wheel --no-build-isolation --no-deps -w dist/ -r <(pip3 freeze | grep -E "CInfer|muxi_layout_kernels")
