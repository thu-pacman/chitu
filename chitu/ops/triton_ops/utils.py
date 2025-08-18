# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import random
import time
import functools
from logging import getLogger

import torch
import triton.language as tl

logger = getLogger(__name__)

# Triton does not support explicitly typed immediate values. Instead, it looks for
# the narrowest type that can hold the value (see https://triton-lang.org/main/python-api/triton-semantics.html).
# This means that if you use a hex value for a nagative signed integer, it will be
# interpreted as a wider unsigned integer. Starting from triton 3.3.1, this results
# in an error when you combine this integer with a signed variable in an operator,
# for example `x & 0x80000000`. Therefore, we need to define these constants here.
SIGNED_INT32_0x87F00000 = tl.constexpr(0x87F00000 - 0x100000000)
SIGNED_INT16_0x81C0 = tl.constexpr(0x81C0 - 0x10000)
SIGNED_INT16_0x87F0 = tl.constexpr(0x87F0 - 0x10000)
SIGNED_INT8_0x9C = tl.constexpr(0x9C - 0x100)


def to_triton_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")


def auto_retry_triton_compilation(fn):
    """
    Avoid file confict introduced by Triton compiler.

    Triton kernels needs to be compiled at the first run, and the Triton compiler uses
    `~/.triton/cache/` for temporary files. However, in distributed envrionment where
    `~` is mounted by NFS, these files may conflict due to the lack of locking mechanism
    in NFS.

    This function simply retries the compilation if the error is related to file conflict.
    """

    # TODO: Use a better way to avoid file conflict. For example, we can create a symlink
    # from `~/.triton/cache` to a local directory, or we can make use of `torch.distributed`
    # to synchronize the compilation.

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        i = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i >= 30:
                    raise e
                i += 1
                msg = str(e)
                if (
                    "cannot stat shared object: Stale file handle" in msg
                    or "No such file or directory"
                    and "/.triton/cache/" in msg
                ):
                    time.sleep(random.random() * 2 + 1)
                    continue
                raise e

    return wrapped


def auto_tuning_logger(args, *, name: str, **kwargs):
    # NOTE: there are more info in `args`, but normally we don't print it,
    # because there are large tensors inside, which is a run time performance
    # overhead to print them. You can temporarily print them if you want to
    # debug.
    logger.debug(
        f"Tuning {name}. Trying: "
        + ", ".join([f"{key}={kwargs[key]}" for key in kwargs])
    )
