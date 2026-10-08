# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import sys

# Enflame GCU early setup (no-op without torch_gcu / triton_gcu). MUST run before any
# other chitu submodule import: chitu.device_type.get_device_name() caches its result,
# and chitu.ops.triton_ops resolves Triton's active driver at import time. The torch_gcu
# shim maps ".cuda()"->".gcu()" and the "nccl" process-group backend->"eccl";
# "import triton_gcu.triton" activates the GCU backend for the stock `triton` package.
try:
    from torch_gcu import transfer_to_gcu  # noqa: F401

    try:
        import triton_gcu.triton  # noqa: F401
    except Exception:
        pass
except Exception:
    pass

# If this attribute exists, we are running inside a PyInstaller bundle, then
# we skip import unneeded dependencies.
if not getattr(sys, "frozen", False):
    # Some special logging functions like `logger.warning_once` are used
    # across chitu functions. In order to make those functions functional,
    # we configure the logging functions here.
    #
    # NOTE: This means the `logging` library is configured once you import
    # `chitu`.
    from chitu.logging_utils import setup_chitu_logging

    setup_chitu_logging()

    # Patch for transformers >= 5.0 compatibility
    # bytes_to_unicode was moved from gpt2.tokenization_gpt2 to convert_slow_tokenizer
    # This patch makes it available at the old location for backward compatibility
    try:
        import transformers.models.gpt2.tokenization_gpt2 as gpt2_module

        if not hasattr(gpt2_module, "bytes_to_unicode"):
            from transformers.convert_slow_tokenizer import bytes_to_unicode

            gpt2_module.bytes_to_unicode = bytes_to_unicode
    except ImportError:
        pass  # transformers not installed, skip patch
