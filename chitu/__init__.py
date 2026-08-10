# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Package-level logging setup is for normal runtime imports. Boot-time and docs
# generation imports avoid runtime dependencies such as torch.
should_setup_logging = not (
    getattr(sys, "frozen", False)
    or (os.environ.get("APPIMAGE") == "SOURCE")
    or (os.environ.get("CHITU_HTTP_API_DOCS") == "1")
)

if should_setup_logging:
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
