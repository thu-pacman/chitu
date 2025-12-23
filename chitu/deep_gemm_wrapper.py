from contextlib import contextmanager
from chitu.utils import try_import_opt_dep
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 SGLang Team
# SPDX-SnippetName: dynamic SM configuration for deep_gemm
#
# This implementation of the dynamic SM configuration for deep_gemm is originally from SGLang,
# (https://github.com/sgl-project/sglang/commit/b4c41f7276e224b03cbcb9121c01a8be88109520),
# licensed under Apache 2.0.
@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)

       
# SPDX-SnippetEnd
