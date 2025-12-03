from contextlib import contextmanager
from chitu.utils import try_import_opt_dep
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")

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