from typing import List

from chitu.utils import try_import_opt_dep

muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)
tbsgemm, has_tbsgemm = try_import_opt_dep("tbsgemm", "muxi_w8a8_kernels")


def preprocess_weights_for_native_layout(
    checkpoint, rpl_names: List[str], cpl_names: List[str]
):
    def is_rpl_weight(key):
        for name in rpl_names:
            if key.endswith(f".{name}.weight"):
                return True
        return False

    def is_cpl_weight(key):
        for name in cpl_names:
            if key.endswith(f".{name}.weight"):
                return True
        return False

    new_checkpoint = {}
    for key in checkpoint.keys():
        if is_rpl_weight(key):  # Row parallel
            m, k = checkpoint[key].shape
            assert m % 64 == 0
            assert k % 128 == 0
            new_checkpoint[key] = (
                checkpoint[key]
                .reshape(m // 16, 16, k // 8, 8)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(m, k)  # Reshape back to for compatibility
            )
        elif is_cpl_weight(key):  # Column parallel
            m, k = checkpoint[key].shape
            assert m % 64 == 0
            assert k % 128 == 0
            new_checkpoint[key] = (
                checkpoint[key]
                .reshape(m // 16, 16, k // 8, 8)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(m, k)  # Reshape back to for compatibility
            )
        else:
            new_checkpoint[key] = checkpoint[key]
    return new_checkpoint


def linear_layout_contig_x_native_y(x, w, b=None):
    assert x.ndim == 2
    x_is_vector = x.shape[0] == 1
    if not x_is_vector:
        x_transposed = muxi_layout_kernels.layoutB(x)
    # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
    w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
    if x_is_vector:
        y = muxi_layout_kernels.gemv_layoutA(w_transposed, x, bias=b)
        # View as 5D to be compatible with "native layout" but make n's tile to be 1.
        y = y.view(y.shape[1] // 32, 1, 4, 1, 8)
    elif x_transposed.shape[1] * 16 > 256:
        y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed, bias=b)
        y = muxi_layout_kernels.layoutB(y)
    else:
        y = muxi_layout_kernels.gemm_layoutABC(w_transposed, x_transposed, bias=b)
    return y


def linear_layout_native_x_contig_y(x_transposed, w, b=None):
    assert x_transposed.ndim == 5
    x_is_vector = x_transposed.shape[1] == 1 and x_transposed.shape[3] == 1
    # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
    w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
    if x_is_vector:
        y = muxi_layout_kernels.gemv_layoutA(
            w_transposed, x_transposed.view(1, -1), bias=b
        )
    elif x_transposed.shape[1] * 16 > 256:
        y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed, bias=b)
    else:
        y = muxi_layout_kernels.gemm_layoutAB_ContinuousC(
            w_transposed, x_transposed, bias=b
        )
    return y


def linear_layout_contig_x_contig_y(x, w, b=None):
    assert x.ndim == 2
    x_is_vector = x.shape[0] == 1
    if not x_is_vector:
        x_transposed = muxi_layout_kernels.layoutB(x)
    # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
    w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
    if x_is_vector:
        y = muxi_layout_kernels.gemv_layoutA(w_transposed, x, bias=b)
    elif x_transposed.shape[1] * 16 > 256:
        y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed, bias=b)
    else:
        y = muxi_layout_kernels.gemm_layoutAB_ContinuousC(
            w_transposed, x_transposed, bias=b
        )
    return y
