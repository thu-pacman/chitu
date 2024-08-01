import torch

import triton
import triton.language as tl

from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
def upcast_if_fp8(a):
    if "fp8" in str(a):
        return torch.float16
    return a

_ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]

def get_higher_dtype(a, b):
    a = upcast_if_fp8(a)
    b = upcast_if_fp8(b)
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_kfp in [32, 64]:
                    for block_n in [32, 64, 128, 256]:
                        num_warps = 2 if block_n <= 64 else 4
                        configs.append(
                            Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,'BLOCK_Kfp': block_kfp, 'SPLIT_K': 1},
                                num_stages=num_stages, num_warps=num_warps))
                        # split_k
                        for split_k in [2, 4, 8, 16]:
                            configs.append(
                                Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'BLOCK_Kfp': block_kfp, 'SPLIT_K': split_k},
                                    num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs





@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=5, num_warps=2),
        # good for int8
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1, 'BLOCK_Kfp': 16}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=['M', 'N'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})

@triton.jit
def matmul_kernelint8(x,w, A, B, C, M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #

            Afp, Bfp, Cfp,
            Kfp,
            stride_amfp, stride_akfp,  #
            stride_bkfp, stride_bnfp,  #
            stride_cmfp, stride_cnfp,

            acc_dtype: tl.constexpr,  #
            allow_tf32: tl.constexpr,  #
            fp8_fast_accum: tl.constexpr,  #
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
            BLOCK_Kfp: tl.constexpr,  
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr  #
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)


    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=False)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)


    #Cfp = Cfp + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    # handles write-back with reduction-splitting

    # x_ = tl.load(x + 0)

    # accfp = acc.to(tl.float32) 
    # accfp *=  x_
    # accfp = accfp.to(tl.float16)
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
        
    else:
        tl.atomic_add(C, acc, mask=mask)
        







def matmulint8_fused_dequant(x,w, a, b, afp, bfp, c, cfp16, M, N, K, Kfp):
 
    grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])

 
    allow_tf32=True
    fp8_fast_accum=True
    matmul_kernelint8[grid](
        x, w,
        a, b, c,  #
        M, N, K,  #
        K, 1,  #
        1, K,  #
        N, 1,  #
        afp, bfp, cfp16, Kfp[0],
        Kfp, 1,  #
        1, Kfp,  #
        N, 1,  #
        allow_tf32=allow_tf32,  #
        fp8_fast_accum=fp8_fast_accum,  #
        GROUP_M=8, acc_dtype=tl.int32,AB_DTYPE=None
    )
    return c, cfp16