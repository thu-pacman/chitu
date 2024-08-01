
import torch

import triton
import triton.language as tl

from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
 
 

def get_configs_fp_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
                for block_kfp in [32, 64]:
                    for block_n in [32, 64, 128, 256]:
                        num_warps = 2 if block_n <= 64 else 4
                        configs.append(
                            Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_kfp, 'SPLIT_K': 1},
                                num_stages=num_stages, num_warps=num_warps))

    return configs

@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_M': 128, 'BLOCK_N': 256,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        Config({'BLOCK_M': 128, 'BLOCK_N': 256,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_fp_io_bound(),
    key=['M', 'N'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)

@triton.jit
def matmul_kernelfp16(A, B, C, M, N, K,
            stride_amfp, stride_akfp,  #
            stride_bkfp, stride_bnfp,  #
            stride_cmfp, stride_cnfp,
            BLOCK_M: tl.constexpr, 
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr, 
            SPLIT_K: tl.constexpr,  
            GROUP_M: tl.constexpr
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
 
    # pointers
 
    rkfp =  tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_amfp + rkfp[None, :] * stride_akfp)
    B = B + (rkfp[:, None] * stride_bkfp + rbn[None, :] * stride_bnfp)

    accfp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rmfp = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rnfp = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)


    afp = tl.zeros((BLOCK_M, BLOCK_K), dtype=C.dtype.element_ty)
    bfp = tl.zeros((BLOCK_K, BLOCK_N), dtype=C.dtype.element_ty)
    C = C + (rmfp[:, None] * stride_cmfp + rnfp[None, :] * stride_cnfp)  
    mask = (rm < M)[:, None] & (rn < N)[None, :]



    K_ = tl.load(K + 0)
    if K_ == 0:

        return 

    maxK = tl.cdiv(K_, BLOCK_K )
    for k in range(0, maxK - 1):

        afp = tl.load(A)
        bfp = tl.load(B)

        A += BLOCK_K   * stride_akfp
        B += BLOCK_K   * stride_bkfp     

        accfp = tl.dot(afp, bfp, accfp, out_dtype=tl.float32, allow_tf32=False)

    k  = maxK - 1
    if  K_ % ( BLOCK_K ) == 0:
        afp = tl.load(A)
        bfp = tl.load(B)
    else:
        k_remainingfp = K_ - k * (BLOCK_K )                
        afp = tl.load(A, mask=rkfp[None, :] < k_remainingfp, other=0.0)
        bfp = tl.load(B, mask=rkfp[:, None] < k_remainingfp, other=0.0)

    accfp = tl.dot(afp, bfp, accfp, out_dtype=tl.float32, allow_tf32=False)

    accfp = accfp.to(tl.float16)

    # rematerialize rm and rn to save registers


    tl.store(C, accfp, mask=mask)
 




def matmulfp16( afp, bfp, cfp16, M, N,  K):
 
    grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])

 

    matmul_kernelfp16[grid](
        afp, bfp, cfp16,M, N,  K,
        1, M,  #
        N, 1,  #
        N, 1,  #
        GROUP_M=8
    )
 
    return 