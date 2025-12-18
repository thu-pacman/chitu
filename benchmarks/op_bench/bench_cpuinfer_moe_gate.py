# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_opt_dep
from chitu.ops.moe_gate import moe_gate

from benchmarks.op_bench.bench_util import Benchmark, do_bench, perf_report

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


def cpuinfer_moe_gate(qlen, scores, correction_bias, CPUInfer, cpu_moe_gate, topk):
    indices = torch.zeros((qlen, topk), dtype=torch.int64).contiguous()
    weights = torch.zeros((qlen, topk), dtype=torch.bfloat16).contiguous()

    CPUInfer.submit(
        cpu_moe_gate.forward(
            qlen,
            scores.data_ptr(),
            correction_bias.data_ptr() if correction_bias is not None else 0,
            indices.data_ptr(),
            weights.data_ptr(),
        )
    )
    CPUInfer.sync()

    return indices, weights


@perf_report(
    Benchmark(
        x_names=["num_experts"],
        x_vals=[8, 16, 32, 64],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe-gate-performance",
        args={
            "num_expert_groups": 2,
            "topk_group": 1,
            "topk_as_topk_group_criteria": 2,
            "topk": 2,
            "qlen": 32,
            "score_func": "softmax",
            "use_correction_bias": True,
        },
    )
)
def benchmark(
    num_experts,
    num_expert_groups,
    topk_group,
    topk_as_topk_group_criteria,
    topk,
    qlen,
    score_func,
    use_correction_bias,
    provider,
):
    if provider == "cpuinfer" and not has_cpuinfer:
        return float("nan")

    if num_experts % num_expert_groups != 0:
        return float("nan")

    group_max_len = 1024
    hidden_type = 30

    scores = torch.randn((qlen, num_experts), dtype=torch.bfloat16).contiguous()

    correction_bias = None
    if use_correction_bias:
        correction_bias = torch.randn((num_experts,), dtype=torch.bfloat16).contiguous()

    if provider == "torch":
        ms = do_bench(
            lambda: moe_gate(
                scores,
                topk,
                num_expert_group=num_expert_groups,
                topk_group=topk_group,
                topk_as_topk_group_criteria=topk_as_topk_group_criteria,
                e_score_correction_bias=correction_bias,
                score_func=score_func,
                impl="torch",
            )
        )
    elif provider == "cpuinfer":
        CPUInfer = cpuinfer.CPUInfer("physical_core")
        config = cpuinfer.moe_gate.MoEGateConfig(
            num_experts,
            num_expert_groups,
            topk,
            topk_group,
            group_max_len,
            score_func,
            use_correction_bias,
            hidden_type,
        )
        cpu_moe_gate = cpuinfer.moe_gate.MoEGate(config)

        CPUInfer.submit(cpu_moe_gate.warm_up())
        CPUInfer.sync()

        ms = do_bench(
            lambda: cpuinfer_moe_gate(
                qlen, scores, correction_bias, CPUInfer, cpu_moe_gate, topk
            )
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if has_cpuinfer:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: cpuinfer module not available")
