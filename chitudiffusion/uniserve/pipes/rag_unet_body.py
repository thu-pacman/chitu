import os

# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["PYTORCH_JIT"] = "0"
import time
import torch
import torchperf
from torch import Tensor
import torch._custom_ops
import uniserve_cuda
import uniserve
import logging
import itertools
from uniserve.models import load_diffusers_pipe
import types

import warnings


def _mock_unet_body(*args, **kwargs):
    class DummyModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            warnings.warn(
                "Using dummy UNet body model; original .sdxl_unet_body_transformed file is missing!"
            )
            return None

    warnings.warn("Importing dummy sdxl_unet_body_transformed; file not found.")
    return DummyModel()


def _mock_unet_body_sd15(*args, **kwargs):
    class DummyModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            warnings.warn(
                "Using dummy UNet body model; original .sd15_unet_body_transformed file is missing!"
            )
            return None

    warnings.warn("Importing dummy sd15_unet_body_transformed; file not found.")
    return DummyModel()


try:
    from .sdxl_unet_body_transformed import sdxl_unet_body_transformed
except ImportError:
    sdxl_unet_body_transformed = _mock_unet_body

try:
    from .sd15_unet_body_transformed import sd15_unet_body_transformed
except ImportError:
    sd15_unet_body_transformed = _mock_unet_body_sd15
from uniserve.models.unet_2d_condition import build_unet, build_unet_input

from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DiffusionPipeline,
)
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile,
    compile_unet,
    CompilationConfig,
)

# torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)


def get_config():
    config = CompilationConfig.Default()
    # Disable fused GEGLU until cutlass kernel stabilizes; the custom op
    # frequently throws `Error Internal` on Ampere+ cards which prevents
    # pipeline execution altogether.  Falling back to the unfused aten
    # sequence only costs a few percent in this path but keeps things running.
    config.enable_fused_linear_geglu = False
    # xformers and Triton are suggested for achieving best performance.
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")
    try:
        import triton

        config.enable_triton = True
    except ImportError:
        print("Triton not installed, skip")
    # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
    config.enable_cuda_graph = True
    return config


def compile_model(m, config):
    device = (
        m.device
        if hasattr(m, "device")
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"
    return compile_unet(m, config)


def move_constant_to_cpu(model):
    for k, v in model.named_buffers():
        print(k, v.device)
        if k in ("_tensor_constant1", "_tensor_constant3", "_tensor_constant5"):
            print(f"Move {k} to cpu")
            setattr(model, k, v.to("cpu"))
            # model[k] = v.to('cpu')


def create_index_2d(hs, ws):
    assert len(hs) == len(ws)
    HxWs = [h * w for h, w in zip(hs, ws)]
    idx_cuda = torch.tensor([hs, ws, HxWs], dtype=torch.int64)
    idx_cpu = idx_cuda.to("cpu")
    return idx_cuda, idx_cpu


def create_index_1d(Lseq):
    idx_cuda = torch.tensor([Lseq], dtype=torch.int64)
    idx_cpu = idx_cuda.to("cpu")
    return idx_cuda, idx_cpu


def create_cum_index_1d(Lseq):
    idx = [0]
    for v in Lseq:
        idx += [idx[-1] + int(v)]
    idx_cuda = torch.tensor(idx, device="cuda", dtype=torch.int32)
    return idx_cuda


def create_unet_body_inputs(hs, ws, model_name: str):
    args, kwargs = build_unet_input(b=len(hs), name=model_name)
    idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
    cum_idx1d_cuda = create_cum_index_1d([h * w for h, w in zip(hs, ws)])
    # emb = full_model.run_head(full_model, *args, **kwargs)
    emb = torch.randn([len(hs), 1280])
    x = torch.randn([4 * sum([h * w for h, w in zip(hs, ws)])])
    return (
        idx1d_cuda,
        idx1d_cpu,
        idx2d_cuda,
        idx2d_cpu,
        cum_idx1d_cuda,
        None,
        None,
        x,
        kwargs["encoder_hidden_states"],
        emb,
    )


def create_unet_head_and_body_inputs(hs, ws, model_name: str):
    args, kwargs = build_unet_input(b=len(hs), h=hs[0], w=ws[0], name=model_name)
    idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
    cum_idx1d_cuda = create_cum_index_1d([h * w for h, w in zip(hs, ws)])
    prompt_cum_idx1d_cuda = create_cum_index_1d([77] * len(hs))
    # emb = full_model.run_head(full_model, *args, **kwargs)
    emb = torch.randn([len(hs), 1280])
    x = torch.randn([4 * sum([h * w for h, w in zip(hs, ws)])])
    args = (x.reshape(args[0].shape), *args[1:])
    return (args, kwargs), (
        idx1d_cuda,
        idx1d_cpu,
        idx2d_cuda,
        idx2d_cpu,
        cum_idx1d_cuda,
        prompt_cum_idx1d_cuda,
        None,
        None,
        x,
        kwargs["encoder_hidden_states"],
        emb,
    )


def create_ragged_unet_head_and_body_inputs(hs, ws, model_name: str):
    args, kwargs = build_unet_input(b=len(hs), h=hs[0], w=ws[0], name=model_name)
    idx2d_cuda, idx2d_cpu = create_index_2d(hs, ws)
    idx1d_cuda, idx1d_cpu = idx2d_cuda[2:], idx2d_cpu[2:]
    cum_idx1d_cuda = create_cum_index_1d([h * w for h, w in zip(hs, ws)])
    prompt_cum_idx1d_cuda = create_cum_index_1d([77] * len(hs))
    emb = torch.randn([len(hs), 1280])
    x = torch.randn([4 * sum([h * w for h, w in zip(hs, ws)])])
    regular_inputs = [
        (
            (x[4 * idx : 4 * idx + 4 * h * w].reshape(1, 4, h, w), *args[1:]),
            {
                "encoder_hidden_states": kwargs["encoder_hidden_states"][i : i + 1],
                "cross_attention_kwargs": None,
                "added_cond_kwargs": (
                    {
                        "text_embeds": kwargs["added_cond_kwargs"]["text_embeds"][
                            i : i + 1
                        ],
                        "time_ids": kwargs["added_cond_kwargs"]["time_ids"][i : i + 1],
                    }
                    if model_name == "sdxl"
                    else None
                ),
                "return_dict": False,
            },
        )
        for i, (h, w, idx) in enumerate(zip(hs, ws, cum_idx1d_cuda[:-1]))
    ]
    if model_name in ["sdxl", "sd15"]:
        return regular_inputs, (
            (args, kwargs),
            (
                idx1d_cuda,
                idx1d_cpu,
                idx2d_cuda,
                idx2d_cpu,
                cum_idx1d_cuda,
                prompt_cum_idx1d_cuda,
                None,
                None,
                x,
                kwargs["encoder_hidden_states"],
                emb,
            ),
        )
    else:
        raise RuntimeError()


# %%
# hss = [v // 8 for v in [256, 256, 256, 512, 512, 512, 768, 768, 768]]
# wss = [v // 8 for v in [256, 512, 768, 256, 512, 768, 256, 512, 768]]

delta = 64
base = 512
hss = [
    v // 8
    for v in [
        base - delta,
        base - delta,
        base - delta,
        base,
        base,
        base,
        base + delta,
        base + delta,
        base + delta,
    ]
]
wss = [
    v // 8
    for v in [
        base - delta,
        base,
        base + delta,
        base - delta,
        base,
        base + delta,
        base - delta,
        base,
        base + delta,
    ]
]

# batch
shape_configs = [
    (
        [hss[j // 2] for j in range(2 * (i + 1))],
        [wss[j // 2] for j in range(2 * (i + 1))],
    )
    for i in range(len(hss))
]

# # # no batch
# shape_configs += [
#     (
#         [hss[j // 2] for j in range(2 * i, 2 * (i + 1))],
#         [wss[j // 2] for j in range(2 * i, 2 * (i + 1))],
#     )
#     for i in range(len(hss))
# ]

# shape_configs = [([64, 64], [64, 64])]
# shape_configs = [([64, 64] * 5, [64, 64] * 5)]
# hs, ws =[32, 32, 64, 64], [32, 32, 64, 64]
# hs, ws = [64, 32, 64, 32], [64, 64, 32, 32]


def get_model(model_name):
    # Save current working directory
    original_cwd = os.getcwd()
    try:
        if model_name == "sdxl":
            # Change to the pipes directory so relative paths in module.py work
            # module.py expects paths like "sdxl_unet_body_transformed/file.pt"
            # which means it should be run from the pipes directory
            pipes_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(pipes_dir)
            model = sdxl_unet_body_transformed().eval().cuda().half()
        elif model_name == "sd15":
            # Change to the pipes directory so relative paths in module.py work
            # module.py expects paths like "sd15_unet_body_transformed/file.pt"
            # which means it should be run from the pipes directory
            pipes_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(pipes_dir)
            model = sd15_unet_body_transformed().eval().cuda().half()
        else:
            raise RuntimeError(f"Unknown model {model_name}")
        return model
    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)


logging.getLogger().setLevel(logging.INFO)


def perf_unet_body(model_name, profiling: bool = False):
    with torch.no_grad():
        model = get_model(model_name)
        move_constant_to_cpu(model)
        model_eager = get_model(model_name)
        move_constant_to_cpu(model_eager)
        config = get_config()
        model = compile_model(model, config)
        print("===== compiled model successfully.=====")
        for bs in [1, 2, 16]:
            for hs, ws in [([64] * bs, [64] * bs)]:
                print(hs, ws)
                # continue

                with torch.device("cuda"):
                    (
                        regular_inputs,
                        (
                            inputs_head,
                            inputs_body,
                        ),
                    ) = create_ragged_unet_head_and_body_inputs(hs, ws, model_name)

                f = lambda: model(*inputs_body)
                t_start = time.time()
                f()
                t_end = time.time()
                t1 = torchperf.cuda_timeit_ms(f)
                # t2 = torchperf.cuda_timeit_host_ms(lambda: model(*inputs))
                print(f"== Time build opt: {t_end-t_start:.2f} s, {t1:.2f} ms", hs, ws)

                if profiling:
                    prof = torchperf.torch_profile_it(
                        f"sfast-ragged-{model_name}-512", f
                    )


def run_unet(head, body, inputs_head, inputs_body, force_no_trace=False):
    emb = head(*inputs_head[0], **inputs_head[1])
    if force_no_trace:
        ret = body(*inputs_body[:-2], emb, inputs_body[-1], hash_key=[0])
    else:
        ret = body(*inputs_body[:-2], emb, inputs_body[-1])
    return ret


def get_compiled_unet_body(model_name: str):
    model = get_model(model_name)
    move_constant_to_cpu(model)
    config = get_config()
    config.enable_cuda_graph = False
    # config.enable_jit = False
    torch._C._get_graph_executor_optimize(False)
    model = compile_model(model, config)
    return model


def get_compiled_unet_head(pipe):
    # pipe = load_diffusers_pipe(model_name)
    pipe.unet.run_head = types.MethodType(
        uniserve.models.unet_2d_condition.run_head, pipe.unet
    )
    # pipe.unet.run_body = types.MethodType(
    #     uniserve.models.unet_2d_condition.run_body, pipe.unet
    # )
    # config = get_config()
    # config.memory_format = None
    # pipe.unet.run_head = compile_model(pipe.unet.run_head, config)
    return pipe.unet.run_head


def get_standard_ans_with_ragged_inputs(pipe, regular_inputs):
    outputs = []
    for args, kwargs in regular_inputs:
        outputs.append(pipe.unet(*args, **kwargs))

    # Consistant with UNet output
    return (torch.cat([v[0].flatten() for v in outputs], dim=0),)


@torch.no_grad()
def perf_unet(model_name, profiling: bool = False):
    pipe = load_diffusers_pipe(model_name)
    head = get_compiled_unet_head(pipe)
    body = get_compiled_unet_body(model_name)
    for bs in [1, 2, 4, 8, 16]:
        for hs, ws in [
            ([64 for i in range(bs)], [64 for i in range(bs)])
            # ([64 + 8 * i for i in range(bs)], [64 + 8 * i for i in range(bs)])
        ]:
            with torch.device("cuda"):
                (
                    regular_inputs,
                    (
                        inputs_head,
                        inputs_body,
                    ),
                ) = create_ragged_unet_head_and_body_inputs(hs, ws, model_name)

            f0 = lambda: run_unet(head, body, inputs_head, inputs_body)
            t_warmup = torchperf.cuda_timeit_ms(f0, warmup=0, iters=1)
            print("== warmup bs", bs, t_warmup, hs, ws)

            y = f0()
            ans = get_standard_ans_with_ragged_inputs(pipe, regular_inputs)
            if torchperf.allclose(y[0], ans[0], 0.01, 0.01, etol=0.01):
                print("== Pass test")
            print("== bs", bs, torchperf.cuda_timeit_ms(f0), hs, ws)
            if profiling:
                torchperf.torch_profile_it(f"sfast-ragged-{model_name}-UNet", f0)


def ragged_modify(pipe, model_name):
    head = get_compiled_unet_head(pipe)
    body = get_compiled_unet_body(model_name)
    return head, body


if __name__ == "__main__":
    # perf_unet_body("sd15", True)
    perf_unet("sd15", False)
