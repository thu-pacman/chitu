from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig

from chitu.attn_backend import AttnBackend
from chitu.kv_cache.cache_manager import KVCacheManagerBase
from chitu.global_vars import get_global_args
from chitu.models.registry import ModelType, register_model
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer import ThresholdParallelDecoder, CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM
logger = getLogger(__name__)

# Guard to avoid double sglang init when model is instantiated multiple times
_sglang_initialized = False
_sglang_server_args = None


def _init_sglang_for_llada():
    """Initialize sglang parallel state for dLLM (LLaDA2) model compatibility.

    torch.distributed is already initialized by chitu above, so sglang's
    init_distributed_environment will skip init_process_group and only set
    its own global variables. This avoids port conflicts.
    """
    global _sglang_initialized, _sglang_server_args
    if _sglang_initialized:
        return _sglang_server_args

    try:
        from sglang.srt.distributed import (
            init_distributed_environment as sglang_init_dist,
            initialize_model_parallel as sglang_init_mp,
        )
        from sglang.srt.layers.dp_attention import initialize_dp_attention
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.server_args import ServerArgs

        args = get_global_args()
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        tensor_parallel_size = args.infer.tp_size
        expert_parallel_size = getattr(args.infer, "ep_size", 1)
        pipeline_parallel_size = args.infer.pp_size
        model_path = args.models.ckpt_dir

        sglang_init_dist(
            world_size=world_size,
            rank=global_rank,
            distributed_init_method="env://",
            local_rank=global_rank % torch.cuda.device_count(),
            backend="nccl",
        )
        sglang_init_mp(
            tensor_model_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=expert_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            backend="nccl",
        )
        server_args = ServerArgs(
            model_path=model_path,
            enable_dp_attention=True,
            trust_remote_code=True,
            tp_size=tensor_parallel_size,
            dp_size=1,
            pp_size=pipeline_parallel_size,
        )
        try:
            from sglang.srt.server_args import set_global_server_args_for_scheduler

            set_global_server_args_for_scheduler(server_args)
        except ImportError:
            pass
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        initialize_dp_attention(
            server_args=server_args,
            model_config=model_config,
        )
        initialize_moe_config(server_args)
        _sglang_initialized = True
        _sglang_server_args = server_args
        logger.info("sglang parallel state initialized for dLLM model")
        return server_args
    except Exception as e:
        logger.warning(f"Failed to initialize sglang parallel state: {e}")
        return None


@register_model(ModelType.LLADA)
class TransformerLLaDA(nn.Module):
    def __init__(
        self,
        params,
        cache_managers: dict[str, KVCacheManagerBase],
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        merge_qkv_gate_up: bool = False,
        **kvargs,
    ):
        super().__init__()
        config_path = params.get("model_config_path") or params.get("ckpt_dir")
        model_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        torch.set_default_dtype(torch.bfloat16)
        self.server_args = _init_sglang_for_llada()
        print(f"{model_config=}")
        self.model = LLaDA2SGLangLM(config=model_config, expert_map_path='.')
        ## hard code here
        self.max_length = 2048
        self.aligned_lengths = [32, 64, 96, 128]
        self.supported_batch_sizes = [1,2,4,8,16,32]
        self.device = torch.device("cuda")
        ## need to add more decoder here, store this option in config (llada.yaml)
        self.decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=156895, eos_id=156892)

    def _materialize_meta_tensors(self, module: nn.Module, device: str) -> None:
        """Convert meta tensors to real device (e.g. expert_bias not in checkpoint)."""
        target = torch.device(device)
        for key in list(module._parameters.keys()):
            param = module._parameters[key]
            if param is not None and param.device.type == "meta":
                module._parameters[key] = nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device=target)
                )
        for key in list(module._buffers.keys()):
            buf = module._buffers[key]
            if buf is not None and buf.device.type == "meta":
                module._buffers[key] = torch.empty(
                    buf.shape, dtype=buf.dtype, device=target
                )
        for child in module.children():
            self._materialize_meta_tensors(child, device)

    def _refresh_correction_bias_refs(self, module: nn.Module) -> None:
        """Refresh correction_bias refs after materialization.

        MLP/TopK hold refs to gate.expert_bias; after replacing meta param they
        still point to old meta tensor, causing moe_fused_gate to fail.
        moe_fused_gate expects input and bias in same dtype (float32).
        """
        for m in module.modules():
            if (
                hasattr(m, "gate")
                and hasattr(m.gate, "expert_bias")
                and m.gate.expert_bias is not None
            ):
                bias = m.gate.expert_bias
                if bias.dtype != torch.float32:
                    m.gate._parameters["expert_bias"] = nn.Parameter(
                        bias.to(torch.float32)
                    )
                    bias = m.gate.expert_bias
                if hasattr(m, "correction_bias"):
                    m.correction_bias = bias
                if hasattr(m, "topk") and hasattr(m.topk, "topk_config"):
                    cfg = m.topk.topk_config
                    if hasattr(cfg, "correction_bias") and cfg.correction_bias is not None:
                        cfg.correction_bias = bias

    def load_weights(self, ckpt_dir: str, device: str = "cuda"):
        # LLaDA 在默认设备上构建（与 dinfer 一致），直接 load_weights，不使用 meta/to_empty。
        inner = self.model
        print("模型结构如下：")
        print(inner)
        torch.set_default_dtype(torch.bfloat16)
        inner.load_weights(ckpt_dir, device=device)
        # 确保 correction_bias 引用与 dtype 正确（与 dinfer 的 dtype 转换一致）。
        self._refresh_correction_bias_refs(inner)
        # ModelRunner is not nn.Module; use object.__setattr__ to bypass nn.Module's check.
        if "model" in self._modules:
            del self._modules["model"]
        object.__setattr__(
            self,
            "model",
            ModelRunner(
                inner,
                self.device,
                server_args=self.server_args,
                max_length=self.max_length,
                prefill_lengths=self.aligned_lengths,
                enable_cuda_graph=True,
                supported_batch_sizes=self.supported_batch_sizes,
                use_cross_block=False,
                enable_compile=True,
            ),
        )
