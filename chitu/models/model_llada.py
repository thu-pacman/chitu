from typing import Optional
import torch
import torch.nn as nn


## directly import dinfer model


from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.decoding.diffusion_runner import ModelRunner

from chitu.attn_backend import AttnBackend
from chitu.cache_manager import KVCacheManagerBase
from chitu.models.model import Attention, RMSNorm, Transformer, TransformerBlock
from chitu.models.registry import ModelType, register_model
from transformers import AutoConfig, AutoModelForCausalLM
from sglang.srt.server_args import ServerArgs


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
        print(f"{params=}")
        config_path = params.get("model_config_path") or params.get("ckpt_dir")
        model_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        torch.set_default_dtype(torch.bfloat16)
        self.model = LLaDA2SGLangLM(config=model_config)

    def apply(self, fn, *args, **kwargs):
        # Skip recursive apply into self.model to avoid method name collision:
        # LLaDA2SGLangLM contains UnquantizedFusedMoEMethod which overrides apply()
        # with an incompatible signature. Since all weights are already on CUDA after
        # load_weights(), the _move_one_module_to_device step is unnecessary.
        fn(self)
        return self

    def load_weights(self, ckpt_dir: str, device: str = "cuda"):
        # Materialize meta-device parameters to real device before loading weights,
        # because chitu builds the model under `torch.device("meta")` context.
        self.model.to_empty(device=device)
        self.model.load_weights(ckpt_dir, device=device)