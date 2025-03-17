from logging import getLogger

import torch

from chitu.models.model import *
from chitu.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from chitu.tokenizer import ChatFormat, Tokenizer

logger = getLogger(__name__)


__all__ = ["quant"]


def replace_with_bnb(model, current_key_name=None):

    import bitsandbytes as bnb

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)
        # if current_key_name_str == "model.model":
        #    continue
        # if (name == "model"):
        #    continue
        if isinstance(
            module, (torch.nn.Linear, ColumnParallelLinear, RowParallelLinear)
        ):
            bnb_module = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            bnb_module.weight.data = module.weight.data
            if module.bias is not None:
                bnb_module.bias.data = module.bias.data
            bnb_module.requires_grad_(False)
            setattr(model, name, bnb_module)
            has_been_replaced = True
        if len(list(module.children())) > 0:
            _, _has_been_replaced = replace_with_bnb(module, current_key_name)
            has_been_replaced = has_been_replaced | _has_been_replaced

        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_gptq(model, current_key_name=None):

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)
        if name != "lm_head":
            if isinstance(
                module, (torch.nn.Linear, ColumnParallelLinear, RowParallelLinear)
            ):
                from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear

                gptq_linear = QuantLinear(
                    8,
                    128,
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    trainable=False,
                    weight_dtype=module.weight.dtype,
                )
                gptq_linear.requires_grad_(False)
                setattr(model, name, gptq_linear)
                has_been_replaced = True
            if len(list(module.children())) > 0:
                _, _has_been_replaced = replace_with_gptq(module, current_key_name)
                has_been_replaced = has_been_replaced | _has_been_replaced

        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_w8a16(model, current_key_name=None):

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)
        if name != "lm_head":
            if isinstance(
                module, (torch.nn.Linear, ColumnParallelLinear, RowParallelLinear)
            ):
                from chitu.quantize.w8a16 import WeightOnlyLinear

                w8a16_linear = WeightOnlyLinear(
                    module.in_features, module.out_features, module.bias is not None
                )
                w8a16_linear.requires_grad_(False)
                setattr(model, name, w8a16_linear)
                has_been_replaced = True
            if len(list(module.children())) > 0:
                _, _has_been_replaced = replace_with_w8a16(module, current_key_name)
                has_been_replaced = has_been_replaced | _has_been_replaced

        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_simple_w8a8(model, current_key_name=None, quant_on_load=False):

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)
        if name != "lm_head":
            if isinstance(
                module, (torch.nn.Linear, ColumnParallelLinear, RowParallelLinear)
            ):
                from chitu.quantize.w8a8 import W8A8Linear

                w8a8_linear = W8A8Linear.from_float(
                    module, model_arch_only=not quant_on_load
                )
                w8a8_linear.requires_grad_(False)
                setattr(model, name, w8a8_linear)
                has_been_replaced = True
            if len(list(module.children())) > 0:
                _, _has_been_replaced = replace_with_simple_w8a8(
                    module, current_key_name, quant_on_load=quant_on_load
                )
                has_been_replaced = has_been_replaced | _has_been_replaced

        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_simple_w8a8_muxi(model, current_key_name=None, quant_on_load=False):

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)
        if name != "lm_head":
            from chitu.models.model import RMSNorm
            from chitu.quantize.muxi_w8a8 import NormAndQuant, W8A8Linear

            if isinstance(
                module, (torch.nn.Linear, ColumnParallelLinear, RowParallelLinear)
            ):

                w8a8_linear = W8A8Linear.from_float(
                    module, model_arch_only=not quant_on_load
                )
                w8a8_linear.requires_grad_(False)
                setattr(model, name, w8a8_linear)
                has_been_replaced = True

            if isinstance(module, (RMSNorm)) and name != "norm":
                new_norm = NormAndQuant.from_float(
                    module, model_arch_only=not quant_on_load
                )
                new_norm.requires_grad_(False)
                setattr(model, name, new_norm)
                has_been_replaced = True

            if len(list(module.children())) > 0:
                _, _has_been_replaced = replace_with_simple_w8a8_muxi(
                    module, current_key_name, quant_on_load=quant_on_load
                )
                has_been_replaced = has_been_replaced | _has_been_replaced

        current_key_name.pop(-1)
    return model, has_been_replaced


def quantize_llmint8(model):
    model, has_been_replaced = replace_with_bnb(model)

    if not has_been_replaced:
        logger.warning("No linear modules were quantized.")

    logger.debug(f"Quantized model: {model}")
    return model


def quantize_gptq(model):
    model, has_been_replaced = replace_with_gptq(model)

    if not has_been_replaced:
        logger.warning("No linear modules were quantized.")

    logger.debug(f"Quantized model: {model}")
    return model


def quantize_awq(model, name="hf-llama"):
    q_config = {"zero_point": True, "q_group_size": 128}

    from chitu import awq

    awq.real_quantize_model_weight(model, w_bit=4, q_config=q_config, init_only=True)

    if name == "llama":
        sd = torch.load(
            "/home/tanyijun/cinfer/quant_cache/Llama3-8B-4bit.pth", map_location="cpu"
        )
        model.load_state_dict(sd)
    elif name == "hf-llama":
        0

    class FP16Trans(torch.nn.Module):
        def __init__(self, toke):
            super().__init__()
            self.tok_embeddings = toke

        def forward(self, x):
            return self.tok_embeddings(x).to(torch.float16)

    if name == "llama":
        model.tok_embeddings = FP16Trans(model.tok_embeddings)
    elif name == "hf-llama":
        model.embed_tokens = FP16Trans(model.embed_tokens)

    logger.debug(f"Quantized model: {model}")

    return model


def quantize_w8a16(model):
    model, has_been_replaced = replace_with_w8a16(model)

    if not has_been_replaced:
        raise NotImplementedError("error load model")

    logger.debug(f"Quantized model: {model}")
    return model


def quantize_simple_w8a8(model, quant_on_load=False):
    model, has_been_replaced = replace_with_simple_w8a8(
        model, quant_on_load=quant_on_load
    )

    if not has_been_replaced:
        raise NotImplementedError("error load model")

    logger.debug(f"Quantized model: {model}")
    return model


def quantize_simple_w8a8_muxi(model, quant_on_load=False):
    model, has_been_replaced = replace_with_simple_w8a8_muxi(
        model, quant_on_load=quant_on_load
    )

    if not has_been_replaced:
        raise NotImplementedError("error load model")

    logger.debug(f"Quantized model: {model}")
    return model


def quant(model, method=None, name="hf-llama", quant_on_load=False):

    if method == "llmint8":
        return quantize_llmint8(model)
    elif method == "awq":
        return quantize_awq(model, name)
    elif method == "gptq":
        return quantize_gptq(model)
    elif method == "w8a16":
        return quantize_w8a16(model)
    elif method == "simple_w8a8":
        return quantize_simple_w8a8(model, quant_on_load=quant_on_load)
    elif method == "simple_w8a8_muxi":
        return quantize_simple_w8a8_muxi(model, quant_on_load=quant_on_load)
    return model
