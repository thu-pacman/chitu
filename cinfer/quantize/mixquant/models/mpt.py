from .base import BaseAWQForCausalLM
from typing import Dict
from transformers.models.mpt.modeling_mpt import MptBlock as OldMptBlock, MptForCausalLM

class MptAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MPTBlock"
    max_new_tokens_key = "max_seq_len"

    @staticmethod
    def fuse_layers(model: MptForCausalLM, quant_config: Dict):
        fuser = MptFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: MptForCausalLM):
        return model.transformer.blocks
    
    @staticmethod
    def get_act_for_scaling(module: OldMptBlock):
        return dict(
            is_scalable=True,
            scale_name="ffn.act",
            scale_layer=module.ffn.act,
            scale_shape=module.ffn.up_proj.out_features
        )
    
    @staticmethod
    def move_embed(model: MptForCausalLM, device: str):
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: OldMptBlock, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.norm_1,
            layers=[module.attn.Wqkv],
            inp=input_feat['attn.Wqkv'],
            module2inspect=module.attn,
            kwargs=module_kwargs
        ))

        # attention output
        layers.append(dict(
            prev_op=module.attn.Wqkv,
            layers=[module.attn.out_proj],
            inp=input_feat['attn.out_proj']
        ))

        # linear 1
        layers.append(dict(
            prev_op=module.norm_2,
            layers=[module.ffn.up_proj],
            inp=input_feat['ffn.up_proj'],
            module2inspect=module.ffn
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.ffn.act,
            layers=[module.ffn.down_proj],
            inp=input_feat['ffn.down_proj']
        ))

        return layers

from typing import List, Tuple
from awq.utils.utils import set_module_name
from awq.modules.fused.block import MPTBlock
from awq.modules.fused.model import MPTModel

class MptFuser:
    def __init__(self, model: MptForCausalLM):
        self.model = model

        self.mpt_blocks: List[Tuple[str, OldMptBlock]] = [
            (name, module) for name, module in self.model.named_modules()
            if 'mptblock' in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldMptBlock
        for module in self.model.transformer.blocks:
            blocks.append(MPTBlock(
                self.model.config.d_model,
                self.model.config.n_heads,
                module.attn.Wqkv,
                module.attn.out_proj,
                module.ffn,
                module.norm_1,
                module.norm_2,
                next(iter(module.state_dict().values())).device, 
                self.model.config.max_new_tokens
            ))

        self.model.transformer = MPTModel(
            self.model.config.vocab_size,
            blocks,
            self.model.transformer.wte,
            self.model.transformer.norm_f,
        )