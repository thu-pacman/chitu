import os
import gc
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Union, Dict
from safetensors.torch import save_file

from huggingface_hub import snapshot_download

from transformers.modeling_utils import shard_checkpoint
from mixquant.modules.linear import   MixLinear_GEMM

 

from mixquant.utils.module import get_named_linears, set_op_by_name, weight_only_map,eightbit_only_name
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from accelerate import init_empty_weights, load_checkpoint_in_model, infer_auto_device_map

class BaseForCausalLM(nn.Module):
    def __init__(self, model, model_type, is_quantized, quant_config):
        super().__init__()
        self.model:PreTrainedModel = model
        self.model_type:str = model_type
        self.is_quantized:bool = is_quantized
        self.search_result = None
        self.quant_config: Dict = quant_config
    
    def to(self, device: str):
        return self.model.to(device)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)


    @torch.no_grad()
    def quantize_mix(self, tokenizer=None, quant_config={},
                       calib_data: Union[str, List[str]]="pileval", 
                       split="train", text_column="text", weight_only_map_str = None):
        self.quant_config = quant_config
        quant_config["version"] = "MIX"
        quant_config["q_group_size"] = 128
        from  mixquant.quantize.mixquant import MixQuantizer

        quantizer = MixQuantizer(
            self, self.model, tokenizer, quant_config["w_bit"], quant_config["q_group_size"],
            quant_config["version"], weight_only_map_str = weight_only_map_str)
        quantizer.quantize()
        self.is_quantized = True




    @torch.no_grad()
    def quantize_quik(self, tokenizer=None, quant_config={},
                       calib_data: Union[str, List[str]]="pileval", 
                       split="train", text_column="text"):
        self.quant_config = quant_config
        quant_config["version"] = "QUIK"
        quant_config["q_group_size"] = -1
        from  mixquant.quantize.mixquant import MixQuantizer

        quantizer = MixQuantizer(
            self, self.model, tokenizer, quant_config["w_bit"], quant_config["q_group_size"],
            quant_config["version"])
        quantizer.quantize_QUIK()
        self.is_quantized = True

    @staticmethod
    def fuse_layers(model, quant_config,mix=False,cache=None):
        pass

    def save_quantized(self, save_dir, safetensors=False, shard_size="10GB"):
        save_dir = save_dir[:-1] if save_dir[-1] == '/' else save_dir

        # Save model
        class EmptyModule(nn.Module):
            def __init__(self): super(EmptyModule, self).__init__()
            def forward(self, x): return x

        # Save model files with empty state dict
        self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

        # Remove empty state dict
        if os.path.exists(f'{save_dir}/pytorch_model.bin'):
            os.remove(f'{save_dir}/pytorch_model.bin')
        else:
            print("do not have the file ", f'{save_dir}/pytorch_model.bin')
        # model_name has no extension, add it when saving state_dict
        model_name = 'model.safetensors' if safetensors else 'pytorch_model.bin'

        # shard checkpoint into chunks (10GB default)
        shards, index = shard_checkpoint(
            self.model.state_dict(), 
            max_shard_size=shard_size, 
            weights_name=model_name
        )

        for shard_file, shard in shards.items():
            if safetensors:
                # safetensors must be in the same memory, so we duplicate and use contiguous memory
                shard = {k: v.clone().contiguous() for k, v in shard.items()}
                save_file(shard, os.path.join(save_dir, shard_file), metadata={"format": "pt"})
            else:
                torch.save(shard, os.path.join(save_dir, shard_file))

        # save shard index
        if index is not None:
            with open(f'{save_dir}/{model_name}.index.json', 'w+') as file:
                file.write(json.dumps(index, indent=4))

        # Save config
        with open(f'{save_dir}/quant_config.json', 'w+') as file:
            file.write(json.dumps(self.quant_config, indent=4))
        
        
    @classmethod
    def from_pretrained(self, model_path, model_type, torch_dtype: torch.dtype = torch.float16, 
                        trust_remote_code=True, safetensors=False, device_map=None,
                        mix = True,
                        **model_init_kwargs):
        

        # Get weights path and quant config
        model_weights_path, config, quant_config = self._load_config(
            self, model_path, '', safetensors, trust_remote_code=trust_remote_code
        )
        if device_map is None:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)

            # Get device map
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=[self.layer_type], 
                dtype=torch_dtype
            )
            del model

        # If not quantized, must load with AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            **model_init_kwargs
        )

        model.eval()

        return self(model, model_type, is_quantized=False, quant_config=quant_config)
    



    @classmethod
    def from_quantized(self, model_path, model_type, model_filename='', 
                             max_new_tokens=None, torch_dtype=torch.float16, 
                             trust_remote_code=True, safetensors=False, is_quantized=True, 
                             fuse_layers=False, version="1.0",
                             max_memory=None, offload_folder=None,
                             mix=False, cache=None):
        # [STEP 1-2] Load weights path and configs
        model_weights_path, config, quant_config = self._load_config(
            self, model_path, model_filename, safetensors, version, 
            trust_remote_code, max_new_tokens=max_new_tokens
        )
        self.quant_config = quant_config
        # [STEP 3] Load model
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        
  
        if mix is True:
            if self.quant_config["version"] == "QUIK":
                print("-------------quik----------")
                self._load_quik_quantized_modules(self, model, cache)
                fuse_layers = False   
                
            else:

                print("-------------mix----------")
                self._load_mix_quantized_modules(self, model, cache)    
        else:
            raise NotImplementedError
        

        model.tie_weights()

        # Get device map
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[self.layer_type], 
            max_memory=max_memory,
            dtype=torch_dtype
        )
        print(device_map)
        # Load checkpoint
        #device_map = {"":"cuda"}

        load_checkpoint_in_model(
            model,
            checkpoint=model_weights_path,
            device_map=device_map,
            offload_folder=offload_folder,
            dtype=torch_dtype
        )
        
        # Dispath to devices

        if fuse_layers:
            self.fuse_layers(model, quant_config, mix, cache)

            # Offloading dispatch
        from accelerate import dispatch_model
        model = dispatch_model(
            model,
            device_map=device_map,
            offload_dir=offload_folder
        )



        return self(model, model_type, is_quantized=is_quantized, quant_config=quant_config)

    def _load_config(self, model_path, model_filename, safetensors=False, 
                           version="Mix", trust_remote_code=True, max_new_tokens=4096):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*"])
            else:
                ignore_patterns.append("*.safetensors*")
            
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
        
        if model_filename != '':
            model_weights_path =  model_filename
        else:
            model_weights_path = model_path

        # [STEP 2] Load config and set sequence length
        quant_config_path = f'{model_path}/quant_config.json'
        if os.path.exists(quant_config_path):
            with open(quant_config_path, 'r') as file:
                quant_config = json.loads(file.read())
        else:
            print("use oneline quant methods")
            quant_config = {"w_bit": 0, "version": version}
        
        
        # Load model config and set max generation length
        if max_new_tokens is None and hasattr(self, 'max_new_tokens_key'):
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            config.max_new_tokens = getattr(config, self.max_new_tokens_key)
        else:
            max_new_tokens = 2048 if max_new_tokens is None else max_new_tokens
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            config.max_new_tokens = max_new_tokens
        
        print(config)
        return model_weights_path, config, quant_config




    def _load_mix_quantized_modules(self, model, MixGemmcache):
        # Real quantization of weights

        # Get blocks of model
        layers = self.get_model_layers(model)

        if isinstance(model.config.architectures,list):
            name = model.config.architectures[0]
        else:
            name = model.config.architectures
        weight_only_name = weight_only_map[ name ]
        for i in tqdm(range(len(layers)), desc="Replacing mixed layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            
            for name, module in named_linears.items():

                

                name_ = str(i) + name

                q_linear =  MixLinear_GEMM.from_linear(module,
                                        bit =  bit,
                                        weight_only = True, 
                                        init_only = True,
                                        cache = MixGemmcache, name = name_, fp_features_num = fp_features_num)
                


 

 
                q_linear.to(next(layer.parameters()).device)
                
                set_op_by_name(layer, name, q_linear)


            torch.cuda.empty_cache()
            gc.collect()



    def _load_quik_quantized_modules(self, model, MixGemmcache):
            # Real quantization of weights
            from mixquant.modules.qlinear import   MixedQLinear
            # Get blocks of model
            layers = self.get_model_layers(model)

            if isinstance(model.config.architectures,list):
                name = model.config.architectures[0]
            else:
                name = model.config.architectures
            weight_only_name = weight_only_map[ name ]
            for i in tqdm(range(len(layers)), desc="Replacing quik layers..."):
                layer = layers[i]

                # Get every linear layer in a block
                named_linears = get_named_linears(layer)

                
                for name, module in named_linears.items():


                    weight_8bit_only = False

 
 

                    for key in eightbit_only_name:
                        if key in  name:
                            weight_8bit_only = False 



                    if weight_8bit_only is True:

                        q_linear =  MixedQLinear.from_linear(module,
                                    module.in_features, module.out_features,  
                                    fp_features_num=256, init_only = True,
                                    bits=8)


                    else:
                        q_linear =  MixedQLinear.from_linear(module,
                                    module.in_features, module.out_features,  
                                    fp_features_num=256, 
                                    init_only = True,
                                    bits=4)

                    
                    q_linear.to(next(layer.parameters()).device)
                    set_op_by_name(layer, name, q_linear)


                torch.cuda.empty_cache()
                gc.collect()
