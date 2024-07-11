import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb


if __name__ == "__main__":

    model_path = "/home/hkz/Meta-Llama-3-8B-Instruct-HF"

    enc = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    #print(model)
    
    from quantize import quant

    qconfig = {}
    qconfig['method'] = 'llmint8'
    qconfig['bits'] = 8
    quant(model, qconfig)

    print(model)

    model = model.to("cuda")

    enc.pad_token = enc.eos_token

    model_inputs = enc(["A list of colors: red, blue", "The capital of Spanish is"], return_tensors="pt", padding=True).to(model.device)
    generate_ids = model.generate(**model_inputs)
    results = enc.batch_decode(generate_ids, skip_special_tokens=True)

    print(results)