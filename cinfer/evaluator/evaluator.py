import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
import os
import json
import numpy as np
import torch.nn as nn
import tqdm
import datasets

def lmeval(model, tasks=["wikitext"]):
    #os.environ['http_proxy'] = "socks5h://localhost:1086" 
    #os.environ['https_proxy'] = "socks5h://localhost:1086" 
    model = HFLM(pretrained=model)
    results = evaluator.simple_evaluate(
        model,
        tasks=tasks,
        batch_size=16,
        num_fewshot=0,
        log_samples=True,
        device="cuda"
    )

    
    def handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)


    dumped = json.dumps(
        results, indent=2, default=handle_non_serializable, ensure_ascii=False
    )
    print(dumped)

    return dumped


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))
    

def ppleval(model, tokenizer, n_samples=40):
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, model.device, n_samples=n_samples)
    #evaluator = Evaluator(dataset, tokenizer, model.device, n_samples=40)
    ppl = evaluator.evaluate(model)
    print(f"Perplexity: {ppl}")
