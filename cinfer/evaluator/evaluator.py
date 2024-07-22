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


META_LLAMA_FLAG = True

from cinfer.model import Backend, VarLens
class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        if META_LLAMA_FLAG:
            #self.dataset = torch.tensor(tokenizer.encode(
            #    "\n\n".join(dataset["text"]), bos=True, eos=True
            #), device="cuda").reshape(1, -1)
            self.dataset = tokenizer.encode(
                "\n\n".join(dataset["text"]), bos=False, eos=False
            )
        else:
            self.dataset = tokenizer(
                "\n\n".join(dataset["text"]), return_tensors="pt"
            ).input_ids.to(device)
            

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        @torch.inference_mode()
        def test_output(self, tokens, device="cuda"):
            varlens = Backend.curr_varlens
            freqs_cis = self.prepare_freqs_cis_prefill(varlens, device)

            # start of model
            if self.rank == 0:
                tokens = torch.from_numpy(np.concatenate(tokens)).to(device)
                h = self.tok_embeddings(tokens)
            else:
                h = tokens
            # layers
            #print("evaluation GPU memory used : ", torch.cuda.memory_allocated())  
            for it, layer in enumerate(self.layers):
                h = layer(h, 0, freqs_cis, None, varlens, cache=False)
            # end of model
            if self.rank == self.world_size - 1:
                h = self.norm(h)
                h = self.output(h)
            return h

        model.test_output = test_output

        
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):    
            if META_LLAMA_FLAG:
                #batch = [ba[(i * 2048) : ((i + 1) * 2048)] for ba in self.dataset]
                batch = [self.dataset[(i * 2048) : ((i + 1) * 2048)]]
            else:
                batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(self.device)

            with torch.no_grad():             
                if META_LLAMA_FLAG:
                    Backend.curr_varlens = VarLens(batch, device=self.device)
                    #lm_logits = model.decode(tokens=torch.tensor(batch, device=self.device), seq_lens=[0]*2048)
                    lm_logits = model.test_output(batch).unsqueeze(dim=0)
                else:
                    lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()       
            if META_LLAMA_FLAG:
                shift_labels = torch.tensor([self.dataset[(i * 2048) : ((i + 1) * 2048)]], device=self.device)[:, 1:]
            else:
                shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

            del batch
            del lm_logits
            del shift_logits
            del shift_labels
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))
    

def ppleval(model, tokenizer, device="cuda", n_samples=40):
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, device, n_samples=n_samples)
    #evaluator = Evaluator(dataset, tokenizer, model.device, n_samples=40)
    ppl = evaluator.evaluate(model)
    print(f"Perplexity: {ppl}")
