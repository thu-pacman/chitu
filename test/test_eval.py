import torch
import os

os.environ["PAGED_SIZE"] = "0"
from chitu.backend import Backend
from chitu.global_vars import *
from chitu.tokenizer import Tokenizer, ChatFormat, TokenizerHF, ChatFormatHF
from chitu.model import Attention, Transformer, apply_rotary_emb
from chitu.model_hf_llama import AttentionHFLlama, apply_rotary_pos_emb_torch
from chitu.utils import VarLens
import yaml
import flash_attn
import hydra

import numpy as np


def prefill_forward(
    self,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    varlens,
):
    bs_seq, _ = x.shape
    xq, xk, xv = self._run_linear(x)
    xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
    xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    output = flash_attn.flash_attn_varlen_func(
        xq,
        xk,
        xv,
        varlens.prefix_lens,
        varlens.prefix_lens,
        varlens.max_len,
        varlens.max_len,
        causal=True,
    ).view(bs_seq, -1)
    return self._run_output_linear(output)


Attention.prefill_forward = prefill_forward


def prefill_forward_qwen(
    self,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    varlens,
):
    bs_seq, _ = x.shape
    xq, xk, xv = self._run_linear(x)
    xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
    xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
    # torch.cuda.synchronize()
    cos, sin = self.rotary_emb(xv, seq_len=bs_seq)
    # torch.cuda.synchronize()
    xq, xk = apply_rotary_pos_emb_torch(
        xq,
        xk,
        cos,
        sin,
        position_ids=self.cache.curr_varlens.position_ids,
    )
    output = self.attn_backend.attn_varlen_func(
        xq,
        xk,
        xv,
        varlens.prefix_lens,
        varlens.prefix_lens,
        varlens.max_len,
        varlens.max_len,
        causal=True,
    ).view(bs_seq, -1)
    return self._run_output_linear(output)


AttentionHFLlama.prefill_forward = prefill_forward_qwen

"""
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
    # print("evaluation GPU memory used : ", torch.cuda.memory_allocated())
    for it, layer in enumerate(self.layers):
        h = layer(h, 0, freqs_cis, None, varlens, cache=False)
    # end of model
    if self.rank == self.world_size - 1:
        h = self.norm(h)
        h = self.output(h)
    return h
"""


@torch.inference_mode()
def prefill_single_device(self, tokens):
    varlens = VarLens(tokens, self.device)
    tokens = torch.from_numpy(np.concatenate(tokens)).to(self.device)
    freqs_cis = self.prepare_freqs_cis_prefill(varlens, self.device)
    h = self._pre_layers(tokens)
    for it, layer in enumerate(self.layers):
        h = layer(h, freqs_cis, varlens)
    # print(h.shape)
    # h = h[[item - 1 for item in tmp]]
    # print(h.shape)
    h = self._post_layers(h)  # Exec post layers AFTER cutting the last token off
    # print(h.shape)

    # print("1 GPU memory used : ", torch.cuda.memory_allocated(0))
    # print(h.shape)
    return h


Transformer.prefill_single_device = prefill_single_device


import tqdm


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # self.dataset = tokenizer(
        #    "\n\n".join(dataset["text"]), return_tensors="pt"
        # ).input_ids.to(device)
        self.dataset = tokenizer.encode(
            "\n\n".join(dataset["text"]), bos=False, eos=False
        )

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            # batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            batch = [self.dataset[(i * 2048) : ((i + 1) * 2048)]]
            with torch.no_grad():
                Backend.curr_varlens = VarLens(batch, device=self.device)
                Backend.cache_manager.curr_varlens = VarLens(batch, device=self.device)
                # lm_logits = model.decode(tokens=torch.tensor(batch, device=self.device), seq_lens=[0]*2048)
                # print("0 GPU memory used : ", torch.cuda.memory_allocated(0))
                lm_logits = model.prefill_single_device(batch).unsqueeze(dim=0)
                # lm_logits = model(batch).logits
            # print(lm_logits.shape)
            # print("2 GPU memory used : ", torch.cuda.memory_allocated(0))
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            # print("3 GPU memory used : ", torch.cuda.memory_allocated(0))
            shift_labels = torch.tensor(
                [self.dataset[(i * 2048) : ((i + 1) * 2048)]], device=self.device
            )[:, 1:]
            # print("4 GPU memory used : ", torch.cuda.memory_allocated(0))
            # print(shift_labels.shape)
            loss_fct = torch.nn.CrossEntropyLoss()
            # print("5 GPU memory used : ", torch.cuda.memory_allocated(0))
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


import datasets


def ppleval(model, tokenizer, n_samples=40):
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, model.device, n_samples=n_samples)
    # evaluator = Evaluator(dataset, tokenizer, model.device, n_samples=40)
    ppl = evaluator.evaluate(model)
    print(f"Perplexity: {ppl}")


@hydra.main(config_path="../example/configs/", config_name="serve_config")
def load_model(args):
    set_global_variables(args)
    Backend.build(args)
    print(Backend.tokenizer)
    print(Backend.model)


if __name__ == "__main__":
    load_model()
    model = Backend.model
    tokenizer = Backend.tokenizer
    print("here")
    print(model, tokenizer)

    ppleval(model, tokenizer)
