import random, xgrammar

from chitu.constraint_decode import apply_bitmask
import torch
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def test_regex_reject_tags():
    tags = ["<｜01｜>", "<｜002｜>"]
    vocab = sorted(list(set("".join(tags))))
    tokenizer_info = xgrammar.TokenizerInfo(vocab, stop_token_ids=len(vocab))
    format = xgrammar.structural_tag.AnyTextFormat(excludes=tags)
    grammar = xgrammar.Grammar.from_structural_tag(
        xgrammar.StructuralTag(format=format)
    )
    grammar_compiler = xgrammar.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_grammar(grammar)
    for _ in range(10000):
        seq = "".join(random.choices(vocab, k=random.randint(1, 12)))
        std = all(t not in seq for t in tags)
        ans = xgrammar.GrammarMatcher(compiled_grammar).accept_string(seq)
        assert std == ans


def test_apply_bit_mask():
    B = 4
    H = 32 * 3 + 8
    DEVICE = "cuda"
    logits = torch.arange(H, dtype=torch.float32)
    logits = logits[None, :] + torch.arange(B)[:, None]
    indices = [2, 0, 1]
    mask = torch.rand((B, H)) > 0.5
    bitmask = torch.zeros((B, (H + 31) // 32), dtype=torch.int32)
    std = logits.clone()
    for i in range(B):
        for j in range(H):
            if mask[i, j].item():
                bitmask[i, j // 32] |= 1 << (j % 32)
            elif i in indices:
                std[i, j] = float("-inf")

    out = logits.to(device=DEVICE)
    std = std.to(device=DEVICE)
    apply_bitmask(out, bitmask.to(device=DEVICE), indices)
    assert torch.allclose(out, std)
