import torch
import pytest
import random

from chitu.ops import apply_frequency_penalty
from chitu.utils import try_import_platform_dep
from chitu.testing import assert_close
from chitu.sampling.sampler import TOKEN_BLOCK_SIZE

triton, has_triton = try_import_platform_dep("triton")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


@pytest.mark.parametrize("batch_size", [0, 1, 8, 128])
@pytest.mark.parametrize("vocab_size", [100, 151936, 129280])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_frequency_penalty(batch_size, vocab_size, impl, record_benchmark):
    if impl == "triton" and not has_triton:
        pytest.skip("Triton is not installed")
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping CUDA tests")

    logits = torch.randn((batch_size * 2, vocab_size), dtype=torch.float, device="cuda")
    indices = [random.randrange(0, batch_size * 2) for _ in range(batch_size)]
    blocks = [
        torch.randint(0, vocab_size, (TOKEN_BLOCK_SIZE,), device="cuda")
        for _ in range(batch_size)
    ]
    sizes = [
        (
            TOKEN_BLOCK_SIZE
            if random.random() > 0.5
            else random.randint(1, TOKEN_BLOCK_SIZE)
        )
        for _ in range(batch_size)
    ]
    penalties = [random.random() * 10 for _ in range(batch_size)]

    logits_ref = logits.clone()
    logits_test = logits.clone()
    apply_frequency_penalty(logits_ref, indices, blocks, sizes, penalties, impl="torch")

    record_benchmark.run(
        lambda: apply_frequency_penalty(
            logits_test,
            indices,
            blocks,
            sizes,
            penalties,
            impl=impl,
        ),
        batch_size=batch_size,
        impl=impl,
    )
    assert_close(logits_test, logits_ref, atol=1e-2, rtol=1e-2)
