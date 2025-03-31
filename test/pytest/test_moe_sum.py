from chitu.fused_moe import moe_sum
import torch
import random


def torch_moe_sum(input_tensor, output_tensor):
    output_tensor.copy_(input_tensor.sum(dim=1))


def test_moe_sum():

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define test dimensions
    M_values = [32, 64, 128]
    topK_values = [2, 4, 8]
    N_values = [256, 512, 1024]

    # Test configuration for moe_sum kernel
    config = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}

    for M in M_values:
        for topK in topK_values:
            for N in N_values:
                # Create random input tensor
                input_tensor = torch.rand(
                    M, topK, N, device="cuda", dtype=torch.float16
                )

                # Create output tensors for both implementations
                torch_output = torch.zeros(M, N, device="cuda", dtype=torch.float16)
                triton_output = torch.zeros(M, N, device="cuda", dtype=torch.float16)

                # Run torch implementation
                torch_moe_sum(input_tensor, torch_output)

                # Run triton implementation
                moe_sum(input_tensor, triton_output, config)

                # Check that results match
                assert torch.allclose(
                    torch_output, triton_output, rtol=1e-3, atol=1e-3
                ), f"Results don't match for shape M={M}, topK={topK}, N={N}"


if __name__ == "__main__":
    test_moe_sum()
