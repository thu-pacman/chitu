import triton
import time
import triton.language as tl
import torch


# Define constants
BATCH_SIZE = 64
NUM_HEAD = 1
HEAD_DIM = 128
TOTAL_SEQ = 1024  # Assuming a total sequence length for demonstration purposes

# Create dummy data
xk = torch.randn(BATCH_SIZE, NUM_HEAD, HEAD_DIM, device="cuda")
xv = torch.randn(BATCH_SIZE, NUM_HEAD, HEAD_DIM, device="cuda")
output = torch.zeros(2, BATCH_SIZE, TOTAL_SEQ, NUM_HEAD, HEAD_DIM, device="cuda")
curr_seq_lens = torch.randint(0, TOTAL_SEQ, (BATCH_SIZE,), device="cuda")
curr_seq_lens_cpu = curr_seq_lens.cpu()
output2 = torch.zeros(2, BATCH_SIZE, TOTAL_SEQ, NUM_HEAD, HEAD_DIM, device="cuda")

# Launch the kernel
grid = (BATCH_SIZE, NUM_HEAD)

move_data_kernel[grid](
    xk_ptr=xk,
    xv_ptr=xv,
    output_ptr=output,
    seq_lens_ptr=curr_seq_lens,
    BATCH_SIZE=BATCH_SIZE,
    NUM_HEAD=NUM_HEAD,
    HEAD_DIM=HEAD_DIM,
    TOTAL_SEQ=TOTAL_SEQ,
)

torch.cuda.synchronize()
t0 = time.time()
for i in range(100):
    move_data_kernel[grid](
        xk_ptr=xk,
        xv_ptr=xv,
        output_ptr=output,
        seq_lens_ptr=curr_seq_lens,
        BATCH_SIZE=BATCH_SIZE,
        NUM_HEAD=NUM_HEAD,
        HEAD_DIM=HEAD_DIM,
        TOTAL_SEQ=TOTAL_SEQ,
    )
torch.cuda.synchronize()
t1 = time.time()
print(t1 - t0)

# Verify the result
# print(output)


torch.cuda.synchronize()
t0 = time.time()
for i in range(100):
    for it in range(xk.shape[0]):
        output2[0][it][curr_seq_lens_cpu[it]] = xk[it]
        output2[1][it][curr_seq_lens_cpu[it]] = xv[it]
torch.cuda.synchronize()
t1 = time.time()

# Verify the result
# print(output)
print(torch.allclose(output, output2))

print(t1 - t0)
