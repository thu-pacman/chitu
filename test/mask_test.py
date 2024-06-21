import torch

seqlen = 4
start_pos = 0

mask = torch.full((seqlen, seqlen), float("-inf"), device="cuda")

mask = torch.triu(mask, diagonal=1)

# When performing key-value caching, we compute the attention scores
# only for the new sequence. Thus, the matrix of scores is of size
# (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
# j > cache_len + i, since row i corresponds to token cache_len + i.
mask = torch.hstack([torch.zeros((seqlen, start_pos), device="cuda"), mask])
print(mask)
