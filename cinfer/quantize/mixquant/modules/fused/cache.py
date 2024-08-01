import torch

class WindowedCache:
    def __init__(self, cache_v_shape, cache_k_shape, device):
        """
        The window size is the same as the max_new_tokens. The window will
        automatically roll once max_new_tokens is exceeded.
        """
        # [batch_size, n_kv_heads, max_seq_len, head_dim]
        self.v = torch.zeros(cache_v_shape).to(device).half()
        # [batch_size, n_kv_heads, head_dim // pack_factor, max_seq_len, pack_factor]
        self.k = torch.zeros(cache_k_shape).to(device).half()
    
    def get_kv(self, batch_size, start_pos, seqlen, head_dim):
        xv = self.v[:batch_size, :, : start_pos + seqlen, :].transpose(1, 2).contiguous()
        xk = self.k[:batch_size, :, :, : start_pos + seqlen, :].transpose(2, 3).contiguous()
        xk = xk.reshape(xk.shape[:-2] + (head_dim,)).transpose(1, 2).contiguous()

        return xv, xk
    
    def update_kv(self, values_store, keys_store, batch_size, start_pos, seqlen):
        self.v[:batch_size, :, start_pos : start_pos + seqlen, :] = values_store
        self.k[:batch_size, :, :, start_pos : start_pos + seqlen, :] = keys_store
    
    def roll_kv(self, roll_len, start_pos):
        # Roll only the necessary part of the cache to the left
        self.v[:, :, :-roll_len, :] = self.v[:, :, roll_len:, :]
        self.k[:, :, :, :-roll_len, :] = self.k[:, :, :, roll_len:, :]

        # Zero out the new part
        self.v[:, :, -roll_len:, :] = 0
        self.k[:, :, :, -roll_len:, :] = 0

        return start_pos - roll_len
    
    def to(self, device):
        self.k = self.k.to(device)
        self.v = self.v.to(device)
    