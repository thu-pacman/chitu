
def get_attention_shapes(attention_shapes, max_seq_len, cache_batch_size, n_heads, n_kv_heads, head_dim):
    if attention_shapes is not None:
        attention_shapes = attention_shapes

    elif n_kv_heads == 0:
        attention_shapes = {
            # following fastertransformer definition
            "cache_v": (cache_batch_size, n_heads, max_seq_len, head_dim,),
            # 8: pack 8 fp16 in FT, if fp32 then use 4
            "cache_k": (cache_batch_size, n_heads, head_dim // 8, max_seq_len, 8,),
            "xqkv_view": (-1, n_heads, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0],
            "xk_slice": lambda xqkv: xqkv[:, :, 1],
            "xv_slice": lambda xqkv: xqkv[:, :, 2],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_heads, head_dim),
            "xv_view": (n_heads, head_dim),
            "xk_reshape": (n_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_heads, head_dim),
            "single_xv_view": (n_heads, head_dim)
        }

    else:
        attention_shapes = {
            # following fastertransformer definition
            "cache_v": (cache_batch_size, n_kv_heads, max_seq_len, head_dim,),
            # 8: pack 8 fp16 in FT, if fp32 then use 4
            "cache_k": (cache_batch_size, n_kv_heads, head_dim // 8, max_seq_len, 8,),
            "xqkv_view": (n_heads + n_kv_heads * 2, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0 : n_heads],
            "xk_slice": lambda xqkv: xqkv[:, :, n_heads : (n_heads + n_kv_heads)],
            "xv_slice": lambda xqkv: xqkv[:, :, -n_kv_heads :],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_kv_heads, head_dim),
            "xv_view": (n_kv_heads, head_dim),
            "xk_reshape": (n_kv_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_kv_heads, head_dim),
            "single_xv_view": (n_kv_heads, head_dim)
        }
    
    return attention_shapes