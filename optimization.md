# 待实现的优化

## DCP KV cache 可配置交错粒度

状态：待实现。当前保留按物理页轮转的布局，不引入请求级 route seed。

### 目标与收益边界

参考 vLLM，将 KV 的交错粒度与物理页大小分开。当前每写满一页才切换 owner，短请求和不足一个完整逻辑组的尾部 KV 容易集中在靠前的 rank。减小交错粒度可以减轻这种偏斜。

例如物理页大小为 64、DCP size 为 4、请求长度为 100：

| 交错粒度（token 数） | rank 0 | rank 1 | rank 2 | rank 3 |
| --- | ---: | ---: | ---: | ---: |
| 64（当前布局） | 64 | 36 | 0 | 0 |
| 16 | 32 | 32 | 20 | 16 |
| 1 | 25 | 25 | 25 | 25 |

交错粒度不会轮转不同请求的起始 owner。长度小于交错粒度的请求仍全部落在 rank 0；即使粒度为 1，长度小于 DCP size 的请求也无法覆盖所有 rank。它改善的是 KV 数量的分布，不保证稀疏 TopK 选中的 KV 同样均衡。

### 配置与映射

候选配置名为 `dcp_kv_cache_interleave_size`。vLLM 当前使用 `cp_kv_cache_interleave_size`，并保留旧的 DCP 配置名；落地时再确定本项目的名称。

- `block_size` 仍表示每个 rank 的物理页 token 数，记为 `S`。
- DCP size 记为 `C`，交错粒度记为 `I`，要求 `1 <= I <= S` 且 `S % I == 0`。
- 默认值解析为 `block_size`，保持现有布局；显式设置较小值时才改变布局。
- 逻辑组仍覆盖 `S * C` 个全局 token，每个 rank 为该组保存一页。

对于请求内的全局 token 位置 `p`，映射为：

```python
owner = (p // I) % C
local_position = (p // (I * C)) * I + p % I
group_index = local_position // S
offset_in_local_page = local_position % S
physical_slot = block_table[request_id, group_index] * S + offset_in_local_page
```

将本地 compact token 位置 `l` 还原为全局位置：

```python
global_id = (l // I * C + dcp_rank) * I + l % I
```

长度为 `L` 的可见前缀在当前 rank 的本地长度为：

```python
local_length = L // (I * C) * I
local_length += min(max(L % (I * C) - dcp_rank * I, 0), I)
```

Indexer 的每个 MTP query 使用自己的可见前缀长度，并继续以 cache 实际有效 token 数为上限。

### 需要贯通的链路

1. 配置校验，以及 cache/accessor 中的静态布局信息。
2. Torch/Triton slot mapping：owner 判断、组块定位和本地页内偏移。
3. Cache 有效长度与 MTP advance：使用交错粒度计算本地 token 数。
4. Indexer：本地因果长度、compact slot 到全局 token ID 的还原。
5. Attention：全局 TopK ID 到 owner-local physical slot 的映射。
6. PD KV transfer、接收字节计数和前缀缓存：源/目标布局对齐及有效范围处理。

其中 PD 传输是主要配套工作。当前 Prefill 和 Decode 按整页配对，一个 Prefill 页只发给一个 owner。`I < S` 时，一个 Prefill 页中的 token 会分散到多个 Decode rank，同一个 Decode 物理页也可能由多个源页的片段组成。因此需要支持分段传输或布局转换，并处理源/目标偏移、传输长度、尾部有效范围，以及 BF16、packed FP8 和 indexer K 的记录布局。不能只修改 owner 公式而继续沿用整页传输计划。

交错粒度属于 KV 存储布局。跨实例传输必须能识别目标布局；已有 KV 或前缀缓存不能在切换交错粒度后直接复用。保持固定起点与确定性映射，不重新引入每请求 seed。

### 验证与性能评估

- 验证 `I=1`、中间粒度和 `I=S`；`I=S` 应与现有布局完全一致。
- 覆盖短请求、交错边界、逻辑组边界、尾部、空 owner，以及 request-major MTP。
- 验证写入、Indexer TopK、Attention 与完整 KV 参考结果一致。
- 覆盖 PD 传输、prefix hit 跳传、共享前缀和 CUDA Graph replay。
- 评估本地 KV/候选分布、RDMA 分段数、传输吞吐和端到端 Decode 延迟。更细的交错可能增加传输片段，需实测收益与开销。

参考：[vLLM 交错粒度配置](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)、[本地序列长度计算](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py)。
