# Chitu PD 分离（Prefill-Decode Disaggregation）

将 Prefill 和 Decode 部署在不同进程，Prefill 完成后通过 RDMA 将 KV Cache 和首 token 直传给 Decode，Decode 开始生成后续 token。

## 整体架构

系统由三类进程组成：

| 进程          | 职责                                                    | 入口                                                  |
| ----------- | ----------------------------------------------------- | --------------------------------------------------- |
| **Router**  | HTTP 入口，接收请求后同时分发给 P 和 D；托管协调服务和 Bootstrap 服务         | `chitu/serve/api_server.py`                         |
| **Prefill** | 执行 prefill 计算，产出 KV Cache 和首 token，通过 RDMA 发送给 Decode | `chitu/distributed/pd_disaggregation/pd_service.py` |
| **Decode**  | 接收 KV Cache 后执行 decode 生成，流式回传 token                  | `chitu/distributed/pd_disaggregation/pd_service.py` |


```
                          ┌──────────────────────────────┐
                          │         Router (节点 A)       │
                          │                              │
                          │  API Server (HTTP :22001)    │
                          │  PDCoordination (ZMQ)        │
                          │  Bootstrap      (HTTP :8080) │
                          └──────┬────────────┬──────────┘
                      ZMQ PUSH  │            │  ZMQ PUSH
                  ┌─────────────┘            └────────────────┐
                  ▼                                           ▼
     ┌────────────────────────┐              ┌────────────────────────────┐
     │   Prefill (节点 B)     │              │    Decode (节点 A)         │
     │                        │    RDMA      │                            │
     │  PDSchedulerService    │─────────────▶│  PDSchedulerService        │
     │  PrefillOnlyManager    │  KV Cache    │  DecodeOnlyManager         │
     │  KVManagerPrefill      │              │  KVManagerDecode           │
     └────────────────────────┘              └────────────────────────────┘
```

运行模式由所有实例的有效 `multi_inst.role` 推导：所有实例均为 `prefill_and_decode` 时使用独立多实例模式；所有实例均为 `prefill` 或 `decode` 时使用经典 PD 分离模式。

> Prefill / Decode 框内列出的是同一进程内的三个层次（由上到下为包含关系）：`PDSchedulerService`（进程入口）→ `PrefillOnlyManager` / `DecodeOnlyManager`（调度器）→ `KVManagerPrefill` / `KVManagerDecode`（KV 传输）。

## 通信分层

系统通信分为**控制面**和**数据面**两层：

### 控制面（ZMQ）

所有控制面消息使用 msgpack 单帧序列化的 dataclass 协议，通过 `type` 字段自动分发。

| 消息                           | 方向             | 作用                              |
| ---------------------------- | -------------- | ------------------------------- |
| DecodePrepare                | Scheduler → Decode | Scheduler 通知 Decode 预分配 KV block |
| DecodeAllocated              | Decode → Prefill | Decode 发送 recv_buffers 和 session_id |
| RankTransferDone             | Prefill rank → Prefill ctrl | 单 rank RDMA 传输完成           |
| PrefillDone                  | Prefill ctrl → Decode | 全部传输完成，携带 first_token 等      |
| Router → P/D 请求             | ZMQ PUSH/PULL  | 分发请求                            |
| P/D 统计心跳                   | ZMQ PUSH       | 向 Router 上报 stats               |

ZMQ 端点通过 `KVManagerEndpoint` 统一抽象，支持三种模式：
- **master**（ctrl rank）：PULL 接收外部消息 + PUB/SUB relay 广播到同组 slave
- **slave**（非 ctrl rank）：SUB 订阅 ctrl rank 广播
- **remote**（对端）：PUSH 发送到远程 endpoint

### 数据面（RDMA）

Prefill 通过 Mooncake Transfer Engine 执行 Device-to-Device 的 RDMA 写入：

- **内存注册**：整个 KV cache tensor 一次性注册，由 `KVManagerBase.register_cache()` 在初始化时完成
- **传输规划**：Prefill 初始化时通过 per-rank `CacheInfo` 交换预计算 `StaticTransferPlan`；per-request 利用 `base_ptrs + block_id × block_stride` 笛卡尔积生成地址，numpy sort/merge 后输出 `TransferPlan`
- **首 token**：通过 `PrefillDone` 消息直接携带

### KV Cache 的 split/replica 模型

每个 cache 由 `split_size` 字段描述 dim3 在 TP ranks 间的分布：

- `split_size = 0`（replica）：所有 rank 持有完整副本。每 rank 为独立 replica。
- `split_size > 0`（split）：dim3 被切分为 `split_size` 份，分布在 TP ranks 间。

## 请求生命周期

以 1P1D 为例（Prefill/Decode 各一个 ctrl rank + 若干 worker rank）：

```
 Router          Prefill Ctrl        Prefill Rank       Decode Ctrl         Decode Rank
   │                 │                   │                   │                   │
   │── prefill req ─▶│                   │                   │                   │
   │── decode req ──────────────────────────────────────────▶│                   │
   │                 │                   │                   │                   │
   │                 │                   │   DecodePrepare   │                   │
   │                 │                   │                   │◀── scheduler      │
   │                 │                   │                   │── PUB relay ────▶│
   │                 │                   │                   │                   │
   │                 │                   │                   │          分配 block
   │                 │                   │                   │    构造 TransferBuffers (block_ids)
   │                 │                   │                   │                   │
   │                 │    DecodeAllocated│                   │                   │
   │                 │◀──────────────────────────────────────────────────────────│ (dp_rank 匹配的)
   │                 │── PUB relay ─────▶│                   │                   │
   │                 │                   │                   │                   │
   │                 │            is_decode_allocated = True  │                   │
   │                 │            scheduler promote → TaskPool│                   │
   │                 │            prefill 计算                 │                   │
   │                 │            产出 KV + logits             │                   │
   │                 │                   │                   │                   │
   │                 │            send_kv_cache               │                   │
   │                 │            create_transfer_plan        │                   │
   │                 │                   │                   │                   │
   │                 │                   │── RDMA write ═══════════════════════▶│
   │                 │                   │                   │                   │
   │                 │  RankTransferDone │                   │                   │
   │                 │◀──────────────────│                   │                   │
   │                 │                   │                   │                   │
   │                 │  (收齐 dp_way_size 个)                  │                   │
   │                 │                   │                   │                   │
   │                 │                   │    PrefillDone    │                   │
   │                 │──────────────────────────────────────▶│── PUB relay ────▶│
   │                 │                   │                   │                   │
   │                 │                   │                   │   校验 rank_bytes
   │                 │                   │                   │   is_prefill_done = True
   │                 │                   │                   │                   │
   │                 │                   │           recv_kv_cache_and_insert   │
   │                 │                   │           insert_kv_cache_from_transfer
   │                 │                   │           kv_recv_reorder           │
   │                 │                   │                   │                   │
   │                 │                   │        scheduler promote → TaskPool  │
   │                 │                   │        Decode 循环                   │
   │                 │                   │                   │                   │
   │◀── tokens ────────────────────────────────────────────────────────────────│
   │                 │                   │                   │                   │
```

### 角色说明

| 角色 | 职责 |
|------|------|
| **Prefill Ctrl Rank**（rank=0） | 接收 `DecodeAllocated`，通过 PUB/SUB relay 广播；收集各 rank 的 `RankTransferDone`，汇总后发送 `PrefillDone` |
| **Prefill Rank** | 接收 relay 的 `DecodeAllocated`，执行 prefill 计算，调用 `send_kv_cache` → `create_transfer_plan` → RDMA 写入 → 发送 `RankTransferDone` |
| **Decode Ctrl Rank**（rank=0） | 接收 scheduler 的 `DecodePrepare`，通过 PUB/SUB relay 广播；接收 `PrefillDone`，通过 relay 广播给所有 rank |
| **Decode Rank** | 接收 relay 的 `DecodePrepare`，dp_rank 匹配时分配 block 并发送 `DecodeAllocated`；接收 `PrefillDone` 后调用 `recv_kv_cache_and_insert` |

### 详细步骤

**1. Router 分发**

`PDRequestRouter._add_pd_request()` 为请求生成 `request_id`，通过轮询选择一个 Prefill 和一个 Decode Scheduler，将同一请求同时发给 P&D。Decode 侧的消息额外携带 `prefill_scheduler_id`，用于后续 PD 配对。

**2. Decode 准备**

Decode Scheduler 收到请求后入队到 `_decode_incoming_q`。后台推进线程将请求推入 `_decode_prealloc_q`，同时分配 KV cache block 并通过 `KVManagerDecode.send_decode_prepare()` 向 Decode ctrl rank 发送 `DecodePrepare`。

Decode ctrl rank 通过 PUB/SUB relay 广播 `DecodePrepare` 到所有 Decode rank。dp_rank 匹配的 rank 调用 `prepare_kv_transfer()`：
- 为每个 cache 分配 block（`PagedKVCache` 或 `SingletonPagedKVCache`）
- 调用 `cache.构造 TransferBuffers (block_ids)()` 收集 recv_buffers（物理地址 + key）
- 记录 `recv_bytes` 用于后续字节校验
- 发送 `DecodeAllocated` 到 Prefill ctrl rank

**3. Prefill 计算与传输**

Prefill ctrl rank 收到 `DecodeAllocated` 后通过 PUB/SUB relay 广播到所有 Prefill rank。Scheduler 侧的队列推进（`_prefill_incoming_q → _prefill_bootstrap_q → _prefill_ready_q`）等待 `is_decode_allocated` 就绪后创建 Task 执行 prefill。

Prefill 完成后触发 `MooncakeKVTransferHook.on_prefill_done()`，每个 Prefill rank 调用 `KVManagerPrefill.send_kv_cache()`：
- 遍历 `Backend.cache_dict` 中所有 cache，调用 `cache.构造 TransferBuffers (block_ids)()` 收集 send_buffers
- `create_transfer_plan(static_plan, send_buffers, recv_buffers)` 生成 `TransferPlan`
- 提交到 `ThreadPoolExecutor` 异步执行 RDMA

`transfer_worker` 线程：
- `TransferPlan.execute_send()` 批量 RDMA 写入
- 发送 `RankTransferDone` 到 Prefill ctrl rank

Prefill ctrl rank 收齐 `dp_way_size` 个 `RankTransferDone` 后，发送 `PrefillDone`（含 `first_token`、`num_hit_tokens`、`rank_bytes`）到 Decode ctrl rank。

**4. Decode 执行**

Decode ctrl rank 收到 `PrefillDone` 后通过 PUB/SUB relay 广播到所有 Decode rank。每个 Decode rank：
- 校验 `rank_bytes[session_id] == recv_bytes`（不一致记录 error）
- 设置 `is_prefill_done = True`

Decode hook `before_decode_step` 调用 `recv_kv_cache_and_insert()`：
- 断言 `is_prefill_done`
- 调用 `cache.insert_kv_cache_from_transfer()` 登记 KV 页表
- 调用 `cache.kv_recv_reorder()`（仅 `split_size > 0` 且 `n_chunks > 1` 时）
- 请求从 `_decode_prealloc_q` 推入 `_decode_ready_q`，创建 Task 进入 decode 循环
- Token 通过 `DPTokenManager` 经 ZMQ 回传给 Router，Router 流式返回给 Client

## 核心组件

### KVManager（`kv_transfer/` 目录）

KV 传输模块，由以下文件组成：

```
kv_transfer/
├── __init__.py              # 公开 KVManagerPrefill, KVManagerDecode
├── base.py                  # KVManagerBase: TransferEngine + TaskInfo + trace + buffer 注册
├── prefill.py               # KVManagerPrefill: send 侧完整逻辑
├── decode.py                # KVManagerDecode: recv 侧完整逻辑
├── endpoint.py              # KVManagerEndpoint: ZMQ master/slave/remote 三模式
├── protocol.py              # 4 种协议消息 dataclass + ProtocolSerializer (msgpack)
├── transfer_buffers.py      # TransferBuffers (per-cache block ID lists)
├── transfer_plan.py         # TransferPlan + create_transfer_plan
├── static_transfer_plan.py  # StaticTransferPlan + build + sort/merge
├── cache_info.py            # CacheInfo / InstanceCacheInfos: per-cache distribution 与 chunk 计算
├── task_info.py             # TaskInfo: per-request 状态聚合
└── mooncake/
    ├── transfer_engine.py   # MooncakeTransferEngine: RDMA 封装 + MooncakeBootstrapServer
    ├── metadata.py          # MetadataBuffers
    └── utils.py             # FastQueue 等工具函数
```

**DisaggregationMode**（`base.py`）：

```python
class DisaggregationMode(Enum):
    NULL = "null"       # 未启用
    PREFILL = "prefill" # Prefill 模式
    DECODE = "decode"   # Decode 模式
```

请求状态通过 `TaskInfo` 中的 `is_decode_allocated` / `is_prefill_done` 等标志位管理。

**Prefill 模式**

```
                   ┌─── control rank (rank=0) ──────────┐
                   │                                     │
  Decode ─ZMQ──▶   │  外部端口: decode_allocated (接收     │  ──ZMQ PUB──▶ 所有 ranks
                   │              DecodeAllocated)       │
                   │                                     │
  Prefill ranks    │  内部端口: rank_transfer_done        │  ──ZMQ PUSH─▶ Decode (PrefillDone)
  ──ZMQ PUSH──▶    │  (汇总各 rank 的传输完成)            │
                   └─────────────────────────────────────┘
```

- **control rank**（rank=0）：接收 Decode 的 `DecodeAllocated`，通过 PUB/SUB relay 广播；收集各 rank 的 `RankTransferDone`，汇总后发送 `PrefillDone`
- **传输线程**（ThreadPoolExecutor）：每个 rank 执行 `transfer_worker`，调用 `TransferPlan.execute_send()` 完成 RDMA 传输，然后发送 `RankTransferDone`

**Decode 模式**

- ctrl rank（rank=0）接收 scheduler 的 `send_decode_prepare()` → 通过 PUB/SUB relay 广播给所有 ranks
- dp_rank 匹配的 rank 执行 `prepare_kv_transfer()` → 分配 block → 发送 `DecodeAllocated` 到 Prefill
- 收到 `PrefillDone` 后校验字节 → `is_prefill_done = True` → hook 触发 `recv_kv_cache_and_insert()`

### 协议消息（`protocol.py`）

所有控制面消息使用 msgpack 单帧序列化，通过 `type` 字段分派到对应 dataclass。

| 消息 | 方向 | 关键字段 |
|------|------|----------|
| `DecodePrepare` | Scheduler → Decode rank | `req_id`, `prefill_sid`, `prefix_len`, `new_cache_ids`, `dp_rank` |
| `DecodeAllocated` | Decode → Prefill | `req_id`, `session_id`, `buffers: TransferBuffers`, `dp_rank`, `rank_num` |
| `RankTransferDone` | Prefill rank → ctrl | `req_id`, `first_token`, `rank_bytes: dict[str,int]` |
| `PrefillDone` | Prefill ctrl → Decode | `req_id`, `first_token`, `num_hit_tokens`, `rank_bytes: dict[str,int]` |

### 传输匹配与规划（`transfer_plan.py` + `static_transfer_plan.py`）

**初始化阶段**：每个 Prefill rank 在 `register_cache()` 后调用 `_build_static_transfer_plans()`。

1. 收集本 rank 各 cache 的 `CacheInfo`（GPU 指针、strides、layer_ids、split/replica）
2. `all_gather_object` 汇聚所有本地 rank 信息，coordinator 交换远端信息 → `InstanceCacheInfos`
3. 对每个 decode instance 调用 `build_static_transfer_plan(local_infos: RankCacheInfos, remote_infos: InstanceCacheInfos)`：
   - 遍历 `(decode_session, cache_name)` pair，chunk 映射 + replica 匹配 + layer 交集
   - chunk offset 预计算入 `src_base_ptrs / dst_base_ptrs`
4. 产出 `StaticTransferPlan`（per decode inst_id），存为 `self.static_transfer_plan: dict[int, StaticTransferPlan]`

**Per-request**：`create_transfer_plan(static_plan, send_buffers, recv_by_session)` —

1. per cache：`generate_and_merge()` — `n_layers * n_blocks` 笛卡尔积 `addr = base_ptrs[layer] + phys_block_id * block_stride`，ravel 后 numpy sort/merge
2. 跨 cache `sort_and_merge()` 统一按 dst 地址排序合并连续区间
3. 产出 `TransferPlan`（`ptrs/lengths/remote_ptrs` 全为 numpy int64 数组，`execute_send()` 时 `.tolist()` 转换为 `list[int]`）

**优势**：无 per-request 临时对象；地址计算向量化（numpy）；`DecodeAllocated` 消息体积 ~10MB → ~1KB。

**字节校验**：`PrefillDone.rank_bytes` 汇总所有 Prefill rank 的 per-session 传输字节数，Decode 侧对比 `recv_bytes`。
### CacheInfo / RankCacheInfos / InstanceCacheInfos（`cache_info.py`）

**`CacheInfo`**（frozen dataclass）描述单一 rank 的单一 cache tensor 的全部静态信息：

```python
@dataclass(frozen=True)
class CacheInfo:
    split_len: int       # 本地 head 数
    split_id: int        # 全局 head 起始偏移
    split_size: int      # 0 = replica; >0 = split
    replica_id: int      # replica 组内序号
    replica_size: int    # replica 组大小
    base_ptr: int        # tensor GPU data_ptr（新增）
    layer_ids: list[int] # 持有的 global layer 列表（新增）
    layer_stride: int    # stride(0) * elem_size（新增）
    block_stride: int    # stride(1) * elem_size（新增）

    def calc_chunking(self, remote: CacheInfo) -> tuple[int, int]:
        align = math.gcd(self.split_len, remote.split_len)
        return self.split_len // align, align
```

**`RankCacheInfos`** — 单个 rank 的 cache 集合：`{session_id, caches: dict[cache_name, CacheInfo]}`。

**`InstanceCacheInfos`** — 一个实例所有 rank 的集合：`{ranks: dict[session_id, dict[cache_name, CacheInfo]]}`。msgpack 序列化通过 coordinator 在 P/D 间交换（key: `inst{id}:all_rank_cache_dists`，type: `"InstanceCacheInfos"`）。

**per-rank 交换**：每个 rank 收集自身 `RankCacheInfos` → `all_gather_object` 汇聚 → coordinator 交换 → Prefill 侧用 `build_static_transfer_plan()` 构建 per-rank `StaticTransferPlan`。`kv_recv_reorder()` 通过 `local_dists.get(key)` 和 `remote_dists.get_any(key)` 动态计算 `n_chunks`。

通过 `gcd(local.split_len, remote.split_len)` 动态对齐——无论 P/D 两端的 TP 大小如何，chunking 总能自动匹配。
### PDCoordinationService（`pd_coordination.py`）

运行在 Router 进程中，负责调度器注册与 P/D 配对的状态管理（`register_scheduler` / `register_pd_pair` / `get_pd_stats`）。

**服务发现**

控制面 endpoint 的注册与查询统一通过 coordinator TCPStore（`chitu/distributed/coordinator.py` 的 `set_endpoint` / `get_endpoint`）完成。

使用的角色：

| 角色                                | 连接名                                             | 作用                          |
| --------------------------------- | ------------------------------------------------ | --------------------------- |
| `prefill{sid}`                    | `decode_allocated` / `rank_transfer_done`        | Prefill 控制面 endpoint       |
| `decode{sid}`                     | `decode_prepare` / `prefill_done`                | Decode 控制面 endpoint        |

### MooncakeBootstrapServer（`mooncake/transfer_engine.py`）

运行在 Router 进程中的轻量 HTTP 服务。Router 启动时只要 `kv_transfer_backend == "mooncake"` 即启动。

MooncakeBootstrapServer vs PDCoordinationService：

- **MooncakeBootstrapServer**：Prefill endpoint 集合。Decode 通过 Bootstrap 来发现 Prefill 的 Mooncake session 地址。这是 Decode 找到 Prefill 的手段。
- **PDCoordinationService**：内部控制面端点同步。用于 Prefill/Decode 内部 ranks 之间发现 endpoint。

唯一存在 fallback 的地方是 Prefill **注册自身** endpoint 这一步：优先通过 Coordination Service 注册，仅当 Coordination 不可用时才 fallback 到 Bootstrap 的 `PUT /route`。


| 方法  | 路径                          | 作用                                                                                |
| --- | --------------------------- | --------------------------------------------------------------------------------- |
| PUT | `/route`                    | Prefill 注册 `{role, rank_ip, rank_port, engine_rank}`（Coordination 不可用时的 fallback） |
| GET | `/route?engine_rank=<rank>` | Decode 查询 Prefill endpoint（主要使用路径）                                                |
| GET | `/route?engine_rank=-1`     | 返回 `dp_size`                                                                      |
| GET | `/health`                   | 健康检查                                                                              |


### PDScheduler（`pd_scheduler.py`）

根据模式分为 `PrefillOnlyManager` 和 `DecodeOnlyManager`。

**Prefill 队列模型**

```
请求到达 → _prefill_incoming_q → _prefill_bootstrap_q → _prefill_ready_q → TaskPool → 执行
                                   (等待 DecodeAllocated)    (可调度)
```

**Decode 队列模型**

```
请求到达 → _decode_incoming_q → _decode_prealloc_q → _decode_ready_q → TaskPool → 执行
             (分配 block, 发送       (等待 PrefillDone)        (可调度)
              DecodePrepare)
```

Decode 侧支持 token budget（`decode_prealloc_token_budget`）和 per-DP 并发限制（`decode_max_running_tasks_per_dp`），控制 prealloc 阶段的内存压力。`DecodePrepare` 每 1 秒重发直到 PrefillDone 就绪。

### MooncakeKVTransferHook（`hooks.py`）

注入到 Executor 中的 Hook：

- `on_prefill_done(tasks)`：Prefill 模式下调用 `kv_manager.send_kv_cache()`
- `before_decode_step(req_ids)`：Decode 模式下逐请求调用 `kv_manager.recv_kv_cache_and_insert()` 并更新 task 的首 token

PP>1 时，非最后一个 PP stage 只发送 KV（不含首 token）。

## 配置

不论什么拓扑（1P1D、2P3D、XP YD），统一使用 `serve_config.yaml` 作为基础配置。启动脚本通过 Hydra 命令行 override 动态覆盖 `multi_inst.inst_overrides` 和 `dp_size` 等字段，无需为每种拓扑维护单独的配置文件。

```yaml
multi_inst:
  n_insts: 2                       # P + D 总实例数（启动时覆盖）
  inst_id: 0                       # 当前实例 ID；Router 启动时设为 null
  role: "prefill_and_decode"
  inst_overrides: {}               # 启动时覆盖为每个实例的有效配置

  pd_disaggregation:
    prefill_scheduler: null
    decode_scheduler: null
    kv_transfer_backend: "mooncake"

    kv_transfer:
      buffer_size: 2048
      transfer_timeout: 30.0
      max_concurrent_transfers: 8
      decode_wait_timeout_s: 300.0     # Decode 等待 KV 传输完成的超时
      decode_resend_interval_s: 0.5    # Decode 重发 DecodePrepare 间隔
      decode_poll_interval_s: 0.05     # Decode 轮询 KV 状态间隔
      queue_max_pending: 32            # 队列最大 pending 数
      queue_log_interval_s: 1.0        # 队列背压日志间隔
      decode_prealloc_max_pending: null # prealloc 并发上限
      decode_prealloc_token_budget: 0  # prealloc token 预算
      decode_prealloc_reserved_tokens: 0
      decode_max_running_tasks_per_dp: null
      prefill_bootstrap_poll_interval_s: 0.01
      prefill_wait_timeout_s: 2400.0

  router:
    is_router: True                # Router 进程设为 True，P/D 设为 False
    host: 0.0.0.0
    port: 22001                    # HTTP 推理入口端口
    routing_algorithm: "prefix_cache_aware"

  # 启动脚本会生成以下形式的实例覆盖配置
  inst_overrides:
    0:
      multi_inst:
        role: "prefill"
        pd_disaggregation:
          prefill_scheduler:
            max_batch_size: 32
            max_total_tokens: 8192
            batching_strategy: "varlen"
    1:
      multi_inst:
        role: "decode"
        pd_disaggregation:
          decode_scheduler:
            scheduling_strategy: "immediate"
```

每个 P/D 调度器启动后绑定随机端口，并通过 coordinator 以角色 `prefill_instance_<id>` / `decode_instance_<id>` 注册 endpoint；Router 从 coordinator 发现各调度器地址，无需在配置中填写 host/port。启动脚本 `srun_pd_disagg_base_apptainer.sh` 会根据 `--prefill` / `--decode` 参数自动生成 `multi_inst.inst_overrides` 的 Hydra override 传给 Router 和每个 P/D 实例，同时为 Router 设置 `multi_inst.inst_id=null`，为每个 P/D 实例设置对应的 `multi_inst.inst_id`。`--pd-spec` 中的参数（如 `decode_wait_timeout_s`）会覆盖上面 `kv_transfer` 下的默认值。

### 端口矩阵（2P3D 示例）


| 组件             | 端口    | 协议   | 节点  | 用途                          |
| -------------- | ----- | ---- | --- | --------------------------- |
| Router API     | 22001 | HTTP | A   | 推理入口 `/v1/chat/completions` |
| Router Stats   | 随机    | ZMQ  | A   | P/D 统计心跳（coordinator 发现）    |
| Router Token   | 随机    | ZMQ  | A   | Decode → Router token 回传（coordinator 发现）    |
| Coordinator    | 21001+offset | TCP  | A   | 端口发现 TCPStore  |
| Bootstrap      | 8080  | HTTP | A   | Mooncake endpoint目录         |
| Prefill P0     | 随机    | ZMQ  | B   | Router → P0 请求（coordinator 发现）  |
| Prefill P1     | 随机    | ZMQ  | B   | Router → P1 请求（coordinator 发现）  |
| Decode D0      | 随机    | ZMQ  | A   | Router → D0 请求（coordinator 发现）  |
| Decode D1      | 随机    | ZMQ  | A   | Router → D1 请求（coordinator 发现）  |
| Decode D2      | 随机    | ZMQ  | A   | Router → D2 请求（coordinator 发现）  |
| P ↔ D RDMA     | —     | RDMA | —   | KV cache 显存直传                 |


## 运行方式

### 环境变量


| 变量                     | 必须       | 说明                           |
| ---------------------- | -------- | ---------------------------- |
| `CUDA_VISIBLE_DEVICES` | 按需       | 控制 GPU 分配                    |


### 统一启动脚本

使用 `script/srun_pd_disagg_base_apptainer.sh`，通过 SLURM + Apptainer 容器启动。脚本自动完成节点分配、GPU 划分、端口编排、Router/Prefill/Decode 全部进程的拉起。

```
bash script/srun_pd_disagg_base_apptainer.sh <MODEL_CONFIG> <MODEL_CKPT_DIR> <SIF_FILE> [options...]
```

**参数分五类：**


| 类别     | 参数                                                                                | 说明                                                        |
| ------ | --------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 集群     | `--nodes N` `--gpus-per-node N` `--partition P` `--exclude NODES` `--log-dir DIR` | SLURM 资源                                                  |
| 模型通用   | `--model-spec "key=val,..."`                                                      | `attn_type`, `mla_absorb`, `use_cuda_graph` 等             |
| PD 传输  | `--pd-spec "key=val,..."`                                                         | `decode_wait_timeout_s`, `decode_prealloc_token_budget` 等 |
| 实例参数   | `--prefill "tp=N,pp=N,..."` `--decode "tp=N,dp=N,..."`                            | 可重复，每次出现定义一个实例                                            |
| Router | `--router-port PORT` `--config-name NAME` `--bind-code 0|1`                       | Router 配置                                                 |


**实例参数支持的 key：** `tp`, `pcp`, `pp`, `dp`, `ep`, `max_seq_len`, `max_batch_size`, `max_new_tokens`, `chunk`(仅 prefill), `full_warmup`, `nnodes`, `nproc`。含 `.` 的 key 自动作为 Hydra override（如 `infer.memory_utilization=0.90`）。

#### 示例 1：DeepSeek-R1（4 节点，1P1D，PP=2）

1 个 Prefill（TP=8, PP=2，占 2 节点 16 卡）+ 1 个 Decode（TP=1, DP=16, EP=16，占 2 节点 16 卡）。

```bash
bash script/srun_pd_disagg_base_apptainer.sh \
  DeepSeek-R1 /data/nfs/DeepSeek-R1 /path/to/chitu-mooncake.sif \
  --nodes 4 --router-port 22006 \
  --model-spec "attn_type=flash_mla,mla_absorb=absorb-without-precomp" \
  --pd-spec "decode_wait_timeout_s=1200,decode_prealloc_max_pending=256,decode_prealloc_token_budget=350000,decode_prealloc_reserved_tokens=1024,decode_max_running_tasks_per_dp=30" \
  --prefill "tp=8,pp=2,dp=1,ep=1,max_seq_len=6144,max_batch_size=256,full_warmup=True,infer.memory_utilization=0.90" \
  --decode "tp=1,pp=1,dp=16,ep=16,max_seq_len=6144,max_batch_size=512,full_warmup=True,infer.use_cuda_graph=True" \
  --bind-code 1
```

#### 示例 2：Qwen3-235B-A22B-FP8（4 节点，2P1D）

2 个 Prefill（各 TP=4, PP=2，各占 1 节点 8 卡）+ 1 个 Decode（TP=1, DP=16, EP=16，占 2 节点 16 卡）。

```bash
bash script/srun_pd_disagg_base_apptainer.sh \
  Qwen3-235B-A22B-fp8 /data/nfs/Qwen3-235B-A22B-FP8 /path/to/chitu-mooncake.sif \
  --nodes 4 --router-port 22006 \
  --model-spec "attn_type=flash_mla" \
  --pd-spec "decode_wait_timeout_s=1200,decode_prealloc_max_pending=256,decode_prealloc_token_budget=350000,decode_prealloc_reserved_tokens=1024,decode_max_running_tasks_per_dp=30" \
  --prefill "tp=4,pp=2,dp=1,ep=1,max_seq_len=6144,max_batch_size=256,full_warmup=True,infer.memory_utilization=0.90" \
  --prefill "tp=4,pp=2,dp=1,ep=1,max_seq_len=6144,max_batch_size=256,full_warmup=True,infer.memory_utilization=0.90" \
  --decode "tp=1,pp=1,dp=16,ep=16,max_seq_len=6144,max_batch_size=512,full_warmup=True,infer.use_cuda_graph=True" \
  --bind-code 1
```

#### 脚本工作原理

1. 解析所有 `--prefill` / `--decode` 规格，推导每个实例的 `world_size = tp * pp * dp`，自动计算 `nnodes` 和 `nproc_per_node`
2. 贪心分配节点：Prefill 优先紧凑放置，Decode 优先使用空闲节点
3. 通过 `srun --ntasks=<节点数>` 在每个节点上运行同一脚本的 `--node` 模式
4. 每个节点根据 `SLURM_PROCID` 判断自己负责哪些实例，分配 GPU，依次启动
5. Node 0 额外启动 Router 进程，等待端口就绪后其他进程才开始连接

#### 本地单机启动

开发调试用，不依赖 SLURM：

```bash
bash script/start_pd_disagg_1p2d_local.sh <MODEL_CONFIG> <MODEL_CKPT_DIR>
```

### 请求测试

```bash
curl -X POST http://<Router_IP>:22001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":true}'
```

## 内置测试

Router 内置了 `pd.test` 冒烟测试模式。当所有 P/D 实例连接就绪后，Router 自动注入一批 mock 请求，监控其完整经过 prefill→decode→finish 流水线，输出每请求结果后退出。

### 配置参数

配置位于 `pd_test` 节点（`chitu/config/test.yaml`）：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enable` | bool | `False` | 是否启用测试模式 |
| `req_num` | int | `8` | 注入的测试请求数量 |
| `req_timeout` | float | `300` | 单请求超时（秒），超时记为失败 |
| `output_len` | int | `128` | 每个请求的 `max_new_tokens` |

### 命令行用法

通过 Hydra override 开启：

```bash
bash script/srun_pd_disagg_base_apptainer.sh <MODEL_CONFIG> <MODEL_CKPT_DIR> <SIF_FILE> \
  --prefill "tp=2,pp=1,dp=1" \
  --decode "tp=1,pp=1,dp=2,ep=2" \
  --pd-spec "pd_test.enable=1,pd_test.req_num=8,pd_test.req_timeout=300,pd_test.output_len=256"
```

### 工作原理

1. Router 启动后等待所有 P/D 实例连接（`_wait_for_pd_instances`）
2. 实例全部就绪后，`PDTestRunner.run()` 启动：
   - **创建请求**：以 `pd_test_000000` ~ `pd_test_000007` 为 `request_id`，从内置问题池中轮询选择 prompt，通过 `PDRequestRouter.add_request()` 分发
   - **监控完成**：每 0.5s 轮询 `pending_pd_requests`，通过 Token Router 的 `active_requests` 判断请求是否完成；超时记为失败
   - **输出结果**：所有请求完成后，逐条打印 `[PD_TEST][result]` 日志（`request_id` / `status` / `input_len` / `output_tokens` / 生成文本）
3. 全部请求成功 → `os._exit(0)`；有任何失败 → `os._exit(1)`

### 日志示例

```
[PD_TEST] test mode enabled: num_requests=8 timeout=300.0s output_len=128
[PD_TEST] creating 8 test requests
[PD_TEST] all 8 test requests dispatched, start_time=1719600000.123
[PD_TEST] monitoring 8 requests, timeout=300.0s
[PD_TEST] all requests finished: completed=8 failed=0 total_elapsed=45.678s
[PD_TEST] all 8 requests completed successfully
[PD_TEST][result] rid=pd_test_000000 status=COMPLETED input_len=22 max_new_tokens=128 output_tokens=128 output=宫保鸡丁是一道著名的川菜...
[PD_TEST][result] rid=pd_test_000001 status=COMPLETED input_len=49 max_new_tokens=128 output_tokens=128 output=Kung Pao chicken is a spicy...
...
[PD_TEST] exiting process
```

### CI 集成

CI 中使用 Hydra 命令行 override 直接覆盖 `pd_test.*` 参数，无需修改配置文件：

```yaml
# 示例：ci/platforms/h20/pd_test.yml
CHITU_CMDLINE: |
  models=GLM-4.7-Flash \
  models.ckpt_dir=/data/nfs2/GLM-4.7-Flash \
  pd_test.enable=1 \
  pd_test.req_num=8 \
  pd_test.req_timeout=600 \
  pd_test.output_len=1024 \
  multi_inst.n_insts=2 \
  +multi_inst.inst_overrides.0.multi_inst.role=prefill \
  +multi_inst.inst_overrides.0.infer.tp_size=2 \
  +multi_inst.inst_overrides.1.multi_inst.role=decode \
  ...
```


## 监控与观测

### Prometheus 指标（`chitu/metrics/prometheus_collector.py`）

每个 P/D 进程启动独立的 Prometheus HTTP exporter。指标名称、标签、类型、派生表达式和展示位置以生成文档为准：[`docs/zh/METRICS.md`](../../../docs/zh/METRICS.md)。


### 代码中的打点方式

```python
from chitu.metrics.prometheus_collector import observe_pd_stage, observe_stage_duration

# 方式 1：context manager（自动计时）
with observe_pd_stage("prefill", "kv_send"):
    kv_manager.send_kv_cache(...)

# 方式 2：手动记录预计算的耗时
observe_stage_duration("decode", "kv_recv", duration_s)
```

### 日志锚点

关键日志标记用于排查时序问题：


| 日志前缀                                     | 出现位置              | 含义              |
| ---------------------------------------- | ----------------- | --------------- |
| `[PD_STAGE][router.recv.start]`          | Router            | 请求进入 PD 路由      |
| `[PD_STAGE][prefill.kv_send.start]`      | Prefill Hook      | KV 发送开始         |
| `processing prefill request`             | Prefill Scheduler | 收到 prefill 请求   |
| `processing decode request`              | Decode Scheduler  | 收到 decode 请求    |
| `[PD_BOOTSTRAP][prefill.wait]`           | Prefill Scheduler | 等待 DecodeAllocated |
| `[PD_BOOTSTRAP][decode.wait]`            | Decode Scheduler  | 等待 PrefillDone   |
| `[PD_QUEUE][decode.ready]`               | Decode Scheduler  | KV ready，准备 decode |
| `created pd request: <rid> -> P{k}-D{m}` | Router            | 请求分配到具体 P/D     |


启用详细日志：设置环境变量 `CHITU_LOGGING_LEVEL=chitu.distributed.pd_disaggregation:DEBUG;chitu.hooks:DEBUG;chitu.scheduler:DEBUG`。

### Prometheus Server 集成

Router 进程可选启动内置的 Prometheus Server（`PrometheusServerManager`），自动抓取所有 P/D 的 exporter。也可使用外部 Prometheus，将各 exporter 地址加入 scrape 配置。

## TP 并行

- 仅 rank 0 对外暴露 ZMQ 端口，处理控制面消息
- 非 rank 0 通过 PUB/SUB 接收 ctrl rank 广播的 `DecodeAllocated` / `DecodePrepare` / `PrefillDone` 等消息
- 每个 TP rank 独立执行自己负责的 KV 层的 RDMA 传输
- 所有 rank 完成后由 Prefill ctrl rank 汇总 `dp_way_size` 个 `RankTransferDone`，发送 `PrefillDone`
- head-repeat 场景（n_head < tp_size）下，同 head 的连续 rank 产生相同 match_prefix，`StaticTransferPlan builder` 自动过滤重复匹配

## CP（Context Parallelism）支持

- `pcp_size > 1` 时 TP 强制为 1，CP group 替代 TP group 参与 split 计算
- `dp_way_size = world_size // dp_group_size` 自动覆盖 CP 场景
- `cp_rank` 记录在 trace 日志中

## 代码导航


| 模块              | 路径                                        | 说明                                       |
| --------------- | ----------------------------------------- | ---------------------------------------- |
| Router          | `pd_request_router.py`                    | 请求路由与 P/D 分发                             |
| Coordination    | `pd_coordination.py`                      | endpoint发现与元数据同步                         |
| Scheduler       | `pd_scheduler.py`                         | Prefill/Decode 调度器与分层队列管理                  |
| Service         | `pd_service.py`                           | 进程入口与初始化                                 |
| KV Manager Base | `kv_transfer/base.py`                     | 共享基类：TransferEngine + TaskInfo + trace     |
| KV Manager Prefill | `kv_transfer/prefill.py`               | Send 侧：DecodeAllocated 接收 + RDMA 发送 + RankTransferDone |
| KV Manager Decode  | `kv_transfer/decode.py`                | Recv 侧：DecodePrepare → block 分配 → DecodeAllocated → 等待 PrefillDone |
| Endpoint        | `kv_transfer/endpoint.py`                 | ZMQ master/slave/remote 三模式抽象              |
| Protocol        | `kv_transfer/protocol.py`                 | 4 种协议消息 dataclass + msgpack 序列化           |
| Transfer Buffers | `kv_transfer/transfer_buffers.py`        | TransferBuffers (per-cache block ID lists)  |
| Transfer Plan   | `kv_transfer/transfer_plan.py`            | StaticTransferPlan builder + TransferPlan + create_transfer_plan |
| Cache Info      | `kv_transfer/cache_info.py`               | CacheInfo / InstanceCacheInfos: per-cache distribution 与 chunk 计算 |
| Task Info       | `kv_transfer/task_info.py`                | TaskInfo: per-request 状态聚合                |
| Transfer Engine | `kv_transfer/mooncake/transfer_engine.py` | Mooncake RDMA 引擎封装 + Bootstrap Server    |
| Metadata        | `kv_transfer/mooncake/metadata.py`        | MetadataBuffers（辅助 buffer 管理）             |
| Hook            | `hooks.py`                                | `MooncakeKVTransferHook`，prefill/decode 时触发传输 |
| 监控              | `metrics/prometheus_collector.py`         | Prometheus 指标定义与辅助函数                     |
| 配置              | `config/serve_config.yaml`                   | 服务和 PD 分离的统一基础配置                         |
| 脚本              | `script/start_pd_disagg_*.sh`             | 本地启动脚本                                   |
| 脚本              | `script/srun_pd_disagg_*.sh`              | SLURM 多机启动脚本                             |


## 常见问题


| 现象                    | 排查方向                                                                        |
| --------------------- | --------------------------------------------------------------------------- |
| P/D 启动卡在 Bootstrap 连接 | 确认 Router 已启动 Bootstrap，且 coordinator.host/port 对 P/D 节点可达 |
| Decode 长时间 WAITING    | 检查 Prefill 是否收到 `DecodeAllocated`；查看 Prefill 传输线程日志；确认 RDMA 设备已正确检测（见 `chitu/distributed/infiniband.py`） |
| RDMA "Bad address"    | 确认 `register_cache()` 在 cache 创建后调用；检查各层 ptr/len 无重叠 |
| transfer bytes mismatch | Decode 日志出现 `rank_bytes` vs `recv_bytes` 不一致；检查 Prefill/Decode 的 `split_size` 配置是否匹配 |
| GLM-5 + auto attn_type + MTP>1 乱码 | `HopperMixedBackend` 与 MTP 不兼容，设置 `attn_type=flash_mla` |
| 同节点多进程端口冲突            | 调度器请求端口为随机分配并通过 coordinator 发现；仅需为每个 torchrun 进程指定不同的 `--master_port`；Router port 使用 22001 + job_offset 基础 |
| head-repeat 模型乱码          | 检查 `CacheInfo` 中的 `replica_size` / `split_id` 计算是否正确（head 按连续分组分布，同 head 的 rank 相邻）；确认 `StaticTransferPlan builder` replica_ratio 过滤逻辑 |
