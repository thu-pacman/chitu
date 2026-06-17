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
                          │  API Server (HTTP :21003)    │
                          │  PDCoordination (ZMQ :29800) │
                          │  MetadataSync   (ZMQ :29801) │
                          │  Bootstrap      (HTTP :8080) │
                          └──────┬────────────┬──────────┘
                      ZMQ PUSH  │            │  ZMQ PUSH
                  ┌─────────────┘            └────────────────┐
                  ▼                                           ▼
     ┌────────────────────────┐              ┌────────────────────────────┐
     │   Prefill (节点 B)     │              │    Decode (节点 A)         │
     │                        │    RDMA      │                            │
     │  PDSchedulerService    │─────────────▶│  PDSchedulerService        │
     │  PrefillOnlyScheduler  │  KV + Token  │  DecodeOnlyScheduler       │
     │  KVManager (PREFILL)   │              │  KVManager (DECODE)        │
     └────────────────────────┘              └────────────────────────────┘
```

## 通信分层

系统通信分为**控制面**和**数据面**两层：

### 控制面（ZMQ + HTTP）


| 通信路径                         | 协议             | 作用                          |
| ---------------------------- | -------------- | --------------------------- |
| Router → P/D Scheduler       | ZMQ PUSH/PULL  | 分发请求                        |
| Router ↔ P/D（PDCoordination） | ZMQ PULL + REP | endpoint发现、元数据同步            |
| D → P（DECODE_REGISTER）       | ZMQ multipart  | Decode 向 Prefill 注册本地显存指针   |
| D → P（TRANSFER_INFO）         | ZMQ multipart  | Decode 发送 per-request 目标页索引 |
| P → D（Status Update）         | ZMQ PUSH       | Prefill 通知 Decode 传输完成      |
| P/D → Bootstrap              | HTTP PUT/GET   | Prefill endpoint注册与查询       |


### 数据面（RDMA）

Prefill 通过 Mooncake Transfer Engine 执行Device到Device的 RDMA 写入：

- **KV Cache**：按层、按页索引映射，支持连续块聚合以减少 RDMA 调用
- **First Token**：通过 MetadataBuffers 的 aux buffer 传输（包含 first token id）

## 请求生命周期

以 1P1D 为例：

```
 Client                Router                 Prefill                  Decode
   │                     │                       │                       │
   │─── HTTP Request ──▶ │                       │                       │
   │                     │── ZMQ prefill req ──▶ │                       │
   │                     │── ZMQ decode req ────────────────────────────▶│
   │                     │                       │                       │
   │                     │                       │   ◄─ DECODE_REGISTER ─│ (一次性注册显存指针)
   │                     │                       │    ◄─ TRANSFER_INFO ──│ (目标页索引 + aux index)
   │                     │                       │                       │
   │                     │                    prefill 计算                │
   │                     │                    产出 KV + logits            │
   │                     │                       │                       │
   │                     │                       │── RDMA: KV Cache ───▶ │
   │                     │                       │── RDMA: aux (token) ─▶│
   │                     │                       │── ZMQ: Success ──────▶│
   │                     │                       │                       │
   │                     │                       │              插入 KV 到 CacheManager
   │                     │                       │              开始 decode 循环
   │◀── stream tokens ── │◀──────────── ZMQ token ───────────────────────│
   │                     │                       │                       │
```

### 详细步骤

**1. Router 分发**

`PDRequestRouter._add_pd_request()` 为请求生成 `request_id`，通过轮询选择一个 Prefill 和一个 Decode Scheduler，将同一请求同时发给 P&D。Decode 侧的消息额外携带 `prefill_scheduler_id`，用于后续PD配对。

**2. Decode 准备**

Decode Scheduler 收到请求后入队到 `_decode_incoming_q`。后台推进线程将请求推入 `_decode_prealloc_q`，同时通过 `PDScheduler._send_pd_prepare_transfer()` 向 Decode 的 KVManager 发送准备指令。KVManager 执行：

- 首次连接时，通过 Bootstrap/Coordination 发现 Prefill endpoint
- 向目标 Prefill 发送 `DECODE_REGISTER`（本地 KV/aux 显存指针，一次性、幂等）
- 为本请求分配page index和 aux index
- 向目标 Prefill 发送 `TRANSFER_INFO`（page index + aux index）

**3. Prefill 计算与传输**

Prefill Scheduler 收到请求后经过 `_prefill_incoming_q → _prefill_bootstrap_q → _prefill_ready_q` 的队列推进，等待对应的 `TRANSFER_INFO` 就绪后创建 Task 执行 prefill。

Prefill 完成后触发 `MooncakeKVTransferHook.on_prefill_done()`，调用 `KVManager.send_kv_cache()`：

- 从 CacheManager 获取源端页索引
- 在 MetadataBuffers 分配 aux index，写入first token
- 入队 `TransferKVChunk` 给后台传输线程

传输线程从队列取出任务，按层通过 `ThreadPoolExecutor` 并行执行 RDMA 传输，完成后通过 ZMQ 发送 `KVPoll.Success` 给 Decode。

**4. Decode 执行**

Decode 的 KVManager 轮询 `request_status`，收到 Success 后：

- 从 aux buffer 取回first token
- 调用 `CacheManager.insert_kv_cache_from_transfer()` 登记 KV 页表
- 请求从 `_decode_prealloc_q` 推入 `_decode_ready_q`，创建 Task 进入 decode 循环
- Token 通过 `DPTokenManager` 经 ZMQ 回传给 Router，Router 流式返回给 Client

## 核心组件

### KVManager（`kv_transfer/kv_manager.py`）

KV 传输的核心管理器，根据 `DisaggregationMode` 区分 Prefill/Decode 行为。

**关键枚举**

```python
class DisaggregationMode(Enum):
    NULL = "null"       # 未启用
    PREFILL = "prefill" # Prefill 模式
    DECODE = "decode"   # Decode 模式

class KVPoll(Enum):
    Waiting = 0  # 等待传输
    Success = 1  # 传输完成
```

**Prefill 模式线程模型**

```
                   ┌─── control rank (pp=0, tp=0) ─────┐
                   │                                   │
  Decode ─ZMQ──▶   │  外部端口: DECODE_REGISTER /       │  ──ZMQ PUB──▶ 所有 PP/TP ranks
                   │            TRANSFER_INFO          │
                   │                                   │
  PP/TP ranks      │  内部端口: STAGE_DONE 收集          │  ──ZMQ PUSH─▶ Decode (Success)
  ──ZMQ PUSH──▶    │                                   │
                   └───────────────────────────────────┘
```

- **control rank**（pp=0, tp=0）：对外暴露 ZMQ 端口，接收 Decode 的注册和传输请求，通过 PUB/SUB 广播给所有 PP/TP ranks
- **传输线程**（每个 rank）：从 `transfer_queue` 取 `TransferKVChunk`，执行 RDMA 传输，完成后发送 `STAGE_DONE` 给 control rank
- **control rank 汇总**：收齐所有 shard 的 `STAGE_DONE` 后，向 Decode 发送 `KVPoll.Success`

**Decode 模式**

- 后台线程发现所有 Prefill engine_rank，对每个执行一次性endpoint注册
- per-request 向目标 Prefill 发送 `TRANSFER_INFO`
- `(tp=0, pp=0)` owner rank 对外暴露 prepare/status endpoint，并通过一个 decode-local ZMQ broadcast 将 `PD_PREPARE_TRANSFER` / `KVPoll.Success` 同步给本地其他 TP/PP ranks
- 轮询 `request_status` 等待 `KVPoll.Success`

### ZMQ 消息协议（`kv_manager.py`）

所有控制面消息使用 ZMQ multipart 格式，固定位置索引，可选字段用空字节 `b""` 作为 placeholder。

**DECODE_REGISTER**（Decode → Prefill，一次性注册）


| Frame索引 | 名称                  | 内容                              |
| ------- | ------------------- | ------------------------------- |
| 0       | ROOM                | `UUID(int=0)` 的 bytes（表示注册而非请求） |
| 1       | TYPE                | `b"DECODE_REGISTER"`            |
| 2       | DECODE_IP           | ASCII 字符串                       |
| 3       | DECODE_PORT         | ASCII 字符串                       |
| 4       | SESSION_ID          | Mooncake session ID（ASCII）      |
| 5       | PACKED_KV_PTRS      | uint64 数组 bytes（各层 KV 地址）       |
| 6       | PACKED_AUX_PTR      | uint64 bytes（aux buffer 地址）     |
| 7       | PACKED_LINEAR_PTRS  | uint64 数组 bytes                 |
| 8       | TP_SIZE_ASCII       | Decode 侧 TP 大小                  |
| 9       | PACKED_INDEXER_PTRS | uint64 数组 bytes                 |


**TRANSFER_INFO**（Decode → Prefill，per-request）


| Frame索引 | 名称                    | 内容                      |
| ------- | --------------------- | ----------------------- |
| 0       | ROOM                  | request_id 的 UUID bytes |
| 1       | TYPE                  | `b"TRANSFER_INFO"`      |
| 2       | DECODE_IP             | ASCII 字符串               |
| 3       | DECODE_PORT           | ASCII 字符串               |
| 4       | SESSION_ID            | Mooncake session ID     |
| 5       | DST_KV_INDICES_BYTES  | int32 数组 bytes（目标页索引）   |
| 6       | AUX_INDEX_ASCII       | aux buffer 槽位索引         |
| 7       | LINEAR_INDICES_BYTES  | int32 数组 bytes          |
| 8       | INDEXER_INDICES_BYTES | int32 数组 bytes          |


**STAGE_DONE**（Prefill PP/TP rank → control rank）


| Frame索引 | 名称       | 内容                         |
| ------- | -------- | -------------------------- |
| 0       | ROOM     | request_id 的 UUID bytes    |
| 1       | TYPE     | `b"STAGE_DONE"`            |
| 2       | PP_STAGE | PP stage 索引                |
| 3       | TP_RANK  | TP rank 索引                 |
| 4       | AUX_DONE | `"0"` 或 `"1"`（是否完成 aux 传输） |


**Status Update**（Prefill control rank → Decode）

```
[room.bytes, "1"]    # "1" = KVPoll.Success
```

### PDCoordinationService（`pd_coordination.py`）

运行在 Router 进程中，负责服务发现和meta data同步。

**两个 ZMQ Socket**


| Socket              | 类型   | 端口    | 用途                                                |
| ------------------- | ---- | ----- | ------------------------------------------------- |
| coordination_socket | PULL | 29800 | 接收 P/D 的协调消息（prefill_complete, decode_complete 等） |
| metadata_socket     | REP  | 29801 | 同步meta data请求（endpoint注册/查询）                      |


**元数据同步请求类型**


| 类型                            | 方向                     | 作用                               |
| ----------------------------- | ---------------------- | -------------------------------- |
| `set_prefill_ctrl_endpoint`   | Prefill → Coordination | 注册 Prefill control rank 的 ZMQ 端口 |
| `get_prefill_ctrl_endpoint`   | P/D → Coordination     | 查询 Prefill control rank endpoint |
| `set_decode_prepare_endpoint` | Decode → Coordination  | 注册 Decode 准备 endpoint            |
| `get_decode_prepare_endpoint` | Prefill → Coordination | 查询 Decode 准备 endpoint            |
| `set_decode_status_endpoint`  | Decode → Coordination  | 注册 Decode 状态 endpoint（附带 internal broadcast port） |
| `get_decode_status_endpoint`  | Prefill / Decode → Coordination | 查询 Decode 状态 endpoint与 internal broadcast |


### MooncakeBootstrapServer（`kv_transfer/mooncake/transfer_engine.py`）

运行在 Router 进程中的轻量 HTTP 服务。Router 启动时只要 `kv_transfer_backend == "mooncake"` 即**启动**。

MooncakeBootstrapServer vs PDCoordinationService：

- **MooncakeBootstrapServer**：Prefill endpoint 集合。Decode 通过 `_get_bootstrap_info()` 查询 Bootstrap 来发现 Prefill 的 ZMQ 地址，通过 `_discover_prefill_engine_ranks()` 探测有多少个 Prefill。这是 Decode 找到 Prefill 的手段。
- **PDCoordinationService：**：内部控制面端点同步。用于 Prefill PP/TP ranks 之间发现 control rank 的广播端口、Decode prepare/status endpoint，以及 Decode 内部 broadcast endpoint。

唯一存在 fallback 的地方是 Prefill **注册自身** endpoint 这一步：优先通过 Coordination Service 注册，仅当 Coordination 不可用时才 fallback 到 Bootstrap 的 `PUT /route`（`_register_to_bootstrap()`）。


| 方法  | 路径                          | 作用                                                                                |
| --- | --------------------------- | --------------------------------------------------------------------------------- |
| PUT | `/route`                    | Prefill 注册 `{role, rank_ip, rank_port, engine_rank}`（Coordination 不可用时的 fallback） |
| GET | `/route?engine_rank=<rank>` | Decode 查询 Prefill endpoint（主要使用路径）                                                |
| GET | `/route?engine_rank=-1`     | 返回 `dp_size`                                                                      |
| GET | `/health`                   | 健康检查                                                                              |


### PDScheduler（`pd_scheduler.py`）

继承自 `Scheduler`，根据模式分为 `PrefillOnlyScheduler` 和 `DecodeOnlyScheduler`。

**Prefill 队列模型**

```
请求到达 → _prefill_incoming_q → _prefill_bootstrap_q → _prefill_ready_q → TaskPool → 执行
                                   (等待 TRANSFER_INFO)    (可调度)
```

**Decode 队列模型**

```
请求到达 → _decode_incoming_q → _decode_prealloc_q → _decode_ready_q → TaskPool → 执行
             (发送 PREPARE)      (等待 KVPoll.Success)   (可调度)
```

### MooncakeKVTransferHook（`hooks.py`）

注入到 Executor 中的 Hook，在 prefill 完成后自动触发 KV 发送：

- `on_prefill_done(req_ids, logits, tasks)`：Prefill 模式下，对 logits 做 sample 得到首 token，调用 `KVManager.send_kv_cache()`
- PP>1 时，非最后一个 PP stage 只发送 KV（不含First token）

## 配置

不论什么拓扑（1P1D、2P3D、XP YD），统一使用 `pd_disagg_serve_config.yaml` 作为基础配置。启动脚本通过 Hydra 命令行 override 动态覆盖 `prefill_schedulers` / `decode_schedulers` 列表和 `dp_size` 等字段，无需为每种拓扑维护单独的配置文件。

配置文件位于 `chitu/config/pd_disagg_serve_config.yaml`，继承 `serve_config.yaml` 的通用项：

```yaml
defaults:
  - serve_config      # 继承模型/推理/校验等通用配置
  - _self_

dp_config:
  enabled: True
  scheduler_base_host: 0.0.0.0
  scheduler_base_port: 29610       # Scheduler ZMQ 基础端口
  dp_size: 2                       # P + D 总实例数（启动时覆盖）
  dp_id: 0                         # 当前进程的 DP ID（启动时覆盖）

  router:
    is_router: True                # Router 进程设为 True，P/D 设为 False
    host: 0.0.0.0
    port: 21003                    # HTTP 推理入口端口
    stats_port: 29600              # 统计上报端口
    token_port: 29700              # Token 回传端口
    routing_algorithm: "power_of_two_choices"

    pd_disaggregation:
      enabled: True
      log_verbose: False
      coordination_port: 29800     # PDCoordination 协调端口
      metadata_sync_port: 29801    # 元数据同步端口
      kv_transfer_backend: "mooncake"
      bootstrap_port: 8080         # Bootstrap HTTP 端口

      kv_transfer:
        buffer_size: 2048
        transfer_timeout: 30.0
        max_concurrent_transfers: 8
        decode_wait_timeout_s: 300.0   # Decode 等待 KV 传输完成的超时
        decode_resend_interval_s: 0.5  # Decode 重发 TRANSFER_INFO 间隔
        decode_poll_interval_s: 0.05   # Decode 轮询 KV 状态间隔

    # 以下列表在启动时由脚本动态覆盖（host/port 按实际节点 IP 和实例索引填充）
    prefill_schedulers:
      - host: 0.0.0.0
        port: 29620
        max_batch_size: 32
        max_total_tokens: 8192
        batching_strategy: "varlen"

    decode_schedulers:
      - host: 0.0.0.0
        port: 29630
        scheduling_strategy: "immediate"
```

启动脚本 `srun_pd_disagg_base_apptainer.sh` 会根据 `--prefill` / `--decode` 参数自动生成 `prefill_schedulers=[{host:...,port:...},...]` 和 `decode_schedulers=[...]` 的 Hydra override 传给 Router，同时为每个 P/D 实例设置对应的 `dp_config.dp_id` 和 `dp_config.scheduler_base_port`。`--pd-spec` 中的参数（如 `decode_wait_timeout_s`）会覆盖上面 `kv_transfer` 下的默认值。

### 端口矩阵（2P3D 示例）


| 组件             | 端口    | 协议   | 节点  | 用途                          |
| -------------- | ----- | ---- | --- | --------------------------- |
| Router API     | 21003 | HTTP | A   | 推理入口 `/v1/chat/completions` |
| Router Stats   | 29600 | ZMQ  | A   | P/D 统计心跳                    |
| Router Token   | 29700 | ZMQ  | A   | Decode → Router token 回传    |
| PDCoordination | 29800 | ZMQ  | A   | 协调消息                        |
| MetadataSync   | 29801 | ZMQ  | A   | 元数据同步                       |
| Bootstrap      | 8080  | HTTP | A   | Mooncake endpoint目录         |
| Prefill P0     | 29620 | ZMQ  | B   | Router → P0 请求              |
| Prefill P1     | 29621 | ZMQ  | B   | Router → P1 请求              |
| Decode D0      | 29630 | ZMQ  | A   | Router → D0 请求              |
| Decode D1      | 29631 | ZMQ  | A   | Router → D1 请求              |
| Decode D2      | 29632 | ZMQ  | A   | Router → D2 请求              |
| P ↔ D RDMA     | —     | RDMA | —   | KV/aux 显存直传                 |


## 运行方式

### 环境变量


| 变量                     | 必须       | 说明                           |
| ---------------------- | -------- | ---------------------------- |
| `PD_MASTER_ADDR`       | P/D 节点必设 | Router 的可达 IP（不要用 127.0.0.1） |
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


**实例参数支持的 key：** `tp`, `pp`, `dp`, `ep`, `max_seq_len`, `max_batch_size`, `max_new_tokens`, `chunk`(仅 prefill), `full_warmup`, `nnodes`, `nproc`。含 `.` 的 key 自动作为 Hydra override（如 `infer.memory_utilization=0.90`）。

#### 示例 1：DeepSeek-R1（4 节点，1P1D，PP=2）

1 个 Prefill（TP=8, PP=2，占 2 节点 16 卡）+ 1 个 Decode（TP=1, DP=16, EP=16，占 2 节点 16 卡）。

```bash
bash script/srun_pd_disagg_base_apptainer.sh \
  DeepSeek-R1 /data/nfs/DeepSeek-R1 /path/to/chitu-mooncake.sif \
  --nodes 4 --router-port 21006 \
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
  --nodes 4 --router-port 21006 \
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
curl -X POST http://<Router_IP>:21003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":true}'
```

## 监控与观测

### Prometheus 指标（`chitu/metrics/prometheus_collector.py`）

每个 P/D 进程启动独立的 Prometheus HTTP exporter。

**延迟分布（Histogram）**


| 指标名                                  | 标签              | 说明                                        |
| ------------------------------------ | --------------- | ----------------------------------------- |
| `chitu_pd_stage_duration_seconds`    | `role`, `stage` | 各阶段延迟（如 `router.recv`, `prefill.kv_send`） |
| `chitu_e2e_request_duration_seconds` | —               | 端到端请求延迟                                   |
| `chitu_time_to_first_token_seconds`  | —               | 首 token 延迟（TTFT）                          |
| `chitu_kv_transfer_duration_seconds` | —               | KV RDMA 传输延迟                              |
| `chitu_kv_transfer_size_bytes`       | —               | 单次 KV 传输大小                                |


**实时状态（Gauge）**


| 指标名                                        | 标签                   | 说明             |
| ------------------------------------------ | -------------------- | -------------- |
| `chitu_pd_queue_size`                      | `role`, `queue_name` | 各队列深度          |
| `chitu_router_pending_requests`            | —                    | Router 待处理请求   |
| `chitu_active_requests`                    | `role`               | 活跃请求数          |
| `chitu_kv_transfer_speed_gbps`             | —                    | 最近一次 KV 传输速度   |
| `chitu_kv_cache_usage_ratio`               | `rank`, `dp_id`      | KV Cache 使用率   |
| `chitu_used_blocks` / `chitu_total_blocks` | `rank`, `dp_id`      | KV Cache 块使用情况 |
| `chitu_cuda_used_bytes` / `chitu_cuda_total_bytes` | `rank`, `dp_id` | GPU 显存使用       |


**计数器（Counter）**


| 指标名                                | 标签              | 说明                 |
| ---------------------------------- | --------------- | ------------------ |
| `chitu_completed_requests_total`   | `role`          | 完成请求数              |
| `chitu_kv_transfer_failures_total` | `role`          | KV 传输失败数           |
| `chitu_request_timeouts_total`     | `stage`         | 请求超时数              |
| `chitu_total_generated_tokens`     | `rank`, `dp_id` | 生成 token 总数        |
| `chitu_total_prompt_tokens`        | `rank`, `dp_id` | 处理 prompt token 总数 |


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
| `decode endpoint registered to prefill`  | Decode KVManager  | 完成endpoint注册    |
| `finished kv cache transfer`             | Prefill KVManager | KV RDMA 传输完成    |
| `finished aux transfer`                  | Prefill KVManager | aux 传输完成        |
| `received kv cache`                      | Decode KVManager  | 收到 KV，准备 decode |
| `created pd request: <rid> -> P{k}-D{m}` | Router            | 请求分配到具体 P/D     |


启用详细日志：配置 `dp_config.router.pd_disaggregation.log_verbose=True`。

### Prometheus Server 集成

Router 进程可选启动内置的 Prometheus Server（`PrometheusServerManager`），自动抓取所有 P/D 的 exporter。也可使用外部 Prometheus，将各 exporter 地址加入 scrape 配置。

## TP 并行

- 仅 TP 主 rank（rank 0）对外暴露 ZMQ 端口，处理控制面消息
- 非主 rank 通过 PUB/SUB 接收 control rank 广播的注册和传输信息
- 每个 TP rank 独立执行自己负责的 KV 层的 RDMA 传输
- 所有 rank 完成后由 control rank 汇总发送 Success

## 代码导航


| 模块              | 路径                                        | 说明                                       |
| --------------- | ----------------------------------------- | ---------------------------------------- |
| Router          | `pd_request_router.py`                    | 请求路由与 P/D 分发                             |
| Coordination    | `pd_coordination.py`                      | endpoint发现与元数据同步                         |
| Scheduler       | `pd_scheduler.py`                         | Prefill/Decode 调度器与队列管理                  |
| Service         | `pd_service.py`                           | 进程入口与初始化                                 |
| KV Manager      | `kv_transfer/kv_manager.py`               | KV 传输核心逻辑                                |
| Transfer Engine | `kv_transfer/mooncake/transfer_engine.py` | Mooncake RDMA 引擎封装 + Bootstrap Server    |
| Metadata        | `kv_transfer/mooncake/metadata.py`        | 首 token aux buffer 管理                    |
| Hook            | `hooks.py`                                | `MooncakeKVTransferHook`，prefill 完成后触发传输 |
| 监控              | `metrics/prometheus_collector.py`         | Prometheus 指标定义与辅助函数                     |
| 配置              | `config/pd_disagg_*.yaml`                 | 各拓扑配置模板                                  |
| 脚本              | `script/start_pd_disagg_*.sh`             | 本地启动脚本                                   |
| 脚本              | `script/srun_pd_disagg_*.sh`              | SLURM 多机启动脚本                             |


## 常见问题


| 现象                    | 排查方向                                                                        |
| --------------------- | --------------------------------------------------------------------------- |
| P/D 启动卡在 Bootstrap 连接 | 确认 Router 已启动 Bootstrap（:8080）；`PD_MASTER_ADDR` 指向 Router IP，非回环地址          |
| Decode 长时间 WAITING    | 检查 Prefill 是否收到 TRANSFER_INFO；查看 Prefill 传输线程日志；确认 RDMA 设备已正确检测（见 `chitu/distributed/infiniband.py`） |
| RDMA "Bad address"    | 确认 `register_buffer_to_engine()` 在 CacheManager 注入后调用；检查各层 base_ptr/len 无重叠 |
| Router 序列化报错          | 确保请求 `messages` 是纯 dict 列表（非 pydantic 对象）                                   |
| 同节点多进程端口冲突            | 为每个 torchrun 进程指定不同的 `scheduler_base_port` 和 `--master_port`                |


