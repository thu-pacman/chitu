## 背景与目标
- 目标：将 Prefill（P）与 Decode（D）物理解耦，通过 RDMA 在不同节点之间传输 KV Cache 与first token logits，实现“Prefill 生成first token、Decode 持续生成tokens”的推理。
- 适配：1P1D 起步，可拓展多 P/D。对 TP/PP/DP 内部并行与 PD 解耦正交。

## 部署拓扑与进程
- Router
  - 进程入口：`chitu/serve/api_server.py` → `chitu/pd_request_router.py`
  - 职责：
    - 接入 HTTP 请求（`/v1/completions`），进入 PD 分流路径。
    - 启动 PDCoordination（ZMQ）：P/D 服务发现、状态协调。
    - 启动 Mooncake Bootstrap Server（HTTP）：为 RDMA 握手提供中心化注册/查询。
- Prefill 节点（P）
  - 进程入口：`chitu/serve/scheduler.py` → `chitu/pd_service.py` → `chitu/pd_scheduler.py`
  - 职责：
    - 执行 prefill，产出 KV Cache 与首 token logits。
    - 通过 `KVManager(disaggregation_mode=PREFILL)` 用 RDMA 向 D 传输 KV/aux，并以 ZMQ 通知状态。
    - TP 并行时仅 TP 主 rank 对外监听端口并处理 ZMQ；其他 TP rank 仅参与计算与组内通信（不绑定端口）。
- Decode 节点（D）
  - 进程入口同 P（decode-only 模式）
  - 职责：
    - 在收到 PD decode 请求后，向 P 注册本地内存指针与通信端口，等待 KV/aux 到齐后进入 decode。
    - 可就绪即解，也可滚动聚合做变长批 decode。
    - TP 并行时仅 TP 主 rank 对外监听端口并处理 ZMQ；其他 TP rank 仅参与计算与组内通信（不绑定端口）。

端口与组件（默认示例，实际以 `chitu/config/pd_disagg_1p1d_multi_node.yaml` 为准）
 
### 端口与网络说明（端口矩阵）

以下为 1P1D 的默认端口规划，实际以 `chitu/config/pd_disagg_1p1d_multi_node.yaml` 为准（键名见“配置键”列）。

| 组件 | 端口 | 协议 | 通信双方 | 用途 | 配置键/环境变量 |
| --- | --- | --- | --- | --- | --- |
| Router API | 21003 | HTTP | Client → Router | 推理入口 `/v1/completions` | `dp_config.router.port` |
| Router Stats | 29600 | ZMQ | P/D → Router | 上报统计/心跳（可选） | `dp_config.router.stats_port` |
| Router Token | 29700 | ZMQ | Router ↔ Router接收P 和 D 节点 Token 的端口 | 传输 Token | `dp_config.router.token_port` |
| Router Coordination | 29800 | ZMQ | Router ↔ P/D | PD 协调/服务发现/配对 | `dp_config.router.pd_disaggregation.coordination_port` |
| Router Metadata Sync | 29801 | ZMQ | Router ↔ P/D | 元数据同步（可选） | `dp_config.router.pd_disaggregation.metadata_sync_port` |
| Bootstrap Server | 8080 | HTTP | P/D → Router | Mooncake handshake服务（查询/注册 endpoint、session） | `dp_config.router.pd_disaggregation.bootstrap_port`；P/D 必设：`PD_MASTER_ADDR=<Router_IP>` |
| Prefill Scheduler | 29620 | ZMQ | Router → Prefill | Prefill 请求分发 | `dp_config.router.prefill_schedulers[0].port` |
| Decode Scheduler | 29630 | ZMQ | Router → Decode | Decode 请求分发 | `dp_config.router.decode_schedulers[0].port` |
| Prefill ↔ Decode | N/A | RDMA | Prefill ↔ Decode | KV/aux 显存直达传输 | `dp_config.router.pd_disaggregation.ib_device`（设备名）；endpoint通过 Bootstrap 服务提供 |

补充说明：
- P/D 需要能访问 `http://<PD_MASTER_ADDR>:<bootstrap_port>`；`PD_MASTER_ADDR` 必须为 Router 的可访问 IP（避免 `127.0.0.1/localhost/0.0.0.0`）。
- 若环境同时存在 `MASTER_ADDR`（torchrun 使用）与 `PD_MASTER_ADDR`，KVManager 优先使用 `PD_MASTER_ADDR`。
- ZMQ 端口用于 Router 与 P/D 的通信；RDMA 不占用以上端口，由 Mooncake 直接在设备上完成传输。
- Token 回传端口（默认 29700）仅由 Decode 节点使用；Prefill 节点不连接该端口且不发送 token。

### 2P3D 两节点端口与进程映射示例

以下为常用的两节点 2P3D 端口布局（Router+3×Decode 在节点 A；2×Prefill 在节点 B）。端口可通过配置/脚本覆盖(可参考chitu/config/pd_disagg_2p3d_multi_node.yaml)，但建议保持不冲突且分段清晰：

| 类别 | 实例 | 节点 | 端口/地址 | 说明 |
| --- | --- | --- | --- | --- |
| Router API | - | 节点A | 21003 | HTTP 推理入口 `/v1/chat/completions` |
| Router Stats | - | 节点A | 29600 | P/D → Router 统计/心跳（ZMQ） |
| Router Token | - | 节点A | 29700 | Router ↔ P/D token 回传通道（ZMQ） |
| PD Coordination | - | 节点A | 29800 | Router ↔ P/D 协调（ZMQ） |
| Metadata Sync | - | 节点A | 29801 | Router ↔ P/D 元数据同步（ZMQ） |
| Bootstrap | - | 节点A | 8080 | Router 托管的 Mooncake Bootstrap（HTTP） |
| Prefill Scheduler | P0 | 节点B | 29620 | Router → Prefill P0（ZMQ） |
| Prefill Scheduler | P1 | 节点B | 29621 | Router → Prefill P1（ZMQ） |
| Decode Scheduler | D0 | 节点A | 29630 | Router → Decode D0（ZMQ） |
| Decode Scheduler | D1 | 节点A | 29631 | Router → Decode D1（ZMQ） |
| Decode Scheduler | D2 | 节点A | 29632 | Router → Decode D2（ZMQ） |

注意：
- 同一物理节点上启动多个 Scheduler 时，请通过 `dp_config.scheduler_base_port` 或脚本参数为每个 `torchrun` 进程指定独立的 base port（例如 D0/D1/D2 分别用 29630/29631/29632），以避免 `Address already in use`。
- 2P3D 的 P/D 数量不影响 Router 自身端口；但 Prefill/Decode 列表的 host/port 需在配置中一一对应。
- TP 并行（任一 Scheduler 内）仅主 rank 绑定对外端口；从属 TP rank 不绑定端口，仅参与计算。

环境变量
- 所有 P/D 节点设置 `PD_MASTER_ADDR=<Router_IP>`（切勿使用 127.0.0.1，跨机不可达）。
- 若未设置 `PD_MASTER_ADDR`，某些组件可能回退到 `MASTER_ADDR`，但为避免与 torchrun 环境冲突，建议显式使用 `PD_MASTER_ADDR`。

## 通信与协议分层
- 控制面（Control Plane）
  - Router ↔ P/D：ZMQ
    - 用途：分发 PD 请求（prefill/decode）、调度配对P/D、P/D状态上报。
  - P ↔ D：ZMQ
    - 用途：传输层handshake与状态同步（decode 向 prefill 注册本地 KV/aux 指针；prefill 完成后回传状态 KVPoll）。
    - request body：multipart（`[room.bytes, ascii parts...]`），二进制 + ASCII。
- 数据面（Data Plane）
  - P ↔ D：RDMA（Mooncake Transfer Engine）
    - 用于直接显存到显存的 KV 与 logits 传输。
  - Bootstrap（HTTP，Router 托管）
    - endpoint（最小实现）：
      - PUT `/route`：Prefill 注册自身 `{role, rank_ip, rank_port, engine_rank}`。
      - GET `/route?engine_rank=<rank>`：返回目标 `{rank_ip, rank_port}`。
      - GET `/health`：健康检查。
    - 用途：为 P/D 提供会话与endpoint的目录服务。

## 组件与数据结构
- `KVManager`（`chitu/kv_transfer/kv_manager.py`）
  - Prefill 模式：
    - `decode_kv_args_table[session_id]`：decode 端回报的本地 KV/aux 基址信息（一次注册）。
    - `transfer_infos[room]`：decode 端 per-request 目的页索引与 aux 槽位。
    - 线程：
      - prefill 通信线程：接收 decode 的注册与 transfer 请求。
      - 传输线程：按层/块调度 RDMA 复制 KV，再复制 logits，并用 ZMQ 通知 KVPoll.Success。
  - Decode 模式：
    - 首次调用时通过 Bootstrap 获取 prefill endpoint，并以 ZMQ 向每个可用 prefill（engine_rank）进行一次性 endpoint 注册（幂等）。
    - 发起 per-request TransferInfo（目的页索引与 aux_index）给“目标 prefill”（由 Router 路由的 `prefill_scheduler_id` 指定；调用 `set_prefill_target_engine_rank(request_id, engine_rank)` 绑定）。
    - 等待 KVPoll.Success；取回 logits；调用 CacheManager 将 KV 插入/登记。
  - request_id 统一映射：`_to_uuid()` 将任意字符串稳定映射为 UUID（原生 UUID 或 UUID5），保证 Router/P/D 三方一致匹配。
  - 元数据缓冲：`MetadataBuffers.allocate/get/free` 管理 logits 的环形槽位。
  - 关键内部结构：
    - `TransferKVChunk(room, prefill_kv_indices, prefill_aux_index)`：P 侧入队的传输单元。
    - `KVArgsRegisterInfo(room, endpoint, dst_port, session_id, dst_kv_ptrs, dst_aux_ptr)`：D 侧注册到 P 的指针元信息。
    - `TransferInfo(room, endpoint, dst_port, session_id, dst_kv_indices, dst_aux_index)`：D → P 的 per-request 传输参数。
    - `_decode_registered_remote_set`：记录 D 已向哪些 Prefill 完成endpoint注册（幂等）。
    - `prefill_target_rank_by_room`：按请求绑定的目标 Prefill engine_rank。
- `PagedKVCacheManager`（`chitu/cache_manager.py`）
  - `get_contiguous_buf_infos()`：提供每层 KV 连续内存的基址、长度、单项长度（用于 RDMA 注册与寻址）。
  - `get_page_indices(req_id)`：prefill 侧导出该请求的页索引（源索引）。
  - `insert_kv_cache_from_transfer(req_id, page_indices, prefix_length)`：decode 侧登记页表与前缀长度（目的索引）。
  - 其他：`block_size/block_table/remove_task` 等用于页粒度管理与生命周期。
- `PDSchedulerService` + `PDScheduler`（`chitu/pd_service.py`、`chitu/pd_scheduler.py`）
  - Prefill-only：prefill → 产出 logits → `kv_manager.send_kv_cache(...)`。
  - Decode-only：`kv_manager.recv_kv_cache_and_insert(...)` → decode 循环。
  - Token Manager：仅在 Decode-only 或 Unified 模式初始化；Prefill-only 模式跳过，不做 Token 回传。

## 启动与 warmup
- warmup统一走本地 direct warmup（不依赖 Router/Bootstrap/P↔D）：`warmup_engine()`
  - 仅warmup算子/JIT/缓存；Router 独立启动 Bootstrap 与 PDCoordination，不参与warmup。
  - 注意：`model.decode(tokens, batch_size)` 的第二个参数是整型“BatchSize”，不是长度列表。

### 启动流程（2P3D，两节点示例）

- Router（节点A）：
  - 启动 API Server（21003）。
  - 启动 PDCoordination（29800/29801）。
  - 解析配置中的 Prefill/Decode 列表，建立至各 Scheduler 的 ZMQ 连接。
  - 启动 Mooncake Bootstrap（8080）。

- Prefill 节点（节点B，P0/P1 两个进程）：
  - `PDSchedulerService` 以 `prefill_only` 模式启动。
  - KVManager：
    - 启动 prefill 通信线程，绑定本地 `rank_port`。
    - 注册 KV 与 AUX 显存区域到 Mooncake 引擎。
    - 将自身 `{role=Prefill, rank_ip, rank_port, engine_rank=dp_id}` 注册到 Bootstrap。
  - 不初始化/不启动 DP Token Manager；不连接 Router Token 端口（29700）。
  - TP 并行：仅 TP 主 rank 绑定 `scheduler_base_port`；其他 TP rank 不绑定。

- Decode 节点（节点A，D0/D1/D2 三个进程）：
  - `PDSchedulerService` 以 `decode_only` 模式启动。
  - KVManager：
    - 启动 decode 通信线程，绑定本地 `rank_port`。
    - 注册 AUX 显存区域（和必要的 KV 接收区域）。
    - 后台线程发现所有 Prefill engine_rank（0..N-1），对每个 Prefill 做一次endpoint注册（幂等）。
  - TP 并行：仅 TP 主 rank 绑定 `scheduler_base_port`；其他 TP rank 不绑定。

脚本参考：`cinfer-ep/script/srun_pd_disagg_2p3d_two_nodes.sh`，会自动解析 `SLURM_NODELIST` 并为同节点多进程设置不同的 `scheduler_base_port` 与 `CUDA_VISIBLE_DEVICES`。
 - Decode 侧 TP=2 时，常见映射为三路进程分别使用 `(0,1) / (2,3) / (4,5)`；Prefill 侧 TP=2 常见映射为两路 `(0,1) / (2,3)`。

## 请求生命周期（1P1D）
- 1) Router 接入与分发
  - `/v1/completions` → PD 分流：生成一个 `request_id`，并将同一 id 的两条子请求分别发送到 P（prefill）与 D（decode）。
  - 序列化：`messages` 递归转纯 dict 列表，整个请求为纯 Python 基本类型，msgpack over ZMQ 传输。
- 2) Decode 端准备
  - D 收到 decode 请求 → 调 `recv_kv_cache_and_insert([rid], cache_mgr)`：
    - 若首次：通过 Bootstrap 查询 prefill endpoint；ZMQ 注册 decode 本地 `kv_ptrs/aux_ptr` 与 `session_id`。
    - 为本 rid 分配 aux 槽；可选：预留目的页索引（如 CacheManager 支持）。
    - 以 ZMQ 将 per-request `{dst_indices, aux_index}` 通知 prefill（TransferInfo）。
- 3) Prefill 计算与传输
  - P 收到 prefill 请求 → prefill → 产出 kv cache 和首 token logits。
  - 调 `send_kv_cache(logits, [rid], cache_mgr)`：
    - 从 CacheManager 获取源页索引（`get_page_indices`）。
    - 在 metadata_buffers 分配该 rid 的 `aux_index`。
    - 入队 `TransferKVChunk(room, prefill_kv_indices, prefill_aux_index)`。
- 4) RDMA 传输
  - Prefill 传输线程拉取队列中的元素TransferKVChunk，等待 D 的 TransferInfo 与 decode_kv_args：
    - 分层计算地址：`src = kv_base + src_idx*item_len`，`dst = dst_base + dst_idx*item_len`。
    - 使用 `transfer_engine.transfer_sync(session_id, src, dst, length)` 执行 KV 拷贝（可合并连续块）。
    - 复制 logits：`src = aux_base + prefill_aux_index*aux_item_len` → `dst = dst_aux + dst_aux_index*aux_item_len`。
- 5) 状态同步与 decode
  - 传输完成后，P 以 ZMQ `PUSH` 向 D 发布 `[room.bytes, "1"]`（`KVPoll.Success`）。
  - D 端等待 rid 状态变为 Success，取回 logits，调用 `insert_kv_cache_from_transfer()` 登记 KV，然后进入 decode。
  - Decode 结束后清理任务与缓存（P 侧也可 `remove_task(room)` 清理源页）。

## 请求生命周期（2P3D）

- 1) Router 接入与路由
  - 为请求生成 `request_id`。
  - 以轮询或其他 LB 策略选择一个 Prefill `P{k}` 与一个 Decode `D{m}`，记录到 `PendingPDRequest` 并打印日志锚点：
    - `created pd request: <rid> -> P{k}-D{m}`。
  - 向 `P{k}` 和 `D{m}` 同时发送 PD 子请求（消息体均为纯 dict 序列）。

- 2) Decode 端准备（D{m}）
  - `PDScheduler._process_decode_request()` 保存 `prefill_scheduler_id=k`，并调用：
    - `kv_manager.set_prefill_target_engine_rank(<rid>, k)`。
  - `kv_manager.recv_kv_cache_and_insert([rid])`：
    - 若首次：遍历发现的所有 Prefill `engine_rank`，对尚未注册的 Prefill 做endpoint注册（一次性、幂等）。
    - 为 `<rid>` 分配 AUX 槽；可选：向 CacheManager 预留目的页索引。
    - 仅向目标 Prefill `P{k}` 发送 per-request `TransferInfo`（包含目的页索引与 `aux_index`）。

- 3) Prefill 计算与出数（P{k}）
  - `PDScheduler._process_prefill_request()` 执行真实 prefill，产出首 token logits。
  - `kv_manager.send_kv_cache(logits, [rid])`：
    - 查询 CacheManager 源端页索引，分配 `prefill_aux_index`。
    - 入队 `TransferKVChunk(room=<rid>, prefill_kv_indices, prefill_aux_index)` 给传输线程。

- 4) RDMA 传输（P{k} → D{m}）
  - 传输线程检测到 `<rid>` 的 `TransferInfo` 与 decode 端指针已就绪：
    - 逐层计算地址并执行 KV 拷贝；随后拷贝 logits（AUX）。
    - 成功后以 ZMQ 通知 D：`[room.bytes, "1"]`（`KVPoll.Success`）。

- 5) Decode 执行（D{m}）
  - D 收到 Success：
    - 从 AUX 读取 logits；
    - 将已接收 KV 插入 CacheManager 并设置 `prefix_length`；
    - 进入 decode 循环直至生成完成；
    - 清理任务与缓存。

多 P/D 关键点：
- D 仅向目标 P 发送 per-request `TransferInfo`（避免 “交叉发送”）；
- D 对所有 P 做一次endpoint注册，保证任意请求都能快速完成handshake；
- Bootstrap 对不存在的 `engine_rank` 返回 404，属正常探测现象（已在 KVManager 中降级为 debug/warn）。

日志锚点（便于排查）：
- Router：`created pd request: <rid> -> Pk-Dm`。
- Prefill：`processing prefill request: <rid>`、`finished kv cache transfer for <rid>`、`finished aux transfer for <rid>`。
- Decode：`processing decode request: <rid>`、`decode endpoint registered to prefill (engine_rank=...)`、`received kv cache for <rid>`。

## 内存注册与时序
- 注册时机（避免 Bad address/重叠）
  - 仅在 `set_cache_manager()` 注入真实 CacheManager 后调用 `register_buffer_to_engine()`。
  - 为 KV 与 AUX 做幂等注册（一次成功后不再重复），失败则熔断该类传输尝试并降级。
- 注册信息来源
  - KV：`get_contiguous_buf_infos()` 提供每层 base_ptr/len/item_len。
  - AUX：来自 `MetadataBuffers.get_buf_infos()`。
- 注意
  - Router 托管 Bootstrap；P/D 请用 `PD_MASTER_ADDR=<Router_IP>` 访问，不要使用回环地址。
  - 传输线程与通信线程分离，避免阻塞。

## Batch策略与性能
- 默认策略（TTFT 优先）
  - P：单请求 prefill 完成即发送 KV/aux。
  - D：某请求状态成功即开始 decode，不等待其它请求。
- 吞吐优化（可选）
  - D 端将若干“已就绪请求”一起调用 `recv_kv_cache_and_insert([...])`，Batch一次性进入 decode；或滚动聚合成变长批。
  - 传输侧合并连续块减少 RDMA 调用次数。
- 删除/回收
  - Prefill 完成后，可在 P 侧 `remove_task(room)` 及时释放源端页表；Decode 结束释放目的端资源。

### Batch 请求时序（P×D）

- 路由与绑定
  - Router 为每个原始请求生成独立 `request_id`，Batch内各请求相互独立。
  - 在 2P3D 等多实例下，Router 为每个请求绑定 `prefill_scheduler_id → decode_scheduler_id`（轮询或其他负载均衡策略），Decode 侧据此调用 `kv_manager.set_prefill_target_engine_rank(rid, prefill_id)` 选取目标 Prefill。

- Decode 端准备（可Batch）
  - D 将若干“待就绪请求”一起调用：`kv_manager.recv_kv_cache_and_insert([rid1, rid2, ...], cache_mgr)`。
  - 对每个 rid：
    - 分配 AUX 槽位（存放 first token logits）。
    - 可选：调用 CacheManager 的 `reserve_blocks_for_transfer()` 预留目的页索引（若实现）。
    - 向“目标 Prefill”发送 per-request `TransferInfo{dst_kv_indices, aux_index}`。
  - 函数内部等待所有 rid 的 `KVPoll.Success`；成功后返回形如 `[B, vocab]` 的 batched logits 张量，随后可选择：
    - 就绪即解：逐个请求进入 decode；
    - 聚合解码：把这批就绪请求组Batch（varlen/定长均可）进入一次 decode。

- Prefill 侧Batch处理与出数
  - Scheduler 将就绪的 Prefill 请求按照后端策略打包（如 PackedTasks），一次完成若干请求的 prefill。
  - Prefill 完成后调用 `kv_manager.send_kv_cache(logits, [rid1, rid2, ...], cache_mgr)`：
    - 为每个 rid 在 `MetadataBuffers` 分配一个 `aux_index` 并写入对应行的 logits。
    - 通过 `cache_mgr.get_page_indices(rid)` 取源端页索引，入队 `TransferKVChunk(room, prefill_kv_indices, prefill_aux_index)`。
  - 传输线程按请求逐个消化队列项；与 Decode 端的 per-request `TransferInfo` 对齐后发起 KV→AUX 的 RDMA 拷贝。

- 顺序与一致性
  - Batch内请求的 P/D 对齐依赖统一的 `_to_uuid()` 映射，保证 Router/P/D 三方对同一 `request_id` 的一致识别。
  - 任何一个请求独立完成即可解码，不受同Batch其它请求的影响；队列与状态均为 per-request 管理。

## 线程模型（Prefill 计算线程与传输线程）

- 通信线程（Prefill 侧）
  - 初始化：`start_prefill_thread()` 在本地 `rank_port` 上绑定 ZMQ `PULL`。
  - 消息分类：
    - `room == UUID(int=0)`：Decode 端一次性注册本地 `KVArgsRegisterInfo{dst_kv_ptrs, dst_aux_ptr, mooncake_session_id}`，以 `session_id` 作为键入表。
    - 其他 `room`：接收 per-request `TransferInfo{dst_kv_indices, dst_aux_index}`，按 `room` 记录。
  - 职责：仅负责握手与登记，不做数据传输。

- 传输线程（Prefill 侧）
  - 队列：`FastQueue` 承接 `send_kv_cache()` 入队的 `TransferKVChunk{room, prefill_kv_indices, prefill_aux_index}`。
  - 调度：后台 `transfer_worker()` 消费队列元素TransferKVChunk，等待对应 `TransferInfo` 与 `KVArgsRegisterInfo` 就绪。
  - RDMA：
    - 先用 `group_concurrent_contiguous()` 对源/目的页索引做连续分段聚合。
    - 每层提交一个任务到 `ThreadPoolExecutor` 并行处理；每个任务内按聚合段调用 `transfer_engine.transfer_sync(session_id, src_addr, dst_addr, length)`。
    - KV 成功后，再拷贝 AUX：`aux_base + prefill_aux_index*item_len → dst_aux + dst_aux_index*item_len`。
  - 回写与清理：
    - 成功后通过 `sync_status_to_decode_endpoint()` 以 ZMQ `PUSH` 写回 `[room.bytes, "1"]`（`KVPoll.Success`）。
    - 更新 `request_status[room]=Success`；可选 `cache_manager.remove_task(room)` 回收源端页；`metadata_buffers.free([room])` 释放 AUX 槽。

- 计算与传输解耦
  - `send_kv_cache()` 只做“分配+入队”，即时返回；真实 RDMA 在后台传输线程执行，避免阻塞 Prefill 计算与后续Batch拼装。

## TP 并行下的传输语义与部署建议

- 控制面约束
  - 仅 TP 主 rank 对外暴露端口并处理 ZMQ/Bootstrap；其他 TP rank 仅参与算子与组内通信，不对外监听端口。
  - Router 与 Decode 仅与 TP 主 rank 完成握手；`dp_id` 仍用于区分多路 Prefill 实例（P0/P1/...）。

- 数据面语义（当前实现）
  - KV/AUX 注册与传输在绑定端口的 rank 上进行：通过 CacheManager 的 `get_contiguous_buf_infos()` 注册该 rank 持有的连续层缓冲区；随后基于 per-request 的“页索引映射”按层搬运。
  - Decode 端在收到 `KVPoll.Success` 后：
    - 读取 batched logits 进入sample；
    - 调用 `insert_kv_cache_from_transfer(rid, page_indices, prefix_length)` 将已搬运的页登记到目的端。
    - 若目标 CacheManager 未提供“预留/插入”能力，或部分 TP 分片未通过 RDMA 搬运，则 Decode 端会执行“fake prefill”以补齐/重建 KV（见 `PDScheduler._execute_decode()`）。

- 使用建议
  - 首次接入 PD 分离：建议 TP=1 验证端到端路径与吞吐收益。
  - 规划演进：每 TP rank 独立注册与传输、主 rank 仅做控制面编排，实现 shard-to-shard 直传以进一步降低 Decode 端重建开销。

## 技术要点
- 控制面/数据面解耦
  - Router（ZMQ 控制面）与 Mooncake（HTTP+RDMA 数据面）职责明确；Bootstrap 由 Router 中心化托管，易运维/可观测。
- 统一 request_id 映射
  - 通过 `_to_uuid` 将任意字符串（时间戳等）稳定映射至 UUID，消除三方不一致的隐患。
- 内存注册延迟与幂等
  - 避免“CacheManager 未就绪/重复注册”导致的 RDMA Bad address/重叠；失败自动熔断降级，保证服务健壮性。
- 边到边解（就绪即解）
  - Prefill 单请求完成即发送；Decode 收到即解，优化首 token 延迟；同时兼容Batch策略以拉高吞吐。
- 显存直达与页粒度调度
  - 以页为单位传输 KV，通过“源页索引→目的页索引”映射实现精准搬运，减少 decode 端重算与拷贝。
- 轻量 warmup（PD 专用）
  - PD 下 warmup 不依赖 Router/Bootstrap/对端，降低启动耦合与复杂度。
- 可观测性
  - 关键阶段有明确日志锚点（prefill/decode 请求处理、注册/传输/状态更新），便于快速定位时序与握手问题。

## 配置与运行要点
- 配置文件：`chitu/config/pd_separation_stage2.yaml`、`pd_separation_example.yaml`
  - `router.pd_separation.bootstrap_port`：Bootstrap 端口（Router 托管）。
  - `ib_device`：RDMA 设备名（可选）。
  - ZMQ 端口：Router 协调/统计、P/D 服务端口。
- 环境变量
  - `PD_MASTER_ADDR=<Router_IP>`（P/D 必配，优先使用）。
  - 同节点多进程端口：可通过 `dp_config.scheduler_base_port` 为每个 `torchrun` 进程指定基准端口（例如 D0/D1/D2 → 29630/29631/29632）。
  - Token 回传：仅 Decode 节点连接 `router_token_port`（默认 29700）。
- 重要接口与函数
  - Router：`PDRequestRouter.add_request()`、`_serialize_original_request()`（深度转纯 dict）
  - P：`KVManager.send_kv_cache()`、prefill 线程接收 decode 注册与 TransferInfo
  - D：`KVManager.recv_kv_cache_and_insert()`、decode 注册/等待/插入
  - Cache：`get_contiguous_buf_infos()`、`get_page_indices()`、`insert_kv_cache_from_transfer()`
  - Bootstrap：`MooncakeBootstrapServer` `/route` 与 `/health`

## 故障现象与排查建议
- P/D 连接 Bootstrap 失败
  - 确认 Router 已启动 Bootstrap；P/D 的 `MASTER_ADDR` 指向 Router IP；端口放通。
- “cache manager does not support get_contiguous_buf_infos”
  - 说明注册时 CacheManager 未注入或未实现；检查 `set_cache_manager()` 调用时序。
- RDMA “Bad address”/“registration failed”
  - 避免过早/重复注册；检查每层 `base_ptr/len` 是否覆盖正确、无重叠；排查设备/驱动与权限。
- Router msgpack 序列化异常（Message 对象）
  - 确保 Router 对 `messages` 做深度 dict 化；不要传 pydantic/自定义对象。
- Decode 一直 WAITING
  - 可能 D 未先注册本地指针给 P；或 per-request TransferInfo 未发出；或 P 未回状态。查看 P/D 对应日志锚点。

## 脚本与示例（2P3D 两节点）

- 脚本：`cinfer-ep/script/srun_pd_disagg_2p3d_two_nodes.sh`
  - Router+3×Decode 在节点0；2×Prefill 在节点1。
  - 自动解析 `SLURM_NODELIST` 并设置：
    - Router 面向的所有 Scheduler host/port；
    - `PD_MASTER_ADDR=<Router_IP>` 注入至所有 P/D 进程；
    - 同节点多进程的 `scheduler_base_port`；
    - 端口就绪探测（`nc -z`）。

运行示例：
```bash
srun -N 2 bash cinfer-ep/script/srun_pd_disagg_2p3d_two_nodes.sh \
  --model /data/nfs/Qwen3-32B \
  --partition <your_partition> \
  --cpus-per-gpu 8
```

## 验证与观测

- 示例（2P3D）：
```bash
python test/verify_xpyd_topology.py \
  --router http://<Router_IP>:21003 \
  --model /your_model_path \
  --concurrency 6 --max_tokens 64 --stream
```
- 判定标准：
  - 所有发现到的 `P{i}` 和 `D{j}` 在日志中均有唯一 `request_id` 命中（去重）；
  - Router 的路由统计包含所有参与的 P 与 D 组合的至少一次出现。

## 相关代码阅读指南

- 配置与脚本：
  - `chitu/config/pd_disagg_2p2d_multi_node.yaml`、`pd_disagg_2p3d_multi_node.yaml`：Prefill/Decode 列表及端口定义。
  - `script/srun_pd_disagg_2p2d_two_nodes.sh`、`srun_pd_disagg_2p3d_two_nodes.sh`：
    - 跨节点 IP 解析（`getent ahostsv4`）、`PD_MASTER_ADDR` 注入；
    - 多进程端口隔离（`dp_config.scheduler_base_port`）；

- Router 侧：`pd_request_router.py`
  - 解析多 P/D；为每个 scheduler 建立 ZMQ；
  - 启动 Bootstrap；
  - 轮询路由 + 日志锚点（`created pd request: ...`）。

- Scheduler/Service：`pd_service.py`、`pd_scheduler.py`
  - `PDSchedulerService` 按模式（prefill_only/decode_only）绑定不同 base port；
  - Decode 路径在 `_process_decode_request` 中将 `prefill_scheduler_id` 注入 `KVManager.set_prefill_target_engine_rank`。
  - Token 回传仅在 Decode-only/Unified 初始化 Token Manager；Prefill-only 明确跳过。

- KV 传输：`kv_manager.py`
  - Decode 侧：
    - 后台发现全部 Prefill engine_ranks 并幂等注册；
    - 按请求精准选取目标 Prefill（避免“P0 只对接 D0”的假设）。
  - Prefill 侧：
    - `TransferKVChunk` 入队；多层并发 RDMA 传输；
    - 传输完成后向 D 回写 `KVPoll.Success`。

## 扩展与演进
- 多 P/D：Router 增加 P/D 组管理与路由选择（负载、就近、亲和），Bootstrap 仍中心化；KVManager 保持不变。
- 可靠性：Bootstrap HA（反代/Keepalived）、ZMQ 重试、状态超时回收。
- 传输优化：异步 RDMA pipeline、跨层Batch传、页对齐聚合策略与自适应切分。

- 参考代码入口：
  - Router：`chitu/pd_request_router.py`、`chitu/serve/api_server.py`
  - P/D 服务：`chitu/pd_service.py`、`chitu/pd_scheduler.py`
  - 传输：`chitu/kv_transfer/kv_manager.py`、`chitu/kv_transfer/mooncake/transfer_engine.py`
  - 缓存：`chitu/cache_manager.py`
  - 启动：`chitu/serve/scheduler.py`、`scripts/start_pd_separation.sh`

---
