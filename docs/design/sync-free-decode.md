# Sync-free decode：让 decode 步的 GPU 不再等 CPU

稳态 decode 时，**让每一步要读的值在发射时就已经在设备上，host 只在下一步的图已经在跑之后
再「得知」上一步的结果** —— 消掉历史实现里每步固定出现的 GPU 气泡。

- 开关：`infer.sync_free_decode`（默认 `false`）—— 关掉时行为与历史实现逐字相同，可以按 `bs` 直接 A/B。
- 覆盖：`pcp = 1`、`mtp_size > 1` 的布局走设备路径（含任意 `tp` / `dp`；`pp > 1` 上是它的**最后一级**，
  见 §2.5）；`pcp > 1`、`mtp_size <= 1`、PD prefill-only，以及读不回 counts 的非 CUDA 后端都不换序
  （§2.5），行为与历史实现一致。
- **来源**：本模块站在 `processors / context` 基座之上（基类把一步拆成处理器列表之后），替代同一特性基于旧
  `Executor` 的早期实现（MR `!2083`）；**基类一行未改**，特性收在一个子类里（§4）。
- **成果**：换序本身带来的 output throughput 提升**视布局而定**，从约 **+1.4%**（GLM-5.2-FP8
  `tp1 / dp16 / ep16 / pp1`，受 §6 那条跨节点 EP 派发地板的限制）到约 **+19%**（GLM-4.7-Flash
  `tp2 / dp1 / ep2 / pp4` 的 bs ≥ 4）。**逐 bs、逐配对的数据与口径在 MR `!2111` 的说明里**，本文只留量级与
  机制。
- **机制证据**（来自一次性的探针构建，探针不在本 MR 里）：`pp = 2` 的最后一级上 **90.8% ~ 96.9%** 的
  decode 步真的走了设备路径，没接上的是 prefill → decode 边界那几步（§2.5）。

一句话：历史实现是「**等结果 → 干活 → 发图**」，现在是「**发图 → 等结果（此时 GPU 已忙）→ 干活**」。

## 0. 名词

| 名词 | 含义 |
|---|---|
| step / 步 | decode 的一次迭代 |
| verify | 一步的模型前向：一次跑 K 个候选 token（`mtp_size = K`），把它们验证掉 |
| draft | MTP 的草稿图：接在上一步的结果之后，猜出 K 个候选 token 与 proposal p' |
| sample | verify 之后的采样：算出接受几个 token、下一步的 token、下一步 draft 的参数 |
| accept counts | 上一步接受了几个 token：`PackedTasksResult.accept_indices`，`(bs,)` 的 device 张量 |
| anchor token | 上一步接受的最后一个 token，也就是下一步 draft 的输入行 |
| mirror（host 镜像） | device 上的值在 host 侧的那份拷贝（长度列表、`task` 字段、sample state） |
| D2H / H2D | device → host / host → device 的拷贝 |
| 气泡（bubble） | GPU 队列为空、没有已发射的工作可做的那段时间 |

## 1. 问题：每步一个气泡

### 1.1 decode 的一步里，host 与 device 互相要什么

一步的骨架（`pp = 1`、`mtp > 1`、schedule-overlap）是四个相位：

- **draft**：读上一步的 accept counts 与 anchor token，发射 draft 图，产生 K 个候选 token
  （跑的是上一步那批请求的列表，见 §2.2）；
- **verify**：把这 K 个候选 token 当作输入跑一次前向，连同 proposal p' 做拒绝采样，接受前若干位；
- **sample**：在 verify 的输出上采样，写出 `PackedTasksResult`（`tokens` / `accept_indices` /
  `next_tokens`），并**异步**启动一次 D2H（`start_sync()`，随后可以用 `finish_sync()` 等它落地）；
- **同步**：等那次 D2H 落地，把这些结果写进 host —— `task.mtp_accept_index`、`task.next_tokens`、
  `tid_to_cached_len`、sample state、请求级结果（BatchResult）、DP 上的结果交换。

也就是说：**上一步的结果先在 device 上算出来（sample 图），host 再把它抄回自己这边**，因为
调度器、grammar、请求答复这些逻辑都只认 host 数据。

### 1.2 气泡是怎么来的

历史顺序（进程队列 = `[同步, draft, model_run]`，`model_run` 里依次是 verify 与 sample）：

```
host :  ├── 等上一步的 D2H ──┤├ draft 的 host 准备 ├┤ verify 的 host 准备 ├┤ sample 前的 host 活 ├
GPU  :  │   （没有已发射的工作） ││     draft 图     ││      verify 图      ││    sample 图    │
```

- **等待本身不是气泡**：它等的就是上一步的 kernel 结果，那是真实计算。
- **气泡是等待返回之后的那段 host 时间**：拼 host 列表、算 accept indices、`create_tensor()` 做 H2D、
  `prepare_cache_decode()` 一整套 python —— 这期间 GPU 队列是空的。它每步固定出现、基本不随 batch 变，
  所以 batch 越大（GPU 每步越久）占比越小：这是收益随 bs 上升的原因。

### 1.3 目标

把「host 必须先知道上一步的结果」从**发射之前**挪到**发射之后**：

1. 一步要读的值（accept counts / anchor token / verify 的输入行 / KV 窗口基准长度）改成**从 device 直接取**；
2. host 的等待与记账**保留**，但排在「本步的 draft / verify 已经发射」之后，于是被它们盖住；
3. host 侧的镜像依然精确（调度器与回退路径还在读它），只是写得晚一点。

## 2. 设计

### 2.1 开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `infer.sync_free_decode` | `false` | 总开关：一步的 decode 状态（KV 长度推进、MTP accept 计数）留在 device 上原地更新、host 镜像保持精确，并把 draft 的输入设备化，让 host 的等待落在本步的图之后而不是让 GPU 空等。要求 `mtp_size > 1`、且布局被支持（§2.5）。 |

队列换不换序（`draft → verify(+sample) → 同步`）由总开关与布局决定：设备路径一接上
（`pcp = 1`、`mtp_size > 1`；`pp > 1` 上是它的最后一级，见 §2.5），draft 的 counts / anchor / 起点页都从
device 上取，host 的等待落在本步的图之后（§3）。

打开后，同一个步变成：

```
step i:
  draft（喂上一步的结果）      verify(i) + sample(i)            同步（收上一步的结果）
  读 device 的 counts/anchor → 读 device 的 counts/输入行 →  等 D2H、补 host 镜像、DP 交换
GPU :   │    draft 图    │      │         verify 图        │      │ sample 图 │ ← host 的等待落在这里
```

### 2.2 换序凭什么成立：要读的 host 值全部设备化

| verify / draft 需要的值 | 历史来源 | sync-free 来源 |
|---|---|---|
| accept counts | 同步相位写好的 `task.mtp_accept_index`，再 H2D 回 device | 上一步 sample 图写在 device 上的 `PackedTasksResult.accept_indices` |
| verify 的输入 token（K 宽行） | `task.next_tokens`（host list）拼回 device | draft 自己那份 device 行，整批一份（`DraftBatch`） |
| draft 的 anchor token | `accepted_tokens`（host list）再 H2D | 从采样行里按 counts gather（`tokens.gather(1, accept.clamp(...))`） |
| draft 的窗口基准长度 | 同步相位修正过的 `tid_to_cached_len`（host list） | cache 自己的 device 长度，由 device 上的 counts 推进 |
| draft 链起点的 page id | `update_mtp_cache_accept()` 里按 host 的 `tid_to_accept_index` / `tid_to_cached_len` 重算 `_read_page_ids` | 上一步的 `_write_page_ids` 的第 `max(accept, 0)` 列，device 上 gather（窗口里落在 ckpt 位置的那一列，写页表里本来就是 ckpt page，与 host 版同值） |
| verify 的窗口基准长度 | 同步相位修正过的 `tid_to_cached_len`（host list） | 同一份 host list；退休列表与本步是同一份时，由 draft 相位读回的 counts 提前修正（§2.4），否则（`pp > 1` 的轮转，§2.5）直接用同步相位抄好的 host 值 |
| GPU block table | 每步按 host 算出来的位置重新 stage | 本步没有请求跨块边界时直接跳过（值比较短路） |
| draft / verify 的任务列表 | draft 用 `TaskCollector.get_postprocess_tasks()`（**上一步**的列表），verify 用本步的列表；批变化时两者不一致，行按 task id 从 `task.next_tokens` 取 | 同样的形状：draft 仍读那份列表，但它的行来自 device —— 整批一份 `DraftBatch`（device 行 + p'）给同名同序的批直接用，另按 task id 记一份（`state.next_tokens_device` / `state.draft_probs`）给批变了的情况。`_sync_free_prev_batch()` 只判退休的那份列表本身可读，不要求它与本步逐 id 相同（§2.5） |

这些值全部由 sample 图算出来，所以 host 看到它们之前它们就已经在 device 上了 —— 这正是可以先发射的原因。
唯一要回 host 的是 accept counts 本身（§2.4 的一次读回）：`tid_to_cached_len`、page id、sampler state
这些 host 账目都只认它 —— 而 draft 要在 counts 到 host 之前就用上那份 page id，所以它不在这本账里，
而是按 `_read_page_ids` 自己的定义（「上一 step 的写页第 accept_index 列」，见
`SingletonPagedKVCache._read_page_ids` 的定义处）直接从 device 上算（表中「draft 链起点的 page id」一行）。

**draft 读哪一份列表。** draft 要读的 anchor 与 accept counts 描述的是「上一步接受了什么」，而它把这些
token 交给的是**下一次 verify**。历史上 draft 读的就是 `TaskCollector.get_postprocess_tasks()`，也就是
同步相位正要退休的那份列表，verify 读的是本步那份。本特性保留这个形状，只把行设备化：draft 仍读退休的
那份列表（`_sync_free_prev_batch()`），但**不再要求它与本步的列表逐 id 相同** —— `pp > 1` 下调度器把
decode 轮转成 `pp_size` 组，退休的那份与本步常是两组不同的请求，逐 id 比只会让设备路径永远接不上
（§2.5）。

行的两本账覆盖这两种情形：整批一份 `DraftBatch`（device 行 + p'，带一份 `task_ids` 快照）给**同名同序
的批**直接用（`_rebuild_mtp_draft_inputs()` 的 batch 分支与 `_get_draft_tokens_device()` 都先试它）；
另一份按 task id 记（`state.next_tokens_device` / `state.draft_probs`），批变了、或两个批不同名 / 不同序
时由 verify 按 task 取自己那一行。这份逐 task 的账也正好是 PP 要的形状：draft 与 sampler 只在最后一级，
本步的列表在别的级上还不完整。

「draft 的行也在 host 上一次」这件事仍然要交代：host 侧的消费者（grammar bitmask、frequency penalty）
确实需要 token 行本身，所以整批行照历史路径挂回上一批的结果（`generated_result.next_tokens` +
`record_sync_event()`）。那次等待排在本步 draft 图之后，到时已被 verify 盖住（§2.4 的同一套手法）。

### 2.3 只有采样 rank 有这些值：一步一次的设备张量广播

只有采样 rank（`tp0` + `pcp0` + 最后一个 pp stage）跑 sampler，但 draft 与 verify 在组内所有 rank 上都要跑、
都要读 counts 与 anchor。做法：draft 相位一开始，采样 rank 把 `(counts, anchor)` 拼成一条 `2 * bs` 的
int64 张量，走**本步自己的那组广播**（`_broadcast_tensor_payload`，NCCL 集合）发给组内所有 rank。
其余 rank 拿到的是同一份 device 值，于是「把等待挪到后面」在整组里同时发生。

非采样 rank 的 host 镜像由它自己那份同步相位补：它没有 sampler，counts 正好来自这次广播，
两者的值一致，所以组内每一步的账都一样。

### 2.4 host 的账：counts 读回一次，其余照历史路径走

设备路径只把「一步要读的值」设备化；host 侧的账目仍然精确，靠的是**一次读回**：

1. draft 相位一开始，采样 rank 把本步的 accept counts **非阻塞地拷进一块 pinned 缓冲**，紧接着在同一个
   stream 上 `record_event()`。这次拷贝排在本步 draft 图**之前**，所以它等的只是上一步 verify / sample
   的尾巴（那段等待本来就有）；
2. verify 的准备（`_prepare_cache_decode()`）里才 `event.synchronize()` —— 那时 draft 图已经发出去，
   等待被它盖住；
3. 拿到 counts 之后，用**基类自己的** `KVCache.update_mtp_cache_accept()` 修正 `tid_to_cached_len`，
   然后 `super()._prepare_cache_decode()` 照常跑：窗口、page id、sampler state 全部是历史路径写出来的
   （page id 在 draft 之前已经按 device 值写过一遍，两处同值）。

**读回之后**，host 侧没有延后的镜像：`tid_to_cached_len` 就是历史实现里那个值，`prepare_cache_decode()` 的
host 路径就是正确的路径 —— 这正是这一步不需要在 KV cache 里开设备分支的原因。**读回之前**（也就是 draft
自己发射的那段窗口内）host 镜像是落后的，见下一段：说「host 镜像精确」只在读回之后成立。

唯一「device 先写、host 暂时不知道」的是 **draft 自己的那个窗口**：draft 在 counts 到 host 之前就发射了，
它的 `mtp_seq_len_delta` 由 cache 从 device 长度直接写（`SyncFreeDecodeState.prepare_mtp_cache_decode()`，
`BatchedSeqLenDelta.set_lens_device()`），host 镜像在这段时间里是**毒化**的（读到就 raise，而不是返回上一步的
值）；verify 的窗口（上面第 3 步的 host 路径）把它重写一遍之后才恢复可读。这就是 §5.1 说的「按设计慢一个
窗口」，两处指的是同一个窗口、同一个时刻。

### 2.5 回退：什么时候这一步走回历史顺序

只有「这一步的输入确实全在 device 上」才换序，否则把同步相位拉回 draft 之前（`_sync_free_ensure_sync_part()`），
这一步与历史顺序等价：

- **布局**：`pcp > 1` 不换序（那些 rank 只看序列的一段，窗口算术是逐段的）；`mtp_size <= 1` 同理（没有
  draft）；**非 CUDA 的 device**（`op_impl = cpu`，或任何 `is_cuda` 为假的后端）也不换序 —— 设备路径的
  counts 读回要求 CUDA 张量，读不回来的话只有基类那两处会调 `update_mtp_cache_accept()`，而设备路径会把
  长度照推，两边会越走越远（`device_path_layout_supported()` 用一次真实张量的 `is_cuda` 判据，与读回同一
  条件）；**PD prefill-only 节点**同样不换序：它不采样，基类的 `postprocess_generate_draft()` 在那里早退，
  accept counts 没有任何人应用。`pp > 1` 换序，但只在它的**最后一级**：draft 与 sampler 都只在那里，它前面
  的那些级要的 accept counts 是流水线原样回传的（最后一级采样 → 直接发回第一级 → 再沿级向前 ZMQ 转派），
  而且它们手上没有「本步的图已经发出去」这件事 —— 没有 draft 可以遮住等 counts 的那段 host 时间 ——
  所以照旧跑历史顺序，不换序。
- **退休列表读不了**（不是 decode 步 / 行数对不上 / 没有 device 结果）：`_sync_free_prev_batch()` 判定
  这一步不走设备路径，draft 与 verify 都保持历史顺序。判据只看**退休的那份列表**，不要求它与本步的列表
  逐 id 相同 —— `pp > 1` 下调度器把 decode 轮转成 `pp_size` 组（`Scheduler` 的 `num_scheduler_groups`），
  相邻两步本来就是**不相交**的两组任务，逐 id 比只会让设备路径永远接不上。
- **verify 的 counts 不是本步那份列表的**：设备路径那一步读回的是**退休列表**的 counts（`SyncFreeStep.tasks`
  就是它），它们进的是那份列表的 `tid_to_cached_len` 账；verify 要的 `Backend.model.mtp_accept_indices`
  另走 `_sync_free_counts_for()` —— 只有设备上的 counts 就是为**本步这份列表**取的（`SyncFreeStep.batch`
  逐 id 相同）才用它，否则退回 host 镜像。`pp > 1` 下后者是常态，而且退得安全：本步的 counts 是上一步的
  同步相位抄到 host 上的，来自同一组任务上一次的 verify，是新鲜的。
- **首步与 prefill→decode 边界**：没有上一步的 device 结果可用。
- **draft 的起点页取不到**：`_read_page_ids` 要从上一 step 的写页表与这一步的行数对上才成立
  （`_sync_free_point_read_pages()`，正常布局下必然对上）。对不上就把这一步的输入整体作废
  （context 上不留 `SyncFreeStep`），不给 draft 读一个说不准的页。
- **组内必须同答**：闸门选中的是**整组**的广播（设备路径走 NCCL 集合，host 路径走 ZMQ 主发余收），
  所以判据只用所有 rank 都有的字段（`PackedTasksBase` 上的 `task_ids` / `output_task_ids`）。
  不允许「采样 rank 说走设备路径、follower 说走 host 路径」—— 那会让两边分别等在 NCCL 与 ZMQ 上。
- **DP**：换序把同步相位整个挪到 sample 之后，所以回退不能把它整个拉回 draft 之前 —— 那样会把这
  「与 DP peer 的一次交换」挪到对方 MoE dispatch 的前面。回退只跑同步相位的 **host 半边**
  （`_sync_free_ensure_sync_part()`：`finish_sync()` + `batch_update_mtp_accept_index()`），
  交换与请求级结果留在队列给同步相位的位置上。两个 DP rank 跑不同的批，是否拉 host 半边可以不同。

## 3. 一个稳态 decode 步：历史 vs sync-free

（下表右列是 sync-free 的设备路径 —— 换序后的队列。）

| 相位 | 历史实现 | sync-free |
|---|---|---|
| 等待上一步结果 | 队列第一个相位，阻塞 | 队列最后一个相位；draft 要读的 counts 由一次读回补齐，那次等待落在 draft 图之后 |
| accept counts | 同步相位写 `task.mtp_accept_index` → 下一步 H2D | 直接用 device 上的 `accept_indices`（采样 rank 广播给全组），同时读回 host 一次 |
| draft 的输入 | `accepted_tokens`（host list）→ H2D | device 上 gather 出来的 anchor 行 |
| draft 窗口 | host 长度列表构造 | device 长度构造，host 镜像毒化到 verify 窗口写入为止 |
| draft 链起点的 page id | `update_mtp_cache_accept()` 按 host 账目重算 `_read_page_ids`，排在 draft 之前 | `share_inputs` 里按 device counts 从上一 step 的写页表 gather，同样排在 draft 之前 |
| verify 的输入行 | `task.next_tokens`（host list）→ H2D | draft 的 device 行（整批一份） |
| draft / verify 的任务列表 | draft 用上一步的列表、verify 用本步的（两批请求，批变化时不一致） | draft 仍用上一步的列表，但不要求它与本步逐 id 相同（§2.2、§2.5）；行整批一份在 device 上，另按 task 记一份 |
| verify 窗口 | host 长度列表构造 | 同一份 host 列表构造，但 counts 已经提前读回并修正过 |
| `tid_to_cached_len` / MTP cache 位置 | 同步相位就地更新 | verify 的准备里就地更新（基类的 `update_mtp_cache_accept()`，counts 已读回） |
| 每步 host 阻塞等待 | 2 次（同步相位、`update_sampler` 里的 `result.sync()`） | 仍然 2 次，但一次落在 draft 图之后、一次落在 verify 图之后 |
| DP 结果交换 | 同步相位里 | 同步相位里（位置不变，只是紧跟 host 半边） |
| 图要读的值 | 部分来自 host 列表 / H2D | 全部已在 device 上 |

要点：**host 的工作量没有消失**（它必须知道结果才能答复请求、填 grammar bitmask、更新调度状态），
消失的是「GPU 在这些工作期间无事可做」。

## 4. 代码地图

| 文件 | 职责 |
|---|---|
| `chitu/sync_free_decode.py` | 本特性的全部代码：开关（`is_sync_free_decode_enabled()` / `device_path_layout_supported()`）、每步状态 `SyncFreeStep`、设备路径本体 `SyncFreeDecodeMixin`，以及按开关选类的 `build_executor(args)` 与 `SyncFreeDecodeExecutor`（只放**跨步**的东西：重排的判据、pinned 缓冲） |
| `chitu/kv_cache/kv_cache_sync_free.py` | `SyncFreeDecodeState`：draft 窗口的 device 长度、draft 起点页的 device 推导、GPU block table 的短路 |
| `chitu/executor.py` | 基类：相位队列与各相位；**本特性一行未改**（全部靠继承与覆盖） |
| `chitu/kv_cache/kv_cache.py` | 三个调用点：`prepare_mtp_cache_decode()` 与 `_upd_gpu_block_table()` 各加一个「设备路径要不要接管」的判断，`_upd_read_page_ids()` 在起点页已由 device 推过时短路 |
| `chitu/sampling/sampler.py` | `DraftBatch`：一步的 draft 行（device 行 + p'）整批一份，verify 在原地读它；另按 task 记一份，供「批变了」的回退用 |
| `chitu/batched_seq_len.py` | 设备侧长度推进（`set_lens_device()` / `update_host_mirror()`），host 镜像的毒化与恢复 |

新基座把一步写成「一串 processor + 一个 `ProcessContext`」，设备路径就按这个形状接进去：**子类自己排自己的
processor 列表，也自己写里面需要的 processor**，基类不开任何缝。

| 方法 | 默认（历史）行为 | sync-free 行为 |
|---|---|---|
| `_build_process_queues`（覆盖） | `overlap_prefix + decode_body + flush_suffix`：会 draft 的 rank 上是 `[同步, draft] + 模型体`，第一级再用 `[flush]` 收尾（`pp = 1` 时同一个 rank 既第一级也最后一级） | 设备路径接上（`pcp = 1`、`mtp_size > 1`；`pp > 1` 上是它的最后一级）就换成 `[share_inputs, draft] + 模型体 + [同步]`（§3 的相位顺序），第一级仍用 `[flush]` 收尾；其余布局原样不动 |
| `_sync_free_generate_draft`（本特性自己的 processor，替掉基类的 `postprocess_generate_draft`） | 跑**上一步**的列表（`TaskCollector.get_postprocess_tasks()`），输入行从 `generated_result.accepted_tokens`（host list）拼出来再 H2D，counts 从 `task.mtp_accept_index` 取 | 同一个 processor、同一份列表，但两个输入换成 `share_inputs` 分享在设备上的行；行整批落在一份 `DraftBatch` 上，verify 直接读它 |
| `_prepare_accept_indices`（覆盖） | `task.mtp_accept_index`（host）→ H2D → 组内广播 → 写 caches | 直接用本步的 device counts（`share_inputs` 已广播）；cache 的长度账留给 `_prepare_cache_decode`，起点页账由 `share_inputs` 按 device 值先做完 |
| `_get_draft_tokens_device`（覆盖） | 逐 task 取 `state.next_tokens_device` 再 `stack` | 整批一份的 `DraftBatch` 行（另有一份逐 task 的账，供「批变了」的回退用） |
| `_prepare_cache_decode`（覆盖） | 直接走基类 | 先把读回的 counts 用 `update_mtp_cache_accept()` 应用，再 `super()` |

每步的状态放在 context 里而不是 executor 上 —— context 随步生灭，所以不存在「读到上一步输入」这种状态。
它挂成一个属性（`_STEP_ATTR`）而不是 `ProcessContext` 的字段：后者等于在基类里开缝。context 上没有它，
就是「本步不走设备路径」，每个读者的回退都只判这一件事。

`SyncFreeDecodeMixin` 自己剩下的部分：`_sync_free_share_inputs()`（本步的输入 processor —— 取 device 输入、
一次组内广播、把 counts 交给 caches、按 device counts 把 draft 起点页推好
（`_sync_free_point_read_pages()` → `SyncFreeDecodeState.set_read_page_ids_device()`）、排读回，
并把 `SyncFreeStep` 放进本步的 context）、
`_sync_free_local_inputs()`（采样 rank 自己那份）、
`_sync_free_queue_counts_readback()` / `_sync_free_take_counts()`（pinned 读回与等待）、
`_sync_free_prev_batch()` / `_sync_free_inputs_for()` / `_sync_free_counts_for()`（这一步能不能走设备路径）、
`_sync_free_ensure_sync_part()`（走不了时的回退）。

## 5. 覆盖与边界

**覆盖**：`pcp = 1`（任意 `tp` / `dp`）、`mtp_size > 1` 的稳态 decode 步走设备路径；`pp > 1` 上是它的
最后一级，它前面的那些级跑的是历史排法（§2.5）。

**不在范围内**：

- `pp > 1` 上最后一级之前的那些级：它们要的 accept counts 由流水线原样回传，手上也没有 draft 能遮住
  「等 counts」的那段 host 时间，所以照旧跑历史排法。把设备路径铺到它们身上（例如让最后一级把 counts
  沿 PP 组广播一次、那些级各自非阻塞读回）是可选扩展，不在本特性范围内；
- `pcp > 1` 的设备路径：换序与设备输入都还没接（§2.5）；
- 每完成一个请求的 drain / 空步 flush、以及 MTP draft 窗口本身的固定开销（约 2.0 ms/步，不随 bs 缩），
  属于另外两条线，与本文的换序无关；
- grammar（xgrammar bitmask）与 frequency penalty 仍然是 **host 逻辑**，它们读的 draft 行由 host 镜像提供；
  这些 host 工作并没有被消除，只是被排到了已发射的图之后（§1.3）。

### 5.1 记账必须排在同步之后：与上游 draft 窗口 guard 的交互

上游 `PagedKVCache.prepare_mtp_cache_decode()` 在 `draft_offset == 1` 时检查 draft 要写的窗口是否落在
请求已分配的页里（`anchor + mtp_size - 2 < allocated_tokens`，`anchor` 取自 host 镜像
`tid_to_cached_len`）。这条检查是上游 `!2115` 为"draft 越界写坏 MTP 的 K/V"加的。设备路径上这份镜像
**按设计慢一个窗口**（§2.2，`accepted_lens_device_for_draft()` 存在的理由就是它），本来就贴着这条检查的
边界，于是**任何让预留或调度状态晚一步的排布都会顶穿它**。换序最初把 `_commit_step_tokens` 留在模型体里、
排在同一步队尾的同步相位**之前**，正好制造了两件事：

1. **`Task.alloc_seq_len` 少一个窗口**：`has_unsync_new_token` 只由 `_commit_step_tokens()` 置位，而清除
   它的 `batch_update_response_sync()` 在同步相位里；排在同步之前 ⇒ 标志刚置上就被清掉 ⇒ 该步之后调度器
   看到的 `num_unsync_tokens` 是 `0`，预留比历史顺序少 `mtp_size - 1 - accept` 个 token。实测（GLM-4.7
   `tp4/dp2`、bs = 64）：guard 读到 `anchor = 191`、`allocated_tokens = 192`，而那一刻真实接受长度是 **189**
   —— draft 写 `[189, 190]`、verify 写 `[189, 192)` 本来都装得下，差额只有镜像多读的那 2 个 token。
2. **`update_decode_status()` 看不到本步的 accept**：它只在 `_commit_step_tokens()` 里刷新，排在同步之前
   ⇒ 判定"这个请求是否已到 `max_seq_len`"用的是旧长度 ⇒ **已经到上限的请求仍被排进下一步**，而设备路径
   按设备长度取窗口 ⇒ draft 写到上限之外。

第 2 条会真的写坏显存，且**不能靠加大分配解决**：同一配置（`max_seq_len = 512`、`out = 512`、bs = 8、开关打开）
实测，把预留加大一个 `mtp_size` 的提前量只让上面那条 host 检查沉默，**越界照旧发生 227 次**
（`IndexKernel` 的 `index out of bounds`）；缓存可寻址上界由 `max_alloc_seq_len(max_seq_len)` 与 RoPE 表钉死，
分配再多也盖不住写到上限之外的位置。

**修法**：在 `_sync_free_decode_queue()` 里把 `_commit_step_tokens` 放在队尾的 `postprocess_sync_part`
**之后**（其余换序不变）。这样记账看到的是本步刚刚同步上来的 accept，两个后果同时消失，也不需要绕过上游那条
guard。实测同一配置：`rc = 0`、guard **0** 次、device 越界 **0** 次（开/关两臂吞吐 516.5 / 982.6 tps 与
490.3 / 840.1 tps）。

**回归口径**（目前唯一能稳定把请求推到上限、从而覆盖这一类的跑法）：`infer.max_seq_len = 512` +
`benchmark.output_len = 512` + `bs = 8` + 开关打开，单节点约 1 分钟；要求 `MTP draft window ... not covered`
与 `index out of bounds` 都是 0 次。

**`full_warmup` 的现状**：改序之前，`infer.full_warmup = auto`（上游完整 warmup）会在 **warmup 路径**撞到
同一个 draft 窗口判定 —— **连关着开关的 base 臂也一样**，所以当时的基准都要用 `full_warmup = False` 绕开。
记账相位排到同步之后，这条也一并消失：`full_warmup = True` 下开/关两臂各跑一次 decode，`rc = 0`、
guard 0 次、越界 0 次（GLM-4.7-Flash `tp1 / dp8 / ep8 / pp1`，1 节点）。现状基准因此可以按默认的
`full_warmup = auto` 跑。

## 6. 成果与机制（数据明细见 MR 说明）

本节只留结论量级与机制；**逐 bs、逐配对的数字、口径与限制都在 MR `!2111` 的说明里**，设计本身不依赖它们。

**口径**：同一棵树「开关关 / 开」两臂，**同一次 srun 内交错**并按相同轮序号配对；`in = 128 / out = 512`、
bs = 1 ~ 32、每 bs **11 轮取后 10 轮平均**、`infer.full_warmup = True`。

**量级**：换序带来的 output throughput 提升视布局而定 —— GLM-4.7-Flash `tp2 / dp1 / ep2 / pp4` 在 bs ≥ 4 上
约 **+17% ~ +19%**、`tp1 / dp8 / ep8 / pp1` 全程约 **+8%**；GLM-5.2-FP8 `tp8 / dp1 / ep1 / pp2` 约 **+5%**、
`tp1 / dp16 / ep16 / pp1` 约 **+1.4%**（受下一条的机制限制）。小 bs（1、2）上各布局的收益在 0 ~ +8% 之间，
且**必须靠多轮平均**才能与噪声分开：这里的噪声底由「同一臂十轮之间的散布」给出（几个百分点的量级），
而**逐臂取迭代最大值不是降噪手段** —— 那是「每臂各挑自己最幸运的一次」，只会抬高最吵的格子。

**机制**：消掉的不是「等待」，而是**等待返回之后的那段 host 记账**（§1.3、§2.4）。把它挪到本步的图已经
发射之后，它就被 GPU 的执行盖住。两条一次性探针读数支持这一点（探针不在本 MR 里）：`pp = 2` 的最后一级上
**90.8% ~ 96.9%** 的 decode 步真的走了设备路径；在 verify 发射前烧 1 ms 只让步周期涨 0.42 ~ 1.97 ms
（该处余量 **0.58 ~ 1.03 ms**），说明这段 host 工作本来就落在 GPU 的空档里。

**为什么 `tp1 / dp16 / ep16 / pp1` 只能拿到约 1.5%**：这条布局的步周期里约 **46.8 ms/步与批无关**
（bs = 32 时占 73%）—— `bs = 1` 时整个作业只有 1 行、15/16 个 rank 空跑，一步仍要 41.8 ms，约
0.54 ms × 78 层，是 DeepEP-LL 每层一次跨节点 all-to-all 的延迟链。而可挪走的 host 记账是**每步固定量**，
`dpep` 每个 rank 每步只有 `bs / 16` 行却摊在这条比 GLM-4.7 长 4.2× 的步里 ⇒ 同等行数上的收益被稀释约
4.2× ⇒ **天花板约 1.5% ~ 2%**（实测与之一致）。要在这条布局上拿数量级改善，杠杆是那 46.8 ms 的 EP 派发
地板（让每层 all-to-all 与计算重叠、或减少 EP 轮次），不是本特性。

**复现性口径**：这类读数只认同一次 srun 内交错的两臂（或 ABBA 次序下方向相反的两对）—— 跨 srun / 跨机器的
同配置绝对值实测可差 10% ~ 27%，所以单点 Δ 不作为结论；`iters ≥ 11` 且取稳态轮次的均值，是本特性在这套
harness 上的分辨率下限（配对的 Δ 的标准误可到 0.2 pp 量级）。

### 6.1 与换序无关的四处 host 减负

这条历史路径上本 MR 另有四处 host 减负。**它们不判开关，关掉开关也在生效**，所以不属于「sync-free 的收益」，
而是顺带修掉的 host 侧浪费（逐条的 `文件:行号`、为什么属于本 MR、怎么关掉，都在 MR 说明里登记）：

| 位置 | 省掉的东西 |
|---|---|
| `kv_cache.py::_upd_gpu_block_table()` | 行内容与上一步相同（这一步没有请求跨 block 边界）就整段短路：跳过后面的 `set_shape()` 与逐行 diff + 拷贝 |
| `sampler.py::prepare_draft_sample_params()` | 逐行 `(temperature, top_k, top_p)` 签名不变就复用上一步的 device 张量，跳过 H2D 与 pin 分配 |
| `prometheus_collector.py` | 「有请求结束」的步都要重建的 gauge（`update_GPU_usage()` 要问一次 NVML 设备进程表）改成按抓取周期重建一次；`CHITU_METRICS_REFRESH_INTERVAL_S = 0` 恢复旧行为 |
| `kv_cache.py::_upd_read_page_ids()` / `_update_write_page_ids()` | 页表上传从 pageable `torch.tensor()` 改成 pinned + `non_blocking` 的 `create_tensor()` |

前三处的判据都是**行为等价**（相等则跳过），不是策略选择；只有 gauge 那条改变了 `/metrics` 的刷新语义，
所以它带一个 env 回退档。

## 7. 怎么开

配置项在 `chitu/config/serve_config.yaml`，也可以在任何一次启动里用命令行覆盖
（`infer.sync_free_decode=true`）：

```yaml
infer:
  sync_free_decode: true   # 打开本特性：状态留在设备侧 + draft 的输入设备化
```

`sync_free_decode` 关掉时，`build_executor()` 构造的是历史 `Executor`，本特性涉及的代码一行都不会跑到。
