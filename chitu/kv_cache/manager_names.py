# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache 及其 manager 的名字（``manager_name``）的出处。

``manager_name`` 是 cache、manager、scheduler 与 PD 传输之间的契约：scheduler 按 manager 名
遍历 ``cache_manager_dict``，把本 step 分配到的 block ids 记进 ``task.new_cache_ids[name]``
（``scheduler._prepare_prefill_metadata`` / ``_prepare_decode_metadata``）；每个 cache 再用
``task.new_cache_ids[cache.manager_name]`` 取自己那一份（``kv_cache.py``；PD 传输见
``kv_transfer/decode.py``）。因此：

* **两个 cache 共用 manager，就是它们的 ``manager_name`` 相同**，没有别的开关；
* 但 manager 的 ``num_blocks`` / ``block_size``(=checkpoint_interval) /
  ``max_blocks_per_req`` 都是从它服务的那个 cache 的 layout 推出来的
  （见 ``SingletonPagedKVCacheManager``），所以共用还要求两边 layout 一致 —— 例如
  prefill 侧带 checkpoint 的 linear cache 与 MTP cache 就不能共用。

下属四个名字同时是 ``cache_dict`` 的 key 和该 cache 的 ``manager_name``：

* ``MAIN_CACHE_NAME``：main KV cache（full attention 层）及其 manager；
* ``LINEAR_CACHE_NAME``：linear attention 的 recurring state（``SingletonPagedKVCache``），
  prefill 侧带 checkpoint（``checkpoint_interval``）；
* ``MTP_CACHE_NAME``：MTP 每个 draft step 的 hidden state（``SingletonPagedKVCache``，
  ``checkpoint_interval`` 为 None）；
* ``INDEXER_CACHE_NAME``：GLM-5.2 / DeepSeek-V3 的 indexer state；block size 与 main 相同时
  它复用 ``MAIN_CACHE_NAME`` 的 manager（见 ``builders._attach_indexer_cache_managers``）。

模型自己生成的名字不在此列，例如 DeepSeek-V4 的 ``compressed_csa`` / ``compressed_hca``
与 ``main_csa`` / ``main_hca``（见 ``providers/deepseek_v4.py``）。
"""

MAIN_CACHE_NAME = "main"
LINEAR_CACHE_NAME = "linear"
MTP_CACHE_NAME = "mtp"
INDEXER_CACHE_NAME = "indexer"
#: 只有 cache_dict key，没有 manager（MMPagedKVCache 自己管显存）
MULTIMODAL_CACHE_NAME = "multimodal"

#: PD 传输支持的 cache（白名单，见 ``pd_disaggregation/pd_service.py``）
SUPPORTED_CACHE_NAMES = frozenset(
    {MAIN_CACHE_NAME, LINEAR_CACHE_NAME, MTP_CACHE_NAME, INDEXER_CACHE_NAME}
)
