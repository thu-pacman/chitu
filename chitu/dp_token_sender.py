# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
数据并行 Token 发送器
负责将 DP 组生成的 token 发送回 Token Router
"""

import logging
import time
from typing import Any, Dict, List, Optional
from chitu.task import Task

logger = logging.getLogger(__name__)


class DPTokenSender:
    """DP模式下的Token发送器，负责发送token到Router"""

    def __init__(
        self, router_address: str = "tcp://localhost:29700", dp_group_id: int = 0
    ):
        self.router_address = router_address
        self.dp_group_id = dp_group_id
        self.socket = None
        self.context = None

        self.instance_id = id(self)
        logger.info(
            f"DPTokenSender created: group={dp_group_id}, instance_id={self.instance_id}, router={router_address}"
        )

        self.request_token_cache: Dict[str, List[int]] = {}

    async def start(self):
        """启动Token发送器"""
        await self._init_socket()
        logger.info(f"DPTokenSender started for group {self.dp_group_id}")

    async def _init_socket(self):
        """初始化ZMQ socket"""
        import zmq.asyncio

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(self.router_address)

    async def send_token(
        self,
        request_id: str,
        token: int,
        top_logprobs: Optional[List[float]] = None,
        top_token_idx: Optional[List[int]] = None,
        task=None,  # Add task parameter to get prompt_len
    ):
        """发送单个token到Router"""
        try:
            from chitu.backend import Backend

            # 检查是否是第一个token，并获取prompt_len
            is_first_token = request_id not in getattr(self, "_first_token_sent", set())
            if not hasattr(self, "_first_token_sent"):
                self._first_token_sent = set()

            if is_first_token:
                self._first_token_sent.add(request_id)

            logger.debug(
                f"DP Token Sender: [request {request_id}] prepare to send token {token}, is_first_token={is_first_token}"
            )

            # 解码token为文本
            text = ""
            if Backend.tokenizer is not None:
                try:
                    text = Backend.tokenizer.decode([token])
                    logger.debug(
                        f"DP Token Sender: [request {request_id}] decode token {token} -> '{text}'"
                    )
                except Exception as decode_error:
                    logger.error(
                        f"DP Token Sender: [request {request_id}] token decode failed: {decode_error}"
                    )
                    text = f"[DECODE_ERROR_{token}]"
            else:
                logger.warning(
                    f"DP Token Sender: [request {request_id}] tokenizer not available, cannot decode token {token}"
                )
                text = f"[TOKEN_{token}]"

            # 解码top_tokens（如果存在）
            top_tokens_text = None
            if top_token_idx is not None and Backend.tokenizer is not None:
                try:
                    top_tokens_text = [
                        Backend.tokenizer.decode([idx]) for idx in top_token_idx
                    ]
                    logger.info(
                        f"DP Token Sender: [request {request_id}] decode top_tokens {top_token_idx} -> {top_tokens_text}"
                    )
                except Exception as decode_error:
                    logger.error(
                        f"DP Token Sender: [request {request_id}] top_tokens decode failed: {decode_error}"
                    )
                    top_tokens_text = [f"[DECODE_ERROR_{idx}]" for idx in top_token_idx]

            data = {
                "type": "token",
                "request_id": request_id,
                "text": text,
                "original_token_id": token,
                "dp_group_id": self.dp_group_id,
                "timestamp": time.time(),
            }

            # 如果是第一个token且有task，包含prompt_len信息
            if (
                is_first_token
                and task is not None
                and hasattr(task, "req")
                and hasattr(task.req, "prompt_len")
            ):
                data["prompt_len"] = task.req.prompt_len
                logger.info(
                    f"DP Token Sender: [request {request_id}] first token, prompt_len={task.req.prompt_len}, token={token}"
                )

            if top_logprobs is not None:
                data["top_logprobs"] = top_logprobs
            if top_tokens_text is not None:
                data["top_tokens_text"] = top_tokens_text

            await self._send_data(data)

        except Exception as e:
            logger.error(
                f"DP Token Sender: [request {request_id}] send token failed: {e}"
            )
            import traceback

            logger.error(
                f"DP Token Sender: [request {request_id}] send token failed details: {traceback.format_exc()}"
            )

    async def send_finish(self, request_id: str, finish_reason: str = "stop"):
        """发送请求完成信号"""
        try:
            data = {
                "type": "finish",
                "request_id": request_id,
                "finish_reason": finish_reason,
                "dp_group_id": self.dp_group_id,
                "timestamp": time.time(),
            }

            await self._send_data(data)

            if request_id in self.request_token_cache:
                cache_size = len(self.request_token_cache[request_id])
                del self.request_token_cache[request_id]

            last_decoded_len_key = f"_last_decoded_len_{request_id}"
            if hasattr(self, last_decoded_len_key):
                delattr(self, last_decoded_len_key)

        except Exception as e:
            logger.error(
                f"DP Token Sender: [request {request_id}] send finish signal failed: {e}"
            )

    async def send_error(self, request_id: str, error_message: str):
        """发送错误信号"""
        try:
            data = {
                "type": "error",
                "request_id": request_id,
                "error": error_message,
                "dp_group_id": self.dp_group_id,
                "timestamp": time.time(),
            }

            await self._send_data(data)

            if request_id in self.request_token_cache:
                cache_size = len(self.request_token_cache[request_id])
                del self.request_token_cache[request_id]

            last_decoded_len_key = f"_last_decoded_len_{request_id}"
            if hasattr(self, last_decoded_len_key):
                delattr(self, last_decoded_len_key)

        except Exception as e:
            logger.error(
                f"DP Token Sender: [request {request_id}] send error signal failed: {e}"
            )

    async def _send_data(self, data: Dict[str, Any]):
        """发送数据到Router"""
        try:
            import msgpack

            packed_data = msgpack.packb(data)
            await self.socket.send(packed_data)
        except Exception as e:
            logger.error(f"DP Token Sender: 数据发送失败: {e}")

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.request_token_cache.clear()
        logger.info(f"DPTokenSender closed for group {self.dp_group_id}")


class DPTaskWrapper:
    """DP Task 包装器，集成 Token Sender 功能"""

    def __init__(self, original_task: Task, token_sender: DPTokenSender):
        self.original_task = original_task
        self.token_sender = token_sender
        self._original_update_response_sync = original_task.update_response_sync

        original_task.update_response_sync = self._dp_update_response_sync

    def _dp_update_response_sync(self, token: int):
        """重写的 update_response_sync 方法，同时发送 token 到 Router"""
        self._original_update_response_sync(token)

        import asyncio

        def send_token_async():
            """在后台异步发送token"""
            try:
                # 创建新的事件循环或使用现有的
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 如果循环正在运行，使用create_task
                        loop.create_task(self._send_token_to_router(token))
                    else:
                        # 如果循环未运行，直接运行
                        loop.run_until_complete(self._send_token_to_router(token))
                except RuntimeError:
                    # 如果没有事件循环，创建新的
                    asyncio.run(self._send_token_to_router(token))
            except Exception as e:
                logger.error(f"[DPTaskWrapper] Token发送失败: {e}")

        # 在线程池中执行异步操作，避免阻塞主线程
        import threading

        token_thread = threading.Thread(target=send_token_async, daemon=True)
        token_thread.start()

    async def _send_token_to_router(self, token: int):
        """异步发送token到Router"""
        try:
            request_id = self.original_task.req.request_id
            top_logprobs = None
            top_token_idx = None

            await self.token_sender.send_token(
                request_id=request_id,
                token=token,
                top_logprobs=top_logprobs,
                top_token_idx=top_token_idx,
                task=self.original_task,  # Pass task for prompt_len info
            )

            logger.debug(f"[DPTaskWrapper] Token发送成功: {request_id} -> {token}")

            if self.original_task.need_remove():
                await self.token_sender.send_finish(
                    request_id=request_id,
                    finish_reason=self.original_task.req.finish_reason or "stop",
                )
                logger.debug(f"[DPTaskWrapper] 任务完成信号发送: {request_id}")

        except Exception as e:
            logger.error(f"[DPTaskWrapper] Token发送过程出错: {e}")
            import traceback

            logger.error(f"[DPTaskWrapper] 错误详情: {traceback.format_exc()}")

    def __getattr__(self, name):
        """代理其他属性到原始 task"""
        return getattr(self.original_task, name)


class DPTokenManager:
    """DP Token 管理器，管理整个 DP 组的 token 发送"""

    def __init__(self, dp_group_id: int, router_address: str = "tcp://localhost:29700"):
        self.dp_group_id = dp_group_id
        self.router_address = router_address
        self.token_sender = DPTokenSender(router_address, dp_group_id)
        self.wrapped_tasks: Dict[str, DPTaskWrapper] = {}

        self.instance_id = id(self)
        logger.info(
            f"DPTokenManager created: group={dp_group_id}, instance_id={self.instance_id}, router={router_address}"
        )

    async def start(self):
        """启动 Token Manager"""
        await self.token_sender.start()
        logger.info(f"DP Token Manager started for group {self.dp_group_id}")

    def wrap_task(self, task: Task) -> DPTaskWrapper:
        """包装 Task 以支持 token 发送"""
        if task.req.request_id in self.wrapped_tasks:
            return self.wrapped_tasks[task.req.request_id]

        wrapped = DPTaskWrapper(task, self.token_sender)
        self.wrapped_tasks[task.req.request_id] = wrapped

        logger.debug(f"Wrapped task {task.req.request_id} for DP token sending")
        return wrapped

    def unwrap_task(self, request_id: str):
        """取消包装 Task"""
        if request_id in self.wrapped_tasks:
            del self.wrapped_tasks[request_id]

    async def send_error_for_request(self, request_id: str, error_message: str):
        """为特定请求发送错误"""
        await self.token_sender.send_error(request_id, error_message)

    def close(self):
        """关闭管理器"""
        self.token_sender.close()


# 全局实例（每个 DP 组一个）
_dp_token_managers: Dict[int, DPTokenManager] = {}


def get_dp_token_manager(
    dp_group_id: int, router_address: str = "tcp://localhost:29700"
) -> DPTokenManager:
    """获取 DP Token Manager 实例"""
    global _dp_token_managers

    if dp_group_id not in _dp_token_managers:
        _dp_token_managers[dp_group_id] = DPTokenManager(dp_group_id, router_address)

    return _dp_token_managers[dp_group_id]


async def start_dp_token_manager(
    dp_group_id: int, router_address: str = "tcp://localhost:29700"
):
    """指定 DP 组的 Token Manager"""
    manager = get_dp_token_manager(dp_group_id, router_address)
    await manager.start()
    return manager
