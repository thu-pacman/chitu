# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Optional
from chitu.global_vars import get_global_args
from chitu.utils import try_import_opt_dep

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")

_cpu_infer_instance = None
_lock = threading.Lock()


def get_cpu_infer(bind_thread_to_cpu: Optional[str] = None):
    global _cpu_infer_instance
    if _cpu_infer_instance is None:
        with _lock:
            if _cpu_infer_instance is None:
                if bind_thread_to_cpu is None:
                    bind_thread_to_cpu = get_global_args().infer.bind_thread_to_cpu
                _cpu_infer_instance = cpuinfer.CPUInfer(bind_thread_to_cpu)
    return _cpu_infer_instance
