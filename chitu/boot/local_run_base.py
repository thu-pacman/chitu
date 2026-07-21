# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, Sequence


class LocalRunCallback(Protocol):
    def __call__(
        self,
        cfg,
        raw_argv: Sequence[str],
        master_addr,
        master_port,
        rdvz_port,
        rdvz_id,
        *,
        is_multi_inst,
        is_router,
        is_master_node,
        torchrun_n_nodes,
        torchrun_nproc_per_node,
        container_name_suffix=None,
        _proc_registry=None,
    ) -> None: ...
