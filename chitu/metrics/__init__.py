# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

if os.environ.get("CHITU_GENERATING_DOCS") != "1":
    from chitu.metrics.prometheus_collector import PrometheusMetricsCollector
    from chitu.metrics.registry import metrics_runtime_context
    from chitu.metrics.prometheus_manager import PrometheusServerManager
    from chitu.metrics.grafana_manager import GrafanaManager
    from chitu.metrics.metrics_monitor import (
        start_prometheus_server_and_metrics_monitor,
        stop_metrics_monitor,
    )
