# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging
import os
import shutil
import subprocess
import threading
import time
import re
from pathlib import Path
from string import Template
from typing import Optional

import requests

from chitu.distributed.tcp_ip import get_free_port, is_port_available
from chitu.global_vars import get_global_args

logger = logging.getLogger(__name__)

_PROVISIONING_DIR = [
    Path(__file__).parent / "grafana",
    Path("/workspace/chitu/grafana"),
]


class GrafanaManager:
    """Manage an embedded Grafana server that auto-provisions a Prometheus
    datasource and the Chitu dashboard.

    Mirrors the lifecycle pattern of ``PrometheusServerManager``:
    singleton created on Rank 0, auto-cleaned up via ``atexit``.
    """

    _instance: Optional["GrafanaManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, prometheus_url: str):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    metrics_args = get_global_args().metrics
                    cls._instance = cls(
                        prometheus_url=prometheus_url,
                        grafana_host=metrics_args.grafana_host,
                        server_port=metrics_args.grafana_port,
                    )
        return cls._instance

    @staticmethod
    def _health_check_host(bind_host: str) -> str:
        """Map a bind address to a connectable address for health checks."""
        if bind_host in ("0.0.0.0", "", "::", "[::]"):
            return "127.0.0.1"
        return bind_host

    def __init__(self, prometheus_url: str, grafana_host: str, server_port: int):
        self.prometheus_url = prometheus_url
        self.grafana_host = grafana_host
        self.server_port = server_port
        self.process: Optional[subprocess.Popen] = None

        pid = os.getpid()
        self.work_dir = Path(os.path.abspath(f"grafana_data_{pid}"))
        self._prepare_work_dir()
        self.start()
        atexit.register(GrafanaManager.cleanup)

    # ------------------------------------------------------------------
    # Directory & provisioning setup
    # ------------------------------------------------------------------

    def _prepare_work_dir(self):
        """Materialise provisioning templates into a per-process work dir."""
        prov_dst = self.work_dir / "provisioning"
        dash_dst = self.work_dir / "dashboards"

        for d in (prov_dst / "datasources", prov_dst / "dashboards", dash_dst):
            d.mkdir(parents=True, exist_ok=True)

        provisioning_dir = None
        for candidate in _PROVISIONING_DIR:
            if candidate.exists():
                provisioning_dir = candidate
                break
        else:
            raise FileNotFoundError("Grafana provisioning directory not found")

        src_ds = provisioning_dir / "provisioning" / "datasources" / "prometheus.yml"
        content = src_ds.read_text()
        rendered = Template(content).safe_substitute(
            PROMETHEUS_HOST=self.prometheus_url.split("//")[-1].rsplit(":", 1)[0],
            PROMETHEUS_PORT=self.prometheus_url.rsplit(":", 1)[-1],
        )
        (prov_dst / "datasources" / "prometheus.yml").write_text(rendered)

        src_dp = provisioning_dir / "provisioning" / "dashboards" / "dashboard.yml"
        content = src_dp.read_text()
        rendered = Template(content).safe_substitute(DASHBOARD_DIR=str(dash_dst))
        (prov_dst / "dashboards" / "dashboard.yml").write_text(rendered)

        src_json = provisioning_dir / "dashboards" / "chitu_overview.json"
        shutil.copy2(src_json, dash_dst / "chitu_overview.json")

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self, timeout: int = 60, max_retries: int = 8):
        data_path = str(self.work_dir / "data")
        logs_path = str(self.work_dir / "log")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)
        logger.info(f"Grafana work dir: {self.work_dir}")
        logger.info(f"Grafana log dir: {logs_path}")

        for attempt in range(max_retries + 1):
            if attempt > 0 or not is_port_available(self.server_port):
                port = get_free_port()
                logger.warning(
                    f"Port[{self.server_port}] is unavailable, change Grafana server port to {port}"
                )
                self.server_port = port

            if self._start_once(timeout):
                return True

        logger.error(f"Failed to start Grafana server after {max_retries + 1} attempts")
        self._cleanup_files()
        return False

    def _start_once(self, timeout=60):
        """Start Grafana once on self.server_port and wait until it is ready."""

        prov_path = str(self.work_dir / "provisioning")
        data_path = str(self.work_dir / "data")
        logs_path = str(self.work_dir / "log")
        cmd = [
            "grafana-server",
            f"--homepath={self._find_grafana_home()}",
            f"--config={self._write_ini()}",
            "cfg:default.paths.provisioning=" + prov_path,
            "cfg:default.paths.data=" + data_path,
            "cfg:default.paths.logs=" + logs_path,
        ]

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
        except FileNotFoundError as e:
            logger.error(
                f"Error occurred while starting Grafana, please verify whether the Grafana binary path is included in the current $PATH variable: {e}"
            )
            return False

        start_time = time.time()
        check_host = self._health_check_host(self.grafana_host)
        health_url = f"http://{check_host}:{self.server_port}/api/health"
        logger.info(f"Waiting Grafana server ready ...")
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                logger.error(
                    f"Grafana server process exited unexpectedly (exit_code={self.process.returncode}):\n{stderr}"
                )
                self.process = None
                return False

            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    # double check
                    time.sleep(0.5)
                    if not self._read_server_start_log():
                        logger.error(
                            f"Grafana server is detected but no server starting log was found. "
                            f"This maybe because other Grafana server takes this port."
                        )
                        self._stop_current_process()
                        self._clean_log_file()
                        return False
                    logger.info(
                        f"Grafana is running: http://{self.grafana_host}:{self.server_port} "
                        f"(PID: {self.process.pid})"
                    )
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)

        logger.error(f"Start Grafana server timeout")
        self._stop_current_process()
        return False

    def _find_grafana_home(self) -> str:
        """Best-effort locate grafana home directory."""
        candidates = [
            os.environ.get("GF_PATHS_HOME", ""),
            "/usr/share/grafana",
            "/opt/grafana",
        ]
        for c in candidates:
            if c and os.path.isdir(c):
                return c
        return "/usr/share/grafana"

    def _write_ini(self) -> str:
        """Write a minimal grafana.ini with auth disabled for local use."""
        ini_path = self.work_dir / "grafana.ini"
        ini_path.write_text(
            "[server]\n"
            f"http_addr = {self.grafana_host}\n"
            f"http_port = {self.server_port}\n"
            "protocol = http\n"
            "\n"
            "[auth.anonymous]\n"
            "enabled = true\n"
            "org_role = Admin\n"
            "\n"
            "[security]\n"
            "admin_user = admin\n"
            "admin_password = admin\n"
            "\n"
            "[users]\n"
            "default_theme = dark\n"
        )
        return str(ini_path)

    def is_running(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        try:
            check_host = self._health_check_host(self.grafana_host)
            resp = requests.get(
                f"http://{check_host}:{self.server_port}/api/health", timeout=1
            )
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _stop_current_process(self):
        if self.process:
            logger.info(f"Stopping Grafana server (PID: {self.process.pid}) ...")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                logger.info("Grafana server stopped")
            except Exception as e:
                logger.error(f"Error stopping Grafana: {e}")
            finally:
                self.process = None

    def _read_server_start_log(self):
        """
        NOTE: Set the stdout=PIPE for Grafana subprocess will cause timeout.
        Here directly reads the status from the log file.
        """
        hosts = (
            ["127.0.0.1", "0.0.0.0", "", "::", "[::]"]
            if GrafanaManager._health_check_host(self.grafana_host) == "127.0.0.1"
            else [self.grafana_host]
        )
        hosts_pattern = "(" + "|".join([re.escape(host) for host in hosts]) + ")"
        pattern = re.compile(
            f'level=info msg="HTTP Server Listen" address={hosts_pattern}:{self.server_port} protocol=http subUrl= socket='
        )
        log_file_path = str(self.work_dir / "log" / "grafana.log")
        with open(log_file_path, "r") as LOG_FILE:
            full_log = LOG_FILE.read()
            if pattern.search(full_log) is not None:
                return True
        return False

    def _clean_log_file(self):
        log_file_path = str(self.work_dir / "log" / "grafana.log")
        try:
            os.remove(log_file_path)
        except OSError as e:
            logger.warning(f"Failed to remove Grafana log file: {e}")

    def _cleanup_files(self):
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir)
            except OSError as e:
                logger.warning(f"Failed to remove Grafana work dir: {e}")

    @classmethod
    def cleanup(cls):
        with cls._lock:
            if cls._instance is None:
                return
            cls._instance._stop_current_process()
            cls._instance._cleanup_files()
            cls._instance = None
