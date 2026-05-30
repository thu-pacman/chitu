# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import yaml
import subprocess
import os
import time
import requests
import atexit
from typing import Optional
import threading
import logging

from chitu.global_vars import get_global_args
from chitu.distributed.tcp_ip import is_port_available, get_free_port

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_INTERVAL: str = "15s"
_DEFAULT_JOB_NAME: str = "chitu_service"


class PrometheusServerManager:
    """
    Manage Prometheus Server, including config, start, stop and query.
    """

    _instance: Optional["PrometheusServerManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        collector_addrs: list[str],
    ):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    metrics_args = get_global_args().metrics
                    server_addr: str = metrics_args.prometheus_listening_host
                    server_port: int = metrics_args.prometheus_listening_port
                    config_file: str = metrics_args.prometheus_config_file
                    data_dir: str = metrics_args.prometheus_data_dir
                    scrape_interval: int = metrics_args.prometheus_scrape_interval
                    cls._instance = cls(
                        collector_addrs,
                        server_addr,
                        server_port,
                        config_file,
                        data_dir,
                        scrape_interval,
                    )
        return cls._instance

    def __init__(
        self,
        collector_addrs,
        server_addr: str,
        server_port: int,
        config_file: str,
        data_dir: str,
        scrape_interval: int,
    ):
        """
        Args:
            collector_addrs: Prometheus Server pull metrics from these addresses.
            server_port: Prometheus Server listening port
            config_file: Prometheus configuration file path
            data_dir: Prometheus data stoarge path
        """
        self.collector_addrs = collector_addrs
        self.server_addr = server_addr
        self.server_port = server_port
        # Use PID to isolate config/data per instance, avoiding TSDB lock
        # conflicts when multiple instances run on the same machine.
        pid = os.getpid()
        base, ext = os.path.splitext(config_file)
        self.config_file = f"{base}_{pid}{ext}"
        self.data_dir = f"{data_dir}_{pid}"
        self.process = None
        self.server_url = f"http://{server_addr}:{server_port}"
        self.query_url = f"{self.server_url}/api/v1/query"
        self.create_config(collector_addrs, scrape_interval)
        self.start()

        atexit.register(self.cleanup)

    def create_config(
        self,
        targets,
        scrape_interval: int,
    ):
        """
        Creat Prometheus configuration file , save to self.config_file
        Args:
            targets: target Collectors, format: ['ip1:addr1',...]
            scrape_interval: interval of Prometheus server's two scrape actions
        """
        config = {
            "global": {
                "scrape_interval": f"{int(scrape_interval)}s",
                "evaluation_interval": _DEFAULT_EVAL_INTERVAL,
            },
            "scrape_configs": [
                {
                    "job_name": _DEFAULT_JOB_NAME,
                    "static_configs": [{"targets": targets}],
                }
            ],
        }

        with open(self.config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(
            f"Prometheus configuration file has been created: {self.config_file}"
        )

    def start(self, timeout=60, max_retries=8):
        """
        Start Prometheus Server
        Args:
            timeout: 等待超时时间(秒)
            max_retries: retry count after a failed start attempt
        Returns:
            bool: whether Prometheus Server is ready
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Prometheus config file doesn't exist: {self.config_file}. "
            )

        os.makedirs(self.data_dir, exist_ok=True)
        lock_file = os.path.join(self.data_dir, "lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info(f"Removed stale TSDB lock file: {lock_file}")
            except OSError as e:
                logger.warning(f"Failed to remove stale TSDB lock file: {e}")

        for attempt in range(max_retries + 1):
            if attempt > 0 or not is_port_available(self.server_port):
                port = get_free_port()
                logger.warning(
                    f"Port[{self.server_port}] is unavailable, change Prometheus server port to {port}"
                )
                self.server_port = port
                self.server_url = f"http://{self.server_addr}:{self.server_port}"
                self.query_url = f"{self.server_url}/api/v1/query"

            if self._start_once(timeout):
                return True

        logger.error(
            f"Failed to start Prometheus server after {max_retries + 1} attempts"
        )
        return False

    def _start_once(self, timeout=60):
        """Start Prometheus once on self.server_port and wait until it is ready."""
        cmd = [
            "prometheus",
            f"--config.file={self.config_file}",
            f"--web.listen-address=:{self.server_port}",
            f"--storage.tsdb.path={self.data_dir}",
        ]

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except FileNotFoundError as e:
            logger.error(
                f"Error occurred while starting Prometheus, please verify whether the Prometheus binary path is included in the current $PATH variable: {e}"
            )
            return False

        start_time = time.time()
        health_url = f"{self.server_url}/-/ready"

        logger.info(f"Waiting Prometheus server ready ...")
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                logger.error(
                    f"Prometheus server process exited unexpectedly (exit_code={self.process.returncode}):\n{stderr}"
                )
                self.process = None
                return False

            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    logger.info(
                        f"Prometheus Server is running: port: {self.server_port}, config: {self.config_file}, data: {self.data_dir}, URL: {self.server_url}, PID: {self.process.pid}"
                    )
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)

        logger.error(f"Start Prometheus server timeout")
        self._stop_current_process()
        return False

    def _stop_current_process(self):
        if not self.process:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        except Exception as e:
            logger.warning(f"Failed to stop Prometheus process: {e}")
        finally:
            self.process = None

    def is_running(self):
        """
        Check whether the Prometheus server is still running

        Returns:
            bool: is still running
        """
        if self.process and self.process.poll() is None:
            return True
        return False

    def query_metric_rate_each_rank(
        self, metric_name: str, time_window: str = "10s"
    ) -> dict[tuple[str, str, str], str]:
        """
        Query metric rate for each rank

        :param metric_name: metric name, eg: chitu_total_generated_tokens
        :type metric_name: str
        :param time_window: 1s means 10 seconds, 1m means 1 minutes
        :type time_window: str
        Returns:
            {(instance_id, dp_id, rank): metric_rate}
        """
        try:
            query = f"rate({metric_name}[{time_window}])"
            response = requests.get(
                self.query_url,
                params={"query": query},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            if data["status"] != "success":
                raise Exception(f"查询失败: {data}")

            ans = {}
            for result in data["data"]["result"]:
                if result["metric"]["job"] == _DEFAULT_JOB_NAME:
                    key = (
                        result["metric"]["instance_id"],
                        result["metric"]["dp_id"],
                        result["metric"]["rank"],
                    )
                    val = result["value"][1]
                    ans[key] = val
            return ans
        except Exception as e:
            logger.error(f"query_metric_rate_each_rank failed: {e}")
            return {}

    def query_metric_latest_value_each_rank(
        self, metric_name: str
    ) -> dict[tuple[str, str, str], str]:
        """
        Query metric value for each dp rank

        :param metric_name: metric name, eg: chitu_total_generated_tokens
        :type metric_name: str
        :param time_window: 1s means 10 seconds, 1m means 1 minutes
        :type time_window: str

        Returns:
            {(instance_id, dp_id, rank): metric_value}
        """
        try:
            query = f"{metric_name} offset 0s"
            response = requests.get(
                self.query_url,
                params={"query": query},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

            if data["status"] != "success":
                raise Exception(f"查询失败: {data}")

            ans = {}
            for result in data["data"]["result"]:
                if result["metric"]["job"] == _DEFAULT_JOB_NAME:
                    key = (
                        result["metric"]["instance_id"],
                        result["metric"]["dp_id"],
                        result["metric"]["rank"],
                    )
                    val = result["value"][1]
                    ans[key] = val
            return ans
        except Exception as e:
            logger.error(f"query_metric_latest_value_each_rank failed: {e}")
            return {}

    def list_all_metrics(self):
        """
        List all metrics in Prometheus server, for debug

        Returns:
            list: list of all metric names
        """
        metadata_url = f"{self.server_url}/api/v1/label/__name__/values"
        logger.warning(f"metadata_url:{metadata_url}")

        response = requests.get(metadata_url, proxies={"http": None, "https": None})
        data = response.json()

        if data["status"] == "success":
            metrics = data["data"]
            logger.warning(f"找到 {len(metrics)} 个指标:")
            for metric in sorted(metrics)[:20]:  # 只显示前 20 个
                logger.warning(f"  - {metric}")
            if len(metrics) > 20:
                logger.warning(f"  ... 还有 {len(metrics) - 20} 个")
            return metrics
        else:
            logger.warning("获取指标列表失败")
            return []

    def stop(self):
        """Stop Prometheus Server and clean up per-instance files."""
        if self.process:
            logger.info(
                f"Stopping Prometheus server process (PID: {self.process.pid})..."
            )
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                    logger.info("Prometheus server has stopped")
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                    logger.info("Prometheus server has been killed")
            except Exception as e:
                logger.error(f"Exception during stopping Prometheus server: {e}")
            finally:
                self.process = None

        # Clean up per-instance config file and data directory
        self._cleanup_files()

    def _cleanup_files(self):
        """Remove per-instance config file and data directory."""
        import shutil

        if self.config_file and os.path.exists(self.config_file):
            try:
                os.remove(self.config_file)
            except OSError as e:
                logger.warning(f"Failed to remove config file {self.config_file}: {e}")

        if self.data_dir and os.path.exists(self.data_dir):
            try:
                shutil.rmtree(self.data_dir)
            except OSError as e:
                logger.warning(f"Failed to remove data dir {self.data_dir}: {e}")

    @classmethod
    def cleanup(cls):
        with cls._lock:
            if cls._instance is None:
                return
            cls._instance.stop()
            cls._instance = None
