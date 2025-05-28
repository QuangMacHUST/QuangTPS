"""
Module theo dõi và tối ưu hóa hiệu năng hệ thống QuangTPS.

Provides comprehensive performance monitoring and optimization for:
- Memory usage tracking
- CPU utilization monitoring
- GPU resource management
- Calculation time profiling
- System health checks
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import gc
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Metrics hiệu năng hệ thống."""

    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_available_mb: float = 0.0

    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_cores: int = 0

    # GPU metrics (if available)
    gpu_memory_mb: float = 0.0
    gpu_utilization_percent: float = 0.0

    # Calculation metrics
    calculation_times: Dict[str, float] = field(default_factory=dict)
    active_calculations: int = 0

    # System health
    disk_usage_percent: float = 0.0
    network_io_mb: float = 0.0

    # Timestamps
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Monitor hiệu năng hệ thống real-time."""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread = None

        # Metrics storage
        self.current_metrics = PerformanceMetrics()
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements

        # Calculation tracking
        self.active_calculations = {}
        self.calculation_history = defaultdict(list)

        # Callbacks for alerts
        self.alert_callbacks = []

        # Thresholds for alerts
        self.memory_threshold_mb = 8000  # 8GB
        self.cpu_threshold_percent = 80
        self.gpu_memory_threshold_mb = 3000  # 3GB

        # GPU detection
        self.has_gpu = self._detect_gpu()

        logger.info(f"PerformanceMonitor initialized - GPU available: {self.has_gpu}")

    def _detect_gpu(self) -> bool:
        """Detect GPU availability."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            try:
                import pynvml

                pynvml.nvmlInit()
                return True
            except:
                return False
        except:
            return False

    def start_monitoring(self):
        """Bắt đầu monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Dừng monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._update_metrics()
                self._check_alerts()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)

    def _update_metrics(self):
        """Cập nhật metrics hiện tại."""
        # Memory metrics
        memory = psutil.virtual_memory()
        self.current_metrics.memory_usage_mb = memory.used / (1024 * 1024)
        self.current_metrics.memory_available_mb = memory.available / (1024 * 1024)

        # Update peak memory
        if self.current_metrics.memory_usage_mb > self.current_metrics.memory_peak_mb:
            self.current_metrics.memory_peak_mb = self.current_metrics.memory_usage_mb

        # CPU metrics
        self.current_metrics.cpu_usage_percent = psutil.cpu_percent()
        self.current_metrics.cpu_cores = psutil.cpu_count()

        # GPU metrics
        if self.has_gpu:
            self._update_gpu_metrics()

        # Disk usage
        disk = psutil.disk_usage("/")
        self.current_metrics.disk_usage_percent = (disk.used / disk.total) * 100

        # Active calculations
        self.current_metrics.active_calculations = len(self.active_calculations)

        # Update timestamp
        self.current_metrics.timestamp = time.time()

        # Store in history
        self.metrics_history.append(
            PerformanceMetrics(
                memory_usage_mb=self.current_metrics.memory_usage_mb,
                memory_peak_mb=self.current_metrics.memory_peak_mb,
                memory_available_mb=self.current_metrics.memory_available_mb,
                cpu_usage_percent=self.current_metrics.cpu_usage_percent,
                cpu_cores=self.current_metrics.cpu_cores,
                gpu_memory_mb=self.current_metrics.gpu_memory_mb,
                gpu_utilization_percent=self.current_metrics.gpu_utilization_percent,
                calculation_times=self.current_metrics.calculation_times.copy(),
                active_calculations=self.current_metrics.active_calculations,
                disk_usage_percent=self.current_metrics.disk_usage_percent,
                timestamp=self.current_metrics.timestamp,
            )
        )

    def _update_gpu_metrics(self):
        """Cập nhật GPU metrics."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.current_metrics.gpu_memory_mb = gpu.memoryUsed
                self.current_metrics.gpu_utilization_percent = gpu.load * 100
        except ImportError:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.current_metrics.gpu_memory_mb = mem_info.used / (1024 * 1024)

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.current_metrics.gpu_utilization_percent = util.gpu
            except:
                pass

    def _check_alerts(self):
        """Kiểm tra và gửi alerts."""
        alerts = []

        # Memory alert
        if self.current_metrics.memory_usage_mb > self.memory_threshold_mb:
            alerts.append(
                f"High memory usage: {self.current_metrics.memory_usage_mb:.1f}MB"
            )

        # CPU alert
        if self.current_metrics.cpu_usage_percent > self.cpu_threshold_percent:
            alerts.append(
                f"High CPU usage: {self.current_metrics.cpu_usage_percent:.1f}%"
            )

        # GPU memory alert
        if (
            self.has_gpu
            and self.current_metrics.gpu_memory_mb > self.gpu_memory_threshold_mb
        ):
            alerts.append(
                f"High GPU memory usage: {self.current_metrics.gpu_memory_mb:.1f}MB"
            )

        # Send alerts
        for alert in alerts:
            self._send_alert(alert)

    def _send_alert(self, message: str):
        """Gửi alert message."""
        logger.warning(f"Performance Alert: {message}")
        for callback in self.alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable[[str], None]):
        """Thêm callback cho alerts."""
        self.alert_callbacks.append(callback)

    def start_calculation(self, name: str) -> str:
        """Bắt đầu tracking calculation."""
        calc_id = f"{name}_{time.time()}"
        self.active_calculations[calc_id] = {
            "name": name,
            "start_time": time.time(),
            "memory_start": self.current_metrics.memory_usage_mb,
        }
        return calc_id

    def end_calculation(self, calc_id: str):
        """Kết thúc tracking calculation."""
        if calc_id in self.active_calculations:
            calc_info = self.active_calculations.pop(calc_id)
            duration = time.time() - calc_info["start_time"]
            memory_used = (
                self.current_metrics.memory_usage_mb - calc_info["memory_start"]
            )

            # Store in history
            self.calculation_history[calc_info["name"]].append(
                {
                    "duration": duration,
                    "memory_used": memory_used,
                    "timestamp": time.time(),
                }
            )

            # Update current metrics
            self.current_metrics.calculation_times[calc_info["name"]] = duration

            logger.info(
                f"Calculation '{calc_info['name']}' completed in {duration:.2f}s, "
                f"memory used: {memory_used:.1f}MB"
            )

    def get_calculation_stats(self, name: str) -> Dict[str, Any]:
        """Lấy thống kê calculation."""
        if name not in self.calculation_history:
            return {}

        history = self.calculation_history[name]
        durations = [h["duration"] for h in history]
        memory_usage = [h["memory_used"] for h in history]

        return {
            "count": len(history),
            "avg_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "avg_memory": np.mean(memory_usage),
            "total_time": np.sum(durations),
        }

    def optimize_memory(self):
        """Tối ưu hóa memory usage."""
        logger.info("Starting memory optimization...")

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")

        # Clear calculation history if too large
        for name in list(self.calculation_history.keys()):
            if len(self.calculation_history[name]) > 100:
                # Keep only last 50 entries
                self.calculation_history[name] = self.calculation_history[name][-50:]

        # Clear old metrics history
        if len(self.metrics_history) > 500:
            # Keep only last 300 entries
            new_history = deque(list(self.metrics_history)[-300:], maxlen=1000)
            self.metrics_history = new_history

        logger.info("Memory optimization completed")

    def get_system_report(self) -> Dict[str, Any]:
        """Tạo báo cáo hệ thống."""
        return {
            "current_metrics": {
                "memory_usage_mb": self.current_metrics.memory_usage_mb,
                "memory_peak_mb": self.current_metrics.memory_peak_mb,
                "cpu_usage_percent": self.current_metrics.cpu_usage_percent,
                "gpu_memory_mb": self.current_metrics.gpu_memory_mb,
                "gpu_utilization_percent": self.current_metrics.gpu_utilization_percent,
                "active_calculations": self.current_metrics.active_calculations,
                "disk_usage_percent": self.current_metrics.disk_usage_percent,
            },
            "system_info": {
                "cpu_cores": self.current_metrics.cpu_cores,
                "has_gpu": self.has_gpu,
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "monitoring_active": self.is_monitoring,
            },
            "calculation_stats": {
                name: self.get_calculation_stats(name)
                for name in self.calculation_history.keys()
            },
            "history_length": len(self.metrics_history),
        }


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Lấy global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        _performance_monitor.start_monitoring()
    return _performance_monitor


def track_calculation(name: str):
    """Decorator để track calculation performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            calc_id = monitor.start_calculation(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end_calculation(calc_id)

        return wrapper

    return decorator


# Context manager for calculation tracking
class CalculationTracker:
    """Context manager để track calculation."""

    def __init__(self, name: str):
        self.name = name
        self.calc_id = None
        self.monitor = get_performance_monitor()

    def __enter__(self):
        self.calc_id = self.monitor.start_calculation(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.calc_id:
            self.monitor.end_calculation(self.calc_id)
