#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance Monitor for QuangTPS

Real-time monitoring của hiệu năng hệ thống bao gồm
CPU, memory, GPU usage và dose calculation performance.
"""

import logging
import psutil
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QGroupBox,
        QGridLayout,
        QTextEdit,
        QPushButton,
        QCheckBox,
    )
    from PyQt5.QtCore import QTimer, pyqtSignal, QThread, QObject
    from PyQt5.QtGui import QFont, QColor

    _PYQT_AVAILABLE = True
except ImportError:
    _PYQT_AVAILABLE = False
    QWidget = object
    QObject = object

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.animation as animation

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml

    _NVIDIA_ML_AVAILABLE = True
except ImportError:
    _NVIDIA_ML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Metrics của hệ thống tại một thời điểm"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None


@dataclass
class PerformanceStats:
    """Thống kê performance của các operations"""

    operation_name: str
    total_calls: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    last_call_time: Optional[datetime] = None


class PerformanceTracker:
    """Tracker để đo performance của các operations"""

    def __init__(self):
        self.stats: Dict[str, PerformanceStats] = {}
        self.active_operations: Dict[str, float] = {}
        self.lock = threading.Lock()

    def start_operation(self, operation_name: str) -> str:
        """Bắt đầu đo một operation"""
        start_time = time.time()
        operation_id = f"{operation_name}_{start_time}"

        with self.lock:
            self.active_operations[operation_id] = start_time

        return operation_id

    def end_operation(self, operation_id: str):
        """Kết thúc đo operation"""
        end_time = time.time()

        with self.lock:
            if operation_id not in self.active_operations:
                return

            start_time = self.active_operations.pop(operation_id)
            operation_name = operation_id.split("_")[0]
            duration = end_time - start_time

            if operation_name not in self.stats:
                self.stats[operation_name] = PerformanceStats(
                    operation_name=operation_name,
                    total_calls=0,
                    total_time=0.0,
                    average_time=0.0,
                    min_time=float("inf"),
                    max_time=0.0,
                )

            stats = self.stats[operation_name]
            stats.total_calls += 1
            stats.total_time += duration
            stats.average_time = stats.total_time / stats.total_calls
            stats.min_time = min(stats.min_time, duration)
            stats.max_time = max(stats.max_time, duration)
            stats.last_call_time = datetime.now()

    def get_stats(self, operation_name: str) -> Optional[PerformanceStats]:
        """Lấy stats của một operation"""
        return self.stats.get(operation_name)

    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """Lấy tất cả stats"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset tất cả stats"""
        with self.lock:
            self.stats.clear()
            self.active_operations.clear()


class SystemMonitor(QThread):
    """Thread để monitor system metrics"""

    metrics_updated = pyqtSignal(object)  # SystemMetrics

    def __init__(self):
        super().__init__()
        self.running = False
        self.update_interval = 1.0  # seconds

        # Initialize GPU monitoring
        self.gpu_available = False
        if _NVIDIA_ML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
                if self.gpu_available:
                    self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                logger.warning(f"Không thể khởi tạo GPU monitoring: {e}")
                self.gpu_available = False

    def run(self):
        """Main monitoring loop"""
        self.running = True

        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics_updated.emit(metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Lỗi trong system monitoring: {e}")
                time.sleep(1.0)

    def stop(self):
        """Dừng monitoring"""
        self.running = False
        self.wait()

    def _collect_metrics(self) -> SystemMetrics:
        """Thu thập system metrics"""
        # CPU và Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # GPU metrics
        gpu_percent = None
        gpu_memory_percent = None
        gpu_temperature = None

        if self.gpu_available:
            try:
                # GPU utilization
                gpu_util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_percent = gpu_util.gpu

                # GPU memory
                gpu_mem = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_percent = (gpu_mem.used / gpu_mem.total) * 100

                # GPU temperature
                gpu_temperature = nvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, nvml.NVML_TEMPERATURE_GPU
                )

            except Exception as e:
                logger.warning(f"Lỗi khi đọc GPU metrics: {e}")

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temperature=gpu_temperature,
        )


class PerformanceWidget(QWidget):
    """Widget hiển thị performance metrics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_history = deque(maxlen=300)  # 5 minutes at 1Hz
        self.performance_tracker = PerformanceTracker()

        self.setup_ui()
        self.setup_monitoring()

    def setup_ui(self):
        """Thiết lập giao diện"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("System Performance Monitor")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("Reset Stats")
        self.reset_btn.clicked.connect(self.reset_stats)
        controls_layout.addWidget(self.reset_btn)

        self.auto_update_cb = QCheckBox("Auto Update")
        self.auto_update_cb.setChecked(True)
        controls_layout.addWidget(self.auto_update_cb)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Current metrics
        self.setup_current_metrics(layout)

        # Performance charts
        if _MATPLOTLIB_AVAILABLE:
            self.setup_charts(layout)

        # Performance stats
        self.setup_performance_stats(layout)

        self.setLayout(layout)

    def setup_current_metrics(self, parent_layout):
        """Thiết lập hiển thị metrics hiện tại"""
        group = QGroupBox("Current System Metrics")
        layout = QGridLayout()

        # CPU
        layout.addWidget(QLabel("CPU Usage:"), 0, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        layout.addWidget(self.cpu_bar, 0, 1)
        self.cpu_label = QLabel("0%")
        layout.addWidget(self.cpu_label, 0, 2)

        # Memory
        layout.addWidget(QLabel("Memory Usage:"), 1, 0)
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        layout.addWidget(self.memory_bar, 1, 1)
        self.memory_label = QLabel("0%")
        layout.addWidget(self.memory_label, 1, 2)

        # GPU (if available)
        self.gpu_widgets = []
        if _NVIDIA_ML_AVAILABLE:
            layout.addWidget(QLabel("GPU Usage:"), 2, 0)
            self.gpu_bar = QProgressBar()
            self.gpu_bar.setRange(0, 100)
            layout.addWidget(self.gpu_bar, 2, 1)
            self.gpu_label = QLabel("N/A")
            layout.addWidget(self.gpu_label, 2, 2)
            self.gpu_widgets.extend([self.gpu_bar, self.gpu_label])

            layout.addWidget(QLabel("GPU Memory:"), 3, 0)
            self.gpu_mem_bar = QProgressBar()
            self.gpu_mem_bar.setRange(0, 100)
            layout.addWidget(self.gpu_mem_bar, 3, 1)
            self.gpu_mem_label = QLabel("N/A")
            layout.addWidget(self.gpu_mem_label, 3, 2)
            self.gpu_widgets.extend([self.gpu_mem_bar, self.gpu_mem_label])

            layout.addWidget(QLabel("GPU Temperature:"), 4, 0)
            self.gpu_temp_label = QLabel("N/A")
            layout.addWidget(self.gpu_temp_label, 4, 1, 1, 2)
            self.gpu_widgets.append(self.gpu_temp_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def setup_charts(self, parent_layout):
        """Thiết lập biểu đồ performance"""
        group = QGroupBox("Performance Charts")
        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Setup subplots
        self.ax_cpu = self.figure.add_subplot(221)
        self.ax_memory = self.figure.add_subplot(222)
        self.ax_gpu = self.figure.add_subplot(223) if _NVIDIA_ML_AVAILABLE else None
        self.ax_operations = self.figure.add_subplot(224)

        self.figure.tight_layout()

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def setup_performance_stats(self, parent_layout):
        """Thiết lập hiển thị performance stats"""
        group = QGroupBox("Operation Performance Statistics")
        layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.stats_text)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def setup_monitoring(self):
        """Thiết lập system monitoring"""
        self.system_monitor = SystemMonitor()
        self.system_monitor.metrics_updated.connect(self.update_metrics)

        # Timer để update charts
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_charts)

        # Timer để update stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_performance_stats)
        self.stats_timer.start(5000)  # Update every 5 seconds

    def start_monitoring(self):
        """Bắt đầu monitoring"""
        self.system_monitor.start()
        if _MATPLOTLIB_AVAILABLE:
            self.chart_timer.start(2000)  # Update charts every 2 seconds

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Dừng monitoring"""
        self.system_monitor.stop()
        self.chart_timer.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        logger.info("Performance monitoring stopped")

    def reset_stats(self):
        """Reset performance stats"""
        self.performance_tracker.reset_stats()
        self.metrics_history.clear()
        self.update_performance_stats()

        if _MATPLOTLIB_AVAILABLE:
            self.update_charts()

        logger.info("Performance stats reset")

    def update_metrics(self, metrics: SystemMetrics):
        """Cập nhật hiển thị metrics"""
        if not self.auto_update_cb.isChecked():
            return

        # Add to history
        self.metrics_history.append(metrics)

        # Update progress bars
        self.cpu_bar.setValue(int(metrics.cpu_percent))
        self.cpu_label.setText(f"{metrics.cpu_percent:.1f}%")

        self.memory_bar.setValue(int(metrics.memory_percent))
        self.memory_label.setText(
            f"{metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB)"
        )

        # Update GPU metrics if available
        if metrics.gpu_percent is not None and self.gpu_widgets:
            self.gpu_bar.setValue(int(metrics.gpu_percent))
            self.gpu_label.setText(f"{metrics.gpu_percent:.1f}%")

            if metrics.gpu_memory_percent is not None:
                self.gpu_mem_bar.setValue(int(metrics.gpu_memory_percent))
                self.gpu_mem_label.setText(f"{metrics.gpu_memory_percent:.1f}%")

            if metrics.gpu_temperature is not None:
                self.gpu_temp_label.setText(f"{metrics.gpu_temperature:.0f}°C")

    def update_charts(self):
        """Cập nhật biểu đồ"""
        if not _MATPLOTLIB_AVAILABLE or not self.metrics_history:
            return

        try:
            # Prepare data
            times = [m.timestamp for m in self.metrics_history]
            cpu_values = [m.cpu_percent for m in self.metrics_history]
            memory_values = [m.memory_percent for m in self.metrics_history]

            # Clear previous plots
            self.ax_cpu.clear()
            self.ax_memory.clear()
            if self.ax_gpu:
                self.ax_gpu.clear()
            self.ax_operations.clear()

            # CPU chart
            self.ax_cpu.plot(times, cpu_values, "b-", linewidth=2)
            self.ax_cpu.set_title("CPU Usage (%)")
            self.ax_cpu.set_ylim(0, 100)
            self.ax_cpu.grid(True, alpha=0.3)

            # Memory chart
            self.ax_memory.plot(times, memory_values, "r-", linewidth=2)
            self.ax_memory.set_title("Memory Usage (%)")
            self.ax_memory.set_ylim(0, 100)
            self.ax_memory.grid(True, alpha=0.3)

            # GPU chart
            if self.ax_gpu and any(
                m.gpu_percent is not None for m in self.metrics_history
            ):
                gpu_values = [m.gpu_percent or 0 for m in self.metrics_history]
                self.ax_gpu.plot(times, gpu_values, "g-", linewidth=2)
                self.ax_gpu.set_title("GPU Usage (%)")
                self.ax_gpu.set_ylim(0, 100)
                self.ax_gpu.grid(True, alpha=0.3)

            # Operation performance chart
            stats = self.performance_tracker.get_all_stats()
            if stats:
                operations = list(stats.keys())[:5]  # Top 5 operations
                avg_times = [
                    stats[op].average_time * 1000 for op in operations
                ]  # Convert to ms

                self.ax_operations.bar(operations, avg_times)
                self.ax_operations.set_title("Average Operation Time (ms)")
                self.ax_operations.tick_params(axis="x", rotation=45)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Lỗi khi update charts: {e}")

    def update_performance_stats(self):
        """Cập nhật performance statistics"""
        try:
            stats = self.performance_tracker.get_all_stats()

            if not stats:
                self.stats_text.setText("No performance data available")
                return

            # Format stats text
            text = "Operation Performance Statistics:\n"
            text += "=" * 60 + "\n"
            text += f"{'Operation':<20} {'Calls':<8} {'Avg(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10}\n"
            text += "-" * 60 + "\n"

            for op_name, stat in sorted(stats.items()):
                text += f"{op_name:<20} {stat.total_calls:<8} "
                text += f"{stat.average_time * 1000:<10.2f} {stat.min_time * 1000:<10.2f} {stat.max_time * 1000:<10.2f}\n"

            # Add system summary
            if self.metrics_history:
                latest = self.metrics_history[-1]
                text += "\n" + "=" * 60 + "\n"
                text += "Current System Status:\n"
                text += f"CPU: {latest.cpu_percent:.1f}%\n"
                text += f"Memory: {latest.memory_percent:.1f}% ({latest.memory_used_gb:.1f}GB)\n"
                if latest.gpu_percent is not None:
                    text += f"GPU: {latest.gpu_percent:.1f}%\n"
                    text += f"GPU Memory: {latest.gpu_memory_percent:.1f}%\n"
                    text += f"GPU Temperature: {latest.gpu_temperature:.0f}°C\n"

            self.stats_text.setText(text)

        except Exception as e:
            logger.error(f"Lỗi khi update performance stats: {e}")

    def track_operation(self, operation_name: str):
        """Context manager để track operation performance"""

        class OperationTracker:
            def __init__(self, tracker, op_name):
                self.tracker = tracker
                self.op_name = op_name
                self.op_id = None

            def __enter__(self):
                self.op_id = self.tracker.performance_tracker.start_operation(
                    self.op_name
                )
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.op_id:
                    self.tracker.performance_tracker.end_operation(self.op_id)

        return OperationTracker(self, operation_name)


# Global performance monitor instance
_performance_widget = None


def get_performance_monitor() -> Optional[PerformanceWidget]:
    """Lấy performance monitor instance"""
    global _performance_widget
    return _performance_widget


def create_performance_monitor(parent=None) -> PerformanceWidget:
    """Tạo performance monitor widget"""
    global _performance_widget
    if not _PYQT_AVAILABLE:
        logger.warning("PyQt5 không khả dụng. Performance monitor không hoạt động.")
        return None

    _performance_widget = PerformanceWidget(parent)
    return _performance_widget


def track_operation(operation_name: str):
    """Decorator để track performance của function"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            if monitor:
                with monitor.track_operation(operation_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
