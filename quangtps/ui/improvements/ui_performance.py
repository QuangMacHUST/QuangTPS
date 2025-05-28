"""
Module cải thiện hiệu năng giao diện người dùng QuangTPS.

Provides:
- Lazy loading for heavy widgets
- Progressive rendering for large datasets
- Memory-efficient UI components
- Responsive design patterns
- Performance monitoring
"""

import time
import threading
import weakref
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
import logging

# PyQt5 imports with fallbacks
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QApplication,
        QProgressBar,
        QLabel,
        QVBoxLayout,
        QHBoxLayout,
        QScrollArea,
        QFrame,
        QSplitter,
        QTabWidget,
        QTimer,
    )
    from PyQt5.QtCore import (
        QThread,
        pyqtSignal,
        QTimer as QCoreTimer,
        QObject,
        QMutex,
        QSize,
    )
    from PyQt5.QtGui import QPixmap, QPainter, QFont, QFontMetrics

    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

    # Create mock classes
    class QWidget:
        pass

    class QThread:
        pass

    class pyqtSignal:
        def __init__(self, *args):
            pass

        def emit(self, *args):
            pass

        def connect(self, *args):
            pass


logger = logging.getLogger(__name__)


@dataclass
class UIPerformanceSettings:
    """Cài đặt hiệu năng UI."""

    # Lazy loading
    enable_lazy_loading: bool = True
    lazy_load_threshold: int = 100  # số widgets để trigger lazy loading
    lazy_load_chunk_size: int = 20

    # Progressive rendering
    enable_progressive_rendering: bool = True
    render_chunk_size: int = 1000  # số elements per chunk
    render_delay_ms: int = 50

    # Memory management
    enable_memory_optimization: bool = True
    widget_cache_size: int = 50
    auto_cleanup_interval_ms: int = 30000  # 30 seconds

    # Responsive design
    enable_responsive_design: bool = True
    breakpoint_mobile: int = 768
    breakpoint_tablet: int = 1024

    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_log_interval_ms: int = 5000


@dataclass
class UIPerformanceMetrics:
    """Metrics hiệu năng UI."""

    widget_creation_time: float = 0.0
    render_time: float = 0.0
    memory_usage_mb: float = 0.0
    widgets_loaded: int = 0
    widgets_cached: int = 0
    refresh_rate_fps: float = 0.0

    # User interaction metrics
    click_response_time_ms: float = 0.0
    scroll_performance_fps: float = 0.0
    resize_time_ms: float = 0.0


class LazyLoadingManager(QObject if PYQT5_AVAILABLE else object):
    """Manager cho lazy loading widgets."""

    widget_loaded = pyqtSignal(str) if PYQT5_AVAILABLE else None

    def __init__(self, settings: UIPerformanceSettings):
        if PYQT5_AVAILABLE:
            super().__init__()
        self.settings = settings
        self._pending_widgets = {}
        self._loaded_widgets = {}
        self._loading_queue = []
        self._is_loading = False

        logger.info("Initialized LazyLoadingManager")

    def register_widget(
        self, widget_id: str, create_function: Callable, priority: int = 0
    ):
        """Đăng ký widget để lazy load."""
        self._pending_widgets[widget_id] = {
            "create_function": create_function,
            "priority": priority,
            "created": False,
        }
        logger.debug(f"Registered widget for lazy loading: {widget_id}")

    def request_widget(self, widget_id: str) -> Optional[QWidget]:
        """Request widget với lazy loading."""
        if widget_id in self._loaded_widgets:
            return self._loaded_widgets[widget_id]

        if (
            widget_id in self._pending_widgets
            and not self._pending_widgets[widget_id]["created"]
        ):
            self._load_widget_now(widget_id)
            return self._loaded_widgets.get(widget_id)

        return None

    def _load_widget_now(self, widget_id: str):
        """Load widget ngay lập tức."""
        if widget_id not in self._pending_widgets:
            return

        start_time = time.time()

        try:
            widget_info = self._pending_widgets[widget_id]
            widget = widget_info["create_function"]()

            self._loaded_widgets[widget_id] = widget
            widget_info["created"] = True

            load_time = time.time() - start_time
            logger.debug(f"Loaded widget {widget_id} in {load_time:.3f}s")

            if self.widget_loaded:
                self.widget_loaded.emit(widget_id)

        except Exception as e:
            logger.error(f"Error loading widget {widget_id}: {e}")

    def preload_priority_widgets(self):
        """Preload các widgets có priority cao."""
        priority_widgets = sorted(
            [
                (wid, info)
                for wid, info in self._pending_widgets.items()
                if not info["created"]
            ],
            key=lambda x: x[1]["priority"],
            reverse=True,
        )

        for widget_id, _ in priority_widgets[: self.settings.lazy_load_chunk_size]:
            self._load_widget_now(widget_id)


class ProgressiveRenderer(QObject if PYQT5_AVAILABLE else object):
    """Renderer cho large datasets với progressive loading."""

    chunk_rendered = pyqtSignal(int) if PYQT5_AVAILABLE else None
    render_completed = pyqtSignal() if PYQT5_AVAILABLE else None

    def __init__(self, settings: UIPerformanceSettings):
        if PYQT5_AVAILABLE:
            super().__init__()
        self.settings = settings
        self._render_timer = QCoreTimer() if PYQT5_AVAILABLE else None
        self._current_chunk = 0
        self._total_chunks = 0
        self._render_data = []
        self._render_widget = None
        self._render_callback = None

        if self._render_timer:
            self._render_timer.timeout.connect(self._render_next_chunk)

    def start_progressive_render(
        self,
        data: List[Any],
        widget: QWidget,
        render_callback: Callable,
        chunk_size: Optional[int] = None,
    ):
        """Bắt đầu progressive rendering."""
        self._render_data = data
        self._render_widget = widget
        self._render_callback = render_callback

        chunk_size = chunk_size or self.settings.render_chunk_size
        self._total_chunks = len(data) // chunk_size + (
            1 if len(data) % chunk_size else 0
        )
        self._current_chunk = 0

        logger.info(
            f"Starting progressive render: {len(data)} items in {self._total_chunks} chunks"
        )

        if self._render_timer:
            self._render_timer.start(self.settings.render_delay_ms)

    def _render_next_chunk(self):
        """Render chunk tiếp theo."""
        if self._current_chunk >= self._total_chunks:
            self._render_timer.stop()
            if self.render_completed:
                self.render_completed.emit()
            logger.info("Progressive rendering completed")
            return

        start_idx = self._current_chunk * self.settings.render_chunk_size
        end_idx = min(
            start_idx + self.settings.render_chunk_size, len(self._render_data)
        )

        chunk_data = self._render_data[start_idx:end_idx]

        try:
            self._render_callback(chunk_data, self._render_widget)

            if self.chunk_rendered:
                self.chunk_rendered.emit(self._current_chunk)

        except Exception as e:
            logger.error(f"Error rendering chunk {self._current_chunk}: {e}")

        self._current_chunk += 1


class ResponsiveLayoutManager:
    """Manager cho responsive design."""

    def __init__(self, settings: UIPerformanceSettings):
        self.settings = settings
        self._layouts = {}
        self._current_breakpoint = "desktop"

    def register_layout(self, name: str, widget: QWidget, layouts: Dict[str, Callable]):
        """Đăng ký layout responsive."""
        self._layouts[name] = {"widget": weakref.ref(widget), "layouts": layouts}

    def update_for_size(self, width: int, height: int):
        """Cập nhật layout cho kích thước mới."""
        new_breakpoint = self._get_breakpoint(width)

        if new_breakpoint != self._current_breakpoint:
            self._current_breakpoint = new_breakpoint
            self._apply_breakpoint_layouts(new_breakpoint)

    def _get_breakpoint(self, width: int) -> str:
        """Xác định breakpoint hiện tại."""
        if width < self.settings.breakpoint_mobile:
            return "mobile"
        elif width < self.settings.breakpoint_tablet:
            return "tablet"
        else:
            return "desktop"

    def _apply_breakpoint_layouts(self, breakpoint: str):
        """Áp dụng layout cho breakpoint."""
        for name, layout_info in self._layouts.items():
            widget = layout_info["widget"]()
            if widget and breakpoint in layout_info["layouts"]:
                try:
                    layout_info["layouts"][breakpoint](widget)
                    logger.debug(f"Applied {breakpoint} layout to {name}")
                except Exception as e:
                    logger.error(f"Error applying layout to {name}: {e}")


class PerformanceMonitor(QObject if PYQT5_AVAILABLE else object):
    """Monitor hiệu năng UI."""

    metrics_updated = pyqtSignal(object) if PYQT5_AVAILABLE else None

    def __init__(self, settings: UIPerformanceSettings):
        if PYQT5_AVAILABLE:
            super().__init__()
        self.settings = settings
        self.metrics = UIPerformanceMetrics()
        self._monitor_timer = QCoreTimer() if PYQT5_AVAILABLE else None
        self._start_time = time.time()

        if self._monitor_timer:
            self._monitor_timer.timeout.connect(self._update_metrics)
            if self.settings.enable_performance_monitoring:
                self._monitor_timer.start(self.settings.performance_log_interval_ms)

    def _update_metrics(self):
        """Cập nhật performance metrics."""
        try:
            # Memory usage
            import psutil

            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024

            # Calculate refresh rate (simplified)
            current_time = time.time()
            elapsed = current_time - self._start_time
            if elapsed > 0:
                self.metrics.refresh_rate_fps = 1.0 / elapsed

            self._start_time = current_time

            if self.metrics_updated:
                self.metrics_updated.emit(self.metrics)

            logger.debug(
                f"Performance: {self.metrics.memory_usage_mb:.1f}MB, "
                f"{self.metrics.refresh_rate_fps:.1f}FPS"
            )

        except Exception as e:
            logger.warning(f"Error updating performance metrics: {e}")


class MemoryOptimizedWidget(QWidget if PYQT5_AVAILABLE else object):
    """Base class cho memory-optimized widgets."""

    def __init__(self, parent=None):
        if PYQT5_AVAILABLE:
            super().__init__(parent)
        self._cached_pixmaps = {}
        self._cleanup_timer = QCoreTimer() if PYQT5_AVAILABLE else None

        if self._cleanup_timer:
            self._cleanup_timer.timeout.connect(self._cleanup_cache)
            self._cleanup_timer.start(30000)  # 30 seconds

    def cache_pixmap(self, key: str, pixmap: QPixmap):
        """Cache pixmap để tái sử dụng."""
        if PYQT5_AVAILABLE:
            self._cached_pixmaps[key] = pixmap

    def get_cached_pixmap(self, key: str) -> Optional[QPixmap]:
        """Lấy cached pixmap."""
        return self._cached_pixmaps.get(key) if PYQT5_AVAILABLE else None

    def _cleanup_cache(self):
        """Cleanup cache để giải phóng memory."""
        if len(self._cached_pixmaps) > 20:  # Keep only 20 recent pixmaps
            # Remove oldest entries
            items = list(self._cached_pixmaps.items())
            for key, _ in items[:-20]:
                del self._cached_pixmaps[key]

            logger.debug(
                f"Cleaned pixmap cache, {len(self._cached_pixmaps)} items remaining"
            )


class UIPerformanceManager:
    """Main manager cho UI performance."""

    def __init__(self, settings: Optional[UIPerformanceSettings] = None):
        self.settings = settings or UIPerformanceSettings()

        # Khởi tạo các managers
        self.lazy_manager = LazyLoadingManager(self.settings)
        self.progressive_renderer = ProgressiveRenderer(self.settings)
        self.responsive_manager = ResponsiveLayoutManager(self.settings)
        self.performance_monitor = PerformanceMonitor(self.settings)

        logger.info("Initialized UIPerformanceManager")

    def optimize_widget(self, widget: QWidget) -> QWidget:
        """Optimize widget với các techniques hiệu năng."""
        if not PYQT5_AVAILABLE:
            return widget

        start_time = time.time()

        # Apply memory optimizations
        if hasattr(widget, "setAttribute"):
            widget.setAttribute(77, True)  # WA_DeleteOnClose

        # Apply lazy loading if applicable
        if hasattr(widget, "setLazyLoading"):
            widget.setLazyLoading(True)

        optimization_time = time.time() - start_time
        logger.debug(f"Optimized widget in {optimization_time:.3f}s")

        return widget

    def create_optimized_layout(self, orientation: str = "vertical") -> QVBoxLayout:
        """Tạo optimized layout."""
        if not PYQT5_AVAILABLE:
            return None

        if orientation == "vertical":
            layout = QVBoxLayout()
        else:
            layout = QHBoxLayout()

        # Optimize layout settings
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        return layout

    def benchmark_ui_performance(self) -> Dict[str, Any]:
        """Benchmark UI performance."""
        logger.info("Running UI performance benchmark...")

        results = {}

        # Test widget creation time
        start_time = time.time()
        test_widgets = []
        for i in range(100):
            if PYQT5_AVAILABLE:
                widget = QLabel(f"Test {i}")
                test_widgets.append(widget)

        widget_creation_time = time.time() - start_time
        results["widget_creation_time"] = widget_creation_time

        # Test layout performance
        start_time = time.time()
        if PYQT5_AVAILABLE:
            layout = QVBoxLayout()
            for widget in test_widgets:
                layout.addWidget(widget)

        layout_time = time.time() - start_time
        results["layout_time"] = layout_time

        # Memory usage
        try:
            import psutil

            process = psutil.Process()
            results["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        except:
            results["memory_usage_mb"] = 0.0

        logger.info(
            f"UI benchmark completed: Widget creation: {widget_creation_time:.3f}s, "
            f"Layout: {layout_time:.3f}s, Memory: {results['memory_usage_mb']:.1f}MB"
        )

        return results


def create_performance_manager(
    enable_lazy_loading: bool = True,
    enable_progressive_rendering: bool = True,
    enable_memory_optimization: bool = True,
    enable_responsive_design: bool = True,
) -> UIPerformanceManager:
    """Factory function tạo UI performance manager."""

    settings = UIPerformanceSettings(
        enable_lazy_loading=enable_lazy_loading,
        enable_progressive_rendering=enable_progressive_rendering,
        enable_memory_optimization=enable_memory_optimization,
        enable_responsive_design=enable_responsive_design,
    )

    return UIPerformanceManager(settings)


# Test function
if __name__ == "__main__":
    print("Testing UI Performance...")

    manager = create_performance_manager()

    # Run benchmark
    benchmark_results = manager.benchmark_ui_performance()

    print("\nUI Performance Benchmark Results:")
    print("=" * 50)
    for metric, value in benchmark_results.items():
        if metric.endswith("_time"):
            print(f"{metric:25}: {value:.3f}s")
        elif metric.endswith("_mb"):
            print(f"{metric:25}: {value:.1f}MB")
        else:
            print(f"{metric:25}: {value}")
