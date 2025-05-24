"""
Chart Widgets Module

Cung cấp các widget chuyên dùng để hiển thị charts và đồ thị
trong giao diện QuangTPS.
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np

# PyQt5 imports với fallback
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QScrollArea,
        QGroupBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QColor

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

    # Fallback classes
    class QWidget:
        def __init__(self, parent=None):
            pass

    class pyqtSignal:
        def __init__(self, *args):
            pass


# Matplotlib imports với fallback
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

    class Figure:
        def __init__(self, *args, **kwargs):
            pass

    class FigureCanvas:
        def __init__(self, figure):
            pass


logger = logging.getLogger(__name__)


class PlanEvaluationWidget(QWidget if HAS_PYQT else object):
    """
    Widget đánh giá kế hoạch với charts và bảng số liệu.
    """

    def __init__(self, parent=None):
        """Khởi tạo PlanEvaluationWidget."""
        if HAS_PYQT:
            super().__init__(parent)
            self.setup_ui()
        else:
            logger.warning("PlanEvaluationWidget yêu cầu PyQt5")

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not HAS_PYQT:
            return

        layout = QVBoxLayout()

        # Title
        title = QLabel("Plan Evaluation")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Main content area
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Charts
        charts_widget = self.create_charts_panel()
        main_splitter.addWidget(charts_widget)

        # Right panel - Tables
        tables_widget = self.create_tables_panel()
        main_splitter.addWidget(tables_widget)

        main_splitter.setSizes([400, 300])
        layout.addWidget(main_splitter)

        self.setLayout(layout)

    def create_charts_panel(self):
        """Tạo panel chứa các chart."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Chart placeholder
        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(8, 6), facecolor="#2B2B2B")
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            placeholder = QLabel("Charts không khả dụng (thiếu matplotlib)")
            placeholder.setStyleSheet("color: #F5A623; padding: 20px;")
            layout.addWidget(placeholder)

        widget.setLayout(layout)
        return widget

    def create_tables_panel(self):
        """Tạo panel chứa bảng dữ liệu."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #3C3C3C;
                color: white;
                border: 1px solid #555555;
                gridline-color: #555555;
            }
            QHeaderView::section {
                background-color: #4A90E2;
                color: white;
                padding: 4px;
                border: 1px solid #555555;
            }
        """)
        layout.addWidget(self.table)

        # Initialize table with sample data
        self.setup_sample_table()

        widget.setLayout(layout)
        return widget

    def setup_sample_table(self):
        """Thiết lập bảng dữ liệu mẫu."""
        if not HAS_PYQT:
            return

        headers = [
            "Structure",
            "Volume (cc)",
            "Mean Dose (Gy)",
            "Max Dose (Gy)",
            "Status",
        ]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Sample data
        sample_data = [
            ["PTV", "150.5", "52.3", "55.8", "PASS"],
            ["Spinal Cord", "25.2", "8.5", "45.2", "PASS"],
            ["Heart", "320.8", "12.4", "28.9", "PASS"],
            ["Lung_L", "1250.3", "15.2", "38.7", "WARNING"],
            ["Lung_R", "1180.9", "14.8", "36.4", "PASS"],
        ]

        self.table.setRowCount(len(sample_data))

        for i, row_data in enumerate(sample_data):
            for j, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))

                # Color coding for status
                if j == 4:  # Status column
                    if value == "PASS":
                        item.setBackground(QColor("#4CAF50"))
                    elif value == "WARNING":
                        item.setBackground(QColor("#FF9800"))
                    elif value == "FAIL":
                        item.setBackground(QColor("#F44336"))

                self.table.setItem(i, j, item)

        # Adjust column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

    def update_plan_data(self, plan_data: Dict[str, Any]):
        """Cập nhật dữ liệu kế hoạch."""
        if not HAS_PYQT:
            return

        logger.info(f"Updating plan evaluation with data: {len(plan_data)} structures")

        # Update charts if available
        if HAS_MATPLOTLIB and hasattr(self, "figure"):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#3C3C3C")

            # Sample DVH plot
            doses = np.linspace(0, 60, 100)
            for i, (structure, data) in enumerate(plan_data.items()):
                volume_fraction = np.exp(-doses / (20 + i * 5))
                ax.plot(doses, volume_fraction * 100, label=structure, linewidth=2)

            ax.set_xlabel("Dose (Gy)", color="white")
            ax.set_ylabel("Volume (%)", color="white")
            ax.set_title("Dose Volume Histogram", color="white")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.tick_params(colors="white")

            self.canvas.draw()

        # Update table data
        self.update_table_data(plan_data)

    def update_table_data(self, plan_data: Dict[str, Any]):
        """Cập nhật dữ liệu bảng."""
        if not HAS_PYQT or not plan_data:
            return

        self.table.setRowCount(len(plan_data))

        for i, (structure, data) in enumerate(plan_data.items()):
            # Structure name
            self.table.setItem(i, 0, QTableWidgetItem(structure))

            # Volume
            volume = data.get("volume", 0)
            self.table.setItem(i, 1, QTableWidgetItem(f"{volume:.1f}"))

            # Mean dose
            mean_dose = data.get("mean_dose", 0)
            self.table.setItem(i, 2, QTableWidgetItem(f"{mean_dose:.1f}"))

            # Max dose
            max_dose = data.get("max_dose", 0)
            self.table.setItem(i, 3, QTableWidgetItem(f"{max_dose:.1f}"))

            # Status
            status = data.get("status", "UNKNOWN")
            status_item = QTableWidgetItem(status)

            if status == "PASS":
                status_item.setBackground(QColor("#4CAF50"))
            elif status == "WARNING":
                status_item.setBackground(QColor("#FF9800"))
            elif status == "FAIL":
                status_item.setBackground(QColor("#F44336"))

            self.table.setItem(i, 4, status_item)


class DVHChartWidget(QWidget if HAS_PYQT else object):
    """
    Widget chuyên dụng để hiển thị Dose Volume Histogram.
    """

    def __init__(self, parent=None):
        """Khởi tạo DVHChartWidget."""
        if HAS_PYQT:
            super().__init__(parent)
            self.setup_ui()
            self.dvh_data = {}
        else:
            logger.warning("DVHChartWidget yêu cầu PyQt5")

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not HAS_PYQT:
            return

        layout = QVBoxLayout()

        # Title
        title = QLabel("Dose Volume Histogram")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Chart area
        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(10, 6), facecolor="#2B2B2B")
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)

            # Initialize empty plot
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor("#3C3C3C")
            self.setup_plot_style()

        else:
            placeholder = QLabel("DVH Chart không khả dụng (thiếu matplotlib)")
            placeholder.setStyleSheet("color: #F5A623; padding: 20px;")
            layout.addWidget(placeholder)

        # Control buttons
        button_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5BA0F2;
            }
        """)
        button_layout.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #666666;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #777777;
            }
        """)
        button_layout.addWidget(self.export_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        if hasattr(self, "refresh_btn"):
            self.refresh_btn.clicked.connect(self.refresh_plot)
            self.export_btn.clicked.connect(self.export_plot)

    def setup_plot_style(self):
        """Thiết lập style cho plot."""
        if not HAS_MATPLOTLIB:
            return

        self.ax.set_xlabel("Dose (Gy)", color="white", fontsize=12)
        self.ax.set_ylabel("Volume (%)", color="white", fontsize=12)
        self.ax.set_title("Dose Volume Histogram", color="white", fontsize=14)
        self.ax.grid(True, alpha=0.3, color="white")
        self.ax.tick_params(colors="white")

        # Set background
        self.ax.set_facecolor("#3C3C3C")
        self.figure.patch.set_facecolor("#2B2B2B")

    def add_dvh_curve(
        self,
        structure_name: str,
        doses: np.ndarray,
        volumes: np.ndarray,
        color: str = None,
    ):
        """Thêm đường cong DVH."""
        if not HAS_MATPLOTLIB:
            return

        self.dvh_data[structure_name] = {
            "doses": doses,
            "volumes": volumes,
            "color": color,
        }

        self.refresh_plot()

    def refresh_plot(self):
        """Làm mới plot."""
        if not HAS_MATPLOTLIB or not hasattr(self, "ax"):
            return

        self.ax.clear()
        self.setup_plot_style()

        # Plot all DVH curves
        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#98D8C8",
        ]

        for i, (structure, data) in enumerate(self.dvh_data.items()):
            color = data.get("color", colors[i % len(colors)])
            self.ax.plot(
                data["doses"],
                data["volumes"],
                label=structure,
                linewidth=2,
                color=color,
            )

        if self.dvh_data:
            self.ax.legend(loc="upper right", facecolor="#3C3C3C", edgecolor="white")
            self.ax.set_xlim(
                0, max([max(data["doses"]) for data in self.dvh_data.values()])
            )
            self.ax.set_ylim(0, 100)

        self.canvas.draw()

    def export_plot(self):
        """Xuất plot ra file."""
        if not HAS_MATPLOTLIB:
            return

        try:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                self.figure.savefig(tmp.name, facecolor="#2B2B2B", dpi=300)
                logger.info(f"DVH plot exported to {tmp.name}")

        except Exception as e:
            logger.error(f"Failed to export DVH plot: {e}")

    def clear_data(self):
        """Xóa tất cả dữ liệu DVH."""
        self.dvh_data.clear()
        self.refresh_plot()


class MetricsTableWidget(QWidget if HAS_PYQT else object):
    """
    Widget hiển thị bảng các metrics đánh giá kế hoạch.
    """

    def __init__(self, parent=None):
        """Khởi tạo MetricsTableWidget."""
        if HAS_PYQT:
            super().__init__(parent)
            self.setup_ui()
        else:
            logger.warning("MetricsTableWidget yêu cầu PyQt5")

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not HAS_PYQT:
            return

        layout = QVBoxLayout()

        # Title
        title = QLabel("Plan Quality Metrics")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Table
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #3C3C3C;
                color: white;
                border: 1px solid #555555;
                gridline-color: #555555;
                selection-background-color: #4A90E2;
            }
            QHeaderView::section {
                background-color: #4A90E2;
                color: white;
                padding: 6px;
                border: 1px solid #555555;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)

        # Setup table headers
        headers = ["Metric", "Value", "Target", "Status", "Deviation"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Adjust column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Metric name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Value
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Target
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Deviation

        layout.addWidget(self.table)
        self.setLayout(layout)

        # Add sample data
        self.load_sample_metrics()

    def load_sample_metrics(self):
        """Tải dữ liệu metrics mẫu."""
        sample_metrics = [
            {
                "metric": "PTV Coverage (V95%)",
                "value": 98.5,
                "target": ">95",
                "status": "PASS",
                "deviation": 3.5,
            },
            {
                "metric": "PTV Conformity Index",
                "value": 0.92,
                "target": ">0.9",
                "status": "PASS",
                "deviation": 0.02,
            },
            {
                "metric": "PTV Homogeneity Index",
                "value": 0.08,
                "target": "<0.1",
                "status": "PASS",
                "deviation": -0.02,
            },
            {
                "metric": "Spinal Cord Max Dose",
                "value": 45.2,
                "target": "<46",
                "status": "PASS",
                "deviation": -0.8,
            },
            {
                "metric": "Heart Mean Dose",
                "value": 12.4,
                "target": "<15",
                "status": "PASS",
                "deviation": -2.6,
            },
            {
                "metric": "Lung V20Gy",
                "value": 28.5,
                "target": "<30",
                "status": "PASS",
                "deviation": -1.5,
            },
        ]

        self.update_metrics(sample_metrics)

    def update_metrics(self, metrics: List[Dict[str, Any]]):
        """Cập nhật bảng metrics."""
        if not HAS_PYQT or not metrics:
            return

        self.table.setRowCount(len(metrics))

        for i, metric in enumerate(metrics):
            # Metric name
            self.table.setItem(i, 0, QTableWidgetItem(metric.get("metric", "")))

            # Value
            value = metric.get("value", 0)
            if isinstance(value, float):
                self.table.setItem(i, 1, QTableWidgetItem(f"{value:.2f}"))
            else:
                self.table.setItem(i, 1, QTableWidgetItem(str(value)))

            # Target
            self.table.setItem(i, 2, QTableWidgetItem(metric.get("target", "")))

            # Status with color coding
            status = metric.get("status", "UNKNOWN")
            status_item = QTableWidgetItem(status)

            if status == "PASS":
                status_item.setBackground(QColor("#4CAF50"))
            elif status == "WARNING":
                status_item.setBackground(QColor("#FF9800"))
            elif status == "FAIL":
                status_item.setBackground(QColor("#F44336"))

            self.table.setItem(i, 3, status_item)

            # Deviation
            deviation = metric.get("deviation", 0)
            if isinstance(deviation, (int, float)):
                deviation_item = QTableWidgetItem(f"{deviation:+.2f}")
                if deviation > 0:
                    deviation_item.setBackground(
                        QColor("#FFEB3B")
                    )  # Yellow for positive deviation
                elif deviation < 0:
                    deviation_item.setBackground(
                        QColor("#4CAF50")
                    )  # Green for negative deviation (better)
            else:
                deviation_item = QTableWidgetItem(str(deviation))

            self.table.setItem(i, 4, deviation_item)


# Factory functions
def create_plan_evaluation_widget(parent=None) -> PlanEvaluationWidget:
    """Tạo PlanEvaluationWidget."""
    return PlanEvaluationWidget(parent)


def create_dvh_chart_widget(parent=None) -> DVHChartWidget:
    """Tạo DVHChartWidget."""
    return DVHChartWidget(parent)


def create_metrics_table_widget(parent=None) -> MetricsTableWidget:
    """Tạo MetricsTableWidget."""
    return MetricsTableWidget(parent)


# Export all classes and functions
__all__ = [
    "PlanEvaluationWidget",
    "DVHChartWidget",
    "MetricsTableWidget",
    "create_plan_evaluation_widget",
    "create_dvh_chart_widget",
    "create_metrics_table_widget",
]
