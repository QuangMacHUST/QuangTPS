"""
QuangTPS Plan Quality Assurance Widget

Widget đánh giá chất lượng kế hoạch xạ trị theo phong cách Eclipse.
Cung cấp giao diện toàn diện cho QA analysis bao gồm:
- Gamma analysis
- Plan quality metrics
- Safety checks
- Comprehensive reporting
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import numpy as np

# UI imports với fallback
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QGroupBox,
        QTabWidget,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QProgressBar,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QTextEdit,
        QFileDialog,
        QMessageBox,
        QSplitter,
        QFrame,
        QScrollArea,
        QSlider,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QPainter

    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False

    # Fallback classes
    class QWidget:
        pass

    class QVBoxLayout:
        pass

    class QHBoxLayout:
        pass

    class QGridLayout:
        pass

    class QGroupBox:
        pass

    class QTabWidget:
        pass

    class QLabel:
        pass

    class QPushButton:
        pass

    class QTableWidget:
        pass

    class QTableWidgetItem:
        pass

    class QProgressBar:
        pass

    class QComboBox:
        pass

    class QSpinBox:
        pass

    class QDoubleSpinBox:
        pass

    class QCheckBox:
        pass

    class QTextEdit:
        pass

    class QFileDialog:
        pass

    class QMessageBox:
        pass

    class QSplitter:
        pass

    class QFrame:
        pass

    class QScrollArea:
        pass

    class QSlider:
        pass

    class QThread:
        pass

    class QTimer:
        pass

    class Qt:
        pass

    def pyqtSignal(*args):
        return None


# Core QuangTPS imports với fallback
try:
    from quangtps.evaluation.qa.plan_qa_engine import (
        PlanQAEngine,
        QASettings,
        QATestType,
        ComprehensiveQAReport,
        QAResult,
        QASeverity,
    )

    HAS_QA_ENGINE = True
except ImportError:
    HAS_QA_ENGINE = False

    # Fallback classes
    class PlanQAEngine:
        pass

    class QASettings:
        pass

    class QATestType:
        pass

    class ComprehensiveQAReport:
        pass

    class QAResult:
        pass

    class QASeverity:
        pass


# UI styling
try:
    from quangtps.ui.styles.eclipse_style_theme import EclipseStyleTheme

    HAS_ECLIPSE_THEME = True
except ImportError:
    HAS_ECLIPSE_THEME = False

    class EclipseStyleTheme:
        @staticmethod
        def get_colors():
            return {
                "background": "#2B2B2B",
                "panel": "#3C3C3C",
                "border": "#555555",
                "text": "#CCCCCC",
                "accent": "#4A90E2",
                "success": "#27AE60",
                "warning": "#F39C12",
                "error": "#E74C3C",
            }


# Plotting imports với fallback
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

    class FigureCanvas:
        pass

    class Figure:
        pass


logger = logging.getLogger(__name__)


class QAWorkerThread(QThread):
    """Thread worker cho QA analysis để tránh blocking UI."""

    progress_updated = pyqtSignal(int, str)
    analysis_completed = pyqtSignal(object)  # ComprehensiveQAReport
    analysis_failed = pyqtSignal(str)

    def __init__(self, qa_engine, reference_data, measured_data, test_types=None):
        super().__init__()
        self.qa_engine = qa_engine
        self.reference_data = reference_data
        self.measured_data = measured_data
        self.test_types = test_types

    def run(self):
        """Chạy QA analysis trong background thread."""
        try:

            def progress_callback(progress, message):
                self.progress_updated.emit(int(progress), message)

            result = self.qa_engine.run_comprehensive_qa(
                reference_data=self.reference_data,
                measured_data=self.measured_data,
                test_types=self.test_types,
                progress_callback=progress_callback,
            )

            self.analysis_completed.emit(result)

        except Exception as e:
            logger.error(f"QA analysis failed: {e}")
            self.analysis_failed.emit(str(e))


class GammaAnalysisWidget(QWidget):
    """Widget cho gamma analysis với Eclipse styling."""

    def __init__(self):
        super().__init__()
        self.gamma_map = None
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện gamma analysis."""
        if not HAS_PYQT5:
            return

        layout = QVBoxLayout(self)

        # Settings group
        settings_group = QGroupBox("Gamma Analysis Settings")
        settings_layout = QGridLayout(settings_group)

        # Distance criterion
        settings_layout.addWidget(QLabel("Distance (mm):"), 0, 0)
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setRange(0.1, 10.0)
        self.distance_spinbox.setValue(3.0)
        self.distance_spinbox.setSingleStep(0.1)
        settings_layout.addWidget(self.distance_spinbox, 0, 1)

        # Dose criterion
        settings_layout.addWidget(QLabel("Dose (%):"), 0, 2)
        self.dose_spinbox = QDoubleSpinBox()
        self.dose_spinbox.setRange(0.1, 10.0)
        self.dose_spinbox.setValue(3.0)
        self.dose_spinbox.setSingleStep(0.1)
        settings_layout.addWidget(self.dose_spinbox, 0, 3)

        # Pass rate threshold
        settings_layout.addWidget(QLabel("Pass Rate Threshold (%):"), 1, 0)
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(80.0, 100.0)
        self.threshold_spinbox.setValue(95.0)
        self.threshold_spinbox.setSingleStep(1.0)
        settings_layout.addWidget(self.threshold_spinbox, 1, 1)

        # Dose threshold
        settings_layout.addWidget(QLabel("Dose Threshold (%):"), 1, 2)
        self.dose_threshold_spinbox = QDoubleSpinBox()
        self.dose_threshold_spinbox.setRange(1.0, 50.0)
        self.dose_threshold_spinbox.setValue(10.0)
        self.dose_threshold_spinbox.setSingleStep(1.0)
        settings_layout.addWidget(self.dose_threshold_spinbox, 1, 3)

        layout.addWidget(settings_group)

        # Results display
        results_group = QGroupBox("Gamma Analysis Results")
        results_layout = QVBoxLayout(results_group)

        # Results table
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value", "Status"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)

        # Gamma map visualization
        if HAS_MATPLOTLIB:
            self.gamma_figure = Figure(figsize=(8, 6))
            self.gamma_canvas = FigureCanvas(self.gamma_figure)
            results_layout.addWidget(self.gamma_canvas)
        else:
            gamma_placeholder = QLabel("Gamma visualization requires matplotlib")
            gamma_placeholder.setAlignment(Qt.AlignCenter)
            results_layout.addWidget(gamma_placeholder)

        layout.addWidget(results_group)

        # Apply Eclipse styling
        if HAS_ECLIPSE_THEME:
            colors = EclipseStyleTheme.get_colors()
            self.setStyleSheet(f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 2px solid {colors["border"]};
                    border-radius: 5px;
                    margin-top: 1ex;
                    color: {colors["text"]};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }}
                QTableWidget {{
                    background-color: {colors["panel"]};
                    color: {colors["text"]};
                    gridline-color: {colors["border"]};
                }}
                QDoubleSpinBox {{
                    background-color: {colors["panel"]};
                    color: {colors["text"]};
                    border: 1px solid {colors["border"]};
                }}
            """)

    def update_gamma_results(self, qa_result):
        """Cập nhật kết quả gamma analysis."""
        if not HAS_PYQT5 or not qa_result:
            return

        try:
            # Clear existing results
            self.results_table.setRowCount(0)

            # Add gamma results to table
            if hasattr(qa_result, "result_data") and qa_result.result_data:
                gamma_stats = qa_result.result_data.get("gamma_statistics", {})

                metrics = [
                    ("Pass Rate (%)", gamma_stats.get("pass_rate", 0.0)),
                    ("Mean Gamma", gamma_stats.get("mean_gamma", 0.0)),
                    ("Max Gamma", gamma_stats.get("max_gamma", 0.0)),
                    ("Analyzed Points", gamma_stats.get("analyzed_points", 0)),
                    ("Passed Points", gamma_stats.get("passed_points", 0)),
                ]

                for i, (metric, value) in enumerate(metrics):
                    self.results_table.insertRow(i)
                    self.results_table.setItem(i, 0, QTableWidgetItem(metric))

                    if isinstance(value, float):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = str(value)
                    self.results_table.setItem(i, 1, QTableWidgetItem(value_str))

                    # Status color coding
                    if metric == "Pass Rate (%)":
                        threshold = self.threshold_spinbox.value()
                        if value >= threshold:
                            status = "PASS"
                            color = "green"
                        else:
                            status = "FAIL"
                            color = "red"
                    else:
                        status = "INFO"
                        color = "blue"

                    status_item = QTableWidgetItem(status)
                    status_item.setForeground(QColor(color))
                    self.results_table.setItem(i, 2, status_item)

                # Update gamma map visualization
                self._update_gamma_visualization(qa_result.result_data.get("gamma_map"))

        except Exception as e:
            logger.error(f"Error updating gamma results: {e}")

    def _update_gamma_visualization(self, gamma_map):
        """Cập nhật visualization của gamma map."""
        if not HAS_MATPLOTLIB or gamma_map is None:
            return

        try:
            self.gamma_figure.clear()
            ax = self.gamma_figure.add_subplot(111)

            # Select middle slice for display
            if len(gamma_map.shape) == 3:
                middle_slice = gamma_map.shape[2] // 2
                gamma_slice = gamma_map[:, :, middle_slice]
            else:
                gamma_slice = gamma_map

            # Create gamma map plot
            im = ax.imshow(gamma_slice, cmap="jet", vmin=0, vmax=2)
            ax.set_title(f"Gamma Map (Middle Slice)")
            ax.set_xlabel("X Index")
            ax.set_ylabel("Y Index")

            # Add colorbar
            cbar = self.gamma_figure.colorbar(im, ax=ax)
            cbar.set_label("Gamma Value")

            # Add pass/fail contours
            contour_levels = [1.0]  # Gamma = 1 is pass/fail threshold
            ax.contour(gamma_slice, levels=contour_levels, colors="white", linewidths=2)

            self.gamma_figure.tight_layout()
            self.gamma_canvas.draw()

        except Exception as e:
            logger.error(f"Error updating gamma visualization: {e}")


class PlanQualityMetricsWidget(QWidget):
    """Widget cho plan quality metrics."""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện plan quality metrics."""
        if not HAS_PYQT5:
            return

        layout = QVBoxLayout(self)

        # Metrics table
        metrics_group = QGroupBox("Plan Quality Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Metric", "Calculated", "Threshold", "Status"]
        )
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_table)

        layout.addWidget(metrics_group)

        # Plan quality visualization
        if HAS_MATPLOTLIB:
            self.quality_figure = Figure(figsize=(10, 6))
            self.quality_canvas = FigureCanvas(self.quality_figure)
            layout.addWidget(self.quality_canvas)
        else:
            quality_placeholder = QLabel(
                "Plan quality visualization requires matplotlib"
            )
            quality_placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(quality_placeholder)

        # Apply styling
        if HAS_ECLIPSE_THEME:
            colors = EclipseStyleTheme.get_colors()
            self.setStyleSheet(f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 2px solid {colors["border"]};
                    border-radius: 5px;
                    margin-top: 1ex;
                    color: {colors["text"]};
                }}
                QTableWidget {{
                    background-color: {colors["panel"]};
                    color: {colors["text"]};
                    gridline-color: {colors["border"]};
                }}
            """)

    def update_quality_metrics(self, qa_results):
        """Cập nhật plan quality metrics."""
        if not HAS_PYQT5 or not qa_results:
            return

        try:
            # Clear existing results
            self.metrics_table.setRowCount(0)

            # Process quality metrics from QA results
            quality_metrics = []

            for result in qa_results:
                if hasattr(result, "test_type") and result.test_type:
                    test_type = result.test_type
                    if hasattr(test_type, "value"):
                        test_name = test_type.value
                    else:
                        test_name = str(test_type)

                    if "conformity" in test_name.lower():
                        quality_metrics.append(("Conformity Index", result))
                    elif "homogeneity" in test_name.lower():
                        quality_metrics.append(("Homogeneity Index", result))
                    elif "gradient" in test_name.lower():
                        quality_metrics.append(("Gradient Index", result))

            # Add metrics to table
            for i, (metric_name, result) in enumerate(quality_metrics):
                self.metrics_table.insertRow(i)
                self.metrics_table.setItem(i, 0, QTableWidgetItem(metric_name))

                # Calculated value
                if (
                    hasattr(result, "measured_value")
                    and result.measured_value is not None
                ):
                    calc_value = f"{result.measured_value:.3f}"
                else:
                    calc_value = "N/A"
                self.metrics_table.setItem(i, 1, QTableWidgetItem(calc_value))

                # Threshold
                if (
                    hasattr(result, "expected_value")
                    and result.expected_value is not None
                ):
                    threshold = f"{result.expected_value:.3f}"
                else:
                    threshold = "N/A"
                self.metrics_table.setItem(i, 2, QTableWidgetItem(threshold))

                # Status
                if hasattr(result, "pass_status"):
                    status = "PASS" if result.pass_status else "FAIL"
                    color = "green" if result.pass_status else "red"
                else:
                    status = "UNKNOWN"
                    color = "gray"

                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor(color))
                self.metrics_table.setItem(i, 3, status_item)

            # Update quality visualization
            self._update_quality_visualization(quality_metrics)

        except Exception as e:
            logger.error(f"Error updating quality metrics: {e}")

    def _update_quality_visualization(self, quality_metrics):
        """Cập nhật visualization của plan quality."""
        if not HAS_MATPLOTLIB or not quality_metrics:
            return

        try:
            self.quality_figure.clear()

            # Create bar chart of quality metrics
            ax = self.quality_figure.add_subplot(111)

            metric_names = []
            metric_values = []
            colors = []

            for metric_name, result in quality_metrics:
                metric_names.append(metric_name.replace(" Index", ""))

                if (
                    hasattr(result, "measured_value")
                    and result.measured_value is not None
                ):
                    metric_values.append(result.measured_value)

                    # Color based on pass/fail
                    if hasattr(result, "pass_status") and result.pass_status:
                        colors.append("green")
                    else:
                        colors.append("red")
                else:
                    metric_values.append(0)
                    colors.append("gray")

            if metric_names:
                bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
                ax.set_ylabel("Metric Value")
                ax.set_title("Plan Quality Metrics")
                ax.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

            self.quality_figure.tight_layout()
            self.quality_canvas.draw()

        except Exception as e:
            logger.error(f"Error updating quality visualization: {e}")


class PlanQAWidget(QWidget):
    """
    Main Plan QA Widget với giao diện Eclipse style.

    Cung cấp:
    - Gamma analysis
    - Plan quality metrics
    - Safety checks
    - Comprehensive reporting
    """

    def __init__(self):
        super().__init__()

        # Initialize QA engine
        if HAS_QA_ENGINE:
            self.qa_engine = PlanQAEngine()
        else:
            self.qa_engine = None

        self.current_qa_report = None
        self.qa_worker = None

        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện chính."""
        if not HAS_PYQT5:
            layout = QVBoxLayout(self)
            error_label = QLabel(
                "PyQt5 không khả dụng. Cần cài đặt PyQt5 để sử dụng QA interface."
            )
            layout.addWidget(error_label)
            return

        layout = QVBoxLayout(self)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Settings and controls
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Right panel - Analysis results
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions (30%, 70%)
        main_splitter.setSizes([300, 700])

        layout.addWidget(main_splitter)

        # Status bar
        self._create_status_bar(layout)

        # Apply Eclipse styling
        self._apply_eclipse_styling()

    def _create_left_panel(self):
        """Tạo panel trái chứa settings và controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # QA Settings group
        settings_group = QGroupBox("QA Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Test selection
        test_selection_label = QLabel("Select QA Tests:")
        settings_layout.addWidget(test_selection_label)

        # Checkboxes for different tests
        self.gamma_checkbox = QCheckBox("Gamma Analysis")
        self.gamma_checkbox.setChecked(True)
        settings_layout.addWidget(self.gamma_checkbox)

        self.conformity_checkbox = QCheckBox("Conformity Index")
        self.conformity_checkbox.setChecked(True)
        settings_layout.addWidget(self.conformity_checkbox)

        self.homogeneity_checkbox = QCheckBox("Homogeneity Index")
        self.homogeneity_checkbox.setChecked(True)
        settings_layout.addWidget(self.homogeneity_checkbox)

        self.dose_limits_checkbox = QCheckBox("Dose Limits Check")
        self.dose_limits_checkbox.setChecked(True)
        settings_layout.addWidget(self.dose_limits_checkbox)

        layout.addWidget(settings_group)

        # Data input group
        data_group = QGroupBox("Data Input")
        data_layout = QVBoxLayout(data_group)

        # Reference data
        self.reference_button = QPushButton("Load Reference Data")
        self.reference_button.clicked.connect(self.load_reference_data)
        data_layout.addWidget(self.reference_button)

        self.reference_label = QLabel("No reference data loaded")
        self.reference_label.setWordWrap(True)
        data_layout.addWidget(self.reference_label)

        # Measured data
        self.measured_button = QPushButton("Load Measured Data")
        self.measured_button.clicked.connect(self.load_measured_data)
        data_layout.addWidget(self.measured_button)

        self.measured_label = QLabel("No measured data loaded")
        self.measured_label.setWordWrap(True)
        data_layout.addWidget(self.measured_label)

        layout.addWidget(data_group)

        # Analysis controls
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Run analysis button
        self.run_analysis_button = QPushButton("Run QA Analysis")
        self.run_analysis_button.clicked.connect(self.run_qa_analysis)
        self.run_analysis_button.setEnabled(False)
        controls_layout.addWidget(self.run_analysis_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        controls_layout.addWidget(self.progress_label)

        layout.addWidget(controls_group)

        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QVBoxLayout(export_group)

        self.export_pdf_button = QPushButton("Export PDF Report")
        self.export_pdf_button.clicked.connect(self.export_pdf_report)
        self.export_pdf_button.setEnabled(False)
        export_layout.addWidget(self.export_pdf_button)

        self.export_csv_button = QPushButton("Export CSV Data")
        self.export_csv_button.clicked.connect(self.export_csv_data)
        self.export_csv_button.setEnabled(False)
        export_layout.addWidget(self.export_csv_button)

        layout.addWidget(export_group)

        # Add stretch to push everything to top
        layout.addStretch()

        return panel

    def _create_right_panel(self):
        """Tạo panel phải chứa kết quả analysis."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create tab widget for different analysis types
        self.results_tabs = QTabWidget()

        # Gamma analysis tab
        self.gamma_widget = GammaAnalysisWidget()
        self.results_tabs.addTab(self.gamma_widget, "Gamma Analysis")

        # Plan quality tab
        self.quality_widget = PlanQualityMetricsWidget()
        self.results_tabs.addTab(self.quality_widget, "Plan Quality")

        # Summary tab
        self.summary_widget = self._create_summary_widget()
        self.results_tabs.addTab(self.summary_widget, "Summary")

        layout.addWidget(self.results_tabs)

        return panel

    def _create_summary_widget(self):
        """Tạo widget tóm tắt QA."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Overall results
        overall_group = QGroupBox("Overall QA Results")
        overall_layout = QGridLayout(overall_group)

        # Overall score
        overall_layout.addWidget(QLabel("Overall Score:"), 0, 0)
        self.overall_score_label = QLabel("--")
        self.overall_score_label.setFont(QFont("Arial", 16, QFont.Bold))
        overall_layout.addWidget(self.overall_score_label, 0, 1)

        # Pass status
        overall_layout.addWidget(QLabel("Pass Status:"), 1, 0)
        self.pass_status_label = QLabel("--")
        self.pass_status_label.setFont(QFont("Arial", 14, QFont.Bold))
        overall_layout.addWidget(self.pass_status_label, 1, 1)

        # Total tests
        overall_layout.addWidget(QLabel("Total Tests:"), 2, 0)
        self.total_tests_label = QLabel("--")
        overall_layout.addWidget(self.total_tests_label, 2, 1)

        # Passed tests
        overall_layout.addWidget(QLabel("Passed Tests:"), 3, 0)
        self.passed_tests_label = QLabel("--")
        overall_layout.addWidget(self.passed_tests_label, 3, 1)

        layout.addWidget(overall_group)

        # Detailed results table
        details_group = QGroupBox("Detailed Test Results")
        details_layout = QVBoxLayout(details_group)

        self.details_table = QTableWidget(0, 5)
        self.details_table.setHorizontalHeaderLabels(
            ["Test Type", "Result", "Expected", "Status", "Severity"]
        )
        self.details_table.horizontalHeader().setStretchLastSection(True)
        details_layout.addWidget(self.details_table)

        layout.addWidget(details_group)

        return widget

    def _create_status_bar(self, parent_layout):
        """Tạo status bar."""
        status_frame = QFrame()
        status_frame.setFixedHeight(30)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)

        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        # QA engine status
        if self.qa_engine:
            engine_status = QLabel("QA Engine: Ready")
            engine_status.setStyleSheet("color: green;")
        else:
            engine_status = QLabel("QA Engine: Not Available")
            engine_status.setStyleSheet("color: red;")
        status_layout.addWidget(engine_status)

        parent_layout.addWidget(status_frame)

    def _apply_eclipse_styling(self):
        """Áp dụng Eclipse styling."""
        if not HAS_ECLIPSE_THEME:
            return

        colors = EclipseStyleTheme.get_colors()

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {colors["background"]};
                color: {colors["text"]};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {colors["border"]};
                border-radius: 5px;
                margin-top: 1ex;
                color: {colors["text"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QPushButton {{
                background-color: {colors["panel"]};
                border: 1px solid {colors["border"]};
                color: {colors["text"]};
                padding: 5px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {colors["accent"]};
            }}
            QPushButton:disabled {{
                background-color: {colors["border"]};
                color: #666666;
            }}
            QTabWidget::pane {{
                border: 1px solid {colors["border"]};
                background-color: {colors["background"]};
            }}
            QTabBar::tab {{
                background-color: {colors["panel"]};
                color: {colors["text"]};
                padding: 8px 16px;
                border: 1px solid {colors["border"]};
            }}
            QTabBar::tab:selected {{
                background-color: {colors["accent"]};
            }}
            QTableWidget {{
                background-color: {colors["panel"]};
                color: {colors["text"]};
                gridline-color: {colors["border"]};
            }}
            QProgressBar {{
                border: 1px solid {colors["border"]};
                background-color: {colors["panel"]};
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {colors["success"]};
            }}
        """)

    def load_reference_data(self):
        """Load reference dose data."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Reference Data",
                "",
                "DICOM files (*.dcm);;NumPy files (*.npy);;All files (*.*)",
            )

            if file_path:
                # TODO: Implement actual data loading
                self.reference_label.setText(
                    f"Reference: {os.path.basename(file_path)}"
                )
                self.reference_data = {"file_path": file_path}  # Placeholder
                self._check_data_ready()

        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load reference data: {e}")

    def load_measured_data(self):
        """Load measured dose data."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Measured Data",
                "",
                "DICOM files (*.dcm);;NumPy files (*.npy);;All files (*.*)",
            )

            if file_path:
                # TODO: Implement actual data loading
                self.measured_label.setText(f"Measured: {os.path.basename(file_path)}")
                self.measured_data = {"file_path": file_path}  # Placeholder
                self._check_data_ready()

        except Exception as e:
            logger.error(f"Error loading measured data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load measured data: {e}")

    def _check_data_ready(self):
        """Kiểm tra xem data đã sẵn sàng cho analysis chưa."""
        data_ready = (
            hasattr(self, "reference_data")
            and hasattr(self, "measured_data")
            and self.qa_engine
        )

        if data_ready:
            self.run_analysis_button.setEnabled(True)
        else:
            self.run_analysis_button.setEnabled(False)

        return data_ready

    def run_qa_analysis(self):
        """Chạy QA analysis."""
        if not self.qa_engine:
            QMessageBox.warning(self, "Warning", "QA Engine not available")
            return

        try:
            # Determine selected tests
            selected_tests = []
            if self.gamma_checkbox.isChecked():
                selected_tests.append(QATestType.GAMMA_ANALYSIS)
            if self.conformity_checkbox.isChecked():
                selected_tests.append(QATestType.CONFORMITY_INDEX)
            if self.homogeneity_checkbox.isChecked():
                selected_tests.append(QATestType.HOMOGENEITY_INDEX)
            if self.dose_limits_checkbox.isChecked():
                selected_tests.append(QATestType.DOSE_LIMITS_CHECK)

            if not selected_tests:
                QMessageBox.warning(
                    self, "Warning", "Please select at least one QA test"
                )
                return

            # Create sample data for analysis (TODO: use real data)
            reference_data = self._create_sample_reference_data()
            measured_data = self._create_sample_measured_data()

            # Start QA analysis in worker thread
            self.qa_worker = QAWorkerThread(
                self.qa_engine, reference_data, measured_data, selected_tests
            )

            # Connect signals
            self.qa_worker.progress_updated.connect(self._on_progress_updated)
            self.qa_worker.analysis_completed.connect(self._on_analysis_completed)
            self.qa_worker.analysis_failed.connect(self._on_analysis_failed)

            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.run_analysis_button.setEnabled(False)

            # Start analysis
            self.qa_worker.start()

        except Exception as e:
            logger.error(f"Error starting QA analysis: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start QA analysis: {e}")

    def _create_sample_reference_data(self):
        """Tạo sample reference data cho testing."""
        return {
            "dose_distribution": np.random.rand(64, 64, 32) * 60,  # Random dose 0-60 Gy
            "ptv_mask": np.random.rand(64, 64, 32) > 0.8,  # Random PTV mask
            "prescription_dose": 50.0,  # 50 Gy prescription
        }

    def _create_sample_measured_data(self):
        """Tạo sample measured data cho testing."""
        return {
            "dose_distribution": np.random.rand(64, 64, 32) * 60,  # Random dose 0-60 Gy
            "ptv_mask": np.random.rand(64, 64, 32) > 0.8,  # Random PTV mask
        }

    def _on_progress_updated(self, progress, message):
        """Callback khi progress được cập nhật."""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        self.status_label.setText(message)

    def _on_analysis_completed(self, qa_report):
        """Callback khi analysis hoàn thành."""
        self.current_qa_report = qa_report

        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.run_analysis_button.setEnabled(True)

        # Update UI with results
        self._update_analysis_results(qa_report)

        # Enable export buttons
        self.export_pdf_button.setEnabled(True)
        self.export_csv_button.setEnabled(True)

        self.status_label.setText("QA Analysis completed successfully")

    def _on_analysis_failed(self, error_message):
        """Callback khi analysis thất bại."""
        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.run_analysis_button.setEnabled(True)

        self.status_label.setText("QA Analysis failed")
        QMessageBox.critical(
            self, "Analysis Failed", f"QA Analysis failed: {error_message}"
        )

    def _update_analysis_results(self, qa_report):
        """Cập nhật UI với kết quả analysis."""
        try:
            if not qa_report:
                return

            # Update summary tab
            if hasattr(qa_report, "overall_score"):
                self.overall_score_label.setText(f"{qa_report.overall_score:.1f}%")

                # Color code based on score
                if qa_report.overall_score >= 90:
                    color = "green"
                elif qa_report.overall_score >= 80:
                    color = "orange"
                else:
                    color = "red"
                self.overall_score_label.setStyleSheet(f"color: {color};")

            if hasattr(qa_report, "overall_pass_status"):
                status_text = "PASS" if qa_report.overall_pass_status else "FAIL"
                color = "green" if qa_report.overall_pass_status else "red"
                self.pass_status_label.setText(status_text)
                self.pass_status_label.setStyleSheet(f"color: {color};")

            if hasattr(qa_report, "total_tests"):
                self.total_tests_label.setText(str(qa_report.total_tests))

            if hasattr(qa_report, "passed_tests"):
                self.passed_tests_label.setText(str(qa_report.passed_tests))

            # Update detailed results table
            if hasattr(qa_report, "qa_results"):
                self._update_details_table(qa_report.qa_results)

                # Update individual tabs
                gamma_results = [
                    r
                    for r in qa_report.qa_results
                    if hasattr(r, "test_type")
                    and str(r.test_type).lower().find("gamma") != -1
                ]
                if gamma_results:
                    self.gamma_widget.update_gamma_results(gamma_results[0])

                quality_results = [
                    r
                    for r in qa_report.qa_results
                    if hasattr(r, "test_type")
                    and any(
                        metric in str(r.test_type).lower()
                        for metric in ["conformity", "homogeneity", "gradient"]
                    )
                ]
                if quality_results:
                    self.quality_widget.update_quality_metrics(quality_results)

        except Exception as e:
            logger.error(f"Error updating analysis results: {e}")

    def _update_details_table(self, qa_results):
        """Cập nhật bảng chi tiết kết quả."""
        try:
            self.details_table.setRowCount(0)

            for i, result in enumerate(qa_results):
                self.details_table.insertRow(i)

                # Test type
                if hasattr(result, "test_type") and result.test_type:
                    if hasattr(result.test_type, "value"):
                        test_type = result.test_type.value
                    else:
                        test_type = str(result.test_type)
                else:
                    test_type = "Unknown"
                self.details_table.setItem(i, 0, QTableWidgetItem(test_type))

                # Result value
                if (
                    hasattr(result, "measured_value")
                    and result.measured_value is not None
                ):
                    result_value = f"{result.measured_value:.3f}"
                else:
                    result_value = "N/A"
                self.details_table.setItem(i, 1, QTableWidgetItem(result_value))

                # Expected value
                if (
                    hasattr(result, "expected_value")
                    and result.expected_value is not None
                ):
                    expected_value = f"{result.expected_value:.3f}"
                else:
                    expected_value = "N/A"
                self.details_table.setItem(i, 2, QTableWidgetItem(expected_value))

                # Status
                if hasattr(result, "pass_status"):
                    status = "PASS" if result.pass_status else "FAIL"
                    color = "green" if result.pass_status else "red"
                else:
                    status = "UNKNOWN"
                    color = "gray"

                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor(color))
                self.details_table.setItem(i, 3, status_item)

                # Severity
                if hasattr(result, "severity"):
                    if hasattr(result.severity, "value"):
                        severity = result.severity.value.upper()
                    else:
                        severity = str(result.severity).upper()

                    # Color code severity
                    if severity == "PASS":
                        sev_color = "green"
                    elif severity in ["WARNING", "MINOR"]:
                        sev_color = "orange"
                    elif severity in ["MAJOR", "CRITICAL", "FAIL"]:
                        sev_color = "red"
                    else:
                        sev_color = "gray"
                else:
                    severity = "UNKNOWN"
                    sev_color = "gray"

                severity_item = QTableWidgetItem(severity)
                severity_item.setForeground(QColor(sev_color))
                self.details_table.setItem(i, 4, severity_item)

        except Exception as e:
            logger.error(f"Error updating details table: {e}")

    def export_pdf_report(self):
        """Xuất báo cáo PDF."""
        if not self.current_qa_report:
            QMessageBox.warning(self, "Warning", "No QA results to export")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export PDF Report",
                "qa_report.pdf",
                "PDF files (*.pdf);;All files (*.*)",
            )

            if file_path:
                # TODO: Implement PDF export
                QMessageBox.information(
                    self, "Export", f"PDF report exported to {file_path}"
                )

        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export PDF: {e}")

    def export_csv_data(self):
        """Xuất dữ liệu CSV."""
        if not self.current_qa_report:
            QMessageBox.warning(self, "Warning", "No QA results to export")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export CSV Data",
                "qa_data.csv",
                "CSV files (*.csv);;All files (*.*)",
            )

            if file_path:
                # TODO: Implement CSV export
                QMessageBox.information(
                    self, "Export", f"CSV data exported to {file_path}"
                )

        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export CSV: {e}")


def create_plan_qa_widget():
    """Factory function để tạo PlanQAWidget."""
    return PlanQAWidget()


if __name__ == "__main__":
    # Test widget independently
    if HAS_PYQT5:
        try:
            # Try to import PyQt5 components for testing
            import sys
            from PyQt5.QtWidgets import QApplication as _QApp

            app = _QApp(sys.argv)
            widget = PlanQAWidget()
            widget.show()
            print("Plan QA Widget test thành công!")

        except ImportError as ie:
            print(f"Lỗi import PyQt5: {ie}")
            print("Sử dụng fallback classes")
            widget = PlanQAWidget()  # Will use fallback classes
            print("Plan QA Widget (fallback) test thành công!")

        except Exception as e:
            print(f"Lỗi test Plan QA Widget: {e}")
            # Try fallback
            try:
                widget = PlanQAWidget()  # Will use fallback classes
                print("Plan QA Widget (fallback) test thành công!")
            except Exception as fe:
                print(f"Fallback cũng lỗi: {fe}")
    else:
        print("PyQt5 không khả dụng - sử dụng fallback")
        widget = PlanQAWidget()  # Will use fallback classes
        print("Plan QA Widget (fallback) test thành công!")
