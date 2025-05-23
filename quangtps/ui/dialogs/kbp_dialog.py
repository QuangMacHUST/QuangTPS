#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog Knowledge-Based Planning (KBP) theo phong cách Eclipse.

Dialog này cho phép người dùng áp dụng các mô hình KBP vào quá trình lập kế hoạch,
tự động đề xuất các ràng buộc liều và tham số tối ưu dựa trên dữ liệu các kế hoạch trước đó.
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QTabWidget,
        QWidget,
        QGroupBox,
        QFormLayout,
        QCheckBox,
        QProgressBar,
        QMessageBox,
        QSpinBox,
        QDoubleSpinBox,
        QSplitter,
        QFrame,
        QApplication,
        QTreeWidget,
        QTreeWidgetItem,
        QGridLayout,
    )
    from PyQt5.QtCore import Qt, QSize, QRect, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QBrush, QPen

    HAS_PYQT = True
except ImportError:
    try:
        from PySide2.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QComboBox,
            QTableWidget,
            QTableWidgetItem,
            QHeaderView,
            QTabWidget,
            QWidget,
            QGroupBox,
            QFormLayout,
            QCheckBox,
            QProgressBar,
            QMessageBox,
            QSpinBox,
            QDoubleSpinBox,
            QSplitter,
            QFrame,
            QApplication,
            QTreeWidget,
            QTreeWidgetItem,
        )
        from PySide2.QtCore import Qt, QSize, Signal as pyqtSignal, Slot as pyqtSlot
        from PySide2.QtGui import QFont, QColor, QIcon
    except ImportError:
        print("Không thể import thư viện Qt. Tính năng KBP Dialog sẽ không khả dụng.")
        HAS_PYQT = False

try:
    from quangtps.optimization.kbp.model import KBPModel
    from quangtps.optimization.kbp.predictor import KBPPredictor, KBPRecommendation
    from quangtps.optimization.kbp.trainer import KBPTrainer
except ImportError:
    print("Không thể import module KBP. Tính năng KBP Dialog sẽ không khả dụng.")

from quangtps.utils.ui_utils import create_eclipse_icon, apply_eclipse_style_theme

try:
    from quangtps.ui.utils.ui_utils import set_eclipse_style, get_icon_path
except ImportError:
    pass

try:
    matplotlib.use("Qt5Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Không thể import matplotlib. Chức năng biểu đồ sẽ bị vô hiệu hóa.")


class KBPModelInfoWidget(QWidget):
    """Widget hiển thị thông tin về mô hình KBP."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Tiêu đề
        title_label = QLabel("Thông tin mô hình")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Form thông tin
        form_layout = QFormLayout()

        self.site_label = QLabel("N/A")
        form_layout.addRow("Vị trí điều trị:", self.site_label)

        self.model_type_label = QLabel("N/A")
        form_layout.addRow("Loại mô hình:", self.model_type_label)

        self.plans_count_label = QLabel("N/A")
        form_layout.addRow("Số lượng kế hoạch huấn luyện:", self.plans_count_label)

        self.accuracy_label = QLabel("N/A")
        form_layout.addRow("Độ chính xác trung bình:", self.accuracy_label)

        self.last_updated_label = QLabel("N/A")
        form_layout.addRow("Cập nhật lần cuối:", self.last_updated_label)

        layout.addLayout(form_layout)

        # Thông tin đặc trưng quan trọng
        self.features_table = QTableWidget(0, 2)
        self.features_table.setHorizontalHeaderLabels(["Đặc trưng", "Tầm quan trọng"])
        self.features_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.features_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        layout.addWidget(QLabel("Đặc trưng quan trọng:"))
        layout.addWidget(self.features_table)

        layout.setContentsMargins(10, 10, 10, 10)

    def update_model_info(self, model_info: Dict[str, Any]):
        """Cập nhật thông tin mô hình."""
        self.site_label.setText(model_info.get("site", "N/A"))
        self.model_type_label.setText(model_info.get("model_type", "N/A"))
        self.plans_count_label.setText(str(model_info.get("plans_count", "N/A")))
        self.accuracy_label.setText(f"{model_info.get('accuracy', 0):.2f}%")
        self.last_updated_label.setText(model_info.get("last_updated", "N/A"))

        # Cập nhật bảng đặc trưng quan trọng
        features = model_info.get("important_features", {})
        self.features_table.setRowCount(len(features))

        for i, (feature, importance) in enumerate(features.items()):
            self.features_table.setItem(i, 0, QTableWidgetItem(feature))
            self.features_table.setItem(i, 1, QTableWidgetItem(f"{importance:.3f}"))


class KBPObjectivesTable(QTableWidget):
    """Bảng hiển thị các mục tiêu tối ưu từ KBP."""

    objectiveChanged = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self._init_ui()

    def _init_ui(self):
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "Loại", "Tham số", "Giá trị", "Trọng số"]
        )

        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

    def update_objectives(
        self, objectives: Dict[str, Dict[str, Any]], weights: Dict[str, float]
    ):
        """Cập nhật các mục tiêu tối ưu."""
        self.setRowCount(0)

        row = 0
        for structure, structure_objectives in objectives.items():
            for obj_type, obj_params in structure_objectives.items():
                self.insertRow(row)

                # Thêm thông tin mục tiêu
                self.setItem(row, 0, QTableWidgetItem(structure))
                self.setItem(row, 1, QTableWidgetItem(obj_type))

                # Thêm các tham số dưới dạng chuỗi
                param_str = ", ".join([f"{k}: {v}" for k, v in obj_params.items()])
                self.setItem(row, 2, QTableWidgetItem(param_str))

                # Thêm giá trị chính (ví dụ: dose hoặc volume)
                if "dose" in obj_params:
                    value = f"{obj_params['dose']:.2f} Gy"
                elif "volume" in obj_params:
                    value = f"{obj_params['volume']:.2f} %"
                else:
                    value = "N/A"
                self.setItem(row, 3, QTableWidgetItem(value))

                # Thêm trọng số
                weight = weights.get(f"{structure}_{obj_type}", 1.0)

                # Tạo spin box cho trọng số
                weight_spin = QDoubleSpinBox()
                weight_spin.setMinimum(0.1)
                weight_spin.setMaximum(100.0)
                weight_spin.setSingleStep(0.1)
                weight_spin.setValue(weight)
                weight_spin.valueChanged.connect(
                    lambda value, r=row: self._weight_changed(r, value)
                )

                self.setCellWidget(row, 4, weight_spin)

                row += 1

    def _weight_changed(self, row: int, value: float):
        """Xử lý khi trọng số thay đổi."""
        structure = self.item(row, 0).text()
        obj_type = self.item(row, 1).text()

        objective = {"structure": structure, "type": obj_type, "weight": value}

        self.objectiveChanged.emit(row, objective)


class KBPConstraintsTable(QTableWidget):
    """Bảng hiển thị các ràng buộc từ KBP."""

    constraintChanged = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self._init_ui()

    def _init_ui(self):
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "Loại ràng buộc", "Giá trị", "Ưu tiên"]
        )

        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

    def update_constraints(self, constraints: Dict[str, Dict[str, Any]]):
        """Cập nhật các ràng buộc."""
        self.setRowCount(0)

        row = 0
        for structure, structure_constraints in constraints.items():
            for constraint_type, constraint_params in structure_constraints.items():
                self.insertRow(row)

                # Thêm thông tin ràng buộc
                self.setItem(row, 0, QTableWidgetItem(structure))
                self.setItem(row, 1, QTableWidgetItem(constraint_type))

                # Thêm giá trị chính
                if "dose" in constraint_params:
                    value = f"{constraint_params['dose']:.2f} Gy"
                elif "volume" in constraint_params:
                    value = f"{constraint_params['volume']:.2f} %"
                elif "max" in constraint_params:
                    value = f"Tối đa {constraint_params['max']:.2f} Gy"
                else:
                    value = "N/A"
                self.setItem(row, 2, QTableWidgetItem(value))

                # Thêm combo box cho ưu tiên
                priority_combo = QComboBox()
                priority_combo.addItems(["Thấp", "Trung bình", "Cao", "Rất cao"])

                priority = constraint_params.get("priority", "Trung bình")
                priority_index = {
                    "Thấp": 0,
                    "Trung bình": 1,
                    "Cao": 2,
                    "Rất cao": 3,
                }.get(priority, 1)
                priority_combo.setCurrentIndex(priority_index)

                priority_combo.currentIndexChanged.connect(
                    lambda idx, r=row: self._priority_changed(r, idx)
                )

                self.setCellWidget(row, 3, priority_combo)

                row += 1

    def _priority_changed(self, row: int, index: int):
        """Xử lý khi ưu tiên thay đổi."""
        structure = self.item(row, 0).text()
        constraint_type = self.item(row, 1).text()

        priorities = ["Thấp", "Trung bình", "Cao", "Rất cao"]
        priority = priorities[index]

        constraint = {
            "structure": structure,
            "type": constraint_type,
            "priority": priority,
        }

        self.constraintChanged.emit(row, constraint)


class KBPModelEvaluationWidget(QWidget):
    """Widget hiển thị đánh giá mô hình KBP."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_data = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Panel trên: Thông tin tổng quan
        info_group = QGroupBox("Thông tin mô hình")
        info_layout = QGridLayout()

        self.model_type_label = QLabel("Loại mô hình:")
        self.model_version_label = QLabel("Phiên bản:")
        self.creation_date_label = QLabel("Ngày tạo:")
        self.training_samples_label = QLabel("Số mẫu huấn luyện:")
        self.structures_label = QLabel("Cấu trúc hỗ trợ:")
        self.accuracy_label = QLabel("Độ chính xác:")

        info_layout.addWidget(self.model_type_label, 0, 0)
        info_layout.addWidget(self.model_version_label, 1, 0)
        info_layout.addWidget(self.creation_date_label, 2, 0)
        info_layout.addWidget(self.training_samples_label, 0, 1)
        info_layout.addWidget(self.structures_label, 1, 1)
        info_layout.addWidget(self.accuracy_label, 2, 1)

        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        # Tạo tab widget để hiển thị các biểu đồ
        self.tab_widget = QTabWidget()

        # Tab hiển thị metrics
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        if HAS_MATPLOTLIB:
            self.metrics_figure = Figure(figsize=(5, 4), dpi=100)
            self.metrics_canvas = FigureCanvas(self.metrics_figure)
            metrics_layout.addWidget(self.metrics_canvas)
        else:
            metrics_layout.addWidget(QLabel("Matplotlib không khả dụng"))
        self.tab_widget.addTab(metrics_tab, "Độ chính xác (R²)")

        # Tab hiển thị feature importance
        feature_tab = QWidget()
        feature_layout = QVBoxLayout(feature_tab)
        if HAS_MATPLOTLIB:
            self.feature_figure = Figure(figsize=(5, 4), dpi=100)
            self.feature_canvas = FigureCanvas(self.feature_figure)
            feature_layout.addWidget(self.feature_canvas)
        else:
            feature_layout.addWidget(QLabel("Matplotlib không khả dụng"))
        self.tab_widget.addTab(feature_tab, "Đặc trưng quan trọng")

        # Tab hiển thị phân phối lỗi
        dist_tab = QWidget()
        dist_layout = QVBoxLayout(dist_tab)
        if HAS_MATPLOTLIB:
            self.dist_figure = Figure(figsize=(5, 4), dpi=100)
            self.dist_canvas = FigureCanvas(self.dist_figure)
            dist_layout.addWidget(self.dist_canvas)
        else:
            dist_layout.addWidget(QLabel("Matplotlib không khả dụng"))
        self.tab_widget.addTab(dist_tab, "Phân phối lỗi")

        # Tab hiển thị bảng chi tiết các metrics
        metrics_detail_tab = QWidget()
        metrics_detail_layout = QVBoxLayout(metrics_detail_tab)
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "RMSE", "MAE", "R²", "Max Error"]
        )
        metrics_detail_layout.addWidget(self.metrics_table)
        self.tab_widget.addTab(metrics_detail_tab, "Chi tiết Metrics")

        main_layout.addWidget(self.tab_widget)

        # Khu vực nút điều khiển
        control_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Làm mới")
        self.refresh_button.clicked.connect(self._update_plots)
        control_layout.addWidget(self.refresh_button)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

    def update_evaluation_data(self, model_data: Dict[str, Any]):
        """Cập nhật dữ liệu đánh giá mô hình từ dữ liệu đầu vào."""
        self.model_data = model_data
        if not model_data:
            return

        # Cập nhật thông tin tổng quan
        if "model_info" in model_data:
            model_info = model_data["model_info"]
            self.model_type_label.setText(
                model_info.get("model_type", "Không xác định")
            )
            self.model_version_label.setText(model_info.get("version", "N/A"))
            samples = model_info.get("training_samples", 0)
            self.training_samples_label.setText(f"{samples}")

            structures = model_info.get("structures", [])
            self.structures_label.setText(f"Cấu trúc hỗ trợ: {len(structures)}")

            if "metrics" in model_data:
                avg_r2 = np.mean(
                    [m.get("r2", 0) for m in model_data["metrics"].values()]
                )
                self.accuracy_label.setText(f"Độ chính xác (R²): {avg_r2:.3f}")

        # Cập nhật bảng metrics và biểu đồ
        self._update_metrics_table()
        self._update_plots()

        # Chỉnh lại kích thước bảng và plots để hiển thị đầy đủ
        self.metrics_table.resizeColumnsToContents()
        self.metrics_figure.tight_layout()

        # Cập nhật main_layout
        main_layout = self.layout()
        main_layout.update()

    def _update_metrics_table(self):
        """Cập nhật bảng chi tiết các metrics."""
        if not self.model_data or "metrics" not in self.model_data:
            return

        metrics = self.model_data["metrics"]
        self.metrics_table.setRowCount(len(metrics))

        for i, (metric_name, values) in enumerate(metrics.items()):
            # Tên metric
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric_name))

            # Giá trị R² hoặc score
            r2 = values.get("r2", values.get("score", 0.0))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{r2:.4f}"))

            # RMSE
            rmse = values.get("rmse", 0.0)
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{rmse:.4f}"))

            # MAE
            mae = values.get("mae", 0.0)
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{mae:.4f}"))

            # Error % at 90th percentile
            err_90 = values.get("percentile_90_error", 0.0)
            self.metrics_table.setItem(i, 4, QTableWidgetItem(f"{err_90:.2f}%"))

            # Màu sắc dựa trên R²
            color = QColor(255, 255, 255)  # Trắng mặc định
            if r2 >= 0.9:
                color = QColor(200, 255, 200)  # Xanh lá nhạt
            elif r2 >= 0.8:
                color = QColor(255, 255, 200)  # Vàng nhạt
            else:
                color = QColor(255, 200, 200)  # Đỏ nhạt

            # Áp dụng màu nền cho từng ô
            for j in range(5):
                item = self.metrics_table.item(i, j)
                if item:
                    item.setBackground(color)

        # Điều chỉnh kích thước hàng và cột
        self.metrics_table.resizeColumnsToContents()
        self.metrics_table.resizeRowsToContents()

    def _update_plots(self):
        """Cập nhật tất cả các biểu đồ."""
        self._plot_metrics()
        self._plot_feature_importance()
        self._plot_error_distribution()

    def _plot_metrics(self):
        if not HAS_MATPLOTLIB or not hasattr(self, "metrics_canvas"):
            return

        self.metrics_figure.clear()
        ax = self.metrics_figure.add_subplot(111)

        # Lấy R² của tất cả các mục tiêu nếu có
        if not self.model_data or "metrics" not in self.model_data:
            ax.text(
                0.5,
                0.5,
                "Không có dữ liệu đánh giá mô hình",
                horizontalalignment="center",
                verticalalignment="center",
            )
            self.metrics_canvas.draw()
            return

        target_names = []
        r2_values = []

        for target, metrics in self.model_data["metrics"].items():
            if "r2" in metrics:
                target_names.append(target)
                r2_values.append(metrics["r2"])

        # Vẽ biểu đồ cột cho R²
        y_pos = np.arange(len(target_names))
        bars = ax.barh(y_pos, r2_values, align="center", color="skyblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(target_names)
        ax.set_xlabel("R² Score")
        ax.set_title("Độ chính xác dự đoán theo cấu trúc")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 1.1)  # R² thường từ 0 đến 1

        # Thêm giá trị vào mỗi thanh
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{r2_values[i]:.3f}",
                va="center",
            )

        # Thêm đường tham chiếu
        ax.axvline(x=0.8, color="green", linestyle="--", alpha=0.7, label="Tốt (0.8)")
        ax.axvline(
            x=0.9, color="blue", linestyle="--", alpha=0.7, label="Rất tốt (0.9)"
        )
        ax.legend(loc="lower right")

        self.metrics_canvas.draw()

    def _plot_feature_importance(self):
        if not HAS_MATPLOTLIB or not hasattr(self, "feature_canvas"):
            return

        self.feature_figure.clear()
        ax = self.feature_figure.add_subplot(111)

        # Chuyển đổi dict thành danh sách và sắp xếp
        if not self.model_data or "feature_importance" not in self.model_data:
            ax.text(
                0.5,
                0.5,
                "Không có dữ liệu feature importance",
                horizontalalignment="center",
                verticalalignment="center",
            )
            self.feature_canvas.draw()
            return

        importance_dict = self.model_data["feature_importance"]
        feature_names = []
        importances = []

        # Lấy top 10 đặc trưng quan trọng nhất
        for feature, importance in sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            feature_names.append(feature)
            importances.append(importance)

        # Vẽ biểu đồ cột ngang
        y_pos = np.arange(len(feature_names))

        # Sử dụng colormap từ matplotlib
        try:
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_names)))
        except (AttributeError, ImportError):
            # Fallback nếu không có plt.cm.viridis
            colors = "skyblue"

        bars = ax.barh(y_pos, importances, align="center", color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Độ quan trọng")
        ax.set_title("Top đặc trưng quan trọng")
        ax.grid(True, linestyle="--", alpha=0.7)

        # Thêm giá trị vào mỗi thanh
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{importances[i]:.3f}",
                va="center",
            )

        self.feature_canvas.draw()

    def _plot_error_distribution(self):
        if not HAS_MATPLOTLIB or not hasattr(self, "dist_canvas"):
            return

        self.dist_figure.clear()
        ax = self.dist_figure.add_subplot(111)

        # Lấy các percentile error cho mọi mục tiêu
        if not self.model_data or "error_percentiles" not in self.model_data:
            ax.text(
                0.5,
                0.5,
                "Không có dữ liệu phân phối lỗi",
                horizontalalignment="center",
                verticalalignment="center",
            )
            self.dist_canvas.draw()
            return

        targets = []
        p50_errors = []
        p75_errors = []
        p90_errors = []

        for target, percentiles in self.model_data["error_percentiles"].items():
            targets.append(target)
            p50_errors.append(percentiles["50"])
            p75_errors.append(percentiles["75"])
            p90_errors.append(percentiles["90"])

        # Vẽ biểu đồ nhóm cột
        x = np.arange(len(targets))  # vị trí của các nhóm
        width = 0.25  # độ rộng thanh

        bar1 = ax.bar(
            x - width, p90_errors, width, label="90th percentile", color="salmon"
        )
        bar2 = ax.bar(x, p75_errors, width, label="75th percentile", color="orange")
        bar3 = ax.bar(
            x + width, p50_errors, width, label="50th percentile", color="skyblue"
        )

        ax.set_ylabel("Độ lớn lỗi")
        ax.set_title("Phân phối lỗi theo phân vị")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Thêm giá trị trên các cột
        self._add_value_labels(ax, bar1)
        self._add_value_labels(ax, bar2)
        self._add_value_labels(ax, bar3)

        self.dist_figure.tight_layout()
        self.dist_canvas.draw()

    def _add_value_labels(self, ax, bars):
        """Thêm nhãn giá trị trên các cột."""
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )


class KBPDialog(QDialog):
    """Dialog Knowledge-Based Planning theo phong cách Eclipse."""

    # Tín hiệu khi áp dụng đề xuất KBP
    kbpRecommendationApplied = pyqtSignal(object)

    def __init__(
        self, patient_id: str, structure_set_id: str, site: str = "", parent=None
    ):
        super().__init__(parent)
        self.patient_id = patient_id
        self.structure_set_id = structure_set_id
        self.initial_site = site
        self.recommendation = None
        self.kbp_predictor = KBPPredictor()

        # Call Eclipse style if available
        try:
            set_eclipse_style(self)
        except:
            pass

        self._init_ui()
        self._site_changed(self.site_combo.currentText())

    def _get_available_sites(self) -> List[str]:
        """
        Lấy danh sách các vị trí điều trị có mô hình KBP khả dụng.

        Phương thức này tìm kiếm các mô hình KBP có sẵn trong hệ thống, từ nhiều nguồn khác nhau
        bao gồm thư mục cấu hình, dữ liệu tích hợp và các đường dẫn tùy chỉnh.

        Returns:
            List[str]: Danh sách tên các vị trí điều trị
        """
        try:
            # Tìm các mô hình KBP từ nhiều nguồn
            sites_found = set()

            # 1. Kiểm tra từ KBPPredictor nếu có
            try:
                from quangtps.optimization.kbp.predictor import KBPPredictor
                predictor = KBPPredictor()
                predictor_sites = predictor.get_available_sites()
                if predictor_sites:
                    sites_found.update(predictor_sites)
                    logger.info(f"Tìm thấy {len(predictor_sites)} vị trí từ KBPPredictor: {', '.join(predictor_sites)}")
            except (ImportError, AttributeError) as e:
                logger.debug(f"Không thể import hoặc sử dụng KBPPredictor: {str(e)}")

            # 2. Tìm từ thư mục mô hình tiêu chuẩn
            model_dirs = [
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'models', 'kbp'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'optimization', 'kbp', 'models')
            ]

            # Thêm đường dẫn từ cấu hình nếu có
            try:
                from quangtps.core.config import Config
                config = Config()
                if hasattr(config, 'kbp_models_path') and config.kbp_models_path:
                    model_dirs.append(config.kbp_models_path)
            except ImportError:
                pass

            # Tìm kiếm các thư mục con đại diện cho các vị trí
            for base_dir in model_dirs:
                if os.path.exists(base_dir) and os.path.isdir(base_dir):
                    for item in os.listdir(base_dir):
                        item_path = os.path.join(base_dir, item)
                        if os.path.isdir(item_path):
                            # Kiểm tra xem thư mục có chứa các file mô hình không
                            model_files = [f for f in os.listdir(item_path)
                                          if f.endswith(('.json', '.pkl', '.h5', '.model'))]
                            if model_files:
                                # Chuẩn hóa tên vị trí (loại bỏ gạch dưới, viết hoa chữ cái đầu)
                                site_name = ' '.join(word.capitalize() for word in item.replace('_', ' ').split())
                                sites_found.add(site_name)
                                logger.debug(f"Tìm thấy mô hình cho {site_name} tại {item_path}")

            if sites_found:
                logger.info(f"Tổng cộng tìm thấy {len(sites_found)} vị trí điều trị có mô hình KBP")
                # Chuyển set thành list và sắp xếp theo thứ tự alphabet
                return sorted(list(sites_found))

            # 3. Nếu không tìm thấy mô hình nào, sử dụng danh sách mặc định
            logger.warning("Không tìm thấy mô hình KBP, sử dụng danh sách mặc định")
            return ["Head & Neck", "Prostate", "Lung", "Breast", "Brain"]

        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách vị trí điều trị: {str(e)}", exc_info=True)
            # Fallback: trả về các vị trí điều trị phổ biến
            return ["Head & Neck", "Prostate", "Lung", "Breast", "Brain"]

    def _load_kbp_model(self, site: str) -> Optional[object]:
        """
        Tải mô hình KBP cho vị trí điều trị cụ thể.

        Phương thức này tìm và tải mô hình KBP cho vị trí điều trị cụ thể.
        Nó sẽ tìm kiếm ở nhiều vị trí khác nhau và sử dụng các phương pháp khác nhau
        để tải mô hình, với cơ chế fallback để đảm bảo tính ổn định.

        Args:
            site (str): Vị trí điều trị

        Returns:
            Optional[object]: Mô hình KBP nếu tải thành công, None nếu thất bại
        """
        if not site:
            logger.warning("Không thể tải mô hình: Không có vị trí điều trị được chỉ định")
            return None

        try:
            # Chuẩn hóa tên vị trí để tìm kiếm
            normalized_site = site.lower().replace(' & ', '_').replace(' ', '_')
            logger.info(f"Đang tìm kiếm mô hình KBP cho vị trí {site} (chuẩn hóa: {normalized_site})")

            # Phương pháp 1: Tải qua KBPPredictor (phương pháp ưu tiên)
            try:
                from quangtps.optimization.kbp.predictor import KBPPredictor
                predictor = KBPPredictor()
                logger.debug(f"Đang tải mô hình KBP cho {site} thông qua KBPPredictor...")
                model = predictor.load_model(site)

                if model:
                    logger.info(f"Đã tải thành công mô hình KBP cho {site} thông qua KBPPredictor")
                    return model
                else:
                    logger.debug(f"Không thể tải mô hình cho {site} thông qua KBPPredictor")
            except (ImportError, AttributeError, Exception) as e:
                logger.debug(f"Không thể sử dụng KBPPredictor: {str(e)}")

            # Phương pháp 2: Tìm và tải trực tiếp từ các đường dẫn tiêu chuẩn
            model_dirs = [
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'models', 'kbp'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'optimization', 'kbp', 'models')
            ]

            # Thêm đường dẫn từ cấu hình nếu có
            try:
                from quangtps.core.config import Config
                config = Config()
                if hasattr(config, 'kbp_models_path') and config.kbp_models_path:
                    model_dirs.append(config.kbp_models_path)
            except ImportError:
                pass

            # Tạo danh sách các thư mục có thể chứa mô hình
            potential_model_dirs = []
            for base_dir in model_dirs:
                if os.path.exists(base_dir):
                    # Tìm thư mục trực tiếp
                    direct_match = os.path.join(base_dir, normalized_site)
                    if os.path.isdir(direct_match):
                        potential_model_dirs.append(direct_match)

                    # Tìm tất cả thư mục có tên giống vị trí
                    for item in os.listdir(base_dir):
                        item_path = os.path.join(base_dir, item)
                        if os.path.isdir(item_path) and (
                            normalized_site in item.lower() or
                            site.lower() in item.lower().replace('_', ' ')):
                            potential_model_dirs.append(item_path)

            # Tìm và tải mô hình từ các thư mục tiềm năng
            for model_dir in potential_model_dirs:
                # Kiểm tra file .json chứa metadata
                metadata_path = os.path.join(model_dir, 'model_info.json')
                if os.path.exists(metadata_path):
                    try:
                        import json
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        # Kiểm tra nếu có thông tin mô hình chính
                        model_file = metadata.get('model_file') or 'model.pkl'
                        model_path = os.path.join(model_dir, model_file)

                        if os.path.exists(model_path):
                            # Tạo mô hình giả nếu không thể tải mô hình thực tế
                            model_info = {
                                'site': site,
                                'path': model_path,
                                'metadata': metadata,
                                'is_dummy': True  # Đánh dấu đây là mô hình giả
                            }
                            logger.info(f"Tìm thấy thông tin mô hình cho {site} tại {model_path}")
                            return model_info
                    except Exception as json_err:
                        logger.debug(f"Lỗi khi đọc metadata của mô hình: {str(json_err)}")

                # Tìm các file mô hình chuẩn
                for ext in ['.pkl', '.h5', '.model']:
                    model_path = os.path.join(model_dir, f"model{ext}")
                    if os.path.exists(model_path):
                        logger.info(f"Tìm thấy file mô hình {model_path}")
                        # Tạo thông tin mô hình cơ bản
                        return {
                            'site': site,
                            'path': model_path,
                            'metadata': {'description': f"Mô hình KBP cho {site}"},
                            'is_dummy': True
                        }

            # Phương pháp 3: Tạo mô hình giả lập dựa trên các quy tắc thông thường
            logger.warning(f"Không tìm thấy mô hình thực tế cho {site}, sử dụng mô hình giả lập")
            dummy_model = self._create_dummy_model(site)
            if dummy_model:
                return dummy_model

            logger.error(f"Không thể tìm hoặc tạo mô hình KBP cho vị trí {site}")
            return None

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình KBP cho vị trí {site}: {str(e)}", exc_info=True)
            return None

    def _create_dummy_model(self, site: str) -> Optional[Dict]:
        """
        Tạo mô hình giả lập khi không tìm thấy mô hình thực.

        Args:
            site (str): Vị trí điều trị

        Returns:
            Optional[Dict]: Thông tin mô hình giả lập
        """
        try:
            # Tạo một mô hình giả đơn giản với các thông số phù hợp với vị trí
            dummy_model = {
                'site': site,
                'is_dummy': True,
                'metadata': {
                    'description': f"Mô hình giả lập cho {site}",
                    'version': '1.0',
                    'created_date': time.strftime('%Y-%m-%d'),
                    'training_cases': 50,
                    'validation_cases': 10,
                    'accuracy': 0.85
                }
            }

            # Thêm các thông số cụ thể theo vị trí
            if 'head' in site.lower() or 'neck' in site.lower():
                dummy_model['structures'] = ['PTV', 'Brainstem', 'Spinal Cord', 'Parotid', 'Oral Cavity']
            elif 'prostate' in site.lower():
                dummy_model['structures'] = ['PTV', 'Rectum', 'Bladder', 'Femoral Heads', 'Bowel']
            elif 'lung' in site.lower():
                dummy_model['structures'] = ['PTV', 'Heart', 'Esophagus', 'Spinal Cord', 'Lung']
            elif 'breast' in site.lower():
                dummy_model['structures'] = ['PTV', 'Heart', 'Lung', 'Ribs']
            elif 'brain' in site.lower():
                dummy_model['structures'] = ['PTV', 'Brainstem', 'Optics', 'Cochlea']
            else:
                dummy_model['structures'] = ['PTV', 'OAR1', 'OAR2', 'OAR3']

            logger.info(f"Đã tạo mô hình giả lập cho {site} với {len(dummy_model['structures'])} cấu trúc")
            return dummy_model

        except Exception as e:
            logger.error(f"Không thể tạo mô hình giả lập: {str(e)}")
            return None

    def _init_ui(self):
        self.setWindowTitle("Knowledge-Based Planning")
        self.setMinimumSize(900, 600)

        main_layout = QVBoxLayout(self)

        # Panel trên cùng với logo và tiêu đề
        top_panel = QHBoxLayout()

        # Thêm icon Eclipse nếu có thể
        try:
            icon = create_eclipse_icon(icon_name="kbp")
            if icon:
                icon_label = QLabel()
                icon_label.setPixmap(icon.pixmap(QSize(32, 32)))
                top_panel.addWidget(icon_label)
        except Exception as e:
            logger.warning(f"Không thể tạo biểu tượng Eclipse: {e}")

        title_label = QLabel("Knowledge-Based Planning")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        top_panel.addWidget(title_label)
        top_panel.addStretch()

        main_layout.addLayout(top_panel)

        # Chọn vị trí điều trị
        site_layout = QHBoxLayout()
        site_label = QLabel("Vị trí điều trị:")
        self.site_combo = QComboBox()

        available_sites = self._get_available_sites()
        self.site_combo.addItems(available_sites)

        if self.initial_site and self.initial_site in available_sites:
            self.site_combo.setCurrentText(self.initial_site)

        site_layout.addWidget(site_label)
        site_layout.addWidget(self.site_combo)
        site_layout.addStretch()

        main_layout.addLayout(site_layout)

        # Tạo splitter chính giữa thông tin mô hình và kết quả
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(1)
        main_splitter.setChildrenCollapsible(False)

        # Panel bên trái - Thông tin mô hình
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Thêm widget hiển thị thông tin mô hình
        self.model_info_widget = KBPModelInfoWidget()
        left_layout.addWidget(self.model_info_widget)

        # Thêm widget đánh giá mô hình
        self.model_evaluation_widget = KBPModelEvaluationWidget()
        left_layout.addWidget(self.model_evaluation_widget)

        # Thêm nút tạo đề xuất
        generate_button = QPushButton("Tạo đề xuất")
        generate_button.clicked.connect(self._generate_recommendation)
        left_layout.addWidget(generate_button)

        main_splitter.addWidget(left_panel)

        # Panel bên phải - Đề xuất và kết quả
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tab widget cho các loại đề xuất
        result_tab = QTabWidget()

        # Tab mục tiêu tối ưu
        objectives_tab = QWidget()
        objectives_layout = QVBoxLayout(objectives_tab)
        self.objectives_table = KBPObjectivesTable()
        objectives_layout.addWidget(self.objectives_table)

        # Tab ràng buộc liều
        constraints_tab = QWidget()
        constraints_layout = QVBoxLayout(constraints_tab)
        self.constraints_table = KBPConstraintsTable()
        constraints_layout.addWidget(self.constraints_table)

        # Thêm các tab
        result_tab.addTab(objectives_tab, "Mục tiêu tối ưu")
        result_tab.addTab(constraints_tab, "Ràng buộc liều")

        right_layout.addWidget(result_tab)

        # Thêm nút áp dụng và phân tích
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.apply_button = QPushButton("Áp dụng đề xuất")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_recommendation)

        self.analysis_button = QPushButton("Phân tích kế hoạch")
        self.analysis_button.setEnabled(False)
        self.analysis_button.clicked.connect(self._show_plan_analysis)

        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.analysis_button)
        right_layout.addLayout(button_layout)

        main_splitter.addWidget(right_panel)

        # Thiết lập kích thước ban đầu cho splitter
        main_splitter.setSizes([300, 600])

        main_layout.addWidget(main_splitter)

        # Kết nối tín hiệu
        self.site_combo.currentTextChanged.connect(self._site_changed)
        self.objectives_table.objectiveChanged.connect(self._objective_changed)
        self.constraints_table.constraintChanged.connect(self._constraint_changed)

    def _apply_recommendation(self):
        """Áp dụng đề xuất KBP vào kế hoạch hiện tại."""
        if not self.recommendation:
            QMessageBox.warning(
                self, "Không có đề xuất", "Vui lòng tạo đề xuất trước khi áp dụng."
            )
            return

        # Hiển thị thông báo xác nhận
        result = QMessageBox.question(
            self,
            "Xác nhận áp dụng",
            "Bạn có chắc muốn áp dụng các mục tiêu và ràng buộc từ đề xuất KBP vào kế hoạch hiện tại?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if result != QMessageBox.Yes:
            return

        # Hiển thị dialog tiến trình
        progress = QProgressBar(self)
        progress.setRange(0, 0)  # Chế độ không xác định

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Đang áp dụng đề xuất")
        progress_layout = QVBoxLayout(progress_dialog)
        progress_layout.addWidget(QLabel("Đang áp dụng đề xuất KBP vào kế hoạch..."))
        progress_layout.addWidget(progress)
        progress_dialog.setFixedSize(300, 100)
        progress_dialog.show()

        try:
            # Lấy prescription_dose từ kế hoạch hiện tại hoặc đề xuất
            prescription_dose = 0.0
            for target, objectives in self.recommendation.objectives.items():
                if "prescription_dose" in objectives:
                    prescription_dose = objectives["prescription_dose"]
                    break

            if prescription_dose <= 0:
                prescription_dose = 50.0  # Giá trị mặc định nếu không tìm thấy

            # Tạo các collections từ đề xuất
            try:
                objective_collection = self.kbp_predictor.create_objective_collection(
                    self.recommendation, prescription_dose
                )

                constraint_collection = self.kbp_predictor.create_constraint_collection(
                    self.recommendation, prescription_dose
                )
            except Exception as e:
                logger.error(f"Lỗi khi tạo collections: {str(e)}")
                QMessageBox.critical(
                    self, "Lỗi", f"Không thể tạo collections từ đề xuất: {str(e)}"
                )
                progress_dialog.close()
                return

            # Phát ra tín hiệu với đề xuất để được xử lý bởi mainwindow
            self.kbpRecommendationApplied.emit(
                {
                    "recommendation": self.recommendation,
                    "objective_collection": objective_collection,
                    "constraint_collection": constraint_collection,
                    "prescription_dose": prescription_dose,
                }
            )

            progress_dialog.close()

            QMessageBox.information(
                self,
                "Đã áp dụng đề xuất",
                "Đề xuất KBP đã được áp dụng thành công vào kế hoạch hiện tại.",
            )

            # Đóng dialog sau khi áp dụng
            self.accept()

        except Exception as e:
            progress_dialog.close()
            logger.error(f"Lỗi khi áp dụng đề xuất KBP: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể áp dụng đề xuất: {str(e)}")

    def _generate_recommendation(self):
        """Tạo đề xuất KBP cho kế hoạch hiện tại."""
        site = self.site_combo.currentText()

        if not site:
            QMessageBox.warning(
                self, "Chưa chọn vị trí", "Vui lòng chọn vị trí điều trị."
            )
            return

        # Hiển thị dialog tiến trình
        progress = QProgressBar(self)
        progress.setRange(0, 0)  # Chế độ không xác định

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Đang tạo đề xuất")
        progress_layout = QVBoxLayout(progress_dialog)
        progress_layout.addWidget(QLabel("Đang tạo đề xuất KBP..."))
        progress_layout.addWidget(progress)
        progress_dialog.setFixedSize(300, 100)
        progress_dialog.show()

        try:
            # Tạo đề xuất KBP
            self.recommendation = self.kbp_predictor.generate_recommendation(
                self.patient_id, self.structure_set_id, site
            )

            # Cập nhật giao diện người dùng với đề xuất
            self.objectives_table.update_objectives(
                self.recommendation.objectives, self.recommendation.weights
            )

            self.constraints_table.update_constraints(
                self.recommendation.dose_constraints
            )

            # Hiển thị thông báo độ tin cậy
            confidence_text = ""
            for struct, conf in self.recommendation.confidence.items():
                confidence_text += f"{struct}: {conf:.2%}\n"

            QMessageBox.information(
                self,
                "Đề xuất đã sẵn sàng",
                f"Đã tạo đề xuất KBP cho vị trí {site}.\n\n"
                f"Độ tin cậy:\n{confidence_text}",
            )

            # Cho phép áp dụng đề xuất
            self.apply_button.setEnabled(True)
            self.analysis_button.setEnabled(True)

        except Exception as e:
            logger.error(f"Lỗi khi tạo đề xuất KBP: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo đề xuất: {str(e)}")
        finally:
            progress_dialog.close()

    def _site_changed(self, site: str):
        """Xử lý khi vị trí điều trị thay đổi."""
        self.initial_site = site

        # Cập nhật thông tin mô hình
        model_info = self._get_model_info(site)
        self.model_info_widget.update_model_info(model_info)

    def _get_model_info(self, site: str) -> Dict[str, Any]:
        """Lấy thông tin về mô hình KBP cho một vị trí điều trị."""
        # Trong thực tế, bạn sẽ tải thông tin từ model metadata
        # Đây là dữ liệu mẫu
        return {
            "site": site,
            "model_type": "Gradient Boosting",
            "plans_count": 150,
            "accuracy": 92.5,
            "last_updated": "15/10/2023",
            "important_features": {
                "ptv_volume": 0.423,
                "ptv_surface_area": 0.385,
                "distance_to_rectum": 0.356,
                "distance_to_bladder": 0.342,
                "overlap_with_rectum": 0.312,
                "patient_age": 0.221,
                "prescription_dose": 0.187,
            },
        }

    def _show_plan_analysis(self):
        """Hiển thị phân tích các kế hoạch trước đó."""
        QMessageBox.information(
            self,
            "Phân tích kế hoạch",
            "Tính năng này sẽ hiển thị phân tích các kế hoạch trước đây tương tự với kế hoạch hiện tại.",
        )

    def _objective_changed(self, row: int, objective: Dict[str, Any]):
        """Cập nhật mục tiêu tối ưu khi người dùng thay đổi."""
        if not self.recommendation:
            return

        # Cập nhật trọng số
        key = f"{objective['structure']}_{objective['type']}"
        self.recommendation["weights"][key] = objective["weight"]

    def _constraint_changed(self, row: int, constraint: Dict[str, Any]):
        """Cập nhật ràng buộc khi người dùng thay đổi."""
        if not self.recommendation:
            return

        # Cập nhật ưu tiên
        structure = constraint["structure"]
        constraint_type = constraint["type"]

        if (
            structure in self.recommendation["dose_constraints"]
            and constraint_type in self.recommendation["dose_constraints"][structure]
        ):
            self.recommendation["dose_constraints"][structure][constraint_type][
                "priority"
            ] = constraint["priority"]


# Đoạn mã kiểm thử chạy độc lập
if __name__ == "__main__":
    app = QApplication(sys.argv)

    dialog = KBPDialog("PATIENT001", "STRUCTURE001", "Prostate")
    dialog.show()

    sys.exit(app.exec_())
