#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dialog cho lập kế hoạch thích ứng thời gian thực tự động.

Module này cung cấp giao diện người dùng để thiết lập, cấu hình và chạy
tính năng lập kế hoạch thích ứng thời gian thực tự động.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QDate
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QGroupBox,
        QCheckBox,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QTabWidget,
        QListWidget,
        QListWidgetItem,
        QDateEdit,
        QProgressBar,
        QScrollArea,
        QWidget,
        QMessageBox,
        QFileDialog,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PyQt5.QtGui import QIcon, QPixmap
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal, QSize, QDate
    from PyQt6.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QGroupBox,
        QCheckBox,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QTabWidget,
        QListWidget,
        QListWidgetItem,
        QDateEdit,
        QProgressBar,
        QScrollArea,
        QWidget,
        QMessageBox,
        QFileDialog,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PyQt6.QtGui import QIcon, QPixmap

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from quangtps.adaptive.real_time_adaptive_planning import RealTimeAdaptivePlanner
from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan
from quangtps.core.services import ServiceRegistry
from quangtps.ui.charts.volume_chart import VolumeChangeChart

logger = logging.getLogger(__name__)


class RealTimeAdaptivePlanningDialog(QDialog):
    """
    Dialog cho thiết lập và quản lý lập kế hoạch thích ứng thời gian thực.
    """

    def __init__(self, patient: Patient, reference_plan: Plan, parent=None):
        """
        Khởi tạo dialog.

        Args:
            patient: Đối tượng bệnh nhân
            reference_plan: Kế hoạch tham chiếu
            parent: Widget cha (nếu có)
        """
        super().__init__(parent)
        self.patient = patient
        self.reference_plan = reference_plan
        self.planner = RealTimeAdaptivePlanner(patient, reference_plan)
        self.predictions = {}  # Kết quả dự đoán
        self.adaptive_plans = []  # Các kế hoạch thích ứng đã tạo

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        self.setWindowTitle("Lập kế hoạch thích ứng thời gian thực")
        self.resize(1000, 800)

        layout = QVBoxLayout(self)

        # Tiêu đề
        header_layout = QHBoxLayout()
        title_label = QLabel("Lập kế hoạch thích ứng thời gian thực tự động")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # Thông tin kế hoạch
        plan_info = QLabel(f"Kế hoạch tham chiếu: {self.reference_plan.name}")
        plan_info.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(plan_info)

        layout.addLayout(header_layout)

        # Tab widget chính
        self.tab_widget = QTabWidget()

        # Tab cấu hình
        self.config_tab = QWidget()
        self._setup_config_tab()
        self.tab_widget.addTab(self.config_tab, "Cấu hình")

        # Tab dự đoán
        self.prediction_tab = QWidget()
        self._setup_prediction_tab()
        self.tab_widget.addTab(self.prediction_tab, "Dự đoán thay đổi")

        # Tab kế hoạch thích ứng
        self.plans_tab = QWidget()
        self._setup_plans_tab()
        self.tab_widget.addTab(self.plans_tab, "Kế hoạch thích ứng")

        layout.addWidget(self.tab_widget)

        # Các nút điều khiển
        button_layout = QHBoxLayout()

        self.close_button = QPushButton("Đóng")
        self.close_button.clicked.connect(self.reject)

        self.help_button = QPushButton("Trợ giúp")
        self.help_button.clicked.connect(self._show_help)

        button_layout.addWidget(self.help_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _setup_config_tab(self):
        """Thiết lập tab cấu hình."""
        layout = QVBoxLayout(self.config_tab)

        # Nhóm thiết lập cấu trúc theo dõi
        structure_group = QGroupBox("Cấu trúc theo dõi")
        structure_layout = QVBoxLayout(structure_group)

        # Danh sách cấu trúc
        self.structure_table = QTableWidget()
        self.structure_table.setColumnCount(3)
        self.structure_table.setHorizontalHeaderLabels(["Chọn", "Tên cấu trúc", "Loại"])
        self.structure_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.structure_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )

        # Điền danh sách cấu trúc từ kế hoạch
        self._populate_structure_table()

        structure_layout.addWidget(self.structure_table)

        # Nút chọn tất cả / bỏ chọn tất cả
        structure_buttons = QHBoxLayout()
        self.select_all_btn = QPushButton("Chọn tất cả")
        self.select_all_btn.clicked.connect(self._select_all_structures)

        self.deselect_all_btn = QPushButton("Bỏ chọn tất cả")
        self.deselect_all_btn.clicked.connect(self._deselect_all_structures)

        structure_buttons.addWidget(self.select_all_btn)
        structure_buttons.addWidget(self.deselect_all_btn)
        structure_buttons.addStretch()

        structure_layout.addLayout(structure_buttons)

        layout.addWidget(structure_group)

        # Nhóm thiết lập tham số
        params_group = QGroupBox("Tham số thích ứng")
        params_layout = QGridLayout(params_group)

        # Thiết lập ngưỡng thích ứng
        params_layout.addWidget(QLabel("Ngưỡng thay đổi thể tích (%):"), 0, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1, 50)
        self.threshold_spin.setValue(5)
        self.threshold_spin.setSuffix("%")
        params_layout.addWidget(self.threshold_spin, 0, 1)

        # Thiết lập khoảng dự đoán
        params_layout.addWidget(QLabel("Số ngày dự đoán:"), 1, 0)
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 30)
        self.horizon_spin.setValue(5)
        params_layout.addWidget(self.horizon_spin, 1, 1)

        # Thiết lập ngày bắt đầu dự đoán
        params_layout.addWidget(QLabel("Ngày bắt đầu dự đoán:"), 2, 0)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate())
        params_layout.addWidget(self.start_date, 2, 1)

        layout.addWidget(params_group)

        # Nhóm mô hình dự đoán
        model_group = QGroupBox("Mô hình dự đoán")
        model_layout = QVBoxLayout(model_group)

        model_grid = QGridLayout()
        model_grid.addWidget(QLabel("Cấu trúc:"), 0, 0)
        self.model_structure_combo = QComboBox()
        self._populate_structure_combo()
        model_grid.addWidget(self.model_structure_combo, 0, 1)

        model_grid.addWidget(QLabel("Đường dẫn mô hình:"), 1, 0)
        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLabel("Chưa chọn mô hình")
        self.model_path_edit.setStyleSheet("font-style: italic;")
        model_path_layout.addWidget(self.model_path_edit)

        self.browse_model_btn = QPushButton("Duyệt...")
        self.browse_model_btn.clicked.connect(self._browse_model)
        model_path_layout.addWidget(self.browse_model_btn)

        model_grid.addLayout(model_path_layout, 1, 1)

        model_layout.addLayout(model_grid)

        # Nút tải mô hình
        self.load_model_btn = QPushButton("Tải mô hình cho cấu trúc")
        self.load_model_btn.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_model_btn)

        # Danh sách mô hình đã tải
        model_layout.addWidget(QLabel("Mô hình đã tải:"))
        self.model_list = QListWidget()
        model_layout.addWidget(self.model_list)

        layout.addWidget(model_group)

        # Nút áp dụng cấu hình
        self.apply_config_btn = QPushButton("Áp dụng cấu hình")
        self.apply_config_btn.clicked.connect(self._apply_config)
        layout.addWidget(self.apply_config_btn)

        layout.addStretch()

    def _setup_prediction_tab(self):
        """Thiết lập tab dự đoán."""
        layout = QVBoxLayout(self.prediction_tab)

        # Nút dự đoán
        predict_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Dự đoán thay đổi cấu trúc")
        self.predict_btn.clicked.connect(self._run_prediction)
        predict_layout.addWidget(self.predict_btn)

        self.check_adapt_btn = QPushButton("Kiểm tra cần thích ứng")
        self.check_adapt_btn.clicked.connect(self._check_adaptation_needed)
        predict_layout.addWidget(self.check_adapt_btn)

        layout.addLayout(predict_layout)

        # Kết quả dự đoán
        result_group = QGroupBox("Kết quả dự đoán")
        result_layout = QVBoxLayout(result_group)

        # Bảng dự đoán thay đổi thể tích
        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(0)
        self.prediction_table.setRowCount(0)
        result_layout.addWidget(self.prediction_table)

        # Biểu đồ thay đổi thể tích
        self.volume_chart = VolumeChangeChart(self)
        result_layout.addWidget(self.volume_chart)

        layout.addWidget(result_group)

        # Nút tạo kế hoạch thích ứng
        self.generate_plan_btn = QPushButton("Tạo kế hoạch thích ứng")
        self.generate_plan_btn.clicked.connect(self._generate_adaptive_plan)
        self.generate_plan_btn.setEnabled(False)
        layout.addWidget(self.generate_plan_btn)

    def _setup_plans_tab(self):
        """Thiết lập tab kế hoạch thích ứng."""
        layout = QVBoxLayout(self.plans_tab)

        # Nút tạo chuỗi kế hoạch
        self.generate_sequence_btn = QPushButton("Tạo chuỗi kế hoạch thích ứng")
        self.generate_sequence_btn.clicked.connect(self._generate_adaptive_sequence)
        layout.addWidget(self.generate_sequence_btn)

        # Danh sách kế hoạch đã tạo
        layout.addWidget(QLabel("Kế hoạch thích ứng đã tạo:"))
        self.plan_list = QListWidget()
        self.plan_list.currentRowChanged.connect(self._on_plan_selected)
        layout.addWidget(self.plan_list)

        # Chi tiết kế hoạch
        details_group = QGroupBox("Chi tiết kế hoạch")
        details_layout = QVBoxLayout(details_group)

        self.plan_details = QLabel("Chọn một kế hoạch để xem chi tiết")
        details_layout.addWidget(self.plan_details)

        # Các nút tương tác với kế hoạch
        plan_actions = QHBoxLayout()

        self.view_plan_btn = QPushButton("Xem kế hoạch")
        self.view_plan_btn.clicked.connect(self._view_selected_plan)
        self.view_plan_btn.setEnabled(False)
        plan_actions.addWidget(self.view_plan_btn)

        self.compare_plans_btn = QPushButton("So sánh với kế hoạch gốc")
        self.compare_plans_btn.clicked.connect(self._compare_with_reference)
        self.compare_plans_btn.setEnabled(False)
        plan_actions.addWidget(self.compare_plans_btn)

        self.approve_plan_btn = QPushButton("Phê duyệt kế hoạch")
        self.approve_plan_btn.clicked.connect(self._approve_selected_plan)
        self.approve_plan_btn.setEnabled(False)
        plan_actions.addWidget(self.approve_plan_btn)

        details_layout.addLayout(plan_actions)

        layout.addWidget(details_group)

    def _connect_signals(self):
        """Kết nối các tín hiệu."""
        # Kết nối các tín hiệu khác nếu cần
        pass

    def _populate_structure_table(self):
        """Điền danh sách cấu trúc vào bảng."""
        structures = self.reference_plan.get_structures()
        self.structure_table.setRowCount(len(structures))

        for i, structure in enumerate(structures):
            # Cột checkbox
            checkbox = QCheckBox()
            self.structure_table.setCellWidget(i, 0, checkbox)

            # Mặc định chọn các cấu trúc target và OAR quan trọng
            structure_type = (
                structure.type.lower() if hasattr(structure, "type") else ""
            )
            if (
                "target" in structure_type
                or "ptv" in structure_type
                or "ctv" in structure_type
                or any(
                    org in structure.name.lower()
                    for org in [
                        "heart",
                        "lung",
                        "spinal",
                        "brain",
                        "kidney",
                        "liver",
                        "bladder",
                        "rectum",
                    ]
                )
            ):
                checkbox.setChecked(True)

            # Cột tên cấu trúc
            name_item = QTableWidgetItem(structure.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.structure_table.setItem(i, 1, name_item)

            # Cột loại cấu trúc
            type_item = QTableWidgetItem(
                structure.type if hasattr(structure, "type") else "Unknown"
            )
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.structure_table.setItem(i, 2, type_item)

    def _populate_structure_combo(self):
        """Điền danh sách cấu trúc vào combo box."""
        self.model_structure_combo.clear()

        structures = self.reference_plan.get_structures()
        for structure in structures:
            self.model_structure_combo.addItem(structure.name, structure.id)

    def _select_all_structures(self):
        """Chọn tất cả các cấu trúc."""
        for row in range(self.structure_table.rowCount()):
            checkbox = self.structure_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(True)

    def _deselect_all_structures(self):
        """Bỏ chọn tất cả các cấu trúc."""
        for row in range(self.structure_table.rowCount()):
            checkbox = self.structure_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(False)

    def _browse_model(self):
        """Duyệt tìm file mô hình dự đoán."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file mô hình",
            "",
            "Model Files (*.pkl *.h5 *.model);;All Files (*)",
        )

        if file_path:
            self.model_path_edit.setText(file_path)

    def _load_model(self):
        """Tải mô hình dự đoán cho cấu trúc đã chọn."""
        model_path = self.model_path_edit.text()
        if model_path == "Chưa chọn mô hình":
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file mô hình")
            return

        structure_idx = self.model_structure_combo.currentIndex()
        if structure_idx < 0:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cấu trúc")
            return

        structure_id = self.model_structure_combo.itemData(structure_idx)
        structure_name = self.model_structure_combo.currentText()

        success = self.planner.load_prediction_model(structure_id, model_path)

        if success:
            self.model_list.addItem(f"{structure_name}: {os.path.basename(model_path)}")
            QMessageBox.information(
                self, "Thành công", f"Đã tải mô hình cho cấu trúc {structure_name}"
            )
        else:
            QMessageBox.warning(
                self, "Lỗi", f"Không thể tải mô hình cho cấu trúc {structure_name}"
            )

    def _apply_config(self):
        """Áp dụng các thiết lập cấu hình."""
        # Thu thập danh sách cấu trúc đã chọn
        selected_structures = []
        for row in range(self.structure_table.rowCount()):
            checkbox = self.structure_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                structure_id = self.reference_plan.get_structures()[row].id
                selected_structures.append(structure_id)

        # Thiết lập cấu trúc theo dõi
        self.planner.set_monitoring_structures(selected_structures)

        # Thiết lập ngưỡng thích ứng
        threshold = self.threshold_spin.value() / 100.0  # Chuyển đổi từ % sang decimal
        self.planner.set_adaptation_threshold(threshold)

        # Thiết lập khoảng dự đoán
        horizon = self.horizon_spin.value()
        self.planner.set_prediction_horizon(horizon)

        QMessageBox.information(self, "Thành công", "Đã áp dụng cấu hình thành công")

    def _run_prediction(self):
        """Chạy dự đoán thay đổi cấu trúc."""
        # Lấy ngày bắt đầu dự đoán
        qdate = self.start_date.date()
        start_date = datetime(qdate.year(), qdate.month(), qdate.day())

        # Dự đoán thay đổi cấu trúc
        self.predictions = self.planner.predict_anatomy_changes(start_date)

        # Hiển thị kết quả dự đoán
        self._display_predictions()

        # Kích hoạt các nút liên quan
        self.check_adapt_btn.setEnabled(True)
        self.generate_plan_btn.setEnabled(True)

    def _display_predictions(self):
        """Hiển thị kết quả dự đoán trong bảng và biểu đồ."""
        if not self.predictions:
            return

        # Xác định số cột dựa trên dự đoán đầu tiên
        first_prediction = next(iter(self.predictions.values()), None)
        if not first_prediction:
            return

        predictions_list = first_prediction.get("predictions", [])
        col_count = len(predictions_list) + 2  # +2 cho cột Structure và Current Volume

        # Thiết lập bảng
        self.prediction_table.setRowCount(len(self.predictions))
        self.prediction_table.setColumnCount(col_count)

        # Thiết lập tiêu đề cột
        headers = ["Cấu trúc", "Thể tích hiện tại"]
        for i, pred in enumerate(predictions_list):
            date = pred.get("date")
            if date:
                date_str = date.strftime("%d/%m/%Y")
                headers.append(date_str)

        self.prediction_table.setHorizontalHeaderLabels(headers)

        # Điền dữ liệu dự đoán
        row = 0
        volume_data = {}  # Dữ liệu cho biểu đồ

        for structure_id, prediction_data in self.predictions.items():
            structure_name = prediction_data.get("structure_name", structure_id)
            current_volume = prediction_data.get("current_volume", 0)
            predictions_list = prediction_data.get("predictions", [])

            # Dữ liệu cho biểu đồ
            volume_series = [current_volume]
            dates = [datetime.now()]

            # Cột tên cấu trúc
            self.prediction_table.setItem(row, 0, QTableWidgetItem(structure_name))

            # Cột thể tích hiện tại
            self.prediction_table.setItem(
                row, 1, QTableWidgetItem(f"{current_volume:.2f} cc")
            )

            # Cột dự đoán
            for i, pred in enumerate(predictions_list):
                volume = pred.get("volume", 0)
                date = pred.get("date")

                # Thêm vào dữ liệu cho biểu đồ
                volume_series.append(volume)
                dates.append(date)

                # Thêm vào bảng
                change_pct = (
                    (volume - current_volume) / current_volume * 100
                    if current_volume > 0
                    else 0
                )
                cell_text = f"{volume:.2f} cc ({change_pct:+.1f}%)"

                # Đánh dấu màu nếu thay đổi vượt ngưỡng
                item = QTableWidgetItem(cell_text)
                if abs(change_pct) > self.threshold_spin.value():
                    item.setBackground(Qt.red if change_pct > 0 else Qt.blue)

                self.prediction_table.setItem(row, i + 2, item)

            # Lưu dữ liệu cho biểu đồ
            volume_data[structure_name] = {"volumes": volume_series, "dates": dates}

            row += 1

        # Cập nhật biểu đồ
        self.volume_chart.clear()
        self.volume_chart.set_volume_data(volume_data)
        self.volume_chart.update_plot()

        # Điều chỉnh kích thước cột
        self.prediction_table.resizeColumnsToContents()

    def _check_adaptation_needed(self):
        """Kiểm tra xem có cần thích ứng kế hoạch không."""
        if not self.predictions:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chạy dự đoán trước")
            return

        needs_adaptation = self.planner.check_adaptation_needed(self.predictions)

        if needs_adaptation:
            QMessageBox.information(
                self,
                "Cần thích ứng",
                "Cần thích ứng kế hoạch dựa trên dự đoán thay đổi cấu trúc.\n"
                "Bạn có thể tạo kế hoạch thích ứng bằng nút 'Tạo kế hoạch thích ứng'.",
            )
        else:
            QMessageBox.information(
                self,
                "Không cần thích ứng",
                "Không cần thích ứng kế hoạch dựa trên dự đoán hiện tại.\n"
                "Các thay đổi dự đoán nằm trong ngưỡng cho phép.",
            )

    def _generate_adaptive_plan(self):
        """Tạo kế hoạch thích ứng dựa trên dự đoán."""
        if not self.predictions:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chạy dự đoán trước")
            return

        # Lấy ngày dự đoán đầu tiên
        qdate = self.start_date.date()
        start_date = datetime(qdate.year(), qdate.month(), qdate.day())
        target_date = start_date + timedelta(days=1)  # Ngày đầu tiên trong dự đoán

        # Tạo kế hoạch thích ứng
        plan = self.planner.generate_adaptive_plan(target_date, self.predictions)

        if plan:
            # Thêm vào danh sách và hiển thị
            self.adaptive_plans.append(plan)
            self._update_plan_list()

            # Chuyển sang tab kế hoạch
            self.tab_widget.setCurrentIndex(2)

            QMessageBox.information(
                self, "Thành công", f"Đã tạo kế hoạch thích ứng: {plan.name}"
            )
        else:
            QMessageBox.warning(self, "Lỗi", "Không thể tạo kế hoạch thích ứng")

    def _generate_adaptive_sequence(self):
        """Tạo chuỗi kế hoạch thích ứng cho toàn bộ khoảng dự đoán."""
        # Lấy ngày bắt đầu dự đoán
        qdate = self.start_date.date()
        start_date = datetime(qdate.year(), qdate.month(), qdate.day())

        # Tạo chuỗi kế hoạch thích ứng
        plans = self.planner.generate_adaptive_plans_sequence(start_date)

        if plans:
            # Thêm vào danh sách và hiển thị
            self.adaptive_plans.extend(plans)
            self._update_plan_list()

            QMessageBox.information(
                self, "Thành công", f"Đã tạo {len(plans)} kế hoạch thích ứng"
            )
        else:
            QMessageBox.information(
                self,
                "Thông báo",
                "Không cần tạo kế hoạch thích ứng nào trong khoảng dự đoán",
            )

    def _update_plan_list(self):
        """Cập nhật danh sách kế hoạch."""
        self.plan_list.clear()

        for plan in self.adaptive_plans:
            item = QListWidgetItem(plan.name)
            item.setData(Qt.UserRole, plan)
            self.plan_list.addItem(item)

    def _on_plan_selected(self, row):
        """Xử lý khi một kế hoạch được chọn từ danh sách."""
        if row < 0 or row >= len(self.adaptive_plans):
            self.plan_details.setText("Chọn một kế hoạch để xem chi tiết")
            self.view_plan_btn.setEnabled(False)
            self.compare_plans_btn.setEnabled(False)
            self.approve_plan_btn.setEnabled(False)
            return

        plan = self.adaptive_plans[row]

        # Hiển thị thông tin chi tiết
        details = f"<b>Tên:</b> {plan.name}<br>"
        details += f"<b>Mô tả:</b> {plan.description}<br>"

        if hasattr(plan, "creation_date"):
            details += (
                f"<b>Ngày tạo:</b> {plan.creation_date.strftime('%d/%m/%Y %H:%M')}<br>"
            )

        details += f"<b>Số beam:</b> {len(plan.beams) if hasattr(plan, 'beams') else 'N/A'}<br>"

        self.plan_details.setText(details)

        # Kích hoạt các nút
        self.view_plan_btn.setEnabled(True)
        self.compare_plans_btn.setEnabled(True)
        self.approve_plan_btn.setEnabled(True)

    def _view_selected_plan(self):
        """Xem kế hoạch đã chọn."""
        row = self.plan_list.currentRow()
        if row < 0 or row >= len(self.adaptive_plans):
            return

        plan = self.adaptive_plans[row]

        # Gửi tín hiệu đến ứng dụng chính để hiển thị kế hoạch
        # Trong một ứng dụng thực, sẽ có cơ chế gửi tín hiệu tới main window
        QMessageBox.information(
            self, "Xem kế hoạch", f"Đang mở kế hoạch {plan.name} trong giao diện chính"
        )

    def _compare_with_reference(self):
        """So sánh kế hoạch đã chọn với kế hoạch tham chiếu."""
        row = self.plan_list.currentRow()
        if row < 0 or row >= len(self.adaptive_plans):
            return

        plan = self.adaptive_plans[row]

        # Trong một ứng dụng thực, sẽ mở một cửa sổ so sánh kế hoạch
        QMessageBox.information(
            self,
            "So sánh kế hoạch",
            f"Đang so sánh kế hoạch {plan.name} với kế hoạch tham chiếu {self.reference_plan.name}",
        )

    def _approve_selected_plan(self):
        """Phê duyệt kế hoạch đã chọn."""
        row = self.plan_list.currentRow()
        if row < 0 or row >= len(self.adaptive_plans):
            return

        plan = self.adaptive_plans[row]

        reply = QMessageBox.question(
            self,
            "Phê duyệt kế hoạch",
            f"Bạn có chắc chắn muốn phê duyệt kế hoạch {plan.name} không?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Trong một ứng dụng thực, sẽ có quy trình phê duyệt kế hoạch
            QMessageBox.information(
                self, "Phê duyệt kế hoạch", f"Kế hoạch {plan.name} đã được phê duyệt"
            )

    def _show_help(self):
        """Hiển thị trợ giúp."""
        help_text = """
        <h3>Lập kế hoạch thích ứng thời gian thực tự động</h3>

        <p>Công cụ này cho phép bạn dự đoán sự thay đổi của cấu trúc giải phẫu theo thời gian
        và tự động tạo kế hoạch thích ứng dựa trên những thay đổi đó.</p>

        <h4>Quy trình sử dụng:</h4>
        <ol>
            <li>Thiết lập cấu hình: Chọn các cấu trúc cần theo dõi và thiết lập các tham số thích ứng</li>
            <li>Dự đoán thay đổi: Chạy dự đoán để xem sự thay đổi của các cấu trúc theo thời gian</li>
            <li>Tạo kế hoạch thích ứng: Tạo kế hoạch thích ứng dựa trên các thay đổi dự đoán</li>
            <li>Xem và phê duyệt: Kiểm tra và phê duyệt kế hoạch thích ứng</li>
        </ol>
        """

        QMessageBox.information(self, "Trợ giúp", help_text)
