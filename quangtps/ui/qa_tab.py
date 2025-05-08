#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho tab Đảm bảo Chất lượng (QA).

Module này cung cấp giao diện để quản lý, thực hiện, và đánh giá
các kiểm tra đảm bảo chất lượng cho kế hoạch điều trị.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

from PyQt5.QtCore import Qt, QDate, QDateTime, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QTabWidget,
    QLineEdit,
    QDateEdit,
    QTextEdit,
    QGroupBox,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QSplitter,
    QScrollArea,
    QFrame,
    QHeaderView,
    QRadioButton,
    QButtonGroup,
    QFileDialog,
    QToolButton,
    QMenu,
    QAction,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QInputDialog,
    QToolBar,
    QListWidgetItem,
)
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap

from quangtps.core.logging import get_logger
from quangtps.evaluation.qa.treatment_qa import (
    TreatmentQAManager,
    TreatmentQATest,
    QATestType,
    QAProtocol,
    QAStatus,
    MetricResult,
)
from quangtps.evaluation.qa.machine_log_analyzer import DeviationSeverity
from quangtps.database.patient_db import PatientDatabase
from quangtps.planning.plan import PlanStatus

# Import Machine Log Analyzer Widget
from quangtps.ui.machine_log_analyzer_widget import MachineLogAnalyzerWidget

logger = get_logger(__name__)


class QATestDetailsWidget(QWidget):
    """Widget hiển thị chi tiết một bài kiểm tra QA."""

    test_updated = pyqtSignal(str)  # Tín hiệu khi bài kiểm tra được cập nhật

    def __init__(self, parent=None):
        """Khởi tạo widget chi tiết bài kiểm tra QA."""
        super().__init__(parent)
        self.test_id = None
        self.qa_manager = TreatmentQAManager()
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện widget."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Khu vực thông tin chung
        info_group = QGroupBox("Thông tin bài kiểm tra")
        info_layout = QFormLayout()

        self.test_name_label = QLabel("")
        self.test_type_label = QLabel("")
        self.protocol_label = QLabel("")
        self.status_label = QLabel("")
        self.created_date_label = QLabel("")
        self.performed_date_label = QLabel("")
        self.performed_by_label = QLabel("")

        info_layout.addRow("Tên bài kiểm tra:", self.test_name_label)
        info_layout.addRow("Loại kiểm tra:", self.test_type_label)
        info_layout.addRow("Giao thức:", self.protocol_label)
        info_layout.addRow("Trạng thái:", self.status_label)
        info_layout.addRow("Ngày tạo:", self.created_date_label)
        info_layout.addRow("Ngày thực hiện:", self.performed_date_label)
        info_layout.addRow("Người thực hiện:", self.performed_by_label)

        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        # Khu vực kết quả metric
        metrics_group = QGroupBox("Kết quả đánh giá")
        metrics_layout = QVBoxLayout()

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(7)
        self.metrics_table.setHorizontalHeaderLabels(
            [
                "Metric",
                "Giá trị",
                "Tham chiếu",
                "Dung sai",
                "Đơn vị",
                "Sai số",
                "Chấp nhận được",
            ]
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        metrics_layout.addWidget(self.metrics_table)
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        # Khu vực thêm metric mới
        add_metric_group = QGroupBox("Thêm metric mới")
        add_metric_layout = QFormLayout()

        self.metric_name_input = QLineEdit()
        self.metric_value_input = QDoubleSpinBox()
        self.metric_value_input.setRange(-10000, 10000)
        self.metric_value_input.setDecimals(3)

        self.metric_reference_input = QDoubleSpinBox()
        self.metric_reference_input.setRange(-10000, 10000)
        self.metric_reference_input.setDecimals(3)

        self.metric_tolerance_input = QDoubleSpinBox()
        self.metric_tolerance_input.setRange(0, 1000)
        self.metric_tolerance_input.setDecimals(3)

        self.metric_unit_input = QLineEdit()
        self.metric_description_input = QLineEdit()

        add_metric_layout.addRow("Tên metric:", self.metric_name_input)
        add_metric_layout.addRow("Giá trị:", self.metric_value_input)
        add_metric_layout.addRow("Tham chiếu:", self.metric_reference_input)
        add_metric_layout.addRow("Dung sai:", self.metric_tolerance_input)
        add_metric_layout.addRow("Đơn vị:", self.metric_unit_input)
        add_metric_layout.addRow("Mô tả:", self.metric_description_input)

        add_metric_btn = QPushButton("Thêm metric")
        add_metric_btn.clicked.connect(self._add_metric)
        add_metric_layout.addRow("", add_metric_btn)

        add_metric_group.setLayout(add_metric_layout)
        main_layout.addWidget(add_metric_group)

        # Khu vực cập nhật trạng thái
        status_group = QGroupBox("Cập nhật trạng thái")
        status_layout = QFormLayout()

        self.status_combo = QComboBox()
        for status in QAStatus:
            self.status_combo.addItem(status.value, status)

        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(100)

        status_layout.addRow("Trạng thái:", self.status_combo)
        status_layout.addRow("Ghi chú:", self.notes_input)

        update_status_btn = QPushButton("Cập nhật trạng thái")
        update_status_btn.clicked.connect(self._update_status)
        status_layout.addRow("", update_status_btn)

        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # Khu vực nút điều khiển chính
        control_layout = QHBoxLayout()

        evaluate_btn = QPushButton("Đánh giá kết quả")
        evaluate_btn.clicked.connect(self._evaluate_test)

        export_btn = QPushButton("Xuất báo cáo")
        export_btn.clicked.connect(self._export_report)

        control_layout.addWidget(evaluate_btn)
        control_layout.addWidget(export_btn)

        main_layout.addLayout(control_layout)
        main_layout.addStretch()

    def load_test(self, test_id: str):
        """Tải thông tin của một bài kiểm tra QA."""
        if not test_id:
            return

        self.test_id = test_id
        test = self.qa_manager.get_test(test_id)
        if not test:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy bài kiểm tra QA.")
            return

        # Cập nhật thông tin hiển thị
        self.test_name_label.setText(test.test_name)
        self.test_type_label.setText(test.test_type.value)
        self.protocol_label.setText(test.protocol.value)
        self.status_label.setText(test.status.value)
        self.created_date_label.setText(test.created_date.strftime("%d/%m/%Y %H:%M"))

        if test.performed_date:
            self.performed_date_label.setText(
                test.performed_date.strftime("%d/%m/%Y %H:%M")
            )
        else:
            self.performed_date_label.setText("Chưa thực hiện")

        self.performed_by_label.setText(test.performed_by if test.performed_by else "")

        # Cập nhật bảng metrics
        self._load_metrics(test)

        # Đặt trạng thái hiện tại trong combo box
        index = self.status_combo.findText(test.status.value)
        if index >= 0:
            self.status_combo.setCurrentIndex(index)

        # Hiển thị ghi chú
        self.notes_input.setText(test.notes)

    def _load_metrics(self, test: TreatmentQATest):
        """Tải các metric của bài kiểm tra vào bảng."""
        self.metrics_table.setRowCount(0)

        for i, metric in enumerate(test.metrics):
            self.metrics_table.insertRow(i)

            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric.name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{metric.value:.3f}"))
            self.metrics_table.setItem(
                i, 2, QTableWidgetItem(f"{metric.reference:.3f}")
            )
            self.metrics_table.setItem(
                i, 3, QTableWidgetItem(f"{metric.tolerance:.3f}")
            )
            self.metrics_table.setItem(i, 4, QTableWidgetItem(metric.unit))
            self.metrics_table.setItem(i, 5, QTableWidgetItem(f"{metric.error:.3f}"))

            acceptable_item = QTableWidgetItem(
                "Có" if metric.is_acceptable else "Không"
            )
            acceptable_item.setForeground(
                QColor("green" if metric.is_acceptable else "red")
            )
            self.metrics_table.setItem(i, 6, acceptable_item)

    def _add_metric(self):
        """Thêm một metric mới vào bài kiểm tra."""
        if not self.test_id:
            QMessageBox.warning(self, "Lỗi", "Không có bài kiểm tra nào được chọn.")
            return

        name = self.metric_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên metric.")
            return

        value = self.metric_value_input.value()
        reference = self.metric_reference_input.value()
        tolerance = self.metric_tolerance_input.value()
        unit = self.metric_unit_input.text()
        description = self.metric_description_input.text()

        # Tạo metric mới
        metric = MetricResult(
            name=name,
            value=value,
            reference=reference,
            tolerance=tolerance,
            unit=unit,
            description=description,
        )

        # Thêm vào bài kiểm tra
        test = self.qa_manager.get_test(self.test_id)
        if test:
            test.add_metric(metric)
            self.qa_manager.update_test(test)
            self._load_metrics(test)

            # Reset các trường nhập liệu
            self.metric_name_input.clear()
            self.metric_value_input.setValue(0)
            self.metric_reference_input.setValue(0)
            self.metric_tolerance_input.setValue(0)
            self.metric_unit_input.clear()
            self.metric_description_input.clear()

            self.test_updated.emit(self.test_id)
            QMessageBox.information(
                self, "Thành công", f"Đã thêm metric '{name}' vào bài kiểm tra."
            )

    def _update_status(self):
        """Cập nhật trạng thái của bài kiểm tra."""
        if not self.test_id:
            QMessageBox.warning(self, "Lỗi", "Không có bài kiểm tra nào được chọn.")
            return

        status_index = self.status_combo.currentIndex()
        status = self.status_combo.itemData(status_index)
        notes = self.notes_input.toPlainText()

        test = self.qa_manager.get_test(self.test_id)
        if test:
            test.set_status(status, notes)

            # Cập nhật ngày thực hiện nếu trạng thái là IN_PROGRESS
            if status == QAStatus.IN_PROGRESS and not test.performed_date:
                test.performed_date = datetime.now()
                test.performed_by = (
                    "Current User"  # Đây có thể thay bằng thông tin người dùng hiện tại
                )

            self.qa_manager.update_test(test)
            self.load_test(self.test_id)  # Tải lại thông tin
            self.test_updated.emit(self.test_id)
            QMessageBox.information(
                self, "Thành công", "Đã cập nhật trạng thái bài kiểm tra."
            )

    def _evaluate_test(self):
        """Đánh giá kết quả tổng thể của bài kiểm tra."""
        if not self.test_id:
            QMessageBox.warning(self, "Lỗi", "Không có bài kiểm tra nào được chọn.")
            return

        test = self.qa_manager.get_test(self.test_id)
        if not test:
            return

        if len(test.metrics) == 0:
            QMessageBox.warning(
                self, "Lỗi", "Bài kiểm tra không có metric nào để đánh giá."
            )
            return

        result = test.evaluate()
        status = QAStatus.PASSED if result else QAStatus.FAILED
        test.set_status(status, f"Đánh giá tự động: {'Đạt' if result else 'Không đạt'}")
        self.qa_manager.update_test(test)

        self.load_test(self.test_id)
        self.test_updated.emit(self.test_id)

        QMessageBox.information(
            self,
            "Kết quả đánh giá",
            f"Kết quả đánh giá: {'ĐẠT' if result else 'KHÔNG ĐẠT'}\n\n"
            + f"Bài kiểm tra có {len(test.metrics)} metric, "
            + f"{sum(1 for m in test.metrics if m.is_acceptable)} đạt yêu cầu.",
        )

    def _export_report(self):
        """Xuất báo cáo QA."""
        if not self.test_id:
            QMessageBox.warning(self, "Lỗi", "Không có bài kiểm tra nào được chọn.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất báo cáo", "", "PDF Files (*.pdf);;HTML Files (*.html)"
        )

        if not file_path:
            return

        test = self.qa_manager.get_test(self.test_id)
        if not test:
            return

        try:
            # Đây sẽ là nơi triển khai việc xuất báo cáo thực tế
            # Có thể sử dụng module reporting để thực hiện việc này
            from quangtps.reporting.qa_report import create_qa_report

            create_qa_report(test, file_path)
            QMessageBox.information(
                self, "Thành công", f"Đã xuất báo cáo thành {file_path}"
            )
        except Exception as e:
            logger.error(f"Lỗi khi xuất báo cáo QA: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể xuất báo cáo: {e}")


class QATab(QWidget):
    """Tab quản lý và thực hiện các bài kiểm tra QA."""

    def __init__(self, parent=None):
        """Khởi tạo tab QA."""
        super().__init__(parent)
        self.qa_manager = TreatmentQAManager()
        self.patient_db = PatientDatabase()
        self._init_ui()
        self._load_qa_tests()

    def _init_ui(self):
        """Khởi tạo giao diện tab."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Tạo tab widget
        self.tab_widget = QTabWidget()

        # Tab quản lý QA
        self.qa_management_tab = self._create_qa_management_tab()
        self.tab_widget.addTab(self.qa_management_tab, "Quản lý QA")

        # Tab phantom QA
        self.phantom_qa_tab = self._create_phantom_qa_tab()
        self.tab_widget.addTab(self.phantom_qa_tab, "Phantom QA")

        # Tab phân tích log máy điều trị
        self.machine_log_tab = self._create_machine_log_tab()
        self.tab_widget.addTab(self.machine_log_tab, "Log File Máy")

        # Tab quản lý quy tắc kiểm tra log file
        self.rules_tab = self._create_rules_tab()
        self.tab_widget.addTab(self.rules_tab, "Quy tắc kiểm tra")

        # Thêm tab widget vào layout chính
        main_layout.addWidget(self.tab_widget)

    def _create_machine_log_tab(self):
        """Tạo tab để phân tích log file máy điều trị."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Tạo widget phân tích log file
        from quangtps.ui.machine_log_analyzer_widget import MachineLogAnalyzerWidget

        self.log_analyzer_widget = MachineLogAnalyzerWidget()

        # Kết nối tín hiệu
        self.log_analyzer_widget.analysis_completed.connect(
            self._on_log_analysis_completed
        )
        self.log_analyzer_widget.analysis_failed.connect(self._on_log_analysis_failed)

        # Thêm nút tùy chọn validator
        validator_layout = QHBoxLayout()

        self.use_validator_check = QCheckBox("Sử dụng quy tắc tự động kiểm tra")
        self.use_validator_check.setChecked(True)
        validator_layout.addWidget(self.use_validator_check)

        self.manage_rules_button = QPushButton("Quản lý quy tắc kiểm tra")
        self.manage_rules_button.clicked.connect(self._show_rule_manager)
        validator_layout.addWidget(self.manage_rules_button)

        validator_layout.addStretch()

        # Thêm vào layout
        layout.addLayout(validator_layout)
        layout.addWidget(self.log_analyzer_widget)

        return tab

    def _create_rules_tab(self):
        """Tạo tab để quản lý quy tắc kiểm tra log file."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Tạo widget quản lý quy tắc
        from quangtps.ui.log_file_validator_widget import LogFileValidatorWidget

        self.rule_manager_widget = LogFileValidatorWidget()

        # Kết nối tín hiệu
        self.rule_manager_widget.rules_changed.connect(self._on_rules_changed)

        # Thêm vào layout
        layout.addWidget(self.rule_manager_widget)

        return tab

    def _create_qa_management_tab(self):
        """Tạo tab quản lý QA."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Panel bên trái: Danh sách bài kiểm tra và điều khiển
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Khu vực bộ lọc
        filter_group = QGroupBox("Bộ lọc")
        filter_layout = QFormLayout(filter_group)

        self.filter_name = QLineEdit()
        self.filter_name.setPlaceholderText("Tìm kiếm theo tên...")
        self.filter_name.textChanged.connect(self._apply_filters)

        self.filter_type = QComboBox()
        self.filter_type.addItem("Tất cả loại", None)
        for test_type in QATestType:
            self.filter_type.addItem(test_type.value, test_type)
        self.filter_type.currentIndexChanged.connect(self._apply_filters)

        self.filter_status = QComboBox()
        self.filter_status.addItem("Tất cả trạng thái", None)
        for status in QAStatus:
            self.filter_status.addItem(status.value, status)
        self.filter_status.currentIndexChanged.connect(self._apply_filters)

        filter_layout.addRow("Tên:", self.filter_name)
        filter_layout.addRow("Loại:", self.filter_type)
        filter_layout.addRow("Trạng thái:", self.filter_status)

        left_layout.addWidget(filter_group)

        # Bảng danh sách bài kiểm tra QA
        self.tests_table = QTableWidget()
        self.tests_table.setColumnCount(5)
        self.tests_table.setHorizontalHeaderLabels(
            ["ID", "Tên bài kiểm tra", "Loại", "Trạng thái", "Ngày tạo"]
        )
        self.tests_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tests_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.tests_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tests_table.setSelectionMode(QTableWidget.SingleSelection)
        self.tests_table.itemSelectionChanged.connect(self._on_test_selected)

        left_layout.addWidget(QLabel("Danh sách bài kiểm tra QA:"))
        left_layout.addWidget(self.tests_table)

        # Các nút điều khiển
        control_layout = QHBoxLayout()

        new_test_btn = QPushButton("Tạo mới")
        new_test_btn.clicked.connect(self._create_new_test)

        delete_test_btn = QPushButton("Xóa")
        delete_test_btn.clicked.connect(self._delete_test)

        refresh_btn = QPushButton("Làm mới")
        refresh_btn.clicked.connect(self._load_qa_tests)

        control_layout.addWidget(new_test_btn)
        control_layout.addWidget(delete_test_btn)
        control_layout.addWidget(refresh_btn)

        left_layout.addLayout(control_layout)

        # Panel bên phải: Chi tiết bài kiểm tra
        self.test_details = QATestDetailsWidget()
        self.test_details.test_updated.connect(self._on_test_updated)

        # Thêm panels vào layout chính
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.test_details)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

        # Tải dữ liệu
        self._load_qa_tests()

        return tab

    def _create_phantom_qa_tab(self):
        """Tạo tab phantom QA."""
        from quangtps.evaluation.qa.phantom import PhantomLibrary

        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Thẻ công cụ
        toolbar = QToolBar()

        # Nút thêm phantom
        add_phantom_action = QAction(QIcon.fromTheme("list-add"), "Thêm Phantom", self)
        add_phantom_action.triggered.connect(self._add_phantom)
        toolbar.addAction(add_phantom_action)

        # Nút xóa phantom
        remove_phantom_action = QAction(
            QIcon.fromTheme("list-remove"), "Xóa Phantom", self
        )
        remove_phantom_action.triggered.connect(self._remove_phantom)
        toolbar.addAction(remove_phantom_action)

        # Nút tải phantom từ DICOM
        import_phantom_action = QAction(
            QIcon.fromTheme("document-open"), "Nhập từ DICOM", self
        )
        import_phantom_action.triggered.connect(self._import_phantom)
        toolbar.addAction(import_phantom_action)

        # Nút xuất phantom ra DICOM
        export_phantom_action = QAction(
            QIcon.fromTheme("document-save"), "Xuất ra DICOM", self
        )
        export_phantom_action.triggered.connect(self._export_phantom)
        toolbar.addAction(export_phantom_action)

        # Thêm thanh công cụ vào layout
        layout.addWidget(toolbar)

        # Tạo layout chính
        main_layout = QHBoxLayout()

        # Panel bên trái: Danh sách phantom
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Label hướng dẫn
        left_layout.addWidget(QLabel("Thư viện Phantom:"))

        # Danh sách phantom
        self.phantom_list = QListWidget()
        self.phantom_list.itemSelectionChanged.connect(self._on_phantom_selected)
        left_layout.addWidget(self.phantom_list)

        # Panel bên phải: Chi tiết phantom
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Thông tin phantom
        info_group = QGroupBox("Thông tin Phantom")
        info_layout = QFormLayout(info_group)

        self.phantom_name = QLineEdit()
        self.phantom_type = QComboBox()
        self.phantom_type.addItems(["Nước", "Chất dẻo", "Tùy chỉnh"])
        self.phantom_description = QTextEdit()

        info_layout.addRow("Tên:", self.phantom_name)
        info_layout.addRow("Loại:", self.phantom_type)
        info_layout.addRow("Mô tả:", self.phantom_description)

        right_layout.addWidget(info_group)

        # Thông số vật lý
        physics_group = QGroupBox("Thông số vật lý")
        physics_layout = QFormLayout(physics_group)

        self.phantom_density = QDoubleSpinBox()
        self.phantom_density.setRange(0.1, 10.0)
        self.phantom_density.setValue(1.0)
        self.phantom_density.setSingleStep(0.1)

        self.phantom_dimensions = QLineEdit()
        self.phantom_dimensions.setPlaceholderText("Width x Height x Depth (mm)")

        physics_layout.addRow("Mật độ (g/cm³):", self.phantom_density)
        physics_layout.addRow("Kích thước:", self.phantom_dimensions)

        right_layout.addWidget(physics_group)

        # Nút lưu thay đổi
        self.save_phantom_button = QPushButton("Lưu thay đổi")
        self.save_phantom_button.clicked.connect(self._save_phantom)
        right_layout.addWidget(self.save_phantom_button)

        # Thêm các panel vào layout chính
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)
        layout.addLayout(main_layout)

        # Tải danh sách phantom
        self._load_phantoms()

        return tab

    def _load_phantoms(self):
        """Tải danh sách phantom từ thư viện."""
        self.phantom_list.clear()

        try:
            from quangtps.evaluation.qa.phantom import PhantomLibrary

            phantoms = PhantomLibrary.get_all_phantoms()

            for phantom in phantoms:
                item = QListWidgetItem(phantom.name)
                item.setData(Qt.UserRole, phantom)
                self.phantom_list.addItem(item)
        except Exception as e:
            logger.error(f"Lỗi khi tải danh sách phantom: {str(e)}")
            QMessageBox.warning(
                self, "Lỗi", f"Không thể tải danh sách phantom: {str(e)}"
            )

    def _on_phantom_selected(self):
        """Xử lý khi chọn một phantom từ danh sách."""
        items = self.phantom_list.selectedItems()
        if not items:
            return

        phantom = items[0].data(Qt.UserRole)

        # Hiển thị thông tin phantom
        self.phantom_name.setText(phantom.name)
        index = self.phantom_type.findText(phantom.type)
        if index >= 0:
            self.phantom_type.setCurrentIndex(index)
        self.phantom_description.setText(phantom.description)
        self.phantom_density.setValue(phantom.density)

        dimensions = f"{phantom.width} x {phantom.height} x {phantom.depth}"
        self.phantom_dimensions.setText(dimensions)

    def _add_phantom(self):
        """Thêm phantom mới vào thư viện."""
        name, ok = QInputDialog.getText(self, "Thêm Phantom", "Tên phantom:")

        if ok and name:
            try:
                from quangtps.evaluation.qa.phantom import Phantom, PhantomLibrary

                # Tạo phantom mới
                phantom = Phantom(
                    name=name,
                    type="Tùy chỉnh",
                    description="",
                    density=1.0,
                    width=300,
                    height=300,
                    depth=300,
                )

                # Thêm vào thư viện
                PhantomLibrary.add_phantom(phantom)

                # Cập nhật danh sách
                self._load_phantoms()

                # Chọn phantom mới
                for i in range(self.phantom_list.count()):
                    if self.phantom_list.item(i).text() == name:
                        self.phantom_list.setCurrentRow(i)
                        break

            except Exception as e:
                logger.error(f"Lỗi khi thêm phantom: {str(e)}")
                QMessageBox.critical(self, "Lỗi", f"Không thể thêm phantom: {str(e)}")

    def _remove_phantom(self):
        """Xóa phantom khỏi thư viện."""
        items = self.phantom_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một phantom để xóa.")
            return

        phantom = items[0].data(Qt.UserRole)

        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa phantom '{phantom.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                from quangtps.evaluation.qa.phantom import PhantomLibrary

                # Xóa phantom
                PhantomLibrary.remove_phantom(phantom.id)

                # Cập nhật danh sách
                self._load_phantoms()

            except Exception as e:
                logger.error(f"Lỗi khi xóa phantom: {str(e)}")
                QMessageBox.critical(self, "Lỗi", f"Không thể xóa phantom: {str(e)}")

    def _save_phantom(self):
        """Lưu thay đổi cho phantom hiện tại."""
        items = self.phantom_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một phantom để lưu.")
            return

        phantom = items[0].data(Qt.UserRole)

        try:
            from quangtps.evaluation.qa.phantom import PhantomLibrary

            # Cập nhật thông tin
            phantom.name = self.phantom_name.text()
            phantom.type = self.phantom_type.currentText()
            phantom.description = self.phantom_description.toPlainText()
            phantom.density = self.phantom_density.value()

            # Xử lý kích thước
            dimensions = self.phantom_dimensions.text().split("x")
            if len(dimensions) == 3:
                phantom.width = float(dimensions[0].strip())
                phantom.height = float(dimensions[1].strip())
                phantom.depth = float(dimensions[2].strip())

            # Lưu vào thư viện
            PhantomLibrary.update_phantom(phantom)

            # Cập nhật danh sách
            self._load_phantoms()

            QMessageBox.information(
                self, "Thành công", f"Đã lưu thay đổi cho phantom '{phantom.name}'."
            )

        except Exception as e:
            logger.error(f"Lỗi khi lưu phantom: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu phantom: {str(e)}")

    def _import_phantom(self):
        """Nhập phantom từ file DICOM."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file DICOM", "", "DICOM Files (*.dcm);;All Files (*.*)"
        )

        if file_path:
            try:
                from quangtps.evaluation.qa.phantom import PhantomLibrary

                # Nhập phantom từ DICOM
                phantom = PhantomLibrary.import_from_dicom(file_path)

                # Cập nhật danh sách
                self._load_phantoms()

                # Chọn phantom mới
                for i in range(self.phantom_list.count()):
                    if self.phantom_list.item(i).text() == phantom.name:
                        self.phantom_list.setCurrentRow(i)
                        break

                QMessageBox.information(
                    self, "Thành công", f"Đã nhập phantom '{phantom.name}' từ DICOM."
                )

            except Exception as e:
                logger.error(f"Lỗi khi nhập phantom từ DICOM: {str(e)}")
                QMessageBox.critical(
                    self, "Lỗi", f"Không thể nhập phantom từ DICOM: {str(e)}"
                )

    def _export_phantom(self):
        """Xuất phantom ra file DICOM."""
        items = self.phantom_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một phantom để xuất.")
            return

        phantom = items[0].data(Qt.UserRole)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu file DICOM",
            f"{phantom.name}.dcm",
            "DICOM Files (*.dcm);;All Files (*.*)",
        )

        if file_path:
            try:
                from quangtps.evaluation.qa.phantom import PhantomLibrary

                # Xuất phantom sang DICOM
                PhantomLibrary.export_to_dicom(phantom, file_path)

                QMessageBox.information(
                    self, "Thành công", f"Đã xuất phantom '{phantom.name}' ra DICOM."
                )

            except Exception as e:
                logger.error(f"Lỗi khi xuất phantom ra DICOM: {str(e)}")
                QMessageBox.critical(
                    self, "Lỗi", f"Không thể xuất phantom ra DICOM: {str(e)}"
                )

    def _load_qa_tests(self):
        """Tải dữ liệu các bài kiểm tra QA."""
        try:
            # Lấy danh sách các bài kiểm tra QA
            self.qa_tests = self.qa_manager.get_all_tests()

            # Xóa các mục cũ trong bảng
            self.tests_table.setRowCount(0)

            # Thêm các bài kiểm tra vào bảng
            for i, test in enumerate(self.qa_tests):
                self.tests_table.insertRow(i)

                # ID
                id_item = QTableWidgetItem(test.test_id)
                id_item.setData(Qt.UserRole, test.test_id)
                self.tests_table.setItem(i, 0, id_item)

                # Tên
                self.tests_table.setItem(i, 1, QTableWidgetItem(test.test_name))

                # Loại
                self.tests_table.setItem(i, 2, QTableWidgetItem(test.test_type.value))

                # Giao thức
                self.tests_table.setItem(i, 3, QTableWidgetItem(test.protocol.value))

                # Trạng thái
                status_item = QTableWidgetItem(test.status.value)

                # Màu sắc trạng thái
                if test.status == QAStatus.PASSED:
                    status_item.setForeground(QColor("green"))
                elif test.status == QAStatus.FAILED:
                    status_item.setForeground(QColor("red"))
                elif test.status == QAStatus.IN_PROGRESS:
                    status_item.setForeground(QColor("blue"))

                self.tests_table.setItem(i, 4, status_item)

                # Ngày tạo
                date_str = test.created_date.strftime("%d/%m/%Y")
                self.tests_table.setItem(i, 5, QTableWidgetItem(date_str))

                # Người thực hiện
                performer = test.performed_by if test.performed_by else "Chưa có"
                self.tests_table.setItem(i, 6, QTableWidgetItem(performer))

            # Nếu không có bài kiểm tra nào, hiển thị thông báo
            if self.tests_table.rowCount() == 0:
                self._show_empty_message()
            else:
                self._hide_empty_message()

            logger.info("Đã tải %d bài kiểm tra QA", len(self.qa_tests))

        except Exception as e:
            logger.exception("Lỗi khi tải danh sách các bài kiểm tra QA: %s", str(e))
            QMessageBox.warning(
                self,
                "Lỗi tải dữ liệu QA",
                f"Không thể tải danh sách các bài kiểm tra QA: {str(e)}\n\n"
                "Tạo dữ liệu mẫu để minh họa.",
            )

            # Tạo dữ liệu mẫu
            self._create_sample_data()

    def _show_empty_message(self):
        """Hiển thị thông báo khi không có bài kiểm tra nào."""
        # Kiểm tra xem đã có label thông báo chưa
        if not hasattr(self, "empty_message_label"):
            self.empty_message_label = QLabel(
                "Không có bài kiểm tra QA nào.\nNhấn nút 'Tạo mới' để tạo bài kiểm tra."
            )
            self.empty_message_label.setAlignment(Qt.AlignCenter)
            self.empty_message_label.setStyleSheet("color: gray; font-size: 14px;")

            # Thêm vào layout giữa bảng và các nút
            layout_index = self.layout().indexOf(self.tests_table)
            if layout_index >= 0:
                self.layout().insertWidget(layout_index + 1, self.empty_message_label)
        else:
            self.empty_message_label.setVisible(True)

    def _hide_empty_message(self):
        """Ẩn thông báo khi có bài kiểm tra."""
        if hasattr(self, "empty_message_label"):
            self.empty_message_label.setVisible(False)

    def _create_sample_data(self):
        """Tạo dữ liệu mẫu khi không thể tải dữ liệu thật."""
        try:
            # Xóa dữ liệu cũ
            self.tests_table.setRowCount(0)

            # Tạo dữ liệu mẫu
            sample_data = [
                {
                    "id": "QA-001",
                    "name": "Kiểm tra output beam 6MV",
                    "type": "Output Verification",
                    "protocol": "Monthly",
                    "status": "Passed",
                    "date": "01/04/2025",
                    "performer": "Nguyễn Văn A",
                },
                {
                    "id": "QA-002",
                    "name": "Kiểm tra MLC positioning",
                    "type": "MLC QA",
                    "protocol": "Weekly",
                    "status": "Failed",
                    "date": "02/04/2025",
                    "performer": "Trần Thị B",
                },
                {
                    "id": "QA-003",
                    "name": "Kiểm tra planar dose",
                    "type": "Patient QA",
                    "protocol": "Pre-treatment",
                    "status": "In progress",
                    "date": "03/04/2025",
                    "performer": "Lê Văn C",
                },
            ]

            for i, test in enumerate(sample_data):
                self.tests_table.insertRow(i)

                # ID
                id_item = QTableWidgetItem(test["id"])
                id_item.setData(Qt.UserRole, test["id"])
                self.tests_table.setItem(i, 0, id_item)

                # Tên
                self.tests_table.setItem(i, 1, QTableWidgetItem(test["name"]))

                # Loại
                self.tests_table.setItem(i, 2, QTableWidgetItem(test["type"]))

                # Giao thức
                self.tests_table.setItem(i, 3, QTableWidgetItem(test["protocol"]))

                # Trạng thái
                status_item = QTableWidgetItem(test["status"])

                # Màu sắc trạng thái
                if test["status"] == "Passed":
                    status_item.setForeground(QColor("green"))
                elif test["status"] == "Failed":
                    status_item.setForeground(QColor("red"))
                elif test["status"] == "In progress":
                    status_item.setForeground(QColor("blue"))

                self.tests_table.setItem(i, 4, status_item)

                # Ngày tạo
                self.tests_table.setItem(i, 5, QTableWidgetItem(test["date"]))

                # Người thực hiện
                self.tests_table.setItem(i, 6, QTableWidgetItem(test["performer"]))

            logger.info("Đã tạo %d bài kiểm tra QA mẫu", len(sample_data))

        except Exception as e:
            logger.exception("Lỗi khi tạo dữ liệu mẫu QA: %s", str(e))
            QMessageBox.critical(
                self, "Lỗi nghiêm trọng", f"Không thể hiển thị dữ liệu QA: {str(e)}"
            )

    def _apply_filters(self):
        """Áp dụng bộ lọc cho danh sách bài kiểm tra QA."""
        # Lấy giá trị các bộ lọc
        name_filter = self.filter_name.text().lower()
        type_filter = self.filter_type.itemData(self.filter_type.currentIndex())
        status_filter = self.filter_status.itemData(self.filter_status.currentIndex())

        # Ẩn/hiện các hàng theo bộ lọc
        for row in range(self.tests_table.rowCount()):
            test_id = self.tests_table.item(row, 0).data(Qt.UserRole)
            if test_id in self.qa_tests:
                test = self.qa_tests[test_id]
                test_name = test.test_name.lower()
                test_type = test.test_type
                test_status = test.status

                # Kiểm tra từng điều kiện lọc
                name_match = name_filter == "" or name_filter in test_name
                type_match = type_filter is None or type_filter == test_type
                status_match = status_filter is None or status_filter == test_status

                self.tests_table.setRowHidden(
                    row, not (name_match and type_match and status_match)
                )

        # Hiển thị thông báo nếu không có kết quả
        visible_count = 0
        for row in range(self.tests_table.rowCount()):
            if not self.tests_table.isRowHidden(row):
                visible_count += 1

        if visible_count == 0 and self.tests_table.rowCount() > 0:
            self._show_empty_message(
                "Không tìm thấy bài kiểm tra QA phù hợp với bộ lọc"
            )
        else:
            self._hide_empty_message()

    def _on_test_selected(self):
        """Xử lý sự kiện khi chọn một bài kiểm tra trong bảng."""
        selected_items = self.tests_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        test_id = self.tests_table.item(row, 0).data(Qt.UserRole)
        self.test_details.load_test(test_id)

    def _on_test_updated(self, test_id: str):
        """Xử lý sự kiện khi một bài kiểm tra được cập nhật."""
        self._load_qa_tests()

        # Chọn lại bài kiểm tra vừa cập nhật
        for i in range(self.tests_table.rowCount()):
            if self.tests_table.item(i, 0).data(Qt.UserRole) == test_id:
                self.tests_table.selectRow(i)
                break

    def _create_new_test(self):
        """Tạo một bài kiểm tra QA mới."""
        # Hiển thị dialog để nhập thông tin bài kiểm tra mới
        dialog = QANewTestDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        # Lấy thông tin từ dialog
        test_data = dialog.get_test_data()

        # Tạo bài kiểm tra mới
        try:
            new_test = TreatmentQATest(
                test_name=test_data["name"],
                test_type=test_data["type"],
                protocol=test_data["protocol"],
                plan_id=test_data.get("plan_id"),
                patient_id=test_data.get("patient_id"),
            )

            # Thêm bài kiểm tra vào TreatmentQAManager
            self.qa_manager.add_test(new_test)

            # Làm mới danh sách và chọn bài kiểm tra mới
            self._load_qa_tests()

            # Tìm và chọn bài kiểm tra mới trong bảng
            for i in range(self.tests_table.rowCount()):
                if self.tests_table.item(i, 0).data(Qt.UserRole) == new_test.test_id:
                    self.tests_table.selectRow(i)
                    break

            QMessageBox.information(
                self, "Thành công", "Đã tạo bài kiểm tra QA mới thành công."
            )
        except Exception as e:
            logger.error(f"Lỗi khi tạo bài kiểm tra QA mới: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể tạo bài kiểm tra mới: {e}")

    def _delete_test(self):
        """Xóa bài kiểm tra QA đã chọn."""
        selected_items = self.tests_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng chọn một bài kiểm tra để xóa."
            )
            return

        row = selected_items[0].row()
        test_id = self.tests_table.item(row, 0).data(Qt.UserRole)
        test_name = self.tests_table.item(row, 1).text()

        # Xác nhận xóa
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa bài kiểm tra '{test_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # Xóa bài kiểm tra
        try:
            self.qa_manager.delete_test(test_id)
            self._load_qa_tests()
            self.test_details.load_test(None)  # Xóa hiển thị chi tiết
            QMessageBox.information(
                self, "Thành công", f"Đã xóa bài kiểm tra '{test_name}'."
            )
        except Exception as e:
            logger.error(f"Lỗi khi xóa bài kiểm tra QA: {e}")
            QMessageBox.warning(self, "Lỗi", f"Không thể xóa bài kiểm tra: {e}")

    def _show_rule_manager(self):
        """Hiển thị tab quản lý quy tắc kiểm tra."""
        self.tab_widget.setCurrentWidget(self.rules_tab)

    def _on_rules_changed(self):
        """Xử lý khi danh sách quy tắc thay đổi."""
        # Lưu quy tắc vào file cấu hình nếu cần
        # Hiện tại chỉ log thông báo
        logger.info("Danh sách quy tắc kiểm tra đã thay đổi")

    def _on_log_analysis_completed(self, results):
        """Xử lý khi phân tích log file hoàn tất."""
        # Kiểm tra nếu đã chọn sử dụng validator
        if (
            hasattr(self, "use_validator_check")
            and self.use_validator_check.isChecked()
        ):
            self._validate_log_results(results)

        logger.info(
            f"Đã phân tích log file thành công với {len(results.get('deviations', []))} sai lệch"
        )

    def _validate_log_results(self, results):
        """Kiểm tra kết quả log file với các quy tắc đã định nghĩa."""
        if not hasattr(self, "rule_manager_widget"):
            return

        try:
            # Lấy validator từ rule manager
            validator = self.rule_manager_widget.get_validator()

            # Lấy dữ liệu log từ kết quả phân tích
            log_data = results.get("log_data")
            if log_data is None:
                logger.warning("Không có dữ liệu log để kiểm tra")
                return

            # Lấy analyzer từ kết quả nếu có
            analyzer = results.get("analyzer")
            if analyzer is None:
                # Tạo analyzer mới nếu không có trong kết quả
                from quangtps.evaluation.qa.machine_log_analyzer import LogFileAnalyzer

                analyzer = LogFileAnalyzer(
                    log_file_path=None,
                    log_type=results.get("log_type"),
                )
                analyzer.deviations = results.get("deviations", [])
                analyzer.summary = results.get("summary", {})
                analyzer.pass_rate = results.get("pass_rate", 0)
                analyzer.max_deviation = results.get("max_deviation", {})

            # Cài đặt ánh xạ tham số phổ biến cho validator
            common_params = {
                "gantry_angle": "gantry_angle",
                "collimator_angle": "collimator_angle",
                "couch_position": "couch_position",
                "mlc_position": "mlc_position",
                "jaw_position": "jaw_position",
                "dose_rate": "dose_rate",
                "mu_delivery": "mu_delivery",
            }
            validator.set_parameter_map(common_params)

            # Thực hiện kiểm tra
            validation_results = validator.validate(log_data, analyzer)

            # Hiển thị kết quả kiểm tra
            self._show_validation_results(validation_results)

        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra log file với validator: {str(e)}")
            import traceback

            traceback.print_exc()

    def _show_validation_results(self, validation_results):
        """Hiển thị kết quả kiểm tra validator."""
        if not validation_results:
            logger.info("Không có kết quả kiểm tra validator")
            return

        # Đếm số lượng quy tắc đạt/không đạt
        passed = sum(1 for r in validation_results if r.passed)
        failed = len(validation_results) - passed

        # Tạo dialog hiển thị kết quả chi tiết
        dialog = QDialog(self)
        dialog.setWindowTitle("Kết quả kiểm tra log file")
        dialog.setMinimumSize(700, 500)
        layout = QVBoxLayout(dialog)

        # Thông tin tóm tắt
        summary_label = QLabel(
            f"<b>Tổng số quy tắc kiểm tra:</b> {len(validation_results)}"
        )
        summary_label.setTextFormat(Qt.RichText)
        layout.addWidget(summary_label)

        passed_label = QLabel(f"<b>Đạt:</b> {passed}")
        passed_label.setTextFormat(Qt.RichText)
        layout.addWidget(passed_label)

        failed_label = QLabel(f"<b>Không đạt:</b> {failed}")
        failed_label.setTextFormat(Qt.RichText)
        layout.addWidget(failed_label)

        # Bảng kết quả chi tiết
        results_table = QTableWidget()
        results_table.setColumnCount(5)
        results_table.setHorizontalHeaderLabels(
            ["Quy tắc", "Tham số", "Kết quả", "Giá trị", "Thông báo"]
        )
        results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        results_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )

        # Thêm dữ liệu vào bảng
        results_table.setRowCount(len(validation_results))

        for i, result in enumerate(validation_results):
            # Tên quy tắc
            name_item = QTableWidgetItem(result.rule.name)
            results_table.setItem(i, 0, name_item)

            # Tham số
            param_item = QTableWidgetItem(result.rule.parameter or "-")
            results_table.setItem(i, 1, param_item)

            # Kết quả
            result_item = QTableWidgetItem("Đạt" if result.passed else "Không đạt")
            result_item.setForeground(QColor("green" if result.passed else "red"))
            result_item.setTextAlignment(Qt.AlignCenter)
            results_table.setItem(i, 2, result_item)

            # Giá trị
            value_text = ""
            if result.value is not None:
                if result.expected is not None:
                    value_text = f"{result.value} (Mong đợi: {result.expected})"
                else:
                    value_text = str(result.value)
            value_item = QTableWidgetItem(value_text)
            results_table.setItem(i, 3, value_item)

            # Thông báo
            message_item = QTableWidgetItem(result.message or "")
            results_table.setItem(i, 4, message_item)

            # Đặt màu nền cho hàng theo kết quả
            if not result.passed:
                for col in range(5):
                    cell = results_table.item(i, col)
                    if cell:
                        severity_colors = {
                            DeviationSeverity.CRITICAL: QColor(255, 200, 200),
                            DeviationSeverity.MAJOR: QColor(255, 220, 200),
                            DeviationSeverity.MODERATE: QColor(255, 255, 200),
                            DeviationSeverity.MINOR: QColor(220, 255, 220),
                            DeviationSeverity.ACCEPTABLE: QColor(200, 255, 255),
                        }
                        cell.setBackground(
                            severity_colors.get(
                                result.rule.severity, QColor(255, 220, 220)
                            )
                        )

        layout.addWidget(results_table, 1)

        # Nút đóng
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Hiển thị thông báo nếu có lỗi nghiêm trọng
        critical_failures = sum(
            1
            for r in validation_results
            if not r.passed and r.rule.severity == DeviationSeverity.CRITICAL
        )

        if critical_failures > 0:
            QMessageBox.critical(
                self,
                "Cảnh báo nghiêm trọng",
                f"Phát hiện {critical_failures} lỗi nghiêm trọng trong kết quả kiểm tra log file!",
            )
        elif failed > 0:
            QMessageBox.warning(
                self,
                "Kết quả kiểm tra log file",
                f"Có {failed}/{len(validation_results)} quy tắc kiểm tra không đạt.",
            )

        # Hiển thị dialog
        dialog.exec_()

    def _on_log_analysis_failed(self, error_message):
        """Xử lý khi phân tích log file thất bại."""
        logger.error(f"Lỗi khi phân tích log file: {error_message}")
        QMessageBox.critical(
            self,
            "Lỗi phân tích log file",
            f"Không thể phân tích log file: {error_message}",
        )


class QANewTestDialog(QDialog):
    """Dialog để tạo một bài kiểm tra QA mới."""

    def __init__(self, parent=None):
        """Khởi tạo dialog tạo bài kiểm tra QA mới."""
        super().__init__(parent)
        self.setWindowTitle("Tạo bài kiểm tra QA mới")
        self.setMinimumWidth(500)

        self.patient_db = PatientDatabase()
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện dialog."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        form_layout = QFormLayout()

        # Trường nhập tên bài kiểm tra
        self.test_name = QLineEdit()
        form_layout.addRow("Tên bài kiểm tra:", self.test_name)

        # Combobox loại kiểm tra
        self.test_type = QComboBox()
        for test_type in QATestType:
            self.test_type.addItem(test_type.value, test_type)
        form_layout.addRow("Loại kiểm tra:", self.test_type)

        # Combobox protocol
        self.protocol = QComboBox()
        for protocol in QAProtocol:
            self.protocol.addItem(protocol.value, protocol)
        form_layout.addRow("Giao thức:", self.protocol)

        # Combobox chọn bệnh nhân (optional)
        self.patient_combo = QComboBox()
        self.patient_combo.addItem("Không áp dụng", None)
        try:
            patients = self.patient_db.get_all_patients()
            for patient in patients:
                self.patient_combo.addItem(
                    f"{patient.patient_id} - {patient.name}", patient.patient_id
                )
        except:
            # Xử lý trường hợp không có kết nối đến cơ sở dữ liệu
            pass

        form_layout.addRow("Bệnh nhân:", self.patient_combo)

        # Combobox chọn kế hoạch điều trị (optional, sẽ được cập nhật khi chọn bệnh nhân)
        self.plan_combo = QComboBox()
        self.plan_combo.addItem("Không áp dụng", None)
        form_layout.addRow("Kế hoạch điều trị:", self.plan_combo)

        # Kết nối sự kiện thay đổi bệnh nhân
        self.patient_combo.currentIndexChanged.connect(self._update_plans)

        layout.addLayout(form_layout)

        # Nút điều khiển
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addSpacing(20)
        layout.addWidget(button_box)

    def _update_plans(self):
        """Cập nhật danh sách kế hoạch khi chọn bệnh nhân."""
        self.plan_combo.clear()
        self.plan_combo.addItem("Không áp dụng", None)

        patient_idx = self.patient_combo.currentIndex()
        if patient_idx <= 0:
            return

        patient_id = self.patient_combo.itemData(patient_idx)
        if not patient_id:
            return

        try:
            # Lấy danh sách kế hoạch điều trị của bệnh nhân
            patient = self.patient_db.get_patient(patient_id)
            if patient and patient.plans:
                for plan in patient.plans:
                    self.plan_combo.addItem(
                        f"{plan.plan_id} - {plan.plan_name}", plan.plan_id
                    )
        except Exception as e:
            logger.error(f"Lỗi khi tải danh sách kế hoạch: {e}")

    def get_test_data(self) -> Dict[str, Any]:
        """Lấy dữ liệu từ dialog."""
        data = {
            "name": self.test_name.text().strip(),
            "type": self.test_type.itemData(self.test_type.currentIndex()),
            "protocol": self.protocol.itemData(self.protocol.currentIndex()),
        }

        # Thêm patient_id nếu có
        patient_idx = self.patient_combo.currentIndex()
        if patient_idx > 0:
            data["patient_id"] = self.patient_combo.itemData(patient_idx)

        # Thêm plan_id nếu có
        plan_idx = self.plan_combo.currentIndex()
        if plan_idx > 0:
            data["plan_id"] = self.plan_combo.itemData(plan_idx)

        return data

    def accept(self):
        """Xác thực dữ liệu trước khi chấp nhận dialog."""
        if not self.test_name.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên bài kiểm tra.")
            return

        super().accept()
