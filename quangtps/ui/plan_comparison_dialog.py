#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Comparison Dialog Module

Module này triển khai dialog so sánh kế hoạch xạ trị
cho phép người dùng so sánh nhiều kế hoạch khác nhau cùng lúc
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any

# Import PyQt
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
        QCheckBox,
        QGroupBox,
        QFrame,
        QSplitter,
        QScrollArea,
        QWidget,
        QTabWidget,
        QFileDialog,
        QMessageBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from quangtps.core.logging import get_logger
from quangtps.evaluation.dvh import DVHAnalyzer
from quangtps.evaluation.clinical_goals import ClinicalGoal
from quangtps.evaluation.plan_quality import PlanQualityEvaluator
from quangtps.evaluation.protocol_manager import ProtocolManager

logger = get_logger(__name__)


class PlanComparisonDialog(QDialog):
    """
    Dialog để so sánh nhiều kế hoạch xạ trị.

    Chức năng:
    - Chọn nhiều kế hoạch để so sánh
    - So sánh DVH của các kế hoạch
    - So sánh thống kê liều của các kế hoạch
    - So sánh các mục tiêu lâm sàng đạt được
    - Xuất báo cáo so sánh sang PDF/HTML
    """

    def __init__(self, parent=None):
        """
        Khởi tạo dialog so sánh kế hoạch.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha của dialog
        """
        super().__init__(parent)

        self.setWindowTitle("So sánh kế hoạch xạ trị")
        self.resize(1000, 700)

        self.plans = []  # Danh sách kế hoạch để so sánh
        self.dvh_analyzers = {}  # Dictionary lưu trữ DVH analyzer cho mỗi kế hoạch
        self.protocol_manager = ProtocolManager()
        self.current_protocol = None

        # Thiết lập UI
        self._init_ui()

        # Kết nối signals và slots
        self._connect_signals()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng"""
        if not HAS_PYQT:
            logger.error("PyQt5 không khả dụng. Không thể khởi tạo UI.")
            return

        # Layout chính
        main_layout = QVBoxLayout(self)

        # Panel chọn kế hoạch
        plan_selection_group = QGroupBox("Chọn kế hoạch để so sánh")
        plan_layout = QHBoxLayout(plan_selection_group)

        # Combo box chọn kế hoạch
        self.plan_combo = QComboBox()
        self.plan_combo.setMinimumWidth(250)

        # Nút thêm kế hoạch
        self.add_plan_button = QPushButton("Thêm kế hoạch")
        self.add_plan_button.setIcon(QIcon("quangtps/ui/icons/new_icons/add.png"))

        # Nút xóa kế hoạch
        self.remove_plan_button = QPushButton("Xóa kế hoạch")
        self.remove_plan_button.setIcon(QIcon("quangtps/ui/icons/new_icons/remove.png"))

        # Nút xóa tất cả
        self.clear_plans_button = QPushButton("Xóa tất cả")
        self.clear_plans_button.setIcon(QIcon("quangtps/ui/icons/new_icons/clear.png"))

        # Thêm widgets vào layout
        plan_layout.addWidget(self.plan_combo)
        plan_layout.addWidget(self.add_plan_button)
        plan_layout.addWidget(self.remove_plan_button)
        plan_layout.addWidget(self.clear_plans_button)

        # Bảng kế hoạch đã chọn
        self.plans_table = QTableWidget(0, 5)
        self.plans_table.setHorizontalHeaderLabels(
            ["Kế hoạch", "Bệnh nhân", "Liều tham chiếu", "Ngày tạo", "Enabled"]
        )
        self.plans_table.setMinimumHeight(150)

        # Chọn protocol để đánh giá
        protocol_layout = QHBoxLayout()

        protocol_label = QLabel("Protocol:")
        self.protocol_combo = QComboBox()
        self.protocol_combo.setMinimumWidth(250)

        # Nút đánh giá
        self.evaluate_button = QPushButton("Đánh giá kế hoạch")
        self.evaluate_button.setIcon(QIcon("quangtps/ui/icons/new_icons/evaluate.png"))

        protocol_layout.addWidget(protocol_label)
        protocol_layout.addWidget(self.protocol_combo)
        protocol_layout.addStretch()
        protocol_layout.addWidget(self.evaluate_button)

        # Tab hiển thị kết quả so sánh
        self.results_tabs = QTabWidget()

        # Tab DVH
        self.dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(self.dvh_tab)

        # Danh sách cấu trúc
        structures_layout = QHBoxLayout()
        structures_label = QLabel("Cấu trúc:")
        self.structures_combo = QComboBox()

        structures_layout.addWidget(structures_label)
        structures_layout.addWidget(self.structures_combo)
        structures_layout.addStretch()

        # Placeholder cho DVH
        self.dvh_placeholder = QLabel("Chọn kế hoạch và đánh giá để hiển thị DVH")
        self.dvh_placeholder.setAlignment(Qt.AlignCenter)
        self.dvh_placeholder.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ddd; padding: 20px;"
        )

        dvh_layout.addLayout(structures_layout)
        dvh_layout.addWidget(self.dvh_placeholder)

        # Tab mục tiêu lâm sàng
        self.goals_tab = QWidget()
        goals_layout = QVBoxLayout(self.goals_tab)

        self.goals_table = QTableWidget(0, 7)
        self.goals_table.setHorizontalHeaderLabels(
            [
                "Cấu trúc",
                "Loại",
                "Điều kiện",
                "Giá trị mục tiêu",
                "Kết quả",
                "Giá trị thực tế",
                "So sánh",
            ]
        )

        goals_layout.addWidget(self.goals_table)

        # Tab thống kê liều
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.stats_tab)

        self.stats_table = QTableWidget(0, 8)
        self.stats_table.setHorizontalHeaderLabels(
            ["Cấu trúc", "Kế hoạch", "Min", "Max", "Mean", "D98", "D95", "D50"]
        )

        stats_layout.addWidget(self.stats_table)

        # Tab chỉ số chất lượng
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)

        self.metrics_table = QTableWidget(0, 5)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Chỉ số", "Kế hoạch", "Giá trị", "Đơn vị", "So sánh"]
        )

        metrics_layout.addWidget(self.metrics_table)

        # Thêm các tab vào tabwidget
        self.results_tabs.addTab(self.dvh_tab, "DVH")
        self.results_tabs.addTab(self.goals_tab, "Mục tiêu lâm sàng")
        self.results_tabs.addTab(self.stats_tab, "Thống kê liều")
        self.results_tabs.addTab(self.metrics_tab, "Chỉ số chất lượng")

        # Nút xuất báo cáo
        buttons_layout = QHBoxLayout()

        self.export_pdf_button = QPushButton("Xuất báo cáo PDF")
        self.export_pdf_button.setIcon(QIcon("quangtps/ui/icons/new_icons/pdf.png"))

        self.export_html_button = QPushButton("Xuất báo cáo HTML")
        self.export_html_button.setIcon(QIcon("quangtps/ui/icons/new_icons/html.png"))

        self.export_csv_button = QPushButton("Xuất dữ liệu CSV")
        self.export_csv_button.setIcon(QIcon("quangtps/ui/icons/new_icons/csv.png"))

        self.close_button = QPushButton("Đóng")
        self.close_button.setIcon(QIcon("quangtps/ui/icons/new_icons/close.png"))

        buttons_layout.addWidget(self.export_pdf_button)
        buttons_layout.addWidget(self.export_html_button)
        buttons_layout.addWidget(self.export_csv_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.close_button)

        # Thêm tất cả vào layout chính
        main_layout.addWidget(plan_selection_group)
        main_layout.addWidget(self.plans_table)
        main_layout.addLayout(protocol_layout)
        main_layout.addWidget(self.results_tabs, 1)
        main_layout.addLayout(buttons_layout)

        # Disable các nút chưa sẵn sàng
        self.remove_plan_button.setEnabled(False)
        self.clear_plans_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)
        self.export_pdf_button.setEnabled(False)
        self.export_html_button.setEnabled(False)
        self.export_csv_button.setEnabled(False)

    def _connect_signals(self):
        """Kết nối các signals và slots."""
        # Kết nối các nút với các hàm xử lý
        self.add_plan_button.clicked.connect(self._add_plan)
        self.remove_plan_button.clicked.connect(self._remove_plan)
        self.clear_plans_button.clicked.connect(self._clear_plans)
        self.evaluate_button.clicked.connect(self._evaluate_plans)
        self.close_button.clicked.connect(self.reject)

        # Kết nối các signals export
        self.export_pdf_button.clicked.connect(lambda: self._export_report("pdf"))
        self.export_html_button.clicked.connect(lambda: self._export_report("html"))
        self.export_csv_button.clicked.connect(lambda: self._export_report("csv"))

        # Kết nối combo box
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
        self.structures_combo.currentIndexChanged.connect(self._on_structure_changed)

    def _update_plans_list(self):
        """Cập nhật danh sách kế hoạch trong UI"""
        # Cập nhật danh sách kế hoạch
        self.plans_table.setRowCount(len(self.plans))

        for i, plan in enumerate(self.plans):
            # Tên kế hoạch
            name_item = QTableWidgetItem(plan.name)
            self.plans_table.setItem(i, 0, name_item)

            # Tên bệnh nhân
            patient_name = getattr(plan, "patient_name", "N/A")
            patient_item = QTableWidgetItem(patient_name)
            self.plans_table.setItem(i, 1, patient_item)

            # Liều tham chiếu
            dose = getattr(plan, "prescription_dose", 0)
            dose_item = QTableWidgetItem(f"{dose:.2f} Gy")
            self.plans_table.setItem(i, 2, dose_item)

            # Ngày tạo
            date = getattr(plan, "creation_date", "N/A")
            date_item = QTableWidgetItem(str(date))
            self.plans_table.setItem(i, 3, date_item)

            # Checkbox Enabled
            checkbox = QTableWidgetItem()
            checkbox.setCheckState(Qt.Checked)
            self.plans_table.setItem(i, 4, checkbox)

        # Cập nhật trạng thái các nút
        has_plans = len(self.plans) > 0
        self.remove_plan_button.setEnabled(has_plans)
        self.clear_plans_button.setEnabled(has_plans)
        self.evaluate_button.setEnabled(has_plans and self.protocol_combo.count() > 0)

        # Điều chỉnh kích thước cột
        self.plans_table.resizeColumnsToContents()

    def _add_plan(self):
        """Thêm kế hoạch vào danh sách so sánh"""
        # Trong thực tế, cần phải lấy kế hoạch từ hệ thống
        # Đây là phương thức giả để minh họa
        plan = self.plan_combo.currentData()

        if plan and plan not in self.plans:
            self.plans.append(plan)
            self._update_plans_list()

    def _remove_plan(self):
        """Xóa kế hoạch khỏi danh sách so sánh"""
        selected_rows = self.plans_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        # Xóa từ cuối lên để không làm thay đổi chỉ số
        for row in sorted([index.row() for index in selected_rows], reverse=True):
            if 0 <= row < len(self.plans):
                del self.plans[row]

        self._update_plans_list()

    def _clear_plans(self):
        """Xóa tất cả kế hoạch khỏi danh sách so sánh"""
        self.plans.clear()
        self._update_plans_list()

    def _on_protocol_changed(self, index):
        """Xử lý khi protocol thay đổi"""
        if index < 0:
            self.current_protocol = None
            return

        # Lấy protocol từ combo box
        self.current_protocol = self.protocol_combo.itemData(index)

        # Cập nhật trạng thái nút đánh giá
        self.evaluate_button.setEnabled(
            len(self.plans) > 0 and self.current_protocol is not None
        )

    def _evaluate_plans(self):
        """Đánh giá tất cả các kế hoạch với protocol hiện tại"""
        if not self.plans or not self.current_protocol:
            QMessageBox.warning(
                self,
                "Không thể đánh giá",
                "Vui lòng chọn ít nhất một kế hoạch và một protocol để đánh giá.",
            )
            return

        # Lấy danh sách kế hoạch được bật (enabled)
        enabled_plans = []
        for i, plan in enumerate(self.plans):
            checkbox = self.plans_table.item(i, 4)
            if checkbox and checkbox.checkState() == Qt.Checked:
                enabled_plans.append(plan)

        if not enabled_plans:
            QMessageBox.warning(
                self,
                "Không có kế hoạch được bật",
                "Vui lòng bật ít nhất một kế hoạch để đánh giá.",
            )
            return

        try:
            # Đánh giá từng kế hoạch
            for plan in enabled_plans:
                # Tạo hoặc lấy DVH analyzer cho kế hoạch
                if plan not in self.dvh_analyzers:
                    self.dvh_analyzers[plan] = DVHAnalyzer(plan)

                # Đánh giá kế hoạch
                evaluator = PlanQualityEvaluator(self.dvh_analyzers[plan])
                results = evaluator.evaluate_with_protocol(plan, self.current_protocol)

                # Lưu kết quả đánh giá
                plan.evaluation_results = results
                plan.evaluation_metrics = evaluator.get_all_metrics()

            # Cập nhật hiển thị
            self._update_dvh_display()
            self._update_goals_display()
            self._update_stats_display()
            self._update_metrics_display()

            # Bật các nút xuất báo cáo
            self.export_pdf_button.setEnabled(True)
            self.export_html_button.setEnabled(True)
            self.export_csv_button.setEnabled(True)

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá kế hoạch: {e}")
            QMessageBox.critical(
                self, "Lỗi đánh giá", f"Xảy ra lỗi khi đánh giá kế hoạch: {str(e)}"
            )

    def _update_dvh_display(self):
        """Cập nhật hiển thị DVH"""
        # TODO: Triển khai hiển thị DVH cho nhiều kế hoạch
        # Trong phiên bản thực tế, cần tích hợp với DVHWidget
        pass

    def _on_structure_changed(self, index):
        """Xử lý khi cấu trúc được chọn thay đổi"""
        # TODO: Cập nhật hiển thị DVH cho cấu trúc được chọn
        pass

    def _update_goals_display(self):
        """Cập nhật hiển thị mục tiêu lâm sàng"""
        # TODO: Hiển thị kết quả đánh giá mục tiêu lâm sàng cho tất cả kế hoạch
        pass

    def _update_stats_display(self):
        """Cập nhật hiển thị thống kê liều"""
        # TODO: Hiển thị thống kê liều cho tất cả kế hoạch
        pass

    def _update_metrics_display(self):
        """Cập nhật hiển thị chỉ số chất lượng"""
        # TODO: Hiển thị chỉ số chất lượng cho tất cả kế hoạch
        pass

    def _export_report(self, format_type):
        """
        Xuất báo cáo so sánh kế hoạch.

        Parameters
        ----------
        format_type : str
            Loại định dạng báo cáo ("pdf", "html", "csv")
        """
        # TODO: Triển khai xuất báo cáo
        pass

    def set_available_plans(self, plans):
        """
        Thiết lập danh sách kế hoạch có sẵn để chọn.

        Parameters
        ----------
        plans : list
            Danh sách các kế hoạch
        """
        self.plan_combo.clear()

        for plan in plans:
            self.plan_combo.addItem(plan.name, plan)

    def set_protocol_manager(self, manager):
        """
        Thiết lập protocol manager.

        Parameters
        ----------
        manager : ProtocolManager
            Protocol manager
        """
        self.protocol_manager = manager
        self._update_protocol_list()

    def _update_protocol_list(self):
        """Cập nhật danh sách protocol trong combobox."""
        if not self.protocol_manager:
            return

        self.protocol_combo.clear()

        protocols = self.protocol_manager.get_all_protocols()
        if not protocols:
            return

        for protocol in protocols:
            self.protocol_combo.addItem(protocol.name, protocol)

        # Chọn protocol đầu tiên
        if self.protocol_combo.count() > 0:
            self.protocol_combo.setCurrentIndex(0)
