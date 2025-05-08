#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tab lập kế hoạch thích ứng trong giao diện chính.

Module này cung cấp tab lập kế hoạch thích ứng cho giao diện người dùng chính,
tích hợp nhiều phương pháp lập kế hoạch thích ứng khác nhau.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QGroupBox,
        QTabWidget,
        QScrollArea,
        QComboBox,
        QSpacerItem,
        QSizePolicy,
        QSplitter,
        QMessageBox,
        QTreeWidget,
        QTreeWidgetItem,
    )
    from PyQt5.QtGui import QIcon, QPixmap
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal, QSize
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QGroupBox,
        QTabWidget,
        QScrollArea,
        QComboBox,
        QSpacerItem,
        QSizePolicy,
        QSplitter,
        QMessageBox,
        QTreeWidget,
        QTreeWidgetItem,
    )
    from PyQt6.QtGui import QIcon, QPixmap

from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan
from quangtps.adaptive.dose_accumulation import DoseAccumulation
from quangtps.adaptive.adaptive_planning import AdaptivePlanner
from quangtps.adaptive.real_time_adaptive_planning import RealTimeAdaptivePlanner
from quangtps.ui.dialogs.real_time_adaptive_planning_dialog import (
    RealTimeAdaptivePlanningDialog,
)

logger = logging.getLogger(__name__)


class AdaptiveTab(QWidget):
    """
    Tab lập kế hoạch thích ứng trong giao diện chính.

    Tích hợp nhiều phương pháp lập kế hoạch thích ứng khác nhau, bao gồm
    lập kế hoạch thích ứng offline, online, và thời gian thực.
    """

    def __init__(self, parent=None):
        """
        Khởi tạo tab lập kế hoạch thích ứng.

        Args:
            parent: Widget cha (nếu có)
        """
        super().__init__(parent)
        self.patient = None
        self.active_plan = None

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Tiêu đề
        header_layout = QHBoxLayout()
        title_label = QLabel("Lập kế hoạch thích ứng")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # Combobox chọn phương pháp thích ứng
        self.method_combo = QComboBox()
        self.method_combo.addItem("Lập kế hoạch thích ứng offline", "offline")
        self.method_combo.addItem("Lập kế hoạch thích ứng online", "online")
        self.method_combo.addItem("Lập kế hoạch thích ứng thời gian thực", "real_time")
        header_layout.addWidget(QLabel("Phương pháp:"))
        header_layout.addWidget(self.method_combo)

        layout.addLayout(header_layout)

        # Splitter chính chia màn hình thành 2 phần
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Phần trái - Danh sách kế hoạch
        self.plans_group = QGroupBox("Kế hoạch")
        plans_layout = QVBoxLayout(self.plans_group)

        # Tree widget hiển thị kế hoạch
        self.plans_tree = QTreeWidget()
        self.plans_tree.setHeaderLabels(["Kế hoạch", "Loại", "Ngày tạo"])
        self.plans_tree.setColumnWidth(0, 200)
        self.plans_tree.setColumnWidth(1, 100)
        self.plans_tree.currentItemChanged.connect(self._on_plan_selected)
        plans_layout.addWidget(self.plans_tree)

        self.main_splitter.addWidget(self.plans_group)

        # Phần phải - Chi tiết và công cụ thích ứng
        self.tools_widget = QWidget()
        tools_layout = QVBoxLayout(self.tools_widget)

        # Thông tin kế hoạch đang chọn
        self.plan_info_group = QGroupBox("Thông tin kế hoạch")
        plan_info_layout = QVBoxLayout(self.plan_info_group)
        self.plan_info_label = QLabel("Chưa chọn kế hoạch")
        plan_info_layout.addWidget(self.plan_info_label)
        tools_layout.addWidget(self.plan_info_group)

        # Các công cụ thích ứng
        self.adaptive_tools_group = QGroupBox("Công cụ lập kế hoạch thích ứng")
        adaptive_layout = QVBoxLayout(self.adaptive_tools_group)

        # Nút mở dialog lập kế hoạch thích ứng thời gian thực
        self.real_time_adaptive_btn = QPushButton(
            "Lập kế hoạch thích ứng thời gian thực"
        )
        self.real_time_adaptive_btn.clicked.connect(
            self._open_real_time_adaptive_dialog
        )
        self.real_time_adaptive_btn.setEnabled(False)
        adaptive_layout.addWidget(self.real_time_adaptive_btn)

        # Nút thích ứng offline
        self.offline_adaptive_btn = QPushButton("Lập kế hoạch thích ứng offline")
        self.offline_adaptive_btn.clicked.connect(self._open_offline_adaptive_dialog)
        self.offline_adaptive_btn.setEnabled(False)
        adaptive_layout.addWidget(self.offline_adaptive_btn)

        # Nút thích ứng online
        self.online_adaptive_btn = QPushButton("Lập kế hoạch thích ứng online")
        self.online_adaptive_btn.clicked.connect(self._open_online_adaptive_dialog)
        self.online_adaptive_btn.setEnabled(False)
        adaptive_layout.addWidget(self.online_adaptive_btn)

        # Tích hợp tính năng tích lũy liều
        self.dose_accumulation_btn = QPushButton("Tích lũy liều từ nhiều kế hoạch")
        self.dose_accumulation_btn.clicked.connect(self._open_dose_accumulation_dialog)
        self.dose_accumulation_btn.setEnabled(False)
        adaptive_layout.addWidget(self.dose_accumulation_btn)

        adaptive_layout.addStretch()
        tools_layout.addWidget(self.adaptive_tools_group)

        self.main_splitter.addWidget(self.tools_widget)

        # Thiết lập kích thước cho splitter
        self.main_splitter.setSizes([300, 700])

        layout.addWidget(self.main_splitter)

    def set_patient(self, patient: Patient):
        """
        Thiết lập bệnh nhân hiện tại.

        Args:
            patient: Đối tượng bệnh nhân
        """
        self.patient = patient
        self._update_plans_tree()

    def set_active_plan(self, plan: Plan):
        """
        Thiết lập kế hoạch đang hoạt động.

        Args:
            plan: Kế hoạch đang hoạt động
        """
        self.active_plan = plan
        self._update_plan_info()

        # Kích hoạt các nút công cụ thích ứng
        enabled = plan is not None
        self.real_time_adaptive_btn.setEnabled(enabled)
        self.offline_adaptive_btn.setEnabled(enabled)
        self.online_adaptive_btn.setEnabled(enabled)
        self.dose_accumulation_btn.setEnabled(enabled)

    def _update_plans_tree(self):
        """Cập nhật danh sách kế hoạch trong tree widget."""
        self.plans_tree.clear()

        if not self.patient:
            return

        # Lấy tất cả kế hoạch của bệnh nhân
        plans = self.patient.get_plans()

        for plan in plans:
            item = QTreeWidgetItem()
            item.setText(0, plan.name)
            item.setText(1, getattr(plan, "type", "Unknown"))

            # Ngày tạo
            if hasattr(plan, "creation_date"):
                item.setText(2, plan.creation_date.strftime("%d/%m/%Y"))

            # Lưu trữ đối tượng kế hoạch
            item.setData(0, Qt.UserRole, plan)

            self.plans_tree.addTopLevelItem(item)

            # Nếu kế hoạch có các kế hoạch thích ứng liên quan
            if hasattr(plan, "adaptive_plans") and plan.adaptive_plans:
                for adaptive_plan in plan.adaptive_plans:
                    child = QTreeWidgetItem()
                    child.setText(0, adaptive_plan.name)
                    child.setText(1, "Thích ứng")

                    if hasattr(adaptive_plan, "creation_date"):
                        child.setText(
                            2, adaptive_plan.creation_date.strftime("%d/%m/%Y")
                        )

                    child.setData(0, Qt.UserRole, adaptive_plan)
                    item.addChild(child)

    def _update_plan_info(self):
        """Cập nhật thông tin kế hoạch đang chọn."""
        if not self.active_plan:
            self.plan_info_label.setText("Chưa chọn kế hoạch")
            return

        info_text = f"<b>Tên kế hoạch:</b> {self.active_plan.name}<br>"

        if hasattr(self.active_plan, "description"):
            info_text += f"<b>Mô tả:</b> {self.active_plan.description}<br>"

        if hasattr(self.active_plan, "creation_date"):
            info_text += f"<b>Ngày tạo:</b> {self.active_plan.creation_date.strftime('%d/%m/%Y')}<br>"

        # Thông tin tổng quan
        if hasattr(self.active_plan, "prescription"):
            info_text += "<b>Chỉ định:</b><br>"
            for target, rx in self.active_plan.prescription.items():
                info_text += (
                    f"- {target}: {rx.dose} {rx.unit} trong {rx.fractions} phiên<br>"
                )

        self.plan_info_label.setText(info_text)

    def _on_plan_selected(self, current, previous):
        """Xử lý khi một kế hoạch được chọn từ tree."""
        if not current:
            return

        # Lấy đối tượng kế hoạch từ item
        plan = current.data(0, Qt.UserRole)
        if plan:
            self.set_active_plan(plan)

    def _open_real_time_adaptive_dialog(self):
        """Mở dialog lập kế hoạch thích ứng thời gian thực."""
        if not self.patient or not self.active_plan:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn bệnh nhân và kế hoạch")
            return

        dialog = RealTimeAdaptivePlanningDialog(self.patient, self.active_plan, self)
        dialog.exec()

        # Cập nhật lại danh sách kế hoạch sau khi dialog đóng
        self._update_plans_tree()

    def _open_offline_adaptive_dialog(self):
        """Mở dialog lập kế hoạch thích ứng offline."""
        if not self.patient or not self.active_plan:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn bệnh nhân và kế hoạch")
            return

        # Trong ứng dụng thực, sẽ mở dialog thích ứng offline
        QMessageBox.information(
            self,
            "Lập kế hoạch thích ứng offline",
            "Tính năng lập kế hoạch thích ứng offline đang được phát triển.",
        )

    def _open_online_adaptive_dialog(self):
        """Mở dialog lập kế hoạch thích ứng online."""
        if not self.patient or not self.active_plan:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn bệnh nhân và kế hoạch")
            return

        # Trong ứng dụng thực, sẽ mở dialog thích ứng online
        QMessageBox.information(
            self,
            "Lập kế hoạch thích ứng online",
            "Tính năng lập kế hoạch thích ứng online đang được phát triển.",
        )

    def _open_dose_accumulation_dialog(self):
        """Mở dialog tích lũy liều."""
        if not self.patient:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn bệnh nhân")
            return

        # Trong ứng dụng thực, sẽ mở dialog tích lũy liều
        QMessageBox.information(
            self, "Tích lũy liều", "Tính năng tích lũy liều đang được phát triển."
        )
