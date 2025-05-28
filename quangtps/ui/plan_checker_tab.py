#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Checker Tab Module

Tab hiển thị tính năng kiểm tra kế hoạch điều trị trong giao diện chính của QuangTPS,
tương tự như tính năng Plan Checker trong Eclipse TPS.
"""

import logging
from typing import Optional, Dict, List, Any

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSplitter,
        QComboBox,
        QFrame,
        QToolBar,
        QAction,
        QStatusBar,
        QMessageBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QIcon

    QT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các thành phần PyQt5: {e}")
    QT_AVAILABLE = False

from quangtps.core.logging import get_logger
from quangtps.ui.plan_checker_widget import PlanCheckerWidget

logger = get_logger(__name__)


class PlanCheckerTab(QWidget):
    """
    Tab hiển thị tính năng kiểm tra kế hoạch điều trị trong giao diện chính của QuangTPS.

    Tab này tích hợp PlanCheckerWidget và cung cấp các điều khiển bổ sung để làm việc
    với các kế hoạch điều trị.
    """

    # Tín hiệu
    planSelected = pyqtSignal(object)  # Phát khi một kế hoạch được chọn

    def __init__(self, parent=None):
        """Khởi tạo tab kiểm tra kế hoạch."""
        super().__init__(parent)

        # Biến thành viên
        self.app = None
        self.current_patient = None
        self.current_plan = None
        self.plan_checker_widget = None

        # Khởi tạo giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng của tab."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar("Plan Checker Toolbar")
        toolbar.setIconSize(QSize(24, 24))

        # Combobox chọn kế hoạch
        toolbar.addWidget(QLabel("Kế hoạch:"))
        self.plan_combo = QComboBox()
        self.plan_combo.setMinimumWidth(200)
        self.plan_combo.currentIndexChanged.connect(self._on_plan_changed)
        toolbar.addWidget(self.plan_combo)

        toolbar.addSeparator()

        # Nút refresh
        refresh_action = QAction("Làm mới", self)
        refresh_action.triggered.connect(self._on_refresh)
        toolbar.addAction(refresh_action)

        # Thêm toolbar vào layout
        main_layout.addWidget(toolbar)

        # Thêm widget Plan Checker
        self.plan_checker_widget = PlanCheckerWidget(self)
        main_layout.addWidget(self.plan_checker_widget)

    def set_app(self, app):
        """
        Thiết lập tham chiếu đến ứng dụng chính.

        Parameters:
            app: Instance của ứng dụng QuangTPS chính
        """
        self.app = app
        self._update_patient_data()

    def set_patient(self, patient):
        """
        Thiết lập bệnh nhân hiện tại.

        Parameters:
            patient: Đối tượng Patient hiện tại
        """
        self.current_patient = patient
        self._update_plan_list()

    def _update_patient_data(self):
        """Cập nhật dữ liệu bệnh nhân từ ứng dụng."""
        if not self.app:
            return

        # Lấy bệnh nhân hiện tại từ app
        try:
            if hasattr(self.app, "get_current_patient"):
                patient = self.app.get_current_patient()
                if patient:
                    self.set_patient(patient)
            else:
                logger.warning("App không có method get_current_patient")
        except Exception as e:
            logger.error(f"Lỗi khi lấy current patient: {e}")

    def _update_plan_list(self):
        """Cập nhật danh sách kế hoạch từ bệnh nhân hiện tại."""
        # Xóa danh sách cũ
        self.plan_combo.clear()

        # Nếu không có bệnh nhân, dừng
        if not self.current_patient:
            return

        # Lấy danh sách kế hoạch từ bệnh nhân
        plans = self.current_patient.get_plans()
        if not plans:
            return

        # Thêm kế hoạch vào combobox
        for plan in plans:
            self.plan_combo.addItem(plan.name, plan)

        # Chọn kế hoạch đầu tiên
        if self.plan_combo.count() > 0:
            self.plan_combo.setCurrentIndex(0)

    def _on_plan_changed(self, index: int):
        """
        Xử lý khi người dùng thay đổi kế hoạch.

        Parameters:
            index: Chỉ số của kế hoạch được chọn trong combobox
        """
        if index < 0:
            self.current_plan = None
            return

        # Lấy kế hoạch từ dữ liệu của combobox
        self.current_plan = self.plan_combo.itemData(index)

        # Cập nhật widget Plan Checker
        if self.current_plan and self.plan_checker_widget:
            self.plan_checker_widget.setPlan(self.current_plan)

            # Phát tín hiệu về kế hoạch đã chọn
            self.planSelected.emit(self.current_plan)

    def _on_refresh(self):
        """Làm mới dữ liệu từ hệ thống."""
        # Cập nhật dữ liệu bệnh nhân
        self._update_patient_data()

        # Thông báo làm mới thành công
        QMessageBox.information(
            self,
            "Làm mới dữ liệu",
            "Đã làm mới dữ liệu kế hoạch.",
        )


# Hàm tiện ích để tạo tab
def create_plan_checker_tab(parent=None, app=None):
    """
    Tạo tab Plan Checker.

    Parameters:
        parent: Widget cha
        app: Instance của ứng dụng QuangTPS chính

    Returns:
        PlanCheckerTab: Tab đã được khởi tạo
    """
    try:
        tab = PlanCheckerTab(parent)
        if app:
            tab.set_app(app)
        return tab
    except Exception as e:
        logger.error(f"Lỗi khi tạo Plan Checker Tab: {str(e)}")
        return None


# Để kiểm thử khi chạy trực tiếp
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    tab = PlanCheckerTab()
    tab.resize(1200, 800)
    tab.show()

    sys.exit(app.exec_())
