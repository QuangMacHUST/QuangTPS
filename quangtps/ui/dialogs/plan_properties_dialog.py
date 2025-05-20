#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Properties Dialog
=====================

Dialog để xem và chỉnh sửa thuộc tính của kế hoạch xạ trị.
"""

import logging
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QFormLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QComboBox,
        QTextEdit,
        QGroupBox,
        QDateEdit,
        QCheckBox,
        QDialogButtonBox,
        QMessageBox,
    )
    from PyQt5.QtCore import Qt, QDate
    from PyQt5.QtGui import QIcon

    HAS_PYQT = True
except ImportError:
    logging.warning("PyQt5 không khả dụng. Sử dụng lớp giả mạch.")
    HAS_PYQT = False

# Import các module của hệ thống QuangTPS
try:
    from quangtps.planning.plan import Plan
    from quangtps.ui.styles.eclipse_style_theme import apply_eclipse_theme_to_widget
except ImportError:
    logging.warning(
        "Không thể import các module QuangTPS cần thiết. Sử dụng lớp giả mạch."
    )

    class Plan:
        """Lớp giả mạch cho Plan khi không thể import."""

        def __init__(self, name=""):
            self.name = name
            self.description = ""
            self.approval_status = ""
            self.created_date = None
            self.modified_date = None
            self.prescription = None
            self.planner = ""
            self.physician = ""


class PlanPropertiesDialog(QDialog):
    """
    Dialog để xem và chỉnh sửa thuộc tính của kế hoạch xạ trị.
    """

    def __init__(self, plan: Plan, parent=None):
        """
        Khởi tạo dialog thuộc tính kế hoạch.

        Args:
            plan: Kế hoạch xạ trị cần chỉnh sửa thuộc tính
            parent: Widget cha
        """
        if not HAS_PYQT:
            logging.warning(
                "PyQt5 không khả dụng. PlanPropertiesDialog sẽ không hoạt động."
            )
            return

        super().__init__(parent)
        self.setWindowTitle("Thuộc tính kế hoạch")
        self.setMinimumSize(500, 400)

        self.plan = plan
        self._init_ui()
        self._load_plan_data()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Form chính
        form_layout = QFormLayout()

        # Tên kế hoạch
        self.name_edit = QLineEdit()
        form_layout.addRow("Tên kế hoạch:", self.name_edit)

        # Mô tả
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        form_layout.addRow("Mô tả:", self.description_edit)

        # Trạng thái phê duyệt
        self.status_combo = QComboBox()
        self.status_combo.addItems(
            ["Nháp", "Đang xem xét", "Đã phê duyệt", "Đã từ chối"]
        )
        form_layout.addRow("Trạng thái:", self.status_combo)

        # Thông tin bác sĩ và người lập kế hoạch
        self.planner_edit = QLineEdit()
        form_layout.addRow("Người lập kế hoạch:", self.planner_edit)

        self.physician_edit = QLineEdit()
        form_layout.addRow("Bác sĩ:", self.physician_edit)

        # Nhóm thông tin liều lượng
        dose_group = QGroupBox("Thông tin liều lượng")
        dose_layout = QFormLayout(dose_group)

        self.prescription_edit = QLineEdit()
        dose_layout.addRow("Liều kê toa (Gy):", self.prescription_edit)

        self.fractions_edit = QLineEdit()
        dose_layout.addRow("Số phân đoạn:", self.fractions_edit)

        self.dose_per_fraction_edit = QLineEdit()
        dose_layout.addRow("Liều mỗi phân đoạn (Gy):", self.dose_per_fraction_edit)

        # Thêm các widget vào layout chính
        main_layout.addLayout(form_layout)
        main_layout.addWidget(dose_group)

        # Thêm các nút
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._save_changes)
        button_box.rejected.connect(self.reject)

        main_layout.addStretch()
        main_layout.addWidget(button_box)

        # Áp dụng theme Eclipse nếu có
        try:
            apply_eclipse_theme_to_widget(self)
        except:
            pass

    def _load_plan_data(self):
        """Tải dữ liệu từ kế hoạch vào các trường nhập liệu."""
        if not self.plan:
            return

        # Tải các thuộc tính cơ bản
        self.name_edit.setText(self.plan.name if hasattr(self.plan, "name") else "")
        self.description_edit.setText(
            self.plan.description if hasattr(self.plan, "description") else ""
        )

        # Trạng thái
        status = (
            self.plan.approval_status if hasattr(self.plan, "approval_status") else ""
        )
        status_index = 0  # Mặc định là "Nháp"

        if status == "Approved":
            status_index = 2
        elif status == "Reviewing":
            status_index = 1
        elif status == "Rejected":
            status_index = 3

        self.status_combo.setCurrentIndex(status_index)

        # Thông tin bác sĩ và người lập kế hoạch
        self.planner_edit.setText(
            self.plan.planner if hasattr(self.plan, "planner") else ""
        )
        self.physician_edit.setText(
            self.plan.physician if hasattr(self.plan, "physician") else ""
        )

        # Thông tin liều lượng
        if hasattr(self.plan, "prescription") and self.plan.prescription:
            self.prescription_edit.setText(str(self.plan.prescription.total_dose))
            self.fractions_edit.setText(str(self.plan.prescription.num_fractions))
            self.dose_per_fraction_edit.setText(
                str(self.plan.prescription.dose_per_fraction)
            )
        else:
            self.prescription_edit.setText("")
            self.fractions_edit.setText("")
            self.dose_per_fraction_edit.setText("")

    def _save_changes(self):
        """Lưu các thay đổi vào đối tượng kế hoạch."""
        try:
            # Cập nhật các thuộc tính cơ bản
            self.plan.name = self.name_edit.text()
            self.plan.description = self.description_edit.toPlainText()

            # Trạng thái
            status_map = {0: "Draft", 1: "Reviewing", 2: "Approved", 3: "Rejected"}
            self.plan.approval_status = status_map[self.status_combo.currentIndex()]

            # Thông tin bác sĩ và người lập kế hoạch
            self.plan.planner = self.planner_edit.text()
            self.plan.physician = self.physician_edit.text()

            # Thông tin liều lượng
            if hasattr(self.plan, "prescription"):
                try:
                    total_dose = float(self.prescription_edit.text())
                    num_fractions = int(self.fractions_edit.text())
                    dose_per_fraction = float(self.dose_per_fraction_edit.text())

                    self.plan.prescription.total_dose = total_dose
                    self.plan.prescription.num_fractions = num_fractions
                    self.plan.prescription.dose_per_fraction = dose_per_fraction
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Lỗi dữ liệu",
                        "Vui lòng nhập số hợp lệ cho các trường liều lượng.",
                    )
                    return

            self.accept()

        except Exception as e:
            logging.error(f"Lỗi khi lưu thay đổi kế hoạch: {e}")
            QMessageBox.warning(
                self, "Lỗi", f"Đã xảy ra lỗi khi lưu thay đổi: {str(e)}"
            )


# Test code
if __name__ == "__main__" and HAS_PYQT:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Tạo kế hoạch giả
    class DummyPlan:
        def __init__(self):
            self.name = "Test Plan"
            self.description = "This is a test plan for demonstration purposes."
            self.approval_status = "Draft"
            self.planner = "John Doe"
            self.physician = "Dr. Smith"
            self.prescription = type(
                "obj",
                (object,),
                {"total_dose": 60.0, "num_fractions": 30, "dose_per_fraction": 2.0},
            )

    plan = DummyPlan()

    # Tạo dialog
    dialog = PlanPropertiesDialog(plan)
    if dialog.exec_():
        print("Plan properties updated:")
        print(f"Name: {plan.name}")
        print(f"Description: {plan.description}")
        print(f"Status: {plan.approval_status}")
        print(f"Planner: {plan.planner}")
        print(f"Physician: {plan.physician}")
        print(f"Total dose: {plan.prescription.total_dose}")
        print(f"Fractions: {plan.prescription.num_fractions}")
        print(f"Dose per fraction: {plan.prescription.dose_per_fraction}")

    sys.exit(app.exec_())
