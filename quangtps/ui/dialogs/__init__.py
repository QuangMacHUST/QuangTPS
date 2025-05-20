#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module dialogs chứa tất cả các dialog phong cách Eclipse cho QuangTPS.
Hệ thống import an toàn được sử dụng để đảm bảo hệ thống vẫn hoạt động
ngay cả khi thiếu một số thành phần.
"""

import logging

# Import các thành phần cơ bản với try-except
try:
    from PyQt5.QtWidgets import QDialog, QMessageBox
except ImportError:
    try:
        from PySide2.QtWidgets import QDialog, QMessageBox
    except ImportError:
        print("Không thể import thư viện Qt. Chức năng dialog sẽ bị giới hạn.")

# Import các dialog chính với try-except
try:
    from .plan_comparison_dialog import PlanComparisonDialog
except ImportError as e:
    print(f"Không thể import PlanComparisonDialog: {e}")

    # Tạo lớp giả mạch
    class PlanComparisonDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho PlanComparisonDialog khi không thể import."""

        def __init__(self, plans=None, parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            print(
                "WARNING: Đang sử dụng lớp giả PlanComparisonDialog. Tính năng so sánh kế hoạch bị giới hạn."
            )


try:
    from .plan_properties_dialog import PlanPropertiesDialog
except ImportError as e:
    print(f"Không thể import PlanPropertiesDialog: {e}")

    # Tạo lớp giả mạch
    class PlanPropertiesDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho PlanPropertiesDialog khi không thể import."""

        def __init__(self, plan=None, parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            print(
                "WARNING: Đang sử dụng lớp giả PlanPropertiesDialog. Tính năng chỉnh sửa thuộc tính kế hoạch bị giới hạn."
            )


try:
    from .structure_properties_dialog import StructurePropertiesDialog, ColorButton
except ImportError as e:
    print(f"Không thể import StructurePropertiesDialog: {e}")

    # Tạo lớp giả mạch
    class StructurePropertiesDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho StructurePropertiesDialog khi không thể import."""

        def __init__(self, structure=None, parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            print(
                "WARNING: Đang sử dụng lớp giả StructurePropertiesDialog. Tính năng chỉnh sửa thuộc tính cấu trúc bị giới hạn."
            )

    class ColorButton:
        """Lớp giả mạch cho ColorButton khi không thể import."""

        def __init__(self, color=None, parent=None):
            print(
                "WARNING: Đang sử dụng lớp giả ColorButton. Tính năng chọn màu bị giới hạn."
            )

        def get_color(self):
            return (255, 0, 0)

        def set_color(self, color):
            pass


try:
    from .kbp_dialog import KBPDialog
except ImportError as e:
    print(f"Không thể import KBPDialog: {e}")

    # Tạo lớp giả mạch
    class KBPDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho KBPDialog khi không thể import."""

        def __init__(self, patient_id="", structure_set_id="", site="", parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            print(
                "WARNING: Đang sử dụng lớp giả KBPDialog. Tính năng Knowledge-Based Planning bị giới hạn."
            )

        def kbpRecommendationApplied(self):
            pass


# Import các dialog bổ sung
try:
    from .dose_constraints_dialog import DoseConstraintsDialog
except ImportError as e:
    print(f"Không thể import DoseConstraintsDialog: {e}")

    # Tạo lớp giả mạch
    class DoseConstraintsDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho DoseConstraintsDialog khi không thể import."""

        def __init__(self, constraints=None, parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            print(
                "WARNING: Đang sử dụng lớp giả DoseConstraintsDialog. Tính năng chỉnh sửa ràng buộc liều bị giới hạn."
            )


try:
    from .protocol_editor_dialog import ProtocolEditorDialog
except ImportError as e:
    print(f"Không thể import ProtocolEditorDialog: {e}")

    # Tạo lớp giả mạch
    class ProtocolEditorDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho ProtocolEditorDialog khi không thể import."""

        def __init__(self, protocol=None, parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            print(
                "WARNING: Đang sử dụng lớp giả ProtocolEditorDialog. Tính năng chỉnh sửa protocol bị giới hạn."
            )


try:
    from .progress_dialog import ProgressDialog
except ImportError as e:
    print(f"Không thể import ProgressDialog: {e}")

    # Tạo lớp giả mạch
    class ProgressDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho ProgressDialog khi không thể import."""

        def __init__(self, title="Đang xử lý...", parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            self._value = 0
            print(
                "WARNING: Đang sử dụng lớp giả ProgressDialog. Tính năng hiển thị tiến trình bị giới hạn."
            )

        def set_value(self, value):
            self._value = value

        def set_maximum(self, maximum):
            pass

        def set_text(self, text):
            pass


try:
    from .machine_selection_dialog import MachineSelectionDialog
except ImportError as e:
    print(f"Không thể import MachineSelectionDialog: {e}")

    # Tạo lớp giả mạch
    class MachineSelectionDialog(QDialog if "QDialog" in locals() else object):
        """Lớp giả mạch cho MachineSelectionDialog khi không thể import."""

        def __init__(self, machines=None, parent=None):
            if "QDialog" in locals():
                super().__init__(parent)
            print(
                "WARNING: Đang sử dụng lớp giả MachineSelectionDialog. Tính năng chọn máy điều trị bị giới hạn."
            )


# Định nghĩa các dialog khả dụng sau khi import hoàn tất
__all__ = [
    "PlanComparisonDialog",
    "PlanPropertiesDialog",
    "StructurePropertiesDialog",
    "ColorButton",
    "DoseConstraintsDialog",
    "ProtocolEditorDialog",
    "ProgressDialog",
    "MachineSelectionDialog",
    "KBPDialog",
]

# Thông báo trạng thái của module
print("Module dialogs đã được khởi tạo với cơ chế import an toàn.")
