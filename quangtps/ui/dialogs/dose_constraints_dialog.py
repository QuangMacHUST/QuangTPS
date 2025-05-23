#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dose Constraints Dialog Module

Module này cung cấp dialog để quản lý dose constraints trong hệ thống QuangTPS.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Try to import PyQt5 với fallback
try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QComboBox,
        QLineEdit,
        QGroupBox,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QHeaderView,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont

    HAS_PYQT5 = True
except ImportError:
    logger.warning("PyQt5 not available, using fallback classes")
    HAS_PYQT5 = False

    # Fallback classes
    class QDialog:
        def __init__(self, parent=None):
            pass

        def setWindowTitle(self, title):
            pass

        def exec_(self):
            return 0

    QVBoxLayout = QHBoxLayout = QLabel = QPushButton = QTableWidget = type(None)
    QTableWidgetItem = QComboBox = QLineEdit = QGroupBox = QSpinBox = type(None)
    QDoubleSpinBox = QCheckBox = QHeaderView = Qt = pyqtSignal = QFont = type(None)


class DoseConstraintsDialog(QDialog):
    """
    Dialog để quản lý dose constraints với Eclipse-style interface.
    """

    # Signals
    constraints_updated = pyqtSignal(list) if HAS_PYQT5 else None

    def __init__(self, parent=None, structure_names=None, existing_constraints=None):
        """
        Khởi tạo DoseConstraintsDialog.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        structure_names : list, optional
            Danh sách tên cấu trúc
        existing_constraints : list, optional
            Danh sách constraints hiện có
        """
        if not HAS_PYQT5:
            logger.warning("PyQt5 not available, DoseConstraintsDialog disabled")
            return

        super().__init__(parent)
        self.setWindowTitle("Dose Constraints Manager")
        self.setModal(True)
        self.resize(800, 500)

        # Data
        self.structure_names = structure_names or [
            "PTV",
            "Heart",
            "Lung_L",
            "Lung_R",
            "Spinal_Cord",
        ]
        self.constraints = existing_constraints or []

        # Constraint types
        self.constraint_types = {
            "Max Dose": "max_dose",
            "Mean Dose": "mean_dose",
            "Min Dose": "min_dose",
            "V20": "V20",
            "V30": "V30",
            "V40": "V40",
            "V50": "V50",
            "D95": "D95",
            "D98": "D98",
            "D2": "D2",
            "D5": "D5",
        }

        self.setup_ui()
        self.apply_eclipse_style()
        self.populate_table()

    def setup_ui(self):
        """Thiết lập giao diện."""
        if not HAS_PYQT5:
            return

        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Dose Constraints Manager")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Instructions
        info_label = QLabel("Define dose constraints for optimization objectives:")
        main_layout.addWidget(info_label)

        # Constraints table group
        table_group = QGroupBox("Dose Constraints")
        table_layout = QVBoxLayout()

        # Table
        self.constraints_table = QTableWidget()
        self.constraints_table.setColumnCount(6)
        headers = [
            "Structure",
            "Constraint Type",
            "Value",
            "Unit",
            "Priority",
            "Hard/Soft",
        ]
        self.constraints_table.setHorizontalHeaderLabels(headers)

        # Set column widths
        header = self.constraints_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        table_layout.addWidget(self.constraints_table)

        # Table buttons
        table_button_layout = QHBoxLayout()

        add_button = QPushButton("Add Constraint")
        add_button.clicked.connect(self.add_constraint)
        table_button_layout.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_constraint)
        table_button_layout.addWidget(remove_button)

        table_button_layout.addStretch()

        load_template_button = QPushButton("Load Template")
        load_template_button.clicked.connect(self.load_template)
        table_button_layout.addWidget(load_template_button)

        table_layout.addLayout(table_button_layout)
        table_group.setLayout(table_layout)
        main_layout.addWidget(table_group)

        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def apply_eclipse_style(self):
        """Áp dụng Eclipse-style theme."""
        if not HAS_PYQT5:
            return

        try:
            from quangtps.utils.ui_utils import apply_eclipse_style_theme

            apply_eclipse_style_theme(self)
        except ImportError:
            # Fallback styling
            self.setStyleSheet("""
            QDialog {
                background-color: #2B2B2B;
                color: #CCCCCC;
            }
            QTableWidget {
                background-color: #2B2B2B;
                border: 1px solid #555555;
                color: #CCCCCC;
                selection-background-color: #4A90E2;
                gridline-color: #404040;
            }
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #404040;
            }
            QHeaderView::section {
                background-color: #3C3C3C;
                color: #CCCCCC;
                padding: 6px;
                border: 1px solid #555555;
                font-weight: bold;
            }
            QPushButton {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                color: #CCCCCC;
                padding: 6px 12px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4A90E2;
            }
            QComboBox {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                color: #CCCCCC;
                padding: 4px 8px;
                border-radius: 3px;
            }
            """)

    def populate_table(self):
        """Populate table với constraints hiện có."""
        if not HAS_PYQT5:
            return

        self.constraints_table.setRowCount(len(self.constraints))

        for row, constraint in enumerate(self.constraints):
            self.set_table_row(row, constraint)

    def set_table_row(self, row, constraint):
        """Đặt data cho một row trong table."""
        if not HAS_PYQT5:
            return

        # Structure combo
        structure_combo = QComboBox()
        structure_combo.addItems(self.structure_names)
        structure_combo.setCurrentText(constraint.get("structure", ""))
        self.constraints_table.setCellWidget(row, 0, structure_combo)

        # Constraint type combo
        type_combo = QComboBox()
        type_combo.addItems(list(self.constraint_types.keys()))
        type_combo.setCurrentText(constraint.get("type", ""))
        self.constraints_table.setCellWidget(row, 1, type_combo)

        # Value spinbox
        value_spinbox = QDoubleSpinBox()
        value_spinbox.setRange(0.0, 10000.0)
        value_spinbox.setSingleStep(0.1)
        value_spinbox.setValue(constraint.get("value", 0.0))
        value_spinbox.setSuffix("")
        self.constraints_table.setCellWidget(row, 2, value_spinbox)

        # Unit combo
        unit_combo = QComboBox()
        unit_combo.addItems(["Gy", "cGy", "%", "cc"])
        unit_combo.setCurrentText(constraint.get("unit", "Gy"))
        self.constraints_table.setCellWidget(row, 3, unit_combo)

        # Priority spinbox
        priority_spinbox = QSpinBox()
        priority_spinbox.setRange(1, 10)
        priority_spinbox.setValue(constraint.get("priority", 1))
        self.constraints_table.setCellWidget(row, 4, priority_spinbox)

        # Hard/Soft checkbox
        hard_checkbox = QCheckBox()
        hard_checkbox.setChecked(constraint.get("is_hard", True))
        hard_checkbox.setText("Hard" if constraint.get("is_hard", True) else "Soft")
        hard_checkbox.stateChanged.connect(
            lambda state, cb=hard_checkbox: cb.setText(
                "Hard" if state == Qt.Checked else "Soft"
            )
        )
        self.constraints_table.setCellWidget(row, 5, hard_checkbox)

    def add_constraint(self):
        """Thêm constraint mới."""
        if not HAS_PYQT5:
            return

        new_constraint = {
            "structure": self.structure_names[0] if self.structure_names else "",
            "type": "Max Dose",
            "value": 50.0,
            "unit": "Gy",
            "priority": 1,
            "is_hard": True,
        }

        self.constraints.append(new_constraint)

        row_count = self.constraints_table.rowCount()
        self.constraints_table.setRowCount(row_count + 1)
        self.set_table_row(row_count, new_constraint)

    def remove_constraint(self):
        """Xóa constraint được chọn."""
        if not HAS_PYQT5:
            return

        current_row = self.constraints_table.currentRow()
        if current_row >= 0:
            self.constraints_table.removeRow(current_row)
            if current_row < len(self.constraints):
                del self.constraints[current_row]

    def load_template(self):
        """Load template constraints phổ biến."""
        if not HAS_PYQT5:
            return

        # Template cho Head & Neck
        template_constraints = [
            {
                "structure": "PTV",
                "type": "Min Dose",
                "value": 95.0,
                "unit": "%",
                "priority": 1,
                "is_hard": True,
            },
            {
                "structure": "PTV",
                "type": "Max Dose",
                "value": 107.0,
                "unit": "%",
                "priority": 1,
                "is_hard": True,
            },
            {
                "structure": "Spinal_Cord",
                "type": "Max Dose",
                "value": 45.0,
                "unit": "Gy",
                "priority": 1,
                "is_hard": True,
            },
            {
                "structure": "Brainstem",
                "type": "Max Dose",
                "value": 54.0,
                "unit": "Gy",
                "priority": 1,
                "is_hard": True,
            },
            {
                "structure": "Parotid_L",
                "type": "Mean Dose",
                "value": 26.0,
                "unit": "Gy",
                "priority": 2,
                "is_hard": False,
            },
            {
                "structure": "Parotid_R",
                "type": "Mean Dose",
                "value": 26.0,
                "unit": "Gy",
                "priority": 2,
                "is_hard": False,
            },
        ]

        self.constraints.extend(template_constraints)
        self.populate_table()

    def get_constraints(self) -> List[Dict[str, Any]]:
        """Lấy danh sách constraints từ table."""
        if not HAS_PYQT5:
            return []

        constraints = []

        for row in range(self.constraints_table.rowCount()):
            structure_combo = self.constraints_table.cellWidget(row, 0)
            type_combo = self.constraints_table.cellWidget(row, 1)
            value_spinbox = self.constraints_table.cellWidget(row, 2)
            unit_combo = self.constraints_table.cellWidget(row, 3)
            priority_spinbox = self.constraints_table.cellWidget(row, 4)
            hard_checkbox = self.constraints_table.cellWidget(row, 5)

            if all(
                [
                    structure_combo,
                    type_combo,
                    value_spinbox,
                    unit_combo,
                    priority_spinbox,
                    hard_checkbox,
                ]
            ):
                constraint = {
                    "structure": structure_combo.currentText(),
                    "type": type_combo.currentText(),
                    "value": value_spinbox.value(),
                    "unit": unit_combo.currentText(),
                    "priority": priority_spinbox.value(),
                    "is_hard": hard_checkbox.isChecked(),
                }
                constraints.append(constraint)

        return constraints

    def accept(self):
        """Chấp nhận dialog và emit constraints."""
        if not HAS_PYQT5:
            return

        constraints = self.get_constraints()
        if self.constraints_updated:
            self.constraints_updated.emit(constraints)
        super().accept()


# Factory function
def show_dose_constraints_dialog(
    parent=None, structure_names=None, existing_constraints=None
):
    """
    Hiển thị dialog dose constraints và trả về constraints.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha
    structure_names : list, optional
        Danh sách tên cấu trúc
    existing_constraints : list, optional
        Constraints hiện có

    Returns
    -------
    list
        Danh sách constraints hoặc None nếu hủy
    """
    if not HAS_PYQT5:
        logger.warning("PyQt5 not available, returning default constraints")
        return [
            {
                "structure": "PTV",
                "type": "Min Dose",
                "value": 95.0,
                "unit": "%",
                "priority": 1,
                "is_hard": True,
            },
            {
                "structure": "Spinal_Cord",
                "type": "Max Dose",
                "value": 45.0,
                "unit": "Gy",
                "priority": 1,
                "is_hard": True,
            },
        ]

    dialog = DoseConstraintsDialog(parent, structure_names, existing_constraints)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return dialog.get_constraints()
    else:
        return None
