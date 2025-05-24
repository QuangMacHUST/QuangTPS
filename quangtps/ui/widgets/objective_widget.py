"""
Objective Widget Module

Cung cấp các widget để chỉnh sửa và quản lý objectives trong tối ưu hóa.
"""

import logging
from typing import List, Dict, Optional, Any, Callable
import numpy as np

# PyQt5 imports với fallback
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QComboBox,
        QDoubleSpinBox,
        QSpinBox,
        QGroupBox,
        QFormLayout,
        QLineEdit,
        QCheckBox,
        QSlider,
        QTabWidget,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QColor

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

    # Fallback classes
    class QWidget:
        def __init__(self, parent=None):
            pass

    class pyqtSignal:
        def __init__(self, *args):
            pass


logger = logging.getLogger(__name__)


class ObjectiveEditorWidget(QWidget if HAS_PYQT else object):
    """
    Widget để chỉnh sửa objectives cho tối ưu hóa kế hoạch.
    """

    objectives_changed = pyqtSignal(list) if HAS_PYQT else None

    def __init__(self, parent=None):
        """Khởi tạo ObjectiveEditorWidget."""
        if HAS_PYQT:
            super().__init__(parent)
            self.objectives = []
            self.structure_names = []
            self.setup_ui()
        else:
            logger.warning("ObjectiveEditorWidget yêu cầu PyQt5")

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not HAS_PYQT:
            return

        layout = QVBoxLayout()

        # Title
        title = QLabel("Objective Functions")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Tab widget cho các loại objectives
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2B2B2B;
            }
            QTabBar::tab {
                background-color: #3C3C3C;
                color: #CCCCCC;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4A90E2;
            }
        """)

        # Dose objectives tab
        self.dose_tab = self.create_dose_objectives_tab()
        self.tab_widget.addTab(self.dose_tab, "Dose")

        # Volume objectives tab
        self.volume_tab = self.create_volume_objectives_tab()
        self.tab_widget.addTab(self.volume_tab, "Volume")

        # DVH objectives tab
        self.dvh_tab = self.create_dvh_objectives_tab()
        self.tab_widget.addTab(self.dvh_tab, "DVH")

        # Biological objectives tab
        self.biological_tab = self.create_biological_objectives_tab()
        self.tab_widget.addTab(self.biological_tab, "Biological")

        layout.addWidget(self.tab_widget)

        # Objectives table
        self.objectives_table = self.create_objectives_table()
        layout.addWidget(self.objectives_table)

        # Control buttons
        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("Add Objective")
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5BA0F2;
            }
        """)
        self.add_btn.clicked.connect(self.add_objective)
        button_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F66356;
            }
        """)
        self.remove_btn.clicked.connect(self.remove_selected_objective)
        button_layout.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #666666;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #777777;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_objectives)
        button_layout.addWidget(self.clear_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_dose_objectives_tab(self):
        """Tạo tab cho dose objectives."""
        widget = QWidget()
        layout = QFormLayout()

        # Structure selection
        self.dose_structure_combo = QComboBox()
        self.dose_structure_combo.addItems(
            ["PTV", "Spinal_Cord", "Heart", "Lung_L", "Lung_R"]
        )
        layout.addRow("Structure:", self.dose_structure_combo)

        # Dose limit
        self.dose_limit_spin = QDoubleSpinBox()
        self.dose_limit_spin.setRange(0.0, 100.0)
        self.dose_limit_spin.setValue(50.0)
        self.dose_limit_spin.setSuffix(" Gy")
        layout.addRow("Dose Limit:", self.dose_limit_spin)

        # Operator
        self.dose_operator_combo = QComboBox()
        self.dose_operator_combo.addItems(["<=", ">=", "="])
        layout.addRow("Operator:", self.dose_operator_combo)

        # Weight
        self.dose_weight_spin = QDoubleSpinBox()
        self.dose_weight_spin.setRange(0.1, 100.0)
        self.dose_weight_spin.setValue(1.0)
        layout.addRow("Weight:", self.dose_weight_spin)

        widget.setLayout(layout)
        return widget

    def create_volume_objectives_tab(self):
        """Tạo tab cho volume objectives."""
        widget = QWidget()
        layout = QFormLayout()

        # Structure selection
        self.volume_structure_combo = QComboBox()
        self.volume_structure_combo.addItems(
            ["PTV", "Spinal_Cord", "Heart", "Lung_L", "Lung_R"]
        )
        layout.addRow("Structure:", self.volume_structure_combo)

        # Volume limit
        self.volume_limit_spin = QDoubleSpinBox()
        self.volume_limit_spin.setRange(0.0, 100.0)
        self.volume_limit_spin.setValue(95.0)
        self.volume_limit_spin.setSuffix(" %")
        layout.addRow("Volume Limit:", self.volume_limit_spin)

        # Operator
        self.volume_operator_combo = QComboBox()
        self.volume_operator_combo.addItems(["<=", ">=", "="])
        layout.addRow("Operator:", self.volume_operator_combo)

        # Weight
        self.volume_weight_spin = QDoubleSpinBox()
        self.volume_weight_spin.setRange(0.1, 100.0)
        self.volume_weight_spin.setValue(1.0)
        layout.addRow("Weight:", self.volume_weight_spin)

        widget.setLayout(layout)
        return widget

    def create_dvh_objectives_tab(self):
        """Tạo tab cho DVH objectives."""
        widget = QWidget()
        layout = QFormLayout()

        # Structure selection
        self.dvh_structure_combo = QComboBox()
        self.dvh_structure_combo.addItems(
            ["PTV", "Spinal_Cord", "Heart", "Lung_L", "Lung_R"]
        )
        layout.addRow("Structure:", self.dvh_structure_combo)

        # Dose percent
        self.dose_percent_spin = QDoubleSpinBox()
        self.dose_percent_spin.setRange(0.0, 100.0)
        self.dose_percent_spin.setValue(95.0)
        self.dose_percent_spin.setSuffix(" %")
        layout.addRow("Dose Percent:", self.dose_percent_spin)

        # Volume percent
        self.volume_percent_spin = QDoubleSpinBox()
        self.volume_percent_spin.setRange(0.0, 100.0)
        self.volume_percent_spin.setValue(95.0)
        self.volume_percent_spin.setSuffix(" %")
        layout.addRow("Volume Percent:", self.volume_percent_spin)

        # Operator
        self.dvh_operator_combo = QComboBox()
        self.dvh_operator_combo.addItems(["<=", ">=", "="])
        layout.addRow("Operator:", self.dvh_operator_combo)

        # Weight
        self.dvh_weight_spin = QDoubleSpinBox()
        self.dvh_weight_spin.setRange(0.1, 100.0)
        self.dvh_weight_spin.setValue(1.0)
        layout.addRow("Weight:", self.dvh_weight_spin)

        widget.setLayout(layout)
        return widget

    def create_biological_objectives_tab(self):
        """Tạo tab cho biological objectives."""
        widget = QWidget()
        layout = QFormLayout()

        # Structure selection
        self.bio_structure_combo = QComboBox()
        self.bio_structure_combo.addItems(
            ["PTV", "Spinal_Cord", "Heart", "Lung_L", "Lung_R"]
        )
        layout.addRow("Structure:", self.bio_structure_combo)

        # Model type
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["TCP", "NTCP", "EUD"])
        layout.addRow("Model Type:", self.model_type_combo)

        # Target value
        self.target_value_spin = QDoubleSpinBox()
        self.target_value_spin.setRange(0.0, 1.0)
        self.target_value_spin.setValue(0.95)
        self.target_value_spin.setDecimals(3)
        layout.addRow("Target Value:", self.target_value_spin)

        # Weight
        self.bio_weight_spin = QDoubleSpinBox()
        self.bio_weight_spin.setRange(0.1, 100.0)
        self.bio_weight_spin.setValue(1.0)
        layout.addRow("Weight:", self.bio_weight_spin)

        widget.setLayout(layout)
        return widget

    def create_objectives_table(self):
        """Tạo bảng hiển thị objectives."""
        group = QGroupBox("Current Objectives")
        group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #3C3C3C;
                color: white;
                border: 1px solid #555555;
                gridline-color: #555555;
                selection-background-color: #4A90E2;
            }
            QHeaderView::section {
                background-color: #4A90E2;
                color: white;
                padding: 6px;
                border: 1px solid #555555;
                font-weight: bold;
            }
        """)

        # Setup table headers
        headers = [
            "Type",
            "Structure",
            "Parameter",
            "Operator",
            "Value",
            "Weight",
            "Status",
        ]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Adjust column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Structure
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Parameter
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Operator
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Value
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Weight
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Status

        layout.addWidget(self.table)
        group.setLayout(layout)

        return group

    def add_objective(self):
        """Thêm objective mới."""
        if not HAS_PYQT:
            return

        current_tab = self.tab_widget.currentIndex()

        if current_tab == 0:  # Dose tab
            objective = {
                "type": "dose",
                "structure": self.dose_structure_combo.currentText(),
                "parameter": "dose_limit",
                "operator": self.dose_operator_combo.currentText(),
                "value": self.dose_limit_spin.value(),
                "weight": self.dose_weight_spin.value(),
                "status": "Active",
            }
        elif current_tab == 1:  # Volume tab
            objective = {
                "type": "volume",
                "structure": self.volume_structure_combo.currentText(),
                "parameter": "volume_limit",
                "operator": self.volume_operator_combo.currentText(),
                "value": self.volume_limit_spin.value(),
                "weight": self.volume_weight_spin.value(),
                "status": "Active",
            }
        elif current_tab == 2:  # DVH tab
            objective = {
                "type": "dvh",
                "structure": self.dvh_structure_combo.currentText(),
                "parameter": f"D{self.dose_percent_spin.value()}%",
                "operator": self.dvh_operator_combo.currentText(),
                "value": self.volume_percent_spin.value(),
                "weight": self.dvh_weight_spin.value(),
                "status": "Active",
            }
        elif current_tab == 3:  # Biological tab
            objective = {
                "type": "biological",
                "structure": self.bio_structure_combo.currentText(),
                "parameter": self.model_type_combo.currentText(),
                "operator": "=",
                "value": self.target_value_spin.value(),
                "weight": self.bio_weight_spin.value(),
                "status": "Active",
            }
        else:
            return

        self.objectives.append(objective)
        self.update_objectives_table()

        if self.objectives_changed:
            self.objectives_changed.emit(self.objectives)

    def remove_selected_objective(self):
        """Xóa objective được chọn."""
        if not HAS_PYQT:
            return

        current_row = self.table.currentRow()
        if current_row >= 0 and current_row < len(self.objectives):
            del self.objectives[current_row]
            self.update_objectives_table()

            if self.objectives_changed:
                self.objectives_changed.emit(self.objectives)

    def clear_objectives(self):
        """Xóa tất cả objectives."""
        self.objectives.clear()
        self.update_objectives_table()

        if self.objectives_changed:
            self.objectives_changed.emit(self.objectives)

    def update_objectives_table(self):
        """Cập nhật bảng objectives."""
        if not HAS_PYQT:
            return

        self.table.setRowCount(len(self.objectives))

        for i, obj in enumerate(self.objectives):
            # Type
            self.table.setItem(i, 0, QTableWidgetItem(obj["type"].upper()))

            # Structure
            self.table.setItem(i, 1, QTableWidgetItem(obj["structure"]))

            # Parameter
            self.table.setItem(i, 2, QTableWidgetItem(obj["parameter"]))

            # Operator
            self.table.setItem(i, 3, QTableWidgetItem(obj["operator"]))

            # Value
            if isinstance(obj["value"], float):
                self.table.setItem(i, 4, QTableWidgetItem(f"{obj['value']:.2f}"))
            else:
                self.table.setItem(i, 4, QTableWidgetItem(str(obj["value"])))

            # Weight
            self.table.setItem(i, 5, QTableWidgetItem(f"{obj['weight']:.1f}"))

            # Status
            status_item = QTableWidgetItem(obj["status"])
            if obj["status"] == "Active":
                status_item.setBackground(QColor("#4CAF50"))
            elif obj["status"] == "Inactive":
                status_item.setBackground(QColor("#666666"))
            elif obj["status"] == "Error":
                status_item.setBackground(QColor("#F44336"))

            self.table.setItem(i, 6, status_item)

    def set_structure_names(self, structure_names: List[str]):
        """Cập nhật danh sách tên cấu trúc."""
        self.structure_names = structure_names

        if HAS_PYQT:
            # Update all structure combo boxes
            for combo in [
                self.dose_structure_combo,
                self.volume_structure_combo,
                self.dvh_structure_combo,
                self.bio_structure_combo,
            ]:
                combo.clear()
                combo.addItems(structure_names)

    def get_objectives(self) -> List[Dict[str, Any]]:
        """Lấy danh sách objectives hiện tại."""
        return self.objectives.copy()

    def set_objectives(self, objectives: List[Dict[str, Any]]):
        """Thiết lập danh sách objectives."""
        self.objectives = objectives.copy()
        self.update_objectives_table()

    def load_sample_objectives(self):
        """Tải objectives mẫu."""
        sample_objectives = [
            {
                "type": "dose",
                "structure": "PTV",
                "parameter": "mean_dose",
                "operator": "=",
                "value": 50.0,
                "weight": 10.0,
                "status": "Active",
            },
            {
                "type": "dose",
                "structure": "Spinal_Cord",
                "parameter": "max_dose",
                "operator": "<=",
                "value": 45.0,
                "weight": 5.0,
                "status": "Active",
            },
            {
                "type": "volume",
                "structure": "PTV",
                "parameter": "coverage",
                "operator": ">=",
                "value": 95.0,
                "weight": 8.0,
                "status": "Active",
            },
        ]

        self.set_objectives(sample_objectives)


# Factory function
def create_objective_editor_widget(parent=None) -> ObjectiveEditorWidget:
    """Tạo ObjectiveEditorWidget."""
    return ObjectiveEditorWidget(parent)


# Export all classes and functions
__all__ = ["ObjectiveEditorWidget", "create_objective_editor_widget"]
