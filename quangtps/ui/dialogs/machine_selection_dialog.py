#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Machine Selection Dialog Module

Cung cấp dialog chọn máy xạ trị với thông tin chi tiết.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Kiểm tra PyQt5 availability
HAS_PYQT5 = False
try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QListWidget,
        QListWidgetItem,
        QLabel,
        QPushButton,
        QTextEdit,
        QSplitter,
        QFrame,
        QWidget,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QFont

    HAS_PYQT5 = True
    logger.info("PyQt5 được tải thành công cho MachineSelectionDialog")
except ImportError as e:
    logger.warning(f"PyQt5 không khả dụng: {e}")

    # Tạo fallback classes
    class QDialog:
        def __init__(self, parent=None):
            self.parent = parent

        def setWindowTitle(self, title):
            pass

        def setModal(self, modal):
            pass

        def exec_(self):
            return 0

        def accept(self):
            pass

        def reject(self):
            pass

        def show(self):
            pass

        def setMinimumSize(self, width, height):
            pass

    class QVBoxLayout:
        def __init__(self):
            pass

        def addWidget(self, widget):
            pass

        def addLayout(self, layout):
            pass

        def setContentsMargins(self, *args):
            pass

    class QHBoxLayout:
        def __init__(self):
            pass

        def addWidget(self, widget):
            pass

        def addStretch(self):
            pass

    class QListWidget:
        def __init__(self):
            self.items = []

        def addItem(self, item):
            self.items.append(item)

        def currentRow(self):
            return 0

        def itemSelectionChanged(self):
            pass

        def connect(self, slot):
            pass

    class QListWidgetItem:
        def __init__(self, text=""):
            self.text = text

        def setText(self, text):
            self.text = text

    class QLabel:
        def __init__(self, text=""):
            self.text = text

        def setText(self, text):
            self.text = text

        def setFont(self, font):
            pass

        def setAlignment(self, alignment):
            pass

    class QPushButton:
        def __init__(self, text=""):
            self.text = text
            self.enabled = True

        def setText(self, text):
            self.text = text

        def setEnabled(self, enabled):
            self.enabled = enabled

        def clicked(self):
            pass

        def connect(self, slot):
            pass

    class QTextEdit:
        def __init__(self):
            self.text = ""

        def setHtml(self, html):
            self.text = html

        def clear(self):
            self.text = ""

        def setReadOnly(self, readonly):
            pass

    class QSplitter:
        def __init__(self, orientation=None):
            pass

        def addWidget(self, widget):
            pass

        def setSizes(self, sizes):
            pass

    class QFrame:
        def __init__(self):
            pass

        def setFrameStyle(self, style):
            pass

    class QWidget:
        def __init__(self, parent=None):
            pass

        def setLayout(self, layout):
            pass

    class Qt:
        AlignCenter = 4
        AlignLeft = 1
        Horizontal = 1
        Vertical = 2

    class QFont:
        def __init__(self, family="", size=9):
            pass

        def setBold(self, bold):
            pass

        def setPointSize(self, size):
            pass

    def pyqtSignal(*args):
        class Signal:
            def emit(self, *args):
                pass

            def connect(self, slot):
                pass

        return Signal()


# Import sau khi định nghĩa fallback classes
try:
    from quangtps.utils.ui_utils import apply_eclipse_style_theme
except ImportError:
    logger.warning("Không thể import apply_eclipse_style_theme")

    def apply_eclipse_style_theme(widget):
        pass


class MachineSelectionDialog(QDialog):
    """
    Dialog để chọn máy xạ trị với thông tin chi tiết.
    """

    # Signals
    machine_selected = pyqtSignal(str) if HAS_PYQT5 else None

    def __init__(self, parent=None, available_machines=None):
        """
        Khởi tạo MachineSelectionDialog.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        available_machines : dict, optional
            Dictionary các máy có sẵn
        """
        if not HAS_PYQT5:
            logger.warning("PyQt5 not available, MachineSelectionDialog disabled")
            return

        super().__init__(parent)
        self.setWindowTitle("Machine Selection")
        self.setModal(True)
        self.resize(600, 400)

        # Default machines nếu không được cung cấp
        self.available_machines = available_machines or self.get_default_machines()
        self.selected_machine = None

        self.setup_ui()
        self.apply_eclipse_style()
        self.load_machines()

    def get_default_machines(self) -> Dict[str, Dict[str, Any]]:
        """Lấy danh sách máy mặc định."""
        return {
            "TrueBeam": {
                "manufacturer": "Varian",
                "model": "TrueBeam",
                "description": "High-performance linear accelerator",
                "energies": {
                    "photon": ["6X", "10X", "15X", "6FFF", "10FFF"],
                    "electron": ["6E", "9E", "12E", "15E", "18E"],
                },
                "max_field_size": 40.0,
                "mlc_type": "HD120",
                "max_dose_rate": 1400,
                "status": "Available",
            },
            "Halcyon": {
                "manufacturer": "Varian",
                "model": "Halcyon",
                "description": "Compact linear accelerator for high-throughput treatments",
                "energies": {"photon": ["6X", "6FFF"], "electron": []},
                "max_field_size": 28.0,
                "mlc_type": "Dual-Layer",
                "max_dose_rate": 800,
                "status": "Available",
            },
            "VersaHD": {
                "manufacturer": "Elekta",
                "model": "Versa HD",
                "description": "Versatile linear accelerator with advanced imaging",
                "energies": {
                    "photon": ["6X", "10X", "15X", "6FFF", "10FFF"],
                    "electron": ["4E", "6E", "8E", "10E", "12E", "15E", "18E"],
                },
                "max_field_size": 40.0,
                "mlc_type": "Agility",
                "max_dose_rate": 1400,
                "status": "Available",
            },
        }

    def setup_ui(self):
        """Thiết lập giao diện."""
        if not HAS_PYQT5:
            return

        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Select Treatment Machine")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Splitter cho machine list và details
        splitter = QSplitter(Qt.Horizontal)

        # Machine list group
        list_group = QFrame()
        list_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        list_layout = QVBoxLayout()

        self.machine_list = QListWidget()
        self.machine_list.currentRowChanged.connect(self.on_machine_selected)
        list_layout.addWidget(self.machine_list)

        list_group.setLayout(list_layout)
        splitter.addWidget(list_group)

        # Machine details group
        details_group = QFrame()
        details_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)

        details_group.setLayout(details_layout)
        splitter.addWidget(details_group)

        splitter.setSizes([250, 350])
        main_layout.addWidget(splitter)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)
        button_layout.addWidget(self.ok_button)

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
            apply_eclipse_style_theme(self)
        except ImportError:
            # Fallback styling
            self.setStyleSheet("""
            QDialog {
                background-color: #2B2B2B;
                color: #CCCCCC;
            }
            QGroupBox {
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                margin-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            QListWidget {
                background-color: #2B2B2B;
                border: 1px solid #555555;
                color: #CCCCCC;
                selection-background-color: #4A90E2;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QListWidget::item:selected {
                background-color: #4A90E2;
            }
            QTextEdit {
                background-color: #2B2B2B;
                border: 1px solid #555555;
                color: #CCCCCC;
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
            QPushButton:disabled {
                background-color: #2B2B2B;
                color: #777777;
            }
            """)

    def load_machines(self):
        """Load danh sách máy."""
        if not HAS_PYQT5:
            return

        for machine_name, machine_info in self.available_machines.items():
            item = QListWidgetItem(machine_name)

            # Set tooltip with brief info
            tooltip = f"{machine_info.get('manufacturer', 'Unknown')} {machine_info.get('model', '')}"
            item.setToolTip(tooltip)

            self.machine_list.addItem(item)

    def on_machine_selected(self, row):
        """Xử lý khi chọn máy."""
        if not HAS_PYQT5 or row < 0:
            return

        machine_names = list(self.available_machines.keys())
        if row < len(machine_names):
            machine_name = machine_names[row]
            self.selected_machine = machine_name
            self.show_machine_details(machine_name)
            self.ok_button.setEnabled(True)

    def show_machine_details(self, machine_name):
        """Hiển thị chi tiết máy."""
        if not HAS_PYQT5:
            return

        machine_info = self.available_machines.get(machine_name, {})

        details_html = f"""
        <h3 style="color: #4A90E2;">{machine_name}</h3>
        <p><b>Manufacturer:</b> {machine_info.get("manufacturer", "Unknown")}</p>
        <p><b>Model:</b> {machine_info.get("model", "Unknown")}</p>
        <p><b>Description:</b> {machine_info.get("description", "No description available")}</p>

        <h4 style="color: #4A90E2;">Technical Specifications</h4>
        <p><b>Max Field Size:</b> {machine_info.get("max_field_size", "Unknown")} x {machine_info.get("max_field_size", "Unknown")} cm</p>
        <p><b>MLC Type:</b> {machine_info.get("mlc_type", "Unknown")}</p>
        <p><b>Max Dose Rate:</b> {machine_info.get("max_dose_rate", "Unknown")} MU/min</p>
        <p><b>Status:</b> <span style="color: {"green" if machine_info.get("status") == "Available" else "red"};">{machine_info.get("status", "Unknown")}</span></p>

        <h4 style="color: #4A90E2;">Available Energies</h4>
        """

        energies = machine_info.get("energies", {})
        if "photon" in energies and energies["photon"]:
            details_html += f"<p><b>Photon:</b> {', '.join(energies['photon'])}</p>"

        if "electron" in energies and energies["electron"]:
            details_html += f"<p><b>Electron:</b> {', '.join(energies['electron'])}</p>"

        self.details_text.setHtml(details_html)

    def get_selected_machine(self) -> Optional[str]:
        """Lấy máy được chọn."""
        return self.selected_machine

    def get_selected_machine_info(self) -> Optional[Dict[str, Any]]:
        """Lấy thông tin máy được chọn."""
        if self.selected_machine:
            return self.available_machines.get(self.selected_machine)
        return None


# Factory function
def show_machine_selection_dialog(parent=None, available_machines=None):
    """
    Hiển thị dialog chọn máy và trả về máy được chọn.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha
    available_machines : dict, optional
        Dictionary các máy có sẵn

    Returns
    -------
    tuple
        (selected_machine_name, machine_info) hoặc (None, None) nếu hủy
    """
    if not HAS_PYQT5:
        logger.warning("PyQt5 not available, returning default machine")
        return "TrueBeam", {"manufacturer": "Varian", "model": "TrueBeam"}

    dialog = MachineSelectionDialog(parent, available_machines)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return dialog.get_selected_machine(), dialog.get_selected_machine_info()
    else:
        return None, None
