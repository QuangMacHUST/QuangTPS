#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protocol Dialog Module

Hộp thoại cho phép người dùng chọn, xem, và quản lý các protocol lâm sàng,
tương tự như tính năng Protocol Selection trong Eclipse TPS.
"""

import os
import logging
from typing import Dict, List, Optional, Any

try:
from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTreeWidget,
        QTreeWidgetItem,
        QHeaderView,
        QSplitter,
        QTextEdit,
        QTabWidget,
        QWidget,
        QComboBox,
        QMessageBox,
        QFileDialog,
        QGroupBox,
        QRadioButton,
        QButtonGroup,
        QFormLayout,
        QLineEdit,
        QDialogButtonBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QIcon, QColor, QFont
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")

# Import QuangTPS modules
try:
    from quangtps.evaluation.clinical_goals import (
        ClinicalGoal,
        ClinicalGoalCollection,
        ClinicalGoalTemplate,
        ClinicalGoalManager,
        GoalType,
        GoalOperator,
        GoalPriority,
    )
except ImportError as e:
    logging.error(f"Không thể import module clinical_goals: {e}")

logger = logging.getLogger(__name__)


class ClinicalProtocolDialog(QDialog):
    """
    Hộp thoại chọn và quản lý protocol lâm sàng.

    Hộp thoại này cho phép người dùng chọn protocol lâm sàng từ danh sách có sẵn,
    xem chi tiết, và quản lý (import, export, v.v.) protocol cho đánh giá kế hoạch.
    """

    def __init__(self, parent=None):
        """Khởi tạo hộp thoại."""
        super().__init__(parent)
        self.setWindowTitle("Clinical Protocol Selection")
        self.resize(800, 600)

        # Khởi tạo biến thành viên
        self.protocol_manager = None
        self.selected_protocol = None
        self.filter_by_site = ""

        # Khởi tạo giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Panel bên trái: Danh sách protocol
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Filter by site
        filter_group = QGroupBox("Filter")
        filter_layout = QFormLayout(filter_group)

        self.site_combo = QComboBox()
        self.site_combo.addItem("All Sites", "")
        self.site_combo.addItem("Head and Neck", "head_and_neck")
        self.site_combo.addItem("Thorax", "thorax")
        self.site_combo.addItem("Breast", "breast")
        self.site_combo.addItem("Abdomen", "abdomen")
        self.site_combo.addItem("Pelvis", "pelvis")
        self.site_combo.addItem("Brain", "brain")
        self.site_combo.addItem("Prostate", "prostate")
        self.site_combo.currentIndexChanged.connect(self._on_site_filter_changed)

        filter_layout.addRow("Treatment Site:", self.site_combo)
        left_layout.addWidget(filter_group)

        # Protocol tree
        self.protocol_tree = QTreeWidget()
        self.protocol_tree.setHeaderLabels(["Protocol", "Site", "Description"])
        self.protocol_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.protocol_tree.setSelectionBehavior(QTreeWidget.SelectRows)
        self.protocol_tree.itemSelectionChanged.connect(self._on_protocol_selected)

        # Thiết lập kích thước cột
        header = self.protocol_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        left_layout.addWidget(self.protocol_tree, 1)

        # Thêm vào splitter
        splitter.addWidget(left_panel)

        # Panel bên phải: Chi tiết protocol
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tab widget cho các chế độ xem khác nhau
        tab_widget = QTabWidget()

        # Tab thông tin chi tiết
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)

        self.protocol_details = QTextEdit()
        self.protocol_details.setReadOnly(True)
        self.protocol_details.setStyleSheet("QTextEdit { background-color: #f8f9fa; }")
        details_layout.addWidget(self.protocol_details)

        tab_widget.addTab(details_tab, "Details")

        # Tab bảng mục tiêu
        goals_tab = QWidget()
        goals_layout = QVBoxLayout(goals_tab)

        self.goals_tree = QTreeWidget()
        self.goals_tree.setHeaderLabels(
            ["Structure", "Type", "Criteria", "Priority", "Notes"]
        )

        # Thiết lập kích thước cột
        header = self.goals_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        goals_layout.addWidget(self.goals_tree)

        tab_widget.addTab(goals_tab, "Goals")

        right_layout.addWidget(tab_widget)

        # Thêm vào splitter
        splitter.addWidget(right_panel)

        # Thiết lập kích thước ban đầu cho splitter
        splitter.setSizes([300, 500])

        # Thêm splitter vào layout chính
        main_layout.addWidget(splitter)

        # Nút điều khiển
        button_layout = QHBoxLayout()

        # Import/Export buttons
        import_button = QPushButton("Import...")
        import_button.clicked.connect(self._on_import)
        button_layout.addWidget(import_button)

        export_button = QPushButton("Export...")
        export_button.clicked.connect(self._on_export)
        button_layout.addWidget(export_button)

        button_layout.addStretch()

        # OK/Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)

        main_layout.addLayout(button_layout)

    def setProtocolManager(self, protocol_manager):
        """
        Thiết lập đối tượng quản lý protocol.

        Parameters
        ----------
        protocol_manager : ClinicalGoalManager
            Đối tượng quản lý protocol
        """
        self.protocol_manager = protocol_manager
        self._populate_protocol_tree()

    def getSelectedProtocol(self):
        """
        Lấy protocol đã chọn.

        Returns
        -------
        ClinicalGoalCollection
            Protocol đã chọn, hoặc None nếu không có protocol nào được chọn
        """
        return self.selected_protocol

    def _populate_protocol_tree(self):
        """Điền danh sách protocol vào cây."""
        self.protocol_tree.clear()

        if not self.protocol_manager:
            return

        try:
            # Lấy danh sách tên protocol
            template_names = self.protocol_manager.get_template_names()

            for name in template_names:
                template = self.protocol_manager.get_template_by_name(name)
                if not template:
                    continue

                # Lọc theo site nếu có
                if (
                    self.filter_by_site
                    and template.treatment_site != self.filter_by_site
                ):
                    continue

                # Tạo item trong cây
                item = QTreeWidgetItem(
                    [
                        name,
                        template.treatment_site,
                        template.description[:50] + "..."
                        if len(template.description) > 50
                        else template.description,
                    ]
                )

                # Lưu trữ dữ liệu
                item.setData(0, Qt.UserRole, name)

                self.protocol_tree.addTopLevelItem(item)
        except Exception as e:
            logger.error(f"Lỗi khi điền danh sách protocol: {e}")

    def _on_site_filter_changed(self, index):
        """
        Xử lý sự kiện khi bộ lọc site thay đổi.

        Parameters
        ----------
        index : int
            Chỉ số của mục đã chọn
        """
        self.filter_by_site = self.site_combo.itemData(index)
        self._populate_protocol_tree()

    def _on_protocol_selected(self):
        """Xử lý sự kiện khi protocol được chọn trong cây."""
        selected_items = self.protocol_tree.selectedItems()
        if not selected_items:
            self.selected_protocol = None
            self.protocol_details.clear()
            self.goals_tree.clear()
            return

        # Lấy tên protocol
        protocol_name = selected_items[0].data(0, Qt.UserRole)

        try:
            # Lấy template
            template = self.protocol_manager.get_template_by_name(protocol_name)
            if not template:
                return

            # Chuyển đổi template thành collection
            self.selected_protocol = template.to_goal_collection()

            # Hiển thị chi tiết
            self._display_protocol_details(template)

            # Hiển thị danh sách mục tiêu
            self._display_protocol_goals(template)
        except Exception as e:
            logger.error(f"Lỗi khi hiển thị chi tiết protocol: {e}")

    def _display_protocol_details(self, template):
        """
        Hiển thị chi tiết protocol.

        Parameters
        ----------
        template : ClinicalGoalTemplate
            Template protocol cần hiển thị
        """
        # Tạo nội dung HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 10px; }}
                h2 {{ color: #2c3e50; }}
                .info {{ margin-bottom: 15px; }}
                .label {{ font-weight: bold; }}
                .value {{ margin-left: 10px; }}
                .description {{ margin-top: 20px; white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <h2>{template.name}</h2>

            <div class="info">
                <span class="label">Treatment Site:</span>
                <span class="value">{template.treatment_site}</span>
            </div>

            <div class="info">
                <span class="label">Number of Goals:</span>
                <span class="value">{len(template.goals)}</span>
            </div>

            <div class="description">
                {template.description}
            </div>
        </body>
        </html>
        """

        self.protocol_details.setHtml(html)

    def _display_protocol_goals(self, template):
        """
        Hiển thị danh sách mục tiêu của protocol.

        Parameters
        ----------
        template : ClinicalGoalTemplate
            Template protocol cần hiển thị
        """
        self.goals_tree.clear()

        # Nhóm mục tiêu theo cấu trúc
        structure_goals = {}

        for goal in template.goals:
            if goal.structure_name not in structure_goals:
                structure_goals[goal.structure_name] = []

            structure_goals[goal.structure_name].append(goal)

        # Thêm vào tree
        for structure_name, goals in structure_goals.items():
            # Tạo item cha cho cấu trúc
            structure_item = QTreeWidgetItem([structure_name])
            structure_item.setExpanded(True)
            self.goals_tree.addTopLevelItem(structure_item)

            # Thêm các mục tiêu của cấu trúc
            for goal in goals:
                # Tạo chuỗi tiêu chí
                criteria = f"{self._get_operator_str(goal.operator)} {goal.value}"
                if (
                    goal.goal_type == GoalType.VOLUME_AT_DOSE
                    and goal.dose_level is not None
                ):
                    criteria = f"V{goal.dose_level}Gy {criteria}%"
                elif (
                    goal.goal_type == GoalType.DOSE_AT_VOLUME
                    and goal.volume_level is not None
                ):
                    criteria = f"D{goal.volume_level}% {criteria}Gy"

                # Tạo item con cho mục tiêu
                goal_item = QTreeWidgetItem(
                    [
                        "",  # Structure đã hiển thị ở item cha
                        self._get_type_str(goal.goal_type),
                        criteria,
                        self._get_priority_str(goal.priority),
                        goal.notes,
                    ]
                )

                structure_item.addChild(goal_item)

    def _get_type_str(self, goal_type):
        """
        Chuyển đổi loại mục tiêu thành chuỗi.

        Parameters
        ----------
        goal_type : GoalType
            Loại mục tiêu

        Returns
        -------
        str
            Chuỗi đại diện cho loại mục tiêu
        """
        type_map = {
            GoalType.VOLUME_AT_DOSE: "Volume at Dose",
            GoalType.DOSE_AT_VOLUME: "Dose at Volume",
            GoalType.MAX_DOSE: "Maximum Dose",
            GoalType.MIN_DOSE: "Minimum Dose",
            GoalType.MEAN_DOSE: "Mean Dose",
            GoalType.CI: "Conformity Index",
            GoalType.HI: "Homogeneity Index",
            GoalType.GI: "Gradient Index",
        }

        return type_map.get(goal_type, "Unknown")

    def _get_operator_str(self, operator):
        """
        Chuyển đổi toán tử thành chuỗi.

        Parameters
        ----------
        operator : GoalOperator
            Toán tử

        Returns
        -------
        str
            Chuỗi đại diện cho toán tử
        """
        operator_map = {
            GoalOperator.LESS_THAN: "<",
            GoalOperator.LESS_THAN_OR_EQUAL: "≤",
            GoalOperator.GREATER_THAN: ">",
            GoalOperator.GREATER_THAN_OR_EQUAL: "≥",
            GoalOperator.EQUAL: "=",
            GoalOperator.NOT_EQUAL: "≠",
        }

        return operator_map.get(operator, "?")

    def _get_priority_str(self, priority):
        """
        Chuyển đổi mức độ ưu tiên thành chuỗi.

        Parameters
        ----------
        priority : GoalPriority
            Mức độ ưu tiên

        Returns
        -------
        str
            Chuỗi đại diện cho mức độ ưu tiên
        """
        priority_map = {
            GoalPriority.CRITICAL: "Critical",
            GoalPriority.MAJOR: "Major",
            GoalPriority.MINOR: "Minor",
        }

        return priority_map.get(priority, "Unknown")

    def _on_import(self):
        """Xử lý sự kiện khi nút Import được nhấn."""
        # Hiển thị hộp thoại chọn tệp
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Protocol", "", "JSON Files (*.json);;All Files (*)"
        )

        if not filename:
            return

        try:
            # Tải template từ tệp
            template = ClinicalGoalTemplate.load_from_file(filename)

            if not template:
                QMessageBox.warning(
                    self, "Import Error", "Could not load protocol from file."
                )
                return

            # Thêm vào manager
            if self.protocol_manager:
                self.protocol_manager.add_template(template)

                # Cập nhật giao diện
                self._populate_protocol_tree()

                # Chọn protocol mới
                self._select_protocol_by_name(template.name)

                QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Protocol '{template.name}' has been imported successfully.",
                )
        except Exception as e:
            logger.error(f"Lỗi khi import protocol: {e}")
            QMessageBox.critical(
                self, "Import Error", f"Error importing protocol: {str(e)}"
            )

    def _on_export(self):
        """Xử lý sự kiện khi nút Export được nhấn."""
        # Kiểm tra xem có protocol nào được chọn không
        if not self.selected_protocol:
            QMessageBox.warning(
                self, "Export Error", "Please select a protocol to export."
            )
            return

        # Hiển thị hộp thoại chọn tệp
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Protocol", "", "JSON Files (*.json);;All Files (*)"
        )

        if not filename:
                return

        try:
            # Lấy template từ protocol đã chọn
            template = None
            selected_items = self.protocol_tree.selectedItems()
            if selected_items:
                protocol_name = selected_items[0].data(0, Qt.UserRole)
                template = self.protocol_manager.get_template_by_name(protocol_name)

            if not template:
                QMessageBox.warning(
                    self, "Export Error", "Could not find selected protocol."
                )
                return

            # Lưu template ra tệp
            template.save_to_file(filename)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Protocol '{template.name}' has been exported successfully.",
            )
        except Exception as e:
            logger.error(f"Lỗi khi export protocol: {e}")
            QMessageBox.critical(
                self, "Export Error", f"Error exporting protocol: {str(e)}"
            )

    def _select_protocol_by_name(self, name):
        """
        Chọn protocol trong tree view theo tên.

        Parameters
        ----------
        name : str
            Tên của protocol cần chọn
        """
        # Duyệt qua các item trong tree
        for i in range(self.protocol_tree.topLevelItemCount()):
            item = self.protocol_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == name:
                # Chọn item
                self.protocol_tree.setCurrentItem(item)
                break


if __name__ == "__main__":
    # Test code
    import sys

    # Sử dụng try/except để xử lý lỗi import và thực thi
    try:
    from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)

        # Tạo manager với một số template mẫu
        manager = ClinicalGoalManager()

        # Tạo template mẫu
        head_neck_template = ClinicalGoalTemplate(
            name="Head and Neck",
            treatment_site="head_and_neck",
            description="Protocol for head and neck treatments.",
        )

        # Thêm mục tiêu cho template
        ptv_goal = ClinicalGoal(
            structure_id="ptv",
            structure_name="PTV70",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=70.0,
            volume_level=95.0,
            priority=GoalPriority.CRITICAL,
        )
        head_neck_template.add_goal(ptv_goal)

        cord_goal = ClinicalGoal(
            structure_id="cord",
            structure_name="Spinal Cord",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=45.0,
            priority=GoalPriority.CRITICAL,
        )
        head_neck_template.add_goal(cord_goal)

        # Thêm template vào manager
        # Kiểm tra phương thức add_template có tồn tại không
        if hasattr(manager, "add_template"):
            manager.add_template(head_neck_template)
        else:
            # Nếu không có phương thức add_template, có thể thử sử dụng phương thức khác hoặc gán trực tiếp
            logger.warning(
                "ClinicalGoalManager không có phương thức add_template, thử phương pháp thay thế"
            )
            if hasattr(manager, "templates"):
                manager.templates = {head_neck_template.name: head_neck_template}
            else:
                logger.error("Không thể thêm template vào manager")

        # Tạo và hiển thị dialog
        dialog = ClinicalProtocolDialog()
        dialog.setProtocolManager(manager)
        dialog.exec_()

        # Lấy protocol đã chọn
        selected_protocol = dialog.getSelectedProtocol()
        if selected_protocol:
            print(f"Selected protocol: {selected_protocol.name}")
    else:
        print("No protocol selected")

    except ImportError as e:
        logger.error(f"Không thể khởi chạy test ClinicalProtocolDialog: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Lỗi không xác định khi chạy test ClinicalProtocolDialog: {e}")
        print(f"Unexpected error: {e}")
