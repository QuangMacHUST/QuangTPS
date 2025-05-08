#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Widget quản lý quy tắc kiểm tra log file máy điều trị.

Module này cung cấp giao diện người dùng để xem, chỉnh sửa và áp dụng
các quy tắc kiểm tra log file máy điều trị.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QPushButton,
        QFileDialog,
        QTableWidget,
        QTableWidgetItem,
        QComboBox,
        QLineEdit,
        QDoubleSpinBox,
        QDialog,
        QDialogButtonBox,
        QGroupBox,
        QFormLayout,
        QStackedWidget,
        QMessageBox,
        QHeaderView,
        QCheckBox,
        QFrame,
        QPlainTextEdit,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QIcon, QColor
except ImportError:
    # Các lớp giả để tránh lỗi khi không có PyQt5
    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass


from quangtps.evaluation.qa.log_file_validator import (
    LogFileValidator,
    ValidationRule,
    ValidationRuleType,
    DeviationSeverity,
)

logger = logging.getLogger(__name__)


class RuleSeverityComboBox(QComboBox):
    """Combobox chọn mức độ nghiêm trọng của quy tắc."""

    def __init__(self, parent=None):
        """Khởi tạo combobox."""
        super().__init__(parent)
        self._init_items()

    def _init_items(self):
        """Thêm các mức độ nghiêm trọng vào combobox."""
        self.clear()

        # Thêm các mức độ nghiêm trọng
        severity_items = [
            (DeviationSeverity.CRITICAL, "Nghiêm trọng", QColor(255, 0, 0, 100)),
            (DeviationSeverity.MAJOR, "Lớn", QColor(255, 128, 0, 100)),
            (DeviationSeverity.MODERATE, "Trung bình", QColor(255, 255, 0, 100)),
            (DeviationSeverity.MINOR, "Nhỏ", QColor(0, 255, 0, 100)),
            (DeviationSeverity.ACCEPTABLE, "Chấp nhận được", QColor(0, 255, 128, 100)),
        ]

        for severity, text, color in severity_items:
            self.addItem(text)
            index = self.count() - 1
            self.setItemData(index, severity, Qt.UserRole)
            self.setItemData(index, color, Qt.BackgroundRole)

    def set_severity(self, severity: DeviationSeverity):
        """Thiết lập mức độ nghiêm trọng được chọn."""
        for i in range(self.count()):
            if self.itemData(i, Qt.UserRole) == severity:
                self.setCurrentIndex(i)
                return

        # Mặc định nếu không tìm thấy
        self.setCurrentIndex(0)

    def get_severity(self) -> DeviationSeverity:
        """Lấy mức độ nghiêm trọng đã chọn."""
        return self.currentData(Qt.UserRole)


class RuleTypeComboBox(QComboBox):
    """Combobox chọn loại quy tắc kiểm tra."""

    def __init__(self, parent=None):
        """Khởi tạo combobox."""
        super().__init__(parent)
        self._init_items()

    def _init_items(self):
        """Thêm các loại quy tắc vào combobox."""
        self.clear()

        # Thêm các loại quy tắc
        rule_types = [
            (ValidationRuleType.PARAMETER_LIMIT, "Giới hạn tham số"),
            (ValidationRuleType.PARAMETER_DEVIATION, "Độ lệch tham số"),
            (ValidationRuleType.PATTERN_MATCH, "Khớp mẫu nội dung"),
        ]

        for rule_type, text in rule_types:
            self.addItem(text)
            self.setItemData(self.count() - 1, rule_type, Qt.UserRole)

    def set_rule_type(self, rule_type: ValidationRuleType):
        """Thiết lập loại quy tắc được chọn."""
        for i in range(self.count()):
            if self.itemData(i, Qt.UserRole) == rule_type:
                self.setCurrentIndex(i)
                return

        # Mặc định nếu không tìm thấy
        self.setCurrentIndex(0)

    def get_rule_type(self) -> ValidationRuleType:
        """Lấy loại quy tắc đã chọn."""
        return self.currentData(Qt.UserRole)


class AddEditRuleDialog(QDialog):
    """Dialog thêm/sửa quy tắc kiểm tra."""

    def __init__(self, parent=None, edit_rule: ValidationRule = None):
        """
        Khởi tạo dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        edit_rule : ValidationRule, optional
            Quy tắc cần chỉnh sửa, None nếu tạo mới
        """
        super().__init__(parent)
        self.edit_rule = edit_rule
        self.is_edit_mode = edit_rule is not None

        self._init_ui()

        # Nếu đang chỉnh sửa, điền thông tin quy tắc
        if self.is_edit_mode:
            self._fill_rule_data()

    def _init_ui(self):
        """Khởi tạo giao diện dialog."""
        self.setWindowTitle(
            "Thêm quy tắc mới" if not self.is_edit_mode else "Chỉnh sửa quy tắc"
        )
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        # Form layout cho các trường cơ bản
        form_group = QGroupBox("Thông tin quy tắc")
        form_layout = QFormLayout(form_group)

        # Tên quy tắc
        self.name_edit = QLineEdit()
        form_layout.addRow("Tên quy tắc:", self.name_edit)

        # Loại quy tắc
        self.rule_type_combo = RuleTypeComboBox()
        self.rule_type_combo.currentIndexChanged.connect(self._on_rule_type_changed)
        form_layout.addRow("Loại quy tắc:", self.rule_type_combo)

        # Mức độ nghiêm trọng
        self.severity_combo = RuleSeverityComboBox()
        form_layout.addRow("Mức độ nghiêm trọng:", self.severity_combo)

        # Thông báo
        self.message_edit = QLineEdit()
        form_layout.addRow("Thông báo lỗi:", self.message_edit)

        # Thêm form vào layout chính
        layout.addWidget(form_group)

        # Stacked widget cho các tham số tùy theo loại quy tắc
        self.params_stack = QStackedWidget()

        # Tham số cho loại PARAMETER_LIMIT
        limit_widget = QWidget()
        limit_layout = QFormLayout(limit_widget)

        self.param_name_edit = QLineEdit()
        limit_layout.addRow("Tên tham số:", self.param_name_edit)

        self.min_value_spin = QDoubleSpinBox()
        self.min_value_spin.setRange(-9999, 9999)
        self.min_value_spin.setDecimals(4)
        self.min_value_spin.setSpecialValueText("Không giới hạn")
        limit_layout.addRow("Giá trị tối thiểu:", self.min_value_spin)

        self.max_value_spin = QDoubleSpinBox()
        self.max_value_spin.setRange(-9999, 9999)
        self.max_value_spin.setDecimals(4)
        self.max_value_spin.setSpecialValueText("Không giới hạn")
        limit_layout.addRow("Giá trị tối đa:", self.max_value_spin)

        self.params_stack.addWidget(limit_widget)

        # Tham số cho loại PARAMETER_DEVIATION
        deviation_widget = QWidget()
        deviation_layout = QFormLayout(deviation_widget)

        self.deviation_param_edit = QLineEdit()
        deviation_layout.addRow("Tên tham số:", self.deviation_param_edit)

        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0, 9999)
        self.tolerance_spin.setDecimals(4)
        deviation_layout.addRow("Dung sai cho phép:", self.tolerance_spin)

        self.params_stack.addWidget(deviation_widget)

        # Tham số cho loại PATTERN_MATCH
        pattern_widget = QWidget()
        pattern_layout = QFormLayout(pattern_widget)

        self.pattern_edit = QPlainTextEdit()
        self.pattern_edit.setMaximumHeight(100)
        pattern_layout.addRow("Mẫu regex:", self.pattern_edit)

        self.params_stack.addWidget(pattern_widget)

        # Thêm stacked widget vào layout chính
        layout.addWidget(self.params_stack)

        # Nút OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Thiết lập tab order
        self.setTabOrder(self.name_edit, self.rule_type_combo)
        self.setTabOrder(self.rule_type_combo, self.severity_combo)
        self.setTabOrder(self.severity_combo, self.message_edit)

        # Hiển thị tham số phù hợp với loại quy tắc
        self._on_rule_type_changed()

    def _on_rule_type_changed(self):
        """Xử lý khi người dùng thay đổi loại quy tắc."""
        rule_type = self.rule_type_combo.get_rule_type()

        if rule_type == ValidationRuleType.PARAMETER_LIMIT:
            self.params_stack.setCurrentIndex(0)
        elif rule_type == ValidationRuleType.PARAMETER_DEVIATION:
            self.params_stack.setCurrentIndex(1)
        elif rule_type == ValidationRuleType.PATTERN_MATCH:
            self.params_stack.setCurrentIndex(2)

    def _fill_rule_data(self):
        """Điền thông tin quy tắc cần chỉnh sửa vào các controls."""
        if not self.edit_rule:
            return

        # Thông tin cơ bản
        self.name_edit.setText(self.edit_rule.name)
        self.rule_type_combo.set_rule_type(self.edit_rule.rule_type)
        self.severity_combo.set_severity(self.edit_rule.severity)

        if self.edit_rule.message:
            self.message_edit.setText(self.edit_rule.message)

        # Thông tin tùy theo loại quy tắc
        rule_type = self.edit_rule.rule_type

        if rule_type == ValidationRuleType.PARAMETER_LIMIT:
            self.param_name_edit.setText(self.edit_rule.parameter or "")

            if self.edit_rule.min_value is not None:
                self.min_value_spin.setValue(self.edit_rule.min_value)

            if self.edit_rule.max_value is not None:
                self.max_value_spin.setValue(self.edit_rule.max_value)

        elif rule_type == ValidationRuleType.PARAMETER_DEVIATION:
            self.deviation_param_edit.setText(self.edit_rule.parameter or "")

            if self.edit_rule.tolerance is not None:
                self.tolerance_spin.setValue(self.edit_rule.tolerance)

        elif rule_type == ValidationRuleType.PATTERN_MATCH:
            if self.edit_rule.pattern:
                self.pattern_edit.setPlainText(self.edit_rule.pattern)

    def get_rule_data(self) -> Dict[str, Any]:
        """
        Lấy dữ liệu quy tắc từ các controls.

        Returns
        -------
        Dict[str, Any]
            Dữ liệu quy tắc
        """
        rule_data = {
            "name": self.name_edit.text().strip(),
            "rule_type": self.rule_type_combo.get_rule_type(),
            "severity": self.severity_combo.get_severity(),
            "message": self.message_edit.text().strip() or None,
        }

        # Tham số tùy theo loại quy tắc
        rule_type = rule_data["rule_type"]

        if rule_type == ValidationRuleType.PARAMETER_LIMIT:
            rule_data["parameter"] = self.param_name_edit.text().strip()

            min_value = self.min_value_spin.value()
            if min_value == self.min_value_spin.minimum():
                min_value = None
            rule_data["min_value"] = min_value

            max_value = self.max_value_spin.value()
            if max_value == self.max_value_spin.minimum():
                max_value = None
            rule_data["max_value"] = max_value

        elif rule_type == ValidationRuleType.PARAMETER_DEVIATION:
            rule_data["parameter"] = self.deviation_param_edit.text().strip()
            rule_data["tolerance"] = self.tolerance_spin.value()

        elif rule_type == ValidationRuleType.PATTERN_MATCH:
            rule_data["pattern"] = self.pattern_edit.toPlainText().strip()

        return rule_data

    def validate(self) -> Tuple[bool, str]:
        """
        Kiểm tra dữ liệu nhập vào.

        Returns
        -------
        Tuple[bool, str]
            (True, "") nếu dữ liệu hợp lệ, ngược lại (False, lỗi)
        """
        rule_data = self.get_rule_data()

        # Kiểm tra tên quy tắc
        if not rule_data["name"]:
            return False, "Tên quy tắc không được để trống"

        # Kiểm tra tham số tùy theo loại quy tắc
        rule_type = rule_data["rule_type"]

        if rule_type in (
            ValidationRuleType.PARAMETER_LIMIT,
            ValidationRuleType.PARAMETER_DEVIATION,
        ):
            if not rule_data.get("parameter"):
                return False, "Tên tham số không được để trống"

        if rule_type == ValidationRuleType.PARAMETER_LIMIT:
            if rule_data["min_value"] is None and rule_data["max_value"] is None:
                return (
                    False,
                    "Phải chỉ định ít nhất một giới hạn (tối thiểu hoặc tối đa)",
                )

        if rule_type == ValidationRuleType.PARAMETER_DEVIATION:
            if rule_data["tolerance"] <= 0:
                return False, "Dung sai phải lớn hơn 0"

        if rule_type == ValidationRuleType.PATTERN_MATCH:
            if not rule_data.get("pattern"):
                return False, "Mẫu regex không được để trống"

            # Kiểm tra mẫu regex hợp lệ
            try:
                import re

                re.compile(rule_data["pattern"])
            except re.error as e:
                return False, f"Mẫu regex không hợp lệ: {str(e)}"

        return True, ""

    def accept(self):
        """Kiểm tra dữ liệu trước khi chấp nhận."""
        valid, error_message = self.validate()

        if not valid:
            QMessageBox.warning(self, "Lỗi", error_message)
            return

        super().accept()


class LogFileValidatorWidget(QWidget):
    """Widget quản lý quy tắc kiểm tra log file."""

    rules_changed = pyqtSignal()  # Tín hiệu khi danh sách quy tắc thay đổi

    def __init__(self, parent=None):
        """Khởi tạo widget."""
        super().__init__(parent)
        self.validator = LogFileValidator()

        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Tiêu đề
        title_label = QLabel("Quản lý quy tắc kiểm tra log file")
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 2)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Layout cho các nút tác vụ
        buttons_layout = QHBoxLayout()

        self.add_rule_button = QPushButton("Thêm quy tắc")
        self.add_rule_button.clicked.connect(self._add_rule)
        buttons_layout.addWidget(self.add_rule_button)

        self.edit_rule_button = QPushButton("Sửa quy tắc")
        self.edit_rule_button.setEnabled(False)
        self.edit_rule_button.clicked.connect(self._edit_rule)
        buttons_layout.addWidget(self.edit_rule_button)

        self.remove_rule_button = QPushButton("Xóa quy tắc")
        self.remove_rule_button.setEnabled(False)
        self.remove_rule_button.clicked.connect(self._remove_rule)
        buttons_layout.addWidget(self.remove_rule_button)

        buttons_layout.addStretch()

        self.import_rules_button = QPushButton("Nhập quy tắc")
        self.import_rules_button.clicked.connect(self._import_rules)
        buttons_layout.addWidget(self.import_rules_button)

        self.export_rules_button = QPushButton("Xuất quy tắc")
        self.export_rules_button.clicked.connect(self._export_rules)
        buttons_layout.addWidget(self.export_rules_button)

        layout.addLayout(buttons_layout)

        # Bảng danh sách quy tắc
        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(5)
        self.rules_table.setHorizontalHeaderLabels(
            ["Tên", "Loại", "Tham số", "Giá trị", "Mức độ"]
        )
        self.rules_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rules_table.setSelectionMode(QTableWidget.SingleSelection)
        self.rules_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rules_table.verticalHeader().setVisible(False)
        self.rules_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rules_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.rules_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeToContents
        )

        self.rules_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.rules_table.doubleClicked.connect(self._edit_rule)

        layout.addWidget(self.rules_table, 1)

        # Tạo các quy tắc mặc định nếu chưa có
        if not self.validator.rules:
            self._create_default_rules()

        # Cập nhật bảng quy tắc
        self._update_rules_table()

    def _create_default_rules(self):
        """Tạo các quy tắc mặc định."""
        default_rules = LogFileValidator.create_default_rules()
        for rule in default_rules:
            self.validator.add_rule(rule)

    def _update_rules_table(self):
        """Cập nhật bảng quy tắc."""
        self.rules_table.setRowCount(0)

        for rule in self.validator.rules:
            row = self.rules_table.rowCount()
            self.rules_table.insertRow(row)

            # Tên quy tắc
            name_item = QTableWidgetItem(rule.name)
            name_item.setData(Qt.UserRole, rule)
            self.rules_table.setItem(row, 0, name_item)

            # Loại quy tắc
            rule_type_map = {
                ValidationRuleType.PARAMETER_LIMIT: "Giới hạn tham số",
                ValidationRuleType.PARAMETER_DEVIATION: "Độ lệch tham số",
                ValidationRuleType.PATTERN_MATCH: "Khớp mẫu nội dung",
                ValidationRuleType.CUSTOM_FUNCTION: "Hàm tùy chỉnh",
            }
            rule_type_item = QTableWidgetItem(rule_type_map.get(rule.rule_type, ""))
            self.rules_table.setItem(row, 1, rule_type_item)

            # Tham số
            param_item = QTableWidgetItem(rule.parameter or "")
            self.rules_table.setItem(row, 2, param_item)

            # Giá trị
            value_text = ""
            if rule.rule_type == ValidationRuleType.PARAMETER_LIMIT:
                if rule.min_value is not None and rule.max_value is not None:
                    value_text = f"{rule.min_value} - {rule.max_value}"
                elif rule.min_value is not None:
                    value_text = f">= {rule.min_value}"
                elif rule.max_value is not None:
                    value_text = f"<= {rule.max_value}"
            elif rule.rule_type == ValidationRuleType.PARAMETER_DEVIATION:
                if rule.tolerance is not None:
                    value_text = f"Dung sai: {rule.tolerance}"
            elif rule.rule_type == ValidationRuleType.PATTERN_MATCH:
                if rule.pattern:
                    value_text = (
                        rule.pattern[:30] + "..."
                        if len(rule.pattern) > 30
                        else rule.pattern
                    )

            value_item = QTableWidgetItem(value_text)
            self.rules_table.setItem(row, 3, value_item)

            # Mức độ nghiêm trọng
            severity_map = {
                DeviationSeverity.CRITICAL: "Nghiêm trọng",
                DeviationSeverity.MAJOR: "Lớn",
                DeviationSeverity.MODERATE: "Trung bình",
                DeviationSeverity.MINOR: "Nhỏ",
                DeviationSeverity.ACCEPTABLE: "Chấp nhận được",
            }
            severity_item = QTableWidgetItem(severity_map.get(rule.severity, ""))

            # Màu nền theo mức độ nghiêm trọng
            severity_colors = {
                DeviationSeverity.CRITICAL: QColor(255, 0, 0, 100),
                DeviationSeverity.MAJOR: QColor(255, 128, 0, 100),
                DeviationSeverity.MODERATE: QColor(255, 255, 0, 100),
                DeviationSeverity.MINOR: QColor(0, 255, 0, 100),
                DeviationSeverity.ACCEPTABLE: QColor(0, 255, 128, 100),
            }

            if rule.severity in severity_colors:
                severity_item.setBackground(severity_colors[rule.severity])

            self.rules_table.setItem(row, 4, severity_item)

        # Cập nhật trạng thái nút xuất
        self.export_rules_button.setEnabled(self.rules_table.rowCount() > 0)

    def _on_selection_changed(self):
        """Xử lý khi người dùng thay đổi lựa chọn."""
        has_selection = len(self.rules_table.selectedItems()) > 0
        self.edit_rule_button.setEnabled(has_selection)
        self.remove_rule_button.setEnabled(has_selection)

    def _get_selected_rule(self) -> Optional[ValidationRule]:
        """Lấy quy tắc đang được chọn."""
        selected_items = self.rules_table.selectedItems()
        if not selected_items:
            return None

        row = selected_items[0].row()
        return self.rules_table.item(row, 0).data(Qt.UserRole)

    def _add_rule(self):
        """Thêm quy tắc mới."""
        dialog = AddEditRuleDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            rule_data = dialog.get_rule_data()

            # Tạo quy tắc mới
            rule = ValidationRule(
                name=rule_data["name"],
                rule_type=rule_data["rule_type"],
                parameter=rule_data.get("parameter"),
                min_value=rule_data.get("min_value"),
                max_value=rule_data.get("max_value"),
                tolerance=rule_data.get("tolerance"),
                pattern=rule_data.get("pattern"),
                message=rule_data.get("message"),
                severity=rule_data["severity"],
            )

            # Thêm vào validator
            self.validator.add_rule(rule)

            # Cập nhật bảng
            self._update_rules_table()

            # Phát tín hiệu thay đổi
            self.rules_changed.emit()

    def _edit_rule(self):
        """Chỉnh sửa quy tắc đã chọn."""
        rule = self._get_selected_rule()
        if not rule:
            return

        dialog = AddEditRuleDialog(self, rule)
        if dialog.exec_() == QDialog.Accepted:
            rule_data = dialog.get_rule_data()

            # Cập nhật quy tắc
            rule.name = rule_data["name"]
            rule.rule_type = rule_data["rule_type"]
            rule.parameter = rule_data.get("parameter")
            rule.min_value = rule_data.get("min_value")
            rule.max_value = rule_data.get("max_value")
            rule.tolerance = rule_data.get("tolerance")
            rule.pattern = rule_data.get("pattern")
            rule.message = rule_data.get("message")
            rule.severity = rule_data["severity"]

            # Cập nhật bảng
            self._update_rules_table()

            # Phát tín hiệu thay đổi
            self.rules_changed.emit()

    def _remove_rule(self):
        """Xóa quy tắc đã chọn."""
        rule = self._get_selected_rule()
        if not rule:
            return

        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa quy tắc '{rule.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Xóa quy tắc
            self.validator.remove_rule(rule.name)

            # Cập nhật bảng
            self._update_rules_table()

            # Phát tín hiệu thay đổi
            self.rules_changed.emit()

    def _import_rules(self):
        """Nhập quy tắc từ file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Nhập quy tắc", "", "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            # Tải quy tắc từ file
            imported_validator = LogFileValidator.load_rules(file_path)

            if not imported_validator.rules:
                QMessageBox.warning(
                    self, "Lỗi", "Không tìm thấy quy tắc nào trong file."
                )
                return

            # Xác nhận ghi đè hoặc thêm vào
            if self.validator.rules:
                reply = QMessageBox.question(
                    self,
                    "Xác nhận nhập",
                    "Bạn muốn ghi đè các quy tắc hiện có hay thêm vào?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.No,
                )

                if reply == QMessageBox.Cancel:
                    return

                if reply == QMessageBox.Yes:
                    # Ghi đè
                    self.validator = imported_validator
                else:
                    # Thêm vào
                    for rule in imported_validator.rules:
                        self.validator.add_rule(rule)
            else:
                # Không có quy tắc hiện tại, ghi đè
                self.validator = imported_validator

            # Cập nhật bảng
            self._update_rules_table()

            # Thông báo thành công
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã nhập {len(imported_validator.rules)} quy tắc từ file.",
            )

            # Phát tín hiệu thay đổi
            self.rules_changed.emit()

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể nhập quy tắc từ file: {str(e)}"
            )
            logger.error(f"Lỗi khi nhập quy tắc từ file {file_path}: {str(e)}")

    def _export_rules(self):
        """Xuất quy tắc ra file."""
        if not self.validator.rules:
            QMessageBox.warning(self, "Cảnh báo", "Không có quy tắc nào để xuất.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất quy tắc", "", "JSON Files (*.json);;All Files (*.*)"
        )

        if not file_path:
            return

        # Thêm phần mở rộng .json nếu chưa có
        if not file_path.lower().endswith(".json"):
            file_path += ".json"

        try:
            # Lưu quy tắc vào file
            success = self.validator.save_rules(file_path)

            if success:
                QMessageBox.information(
                    self,
                    "Thành công",
                    f"Đã xuất {len(self.validator.rules)} quy tắc ra file.",
                )
            else:
                QMessageBox.warning(self, "Lỗi", "Không thể xuất quy tắc ra file.")

        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Không thể xuất quy tắc ra file: {str(e)}"
            )
            logger.error(f"Lỗi khi xuất quy tắc ra file {file_path}: {str(e)}")

    def set_validator(self, validator: LogFileValidator):
        """
        Thiết lập validator cho widget.

        Parameters
        ----------
        validator : LogFileValidator
            Validator cần thiết lập
        """
        self.validator = validator
        self._update_rules_table()

    def get_validator(self) -> LogFileValidator:
        """
        Lấy validator hiện tại.

        Returns
        -------
        LogFileValidator
            Validator hiện tại
        """
        return self.validator


if __name__ == "__main__":
    # Chạy widget để kiểm thử
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = LogFileValidatorWidget()
    widget.show()
    sys.exit(app.exec_())
