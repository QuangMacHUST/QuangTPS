#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog Knowledge-Based Planning (KBP) theo phong cách Eclipse.

Dialog này cho phép người dùng áp dụng các mô hình KBP vào quá trình lập kế hoạch,
tự động đề xuất các ràng buộc liều và tham số tối ưu dựa trên dữ liệu các kế hoạch trước đó.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union, Set

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
        QHeaderView,
        QTabWidget,
        QWidget,
        QGroupBox,
        QFormLayout,
        QCheckBox,
        QProgressBar,
        QMessageBox,
        QSpinBox,
        QDoubleSpinBox,
        QSplitter,
        QFrame,
        QApplication,
        QTreeWidget,
        QTreeWidgetItem,
    )
    from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QFont, QColor, QIcon
except ImportError:
    try:
        from PySide2.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QComboBox,
            QTableWidget,
            QTableWidgetItem,
            QHeaderView,
            QTabWidget,
            QWidget,
            QGroupBox,
            QFormLayout,
            QCheckBox,
            QProgressBar,
            QMessageBox,
            QSpinBox,
            QDoubleSpinBox,
            QSplitter,
            QFrame,
            QApplication,
            QTreeWidget,
            QTreeWidgetItem,
        )
        from PySide2.QtCore import Qt, QSize, Signal as pyqtSignal, Slot as pyqtSlot
        from PySide2.QtGui import QFont, QColor, QIcon
    except ImportError:
        print("Không thể import thư viện Qt. Tính năng KBP Dialog sẽ không khả dụng.")

try:
    from quangtps.optimization.kbp.model import KBPModel
    from quangtps.optimization.kbp.predictor import KBPPredictor, KBPRecommendation
    from quangtps.optimization.kbp.trainer import KBPTrainer
except ImportError:
    print("Không thể import module KBP. Tính năng KBP Dialog sẽ không khả dụng.")

from quangtps.utils.ui_utils import create_eclipse_icon

try:
    from quangtps.ui.utils.ui_utils import set_eclipse_style, get_icon_path
except ImportError:
    pass

logger = logging.getLogger(__name__)


class KBPModelInfoWidget(QWidget):
    """Widget hiển thị thông tin về mô hình KBP."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Tiêu đề
        title_label = QLabel("Thông tin mô hình")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Form thông tin
        form_layout = QFormLayout()

        self.site_label = QLabel("N/A")
        form_layout.addRow("Vị trí điều trị:", self.site_label)

        self.model_type_label = QLabel("N/A")
        form_layout.addRow("Loại mô hình:", self.model_type_label)

        self.plans_count_label = QLabel("N/A")
        form_layout.addRow("Số lượng kế hoạch huấn luyện:", self.plans_count_label)

        self.accuracy_label = QLabel("N/A")
        form_layout.addRow("Độ chính xác trung bình:", self.accuracy_label)

        self.last_updated_label = QLabel("N/A")
        form_layout.addRow("Cập nhật lần cuối:", self.last_updated_label)

        layout.addLayout(form_layout)

        # Thông tin đặc trưng quan trọng
        self.features_table = QTableWidget(0, 2)
        self.features_table.setHorizontalHeaderLabels(["Đặc trưng", "Tầm quan trọng"])
        self.features_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.features_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        layout.addWidget(QLabel("Đặc trưng quan trọng:"))
        layout.addWidget(self.features_table)

        layout.setContentsMargins(10, 10, 10, 10)

    def update_model_info(self, model_info: Dict[str, Any]):
        """Cập nhật thông tin mô hình."""
        self.site_label.setText(model_info.get("site", "N/A"))
        self.model_type_label.setText(model_info.get("model_type", "N/A"))
        self.plans_count_label.setText(str(model_info.get("plans_count", "N/A")))
        self.accuracy_label.setText(f"{model_info.get('accuracy', 0):.2f}%")
        self.last_updated_label.setText(model_info.get("last_updated", "N/A"))

        # Cập nhật bảng đặc trưng quan trọng
        features = model_info.get("important_features", {})
        self.features_table.setRowCount(len(features))

        for i, (feature, importance) in enumerate(features.items()):
            self.features_table.setItem(i, 0, QTableWidgetItem(feature))
            self.features_table.setItem(i, 1, QTableWidgetItem(f"{importance:.3f}"))


class KBPObjectivesTable(QTableWidget):
    """Bảng hiển thị các mục tiêu tối ưu từ KBP."""

    objectiveChanged = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self._init_ui()

    def _init_ui(self):
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "Loại", "Tham số", "Giá trị", "Trọng số"]
        )

        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

    def update_objectives(
        self, objectives: Dict[str, Dict[str, Any]], weights: Dict[str, float]
    ):
        """Cập nhật các mục tiêu tối ưu."""
        self.setRowCount(0)

        row = 0
        for structure, structure_objectives in objectives.items():
            for obj_type, obj_params in structure_objectives.items():
                self.insertRow(row)

                # Thêm thông tin mục tiêu
                self.setItem(row, 0, QTableWidgetItem(structure))
                self.setItem(row, 1, QTableWidgetItem(obj_type))

                # Thêm các tham số dưới dạng chuỗi
                param_str = ", ".join([f"{k}: {v}" for k, v in obj_params.items()])
                self.setItem(row, 2, QTableWidgetItem(param_str))

                # Thêm giá trị chính (ví dụ: dose hoặc volume)
                if "dose" in obj_params:
                    value = f"{obj_params['dose']:.2f} Gy"
                elif "volume" in obj_params:
                    value = f"{obj_params['volume']:.2f} %"
                else:
                    value = "N/A"
                self.setItem(row, 3, QTableWidgetItem(value))

                # Thêm trọng số
                weight = weights.get(f"{structure}_{obj_type}", 1.0)

                # Tạo spin box cho trọng số
                weight_spin = QDoubleSpinBox()
                weight_spin.setMinimum(0.1)
                weight_spin.setMaximum(100.0)
                weight_spin.setSingleStep(0.1)
                weight_spin.setValue(weight)
                weight_spin.valueChanged.connect(
                    lambda value, r=row: self._weight_changed(r, value)
                )

                self.setCellWidget(row, 4, weight_spin)

                row += 1

    def _weight_changed(self, row: int, value: float):
        """Xử lý khi trọng số thay đổi."""
        structure = self.item(row, 0).text()
        obj_type = self.item(row, 1).text()

        objective = {"structure": structure, "type": obj_type, "weight": value}

        self.objectiveChanged.emit(row, objective)


class KBPConstraintsTable(QTableWidget):
    """Bảng hiển thị các ràng buộc từ KBP."""

    constraintChanged = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self._init_ui()

    def _init_ui(self):
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "Loại ràng buộc", "Giá trị", "Ưu tiên"]
        )

        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

    def update_constraints(self, constraints: Dict[str, Dict[str, Any]]):
        """Cập nhật các ràng buộc."""
        self.setRowCount(0)

        row = 0
        for structure, structure_constraints in constraints.items():
            for constraint_type, constraint_params in structure_constraints.items():
                self.insertRow(row)

                # Thêm thông tin ràng buộc
                self.setItem(row, 0, QTableWidgetItem(structure))
                self.setItem(row, 1, QTableWidgetItem(constraint_type))

                # Thêm giá trị chính
                if "dose" in constraint_params:
                    value = f"{constraint_params['dose']:.2f} Gy"
                elif "volume" in constraint_params:
                    value = f"{constraint_params['volume']:.2f} %"
                elif "max" in constraint_params:
                    value = f"Tối đa {constraint_params['max']:.2f} Gy"
                else:
                    value = "N/A"
                self.setItem(row, 2, QTableWidgetItem(value))

                # Thêm combo box cho ưu tiên
                priority_combo = QComboBox()
                priority_combo.addItems(["Thấp", "Trung bình", "Cao", "Rất cao"])

                priority = constraint_params.get("priority", "Trung bình")
                priority_index = {
                    "Thấp": 0,
                    "Trung bình": 1,
                    "Cao": 2,
                    "Rất cao": 3,
                }.get(priority, 1)
                priority_combo.setCurrentIndex(priority_index)

                priority_combo.currentIndexChanged.connect(
                    lambda idx, r=row: self._priority_changed(r, idx)
                )

                self.setCellWidget(row, 3, priority_combo)

                row += 1

    def _priority_changed(self, row: int, index: int):
        """Xử lý khi ưu tiên thay đổi."""
        structure = self.item(row, 0).text()
        constraint_type = self.item(row, 1).text()

        priorities = ["Thấp", "Trung bình", "Cao", "Rất cao"]
        priority = priorities[index]

        constraint = {
            "structure": structure,
            "type": constraint_type,
            "priority": priority,
        }

        self.constraintChanged.emit(row, constraint)


class KBPDialog(QDialog):
    """Dialog Knowledge-Based Planning theo phong cách Eclipse."""

    # Tín hiệu khi áp dụng đề xuất KBP
    kbpRecommendationApplied = pyqtSignal(object)

    def __init__(
        self, patient_id: str, structure_set_id: str, site: str = "", parent=None
    ):
        super().__init__(parent)

        self.patient_id = patient_id
        self.structure_set_id = structure_set_id
        self.site = site

        self.predictor = KBPPredictor()
        self.available_sites = self._get_available_sites()
        self.current_recommendation = None

        self._init_ui()
        set_eclipse_style(self)

    def _init_ui(self):
        self.setWindowTitle("Knowledge-Based Planning")
        self.setMinimumSize(900, 600)

        main_layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        icon_label = QLabel()
        icon_label.setPixmap(create_eclipse_icon("kbp").pixmap(QSize(32, 32)))
        header_layout.addWidget(icon_label)

        title_label = QLabel("Knowledge-Based Planning")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Combo box chọn vị trí điều trị
        site_layout = QHBoxLayout()
        site_layout.addWidget(QLabel("Vị trí điều trị:"))
        self.site_combo = QComboBox()
        self.site_combo.addItems(self.available_sites)

        if self.site and self.site in self.available_sites:
            self.site_combo.setCurrentText(self.site)

        self.site_combo.currentTextChanged.connect(self._site_changed)
        site_layout.addWidget(self.site_combo)

        header_layout.addLayout(site_layout)

        main_layout.addLayout(header_layout)

        # Splitter chính
        main_splitter = QSplitter(Qt.Horizontal)

        # Panel trái - thông tin mô hình
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.model_info_widget = KBPModelInfoWidget()
        left_layout.addWidget(self.model_info_widget)

        # Nút tạo đề xuất
        generate_button = QPushButton("Tạo đề xuất KBP")
        generate_button.setIcon(create_eclipse_icon("generate"))
        generate_button.clicked.connect(self._generate_recommendation)
        left_layout.addWidget(generate_button)

        # Thêm nút xem phân tích
        analysis_button = QPushButton("Xem phân tích kế hoạch trước")
        analysis_button.setIcon(create_eclipse_icon("analysis"))
        analysis_button.clicked.connect(self._show_plan_analysis)
        left_layout.addWidget(analysis_button)

        main_splitter.addWidget(left_panel)

        # Panel phải - kết quả đề xuất
        right_panel = QTabWidget()

        # Tab mục tiêu tối ưu
        objectives_tab = QWidget()
        objectives_layout = QVBoxLayout(objectives_tab)

        self.objectives_table = KBPObjectivesTable()
        self.objectives_table.objectiveChanged.connect(self._objective_changed)
        objectives_layout.addWidget(self.objectives_table)

        right_panel.addTab(objectives_tab, "Mục tiêu tối ưu")

        # Tab ràng buộc liều
        constraints_tab = QWidget()
        constraints_layout = QVBoxLayout(constraints_tab)

        self.constraints_table = KBPConstraintsTable()
        self.constraints_table.constraintChanged.connect(self._constraint_changed)
        constraints_layout.addWidget(self.constraints_table)

        right_panel.addTab(constraints_tab, "Ràng buộc liều")

        main_splitter.addWidget(right_panel)

        # Set tỷ lệ ban đầu cho splitter
        main_splitter.setSizes([300, 600])

        main_layout.addWidget(main_splitter)

        # Các nút điều khiển
        buttons_layout = QHBoxLayout()

        info_label = QLabel("Đề xuất dựa trên các kế hoạch tương tự")
        info_label.setStyleSheet("color: #666;")
        buttons_layout.addWidget(info_label)

        buttons_layout.addStretch()

        apply_button = QPushButton("Áp dụng đề xuất")
        apply_button.setIcon(create_eclipse_icon("apply"))
        apply_button.clicked.connect(self._apply_recommendation)
        buttons_layout.addWidget(apply_button)

        cancel_button = QPushButton("Hủy")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)

        main_layout.addLayout(buttons_layout)

        # Thanh tiến trình
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

    def _get_available_sites(self) -> List[str]:
        """Lấy danh sách các vị trí điều trị có sẵn."""
        try:
            # Đường dẫn thư mục chứa mô hình KBP
            models_dir = "models/kbp"

            if not os.path.exists(models_dir):
                return [
                    "Prostate",
                    "Head and Neck",
                    "Breast",
                    "Lung",
                    "Brain",
                ]  # Mặc định

            # Lấy các thư mục con (mỗi thư mục là một vị trí điều trị)
            sites = [
                d
                for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d))
            ]

            # Chuẩn hóa tên vị trí (viết hoa chữ cái đầu)
            sites = [site.replace("_", " ").title() for site in sites]

            if not sites:
                return [
                    "Prostate",
                    "Head and Neck",
                    "Breast",
                    "Lung",
                    "Brain",
                ]  # Mặc định

            return sites

        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách vị trí điều trị: {str(e)}")
            return ["Prostate", "Head and Neck", "Breast", "Lung", "Brain"]  # Mặc định

    def _get_model_info(self, site: str) -> Dict[str, Any]:
        """Lấy thông tin về mô hình KBP cho một vị trí điều trị."""
        # Trong thực tế, bạn sẽ tải thông tin từ model metadata
        # Đây là dữ liệu mẫu
        return {
            "site": site,
            "model_type": "Gradient Boosting",
            "plans_count": 150,
            "accuracy": 92.5,
            "last_updated": "15/10/2023",
            "important_features": {
                "ptv_volume": 0.423,
                "ptv_surface_area": 0.385,
                "distance_to_rectum": 0.356,
                "distance_to_bladder": 0.342,
                "overlap_with_rectum": 0.312,
                "patient_age": 0.221,
                "prescription_dose": 0.187,
            },
        }

    def _site_changed(self, site: str):
        """Xử lý khi vị trí điều trị thay đổi."""
        self.site = site

        # Cập nhật thông tin mô hình
        model_info = self._get_model_info(site)
        self.model_info_widget.update_model_info(model_info)

    def _generate_recommendation(self):
        """Tạo đề xuất KBP cho kế hoạch hiện tại."""
        site = self.site_combo.currentText()

        if not site:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn vị trí điều trị")
            return

        # Hiển thị thanh tiến trình
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Trong thực tế, bạn sẽ gọi KBPPredictor để tạo đề xuất
        # Đây là dữ liệu mẫu

        # Giả lập tiến trình
        for i in range(1, 11):
            QApplication.processEvents()
            self.progress_bar.setValue(i * 10)
            import time

            time.sleep(0.1)

        # Tạo đề xuất mẫu
        self.current_recommendation = {
            "patient_id": self.patient_id,
            "structure_set_id": self.structure_set_id,
            "dose_constraints": {
                "Rectum": {
                    "Max Dose": {"dose": 50.0, "priority": "Cao"},
                    "V75": {"volume": 15.0, "priority": "Cao"},
                    "Mean Dose": {"dose": 30.0, "priority": "Trung bình"},
                },
                "Bladder": {
                    "Max Dose": {"dose": 55.0, "priority": "Cao"},
                    "V80": {"volume": 10.0, "priority": "Cao"},
                    "Mean Dose": {"dose": 25.0, "priority": "Trung bình"},
                },
                "Femur L": {"Max Dose": {"dose": 40.0, "priority": "Trung bình"}},
                "Femur R": {"Max Dose": {"dose": 40.0, "priority": "Trung bình"}},
            },
            "objectives": {
                "PTV": {
                    "Min Dose": {"dose": 75.0, "volume": 99.0},
                    "Max Dose": {"dose": 82.0, "volume": 1.0},
                    "Uniform Dose": {"dose": 78.0},
                },
                "Rectum": {
                    "Max DVH": {"dose": 70.0, "volume": 20.0},
                    "Max DVH": {"dose": 50.0, "volume": 50.0},
                },
                "Bladder": {
                    "Max DVH": {"dose": 75.0, "volume": 15.0},
                    "Max DVH": {"dose": 55.0, "volume": 40.0},
                },
                "Femur L": {"Max Dose": {"dose": 45.0}},
                "Femur R": {"Max Dose": {"dose": 45.0}},
            },
            "weights": {
                "PTV_Min Dose": 10.0,
                "PTV_Max Dose": 8.0,
                "PTV_Uniform Dose": 5.0,
                "Rectum_Max DVH": 5.0,
                "Bladder_Max DVH": 4.0,
                "Femur L_Max Dose": 2.0,
                "Femur R_Max Dose": 2.0,
            },
            "confidence": {
                "overall": 0.92,
                "PTV": 0.95,
                "Rectum": 0.90,
                "Bladder": 0.88,
                "Femur L": 0.85,
                "Femur R": 0.85,
            },
            "structures_used": {
                "targets": ["PTV"],
                "oars": ["Rectum", "Bladder", "Femur L", "Femur R"],
            },
        }

        # Cập nhật bảng mục tiêu và ràng buộc
        self.objectives_table.update_objectives(
            self.current_recommendation["objectives"],
            self.current_recommendation["weights"],
        )
        self.constraints_table.update_constraints(
            self.current_recommendation["dose_constraints"]
        )

        # Ẩn thanh tiến trình
        self.progress_bar.setVisible(False)

        # Hiển thị thông báo thành công
        QMessageBox.information(
            self,
            "Thành công",
            f"Đã tạo đề xuất KBP cho kế hoạch với độ tin cậy {self.current_recommendation['confidence']['overall']:.2f}",
        )

    def _apply_recommendation(self):
        """Áp dụng đề xuất KBP vào kế hoạch."""
        if not self.current_recommendation:
            QMessageBox.warning(self, "Lỗi", "Chưa có đề xuất KBP nào được tạo")
            return

        # Phát tín hiệu với đề xuất KBP
        self.kbpRecommendationApplied.emit(self.current_recommendation)

        QMessageBox.information(
            self, "Thành công", "Đã áp dụng đề xuất KBP vào kế hoạch"
        )
        self.accept()

    def _show_plan_analysis(self):
        """Hiển thị phân tích các kế hoạch trước đó."""
        QMessageBox.information(
            self,
            "Phân tích kế hoạch",
            "Tính năng này sẽ hiển thị phân tích các kế hoạch trước đây tương tự với kế hoạch hiện tại.",
        )

    def _objective_changed(self, row: int, objective: Dict[str, Any]):
        """Cập nhật mục tiêu tối ưu khi người dùng thay đổi."""
        if not self.current_recommendation:
            return

        # Cập nhật trọng số
        key = f"{objective['structure']}_{objective['type']}"
        self.current_recommendation["weights"][key] = objective["weight"]

    def _constraint_changed(self, row: int, constraint: Dict[str, Any]):
        """Cập nhật ràng buộc khi người dùng thay đổi."""
        if not self.current_recommendation:
            return

        # Cập nhật ưu tiên
        structure = constraint["structure"]
        constraint_type = constraint["type"]

        if (
            structure in self.current_recommendation["dose_constraints"]
            and constraint_type
            in self.current_recommendation["dose_constraints"][structure]
        ):
            self.current_recommendation["dose_constraints"][structure][constraint_type][
                "priority"
            ] = constraint["priority"]


# Đoạn mã kiểm thử chạy độc lập
if __name__ == "__main__":
    app = QApplication(sys.argv)

    dialog = KBPDialog("PATIENT001", "STRUCTURE001", "Prostate")
    dialog.show()

    sys.exit(app.exec_())
