#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog tối ưu hóa dựa trên kiến thức (Knowledge-Based Planning) cho QuangTPS.

Dialog này cho phép người dùng áp dụng các mô hình KBP vào quá trình lập kế hoạch,
xem các đề xuất, và điều chỉnh các tham số trước khi áp dụng vào kế hoạch.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                            QPushButton, QGroupBox, QSplitter, QTabWidget, 
                            QWidget, QComboBox, QFrame, QRadioButton, 
                            QButtonGroup, QMessageBox, QGridLayout, QScrollArea,
                            QSpacerItem, QSizePolicy, QCheckBox, QFileDialog,
                            QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox,
                            QProgressBar, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from quangtps.optimization.kbp.predictor import KBPPredictor, KBPRecommendation
from quangtps.optimization.objectives import ObjectiveCollection, create_objective
from quangtps.optimization.constraints import ConstraintCollection, create_constraint
from quangtps.core.exceptions import ModelError, PredictionError
from quangtps.evaluation.dvh import DVHCalculator
from quangtps.common.widgets import create_info_label

logger = logging.getLogger(__name__)

class ConstraintTable(QTableWidget):
    """Bảng hiển thị các ràng buộc liều được đề xuất."""
    
    constraintChanged = pyqtSignal(str, str, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Thiết lập bảng
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Cấu trúc", "Loại ràng buộc", "Giá trị", "Sử dụng"])
        
        # Thiết lập cách hiển thị
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.cellChanged.connect(self._handle_cell_changed)
    
    def populate_constraints(self, constraints: Dict[str, Dict[str, Any]]):
        """Đổ dữ liệu ràng buộc vào bảng."""
        self.blockSignals(True)
        self.setRowCount(0)
        
        row = 0
        for structure, struct_constraints in constraints.items():
            for constraint_type, value in struct_constraints.items():
                self.insertRow(row)
                
                # Cột cấu trúc
                self.setItem(row, 0, QTableWidgetItem(structure))
                
                # Cột loại ràng buộc
                self.setItem(row, 1, QTableWidgetItem(constraint_type))
                
                # Cột giá trị
                value_item = QTableWidgetItem(str(round(value, 2)))
                value_item.setData(Qt.UserRole, value)
                self.setItem(row, 2, value_item)
                
                # Cột sử dụng
                checkbox = QCheckBox()
                checkbox.setChecked(True)
                self.setCellWidget(row, 3, checkbox)
                
                row += 1
        
        self.blockSignals(False)
    
    def get_selected_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Lấy các ràng buộc đã chọn."""
        constraints = {}
        
        for row in range(self.rowCount()):
            # Kiểm tra xem ràng buộc có được chọn không
            checkbox = self.cellWidget(row, 3)
            if not checkbox.isChecked():
                continue
            
            structure = self.item(row, 0).text()
            constraint_type = self.item(row, 1).text()
            value = self.item(row, 2).data(Qt.UserRole)
            
            if structure not in constraints:
                constraints[structure] = {}
            
            constraints[structure][constraint_type] = value
        
        return constraints
    
    def _handle_cell_changed(self, row, column):
        """Xử lý khi giá trị ô thay đổi."""
        if column == 2:  # Cột giá trị
            structure = self.item(row, 0).text()
            constraint_type = self.item(row, 1).text()
            value_item = self.item(row, 2)
            
            try:
                value = float(value_item.text())
                value_item.setData(Qt.UserRole, value)
                self.constraintChanged.emit(structure, constraint_type, value)
            except ValueError:
                # Khôi phục giá trị cũ
                old_value = value_item.data(Qt.UserRole)
                value_item.setText(str(round(old_value, 2)))
                QMessageBox.warning(self, "Lỗi", "Giá trị phải là số thực")


class ObjectiveTable(QTableWidget):
    """Bảng hiển thị các mục tiêu tối ưu được đề xuất."""
    
    objectiveChanged = pyqtSignal(str, str, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Thiết lập bảng
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Cấu trúc", "Loại mục tiêu", "Giá trị", "Trọng số", "Sử dụng"])
        
        # Thiết lập cách hiển thị
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.cellChanged.connect(self._handle_cell_changed)
    
    def populate_objectives(
        self, 
        objectives: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ):
        """Đổ dữ liệu mục tiêu vào bảng."""
        self.blockSignals(True)
        self.setRowCount(0)
        
        row = 0
        for structure, struct_objectives in objectives.items():
            for objective_type, value in struct_objectives.items():
                self.insertRow(row)
                
                # Cột cấu trúc
                self.setItem(row, 0, QTableWidgetItem(structure))
                
                # Cột loại mục tiêu
                self.setItem(row, 1, QTableWidgetItem(objective_type))
                
                # Cột giá trị
                value_item = QTableWidgetItem(str(round(value, 2)))
                value_item.setData(Qt.UserRole, value)
                self.setItem(row, 2, value_item)
                
                # Cột trọng số
                weight_key = f"weight_{structure}_{objective_type}"
                weight = weights.get(weight_key, 1.0)
                
                weight_spin = QDoubleSpinBox()
                weight_spin.setRange(0.1, 10.0)
                weight_spin.setSingleStep(0.1)
                weight_spin.setValue(weight)
                weight_spin.valueChanged.connect(
                    lambda w, r=row: self._handle_weight_changed(r, w)
                )
                self.setCellWidget(row, 3, weight_spin)
                
                # Cột sử dụng
                checkbox = QCheckBox()
                checkbox.setChecked(True)
                self.setCellWidget(row, 4, checkbox)
                
                row += 1
        
        self.blockSignals(False)
    
    def get_selected_objectives(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """Lấy các mục tiêu và trọng số đã chọn."""
        objectives = {}
        weights = {}
        
        for row in range(self.rowCount()):
            # Kiểm tra xem mục tiêu có được chọn không
            checkbox = self.cellWidget(row, 4)
            if not checkbox.isChecked():
                continue
            
            structure = self.item(row, 0).text()
            objective_type = self.item(row, 1).text()
            value = self.item(row, 2).data(Qt.UserRole)
            
            # Lấy trọng số
            weight_spin = self.cellWidget(row, 3)
            weight = weight_spin.value()
            
            if structure not in objectives:
                objectives[structure] = {}
            
            objectives[structure][objective_type] = value
            
            # Lưu trọng số
            weight_key = f"weight_{structure}_{objective_type}"
            weights[weight_key] = weight
        
        return objectives, weights
    
    def _handle_cell_changed(self, row, column):
        """Xử lý khi giá trị ô thay đổi."""
        if column == 2:  # Cột giá trị
            structure = self.item(row, 0).text()
            objective_type = self.item(row, 1).text()
            value_item = self.item(row, 2)
            
            try:
                value = float(value_item.text())
                value_item.setData(Qt.UserRole, value)
                
                # Lấy trọng số
                weight_spin = self.cellWidget(row, 3)
                weight = weight_spin.value()
                
                self.objectiveChanged.emit(structure, objective_type, value, weight)
            except ValueError:
                # Khôi phục giá trị cũ
                old_value = value_item.data(Qt.UserRole)
                value_item.setText(str(round(old_value, 2)))
                QMessageBox.warning(self, "Lỗi", "Giá trị phải là số thực")
    
    def _handle_weight_changed(self, row, value):
        """Xử lý khi trọng số thay đổi."""
        structure = self.item(row, 0).text()
        objective_type = self.item(row, 1).text()
        val_item = self.item(row, 2)
        obj_value = val_item.data(Qt.UserRole)
        
        self.objectiveChanged.emit(structure, objective_type, obj_value, value)


class KBPDialog(QDialog):
    """
    Dialog tối ưu hóa dựa trên kiến thức (KBP).
    
    Dialog này cho phép người dùng áp dụng các mô hình KBP vào kế hoạch điều trị,
    xem các đề xuất, chỉnh sửa các tham số, và áp dụng vào kế hoạch.
    """
    
    # Tín hiệu khi áp dụng đề xuất
    recommendationApplied = pyqtSignal(KBPRecommendation, ObjectiveCollection, ConstraintCollection)
    
    def __init__(
        self, 
        patient_id: str, 
        structure_set_id: str, 
        prescription_dose: float,
        parent=None
    ):
        """
        Khởi tạo dialog KBP.
        
        Args:
            patient_id: ID bệnh nhân
            structure_set_id: ID tập cấu trúc
            prescription_dose: Liều kê đơn (Gy)
            parent: Widget cha
        """
        super().__init__(parent)
        
        self.setWindowTitle("Tối ưu hóa dựa trên kiến thức (KBP)")
        self.setMinimumSize(900, 700)
        
        self.patient_id = patient_id
        self.structure_set_id = structure_set_id
        self.prescription_dose = prescription_dose
        
        self.kbp_predictor = KBPPredictor()
        self.recommendation = None
        
        self.init_ui()
    
    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Phần trên: thông tin và nút điều khiển
        top_layout = QHBoxLayout()
        
        # Thông tin bệnh nhân
        info_group = QGroupBox("Thông tin")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("ID bệnh nhân:"), 0, 0)
        info_layout.addWidget(QLabel(self.patient_id), 0, 1)
        
        info_layout.addWidget(QLabel("ID tập cấu trúc:"), 1, 0)
        info_layout.addWidget(QLabel(self.structure_set_id), 1, 1)
        
        info_layout.addWidget(QLabel("Liều kê đơn:"), 2, 0)
        info_layout.addWidget(QLabel(f"{self.prescription_dose} Gy"), 2, 1)
        
        top_layout.addWidget(info_group)
        
        # Chọn vị trí điều trị
        site_group = QGroupBox("Vị trí điều trị")
        site_layout = QVBoxLayout(site_group)
        
        self.site_combo = QComboBox()
        self.site_combo.addItems(["Prostate", "H&N", "Lung", "Breast", "Brain", "GI"])
        site_layout.addWidget(self.site_combo)
        
        # Nút tạo đề xuất
        self.generate_button = QPushButton("Tạo đề xuất KBP")
        self.generate_button.clicked.connect(self.generate_recommendation)
        site_layout.addWidget(self.generate_button)
        
        # Nút lưu đề xuất
        self.save_button = QPushButton("Lưu đề xuất")
        self.save_button.clicked.connect(self.save_recommendation)
        self.save_button.setEnabled(False)
        site_layout.addWidget(self.save_button)
        
        # Nút tải đề xuất
        self.load_button = QPushButton("Tải đề xuất")
        self.load_button.clicked.connect(self.load_recommendation)
        site_layout.addWidget(self.load_button)
        
        top_layout.addWidget(site_group)
        
        # Thêm layout trên vào layout chính
        main_layout.addLayout(top_layout)
        
        # Phần giữa: Tab widget cho các đề xuất
        tab_widget = QTabWidget()
        
        # Tab ràng buộc
        constraints_tab = QWidget()
        constraints_layout = QVBoxLayout(constraints_tab)
        
        # Thêm label thông tin
        constraints_info = create_info_label(
            "Bảng dưới đây hiển thị các ràng buộc liều được đề xuất cho các cơ quan nguy cấp."
            " Bạn có thể điều chỉnh giá trị và chọn ràng buộc muốn sử dụng."
        )
        constraints_layout.addWidget(constraints_info)
        
        # Tạo và thêm bảng ràng buộc
        self.constraints_table = ConstraintTable()
        constraints_layout.addWidget(self.constraints_table)
        
        tab_widget.addTab(constraints_tab, "Ràng buộc liều")
        
        # Tab mục tiêu
        objectives_tab = QWidget()
        objectives_layout = QVBoxLayout(objectives_tab)
        
        # Thêm label thông tin
        objectives_info = create_info_label(
            "Bảng dưới đây hiển thị các mục tiêu tối ưu được đề xuất."
            " Bạn có thể điều chỉnh giá trị, trọng số và chọn mục tiêu muốn sử dụng."
        )
        objectives_layout.addWidget(objectives_info)
        
        # Tạo và thêm bảng mục tiêu
        self.objectives_table = ObjectiveTable()
        objectives_layout.addWidget(self.objectives_table)
        
        tab_widget.addTab(objectives_tab, "Mục tiêu tối ưu")
        
        # Tab thông tin
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        # Thêm label thông tin
        model_info = create_info_label(
            "Thông tin về mô hình KBP và mức độ tin cậy của các đề xuất."
        )
        info_layout.addWidget(model_info)
        
        # Tạo và thêm bảng thông tin
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Tham số", "Giá trị"])
        self.info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.info_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.info_table.verticalHeader().setVisible(False)
        
        info_layout.addWidget(self.info_table)
        
        tab_widget.addTab(info_tab, "Thông tin")
        
        # Thêm tab widget vào layout chính
        main_layout.addWidget(tab_widget)
        
        # Phần dưới: Nút điều khiển
        buttons_layout = QHBoxLayout()
        
        # Thêm spacer bên trái
        buttons_layout.addStretch()
        
        # Nút áp dụng
        self.apply_button = QPushButton("Áp dụng vào kế hoạch")
        self.apply_button.clicked.connect(self.apply_recommendation)
        self.apply_button.setEnabled(False)
        buttons_layout.addWidget(self.apply_button)
        
        # Nút hủy
        cancel_button = QPushButton("Hủy")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        
        # Thêm layout nút vào layout chính
        main_layout.addLayout(buttons_layout)
    
    def generate_recommendation(self):
        """Tạo đề xuất KBP cho kế hoạch hiện tại."""
        site = self.site_combo.currentText()
        
        try:
            # Disable nút để tránh nhiều lần nhấn
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Đang tạo đề xuất...")
            self.setCursor(Qt.WaitCursor)
            
            # Tạo đề xuất
            self.recommendation = self.kbp_predictor.generate_recommendation(
                self.patient_id, self.structure_set_id, site
            )
            
            # Hiển thị đề xuất
            self.display_recommendation()
            
            # Enable các nút
            self.save_button.setEnabled(True)
            self.apply_button.setEnabled(True)
            
            QMessageBox.information(self, "Thành công", "Đã tạo đề xuất KBP thành công!")
            
        except (ModelError, PredictionError) as e:
            QMessageBox.critical(self, "Lỗi", str(e))
        finally:
            # Khôi phục trạng thái nút
            self.generate_button.setEnabled(True)
            self.generate_button.setText("Tạo đề xuất KBP")
            self.setCursor(Qt.ArrowCursor)
    
    def display_recommendation(self):
        """Hiển thị đề xuất KBP trên giao diện."""
        if not self.recommendation:
            return
        
        # Hiển thị ràng buộc liều
        self.constraints_table.populate_constraints(self.recommendation.dose_constraints)
        
        # Hiển thị mục tiêu tối ưu
        self.objectives_table.populate_objectives(
            self.recommendation.objectives, 
            self.recommendation.weights
        )
        
        # Hiển thị thông tin
        self.populate_info_table()
    
    def populate_info_table(self):
        """Đổ dữ liệu vào bảng thông tin."""
        self.info_table.setRowCount(0)
        
        # Thêm thông tin cơ bản
        self._add_info_row("ID bệnh nhân", self.recommendation.patient_id)
        self._add_info_row("ID tập cấu trúc", self.recommendation.structure_set_id)
        self._add_info_row("Liều kê đơn", f"{self.prescription_dose} Gy")
        
        # Thêm thông tin cấu trúc
        targets = self.recommendation.structures_used.get("targets", [])
        oars = self.recommendation.structures_used.get("oars", [])
        
        self._add_info_row("Cấu trúc mục tiêu", ", ".join(targets))
        self._add_info_row("Cơ quan nguy cấp", ", ".join(oars))
        
        # Thêm thông tin độ tin cậy
        confidence_avg = np.mean(list(self.recommendation.confidence.values())) if self.recommendation.confidence else 0
        self._add_info_row("Độ tin cậy trung bình", f"{confidence_avg:.2f}")
        
        # Thêm thông tin về đề xuất
        num_constraints = sum(len(c) for c in self.recommendation.dose_constraints.values())
        num_objectives = sum(len(o) for o in self.recommendation.objectives.values())
        
        self._add_info_row("Số ràng buộc liều", str(num_constraints))
        self._add_info_row("Số mục tiêu tối ưu", str(num_objectives))
    
    def _add_info_row(self, param, value):
        """Thêm một hàng vào bảng thông tin."""
        row = self.info_table.rowCount()
        self.info_table.insertRow(row)
        
        # Thêm tham số
        param_item = QTableWidgetItem(param)
        param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)
        self.info_table.setItem(row, 0, param_item)
        
        # Thêm giá trị
        value_item = QTableWidgetItem(str(value))
        value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
        self.info_table.setItem(row, 1, value_item)
    
    def save_recommendation(self):
        """Lưu đề xuất KBP vào file."""
        if not self.recommendation:
            QMessageBox.warning(self, "Cảnh báo", "Không có đề xuất nào để lưu!")
            return
        
        # Hiển thị dialog chọn file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu đề xuất KBP", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            # Lưu đề xuất
            self.recommendation.save(file_path)
            QMessageBox.information(self, "Thành công", f"Đã lưu đề xuất vào {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu đề xuất: {str(e)}")
    
    def load_recommendation(self):
        """Tải đề xuất KBP từ file."""
        # Hiển thị dialog chọn file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Tải đề xuất KBP", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            # Tải đề xuất
            self.recommendation = KBPRecommendation.load(file_path)
            
            # Kiểm tra xem đề xuất có phù hợp với bệnh nhân hiện tại không
            if (self.recommendation.patient_id != self.patient_id or 
                self.recommendation.structure_set_id != self.structure_set_id):
                
                reply = QMessageBox.question(
                    self, "Cảnh báo", 
                    "Đề xuất này được tạo cho bệnh nhân khác. Bạn vẫn muốn tiếp tục?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return
            
            # Hiển thị đề xuất
            self.display_recommendation()
            
            # Enable các nút
            self.save_button.setEnabled(True)
            self.apply_button.setEnabled(True)
            
            QMessageBox.information(self, "Thành công", "Đã tải đề xuất KBP thành công!")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải đề xuất: {str(e)}")
    
    def apply_recommendation(self):
        """Áp dụng đề xuất KBP vào kế hoạch."""
        if not self.recommendation:
            QMessageBox.warning(self, "Cảnh báo", "Không có đề xuất nào để áp dụng!")
            return
        
        # Lấy các ràng buộc và mục tiêu đã chọn
        selected_constraints = self.constraints_table.get_selected_constraints()
        selected_objectives, selected_weights = self.objectives_table.get_selected_objectives()
        
        # Cập nhật đề xuất
        updated_recommendation = KBPRecommendation(
            patient_id=self.recommendation.patient_id,
            structure_set_id=self.recommendation.structure_set_id,
            dose_constraints=selected_constraints,
            objectives=selected_objectives,
            weights=selected_weights,
            confidence=self.recommendation.confidence,
            structures_used=self.recommendation.structures_used
        )
        
        # Tạo tập hợp mục tiêu và ràng buộc
        objectives = self.kbp_predictor.create_objective_collection(
            updated_recommendation, self.prescription_dose
        )
        
        constraints = self.kbp_predictor.create_constraint_collection(
            updated_recommendation, self.prescription_dose
        )
        
        # Phát tín hiệu
        self.recommendationApplied.emit(updated_recommendation, objectives, constraints)
        
        # Đóng dialog
        self.accept() 