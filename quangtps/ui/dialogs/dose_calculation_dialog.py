#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module dialog tính toán liều xạ trị.

Module này cung cấp giao diện người dùng để cấu hình và thực hiện tính toán liều xạ trị.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                           QComboBox, QGroupBox, QRadioButton, QProgressBar, 
                           QLineEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from quangtps.core.types import DoseCalculationAlgorithm
from quangtps.dose.calculation import DoseCalculator
from quangtps.treatment.beams.beam_data_processor import BeamDataProcessor
from quangtps.ui.dialogs.beam_data_input_dialog import BeamDataInputDialog
from quangtps.ui.widgets.helpers import center_dialog, set_icon

logger = logging.getLogger(__name__)

class DoseCalculationDialog(QDialog):
    """
    Dialog tính toán liều xạ trị.
    
    Dialog này cho phép người dùng chọn thuật toán tính toán liều, cấu hình các
    tham số cần thiết, và thực hiện tính toán liều cho các kế hoạch điều trị.
    """
    
    def __init__(self, parent=None, plan_data=None):
        """
        Khởi tạo dialog tính toán liều.
        
        Args:
            parent: Widget cha của dialog.
            plan_data: Dữ liệu kế hoạch điều trị (nếu có).
        """
        super().__init__(parent)
        
        self.setWindowTitle("Tính toán liều xạ trị")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        set_icon(self)
        
        # Lưu dữ liệu kế hoạch
        self.plan_data = plan_data
        
        # Tạo UI
        self._init_ui()
        
        # Căn giữa dialog
        center_dialog(self)
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Phần chọn mô hình chùm tia và thuật toán
        beam_model_group = QGroupBox("Mô hình chùm tia")
        beam_model_layout = QVBoxLayout()
        beam_model_group.setLayout(beam_model_layout)
        
        # Combobox chọn mô hình chùm tia
        beam_model_layout.addWidget(QLabel("Mô hình chùm tia:"))
        self.beam_model_combo = QComboBox()
        self._load_beam_models()  # Tải danh sách mô hình
        beam_model_layout.addWidget(self.beam_model_combo)
        
        # Nút nhập dữ liệu chùm tia mới
        import_beam_data_button = QPushButton("Nhập dữ liệu chùm tia mới...")
        import_beam_data_button.clicked.connect(self._show_beam_data_input_dialog)
        beam_model_layout.addWidget(import_beam_data_button)
        
        # Phần chọn thuật toán
        algorithm_group = QGroupBox("Thuật toán tính toán liều")
        algorithm_layout = QVBoxLayout()
        algorithm_group.setLayout(algorithm_layout)
        
        # Các thuật toán tính toán liều
        self.algo_pencil_beam = QRadioButton("Pencil Beam")
        self.algo_collapsed_cone = QRadioButton("Collapsed Cone")
        self.algo_monte_carlo = QRadioButton("Monte Carlo")
        
        # Chọn Pencil Beam làm mặc định
        self.algo_pencil_beam.setChecked(True)
        
        algorithm_layout.addWidget(self.algo_pencil_beam)
        algorithm_layout.addWidget(self.algo_collapsed_cone)
        algorithm_layout.addWidget(self.algo_monte_carlo)
        
        # Phần tiến trình
        progress_group = QGroupBox("Tiến trình")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Sẵn sàng.")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        # Các nút hành động
        buttons_layout = QHBoxLayout()
        
        self.calculate_button = QPushButton("Tính toán")
        self.calculate_button.clicked.connect(self._calculate_dose)
        
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self.reject)
        
        buttons_layout.addWidget(self.calculate_button)
        buttons_layout.addWidget(self.cancel_button)
        
        # Thêm các phần vào layout chính
        main_layout.addWidget(beam_model_group)
        main_layout.addWidget(algorithm_group)
        main_layout.addWidget(progress_group)
        main_layout.addLayout(buttons_layout)
    
    def _load_beam_models(self):
        """Tải danh sách các mô hình chùm tia có sẵn."""
        self.beam_model_combo.clear()
        
        try:
            # Tạo processor để lấy danh sách năng lượng
            processor = BeamDataProcessor()
            energies = processor.get_available_energies()
            
            if energies:
                for energy in energies:
                    self.beam_model_combo.addItem(f"TrueBeam {energy}", energy)
            else:
                self.beam_model_combo.addItem("Không có mô hình nào", "")
                
        except Exception as e:
            logger.error(f"Lỗi khi tải danh sách mô hình chùm tia: {str(e)}")
            self.beam_model_combo.addItem("Lỗi khi tải mô hình", "")
    
    def _show_beam_data_input_dialog(self):
        """Hiển thị dialog nhập dữ liệu chùm tia."""
        dialog = BeamDataInputDialog(self)
        result = dialog.exec_()
        
        # Nếu dialog đóng thành công, tải lại danh sách mô hình
        if result == QDialog.Accepted:
            self._load_beam_models()
    
    def _calculate_dose(self):
        """Tính toán liều xạ trị."""
        # Kiểm tra dữ liệu kế hoạch
        if not self.plan_data:
            QMessageBox.warning(
                self, "Cảnh báo",
                "Không có dữ liệu kế hoạch điều trị."
            )
            return
            
        # Lấy mô hình chùm tia đã chọn
        beam_model_index = self.beam_model_combo.currentIndex()
        beam_model_data = self.beam_model_combo.itemData(beam_model_index)
        
        if not beam_model_data:
            QMessageBox.warning(
                self, "Cảnh báo",
                "Vui lòng chọn mô hình chùm tia hợp lệ."
            )
            return
        
        # Xác định thuật toán tính toán liều
        algorithm = DoseCalculationAlgorithm.PENCIL_BEAM  # Mặc định
        
        if self.algo_collapsed_cone.isChecked():
            algorithm = DoseCalculationAlgorithm.COLLAPSED_CONE
        elif self.algo_monte_carlo.isChecked():
            algorithm = DoseCalculationAlgorithm.MONTE_CARLO
        
        # Cập nhật trạng thái
        self.status_label.setText("Đang tính toán liều...")
        self.progress_bar.setValue(10)
        
        try:
            # Tạo bộ tính toán liều
            calculator = DoseCalculator(algorithm)
            
            # Tải mô hình chùm tia
            processor = BeamDataProcessor()
            beam_model = processor.load_beam_model(beam_model_data)
            
            # Cấu hình bộ tính toán liều
            calculator.set_beam_model(beam_model)
            calculator.set_patient_data(self.plan_data.get("patient_data"))
            calculator.set_structures(self.plan_data.get("structures"))
            
            # Tính toán liều
            result = calculator.calculate(self.plan_data.get("beams"))
            
            # Cập nhật trạng thái
            self.progress_bar.setValue(100)
            self.status_label.setText("Đã hoàn thành tính toán liều.")
            
            # Trả về kết quả và đóng dialog
            self.plan_data["dose_result"] = result
            self.accept()
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều: {str(e)}")
            QMessageBox.critical(
                self, "Lỗi",
                f"Đã xảy ra lỗi khi tính toán liều: {str(e)}"
            )
            
            # Cập nhật trạng thái
            self.status_label.setText("Đã xảy ra lỗi.")
            self.progress_bar.setValue(0)

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = DoseCalculationDialog()
    dialog.show()
    sys.exit(app.exec_()) 