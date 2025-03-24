#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog for dose calculation settings and execution.

This module provides a dialog for configuring and executing dose
calculation using various algorithms.
"""

import os
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QFileDialog, QMessageBox, QTabWidget,
    QWidget, QButtonGroup, QGridLayout, QFrame, QSizePolicy, QTreeWidget,
    QTreeWidgetItem, QLineEdit, QRadioButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon

from quangtps.core.exceptions import DoseCalculationError
from quangtps.dose.dose_calculator import DoseCalculator
from quangtps.dose.algorithms import AVAILABLE_ALGORITHMS
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.imaging.image import Image
from quangtps.treatment.beams.beam_data_importer import TrueBeamDataReader
from quangtps.core.constants import DoseCalculationAlgorithm
from quangtps.core.config import Config
from quangtps.treatment.beams.truebeam_data_processor import TrueBeamDataProcessor
from quangtps.ui.dialogs.beam_data_import_dialog import BeamDataImportDialog
from quangtps.dose.beam_data_processor import BeamModel, BeamDataManager
from quangtps.core.types import BeamEnergyType
from quangtps.common.paths import get_beam_data_dir
from quangtps.ui.dialogs.beam_data_input_dialog import BeamDataInputDialog

logger = logging.getLogger(__name__)

class CalculationThread(QThread):
    """Thread để tính toán liều."""
    
    update_signal = pyqtSignal(int, str)  # (tiến độ, thông báo)
    finished_signal = pyqtSignal(bool, str, object)  # (thành công, thông báo, kết quả)
    
    def __init__(self, calculator: DoseCalculator, plan: Plan, algorithm: str, parameters: Dict[str, Any]):
        """
        Khởi tạo thread tính toán.
        
        Parameters
        ----------
        calculator : DoseCalculator
            Đối tượng tính toán liều
        plan : Plan
            Kế hoạch điều trị
        algorithm : str
            Thuật toán tính toán liều
        parameters : Dict[str, Any]
            Các tham số cho thuật toán
        """
        super().__init__()
        self.calculator = calculator
        self.plan = plan
        self.algorithm = algorithm
        self.parameters = parameters
        
    def run(self):
        """Chạy tính toán trong thread riêng."""
        try:
            # Cập nhật trạng thái
            self.update_signal.emit(10, "Đang chuẩn bị dữ liệu...")
            
            # Thiết lập thuật toán và tham số
            self.calculator.set_algorithm(self.algorithm)
            for name, value in self.parameters.items():
                self.calculator.set_parameter(name, value)
                
            # Cập nhật trạng thái
            self.update_signal.emit(20, "Đang tính toán liều...")
            
            # Tính toán liều
            result = self.calculator.calculate_plan_dose(self.plan)
            
            # Hoàn thành
            self.update_signal.emit(100, "Đã hoàn thành tính toán.")
            self.finished_signal.emit(True, "Đã tính toán liều thành công.", result)
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều: {str(e)}", exc_info=True)
            self.finished_signal.emit(False, f"Lỗi khi tính toán: {str(e)}", None)

class DoseCalculationDialog(QDialog):
    """
    Dialog for configuring and executing dose calculation.
    """
    
    # Tín hiệu khi người dùng chọn xong thuật toán và tham số
    algorithmSelected = pyqtSignal(dict)
    
    dose_calculated = pyqtSignal(object)  # Phát tín hiệu khi tính toán hoàn thành

    def __init__(self, parent=None, plan=None, ct_image=None, config=None):
        """
        Initialize the dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        plan : TreatmentPlan, optional
            Treatment plan to calculate dose for
        ct_image : Image, optional
            CT image for dose calculation
        config : Config, optional
            Configuration object
        """
        super().__init__(parent)
        
        self.plan = plan
        self.ct_image = ct_image
        self.calculator = DoseCalculator()
        self.result_dose = None
        self.config = config or Config()
        
        self.setWindowTitle("Tính toán liều")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # Khởi tạo processor để lấy thông tin mô hình chùm tia
        beam_model_dir = self.config.get_path('BEAM_MODEL_DIR')
        self.beam_processor = TrueBeamDataProcessor(beam_model_dir)
        
        self.thread = None
        self.beam_data_manager = BeamDataManager()
        
        # Load các mô hình chùm tia
        self.beam_models = self._load_beam_models()
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Main tabs
        self.tab_widget = QTabWidget()
        
        # Basic settings tab
        self.algorithm_tab = QWidget()
        self.init_algorithm_tab()
        self.tab_widget.addTab(self.algorithm_tab, "Thuật toán")
        
        # Beam model tab
        self.beam_model_tab = QWidget()
        self.init_beam_model_tab()
        self.tab_widget.addTab(self.beam_model_tab, "Mô hình chùm tia")
        
        # Advanced settings tab
        self.advanced_tab = QWidget()
        self.init_advanced_tab()
        self.tab_widget.addTab(self.advanced_tab, "Tùy chọn nâng cao")
        
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Tính toán")
        self.calculate_button.clicked.connect(self._start_calculation)
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self._cancel_calculation)
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.calculate_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Initial setup
        self.update_beam_models()
    
    def init_algorithm_tab(self):
        """Khởi tạo tab thuật toán."""
        layout = QVBoxLayout(self.algorithm_tab)
        
        # Nhóm thuật toán
        algorithm_group = QGroupBox("Thuật toán tính toán liều")
        algorithm_layout = QVBoxLayout(algorithm_group)
        
        self.algorithm_buttons = QButtonGroup(self)
        
        # Các thuật toán
        for algo in DoseCalculationAlgorithm:
            radio_button = QRadioButton(algo.value)
            if algo == DoseCalculationAlgorithm.COLLAPSED_CONE:
                radio_button.setChecked(True)
            
            self.algorithm_buttons.addButton(radio_button)
            algorithm_layout.addWidget(radio_button)
        
        layout.addWidget(algorithm_group)
        
        # Nhóm độ phân giải
        resolution_group = QGroupBox("Độ phân giải tính toán")
        resolution_layout = QFormLayout(resolution_group)
        
        self.grid_size_combo = QComboBox()
        self.grid_size_combo.addItems(["2 mm", "3 mm", "4 mm", "5 mm"])
        self.grid_size_combo.setCurrentIndex(1)  # 3 mm mặc định
        resolution_layout.addRow("Kích thước lưới:", self.grid_size_combo)
        
        layout.addWidget(resolution_group)
        
        # Thêm không gian linh hoạt
        layout.addStretch(1)
    
    def init_beam_model_tab(self):
        """Khởi tạo tab mô hình chùm tia."""
        layout = QVBoxLayout(self.beam_model_tab)
        
        # Nhóm mô hình chùm tia
        beam_model_group = QGroupBox("Mô hình chùm tia hiện có")
        beam_model_layout = QVBoxLayout(beam_model_group)
        
        # Danh sách mô hình chùm tia
        self.beam_model_combo = QComboBox()
        self.update_beam_models()
        beam_model_layout.addWidget(self.beam_model_combo)
        
        # Nút nhập dữ liệu chùm tia mới
        import_button_layout = QHBoxLayout()
        
        self.import_beam_data_button = QPushButton("Nhập dữ liệu chùm tia mới...")
        self.import_beam_data_button.clicked.connect(self.show_beam_data_import_dialog)
        import_button_layout.addWidget(self.import_beam_data_button)
        
        self.refresh_models_button = QPushButton("Làm mới")
        self.refresh_models_button.clicked.connect(self.update_beam_models)
        import_button_layout.addWidget(self.refresh_models_button)
        
        beam_model_layout.addLayout(import_button_layout)
        
        layout.addWidget(beam_model_group)
        
        # Thông tin mô hình chùm tia
        info_group = QGroupBox("Thông tin mô hình")
        info_layout = QFormLayout(info_group)
        
        self.energy_label = QLabel("")
        info_layout.addRow("Năng lượng:", self.energy_label)
        
        self.beam_type_label = QLabel("")
        info_layout.addRow("Loại chùm tia:", self.beam_type_label)
        
        layout.addWidget(info_group)
        
        # Cập nhật thông tin khi chọn mô hình
        self.beam_model_combo.currentIndexChanged.connect(self.update_model_info)
        if self.beam_model_combo.count() > 0:
            self.update_model_info(0)
        
        # Thêm không gian linh hoạt
        layout.addStretch(1)
    
    def init_advanced_tab(self):
        """Khởi tạo tab tùy chọn nâng cao."""
        layout = QVBoxLayout(self.advanced_tab)
        
        # Nhóm tùy chọn tính toán
        calc_options_group = QGroupBox("Tùy chọn tính toán")
        calc_options_layout = QFormLayout(calc_options_group)
        
        self.threads_spinbox = QSpinBox()
        self.threads_spinbox.setMinimum(1)
        self.threads_spinbox.setMaximum(32)
        self.threads_spinbox.setValue(4)
        calc_options_layout.addRow("Số luồng tính toán:", self.threads_spinbox)
        
        self.density_correction_checkbox = QCheckBox("Hiệu chỉnh mật độ")
        self.density_correction_checkbox.setChecked(True)
        calc_options_layout.addRow("", self.density_correction_checkbox)
        
        self.use_gpu_checkbox = QCheckBox("Sử dụng GPU (nếu có)")
        self.use_gpu_checkbox.setChecked(True)
        calc_options_layout.addRow("", self.use_gpu_checkbox)
        
        layout.addWidget(calc_options_group)
        
        # Nhóm tùy chọn báo cáo
        report_group = QGroupBox("Tùy chọn báo cáo")
        report_layout = QFormLayout(report_group)
        
        self.save_intermediate_checkbox = QCheckBox("Lưu các bước trung gian")
        report_layout.addRow("", self.save_intermediate_checkbox)
        
        self.generate_report_checkbox = QCheckBox("Tạo báo cáo tính toán liều")
        self.generate_report_checkbox.setChecked(True)
        report_layout.addRow("", self.generate_report_checkbox)
        
        layout.addWidget(report_group)
        
        # Thêm không gian linh hoạt
        layout.addStretch(1)
    
    def update_beam_models(self):
        """Cập nhật danh sách mô hình chùm tia từ processor."""
        self.beam_model_combo.clear()
        
        # Tải lại các mô hình
        self.beam_processor.load_beam_models()
        
        # Lấy danh sách năng lượng
        energies = self.beam_processor.get_available_energies()
        
        if energies:
            for energy in sorted(energies):
                self.beam_model_combo.addItem(f"TrueBeam {energy}")
        else:
            self.beam_model_combo.addItem("Không có mô hình nào")
    
    def update_model_info(self, index):
        """Cập nhật thông tin mô hình khi chọn mô hình khác."""
        if index < 0 or self.beam_model_combo.count() == 0:
            self.energy_label.setText("")
            self.beam_type_label.setText("")
            return
        
        text = self.beam_model_combo.currentText()
        
        if "TrueBeam" in text:
            energy = text.replace("TrueBeam ", "")
            model = self.beam_processor.get_beam_model(energy)
            
            if model:
                self.energy_label.setText(model.energy)
                self.beam_type_label.setText(model.beam_type)
            else:
                self.energy_label.setText(energy)
                self.beam_type_label.setText("Không xác định")
        else:
            self.energy_label.setText("")
            self.beam_type_label.setText("")
    
    def show_beam_data_import_dialog(self):
        """Hiển thị dialog nhập dữ liệu chùm tia."""
        import_dialog = BeamDataInputDialog(self)
        result = import_dialog.exec_()
        
        if result == QDialog.Accepted:
            # Cập nhật lại danh sách mô hình
            self.update_beam_models()
    
    def _start_calculation(self):
        """Bắt đầu tính toán liều."""
        # Lấy thuật toán
        button = self.algorithm_buttons.checkedButton()
        if not button:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn thuật toán tính toán liều.")
            return
        
        algorithm_text = button.text()
        algorithm = next((algo for algo in DoseCalculationAlgorithm if algo.value == algorithm_text), None)
        
        # Lấy kích thước lưới
        grid_size_text = self.grid_size_combo.currentText()
        grid_size = float(grid_size_text.split()[0])
        
        # Lấy mô hình chùm tia
        beam_model_text = self.beam_model_combo.currentText()
        beam_model = None
        
        if "TrueBeam" in beam_model_text:
            energy = beam_model_text.replace("TrueBeam ", "")
            beam_model = self.beam_processor.get_beam_model(energy)
        
        if not beam_model and "Không có mô hình nào" not in beam_model_text:
            QMessageBox.warning(self, "Cảnh báo", "Không tìm thấy mô hình chùm tia đã chọn.")
            return
        
        # Tạo dictionary tham số
        parameters = {
            "algorithm": algorithm,
            "grid_size": grid_size,
            "beam_model": beam_model,
            "threads": self.threads_spinbox.value(),
            "density_correction": self.density_correction_checkbox.isChecked(),
            "use_gpu": self.use_gpu_checkbox.isChecked(),
            "save_intermediate": self.save_intermediate_checkbox.isChecked(),
            "generate_report": self.generate_report_checkbox.isChecked()
        }
        
        # Vô hiệu hóa UI trong khi tính toán
        self._set_calculating_ui(True)
        
        # Tạo và chạy thread tính toán
        self.thread = CalculationThread(self.calculator, self.plan, algorithm_text, parameters)
        self.thread.update_signal.connect(self._update_progress)
        self.thread.finished_signal.connect(self._calculation_completed)
        self.thread.start()
    
    def _update_progress(self, progress: int, message: str):
        """
        Cập nhật tiến trình tính toán.
        
        Parameters
        ----------
        progress : int
            Giá trị tiến trình (0-100)
        message : str
            Thông báo trạng thái
        """
        self.progress_bar.setValue(progress)
        self.energy_label.setText(message)
    
    def _calculation_completed(self, success: bool, message: str, result: Any):
        """
        Xử lý khi hoàn thành tính toán.
        
        Parameters
        ----------
        success : bool
            Thành công hay không
        message : str
            Thông báo
        result : Any
            Kết quả tính toán
        """
        self._set_calculating_ui(False)
        
        if success and result:
            # Hiển thị thông báo thành công
            QMessageBox.information(self, "Thành công", "Đã tính toán liều thành công.")
            
            # Phát tín hiệu với kết quả
            self.dose_calculated.emit(result)
            
            # Đóng dialog
            self.accept()
            
        else:
            # Hiển thị thông báo lỗi
            QMessageBox.warning(self, "Lỗi", f"Tính toán thất bại: {message}")
    
    def _cancel_calculation(self):
        """Hủy tính toán đang chạy."""
        if self.thread and self.thread.isRunning():
            # Hủy thread
            self.thread.terminate()
            self.thread.wait()
            
            # Cập nhật UI
            self._set_calculating_ui(False)
            self.progress_bar.setValue(0)
            self.energy_label.setText("Đã hủy tính toán")
    
    def _set_calculating_ui(self, is_calculating: bool):
        """
        Thiết lập trạng thái UI khi đang tính toán.
        
        Parameters
        ----------
        is_calculating : bool
            True nếu đang tính toán, False nếu không
        """
        self.calculate_button.setEnabled(not is_calculating)
        self.cancel_button.setEnabled(is_calculating)
        
        # Disable các tab và tham số
        self.grid_size_combo.setEnabled(not is_calculating)
        self.beam_model_combo.setEnabled(not is_calculating)
        self.threads_spinbox.setEnabled(not is_calculating)
        self.density_correction_checkbox.setEnabled(not is_calculating)
        self.use_gpu_checkbox.setEnabled(not is_calculating)
        self.save_intermediate_checkbox.setEnabled(not is_calculating)
        self.generate_report_checkbox.setEnabled(not is_calculating)
        
        # Disable các tham số nâng cao
        for i in range(self.threads_spinbox.layout().count()):
            widget = self.threads_spinbox.layout().itemAt(i, QFormLayout.FieldRole).widget()
            if widget:
                widget.setEnabled(not is_calculating)

    def _load_beam_models(self) -> Dict[str, List[BeamModel]]:
        """
        Tải các mô hình chùm tia.
        
        Returns
        -------
        Dict[str, List[BeamModel]]
            Dictionary chứa các mô hình chùm tia, với khóa là loại chùm tia
        """
        beam_models = {}
        
        try:
            # Lấy tất cả các mô hình từ BeamDataManager
            for beam_type in BeamEnergyType:
                beam_type_name = beam_type.value
                beam_models[beam_type_name] = []
                
                # Tải các mô hình cho loại chùm tia này
                models = self.beam_data_manager.get_beam_models(beam_type_name)
                beam_models[beam_type_name].extend(models)
                
            logger.info(f"Đã tải {sum(len(models) for models in beam_models.values())} mô hình chùm tia")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình chùm tia: {str(e)}", exc_info=True)
            
        return beam_models 