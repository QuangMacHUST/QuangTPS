#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dialog cho phép nhập và xử lý dữ liệu chùm tia từ máy gia tốc TrueBeam.
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import traceback

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QFileDialog, QTableWidget, 
                            QTableWidgetItem, QProgressBar, QMessageBox,
                            QGroupBox, QCheckBox, QApplication, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon

from ...treatment.beams.truebeam_data_processor import TrueBeamDataProcessor, TrueBeamDataReader
from ...dose.beam_data_processor import BeamModel
from ...core.config import Config
from ...core.exceptions import BeamDataError

logger = logging.getLogger(__name__)

class BeamDataProcessingThread(QThread):
    """Thread riêng để xử lý dữ liệu chùm tia để tránh treo giao diện"""
    
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, bool)
    error_signal = pyqtSignal(str)
    
    def __init__(self, input_dir: str, output_dir: str, selected_energies: List[str] = None):
        """
        Khởi tạo thread xử lý dữ liệu
        
        Parameters
        ----------
        input_dir : str
            Thư mục chứa dữ liệu đầu vào
        output_dir : str
            Thư mục lưu kết quả
        selected_energies : List[str], optional
            Danh sách các năng lượng được chọn, by default None (xử lý tất cả)
        """
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.selected_energies = selected_energies
        self.processor = TrueBeamDataProcessor(output_dir)
        self.beam_models = {}
    
    def run(self):
        """Xử lý dữ liệu chùm tia trong thread riêng"""
        try:
            # Quét thư mục đầu vào
            reader = TrueBeamDataReader()
            energy_files = reader.scan_directory(self.input_dir)
            
            # Lọc theo danh sách năng lượng được chọn
            if self.selected_energies:
                energy_files = {k: v for k, v in energy_files.items() if k in self.selected_energies}
            
            # Kiểm tra xem có file nào để xử lý không
            if not energy_files:
                self.error_signal.emit("Không tìm thấy file dữ liệu phù hợp trong thư mục đã chọn.")
                return
            
            # Tạo thư mục đầu ra nếu chưa tồn tại
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Xử lý từng năng lượng
            total_energies = len(energy_files)
            for i, (energy_name, file_path) in enumerate(energy_files.items()):
                try:
                    # Cập nhật tiến trình
                    progress = int((i / total_energies) * 100)
                    self.progress_signal.emit(progress, f"Đang xử lý {energy_name}...")
                    
                    # Đọc dữ liệu
                    beam_data = reader.read_beam_data(file_path)
                    
                    # Tạo mô hình chùm tia
                    beam_model = reader.create_beam_model(beam_data)
                    
                    # Lưu vào dictionary
                    self.beam_models[energy_name] = beam_model
                    
                    # Lưu vào file
                    output_path = os.path.join(self.output_dir, f"TrueBeam_{energy_name}_beam_model.json")
                    beam_model.save_to_json(output_path)
                    
                    # Cập nhật tiến trình
                    self.progress_signal.emit(progress + 1, f"Đã xử lý xong {energy_name}")
                    
                except Exception as e:
                    self.progress_signal.emit(progress, f"Lỗi khi xử lý {energy_name}: {str(e)}")
                    logger.error(f"Lỗi khi xử lý năng lượng {energy_name}: {str(e)}")
                    logger.debug(traceback.format_exc())
            
            # Hoàn thành
            self.progress_signal.emit(100, "Hoàn thành xử lý dữ liệu")
            self.finished_signal.emit(self.beam_models, True)
            
        except Exception as e:
            self.error_signal.emit(f"Lỗi khi xử lý dữ liệu: {str(e)}")
            logger.error(f"Lỗi khi xử lý dữ liệu chùm tia: {str(e)}")
            logger.debug(traceback.format_exc())
            self.finished_signal.emit({}, False)


class BeamDataImportDialog(QDialog):
    """Dialog cho phép nhập và xử lý dữ liệu chùm tia"""
    
    def __init__(self, parent=None):
        """Khởi tạo dialog"""
        super().__init__(parent)
        
        self.config = Config()
        self.beam_models = {}
        self.setWindowTitle("Nhập Dữ Liệu Chùm Tia TrueBeam")
        self.setMinimumSize(700, 500)
        
        self.init_ui()
    
    def init_ui(self):
        """Khởi tạo giao diện người dùng"""
        main_layout = QVBoxLayout(self)
        
        # Nhóm thư mục dữ liệu
        directory_group = QGroupBox("Thư Mục Dữ Liệu")
        directory_layout = QVBoxLayout(directory_group)
        
        # Thư mục đầu vào
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Thư mục dữ liệu TrueBeam:"))
        
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        input_layout.addWidget(self.input_dir_edit)
        
        self.browse_input_btn = QPushButton("Chọn thư mục")
        self.browse_input_btn.clicked.connect(self.browse_input_directory)
        input_layout.addWidget(self.browse_input_btn)
        
        directory_layout.addLayout(input_layout)
        
        # Thư mục đầu ra
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Thư mục lưu mô hình:"))
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        # Thiết lập thư mục mặc định từ cấu hình
        default_output_dir = self.config.get_path('BEAM_MODEL_DIR')
        if os.path.exists(default_output_dir):
            self.output_dir_edit.setText(default_output_dir)
        output_layout.addWidget(self.output_dir_edit)
        
        self.browse_output_btn = QPushButton("Chọn thư mục")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.browse_output_btn)
        
        directory_layout.addLayout(output_layout)
        
        main_layout.addWidget(directory_group)
        
        # Bảng danh sách năng lượng
        energies_group = QGroupBox("Danh Sách Năng Lượng")
        energies_layout = QVBoxLayout(energies_group)
        
        self.energy_table = QTableWidget(0, 3)
        self.energy_table.setHorizontalHeaderLabels(["Năng Lượng", "Đường Dẫn", "Chọn"])
        self.energy_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.energy_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.energy_table.horizontalHeader().setStretchLastSection(False)
        self.energy_table.horizontalHeader().setDefaultSectionSize(100)
        self.energy_table.horizontalHeader().setSectionResizeMode(1, QTableWidget.Stretch)
        
        energies_layout.addWidget(self.energy_table)
        
        # Nút quét thư mục
        scan_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Quét Thư Mục")
        self.scan_btn.clicked.connect(self.scan_directory)
        scan_layout.addWidget(self.scan_btn)
        
        # Nút chọn/bỏ chọn tất cả
        self.select_all_btn = QPushButton("Chọn Tất Cả")
        self.select_all_btn.clicked.connect(self.toggle_all_energies)
        self.select_all_btn.setEnabled(False)
        scan_layout.addWidget(self.select_all_btn)
        
        energies_layout.addLayout(scan_layout)
        
        main_layout.addWidget(energies_group)
        
        # Thanh tiến trình
        progress_group = QGroupBox("Tiến Trình")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Sẵn sàng")
        progress_layout.addWidget(self.status_label)
        
        main_layout.addWidget(progress_group)
        
        # Nút điều khiển
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Xử Lý Dữ Liệu")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)
        
        self.cancel_btn = QPushButton("Hủy")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        # Thanh trạng thái
        self.status_label = QLabel("Sẵn sàng")
        
        button_layout.addWidget(self.status_label)
    
    def browse_input_directory(self):
        """Mở hộp thoại chọn thư mục đầu vào"""
        directory = QFileDialog.getExistingDirectory(
            self, "Chọn Thư Mục Dữ Liệu TrueBeam", 
            self.input_dir_edit.text() or str(Path.home())
        )
        
        if directory:
            self.input_dir_edit.setText(directory)
            # Tự động quét thư mục sau khi chọn
            self.scan_directory()
    
    def browse_output_directory(self):
        """Mở hộp thoại chọn thư mục đầu ra"""
        directory = QFileDialog.getExistingDirectory(
            self, "Chọn Thư Mục Lưu Mô Hình", 
            self.output_dir_edit.text() or str(Path.home())
        )
        
        if directory:
            self.output_dir_edit.setText(directory)
    
    def scan_directory(self):
        """Quét thư mục để tìm các file dữ liệu chùm tia"""
        input_dir = self.input_dir_edit.text()
        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn thư mục dữ liệu hợp lệ!")
            return
        
        try:
            # Quét thư mục
            reader = TrueBeamDataReader()
            energy_files = reader.scan_directory(input_dir)
            
            # Cập nhật bảng
            self.energy_table.setRowCount(0)
            
            if not energy_files:
                self.status_label.setText("Không tìm thấy file dữ liệu phù hợp trong thư mục!")
                self.select_all_btn.setEnabled(False)
                self.process_btn.setEnabled(False)
                return
            
            # Thêm các năng lượng vào bảng
            for i, (energy_name, file_path) in enumerate(energy_files.items()):
                self.energy_table.insertRow(i)
                
                # Cột năng lượng
                energy_item = QTableWidgetItem(energy_name)
                self.energy_table.setItem(i, 0, energy_item)
                
                # Cột đường dẫn
                path_item = QTableWidgetItem(file_path)
                self.energy_table.setItem(i, 1, path_item)
                
                # Cột checkbox
                checkbox = QCheckBox()
                checkbox.setChecked(True)
                self.energy_table.setCellWidget(i, 2, checkbox)
            
            # Kích hoạt các nút
            self.select_all_btn.setEnabled(True)
            self.process_btn.setEnabled(True)
            
            # Cập nhật trạng thái
            num_energies = len(energy_files)
            self.status_label.setText(f"Đã tìm thấy {num_energies} năng lượng chùm tia.")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi quét thư mục: {str(e)}")
            logger.error(f"Lỗi khi quét thư mục: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def toggle_all_energies(self):
        """Chọn hoặc bỏ chọn tất cả các năng lượng"""
        # Kiểm tra trạng thái nút
        if self.select_all_btn.text() == "Chọn Tất Cả":
            # Chọn tất cả
            check_state = True
            self.select_all_btn.setText("Bỏ Chọn Tất Cả")
        else:
            # Bỏ chọn tất cả
            check_state = False
            self.select_all_btn.setText("Chọn Tất Cả")
        
        # Cập nhật các checkbox
        for i in range(self.energy_table.rowCount()):
            checkbox = self.energy_table.cellWidget(i, 2)
            if checkbox:
                checkbox.setChecked(check_state)
    
    def get_selected_energies(self) -> List[str]:
        """Lấy danh sách các năng lượng được chọn"""
        selected_energies = []
        
        for i in range(self.energy_table.rowCount()):
            checkbox = self.energy_table.cellWidget(i, 2)
            if checkbox and checkbox.isChecked():
                energy_name = self.energy_table.item(i, 0).text()
                selected_energies.append(energy_name)
        
        return selected_energies
    
    def process_data(self):
        """Xử lý dữ liệu chùm tia cho các năng lượng được chọn"""
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        
        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn thư mục dữ liệu đầu vào hợp lệ!")
            return
        
        if not output_dir:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn thư mục lưu kết quả!")
            return
        
        # Lấy danh sách năng lượng được chọn
        selected_energies = self.get_selected_energies()
        
        if not selected_energies:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ít nhất một năng lượng để xử lý!")
            return
        
        # Tạo thread xử lý
        self.processing_thread = BeamDataProcessingThread(input_dir, output_dir, selected_energies)
        
        # Kết nối tín hiệu
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.process_completed)
        self.processing_thread.error_signal.connect(self.show_error)
        
        # Vô hiệu hóa các nút trong quá trình xử lý
        self.browse_input_btn.setEnabled(False)
        self.browse_output_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.select_all_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.cancel_btn.setText("Đóng")
        
        # Bắt đầu xử lý
        self.progress_bar.setValue(0)
        self.status_label.setText("Đang bắt đầu xử lý...")
        self.processing_thread.start()
    
    def update_progress(self, value: int, message: str):
        """Cập nhật thanh tiến trình và thông báo trạng thái"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def show_error(self, message: str):
        """Hiển thị thông báo lỗi"""
        QMessageBox.critical(self, "Lỗi", message)
        self.status_label.setText(message)
        
        # Kích hoạt lại các nút
        self.browse_input_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.select_all_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
    
    def process_completed(self, beam_models: Dict[str, BeamModel], success: bool):
        """Xử lý khi hoàn thành"""
        if success:
            # Lưu các mô hình chùm tia
            self.beam_models = beam_models
            
            # Hiển thị thông báo thành công
            num_energies = len(beam_models)
            message = f"Đã xử lý thành công {num_energies} mô hình chùm tia!"
            QMessageBox.information(self, "Hoàn Thành", message)
            self.status_label.setText(message)
            
            # Đặt trạng thái thành công
            self.setResult(QDialog.Accepted)
        else:
            # Hiển thị thông báo lỗi
            self.status_label.setText("Xử lý dữ liệu không thành công!")
        
        # Kích hoạt lại các nút
        self.browse_input_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.select_all_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
    
    def get_beam_models(self) -> Dict[str, BeamModel]:
        """Lấy các mô hình chùm tia đã xử lý"""
        return self.beam_models


# Test dialog
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = BeamDataImportDialog()
    result = dialog.exec_()
    if result == QDialog.Accepted:
        print(f"Đã xử lý {len(dialog.get_beam_models())} mô hình chùm tia")
    sys.exit(app.exec_()) 