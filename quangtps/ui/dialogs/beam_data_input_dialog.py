#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog nhập dữ liệu chùm tia cho QuangTPS.

Dialog này cho phép người dùng nhập dữ liệu chùm tia từ các file Excel của TrueBeam
và chuyển đổi thành mô hình chùm tia để sử dụng trong QuangTPS.
"""

import os
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFileDialog, QProgressBar, QListWidget,
    QComboBox, QGridLayout, QGroupBox, QCheckBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QSize
from PyQt5.QtGui import QFont

from quangtps.treatment.beams.truebeam_data_processor import TrueBeamDataProcessor
from quangtps.dose.beam_data_processor import BeamModel
from quangtps.core.config import Config
from quangtps.common.paths import get_beam_data_dir

logger = logging.getLogger(__name__)


class ProcessThread(QThread):
    """Thread để xử lý dữ liệu chùm tia."""
    
    update_signal = pyqtSignal(int, str)  # (tiến độ, thông báo)
    finished_signal = pyqtSignal(bool, str, object)  # (thành công, thông báo, kết quả)
    
    def __init__(self, file_path: str):
        """
        Khởi tạo thread xử lý.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file Excel
        """
        super().__init__()
        self.file_path = file_path
        self.processor = TrueBeamDataProcessor()
        
    def run(self):
        """Chạy xử lý trong thread riêng."""
        try:
            # Cập nhật trạng thái
            self.update_signal.emit(10, "Đang đọc file Excel...")
            
            # Đọc file Excel
            success = self.processor.read_excel_file(self.file_path)
            
            if not success:
                self.finished_signal.emit(False, "Không thể đọc file Excel.", None)
                return
                
            # Cập nhật trạng thái
            self.update_signal.emit(50, "Đang tạo mô hình chùm tia...")
            
            # Tạo mô hình chùm tia
            beam_model = self.processor.create_beam_model()
            
            if beam_model is None:
                self.finished_signal.emit(False, "Không thể tạo mô hình chùm tia.", None)
                return
                
            # Hoàn thành
            self.update_signal.emit(100, "Hoàn thành xử lý.")
            self.finished_signal.emit(True, "Đã tạo mô hình chùm tia thành công.", beam_model)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu chùm tia: {str(e)}", exc_info=True)
            self.finished_signal.emit(False, f"Lỗi khi xử lý: {str(e)}", None)


class BeamDataInputDialog(QDialog):
    """Dialog nhập dữ liệu chùm tia từ TrueBeam."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        self.setWindowTitle("Nhập dữ liệu chùm tia TrueBeam")
        self.setMinimumSize(800, 600)
        
        self.config = Config.get_instance()
        self.beam_model = None
        self.thread = None
        
        self._init_ui()
        
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Phần chọn thư mục
        folder_group = QGroupBox("Thư mục dữ liệu")
        folder_layout = QHBoxLayout(folder_group)
        
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("Chọn thư mục chứa dữ liệu TrueBeam...")
        
        self.browse_btn = QPushButton("Duyệt...")
        self.browse_btn.clicked.connect(self._browse_folder)
        
        self.scan_btn = QPushButton("Quét")
        self.scan_btn.clicked.connect(self._scan_folder)
        self.scan_btn.setEnabled(False)
        
        folder_layout.addWidget(self.folder_edit)
        folder_layout.addWidget(self.browse_btn)
        folder_layout.addWidget(self.scan_btn)
        
        # Phần danh sách file
        files_group = QGroupBox("File dữ liệu")
        files_layout = QVBoxLayout(files_group)
        
        self.files_list = QListWidget()
        self.files_list.setAlternatingRowColors(True)
        self.files_list.itemSelectionChanged.connect(self._file_selected)
        
        files_layout.addWidget(self.files_list)
        
        # Phần lọc năng lượng
        energy_group = QGroupBox("Năng lượng")
        energy_layout = QVBoxLayout(energy_group)
        
        self.energy_combo = QComboBox()
        self.energy_combo.addItem("Tất cả")
        self.energy_combo.currentIndexChanged.connect(self._filter_energy)
        
        energy_layout.addWidget(self.energy_combo)
        
        # Phần thông tin
        info_group = QGroupBox("Thông tin")
        info_layout = QGridLayout(info_group)
        
        self.beam_type_label = QLabel("Loại chùm tia:")
        self.beam_type_value = QLabel("")
        
        self.energy_label = QLabel("Năng lượng:")
        self.energy_value = QLabel("")
        
        self.has_pdd_label = QLabel("Dữ liệu PDD:")
        self.has_pdd_value = QLabel("")
        
        self.has_profile_label = QLabel("Dữ liệu Profile:")
        self.has_profile_value = QLabel("")
        
        self.has_output_label = QLabel("Output Factor:")
        self.has_output_value = QLabel("")
        
        info_layout.addWidget(self.beam_type_label, 0, 0)
        info_layout.addWidget(self.beam_type_value, 0, 1)
        info_layout.addWidget(self.energy_label, 1, 0)
        info_layout.addWidget(self.energy_value, 1, 1)
        info_layout.addWidget(self.has_pdd_label, 2, 0)
        info_layout.addWidget(self.has_pdd_value, 2, 1)
        info_layout.addWidget(self.has_profile_label, 3, 0)
        info_layout.addWidget(self.has_profile_value, 3, 1)
        info_layout.addWidget(self.has_output_label, 4, 0)
        info_layout.addWidget(self.has_output_value, 4, 1)
        
        # Phần tiến trình
        progress_group = QGroupBox("Tiến trình")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Chưa xử lý")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        # Phần nút điều khiển
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Xử lý")
        self.process_btn.clicked.connect(self._process_file)
        self.process_btn.setEnabled(False)
        
        self.save_btn = QPushButton("Lưu mô hình")
        self.save_btn.clicked.connect(self._save_model)
        self.save_btn.setEnabled(False)
        
        self.close_btn = QPushButton("Đóng")
        self.close_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        # Layout bên trái và phải
        top_layout = QHBoxLayout()
        top_layout.addWidget(files_group, 2)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(energy_group)
        right_layout.addWidget(info_group, 1)
        top_layout.addLayout(right_layout, 1)
        
        # Thêm vào layout chính
        main_layout.addWidget(folder_group)
        main_layout.addLayout(top_layout, 3)
        main_layout.addWidget(progress_group)
        main_layout.addLayout(button_layout)
        
        # Thiết lập ban đầu
        self._reset_ui()
        
    def _reset_ui(self):
        """Thiết lập lại trạng thái UI."""
        self.folder_edit.clear()
        self.files_list.clear()
        self.energy_combo.clear()
        self.energy_combo.addItem("Tất cả")
        
        self.beam_type_value.setText("")
        self.energy_value.setText("")
        self.has_pdd_value.setText("")
        self.has_profile_value.setText("")
        self.has_output_value.setText("")
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Chưa xử lý")
        
        self.scan_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
    def _browse_folder(self):
        """Mở hộp thoại chọn thư mục."""
        folder = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục dữ liệu TrueBeam", 
            self.config.get_last_used_dir('beam_data_dir', os.path.expanduser("~"))
        )
        
        if folder:
            self.folder_edit.setText(folder)
            self.config.set_last_used_dir('beam_data_dir', folder)
            self.scan_btn.setEnabled(True)
            self._scan_folder()
            
    def _scan_folder(self):
        """Quét thư mục để tìm các file Excel."""
        folder = self.folder_edit.text()
        if not folder or not os.path.isdir(folder):
            return
            
        # Xóa danh sách cũ
        self.files_list.clear()
        self.energy_combo.clear()
        self.energy_combo.addItem("Tất cả")
        
        # Tìm tất cả các file Excel trong thư mục
        excel_files = []
        energy_set = set()
        
        for file in os.listdir(folder):
            if file.endswith(('.xlsx', '.xls')) and not file.startswith('~$'):
                file_path = os.path.join(folder, file)
                
                # Xác định loại năng lượng từ tên file
                processor = TrueBeamDataProcessor()
                beam_type, energy = processor._determine_beam_type_from_filename(file_path)
                
                # Thêm vào danh sách
                excel_files.append((file, file_path, beam_type, energy))
                
                # Thêm vào tập năng lượng
                if energy != "Unknown":
                    energy_set.add(energy)
        
        # Sắp xếp file theo năng lượng
        excel_files.sort(key=lambda x: x[3])
        
        # Thêm vào danh sách
        for file, file_path, beam_type, energy in excel_files:
            item = f"{file} ({energy})"
            self.files_list.addItem(item)
            # Lưu đường dẫn đầy đủ vào userData
            self.files_list.item(self.files_list.count() - 1).setData(Qt.UserRole, file_path)
        
        # Thêm vào combobox năng lượng
        for energy in sorted(energy_set):
            self.energy_combo.addItem(energy)
            
        # Cập nhật UI
        if self.files_list.count() > 0:
            self.files_list.setCurrentRow(0)
            
    def _file_selected(self):
        """Xử lý khi chọn file từ danh sách."""
        if not self.files_list.currentItem():
            self.process_btn.setEnabled(False)
            return
            
        # Lấy đường dẫn file
        file_path = self.files_list.currentItem().data(Qt.UserRole)
        
        if file_path and os.path.exists(file_path):
            # Hiển thị thông tin file
            processor = TrueBeamDataProcessor()
            beam_type, energy = processor._determine_beam_type_from_filename(file_path)
            
            self.beam_type_value.setText(beam_type)
            self.energy_value.setText(energy)
            
            # Các thông tin khác sẽ được hiển thị sau khi xử lý
            self.has_pdd_value.setText("Chưa xác định")
            self.has_profile_value.setText("Chưa xác định")
            self.has_output_value.setText("Chưa xác định")
            
            # Cho phép xử lý
            self.process_btn.setEnabled(True)
            
    def _filter_energy(self):
        """Lọc danh sách file theo năng lượng."""
        selected_energy = self.energy_combo.currentText()
        
        for i in range(self.files_list.count()):
            item = self.files_list.item(i)
            if selected_energy == "Tất cả" or selected_energy in item.text():
                item.setHidden(False)
            else:
                item.setHidden(True)
                
    def _process_file(self):
        """Xử lý file đã chọn."""
        if not self.files_list.currentItem():
            return
            
        # Lấy đường dẫn file
        file_path = self.files_list.currentItem().data(Qt.UserRole)
        
        if not file_path or not os.path.exists(file_path):
            return
            
        # Vô hiệu hóa UI trong khi xử lý
        self._set_processing_ui(True)
        
        # Tạo và chạy thread xử lý
        self.thread = ProcessThread(file_path)
        self.thread.update_signal.connect(self._update_progress)
        self.thread.finished_signal.connect(self._process_completed)
        self.thread.start()
        
    def _update_progress(self, progress: int, message: str):
        """Cập nhật tiến trình xử lý."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        
    def _process_completed(self, success: bool, message: str, beam_model: Optional[BeamModel]):
        """Xử lý khi hoàn thành."""
        self._set_processing_ui(False)
        
        if success and beam_model:
            self.beam_model = beam_model
            self.save_btn.setEnabled(True)
            
            # Cập nhật thông tin
            self.has_pdd_value.setText("Có" if any("pdd" in param for param in beam_model.parameters) else "Không")
            self.has_profile_value.setText("Có" if any("profile" in param for param in beam_model.parameters) else "Không")
            self.has_output_value.setText("Có" if any("output" in param for param in beam_model.parameters) else "Không")
            
            # Hiển thị thông báo thành công
            QMessageBox.information(self, "Thành công", f"Đã xử lý thành công dữ liệu chùm tia {beam_model.energy}.")
            
        else:
            # Hiển thị thông báo lỗi
            QMessageBox.warning(self, "Lỗi", message)
            
    def _set_processing_ui(self, is_processing: bool):
        """Thiết lập trạng thái UI khi đang xử lý."""
        self.browse_btn.setEnabled(not is_processing)
        self.scan_btn.setEnabled(not is_processing and self.folder_edit.text())
        self.process_btn.setEnabled(not is_processing and self.files_list.currentItem())
        self.save_btn.setEnabled(not is_processing and self.beam_model is not None)
        self.close_btn.setEnabled(not is_processing)
        self.files_list.setEnabled(not is_processing)
        self.energy_combo.setEnabled(not is_processing)
        
    def _save_model(self):
        """Lưu mô hình chùm tia."""
        if not self.beam_model:
            return
            
        # Mở hộp thoại lưu file
        beam_data_dir = get_beam_data_dir()
        beam_type_dir = os.path.join(beam_data_dir, self.beam_model.beam_type.lower())
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(beam_type_dir, exist_ok=True)
        
        # Đề xuất tên file
        default_name = f"truebeam_{self.beam_model.energy.lower().replace(' ', '_')}.json"
        default_path = os.path.join(beam_type_dir, default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu mô hình chùm tia", 
            default_path, 
            "JSON Files (*.json)"
        )
        
        if file_path:
            # Lưu mô hình
            try:
                self.beam_model.save_to_file(file_path)
                QMessageBox.information(
                    self, "Thành công", 
                    f"Đã lưu mô hình chùm tia vào:\n{file_path}"
                )
                
                # Đóng dialog
                self.accept()
                
            except Exception as e:
                logger.error(f"Lỗi khi lưu mô hình: {str(e)}", exc_info=True)
                QMessageBox.warning(
                    self, "Lỗi", 
                    f"Không thể lưu mô hình: {str(e)}"
                )
                
    @staticmethod
    def get_beam_model(parent=None) -> Optional[BeamModel]:
        """
        Hiển thị dialog và trả về mô hình chùm tia.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
            
        Returns
        -------
        Optional[BeamModel]
            Mô hình chùm tia, hoặc None nếu người dùng hủy
        """
        dialog = BeamDataInputDialog(parent)
        if dialog.exec_() == QDialog.Accepted and dialog.beam_model:
            return dialog.beam_model
        return None

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    dialog = BeamDataInputDialog()
    dialog.show()
    sys.exit(app.exec_()) 