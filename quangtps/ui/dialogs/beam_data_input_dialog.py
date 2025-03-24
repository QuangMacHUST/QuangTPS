#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module dialog nhập dữ liệu chùm tia TrueBeam.

Module này cung cấp giao diện người dùng để nhập và xử lý dữ liệu chùm tia
từ máy gia tốc TrueBeam.
"""

import os
import sys
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                          QLineEdit, QComboBox, QFileDialog, QProgressBar, 
                          QTextEdit, QGroupBox, QCheckBox, QMessageBox, QListWidget,
                          QListWidgetItem, QAbstractItemView, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from quangtps.core.exceptions import DataImportError
from quangtps.core.types import BeamEnergyType
from quangtps.treatment.beams.beam_data_importer import TrueBeamDataReader
from quangtps.treatment.beams.beam_data_processor import BeamDataProcessor
from quangtps.ui.widgets.helpers import center_dialog, set_icon

logger = logging.getLogger(__name__)

class BeamDataImportWorker(QThread):
    """Thread worker để nhập dữ liệu chùm tia trong nền."""
    progress_updated = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, source_dir: str, energies: List[str]):
        """
        Khởi tạo worker với thư mục nguồn và danh sách năng lượng.
        
        Args:
            source_dir (str): Thư mục chứa dữ liệu TrueBeam.
            energies (List[str]): Danh sách các năng lượng cần nhập.
        """
        super().__init__()
        self.source_dir = source_dir
        self.energies = energies
        self.cancelled = False
    
    def run(self):
        """Thực thi nhập dữ liệu chùm tia."""
        try:
            # Tạo đối tượng TrueBeamDataReader
            reader = TrueBeamDataReader(self.source_dir)
            
            # Quét thư mục dữ liệu
            reader.scan_data_directory()
            
            # Nếu không chỉ định năng lượng, sử dụng tất cả các năng lượng có sẵn
            if not self.energies:
                self.energies = reader.available_energies
            
            # Nhập dữ liệu cho từng năng lượng
            result = {}
            total_energies = len(self.energies)
            
            for i, energy in enumerate(self.energies):
                if self.cancelled:
                    self.progress_updated.emit(100, "Đã hủy.")
                    return
                
                self.progress_updated.emit(
                    int((i / total_energies) * 100),
                    f"Đang nhập dữ liệu cho năng lượng {energy}..."
                )
                
                try:
                    # Nhập dữ liệu chùm tia
                    beam_data = reader.import_beam_data(energy)
                    
                    # Xuất sang JSON
                    json_file = reader.export_to_json(energy)
                    
                    result[energy] = json_file
                    
                    self.progress_updated.emit(
                        int(((i + 1) / total_energies) * 100),
                        f"Đã nhập xong năng lượng {energy}"
                    )
                except Exception as e:
                    self.progress_updated.emit(
                        int(((i + 1) / total_energies) * 100),
                        f"Lỗi khi nhập dữ liệu {energy}: {str(e)}"
                    )
            
            # Gửi tín hiệu hoàn thành
            self.finished_signal.emit(result, "Đã hoàn thành nhập dữ liệu chùm tia.")
            
        except Exception as e:
            self.error_signal.emit(f"Lỗi khi nhập dữ liệu chùm tia: {str(e)}")
    
    def cancel(self):
        """Hủy quá trình nhập dữ liệu."""
        self.cancelled = True

class BeamDataProcessWorker(QThread):
    """Thread worker để xử lý dữ liệu chùm tia trong nền."""
    progress_updated = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, data_dir: str, energies: List[str]):
        """
        Khởi tạo worker với thư mục dữ liệu và danh sách năng lượng.
        
        Args:
            data_dir (str): Thư mục chứa dữ liệu chùm tia.
            energies (List[str]): Danh sách các năng lượng cần xử lý.
        """
        super().__init__()
        self.data_dir = data_dir
        self.energies = energies
        self.cancelled = False
    
    def run(self):
        """Thực thi xử lý dữ liệu chùm tia."""
        try:
            # Tạo đối tượng BeamDataProcessor
            processor = BeamDataProcessor(self.data_dir)
            
            # Nếu không chỉ định năng lượng, sử dụng tất cả các năng lượng có sẵn
            if not self.energies:
                self.energies = processor.get_available_energies()
            
            # Xử lý dữ liệu cho từng năng lượng
            result = {}
            total_energies = len(self.energies)
            
            for i, energy in enumerate(self.energies):
                if self.cancelled:
                    self.progress_updated.emit(100, "Đã hủy.")
                    return
                
                self.progress_updated.emit(
                    int((i / total_energies) * 100),
                    f"Đang xử lý dữ liệu cho năng lượng {energy}..."
                )
                
                try:
                    # Tải mô hình chùm tia
                    model = processor.load_beam_model(energy)
                    
                    # Xuất mô hình
                    model_file = processor.export_beam_model(energy)
                    
                    result[energy] = model_file
                    
                    self.progress_updated.emit(
                        int(((i + 1) / total_energies) * 100),
                        f"Đã xử lý xong năng lượng {energy}"
                    )
                except Exception as e:
                    self.progress_updated.emit(
                        int(((i + 1) / total_energies) * 100),
                        f"Lỗi khi xử lý dữ liệu {energy}: {str(e)}"
                    )
            
            # Gửi tín hiệu hoàn thành
            self.finished_signal.emit(result, "Đã hoàn thành xử lý dữ liệu chùm tia.")
            
        except Exception as e:
            self.error_signal.emit(f"Lỗi khi xử lý dữ liệu chùm tia: {str(e)}")
    
    def cancel(self):
        """Hủy quá trình xử lý dữ liệu."""
        self.cancelled = True

class BeamDataInputDialog(QDialog):
    """
    Dialog nhập dữ liệu chùm tia TrueBeam.
    
    Dialog này cho phép người dùng chọn thư mục chứa dữ liệu chùm tia TrueBeam,
    xem các năng lượng có sẵn, và nhập dữ liệu vào hệ thống.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo dialog nhập dữ liệu chùm tia.
        
        Args:
            parent: Widget cha của dialog.
        """
        super().__init__(parent)
        
        self.setWindowTitle("Nhập dữ liệu chùm tia TrueBeam")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        set_icon(self)
        
        # Các biến thành viên
        self.source_dir = ""
        self.data_dir = self._get_default_data_dir()
        self.available_energies = []
        self.selected_energies = []
        
        self.import_worker = None
        self.process_worker = None
        
        # Tạo UI
        self._init_ui()
        
        # Căn giữa dialog
        center_dialog(self)
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Phần chọn thư mục nguồn
        source_group = QGroupBox("Thư mục nguồn")
        source_layout = QHBoxLayout()
        source_group.setLayout(source_layout)
        
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self.source_edit.setPlaceholderText("Chọn thư mục chứa dữ liệu TrueBeam...")
        
        browse_button = QPushButton("Duyệt...")
        browse_button.clicked.connect(self._browse_source_dir)
        
        scan_button = QPushButton("Quét")
        scan_button.clicked.connect(self._scan_source_dir)
        
        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(browse_button)
        source_layout.addWidget(scan_button)
        
        main_layout.addWidget(source_group)
        
        # Phần danh sách năng lượng
        energy_group = QGroupBox("Năng lượng có sẵn")
        energy_layout = QVBoxLayout()
        energy_group.setLayout(energy_layout)
        
        self.energy_list = QListWidget()
        self.energy_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.energy_list.itemSelectionChanged.connect(self._update_selected_energies)
        
        self.select_all_button = QPushButton("Chọn tất cả")
        self.select_all_button.clicked.connect(self._select_all_energies)
        
        self.clear_selection_button = QPushButton("Bỏ chọn tất cả")
        self.clear_selection_button.clicked.connect(self._clear_energy_selection)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_all_button)
        button_layout.addWidget(self.clear_selection_button)
        
        energy_layout.addWidget(self.energy_list)
        energy_layout.addLayout(button_layout)
        
        main_layout.addWidget(energy_group)
        
        # Phần trạng thái và tiến trình
        status_group = QGroupBox("Trạng thái")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Sẵn sàng.")
        
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        
        main_layout.addWidget(status_group)
        
        # Phần log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_group)
        
        # Các nút hành động
        action_layout = QHBoxLayout()
        
        self.import_button = QPushButton("Nhập dữ liệu")
        self.import_button.clicked.connect(self._import_data)
        self.import_button.setEnabled(False)
        
        self.process_button = QPushButton("Xử lý dữ liệu")
        self.process_button.clicked.connect(self._process_data)
        self.process_button.setEnabled(False)
        
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self._cancel_operation)
        self.cancel_button.setEnabled(False)
        
        close_button = QPushButton("Đóng")
        close_button.clicked.connect(self.close)
        
        action_layout.addWidget(self.import_button)
        action_layout.addWidget(self.process_button)
        action_layout.addWidget(self.cancel_button)
        action_layout.addWidget(close_button)
        
        main_layout.addLayout(action_layout)
    
    def _get_default_data_dir(self) -> str:
        """Lấy thư mục dữ liệu mặc định."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, "data", "beam_data")
    
    def _browse_source_dir(self):
        """Duyệt thư mục nguồn."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục chứa dữ liệu TrueBeam",
            "", QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            self.source_dir = dir_path
            self.source_edit.setText(dir_path)
            
            # Tự động quét sau khi chọn
            self._scan_source_dir()
    
    def _scan_source_dir(self):
        """Quét thư mục nguồn để tìm dữ liệu chùm tia."""
        # Kiểm tra xem đã chọn thư mục chưa
        if not self.source_dir:
            QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng chọn thư mục chứa dữ liệu TrueBeam."
            )
            return
        
        # Cập nhật trạng thái
        self.status_label.setText("Đang quét thư mục dữ liệu...")
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # Tạo đối tượng TrueBeamDataReader
        reader = TrueBeamDataReader(self.source_dir)
        
        try:
            # Quét thư mục dữ liệu
            excel_files = reader.scan_data_directory()
            
            # Cập nhật danh sách năng lượng
            self.available_energies = reader.available_energies
            self._update_energy_list()
            
            # Cập nhật trạng thái
            if excel_files:
                self.status_label.setText(f"Tìm thấy {len(excel_files)} file Excel và {len(self.available_energies)} năng lượng.")
                self.log_text.append(f"Các file Excel đã tìm thấy:")
                for file_path in excel_files:
                    self.log_text.append(f"  - {os.path.basename(file_path)}")
                
                self.log_text.append("\nCác năng lượng có sẵn:")
                for energy in self.available_energies:
                    self.log_text.append(f"  - {energy}")
                
                # Kích hoạt nút nhập dữ liệu
                self.import_button.setEnabled(True)
            else:
                self.status_label.setText("Không tìm thấy file Excel nào trong thư mục.")
                self.log_text.append("Không tìm thấy file Excel nào trong thư mục.")
                
                # Vô hiệu hóa nút nhập dữ liệu
                self.import_button.setEnabled(False)
            
            # Cập nhật giá trị tiến trình
            self.progress_bar.setValue(100)
            
        except Exception as e:
            self.status_label.setText(f"Lỗi khi quét thư mục: {str(e)}")
            self.log_text.append(f"Lỗi khi quét thư mục: {str(e)}")
    
    def _update_energy_list(self):
        """Cập nhật danh sách năng lượng trong UI."""
        self.energy_list.clear()
        
        for energy in self.available_energies:
            item = QListWidgetItem(energy)
            self.energy_list.addItem(item)
    
    def _update_selected_energies(self):
        """Cập nhật danh sách các năng lượng đã chọn."""
        self.selected_energies = [item.text() for item in self.energy_list.selectedItems()]
    
    def _select_all_energies(self):
        """Chọn tất cả các năng lượng."""
        for i in range(self.energy_list.count()):
            self.energy_list.item(i).setSelected(True)
    
    def _clear_energy_selection(self):
        """Bỏ chọn tất cả các năng lượng."""
        for i in range(self.energy_list.count()):
            self.energy_list.item(i).setSelected(False)
    
    def _import_data(self):
        """Nhập dữ liệu chùm tia."""
        # Kiểm tra xem đã chọn năng lượng chưa
        if not self.selected_energies:
            reply = QMessageBox.question(
                self, "Xác nhận",
                "Bạn chưa chọn năng lượng nào. Bạn có muốn nhập tất cả các năng lượng có sẵn?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._select_all_energies()
                self._update_selected_energies()
            else:
                return
        
        # Cập nhật trạng thái
        self.status_label.setText("Đang chuẩn bị nhập dữ liệu...")
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # Vô hiệu hóa các nút
        self.import_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # Tạo worker để nhập dữ liệu trong nền
        self.import_worker = BeamDataImportWorker(self.source_dir, self.selected_energies)
        
        # Kết nối các tín hiệu
        self.import_worker.progress_updated.connect(self._update_progress)
        self.import_worker.finished_signal.connect(self._import_completed)
        self.import_worker.error_signal.connect(self._handle_error)
        
        # Bắt đầu worker
        self.import_worker.start()
    
    def _process_data(self):
        """Xử lý dữ liệu chùm tia."""
        # Kiểm tra xem đã chọn năng lượng chưa
        if not self.selected_energies:
            reply = QMessageBox.question(
                self, "Xác nhận",
                "Bạn chưa chọn năng lượng nào. Bạn có muốn xử lý tất cả các năng lượng có sẵn?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._select_all_energies()
                self._update_selected_energies()
            else:
                return
        
        # Cập nhật trạng thái
        self.status_label.setText("Đang chuẩn bị xử lý dữ liệu...")
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # Vô hiệu hóa các nút
        self.import_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # Tạo worker để xử lý dữ liệu trong nền
        self.process_worker = BeamDataProcessWorker(self.data_dir, self.selected_energies)
        
        # Kết nối các tín hiệu
        self.process_worker.progress_updated.connect(self._update_progress)
        self.process_worker.finished_signal.connect(self._process_completed)
        self.process_worker.error_signal.connect(self._handle_error)
        
        # Bắt đầu worker
        self.process_worker.start()
    
    def _cancel_operation(self):
        """Hủy hoạt động hiện tại."""
        if self.import_worker and self.import_worker.isRunning():
            self.import_worker.cancel()
        
        if self.process_worker and self.process_worker.isRunning():
            self.process_worker.cancel()
        
        self.status_label.setText("Đang hủy...")
    
    @pyqtSlot(int, str)
    def _update_progress(self, progress: int, status: str):
        """Cập nhật tiến trình và trạng thái."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        self.log_text.append(status)
    
    @pyqtSlot(dict, str)
    def _import_completed(self, result: Dict[str, str], status: str):
        """Xử lý khi nhập dữ liệu hoàn thành."""
        # Cập nhật trạng thái
        self.status_label.setText(status)
        self.log_text.append("\n" + status)
        
        # Hiển thị kết quả
        if result:
            self.log_text.append("\nKết quả nhập dữ liệu:")
            for energy, file_path in result.items():
                self.log_text.append(f"  - {energy}: {file_path}")
        
        # Kích hoạt nút xử lý dữ liệu
        self.process_button.setEnabled(True)
        
        # Kích hoạt lại nút nhập dữ liệu
        self.import_button.setEnabled(True)
        
        # Vô hiệu hóa nút hủy
        self.cancel_button.setEnabled(False)
    
    @pyqtSlot(dict, str)
    def _process_completed(self, result: Dict[str, str], status: str):
        """Xử lý khi xử lý dữ liệu hoàn thành."""
        # Cập nhật trạng thái
        self.status_label.setText(status)
        self.log_text.append("\n" + status)
        
        # Hiển thị kết quả
        if result:
            self.log_text.append("\nKết quả xử lý dữ liệu:")
            for energy, file_path in result.items():
                self.log_text.append(f"  - {energy}: {file_path}")
        
        # Kích hoạt lại các nút
        self.import_button.setEnabled(True)
        self.process_button.setEnabled(True)
        
        # Vô hiệu hóa nút hủy
        self.cancel_button.setEnabled(False)
        
        # Hiển thị thông báo thành công
        QMessageBox.information(
            self, "Hoàn thành",
            f"Đã hoàn thành xử lý dữ liệu cho {len(result)} năng lượng."
        )
    
    @pyqtSlot(str)
    def _handle_error(self, error_message: str):
        """Xử lý lỗi."""
        # Cập nhật trạng thái
        self.status_label.setText("Đã xảy ra lỗi.")
        self.log_text.append("\nLỗi: " + error_message)
        
        # Kích hoạt lại các nút
        self.import_button.setEnabled(True)
        self.process_button.setEnabled(True)
        
        # Vô hiệu hóa nút hủy
        self.cancel_button.setEnabled(False)
        
        # Hiển thị thông báo lỗi
        QMessageBox.critical(
            self, "Lỗi",
            f"Đã xảy ra lỗi: {error_message}"
        )

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    dialog = BeamDataInputDialog()
    dialog.show()
    sys.exit(app.exec_()) 