#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp công cụ phân đoạn tự động cho QuangTPS.

Module này triển khai giao diện người dùng để tương tác với engine
phân đoạn tự động dựa trên deep learning cho hệ thống QuangTPS.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import time
import json

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPalette, QFont, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QProgressBar, QGroupBox, QCheckBox, QSpinBox,
                           QDoubleSpinBox, QFileDialog, QMessageBox, QSplitter, 
                           QFrame, QScrollArea, QTabWidget, QListWidget, QListWidgetItem,
                           QDialog, QTableWidget, QTableWidgetItem, QHeaderView)

from quangtps.ui.base_contour_tool import ContourTool
from quangtps.segmentation.auto.engine import AutoSegmentationEngine
from quangtps.common.widgets import CollapsibleBox, IconButton, ColorButton
from quangtps.segmentation.deep_learning_segmentation import SegmentationModel, available_models
from quangtps.segmentation.model_downloader import ensure_default_models
from quangtps.ui.segmentation_model_manager import SegmentationModelManager

logger = logging.getLogger(__name__)


class AutoSegmentationTask(QThread):
    """
    Class thực hiện phân đoạn tự động trong một luồng riêng.
    
    Luồng này được sử dụng để thực hiện các tác vụ phân đoạn tự động
    mà không làm đứng giao diện người dùng.
    """
    # Tín hiệu cập nhật tiến trình
    progress_updated = pyqtSignal(int)
    # Tín hiệu khi hoàn thành
    task_finished = pyqtSignal(dict)
    # Tín hiệu khi có lỗi
    task_error = pyqtSignal(str)
    
    def __init__(self, 
                engine: AutoSegmentationEngine,
                task_type: str,
                structure: str,
                data: Any,
                params: Dict = None):
        """
        Khởi tạo task phân đoạn.
        
        Parameters
        ----------
        engine : AutoSegmentationEngine
            Engine phân đoạn tự động
        task_type : str
            Loại task ('segment_slice', 'segment_volume', 'segment_dicom')
        structure : str
            Cấu trúc cần phân đoạn
        data : Any
            Dữ liệu đầu vào (hình ảnh, khối 3D, hoặc đường dẫn DICOM)
        params : Dict, optional
            Các tham số bổ sung
        """
        super().__init__()
        
        self.engine = engine
        self.task_type = task_type
        self.structure = structure
        self.data = data
        self.params = params or {}
        self.is_running = False
    
    def run(self):
        """Thực hiện task phân đoạn."""
        self.is_running = True
        result = {'success': False, 'error': 'Unknown error'}
        
        try:
            if self.task_type == 'segment_slice':
                # Phân đoạn một lát cắt
                slice_image = self.data
                spacing = self.params.get('spacing')
                
                # Cập nhật tiến trình
                self.progress_updated.emit(10)
                
                # Thực hiện phân đoạn
                result = self.engine.segment_slice(slice_image, self.structure, spacing)
                
                # Cập nhật tiến trình
                self.progress_updated.emit(100)
                
            elif self.task_type == 'segment_volume':
                # Phân đoạn toàn bộ khối 3D
                volume = self.data
                spacing = self.params.get('spacing')
                
                # Lấy số lát cắt để cập nhật tiến trình
                num_slices = volume.shape[0]
                
                # Tiến trình ban đầu
                self.progress_updated.emit(5)
                
                # Phân đoạn từng lát cắt và cập nhật tiến trình
                # Lưu ý: đây là phiên bản đơn giản, engine thực tế sẽ làm việc này
                for i in range(num_slices):
                    if not self.is_running:
                        return
                    
                    # Cập nhật tiến trình
                    progress = 5 + int(90 * (i + 1) / num_slices)
                    self.progress_updated.emit(progress)
                
                # Thực hiện phân đoạn khối
                result = self.engine.segment_volume(volume, self.structure, spacing)
                
                # Hoàn thành
                self.progress_updated.emit(100)
                
            elif self.task_type == 'segment_dicom':
                # Phân đoạn từ dữ liệu DICOM
                dicom_folder = self.data
                output_folder = self.params.get('output_folder')
                
                # Cập nhật tiến trình
                self.progress_updated.emit(10)
                
                # Thực hiện phân đoạn
                result = self.engine.segment_from_dicom(dicom_folder, self.structure, output_folder)
                
                # Hoàn thành
                self.progress_updated.emit(100)
                
            else:
                result = {'success': False, 'error': f"Unknown task type: {self.task_type}"}
                
            # Đã hoàn thành, gửi kết quả
            self.task_finished.emit(result)
            
        except Exception as e:
            logger.error(f"Error in auto segmentation task: {str(e)}")
            self.task_error.emit(str(e))
        
        finally:
            self.is_running = False
    
    def stop(self):
        """Dừng task phân đoạn."""
        self.is_running = False


class AutoSegmentationTool(ContourTool):
    """
    Công cụ phân đoạn tự động sử dụng deep learning.
    
    Công cụ này cung cấp giao diện người dùng cho phép:
    - Chọn cấu trúc để phân đoạn
    - Phân đoạn tự động trên lát cắt hiện tại
    - Phân đoạn tự động trên toàn bộ khối 3D
    - Quản lý và tải xuống các mô hình phân đoạn
    """
    
    def __init__(self):
        """Khởi tạo công cụ phân đoạn tự động."""
        super().__init__("Auto Segmentation")
        
        # Ensure default models are available
        try:
            ensure_default_models()
        except Exception as e:
            logger.warning(f"Could not ensure default models: {str(e)}")
        
        # Khởi tạo engine phân đoạn
        self.engine = AutoSegmentationEngine()
        
        # Task hiện tại
        self.current_task = None
        
        # Danh sách các mô hình có sẵn
        self.available_models = []
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Cập nhật danh sách mô hình
        self._update_model_list()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout()
        
        # Nhóm chọn cấu trúc
        structure_group = QGroupBox("Chọn cấu trúc cần phân đoạn")
        structure_layout = QVBoxLayout()
        
        # Dropdown chọn cấu trúc
        self.structure_combo = QComboBox()
        self.structure_combo.setMinimumWidth(200)
        structure_layout.addWidget(self.structure_combo)
        
        # Nút làm mới danh sách cấu trúc
        refresh_button = QPushButton("Làm mới danh sách")
        refresh_button.clicked.connect(self._update_structure_list)
        structure_layout.addWidget(refresh_button)
        
        # Thiết lập layout cho nhóm
        structure_group.setLayout(structure_layout)
        
        # Thêm nhóm vào layout chính
        main_layout.addWidget(structure_group)
        
        # Nhóm tùy chọn phân đoạn
        options_group = QGroupBox("Tùy chọn phân đoạn")
        options_layout = QVBoxLayout()
        
        # Checkbox cho việc dùng GPU
        self.use_gpu_checkbox = QCheckBox("Sử dụng GPU nếu có")
        self.use_gpu_checkbox.setChecked(True)
        options_layout.addWidget(self.use_gpu_checkbox)
        
        # Slider điều chỉnh ngưỡng
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Ngưỡng:"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.05)
        self.threshold_spinbox.setValue(0.5)
        threshold_layout.addWidget(self.threshold_spinbox)
        options_layout.addLayout(threshold_layout)
        
        # Slider điều chỉnh smooth
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Làm mịn:"))
        self.smooth_spinbox = QSpinBox()
        self.smooth_spinbox.setRange(0, 10)
        self.smooth_spinbox.setValue(2)
        smooth_layout.addWidget(self.smooth_spinbox)
        options_layout.addLayout(smooth_layout)
        
        # Thiết lập layout cho nhóm tùy chọn
        options_group.setLayout(options_layout)
        
        # Thêm nhóm tùy chọn vào layout chính
        main_layout.addWidget(options_group)
        
        # Nút hành động
        action_group = QGroupBox("Hành động")
        action_layout = QVBoxLayout()
        
        # Nút phân đoạn lát cắt hiện tại
        self.segment_slice_button = QPushButton("Phân đoạn lát cắt hiện tại")
        self.segment_slice_button.clicked.connect(self._segment_current_slice)
        action_layout.addWidget(self.segment_slice_button)
        
        # Nút phân đoạn toàn bộ khối
        self.segment_volume_button = QPushButton("Phân đoạn toàn bộ khối")
        self.segment_volume_button.clicked.connect(self._segment_entire_volume)
        action_layout.addWidget(self.segment_volume_button)
        
        # Nút hủy phân đoạn
        self.cancel_button = QPushButton("Hủy phân đoạn")
        self.cancel_button.clicked.connect(self._cancel_segmentation)
        self.cancel_button.setEnabled(False)
        action_layout.addWidget(self.cancel_button)
        
        # Thiết lập layout cho nhóm hành động
        action_group.setLayout(action_layout)
        
        # Thêm nhóm hành động vào layout chính
        main_layout.addWidget(action_group)
        
        # Nhóm tiến trình
        progress_group = QGroupBox("Tiến trình")
        progress_layout = QVBoxLayout()
        
        # Thanh tiến trình
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Label trạng thái
        self.status_label = QLabel("Sẵn sàng")
        progress_layout.addWidget(self.status_label)
        
        # Thiết lập layout cho nhóm tiến trình
        progress_group.setLayout(progress_layout)
        
        # Thêm nhóm tiến trình vào layout chính
        main_layout.addWidget(progress_group)
        
        # Nhóm tải và quản lý mô hình
        model_group = QGroupBox("Quản lý mô hình")
        model_layout = QVBoxLayout()
        
        # Nút tải mô hình mới
        self.download_model_button = QPushButton("Tải và quản lý mô hình")
        self.download_model_button.clicked.connect(self._show_model_manager)
        model_layout.addWidget(self.download_model_button)
        
        # Thiết lập layout cho nhóm tải mô hình
        model_group.setLayout(model_layout)
        
        # Thêm nhóm tải mô hình vào layout chính
        main_layout.addWidget(model_group)
        
        # Thêm khoảng trống co giãn
        main_layout.addStretch()
        
        # Thiết lập layout chính
        self.setLayout(main_layout)
    
    def _update_model_list(self):
        """Cập nhật danh sách mô hình có sẵn."""
        try:
            # Lấy danh sách mô hình
            self.available_models = available_models()
            
            # Cập nhật danh sách cấu trúc
            self._update_structure_list()
            
            # Log thông tin
            logger.info(f"Loaded {len(self.available_models)} segmentation models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.status_label.setText(f"Lỗi: {str(e)}")
    
    def _update_structure_list(self):
        """Cập nhật danh sách cấu trúc từ các mô hình có sẵn."""
        # Lưu lại lựa chọn hiện tại (nếu có)
        current_structure = self.structure_combo.currentText()
        
        # Xóa danh sách cũ
        self.structure_combo.clear()
        
        # Danh sách tất cả các cấu trúc
        all_structures = set()
        
        # Thêm cấu trúc từ các mô hình có sẵn
        for model in self.available_models:
            structures = model.get('structures', [])
            all_structures.update(structures)
        
        # Thêm vào combobox
        for structure in sorted(all_structures):
            self.structure_combo.addItem(structure)
        
        # Khôi phục lựa chọn trước đó nếu có thể
        if current_structure and self.structure_combo.findText(current_structure) >= 0:
            self.structure_combo.setCurrentText(current_structure)
    
    def _show_model_manager(self):
        """Hiển thị hộp thoại quản lý mô hình."""
        # Tạo hộp thoại quản lý mô hình
        dialog = SegmentationModelManager(self)
        
        # Kết nối sự kiện thay đổi mô hình
        dialog.models_changed.connect(self._update_model_list)
        
        # Hiển thị hộp thoại
        dialog.exec_()
    
    def _segment_current_slice(self):
        """Phân đoạn lát cắt hiện tại."""
        # Check for valid selection
        if not self.structure_combo.currentText():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn cấu trúc để phân đoạn")
            return
            
        # Check if viewer has image
        if not hasattr(self, 'current_slice') or self.current_slice is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có hình ảnh để phân đoạn")
            return
        
        # Get parameters
        structure_name = self.structure_combo.currentText()
        use_gpu = self.use_gpu_checkbox.isChecked()
        threshold = self.threshold_spinbox.value()
        smooth = self.smooth_spinbox.value()
        
        # Update UI
        self.status_label.setText(f"Đang phân đoạn {structure_name}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._update_button_states(True)
        
        # Create task
        self.current_task = AutoSegmentationTask(
            self.engine,
            'segment_slice',
            structure_name,
            self.current_slice,
            {
                'use_gpu': use_gpu,
                'threshold': threshold,
                'smooth': smooth
            }
        )
        
        # Connect signals
        self.current_task.progress_updated.connect(self._update_progress)
        self.current_task.task_finished.connect(self._handle_segmentation_result)
        self.current_task.task_error.connect(self._handle_segmentation_error)
        
        # Start task
        self.current_task.start()
    
    def _segment_entire_volume(self):
        """Phân đoạn toàn bộ khối 3D."""
        # Check for valid selection
        if not self.structure_combo.currentText():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn cấu trúc để phân đoạn")
            return
            
        # Check if viewer has volume
        if not hasattr(self, 'image_volume') or self.image_volume is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có khối 3D để phân đoạn")
            return
        
        # Get parameters
        structure_name = self.structure_combo.currentText()
        use_gpu = self.use_gpu_checkbox.isChecked()
        threshold = self.threshold_spinbox.value()
        smooth = self.smooth_spinbox.value()
        
        # Update UI
        self.status_label.setText(f"Đang phân đoạn {structure_name} (toàn bộ khối)...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._update_button_states(True)
        
        # Create task
        self.current_task = AutoSegmentationTask(
            self.engine,
            'segment_volume',
            structure_name,
            self.image_volume,
            {
                'use_gpu': use_gpu,
                'threshold': threshold,
                'smooth': smooth
            }
        )
        
        # Connect signals
        self.current_task.progress_updated.connect(self._update_progress)
        self.current_task.task_finished.connect(self._handle_segmentation_result)
        self.current_task.task_error.connect(self._handle_segmentation_error)
        
        # Start task
        self.current_task.start()
    
    def _cancel_segmentation(self):
        """Hủy phân đoạn đang thực hiện."""
        if self.current_task and self.current_task.is_running:
            self.current_task.stop()
            self.current_task = None
            
            # Update UI
            self.status_label.setText("Đã hủy phân đoạn")
            self.progress_bar.setVisible(False)
            self._update_button_states(False)
    
    def _update_progress(self, progress: int):
        """Cập nhật thanh tiến trình."""
        self.progress_bar.setValue(progress)
    
    def _update_button_states(self, is_processing: bool):
        """Cập nhật trạng thái các nút."""
        self.segment_slice_button.setEnabled(not is_processing)
        self.segment_volume_button.setEnabled(not is_processing)
        self.cancel_button.setEnabled(is_processing)
        self.structure_combo.setEnabled(not is_processing)
        self.use_gpu_checkbox.setEnabled(not is_processing)
        self.threshold_spinbox.setEnabled(not is_processing)
        self.smooth_spinbox.setEnabled(not is_processing)
        self.download_model_button.setEnabled(not is_processing)
    
    def _handle_segmentation_result(self, result: Dict):
        """Xử lý kết quả phân đoạn."""
        # Reset UI
        self.progress_bar.setVisible(False)
        self._update_button_states(False)
        
        # Check if successful
        if result.get('success', False):
            # Get segmentation result
            structure = result.get('structure')
            mask = result.get('mask')
            
            if structure and mask is not None:
                # Update status
                self.status_label.setText(f"Đã phân đoạn {structure}")
                
                # Add structure to image
                self._add_structure_to_image(structure, mask)
            else:
                # No structure or mask
                self.status_label.setText("Không tìm thấy kết quả phân đoạn")
        else:
            # Error
            error_msg = result.get('error', 'Unknown error')
            self.status_label.setText(f"Lỗi: {error_msg}")
            QMessageBox.warning(self, "Lỗi", f"Lỗi khi phân đoạn: {error_msg}")
        
        # Clear current task
        self.current_task = None
    
    def _handle_segmentation_error(self, error_msg: str):
        """Xử lý lỗi phân đoạn."""
        # Ẩn thanh tiến trình
        self.progress_bar.setVisible(False)
        
        # Hiển thị thông báo lỗi
        QMessageBox.critical(self, "Lỗi", f"Lỗi khi phân đoạn: {error_msg}")
        
        # Làm sạch task hiện tại
        self.current_task = None
        
        # Cập nhật trạng thái nút
        self._update_button_states(False)
    
    def _add_structure_to_image(self, structure_name: str, mask):
        """
        Thêm cấu trúc vào hình ảnh.
        
        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        mask : ndarray
            Mặt nạ phân đoạn
        """
        # This is a placeholder - the actual implementation would add the structure
        # to the current image in the main application
        QMessageBox.information(
            self, 
            "Kết quả phân đoạn", 
            f"Đã phân đoạn cấu trúc: {structure_name}\nKích thước mặt nạ: {mask.shape}"
        )
    
    # Mouse and keyboard event handlers (unchanged)
    def mouse_press(self, pos: Tuple[int, int], button: int):
        pass
    
    def mouse_move(self, pos: Tuple[int, int], buttons: int):
        pass
    
    def mouse_release(self, pos: Tuple[int, int], button: int):
        pass
    
    def key_press(self, key: int):
        pass
