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
        
        # Nút làm mới danh sách mô hình
        refresh_button = IconButton(icon_name="refresh", tooltip="Làm mới danh sách mô hình")
        refresh_button.clicked.connect(self._update_model_list)
        
        # Nút tải mô hình
        download_button = IconButton(icon_name="download", tooltip="Tải mô hình mới")
        download_button.clicked.connect(self._show_model_download_dialog)
        
        # Layout cho các nút
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(refresh_button)
        buttons_layout.addWidget(download_button)
        buttons_layout.addStretch()
        
        structure_layout.addLayout(buttons_layout)
        structure_group.setLayout(structure_layout)
        main_layout.addWidget(structure_group)
        
        # Nhóm tùy chọn phân đoạn
        options_group = QGroupBox("Tùy chọn phân đoạn")
        options_layout = QVBoxLayout()
        
        # Checkbox cho các tùy chọn
        self.post_process_check = QCheckBox("Hậu xử lý (loại bỏ các vùng nhỏ)")
        self.post_process_check.setChecked(True)
        options_layout.addWidget(self.post_process_check)
        
        self.smooth_check = QCheckBox("Làm mịn contour")
        self.smooth_check.setChecked(True)
        options_layout.addWidget(self.smooth_check)
        
        # Layout điều chỉnh ngưỡng
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Ngưỡng:"))
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.05)
        threshold_layout.addWidget(self.threshold_spin)
        
        options_layout.addLayout(threshold_layout)
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # Nhóm nút thực hiện
        action_group = QGroupBox("Thực hiện phân đoạn")
        action_layout = QVBoxLayout()
        
        # Nút phân đoạn lát cắt hiện tại
        self.segment_slice_button = QPushButton("Phân đoạn lát cắt hiện tại")
        self.segment_slice_button.clicked.connect(self._segment_current_slice)
        action_layout.addWidget(self.segment_slice_button)
        
        # Nút phân đoạn toàn bộ khối 3D
        self.segment_volume_button = QPushButton("Phân đoạn toàn bộ khối 3D")
        self.segment_volume_button.clicked.connect(self._segment_volume)
        action_layout.addWidget(self.segment_volume_button)
        
        # Thanh tiến trình
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)
        
        action_group.setLayout(action_layout)
        main_layout.addWidget(action_group)
        
        # Thêm khoảng trống co giãn ở cuối
        main_layout.addStretch()
        
        # Gán layout cho widget
        self.setLayout(main_layout)
    
    def _update_model_list(self):
        """Cập nhật danh sách mô hình có sẵn."""
        # Lấy danh sách các mô hình có sẵn từ engine
        self.available_models = self.engine.get_available_models()
        
        # Lưu chọn hiện tại
        current_selection = self.structure_combo.currentText()
        
        # Xóa các mục hiện tại
        self.structure_combo.clear()
        
        # Thêm các mô hình có sẵn vào dropdown
        for model_info in self.available_models:
            if model_info['available']:
                self.structure_combo.addItem(model_info['name'])
        
        # Khôi phục lựa chọn nếu có thể
        if current_selection and self.structure_combo.findText(current_selection) >= 0:
            self.structure_combo.setCurrentText(current_selection)
        
        # Cập nhật trạng thái các nút
        self._update_button_states()
        
        # Log thông tin
        logger.info(f"Đã cập nhật danh sách mô hình: {len(self.available_models)} mô hình")
    
    def _update_button_states(self):
        """Cập nhật trạng thái các nút dựa trên điều kiện hiện tại."""
        # Kiểm tra xem có mô hình nào được chọn không
        has_model = self.structure_combo.count() > 0
        
        # Kích hoạt/vô hiệu hóa các nút phân đoạn
        self.segment_slice_button.setEnabled(has_model and not self._is_task_running())
        self.segment_volume_button.setEnabled(has_model and not self._is_task_running())
    
    def _is_task_running(self) -> bool:
        """Kiểm tra xem có task nào đang chạy không."""
        return self.current_task is not None and self.current_task.is_running
    
    def _get_segmentation_params(self) -> Dict:
        """Lấy các tham số phân đoạn từ giao diện người dùng."""
        params = {
            'threshold': self.threshold_spin.value(),
            'post_process': self.post_process_check.isChecked(),
            'smooth': self.smooth_check.isChecked()
        }
        return params
    
    def _segment_current_slice(self):
        """Phân đoạn lát cắt hiện tại."""
        # Lấy cấu trúc được chọn
        structure = self.structure_combo.currentText()
        if not structure:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một cấu trúc để phân đoạn")
            return
        
        # Kiểm tra xem task đã đang chạy chưa
        if self._is_task_running():
            QMessageBox.warning(self, "Cảnh báo", "Một task phân đoạn khác đang chạy")
            return
        
        try:
            # Lấy dữ liệu hình ảnh từ lát cắt hiện tại
            image_data = self.window().get_current_slice_data()
            if image_data is None:
                QMessageBox.warning(self, "Cảnh báo", "Không thể lấy dữ liệu lát cắt hiện tại")
                return
            
            # Lấy thông tin khoảng cách pixel
            pixel_spacing = self.window().get_pixel_spacing()
            
            # Lấy các tham số phân đoạn
            params = self._get_segmentation_params()
            params['spacing'] = pixel_spacing
            
            # Hiển thị thanh tiến trình
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Tạo và bắt đầu task phân đoạn
            self.current_task = AutoSegmentationTask(
                engine=self.engine,
                task_type='segment_slice',
                structure=structure,
                data=image_data,
                params=params
            )
            
            # Kết nối các tín hiệu
            self.current_task.progress_updated.connect(self.progress_bar.setValue)
            self.current_task.task_finished.connect(self._handle_segment_result)
            self.current_task.task_error.connect(self._handle_segment_error)
            
            # Bắt đầu task
            self.current_task.start()
            
            # Cập nhật trạng thái nút
            self._update_button_states()
            
        except Exception as e:
            logger.error(f"Error starting segmentation: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi bắt đầu phân đoạn: {str(e)}")
    
    def _segment_volume(self):
        """Phân đoạn toàn bộ khối 3D."""
        # Lấy cấu trúc được chọn
        structure = self.structure_combo.currentText()
        if not structure:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một cấu trúc để phân đoạn")
            return
        
        # Kiểm tra xem task đã đang chạy chưa
        if self._is_task_running():
            QMessageBox.warning(self, "Cảnh báo", "Một task phân đoạn khác đang chạy")
            return
        
        # Hiển thị hộp thoại xác nhận
        confirm = QMessageBox.question(
            self, 
            "Xác nhận", 
            f"Phân đoạn cấu trúc '{structure}' trên toàn bộ khối 3D?\n\nLưu ý: Quá trình này có thể mất nhiều thời gian.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.No:
            return
        
        try:
            # Lấy dữ liệu khối 3D
            volume_data = self.window().get_volume_data()
            if volume_data is None:
                QMessageBox.warning(self, "Cảnh báo", "Không thể lấy dữ liệu khối 3D")
                return
            
            # Lấy thông tin khoảng cách voxel
            voxel_spacing = self.window().get_voxel_spacing()
            
            # Lấy các tham số phân đoạn
            params = self._get_segmentation_params()
            params['spacing'] = voxel_spacing
            
            # Hiển thị thanh tiến trình
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Tạo và bắt đầu task phân đoạn
            self.current_task = AutoSegmentationTask(
                engine=self.engine,
                task_type='segment_volume',
                structure=structure,
                data=volume_data,
                params=params
            )
            
            # Kết nối các tín hiệu
            self.current_task.progress_updated.connect(self.progress_bar.setValue)
            self.current_task.task_finished.connect(self._handle_segment_result)
            self.current_task.task_error.connect(self._handle_segment_error)
            
            # Bắt đầu task
            self.current_task.start()
            
            # Cập nhật trạng thái nút
            self._update_button_states()
            
        except Exception as e:
            logger.error(f"Error starting volume segmentation: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi bắt đầu phân đoạn khối: {str(e)}")
    
    def _handle_segment_result(self, result: Dict):
        """Xử lý kết quả phân đoạn."""
        # Ẩn thanh tiến trình
        self.progress_bar.setVisible(False)
        
        # Kiểm tra xem có thành công không
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error')
            QMessageBox.warning(self, "Cảnh báo", f"Phân đoạn thất bại: {error_msg}")
            return
        
        try:
            # Lấy tên cấu trúc
            structure = result.get('structure', self.structure_combo.currentText())
            
            if 'contours' in result:
                # Xử lý kết quả phân đoạn lát cắt
                contours = result['contours']
                
                # Áp dụng contours vào lát cắt hiện tại
                self.window().add_contours_to_current_slice(structure, contours)
                
                # Thông báo thành công
                QMessageBox.information(
                    self, 
                    "Thành công", 
                    f"Đã phân đoạn cấu trúc '{structure}' trên lát cắt hiện tại"
                )
                
            elif 'contours_3d' in result:
                # Xử lý kết quả phân đoạn khối 3D
                contours_3d = result['contours_3d']
                
                # Áp dụng contours vào khối 3D
                self.window().add_contours_to_volume(structure, contours_3d)
                
                # Thông báo thành công
                QMessageBox.information(
                    self, 
                    "Thành công", 
                    f"Đã phân đoạn cấu trúc '{structure}' trên toàn bộ khối 3D"
                )
            
            # Làm sạch task hiện tại
            self.current_task = None
            
            # Cập nhật trạng thái nút
            self._update_button_states()
            
        except Exception as e:
            logger.error(f"Error handling segmentation result: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi xử lý kết quả phân đoạn: {str(e)}")
    
    def _handle_segment_error(self, error_msg: str):
        """Xử lý lỗi phân đoạn."""
        # Ẩn thanh tiến trình
        self.progress_bar.setVisible(False)
        
        # Hiển thị thông báo lỗi
        QMessageBox.critical(self, "Lỗi", f"Lỗi khi phân đoạn: {error_msg}")
        
        # Làm sạch task hiện tại
        self.current_task = None
        
        # Cập nhật trạng thái nút
        self._update_button_states()
    
    def _show_model_download_dialog(self):
        """Hiển thị hộp thoại tải mô hình mới."""
        # Tạo hộp thoại tải mô hình
        dialog = QDialog(self)
        dialog.setWindowTitle("Tải mô hình phân đoạn")
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(400)
        
        # Tạo layout chính
        main_layout = QVBoxLayout()
        
        # Thêm chú thích
        info_label = QLabel("Chọn mô hình phân đoạn để tải xuống:")
        main_layout.addWidget(info_label)
        
        # Tạo bảng danh sách mô hình
        model_table = QTableWidget()
        model_table.setColumnCount(4)
        model_table.setHorizontalHeaderLabels(["Mô hình", "Mô tả", "Phiên bản", "Trạng thái"])
        model_table.setSelectionBehavior(QTableWidget.SelectRows)
        model_table.setSelectionMode(QTableWidget.SingleSelection)
        model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        main_layout.addWidget(model_table)
        
        # Thêm nút làm mới danh sách
        refresh_button = QPushButton("Làm mới danh sách")
        
        # Thêm thanh tiến trình
        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        
        # Thêm nút tải xuống
        download_button = QPushButton("Tải xuống")
        download_button.setEnabled(False)
        
        # Tạo layout nút
        button_layout = QHBoxLayout()
        button_layout.addWidget(refresh_button)
        button_layout.addStretch()
        button_layout.addWidget(download_button)
        
        # Thêm thanh tiến trình và nút vào layout chính
        main_layout.addWidget(progress_bar)
        main_layout.addLayout(button_layout)
        
        # Thiết lập layout cho hộp thoại
        dialog.setLayout(main_layout)
        
        # Hàm cập nhật danh sách mô hình
        def update_model_list():
            # Hiển thị thanh tiến trình trong khi cập nhật
            progress_bar.setVisible(True)
            progress_bar.setRange(0, 0)  # Hiển thị chuyển động không xác định
            
            # Vô hiệu hóa các nút trong khi cập nhật
            refresh_button.setEnabled(False)
            download_button.setEnabled(False)
            
            # Sử dụng QTimer để cập nhật trong một luồng riêng (cho phép UI responsive)
            def update_in_thread():
                # Cập nhật danh sách mô hình từ kho từ xa
                success = self.engine.model_repository.update_model_list(force_reload=True)
                
                # Hiển thị các mô hình trong bảng
                model_table.setRowCount(0)  # Xóa tất cả các hàng
                
                for i, model in enumerate(self.engine.model_repository.get_available_models()):
                    model_table.insertRow(i)
                    
                    # Tên mô hình
                    model_table.setItem(i, 0, QTableWidgetItem(model['name']))
                    
                    # Mô tả
                    description = model.get('description', '')
                    model_table.setItem(i, 1, QTableWidgetItem(description))
                    
                    # Phiên bản
                    version = model.get('version', '1.0.0')
                    model_table.setItem(i, 2, QTableWidgetItem(version))
                    
                    # Trạng thái
                    status = "Đã tải" if model.get('available', False) else "Chưa tải"
                    status_item = QTableWidgetItem(status)
                    status_item.setForeground(QColor("green" if model.get('available', False) else "red"))
                    model_table.setItem(i, 3, status_item)
                
                # Tự động điều chỉnh kích thước cột
                model_table.resizeColumnsToContents()
                
                # Kích hoạt lại các nút
                refresh_button.setEnabled(True)
                
                # Ẩn thanh tiến trình
                progress_bar.setVisible(False)
                
                # Hiển thị thông báo nếu không thành công
                if not success:
                    QMessageBox.warning(dialog, "Cảnh báo", "Không thể kết nối với kho mô hình từ xa")
                
                # Cập nhật danh sách mô hình trong công cụ chính
                self._update_model_list()
            
            # Sử dụng QTimer để cập nhật trong một luồng riêng
            QTimer.singleShot(100, update_in_thread)
        
        # Kết nối sự kiện chọn hàng trong bảng
        def on_selection_changed():
            selected_rows = model_table.selectionModel().selectedRows()
            if selected_rows:
                row = selected_rows[0].row()
                model_name = model_table.item(row, 0).text()
                model_available = model_table.item(row, 3).text() == "Đã tải"
                download_button.setEnabled(not model_available)
            else:
                download_button.setEnabled(False)
        
        model_table.selectionModel().selectionChanged.connect(on_selection_changed)
        
        # Hàm tải xuống mô hình
        def download_selected_model():
            selected_rows = model_table.selectionModel().selectedRows()
            if not selected_rows:
                return
            
            row = selected_rows[0].row()
            model_name = model_table.item(row, 0).text()
            
            # Hiển thị thanh tiến trình
            progress_bar.setVisible(True)
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            
            # Vô hiệu hóa các nút trong khi tải xuống
            refresh_button.setEnabled(False)
            download_button.setEnabled(False)
            
            # Hàm callback cập nhật tiến trình
            def update_progress(progress):
                progress_bar.setValue(progress)
            
            # Sử dụng QTimer để tải xuống trong một luồng riêng
            def download_in_thread():
                # Tải xuống mô hình
                success = self.engine.model_repository.download_model(
                    model_name, 
                    progress_callback=update_progress
                )
                
                # Hiển thị thông báo
                if success:
                    QMessageBox.information(
                        dialog, 
                        "Thành công", 
                        f"Đã tải xuống mô hình '{model_name}' thành công"
                    )
                    
                    # Cập nhật trạng thái trong bảng
                    status_item = QTableWidgetItem("Đã tải")
                    status_item.setForeground(QColor("green"))
                    model_table.setItem(row, 3, status_item)
                    
                    # Cập nhật danh sách mô hình trong công cụ chính
                    self._update_model_list()
                else:
                    QMessageBox.critical(
                        dialog, 
                        "Lỗi", 
                        f"Không thể tải xuống mô hình '{model_name}'"
                    )
                
                # Kích hoạt lại các nút
                refresh_button.setEnabled(True)
                download_button.setEnabled(True)
                
                # Ẩn thanh tiến trình
                progress_bar.setVisible(False)
            
            # Sử dụng QTimer để tải xuống trong một luồng riêng
            QTimer.singleShot(100, download_in_thread)
        
        # Kết nối các sự kiện
        refresh_button.clicked.connect(update_model_list)
        download_button.clicked.connect(download_selected_model)
        
        # Cập nhật danh sách mô hình khi hiển thị hộp thoại
        update_model_list()
        
        # Hiển thị hộp thoại
        dialog.exec_()
    
    def mouse_press(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi nhấn chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        # Không cần xử lý sự kiện chuột cho công cụ này
        pass
    
    def mouse_move(self, pos: Tuple[int, int], buttons: int):
        """
        Xử lý sự kiện khi di chuyển chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        buttons : int
            Các nút chuột đang được nhấn (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        # Không cần xử lý sự kiện chuột cho công cụ này
        pass
    
    def mouse_release(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi thả chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        # Không cần xử lý sự kiện chuột cho công cụ này
        pass
    
    def key_press(self, key: int):
        """
        Xử lý sự kiện khi nhấn phím.
        
        Parameters
        ----------
        key : int
            Mã phím
        """
        # Không cần xử lý sự kiện phím cho công cụ này
        pass
    
    def paint(self, painter):
        """
        Vẽ lên hình ảnh.
        
        Parameters
        ----------
        painter : QPainter
            Đối tượng QPainter để vẽ
        """
        # Không cần vẽ gì cho công cụ này
        pass
    
    def apply_to_current_slice(self):
        """Áp dụng contour vào lát cắt hiện tại."""
        # Sẽ được gọi khi phân đoạn hoàn tất
        pass
