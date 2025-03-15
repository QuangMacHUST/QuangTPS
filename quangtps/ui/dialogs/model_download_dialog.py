#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp hộp thoại tải mô hình phân đoạn tự động.

Module này triển khai giao diện người dùng cho phép tải xuống
các mô hình phân đoạn tự động từ kho lưu trữ.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import threading

from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QDialog, QPushButton, QVBoxLayout, QHBoxLayout, 
                           QLabel, QListWidget, QListWidgetItem, QProgressBar,
                           QMessageBox, QGroupBox, QCheckBox, QWidget)

from quangtps.segmentation.auto.model_repository import ModelRepository
from quangtps.common.widgets import IconButton

logger = logging.getLogger(__name__)


class DownloadModelTask(QThread):
    """
    Class thực hiện tải xuống mô hình trong một luồng riêng.
    
    Luồng này được sử dụng để tải xuống mô hình mà không làm đứng
    giao diện người dùng.
    """
    # Tín hiệu cập nhật tiến trình
    progress_updated = pyqtSignal(int)
    # Tín hiệu khi hoàn thành
    task_finished = pyqtSignal(bool, str)
    
    def __init__(self, repository: ModelRepository, model_name: str):
        """
        Khởi tạo task tải xuống mô hình.
        
        Parameters
        ----------
        repository : ModelRepository
            Kho lưu trữ mô hình
        model_name : str
            Tên mô hình cần tải xuống
        """
        super().__init__()
        
        self.repository = repository
        self.model_name = model_name
    
    def run(self):
        """Thực hiện task tải xuống."""
        try:
            # Tiến hành tải xuống với cập nhật tiến trình
            success = self.repository.download_model(
                self.model_name,
                progress_callback=self.progress_updated.emit
            )
            
            # Thông báo kết quả
            if success:
                self.task_finished.emit(True, f"Đã tải xuống mô hình {self.model_name} thành công")
            else:
                self.task_finished.emit(False, f"Lỗi khi tải xuống mô hình {self.model_name}")
                
        except Exception as e:
            logger.error(f"Error in download task: {str(e)}")
            self.task_finished.emit(False, f"Lỗi khi tải xuống: {str(e)}")


class ModelDownloadDialog(QDialog):
    """
    Hộp thoại tải xuống mô hình phân đoạn tự động.
    
    Hộp thoại này cho phép người dùng xem danh sách các mô hình có sẵn
    và tải xuống các mô hình mới.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo hộp thoại tải xuống mô hình.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Khởi tạo kho lưu trữ mô hình
        self.repository = ModelRepository()
        
        # Task tải xuống hiện tại
        self.current_task = None
        
        # Thiết lập hộp thoại
        self.setWindowTitle("Tải xuống mô hình phân đoạn")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Cập nhật danh sách mô hình
        self._update_model_list()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout()
        
        # Nhóm danh sách mô hình
        models_group = QGroupBox("Danh sách mô hình có sẵn")
        models_layout = QVBoxLayout()
        
        # Danh sách mô hình
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SingleSelection)
        self.model_list.itemSelectionChanged.connect(self._update_button_states)
        models_layout.addWidget(self.model_list)
        
        # Nút làm mới danh sách mô hình
        refresh_button = IconButton(icon_name="refresh", tooltip="Làm mới danh sách mô hình")
        refresh_button.clicked.connect(self._refresh_model_list)
        
        # Layout cho các nút
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(refresh_button)
        buttons_layout.addStretch()
        
        models_layout.addLayout(buttons_layout)
        models_group.setLayout(models_layout)
        main_layout.addWidget(models_group)
        
        # Nhóm tải xuống
        download_group = QGroupBox("Tải xuống mô hình")
        download_layout = QVBoxLayout()
        
        # Thanh tiến trình
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        download_layout.addWidget(self.progress_bar)
        
        # Nút tải xuống
        self.download_button = QPushButton("Tải xuống mô hình đã chọn")
        self.download_button.clicked.connect(self._download_selected_model)
        self.download_button.setEnabled(False)
        download_layout.addWidget(self.download_button)
        
        download_group.setLayout(download_layout)
        main_layout.addWidget(download_group)
        
        # Nút đóng
        self.close_button = QPushButton("Đóng")
        self.close_button.clicked.connect(self.close)
        main_layout.addWidget(self.close_button)
        
        # Gán layout cho hộp thoại
        self.setLayout(main_layout)
    
    def _update_model_list(self):
        """Cập nhật danh sách mô hình trong UI."""
        # Xóa các mục hiện tại
        self.model_list.clear()
        
        # Lấy danh sách mô hình
        models = self.repository.get_available_models()
        
        # Thêm các mô hình vào danh sách
        for model in models:
            item = QListWidgetItem()
            
            # Tạo widget cho mục danh sách
            widget = QWidget()
            layout = QHBoxLayout()
            layout.setContentsMargins(5, 5, 5, 5)
            
            # Icon trạng thái
            icon_label = QLabel()
            if model.get('available', False):
                icon_label.setPixmap(QIcon.fromTheme("dialog-ok").pixmap(16, 16))
                item.setData(Qt.UserRole, "installed")
            else:
                icon_label.setPixmap(QIcon.fromTheme("dialog-question").pixmap(16, 16))
                item.setData(Qt.UserRole, "not_installed")
            
            layout.addWidget(icon_label)
            
            # Thông tin mô hình
            info_layout = QVBoxLayout()
            
            # Tên mô hình
            name_label = QLabel(f"<b>{model['name']}</b>")
            info_layout.addWidget(name_label)
            
            # Mô tả mô hình
            if 'description' in model:
                desc_label = QLabel(model['description'])
                info_layout.addWidget(desc_label)
            
            # Thông tin phiên bản
            if 'version' in model:
                version_label = QLabel(f"Phiên bản: {model['version']}")
                info_layout.addWidget(version_label)
            
            layout.addLayout(info_layout)
            layout.addStretch()
            
            # Trạng thái
            status_label = QLabel()
            if model.get('available', False):
                status_label.setText("Đã cài đặt")
            else:
                status_label.setText("Chưa cài đặt")
            
            layout.addWidget(status_label)
            
            widget.setLayout(layout)
            
            # Thiết lập kích thước mục
            item.setSizeHint(widget.sizeHint())
            
            # Lưu tên mô hình
            item.setData(Qt.UserRole + 1, model['name'])
            
            # Thêm vào danh sách
            self.model_list.addItem(item)
            self.model_list.setItemWidget(item, widget)
    
    def _refresh_model_list(self):
        """Làm mới danh sách mô hình từ kho lưu trữ."""
        # Cập nhật danh sách mô hình từ kho từ xa
        success = self.repository.update_model_list(force_reload=True)
        
        if success:
            # Cập nhật UI
            self._update_model_list()
            QMessageBox.information(self, "Thành công", "Đã cập nhật danh sách mô hình")
        else:
            QMessageBox.warning(self, "Cảnh báo", "Không thể cập nhật danh sách mô hình từ kho lưu trữ")
    
    def _update_button_states(self):
        """Cập nhật trạng thái các nút dựa trên lựa chọn hiện tại."""
        # Lấy mục được chọn
        selected_items = self.model_list.selectedItems()
        
        # Kích hoạt/vô hiệu hóa nút tải xuống
        if selected_items:
            item = selected_items[0]
            is_installed = (item.data(Qt.UserRole) == "installed")
            self.download_button.setEnabled(not is_installed and not self._is_task_running())
        else:
            self.download_button.setEnabled(False)
    
    def _is_task_running(self) -> bool:
        """Kiểm tra xem có task nào đang chạy không."""
        return self.current_task is not None and self.current_task.isRunning()
    
    def _download_selected_model(self):
        """Tải xuống mô hình đã chọn."""
        # Kiểm tra xem task đã đang chạy chưa
        if self._is_task_running():
            QMessageBox.warning(self, "Cảnh báo", "Một task tải xuống khác đang chạy")
            return
        
        # Lấy mô hình được chọn
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        model_name = item.data(Qt.UserRole + 1)
        
        # Hiển thị thanh tiến trình
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Vô hiệu hóa các nút trong khi tải xuống
        self.download_button.setEnabled(False)
        self.close_button.setEnabled(False)
        
        # Tạo và bắt đầu task tải xuống
        self.current_task = DownloadModelTask(self.repository, model_name)
        
        # Kết nối các tín hiệu
        self.current_task.progress_updated.connect(self.progress_bar.setValue)
        self.current_task.task_finished.connect(self._handle_download_result)
        
        # Bắt đầu task
        self.current_task.start()
    
    def _handle_download_result(self, success: bool, message: str):
        """
        Xử lý kết quả tải xuống.
        
        Parameters
        ----------
        success : bool
            True nếu tải xuống thành công, False nếu có lỗi
        message : str
            Thông báo kết quả
        """
        # Ẩn thanh tiến trình
        self.progress_bar.setVisible(False)
        
        # Kích hoạt lại các nút
        self.close_button.setEnabled(True)
        
        # Hiển thị thông báo kết quả
        if success:
            QMessageBox.information(self, "Thành công", message)
        else:
            QMessageBox.critical(self, "Lỗi", message)
        
        # Làm sạch task hiện tại
        self.current_task = None
        
        # Cập nhật lại danh sách mô hình
        self._update_model_list()
        
        # Cập nhật trạng thái nút
        self._update_button_states()
