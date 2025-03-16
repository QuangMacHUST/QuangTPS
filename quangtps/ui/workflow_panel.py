#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý luồng công việc (workflow) cho QuangTPS.

Module này cung cấp giao diện tích hợp để hướng dẫn người dùng qua các bước 
trong quy trình lập kế hoạch xạ trị, từ nhập dữ liệu, phân đoạn cấu trúc, 
lập kế hoạch, tính toán liều đến đánh giá và xuất kế hoạch.
"""

import os
import logging
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QStackedWidget, QFrame, QScrollArea, QSplitter,
    QSizePolicy, QSpacerItem, QGroupBox, QMessageBox, QProgressBar,
    QDialog, QToolButton, QMenu, QAction, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap

from quangtps.core.logging import get_logger
from quangtps.ui.dicom_loader import DicomLoaderWidget
from quangtps.ui.image_viewer import ImageViewer
from quangtps.ui.structure_view import StructureView

logger = get_logger(__name__)


class WorkflowStep(Enum):
    """Định nghĩa các bước trong quy trình lập kế hoạch xạ trị."""
    DATA_IMPORT = auto()
    CONTOURING = auto()
    PLANNING_SETUP = auto()
    DOSE_CALCULATION = auto()
    PLAN_OPTIMIZATION = auto()
    EVALUATION = auto()
    PLAN_APPROVAL = auto()
    REPORTING = auto()


class WorkflowPanel(QWidget):
    """
    Panel quản lý luồng công việc (workflow).
    
    Widget này hiển thị các bước trong quy trình lập kế hoạch xạ trị
    và cho phép người dùng điều hướng qua các bước.
    """
    
    # Tín hiệu khi thay đổi bước
    step_changed = pyqtSignal(WorkflowStep)
    
    def __init__(self, parent=None):
        """Khởi tạo workflow panel."""
        super().__init__(parent)
        
        # Khởi tạo các thuộc tính
        self.current_step = None
        self.steps = []
        self.step_widgets = {}
        self.step_buttons = {}
        self.step_status = {}
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Thiết lập bước mặc định
        self.set_current_step(WorkflowStep.DATA_IMPORT)
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tiêu đề
        title_label = QLabel("Quy trình lập kế hoạch")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("padding: 5px; background-color: #2a82da; color: white;")
        main_layout.addWidget(title_label)
        
        # Tạo các bước
        self._create_workflow_steps()
        
        # Widget hiển thị các bước
        step_widget = QWidget()
        step_layout = QVBoxLayout(step_widget)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.setSpacing(1)
        
        # Thêm các nút bước
        for step in self.steps:
            button = self._create_step_button(step)
            step_layout.addWidget(button)
            self.step_buttons[step] = button
        
        # Thêm khoảng trống
        step_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Nút điều hướng
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        
        self.prev_button = QPushButton("Trước")
        self.prev_button.clicked.connect(self._on_prev_step)
        
        self.next_button = QPushButton("Tiếp theo")
        self.next_button.clicked.connect(self._on_next_step)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        
        # Thêm các widget vào layout chính
        main_layout.addWidget(step_widget)
        main_layout.addWidget(nav_widget)
    
    def _create_workflow_steps(self):
        """Tạo các bước trong quy trình làm việc."""
        self.steps = [
            WorkflowStep.DATA_IMPORT,
            WorkflowStep.CONTOURING,
            WorkflowStep.PLANNING_SETUP,
            WorkflowStep.DOSE_CALCULATION,
            WorkflowStep.PLAN_OPTIMIZATION,
            WorkflowStep.EVALUATION,
            WorkflowStep.PLAN_APPROVAL,
            WorkflowStep.REPORTING
        ]
        
        # Khởi tạo trạng thái của các bước
        for step in self.steps:
            self.step_status[step] = {
                "completed": False,
                "current": False,
                "enabled": False
            }
        
        # Bước đầu tiên luôn được kích hoạt
        self.step_status[WorkflowStep.DATA_IMPORT]["enabled"] = True
    
    def _create_step_button(self, step: WorkflowStep) -> QPushButton:
        """
        Tạo nút cho một bước trong quy trình.
        
        Args:
            step: Bước trong quy trình
            
        Returns:
            Nút đại diện cho bước
        """
        # Lấy tên hiển thị của bước
        step_name = self._get_step_display_name(step)
        
        # Tạo nút
        button = QPushButton(step_name)
        button.setCheckable(True)
        button.setFlat(True)
        button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                border: none;
                border-left: 5px solid transparent;
            }
            QPushButton:checked {
                background-color: #e6f2ff;
                border-left: 5px solid #2a82da;
                font-weight: bold;
            }
            QPushButton:disabled {
                color: #888888;
            }
        """)
        
        # Kết nối sự kiện
        button.clicked.connect(lambda: self._on_step_button_clicked(step))
        
        return button
    
    def _get_step_display_name(self, step: WorkflowStep) -> str:
        """
        Lấy tên hiển thị của bước.
        
        Args:
            step: Bước trong quy trình
            
        Returns:
            Tên hiển thị của bước
        """
        step_names = {
            WorkflowStep.DATA_IMPORT: "1. Nhập dữ liệu",
            WorkflowStep.CONTOURING: "2. Phân đoạn cấu trúc",
            WorkflowStep.PLANNING_SETUP: "3. Thiết lập kế hoạch",
            WorkflowStep.DOSE_CALCULATION: "4. Tính toán liều",
            WorkflowStep.PLAN_OPTIMIZATION: "5. Tối ưu hóa kế hoạch",
            WorkflowStep.EVALUATION: "6. Đánh giá kế hoạch",
            WorkflowStep.PLAN_APPROVAL: "7. Phê duyệt kế hoạch",
            WorkflowStep.REPORTING: "8. Báo cáo"
        }
        
        return step_names.get(step, str(step))
    
    def _on_step_button_clicked(self, step: WorkflowStep):
        """
        Xử lý sự kiện khi nút bước được nhấn.
        
        Args:
            step: Bước được chọn
        """
        if self.step_status[step]["enabled"]:
            self.set_current_step(step)
    
    def _on_prev_step(self):
        """Xử lý sự kiện khi nút Trước được nhấn."""
        if self.current_step is None:
            return
            
        # Tìm bước trước
        current_index = self.steps.index(self.current_step)
        if current_index > 0:
            prev_step = self.steps[current_index - 1]
            self.set_current_step(prev_step)
    
    def _on_next_step(self):
        """Xử lý sự kiện khi nút Tiếp theo được nhấn."""
        if self.current_step is None:
            return
            
        # Tìm bước tiếp theo
        current_index = self.steps.index(self.current_step)
        if current_index < len(self.steps) - 1:
            next_step = self.steps[current_index + 1]
            
            # Kích hoạt bước tiếp theo
            self.step_status[next_step]["enabled"] = True
            
            # Đánh dấu bước hiện tại là đã hoàn thành
            self.step_status[self.current_step]["completed"] = True
            
            # Chuyển đến bước tiếp theo
            self.set_current_step(next_step)
    
    def set_current_step(self, step: WorkflowStep):
        """
        Thiết lập bước hiện tại.
        
        Args:
            step: Bước cần thiết lập
        """
        if self.current_step == step:
            return
            
        # Cập nhật trạng thái
        if self.current_step is not None:
            self.step_status[self.current_step]["current"] = False
            self.step_buttons[self.current_step].setChecked(False)
        
        self.current_step = step
        self.step_status[step]["current"] = True
        self.step_buttons[step].setChecked(True)
        
        # Cập nhật nút điều hướng
        current_index = self.steps.index(step)
        self.prev_button.setEnabled(current_index > 0)
        self.next_button.setEnabled(current_index < len(self.steps) - 1)
        
        # Phát tín hiệu
        self.step_changed.emit(step)
        
        logger.info(f"Đã chuyển sang bước: {self._get_step_display_name(step)}")
    
    def set_step_completed(self, step: WorkflowStep, completed: bool = True):
        """
        Đánh dấu một bước là đã hoàn thành.
        
        Args:
            step: Bước cần đánh dấu
            completed: Trạng thái hoàn thành
        """
        self.step_status[step]["completed"] = completed
        
        # Kích hoạt bước tiếp theo nếu bước hiện tại đã hoàn thành
        if completed:
            current_index = self.steps.index(step)
            if current_index < len(self.steps) - 1:
                next_step = self.steps[current_index + 1]
                self.step_status[next_step]["enabled"] = True
                
                # Cập nhật trạng thái nút
                self.step_buttons[next_step].setEnabled(True)
    
    def reset_workflow(self):
        """Reset trạng thái quy trình làm việc."""
        for step in self.steps:
            self.step_status[step] = {
                "completed": False,
                "current": False,
                "enabled": False
            }
        
        # Bước đầu tiên luôn được kích hoạt
        self.step_status[WorkflowStep.DATA_IMPORT]["enabled"] = True
        
        # Thiết lập bước đầu tiên là bước hiện tại
        self.set_current_step(WorkflowStep.DATA_IMPORT)


class WorkflowManager(QWidget):
    """
    Trình quản lý luồng công việc (workflow).
    
    Widget này tích hợp panel workflow với các widget tương ứng
    cho từng bước trong quy trình làm việc.
    """
    
    def __init__(self, parent=None):
        """Khởi tạo workflow manager."""
        super().__init__(parent)
        
        # Khởi tạo các thuộc tính
        self.workflow_panel = WorkflowPanel()
        self.content_stack = QStackedWidget()
        self.step_widgets = {}
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Kết nối tín hiệu
        self.workflow_panel.step_changed.connect(self._on_step_changed)
        
        # Khởi tạo các widget cho từng bước
        self._init_step_widgets()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tạo splitter để điều chỉnh kích thước
        splitter = QSplitter(Qt.Horizontal)
        
        # Thêm workflow panel bên trái
        splitter.addWidget(self.workflow_panel)
        
        # Thêm content stack bên phải
        splitter.addWidget(self.content_stack)
        
        # Thiết lập kích thước ban đầu
        splitter.setSizes([200, 800])
        
        # Thêm splitter vào layout chính
        main_layout.addWidget(splitter)
    
    def _init_step_widgets(self):
        """Khởi tạo các widget cho từng bước."""
        # Widget nhập dữ liệu
        data_import_widget = DicomLoaderWidget()
        self.add_step_widget(WorkflowStep.DATA_IMPORT, data_import_widget)
        
        # Widget phân đoạn cấu trúc
        contouring_widget = QWidget()
        contouring_layout = QHBoxLayout(contouring_widget)
        
        # Bên trái: ImageViewer
        image_viewer = ImageViewer()
        
        # Bên phải: StructureView
        structure_view = StructureView()
        
        contouring_layout.addWidget(image_viewer, 2)
        contouring_layout.addWidget(structure_view, 1)
        
        self.add_step_widget(WorkflowStep.CONTOURING, contouring_widget)
        
        # Widget thiết lập kế hoạch
        planning_widget = QWidget()
        planning_layout = QVBoxLayout(planning_widget)
        planning_layout.addWidget(QLabel("Thiết lập kế hoạch (Đang phát triển)"))
        self.add_step_widget(WorkflowStep.PLANNING_SETUP, planning_widget)
        
        # Widget tính toán liều
        dose_widget = QWidget()
        dose_layout = QVBoxLayout(dose_widget)
        dose_layout.addWidget(QLabel("Tính toán liều (Đang phát triển)"))
        self.add_step_widget(WorkflowStep.DOSE_CALCULATION, dose_widget)
        
        # Widget tối ưu hóa kế hoạch
        optimization_widget = QWidget()
        optimization_layout = QVBoxLayout(optimization_widget)
        optimization_layout.addWidget(QLabel("Tối ưu hóa kế hoạch (Đang phát triển)"))
        self.add_step_widget(WorkflowStep.PLAN_OPTIMIZATION, optimization_widget)
        
        # Widget đánh giá kế hoạch
        evaluation_widget = QWidget()
        evaluation_layout = QVBoxLayout(evaluation_widget)
        evaluation_layout.addWidget(QLabel("Đánh giá kế hoạch (Đang phát triển)"))
        self.add_step_widget(WorkflowStep.EVALUATION, evaluation_widget)
        
        # Widget phê duyệt kế hoạch
        approval_widget = QWidget()
        approval_layout = QVBoxLayout(approval_widget)
        approval_layout.addWidget(QLabel("Phê duyệt kế hoạch (Đang phát triển)"))
        self.add_step_widget(WorkflowStep.PLAN_APPROVAL, approval_widget)
        
        # Widget báo cáo
        reporting_widget = QWidget()
        reporting_layout = QVBoxLayout(reporting_widget)
        reporting_layout.addWidget(QLabel("Báo cáo (Đang phát triển)"))
        self.add_step_widget(WorkflowStep.REPORTING, reporting_widget)
    
    def add_step_widget(self, step: WorkflowStep, widget: QWidget):
        """
        Thêm widget cho một bước.
        
        Args:
            step: Bước cần thêm widget
            widget: Widget của bước
        """
        self.step_widgets[step] = widget
        self.content_stack.addWidget(widget)
    
    def _on_step_changed(self, step: WorkflowStep):
        """
        Xử lý sự kiện khi bước thay đổi.
        
        Args:
            step: Bước mới
        """
        # Chuyển đến widget tương ứng
        if step in self.step_widgets:
            widget = self.step_widgets[step]
            self.content_stack.setCurrentWidget(widget)
    
    def reset(self):
        """Reset trình quản lý workflow."""
        self.workflow_panel.reset_workflow()
        
        # Reset các widget riêng lẻ
        # TODO: Thêm các phương thức reset cho từng widget
