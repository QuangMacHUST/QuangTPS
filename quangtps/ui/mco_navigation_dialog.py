#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog điều hướng tối ưu đa tiêu chí (MCO) cho hệ thống QuangTPS.

Dialog này cho phép người dùng khám phá và điều hướng trên mặt Pareto, điều chỉnh
trọng số của các tiêu chí khác nhau và xem kết quả theo thời gian thực.
"""

import os
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                            QPushButton, QGroupBox, QSplitter, QTabWidget, 
                            QWidget, QComboBox, QFrame, QRadioButton, 
                            QButtonGroup, QMessageBox, QGridLayout, QScrollArea,
                            QSpacerItem, QSizePolicy, QCheckBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from quangtps.optimization.methods.mco import MCOEngine, MCONavigator, MCOTrade
from quangtps.evaluation.dvh.dvh_visualization import plot_dvh
from quangtps.imaging.image_viewer import ImageViewer
from quangtps.dose.dose_visualization import DoseColorwash
from quangtps.core.exceptions import OptimizationError
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

class MCOTradeoffPlot(FigureCanvasQTAgg):
    """Biểu đồ đánh đổi giữa các tiêu chí tối ưu."""
    
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()
        
    def plot_tradeoff(self, x_values, y_values, x_label, y_label, 
                     current_point=None, title=None):
        """Vẽ biểu đồ đánh đổi."""
        self.axes.clear()
        
        # Vẽ các điểm trên mặt Pareto
        self.axes.scatter(x_values, y_values, s=50, alpha=0.7, c='blue')
        
        # Nếu có điểm hiện tại, đánh dấu nó
        if current_point:
            self.axes.scatter([current_point[0]], [current_point[1]], 
                             s=100, c='red', marker='*')
        
        # Đặt nhãn và tiêu đề
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        if title:
            self.axes.set_title(title)
        else:
            self.axes.set_title(f"Tradeoff: {x_label} vs {y_label}")
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.fig.tight_layout()
        self.draw()


class ObjectiveSlider(QWidget):
    """Widget thanh trượt cho một tiêu chí tối ưu."""
    
    valueChanged = pyqtSignal(str, float)
    
    def __init__(self, objective_name, min_value=0, max_value=100, 
                default_value=50, description=None, parent=None):
        super().__init__(parent)
        
        self.objective_name = objective_name
        self.description = description or objective_name
        
        # Tạo layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tạo label hiển thị tên và giá trị
        self.value_label = QLabel(f"{self.description}: {default_value}%")
        layout.addWidget(self.value_label)
        
        # Tạo thanh trượt
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setValue(default_value)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self._value_changed)
        layout.addWidget(self.slider)
        
    def _value_changed(self, value):
        """Xử lý khi giá trị thanh trượt thay đổi."""
        self.value_label.setText(f"{self.description}: {value}%")
        self.valueChanged.emit(self.objective_name, value / 100.0)  # Chuẩn hóa về [0, 1]
        
    def get_value(self):
        """Lấy giá trị hiện tại."""
        return self.slider.value() / 100.0
    
    def set_value(self, value):
        """Đặt giá trị."""
        self.slider.setValue(int(value * 100))


class MCONavigationDialog(QDialog):
    """
    Dialog điều hướng tối ưu đa tiêu chí.
    
    Dialog này cho phép người dùng điều chỉnh trọng số giữa các tiêu chí tối ưu
    và khám phá không gian các kế hoạch khả thi.
    """
    
    tradeAccepted = pyqtSignal(MCOTrade)
    
    def __init__(self, mco_engine, parent=None):
        """
        Khởi tạo dialog điều hướng MCO.
        
        Args:
            mco_engine: Động cơ tối ưu đa tiêu chí
            parent: Widget cha
        """
        super().__init__(parent)
        
        self.setWindowTitle("Điều Hướng Tối Ưu Đa Tiêu Chí (MCO)")
        self.setMinimumSize(1200, 800)
        
        self.mco_engine = mco_engine
        self.navigator = MCONavigator(mco_engine)
        
        self.current_trade = None
        self.objective_sliders = {}
        self.selected_objectives = []
        
        self.init_ui()
        
        # Thiết lập hẹn giờ để cập nhật kế hoạch khi người dùng điều chỉnh
        self.update_timer = QTimer()
        self.update_timer.setInterval(500)  # 500ms
        self.update_timer.timeout.connect(self.delayed_update_plan)
        self.update_pending = False
        
    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Splitter chính chia đôi giao diện
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Phần điều khiển bên trái
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Phần điều chỉnh trọng số
        weights_group = QGroupBox("Điều Chỉnh Trọng Số")
        weights_layout = QVBoxLayout(weights_group)
        
        # Tạo scroll area cho các thanh trượt
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Khởi tạo các thanh trượt cho các mục tiêu
        for obj in self.mco_engine.objectives:
            if obj.show_in_navigation:
                slider = ObjectiveSlider(
                    obj.name, 
                    default_value=int(obj.current_weight * 100),
                    description=obj.name
                )
                slider.valueChanged.connect(self.on_weight_changed)
                
                self.objective_sliders[obj.name] = slider
                scroll_layout.addWidget(slider)
        
        scroll_area.setWidget(scroll_content)
        weights_layout.addWidget(scroll_area)
        
        # Nút Reset và Cân bằng
        buttons_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Đặt Lại")
        self.reset_button.clicked.connect(self.reset_weights)
        buttons_layout.addWidget(self.reset_button)
        
        self.balance_button = QPushButton("Cân Bằng")
        self.balance_button.clicked.connect(self.balance_weights)
        buttons_layout.addWidget(self.balance_button)
        
        weights_layout.addLayout(buttons_layout)
        left_layout.addWidget(weights_group)
        
        # Phần lựa chọn tiêu chí để hiển thị
        objectives_group = QGroupBox("Hiển Thị Đánh Đổi")
        objectives_layout = QVBoxLayout(objectives_group)
        
        # Combobox cho X và Y
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("Trục X:"))
        self.x_combo = QComboBox()
        x_layout.addWidget(self.x_combo)
        objectives_layout.addLayout(x_layout)
        
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Trục Y:"))
        self.y_combo = QComboBox()
        y_layout.addWidget(self.y_combo)
        objectives_layout.addLayout(y_layout)
        
        # Nút cập nhật biểu đồ
        self.update_plot_button = QPushButton("Cập Nhật Biểu Đồ")
        self.update_plot_button.clicked.connect(self.update_plot)
        objectives_layout.addWidget(self.update_plot_button)
        
        left_layout.addWidget(objectives_group)
        
        # Nút điều khiển chính
        controls_layout = QHBoxLayout()
        
        self.accept_plan_button = QPushButton("Chấp Nhận Kế Hoạch")
        self.accept_plan_button.clicked.connect(self.accept_plan)
        controls_layout.addWidget(self.accept_plan_button)
        
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self.reject)
        controls_layout.addWidget(self.cancel_button)
        
        left_layout.addLayout(controls_layout)
        
        # Thêm spacer để đẩy các widget lên trên
        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Phần hiển thị bên phải
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Tab widget để chứa các visualizations
        self.results_tabs = QTabWidget()
        
        # Tab đánh đổi
        tradeoff_tab = QWidget()
        tradeoff_layout = QVBoxLayout(tradeoff_tab)
        
        self.tradeoff_plot = MCOTradeoffPlot(width=6, height=5)
        tradeoff_layout.addWidget(self.tradeoff_plot)
        
        self.results_tabs.addTab(tradeoff_tab, "Biểu Đồ Đánh Đổi")
        
        # Tab DVH
        dvh_tab = QWidget()
        self.results_tabs.addTab(dvh_tab, "DVH")
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # Tạo figure matplotlib cho DVH
        self.dvh_fig = Figure(figsize=(5, 4), dpi=100)
        self.dvh_canvas = FigureCanvasQTAgg(self.dvh_fig)
        self.dvh_axes = self.dvh_fig.add_subplot(111)
        dvh_layout.addWidget(self.dvh_canvas)
        
        # Tab Dose
        dose_tab = QWidget()
        dose_layout = QVBoxLayout(dose_tab)
        
        # Widget hiển thị liều
        self.dose_colorwash = DoseColorwash()
        self.dose_img = ImageViewer()
        dose_layout.addWidget(self.dose_img)
        
        self.results_tabs.addTab(dose_tab, "Phân Bố Liều")
        
        # Thêm tabs vào layout bên phải
        right_layout.addWidget(self.results_tabs)
        
        # Thêm widgets vào splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        
        # Thiết lập kích thước khởi tạo cho splitter
        main_splitter.setSizes([400, 800])
        
        # Thêm các mục tiêu vào combo boxes
        self.populate_objective_combos()
        
        # Khởi tạo biểu đồ ban đầu nếu có dữ liệu
        if self.mco_engine.trades:
            self.update_plot()
    
    def populate_objective_combos(self):
        """Thêm các mục tiêu vào combo boxes."""
        objectives = [obj.name for obj in self.mco_engine.objectives if obj.show_in_navigation]
        
        self.x_combo.clear()
        self.y_combo.clear()
        
        self.x_combo.addItems(objectives)
        self.y_combo.addItems(objectives)
        
        # Chọn mục tiêu mặc định cho x và y
        if len(objectives) >= 2:
            self.x_combo.setCurrentIndex(0)
            self.y_combo.setCurrentIndex(1)
        
    def on_weight_changed(self, objective_name, value):
        """Xử lý khi trọng số thay đổi."""
        # Lên lịch cập nhật kế hoạch
        self.update_pending = True
        self.update_timer.start()
    
    def delayed_update_plan(self):
        """Cập nhật kế hoạch sau một khoảng thời gian."""
        if self.update_pending:
            # Dừng hẹn giờ
            self.update_timer.stop()
            self.update_pending = False
            
            # Thu thập trọng số hiện tại
            weights = {}
            for obj_name, slider in self.objective_sliders.items():
                weights[obj_name] = slider.get_value()
            
            # Chuẩn hóa trọng số
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            
            # Cập nhật kế hoạch
            try:
                self.current_trade = self.navigator.update_weights(weights)
                
                # Cập nhật hiển thị
                self.update_dvh()
                self.update_dose_view()
                
                # Cập nhật điểm hiện tại trên biểu đồ đánh đổi
                self.update_current_point()
                
            except OptimizationError as e:
                QMessageBox.warning(self, "Lỗi Cập Nhật", str(e))
    
    def reset_weights(self):
        """Đặt lại trọng số về giá trị ban đầu."""
        for obj in self.mco_engine.objectives:
            if obj.name in self.objective_sliders:
                self.objective_sliders[obj.name].set_value(obj.current_weight)
        
        # Cập nhật kế hoạch
        self.delayed_update_plan()
    
    def balance_weights(self):
        """Đặt các trọng số bằng nhau."""
        if not self.objective_sliders:
            return
        
        # Chia đều trọng số
        num_objectives = len(self.objective_sliders)
        equal_weight = 1.0 / num_objectives
        
        for slider in self.objective_sliders.values():
            slider.set_value(equal_weight)
        
        # Cập nhật kế hoạch
        self.delayed_update_plan()
    
    def update_plot(self):
        """Cập nhật biểu đồ đánh đổi."""
        if not self.mco_engine.trades:
            return
        
        # Lấy mục tiêu đã chọn
        x_objective = self.x_combo.currentText()
        y_objective = self.y_combo.currentText()
        
        if not x_objective or not y_objective:
            return
        
        # Lấy dữ liệu
        x_values = []
        y_values = []
        
        for trade in self.mco_engine.trades:
            if x_objective in trade.objective_values and y_objective in trade.objective_values:
                x_values.append(trade.objective_values[x_objective])
                y_values.append(trade.objective_values[y_objective])
        
        # Vẽ biểu đồ
        current_point = None
        if self.current_trade:
            if x_objective in self.current_trade.objective_values and y_objective in self.current_trade.objective_values:
                current_point = (
                    self.current_trade.objective_values[x_objective],
                    self.current_trade.objective_values[y_objective]
                )
        
        self.tradeoff_plot.plot_tradeoff(
            x_values, y_values, 
            x_objective, y_objective,
            current_point=current_point,
            title=f"Tradeoff: {x_objective} vs {y_objective}"
        )
    
    def update_current_point(self):
        """Cập nhật điểm hiện tại trên biểu đồ đánh đổi."""
        # Kiểm tra xem có biểu đồ hiện tại không
        if not hasattr(self.tradeoff_plot.axes, 'collections') or len(self.tradeoff_plot.axes.collections) < 2:
            # Nếu không, cập nhật toàn bộ biểu đồ
            self.update_plot()
            return
        
        # Lấy mục tiêu đã chọn
        x_objective = self.x_combo.currentText()
        y_objective = self.y_combo.currentText()
        
        if not x_objective or not y_objective or not self.current_trade:
            return
        
        # Cập nhật vị trí điểm hiện tại
        if x_objective in self.current_trade.objective_values and y_objective in self.current_trade.objective_values:
            current_x = self.current_trade.objective_values[x_objective]
            current_y = self.current_trade.objective_values[y_objective]
            
            # Cập nhật vị trí điểm hiện tại (điểm thứ hai trong collections)
            if len(self.tradeoff_plot.axes.collections) >= 2:
                self.tradeoff_plot.axes.collections[1].set_offsets([(current_x, current_y)])
                self.tradeoff_plot.draw()
    
    def update_dvh(self):
        """Cập nhật biểu đồ DVH."""
        if not self.current_trade or not self.current_trade.dvh_data:
            return
        
        self.dvh_axes.clear()
        
        # Thêm các đường DVH
        for struct_name, dvh_data in self.current_trade.dvh_data.items():
            self.dvh_axes.plot(
                dvh_data['dose'],
                dvh_data['volume_percent'],
                label=struct_name
            )
        
        self.dvh_axes.legend()
        self.dvh_axes.set_xlabel('Dose (Gy)')
        self.dvh_axes.set_ylabel('Volume Percent')
        self.dvh_axes.set_title('DVH')
        self.dvh_canvas.draw()
    
    def update_dose_view(self):
        """Cập nhật hiển thị phân bố liều."""
        if not hasattr(self, 'current_trade') or self.current_trade is None:
            return
        
        # Lấy phân bố liều từ MCOTrade hiện tại
        dose_grid = self.current_trade.dose_grid
        if dose_grid is None:
            return
        
        # Hiển thị lát cắt giữa của phân bố liều
        dose_array = dose_grid.dose_array
        if dose_array.ndim == 3:
            slice_idx = dose_array.shape[2] // 2
            dose_slice = dose_array[:, :, slice_idx]
            
            # Tạo một hình từ lát cắt dose
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            # Sử dụng DoseColorwash để hiển thị màu liều
            self.dose_colorwash.display_2d(
                dose_slice=dose_slice,
                figure=fig,
                dose_max=dose_array.max() if dose_array.max() > 0 else None
            )
            
            # Chuyển đổi figure matplotlib thành hình ảnh để hiển thị
            fig.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (h, w, 3)
            
            # Hiển thị hình ảnh trong ImageViewer
            self.dose_img.set_image(buf)
            self.dose_img.update_view()
    
    def accept_plan(self):
        """Chấp nhận kế hoạch hiện tại."""
        if not self.current_trade:
            QMessageBox.warning(self, "Cảnh Báo", "Không có kế hoạch hiện tại để chấp nhận.")
            return
        
        # Phát tín hiệu với trade đã chọn
        self.tradeAccepted.emit(self.current_trade)
        
        # Đóng dialog
        self.accept()
    
    def export_tradeoff_plot(self):
        """Xuất biểu đồ đánh đổi ra file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Biểu Đồ", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)"
        )
        
        if file_path:
            self.tradeoff_plot.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Thông Báo", f"Đã lưu biểu đồ vào {file_path}")
    
    def export_dvh_plot(self):
        """Xuất biểu đồ DVH ra file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Biểu Đồ DVH", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)"
        )
        
        if file_path:
            self.dvh_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Thông Báo", f"Đã lưu biểu đồ DVH vào {file_path}") 