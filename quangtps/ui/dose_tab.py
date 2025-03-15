#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab liều lượng (Dose Tab) cho QuangTPS.

Module này cung cấp giao diện để hiển thị và phân tích phân bố liều lượng
trong kế hoạch xạ trị, bao gồm các công cụ như DVH, phân tích liều, và so sánh liều.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
    QSlider, QColorDialog, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPen

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class DoseVisualizationWidget(QWidget):
    """Widget để hiển thị phân bố liều trong không gian 2D và 3D."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị liều.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        if MATPLOTLIB_AVAILABLE:
            # Tạo figure cho matplotlib
            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            # Thêm subplot cho hiển thị 2D
            self.ax = self.figure.add_subplot(111)
            
            # Thêm vào layout
            self.main_layout.addWidget(self.toolbar)
            self.main_layout.addWidget(self.canvas)
            
            # Label ban đầu
            self.ax.set_xlabel('X (mm)')
            self.ax.set_ylabel('Y (mm)')
            self.ax.set_title('Phân bố liều')
            self.canvas.draw()
        else:
            # Label thông báo nếu không có matplotlib
            self.label = QLabel("Thư viện matplotlib không khả dụng để hiển thị phân bố liều")
            self.label.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.label)
    
    def display_dose_2d(self, dose_data, x_coords, y_coords, colormap='jet'):
        """
        Hiển thị phân bố liều 2D.
        
        Parameters
        ----------
        dose_data : numpy.ndarray
            Dữ liệu phân bố liều 2D
        x_coords : numpy.ndarray
            Tọa độ X
        y_coords : numpy.ndarray
            Tọa độ Y
        colormap : str, optional
            Bảng màu
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        
        # Tạo hình ảnh phân bố liều
        im = self.ax.imshow(
            dose_data,
            extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
            origin='lower',
            cmap=colormap,
            interpolation='bilinear'
        )
        
        # Thêm colorbar
        self.figure.colorbar(im, ax=self.ax, label='Liều (Gy)')
        
        # Cập nhật nhãn
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_title('Phân bố liều')
        
        # Vẽ lại canvas
        self.canvas.draw()
    
    def clear(self):
        """Xóa hiển thị."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_title('Phân bố liều')
        self.canvas.draw()


class DVHWidget(QWidget):
    """Widget để hiển thị biểu đồ thể tích liều (DVH)."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget DVH.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        if MATPLOTLIB_AVAILABLE:
            # Tạo figure cho matplotlib
            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            # Thêm subplot cho DVH
            self.ax = self.figure.add_subplot(111)
            
            # Thêm vào layout
            self.main_layout.addWidget(self.toolbar)
            self.main_layout.addWidget(self.canvas)
            
            # Label ban đầu
            self.ax.set_xlabel('Liều (Gy)')
            self.ax.set_ylabel('Thể tích (%)')
            self.ax.set_title('Biểu đồ thể tích liều (DVH)')
            self.ax.grid(True)
            self.canvas.draw()
        else:
            # Label thông báo nếu không có matplotlib
            self.label = QLabel("Thư viện matplotlib không khả dụng để hiển thị DVH")
            self.label.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.label)
    
    def plot_dvh(self, structures, dose_bins, cumulative=True):
        """
        Vẽ biểu đồ DVH.
        
        Parameters
        ----------
        structures : list
            Danh sách các cấu trúc và dữ liệu DVH của chúng
        dose_bins : numpy.ndarray
            Các bin liều lượng
        cumulative : bool, optional
            Nếu True, vẽ DVH tích lũy, ngược lại vẽ DVH vi phân
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, structure in enumerate(structures):
            color = colors[i % len(colors)]
            if cumulative:
                self.ax.plot(
                    dose_bins, 
                    structure['dvh'], 
                    label=structure['name'], 
                    color=color,
                    linewidth=2
                )
            else:
                # Tính DVH vi phân
                dvh_diff = np.diff(structure['dvh'])
                dvh_diff = np.append(dvh_diff, 0)  # Thêm giá trị 0 cuối cùng
                self.ax.plot(
                    dose_bins, 
                    -dvh_diff,  # Đảo dấu vì DVH tích lũy giảm dần
                    label=structure['name'], 
                    color=color,
                    linewidth=2
                )
        
        # Cập nhật nhãn
        self.ax.set_xlabel('Liều (Gy)')
        self.ax.set_ylabel('Thể tích (%)')
        if cumulative:
            self.ax.set_title('Biểu đồ thể tích liều tích lũy (cDVH)')
        else:
            self.ax.set_title('Biểu đồ thể tích liều vi phân (dDVH)')
        
        self.ax.grid(True)
        self.ax.legend(loc='best')
        
        # Đặt giới hạn trục
        self.ax.set_xlim(0, dose_bins.max())
        self.ax.set_ylim(0, 105)
        
        # Vẽ lại canvas
        self.canvas.draw()
    
    def clear(self):
        """Xóa biểu đồ."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        self.ax.set_xlabel('Liều (Gy)')
        self.ax.set_ylabel('Thể tích (%)')
        self.ax.set_title('Biểu đồ thể tích liều (DVH)')
        self.ax.grid(True)
        self.canvas.draw()


class DoseStatisticsWidget(QWidget):
    """Widget để hiển thị thống kê liều lượng của các cấu trúc."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget thống kê liều.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Bảng thống kê liều
        self.stats_table = QTableWidget(0, 7)
        self.stats_table.setHorizontalHeaderLabels([
            "Cấu trúc", "Min (Gy)", "Max (Gy)", "Mean (Gy)", 
            "D95 (Gy)", "V20 (%)", "Thể tích (cc)"
        ])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.main_layout.addWidget(self.stats_table)
        
        # Nút cập nhật thống kê
        self.update_stats_button = QPushButton("Cập nhật thống kê")
        self.update_stats_button.clicked.connect(self._update_statistics)
        self.main_layout.addWidget(self.update_stats_button, alignment=Qt.AlignRight)
    
    def set_structures(self, structures):
        """
        Thiết lập danh sách cấu trúc và cập nhật bảng thống kê.
        
        Parameters
        ----------
        structures : list
            Danh sách các cấu trúc
        """
        # Xóa dữ liệu cũ
        self.stats_table.setRowCount(0)
        
        # Thêm dữ liệu mới
        for i, structure in enumerate(structures):
            self.stats_table.insertRow(i)
            
            # Tên cấu trúc
            self.stats_table.setItem(i, 0, QTableWidgetItem(structure['name']))
            
            # Các thống kê cơ bản (giả định)
            self.stats_table.setItem(i, 1, QTableWidgetItem(f"{structure.get('min_dose', 0):.2f}"))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{structure.get('max_dose', 0):.2f}"))
            self.stats_table.setItem(i, 3, QTableWidgetItem(f"{structure.get('mean_dose', 0):.2f}"))
            self.stats_table.setItem(i, 4, QTableWidgetItem(f"{structure.get('d95', 0):.2f}"))
            self.stats_table.setItem(i, 5, QTableWidgetItem(f"{structure.get('v20', 0):.2f}"))
            self.stats_table.setItem(i, 6, QTableWidgetItem(f"{structure.get('volume', 0):.2f}"))
    
    def _update_statistics(self):
        """Cập nhật thống kê liều."""
        # Trong một ứng dụng thực tế, điều này sẽ truy xuất dữ liệu mới nhất
        # và cập nhật bảng thống kê
        logging.info("Cập nhật thống kê liều")
    
    def clear(self):
        """Xóa bảng thống kê."""
        self.stats_table.setRowCount(0)


class DoseComparisonWidget(QWidget):
    """Widget để so sánh các phân bố liều khác nhau."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget so sánh liều.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Nhóm lựa chọn kế hoạch
        self.plan_selection_group = QGroupBox("Lựa chọn kế hoạch")
        self.plan_selection_layout = QHBoxLayout(self.plan_selection_group)
        
        self.plan_1_combo = QComboBox()
        self.plan_selection_layout.addWidget(QLabel("Kế hoạch 1:"))
        self.plan_selection_layout.addWidget(self.plan_1_combo)
        
        self.plan_2_combo = QComboBox()
        self.plan_selection_layout.addWidget(QLabel("Kế hoạch 2:"))
        self.plan_selection_layout.addWidget(self.plan_2_combo)
        
        self.compare_button = QPushButton("So sánh")
        self.compare_button.clicked.connect(self._compare_plans)
        self.plan_selection_layout.addWidget(self.compare_button)
        
        self.main_layout.addWidget(self.plan_selection_group)
        
        if MATPLOTLIB_AVAILABLE:
            # Tạo figure cho matplotlib
            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            # Thêm subplot
            self.ax1 = self.figure.add_subplot(121)
            self.ax2 = self.figure.add_subplot(122)
            
            # Thêm vào layout
            self.main_layout.addWidget(self.toolbar)
            self.main_layout.addWidget(self.canvas)
            
            # Label ban đầu
            self.ax1.set_title('Kế hoạch 1')
            self.ax2.set_title('Kế hoạch 2')
            self.canvas.draw()
        else:
            # Label thông báo nếu không có matplotlib
            self.label = QLabel("Thư viện matplotlib không khả dụng để so sánh phân bố liều")
            self.label.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(self.label)
    
    def set_plans(self, plans):
        """
        Thiết lập danh sách kế hoạch.
        
        Parameters
        ----------
        plans : list
            Danh sách các kế hoạch
        """
        # Xóa danh sách cũ
        self.plan_1_combo.clear()
        self.plan_2_combo.clear()
        
        # Thêm kế hoạch vào combo box
        for plan in plans:
            self.plan_1_combo.addItem(plan['name'], plan['id'])
            self.plan_2_combo.addItem(plan['name'], plan['id'])
    
    def _compare_plans(self):
        """So sánh hai kế hoạch."""
        # Trong một ứng dụng thực tế, điều này sẽ so sánh hai kế hoạch
        # và hiển thị kết quả
        logging.info("So sánh kế hoạch")
        
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Xóa dữ liệu cũ
        self.ax1.clear()
        self.ax2.clear()
        
        # Thiết lập nhãn
        self.ax1.set_title(f"Kế hoạch: {self.plan_1_combo.currentText()}")
        self.ax2.set_title(f"Kế hoạch: {self.plan_2_combo.currentText()}")
        
        # Mô phỏng dữ liệu - trong ứng dụng thực tế, dữ liệu sẽ đến từ kế hoạch
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z1 = np.sin(X) * np.cos(Y)
        Z2 = np.cos(X) * np.sin(Y)
        
        # Vẽ dữ liệu
        im1 = self.ax1.imshow(Z1, cmap='jet', origin='lower', extent=[0, 10, 0, 10])
        im2 = self.ax2.imshow(Z2, cmap='jet', origin='lower', extent=[0, 10, 0, 10])
        
        # Thêm colorbar
        self.figure.colorbar(im1, ax=self.ax1, label='Liều (Gy)')
        self.figure.colorbar(im2, ax=self.ax2, label='Liều (Gy)')
        
        # Vẽ lại canvas
        self.canvas.draw()
    
    def clear(self):
        """Xóa hiển thị."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title('Kế hoạch 1')
        self.ax2.set_title('Kế hoạch 2')
        self.canvas.draw()


class DoseTab(QWidget):
    """
    Tab liều lượng.
    
    Tab này bao gồm các công cụ để hiển thị và phân tích phân bố liều lượng
    trong kế hoạch xạ trị, bao gồm hiển thị 2D/3D, DVH, và thống kê liều.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo tab liều lượng.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Trạng thái
        self.current_plan = None
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab liều lượng hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Tab hiển thị liều
        self.dose_display_widget = QWidget()
        self.dose_display_layout = QVBoxLayout(self.dose_display_widget)
        
        # Thiết lập hiển thị
        self.view_control_group = QGroupBox("Điều khiển hiển thị")
        self.view_control_layout = QHBoxLayout(self.view_control_group)
        
        self.slice_label = QLabel("Lát cắt:")
        self.view_control_layout.addWidget(self.slice_label)
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._slice_changed)
        self.view_control_layout.addWidget(self.slice_slider)
        
        self.view_control_layout.addWidget(QLabel("Mặt phẳng:"))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.plane_combo.currentIndexChanged.connect(self._plane_changed)
        self.view_control_layout.addWidget(self.plane_combo)
        
        self.dose_display_layout.addWidget(self.view_control_group)
        
        # Widget hiển thị liều
        self.dose_visualization = DoseVisualizationWidget()
        self.dose_display_layout.addWidget(self.dose_visualization)
        
        # Thêm tab hiển thị liều
        self.tab_widget.addTab(self.dose_display_widget, "Hiển thị liều")
        
        # Tab DVH
        self.dvh_widget = QWidget()
        self.dvh_layout = QVBoxLayout(self.dvh_widget)
        
        # Nhóm điều khiển DVH
        self.dvh_control_group = QGroupBox("Điều khiển DVH")
        self.dvh_control_layout = QHBoxLayout(self.dvh_control_group)
        
        self.dvh_type_label = QLabel("Loại DVH:")
        self.dvh_control_layout.addWidget(self.dvh_type_label)
        
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Tích lũy", "Vi phân"])
        self.dvh_type_combo.currentIndexChanged.connect(self._dvh_type_changed)
        self.dvh_control_layout.addWidget(self.dvh_type_combo)
        
        self.dvh_structures_label = QLabel("Cấu trúc:")
        self.dvh_control_layout.addWidget(self.dvh_structures_label)
        
        self.dvh_layout.addWidget(self.dvh_control_group)
        
        # Widget DVH
        self.dvh_plot = DVHWidget()
        self.dvh_layout.addWidget(self.dvh_plot)
        
        # Thêm tab DVH
        self.tab_widget.addTab(self.dvh_widget, "DVH")
        
        # Tab thống kê liều
        self.stats_widget = DoseStatisticsWidget()
        self.tab_widget.addTab(self.stats_widget, "Thống kê liều")
        
        # Tab so sánh liều
        self.comparison_widget = DoseComparisonWidget()
        self.tab_widget.addTab(self.comparison_widget, "So sánh liều")
    
    def set_plan(self, plan):
        """
        Thiết lập kế hoạch hiện tại và cập nhật giao diện.
        
        Parameters
        ----------
        plan : Any
            Đối tượng kế hoạch
        """
        self.current_plan = plan
        if plan:
            self._populate_dose_data()
        else:
            self._clear_dose_data()
    
    def _populate_dose_data(self):
        """Điền dữ liệu liều lượng vào giao diện."""
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
        pass
    
    def _clear_dose_data(self):
        """Xóa dữ liệu liều lượng khỏi giao diện."""
        # Xóa hiển thị liều
        self.dose_visualization.clear()
        
        # Xóa DVH
        self.dvh_plot.clear()
        
        # Xóa thống kê liều
        self.stats_widget.clear()
        
        # Xóa so sánh liều
        self.comparison_widget.clear()
    
    def _slice_changed(self, value):
        """
        Xử lý sự kiện khi lát cắt thay đổi.
        
        Parameters
        ----------
        value : int
            Giá trị lát cắt mới
        """
        logger.debug(f"Lát cắt thay đổi: {value}")
        # Cập nhật hiển thị dựa trên lát cắt mới - sẽ được triển khai khi có dữ liệu
    
    def _plane_changed(self, index):
        """
        Xử lý sự kiện khi mặt phẳng thay đổi.
        
        Parameters
        ----------
        index : int
            Chỉ số mặt phẳng mới
        """
        plane = self.plane_combo.currentText()
        logger.debug(f"Mặt phẳng thay đổi: {plane}")
        # Cập nhật hiển thị dựa trên mặt phẳng mới - sẽ được triển khai khi có dữ liệu
    
    def _dvh_type_changed(self, index):
        """
        Xử lý sự kiện khi loại DVH thay đổi.
        
        Parameters
        ----------
        index : int
            Chỉ số loại DVH mới
        """
        dvh_type = self.dvh_type_combo.currentText()
        logger.debug(f"Loại DVH thay đổi: {dvh_type}")
        # Cập nhật hiển thị DVH - sẽ được triển khai khi có dữ liệu
