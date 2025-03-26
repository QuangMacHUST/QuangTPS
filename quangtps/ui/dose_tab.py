#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab liều lượng (Dose Tab) cho QuangTPS.

Module này cung cấp giao diện để hiển thị và phân tích phân bố liều lượng
trong kế hoạch xạ trị, bao gồm các công cụ như DVH, phân tích liều, và so sánh liều.
"""

import logging
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QTabWidget,
    QSlider, QScrollArea, QSpinBox, QDoubleSpinBox,
    QTextEdit, QCheckBox, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Internal imports
from quangtps.treatment.techniques.bnct import BNCT
from quangtps.ui.bnct_widget import BNCTDoseAnalysisWidget

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
        
        # Dữ liệu 
        self.current_plan = None
        self.dose_data = None
        self.structures = []
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab liều lượng hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # === Tab container ===
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # === Tab hiển thị liều 2D ===
        dose_display_tab = QWidget()
        self.dose_display_layout = QVBoxLayout(dose_display_tab)
        
        # Panel điều khiển ở trên
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        
        # Chọn mặt phẳng
        plane_label = QLabel("Mặt phẳng:")
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.plane_combo.currentIndexChanged.connect(self._plane_changed)
        
        # Thanh trượt slice
        slice_label = QLabel("Slice:")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)  # Mặc định
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._slice_changed)
        
        # Thêm vào layout điều khiển
        control_layout.addWidget(plane_label)
        control_layout.addWidget(self.plane_combo)
        control_layout.addWidget(slice_label)
        control_layout.addWidget(self.slice_slider, 1)
        
        # Thêm panel điều khiển vào layout tab
        self.dose_display_layout.addWidget(control_widget)
        
        # Widget hiển thị liều
        self.dose_viz_widget = DoseVisualizationWidget()
        self.dose_display_layout.addWidget(self.dose_viz_widget)
        
        # Thêm tab hiển thị liều
        self.tab_widget.addTab(dose_display_tab, "Hiển thị liều")
        
        # === Tab DVH ===
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # Panel điều khiển DVH
        dvh_control_widget = QWidget()
        dvh_control_layout = QHBoxLayout(dvh_control_widget)
        
        # Chọn loại DVH
        dvh_type_label = QLabel("Loại DVH:")
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Tích lũy", "Vi phân"])
        self.dvh_type_combo.currentIndexChanged.connect(self._dvh_type_changed)
        
        # Thêm vào layout điều khiển DVH
        dvh_control_layout.addWidget(dvh_type_label)
        dvh_control_layout.addWidget(self.dvh_type_combo)
        dvh_control_layout.addStretch()
        
        # Thêm panel điều khiển vào layout tab DVH
        dvh_layout.addWidget(dvh_control_widget)
        
        # Widget DVH
        self.dvh_widget = DVHWidget()
        dvh_layout.addWidget(self.dvh_widget)
        
        # Thêm tab DVH
        self.tab_widget.addTab(dvh_tab, "DVH")
        
        # Tab thống kê liều
        dose_stats_tab = QWidget()
        dose_stats_layout = QVBoxLayout(dose_stats_tab)
        self.dose_stats_widget = DoseStatisticsWidget()
        dose_stats_layout.addWidget(self.dose_stats_widget)
        self.tab_widget.addTab(dose_stats_tab, "Thống kê liều")
        
        # Tab so sánh liều
        dose_comparison_tab = QWidget()
        dose_comparison_layout = QVBoxLayout(dose_comparison_tab)
        self.dose_comparison_widget = DoseComparisonWidget()
        dose_comparison_layout.addWidget(self.dose_comparison_widget)
        self.tab_widget.addTab(dose_comparison_tab, "So sánh liều")
        
        # Tab phân tích BNCT (sẽ được thêm khi cần)
        self.bnct_widget = None
    
    def set_plan(self, plan):
        """
        Thiết lập kế hoạch để hiển thị phân bố liều.
        
        Parameters
        ----------
        plan : dict
            Dữ liệu kế hoạch
        """
        try:
            self.current_plan = plan
            
            if plan:
                self._populate_dose_data()
            else:
                self._clear_dose_data()
                logger.warning("Không có kế hoạch được cung cấp cho DoseTab")
                
        except Exception as e:
            logger.exception("Lỗi khi thiết lập kế hoạch trong DoseTab: %s", str(e))
            QMessageBox.critical(
                self,
                "Lỗi tải dữ liệu liều",
                f"Không thể tải dữ liệu liều lượng: {str(e)}\n\nVui lòng kiểm tra kế hoạch điều trị của bạn."
            )
            self._clear_dose_data()
    
    def _populate_dose_data(self):
        """Tải và hiển thị dữ liệu liều từ kế hoạch hiện tại."""
        if not self.current_plan:
            return
            
        try:
            # Mô phỏng tải dữ liệu liều (trong thực tế sẽ tải từ DICOM hoặc cơ sở dữ liệu)
            logger.info("Đang tải dữ liệu liều cho kế hoạch ID: %s", self.current_plan.get('id', 'unknown'))
            
            # Giả lập dữ liệu liều
            dose_shape = (100, 100, 50)  # (z, y, x)
            self.dose_data = np.random.rand(*dose_shape) * 60.0  # Giả lập liều 0-60 Gy
            
            # Tạo tọa độ (trong thực tế sẽ lấy từ DICOM)
            pixel_spacing = 2.0  # mm
            x_coords = np.arange(dose_shape[2]) * pixel_spacing
            y_coords = np.arange(dose_shape[1]) * pixel_spacing
            z_coords = np.arange(dose_shape[0]) * pixel_spacing
            
            # Giả lập dữ liệu cấu trúc (trong thực tế sẽ tải từ DICOM RTSTRUCT)
            self.structures = [
                {
                    'name': 'PTV',
                    'color': QColor(255, 0, 0),
                    'dvh': np.clip(1.0 - np.arange(0, 61) / 50.0, 0, 1) * 100  # Giả lập DVH
                },
                {
                    'name': 'Bladder',
                    'color': QColor(255, 255, 0),
                    'dvh': np.clip(1.0 - np.arange(0, 61) / 30.0, 0, 1) * 100
                },
                {
                    'name': 'Rectum',
                    'color': QColor(0, 255, 0),
                    'dvh': np.clip(1.0 - np.arange(0, 61) / 20.0, 0, 1) * 100
                }
            ]
            
            # Hiển thị dữ liệu liều
            # 1. Cập nhật hiển thị 2D
            current_slice = 25  # Mặc định slice giữa
            self.slice_slider.setMaximum(dose_shape[0] - 1)
            self.slice_slider.setValue(current_slice)
            
            current_plane_index = self.plane_combo.currentIndex()
            if current_plane_index == 0:  # Axial
                self.dose_viz_widget.display_dose_2d(
                    self.dose_data[current_slice, :, :], 
                    x_coords, y_coords
                )
            elif current_plane_index == 1:  # Coronal
                self.dose_viz_widget.display_dose_2d(
                    self.dose_data[:, current_slice, :], 
                    x_coords, z_coords
                )
            else:  # Sagittal
                self.dose_viz_widget.display_dose_2d(
                    self.dose_data[:, :, current_slice], 
                    y_coords, z_coords
                )
            
            # 2. Cập nhật DVH
            dose_bins = np.arange(0, 61)  # 0-60 Gy
            self.dvh_widget.plot_dvh(self.structures, dose_bins, 
                                   cumulative=(self.dvh_type_combo.currentIndex()==0))
            
            # 3. Cập nhật thống kê liều
            self.dose_stats_widget.set_structures(self.structures)
            
            # 4. Cập nhật hiển thị so sánh (nếu có dữ liệu)
            if hasattr(self, 'dose_comparison_widget'):
                plans_for_comparison = [
                    {'id': self.current_plan.get('id'), 'name': self.current_plan.get('name')},
                    # Trong thực tế có thể lấy thêm các kế hoạch khác để so sánh
                ]
                self.dose_comparison_widget.set_plans(plans_for_comparison)
                
            logger.info("Đã tải dữ liệu liều thành công cho kế hoạch ID: %s", 
                      self.current_plan.get('id', 'unknown'))
            
        except Exception as e:
            logger.exception("Lỗi khi tải dữ liệu liều: %s", str(e))
            QMessageBox.warning(
                self,
                "Lỗi hiển thị dữ liệu",
                f"Không thể hiển thị dữ liệu liều: {str(e)}\n\n"
                "Hiển thị dữ liệu mẫu để minh họa thay thế."
            )
            
            # Tạo dữ liệu mẫu đơn giản để hiển thị
            if self.dose_data is None:
                # Tạo dữ liệu mẫu cơ bản nếu không tải được dữ liệu
                dose_shape = (50, 50, 50)
                self.dose_data = np.zeros(dose_shape)
                central_area = np.indices(dose_shape) - np.array([25, 25, 25])[:, None, None, None]
                distance = np.sqrt(np.sum(central_area**2, axis=0))
                self.dose_data = 50 * np.exp(-distance/15)
                
                # Cập nhật thanh trượt
                self.slice_slider.setMaximum(dose_shape[0] - 1)
                self.slice_slider.setValue(dose_shape[0] // 2)
                
                # Hiển thị slice mặc định
                x_coords = np.arange(dose_shape[2])
                y_coords = np.arange(dose_shape[1])
                self.dose_viz_widget.display_dose_2d(
                    self.dose_data[dose_shape[0]//2, :, :], 
                    x_coords, y_coords
                )
    
    def _clear_dose_data(self):
        """Xóa dữ liệu liều lượng khỏi giao diện."""
        # Xóa hiển thị liều
        self.dose_viz_widget.clear()
        
        # Xóa DVH
        self.dvh_widget.clear()
        
        # Xóa thống kê liều
        self.dose_stats_widget.clear()
        
        # Xóa so sánh liều
        self.dose_comparison_widget.clear()
        
        # Xóa dữ liệu BNCT (nếu có)
        if self.bnct_widget:
            self.bnct_widget.clear()
            idx = self.tab_widget.indexOf(self.bnct_widget)
            if idx != -1:  # Nếu tab tồn tại trong tab_widget
                self.tab_widget.setTabVisible(idx, False)
    
    def _slice_changed(self, value):
        """
        Xử lý khi người dùng thay đổi lát cắt hiển thị.
        
        Parameters
        ----------
        value : int
            Chỉ số lát cắt mới
        """
        if not self.current_dose or not hasattr(self.current_dose, 'data') or self.current_dose.data is None or self.current_dose.data.size == 0:
            return
            
        try:
            # Lấy dữ liệu mặt phẳng hiện tại
            if self.current_plane == "Axial":
                if value < 0 or value >= self.current_dose.data.shape[0]:
                    logger.warning(f"Chỉ số lát cắt axial không hợp lệ: {value}")
                    return
                dose_slice = self.current_dose.data[value, :, :]
                if self.current_image and hasattr(self.current_image, 'data') and self.current_image.data is not None:
                    if value < self.current_image.data.shape[0]:
                        image_slice = self.current_image.data[value, :, :]
                    else:
                        image_slice = None
                else:
                    image_slice = None
                    
            elif self.current_plane == "Coronal":
                if self.current_dose.data.shape[1] <= 0:
                    logger.warning("Không có dữ liệu cho mặt phẳng coronal")
                    return
                if value < 0 or value >= self.current_dose.data.shape[1]:
                    logger.warning(f"Chỉ số lát cắt coronal không hợp lệ: {value}")
                    return
                dose_slice = self.current_dose.data[:, value, :]
                if self.current_image and hasattr(self.current_image, 'data') and self.current_image.data is not None:
                    if value < self.current_image.data.shape[1]:
                        image_slice = self.current_image.data[:, value, :]
                    else:
                        image_slice = None
                else:
                    image_slice = None
                    
            elif self.current_plane == "Sagittal":
                if self.current_dose.data.shape[2] <= 0:
                    logger.warning("Không có dữ liệu cho mặt phẳng sagittal")
                    return
                if value < 0 or value >= self.current_dose.data.shape[2]:
                    logger.warning(f"Chỉ số lát cắt sagittal không hợp lệ: {value}")
                    return
                dose_slice = self.current_dose.data[:, :, value]
                if self.current_image and hasattr(self.current_image, 'data') and self.current_image.data is not None:
                    if value < self.current_image.data.shape[2]:
                        image_slice = self.current_image.data[:, :, value]
                    else:
                        image_slice = None
                else:
                    image_slice = None
            else:
                logger.warning(f"Mặt phẳng không hợp lệ: {self.current_plane}")
                return
                
            # Hiển thị dữ liệu
            if dose_slice is not None and dose_slice.size > 0:
                # Hiển thị lát cắt liều
                self.dose_viz_widget.display_dose_2d(dose_slice, None, None)
                
                # Cập nhật thông tin liều
                self._update_dose_info(dose_slice)
            
        except Exception as e:
            logger.error(f"Lỗi khi thay đổi lát cắt: {str(e)}")
    
    def _plane_changed(self, index):
        """
        Xử lý khi người dùng thay đổi mặt phẳng hiển thị.
        
        Parameters
        ----------
        index : int
            Chỉ số mặt phẳng mới
        """
        if not self.current_dose or not hasattr(self.current_dose, 'data') or self.current_dose.data is None or self.current_dose.data.size == 0:
            return
            
        try:
            # Đặt mặt phẳng hiện tại
            self.current_plane = self.plane_combo.currentText()
            
            # Cập nhật phạm vi thanh trượt dựa trên mặt phẳng
            if self.current_plane == "Axial":
                if self.current_dose.data.shape[0] > 0:
                    self.slice_slider.setMinimum(0)
                    self.slice_slider.setMaximum(self.current_dose.data.shape[0] - 1)
                    self.slice_slider.setValue(self.current_dose.data.shape[0] // 2)
                else:
                    logger.warning("Không có dữ liệu cho mặt phẳng axial")
                    return
                    
            elif self.current_plane == "Coronal":
                if len(self.current_dose.data.shape) > 1 and self.current_dose.data.shape[1] > 0:
                    self.slice_slider.setMinimum(0)
                    self.slice_slider.setMaximum(self.current_dose.data.shape[1] - 1)
                    self.slice_slider.setValue(self.current_dose.data.shape[1] // 2)
                else:
                    logger.warning("Không có dữ liệu cho mặt phẳng coronal")
                    return
                    
            elif self.current_plane == "Sagittal":
                if len(self.current_dose.data.shape) > 2 and self.current_dose.data.shape[2] > 0:
                    self.slice_slider.setMinimum(0)
                    self.slice_slider.setMaximum(self.current_dose.data.shape[2] - 1)
                    self.slice_slider.setValue(self.current_dose.data.shape[2] // 2)
                else:
                    logger.warning("Không có dữ liệu cho mặt phẳng sagittal")
                    return
            
            # Hiển thị lát cắt hiện tại
            self._slice_changed(self.slice_slider.value())
            
        except Exception as e:
            logger.error(f"Lỗi khi thay đổi mặt phẳng: {str(e)}")
    
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
