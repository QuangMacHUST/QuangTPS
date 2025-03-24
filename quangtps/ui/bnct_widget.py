#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module widget BNCT cho QuangTPS.

Module này cung cấp giao diện để hiển thị và phân tích các thông số đặc thù
của kỹ thuật điều trị BNCT (Boron Neutron Capture Therapy).
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
    QSlider, QColorDialog, QGridLayout, QFrame
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

from quangtps.treatment.techniques.bnct import BNCT
from quangtps.specialized.bnct.boron import BoronCompoundType, BoronDistributionModel, TwoCompartmentModel
from quangtps.specialized.bnct.neutron import BaseNeutronModel, ReactorNeutronModel, AcceleratorNeutronModel

logger = logging.getLogger(__name__)


class BNCTDoseAnalysisWidget(QWidget):
    """Widget để phân tích và hiển thị thành phần liều BNCT."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget phân tích liều BNCT.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Lưu trữ dữ liệu
        self.bnct_plan = None
        self.dose_components = {}
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Tab widget để phân tách các thành phần giao diện
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Tab thành phần liều
        self.component_widget = QWidget()
        self.component_layout = QVBoxLayout(self.component_widget)
        self.tab_widget.addTab(self.component_widget, "Thành phần liều")
        
        # Tab phân phối Bo
        self.boron_widget = QWidget()
        self.boron_layout = QVBoxLayout(self.boron_widget)
        self.tab_widget.addTab(self.boron_widget, "Phân phối Bo")
        
        # Tab thông số neutron
        self.neutron_widget = QWidget()
        self.neutron_layout = QVBoxLayout(self.neutron_widget)
        self.tab_widget.addTab(self.neutron_widget, "Thông số neutron")
        
        # Tab RBE và quyền số sinh học
        self.rbe_widget = QWidget()
        self.rbe_layout = QVBoxLayout(self.rbe_widget)
        self.tab_widget.addTab(self.rbe_widget, "RBE và CBE")
        
        # Thiết lập nội dung cho các tab
        self._setup_component_tab()
        self._setup_boron_tab()
        self._setup_neutron_tab()
        self._setup_rbe_tab()
    
    def _setup_component_tab(self):
        """Thiết lập nội dung tab thành phần liều."""
        # Bảng thành phần liều
        component_group = QGroupBox("Thành phần liều BNCT")
        component_layout = QVBoxLayout(component_group)
        
        self.component_table = QTableWidget()
        self.component_table.setColumnCount(3)
        self.component_table.setHorizontalHeaderLabels(["Thành phần", "Giá trị (Gy)", "Tỷ lệ (%)"])
        self.component_table.horizontalHeader().setStretchLastSection(True)
        component_layout.addWidget(self.component_table)
        
        # Thêm mô tả
        description_label = QLabel(
            "BNCT bao gồm 4 thành phần liều chính: Liều Bo-10, liều neutron nhanh, "
            "liều neutron nhiệt và liều gamma. Mỗi thành phần có hiệu ứng sinh học khác nhau."
        )
        description_label.setWordWrap(True)
        component_layout.addWidget(description_label)
        
        # Thêm vào layout chính
        self.component_layout.addWidget(component_group)
        
        # Biểu đồ thành phần liều
        if MATPLOTLIB_AVAILABLE:
            chart_group = QGroupBox("Biểu đồ thành phần liều")
            chart_layout = QVBoxLayout(chart_group)
            
            self.component_figure = Figure(figsize=(6, 4))
            self.component_canvas = FigureCanvas(self.component_figure)
            self.component_toolbar = NavigationToolbar(self.component_canvas, self)
            
            chart_layout.addWidget(self.component_toolbar)
            chart_layout.addWidget(self.component_canvas)
            
            self.component_layout.addWidget(chart_group)
            
            # Khởi tạo biểu đồ
            self.component_ax = self.component_figure.add_subplot(111)
            self.component_ax.set_title("Phân tích thành phần liều BNCT")
            self.component_canvas.draw()

    def _setup_boron_tab(self):
        """Thiết lập nội dung tab phân phối Bo."""
        # Thông tin hợp chất Bo
        boron_group = QGroupBox("Thông tin hợp chất Bo")
        boron_form = QFormLayout(boron_group)
        
        # Chọn hợp chất Bo
        self.boron_compound_combo = QComboBox()
        self.boron_compound_combo.addItem(BoronCompoundType.BPA)
        self.boron_compound_combo.addItem(BoronCompoundType.BSH)
        self.boron_compound_combo.addItem("Kết hợp BPA và BSH")
        self.boron_compound_combo.addItem(BoronCompoundType.CUSTOM)
        boron_form.addRow("Hợp chất Bo:", self.boron_compound_combo)
        
        # Nồng độ Bo
        self.boron_concentration_spin = QDoubleSpinBox()
        self.boron_concentration_spin.setRange(0, 1000)
        self.boron_concentration_spin.setValue(30)
        self.boron_concentration_spin.setSuffix(" ppm")
        boron_form.addRow("Nồng độ Bo mô u:", self.boron_concentration_spin)
        
        # Tỷ lệ u/chuẩn (T/N ratio)
        self.tn_ratio_spin = QDoubleSpinBox()
        self.tn_ratio_spin.setRange(0, 10)
        self.tn_ratio_spin.setValue(3.5)
        self.tn_ratio_spin.setSingleStep(0.1)
        boron_form.addRow("Tỷ lệ u/chuẩn (T/N):", self.tn_ratio_spin)
        
        # Thêm vào layout chính
        self.boron_layout.addWidget(boron_group)
        
        # Phân bố Bo
        distribution_group = QGroupBox("Phân bố Bo theo độ sâu")
        distribution_layout = QVBoxLayout(distribution_group)
        
        if MATPLOTLIB_AVAILABLE:
            self.boron_figure = Figure(figsize=(6, 4))
            self.boron_canvas = FigureCanvas(self.boron_figure)
            self.boron_toolbar = NavigationToolbar(self.boron_canvas, self)
            
            distribution_layout.addWidget(self.boron_toolbar)
            distribution_layout.addWidget(self.boron_canvas)
            
            # Khởi tạo biểu đồ
            self.boron_ax = self.boron_figure.add_subplot(111)
            self.boron_ax.set_title("Phân bố Bo theo độ sâu")
            self.boron_ax.set_xlabel("Độ sâu (mm)")
            self.boron_ax.set_ylabel("Nồng độ Bo (ppm)")
            self.boron_canvas.draw()
        
        self.boron_layout.addWidget(distribution_group)
        
        # Mô tả
        description_label = QLabel(
            "Phân bố Bo là yếu tố quyết định hiệu quả điều trị BNCT. Tỷ lệ u/chuẩn (T/N) "
            "càng cao thì tính chọn lọc càng tốt. BPA và BSH có đặc tính phân bố khác nhau."
        )
        description_label.setWordWrap(True)
        self.boron_layout.addWidget(description_label)
    
    def _setup_neutron_tab(self):
        """Thiết lập nội dung tab thông số neutron."""
        # Thông tin nguồn neutron
        source_group = QGroupBox("Thông tin nguồn neutron")
        source_form = QFormLayout(source_group)
        
        # Chọn loại nguồn
        self.neutron_source_combo = QComboBox()
        self.neutron_source_combo.addItems([
            "Máy gia tốc (ACCELERATOR)", 
            "Lò phản ứng (REACTOR)", 
            "Máy phát neutron D-D (DD_GENERATOR)", 
            "Máy phát neutron D-T (DT_GENERATOR)"
        ])
        source_form.addRow("Loại nguồn neutron:", self.neutron_source_combo)
        
        # Thông số nguồn
        self.neutron_energy = QDoubleSpinBox()
        self.neutron_energy.setRange(0.01, 50.0)
        self.neutron_energy.setValue(10.0)
        self.neutron_energy.setSuffix(" MeV")
        source_form.addRow("Năng lượng neutron:", self.neutron_energy)
        
        self.neutron_flux = QDoubleSpinBox()
        self.neutron_flux.setRange(1e6, 1e14)
        self.neutron_flux.setValue(1e9)
        self.neutron_flux.setDecimals(2)
        self.neutron_flux.setSingleStep(1e6)
        self.neutron_flux.setSuffix(" n/cm²/s")
        source_form.addRow("Thông lượng neutron:", self.neutron_flux)
        
        # Nút cập nhật
        self.update_neutron_btn = QPushButton("Cập nhật mô hình")
        self.update_neutron_btn.clicked.connect(self._update_neutron_model)
        source_form.addRow("", self.update_neutron_btn)
        
        # Thêm vào layout chính
        self.neutron_layout.addWidget(source_group)
        
        # Biểu đồ phân bố neutron
        distribution_group = QGroupBox("Phân bố trường neutron theo độ sâu")
        distribution_layout = QVBoxLayout(distribution_group)
        
        if MATPLOTLIB_AVAILABLE:
            self.neutron_figure = Figure(figsize=(6, 4))
            self.neutron_canvas = FigureCanvas(self.neutron_figure)
            self.neutron_toolbar = NavigationToolbar(self.neutron_canvas, self)
            
            distribution_layout.addWidget(self.neutron_toolbar)
            distribution_layout.addWidget(self.neutron_canvas)
            
            # Khởi tạo biểu đồ
            self.neutron_ax = self.neutron_figure.add_subplot(111)
            self.neutron_ax.set_title("Thông lượng neutron theo độ sâu")
            self.neutron_ax.set_xlabel("Độ sâu (mm)")
            self.neutron_ax.set_ylabel("Thông lượng (n/cm²/s)")
            self.neutron_canvas.draw()
        
        self.neutron_layout.addWidget(distribution_group)
        
        # Chất điều tiết
        moderator_group = QGroupBox("Chất điều tiết")
        moderator_layout = QFormLayout(moderator_group)
        
        self.moderator_combo = QComboBox()
        self.moderator_combo.addItems(["Nước", "Nhôm", "Graphite", "BeO", "Fluoride", "Không sử dụng"])
        moderator_layout.addRow("Loại chất điều tiết:", self.moderator_combo)
        
        self.moderator_thickness = QDoubleSpinBox()
        self.moderator_thickness.setRange(0, 30)
        self.moderator_thickness.setValue(5)
        self.moderator_thickness.setSuffix(" cm")
        moderator_layout.addRow("Độ dày chất điều tiết:", self.moderator_thickness)
        
        self.neutron_layout.addWidget(moderator_group)
        
    def _setup_rbe_tab(self):
        """Thiết lập nội dung tab RBE và quyền số sinh học."""
        # Hệ số RBE
        rbe_group = QGroupBox("Hệ số hiệu quả sinh học tương đối (RBE)")
        rbe_layout = QFormLayout(rbe_group)
        
        self.fast_neutron_rbe = QDoubleSpinBox()
        self.fast_neutron_rbe.setRange(1.0, 10.0)
        self.fast_neutron_rbe.setValue(3.2)
        self.fast_neutron_rbe.setSingleStep(0.1)
        rbe_layout.addRow("RBE neutron nhanh:", self.fast_neutron_rbe)
        
        self.thermal_neutron_rbe = QDoubleSpinBox()
        self.thermal_neutron_rbe.setRange(1.0, 5.0)
        self.thermal_neutron_rbe.setValue(2.3)
        self.thermal_neutron_rbe.setSingleStep(0.1)
        rbe_layout.addRow("RBE neutron nhiệt:", self.thermal_neutron_rbe)
        
        self.gamma_rbe = QDoubleSpinBox()
        self.gamma_rbe.setRange(1.0, 2.0)
        self.gamma_rbe.setValue(1.0)
        self.gamma_rbe.setSingleStep(0.1)
        rbe_layout.addRow("RBE tia gamma:", self.gamma_rbe)
        
        # Thêm vào layout chính
        self.rbe_layout.addWidget(rbe_group)
        
        # Hệ số CBE cho Bo
        cbe_group = QGroupBox("Hệ số hiệu quả hợp chất Bo (CBE)")
        cbe_layout = QFormLayout(cbe_group)
        
        self.tumor_cbe = QDoubleSpinBox()
        self.tumor_cbe.setRange(1.0, 8.0)
        self.tumor_cbe.setValue(3.8)
        self.tumor_cbe.setSingleStep(0.1)
        cbe_layout.addRow("CBE mô u:", self.tumor_cbe)
        
        self.skin_cbe = QDoubleSpinBox()
        self.skin_cbe.setRange(1.0, 5.0)
        self.skin_cbe.setValue(2.5)
        self.skin_cbe.setSingleStep(0.1)
        cbe_layout.addRow("CBE da:", self.skin_cbe)
        
        self.brain_cbe = QDoubleSpinBox()
        self.brain_cbe.setRange(1.0, 5.0)
        self.brain_cbe.setValue(1.3)
        self.brain_cbe.setSingleStep(0.1)
        cbe_layout.addRow("CBE não:", self.brain_cbe)
        
        # Thêm vào layout chính
        self.rbe_layout.addWidget(cbe_group)
        
        # Biểu đồ so sánh
        if MATPLOTLIB_AVAILABLE:
            chart_group = QGroupBox("So sánh hiệu quả sinh học của các thành phần")
            chart_layout = QVBoxLayout(chart_group)
            
            self.rbe_figure = Figure(figsize=(6, 4))
            self.rbe_canvas = FigureCanvas(self.rbe_figure)
            self.rbe_toolbar = NavigationToolbar(self.rbe_canvas, self)
            
            chart_layout.addWidget(self.rbe_toolbar)
            chart_layout.addWidget(self.rbe_canvas)
            
            # Khởi tạo biểu đồ
            self.rbe_ax = self.rbe_figure.add_subplot(111)
            self.rbe_ax.set_title("So sánh hiệu quả sinh học")
            self.rbe_ax.set_xlabel("Loại bức xạ/hợp chất")
            self.rbe_ax.set_ylabel("Hệ số hiệu quả sinh học")
            self.rbe_canvas.draw()
            
            self.rbe_layout.addWidget(chart_group)
    
    def set_bnct_plan(self, bnct_plan):
        """
        Thiết lập kế hoạch BNCT và cập nhật giao diện.
        
        Parameters
        ----------
        bnct_plan : BNCT
            Đối tượng kế hoạch BNCT
        """
        self.bnct_plan = bnct_plan
        if bnct_plan is None:
            self._clear_data()
            return
        
        # Cập nhật thông tin hợp chất Bo
        compound_type = bnct_plan.boron_compound
        if compound_type == BoronCompoundType.BPA:
            self.boron_compound_combo.setCurrentIndex(0)
        elif compound_type == BoronCompoundType.BSH:
            self.boron_compound_combo.setCurrentIndex(1)
        elif compound_type == "MIXED":
            self.boron_compound_combo.setCurrentIndex(2)
        else:
            self.boron_compound_combo.setCurrentIndex(3)
        
        self.boron_concentration_spin.setValue(bnct_plan.boron_concentration)
        self.tn_ratio_spin.setValue(bnct_plan.tumor_to_normal_ratio)
        
        # Cập nhật thông tin nguồn neutron
        source_type = bnct_plan.neutron_source
        if source_type == "ACCELERATOR":
            self.neutron_source_combo.setCurrentIndex(0)
        elif source_type == "REACTOR":
            self.neutron_source_combo.setCurrentIndex(1)
        elif source_type == "DD_GENERATOR":
            self.neutron_source_combo.setCurrentIndex(2)
        elif source_type == "DT_GENERATOR":
            self.neutron_source_combo.setCurrentIndex(3)
        
        # Cập nhật hiển thị thành phần liều
        self._update_dose_components()
        
        # Cập nhật biểu đồ
        self._update_all_plots()
    
    def _update_dose_components(self):
        """Cập nhật bảng thành phần liều từ kế hoạch BNCT."""
        if self.bnct_plan is None:
            return
        
        # Lấy các thành phần liều tại độ sâu mặc định
        depth = 20.0  # Độ sâu mẫu (mm)
        components = self.bnct_plan.calculate_dose_components(depth)
        
        # Tổng liều vật lý
        total_dose = sum(components.values())
        
        # Cập nhật bảng
        self.component_table.setRowCount(len(components) + 1)  # +1 cho dòng tổng
        
        # Thêm các thành phần
        row = 0
        for component, value in components.items():
            percentage = (value / total_dose * 100) if total_dose > 0 else 0
            
            name_item = QTableWidgetItem(self._format_component_name(component))
            value_item = QTableWidgetItem(f"{value:.3f}")
            percent_item = QTableWidgetItem(f"{percentage:.1f}")
            
            self.component_table.setItem(row, 0, name_item)
            self.component_table.setItem(row, 1, value_item)
            self.component_table.setItem(row, 2, percent_item)
            
            row += 1
        
        # Thêm dòng tổng
        name_item = QTableWidgetItem("Tổng cộng")
        name_item.setFont(QFont("", -1, QFont.Bold))
        
        value_item = QTableWidgetItem(f"{total_dose:.3f}")
        value_item.setFont(QFont("", -1, QFont.Bold))
        
        percent_item = QTableWidgetItem("100.0")
        percent_item.setFont(QFont("", -1, QFont.Bold))
        
        self.component_table.setItem(row, 0, name_item)
        self.component_table.setItem(row, 1, value_item)
        self.component_table.setItem(row, 2, percent_item)
        
        # Cập nhật biểu đồ
        if MATPLOTLIB_AVAILABLE:
            self._update_component_chart(components)
    
    def _update_component_chart(self, components):
        """Cập nhật biểu đồ thành phần liều."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Xóa biểu đồ cũ
        self.component_ax.clear()
        
        # Chuẩn bị dữ liệu
        labels = [self._format_component_name(c) for c in components.keys()]
        values = list(components.values())
        colors = ['#1a73e8', '#fbbc04', '#ea4335', '#34a853', '#9c27b0', '#ff7043']
        
        # Vẽ biểu đồ
        self.component_ax.pie(
            values, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            shadow=False
        )
        self.component_ax.set_title("Phân tích thành phần liều BNCT")
        self.component_ax.axis('equal')  # Equal aspect ratio
        
        # Cập nhật canvas
        self.component_canvas.draw()
    
    def _format_component_name(self, component_key):
        """Định dạng tên thành phần liều."""
        name_map = {
            'neutron_thermal': 'Neutron nhiệt',
            'neutron_epithermal': 'Neutron biên nhiệt',
            'neutron_fast': 'Neutron nhanh',
            'gamma': 'Tia gamma',
            'alpha': 'Hạt alpha (Bo)',
            'lithium': 'Lithium (Bo)'
        }
        return name_map.get(component_key, component_key)
        
    def _update_all_plots(self):
        """Cập nhật tất cả các biểu đồ."""
        if not MATPLOTLIB_AVAILABLE or self.bnct_plan is None:
            return
        
        # Cập nhật biểu đồ phân bố Bo
        self._update_boron_distribution_plot()
        
        # Cập nhật biểu đồ phân bố neutron
        self._update_neutron_distribution_plot()
        
        # Cập nhật biểu đồ RBE và CBE
        self._update_rbe_chart()
    
    def _update_boron_distribution_plot(self):
        """Cập nhật biểu đồ phân bố Bo."""
        if not MATPLOTLIB_AVAILABLE or self.bnct_plan is None:
            return
        
        # Xóa biểu đồ cũ
        self.boron_ax.clear()
        
        # Tạo dữ liệu
        depths = np.linspace(0, 100, 101)  # 0-100 mm
        
        # Mô phỏng phân bố Bo đơn giản
        base_concentration = self.boron_concentration_spin.value()
        tumor_ratio = self.tn_ratio_spin.value()
        
        # Phân bố Bo trong mô u (mô phỏng)
        tumor_peak = 30  # Giả sử đỉnh u ở độ sâu 30mm
        tumor_width = 15  # Độ rộng của u
        tumor_profile = base_concentration * np.exp(-((depths - tumor_peak) ** 2) / (2 * tumor_width ** 2))
        
        # Phân bố Bo trong mô lành (mô phỏng đơn giản)
        normal_concentration = base_concentration / tumor_ratio
        normal_profile = np.ones_like(depths) * normal_concentration
        
        # Vẽ biểu đồ
        self.boron_ax.plot(depths, tumor_profile, 'r-', label='Mô u')
        self.boron_ax.plot(depths, normal_profile, 'b-', label='Mô lành')
        
        self.boron_ax.set_xlabel('Độ sâu (mm)')
        self.boron_ax.set_ylabel('Nồng độ Bo (ppm)')
        self.boron_ax.set_title('Phân bố Bo theo độ sâu')
        self.boron_ax.grid(True)
        self.boron_ax.legend()
        
        # Cập nhật canvas
        self.boron_canvas.draw()
    
    def _update_neutron_distribution_plot(self):
        """Cập nhật biểu đồ phân bố neutron."""
        if not MATPLOTLIB_AVAILABLE or self.bnct_plan is None:
            return
        
        # Xóa biểu đồ cũ
        self.neutron_ax.clear()
        
        # Tạo dữ liệu
        depths = np.linspace(0, 100, 101)  # 0-100 mm
        
        # Mô phỏng phân bố neutron đơn giản
        if not hasattr(self.bnct_plan, '_neutron_model') or self.bnct_plan._neutron_model is None:
            return
            
        thermal_flux = np.array([self.bnct_plan._neutron_model.calculate_thermal_flux(d/10) for d in depths])
        epithermal_flux = np.array([self.bnct_plan._neutron_model.calculate_epithermal_flux(d/10) for d in depths])
        fast_flux = np.array([self.bnct_plan._neutron_model.calculate_fast_flux(d/10) for d in depths])
        
        # Vẽ biểu đồ
        self.neutron_ax.semilogy(depths, thermal_flux, 'b-', label='Neutron nhiệt')
        self.neutron_ax.semilogy(depths, epithermal_flux, 'g-', label='Neutron biên nhiệt')
        self.neutron_ax.semilogy(depths, fast_flux, 'r-', label='Neutron nhanh')
        
        self.neutron_ax.set_xlabel('Độ sâu (mm)')
        self.neutron_ax.set_ylabel('Thông lượng (n/cm²/s)')
        self.neutron_ax.set_title('Phân bố neutron theo độ sâu')
        self.neutron_ax.grid(True)
        self.neutron_ax.legend()
        
        # Cập nhật canvas
        self.neutron_canvas.draw()
    
    def _update_rbe_chart(self):
        """Cập nhật biểu đồ RBE và CBE."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Xóa biểu đồ cũ
        self.rbe_ax.clear()
        
        # Chuẩn bị dữ liệu
        categories = ['Gamma', 'Neutron nhiệt', 'Neutron nhanh', 'Bo (u)', 'Bo (não)', 'Bo (da)']
        values = [
            self.gamma_rbe.value(),
            self.thermal_neutron_rbe.value(),
            self.fast_neutron_rbe.value(),
            self.tumor_cbe.value(),
            self.brain_cbe.value(),
            self.skin_cbe.value()
        ]
        colors = ['#34a853', '#4285f4', '#ea4335', '#fbbc04', '#9c27b0', '#ff7043']
        
        # Vẽ biểu đồ
        bars = self.rbe_ax.bar(categories, values, color=colors)
        
        # Thêm nhãn giá trị trên các cột
        for bar in bars:
            height = bar.get_height()
            self.rbe_ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom'
            )
        
        self.rbe_ax.set_ylim(0, max(values) * 1.2)  # Thêm không gian cho nhãn
        self.rbe_ax.set_ylabel('Hệ số hiệu quả sinh học')
        self.rbe_ax.set_title('So sánh hệ số RBE/CBE')
        self.rbe_ax.set_axisbelow(True)
        self.rbe_ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Cập nhật canvas
        self.rbe_canvas.draw()
        
    def _update_neutron_model(self):
        """Cập nhật mô hình neutron dựa trên thông tin nhập từ giao diện."""
        if self.bnct_plan is None:
            return
            
        # Cập nhật loại nguồn neutron
        source_index = self.neutron_source_combo.currentIndex()
        if source_index == 0:
            self.bnct_plan.neutron_source = "ACCELERATOR"
        elif source_index == 1:
            self.bnct_plan.neutron_source = "REACTOR"
        elif source_index == 2:
            self.bnct_plan.neutron_source = "DD_GENERATOR"
        elif source_index == 3:
            self.bnct_plan.neutron_source = "DT_GENERATOR"
            
        # Thiết lập lại mô hình neutron
        self.bnct_plan.setup_neutron_source()
        
        # Cập nhật giao diện
        self._update_dose_components()
        self._update_neutron_distribution_plot()
        
    def _clear_data(self):
        """Xóa tất cả dữ liệu và đặt lại giao diện."""
        # Xóa bảng thành phần liều
        self.component_table.setRowCount(0)
        
        # Xóa các biểu đồ nếu matplotlib khả dụng
        if MATPLOTLIB_AVAILABLE:
            self.component_ax.clear()
            self.component_ax.set_title("Phân tích thành phần liều BNCT")
            self.component_canvas.draw()
            
            self.boron_ax.clear()
            self.boron_ax.set_title("Phân bố Bo theo độ sâu")
            self.boron_ax.set_xlabel("Độ sâu (mm)")
            self.boron_ax.set_ylabel("Nồng độ Bo (ppm)")
            self.boron_canvas.draw()
            
            self.neutron_ax.clear()
            self.neutron_ax.set_title("Thông lượng neutron theo độ sâu")
            self.neutron_ax.set_xlabel("Độ sâu (mm)")
            self.neutron_ax.set_ylabel("Thông lượng (n/cm²/s)")
            self.neutron_canvas.draw()
            
            self.rbe_ax.clear()
            self.rbe_ax.set_title("So sánh hiệu quả sinh học")
            self.rbe_ax.set_xlabel("Loại bức xạ/hợp chất")
            self.rbe_ax.set_ylabel("Hệ số hiệu quả sinh học")
            self.rbe_canvas.draw()
