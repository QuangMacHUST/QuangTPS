#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đánh giá kế hoạch xạ trị (Plan Evaluation) cho QuangTPS.

Module này cung cấp các công cụ và giao diện người dùng để đánh giá
chất lượng của kế hoạch xạ trị thông qua các chỉ số lâm sàng, biểu đồ DVH,
và phân tích phân bố liều.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import SimpleITK as sitk

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QTabWidget, QSplitter,
    QComboBox, QGroupBox, QFrame, QScrollArea, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont

from quangtps.core.logging import get_logger
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.metrics.conformity import ConformityIndices
from quangtps.evaluation.metrics.homogeneity import HomogeneityIndices
from quangtps.evaluation.metrics.gradient import GradientIndices
from quangtps.evaluation.metrics.biological import TCP_LKB, NTCP_LKB
from quangtps.dose.dose_visualization import DoseColorwash, DVHPlotter
from quangtps.ui.image_display import ImageSliceWidget

logger = get_logger(__name__)


class DVHCanvas(FigureCanvasQTAgg):
    """Canvas để hiển thị biểu đồ DVH."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Khởi tạo canvas DVH.
        
        Args:
            parent: Widget cha
            width: Chiều rộng hình (inches)
            height: Chiều cao hình (inches)
            dpi: Độ phân giải điểm ảnh
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(DVHCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Cấu hình biểu đồ DVH
        self.axes.set_xlabel("Liều (Gy)")
        self.axes.set_ylabel("Thể tích (%)")
        self.axes.set_title("Biểu đồ Liều-Thể tích (DVH)")
        self.axes.grid(True)
        self.axes.set_xlim(0, 80)  # Mặc định từ 0 đến 80 Gy
        self.axes.set_ylim(0, 105)  # Mặc định từ 0 đến 105%
        
        self.fig.tight_layout()


class MetricsTableWidget(QTableWidget):
    """Bảng hiển thị các chỉ số đánh giá kế hoạch."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget bảng chỉ số.
        
        Args:
            parent: Widget cha
        """
        super().__init__(parent)
        
        # Cấu hình bảng
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Cấu trúc", "Chỉ số", "Giá trị"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        
        # Dữ liệu mẫu
        self.init_sample_data()
    
    def init_sample_data(self):
        """Khởi tạo dữ liệu mẫu cho bảng."""
        # Xóa dữ liệu hiện tại
        self.clearContents()
        self.setRowCount(0)
        
        # Dữ liệu mẫu
        sample_data = [
            ("PTV", "Dmin", "47.5 Gy"),
            ("PTV", "Dmax", "53.2 Gy"),
            ("PTV", "Dmean", "50.8 Gy"),
            ("PTV", "V95%", "98.7%"),
            ("PTV", "D95%", "50.2 Gy"),
            ("PTV", "CI", "0.92"),
            ("PTV", "HI", "1.07"),
            ("Parotid L", "Dmean", "26.4 Gy"),
            ("Parotid R", "Dmean", "25.1 Gy"),
            ("Spinal Cord", "Dmax", "38.6 Gy"),
        ]
        
        # Thêm dữ liệu vào bảng
        self.setRowCount(len(sample_data))
        for i, (structure, metric, value) in enumerate(sample_data):
            self.setItem(i, 0, QTableWidgetItem(structure))
            self.setItem(i, 1, QTableWidgetItem(metric))
            self.setItem(i, 2, QTableWidgetItem(value))


class IsodoseControlWidget(QWidget):
    """Widget điều khiển đường đồng liều (isodose)."""
    
    # Tín hiệu khi thay đổi cài đặt
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget điều khiển isodose.
        
        Args:
            parent: Widget cha
        """
        super().__init__(parent)
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Tiêu đề
        title_label = QLabel("Điều khiển Isodose")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Form layout cho các cài đặt
        form_layout = QFormLayout()
        
        # Danh sách các mức isodose
        self.isodose_combo = QComboBox()
        self.isodose_combo.addItems(["100%, 95%, 90%, 80%, 70%, 50%, 30%", "95%, 80%, 50%, 20%", "Tùy chỉnh"])
        self.isodose_combo.currentIndexChanged.connect(self._on_isodose_preset_changed)
        form_layout.addRow("Cài đặt sẵn:", self.isodose_combo)
        
        # Liều tham chiếu
        self.reference_dose_spin = QDoubleSpinBox()
        self.reference_dose_spin.setRange(0.1, 1000.0)
        self.reference_dose_spin.setValue(50.0)
        self.reference_dose_spin.setSuffix(" Gy")
        self.reference_dose_spin.valueChanged.connect(self._on_settings_changed)
        form_layout.addRow("Liều tham chiếu:", self.reference_dose_spin)
        
        # Độ trong suốt
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setValue(0.7)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.valueChanged.connect(self._on_settings_changed)
        form_layout.addRow("Độ trong suốt:", self.opacity_spin)
        
        # Thêm form layout vào layout chính
        main_layout.addLayout(form_layout)
        
        # Checkbox cho việc hiển thị
        self.show_isodose_check = QCheckBox("Hiển thị đường đồng liều")
        self.show_isodose_check.setChecked(True)
        self.show_isodose_check.stateChanged.connect(self._on_settings_changed)
        main_layout.addWidget(self.show_isodose_check)
        
        self.show_colorwash_check = QCheckBox("Hiển thị colorwash")
        self.show_colorwash_check.setChecked(True)
        self.show_colorwash_check.stateChanged.connect(self._on_settings_changed)
        main_layout.addWidget(self.show_colorwash_check)
        
        # Thêm khoảng trống
        main_layout.addStretch()
        
        # Nút cập nhật
        self.update_button = QPushButton("Cập nhật hiển thị")
        self.update_button.clicked.connect(self._on_update_clicked)
        main_layout.addWidget(self.update_button)
    
    def _on_isodose_preset_changed(self):
        """Xử lý sự kiện khi thay đổi cài đặt sẵn isodose."""
        # TODO: Triển khai việc thay đổi cài đặt isodose
        self._on_settings_changed()
    
    def _on_settings_changed(self):
        """Xử lý sự kiện khi thay đổi cài đặt."""
        # Chưa phát tín hiệu để tránh cập nhật liên tục
        pass
    
    def _on_update_clicked(self):
        """Xử lý sự kiện khi nhấn nút cập nhật."""
        # Tạo từ điển cài đặt
        settings = {
            "reference_dose": self.reference_dose_spin.value(),
            "opacity": self.opacity_spin.value(),
            "show_isodose": self.show_isodose_check.isChecked(),
            "show_colorwash": self.show_colorwash_check.isChecked(),
            "isodose_preset": self.isodose_combo.currentText()
        }
        
        # Phát tín hiệu với cài đặt
        self.settings_changed.emit(settings)


class DoseDisplayWidget(QWidget):
    """Widget hiển thị phân bố liều."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị liều.
        
        Args:
            parent: Widget cha
        """
        super().__init__(parent)
        
        # Khởi tạo các thuộc tính
        self.dose_data = None
        self.anatomy_data = None
        self.structures = {}
        self.dose_colorwash = DoseColorwash()
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Widget hiển thị lát cắt
        self.image_widget = ImageSliceWidget()
        main_layout.addWidget(self.image_widget)
        
        # Thêm dữ liệu mẫu
        self._add_sample_data()
    
    def _add_sample_data(self):
        """Thêm dữ liệu mẫu cho hiển thị."""
        # Tạo dữ liệu mẫu
        image_size = (128, 128, 20)
        self.anatomy_data = np.ones(image_size) * 100  # Mô phỏng CT với giá trị HU
        
        # Tạo cấu trúc hình cầu ở giữa
        center = (image_size[0]//2, image_size[1]//2, image_size[2]//2)
        radius = min(image_size) // 4
        
        x, y, z = np.ogrid[:image_size[0], :image_size[1], :image_size[2]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        sphere = dist_from_center <= radius
        
        # Tạo phân bố liều mẫu (cao nhất ở trung tâm, giảm dần ra ngoài)
        self.dose_data = np.zeros(image_size)
        self.dose_data[sphere] = 50 * (1 - dist_from_center[sphere] / radius)
        
        # Cập nhật hiển thị
        self.set_slice_data(10)  # Hiển thị lát cắt ở giữa
    
    def set_slice_data(self, slice_index):
        """
        Cập nhật dữ liệu lát cắt để hiển thị.
        
        Args:
            slice_index: Chỉ số lát cắt
        """
        if self.dose_data is None or self.anatomy_data is None:
            return
        
        if slice_index >= 0 and slice_index < self.dose_data.shape[2]:
            # Lấy dữ liệu lát cắt
            dose_slice = self.dose_data[:, :, slice_index]
            anatomy_slice = self.anatomy_data[:, :, slice_index]
            
            # Hiển thị dữ liệu
            self.image_widget.set_background_data(anatomy_slice)
            self.image_widget.set_dose_data(dose_slice)
            self.image_widget.update_display()


class BiologicalModelWidget(QWidget):
    """Widget cho các mô hình sinh học."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget mô hình sinh học.
        
        Args:
            parent: Widget cha
        """
        super().__init__(parent)
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Tiêu đề
        title_label = QLabel("Các chỉ số sinh học")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Phân chia thành hai cột
        columns_layout = QHBoxLayout()
        
        # Cột trái: TCP
        tcp_group = QGroupBox("TCP (Tumor Control Probability)")
        tcp_layout = QVBoxLayout(tcp_group)
        
        # Bảng TCP
        self.tcp_table = QTableWidget(3, 2)
        self.tcp_table.setHorizontalHeaderLabels(["Cấu trúc", "TCP"])
        self.tcp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Dữ liệu mẫu
        self.tcp_table.setItem(0, 0, QTableWidgetItem("GTV"))
        self.tcp_table.setItem(0, 1, QTableWidgetItem("94.2%"))
        self.tcp_table.setItem(1, 0, QTableWidgetItem("CTV"))
        self.tcp_table.setItem(1, 1, QTableWidgetItem("92.7%"))
        self.tcp_table.setItem(2, 0, QTableWidgetItem("PTV"))
        self.tcp_table.setItem(2, 1, QTableWidgetItem("89.5%"))
        
        tcp_layout.addWidget(self.tcp_table)
        
        # Cột phải: NTCP
        ntcp_group = QGroupBox("NTCP (Normal Tissue Complication Probability)")
        ntcp_layout = QVBoxLayout(ntcp_group)
        
        # Bảng NTCP
        self.ntcp_table = QTableWidget(4, 2)
        self.ntcp_table.setHorizontalHeaderLabels(["Cơ quan", "NTCP"])
        self.ntcp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Dữ liệu mẫu
        self.ntcp_table.setItem(0, 0, QTableWidgetItem("Parotid L"))
        self.ntcp_table.setItem(0, 1, QTableWidgetItem("14.3%"))
        self.ntcp_table.setItem(1, 0, QTableWidgetItem("Parotid R"))
        self.ntcp_table.setItem(1, 1, QTableWidgetItem("12.8%"))
        self.ntcp_table.setItem(2, 0, QTableWidgetItem("Spinal Cord"))
        self.ntcp_table.setItem(2, 1, QTableWidgetItem("2.1%"))
        self.ntcp_table.setItem(3, 0, QTableWidgetItem("Brainstem"))
        self.ntcp_table.setItem(3, 1, QTableWidgetItem("0.5%"))
        
        ntcp_layout.addWidget(self.ntcp_table)
        
        # Thêm các cột vào layout
        columns_layout.addWidget(tcp_group)
        columns_layout.addWidget(ntcp_group)
        
        main_layout.addLayout(columns_layout)
        
        # Thêm giải thích
        note_label = QLabel(
            "Lưu ý: Các giá trị TCP/NTCP được tính dựa trên mô hình LKB "
            "(Lyman-Kutcher-Burman). Các tham số mô hình được lấy từ "
            "dữ liệu lâm sàng đã công bố."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: gray;")
        main_layout.addWidget(note_label)
        
        # Thêm nút Tính toán lại
        recalculate_button = QPushButton("Tính toán lại")
        recalculate_button.clicked.connect(self._on_recalculate)
        main_layout.addWidget(recalculate_button)
    
    def _on_recalculate(self):
        """Xử lý sự kiện khi nhấn nút Tính toán lại."""
        # TODO: Triển khai việc tính toán lại TCP/NTCP
        logger.info("Đang tính toán lại các chỉ số sinh học...")


class PlanEvaluationWidget(QWidget):
    """
    Widget đánh giá kế hoạch xạ trị.
    
    Widget này tích hợp các công cụ để đánh giá chất lượng của kế hoạch
    xạ trị, bao gồm DVH, các chỉ số lâm sàng, và phân tích phân bố liều.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget đánh giá kế hoạch.
        
        Args:
            parent: Widget cha
        """
        super().__init__(parent)
        
        # Khởi tạo các thuộc tính
        self.current_plan = None
        self.dvh_calculator = DVHCalculator()
        self.dvh_plotter = DVHPlotter()
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Tạo tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab 1: DVH và chỉ số
        dvh_tab = QWidget()
        dvh_layout = QHBoxLayout(dvh_tab)
        
        # Bên trái: Biểu đồ DVH
        dvh_canvas_container = QWidget()
        dvh_canvas_layout = QVBoxLayout(dvh_canvas_container)
        
        self.dvh_canvas = DVHCanvas(width=6, height=5)
        dvh_canvas_layout.addWidget(self.dvh_canvas)
        
        # Điều khiển DVH
        dvh_controls = QWidget()
        dvh_controls_layout = QHBoxLayout(dvh_controls)
        
        self.relative_volume_check = QCheckBox("Thể tích tương đối")
        self.relative_volume_check.setChecked(True)
        dvh_controls_layout.addWidget(self.relative_volume_check)
        
        self.relative_dose_check = QCheckBox("Liều tương đối")
        dvh_controls_layout.addWidget(self.relative_dose_check)
        
        self.cumulative_radio = QCheckBox("DVH tích lũy")
        self.cumulative_radio.setChecked(True)
        dvh_controls_layout.addWidget(self.cumulative_radio)
        
        self.differential_radio = QCheckBox("DVH vi phân")
        dvh_controls_layout.addWidget(self.differential_radio)
        
        dvh_canvas_layout.addWidget(dvh_controls)
        
        # Bên phải: Bảng chỉ số
        self.metrics_table = MetricsTableWidget()
        
        # Thêm vào layout DVH
        dvh_layout.addWidget(dvh_canvas_container, 2)
        dvh_layout.addWidget(self.metrics_table, 1)
        
        # Tab 2: Phân bố liều
        dose_tab = QWidget()
        dose_layout = QHBoxLayout(dose_tab)
        
        # Bên trái: Hiển thị liều
        self.dose_display = DoseDisplayWidget()
        
        # Bên phải: Điều khiển
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        
        self.isodose_control = IsodoseControlWidget()
        self.isodose_control.settings_changed.connect(self._on_isodose_settings_changed)
        
        # Điều khiển lát cắt
        slice_control = QWidget()
        slice_control_layout = QHBoxLayout(slice_control)
        
        slice_label = QLabel("Lát cắt:")
        self.slice_slider = QSpinBox()
        self.slice_slider.setRange(0, 19)  # Mặc định 20 lát cắt
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        
        slice_control_layout.addWidget(slice_label)
        slice_control_layout.addWidget(self.slice_slider)
        
        right_panel_layout.addWidget(self.isodose_control)
        right_panel_layout.addWidget(slice_control)
        right_panel_layout.addStretch()
        
        # Thêm vào layout Dose
        dose_layout.addWidget(self.dose_display, 3)
        dose_layout.addWidget(right_panel, 1)
        
        # Tab 3: Các chỉ số sinh học
        bio_tab = QWidget()
        bio_layout = QVBoxLayout(bio_tab)
        
        self.bio_widget = BiologicalModelWidget()
        bio_layout.addWidget(self.bio_widget)
        
        # Thêm các tab vào tab widget
        self.tab_widget.addTab(dvh_tab, "DVH & Chỉ số")
        self.tab_widget.addTab(dose_tab, "Phân bố liều")
        self.tab_widget.addTab(bio_tab, "Mô hình sinh học")
        
        # Thêm nút báo cáo
        report_button = QPushButton("Tạo báo cáo đánh giá")
        report_button.clicked.connect(self._on_create_report)
        main_layout.addWidget(report_button)
    
    def _on_slice_changed(self, value):
        """
        Xử lý sự kiện khi thay đổi lát cắt.
        
        Args:
            value: Chỉ số lát cắt mới
        """
        self.dose_display.set_slice_data(value)
    
    def _on_isodose_settings_changed(self, settings):
        """
        Xử lý sự kiện khi thay đổi cài đặt isodose.
        
        Args:
            settings: Từ điển chứa các cài đặt
        """
        # TODO: Cập nhật hiển thị isodose dựa trên cài đặt
        logger.info("Cập nhật cài đặt isodose: %s", settings)
    
    def _on_create_report(self):
        """Xử lý sự kiện khi nhấn nút tạo báo cáo."""
        # TODO: Triển khai việc tạo báo cáo đánh giá
        logger.info("Đang tạo báo cáo đánh giá kế hoạch...")
    
    def set_plan_data(self, plan_data):
        """
        Thiết lập dữ liệu kế hoạch để đánh giá.
        
        Args:
            plan_data: Dữ liệu kế hoạch xạ trị
        """
        self.current_plan = plan_data
        # TODO: Cập nhật tất cả các hiển thị dựa trên dữ liệu kế hoạch
        logger.info("Đã cập nhật dữ liệu kế hoạch mới")


if __name__ == "__main__":
    """Kiểm thử widget đánh giá kế hoạch."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    window = PlanEvaluationWidget()
    window.setWindowTitle("Đánh giá kế hoạch xạ trị")
    window.resize(1200, 800)
    window.show()
    
    sys.exit(app.exec_())
