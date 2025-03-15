#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab lập kế hoạch (Planning Tab) cho QuangTPS.

Module này cung cấp giao diện để tạo và chỉnh sửa kế hoạch xạ trị,
bao gồm các công cụ để thiết lập kỹ thuật điều trị, thông số chùm tia,
và tối ưu hóa kế hoạch.
"""

import os
import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTreeWidget, QTreeWidgetItem, QSlider, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QDate, QSize
from PyQt5.QtGui import QFont, QIcon

from quangtps.treatment.techniques.dcat import DCAT
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.stereotactic import SRS, SBRT
from quangtps.treatment.treatment_technique_selector import TechniqueSuitabilityCalculator
from quangtps.ui.structure_view import StructureView

logger = logging.getLogger(__name__)


class PlanningTab(QWidget):
    """
    Tab lập kế hoạch xạ trị.
    
    Tab này bao gồm các công cụ để tạo và chỉnh sửa kế hoạch xạ trị,
    thiết lập kỹ thuật điều trị, thông số chùm tia, và tối ưu hóa kế hoạch.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo tab lập kế hoạch.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Trạng thái
        self.current_plan = None
        self.current_technique = None
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab lập kế hoạch hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QHBoxLayout(self)
        
        # Splitter chính
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Panel bên trái (cấu trúc và thông tin kế hoạch)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Tab widget cho panel bên trái
        self.left_tabs = QTabWidget()
        self.left_layout.addWidget(self.left_tabs)
        
        # Tab kế hoạch
        self.plan_info_widget = QWidget()
        self.plan_info_layout = QVBoxLayout(self.plan_info_widget)
        
        # Nhóm thông tin kế hoạch
        self.plan_group = QGroupBox("Thông tin kế hoạch")
        self.plan_form = QFormLayout(self.plan_group)
        
        self.plan_name_field = QLineEdit()
        self.plan_form.addRow("Tên kế hoạch:", self.plan_name_field)
        
        self.plan_description_field = QTextEdit()
        self.plan_description_field.setMaximumHeight(100)
        self.plan_form.addRow("Mô tả:", self.plan_description_field)
        
        self.plan_date_field = QDateEdit()
        self.plan_date_field.setDisplayFormat("dd/MM/yyyy")
        self.plan_date_field.setCalendarPopup(True)
        self.plan_date_field.setDate(QDate.currentDate())
        self.plan_form.addRow("Ngày tạo:", self.plan_date_field)
        
        self.plan_intent_field = QComboBox()
        self.plan_intent_field.addItems(["Điều trị triệt căn", "Điều trị triệu chứng", "Điều trị bổ trợ", "Khác"])
        self.plan_form.addRow("Mục đích:", self.plan_intent_field)
        
        self.plan_status_field = QComboBox()
        self.plan_status_field.addItems(["Đang dự thảo", "Đang xem xét", "Đã phê duyệt", "Hoàn thành", "Hủy bỏ"])
        self.plan_form.addRow("Trạng thái:", self.plan_status_field)
        
        self.plan_info_layout.addWidget(self.plan_group)
        
        # Nhóm liều lượng
        self.dose_group = QGroupBox("Liều lượng")
        self.dose_form = QFormLayout(self.dose_group)
        
        self.prescribed_dose_field = QDoubleSpinBox()
        self.prescribed_dose_field.setRange(0, 100)
        self.prescribed_dose_field.setSuffix(" Gy")
        self.prescribed_dose_field.setDecimals(2)
        self.dose_form.addRow("Liều chỉ định:", self.prescribed_dose_field)
        
        self.fractions_field = QSpinBox()
        self.fractions_field.setRange(1, 40)
        self.fractions_field.setValue(1)
        self.dose_form.addRow("Số phân liều:", self.fractions_field)
        
        self.dose_per_fraction_field = QDoubleSpinBox()
        self.dose_per_fraction_field.setRange(0, 20)
        self.dose_per_fraction_field.setSuffix(" Gy")
        self.dose_per_fraction_field.setDecimals(2)
        self.dose_form.addRow("Liều/phân liều:", self.dose_per_fraction_field)
        
        self.plan_info_layout.addWidget(self.dose_group)
        
        # Nút lưu kế hoạch
        self.save_plan_button = QPushButton("Lưu kế hoạch")
        self.save_plan_button.clicked.connect(self._save_plan)
        self.plan_info_layout.addWidget(self.save_plan_button, alignment=Qt.AlignRight)
        
        # Thêm tab kế hoạch
        self.left_tabs.addTab(self.plan_info_widget, "Kế hoạch")
        
        # Tab cấu trúc
        self.structures_widget = QWidget()
        self.structures_layout = QVBoxLayout(self.structures_widget)
        
        # Hiển thị cấu trúc
        self.structure_view = StructureView()
        self.structures_layout.addWidget(self.structure_view)
        
        # Thêm tab cấu trúc
        self.left_tabs.addTab(self.structures_widget, "Cấu trúc")
        
        # Tab ràng buộc
        self.constraints_widget = QWidget()
        self.constraints_layout = QVBoxLayout(self.constraints_widget)
        
        # Bảng ràng buộc
        self.constraints_table = QTableWidget(0, 4)
        self.constraints_table.setHorizontalHeaderLabels(["Cấu trúc", "Loại", "Giá trị", "Mức độ ưu tiên"])
        self.constraints_table.horizontalHeader().setStretchLastSection(True)
        self.constraints_layout.addWidget(self.constraints_table)
        
        # Nút thêm ràng buộc
        self.add_constraint_button = QPushButton("Thêm ràng buộc")
        self.add_constraint_button.clicked.connect(self._add_constraint)
        self.constraints_layout.addWidget(self.add_constraint_button, alignment=Qt.AlignRight)
        
        # Thêm tab ràng buộc
        self.left_tabs.addTab(self.constraints_widget, "Ràng buộc")
        
        # Panel bên phải (thiết lập kỹ thuật và chùm tia)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Tab widget cho panel bên phải
        self.right_tabs = QTabWidget()
        self.right_layout.addWidget(self.right_tabs)
        
        # Tab kỹ thuật
        self.technique_widget = QWidget()
        self.technique_layout = QVBoxLayout(self.technique_widget)
        
        # Nhóm lựa chọn kỹ thuật
        self.technique_group = QGroupBox("Kỹ thuật xạ trị")
        self.technique_button_layout = QVBoxLayout(self.technique_group)
        
        # Radio buttons cho các kỹ thuật
        self.technique_button_group = QButtonGroup(self)
        
        self.dcat_radio = QRadioButton("DCAT - Dynamic Conformal Arc Therapy")
        self.technique_button_group.addButton(self.dcat_radio)
        self.technique_button_layout.addWidget(self.dcat_radio)
        
        self.imrt_radio = QRadioButton("IMRT - Intensity Modulated Radiation Therapy")
        self.technique_button_group.addButton(self.imrt_radio)
        self.technique_button_layout.addWidget(self.imrt_radio)
        
        self.vmat_radio = QRadioButton("VMAT - Volumetric Modulated Arc Therapy")
        self.technique_button_group.addButton(self.vmat_radio)
        self.technique_button_layout.addWidget(self.vmat_radio)
        
        self.srs_radio = QRadioButton("SRS - Stereotactic Radiosurgery")
        self.technique_button_group.addButton(self.srs_radio)
        self.technique_button_layout.addWidget(self.srs_radio)
        
        self.sbrt_radio = QRadioButton("SBRT - Stereotactic Body Radiation Therapy")
        self.technique_button_group.addButton(self.sbrt_radio)
        self.technique_button_layout.addWidget(self.sbrt_radio)
        
        # Kết nối sự kiện thay đổi lựa chọn
        self.technique_button_group.buttonClicked.connect(self._technique_selected)
        
        self.technique_layout.addWidget(self.technique_group)
        
        # Nhóm tính toán độ phù hợp
        self.suitability_group = QGroupBox("Độ phù hợp của kỹ thuật")
        self.suitability_layout = QVBoxLayout(self.suitability_group)
        
        # Label hiển thị mức độ phù hợp
        self.suitability_label = QLabel("Vui lòng nhập thông tin bệnh nhân và kế hoạch để tính toán độ phù hợp")
        self.suitability_label.setWordWrap(True)
        self.suitability_layout.addWidget(self.suitability_label)
        
        # Nút tính toán độ phù hợp
        self.calculate_suitability_button = QPushButton("Tính toán độ phù hợp")
        self.calculate_suitability_button.clicked.connect(self._calculate_technique_suitability)
        self.suitability_layout.addWidget(self.calculate_suitability_button)
        
        self.technique_layout.addWidget(self.suitability_group)
        
        # Thêm tab kỹ thuật
        self.right_tabs.addTab(self.technique_widget, "Kỹ thuật")
        
        # Tab chùm tia
        self.beams_widget = QWidget()
        self.beams_layout = QVBoxLayout(self.beams_widget)
        
        # Bảng chùm tia
        self.beams_table = QTableWidget(0, 6)
        self.beams_table.setHorizontalHeaderLabels(["ID", "Tên", "Góc gantry", "Góc bàn", "MU", "Trạng thái"])
        self.beams_table.horizontalHeader().setStretchLastSection(True)
        self.beams_layout.addWidget(self.beams_table)
        
        # Nút thêm chùm tia
        self.beam_buttons_layout = QHBoxLayout()
        
        self.add_beam_button = QPushButton("Thêm chùm tia")
        self.add_beam_button.clicked.connect(self._add_beam)
        self.beam_buttons_layout.addWidget(self.add_beam_button)
        
        self.add_arc_button = QPushButton("Thêm cung")
        self.add_arc_button.clicked.connect(self._add_arc)
        self.beam_buttons_layout.addWidget(self.add_arc_button)
        
        self.edit_beam_button = QPushButton("Chỉnh sửa")
        self.edit_beam_button.clicked.connect(self._edit_beam)
        self.beam_buttons_layout.addWidget(self.edit_beam_button)
        
        self.delete_beam_button = QPushButton("Xóa")
        self.delete_beam_button.clicked.connect(self._delete_beam)
        self.beam_buttons_layout.addWidget(self.delete_beam_button)
        
        self.beams_layout.addLayout(self.beam_buttons_layout)
        
        # Thêm tab chùm tia
        self.right_tabs.addTab(self.beams_widget, "Chùm tia")
        
        # Tab tối ưu hóa
        self.optimization_widget = QWidget()
        self.optimization_layout = QVBoxLayout(self.optimization_widget)
        
        # Nhóm thiết lập tối ưu hóa
        self.optimization_group = QGroupBox("Thiết lập tối ưu hóa")
        self.optimization_form = QFormLayout(self.optimization_group)
        
        self.opt_algorithm_field = QComboBox()
        self.opt_algorithm_field.addItems(["Simulated Annealing", "Genetic Algorithm", "Gradient Descent", "IPOPT"])
        self.optimization_form.addRow("Thuật toán:", self.opt_algorithm_field)
        
        self.opt_iterations_field = QSpinBox()
        self.opt_iterations_field.setRange(10, 1000)
        self.opt_iterations_field.setValue(100)
        self.optimization_form.addRow("Số lần lặp:", self.opt_iterations_field)
        
        self.opt_convergence_field = QDoubleSpinBox()
        self.opt_convergence_field.setRange(0.001, 0.1)
        self.opt_convergence_field.setValue(0.01)
        self.opt_convergence_field.setDecimals(4)
        self.optimization_form.addRow("Ngưỡng hội tụ:", self.opt_convergence_field)
        
        self.optimization_layout.addWidget(self.optimization_group)
        
        # Nút tối ưu hóa
        self.run_optimization_button = QPushButton("Chạy tối ưu hóa")
        self.run_optimization_button.clicked.connect(self._run_optimization)
        self.optimization_layout.addWidget(self.run_optimization_button, alignment=Qt.AlignRight)
        
        # Thêm tab tối ưu hóa
        self.right_tabs.addTab(self.optimization_widget, "Tối ưu hóa")
        
        # Thêm các panel vào splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        
        # Thiết lập kích thước ban đầu
        self.main_splitter.setSizes([400, 600])
        
        # Vô hiệu hóa các tab liên quan đến kỹ thuật khi chưa có kế hoạch
        self.right_tabs.setEnabled(False)
    
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
            self._populate_plan_data()
            self.right_tabs.setEnabled(True)
        else:
            self._clear_plan_data()
            self.right_tabs.setEnabled(False)
    
    def _populate_plan_data(self):
        """Điền thông tin kế hoạch vào giao diện."""
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
        pass
    
    def _clear_plan_data(self):
        """Xóa thông tin kế hoạch khỏi giao diện."""
        # Xóa thông tin kế hoạch
        self.plan_name_field.clear()
        self.plan_description_field.clear()
        self.plan_date_field.setDate(QDate.currentDate())
        self.plan_intent_field.setCurrentIndex(0)
        self.plan_status_field.setCurrentIndex(0)
        
        # Xóa thông tin liều lượng
        self.prescribed_dose_field.setValue(0)
        self.fractions_field.setValue(1)
        self.dose_per_fraction_field.setValue(0)
        
        # Xóa bảng ràng buộc
        self.constraints_table.setRowCount(0)
        
        # Xóa lựa chọn kỹ thuật
        self.technique_button_group.setExclusive(False)
        for button in self.technique_button_group.buttons():
            button.setChecked(False)
        self.technique_button_group.setExclusive(True)
        
        # Xóa bảng chùm tia
        self.beams_table.setRowCount(0)
    
    def _save_plan(self):
        """Lưu thông tin kế hoạch."""
        logger.info("Lưu thông tin kế hoạch")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _add_constraint(self):
        """Thêm ràng buộc mới."""
        logger.info("Thêm ràng buộc")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _technique_selected(self, button):
        """
        Xử lý sự kiện khi một kỹ thuật được chọn.
        
        Parameters
        ----------
        button : QRadioButton
            Nút radio được chọn
        """
        logger.info(f"Kỹ thuật được chọn: {button.text()}")
        
        # Xác định kỹ thuật dựa trên nút được chọn
        if button is self.dcat_radio:
            self.current_technique = "DCAT"
        elif button is self.imrt_radio:
            self.current_technique = "IMRT"
        elif button is self.vmat_radio:
            self.current_technique = "VMAT"
        elif button is self.srs_radio:
            self.current_technique = "SRS"
        elif button is self.sbrt_radio:
            self.current_technique = "SBRT"
        
        # Cập nhật giao diện dựa trên kỹ thuật được chọn
        self._update_beam_controls()
    
    def _update_beam_controls(self):
        """Cập nhật các điều khiển chùm tia dựa trên kỹ thuật được chọn."""
        # Kích hoạt/vô hiệu hóa các nút dựa trên kỹ thuật
        if self.current_technique in ["DCAT", "VMAT", "SRS", "SBRT"]:
            self.add_arc_button.setEnabled(True)
        else:
            self.add_arc_button.setEnabled(False)
        
        # Xóa bảng chùm tia
        self.beams_table.setRowCount(0)
    
    def _calculate_technique_suitability(self):
        """Tính toán độ phù hợp của các kỹ thuật."""
        logger.info("Tính toán độ phù hợp của kỹ thuật")
        
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
        # Hiện tại chỉ mô phỏng kết quả
        
        site = self.plan_intent_field.currentText()
        prescription = self.prescribed_dose_field.value()
        
        # Mô phỏng tính toán độ phù hợp
        suitability_text = (
            "Độ phù hợp của các kỹ thuật (thang điểm 1-10):\n"
            "- DCAT: 7/10 (Phù hợp cho hầu hết các trường hợp)\n"
            "- IMRT: 9/10 (Rất phù hợp cho các trường hợp phức tạp)\n"
            "- VMAT: 8/10 (Phù hợp cho các trường hợp cần phân bố liều đồng đều)\n"
            "- SRS: 5/10 (Phù hợp cho các tổn thương nhỏ trong não)\n"
            "- SBRT: 6/10 (Phù hợp cho các tổn thương ở thân)"
        )
        
        self.suitability_label.setText(suitability_text)
    
    def _add_beam(self):
        """Thêm chùm tia mới."""
        logger.info("Thêm chùm tia")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _add_arc(self):
        """Thêm cung mới."""
        logger.info("Thêm cung")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _edit_beam(self):
        """Chỉnh sửa chùm tia được chọn."""
        logger.info("Chỉnh sửa chùm tia")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _delete_beam(self):
        """Xóa chùm tia được chọn."""
        logger.info("Xóa chùm tia")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
    
    def _run_optimization(self):
        """Chạy tối ưu hóa kế hoạch."""
        logger.info("Chạy tối ưu hóa kế hoạch")
        # Chưa có dữ liệu thực tế, sẽ được triển khai khi có dữ liệu
