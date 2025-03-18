#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho Dialog Beam.

Module này cung cấp dialog thêm mới hoặc chỉnh sửa chùm tia.
"""

import logging
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLineEdit, QDoubleSpinBox, QComboBox,
    QPushButton, QGroupBox, QMessageBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QTabWidget, QWidget, QDialogButtonBox
)
from PyQt5.QtCore import Qt

from quangtps.core.beam_types import BeamType
from quangtps.planning.beam import BeamSetup
from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.beams.beam_modifiers import Wedge, Bolus, Block

logger = logging.getLogger(__name__)


class BeamDialog(QDialog):
    """
    Dialog cho thêm mới hoặc chỉnh sửa chùm tia.
    
    Dialog này cung cấp giao diện người dùng để người dùng có thể thêm mới
    hoặc chỉnh sửa thông tin của một chùm tia, bao gồm thông số hình học,
    năng lượng, và các bộ điều chỉnh (modifiers).
    """
    
    def __init__(self, parent=None, beam_setup: Optional[BeamSetup] = None):
        """
        Khởi tạo dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        beam_setup : BeamSetup, optional
            Đối tượng BeamSetup cần chỉnh sửa, hoặc None nếu thêm mới
        """
        super().__init__(parent)
        self.setWindowTitle("Thêm/Sửa Chùm Tia")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        self.beam_setup = beam_setup if beam_setup else BeamSetup()
        
        # Tạo giao diện
        self._create_ui()
        
        # Nếu là chỉnh sửa, load dữ liệu từ beam_setup
        if beam_setup:
            self._load_beam_data()
    
    def _create_ui(self):
        """Tạo giao diện người dùng cho dialog."""
        layout = QVBoxLayout(self)
        
        # Thông tin cơ bản
        basic_group = QGroupBox("Thông tin cơ bản")
        basic_form = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        basic_form.addRow("Tên chùm tia:", self.name_edit)
        
        self.beam_type_combo = QComboBox()
        for beam_type in BeamType:
            self.beam_type_combo.addItem(beam_type.value, beam_type)
        basic_form.addRow("Loại chùm tia:", self.beam_type_combo)
        
        self.energy_combo = QComboBox()
        self.energy_combo.setEditable(True)
        energies = ["6MV", "10MV", "15MV", "6MeV", "9MeV", "12MeV", "15MeV", "18MeV", "21MeV"]
        for energy in energies:
            self.energy_combo.addItem(energy)
        basic_form.addRow("Năng lượng:", self.energy_combo)
        
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0, 10000)
        self.mu_spin.setValue(100)
        self.mu_spin.setSuffix(" MU")
        basic_form.addRow("Monitor Units:", self.mu_spin)
        
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 1)
        self.weight_spin.setValue(1)
        self.weight_spin.setSingleStep(0.1)
        self.weight_spin.setDecimals(2)
        basic_form.addRow("Trọng số:", self.weight_spin)
        
        # Tab widget cho các nhóm thông số khác
        tab_widget = QTabWidget()
        
        # Tab hình học
        geometry_tab = QWidget()
        geometry_layout = QFormLayout(geometry_tab)
        
        self.gantry_angle_spin = QDoubleSpinBox()
        self.gantry_angle_spin.setRange(0, 360)
        self.gantry_angle_spin.setValue(0)
        self.gantry_angle_spin.setSuffix("°")
        geometry_layout.addRow("Góc gantry:", self.gantry_angle_spin)
        
        self.collimator_angle_spin = QDoubleSpinBox()
        self.collimator_angle_spin.setRange(0, 360)
        self.collimator_angle_spin.setValue(0)
        self.collimator_angle_spin.setSuffix("°")
        geometry_layout.addRow("Góc collimator:", self.collimator_angle_spin)
        
        self.couch_angle_spin = QDoubleSpinBox()
        self.couch_angle_spin.setRange(0, 360)
        self.couch_angle_spin.setValue(0)
        self.couch_angle_spin.setSuffix("°")
        geometry_layout.addRow("Góc bàn:", self.couch_angle_spin)
        
        self.ssd_spin = QDoubleSpinBox()
        self.ssd_spin.setRange(50, 200)
        self.ssd_spin.setValue(100)
        self.ssd_spin.setSuffix(" cm")
        geometry_layout.addRow("SSD:", self.ssd_spin)
        
        # Kích thước trường
        field_size_widget = QWidget()
        field_size_layout = QHBoxLayout(field_size_widget)
        field_size_layout.setContentsMargins(0, 0, 0, 0)
        
        self.field_width_spin = QDoubleSpinBox()
        self.field_width_spin.setRange(1, 40)
        self.field_width_spin.setValue(10)
        self.field_width_spin.setSuffix(" cm")
        field_size_layout.addWidget(QLabel("Rộng:"))
        field_size_layout.addWidget(self.field_width_spin)
        
        self.field_height_spin = QDoubleSpinBox()
        self.field_height_spin.setRange(1, 40)
        self.field_height_spin.setValue(10)
        self.field_height_spin.setSuffix(" cm")
        field_size_layout.addWidget(QLabel("Cao:"))
        field_size_layout.addWidget(self.field_height_spin)
        
        geometry_layout.addRow("Kích thước trường:", field_size_widget)

        # Tọa độ isocenter
        isocenter_widget = QWidget()
        isocenter_layout = QGridLayout(isocenter_widget)
        isocenter_layout.setContentsMargins(0, 0, 0, 0)
        
        self.isocenter_x_spin = QDoubleSpinBox()
        self.isocenter_x_spin.setRange(-500, 500)
        self.isocenter_x_spin.setValue(0)
        self.isocenter_x_spin.setSuffix(" mm")
        isocenter_layout.addWidget(QLabel("X:"), 0, 0)
        isocenter_layout.addWidget(self.isocenter_x_spin, 0, 1)
        
        self.isocenter_y_spin = QDoubleSpinBox()
        self.isocenter_y_spin.setRange(-500, 500)
        self.isocenter_y_spin.setValue(0)
        self.isocenter_y_spin.setSuffix(" mm")
        isocenter_layout.addWidget(QLabel("Y:"), 0, 2)
        isocenter_layout.addWidget(self.isocenter_y_spin, 0, 3)
        
        self.isocenter_z_spin = QDoubleSpinBox()
        self.isocenter_z_spin.setRange(-500, 500)
        self.isocenter_z_spin.setValue(0)
        self.isocenter_z_spin.setSuffix(" mm")
        isocenter_layout.addWidget(QLabel("Z:"), 1, 0)
        isocenter_layout.addWidget(self.isocenter_z_spin, 1, 1)
        
        geometry_layout.addRow("Isocenter:", isocenter_widget)
        
        tab_widget.addTab(geometry_tab, "Hình học")
        
        # Tab bộ điều chỉnh (modifiers)
        modifiers_tab = QWidget()
        modifiers_layout = QVBoxLayout(modifiers_tab)
        
        # Nêm (Wedge)
        wedge_group = QGroupBox("Nêm (Wedge)")
        wedge_layout = QFormLayout(wedge_group)
        
        self.use_wedge_check = QPushButton("Thêm nêm")
        self.use_wedge_check.setCheckable(True)
        wedge_layout.addRow(self.use_wedge_check)
        
        self.wedge_angle_spin = QDoubleSpinBox()
        self.wedge_angle_spin.setRange(0, 60)
        self.wedge_angle_spin.setValue(15)
        self.wedge_angle_spin.setSuffix("°")
        self.wedge_angle_spin.setEnabled(False)
        wedge_layout.addRow("Góc nêm:", self.wedge_angle_spin)
        
        self.wedge_orientation_combo = QComboBox()
        orientations = ["IN", "OUT", "LEFT", "RIGHT"]
        for orient in orientations:
            self.wedge_orientation_combo.addItem(orient)
        self.wedge_orientation_combo.setEnabled(False)
        wedge_layout.addRow("Hướng nêm:", self.wedge_orientation_combo)
        
        modifiers_layout.addWidget(wedge_group)
        
        # Bolus
        bolus_group = QGroupBox("Bolus")
        bolus_layout = QFormLayout(bolus_group)
        
        self.use_bolus_check = QPushButton("Thêm bolus")
        self.use_bolus_check.setCheckable(True)
        bolus_layout.addRow(self.use_bolus_check)
        
        self.bolus_thickness_spin = QDoubleSpinBox()
        self.bolus_thickness_spin.setRange(0, 20)
        self.bolus_thickness_spin.setValue(0.5)
        self.bolus_thickness_spin.setSuffix(" cm")
        self.bolus_thickness_spin.setEnabled(False)
        bolus_layout.addRow("Độ dày:", self.bolus_thickness_spin)
        
        self.bolus_material_combo = QComboBox()
        materials = ["Water", "Wax", "Custom"]
        for material in materials:
            self.bolus_material_combo.addItem(material)
        self.bolus_material_combo.setEnabled(False)
        bolus_layout.addRow("Vật liệu:", self.bolus_material_combo)
        
        modifiers_layout.addWidget(bolus_group)
        
        # Block
        block_group = QGroupBox("Chặn (Block)")
        block_layout = QFormLayout(block_group)
        
        self.use_block_check = QPushButton("Thêm block")
        self.use_block_check.setCheckable(True)
        block_layout.addRow(self.use_block_check)
        
        self.block_thickness_spin = QDoubleSpinBox()
        self.block_thickness_spin.setRange(0, 10)
        self.block_thickness_spin.setValue(7.5)
        self.block_thickness_spin.setSuffix(" cm")
        self.block_thickness_spin.setEnabled(False)
        block_layout.addRow("Độ dày:", self.block_thickness_spin)
        
        modifiers_layout.addWidget(block_group)
        
        tab_widget.addTab(modifiers_tab, "Bộ điều chỉnh")
        
        # Tab metadata
        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        
        # Bảng metadata
        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Tên", "Giá trị"])
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.metadata_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        metadata_layout.addWidget(self.metadata_table)
        
        # Nút thêm/xóa metadata
        metadata_buttons = QHBoxLayout()
        self.add_metadata_button = QPushButton("Thêm")
        self.remove_metadata_button = QPushButton("Xóa")
        metadata_buttons.addWidget(self.add_metadata_button)
        metadata_buttons.addWidget(self.remove_metadata_button)
        metadata_layout.addLayout(metadata_buttons)
        
        tab_widget.addTab(metadata_tab, "Metadata")
        
        # Thêm các widget vào layout chính
        layout.addWidget(basic_group)
        layout.addWidget(tab_widget)
        
        # Nút Đồng ý/Hủy
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Kết nối các sự kiện
        self.beam_type_combo.currentIndexChanged.connect(self._on_beam_type_changed)
        self.use_wedge_check.toggled.connect(self._on_use_wedge_toggled)
        self.use_bolus_check.toggled.connect(self._on_use_bolus_toggled)
        self.use_block_check.toggled.connect(self._on_use_block_toggled)
        self.add_metadata_button.clicked.connect(self._on_add_metadata)
        self.remove_metadata_button.clicked.connect(self._on_remove_metadata)
    
    def _load_beam_data(self):
        """Load dữ liệu từ beam_setup vào giao diện."""
        if not self.beam_setup:
            return
        
        # Thông tin cơ bản
        self.name_edit.setText(self.beam_setup.name)
        
        # Tìm index của beam type trong combobox
        beam_type = self.beam_setup.beam.beam_type if hasattr(self.beam_setup.beam, 'beam_type') else BeamType.PHOTON
        for i in range(self.beam_type_combo.count()):
            if self.beam_type_combo.itemData(i) == beam_type:
                self.beam_type_combo.setCurrentIndex(i)
                break
        
        # Năng lượng
        energy = self.beam_setup.beam.energy if hasattr(self.beam_setup.beam, 'energy') else "6MV"
        index = self.energy_combo.findText(energy)
        if index >= 0:
            self.energy_combo.setCurrentIndex(index)
        else:
            self.energy_combo.setCurrentText(energy)
        
        # MU và trọng số
        self.mu_spin.setValue(self.beam_setup.monitor_units)
        self.weight_spin.setValue(self.beam_setup.weight)
        
        # Hình học
        if self.beam_setup.beam_geometry:
            self.gantry_angle_spin.setValue(self.beam_setup.beam_geometry.gantry_angle)
            self.collimator_angle_spin.setValue(self.beam_setup.beam_geometry.collimator_angle)
            self.couch_angle_spin.setValue(self.beam_setup.beam_geometry.couch_angle)
            self.ssd_spin.setValue(self.beam_setup.beam_geometry.ssd)
        
        # Kích thước trường
        width, height = self.beam_setup.field_size
        self.field_width_spin.setValue(width)
        self.field_height_spin.setValue(height)
        
        # Isocenter
        x, y, z = self.beam_setup.isocenter_position
        self.isocenter_x_spin.setValue(x)
        self.isocenter_y_spin.setValue(y)
        self.isocenter_z_spin.setValue(z)
        
        # Bộ điều chỉnh
        if self.beam_setup.wedge:
            self.use_wedge_check.setChecked(True)
            self.wedge_angle_spin.setValue(self.beam_setup.wedge.angle)
            index = self.wedge_orientation_combo.findText(self.beam_setup.wedge.orientation)
            if index >= 0:
                self.wedge_orientation_combo.setCurrentIndex(index)
        
        if self.beam_setup.bolus:
            self.use_bolus_check.setChecked(True)
            self.bolus_thickness_spin.setValue(self.beam_setup.bolus.thickness)
            index = self.bolus_material_combo.findText(self.beam_setup.bolus.material)
            if index >= 0:
                self.bolus_material_combo.setCurrentIndex(index)
        
        if self.beam_setup.blocks:
            self.use_block_check.setChecked(True)
            if self.beam_setup.blocks:
                self.block_thickness_spin.setValue(self.beam_setup.blocks[0].thickness)
        
        # Metadata
        for key, value in self.beam_setup.metadata.items():
            row = self.metadata_table.rowCount()
            self.metadata_table.insertRow(row)
            self.metadata_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.metadata_table.setItem(row, 1, QTableWidgetItem(str(value)))
    
    def _on_beam_type_changed(self, index):
        """Xử lý khi loại chùm tia thay đổi."""
        beam_type = self.beam_type_combo.itemData(index)
        
        # Cập nhật danh sách năng lượng dựa trên loại chùm tia
        self.energy_combo.clear()
        if beam_type == BeamType.PHOTON:
            energies = ["6MV", "10MV", "15MV", "18MV"]
        elif beam_type == BeamType.ELECTRON:
            energies = ["6MeV", "9MeV", "12MeV", "15MeV", "18MeV", "21MeV"]
        elif beam_type == BeamType.PROTON:
            energies = ["70MeV", "100MeV", "150MeV", "200MeV", "250MeV"]
        elif beam_type == BeamType.CARBON:
            energies = ["100MeV/u", "150MeV/u", "200MeV/u", "300MeV/u", "400MeV/u"]
        else:
            energies = []
        
        for energy in energies:
            self.energy_combo.addItem(energy)
    
    def _on_use_wedge_toggled(self, checked):
        """Xử lý khi nút thêm nêm được bật/tắt."""
        self.wedge_angle_spin.setEnabled(checked)
        self.wedge_orientation_combo.setEnabled(checked)
    
    def _on_use_bolus_toggled(self, checked):
        """Xử lý khi nút thêm bolus được bật/tắt."""
        self.bolus_thickness_spin.setEnabled(checked)
        self.bolus_material_combo.setEnabled(checked)
    
    def _on_use_block_toggled(self, checked):
        """Xử lý khi nút thêm block được bật/tắt."""
        self.block_thickness_spin.setEnabled(checked)
    
    def _on_add_metadata(self):
        """Xử lý khi nhấn nút thêm metadata."""
        row = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row)
        self.metadata_table.setItem(row, 0, QTableWidgetItem(""))
        self.metadata_table.setItem(row, 1, QTableWidgetItem(""))
    
    def _on_remove_metadata(self):
        """Xử lý khi nhấn nút xóa metadata."""
        selected_rows = self.metadata_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Xóa từ dưới lên để không ảnh hưởng đến chỉ số
        rows = sorted([index.row() for index in selected_rows], reverse=True)
        for row in rows:
            self.metadata_table.removeRow(row)
    
    def accept(self):
        """Xử lý khi người dùng nhấn nút Đồng ý."""
        try:
            # Kiểm tra thông tin cơ bản
            name = self.name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên chùm tia.")
                return
            
            # Cập nhật beam_setup
            if not self.beam_setup:
                self.beam_setup = BeamSetup()
            
            # Thông tin cơ bản
            self.beam_setup.name = name
            self.beam_setup.monitor_units = self.mu_spin.value()
            self.beam_setup.weight = self.weight_spin.value()
            
            # Thông tin chùm tia
            beam_type = self.beam_type_combo.currentData()
            energy = self.energy_combo.currentText()
            
            # Tạo hoặc cập nhật đối tượng Beam
            if not self.beam_setup.beam:
                self.beam_setup.beam = Beam(beam_name=name)
            
            self.beam_setup.beam.beam_type = beam_type
            self.beam_setup.beam.energy = energy
            
            # Thông tin hình học
            if not self.beam_setup.beam_geometry:
                self.beam_setup.beam_geometry = BeamGeometry()
            
            self.beam_setup.beam_geometry.gantry_angle = self.gantry_angle_spin.value()
            self.beam_setup.beam_geometry.collimator_angle = self.collimator_angle_spin.value()
            self.beam_setup.beam_geometry.couch_angle = self.couch_angle_spin.value()
            self.beam_setup.beam_geometry.ssd = self.ssd_spin.value()
            
            # Kích thước trường
            width = self.field_width_spin.value()
            height = self.field_height_spin.value()
            self.beam_setup.field_size = (width, height)
            
            # Isocenter
            x = self.isocenter_x_spin.value()
            y = self.isocenter_y_spin.value()
            z = self.isocenter_z_spin.value()
            self.beam_setup.isocenter_position = (x, y, z)
            
            # Bộ điều chỉnh (modifiers)
            if self.use_wedge_check.isChecked():
                angle = self.wedge_angle_spin.value()
                orientation = self.wedge_orientation_combo.currentText()
                self.beam_setup.wedge = Wedge(angle=angle, orientation=orientation)
            else:
                self.beam_setup.wedge = None
            
            if self.use_bolus_check.isChecked():
                thickness = self.bolus_thickness_spin.value()
                material = self.bolus_material_combo.currentText()
                self.beam_setup.bolus = Bolus(thickness=thickness, material=material)
            else:
                self.beam_setup.bolus = None
            
            if self.use_block_check.isChecked():
                thickness = self.block_thickness_spin.value()
                # Tạo một block đơn giản, trong thực tế sẽ cần thêm thông tin về hình dạng
                self.beam_setup.blocks = [Block(thickness=thickness)]
            else:
                self.beam_setup.blocks = []
            
            # Metadata
            self.beam_setup.metadata = {}
            for row in range(self.metadata_table.rowCount()):
                key_item = self.metadata_table.item(row, 0)
                value_item = self.metadata_table.item(row, 1)
                
                if key_item and value_item and key_item.text().strip():
                    key = key_item.text().strip()
                    value = value_item.text().strip()
                    self.beam_setup.metadata[key] = value
            
            super().accept()
        except Exception as e:
            logger.error("Lỗi khi lưu thông tin chùm tia: %s", str(e), exc_info=True)
            QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi khi lưu thông tin chùm tia: {str(e)}")
