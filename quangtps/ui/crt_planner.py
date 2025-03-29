#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Giao diện người dùng để tạo và quản lý kế hoạch xạ trị 3D CRT.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox, 
    QTabWidget, QSplitter, QFrame, QMessageBox, QListWidget,
    QListWidgetItem, QCheckBox, QSpinBox, QLineEdit, QDialog,
    QDialogButtonBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon

from quangtps.treatment.techniques.crt_manager import CRTManager
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.ui.beam_visualization import BeamVisualization
from quangtps.imaging.structures import Structure
from quangtps.ui.mlc_editor import MLCEditor
from quangtps.common.paths import get_icon_path

logger = logging.getLogger(__name__)

class BeamConfigDialog(QDialog):
    """Dialog để cấu hình chùm tia."""
    
    def __init__(self, beam=None, parent=None):
        """
        Khởi tạo dialog cấu hình chùm tia.
        
        Parameters
        ----------
        beam : Beam, optional
            Chùm tia cần cấu hình, hoặc None để tạo chùm tia mới
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        self.beam = beam or Beam()
        self._init_ui()
        
        # Điền thông tin chùm tia nếu có
        if beam:
            self._fill_beam_info()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        self.setWindowTitle("Cấu hình chùm tia")
        self.resize(400, 350)
        
        layout = QVBoxLayout(self)
        
        # Form layout cho các thông số
        form_layout = QFormLayout()
        
        # Tên chùm tia
        self.name_edit = QLineEdit()
        form_layout.addRow("Tên:", self.name_edit)
        
        # Góc gantry
        self.gantry_angle_spin = QDoubleSpinBox()
        self.gantry_angle_spin.setRange(0, 359.9)
        self.gantry_angle_spin.setDecimals(1)
        self.gantry_angle_spin.setSingleStep(10)
        form_layout.addRow("Góc gantry (độ):", self.gantry_angle_spin)
        
        # Góc collimator
        self.collimator_angle_spin = QDoubleSpinBox()
        self.collimator_angle_spin.setRange(0, 359.9)
        self.collimator_angle_spin.setDecimals(1)
        self.collimator_angle_spin.setSingleStep(10)
        form_layout.addRow("Góc collimator (độ):", self.collimator_angle_spin)
        
        # Góc table
        self.table_angle_spin = QDoubleSpinBox()
        self.table_angle_spin.setRange(0, 359.9)
        self.table_angle_spin.setDecimals(1)
        self.table_angle_spin.setSingleStep(10)
        form_layout.addRow("Góc table (độ):", self.table_angle_spin)
        
        # Kích thước trường X
        self.field_size_x_spin = QDoubleSpinBox()
        self.field_size_x_spin.setRange(1, 40)
        self.field_size_x_spin.setDecimals(1)
        self.field_size_x_spin.setSingleStep(1)
        self.field_size_x_spin.setSuffix(" cm")
        form_layout.addRow("Kích thước trường X:", self.field_size_x_spin)
        
        # Kích thước trường Y
        self.field_size_y_spin = QDoubleSpinBox()
        self.field_size_y_spin.setRange(1, 40)
        self.field_size_y_spin.setDecimals(1)
        self.field_size_y_spin.setSingleStep(1)
        self.field_size_y_spin.setSuffix(" cm")
        form_layout.addRow("Kích thước trường Y:", self.field_size_y_spin)
        
        # Năng lượng
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["6 MV", "10 MV", "15 MV", "18 MV", "6 MeV", "9 MeV", "12 MeV", "15 MeV"])
        form_layout.addRow("Năng lượng:", self.energy_combo)
        
        # SSD hoặc SAD
        self.ssd_spin = QDoubleSpinBox()
        self.ssd_spin.setRange(80, 120)
        self.ssd_spin.setDecimals(1)
        self.ssd_spin.setSingleStep(1)
        self.ssd_spin.setSuffix(" cm")
        self.ssd_spin.setValue(100.0)
        form_layout.addRow("SSD:", self.ssd_spin)
        
        # Trọng số
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 1)
        self.weight_spin.setDecimals(2)
        self.weight_spin.setSingleStep(0.1)
        self.weight_spin.setValue(1.0)
        form_layout.addRow("Trọng số:", self.weight_spin)
        
        # Wedge
        self.wedge_group = QGroupBox("Wedge")
        self.wedge_group.setCheckable(True)
        self.wedge_group.setChecked(False)
        
        wedge_layout = QFormLayout(self.wedge_group)
        
        self.wedge_angle_spin = QDoubleSpinBox()
        self.wedge_angle_spin.setRange(0, 60)
        self.wedge_angle_spin.setDecimals(0)
        self.wedge_angle_spin.setSingleStep(15)
        self.wedge_angle_spin.setValue(30)
        wedge_layout.addRow("Góc wedge (độ):", self.wedge_angle_spin)
        
        self.wedge_orientation_combo = QComboBox()
        self.wedge_orientation_combo.addItems(["IN", "OUT", "LEFT", "RIGHT"])
        wedge_layout.addRow("Hướng:", self.wedge_orientation_combo)
        
        # Thêm các layout vào layout chính
        layout.addLayout(form_layout)
        layout.addWidget(self.wedge_group)
        
        # Nút
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _fill_beam_info(self):
        """Điền thông tin chùm tia vào các controls."""
        # Thông tin cơ bản
        if hasattr(self.beam, 'name') and self.beam.name:
            self.name_edit.setText(self.beam.name)
        
        if hasattr(self.beam, 'gantry_angle'):
            self.gantry_angle_spin.setValue(self.beam.gantry_angle)
        
        if hasattr(self.beam, 'collimator_angle'):
            self.collimator_angle_spin.setValue(self.beam.collimator_angle)
        
        if hasattr(self.beam, 'table_angle'):
            self.table_angle_spin.setValue(self.beam.table_angle)
        
        # Kích thước trường
        if hasattr(self.beam, 'field_size') and self.beam.field_size:
            self.field_size_x_spin.setValue(self.beam.field_size[0])
            self.field_size_y_spin.setValue(self.beam.field_size[1])
        
        # Năng lượng
        if hasattr(self.beam, 'energy') and self.beam.energy:
            energy_text = f"{self.beam.energy}"
            if not energy_text.endswith("MV") and not energy_text.endswith("MeV"):
                if float(self.beam.energy) < 20:  # Heuristic để phân biệt photon và electron
                    energy_text += " MV"
                else:
                    energy_text += " MeV"
            
            index = self.energy_combo.findText(energy_text, Qt.MatchContains)
            if index >= 0:
                self.energy_combo.setCurrentIndex(index)
        
        # SSD/SAD
        if hasattr(self.beam, 'ssd') and self.beam.ssd:
            self.ssd_spin.setValue(self.beam.ssd)
        
        # Trọng số
        if hasattr(self.beam, 'weight') and self.beam.weight:
            self.weight_spin.setValue(self.beam.weight)
        
        # Wedge
        if hasattr(self.beam, 'modifiers') and self.beam.modifiers:
            for modifier in self.beam.modifiers:
                if hasattr(modifier, '__class__') and modifier.__class__.__name__ == 'Wedge':
                    self.wedge_group.setChecked(True)
                    
                    if hasattr(modifier, 'angle'):
                        self.wedge_angle_spin.setValue(modifier.angle)
                    
                    if hasattr(modifier, 'orientation'):
                        index = self.wedge_orientation_combo.findText(modifier.orientation, Qt.MatchContains)
                        if index >= 0:
                            self.wedge_orientation_combo.setCurrentIndex(index)
    
    def get_beam(self) -> Beam:
        """
        Lấy chùm tia với thông tin đã cấu hình.
        
        Returns
        -------
        Beam
            Chùm tia đã cấu hình
        """
        # Cập nhật thông tin cơ bản
        self.beam.name = self.name_edit.text()
        self.beam.gantry_angle = self.gantry_angle_spin.value()
        self.beam.collimator_angle = self.collimator_angle_spin.value()
        self.beam.table_angle = self.table_angle_spin.value()
        
        # Kích thước trường
        self.beam.field_size = (
            self.field_size_x_spin.value(),
            self.field_size_y_spin.value()
        )
        
        # Năng lượng
        energy_text = self.energy_combo.currentText()
        if "MV" in energy_text:
            self.beam.energy = float(energy_text.split()[0])
        elif "MeV" in energy_text:
            self.beam.energy = float(energy_text.split()[0])
        
        # SSD/SAD
        self.beam.ssd = self.ssd_spin.value()
        
        # Trọng số
        self.beam.weight = self.weight_spin.value()
        
        # Xóa các wedge cũ
        if hasattr(self.beam, 'modifiers'):
            self.beam.modifiers = [m for m in self.beam.modifiers if not hasattr(m, '__class__') or m.__class__.__name__ != 'Wedge']
        
        # Thêm wedge nếu được chọn
        if self.wedge_group.isChecked():
            from quangtps.treatment.beams.beam_modifiers import Wedge
            wedge = Wedge(
                "Enhanced Dynamic Wedge",
                self.wedge_angle_spin.value(),
                self.wedge_orientation_combo.currentText()
            )
            self.beam.add_modifier(wedge)
        
        return self.beam

class CRTPlanner(QWidget):
    """
    Giao diện lập kế hoạch 3D CRT.
    
    Widget này cung cấp giao diện người dùng để tạo và quản lý 
    kế hoạch xạ trị 3D Conformal (3D CRT).
    """
    
    plan_created = pyqtSignal(Plan)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget lập kế hoạch 3D CRT.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        self.crt_manager = CRTManager()
        self.plan = None
        self.structures = []
        self.current_beam = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QHBoxLayout(self)
        
        # Phần trái: Danh sách chùm tia và các công cụ
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Nhóm kế hoạch
        plan_group = QGroupBox("Kế hoạch")
        plan_layout = QVBoxLayout(plan_group)
        
        # Form layout cho thông tin kế hoạch
        plan_form = QFormLayout()
        
        self.plan_name_edit = QLineEdit("Kế hoạch 3D CRT")
        plan_form.addRow("Tên kế hoạch:", self.plan_name_edit)
        
        self.site_combo = QComboBox()
        self.site_combo.addItems(["Sọ (Skull)", "Ngực (Chest)", "Bụng (Abdomen)", "Chậu (Pelvis)"])
        plan_form.addRow("Vị trí giải phẫu:", self.site_combo)
        
        self.prescription_spin = QDoubleSpinBox()
        self.prescription_spin.setRange(0, 100)
        self.prescription_spin.setDecimals(1)
        self.prescription_spin.setValue(60.0)
        self.prescription_spin.setSuffix(" Gy")
        plan_form.addRow("Liều kê toa:", self.prescription_spin)
        
        self.fractions_spin = QSpinBox()
        self.fractions_spin.setRange(1, 40)
        self.fractions_spin.setValue(30)
        plan_form.addRow("Số phân liều:", self.fractions_spin)
        
        plan_layout.addLayout(plan_form)
        
        # Nút tạo kế hoạch
        self.create_plan_button = QPushButton("Tạo kế hoạch")
        self.create_plan_button.clicked.connect(self._on_create_plan)
        plan_layout.addWidget(self.create_plan_button)
        
        left_layout.addWidget(plan_group)
        
        # Nhóm chùm tia
        beams_group = QGroupBox("Chùm tia")
        beams_layout = QVBoxLayout(beams_group)
        
        # Mẫu chùm tia
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Mẫu:"))
        
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Chọn mẫu...",
            "AP", "PA", 
            "Left Lateral", "Right Lateral",
            "AP/PA", "4-Field Box",
            "3-Field Skull"
        ])
        template_layout.addWidget(self.template_combo)
        
        self.add_template_button = QPushButton("Thêm")
        self.add_template_button.clicked.connect(self._on_add_template)
        template_layout.addWidget(self.add_template_button)
        
        beams_layout.addLayout(template_layout)
        
        # Danh sách chùm tia
        self.beams_list = QListWidget()
        self.beams_list.currentItemChanged.connect(self._on_beam_selected)
        beams_layout.addWidget(self.beams_list)
        
        # Các nút tương tác với chùm tia
        beam_buttons_layout = QHBoxLayout()
        
        self.add_beam_button = QPushButton("Thêm")
        self.add_beam_button.clicked.connect(self._on_add_beam)
        beam_buttons_layout.addWidget(self.add_beam_button)
        
        self.edit_beam_button = QPushButton("Sửa")
        self.edit_beam_button.clicked.connect(self._on_edit_beam)
        beam_buttons_layout.addWidget(self.edit_beam_button)
        
        self.remove_beam_button = QPushButton("Xóa")
        self.remove_beam_button.clicked.connect(self._on_remove_beam)
        beam_buttons_layout.addWidget(self.remove_beam_button)
        
        beams_layout.addLayout(beam_buttons_layout)
        
        left_layout.addWidget(beams_group)
        
        # Nhóm tính toán
        calc_group = QGroupBox("Tính toán")
        calc_layout = QVBoxLayout(calc_group)
        
        self.calculate_button = QPushButton("Tính toán liều")
        self.calculate_button.clicked.connect(self._on_calculate_dose)
        calc_layout.addWidget(self.calculate_button)
        
        self.normalize_button = QPushButton("Chuẩn hóa kế hoạch")
        self.normalize_button.clicked.connect(self._on_normalize_plan)
        calc_layout.addWidget(self.normalize_button)
        
        self.apply_button = QPushButton("Áp dụng kế hoạch")
        self.apply_button.clicked.connect(self._on_apply_plan)
        calc_layout.addWidget(self.apply_button)
        
        left_layout.addWidget(calc_group)
        
        # Phần phải: Hiển thị chùm tia
        self.beam_visualization = BeamVisualization()
        
        # Thêm các panel vào layout chính
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.beam_visualization, 3)
        
        # Cập nhật trạng thái UI
        self._update_ui_state()
    
    def set_structures(self, structures: List[Structure]):
        """
        Thiết lập danh sách cấu trúc để hiển thị.
        
        Parameters
        ----------
        structures : List[Structure]
            Danh sách các cấu trúc
        """
        self.structures = structures
        self.beam_visualization.set_structures(structures)
    
    def _update_ui_state(self):
        """Cập nhật trạng thái UI dựa trên kế hoạch hiện tại."""
        has_plan = self.plan is not None
        has_beams = has_plan and len(self.plan.beams) > 0
        has_selected_beam = self.current_beam is not None
        
        # Cập nhật trạng thái của các nút
        self.edit_beam_button.setEnabled(has_selected_beam)
        self.remove_beam_button.setEnabled(has_selected_beam)
        self.calculate_button.setEnabled(has_beams)
        self.normalize_button.setEnabled(has_beams)
        self.apply_button.setEnabled(has_beams)
    
    def _on_create_plan(self):
        """Xử lý khi nút tạo kế hoạch được nhấn."""
        plan_name = self.plan_name_edit.text()
        
        if not plan_name:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên kế hoạch")
            return
        
        self.plan = Plan()
        self.plan.name = plan_name
        self.plan.technique = "3D-CRT"
        
        # Thiết lập liều kê toa
        self.plan.prescription_dose = self.prescription_spin.value()
        self.plan.num_fractions = self.fractions_spin.value()
        
        # Cập nhật danh sách chùm tia (xóa sạch)
        self.beams_list.clear()
        self.current_beam = None
        
        # Cập nhật trạng thái UI
        self._update_ui_state()
        
        # Hiển thị thông báo
        QMessageBox.information(self, "Thông báo", f"Đã tạo kế hoạch {plan_name}")
    
    def _on_add_template(self):
        """Xử lý khi nút thêm mẫu chùm tia được nhấn."""
        template_name = self.template_combo.currentText()
        
        if template_name == "Chọn mẫu...":
            return
        
        if not self.plan:
            QMessageBox.warning(self, "Lỗi", "Vui lòng tạo kế hoạch trước")
            return
        
        beams = []
        
        if template_name == "AP":
            beams = self.crt_manager.create_beams_from_template("skull_ap")
        elif template_name == "PA":
            beams = self.crt_manager.create_beams_from_template("skull_pa")
        elif template_name == "Left Lateral":
            beams = self.crt_manager.create_beams_from_template("skull_lateral_left")
        elif template_name == "Right Lateral":
            beams = self.crt_manager.create_beams_from_template("skull_lateral_right")
        elif template_name == "AP/PA":
            beams = self.crt_manager.create_beams_from_template("skull_ap")
            beams += self.crt_manager.create_beams_from_template("skull_pa")
        elif template_name == "4-Field Box":
            beams = self.crt_manager.create_beams_from_template("box_technique")
        elif template_name == "3-Field Skull":
            beams = self.crt_manager.create_beams_from_template("skull_3field")
        
        for beam in beams:
            self.plan.add_beam(beam)
            self._add_beam_to_list(beam)
        
        # Cập nhật trạng thái UI
        self._update_ui_state()
    
    def _on_add_beam(self):
        """Xử lý khi nút thêm chùm tia được nhấn."""
        if not self.plan:
            QMessageBox.warning(self, "Lỗi", "Vui lòng tạo kế hoạch trước")
            return
        
        dialog = BeamConfigDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            beam = dialog.get_beam()
            self.plan.add_beam(beam)
            self._add_beam_to_list(beam)
            
            # Cập nhật trạng thái UI
            self._update_ui_state()
    
    def _on_edit_beam(self):
        """Xử lý khi nút sửa chùm tia được nhấn."""
        if not self.current_beam:
            return
        
        dialog = BeamConfigDialog(self.current_beam, self)
        if dialog.exec_() == QDialog.Accepted:
            beam = dialog.get_beam()
            
            # Cập nhật thông tin trong danh sách
            current_item = self.beams_list.currentItem()
            if current_item:
                current_item.setText(beam.name)
                current_item.setData(Qt.UserRole, beam)
            
            # Cập nhật hiển thị
            self.beam_visualization.set_beam(beam)
    
    def _on_remove_beam(self):
        """Xử lý khi nút xóa chùm tia được nhấn."""
        if not self.current_beam or not self.plan:
            return
        
        # Xóa chùm tia khỏi kế hoạch
        if self.current_beam in self.plan.beams:
            self.plan.beams.remove(self.current_beam)
        
        # Xóa khỏi danh sách
        current_row = self.beams_list.currentRow()
        self.beams_list.takeItem(current_row)
        
        # Cập nhật chùm tia hiện tại
        self.current_beam = None
        
        # Cập nhật trạng thái UI
        self._update_ui_state()
    
    def _on_beam_selected(self, current, previous):
        """Xử lý khi một chùm tia được chọn trong danh sách."""
        if current:
            self.current_beam = current.data(Qt.UserRole)
            self.beam_visualization.set_beam(self.current_beam)
        else:
            self.current_beam = None
            self.beam_visualization.set_beam(None)
        
        # Cập nhật trạng thái UI
        self._update_ui_state()
    
    def _on_calculate_dose(self):
        """Xử lý khi nút tính toán liều được nhấn."""
        if not self.plan or not self.plan.beams:
            return
        
        try:
            # TODO: Gọi phương thức tính toán liều
            success = self.crt_manager.calculate_dose(self.plan)
            
            if success:
                QMessageBox.information(self, "Thông báo", "Đã tính toán liều lượng")
            else:
                QMessageBox.warning(self, "Cảnh báo", "Không thể tính toán liều lượng")
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tính toán liều: {e}")
    
    def _on_normalize_plan(self):
        """Xử lý khi nút chuẩn hóa kế hoạch được nhấn."""
        if not self.plan or not self.plan.beams:
            return
        
        try:
            # Chuẩn hóa trọng số của các chùm tia
            total_weight = sum(beam.weight for beam in self.plan.beams)
            
            if total_weight > 0:
                for beam in self.plan.beams:
                    beam.weight = beam.weight / total_weight
            else:
                # Trường hợp tổng trọng số = 0, phân bố đều
                for beam in self.plan.beams:
                    beam.weight = 1.0 / len(self.plan.beams)
            
            QMessageBox.information(self, "Thông báo", "Đã chuẩn hóa kế hoạch")
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn hóa kế hoạch: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi chuẩn hóa kế hoạch: {e}")
    
    def _on_apply_plan(self):
        """Xử lý khi nút áp dụng kế hoạch được nhấn."""
        if not self.plan or not self.plan.beams:
            return
        
        try:
            # Phát tín hiệu để thông báo kế hoạch đã được tạo
            self.plan_created.emit(self.plan)
            
            QMessageBox.information(
                self, 
                "Thông báo", 
                f"Đã áp dụng kế hoạch {self.plan.name} với {len(self.plan.beams)} chùm tia"
            )
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng kế hoạch: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi áp dụng kế hoạch: {e}")
    
    def _add_beam_to_list(self, beam):
        """
        Thêm chùm tia vào danh sách.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia cần thêm
        """
        item = QListWidgetItem(beam.name)
        item.setData(Qt.UserRole, beam)
        self.beams_list.addItem(item)

if __name__ == "__main__":
    # Chạy giao diện để kiểm tra
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    window = CRTPlanner()
    window.setWindowTitle("3D CRT Planner")
    window.resize(1200, 800)
    window.show()
    
    sys.exit(app.exec_()) 