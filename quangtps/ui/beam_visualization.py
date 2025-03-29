#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý hiển thị trực quan chùm tia xạ trị.

Module này cung cấp các lớp và chức năng để hiển thị 
chùm tia xạ trị từ nhiều góc nhìn khác nhau, bao gồm BEV
(Beam's Eye View) và hiển thị 3D.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge
from typing import List, Dict, Tuple, Any, Optional, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QLabel, QComboBox, QCheckBox, QTabWidget, QSplitter, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.planning.mlc import MLC
from quangtps.treatment.beams.beam_modifiers import Wedge, Block
from quangtps.treatment.techniques.crt_visualizer import CRTVisualizer
from quangtps.imaging.structures import Structure
from quangtps.imaging.dicom_series import DicomSeries

logger = logging.getLogger(__name__)

class BeamVisualization(QWidget):
    """
    Widget hiển thị trực quan chùm tia xạ trị.
    
    Widget này cung cấp giao diện người dùng để hiển thị trực quan 
    chùm tia xạ trị từ nhiều góc nhìn khác nhau, bao gồm BEV (Beam's Eye View)
    và hiển thị 3D.
    """
    
    beam_changed = pyqtSignal(Beam)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị trực quan chùm tia.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        self.current_beam_index = 0
        self.beams = []
        self.plan = None
        self.ct_data = None
        self.structures = {}
        
        # Tạo đối tượng visualizer
        self.crt_visualizer = CRTVisualizer()
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # Tạo tabs
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab BEV
        self.bev_widget = QWidget()
        self.tab_widget.addTab(self.bev_widget, "Beam's Eye View")
        
        # Tab 3D
        self.view_3d_widget = QWidget()
        self.tab_widget.addTab(self.view_3d_widget, "3D View")
        
        # Tab Room's Eye View
        self.room_view_widget = QWidget()
        self.tab_widget.addTab(self.room_view_widget, "Room View")
        
        # Khởi tạo tab BEV
        self._init_bev_tab()
        
        # Khởi tạo tab 3D
        self._init_3d_tab()
        
        # Khởi tạo tab Room View
        self._init_room_view_tab()
        
        # Điều khiển chung
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        
        # Điều khiển chùm tia
        beam_control_group = QGroupBox("Điều khiển chùm tia")
        beam_control_layout = QHBoxLayout(beam_control_group)
        
        self.prev_beam_button = QPushButton("Chùm tia trước")
        self.prev_beam_button.clicked.connect(self._on_prev_beam)
        beam_control_layout.addWidget(self.prev_beam_button)
        
        self.beam_selector = QComboBox()
        self.beam_selector.currentIndexChanged.connect(self._on_beam_selected)
        beam_control_layout.addWidget(self.beam_selector)
        
        self.next_beam_button = QPushButton("Chùm tia tiếp")
        self.next_beam_button.clicked.connect(self._on_next_beam)
        beam_control_layout.addWidget(self.next_beam_button)
        
        control_layout.addWidget(beam_control_group)
        
        # Hiển thị lựa chọn
        display_option_group = QGroupBox("Tùy chọn hiển thị")
        display_option_layout = QHBoxLayout(display_option_group)
        
        self.show_mlc_checkbox = QCheckBox("Hiển thị MLC")
        self.show_mlc_checkbox.setChecked(True)
        self.show_mlc_checkbox.stateChanged.connect(self._update_visualization)
        display_option_layout.addWidget(self.show_mlc_checkbox)
        
        self.show_blocks_checkbox = QCheckBox("Hiển thị Blocks")
        self.show_blocks_checkbox.setChecked(True)
        self.show_blocks_checkbox.stateChanged.connect(self._update_visualization)
        display_option_layout.addWidget(self.show_blocks_checkbox)
        
        self.show_wedge_checkbox = QCheckBox("Hiển thị Wedge")
        self.show_wedge_checkbox.setChecked(True)
        self.show_wedge_checkbox.stateChanged.connect(self._update_visualization)
        display_option_layout.addWidget(self.show_wedge_checkbox)
        
        self.show_structures_checkbox = QCheckBox("Hiển thị cấu trúc")
        self.show_structures_checkbox.setChecked(True)
        self.show_structures_checkbox.stateChanged.connect(self._update_visualization)
        display_option_layout.addWidget(self.show_structures_checkbox)
        
        control_layout.addWidget(display_option_group)
        
        # Khởi tạo trạng thái
        self._update_ui_state()
    
    def _init_bev_tab(self):
        """Khởi tạo tab BEV."""
        bev_layout = QVBoxLayout(self.bev_widget)
        
        # Tạo figure
        self.bev_figure = Figure(figsize=(8, 8))
        self.bev_canvas = FigureCanvas(self.bev_figure)
        self.bev_toolbar = NavigationToolbar(self.bev_canvas, self)
        
        bev_layout.addWidget(self.bev_toolbar)
        bev_layout.addWidget(self.bev_canvas)
    
    def _init_3d_tab(self):
        """Khởi tạo tab 3D."""
        view_3d_layout = QVBoxLayout(self.view_3d_widget)
        
        # Tạo figure
        self.view_3d_figure = Figure(figsize=(8, 8))
        self.view_3d_canvas = FigureCanvas(self.view_3d_figure)
        self.view_3d_toolbar = NavigationToolbar(self.view_3d_canvas, self)
        
        view_3d_layout.addWidget(self.view_3d_toolbar)
        view_3d_layout.addWidget(self.view_3d_canvas)
    
    def _init_room_view_tab(self):
        """Khởi tạo tab Room View."""
        room_view_layout = QVBoxLayout(self.room_view_widget)
        
        # Tạo figures cho 3 góc nhìn
        self.room_view_splitter = QSplitter(Qt.Horizontal)
        
        # Góc nhìn từ trên
        top_view_widget = QWidget()
        top_view_layout = QVBoxLayout(top_view_widget)
        self.top_view_figure = Figure(figsize=(5, 5))
        self.top_view_canvas = FigureCanvas(self.top_view_figure)
        top_view_layout.addWidget(QLabel("Góc nhìn từ trên"))
        top_view_layout.addWidget(self.top_view_canvas)
        
        # Góc nhìn từ trước
        front_view_widget = QWidget()
        front_view_layout = QVBoxLayout(front_view_widget)
        self.front_view_figure = Figure(figsize=(5, 5))
        self.front_view_canvas = FigureCanvas(self.front_view_figure)
        front_view_layout.addWidget(QLabel("Góc nhìn từ trước"))
        front_view_layout.addWidget(self.front_view_canvas)
        
        # Góc nhìn từ bên
        side_view_widget = QWidget()
        side_view_layout = QVBoxLayout(side_view_widget)
        self.side_view_figure = Figure(figsize=(5, 5))
        self.side_view_canvas = FigureCanvas(self.side_view_figure)
        side_view_layout.addWidget(QLabel("Góc nhìn từ bên"))
        side_view_layout.addWidget(self.side_view_canvas)
        
        self.room_view_splitter.addWidget(top_view_widget)
        self.room_view_splitter.addWidget(front_view_widget)
        self.room_view_splitter.addWidget(side_view_widget)
        
        room_view_layout.addWidget(self.room_view_splitter)
        
        # Thanh trượt để điều chỉnh góc
        angle_control_group = QGroupBox("Điều khiển góc nhìn")
        angle_control_layout = QVBoxLayout(angle_control_group)
        
        gantry_layout = QHBoxLayout()
        gantry_layout.addWidget(QLabel("Góc Gantry:"))
        self.gantry_slider = QSlider(Qt.Horizontal)
        self.gantry_slider.setRange(0, 360)
        self.gantry_slider.setTickPosition(QSlider.TicksBelow)
        self.gantry_slider.setTickInterval(45)
        self.gantry_slider.valueChanged.connect(self._update_room_view)
        gantry_layout.addWidget(self.gantry_slider)
        self.gantry_value_label = QLabel("0°")
        gantry_layout.addWidget(self.gantry_value_label)
        angle_control_layout.addLayout(gantry_layout)
        
        collimator_layout = QHBoxLayout()
        collimator_layout.addWidget(QLabel("Góc Collimator:"))
        self.collimator_slider = QSlider(Qt.Horizontal)
        self.collimator_slider.setRange(0, 360)
        self.collimator_slider.setTickPosition(QSlider.TicksBelow)
        self.collimator_slider.setTickInterval(45)
        self.collimator_slider.valueChanged.connect(self._update_room_view)
        collimator_layout.addWidget(self.collimator_slider)
        self.collimator_value_label = QLabel("0°")
        collimator_layout.addWidget(self.collimator_value_label)
        angle_control_layout.addLayout(collimator_layout)
        
        couch_layout = QHBoxLayout()
        couch_layout.addWidget(QLabel("Góc Couch:"))
        self.couch_slider = QSlider(Qt.Horizontal)
        self.couch_slider.setRange(0, 360)
        self.couch_slider.setTickPosition(QSlider.TicksBelow)
        self.couch_slider.setTickInterval(45)
        self.couch_slider.valueChanged.connect(self._update_room_view)
        couch_layout.addWidget(self.couch_slider)
        self.couch_value_label = QLabel("0°")
        couch_layout.addWidget(self.couch_value_label)
        angle_control_layout.addLayout(couch_layout)
        
        room_view_layout.addWidget(angle_control_group)
    
    def set_plan(self, plan: Plan, ct_data: Optional[DicomSeries] = None):
        """
        Thiết lập kế hoạch để hiển thị trực quan.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị
        ct_data : DicomSeries, optional
            Dữ liệu CT để hiển thị
        """
        self.plan = plan
        self.ct_data = ct_data
        
        if plan and hasattr(plan, 'beams'):
            self.beams = plan.beams
            self.current_beam_index = 0
            
            # Cập nhật combobox chọn chùm tia
            self.beam_selector.clear()
            for i, beam in enumerate(self.beams):
                beam_name = getattr(beam, 'name', f"Beam {i+1}")
                self.beam_selector.addItem(beam_name)
        else:
            self.beams = []
            self.current_beam_index = 0
            self.beam_selector.clear()
        
        # Cập nhật cấu trúc
        if plan and hasattr(plan, 'structures'):
            self.structures = plan.structures
        
        # Cập nhật trạng thái UI
        self._update_ui_state()
        
        # Cập nhật hiển thị
        self._update_visualization()
    
    def set_structures(self, structures: Dict[str, Structure]):
        """
        Thiết lập cấu trúc để hiển thị trong BEV.
        
        Parameters
        ----------
        structures : Dict[str, Structure]
            Dictionary các cấu trúc
        """
        self.structures = structures
        self._update_visualization()
    
    def _update_ui_state(self):
        """Cập nhật trạng thái UI dựa trên dữ liệu hiện tại."""
        has_beams = len(self.beams) > 0
        
        self.prev_beam_button.setEnabled(has_beams and self.current_beam_index > 0)
        self.next_beam_button.setEnabled(has_beams and self.current_beam_index < len(self.beams) - 1)
        self.beam_selector.setEnabled(has_beams)
        
        if has_beams and self.current_beam_index < len(self.beams):
            current_beam = self.beams[self.current_beam_index]
            
            # Cập nhật sliders với góc hiện tại
            if hasattr(current_beam, 'gantry_angle'):
                self.gantry_slider.setValue(int(current_beam.gantry_angle))
                self.gantry_value_label.setText(f"{current_beam.gantry_angle}°")
            
            if hasattr(current_beam, 'collimator_angle'):
                self.collimator_slider.setValue(int(current_beam.collimator_angle))
                self.collimator_value_label.setText(f"{current_beam.collimator_angle}°")
            
            if hasattr(current_beam, 'couch_angle'):
                self.couch_slider.setValue(int(current_beam.couch_angle))
                self.couch_value_label.setText(f"{current_beam.couch_angle}°")
    
    def _on_prev_beam(self):
        """Xử lý sự kiện khi nhấn nút chùm tia trước."""
        if self.current_beam_index > 0:
            self.current_beam_index -= 1
            self.beam_selector.setCurrentIndex(self.current_beam_index)
            self._update_ui_state()
            self._update_visualization()
    
    def _on_next_beam(self):
        """Xử lý sự kiện khi nhấn nút chùm tia tiếp."""
        if self.current_beam_index < len(self.beams) - 1:
            self.current_beam_index += 1
            self.beam_selector.setCurrentIndex(self.current_beam_index)
            self._update_ui_state()
            self._update_visualization()
    
    def _on_beam_selected(self, index):
        """
        Xử lý sự kiện khi chọn chùm tia từ dropdown.
        
        Parameters
        ----------
        index : int
            Index của chùm tia được chọn
        """
        if 0 <= index < len(self.beams):
            self.current_beam_index = index
            self._update_ui_state()
            self._update_visualization()
            
            # Emit signal
            self.beam_changed.emit(self.beams[self.current_beam_index])
    
    def _update_visualization(self):
        """Cập nhật tất cả các hiển thị trực quan."""
        self._update_bev()
        self._update_3d_view()
        self._update_room_view()
    
    def _update_bev(self):
        """Cập nhật hiển thị BEV."""
        # Xóa figure
        self.bev_figure.clear()
        
        # Kiểm tra xem có chùm tia không
        if not self.beams or self.current_beam_index >= len(self.beams):
            logger.warning("Không có chùm tia để hiển thị")
            self.bev_canvas.draw()
            return
        
        # Lấy chùm tia hiện tại
        current_beam = self.beams[self.current_beam_index]
        
        # Lấy options hiển thị
        show_mlc = self.show_mlc_checkbox.isChecked()
        show_blocks = self.show_blocks_checkbox.isChecked()
        show_wedge = self.show_wedge_checkbox.isChecked()
        
        try:
            # Vẽ BEV sử dụng CRTVisualizer
            self.crt_visualizer.visualize_beam_eye_view(
                current_beam, 
                figure=self.bev_figure,
                show_mlc=show_mlc,
                show_blocks=show_blocks,
                show_wedge=show_wedge
            )
            
            # Hiển thị cấu trúc nếu cần
            if self.show_structures_checkbox.isChecked() and self.structures:
                self._add_structures_to_bev(current_beam)
            
            self.bev_canvas.draw()
        except Exception as e:
            logger.error(f"Lỗi khi hiển thị BEV: {e}")
    
    def _add_structures_to_bev(self, beam: Beam):
        """
        Thêm cấu trúc vào hiển thị BEV.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia hiện tại
        """
        # Tạm thời giản lược để triển khai sau
        # Trong cài đặt thực tế, cần chuyển đổi cấu trúc 3D sang hình chiếu 2D
        # theo hướng chùm tia
        pass
    
    def _update_3d_view(self):
        """Cập nhật hiển thị 3D."""
        if not self.beams or self.current_beam_index >= len(self.beams):
            return
        
        # Lấy chùm tia hiện tại
        beam = self.beams[self.current_beam_index]
        
        # Xóa figure cũ
        self.view_3d_figure.clear()
        
        # Tạo trục 3D
        ax = self.view_3d_figure.add_subplot(111, projection='3d')
        
        # Thiết lập các giới hạn
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-5, 60])
        
        # Thiết lập nhãn
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        
        # Thiết lập tiêu đề
        ax.set_title(f'Hiển thị 3D cho chùm tia: {beam.name}')
        
        # Vẽ mô phỏng bệnh nhân
        self._draw_patient_outline(ax)
        
        # Vẽ máy gia tốc
        self._draw_linac(ax, beam)
        
        # Vẽ chùm tia
        self._draw_beam_3d(ax, beam)
        
        # Cập nhật canvas
        self.view_3d_canvas.draw()
    
    def _draw_patient_outline(self, ax):
        """
        Vẽ đường viền bệnh nhân trong không gian 3D.
        
        Parameters
        ----------
        ax : Axes3D
            Trục 3D để vẽ
        """
        # Tạo một hình dạng cơ bản cho bệnh nhân (hình trụ)
        radius = 15
        height = 40
        z_bottom = -5
        
        # Tạo hình trụ
        theta = np.linspace(0, 2*np.pi, 32)
        z = np.array([z_bottom, z_bottom + height])
        
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        
        # Vẽ thân hình trụ
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='cyan')
        
        # Vẽ nắp dưới
        x_bottom = radius * np.cos(theta)
        y_bottom = radius * np.sin(theta)
        ax.plot_surface(x_bottom, y_bottom, np.ones_like(x_bottom) * z_bottom, alpha=0.2, color='cyan')
        
        # Vẽ nắp trên
        ax.plot_surface(x_bottom, y_bottom, np.ones_like(x_bottom) * (z_bottom + height), alpha=0.2, color='cyan')
    
    def _draw_linac(self, ax, beam):
        """
        Vẽ máy gia tốc trong không gian 3D.
        
        Parameters
        ----------
        ax : Axes3D
            Trục 3D để vẽ
        beam : Beam
            Chùm tia để xác định vị trí máy gia tốc
        """
        # Thiết lập các tham số
        sad = beam.sad if hasattr(beam, 'sad') else 100.0  # cm
        gantry_angle = np.radians(beam.gantry_angle)
        
        # Tính toán vị trí đầu máy gia tốc
        gantry_head_x = -sad * np.sin(gantry_angle)
        gantry_head_y = 0
        gantry_head_z = sad * np.cos(gantry_angle)
        
        # Vẽ gantry (cung)
        theta = np.linspace(0, 2*np.pi, 100)
        gantry_radius = sad + 20  # Bán kính gantry lớn hơn SAD
        
        gantry_x = -gantry_radius * np.sin(theta)
        gantry_y = np.zeros_like(theta)
        gantry_z = gantry_radius * np.cos(theta)
        
        # Chỉ vẽ nửa cung trên
        mask = gantry_z >= 0
        ax.plot(gantry_x[mask], gantry_y[mask], gantry_z[mask], 'k-', alpha=0.3, linewidth=2)
        
        # Vẽ đầu máy gia tốc (hình hộp)
        head_size = 10
        
        # Tính toán các góc và hướng
        collimator_angle = np.radians(beam.collimator_angle)
        
        # Các vector định hướng (chuyển đổi từ hệ tọa độ gantry sang hệ tọa độ phòng)
        direction_vector = np.array([np.sin(gantry_angle), 0, -np.cos(gantry_angle)])
        up_vector = np.array([0, 1, 0])  # Hướng lên
        right_vector = np.cross(direction_vector, up_vector)
        
        # Áp dụng góc collimator
        rotated_up = up_vector * np.cos(collimator_angle) + right_vector * np.sin(collimator_angle)
        rotated_right = right_vector * np.cos(collimator_angle) - up_vector * np.sin(collimator_angle)
        
        # Vẽ đầu máy gia tốc
        head_vertices = []
        
        # Mặt gần isocenter
        center = np.array([gantry_head_x, gantry_head_y, gantry_head_z])
        
        # Các đỉnh của mặt gần
        p1 = center + head_size/2 * rotated_up + head_size/2 * rotated_right
        p2 = center + head_size/2 * rotated_up - head_size/2 * rotated_right
        p3 = center - head_size/2 * rotated_up - head_size/2 * rotated_right
        p4 = center - head_size/2 * rotated_up + head_size/2 * rotated_right
        
        # Các đỉnh của mặt xa (back)
        back_offset = 15
        p5 = p1 + back_offset * direction_vector
        p6 = p2 + back_offset * direction_vector
        p7 = p3 + back_offset * direction_vector
        p8 = p4 + back_offset * direction_vector
        
        # Vẽ các cạnh
        linac_color = 'gray'
        
        # Mặt gần
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=linac_color, linewidth=2)
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], color=linac_color, linewidth=2)
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=linac_color, linewidth=2)
        ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], color=linac_color, linewidth=2)
        
        # Mặt xa
        ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], color=linac_color, linewidth=2)
        ax.plot([p6[0], p7[0]], [p6[1], p7[1]], [p6[2], p7[2]], color=linac_color, linewidth=2)
        ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], color=linac_color, linewidth=2)
        ax.plot([p8[0], p5[0]], [p8[1], p5[1]], [p8[2], p5[2]], color=linac_color, linewidth=2)
        
        # Cạnh nối
        ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], color=linac_color, linewidth=2)
        ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], color=linac_color, linewidth=2)
        ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], color=linac_color, linewidth=2)
        ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color=linac_color, linewidth=2)
        
        # Vẽ đường từ đầu máy gia tốc đến tâm
        ax.plot([0, gantry_head_x], [0, gantry_head_y], [0, gantry_head_z], 'r--', linewidth=1)
    
    def _draw_beam_3d(self, ax, beam):
        """
        Vẽ chùm tia trong không gian 3D.
        
        Parameters
        ----------
        ax : Axes3D
            Trục 3D để vẽ
        beam : Beam
            Chùm tia để vẽ
        """
        # Thiết lập các tham số
        sad = beam.sad if hasattr(beam, 'sad') else 100.0  # cm
        gantry_angle = np.radians(beam.gantry_angle)
        field_size = beam.field_size
        
        # Tính toán vị trí nguồn
        source_x = -sad * np.sin(gantry_angle)
        source_y = 0
        source_z = sad * np.cos(gantry_angle)
        source = np.array([source_x, source_y, source_z])
        
        # Tính toán các vector định hướng
        direction_vector = np.array([np.sin(gantry_angle), 0, -np.cos(gantry_angle)])
        
        # Thiết lập vector lên ban đầu là [0, 1, 0]
        up_vector = np.array([0, 1, 0])
        
        # Tính toán vector phải (vuông góc với direction và up)
        right_vector = np.cross(direction_vector, up_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)
        
        # Điều chỉnh vector lên để đảm bảo vuông góc
        up_vector = np.cross(right_vector, direction_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)
        
        # Áp dụng góc collimator
        if hasattr(beam, 'collimator_angle'):
            collimator_angle = np.radians(beam.collimator_angle)
            
            # Xoay up_vector và right_vector theo góc collimator
            rotated_up = up_vector * np.cos(collimator_angle) + right_vector * np.sin(collimator_angle)
            rotated_right = right_vector * np.cos(collimator_angle) - up_vector * np.sin(collimator_angle)
            
            up_vector = rotated_up
            right_vector = rotated_right
        
        # Tính toán các góc của trường chiếu
        # Nửa chiều rộng và chiều cao trường
        half_width = field_size[0] / 2.0
        half_height = field_size[1] / 2.0
        
        # Các điểm góc tại isocenter
        p1 = np.array([0, 0, 0]) + half_width * right_vector + half_height * up_vector
        p2 = np.array([0, 0, 0]) - half_width * right_vector + half_height * up_vector
        p3 = np.array([0, 0, 0]) - half_width * right_vector - half_height * up_vector
        p4 = np.array([0, 0, 0]) + half_width * right_vector - half_height * up_vector
        
        # Kéo dài các tia từ nguồn qua các điểm góc
        # Khoảng cách kéo dài
        extension = 50.0
        
        e1 = p1 + extension * self._normalize(p1 - source)
        e2 = p2 + extension * self._normalize(p2 - source)
        e3 = p3 + extension * self._normalize(p3 - source)
        e4 = p4 + extension * self._normalize(p4 - source)
        
        # Vẽ các cạnh của trường tại isocenter
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'y-', linewidth=2)
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], 'y-', linewidth=2)
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 'y-', linewidth=2)
        ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], 'y-', linewidth=2)
        
        # Vẽ các cạnh của trường ở extension
        ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], 'y-', linewidth=2)
        ax.plot([e2[0], e3[0]], [e2[1], e3[1]], [e2[2], e3[2]], 'y-', linewidth=2)
        ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], 'y-', linewidth=2)
        ax.plot([e4[0], e1[0]], [e4[1], e1[1]], [e4[2], e1[2]], 'y-', linewidth=2)
        
        # Vẽ các đường nối từ isocenter đến extension
        ax.plot([p1[0], e1[0]], [p1[1], e1[1]], [p1[2], e1[2]], 'y-', linewidth=2)
        ax.plot([p2[0], e2[0]], [p2[1], e2[1]], [p2[2], e2[2]], 'y-', linewidth=2)
        ax.plot([p3[0], e3[0]], [p3[1], e3[1]], [p3[2], e3[2]], 'y-', linewidth=2)
        ax.plot([p4[0], e4[0]], [p4[1], e4[1]], [p4[2], e4[2]], 'y-', linewidth=2)
        
        # Vẽ các tia từ nguồn đến các góc tại isocenter
        ax.plot([source[0], p1[0]], [source[1], p1[1]], [source[2], p1[2]], 'r-', alpha=0.5, linewidth=1)
        ax.plot([source[0], p2[0]], [source[1], p2[1]], [source[2], p2[2]], 'r-', alpha=0.5, linewidth=1)
        ax.plot([source[0], p3[0]], [source[1], p3[1]], [source[2], p3[2]], 'r-', alpha=0.5, linewidth=1)
        ax.plot([source[0], p4[0]], [source[1], p4[1]], [source[2], p4[2]], 'r-', alpha=0.5, linewidth=1)
        
        # Hiển thị trục trung tâm
        ax.plot([source[0], 0], [source[1], 0], [source[2], 0], 'r-', linewidth=2)
        
        # Vẽ isocenter
        ax.scatter([0], [0], [0], color='red', s=50, marker='o')
        
        # Thêm nhãn "Isocenter"
        ax.text(0, 0, 0, 'Isocenter', color='red', fontsize=10)
    
    def _normalize(self, v):
        """
        Chuẩn hóa vector.
        
        Parameters
        ----------
        v : ndarray
            Vector cần chuẩn hóa
            
        Returns
        -------
        ndarray
            Vector đã chuẩn hóa
        """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    
    def _update_room_view(self):
        """Cập nhật hiển thị Room View."""
        # Xóa figures
        self.top_view_figure.clear()
        self.front_view_figure.clear()
        self.side_view_figure.clear()
        
        # Kiểm tra xem có chùm tia không
        if not self.beams or self.current_beam_index >= len(self.beams):
            logger.warning("Không có chùm tia để hiển thị")
            self.top_view_canvas.draw()
            self.front_view_canvas.draw()
            self.side_view_canvas.draw()
            return
        
        # Lấy chùm tia hiện tại
        current_beam = self.beams[self.current_beam_index]
        
        try:
            # Vẽ góc nhìn từ trên (top view)
            ax_top = self.top_view_figure.add_subplot(111)
            self._draw_top_view(ax_top, current_beam)
            self.top_view_canvas.draw()
            
            # Vẽ góc nhìn từ trước (front view)
            ax_front = self.front_view_figure.add_subplot(111)
            self._draw_front_view(ax_front, current_beam)
            self.front_view_canvas.draw()
            
            # Vẽ góc nhìn từ bên (side view)
            ax_side = self.side_view_figure.add_subplot(111)
            self._draw_side_view(ax_side, current_beam)
            self.side_view_canvas.draw()
        except Exception as e:
            logger.error(f"Lỗi khi hiển thị Room View: {e}")
    
    def _draw_top_view(self, ax, beam: Beam):
        """
        Vẽ góc nhìn từ trên của chùm tia.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Trục để vẽ
        beam : Beam
            Chùm tia cần vẽ
        """
        # Tạm thời giản lược để triển khai sau
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Z (cm)')
        ax.set_title('Top View')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Vẽ bàn điều trị
        ax.add_patch(Rectangle((-40, -80), 80, 160, fill=True, color='lightgray', alpha=0.3))
        
        # Vẽ isocenter
        ax.plot(0, 0, 'o', color='blue', markersize=10)
        
        # Vẽ hướng chùm tia
        if hasattr(beam, 'gantry_angle'):
            angle_rad = np.radians(beam.gantry_angle)
            dx = 50 * np.sin(angle_rad)
            dy = -50 * np.cos(angle_rad)
            ax.arrow(0, 0, dx, dy, head_width=5, head_length=5, fc='yellow', ec='yellow', linewidth=2)
    
    def _draw_front_view(self, ax, beam: Beam):
        """
        Vẽ góc nhìn từ trước của chùm tia.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Trục để vẽ
        beam : Beam
            Chùm tia cần vẽ
        """
        # Tạm thời giản lược để triển khai sau
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title('Front View')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Vẽ bàn điều trị
        ax.add_patch(Rectangle((-40, -15), 80, 10, fill=True, color='lightgray', alpha=0.3))
        
        # Vẽ isocenter
        ax.plot(0, 0, 'o', color='blue', markersize=10)
    
    def _draw_side_view(self, ax, beam: Beam):
        """
        Vẽ góc nhìn từ bên của chùm tia.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Trục để vẽ
        beam : Beam
            Chùm tia cần vẽ
        """
        # Tạm thời giản lược để triển khai sau
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel('Z (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title('Side View')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Vẽ bàn điều trị
        ax.add_patch(Rectangle((-80, -15), 160, 10, fill=True, color='lightgray', alpha=0.3))
        
        # Vẽ isocenter
        ax.plot(0, 0, 'o', color='blue', markersize=10)

if __name__ == "__main__":
    # Chạy giao diện để kiểm tra
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Tạo chùm tia mẫu
    beam = Beam()
    beam.name = "Beam 1"
    beam.technique = "3D-CRT"
    beam.energy = 6
    beam.field_size = (10.0, 10.0)
    
    # Tạo MLC
    mlc = MLC()
    beam.mlc = mlc
    
    # Tạo Wedge
    wedge = Wedge("Enhanced Dynamic Wedge", 15, "IN")
    beam.add_modifier(wedge)
    
    # Tạo Block
    block = Block("Custom Block")
    points = [(-3, -3), (3, -3), (3, 3), (-3, 3), (-3, -3)]
    block.set_contour(points)
    beam.add_modifier(block)
    
    # Tạo cấu trúc mẫu
    class StructureSample:
        def __init__(self, name, center=None):
            self.name = name
            self.center = center or [0, 0, 0]
    
    structures = [
        StructureSample("PTV", [1, 1, 0]),
        StructureSample("GTV", [0, 0, 0]),
        StructureSample("Spinal Cord", [-2, 0, 0]),
        StructureSample("Left Lung", [-3, 2, 0])
    ]
    
    # Tạo giao diện
    window = BeamVisualization()
    window.setWindowTitle("Beam Visualization")
    window.resize(1000, 800)
    window.set_plan(beam)
    window.show()
    
    sys.exit(app.exec_()) 