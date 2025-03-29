#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp giao diện chỉnh sửa MLC (Multi-Leaf Collimator).

Giao diện này cho phép người dùng thiết kế và chỉnh sửa hình dạng MLC 
cho các chùm tia xạ trị, bao gồm cả việc tạo hình dạng từ các cấu trúc và
các hình dạng cơ bản như hình chữ nhật, hình tròn.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, 
    QPushButton, QGroupBox, QRadioButton, QFormLayout, QDoubleSpinBox, 
    QSlider, QTabWidget, QToolBar, QAction, QCheckBox, QMessageBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QSplitter, QStackedWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon

from quangtps.planning.mlc import MLC, MLCLeaf, MLCSequence, MLC_CONFIGURATIONS, create_shape_based_mlc
from quangtps.planning.beam_configurator import BeamConfigurator
from quangtps.imaging.structures import Structure
from quangtps.core.config import Config
from quangtps.common.paths import get_icon_path

logger = logging.getLogger(__name__)

class MLCCanvas(FigureCanvas):
    """Widget hiển thị MLC dựa trên Matplotlib."""
    
    leaf_position_changed = pyqtSignal(int, float)  # (leaf_index, position)
    
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        """
        Khởi tạo canvas MLC.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        width : int, optional
            Chiều rộng của hình (inch)
        height : int, optional
            Chiều cao của hình (inch)
        dpi : int, optional
            Độ phân giải hình (dots per inch)
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_aspect('equal')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.mlc = None
        self.drag_leaf = None
        self.drag_bank = None
        self.field_size = 40.0
        self.show_leaf_numbers = False
        self.selected_leaf = None
        
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
    
    def set_mlc(self, mlc):
        """Thiết lập MLC để hiển thị."""
        self.mlc = mlc
        self.update_display()
        
    def update_display(self):
        """Cập nhật hiển thị của MLC."""
        if self.mlc is None:
            return
            
        self.axes.clear()
        
        # Thiết lập giới hạn trục
        half_size = self.field_size / 2
        self.axes.set_xlim(-half_size, half_size)
        self.axes.set_ylim(-half_size, half_size)
        
        # Tên trục
        self.axes.set_xlabel('X (cm)')
        self.axes.set_ylabel('Y (cm)')
        
        # Vẽ các lá MLC
        for leaf in self.mlc.leaves:
            y_min, y_max, x_pos, bank_factor = leaf.get_physical_coordinates()
            leaf_width = leaf.width
            
            # Xác định màu dựa trên ngân hàng lá
            color = 'lightblue' if bank_factor > 0 else 'lightcoral'
            
            # Đánh dấu lá được chọn
            if self.selected_leaf == leaf.index:
                color = 'yellow' if bank_factor > 0 else 'orange'
            
            # Vẽ hình chữ nhật đại diện cho lá
            if bank_factor > 0:  # Bank A (Left)
                rect = Rectangle(
                    (-half_size, y_min),
                    half_size + x_pos,
                    leaf_width,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7
                )
            else:  # Bank B (Right)
                rect = Rectangle(
                    (x_pos, y_min),
                    half_size - x_pos,
                    leaf_width,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7
                )
            
            self.axes.add_patch(rect)
            
            # Thêm số lá nếu được yêu cầu
            if self.show_leaf_numbers and y_max - y_min >= 0.5:
                if bank_factor > 0:
                    self.axes.text(
                        -half_size + 0.5, 
                        (y_min + y_max) / 2, 
                        str(leaf.index),
                        ha='left', 
                        va='center',
                        fontsize=8
                    )
                else:
                    self.axes.text(
                        half_size - 0.5, 
                        (y_min + y_max) / 2, 
                        str(leaf.index),
                        ha='right', 
                        va='center',
                        fontsize=8
                    )
        
        # Vẽ hệ tọa độ
        self.axes.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        self.axes.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Tính toán và hiển thị phần trăm mở của trường
        if self.mlc.leaves:
            field_area = self.field_size * self.field_size
            transmission_map = self.mlc.get_transmission_map(resolution=100)
            open_area = np.sum(transmission_map > 0.5) / transmission_map.size * field_area
            open_percent = (open_area / field_area) * 100
            
            self.axes.set_title(f"MLC Field - {open_percent:.1f}% Open")
        
        self.fig.tight_layout()
        self.draw()
    
    def on_press(self, event):
        """Xử lý sự kiện nhấn chuột."""
        if event.inaxes != self.axes or self.mlc is None:
            return
            
        # Chuyển đổi tọa độ chuột thành tọa độ MLC
        x, y = event.xdata, event.ydata
        
        # Kiểm tra xem đã nhấn vào lá nào
        for leaf in self.mlc.leaves:
            y_min, y_max, x_pos, bank_factor = leaf.get_physical_coordinates()
            
            if y_min <= y <= y_max:
                if (bank_factor > 0 and abs(x_pos - x) < 0.5) or \
                   (bank_factor < 0 and abs(x_pos - x) < 0.5):
                    self.drag_leaf = leaf.index
                    self.drag_bank = leaf.bank
                    self.selected_leaf = leaf.index
                    self.update_display()
                    break
    
    def on_release(self, event):
        """Xử lý sự kiện thả chuột."""
        self.drag_leaf = None
        self.drag_bank = None
    
    def on_motion(self, event):
        """Xử lý sự kiện di chuyển chuột."""
        if event.inaxes != self.axes or self.drag_leaf is None or self.mlc is None:
            return
            
        # Chuyển đổi tọa độ chuột thành tọa độ MLC
        x = event.xdata
        
        # Giới hạn tọa độ trong phạm vi hợp lệ
        half_size = self.field_size / 2
        x = max(min(x, half_size), -half_size)
        
        # Cập nhật vị trí lá
        leaf = self.mlc.get_leaf(self.drag_leaf)
        if leaf:
            position = x
            if self.mlc.set_leaf_position(self.drag_leaf, position):
                self.leaf_position_changed.emit(self.drag_leaf, position)
                self.update_display()

class MLCEditor(QWidget):
    """
    Widget chỉnh sửa MLC cho kế hoạch xạ trị.
    """
    
    mlc_changed = pyqtSignal(MLC)  # Phát khi MLC thay đổi
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget chỉnh sửa MLC.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        self.mlc = None
        self.current_mlc_type = "HD120"
        self.field_size = 40.0
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QHBoxLayout(self)
        
        # Panel bên trái với các cài đặt
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        
        # Nhóm loại MLC
        mlc_type_group = QGroupBox("Loại MLC")
        mlc_type_layout = QVBoxLayout(mlc_type_group)
        
        self.mlc_type_combo = QComboBox()
        for mlc_type in MLC_CONFIGURATIONS.keys():
            self.mlc_type_combo.addItem(MLC_CONFIGURATIONS[mlc_type]["name"], mlc_type)
        self.mlc_type_combo.currentIndexChanged.connect(self._on_mlc_type_changed)
        
        mlc_type_layout.addWidget(self.mlc_type_combo)
        settings_layout.addWidget(mlc_type_group)
        
        # Nhóm các hình dạng cơ bản
        shapes_group = QGroupBox("Hình dạng")
        shapes_layout = QVBoxLayout(shapes_group)
        
        # Nút tạo hình chữ nhật
        rect_layout = QHBoxLayout()
        rect_layout.addWidget(QLabel("Hình chữ nhật:"))
        self.rect_width_spin = QDoubleSpinBox()
        self.rect_width_spin.setRange(0.5, 40.0)
        self.rect_width_spin.setValue(10.0)
        self.rect_width_spin.setSuffix(" cm")
        rect_layout.addWidget(self.rect_width_spin)
        
        self.rect_height_spin = QDoubleSpinBox()
        self.rect_height_spin.setRange(0.5, 40.0)
        self.rect_height_spin.setValue(10.0)
        self.rect_height_spin.setSuffix(" cm")
        rect_layout.addWidget(self.rect_height_spin)
        
        self.create_rect_button = QPushButton("Tạo")
        self.create_rect_button.clicked.connect(self._create_rectangular_field)
        rect_layout.addWidget(self.create_rect_button)
        
        shapes_layout.addLayout(rect_layout)
        
        # Nút tạo hình tròn
        circle_layout = QHBoxLayout()
        circle_layout.addWidget(QLabel("Hình tròn:"))
        self.circle_radius_spin = QDoubleSpinBox()
        self.circle_radius_spin.setRange(0.5, 20.0)
        self.circle_radius_spin.setValue(5.0)
        self.circle_radius_spin.setSuffix(" cm")
        circle_layout.addWidget(self.circle_radius_spin)
        
        self.create_circle_button = QPushButton("Tạo")
        self.create_circle_button.clicked.connect(self._create_circular_field)
        circle_layout.addWidget(self.create_circle_button)
        
        shapes_layout.addLayout(circle_layout)
        
        # Nút xóa tất cả
        self.clear_button = QPushButton("Mở toàn bộ trường")
        self.clear_button.clicked.connect(self._clear_field)
        shapes_layout.addWidget(self.clear_button)
        
        # Nút đóng tất cả
        self.close_button = QPushButton("Đóng toàn bộ trường")
        self.close_button.clicked.connect(self._close_field)
        shapes_layout.addWidget(self.close_button)
        
        settings_layout.addWidget(shapes_group)
        
        # Bảng vị trí lá
        leaf_group = QGroupBox("Điều chỉnh lá")
        leaf_layout = QVBoxLayout(leaf_group)
        
        self.leaf_table = QTableWidget(0, 3)
        self.leaf_table.setHorizontalHeaderLabels(["Lá", "Bank", "Vị trí (cm)"])
        self.leaf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leaf_table.verticalHeader().setVisible(False)
        self.leaf_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.leaf_table.itemChanged.connect(self._on_leaf_position_edited)
        
        leaf_layout.addWidget(self.leaf_table)
        
        # Checkbox hiển thị số lá
        self.show_numbers_check = QCheckBox("Hiển thị số lá")
        self.show_numbers_check.stateChanged.connect(self._toggle_leaf_numbers)
        leaf_layout.addWidget(self.show_numbers_check)
        
        settings_layout.addWidget(leaf_group)
        
        # Khu vực hiển thị MLC
        self.mlc_canvas = MLCCanvas(self)
        self.mlc_canvas.leaf_position_changed.connect(self._on_canvas_leaf_position_changed)
        
        # Thêm hai panel vào layout chính
        main_layout.addWidget(settings_panel, 1)
        main_layout.addWidget(self.mlc_canvas, 3)
        
        # Khởi tạo MLC mặc định
        self._create_default_mlc()
    
    def _create_default_mlc(self):
        """Tạo MLC mặc định."""
        self.mlc = MLC(self.current_mlc_type)
        self._update_ui()
        self.mlc_changed.emit(self.mlc)
    
    def _update_ui(self):
        """Cập nhật giao diện người dùng với MLC hiện tại."""
        if self.mlc is None:
            return
            
        # Cập nhật canvas
        self.mlc_canvas.set_mlc(self.mlc)
        
        # Cập nhật bảng lá
        self.leaf_table.setRowCount(0)
        self.leaf_table.blockSignals(True)
        
        for leaf in self.mlc.leaves:
            row = self.leaf_table.rowCount()
            self.leaf_table.insertRow(row)
            
            self.leaf_table.setItem(row, 0, QTableWidgetItem(str(leaf.index)))
            self.leaf_table.setItem(row, 1, QTableWidgetItem(leaf.bank))
            
            position_item = QTableWidgetItem(f"{leaf.position:.2f}")
            self.leaf_table.setItem(row, 2, position_item)
        
        self.leaf_table.blockSignals(False)
    
    def _on_mlc_type_changed(self, index):
        """Xử lý khi loại MLC thay đổi."""
        mlc_type = self.mlc_type_combo.currentData()
        if mlc_type != self.current_mlc_type:
            self.current_mlc_type = mlc_type
            self.mlc = MLC(self.current_mlc_type)
            self._update_ui()
            self.mlc_changed.emit(self.mlc)
    
    def _create_rectangular_field(self):
        """Tạo trường hình chữ nhật."""
        if self.mlc is None:
            return
            
        width = self.rect_width_spin.value()
        height = self.rect_height_spin.value()
        
        # Chuyển đổi từ kích thước sang tọa độ
        x1 = -width / 2
        x2 = width / 2
        y1 = -height / 2
        y2 = height / 2
        
        # Thiết lập trường hình chữ nhật
        self.mlc.set_rectangular_field(x1, x2, y1, y2)
        self._update_ui()
        self.mlc_changed.emit(self.mlc)
    
    def _create_circular_field(self):
        """Tạo trường hình tròn."""
        if self.mlc is None:
            return
            
        radius = self.circle_radius_spin.value()
        
        # Thiết lập trường hình tròn
        self.mlc.set_circular_field(0, 0, radius)
        self._update_ui()
        self.mlc_changed.emit(self.mlc)
    
    def _clear_field(self):
        """Mở toàn bộ trường (tất cả các lá đều mở hết mức)."""
        if self.mlc is None:
            return
            
        for leaf in self.mlc.leaves:
            if leaf.bank == "A":
                self.mlc.set_leaf_position(leaf.index, -20.0)
            else:
                self.mlc.set_leaf_position(leaf.index, 20.0)
                
        self._update_ui()
        self.mlc_changed.emit(self.mlc)
    
    def _close_field(self):
        """Đóng toàn bộ trường (tất cả các lá đều ở giữa)."""
        if self.mlc is None:
            return
            
        for leaf in self.mlc.leaves:
            self.mlc.set_leaf_position(leaf.index, 0.0)
                
        self._update_ui()
        self.mlc_changed.emit(self.mlc)
    
    def _on_leaf_position_edited(self, item):
        """Xử lý khi vị trí lá được chỉnh sửa trong bảng."""
        if item.column() != 2 or self.mlc is None:
            return
            
        row = item.row()
        leaf_index = int(self.leaf_table.item(row, 0).text())
        
        try:
            position = float(item.text())
            if self.mlc.set_leaf_position(leaf_index, position):
                self.mlc_canvas.update_display()
                self.mlc_changed.emit(self.mlc)
        except ValueError:
            # Khôi phục giá trị cũ
            leaf = self.mlc.get_leaf(leaf_index)
            if leaf:
                item.setText(f"{leaf.position:.2f}")
    
    def _on_canvas_leaf_position_changed(self, leaf_index, position):
        """Xử lý khi vị trí lá thay đổi từ canvas."""
        # Cập nhật giá trị trong bảng
        for row in range(self.leaf_table.rowCount()):
            if int(self.leaf_table.item(row, 0).text()) == leaf_index:
                self.leaf_table.blockSignals(True)
                self.leaf_table.item(row, 2).setText(f"{position:.2f}")
                self.leaf_table.blockSignals(False)
                break
                
        self.mlc_changed.emit(self.mlc)
    
    def _toggle_leaf_numbers(self, state):
        """Bật/tắt hiển thị số lá."""
        self.mlc_canvas.show_leaf_numbers = (state == Qt.Checked)
        self.mlc_canvas.update_display()
    
    def set_mlc(self, mlc):
        """Thiết lập MLC từ bên ngoài."""
        self.mlc = mlc
        
        # Cập nhật combo loại MLC
        index = self.mlc_type_combo.findData(mlc.mlc_type)
        if index >= 0:
            self.mlc_type_combo.setCurrentIndex(index)
            
        self._update_ui()
    
    def get_mlc(self):
        """Lấy MLC hiện tại."""
        return self.mlc

if __name__ == "__main__":
    # Chạy giao diện để kiểm tra
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    window = MLCEditor()
    window.setWindowTitle("MLC Editor")
    window.resize(1000, 600)
    window.show()
    
    sys.exit(app.exec_()) 