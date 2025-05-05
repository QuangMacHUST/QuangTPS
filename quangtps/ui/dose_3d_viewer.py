#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp khả năng hiển thị phân phối liều 3D trong QuangTPS
tương tự như Eclipse của Varian.
"""

import os
import sys
import logging
import numpy as np
import vtk
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QSplitter,
    QToolBar,
    QAction,
    QColorDialog,
    QMenu,
    QToolButton,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap, QColor

try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    logging.error(
        "Không thể import QVTKRenderWindowInteractor. Hãy đảm bảo đã cài đặt VTK với hỗ trợ Qt."
    )
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Import nội bộ
try:
    from quangtps.ui.vtk_viewer_3d import VTKViewer3D
    from quangtps.ui.dose_visualization_3d import DoseVisualization3D
    from quangtps.ui.utils.color_map import ColorMap, get_eclipse_colormap
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure
    from quangtps.ui.utils.ui_helpers import create_button, create_slider, create_label
    from quangtps.ui.widgets.isodose_selector import IsodoseSelector
    from quangtps.ui.widgets.structure_visibility_panel import StructureVisibilityPanel
except ImportError as e:
    logging.error(f"Không thể import module nội bộ: {e}")

    # Fallback để ít nhất module có thể được import mà không gây lỗi
    class VTKViewer3D:
        pass

    class DoseVisualization3D:
        pass

    ColorMap = lambda: None
    get_eclipse_colormap = lambda: {}
    DoseGrid = lambda: None
    StructureSet = lambda: None
    Structure = lambda: None
    create_button = lambda *args, **kwargs: QPushButton()
    create_slider = lambda *args, **kwargs: QSlider()
    create_label = lambda *args, **kwargs: QLabel()

    class IsodoseSelector(QWidget):
        pass

    class StructureVisibilityPanel(QWidget):
        pass


class Dose3DViewer(QWidget):
    """
    Widget hiển thị phân phối liều 3D với giao diện tương tự Eclipse.
    Cung cấp khả năng tương tác và tùy chỉnh hiển thị.
    """

    # Tín hiệu
    dose_level_changed = pyqtSignal(float)
    dose_display_mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        """Khởi tạo widget hiển thị liều 3D."""
        super(Dose3DViewer, self).__init__(parent)

        # Dữ liệu
        self._dose_grid = None
        self._structure_set = None
        self._prescription = None
        self._active_structures = {}
        self._isodose_levels = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        self._isodose_colors = get_eclipse_colormap()
        self._display_mode = "Surface"  # "Surface", "Volume" hoặc "Contour"
        self._dose_opacity = 0.8
        self._structure_opacity = 0.5
        self._colorwash_enabled = True
        self._structures_enabled = True
        self._show_isodose_labels = True

        # Thiết lập giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QToolBar("Dose 3D Controls")
        toolbar.setIconSize(QSize(16, 16))

        # Các nút công cụ
        # Nút chế độ hiển thị
        display_mode_button = QToolButton()
        display_mode_button.setText("Mode")
        display_mode_button.setPopupMode(QToolButton.InstantPopup)
        display_mode_menu = QMenu(display_mode_button)

        surface_action = QAction("Surface", self)
        surface_action.triggered.connect(lambda: self.set_display_mode("Surface"))
        display_mode_menu.addAction(surface_action)

        volume_action = QAction("Volume", self)
        volume_action.triggered.connect(lambda: self.set_display_mode("Volume"))
        display_mode_menu.addAction(volume_action)

        contour_action = QAction("Contour", self)
        contour_action.triggered.connect(lambda: self.set_display_mode("Contour"))
        display_mode_menu.addAction(contour_action)

        display_mode_button.setMenu(display_mode_menu)
        toolbar.addWidget(display_mode_button)

        # Toggle dose
        toggle_dose_action = QAction("Toggle Dose", self)
        toggle_dose_action.setCheckable(True)
        toggle_dose_action.setChecked(True)
        toggle_dose_action.triggered.connect(self.toggle_dose)
        toolbar.addAction(toggle_dose_action)

        # Toggle structures
        toggle_structures_action = QAction("Toggle Structures", self)
        toggle_structures_action.setCheckable(True)
        toggle_structures_action.setChecked(True)
        toggle_structures_action.triggered.connect(self.toggle_structures)
        toolbar.addAction(toggle_structures_action)

        toolbar.addSeparator()

        # Export screenshot
        export_action = QAction("Export Image", self)
        export_action.triggered.connect(self.export_screenshot)
        toolbar.addAction(export_action)

        # Reset view
        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_view_action)

        main_layout.addWidget(toolbar)

        # Phần chính: splitter giữa 3D view và control panel
        splitter = QSplitter(Qt.Horizontal)

        # VTK Viewer
        self.viewer_3d = VTKViewer3D(parent=self)
        self.dose_vis = DoseVisualization3D(self.viewer_3d)
        splitter.addWidget(self.viewer_3d)

        # Panel điều khiển
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # Isodose selector
        isodose_group = QGroupBox("Isodose Levels")
        isodose_layout = QVBoxLayout(isodose_group)

        self.isodose_selector = IsodoseSelector(
            self._isodose_levels, self._isodose_colors
        )
        self.isodose_selector.isodose_levels_changed.connect(self.update_isodose_levels)
        isodose_layout.addWidget(self.isodose_selector)

        # Isodose opacity
        dose_opacity_layout = QHBoxLayout()
        dose_opacity_layout.addWidget(QLabel("Dose Opacity:"))
        self.dose_opacity_slider = QSlider(Qt.Horizontal)
        self.dose_opacity_slider.setRange(0, 100)
        self.dose_opacity_slider.setValue(int(self._dose_opacity * 100))
        self.dose_opacity_slider.valueChanged.connect(self.set_dose_opacity)
        dose_opacity_layout.addWidget(self.dose_opacity_slider)
        isodose_layout.addLayout(dose_opacity_layout)

        # Show isodose labels
        self.show_labels_checkbox = QCheckBox("Show Isodose Labels")
        self.show_labels_checkbox.setChecked(self._show_isodose_labels)
        self.show_labels_checkbox.toggled.connect(self.toggle_isodose_labels)
        isodose_layout.addWidget(self.show_labels_checkbox)

        control_layout.addWidget(isodose_group)

        # Structure visibility panel
        structure_group = QGroupBox("Structures")
        structure_layout = QVBoxLayout(structure_group)

        self.structure_panel = StructureVisibilityPanel()
        self.structure_panel.visibility_changed.connect(
            self.update_structure_visibility
        )
        structure_layout.addWidget(self.structure_panel)

        # Structure opacity
        struct_opacity_layout = QHBoxLayout()
        struct_opacity_layout.addWidget(QLabel("Structure Opacity:"))
        self.struct_opacity_slider = QSlider(Qt.Horizontal)
        self.struct_opacity_slider.setRange(0, 100)
        self.struct_opacity_slider.setValue(int(self._structure_opacity * 100))
        self.struct_opacity_slider.valueChanged.connect(self.set_structure_opacity)
        struct_opacity_layout.addWidget(self.struct_opacity_slider)
        structure_layout.addLayout(struct_opacity_layout)

        control_layout.addWidget(structure_group)

        # Placeholder cho các điều khiển bổ sung
        control_layout.addStretch(1)

        splitter.addWidget(control_panel)
        splitter.setSizes([700, 300])  # Phân chia mặc định

        main_layout.addWidget(splitter)

        # Set initial state
        self.viewer_3d.reset_camera()

    def load_data(self, dose_grid, structure_set=None, prescription=None):
        """
        Nạp dữ liệu liều và cấu trúc để hiển thị.

        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều 3D cần hiển thị
        structure_set : StructureSet, optional
            Tập cấu trúc cần hiển thị
        prescription : float, optional
            Liều chỉ định (dùng để chuẩn hóa)
        """
        if not dose_grid:
            logging.error("Không có dữ liệu liều để hiển thị.")
            return

        self._dose_grid = dose_grid
        self._structure_set = structure_set
        self._prescription = prescription

        # Chuẩn bị dữ liệu
        self._prepare_dose_data()
        self._prepare_structure_data()

        # Cập nhật giao diện
        self.update_display()

    def _prepare_dose_data(self):
        """Chuẩn bị dữ liệu liều để hiển thị."""
        if not self._dose_grid:
            return

        try:
            # Chuẩn hóa liều nếu cần
            if self._prescription:
                norm_factor = 100.0 / self._prescription
                dose_data = self._dose_grid.data * norm_factor
            else:
                dose_data = self._dose_grid.data

            # Cập nhật dữ liệu liều trong visualizer
            self.dose_vis.set_dose_data(
                dose_data,
                origin=self._dose_grid.origin,
                spacing=self._dose_grid.spacing,
            )

            # Cập nhật isodose_selector với giá trị dựa trên dữ liệu thực
            max_dose = np.max(dose_data)
            if max_dose > 0:
                # Tạo các mức isodose dựa trên % của liều tối đa
                if self._prescription:
                    # Nếu đã chuẩn hóa theo liều chỉ định thì dùng % liều chỉ định
                    levels = [
                        level for level in self._isodose_levels if level <= max_dose
                    ]
                else:
                    # Nếu không thì tự động tạo các mức
                    levels = [
                        round(p * max_dose / 100, 1)
                        for p in [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
                    ]

                self.isodose_selector.set_isodose_levels(levels)
        except Exception as e:
            logging.error(f"Lỗi khi chuẩn bị dữ liệu liều: {e}")

    def _prepare_structure_data(self):
        """Chuẩn bị dữ liệu cấu trúc để hiển thị."""
        if not self._structure_set:
            return

        try:
            # Xóa list cấu trúc hiện tại
            self.structure_panel.clear()

            # Thêm từng cấu trúc vào panel và visualizer
            for struct in self._structure_set.structures:
                if not struct.name:
                    continue

                # Thêm vào structure_panel
                color = (
                    struct.color if struct.color else (1.0, 0.0, 0.0)
                )  # Mặc định là đỏ
                self.structure_panel.add_structure(struct.id, struct.name, color, True)

                # Thêm vào active_structures
                self._active_structures[struct.id] = True

                # Thêm vào visualizer
                if hasattr(struct, "mesh") and struct.mesh is not None:
                    self.dose_vis.add_structure_from_mesh(struct.id, struct.mesh, color)
                elif hasattr(struct, "mask") and struct.mask is not None:
                    self.dose_vis.add_structure_from_mask(
                        struct.id,
                        struct.mask,
                        color,
                        origin=self._dose_grid.origin if self._dose_grid else None,
                        spacing=self._dose_grid.spacing if self._dose_grid else None,
                    )
        except Exception as e:
            logging.error(f"Lỗi khi chuẩn bị dữ liệu cấu trúc: {e}")

    def update_display(self):
        """Cập nhật hiển thị dựa trên cài đặt hiện tại."""
        if not self._dose_grid:
            return

        try:
            # Cập nhật isodose
            if self._colorwash_enabled:
                isodose_levels = self.isodose_selector.get_isodose_levels()
                isodose_colors = self.isodose_selector.get_isodose_colors()

                self.dose_vis.set_isodose_levels(isodose_levels)
                self.dose_vis.set_isodose_colors(isodose_colors)
                self.dose_vis.set_dose_opacity(self._dose_opacity)
                self.dose_vis.set_display_mode(self._display_mode.lower())
                self.dose_vis.show_dose(True)
            else:
                self.dose_vis.show_dose(False)

            # Cập nhật cấu trúc
            if self._structures_enabled:
                self.dose_vis.set_structure_opacity(self._structure_opacity)
                self.dose_vis.show_structures(True)

                # Cập nhật visibility của từng cấu trúc
                for struct_id, visible in self._active_structures.items():
                    self.dose_vis.set_structure_visibility(struct_id, visible)
            else:
                self.dose_vis.show_structures(False)

            # Cập nhật hiển thị nhãn
            self.dose_vis.show_isodose_labels(self._show_isodose_labels)

            # Kích hoạt render
            self.viewer_3d.render()
        except Exception as e:
            logging.error(f"Lỗi khi cập nhật hiển thị: {e}")

    # Handlers cho sự kiện UI

    def set_display_mode(self, mode):
        """Đặt chế độ hiển thị liều."""
        if mode in ["Surface", "Volume", "Contour"]:
            self._display_mode = mode
            self.dose_display_mode_changed.emit(mode)
            self.update_display()

    def update_isodose_levels(self, levels, colors):
        """Cập nhật các mức isodose và màu sắc."""
        self._isodose_levels = levels
        self._isodose_colors = colors
        self.update_display()

    def set_dose_opacity(self, value):
        """Đặt độ trong suốt cho hiển thị liều."""
        self._dose_opacity = value / 100.0
        self.update_display()

    def set_structure_opacity(self, value):
        """Đặt độ trong suốt cho hiển thị cấu trúc."""
        self._structure_opacity = value / 100.0
        self.update_display()

    def toggle_dose(self, enabled):
        """Bật/tắt hiển thị liều."""
        self._colorwash_enabled = enabled
        self.update_display()

    def toggle_structures(self, enabled):
        """Bật/tắt hiển thị cấu trúc."""
        self._structures_enabled = enabled
        self.update_display()

    def toggle_isodose_labels(self, enabled):
        """Bật/tắt hiển thị nhãn isodose."""
        self._show_isodose_labels = enabled
        self.update_display()

    def update_structure_visibility(self, struct_id, visible):
        """Cập nhật hiển thị cho một cấu trúc cụ thể."""
        self._active_structures[struct_id] = visible
        self.update_display()

    def export_screenshot(self):
        """Xuất ảnh chụp màn hình hiện tại."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Screenshot",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
            )

            if filename:
                self.viewer_3d.save_screenshot(filename)
        except Exception as e:
            logging.error(f"Lỗi khi xuất ảnh chụp màn hình: {e}")

    def reset_view(self):
        """Đặt lại góc nhìn camera về vị trí mặc định."""
        self.viewer_3d.reset_camera()

    def clear(self):
        """Xóa tất cả dữ liệu hiển thị."""
        try:
            self.dose_vis.clear()
            self.structure_panel.clear()
            self._dose_grid = None
            self._structure_set = None
            self._prescription = None
            self._active_structures = {}
            self.viewer_3d.render()
        except Exception as e:
            logging.error(f"Lỗi khi xóa dữ liệu: {e}")

    # Phương thức demo và test

    @staticmethod
    def create_test_data():
        """Tạo dữ liệu mẫu để test hiển thị."""
        # Tạo lưới liều mẫu
        dose_shape = (50, 50, 30)
        dose_data = np.zeros(dose_shape)

        # Tạo phân phối liều tương tự hình cầu
        center = np.array(dose_shape) / 2
        max_radius = min(dose_shape) / 3

        xx, yy, zz = np.meshgrid(
            np.arange(dose_shape[0]), np.arange(dose_shape[1]), np.arange(dose_shape[2])
        )

        radius = np.sqrt(
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
        )

        # Tạo phân phối liều hình cầu giảm dần từ tâm
        dose_data = 100 * np.exp(-((radius / max_radius) ** 2))

        # Create dose grid
        dose_grid = type(
            "DoseGrid",
            (),
            {"data": dose_data, "origin": (-125, -125, -75), "spacing": (5, 5, 5)},
        )

        # Tạo các cấu trúc mẫu
        structure_set = type("StructureSet", (), {"structures": []})

        # Tạo cấu trúc PTV
        ptv_radius = max_radius * 0.7
        ptv_mask = radius <= ptv_radius

        ptv = type(
            "Structure",
            (),
            {
                "id": "PTV",
                "name": "PTV",
                "color": (1.0, 0.0, 0.0),  # Red
                "mask": ptv_mask,
                "mesh": None,
            },
        )

        # Tạo cấu trúc OAR
        oar_center = center + np.array([0, 0, max_radius * 0.5])
        xx, yy, zz = np.meshgrid(
            np.arange(dose_shape[0]), np.arange(dose_shape[1]), np.arange(dose_shape[2])
        )

        oar_radius = max_radius * 0.4
        oar_mask = (
            (xx - oar_center[0]) ** 2
            + (yy - oar_center[1]) ** 2
            + (zz - oar_center[2]) ** 2
        ) <= oar_radius**2

        oar = type(
            "Structure",
            (),
            {
                "id": "OAR",
                "name": "Organ at Risk",
                "color": (0.0, 0.0, 1.0),  # Blue
                "mask": oar_mask,
                "mesh": None,
            },
        )

        # Thêm cấu trúc vào tập hợp
        structure_set.structures = [ptv, oar]

        return dose_grid, structure_set, 70.0  # 70 Gy prescription

    @staticmethod
    def run_standalone():
        """Chạy widget như một ứng dụng độc lập để test."""
        app = QApplication(sys.argv)

        viewer = Dose3DViewer()
        viewer.setWindowTitle("QuangTPS - 3D Dose Viewer")
        viewer.resize(1200, 800)

        # Tạo dữ liệu mẫu
        dose_grid, structure_set, prescription = Dose3DViewer.create_test_data()

        # Nạp dữ liệu vào viewer
        viewer.load_data(dose_grid, structure_set, prescription)

        viewer.show()
        sys.exit(app.exec_())


# Để chạy module này như ứng dụng độc lập cho mục đích test
if __name__ == "__main__":
    Dose3DViewer.run_standalone()
