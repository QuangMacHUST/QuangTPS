#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module hiển thị ảnh 3D cho QuangTPS.

Module này cung cấp widget hiển thị 3D cho hình ảnh y tế, bao gồm nhiều chế độ
hiển thị như bề mặt (surface), thể tích (volume), chiếu cường độ cực đại (MIP)
và X-ray. Được thiết kế để tích hợp với các module khác trong QuangTPS.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from PyQt5.QtWidgets import (  # pylint: disable=no-name-in-module
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QPushButton,
    QToolBar,
    QAction,
    QFrame,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt5.QtGui import QIcon, QColor  # pylint: disable=no-name-in-module


# Thử import VTK
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from quangtps.ui.vtk_viewer_3d import VTKViewer3D

    HAS_VTK = True
except ImportError:
    logging.warning("Không thể import VTK. Chức năng hiển thị 3D sẽ bị giới hạn.")
    HAS_VTK = False

# Thử import PyVista
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor, BackgroundPlotter

    HAS_PYVISTA = True
except ImportError:
    logging.warning("Không thể import PyVista. Một số chức năng 3D sẽ bị giới hạn.")
    HAS_PYVISTA = False

# Import các module nội bộ
from quangtps.core.logging import get_logger
from quangtps.imaging.image import Image
from quangtps.imaging.structures import Structure, StructureSet
from quangtps.dose.dose_grid import DoseGrid

logger = get_logger(__name__)


class Image3DWidget(QWidget):
    """
    Widget hiển thị dữ liệu hình ảnh y tế 3D với nhiều chế độ hiển thị.

    Hỗ trợ các chế độ như surface rendering, volume rendering, MIP và X-ray,
    tương thích với các hệ thống TPS thương mại.
    """

    # Tín hiệu
    render_complete = pyqtSignal()
    view_changed = pyqtSignal(str)  # Tín hiệu khi chế độ xem thay đổi

    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị 3D.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)

        # Thuộc tính dữ liệu
        self._image = None  # Dữ liệu hình ảnh
        self._structures = {}  # Cấu trúc {id: Structure}
        self._dose = None  # Dữ liệu liều
        self._mode = "surface"  # Chế độ hiển thị mặc định

        # Thuộc tính trực quan hóa
        self._opacity = 0.7
        self._structure_opacity = 0.5
        self._dose_opacity = 0.6
        self._structure_visibility = {}
        self._isodose_levels = [10, 20, 30, 50, 70, 80, 90, 95, 100]
        self._show_structures = True
        self._show_dose = True
        self._colormap = "viridis"
        self._invert = False

        # Thiết lập UI
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar cho các điều khiển
        self._toolbar = QToolBar()
        self._toolbar.setIconSize(QSize(16, 16))
        layout.addWidget(self._toolbar)

        # Thêm các điều khiển cho chế độ hiển thị
        mode_label = QLabel("Chế độ:")
        self._toolbar.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Surface", "Volume", "MIP", "X-Ray"])
        self._mode_combo.currentTextChanged.connect(self._change_mode)
        self._toolbar.addWidget(self._mode_combo)
        self._toolbar.addSeparator()

        # Các chế độ xem tiêu chuẩn
        view_label = QLabel("Góc nhìn:")
        self._toolbar.addWidget(view_label)

        for view in ["Axial", "Coronal", "Sagittal", "3D"]:
            action = QAction(view, self)
            action.triggered.connect(lambda checked, v=view.lower(): self._set_view(v))
            self._toolbar.addAction(action)

        self._toolbar.addSeparator()

        # Điều khiển hiển thị
        self._toggle_structures = QAction("Structures", self)
        self._toggle_structures.setCheckable(True)
        self._toggle_structures.setChecked(True)
        self._toggle_structures.triggered.connect(self._toggle_structure_display)
        self._toolbar.addAction(self._toggle_structures)

        self._toggle_dose = QAction("Dose", self)
        self._toggle_dose.setCheckable(True)
        self._toggle_dose.setChecked(True)
        self._toggle_dose.triggered.connect(self._toggle_dose_display)
        self._toolbar.addAction(self._toggle_dose)

        # Khu vực hiển thị chính
        self._frame = QFrame()
        self._frame.setFrameStyle(QFrame.StyledPanel)
        self._frame.setStyleSheet("background-color: black;")
        self._frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._frame)

        # Thiết lập renderer
        self._setup_renderer()

    def _setup_renderer(self):
        """Thiết lập renderer 3D dựa trên các thư viện có sẵn."""
        renderer_layout = QVBoxLayout(self._frame)
        renderer_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYVISTA:
            try:
                # PyVista renderer
                self._plotter = QtInteractor(self._frame)
                renderer_layout.addWidget(self._plotter)
                self._render_engine = "pyvista"
                logger.info("Sử dụng PyVista làm engine render 3D")

                # Thiết lập màu nền
                self._plotter.set_background("black")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo PyVista renderer: {e}")
                self._setup_fallback_renderer(renderer_layout)
        elif HAS_VTK:
            try:
                # Sử dụng lớp VTK Viewer từ QuangTPS
                self._vtk_viewer = VTKViewer3D(self._frame)
                renderer_layout.addWidget(self._vtk_viewer)
                self._render_engine = "vtk"
                logger.info("Sử dụng VTK làm engine render 3D")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo VTK renderer: {e}")
                self._setup_fallback_renderer(renderer_layout)
        else:
            # Không có engine 3D, sử dụng renderer giả
            self._setup_fallback_renderer(renderer_layout)

    def _setup_fallback_renderer(self, layout):
        """Thiết lập renderer dự phòng khi không có engine 3D."""
        self._render_engine = "none"
        message = QLabel(
            "Hiển thị 3D không khả dụng.\nVui lòng cài đặt VTK hoặc PyVista."
        )
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(message)
        logger.warning("Sử dụng renderer giả do không có engine 3D")

    def set_image(self, image):
        """
        Thiết lập hình ảnh để hiển thị.

        Parameters
        ----------
        image : Image hoặc numpy.ndarray
            Hình ảnh để hiển thị
        """
        self._image = image
        self._update_display()

    def add_structure(self, structure_id, structure, color=None):
        """
        Thêm cấu trúc để hiển thị.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        structure : Structure
            Đối tượng Structure
        color : tuple hoặc QColor, optional
            Màu sắc hiển thị
        """
        self._structures[structure_id] = structure
        self._structure_visibility[structure_id] = True
        self._update_display()

    def set_structure_set(self, structure_set):
        """
        Thiết lập tập cấu trúc để hiển thị.

        Parameters
        ----------
        structure_set : StructureSet hoặc dict
            Tập cấu trúc để hiển thị
        """
        if isinstance(structure_set, dict):
            self._structures = structure_set.copy()
        else:
            # Giả sử structure_set có phương thức get_structures() trả về dict
            self._structures = structure_set.get_structures()

        # Khởi tạo trạng thái hiển thị
        for struct_id in self._structures:
            self._structure_visibility[struct_id] = True

        self._update_display()

    def set_dose(self, dose):
        """
        Thiết lập dữ liệu liều để hiển thị.

        Parameters
        ----------
        dose : DoseGrid hoặc numpy.ndarray
            Dữ liệu liều để hiển thị
        """
        self._dose = dose
        self._update_display()

    def set_mode(self, mode):
        """
        Thiết lập chế độ hiển thị.

        Parameters
        ----------
        mode : str
            Chế độ hiển thị ('surface', 'volume', 'mip', 'xray')
        """
        mode = mode.lower()
        if mode in ["surface", "volume", "mip", "xray"]:
            self._mode = mode
            self._mode_combo.setCurrentText(mode.capitalize())
            self._update_display()

    def _change_mode(self, mode_text):
        """Xử lý khi người dùng thay đổi chế độ hiển thị từ combo box."""
        self.set_mode(mode_text.lower())

    def _set_view(self, view):
        """Thiết lập góc nhìn tiêu chuẩn."""
        if self._render_engine == "pyvista" and hasattr(self, "_plotter"):
            if view == "axial":
                self._plotter.view_xy()
            elif view == "coronal":
                self._plotter.view_yz()
            elif view == "sagittal":
                self._plotter.view_xz()
            elif view == "3d":
                self._plotter.view_isometric()
        elif self._render_engine == "vtk" and hasattr(self, "_vtk_viewer"):
            self._vtk_viewer.set_view(view)

        self.view_changed.emit(view)

    def _toggle_structure_display(self, enabled):
        """Bật/tắt hiển thị cấu trúc."""
        self._show_structures = enabled
        self._update_display()

    def _toggle_dose_display(self, enabled):
        """Bật/tắt hiển thị liều."""
        self._show_dose = enabled
        self._update_display()

    def set_structure_visibility(self, structure_id, visible):
        """
        Thiết lập khả năng hiển thị của cấu trúc cụ thể.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        visible : bool
            Trạng thái hiển thị
        """
        if structure_id in self._structure_visibility:
            self._structure_visibility[structure_id] = visible
            self._update_display()

    def set_structure_opacity(self, opacity):
        """
        Thiết lập độ mờ đục của tất cả cấu trúc.

        Parameters
        ----------
        opacity : float
            Độ mờ đục (0-1)
        """
        self._structure_opacity = max(0.0, min(1.0, opacity))
        self._update_display()

    def set_dose_opacity(self, opacity):
        """
        Thiết lập độ mờ đục của hiển thị liều.

        Parameters
        ----------
        opacity : float
            Độ mờ đục (0-1)
        """
        self._dose_opacity = max(0.0, min(1.0, opacity))
        self._update_display()

    def set_isodose_levels(self, levels):
        """
        Thiết lập các mức isodose để hiển thị.

        Parameters
        ----------
        levels : List[float]
            Danh sách các mức isodose (%)
        """
        self._isodose_levels = sorted(levels)
        if self._show_dose:
            self._update_display()

    def _update_display(self):
        """Cập nhật hiển thị 3D dựa trên thuộc tính hiện tại."""
        if self._render_engine == "none":
            return

        # Xóa hiển thị hiện tại
        self._clear_display()

        # Cập nhật dựa trên engine render
        if self._render_engine == "pyvista":
            self._update_pyvista_display()
        elif self._render_engine == "vtk":
            self._update_vtk_display()

    def _clear_display(self):
        """Xóa tất cả đối tượng đang hiển thị."""
        if self._render_engine == "pyvista" and hasattr(self, "_plotter"):
            self._plotter.clear()
        elif self._render_engine == "vtk" and hasattr(self, "_vtk_viewer"):
            self._vtk_viewer.clear_scene()

    def _update_pyvista_display(self):
        """Cập nhật hiển thị sử dụng PyVista."""
        if not hasattr(self, "_plotter") or self._image is None:
            return

        # Xử lý hiển thị hình ảnh dựa trên chế độ
        try:
            # Chuyển đổi dữ liệu hình ảnh sang định dạng PyVista
            if hasattr(self._image, "get_array"):
                image_data = self._image.get_array()
                spacing = self._image.get_spacing()
            else:
                image_data = self._image
                spacing = (1.0, 1.0, 1.0)  # Mặc định

            # Tạo grid từ mảng numpy
            grid = pv.UniformGrid()
            grid.dimensions = np.array(image_data.shape) + 1
            grid.spacing = spacing
            grid.point_data["values"] = image_data.flatten(order="F")

            # Hiển thị dựa trên chế độ
            if self._mode == "surface":
                # Hiển thị dạng bề mặt
                contours = grid.contour(5)
                self._plotter.add_mesh(
                    contours, opacity=self._opacity, cmap=self._colormap
                )
            elif self._mode == "volume":
                # Hiển thị dạng thể tích
                self._plotter.add_volume(
                    grid, cmap=self._colormap, opacity=self._opacity
                )
            elif self._mode == "mip" or self._mode == "xray":
                # Hiển thị dạng MIP hoặc X-ray
                volume = self._plotter.add_volume(
                    grid, cmap=self._colormap, opacity=self._opacity
                )

                # Điều chỉnh chế độ blend dựa trên loại
                if self._mode == "mip":
                    volume.mapper.blend_mode = "maximum_intensity"
                else:
                    volume.mapper.blend_mode = "average_intensity"

            # Hiển thị cấu trúc nếu được bật
            if self._show_structures and self._structures:
                self._add_structures_pyvista()

            # Hiển thị liều nếu được bật
            if self._show_dose and self._dose is not None:
                self._add_dose_pyvista()

            # Cập nhật khung nhìn
            self._plotter.reset_camera()
            self._plotter.update()

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị PyVista: {e}")

    def _add_structures_pyvista(self):
        """Thêm cấu trúc vào hiển thị PyVista."""
        if not hasattr(self, "_plotter"):
            return

        # Lặp qua các cấu trúc và hiển thị
        for struct_id, structure in self._structures.items():
            if not self._structure_visibility.get(struct_id, True):
                continue

            try:
                # Lấy lưới đa giác từ cấu trúc
                if hasattr(structure, "get_mesh"):
                    mesh = structure.get_mesh()
                    if mesh is None:
                        continue

                    # Chuyển đổi sang định dạng PyVista
                    points = mesh.get("points", [])
                    faces = mesh.get("faces", [])

                    if len(points) > 0 and len(faces) > 0:
                        # Tạo lưới PyVista
                        pv_mesh = pv.PolyData(np.array(points), np.array(faces))

                        # Lấy màu cấu trúc hoặc màu mặc định
                        color = (
                            structure.get_color()
                            if hasattr(structure, "get_color")
                            else (1.0, 0.0, 0.0)
                        )

                        # Thêm vào plotter
                        self._plotter.add_mesh(
                            pv_mesh,
                            color=color,
                            opacity=self._structure_opacity,
                            smooth_shading=True,
                        )
            except Exception as e:
                logger.error(f"Lỗi khi thêm cấu trúc {struct_id} vào PyVista: {e}")

    def _add_dose_pyvista(self):
        """Thêm hiển thị liều vào PyVista."""
        if not hasattr(self, "_plotter") or self._dose is None:
            return

        try:
            # Lấy dữ liệu liều
            if hasattr(self._dose, "get_grid"):
                dose_data = self._dose.get_grid()
                spacing = self._dose.get_spacing()
            else:
                dose_data = self._dose
                spacing = (1.0, 1.0, 1.0)  # Mặc định

            # Tạo grid PyVista từ dữ liệu liều
            dose_grid = pv.UniformGrid()
            dose_grid.dimensions = np.array(dose_data.shape) + 1
            dose_grid.spacing = spacing
            dose_grid.point_data["values"] = dose_data.flatten(order="F")

            # Hiển thị các đường isodose
            for level in self._isodose_levels:
                try:
                    if np.max(dose_data) > 0:
                        normalized_level = level / 100.0 * np.max(dose_data)
                        contour = dose_grid.contour([normalized_level])
                        if contour.n_points > 0:
                            # Màu dựa theo mức liều
                            color = self._get_isodose_color(level)
                            self._plotter.add_mesh(
                                contour,
                                color=color,
                                opacity=self._dose_opacity,
                                label=f"{level}%",
                            )
                except Exception as e:
                    logger.debug(f"Lỗi khi hiển thị isodose {level}%: {e}")
        except Exception as e:
            logger.error(f"Lỗi khi thêm hiển thị liều vào PyVista: {e}")

    def _update_vtk_display(self):
        """Cập nhật hiển thị sử dụng VTK Viewer."""
        if not hasattr(self, "_vtk_viewer"):
            return

        try:
            # Gọi các phương thức thích hợp của VTKViewer3D
            if self._image is not None:
                # Giả định VTKViewer3D có các phương thức này
                self._vtk_viewer.set_image(self._image)

            if self._show_structures and self._structures:
                for struct_id, structure in self._structures.items():
                    if self._structure_visibility.get(struct_id, True):
                        self._vtk_viewer.add_structure(struct_id, structure)

            if self._show_dose and self._dose is not None:
                self._vtk_viewer.set_dose(self._dose)
                self._vtk_viewer.set_isodose_levels(self._isodose_levels)

            # Cập nhật hiển thị với mode tương ứng
            if self._mode == "surface":
                self._vtk_viewer.set_display_mode("surface")
            elif self._mode == "volume":
                self._vtk_viewer.set_display_mode("volume")
            elif self._mode == "mip":
                self._vtk_viewer.set_display_mode("mip")
            elif self._mode == "xray":
                self._vtk_viewer.set_display_mode("xray")

            # Cập nhật độ mờ đục
            self._vtk_viewer.set_structure_opacity(self._structure_opacity)
            self._vtk_viewer.set_dose_opacity(self._dose_opacity)

            # Render scene
            self._vtk_viewer.render()
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị VTK: {e}")

    def _get_isodose_color(self, level):
        """
        Lấy màu cho mức isodose cụ thể.

        Parameters
        ----------
        level : float
            Mức isodose (%)

        Returns
        -------
        tuple
            Màu RGB (0-1)
        """
        # Bảng màu mặc định theo Eclipse của Varian
        if level >= 100:
            return (1.0, 0.0, 0.0)  # Đỏ
        elif level >= 95:
            return (1.0, 0.5, 0.0)  # Cam đỏ
        elif level >= 90:
            return (1.0, 0.75, 0.0)  # Cam vàng
        elif level >= 80:
            return (1.0, 1.0, 0.0)  # Vàng
        elif level >= 70:
            return (0.0, 1.0, 0.0)  # Xanh lá
        elif level >= 50:
            return (0.0, 1.0, 1.0)  # Cyan
        elif level >= 30:
            return (0.0, 0.5, 1.0)  # Xanh dương nhạt
        else:
            return (0.0, 0.0, 1.0)  # Xanh dương

    def take_screenshot(self, filename=None):
        """
        Chụp ảnh màn hình hiển thị 3D hiện tại.

        Parameters
        ----------
        filename : str, optional
            Tên file để lưu. Nếu None, trả về dữ liệu ảnh.

        Returns
        -------
        np.ndarray hoặc None
            Dữ liệu ảnh nếu filename là None, ngược lại trả về None
        """
        if self._render_engine == "pyvista" and hasattr(self, "_plotter"):
            return self._plotter.screenshot(filename, return_img=filename is None)
        elif self._render_engine == "vtk" and hasattr(self, "_vtk_viewer"):
            return self._vtk_viewer.get_screenshot(filename)
        return None

    def reset_view(self):
        """Đặt lại góc nhìn camera về mặc định."""
        if self._render_engine == "pyvista" and hasattr(self, "_plotter"):
            self._plotter.reset_camera()
        elif self._render_engine == "vtk" and hasattr(self, "_vtk_viewer"):
            self._vtk_viewer.reset_camera()
