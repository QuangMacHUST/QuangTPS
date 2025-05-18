#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module hiển thị 3D cho cấu trúc giải phẫu trong QuangTPS.

Module này cung cấp các lớp và hàm để hiển thị cấu trúc giải phẫu trong không gian 3D,
với các tính năng hiển thị chuyên nghiệp tương tự Eclipse của Varian.
"""

import logging
import numpy as np
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import traceback

logger = logging.getLogger(__name__)

# Thử import các thư viện hiển thị 3D với xử lý lỗi
VTK_AVAILABLE = False
PYVISTA_AVAILABLE = False
PYQT_VTK_AVAILABLE = False

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_AVAILABLE = True
    logger.info("Đã import VTK thành công")
except ImportError:
    logger.warning("Không thể import VTK. Một số tính năng 3D sẽ bị hạn chế.")

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
    logger.info("Đã import PyVista thành công")
except ImportError:
    logger.warning("Không thể import PyVista. Sẽ sử dụng VTK thuần nếu có thể.")

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QComboBox,
        QLabel,
        QSlider,
    )
    from PyQt5.QtCore import Qt, pyqtSignal

    if VTK_AVAILABLE:
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        PYQT_VTK_AVAILABLE = True
        logger.info("Đã import QVTKRenderWindowInteractor thành công")
except ImportError:
    logger.warning(
        "Không thể import PyQt5 hoặc QVTKRenderWindowInteractor. Hiển thị 3D sẽ bị hạn chế."
    )


class StructureViewer3D(QWidget):
    """
    Widget hiển thị cấu trúc giải phẫu trong không gian 3D.

    Widget này cung cấp giao diện hiển thị và tương tác với các cấu trúc giải phẫu,
    tương tự như chức năng hiển thị 3D của Eclipse (Varian).
    """

    # Tín hiệu
    structureClicked = pyqtSignal(str)  # Phát ra khi nhấp chuột vào cấu trúc
    viewChanged = pyqtSignal(dict)  # Phát ra khi thay đổi góc nhìn

    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị 3D.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        self.parent = parent
        self.structures = {}  # Dict structure_id -> cấu trúc
        self.structure_actors = {}  # Dict structure_id -> actor
        self.structure_colors = {}  # Dict structure_id -> màu sắc
        self.structure_opacities = {}  # Dict structure_id -> độ trong suốt

        # Đặt kích thước tối thiểu
        self.setMinimumSize(400, 300)

        # Thiết lập UI
        self.init_ui()

    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Nếu không có thư viện VTK, hiển thị thông báo
        if not (VTK_AVAILABLE and PYQT_VTK_AVAILABLE):
            label = QLabel("Không thể hiển thị 3D. Vui lòng cài đặt VTK và PyQt5.")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self.setLayout(layout)
            return

        # Tạo layout cho điều khiển và hiển thị
        main_layout = QVBoxLayout()

        # Thêm controls
        controls_layout = QHBoxLayout()

        # Combo box chọn kiểu hiển thị
        view_type_label = QLabel("Kiểu hiển thị:")
        self.view_type_combo = QComboBox()
        self.view_type_combo.addItems(
            ["Surface", "Wireframe", "Surface + Wireframe", "Points"]
        )
        self.view_type_combo.currentIndexChanged.connect(self._on_view_type_changed)
        controls_layout.addWidget(view_type_label)
        controls_layout.addWidget(self.view_type_combo)

        # Nút hiển thị tiêu chuẩn
        self.standard_views = QComboBox()
        self.standard_views.addItems(
            ["Anterior", "Posterior", "Left", "Right", "Superior", "Inferior"]
        )
        self.standard_views.currentIndexChanged.connect(self._on_standard_view_changed)
        controls_layout.addWidget(QLabel("Góc nhìn:"))
        controls_layout.addWidget(self.standard_views)

        # Nút reset camera
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_camera)
        controls_layout.addWidget(self.reset_btn)

        # Slider điều chỉnh độ trong suốt
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(10)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Độ trong suốt:"))
        opacity_layout.addWidget(self.opacity_slider)

        # Thêm các layout con vào layout chính
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(opacity_layout)

        # Tạo VTK render window
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Thiết lập interactor style để xử lý sự kiện
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.interactor_style)

        # Thêm callback cho tương tác chuột
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.interactor.SetPicker(self.picker)

        # Biến lưu trạng thái hiển thị
        self.display_mode = "surface"  # surface, wireframe, surface_wireframe, points

        # Thêm VTK widget vào layout
        main_layout.addWidget(self.vtk_widget)

        # Thiết lập layout
        layout.addLayout(main_layout)
        self.setLayout(layout)

        # Khởi tạo renderer
        self.renderer.SetBackground(0.1, 0.1, 0.2)  # Màu nền xanh đậm
        self.interactor.Initialize()

        # Cài đặt callback cho picker
        self.setup_picker_callback()

    def setup_picker_callback(self):
        """Thiết lập callback cho picker để xử lý sự kiện click chuột."""
        if not (VTK_AVAILABLE and PYQT_VTK_AVAILABLE):
            return

        def on_click(obj, event):
            click_pos = obj.GetEventPosition()
            self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
            actor = self.picker.GetActor()

            if actor:
                # Tìm structure_id từ actor
                for struct_id, struct_actor in self.structure_actors.items():
                    if struct_actor == actor:
                        # Phát tín hiệu với ID của cấu trúc
                        self.structureClicked.emit(struct_id)
                        logger.debug(f"Đã click vào cấu trúc: {struct_id}")
                        return

        # Thêm observer cho sự kiện click chuột
        click_observer = self.interactor.AddObserver("LeftButtonPressEvent", on_click)

    def add_structure(self, structure, color=None):
        """
        Thêm cấu trúc vào hiển thị 3D.

        Parameters
        ----------
        structure : Structure
            Cấu trúc cần hiển thị
        color : tuple or str, optional
            Màu sắc của cấu trúc, định dạng RGB (tuple) hoặc hex (str)
        """
        if not (VTK_AVAILABLE and PYQT_VTK_AVAILABLE):
            logger.warning(
                "Không thể thêm cấu trúc 3D vì thiếu thư viện VTK hoặc PyQt-VTK"
            )
            return

        try:
            # Lấy ID của cấu trúc
            structure_id = getattr(structure, "id", str(id(structure)))

            # Lưu cấu trúc vào dictionary
            self.structures[structure_id] = structure

            # Lấy mask 3D
            mask_3d = getattr(structure, "mask_3d", None)
            if mask_3d is None:
                logger.warning(f"Cấu trúc {structure_id} không có mask 3D")
                return

            # Chuyển đổi mask thành mesh bề mặt
            mesh = self._create_mesh_from_mask(mask_3d)
            if mesh is None:
                logger.warning(f"Không thể tạo mesh cho cấu trúc {structure_id}")
                return

            # Tạo actor và thêm vào renderer
            actor = self._add_mesh_to_renderer(mesh)

            # Lưu actor vào dictionary
            self.structure_actors[structure_id] = actor

            # Thiết lập màu nếu có
            if color:
                self.set_structure_color(structure_id, color)
            else:
                # Lấy màu từ thuộc tính của structure nếu có
                struct_color = getattr(structure, "color", None)
                if struct_color:
                    self.set_structure_color(structure_id, struct_color)
                else:
                    # Sử dụng màu mặc định
                    default_colors = {
                        "ptv": (1.0, 0.2, 0.2),  # Đỏ
                        "ctv": (0.8, 0.5, 0.2),  # Cam
                        "gtv": (1.0, 0.0, 0.0),  # Đỏ đậm
                        "organ": (0.2, 0.8, 0.2),  # Xanh lá
                        "external": (0.7, 0.7, 0.7, 0.1),  # Xám trong suốt
                    }

                    # Mặc định là màu xanh lá
                    color = (0.2, 0.6, 0.8)  # Xanh dương

                    # Tìm màu theo tên cấu trúc
                    struct_name = getattr(structure, "name", "").lower()
                    struct_type = getattr(structure, "type", "").lower()

                    for key, clr in default_colors.items():
                        if key in struct_name or key in struct_type:
                            color = clr
                            break

                    self.set_structure_color(structure_id, color)

            # Thiết lập độ trong suốt mặc định
            self.set_structure_opacity(structure_id, 0.8)

            # Cập nhật view
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            logger.info(f"Đã thêm cấu trúc {structure_id} vào hiển thị 3D")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi thêm cấu trúc 3D: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _create_mesh_from_mask(self, mask_3d):
        """
        Tạo mesh từ mask 3D.

        Parameters
        ----------
        mask_3d : ndarray
            Mask 3D của cấu trúc

        Returns
        -------
        vtkPolyData
            Mesh VTK tạo từ mask
        """
        if not VTK_AVAILABLE:
            return None

        try:
            # Chuyển đổi mask thành vtk image data
            dims = mask_3d.shape
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(dims[2], dims[1], dims[0])
            vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

            # Chuyển đổi mask từ numpy sang vtk
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        vtk_image.SetScalarComponentFromFloat(
                            k, j, i, 0, mask_3d[i, j, k]
                        )

            # Tạo bề mặt bằng marching cubes
            mc = vtk.vtkMarchingCubes()
            mc.SetInputData(vtk_image)
            mc.SetValue(0, 0.5)  # Giá trị ngưỡng
            mc.Update()

            # Tạo mesh
            mesh = vtk.vtkPolyData()
            mesh.DeepCopy(mc.GetOutput())

            # Làm mịn mesh
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputData(mesh)
            smoother.SetNumberOfIterations(15)
            smoother.SetRelaxationFactor(0.1)
            smoother.Update()

            # Giảm số lượng tam giác
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputData(smoother.GetOutput())
            decimate.SetTargetReduction(0.5)  # Giảm 50% số tam giác
            decimate.PreserveTopologyOn()
            decimate.Update()

            # Tính toán vector pháp tuyến
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(decimate.GetOutput())
            normals.SetFeatureAngle(60.0)
            normals.ComputePointNormalsOn()
            normals.Update()

            return normals.GetOutput()

        except Exception as e:
            logger.error(f"Lỗi khi tạo mesh từ mask: {str(e)}")
            return None

    def _add_mesh_to_renderer(self, mesh):
        """
        Thêm mesh vào renderer.

        Parameters
        ----------
        mesh : vtkPolyData
            Mesh VTK cần thêm vào renderer

        Returns
        -------
        vtkActor
            Actor VTK đã tạo
        """
        if not VTK_AVAILABLE or mesh is None:
            return None

        try:
            # Tạo mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)

            # Tạo actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Thiết lập thuộc tính bề mặt
            actor.GetProperty().SetSpecular(0.3)
            actor.GetProperty().SetSpecularPower(20)
            actor.GetProperty().SetInterpolationToPhong()

            # Thêm actor vào renderer
            self.renderer.AddActor(actor)

            return actor

        except Exception as e:
            logger.error(f"Lỗi khi thêm mesh vào renderer: {str(e)}")
            return None

    def set_structure_color(self, structure_id, color):
        """
        Thiết lập màu cho cấu trúc.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        color : tuple or str
            Màu sắc định dạng RGB (0-1) hoặc hex
        """
        if not VTK_AVAILABLE or structure_id not in self.structure_actors:
            return

        try:
            # Chuyển đổi màu từ hex sang RGB nếu cần
            if isinstance(color, str) and color.startswith("#"):
                r = int(color[1:3], 16) / 255.0
                g = int(color[3:5], 16) / 255.0
                b = int(color[5:7], 16) / 255.0
                color = (r, g, b)

            # Thiết lập màu cho actor
            actor = self.structure_actors[structure_id]
            if len(color) == 3:
                actor.GetProperty().SetColor(color[0], color[1], color[2])
            elif len(color) == 4:
                actor.GetProperty().SetColor(color[0], color[1], color[2])
                actor.GetProperty().SetOpacity(color[3])

            # Lưu màu
            self.structure_colors[structure_id] = color

            # Render lại
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập màu cho cấu trúc {structure_id}: {str(e)}")

    def set_structure_opacity(self, structure_id, opacity):
        """
        Thiết lập độ trong suốt cho cấu trúc.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        opacity : float
            Độ trong suốt từ 0.0 (trong suốt hoàn toàn) đến 1.0 (không trong suốt)
        """
        if not VTK_AVAILABLE or structure_id not in self.structure_actors:
            return

        try:
            # Thiết lập độ trong suốt
            actor = self.structure_actors[structure_id]
            actor.GetProperty().SetOpacity(opacity)

            # Lưu độ trong suốt
            self.structure_opacities[structure_id] = opacity

            # Render lại
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            logger.error(
                f"Lỗi khi thiết lập độ trong suốt cho cấu trúc {structure_id}: {str(e)}"
            )

    def remove_structure(self, structure_id):
        """
        Xóa cấu trúc khỏi hiển thị 3D.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc cần xóa
        """
        if not VTK_AVAILABLE or structure_id not in self.structure_actors:
            return

        try:
            # Xóa actor khỏi renderer
            actor = self.structure_actors[structure_id]
            self.renderer.RemoveActor(actor)

            # Xóa khỏi dictionaries
            del self.structure_actors[structure_id]
            if structure_id in self.structures:
                del self.structures[structure_id]
            if structure_id in self.structure_colors:
                del self.structure_colors[structure_id]
            if structure_id in self.structure_opacities:
                del self.structure_opacities[structure_id]

            # Render lại
            self.vtk_widget.GetRenderWindow().Render()
            logger.info(f"Đã xóa cấu trúc {structure_id} khỏi hiển thị 3D")

        except Exception as e:
            logger.error(f"Lỗi khi xóa cấu trúc {structure_id}: {str(e)}")

    def clear(self):
        """Xóa tất cả các cấu trúc khỏi hiển thị 3D."""
        if not VTK_AVAILABLE:
            return

        try:
            # Xóa tất cả các actor khỏi renderer
            for actor in self.structure_actors.values():
                self.renderer.RemoveActor(actor)

            # Xóa tất cả dictionaries
            self.structure_actors = {}
            self.structures = {}
            self.structure_colors = {}
            self.structure_opacities = {}

            # Render lại
            self.vtk_widget.GetRenderWindow().Render()
            logger.info("Đã xóa tất cả cấu trúc khỏi hiển thị 3D")

        except Exception as e:
            logger.error(f"Lỗi khi xóa tất cả cấu trúc: {str(e)}")

    def reset_camera(self):
        """Đặt lại camera về vị trí mặc định."""
        if not VTK_AVAILABLE:
            return

        try:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            logger.error(f"Lỗi khi reset camera: {str(e)}")

    def set_display_mode(self, mode):
        """
        Thiết lập chế độ hiển thị.

        Parameters
        ----------
        mode : str
            Chế độ hiển thị: 'surface', 'wireframe', 'surface_wireframe', 'points'
        """
        if not VTK_AVAILABLE:
            return

        try:
            self.display_mode = mode

            # Áp dụng chế độ hiển thị cho tất cả actor
            for actor in self.structure_actors.values():
                if mode == "wireframe":
                    actor.GetProperty().SetRepresentationToWireframe()
                elif mode == "surface":
                    actor.GetProperty().SetRepresentationToSurface()
                elif mode == "surface_wireframe":
                    actor.GetProperty().SetRepresentationToSurface()
                    actor.GetProperty().EdgeVisibilityOn()
                    actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
                    actor.GetProperty().SetLineWidth(1.0)
                elif mode == "points":
                    actor.GetProperty().SetRepresentationToPoints()
                    actor.GetProperty().SetPointSize(3)

            # Render lại
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập chế độ hiển thị {mode}: {str(e)}")

    def _on_view_type_changed(self, index):
        """Xử lý khi thay đổi kiểu hiển thị."""
        if index == 0:
            self.set_display_mode("surface")
        elif index == 1:
            self.set_display_mode("wireframe")
        elif index == 2:
            self.set_display_mode("surface_wireframe")
        elif index == 3:
            self.set_display_mode("points")

    def _on_standard_view_changed(self, index):
        """Xử lý khi chọn góc nhìn tiêu chuẩn."""
        if not VTK_AVAILABLE:
            return

        try:
            view_name = self.standard_views.currentText()
            camera = self.renderer.GetActiveCamera()

            if view_name == "Anterior":
                camera.SetPosition(0, -1000, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "Posterior":
                camera.SetPosition(0, 1000, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "Left":
                camera.SetPosition(-1000, 0, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "Right":
                camera.SetPosition(1000, 0, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "Superior":
                camera.SetPosition(0, 0, 1000)
                camera.SetViewUp(0, 1, 0)
            elif view_name == "Inferior":
                camera.SetPosition(0, 0, -1000)
                camera.SetViewUp(0, 1, 0)

            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập góc nhìn tiêu chuẩn: {str(e)}")

    def _on_opacity_changed(self, value):
        """Xử lý khi thay đổi độ trong suốt chung."""
        opacity = value / 100.0

        try:
            # Áp dụng độ trong suốt cho tất cả actor
            for struct_id, actor in self.structure_actors.items():
                # Tính độ trong suốt tương đối với giá trị đã lưu
                saved_opacity = self.structure_opacities.get(struct_id, 0.8)
                relative_opacity = saved_opacity * opacity

                actor.GetProperty().SetOpacity(relative_opacity)

            # Render lại
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            logger.error(f"Lỗi khi thay đổi độ trong suốt: {str(e)}")

    def update_view(self):
        """Cập nhật hiển thị."""
        if not VTK_AVAILABLE:
            return

        try:
            self.vtk_widget.GetRenderWindow().Render()

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị: {str(e)}")

    def export_image(self, filename, width=1920, height=1080):
        """
        Xuất hình ảnh hiển thị hiện tại.

        Parameters
        ----------
        filename : str
            Đường dẫn tệp tin để lưu hình ảnh
        width : int
            Chiều rộng hình ảnh
        height : int
            Chiều cao hình ảnh
        """
        if not VTK_AVAILABLE:
            logger.warning("Không thể xuất ảnh vì thiếu thư viện VTK")
            return False

        try:
            # Tạo window to image filter để chụp màn hình
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.vtk_widget.GetRenderWindow())
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()

            # Xác định định dạng xuất dựa trên phần mở rộng tệp
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext == ".png":
                writer = vtk.vtkPNGWriter()
            elif ext == ".jpg" or ext == ".jpeg":
                writer = vtk.vtkJPEGWriter()
            elif ext == ".tif" or ext == ".tiff":
                writer = vtk.vtkTIFFWriter()
            else:
                # Mặc định là PNG
                writer = vtk.vtkPNGWriter()
                if not filename.endswith(".png"):
                    filename += ".png"

            # Thiết lập và ghi file
            writer.SetFileName(filename)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()

            logger.info(f"Đã xuất ảnh thành công: {filename}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xuất ảnh: {str(e)}")
            return False

    def closeEvent(self, event):
        """Xử lý sự kiện đóng widget."""
        if hasattr(self, "vtk_widget") and hasattr(self.vtk_widget, "GetRenderWindow"):
            if self.vtk_widget.GetRenderWindow() is not None:
                self.vtk_widget.GetRenderWindow().Finalize()

        if hasattr(self, "interactor") and self.interactor is not None:
            self.interactor.TerminateApp()

        super().closeEvent(event)
