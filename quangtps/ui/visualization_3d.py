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
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple

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

    # Kiểm tra các lớp quan trọng
    test_classes = [
        "vtkRenderer",
        "vtkRenderWindow",
        "vtkRenderWindowInteractor",
        "vtkInteractorStyleTrackballCamera",
        "vtkCellPicker",
        "vtkImageData",
        "vtkMarchingCubes",
        "vtkPolyData",
        "vtkSmoothPolyDataFilter",
        "vtkDecimatePro",
        "vtkPolyDataNormals",
        "vtkPolyDataMapper",
        "vtkActor",
    ]

    for cls_name in test_classes:
        if not hasattr(vtk, cls_name):
            logger.warning(f"VTK không có lớp {cls_name}")
            VTK_AVAILABLE = False
            break
except ImportError as e:
    logger.warning(f"Không thể import VTK: {e}")
    VTK_AVAILABLE = False

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
    logger.info("Đã import PyVista thành công")
except ImportError as e:
    logger.warning(f"Không thể import PyVista: {e}")
    PYVISTA_AVAILABLE = False

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QComboBox,
        QLabel,
        QSlider,
        QDialog,
        QColorDialog,
        QFileDialog,
        QMessageBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize

    if VTK_AVAILABLE:
        try:
            from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

            PYQT_VTK_AVAILABLE = True
            logger.info("Đã import QVTKRenderWindowInteractor thành công")
        except ImportError as e:
            logger.warning(f"Không thể import QVTKRenderWindowInteractor: {e}")
            PYQT_VTK_AVAILABLE = False
except ImportError as e:
    logger.warning(f"Không thể import PyQt5: {e}")
    PYQT_VTK_AVAILABLE = False

    # Tạo lớp giả cho QWidget
    class QWidget:
        def __init__(self, parent=None):
            pass

    # Tạo lớp giả cho pyqtSignal
    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass


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

        # Các đối tượng VTK
        self.vtk_widget = None
        self.renderer = None
        self.interactor = None
        self.interactor_style = None
        self.picker = None

        # Thiết lập UI
        self.init_ui()

    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Nếu không có thư viện VTK hoặc PyQt-VTK, hiển thị thông báo
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

        try:
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
            self.display_mode = (
                "surface"  # surface, wireframe, surface_wireframe, points
            )

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
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo VTK widget: {str(e)}")
            logger.error(traceback.format_exc())
            label = QLabel(f"Lỗi khởi tạo VTK: {str(e)}")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self.setLayout(layout)

    def setup_picker_callback(self):
        """Thiết lập callback cho picker để xử lý sự kiện click chuột."""
        if not (VTK_AVAILABLE and PYQT_VTK_AVAILABLE) or not self.interactor:
            return

        def on_click(obj, event):
            try:
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
            except Exception as e:
                logger.error(f"Lỗi trong xử lý sự kiện click: {str(e)}")

        # Thêm observer cho sự kiện click chuột
        if hasattr(self.interactor, "AddObserver"):
            click_observer = self.interactor.AddObserver(
                "LeftButtonPressEvent", on_click
            )

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
            if actor is None:
                logger.warning(f"Không thể tạo actor cho cấu trúc {structure_id}")
                return

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

                    # Kiểm tra loại cấu trúc
                    structure_type = getattr(structure, "type", "").lower()

                    # Mặc định là màu xanh dương
                    color = default_colors.get(structure_type, (0.2, 0.6, 0.8))
                    self.set_structure_color(structure_id, color)

            # Thiết lập độ trong suốt mặc định
            self.set_structure_opacity(structure_id, 0.8)

            # Cập nhật view
            self.update_view()

        except Exception as e:
            logger.error(
                f"Lỗi khi thêm cấu trúc {getattr(structure, 'id', 'unknown')}: {str(e)}"
            )
            logger.error(traceback.format_exc())

    def _create_mesh_from_mask(self, mask_3d):
        """Tạo mesh 3D từ mask numpy 3D."""
        if not VTK_AVAILABLE:
            logger.warning("Không thể tạo mesh vì thiếu VTK")
            return None

        try:
            # Kiểm tra mask 3D
            if (
                mask_3d is None
                or not isinstance(mask_3d, np.ndarray)
                or mask_3d.ndim != 3
            ):
                logger.warning(f"Mask không hợp lệ: {type(mask_3d)}")
                return None

            # Chuyển đổi mask thành vtk image data
            dims = mask_3d.shape
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(dims[2], dims[1], dims[0])
            vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

            # Chuyển đổi mask từ numpy sang vtk
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        value = 255 if mask_3d[i, j, k] else 0
                        vtk_image.SetScalarComponentFromDouble(k, j, i, 0, value)

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
            smoother.FeatureEdgeSmoothingOff()
            smoother.BoundarySmoothingOn()
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
            normals.ComputeCellNormalsOn()
            normals.ConsistencyOn()
            normals.SplittingOff()
            normals.Update()

            # Trả về mesh cuối cùng
            result_mesh = normals.GetOutput()
            return result_mesh

        except Exception as e:
            logger.error(f"Lỗi khi tạo mesh từ mask: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _add_mesh_to_renderer(self, mesh):
        """Thêm mesh vào renderer và trả về actor."""
        if not VTK_AVAILABLE or not mesh:
            return None

        try:
            # Tạo mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)

            # Tạo actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Thiết lập thuộc tính bề mặt
            actor.GetProperty().SetInterpolationToPhong()
            actor.GetProperty().SetAmbient(0.1)
            actor.GetProperty().SetDiffuse(0.7)
            actor.GetProperty().SetSpecular(0.2)
            actor.GetProperty().SetSpecularPower(10.0)

            # Thêm actor vào renderer
            if self.renderer:
                self.renderer.AddActor(actor)
                self.renderer.ResetCamera()

            return actor

        except Exception as e:
            logger.error(f"Lỗi khi thêm mesh vào renderer: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def set_structure_color(self, structure_id, color):
        """Thiết lập màu cho cấu trúc."""
        if not VTK_AVAILABLE or structure_id not in self.structure_actors:
            return

        try:
            # Chuyển đổi màu từ nhiều định dạng sang tuple RGB
            if isinstance(color, str):
                # Màu dạng hex
                if color.startswith("#"):
                    color = color[1:]
                    r = int(color[0:2], 16) / 255.0
                    g = int(color[2:4], 16) / 255.0
                    b = int(color[4:6], 16) / 255.0
                    color = (r, g, b)
            elif isinstance(color, (list, tuple)):
                # Chuyển từ 0-255 sang 0-1 nếu cần
                if any(c > 1.0 for c in color[:3]):
                    color = tuple(c / 255.0 for c in color[:3])
            else:
                # Màu không hợp lệ, sử dụng màu mặc định
                color = (0.0, 0.7, 0.9)  # Xanh dương

            # Lưu màu
            self.structure_colors[structure_id] = color

            # Thiết lập màu cho actor
            actor = self.structure_actors[structure_id]
            actor.GetProperty().SetColor(color[0], color[1], color[2])

            # Cập nhật view
            self.update_view()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập màu cho cấu trúc {structure_id}: {str(e)}")

    def set_structure_opacity(self, structure_id, opacity):
        """Thiết lập độ trong suốt cho cấu trúc."""
        if not VTK_AVAILABLE or structure_id not in self.structure_actors:
            return

        try:
            # Đảm bảo opacity trong khoảng [0, 1]
            opacity = max(0.0, min(1.0, float(opacity)))

            # Lưu giá trị độ trong suốt
            self.structure_opacities[structure_id] = opacity

            # Thiết lập độ trong suốt cho actor
            actor = self.structure_actors[structure_id]
            actor.GetProperty().SetOpacity(opacity)

            # Cập nhật view
            self.update_view()

        except Exception as e:
            logger.error(
                f"Lỗi khi thiết lập độ trong suốt cho cấu trúc {structure_id}: {str(e)}"
            )

    def remove_structure(self, structure_id):
        """Xóa cấu trúc khỏi hiển thị."""
        if not VTK_AVAILABLE or structure_id not in self.structure_actors:
            return

        try:
            # Lấy actor từ dictionary
            actor = self.structure_actors[structure_id]

            # Xóa actor khỏi renderer
            if self.renderer:
                self.renderer.RemoveActor(actor)

            # Xóa khỏi các dictionaries
            if structure_id in self.structure_actors:
                del self.structure_actors[structure_id]
            if structure_id in self.structures:
                del self.structures[structure_id]
            if structure_id in self.structure_colors:
                del self.structure_colors[structure_id]
            if structure_id in self.structure_opacities:
                del self.structure_opacities[structure_id]

            # Cập nhật view
            self.update_view()

        except Exception as e:
            logger.error(f"Lỗi khi xóa cấu trúc {structure_id}: {str(e)}")

    def clear(self):
        """Xóa tất cả cấu trúc khỏi hiển thị."""
        if not VTK_AVAILABLE or not self.renderer:
            return

        try:
            # Xóa tất cả actor khỏi renderer
            for actor in self.structure_actors.values():
                self.renderer.RemoveActor(actor)

            # Xóa các dictionaries
            self.structure_actors.clear()
            self.structures.clear()
            self.structure_colors.clear()
            self.structure_opacities.clear()

            # Cập nhật view
            self.update_view()

        except Exception as e:
            logger.error(f"Lỗi khi xóa tất cả cấu trúc: {str(e)}")

    def reset_camera(self):
        """Đặt lại góc nhìn camera về mặc định."""
        if not VTK_AVAILABLE or not self.renderer:
            return

        try:
            self.renderer.ResetCamera()
            self.update_view()

        except Exception as e:
            logger.error(f"Lỗi khi đặt lại camera: {str(e)}")

    def set_display_mode(self, mode):
        """
        Thiết lập chế độ hiển thị cho tất cả cấu trúc.

        Parameters
        ----------
        mode : str
            Chế độ hiển thị: "surface", "wireframe", "surface_wireframe", hoặc "points"
        """
        if not VTK_AVAILABLE:
            return

        try:
            # Lưu chế độ hiển thị
            self.display_mode = mode

            # Thiết lập chế độ hiển thị cho tất cả actor
            for actor in self.structure_actors.values():
                if mode == "wireframe":
                    actor.GetProperty().SetRepresentationToWireframe()
                elif mode == "points":
                    actor.GetProperty().SetRepresentationToPoints()
                elif mode == "surface_wireframe":
                    actor.GetProperty().SetRepresentationToSurface()
                    actor.GetProperty().EdgeVisibilityOn()
                    actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)  # Đen
                else:  # surface
                    actor.GetProperty().SetRepresentationToSurface()
                    actor.GetProperty().EdgeVisibilityOff()

            # Cập nhật view
            self.update_view()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập chế độ hiển thị {mode}: {str(e)}")

    def _on_view_type_changed(self, index):
        """Xử lý khi người dùng thay đổi loại hiển thị."""
        modes = ["surface", "wireframe", "surface_wireframe", "points"]
        if index < len(modes):
            self.set_display_mode(modes[index])

    def _on_standard_view_changed(self, index):
        """Thiết lập góc nhìn tiêu chuẩn."""
        if not VTK_AVAILABLE or not self.renderer:
            return

        try:
            camera = self.renderer.GetActiveCamera()
            if not camera:
                return

            # Đặt lại camera
            self.renderer.ResetCamera()

            # Thiết lập vị trí và hướng camera dựa trên góc nhìn
            view_name = self.standard_views.currentText().lower()
            if view_name == "anterior":
                camera.SetPosition(0, -1, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "posterior":
                camera.SetPosition(0, 1, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "left":
                camera.SetPosition(-1, 0, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "right":
                camera.SetPosition(1, 0, 0)
                camera.SetViewUp(0, 0, 1)
            elif view_name == "superior":
                camera.SetPosition(0, 0, 1)
                camera.SetViewUp(0, 1, 0)
            elif view_name == "inferior":
                camera.SetPosition(0, 0, -1)
                camera.SetViewUp(0, 1, 0)

            # Cập nhật view
            self.update_view()

            # Phát tín hiệu thay đổi góc nhìn
            self.viewChanged.emit({"view": view_name})

        except Exception as e:
            logger.error(f"Lỗi khi thay đổi góc nhìn: {str(e)}")

    def _on_opacity_changed(self, value):
        """Xử lý khi người dùng thay đổi độ trong suốt."""
        if not VTK_AVAILABLE:
            return

        try:
            # Chuyển đổi giá trị từ 10-100 sang 0.1-1.0
            opacity = value / 100.0

            # Thiết lập độ trong suốt cho tất cả cấu trúc
            for structure_id in self.structure_actors:
                self.set_structure_opacity(structure_id, opacity)

            # Cập nhật view
            self.update_view()

        except Exception as e:
            logger.error(f"Lỗi khi thay đổi độ trong suốt: {str(e)}")

    def update_view(self):
        """Cập nhật hiển thị."""
        if not VTK_AVAILABLE or not self.vtk_widget:
            return

        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật view: {str(e)}")

    def export_image(self, filename, width=1920, height=1080):
        """
        Xuất ảnh hiện tại thành file.

        Parameters
        ----------
        filename : str
            Đường dẫn đến file ảnh cần lưu (hỗ trợ .png, .jpg, .tif)
        width : int, optional
            Chiều rộng của ảnh (pixel)
        height : int, optional
            Chiều cao của ảnh (pixel)
        """
        if not VTK_AVAILABLE or not self.vtk_widget:
            logger.warning("Không thể xuất ảnh vì thiếu VTK")
            return

        try:
            # Tạo window to image filter để chụp màn hình
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.vtk_widget.GetRenderWindow())
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()

            # Lấy đuôi file
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            # Chọn writer dựa trên đuôi file
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

            # Thiết lập và lưu ảnh
            writer.SetFileName(filename)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()

            logger.info(f"Đã xuất ảnh thành: {filename}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xuất ảnh: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def closeEvent(self, event):
        """Xử lý khi đóng widget."""
        if VTK_AVAILABLE and hasattr(self, "vtk_widget") and self.vtk_widget:
            try:
                # Tắt interactor để tránh lỗi khi đóng widget
                self.vtk_widget.GetRenderWindow().Finalize()
                self.vtk_widget.close()
            except Exception as e:
                logger.debug(f"Lỗi khi đóng VTK widget: {str(e)}")

        # Gọi hàm closeEvent của lớp cha
        super().closeEvent(event)
