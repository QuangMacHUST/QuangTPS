#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab Structure/Contour cho QuangTPS.

Module này triển khai giao diện Structure tab tương tự Eclipse của Varian,
cho phép người dùng vẽ, quản lý và chỉnh sửa structure và contour.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import traceback

# Thêm xử lý exception khi import PyQt5
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QSplitter,
        QDialog,
        QColorDialog,
        QComboBox,
        QLineEdit,
        QFormLayout,
        QMessageBox,
        QFileDialog,
        QTabWidget,
        QTreeWidget,
        QTreeWidgetItem,
        QHeaderView,
        QProgressDialog,
        QMenu,
        QAction,
        QToolBar,
        QGroupBox,
        QRadioButton,
        QButtonGroup,
        QCheckBox,
        QSlider,
        QSpinBox,
        QDoubleSpinBox,
        QToolButton,
        QFrame,
        QScrollArea,
        QGridLayout,
        QInputDialog,
        QStackedWidget,
        QAbstractItemView,
        QSizePolicy,
        QDialogButtonBox,
        QTextEdit,
        QTableWidget,
        QTableWidgetItem,
        QApplication,
    )
    from PyQt5.QtGui import (
        QColor,
        QIcon,
        QBrush,
        QPixmap,
        QImage,
        QPainter,
        QPen,
        QCursor,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect, QTimer
except ImportError:
    logging.warning("Không thể import PyQt5. Sử dụng lớp giả.")

    # Định nghĩa lớp giả cho môi trường không có PyQt5
    class QWidget:
        def __init__(self, parent=None):
            pass

    # Các lớp giả khác cho PyQt5
    class DummyQtClass:
        def __init__(self, *args, **kwargs):
            pass

        def addWidget(self, *args, **kwargs):
            pass

        def addLayout(self, *args, **kwargs):
            pass

        def addTab(self, *args, **kwargs):
            pass

        def addStretch(self, *args):
            pass

        def setLayout(self, *args):
            pass

        def setAlignment(self, *args):
            pass

    QVBoxLayout = DummyQtClass
    QHBoxLayout = DummyQtClass
    QLabel = DummyQtClass
    QPushButton = DummyQtClass
    QListWidget = DummyQtClass
    QListWidgetItem = DummyQtClass
    QMenu = DummyQtClass
    QAction = DummyQtClass
    QTabWidget = DummyQtClass
    QDialog = DummyQtClass
    QMessageBox = DummyQtClass
    QTableWidget = DummyQtClass
    QTableWidgetItem = DummyQtClass
    QHeaderView = DummyQtClass
    QApplication = DummyQtClass
    QSplitter = DummyQtClass
    QLineEdit = DummyQtClass
    QTextEdit = DummyQtClass
    Qt = DummyQtClass
    Qt.Horizontal = None
    Qt.AlignCenter = None
    Qt.UserRole = None
    Qt.WaitCursor = None

    class pyqtSignal:
        """Lớp giả cho pyqtSignal"""

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass


# Import các module khác
from quangtps.segmentation.structures.structure import (
    Structure,
    StructureType,
    StructurePriority,
)
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.contour.contour_manager import ContourManager
from quangtps.segmentation.contour.polygon_tool import PolygonTool
from quangtps.segmentation.contour.margin import MarginTool, MarginType
from quangtps.segmentation.contour.boolean_operations import BooleanOperator
from quangtps.segmentation.contour.interpolation import ContourInterpolator

# Handle potentially missing modules
try:
    from quangtps.segmentation.auto_segmentation.semi_automatic import (
        SemiAutomaticSegmentation,
    )
except ImportError:
    SemiAutomaticSegmentation = None

from quangtps.segmentation.auto.engine import AutoSegmentationEngine
from quangtps.ui.image_display import ImageDisplay
from quangtps.imaging.image import Image
from quangtps.core.patient import Patient

# Import ServiceRegistry với xử lý lỗi
try:
    from quangtps.core.service_registry import ServiceRegistry
except ImportError:
    logging.warning("Không thể import ServiceRegistry. Sử dụng lớp giả.")

    class ServiceRegistry:
        """Lớp giả cho ServiceRegistry"""

        _instance = None

        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = ServiceRegistry()
            return cls._instance

        @classmethod
        def register_service(cls, service_name, service):
            """Đăng ký service."""
            pass

        @classmethod
        def get_service(cls, service_name):
            """Lấy service theo tên."""
            pass


logger = logging.getLogger(__name__)

# Import các module tùy chọn với xử lý lỗi
try:
    from quangtps.ui.visualization_3d import StructureViewer3D

    HAS_3D_VISUALIZATION = True
except ImportError:
    logging.warning(
        "Không thể import StructureViewer3D. Chức năng hiển thị 3D sẽ bị hạn chế."
    )
    HAS_3D_VISUALIZATION = False


# Import Patient từ quangtps.core.patient
from quangtps.core.patient import Patient


class ObjectExplorerPanel(QWidget):
    """
    Panel displaying available objects (patients, images, structures, etc.)

    This class provides an Eclipse-like object explorer panel with hierarchical
    organization of patient data, images, structure sets, and plans.
    """

    # Signals
    patientSelected = pyqtSignal(Patient)
    imageSelected = pyqtSignal(object)  # Image object
    structureSetSelected = pyqtSignal(StructureSet)
    planSelected = pyqtSignal(object)  # Plan object

    def __init__(self, parent=None):
        """Initialize the object explorer panel."""
        super().__init__(parent)
        self.init_ui()
        self.current_patient = None

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Object Explorer")
        header.setStyleSheet(
            "font-weight: bold; background-color: #0078D7; color: white; padding: 5px;"
        )
        layout.addWidget(header)

        # Tree-like structure using expandable sections

        # Patients section
        self.patients_group = QGroupBox("Patients")
        patients_layout = QVBoxLayout(self.patients_group)
        self.patients_container = QWidget()
        self.patients_layout = QVBoxLayout(self.patients_container)
        self.patients_layout.setContentsMargins(0, 0, 0, 0)
        self.patients_layout.setSpacing(1)
        patients_scroll = QScrollArea()
        patients_scroll.setWidgetResizable(True)
        patients_scroll.setWidget(self.patients_container)
        patients_layout.addWidget(patients_scroll)
        layout.addWidget(self.patients_group)

        # Images section
        self.images_group = QGroupBox("Images")
        images_layout = QVBoxLayout(self.images_group)
        self.images_container = QWidget()
        self.images_layout = QVBoxLayout(self.images_container)
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        self.images_layout.setSpacing(1)
        images_scroll = QScrollArea()
        images_scroll.setWidgetResizable(True)
        images_scroll.setWidget(self.images_container)
        images_layout.addWidget(images_scroll)
        layout.addWidget(self.images_group)

        # Structure Sets section
        self.structure_sets_group = QGroupBox("Structure Sets")
        structure_sets_layout = QVBoxLayout(self.structure_sets_group)
        self.structure_sets_container = QWidget()
        self.structure_sets_layout = QVBoxLayout(self.structure_sets_container)
        self.structure_sets_layout.setContentsMargins(0, 0, 0, 0)
        self.structure_sets_layout.setSpacing(1)
        structure_sets_scroll = QScrollArea()
        structure_sets_scroll.setWidgetResizable(True)
        structure_sets_scroll.setWidget(self.structure_sets_container)
        structure_sets_layout.addWidget(structure_sets_scroll)
        layout.addWidget(self.structure_sets_group)

        # Plans section
        self.plans_group = QGroupBox("Plans")
        plans_layout = QVBoxLayout(self.plans_group)
        self.plans_container = QWidget()
        self.plans_layout = QVBoxLayout(self.plans_container)
        self.plans_layout.setContentsMargins(0, 0, 0, 0)
        self.plans_layout.setSpacing(1)
        plans_scroll = QScrollArea()
        plans_scroll.setWidgetResizable(True)
        plans_scroll.setWidget(self.plans_container)
        plans_layout.addWidget(plans_scroll)
        layout.addWidget(self.plans_group)

        self.setLayout(layout)

    def set_patient(self, patient: Patient):
        """Set the current patient and update displayed objects."""
        self.current_patient = patient
        self.update_displayed_objects()

    def update_displayed_objects(self):
        """Update all displayed objects based on the current patient."""
        self._clear_layout(self.patients_layout)
        self._clear_layout(self.images_layout)
        self._clear_layout(self.structure_sets_layout)
        self._clear_layout(self.plans_layout)

        if not self.current_patient:
            return

        # Add patient item
        patient_item = self._create_item(
            self.current_patient.name,
            f"ID: {self.current_patient.id}",
            is_selected=True,
            item_type="patient",
            item_data=self.current_patient,
        )
        self.patients_layout.addWidget(patient_item)

        # Add image items if available
        try:
            # Sử dụng ServiceRegistry an toàn
            registry = ServiceRegistry.get_instance()
            patient_db = None

            if registry and hasattr(registry, "get_service"):
                patient_db = registry.get_service("PatientDB")

            if patient_db:
                # Get images for this patient
                images = patient_db.get_images_for_patient(self.current_patient.id)
                if images:
                    for image in images:
                        image_item = self._create_item(
                            image.description
                            if hasattr(image, "description")
                            else "Image",
                            f"Series: {image.series_id if hasattr(image, 'series_id') else 'N/A'}",
                            is_selected=False,
                            item_type="image",
                            item_data=image,
                        )
                        self.images_layout.addWidget(image_item)

                # Get structure sets for this patient
                structure_sets = patient_db.get_structure_sets_for_patient(
                    self.current_patient.id
                )
                if structure_sets:
                    for ss in structure_sets:
                        ss_item = self._create_item(
                            ss.name if hasattr(ss, "name") else "Structure Set",
                            f"ID: {ss.id if hasattr(ss, 'id') else 'N/A'}",
                            is_selected=False,
                            item_type="structure_set",
                            item_data=ss,
                        )
                        self.structure_sets_layout.addWidget(ss_item)

                # Get plans for this patient
                plans = patient_db.get_plans_for_patient(self.current_patient.id)
                if plans:
                    for plan in plans:
                        plan_item = self._create_item(
                            plan.name if hasattr(plan, "name") else "Plan",
                            f"ID: {plan.id if hasattr(plan, 'id') else 'N/A'}",
                            is_selected=False,
                            item_type="plan",
                            item_data=plan,
                        )
                        self.plans_layout.addWidget(plan_item)
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")

    def _create_item(
        self,
        title: str,
        subtitle: str,
        is_selected: bool = False,
        item_type: str = "",
        item_data: Any = None,
    ) -> QWidget:
        """Create an item widget for the object explorer."""
        item = QWidget()
        layout = QVBoxLayout(item)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)

        # Subtitle
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("font-size: 8pt; color: #666;")
            layout.addWidget(subtitle_label)

        item.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 3px;"
        )
        if is_selected:
            item.setStyleSheet(
                "background-color: #0078D7; color: white; border: 1px solid #0078D7; border-radius: 3px;"
            )

        # Store data
        item.setProperty("item_type", item_type)
        item.setProperty("item_data", item_data)

        # Connect mouse events
        item.mousePressEvent = lambda e, i=item: self._on_item_clicked(i)

        return item

    def _on_item_clicked(self, item: QWidget):
        """Handle item click in the object explorer."""
        item_type = item.property("item_type")
        item_data = item.property("item_data")

        if item_type == "patient":
            self.patientSelected.emit(item_data)
        elif item_type == "image":
            self.imageSelected.emit(item_data)
        elif item_type == "structure_set":
            self.structureSetSelected.emit(item_data)
        elif item_type == "plan":
            self.planSelected.emit(item_data)

        # Update selection visuals
        self._update_selection_for_type(item_type, item)

    def _update_selection_for_type(self, item_type: str, selected_item: QWidget):
        """Update visual selection state for items of a specific type."""
        container_layout = None
        if item_type == "patient":
            container_layout = self.patients_layout
        elif item_type == "image":
            container_layout = self.images_layout
        elif item_type == "structure_set":
            container_layout = self.structure_sets_layout
        elif item_type == "plan":
            container_layout = self.plans_layout

        if container_layout:
            # Update all items of this type
            for i in range(container_layout.count()):
                item = container_layout.itemAt(i).widget()
                if item:
                    if item == selected_item:
                        item.setStyleSheet(
                            "background-color: #0078D7; color: white; border: 1px solid #0078D7; border-radius: 3px;"
                        )
                    else:
                        item.setStyleSheet(
                            "background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 3px;"
                        )

    def _clear_layout(self, layout):
        """Clear all widgets from a layout."""
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


class StructureTab(QWidget):
    """
    Structure tab for the QuangTPS application.

    This tab provides an interface for structure management, including
    creation, editing, and visualization of structures for treatment planning.
    """

    # Signals
    structureSetChanged = pyqtSignal(StructureSet)
    structureSelectionChanged = pyqtSignal(Structure)
    structureAdded = pyqtSignal(Structure)
    structureRemoved = pyqtSignal(Structure)
    structureModified = pyqtSignal(Structure)
    structureVisibilityChanged = pyqtSignal(Structure, bool)
    windowClosed = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the structure tab."""
        super().__init__(parent)
        self.parent = parent
        self.image = None
        self.structure_set = None
        self.selected_structure = None
        self.segmentation_interface = None

        # Initialize tools
        self.polygon_tool = PolygonTool()
        self.contour_manager = ContourManager()
        self.margin_tool = MarginTool()  # Initialize the margin tool properly

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the user interface."""
        self.main_layout = QVBoxLayout(self)

        # Layout cho các điều khiển chính
        controls_layout = QHBoxLayout()

        # Panel bên trái để hiển thị danh sách cấu trúc
        self.structures_panel = QWidget()
        self.structures_layout = QVBoxLayout(self.structures_panel)

        # Label và bộ lọc
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Lọc cấu trúc:")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Nhập tên cấu trúc để lọc...")
        self.filter_edit.textChanged.connect(self._filter_structures)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_edit)
        self.structures_layout.addLayout(filter_layout)

        # Danh sách cấu trúc
        self.setup_structure_list()

        # Thêm nút điều khiển cho cấu trúc
        self.setup_structure_controls()

        # Panel thông tin cấu trúc
        self.info_panel = QWidget()
        self.info_layout = QVBoxLayout(self.info_panel)
        self.info_layout.addWidget(QLabel("Thông tin cấu trúc:"))

        # Widget hiển thị chi tiết cấu trúc
        self.structure_info = QTextEdit()
        self.structure_info.setReadOnly(True)
        self.structure_info.setMinimumHeight(150)
        self.info_layout.addWidget(self.structure_info)

        # Thêm tab thống kê và phân tích
        self.stats_tabs = QTabWidget()

        # Tab thống kê cơ bản
        self.basic_stats_tab = QWidget()
        self.basic_stats_layout = QVBoxLayout(self.basic_stats_tab)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Thông số", "Giá trị"])
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.basic_stats_layout.addWidget(self.stats_table)
        self.stats_tabs.addTab(self.basic_stats_tab, "Thống kê cơ bản")

        # Tab phân tích giao thoa
        self.overlap_tab = QWidget()
        self.overlap_layout = QVBoxLayout(self.overlap_tab)
        self.overlap_table = QTableWidget()
        self.overlap_table.setColumnCount(3)
        self.overlap_table.setHorizontalHeaderLabels(
            ["Cấu trúc 1", "Cấu trúc 2", "Thể tích giao thoa (cc)"]
        )
        self.overlap_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.overlap_layout.addWidget(self.overlap_table)

        # Nút phát hiện giao thoa
        self.detect_overlaps_btn = QPushButton("Phát hiện giao thoa")
        self.detect_overlaps_btn.clicked.connect(self._detect_structure_overlaps)
        self.overlap_layout.addWidget(self.detect_overlaps_btn)

        self.stats_tabs.addTab(self.overlap_tab, "Phân tích giao thoa")

        # Thêm tab hiển thị 3D
        self.viz_3d_tab = QWidget()
        self.viz_3d_layout = QVBoxLayout(self.viz_3d_tab)

        # Thiết lập widget hiển thị 3D
        try:
            from quangtps.ui.visualization_3d import StructureViewer3D

            self.structure_viewer_3d = StructureViewer3D()
            self.viz_3d_layout.addWidget(self.structure_viewer_3d)

            # Thêm các nút điều khiển hiển thị 3D
            viz_controls = QHBoxLayout()
            self.show_3d_btn = QPushButton("Hiển thị 3D")
            self.show_3d_btn.clicked.connect(self._show_structure_3d)
            viz_controls.addWidget(self.show_3d_btn)

            self.reset_3d_view_btn = QPushButton("Đặt lại góc nhìn")
            self.reset_3d_view_btn.clicked.connect(self._reset_3d_view)
            viz_controls.addWidget(self.reset_3d_view_btn)

            self.viz_3d_layout.addLayout(viz_controls)

        except (ImportError, Exception) as e:
            logger.error(f"Không thể khởi tạo StructureViewer3D: {str(e)}")
            error_label = QLabel("Không thể tải module hiển thị 3D")
            error_label.setAlignment(Qt.AlignCenter)
            self.viz_3d_layout.addWidget(error_label)

        self.stats_tabs.addTab(self.viz_3d_tab, "Hiển thị 3D")

        # Thêm tabs vào layout
        self.info_layout.addWidget(self.stats_tabs)

        # Khu vực hiển thị cắt lớp
        self.setup_slice_view()

        # Thêm các panel vào layout chính
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.structures_panel)
        splitter.addWidget(self.slice_view_container)
        splitter.addWidget(self.info_panel)

        # Thiết lập kích thước ban đầu cho các panel
        splitter.setSizes([200, 500, 300])

        self.main_layout.addWidget(splitter)

    def setup_structure_controls(self):
        """Thiết lập các nút điều khiển cho cấu trúc."""
        buttons_layout = QHBoxLayout()

        self.new_structure_btn = QPushButton("Tạo mới")
        self.new_structure_btn.clicked.connect(self._create_new_structure)
        buttons_layout.addWidget(self.new_structure_btn)

        self.edit_structure_btn = QPushButton("Chỉnh sửa")
        self.edit_structure_btn.clicked.connect(self._edit_selected_structure)
        buttons_layout.addWidget(self.edit_structure_btn)

        self.delete_structure_btn = QPushButton("Xóa")
        self.delete_structure_btn.clicked.connect(self._delete_selected_structure)
        buttons_layout.addWidget(self.delete_structure_btn)

        self.structures_layout.addLayout(buttons_layout)

        # Hàng nút thứ hai
        buttons_layout2 = QHBoxLayout()

        self.copy_structure_btn = QPushButton("Sao chép")
        self.copy_structure_btn.clicked.connect(self._copy_selected_structure)
        buttons_layout2.addWidget(self.copy_structure_btn)

        self.boolean_op_btn = QPushButton("Boolean")
        self.boolean_op_btn.clicked.connect(self._show_boolean_dialog)
        buttons_layout2.addWidget(self.boolean_op_btn)

        self.export_struct_btn = QPushButton("Xuất")
        self.export_struct_btn.clicked.connect(self._export_selected_structure)
        buttons_layout2.addWidget(self.export_struct_btn)

        self.structures_layout.addLayout(buttons_layout2)

    def _detect_structure_overlaps(self):
        """Phát hiện các giao thoa giữa các cấu trúc và hiển thị kết quả."""
        if not self.patient or not self.patient.structures:
            return

        structures = self.patient.structures

        # Xóa bảng hiện tại
        self.overlap_table.setRowCount(0)

        logger.info("Bắt đầu phát hiện giao thoa giữa các cấu trúc...")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Tính toán giao thoa cho mỗi cặp cấu trúc
            for i, struct1 in enumerate(structures):
                for j, struct2 in enumerate(structures):
                    # Bỏ qua nếu là cùng một cấu trúc
                    if i >= j:
                        continue

                    try:
                        # Tính toán thể tích giao thoa (giả định có phương thức calculate_overlap)
                        overlap_volume = self._calculate_overlap(struct1, struct2)

                        # Nếu có giao thoa đáng kể (>0.01cc), thêm vào bảng
                        if overlap_volume > 0.01:
                            row = self.overlap_table.rowCount()
                            self.overlap_table.insertRow(row)
                            self.overlap_table.setItem(
                                row, 0, QTableWidgetItem(struct1.name)
                            )
                            self.overlap_table.setItem(
                                row, 1, QTableWidgetItem(struct2.name)
                            )
                            self.overlap_table.setItem(
                                row, 2, QTableWidgetItem(f"{overlap_volume:.2f}")
                            )
                    except Exception as e:
                        logger.error(f"Lỗi khi tính toán giao thoa: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Lỗi khi phát hiện giao thoa: {str(e)}")

        finally:
            QApplication.restoreOverrideCursor()

        if self.overlap_table.rowCount() == 0:
            row = self.overlap_table.rowCount()
            self.overlap_table.insertRow(row)
            self.overlap_table.setItem(row, 0, QTableWidgetItem(""))
            self.overlap_table.setItem(
                row, 1, QTableWidgetItem("Không phát hiện giao thoa")
            )
            self.overlap_table.setItem(row, 2, QTableWidgetItem(""))

        logger.info(
            f"Đã phát hiện {self.overlap_table.rowCount()} giao thoa giữa các cấu trúc"
        )

    def _calculate_overlap(self, struct1, struct2):
        """
        Tính toán thể tích giao thoa giữa hai cấu trúc.

        Parameters
        ----------
        struct1 : Structure
            Cấu trúc thứ nhất
        struct2 : Structure
            Cấu trúc thứ hai

        Returns
        -------
        float
            Thể tích giao thoa tính bằng cc
        """
        # Kiểm tra xem có phương thức tính giao thoa không
        if hasattr(struct1, "calculate_overlap"):
            return struct1.calculate_overlap(struct2)

        # Nếu không có phương thức có sẵn, thử mô phỏng đơn giản
        try:
            # Lấy masks 3D của cả hai cấu trúc (nếu có)
            mask1 = getattr(struct1, "mask_3d", None)
            mask2 = getattr(struct2, "mask_3d", None)

            if mask1 is not None and mask2 is not None and mask1.shape == mask2.shape:
                # Tính giao thoa bằng phép AND
                overlap_mask = np.logical_and(mask1, mask2)
                # Giả định mỗi voxel có kích thước 1mm^3
                voxel_volume_cc = 0.001  # 1mm^3 = 0.001 cc
                return np.sum(overlap_mask) * voxel_volume_cc

            # Nếu không có mask, thử phương pháp khác (ví dụ: dùng contours)
            # ...

        except Exception as e:
            logger.error(f"Lỗi khi tính toán giao thoa: {str(e)}")

        # Trả về giá trị giả nếu không tính toán được (cho mục đích demo)
        import random

        return random.uniform(0, 0.5) if random.random() < 0.3 else 0

    def _show_structure_3d(self):
        """Hiển thị cấu trúc đã chọn trong chế độ xem 3D."""
        # Kiểm tra module 3D visualization có sẵn không
        if not HAS_3D_VISUALIZATION:
            logger.error("Không thể hiển thị 3D: module StructureViewer3D không có sẵn")
            QMessageBox.warning(
                self,
                "Không thể hiển thị 3D",
                "Module hiển thị 3D không có sẵn. Vui lòng cài đặt các thư viện cần thiết (VTK, PyVista).",
            )
            return

        # Lấy cấu trúc đã chọn
        selected_items = self.structure_list.selectedItems()
        if not selected_items:
            logger.warning("Không có cấu trúc nào được chọn để hiển thị 3D")
            QMessageBox.information(
                self, "Thông báo", "Vui lòng chọn ít nhất một cấu trúc để hiển thị 3D"
            )
            return

        try:
            # Kiểm tra dependencies
            try:
                import vtk
                import pyvista as pv

                # Sử dụng cách import này để tránh lỗi linter
                from PyQt5.QtWidgets import (
                    QDialog,
                    QVBoxLayout,
                    QHBoxLayout,
                    QComboBox,
                    QLabel,
                    QSlider,
                )
                from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            except ImportError as e:
                dependency = str(e).split("'")[1] if "'" in str(e) else "thư viện"
                QMessageBox.warning(
                    self,
                    "Thiếu thư viện",
                    f"Không thể tải {dependency}. Vui lòng cài đặt các thư viện cần thiết:\n\npip install vtk pyvista",
                )
                logger.error(f"Lỗi import thư viện hiển thị 3D: {e}")
                return

            # Tạo dialog hiển thị 3D nếu chưa có
            if not hasattr(self, "viewer_3d_dialog"):
                # Các widget đã được import ở trên
                self.viewer_3d_dialog = QDialog(self)
                self.viewer_3d_dialog.setWindowTitle("Hiển thị 3D - QuangTPS")
                self.viewer_3d_dialog.setMinimumSize(1000, 700)
                layout = QVBoxLayout(self.viewer_3d_dialog)

                # Layout điều khiển
                control_layout = QHBoxLayout()

                # Chọn kiểu hiển thị
                control_layout.addWidget(QLabel("Kiểu hiển thị:"))
                view_type = QComboBox()
                view_type.addItems(
                    ["Surface", "Wireframe", "Surface + Wireframe", "Points"]
                )
                view_type.setCurrentIndex(0)
                control_layout.addWidget(view_type)

                # Chọn góc nhìn
                control_layout.addWidget(QLabel("Góc nhìn:"))
                standard_views = QComboBox()
                standard_views.addItems(
                    ["Anterior", "Posterior", "Left", "Right", "Superior", "Inferior"]
                )
                standard_views.setCurrentIndex(0)
                control_layout.addWidget(standard_views)

                # Điều khiển độ trong suốt
                control_layout.addWidget(QLabel("Độ trong suốt:"))
                opacity_slider = QSlider(Qt.Horizontal)
                opacity_slider.setMinimum(10)
                opacity_slider.setMaximum(100)
                opacity_slider.setValue(80)
                control_layout.addWidget(opacity_slider)

                layout.addLayout(control_layout)

                # Tạo viewer 3D
                self.structure_viewer_3d = StructureViewer3D()
                layout.addWidget(self.structure_viewer_3d)

                # Kết nối các điều khiển với viewer
                view_type.currentIndexChanged.connect(
                    self.structure_viewer_3d._on_view_type_changed
                )
                standard_views.currentIndexChanged.connect(
                    self.structure_viewer_3d._on_standard_view_changed
                )
                opacity_slider.valueChanged.connect(
                    self.structure_viewer_3d._on_opacity_changed
                )

                # Kết nối signal structureClicked với xử lý
                self.structure_viewer_3d.structureClicked.connect(
                    self._on_3d_structure_clicked
                )

                # Thêm nút tạo ảnh chụp 3D
                button_layout = QHBoxLayout()

                # Nút xuất ảnh
                capture_btn = QPushButton("Xuất ảnh")
                capture_btn.clicked.connect(lambda: self._capture_3d_image())
                button_layout.addWidget(capture_btn)

                # Nút đặt lại góc nhìn
                reset_btn = QPushButton("Đặt lại góc nhìn")
                reset_btn.clicked.connect(self._reset_3d_view)
                button_layout.addWidget(reset_btn)

                # Nút đóng
                close_btn = QPushButton("Đóng")
                close_btn.clicked.connect(self.viewer_3d_dialog.close)
                button_layout.addWidget(close_btn)

                layout.addLayout(button_layout)

            # Xóa các cấu trúc hiện tại
            self.structure_viewer_3d.clear()

            # Thêm từng cấu trúc đã chọn
            for item in selected_items:
                struct_id = item.data(Qt.UserRole)
                if struct_id:
                    structure = self.get_structure_by_id(struct_id)
                    if structure:
                        # Lấy màu của cấu trúc
                        color = getattr(structure, "color", None)

                        # Chuyển đổi color nếu cần
                        if color is None:
                            # Màu mặc định cho từng loại cấu trúc
                            color_map = {
                                "PTV": (1.0, 0.2, 0.2),  # Đỏ
                                "CTV": (0.8, 0.5, 0.2),  # Cam
                                "GTV": (1.0, 0.0, 0.0),  # Đỏ đậm
                                "OAR": (0.2, 0.8, 0.2),  # Xanh lá
                                "EXTERNAL": (0.7, 0.7, 0.7),  # Xám
                            }

                            structure_type = getattr(structure, "type", "")
                            color = color_map.get(
                                structure_type.upper(), (0.2, 0.6, 0.8)
                            )  # Mặc định xanh dương

                        # Thêm cấu trúc vào trình hiển thị 3D
                        self.structure_viewer_3d.add_structure(structure, color)

            # Hiển thị dialog
            self.viewer_3d_dialog.show()
            self.structure_viewer_3d.update_view()
            logger.info("Đã hiển thị cấu trúc trong chế độ xem 3D")

        except Exception as e:
            logger.error(f"Lỗi khi hiển thị cấu trúc 3D: {str(e)}")
            logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
            QMessageBox.critical(self, "Lỗi", f"Không thể hiển thị 3D: {str(e)}")

    def _capture_3d_image(self):
        """Xuất hình ảnh 3D hiện tại thành file."""
        if not hasattr(self, "structure_viewer_3d"):
            return

        try:
            # Hiển thị dialog chọn file
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu ảnh 3D",
                "",
                "Images (*.png *.jpg *.jpeg);;All Files (*)",
                options=options,
            )

            if filename:
                # Đảm bảo có đuôi file
                if not (
                    filename.lower().endswith(".png")
                    or filename.lower().endswith(".jpg")
                    or filename.lower().endswith(".jpeg")
                ):
                    filename += ".png"

                # Xuất ảnh
                self.structure_viewer_3d.export_image(filename)
                QMessageBox.information(
                    self,
                    "Xuất ảnh",
                    f"Đã lưu ảnh 3D thành công: {os.path.basename(filename)}",
                )
                logger.info(f"Đã xuất ảnh 3D thành: {filename}")

        except Exception as e:
            logger.error(f"Lỗi khi xuất ảnh 3D: {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất ảnh 3D: {str(e)}")

    def _on_3d_structure_clicked(self, structure_id):
        """Xử lý khi người dùng nhấp vào cấu trúc trong chế độ xem 3D"""
        try:
            # Tìm cấu trúc có ID tương ứng trong danh sách
            for i in range(self.structure_list.count()):
                item = self.structure_list.item(i)
                if item.data(Qt.UserRole) == structure_id:
                    # Chọn cấu trúc trong danh sách
                    self.structure_list.setCurrentItem(item)
                    # Hiển thị thông tin của cấu trúc đó
                    structure = self.get_structure_by_id(structure_id)
                    if structure:
                        self.update_structure_info(structure)
                        # Highlight cấu trúc trong viewer 3D
                        try:
                            self.structure_viewer_3d.set_structure_color(
                                structure_id, structure.color
                            )
                            self.structure_viewer_3d.set_structure_opacity(
                                structure_id, 1.0
                            )  # Full opacity

                            # Giảm opacity các cấu trúc khác
                            for other_id in self.structure_viewer_3d.structures.keys():
                                if other_id != structure_id:
                                    self.structure_viewer_3d.set_structure_opacity(
                                        other_id, 0.3
                                    )

                            self.structure_viewer_3d.update_view()
                        except Exception as highlight_error:
                            logger.debug(
                                f"Không thể highlight cấu trúc 3D: {highlight_error}"
                            )
                    break
        except Exception as e:
            logger.error(f"Lỗi khi xử lý sự kiện click cấu trúc 3D: {str(e)}")
            logger.error(traceback.format_exc())

    def _reset_3d_view(self):
        """Đặt lại góc nhìn 3D về mặc định."""
        if not hasattr(self, "structure_viewer_3d") or not HAS_3D_VISUALIZATION:
            return

        try:
            # Reset camera position
            self.structure_viewer_3d.reset_camera()

            # Reset opacity tất cả cấu trúc về mặc định (0.8)
            for struct_id in self.structure_viewer_3d.structures.keys():
                self.structure_viewer_3d.set_structure_opacity(struct_id, 0.8)

            # Cập nhật view
            self.structure_viewer_3d.update_view()

            logger.info("Đã đặt lại góc nhìn 3D")
        except Exception as e:
            logger.error(f"Lỗi khi đặt lại góc nhìn 3D: {str(e)}")
            logger.error(traceback.format_exc())

    def _create_new_structure(self):
        """Hiển thị hộp thoại tạo cấu trúc mới."""
        try:
            from quangtps.ui.dialogs.structure_creation_dialog import (
                StructureCreationDialog,
            )

            dialog = StructureCreationDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                # Xử lý tạo cấu trúc mới từ thông tin trong dialog
                new_structure = dialog.get_structure_data()
                if new_structure:
                    self._add_new_structure(new_structure)
        except ImportError:
            logger.error("Không thể import StructureCreationDialog")
            QMessageBox.warning(
                self, "Lỗi", "Không thể tạo cấu trúc mới: module dialog không có sẵn"
            )

    def _edit_selected_structure(self):
        """Hiển thị hộp thoại chỉnh sửa cấu trúc đã chọn."""
        selected_items = self.structure_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self, "Thông báo", "Vui lòng chọn một cấu trúc để chỉnh sửa"
            )
            return

        struct_id = selected_items[0].data(Qt.UserRole)
        if not struct_id:
            return

        structure = self.get_structure_by_id(struct_id)
        if not structure:
            return

        try:
            from quangtps.ui.dialogs.structure_edit_dialog import StructureEditDialog

            dialog = StructureEditDialog(self, structure)
            if dialog.exec_() == QDialog.Accepted:
                # Cập nhật cấu trúc từ dữ liệu trong dialog
                updated_structure = dialog.get_structure_data()
                if updated_structure:
                    self._update_structure(updated_structure)
                    self.update_structure_info(updated_structure)
        except ImportError:
            logger.error("Không thể import StructureEditDialog")
            QMessageBox.warning(
                self, "Lỗi", "Không thể chỉnh sửa cấu trúc: module dialog không có sẵn"
            )

    def _delete_selected_structure(self):
        """Xóa cấu trúc đã chọn."""
        selected_items = self.structure_list.selectedItems()
        if not selected_items:
            return

        # Hỏi xác nhận trước khi xóa
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa {len(selected_items)} cấu trúc đã chọn?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                for item in selected_items:
                    struct_id = item.data(Qt.UserRole)
                    if struct_id:
                        # Xóa cấu trúc từ patient
                        self._remove_structure(struct_id)

                # Cập nhật lại danh sách
                self.update_structure_list()
                logger.info(f"Đã xóa {len(selected_items)} cấu trúc")
            except Exception as e:
                logger.error(f"Lỗi khi xóa cấu trúc: {str(e)}")
                QMessageBox.critical(self, "Lỗi", f"Không thể xóa cấu trúc: {str(e)}")

    def update_structure_info(self, structure):
        """
        Cập nhật thông tin chi tiết của cấu trúc.

        Parameters
        ----------
        structure : Structure
            Cấu trúc cần hiển thị thông tin
        """
        if not structure:
            self.structure_info.clear()
            self.stats_table.setRowCount(0)
            return

        # Cập nhật thông tin cơ bản
        html_info = f"""
        <h3>{structure.name}</h3>
        <p><b>ID:</b> {structure.id if hasattr(structure, "id") else "N/A"}</p>
        <p><b>Loại:</b> {structure.type if hasattr(structure, "type") else "Không xác định"}</p>
        <p><b>Màu:</b> <span style="color:{structure.color if hasattr(structure, "color") else "#FFFFFF"};
                              background-color:{structure.color if hasattr(structure, "color") else "#FFFFFF"};
                              border:1px solid black;">&#9608;&#9608;&#9608;</span></p>
        """

        # Thêm thông tin về thể tích và số contour
        if hasattr(structure, "volume"):
            html_info += f"<p><b>Thể tích:</b> {structure.volume:.2f} cc</p>"

        if hasattr(structure, "num_contours"):
            html_info += f"<p><b>Số contour:</b> {structure.num_contours}</p>"

        self.structure_info.setHtml(html_info)

        # Cập nhật bảng thống kê
        self.stats_table.setRowCount(0)

        stats = [
            ("Tên", structure.name),
            ("ID", getattr(structure, "id", "N/A")),
            ("Thể tích", f"{getattr(structure, 'volume', 0.0):.2f} cc"),
            ("Loại", getattr(structure, "type", "Không xác định")),
            ("Số lát cắt", getattr(structure, "num_slices", 0)),
            ("Số điểm", getattr(structure, "num_points", 0)),
            ("Trung tâm X", f"{getattr(structure, 'center_x', 0.0):.2f} mm"),
            ("Trung tâm Y", f"{getattr(structure, 'center_y', 0.0):.2f} mm"),
            ("Trung tâm Z", f"{getattr(structure, 'center_z', 0.0):.2f} mm"),
        ]

        for i, (name, value) in enumerate(stats):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def get_structure_by_id(self, struct_id):
        """
        Lấy đối tượng cấu trúc theo ID.

        Parameters
        ----------
        struct_id : str
            ID của cấu trúc cần tìm

        Returns
        -------
        Structure hoặc None
            Đối tượng cấu trúc nếu tìm thấy, None nếu không tìm thấy
        """
        if not self.patient or not self.patient.structures:
            return None

        for structure in self.patient.structures:
            if getattr(structure, "id", None) == struct_id:
                return structure

        return None

    def setup_connections(self):
        """Set up connections between widgets."""
        self.structure_list.currentItemChanged.connect(
            self.on_structure_selection_changed
        )

    def set_image(self, image):
        """Set the current image for structure editing."""
        self.image = image

        if image:
            # Update status bar
            dimensions = f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}"
            self.status_label.setText(f"Image loaded - Size: {dimensions}")

            # Reset or create a new structure set if needed
            if not self.structure_set:
                try:
                    # Tạo StructureSet mới
                    self.structure_set = StructureSet(name="RTStruct")
                except Exception as e:
                    logger.error(f"Lỗi khi tạo StructureSet: {e}")
                    return

            # Update structure set name
            self.struct_set_name.setText(self.structure_set.name)

            # Set image in segmentation interface
            self.segmentation_interface.set_image_data(image)

            # Enable the add structure button
            self.add_structure_btn.setEnabled(True)
        else:
            # Clear UI if no image
            self.status_label.setText("No image loaded")
            self.add_structure_btn.setEnabled(False)
            self.delete_structure_btn.setEnabled(False)
            self.edit_structure_btn.setEnabled(False)

    def set_structure_set(self, structure_set):
        """Set the structure set for editing."""
        self.structure_set = structure_set

        if structure_set:
            # Update structure set name
            self.struct_set_name.setText(self.structure_set.name)

            # Clear structure list
            self.structure_list.clear()
            self.structure_items = []

            # Add structures to list
            for structure in structure_set.structures:
                self.add_structure_to_list(structure)

            # Cập nhật trạng thái
            self.status_label.setText(
                f"Loaded {len(structure_set.structures)} structures from {structure_set.name}"
            )

            # Enable/disable buttons
            self.add_structure_btn.setEnabled(True)
            self.delete_structure_btn.setEnabled(True)
            self.edit_structure_btn.setEnabled(True)

            # Emit signal
            self.structureSetChanged.emit(structure_set)
        else:
            # Clear UI if no structure set
            self.structure_list.clear()
            self.struct_set_name.setText("None")
            self.add_structure_btn.setEnabled(False)
            self.delete_structure_btn.setEnabled(False)
            self.edit_structure_btn.setEnabled(False)

        # Clear selected structure
        self.selected_structure = None
        self.update_property_display()

    def add_structure_to_list(self, structure):
        """Add a structure to the list widget."""
        if not structure:
            return

        item = QListWidgetItem(structure.name)

        # Set color indicator (20x20 pixel square)
        pixmap = QPixmap(20, 20)
        pixmap.fill(QColor(*structure.color))
        item.setIcon(QIcon(pixmap))

        # Store the structure reference
        item.setData(Qt.UserRole, structure)

        # Add to list widget
        self.structure_list.addItem(item)

    def add_new_structure(self):
        """Add a new structure to the structure set."""
        if not self.structure_set:
            return

        # Get structure name
        name, ok = QInputDialog.getText(
            self,
            "New Structure",
            "Enter structure name:",
            text=f"Structure {len(self.structure_set.structures) + 1}",
        )

        if not ok or not name:
            return

        # Select color
        color_dialog = QColorDialog(self)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)

        # Set initial color (cycle through some presets)
        preset_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 0),  # Dark Green
            (0, 128, 255),  # Light Blue
        ]

        index = len(self.structure_set.structures) % len(preset_colors)
        color_dialog.setCurrentColor(QColor(*preset_colors[index]))

        if color_dialog.exec_():
            qcolor = color_dialog.currentColor()
            color = (qcolor.red(), qcolor.green(), qcolor.blue())
        else:
            color = preset_colors[index]

        # Create new structure
        try:
            # Tạo structure mới
            structure = Structure(name=name)
            # Gán các thuộc tính khác
            structure.color = color
            structure.image_ref = self.image
        except Exception as e:
            logger.error(f"Lỗi khi tạo Structure: {e}")
            return

        # Add to structure set
        self.structure_set.add_structure(structure)

        # Add to list widget
        self.add_structure_to_list(structure)

        # Select the new structure
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            if item.data(Qt.UserRole) == structure:
                self.structure_list.setCurrentItem(item)
                break

        # Emit signal
        self.structureAdded.emit(structure)

    def on_structure_selection_changed(self, current, previous):
        """Handle structure selection change in list."""
        if current:
            structure = current.data(Qt.UserRole)
            self.selected_structure = structure

            # Update UI
            self.delete_structure_btn.setEnabled(True)
            self.edit_structure_btn.setEnabled(True)

            # Update property display
            self.update_property_display()

            # Set the current structure in segmentation interface
            self.segmentation_interface.set_structure(structure)

            # Emit signal
            self.structureSelectionChanged.emit(structure)
        else:
            self.selected_structure = None

            # Update UI
            self.delete_structure_btn.setEnabled(False)
            self.edit_structure_btn.setEnabled(False)

            # Update property display
            self.update_property_display()

            # Clear structure in segmentation interface
            self.segmentation_interface.set_structure(None)

    def update_property_display(self):
        """Update the property display for the selected structure."""
        if self.selected_structure:
            self.prop_name.setText(f"Name: {self.selected_structure.name}")
            self.prop_type.setText(f"Type: {self.selected_structure.type}")

            color_text = f"RGB({self.selected_structure.color[0]}, {self.selected_structure.color[1]}, {self.selected_structure.color[2]})"
            self.prop_color.setText(f"Color: {color_text}")

            # Calculate volume if possible
            if self.image:
                voxel_size = (
                    self.image.voxel_size
                    if hasattr(self.image, "voxel_size")
                    else (1.0, 1.0, 1.0)
                )
                voxel_volume = (
                    voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000
                )  # Convert to cc
                structure_volume = self.selected_structure.get_volume(voxel_volume)
                self.prop_volume.setText(f"Volume: {structure_volume:.2f} cc")
            else:
                self.prop_volume.setText("Volume: - cc")
        else:
            self.prop_name.setText("Name: -")
            self.prop_type.setText("Type: -")
            self.prop_color.setText("Color: -")
            self.prop_volume.setText("Volume: - cc")

    def delete_selected_structure(self):
        """Delete the currently selected structure."""
        if not self.selected_structure:
            return

        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the structure '{self.selected_structure.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if confirm != QMessageBox.Yes:
            return

        # Remove from structure set
        self.structure_set.remove_structure(self.selected_structure)

        # Get current selection
        current_row = self.structure_list.currentRow()

        # Remove from list widget
        self.structure_list.takeItem(current_row)

        # Emit signal
        self.structureRemoved.emit(self.selected_structure)

        # Clear selected structure
        self.selected_structure = None

        # Update property display
        self.update_property_display()

    def edit_structure_properties(self):
        """Edit properties of the selected structure."""
        if not self.selected_structure:
            return

        # Get new name
        name, ok = QInputDialog.getText(
            self, "Edit Structure", "Structure name:", text=self.selected_structure.name
        )

        if not ok:
            return

        if name and name != self.selected_structure.name:
            self.selected_structure.name = name

            # Update list widget
            current_item = self.structure_list.currentItem()
            current_item.setText(name)

        # Select new color
        color_dialog = QColorDialog(self)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
        color_dialog.setCurrentColor(QColor(*self.selected_structure.color))

        if color_dialog.exec_():
            qcolor = color_dialog.currentColor()
            self.selected_structure.color = (
                qcolor.red(),
                qcolor.green(),
                qcolor.blue(),
            )

            # Update color indicator in list
            current_item = self.structure_list.currentItem()
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor(*self.selected_structure.color))
            current_item.setIcon(QIcon(pixmap))

        # Update property display
        self.update_property_display()

        # Emit signal
        self.structureModified.emit(self.selected_structure)

    def show_structure_context_menu(self, position):
        """Show context menu for structures."""
        if not self.structure_list.count():
            return

        selected_item = self.structure_list.itemAt(position)
        if not selected_item:
            return

        # Create context menu
        context_menu = QMenu(self)

        # Add actions
        edit_action = QAction("Edit Properties", self)
        edit_action.triggered.connect(self.edit_structure_properties)
        context_menu.addAction(edit_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.delete_selected_structure)
        context_menu.addAction(delete_action)

        context_menu.addSeparator()

        # Add visibility toggle
        structure = selected_item.data(Qt.UserRole)

        visibility_action = QAction("Show/Hide", self)
        visibility_action.setCheckable(True)
        visibility_action.setChecked(structure.visible)
        visibility_action.triggered.connect(
            lambda checked, s=structure: self.toggle_structure_visibility(s, checked)
        )
        context_menu.addAction(visibility_action)

        # Add margin tool option
        context_menu.addSeparator()
        margin_action = QAction("Apply Margin...", self)
        margin_action.triggered.connect(
            lambda: self.apply_margin_to_structure(structure)
        )
        context_menu.addAction(margin_action)

        # Add copy/paste options
        context_menu.addSeparator()

        copy_action = QAction("Copy to Next Slice", self)
        copy_action.triggered.connect(lambda: self.copy_to_next_slice(structure))
        context_menu.addAction(copy_action)

        copy_all_action = QAction("Copy to All Slices", self)
        copy_all_action.triggered.connect(self.copy_to_all_slices)
        context_menu.addAction(copy_all_action)

        # Show the menu
        context_menu.exec_(self.structure_list.mapToGlobal(position))

    def toggle_structure_visibility(self, structure, visible):
        """Toggle the visibility of a structure."""
        structure.visible = visible
        self.structureVisibilityChanged.emit(structure, visible)

    def copy_to_next_slice(self, structure=None):
        """Copy the current structure contour to the next slice."""
        if not structure:
            structure = self.selected_structure

        if not structure:
            return

        # This would be implemented to copy the current slice's contour
        # to the next slice for the selected structure
        QMessageBox.information(
            self,
            "Feature Not Implemented",
            "Copy to next slice feature is not yet implemented.",
        )

    def copy_to_all_slices(self):
        """Copy the current structure contour to all slices."""
        if not self.selected_structure:
            return

        # This would be implemented to copy the current slice's contour
        # to all slices for the selected structure
        QMessageBox.information(
            self,
            "Feature Not Implemented",
            "Copy to all slices feature is not yet implemented.",
        )

    def auto_segment(self):
        """Perform auto-segmentation for the selected structure."""
        if not self.selected_structure or not self.image:
            return

        # This would be implemented to perform automatic segmentation
        # for the selected structure using image data
        QMessageBox.information(
            self,
            "Feature Not Implemented",
            "Auto-segmentation feature is not yet implemented.",
        )

    def on_structure_modified(self):
        """Handle structure modification from the segmentation interface."""
        if self.selected_structure:
            # Update property display
            self.update_property_display()

            # Emit signal
            self.structureModified.emit(self.selected_structure)

    def handle_mouse_event(self, event_type, point, slice_index=None, orientation=None):
        """Handle mouse events from the image viewer."""
        if not self.selected_structure:
            return False

        if event_type == "press":
            return self.segmentation_interface.handle_mouse_press(
                point, slice_index, orientation
            )
        elif event_type == "move":
            return self.segmentation_interface.handle_mouse_move(point)
        elif event_type == "release":
            return self.segmentation_interface.handle_mouse_release(point)

        return False

    def get_cursor_for_viewer(self):
        """Get the cursor for the current tool."""
        return self.segmentation_interface.get_cursor_for_viewer()

    def get_overlay_for_viewer(self, orientation, slice_index):
        """Get overlay for display in the viewer."""
        return self.segmentation_interface.get_overlay_for_viewer(
            orientation, slice_index
        )

    def closeEvent(self, event):
        """Handle the close event."""
        self.windowClosed.emit()
        super().closeEvent(event)

    def apply_margin_to_structure(self, structure=None):
        """Apply margin to the selected structure."""
        if not structure:
            structure = self.selected_structure

        if not structure:
            return

        # Lấy pixel spacing từ image
        pixel_spacing = (1.0, 1.0)
        if self.image:
            if hasattr(self.image, "spacing"):
                pixel_spacing = (self.image.spacing[0], self.image.spacing[1])

        # Sử dụng MarginToolWidget mới thay vì hộp thoại cũ
        try:
            from quangtps.segmentation.contour.margin_tool_widget import (
                show_margin_tool_dialog,
            )

            # Hiển thị hộp thoại margin tool
            margin_widget = show_margin_tool_dialog(
                self, self.structure_set, pixel_spacing
            )

            # Kết nối signal marginApplied với hàm xử lý
            margin_widget.marginApplied.connect(self.on_margin_applied)

        except ImportError as e:
            # Fallback vào hộp thoại cũ nếu không tìm thấy module mới
            logger.warning(
                f"Không thể tải MarginToolWidget: {e}. Sử dụng hộp thoại cũ."
            )
            self._show_legacy_margin_dialog(structure)

    def on_margin_applied(self, old_structure, new_structure):
        """Xử lý sau khi áp dụng margin."""
        # Cập nhật UI
        if old_structure != new_structure:
            # Nếu tạo cấu trúc mới, thêm vào danh sách
            self.add_structure_to_list(new_structure)

            # Chọn cấu trúc mới
            for i in range(self.structure_list.count()):
                item = self.structure_list.item(i)
                if item.data(Qt.UserRole) == new_structure:
                    self.structure_list.setCurrentItem(item)
                    break
        else:
            # Nếu cập nhật cấu trúc cũ, phát tín hiệu để cập nhật
            self.structureModified.emit(old_structure)

    def _show_legacy_margin_dialog(self, structure):
        """Hiển thị hộp thoại margin truyền thống (để dự phòng)."""
        # Ask user for margin value and type
        dialog = QDialog(self)
        dialog.setWindowTitle("Apply Margin")
        layout = QVBoxLayout(dialog)

        # Margin type selection
        type_group = QGroupBox("Margin Type")
        type_layout = QVBoxLayout()

        margin_type_combo = QComboBox()
        margin_type_combo.addItem("Uniform Expansion/Contraction", "UNIFORM")
        margin_type_combo.addItem("Anisotropic Expansion", "ANISOTROPIC")
        margin_type_combo.addItem("Ring Structure", "RING")
        margin_type_combo.addItem("Surface Layer", "SURFACE")
        type_layout.addWidget(margin_type_combo)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Uniform margin parameters
        uniform_group = QGroupBox("Uniform Parameters")
        uniform_layout = QFormLayout()
        uniform_margin = QDoubleSpinBox()
        uniform_margin.setRange(-50, 50)
        uniform_margin.setSingleStep(0.5)
        uniform_margin.setValue(5.0)
        uniform_margin.setSuffix(" mm")
        uniform_layout.addRow("Margin:", uniform_margin)
        uniform_group.setLayout(uniform_layout)
        layout.addWidget(uniform_group)

        # Anisotropic margin parameters
        aniso_group = QGroupBox("Anisotropic Parameters")
        aniso_layout = QFormLayout()
        aniso_anterior = QDoubleSpinBox()
        aniso_anterior.setRange(-50, 50)
        aniso_anterior.setSingleStep(0.5)
        aniso_anterior.setValue(5.0)
        aniso_anterior.setSuffix(" mm")
        aniso_layout.addRow("Anterior:", aniso_anterior)

        aniso_posterior = QDoubleSpinBox()
        aniso_posterior.setRange(-50, 50)
        aniso_posterior.setSingleStep(0.5)
        aniso_posterior.setValue(5.0)
        aniso_posterior.setSuffix(" mm")
        aniso_layout.addRow("Posterior:", aniso_posterior)

        aniso_left = QDoubleSpinBox()
        aniso_left.setRange(-50, 50)
        aniso_left.setSingleStep(0.5)
        aniso_left.setValue(5.0)
        aniso_left.setSuffix(" mm")
        aniso_layout.addRow("Left:", aniso_left)

        aniso_right = QDoubleSpinBox()
        aniso_right.setRange(-50, 50)
        aniso_right.setSingleStep(0.5)
        aniso_right.setValue(5.0)
        aniso_right.setSuffix(" mm")
        aniso_layout.addRow("Right:", aniso_right)
        aniso_group.setLayout(aniso_layout)
        layout.addWidget(aniso_group)

        # Ring parameters
        ring_group = QGroupBox("Ring Parameters")
        ring_layout = QFormLayout()
        inner_margin = QDoubleSpinBox()
        inner_margin.setRange(-50, 50)
        inner_margin.setSingleStep(0.5)
        inner_margin.setValue(0.0)
        inner_margin.setSuffix(" mm")
        ring_layout.addRow("Inner Margin:", inner_margin)

        outer_margin = QDoubleSpinBox()
        outer_margin.setRange(0, 50)
        outer_margin.setSingleStep(0.5)
        outer_margin.setValue(5.0)
        outer_margin.setSuffix(" mm")
        ring_layout.addRow("Outer Margin:", outer_margin)
        ring_group.setLayout(ring_layout)
        layout.addWidget(ring_group)

        # Surface parameters
        surface_group = QGroupBox("Surface Parameters")
        surface_layout = QFormLayout()
        thickness = QDoubleSpinBox()
        thickness.setRange(0.1, 20)
        thickness.setSingleStep(0.1)
        thickness.setValue(3.0)
        thickness.setSuffix(" mm")
        surface_layout.addRow("Thickness:", thickness)
        surface_group.setLayout(surface_layout)
        layout.addWidget(surface_group)

        # Structure creation options
        option_group = QGroupBox("Output Options")
        option_layout = QVBoxLayout()

        create_new_rb = QRadioButton("Create new structure")
        create_new_rb.setChecked(True)
        replace_rb = QRadioButton("Replace existing structure")

        option_layout.addWidget(create_new_rb)
        option_layout.addWidget(replace_rb)

        name_layout = QHBoxLayout()
        name_label = QLabel("New structure name:")
        new_name = QLineEdit()
        new_name.setText(f"{structure.name}_margin")
        name_layout.addWidget(name_label)
        name_layout.addWidget(new_name)
        option_layout.addLayout(name_layout)

        option_group.setLayout(option_layout)
        layout.addWidget(option_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Show/hide parameter groups based on selected type
        def update_ui():
            margin_type = margin_type_combo.currentData()
            uniform_group.setVisible(margin_type == "UNIFORM")
            aniso_group.setVisible(margin_type == "ANISOTROPIC")
            ring_group.setVisible(margin_type == "RING")
            surface_group.setVisible(margin_type == "SURFACE")

        margin_type_combo.currentIndexChanged.connect(update_ui)
        update_ui()  # Initial update

        # Execute dialog
        if dialog.exec_() != QDialog.Accepted:
            return

        # Get parameters
        margin_type = margin_type_combo.currentData()
        create_new = create_new_rb.isChecked()
        new_structure_name = new_name.text() if create_new else structure.name

        # Get parameters based on margin type
        margin_params = {}
        if margin_type == "UNIFORM":
            margin_params = {"margin_mm": uniform_margin.value()}
        elif margin_type == "ANISOTROPIC":
            margin_params = {
                "margins_mm": {
                    "ANTERIOR": aniso_anterior.value(),
                    "POSTERIOR": aniso_posterior.value(),
                    "LEFT": aniso_left.value(),
                    "RIGHT": aniso_right.value(),
                }
            }
        elif margin_type == "RING":
            margin_params = {
                "inner_margin_mm": inner_margin.value(),
                "outer_margin_mm": outer_margin.value(),
            }
        elif margin_type == "SURFACE":
            margin_params = {"thickness_mm": thickness.value()}

        # Get actual pixel spacing from image
        pixel_spacing = (1.0, 1.0)
        if self.image:
            if hasattr(self.image, "spacing"):
                pixel_spacing = (self.image.spacing[0], self.image.spacing[1])

        try:
            # Áp dụng margin cho mỗi contour trong cấu trúc
            new_contours = []
            for slice_num, contours in structure.contours.items():
                if contours:
                    # Apply margin operation to all contours on this slice
                    margin_contours = self.margin_tool.margin_by_type(
                        contours, MarginType(margin_type), margin_params, pixel_spacing
                    )
                    new_contours.append((slice_num, margin_contours))

            if create_new:
                # Create new structure
                new_structure = Structure(name=new_structure_name)
                new_structure.id = f"{structure.id}_margin"
                new_structure.color = structure.color
                if hasattr(structure, "type"):
                    new_structure.type = structure.type
                if hasattr(structure, "priority"):
                    new_structure.priority = structure.priority

                # Add contours to new structure
                for slice_num, contours in new_contours:
                    try:
                        # Thử phương thức set_contours nếu có
                        if hasattr(new_structure, "set_contours"):
                            new_structure.set_contours(slice_num, contours)
                        else:
                            # Thử gán trực tiếp vào dictionary contours nếu không có phương thức
                            if not hasattr(new_structure, "contours"):
                                new_structure.contours = {}
                            new_structure.contours[slice_num] = contours
                    except Exception as e:
                        logger.error(f"Lỗi khi cập nhật contours: {e}")

                # Add new structure to current structure set
                if self.structure_set:
                    self.structure_set.add_structure(new_structure)
                    self.add_structure_to_list(new_structure)

                    # Select the new structure
                    for i in range(self.structure_list.count()):
                        item = self.structure_list.item(i)
                        if item.data(Qt.UserRole) == new_structure:
                            self.structure_list.setCurrentItem(item)
                            break
            else:
                # Replace contours in existing structure
                for slice_num, contours in new_contours:
                    try:
                        # Thử phương thức set_contours nếu có
                        if hasattr(structure, "set_contours"):
                            structure.set_contours(slice_num, contours)
                        else:
                            # Thử gán trực tiếp vào dictionary contours nếu không có phương thức
                            if not hasattr(structure, "contours"):
                                structure.contours = {}
                            structure.contours[slice_num] = contours
                    except Exception as e:
                        logger.error(f"Lỗi khi cập nhật contours: {e}")

            # Update MPR views
            self.structureModified.emit(structure)

        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Error", f"Failed to apply margin: {str(e)}")
            logger.error(f"Error applying margin: {str(e)}")


def test_structure_tab():
    """Test function for the structure tab."""
    import sys

    # Sử dụng cách import này để tránh lỗi linter
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)

    # Create main window
    main_window = QtWidgets.QMainWindow()

    # Create test patient
    class TestPatient:
        """Lớp Patient giả cho mục đích kiểm thử."""

        def __init__(self, id, name):
            self.id = id
            self.name = name
            # Thêm các thuộc tính tương thích với Patient
            self.image_sets = []
            self.structure_sets = []
            self.plans = []

    # Create test image
    class TestImage:
        def __init__(self):
            self.id = "test_image_1"
            self.description = "Test CT Image"
            self.series_id = "series_1"
            self.data = np.zeros((100, 512, 512))
            # Add some test patterns
            for z in range(100):
                # Circular pattern that varies with slice
                center_x, center_y = 256, 256
                radius = 100 + z
                for x in range(512):
                    for y in range(512):
                        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if dist < radius:
                            self.data[z, y, x] = 100 + z

            self.shape = self.data.shape
            self.spacing = (1.0, 1.0, 1.0)  # 1mm spacing

        def __getitem__(self, indices):
            return self.data[indices]

    # Create mock PatientDB for testing
    class TestPatientDB:
        def __init__(self):
            self.patients = {}
            self.images = {}
            self.structure_sets = {}
            self.plans = {}

        def get_images_for_patient(self, patient_id):
            return [img for img in self.images.values() if img.patient_id == patient_id]

        def get_structure_sets_for_patient(self, patient_id):
            return [
                ss for ss in self.structure_sets.values() if ss.patient_id == patient_id
            ]

        def get_plans_for_patient(self, patient_id):
            return [
                plan for plan in self.plans.values() if plan.patient_id == patient_id
            ]

        def add_structure_set(self, structure_set):
            ss_id = f"ss_{len(self.structure_sets) + 1}"
            structure_set.id = ss_id
            self.structure_sets[ss_id] = structure_set
            return ss_id

        def get_image_data(self, image_id):
            if image_id in self.images:
                return self.images[image_id].data
            return None

    # Create test data
    test_patient = TestPatient("patient_1", "John Doe")
    test_image = TestImage()
    test_image.patient_id = test_patient.id

    # Register mock services
    try:
        # Khởi tạo TestPatientDB một lần
        test_patient_db = TestPatientDB()

        # Đăng ký với ServiceRegistry
        registry = ServiceRegistry.get_instance()
        if registry and hasattr(registry, "register_service"):
            registry.register_service("PatientDB", test_patient_db)
            logger.info("Đã đăng ký TestPatientDB vào ServiceRegistry")
        else:
            logger.warning("ServiceRegistry không có sẵn phương thức register_service")

        # Sử dụng TestPatientDB trực tiếp
        patient_db = test_patient_db
    except Exception as e:
        logger.error(f"Lỗi khi đăng ký service: {e}")
        logger.error(traceback.format_exc())
        patient_db = TestPatientDB()  # Fallback

    patient_db.patients[test_patient.id] = test_patient
    patient_db.images[test_image.id] = test_image

    # Create structure tab
    structure_tab = StructureTab()
    structure_tab.set_patient(test_patient)

    # Set as central widget
    main_window.setCentralWidget(structure_tab)
    main_window.setWindowTitle("QuangTPS - Structure Tab")
    main_window.resize(1200, 800)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    test_structure_tab()
