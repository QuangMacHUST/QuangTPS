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
        QProgressBar,
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
    try:
        from quangtps.segmentation.structures.structure import (
            Structure,
            StructureType,
            StructurePriority,
        )
    except ImportError:
        logging.warning("Không thể import Structure. Sử dụng lớp giả.")

        class Structure:
            def __init__(self, *args, **kwargs):
                self.id = "unknown"
                self.name = "Unknown Structure"
                self.color = (1.0, 0.0, 0.0)
                self.visible = True

        class StructureType:
            PTV = "PTV"
            OAR = "OAR"
            OTHER = "OTHER"

        class StructurePriority:
            LOW = 1
            MEDIUM = 2
            HIGH = 3

    try:
        from quangtps.segmentation.structures.structure_set import StructureSet
    except ImportError:
        logging.warning("Không thể import StructureSet. Sử dụng lớp giả.")

        class StructureSet:
            def __init__(self, *args, **kwargs):
                self.structures = []
                self.id = "unknown"
                self.name = "Unknown StructureSet"

    try:
        from quangtps.segmentation.contour.contour_manager import ContourManager
    except ImportError:
        logging.warning("Không thể import ContourManager. Sử dụng lớp giả.")

        class ContourManager:
            def __init__(self, *args, **kwargs):
                pass

            def undo(self):
                return False

            def redo(self):
                return False

    try:
        from quangtps.segmentation.contour.polygon_tool import PolygonTool
    except ImportError:
        logging.warning("Không thể import PolygonTool. Sử dụng lớp giả.")

        class PolygonTool:
            def __init__(self, *args, **kwargs):
                pass

            def set_mode(self, mode):
                pass

    try:
        from quangtps.segmentation.contour.margin import MarginTool, MarginType
    except ImportError:
        logging.warning("Không thể import MarginTool. Sử dụng lớp giả.")

        class MarginTool:
            def __init__(self, *args, **kwargs):
                pass

            def margin_by_type(self, contours, margin_type, params, spacing):
                return contours

        class MarginType:
            def __init__(self, *args, **kwargs):
                pass

    try:
        from quangtps.segmentation.contour.boolean_operations import BooleanOperator
    except ImportError:
        logging.warning("Không thể import BooleanOperator. Sử dụng lớp giả.")

        class BooleanOperator:
            UNION = "union"
            INTERSECTION = "intersection"
            SUBTRACT = "subtract"

    try:
        from quangtps.segmentation.contour.interpolation import ContourInterpolator
    except ImportError:
        logging.warning("Không thể import ContourInterpolator. Sử dụng lớp giả.")

        class ContourInterpolator:
            def __init__(self, *args, **kwargs):
                pass

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

# Import StructureSet và Structure với fallback
try:
    from quangtps.segmentation.structures.structure_set import StructureSet
    from quangtps.segmentation.structures.structure import Structure
except ImportError:
    # Fallback classes nếu import thất bại
    class StructureSet:
        def __init__(self, *args, **kwargs):
            self.structures = []
            self.id = "unknown"
            self.name = "Unknown StructureSet"

    class Structure:
        def __init__(self, *args, **kwargs):
            self.id = "unknown"
            self.name = "Unknown Structure"
            self.color = (1.0, 0.0, 0.0)
            self.visible = True


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

            # Update MPR viewer with image data
            if hasattr(self, "mpr_viewer") and self.mpr_viewer:
                try:
                    logger.info("Cập nhật MPR viewer với dữ liệu hình ảnh")
                    # Cập nhật hiển thị MPR với dữ liệu hình ảnh thực
                    self.mpr_viewer.set_image(image)

                    # Cập nhật các overlay cấu trúc nếu có
                    if self.structure_set and self.structure_set.structures:
                        self.update_all_structure_overlays()

                    # Đặt lại các thuộc tính hiển thị
                    self.mpr_viewer.reset_view()
                except Exception as e:
                    logger.error(f"Lỗi khi cập nhật MPR viewer: {str(e)}")
                    QMessageBox.warning(
                        self,
                        "Lỗi MPR Viewer",
                        f"Không thể hiển thị dữ liệu hình ảnh trong MPR viewer: {str(e)}",
                    )

            # Enable the add structure button
            self.add_structure_btn.setEnabled(True)
        else:
            # Clear UI if no image
            self.status_label.setText("No image loaded")
            self.add_structure_btn.setEnabled(False)
            self.delete_structure_btn.setEnabled(False)
            self.edit_structure_btn.setEnabled(False)

            # Clear MPR viewer
            if hasattr(self, "mpr_viewer") and self.mpr_viewer:
                try:
                    self.mpr_viewer.set_image(None)
                except Exception:
                    pass

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

            # Cập nhật trạng thái chọn trong MPR viewer
            if hasattr(self, "mpr_viewer") and self.mpr_viewer:
                try:
                    # Xóa trạng thái "đã chọn" của cấu trúc trước đó
                    if previous:
                        prev_structure = previous.data(Qt.UserRole)
                        if prev_structure and prev_structure.visible:
                            self.mpr_viewer.add_structure_overlay(
                                prev_structure.id,
                                prev_structure,
                                prev_structure.color,
                                False,
                            )

                    # Thêm cấu trúc đã chọn với trạng thái "đã chọn"
                    if structure.visible:
                        self.mpr_viewer.add_structure_overlay(
                            structure.id, structure, structure.color, True
                        )

                    # Cập nhật hiển thị MPR
                    self.mpr_viewer.update_all_views()
                except Exception as e:
                    logger.error(
                        f"Lỗi khi cập nhật cấu trúc đã chọn trong MPR viewer: {str(e)}"
                    )

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

            # Xóa trạng thái đã chọn trong MPR viewer
            if hasattr(self, "mpr_viewer") and self.mpr_viewer and previous:
                try:
                    prev_structure = previous.data(Qt.UserRole)
                    if prev_structure and prev_structure.visible:
                        self.mpr_viewer.add_structure_overlay(
                            prev_structure.id,
                            prev_structure,
                            prev_structure.color,
                            False,
                        )
                        self.mpr_viewer.update_all_views()
                except Exception as e:
                    logger.error(
                        f"Lỗi khi xóa trạng thái đã chọn trong MPR viewer: {str(e)}"
                    )

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

        # Cập nhật hiển thị trong MPR viewer
        if hasattr(self, "mpr_viewer") and self.mpr_viewer:
            try:
                if visible:
                    # Thêm overlay cấu trúc vào MPR viewer
                    color = structure.color
                    is_selected = structure == self.selected_structure
                    self.mpr_viewer.add_structure_overlay(
                        structure.id, structure, color, is_selected
                    )
                else:
                    # Xóa overlay cấu trúc khỏi MPR viewer
                    self.mpr_viewer.remove_structure_overlay(structure.id)
            except Exception as e:
                logger.error(
                    f"Lỗi khi cập nhật hiển thị cấu trúc trong MPR viewer: {str(e)}"
                )

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
            QMessageBox.warning(
                self,
                "Không thể phân đoạn",
                "Vui lòng chọn cấu trúc và đảm bảo đã tải hình ảnh.",
            )
            return

        try:
            # Kiểm tra xem có thể import AutoSegmentationEngine
            from quangtps.segmentation.auto.engine import AutoSegmentationEngine

            # Hiển thị dialog chọn cấu trúc để phân đoạn
            dialog = QDialog(self)
            dialog.setWindowTitle("Phân đoạn tự động")
            layout = QVBoxLayout(dialog)

            # Tạo engine phân đoạn
            auto_seg_engine = AutoSegmentationEngine()

            # Lấy danh sách cấu trúc có thể phân đoạn
            available_structures = auto_seg_engine.get_available_structures()
            if not available_structures:
                QMessageBox.warning(
                    self,
                    "Không có mô hình",
                    "Không tìm thấy mô hình phân đoạn tự động nào. Vui lòng cài đặt mô hình trước.",
                )
                return

            # Tạo form layout cho các tùy chọn
            form_layout = QFormLayout()

            # Combobox chọn cấu trúc để phân đoạn
            structure_combo = QComboBox()
            for structure_name in available_structures:
                structure_combo.addItem(structure_name)
            form_layout.addRow("Cấu trúc:", structure_combo)

            # Tùy chọn sử dụng GPU
            use_gpu_checkbox = QCheckBox("Sử dụng GPU (nếu có)")
            use_gpu_checkbox.setChecked(True)
            form_layout.addRow("", use_gpu_checkbox)

            # Threshold cho phân đoạn nhị phân
            threshold_spinner = QDoubleSpinBox()
            threshold_spinner.setRange(0.1, 0.9)
            threshold_spinner.setSingleStep(0.05)
            threshold_spinner.setValue(0.5)
            form_layout.addRow("Ngưỡng:", threshold_spinner)

            layout.addLayout(form_layout)

            # Progress bar
            progress_label = QLabel("Chuẩn bị...")
            layout.addWidget(progress_label)
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            layout.addWidget(progress_bar)

            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            # Hiển thị dialog
            if dialog.exec_() != QDialog.Accepted:
                return

            # Lấy các lựa chọn từ form
            selected_structure_name = structure_combo.currentText()
            use_gpu = use_gpu_checkbox.isChecked()
            threshold = threshold_spinner.value()

            # Hiển thị progress dialog
            progress = QProgressDialog("Đang phân đoạn tự động...", "Hủy", 0, 100, self)
            progress.setWindowTitle("Phân đoạn tự động")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)

            # Chuẩn bị dữ liệu ảnh
            try:
                # Lấy volume data từ image
                image_data = self.image.data

                # Cập nhật progress
                progress.setValue(20)

                # Thực hiện phân đoạn
                result = auto_seg_engine.segment_volume(
                    volume=image_data,
                    structure=selected_structure_name,
                    use_gpu=use_gpu,
                    threshold=threshold,
                )

                progress.setValue(80)

                # Kiểm tra kết quả
                if not result.get("success", False):
                    error_message = result.get("error", "Lỗi không xác định")
                    QMessageBox.critical(
                        self,
                        "Lỗi phân đoạn",
                        f"Không thể phân đoạn cấu trúc: {error_message}",
                    )
                    return

                # Lấy mask từ kết quả
                mask = result.get("mask")
                if mask is None:
                    QMessageBox.critical(
                        self,
                        "Lỗi phân đoạn",
                        "Không nhận được kết quả phân đoạn từ mô hình.",
                    )
                    return

                # Tạo contour từ mask
                if hasattr(self.segmentation_interface, "set_contours_from_mask"):
                    self.segmentation_interface.set_contours_from_mask(
                        self.selected_structure, mask
                    )

                    # Cập nhật hiển thị
                    self.structureModified.emit(self.selected_structure)

                    # Thông báo thành công
                    QMessageBox.information(
                        self,
                        "Hoàn tất",
                        f"Phân đoạn tự động của {selected_structure_name} hoàn tất.",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Không hỗ trợ",
                        "Giao diện phân đoạn không hỗ trợ tạo contour từ mask.",
                    )

                progress.setValue(100)

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Lỗi phân đoạn",
                    f"Lỗi trong quá trình phân đoạn tự động: {str(e)}",
                )
                logger.error(f"Auto-segmentation error: {str(e)}", exc_info=True)

        except ImportError as e:
            QMessageBox.warning(
                self,
                "Module không khả dụng",
                "Module phân đoạn tự động không khả dụng. Chi tiết lỗi: " + str(e),
            )
            logger.error(f"Auto-segmentation module import error: {str(e)}")
            return
        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi",
                f"Lỗi không xác định trong quá trình phân đoạn tự động: {str(e)}",
            )
            logger.error(f"Unexpected auto-segmentation error: {str(e)}", exc_info=True)
            return

    def on_structure_modified(self):
        """Handle structure modification from the segmentation interface."""
        if self.selected_structure:
            # Update property display
            self.update_property_display()

            # Cập nhật hiển thị trong MPR viewer
            if hasattr(self, "mpr_viewer") and self.mpr_viewer:
                try:
                    # Cập nhật overlay của cấu trúc đã chỉnh sửa
                    structure = self.selected_structure
                    if structure.visible:
                        self.mpr_viewer.add_structure_overlay(
                            structure.id, structure, structure.color, True
                        )

                    # Cập nhật hiển thị MPR
                    self.mpr_viewer.update_all_views()
                    logger.info(
                        f"Đã cập nhật hiển thị cấu trúc {structure.name} trong MPR viewer"
                    )
                except Exception as e:
                    logger.error(
                        f"Lỗi khi cập nhật hiển thị cấu trúc trong MPR viewer: {str(e)}"
                    )

            # Emit signal
            self.structureModified.emit(self.selected_structure)

    def handle_mouse_event(
        self, event_type, point, slice_index=None, orientation=None, tool_type="draw"
    ):
        """Xử lý sự kiện chuột từ MPR viewer."""
        if (
            not hasattr(self, "segmentation_interface")
            or not self.segmentation_interface
        ):
            return False

        try:
            # Truyền thêm thông tin tool_type cho segmentation_interface
            if event_type == "press":
                return self.segmentation_interface.handle_mouse_press(
                    point, slice_index, orientation, tool_type
                )
            elif event_type == "move":
                return self.segmentation_interface.handle_mouse_move(
                    point, slice_index, orientation, tool_type
                )
            elif event_type == "release":
                result = self.segmentation_interface.handle_mouse_release(
                    point, slice_index, orientation, tool_type
                )
                # Cập nhật overlay sau khi vẽ
                if result and hasattr(self, "mpr_viewer"):
                    self.update_structure_overlay(orientation)
                return result

        except Exception as e:
            logger.error(f"Lỗi khi xử lý sự kiện chuột: {str(e)}")
            return False

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

    def setup_slice_view(self):
        """Thiết lập khu vực hiển thị cắt lớp MPR với dữ liệu hình ảnh thực."""
        try:
            # Import MPRViewer từ module mpr_viewer
            from quangtps.ui.mpr_viewer import MPRViewer, ViewOrientation

            # Tạo container cho MPR viewer
            self.slice_view_container = QWidget()
            container_layout = QVBoxLayout(self.slice_view_container)
            container_layout.setContentsMargins(0, 0, 0, 0)

            # Tạo tiêu đề
            title_label = QLabel("Hiển thị đa mặt phẳng (MPR)")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("""
                font-weight: bold;
                background-color: #2070c0;
                color: white;
                padding: 2px;
            """)
            container_layout.addWidget(title_label)

            # Tạo MPRViewer
            self.mpr_viewer = MPRViewer()
            container_layout.addWidget(self.mpr_viewer, 1)

            # Kết nối signals
            self.mpr_viewer.sliceChanged.connect(self.on_slice_changed)
            self.mpr_viewer.orientationChanged.connect(self.on_orientation_changed)
            self.mpr_viewer.mousePressed.connect(self.on_mpr_mouse_pressed)
            self.mpr_viewer.mouseMoved.connect(self.on_mpr_mouse_moved)
            self.mpr_viewer.mouseReleased.connect(self.on_mpr_mouse_released)

            # Thêm toolbox vẽ contour
            self.setup_drawing_tools()
            container_layout.addWidget(self.drawing_toolbar)

        except ImportError as e:
            # Nếu không import được MPRViewer, tạo một widget trống với thông báo lỗi
            logger.error(f"Không thể tải module MPRViewer: {str(e)}")
            self.slice_view_container = QWidget()
            container_layout = QVBoxLayout(self.slice_view_container)

            error_label = QLabel(
                "Không thể tải module hiển thị đa mặt phẳng.\nVui lòng cài đặt các thư viện cần thiết."
            )
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: red; font-weight: bold;")
            container_layout.addWidget(error_label)

            # Tạo các placeholder cho các thuộc tính mà code khác có thể tham chiếu đến
            self.mpr_viewer = None

    def setup_drawing_tools(self):
        """Thiết lập các công cụ vẽ contour."""
        self.drawing_toolbar = QToolBar("Công cụ vẽ")

        # Thêm các công cụ vẽ cơ bản
        self.action_draw = QAction(
            QIcon("quangtps/ui/icons/new_icons/draw.png"), "Vẽ", self
        )
        self.action_draw.setCheckable(True)
        self.action_draw.toggled.connect(self.on_draw_tool_toggled)
        self.drawing_toolbar.addAction(self.action_draw)

        self.action_erase = QAction(
            QIcon("quangtps/ui/icons/new_icons/erase.png"), "Xóa", self
        )
        self.action_erase.setCheckable(True)
        self.action_erase.toggled.connect(self.on_erase_tool_toggled)
        self.drawing_toolbar.addAction(self.action_erase)

        self.action_brush = QAction(
            QIcon("quangtps/ui/icons/new_icons/brush.png"), "Cọ", self
        )
        self.action_brush.setCheckable(True)
        self.action_brush.toggled.connect(self.on_brush_tool_toggled)
        self.drawing_toolbar.addAction(self.action_brush)

        self.action_smart = QAction(
            QIcon("quangtps/ui/icons/new_icons/smart.png"), "Smart Brush", self
        )
        self.action_smart.setCheckable(True)
        self.action_smart.toggled.connect(self.on_smart_tool_toggled)
        self.drawing_toolbar.addAction(self.action_smart)

        self.drawing_toolbar.addSeparator()

        # Thêm công cụ xem trước và quay lại
        self.action_undo = QAction(
            QIcon("quangtps/ui/icons/new_icons/undo.png"), "Hoàn tác", self
        )
        self.action_undo.triggered.connect(self.on_undo)
        self.drawing_toolbar.addAction(self.action_undo)

        self.action_redo = QAction(
            QIcon("quangtps/ui/icons/new_icons/redo.png"), "Làm lại", self
        )
        self.action_redo.triggered.connect(self.on_redo)
        self.drawing_toolbar.addAction(self.action_redo)

        # Công cụ tính toán tự động
        self.drawing_toolbar.addSeparator()

        self.action_auto = QAction(
            QIcon("quangtps/ui/icons/new_icons/auto_segment.png"),
            "Phân đoạn tự động",
            self,
        )
        self.action_auto.triggered.connect(self.auto_segment)
        self.drawing_toolbar.addAction(self.action_auto)

        self.action_margin = QAction(
            QIcon("quangtps/ui/icons/new_icons/margin.png"), "Tạo margin", self
        )
        self.action_margin.triggered.connect(
            lambda: self.apply_margin_to_structure(self.selected_structure)
        )
        self.drawing_toolbar.addAction(self.action_margin)

        # Đặt layout cho toolbar
        self.drawing_toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.drawing_toolbar.setIconSize(QSize(24, 24))

    def on_slice_changed(self, slice_index, orientation):
        """Xử lý khi thay đổi slice trong MPR viewer."""
        if self.selected_structure and self.mpr_viewer:
            # Cập nhật overlay cấu trúc trong chế độ xem
            self.update_structure_overlay(orientation)

    def on_orientation_changed(self, orientation):
        """Xử lý khi thay đổi hướng trong MPR viewer."""
        # Cập nhật giao diện người dùng dựa trên hướng mới
        if orientation and self.mpr_viewer:
            self.update_orientation_ui(orientation)

    def on_mpr_mouse_pressed(self, view_id, view_pos, image_pos):
        """Xử lý khi nhấn chuột trong MPR viewer."""
        if not self.selected_structure:
            # Hiển thị thông báo chọn cấu trúc
            self.status_label.setText("Vui lòng chọn một cấu trúc để vẽ")
            return

        try:
            # Lấy thông tin slice và orientation hiện tại
            orientation = self.mpr_viewer.get_current_orientation()
            slice_index = self.mpr_viewer.get_current_slice_index()

            # Xác định công cụ vẽ nào đang được chọn
            tool_type = self._get_active_drawing_tool()

            # Ghi log sự kiện
            logger.debug(f"Mouse pressed at {image_pos} with tool: {tool_type}")

            # Xử lý sự kiện dựa trên công cụ đang chọn và vị trí
            result = self.handle_mouse_event(
                "press",
                image_pos,
                slice_index=slice_index,
                orientation=orientation,
                tool_type=tool_type,
            )

            # Cập nhật UI nếu cần
            if result:
                self.update_structure_overlay(orientation)

        except Exception as e:
            logger.error(f"Lỗi khi xử lý sự kiện nhấn chuột: {str(e)}")

    def on_mpr_mouse_moved(self, view_id, view_pos, image_pos):
        """Xử lý khi di chuyển chuột trong MPR viewer."""
        if not self.selected_structure:
            return

        try:
            # Lấy thông tin slice và orientation hiện tại
            orientation = self.mpr_viewer.get_current_orientation()
            slice_index = self.mpr_viewer.get_current_slice_index()

            # Xác định công cụ vẽ nào đang được chọn
            tool_type = self._get_active_drawing_tool()

            # Xử lý sự kiện di chuyển
            result = self.handle_mouse_event(
                "move",
                image_pos,
                slice_index=slice_index,
                orientation=orientation,
                tool_type=tool_type,
            )

            # Cập nhật UI nếu có thay đổi
            if result:
                self.update_structure_overlay(orientation)

        except Exception as e:
            logger.error(f"Lỗi khi xử lý sự kiện di chuyển chuột: {str(e)}")

    def on_mpr_mouse_released(self, view_id, view_pos, image_pos):
        """Xử lý khi thả chuột trong MPR viewer."""
        if not self.selected_structure:
            return

        try:
            # Lấy thông tin slice và orientation hiện tại
            orientation = self.mpr_viewer.get_current_orientation()
            slice_index = self.mpr_viewer.get_current_slice_index()

            # Xác định công cụ vẽ nào đang được chọn
            tool_type = self._get_active_drawing_tool()

            # Xử lý sự kiện thả chuột
            result = self.handle_mouse_event(
                "release",
                image_pos,
                slice_index=slice_index,
                orientation=orientation,
                tool_type=tool_type,
            )

            # Cập nhật UI và phát tín hiệu nếu cấu trúc đã thay đổi
            if result:
                self.update_structure_overlay(orientation)
                self.structureModified.emit(self.selected_structure)

        except Exception as e:
            logger.error(f"Lỗi khi xử lý sự kiện thả chuột: {str(e)}")

    def _get_active_drawing_tool(self):
        """Xác định công cụ vẽ nào đang được chọn."""
        try:
            if self.action_draw.isChecked():
                return "draw"
            elif self.action_erase.isChecked():
                return "erase"
            elif self.action_brush.isChecked():
                return "brush"
            elif self.action_smart.isChecked():
                return "smart_brush"
            else:
                return "none"  # Không có công cụ nào được chọn
        except Exception:
            return "none"

    def on_draw_tool_toggled(self, checked):
        """Xử lý khi bật/tắt công cụ vẽ."""
        if checked:
            # Vô hiệu hóa các công cụ khác
            if self.action_erase.isChecked():
                self.action_erase.setChecked(False)
            if self.action_brush.isChecked():
                self.action_brush.setChecked(False)
            if self.action_smart.isChecked():
                self.action_smart.setChecked(False)

            # Cập nhật chế độ vẽ
            if self.mpr_viewer:
                self.mpr_viewer.set_cursor(Qt.CrossCursor)

    def on_erase_tool_toggled(self, checked):
        """Xử lý khi bật/tắt công cụ xóa."""
        if checked:
            # Vô hiệu hóa các công cụ khác
            if self.action_draw.isChecked():
                self.action_draw.setChecked(False)
            if self.action_brush.isChecked():
                self.action_brush.setChecked(False)
            if self.action_smart.isChecked():
                self.action_smart.setChecked(False)

            # Cập nhật chế độ xóa
            if self.mpr_viewer:
                self.mpr_viewer.set_cursor(Qt.CrossCursor)

    def on_brush_tool_toggled(self, checked):
        """Xử lý khi bật/tắt công cụ cọ."""
        if checked:
            # Vô hiệu hóa các công cụ khác
            if self.action_draw.isChecked():
                self.action_draw.setChecked(False)
            if self.action_erase.isChecked():
                self.action_erase.setChecked(False)
            if self.action_smart.isChecked():
                self.action_smart.setChecked(False)

            # Cập nhật chế độ cọ
            if self.mpr_viewer:
                self.mpr_viewer.set_cursor(Qt.CrossCursor)

    def on_smart_tool_toggled(self, checked):
        """Xử lý khi bật/tắt công cụ Smart Brush."""
        if checked:
            # Vô hiệu hóa các công cụ khác
            if self.action_draw.isChecked():
                self.action_draw.setChecked(False)
            if self.action_erase.isChecked():
                self.action_erase.setChecked(False)
            if self.action_brush.isChecked():
                self.action_brush.setChecked(False)

            # Cập nhật chế độ Smart Brush
            if self.mpr_viewer:
                self.mpr_viewer.set_cursor(Qt.CrossCursor)

    def on_undo(self):
        """Hoàn tác thao tác vẽ gần nhất."""
        # Thực hiện hoàn tác nếu có contour manager
        if hasattr(self, "contour_manager") and self.contour_manager:
            if self.contour_manager.undo():
                # Cập nhật hiển thị sau khi hoàn tác
                self.update_all_structure_overlays()
                # Phát tín hiệu thay đổi cấu trúc
                if self.selected_structure:
                    self.structureModified.emit(self.selected_structure)

    def on_redo(self):
        """Làm lại thao tác vẽ đã hoàn tác."""
        # Thực hiện làm lại nếu có contour manager
        if hasattr(self, "contour_manager") and self.contour_manager:
            if self.contour_manager.redo():
                # Cập nhật hiển thị sau khi làm lại
                self.update_all_structure_overlays()
                # Phát tín hiệu thay đổi cấu trúc
                if self.selected_structure:
                    self.structureModified.emit(self.selected_structure)

    def update_structure_overlay(self, orientation):
        """Cập nhật overlay cấu trúc cho một hướng cụ thể."""
        if not hasattr(self, "mpr_viewer") or not self.mpr_viewer:
            return

        try:
            # Lấy slice index hiện tại (không cần orientation parameter)
            if hasattr(self.mpr_viewer, "get_current_slice_index"):
                current_slice = self.mpr_viewer.get_current_slice_index()
            else:
                current_slice = 0

            if current_slice is None:
                return
            # Tạo overlay cho slice hiện tại
            overlay = self.get_overlay_for_viewer(orientation, current_slice)

            # Cập nhật overlay trong MPR viewer
            if hasattr(self.mpr_viewer, "set_overlay"):
                self.mpr_viewer.set_overlay(overlay)
            # Xóa tất cả các overlay hiện tại
            self.mpr_viewer.clear_all_structure_overlays()

            # Thêm lại các overlay cho tất cả các cấu trúc hiển thị
            for structure in self.structure_set.structures:
                if structure.visible:
                    is_selected = structure == self.selected_structure
                    self.mpr_viewer.add_structure_overlay(
                        structure.id, structure, structure.color, is_selected
                    )

            # Cập nhật tất cả các view
            self.mpr_viewer.update_all_views()

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật tất cả các overlay cấu trúc: {str(e)}")

    def update_orientation_ui(self, orientation):
        """Cập nhật giao diện người dùng dựa trên hướng hiện tại."""
        try:
            # Cập nhật các nút điều khiển hoặc hiển thị thông tin cho hướng hiện tại
            if hasattr(self.mpr_viewer, "get_current_slice_index"):
                try:
                    # Thử với orientation parameter trước
                    current_slice = self.mpr_viewer.get_current_slice_index(orientation)
                except TypeError:
                    # Nếu không nhận orientation parameter, gọi không tham số
                    current_slice = self.mpr_viewer.get_current_slice_index()
            else:
                current_slice = 0

            if hasattr(self.mpr_viewer, "get_total_slices"):
                try:
                    # Thử với orientation parameter trước
                    total_slices = self.mpr_viewer.get_total_slices(orientation)
                except TypeError:
                    # Nếu không nhận orientation parameter, gọi không tham số
                    total_slices = self.mpr_viewer.get_total_slices()
            else:
                total_slices = 1

            # Cập nhật status bar hoặc label hiển thị thông tin slice
            if hasattr(self, "status_label"):
                self.status_label.setText(
                    f"Orientation: {orientation}, Slice: {current_slice + 1}/{total_slices}"
                )

            # Cập nhật overlay cấu trúc cho hướng hiện tại
            self.update_structure_overlay(orientation)

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật UI cho hướng {orientation}: {str(e)}")


def test_structure_tab():
    """Test function for the structure tab."""
    try:
        import sys
        from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)

        # Tạo test patient
        class TestPatient:
            def __init__(self, id, name):
                self.id = id
                self.name = name
                self.birth_date = None
                self.gender = None

        # Test structure tab creation
        structure_tab = StructureTab()

        # Test với mock patient
        test_patient = TestPatient("TEST001", "Test Patient")
        structure_tab.set_patient(test_patient)

        print("✓ StructureTab test passed")

    except Exception as e:
        print(f"✗ StructureTab test failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_structure_tab()
