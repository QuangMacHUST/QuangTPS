import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QGroupBox,
    QTabWidget,
    QSplitter,
    QFrame,
    QGridLayout,
    QCheckBox,
    QToolButton,
    QSpinBox,
    QDoubleSpinBox,
    QRadioButton,
    QButtonGroup,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QToolBar,
    QAction,
    QMenu,
    QApplication,
    QFormLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap, QImage, QColor, QPalette

from quangtps.core.logging import get_logger
from quangtps.imaging.image import DicomImage
from quangtps.dicom.dicom_series import DicomSeries
from quangtps.imaging.fusion import ImageFusion
from quangtps.imaging.image_viewer import ImageViewer
from quangtps.ui.image_display import MPRDisplay

logger = get_logger(__name__)


class RegistrationMethod(Enum):
    """Enum for image registration methods."""

    MANUAL = "Manual Registration"
    AUTO = "Automatic Registration"
    POINT_BASED = "Point-based Registration"
    RIGID = "Rigid Registration"
    DEFORMABLE = "Deformable Registration"


class FusionColormap(Enum):
    """Enum for image fusion colormaps."""

    RED_BLUE = "Red-Blue"
    GREEN_PURPLE = "Green-Purple"
    YELLOW_BLUE = "Yellow-Blue"
    HOT_COLD = "Hot-Cold"
    RAINBOW = "Rainbow"


class FusionLayout(Enum):
    """Enum for image fusion display layouts."""

    SIDE_BY_SIDE = "Side by Side"
    OVERLAY = "Overlay"
    SPLIT = "Split View"
    CHECKERBOARD = "Checkerboard"


class ImageRegistrationPanel(QWidget):
    """
    Panel for image registration and fusion, providing Eclipse-like functionality.

    This panel allows users to register multiple image sets (CT, MRI, PET, etc.)
    and visualize the fused images in various layouts with different color schemes.
    """

    registration_completed = pyqtSignal(str, str, np.ndarray)
    fusion_updated = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the image registration panel."""
        super().__init__(parent)
        self.parent = parent

        # Store image data
        self.primary_image = None
        self.secondary_images = {}  # name: image data
        self.current_secondary = None
        self.registration_results = {}  # (primary, secondary): transformation matrix

        self.fusion_engine = ImageFusion()
        self.alpha_blend = 0.5  # Default opacity for fusion
        self.current_fusion_mode = FusionLayout.OVERLAY
        self.current_colormap = FusionColormap.RED_BLUE

        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter: controls on left, viewers on right
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side: control panels
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # Image selection group
        selection_group = QGroupBox("Image Selection")
        selection_layout = QVBoxLayout()

        # Primary image selector
        primary_layout = QHBoxLayout()
        primary_layout.addWidget(QLabel("Primary:"))
        self.primary_combo = QComboBox()
        self.primary_combo.currentIndexChanged.connect(self._on_primary_changed)
        primary_layout.addWidget(self.primary_combo)
        selection_layout.addLayout(primary_layout)

        # Secondary image selector
        secondary_layout = QHBoxLayout()
        secondary_layout.addWidget(QLabel("Secondary:"))
        self.secondary_combo = QComboBox()
        self.secondary_combo.currentIndexChanged.connect(self._on_secondary_changed)
        secondary_layout.addWidget(self.secondary_combo)
        selection_layout.addLayout(secondary_layout)

        selection_group.setLayout(selection_layout)
        controls_layout.addWidget(selection_group)

        # Registration controls group
        reg_group = QGroupBox("Registration Controls")
        reg_layout = QVBoxLayout()

        # Registration method selector
        reg_method_layout = QHBoxLayout()
        reg_method_layout.addWidget(QLabel("Method:"))
        self.reg_method_combo = QComboBox()
        for method in RegistrationMethod:
            self.reg_method_combo.addItem(method.value, method)
        self.reg_method_combo.currentIndexChanged.connect(self._on_reg_method_changed)
        reg_method_layout.addWidget(self.reg_method_combo)
        reg_layout.addLayout(reg_method_layout)

        # Create registration method specific controls (initially showing manual controls)
        self.reg_controls_stack = QTabWidget()

        # Manual registration controls
        manual_widget = QWidget()
        manual_layout = QGridLayout(manual_widget)

        # Translations
        manual_layout.addWidget(QLabel("Translation (mm):"), 0, 0)
        manual_layout.addWidget(QLabel("X:"), 1, 0)
        self.trans_x_spin = QDoubleSpinBox()
        self.trans_x_spin.setRange(-500, 500)
        self.trans_x_spin.setSingleStep(1)
        self.trans_x_spin.valueChanged.connect(self._on_manual_transform_changed)
        manual_layout.addWidget(self.trans_x_spin, 1, 1)

        manual_layout.addWidget(QLabel("Y:"), 2, 0)
        self.trans_y_spin = QDoubleSpinBox()
        self.trans_y_spin.setRange(-500, 500)
        self.trans_y_spin.setSingleStep(1)
        self.trans_y_spin.valueChanged.connect(self._on_manual_transform_changed)
        manual_layout.addWidget(self.trans_y_spin, 2, 1)

        manual_layout.addWidget(QLabel("Z:"), 3, 0)
        self.trans_z_spin = QDoubleSpinBox()
        self.trans_z_spin.setRange(-500, 500)
        self.trans_z_spin.setSingleStep(1)
        self.trans_z_spin.valueChanged.connect(self._on_manual_transform_changed)
        manual_layout.addWidget(self.trans_z_spin, 3, 1)

        # Rotations
        manual_layout.addWidget(QLabel("Rotation (deg):"), 0, 2)
        manual_layout.addWidget(QLabel("X:"), 1, 2)
        self.rot_x_spin = QDoubleSpinBox()
        self.rot_x_spin.setRange(-180, 180)
        self.rot_x_spin.setSingleStep(1)
        self.rot_x_spin.valueChanged.connect(self._on_manual_transform_changed)
        manual_layout.addWidget(self.rot_x_spin, 1, 3)

        manual_layout.addWidget(QLabel("Y:"), 2, 2)
        self.rot_y_spin = QDoubleSpinBox()
        self.rot_y_spin.setRange(-180, 180)
        self.rot_y_spin.setSingleStep(1)
        self.rot_y_spin.valueChanged.connect(self._on_manual_transform_changed)
        manual_layout.addWidget(self.rot_y_spin, 2, 3)

        manual_layout.addWidget(QLabel("Z:"), 3, 2)
        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(-180, 180)
        self.rot_z_spin.setSingleStep(1)
        self.rot_z_spin.valueChanged.connect(self._on_manual_transform_changed)
        manual_layout.addWidget(self.rot_z_spin, 3, 3)

        # Quick adjustment buttons
        adjust_layout = QGridLayout()

        # X translation
        x_minus_btn = QPushButton("X-")
        x_minus_btn.clicked.connect(lambda: self._adjust_transform("trans_x", -1))
        adjust_layout.addWidget(x_minus_btn, 0, 0)

        x_plus_btn = QPushButton("X+")
        x_plus_btn.clicked.connect(lambda: self._adjust_transform("trans_x", 1))
        adjust_layout.addWidget(x_plus_btn, 0, 1)

        # Y translation
        y_minus_btn = QPushButton("Y-")
        y_minus_btn.clicked.connect(lambda: self._adjust_transform("trans_y", -1))
        adjust_layout.addWidget(y_minus_btn, 1, 0)

        y_plus_btn = QPushButton("Y+")
        y_plus_btn.clicked.connect(lambda: self._adjust_transform("trans_y", 1))
        adjust_layout.addWidget(y_plus_btn, 1, 1)

        # Z translation
        z_minus_btn = QPushButton("Z-")
        z_minus_btn.clicked.connect(lambda: self._adjust_transform("trans_z", -1))
        adjust_layout.addWidget(z_minus_btn, 2, 0)

        z_plus_btn = QPushButton("Z+")
        z_plus_btn.clicked.connect(lambda: self._adjust_transform("trans_z", 1))
        adjust_layout.addWidget(z_plus_btn, 2, 1)

        # Rotations
        rx_minus_btn = QPushButton("RX-")
        rx_minus_btn.clicked.connect(lambda: self._adjust_transform("rot_x", -1))
        adjust_layout.addWidget(rx_minus_btn, 0, 2)

        rx_plus_btn = QPushButton("RX+")
        rx_plus_btn.clicked.connect(lambda: self._adjust_transform("rot_x", 1))
        adjust_layout.addWidget(rx_plus_btn, 0, 3)

        ry_minus_btn = QPushButton("RY-")
        ry_minus_btn.clicked.connect(lambda: self._adjust_transform("rot_y", -1))
        adjust_layout.addWidget(ry_minus_btn, 1, 2)

        ry_plus_btn = QPushButton("RY+")
        ry_plus_btn.clicked.connect(lambda: self._adjust_transform("rot_y", 1))
        adjust_layout.addWidget(ry_plus_btn, 1, 3)

        rz_minus_btn = QPushButton("RZ-")
        rz_minus_btn.clicked.connect(lambda: self._adjust_transform("rot_z", -1))
        adjust_layout.addWidget(rz_minus_btn, 2, 2)

        rz_plus_btn = QPushButton("RZ+")
        rz_plus_btn.clicked.connect(lambda: self._adjust_transform("rot_z", 1))
        adjust_layout.addWidget(rz_plus_btn, 2, 3)

        manual_layout.addLayout(adjust_layout, 4, 0, 1, 4)

        # Reset button
        reset_btn = QPushButton("Reset Transform")
        reset_btn.clicked.connect(self._reset_transform)
        manual_layout.addWidget(reset_btn, 5, 0, 1, 4)

        self.reg_controls_stack.addTab(manual_widget, "Manual")

        # Automatic registration tab
        auto_widget = QWidget()
        auto_layout = QVBoxLayout(auto_widget)

        # Registration parameters
        param_group = QGroupBox("Registration Parameters")
        param_layout = QFormLayout()

        # Resolution level
        self.res_level_combo = QComboBox()
        self.res_level_combo.addItems(["Coarse", "Medium", "Fine", "Ultra Fine"])
        self.res_level_combo.setCurrentIndex(1)  # Medium by default
        param_layout.addRow("Resolution Level:", self.res_level_combo)

        # Metric
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(
            ["Mutual Information", "Normalized Cross Correlation", "Mean Squares"]
        )
        param_layout.addRow("Similarity Metric:", self.metric_combo)

        # Max iterations
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(100)
        param_layout.addRow("Max Iterations:", self.max_iter_spin)

        param_group.setLayout(param_layout)
        auto_layout.addWidget(param_group)

        # ROI-based registration option
        self.roi_reg_check = QCheckBox("Use ROI for Registration")
        self.roi_reg_check.setToolTip(
            "Limit registration to a specific region of interest"
        )
        auto_layout.addWidget(self.roi_reg_check)

        # ROI selector (only enabled if roi_reg_check is checked)
        self.roi_selector = QComboBox()
        self.roi_selector.setEnabled(False)
        self.roi_reg_check.toggled.connect(self.roi_selector.setEnabled)
        auto_layout.addWidget(self.roi_selector)

        # Run auto registration button
        run_auto_btn = QPushButton("Run Automatic Registration")
        run_auto_btn.clicked.connect(self._run_auto_registration)
        auto_layout.addWidget(run_auto_btn)

        auto_layout.addStretch()

        self.reg_controls_stack.addTab(auto_widget, "Automatic")

        # Point-based registration tab
        point_widget = QWidget()
        point_layout = QVBoxLayout(point_widget)

        # Instructions
        point_layout.addWidget(
            QLabel("Create corresponding points in both image sets:")
        )

        # Points table
        self.points_table = QTableWidget(0, 3)
        self.points_table.setHorizontalHeaderLabels(
            ["Point Name", "Primary (mm)", "Secondary (mm)"]
        )
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        point_layout.addWidget(self.points_table)

        # Point control buttons
        point_btn_layout = QHBoxLayout()

        add_point_btn = QPushButton("Add Point")
        add_point_btn.clicked.connect(self._add_registration_point)
        point_btn_layout.addWidget(add_point_btn)

        remove_point_btn = QPushButton("Remove Point")
        remove_point_btn.clicked.connect(self._remove_registration_point)
        point_btn_layout.addWidget(remove_point_btn)

        point_layout.addLayout(point_btn_layout)

        # Run point-based registration button
        run_point_btn = QPushButton("Run Point-based Registration")
        run_point_btn.clicked.connect(self._run_point_registration)
        point_layout.addWidget(run_point_btn)

        self.reg_controls_stack.addTab(point_widget, "Point-based")

        reg_layout.addWidget(self.reg_controls_stack)

        # Registration actions
        reg_actions_layout = QHBoxLayout()

        apply_btn = QPushButton("Apply Registration")
        apply_btn.clicked.connect(self._apply_registration)
        reg_actions_layout.addWidget(apply_btn)

        save_btn = QPushButton("Save Registration")
        save_btn.clicked.connect(self._save_registration)
        reg_actions_layout.addWidget(save_btn)

        reg_layout.addLayout(reg_actions_layout)

        reg_group.setLayout(reg_layout)
        controls_layout.addWidget(reg_group)

        # Fusion controls group
        fusion_group = QGroupBox("Fusion Controls")
        fusion_layout = QVBoxLayout()

        # Fusion display mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Display Mode:"))
        self.fusion_mode_combo = QComboBox()
        for mode in FusionLayout:
            self.fusion_mode_combo.addItem(mode.value, mode)
        self.fusion_mode_combo.currentIndexChanged.connect(self._on_fusion_mode_changed)
        mode_layout.addWidget(self.fusion_mode_combo)
        fusion_layout.addLayout(mode_layout)

        # Fusion colormap
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color Scheme:"))
        self.colormap_combo = QComboBox()
        for cmap in FusionColormap:
            self.colormap_combo.addItem(cmap.value, cmap)
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        color_layout.addWidget(self.colormap_combo)
        fusion_layout.addLayout(color_layout)

        # Fusion blending control
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(QLabel("Primary"))

        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)  # Default 50% blend
        self.blend_slider.valueChanged.connect(self._on_blend_changed)
        blend_layout.addWidget(self.blend_slider)

        blend_layout.addWidget(QLabel("Secondary"))
        fusion_layout.addLayout(blend_layout)

        fusion_group.setLayout(fusion_layout)
        controls_layout.addWidget(fusion_group)

        # Add stretch to push controls to the top
        controls_layout.addStretch()

        # Right side: Image viewers
        viewers_widget = QWidget()
        viewers_layout = QVBoxLayout(viewers_widget)
        viewers_layout.setContentsMargins(0, 0, 0, 0)

        # Fusion display tabs
        self.view_tabs = QTabWidget()

        # MPR views (axial, sagittal, coronal)
        self.mpr_widget = MPRDisplay()
        self.view_tabs.addTab(self.mpr_widget, "MPR Views")

        # Side-by-side view
        self.side_widget = QWidget()
        side_layout = QHBoxLayout(self.side_widget)
        side_layout.setContentsMargins(0, 0, 0, 0)

        self.primary_viewer = ImageViewer()
        self.secondary_viewer = ImageViewer()

        side_layout.addWidget(self.primary_viewer)
        side_layout.addWidget(self.secondary_viewer)

        self.view_tabs.addTab(self.side_widget, "Side by Side")

        # Add viewer tabs to layout
        viewers_layout.addWidget(self.view_tabs)

        # Add widgets to splitter
        main_splitter.addWidget(controls_widget)
        main_splitter.addWidget(viewers_widget)

        # Set initial sizes (30% controls, 70% viewers)
        main_splitter.setSizes([300, 700])

        main_layout.addWidget(main_splitter)

    def set_images(
        self, primary_image: DicomSeries, secondary_images: Dict[str, DicomSeries]
    ):
        """
        Set the images for registration.

        Args:
            primary_image: Primary image series
            secondary_images: Dictionary of secondary image series keyed by name
        """
        self.primary_image = primary_image
        self.secondary_images = secondary_images

        # Update image selectors
        self._update_image_selectors()

        # Update viewers
        self._update_viewers()

    def _update_image_selectors(self):
        """Update the image selection comboboxes."""
        # Block signals to prevent triggering selection change events
        self.primary_combo.blockSignals(True)
        self.secondary_combo.blockSignals(True)

        # Clear existing items
        self.primary_combo.clear()
        self.secondary_combo.clear()

        # Add all images to both selectors
        if self.primary_image:
            self.primary_combo.addItem(
                self.primary_image.series_description or "Primary Image",
                self.primary_image,
            )

        for name, image in self.secondary_images.items():
            self.primary_combo.addItem(name, image)
            self.secondary_combo.addItem(name, image)

        # Select initial images
        self.primary_combo.setCurrentIndex(0)
        if self.secondary_combo.count() > 0:
            self.secondary_combo.setCurrentIndex(0)
            self.current_secondary = self.secondary_combo.currentData()

        # Re-enable signals
        self.primary_combo.blockSignals(False)
        self.secondary_combo.blockSignals(False)

        # Update ROI selector
        self._update_roi_selector()

    def _update_roi_selector(self):
        """Update the ROI selector with available structures."""
        self.roi_selector.clear()

        if (
            hasattr(self, "parent")
            and self.parent
            and hasattr(self.parent, "current_patient")
        ):
            patient = self.parent.current_patient
            if patient and hasattr(patient, "structures"):
                for structure in patient.structures:
                    self.roi_selector.addItem(structure.name, structure)

    def _on_primary_changed(self, index):
        """Handle primary image selection change."""
        self.primary_image = self.primary_combo.currentData()
        self._update_viewers()

    def _on_secondary_changed(self, index):
        """Handle secondary image selection change."""
        self.current_secondary = self.secondary_combo.currentData()
        self._update_viewers()
        self._reset_transform()

    def _update_viewers(self):
        """Update the image viewers with current images."""
        if not self.primary_image or not self.current_secondary:
            return

        # Update MPR view
        if hasattr(self.mpr_widget, "set_primary_image"):
            self.mpr_widget.set_primary_image(self.primary_image)

        if hasattr(self.mpr_widget, "set_secondary_image"):
            self.mpr_widget.set_secondary_image(self.current_secondary)

        if hasattr(self.mpr_widget, "set_fusion_parameters"):
            self.mpr_widget.set_fusion_parameters(
                alpha=self.alpha_blend,
                mode=self.current_fusion_mode.value,
                colormap=self.current_colormap.value,
            )

        # Update side-by-side view
        if self.primary_image.image_data is not None:
            self.primary_viewer.set_image(self.primary_image.image_data)

        if self.current_secondary.image_data is not None:
            self.secondary_viewer.set_image(self.current_secondary.image_data)

    def _on_reg_method_changed(self, index):
        """Handle registration method selection change."""
        # Show the appropriate tab based on the selected method
        method = self.reg_method_combo.currentData()

        if method == RegistrationMethod.MANUAL:
            self.reg_controls_stack.setCurrentIndex(0)
        elif method == RegistrationMethod.AUTO:
            self.reg_controls_stack.setCurrentIndex(1)
        elif method == RegistrationMethod.POINT_BASED:
            self.reg_controls_stack.setCurrentIndex(2)
        elif method == RegistrationMethod.RIGID:
            # Rigid uses the automatic tab with limited parameters
            self.reg_controls_stack.setCurrentIndex(1)
        elif method == RegistrationMethod.DEFORMABLE:
            # Deformable uses the automatic tab with different parameters
            self.reg_controls_stack.setCurrentIndex(1)

    def _on_manual_transform_changed(self):
        """Handle manual transformation parameter changes."""
        if not self.primary_image or not self.current_secondary:
            return

        # Create transformation matrix
        transform = self._get_current_transform_matrix()

        # Update registration and viewers
        self._update_registration(transform)

    def _get_current_transform_matrix(self) -> np.ndarray:
        """
        Get the current transformation matrix from UI parameters.

        Returns:
            4x4 transformation matrix
        """
        # Get translation parameters
        tx = self.trans_x_spin.value()
        ty = self.trans_y_spin.value()
        tz = self.trans_z_spin.value()

        # Get rotation parameters (in degrees)
        rx = np.radians(self.rot_x_spin.value())
        ry = np.radians(self.rot_y_spin.value())
        rz = np.radians(self.rot_z_spin.value())

        # Create rotation matrices
        Rx = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(rx), -np.sin(rx), 0],
                [0, np.sin(rx), np.cos(rx), 0],
                [0, 0, 0, 1],
            ]
        )

        Ry = np.array(
            [
                [np.cos(ry), 0, np.sin(ry), 0],
                [0, 1, 0, 0],
                [-np.sin(ry), 0, np.cos(ry), 0],
                [0, 0, 0, 1],
            ]
        )

        Rz = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0, 0],
                [np.sin(rz), np.cos(rz), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Create translation matrix
        T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

        # Combine transformations: T * Rz * Ry * Rx
        transform = T @ Rz @ Ry @ Rx

        return transform

    def _update_registration(self, transform: np.ndarray):
        """
        Update the registration with a new transformation matrix.

        Args:
            transform: 4x4 transformation matrix
        """
        if not self.primary_image or not self.current_secondary:
            return

        # Store the transform in registration results
        key = (self.primary_image.series_uid, self.current_secondary.series_uid)
        self.registration_results[key] = transform

        # Update the fusion engine
        self.fusion_engine.set_transform(transform)

        # Update viewers
        if hasattr(self.mpr_widget, "update_transform"):
            self.mpr_widget.update_transform(transform)

        # Emit signal
        self.fusion_updated.emit()

    def _adjust_transform(self, param: str, delta: float):
        """
        Adjust a transformation parameter by a delta value.

        Args:
            param: Parameter to adjust (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)
            delta: Amount to adjust by
        """
        # Determine step size based on parameter type
        step = (
            1.0 if param.startswith("trans") else 1.0
        )  # 1mm for translation, 1deg for rotation

        # Adjust the corresponding spinbox
        if param == "trans_x":
            self.trans_x_spin.setValue(self.trans_x_spin.value() + delta * step)
        elif param == "trans_y":
            self.trans_y_spin.setValue(self.trans_y_spin.value() + delta * step)
        elif param == "trans_z":
            self.trans_z_spin.setValue(self.trans_z_spin.value() + delta * step)
        elif param == "rot_x":
            self.rot_x_spin.setValue(self.rot_x_spin.value() + delta * step)
        elif param == "rot_y":
            self.rot_y_spin.setValue(self.rot_y_spin.value() + delta * step)
        elif param == "rot_z":
            self.rot_z_spin.setValue(self.rot_z_spin.value() + delta * step)

    def _reset_transform(self):
        """Reset the transformation to identity."""
        # Reset spinboxes to zero
        self.trans_x_spin.setValue(0)
        self.trans_y_spin.setValue(0)
        self.trans_z_spin.setValue(0)
        self.rot_x_spin.setValue(0)
        self.rot_y_spin.setValue(0)
        self.rot_z_spin.setValue(0)

        # Apply identity transform
        identity = np.identity(4)
        self._update_registration(identity)

    def _run_auto_registration(self):
        """Run automatic image registration."""
        if not self.primary_image or not self.current_secondary:
            return

        # Get registration parameters
        resolution_level = self.res_level_combo.currentIndex()
        metric = self.metric_combo.currentText()
        max_iterations = self.max_iter_spin.value()

        # Get ROI mask if selected
        roi_mask = None
        if self.roi_reg_check.isChecked() and self.roi_selector.currentData():
            roi = self.roi_selector.currentData()
            if hasattr(roi, "get_mask"):
                roi_mask = roi.get_mask()

        # Show progress indicator
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Call registration algorithm (placeholder - actual implementation depends on backend)
            # In a real implementation, this would call an actual registration algorithm
            logger.info(
                f"Running auto registration with resolution level {resolution_level}, "
                f"metric {metric}, max iterations {max_iterations}"
            )

            # Simulate a registration result (identity transform with slight offset)
            # In a real implementation, this would be the result of the registration algorithm
            transform = np.identity(4)
            transform[0, 3] = 5.0  # Add 5mm X translation as an example

            # Update registration with the result
            self._update_registration(transform)

            # Update UI to reflect the result
            self.trans_x_spin.setValue(transform[0, 3])
            self.trans_y_spin.setValue(transform[1, 3])
            self.trans_z_spin.setValue(transform[2, 3])

        except Exception as e:
            logger.error(f"Auto registration failed: {e}")
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def _add_registration_point(self):
        """Add a new point for point-based registration."""
        row = self.points_table.rowCount()
        self.points_table.insertRow(row)

        # Point name
        name_item = QTableWidgetItem(f"Point {row + 1}")
        self.points_table.setItem(row, 0, name_item)

        # Primary coordinates (initially empty)
        primary_item = QTableWidgetItem("Click to set")
        primary_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.points_table.setItem(row, 1, primary_item)

        # Secondary coordinates (initially empty)
        secondary_item = QTableWidgetItem("Click to set")
        secondary_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.points_table.setItem(row, 2, secondary_item)

    def _remove_registration_point(self):
        """Remove the selected point from the point-based registration table."""
        selected_rows = self.points_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        # Remove selected rows in reverse order to avoid index issues
        for row in sorted([index.row() for index in selected_rows], reverse=True):
            self.points_table.removeRow(row)

    def _run_point_registration(self):
        """Run point-based registration using the defined points."""
        # Need at least 3 points for 3D registration
        if self.points_table.rowCount() < 3:
            logger.warning("At least 3 points are needed for point-based registration")
            return

        # Collect point pairs
        points_primary = []
        points_secondary = []

        for row in range(self.points_table.rowCount()):
            primary_text = self.points_table.item(row, 1).text()
            secondary_text = self.points_table.item(row, 2).text()

            # Skip points that haven't been set
            if primary_text == "Click to set" or secondary_text == "Click to set":
                continue

            try:
                # Parse coordinates from text (format: "X, Y, Z")
                primary_coords = [float(x) for x in primary_text.split(",")]
                secondary_coords = [float(x) for x in secondary_text.split(",")]

                if len(primary_coords) == 3 and len(secondary_coords) == 3:
                    points_primary.append(primary_coords)
                    points_secondary.append(secondary_coords)
            except ValueError:
                logger.warning(f"Invalid coordinate format at row {row}")

        # Check if we have enough valid points
        if len(points_primary) < 3:
            logger.warning(
                "At least 3 valid points are needed for point-based registration"
            )
            return

        # Show progress indicator
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Convert to numpy arrays
            primary_array = np.array(points_primary)
            secondary_array = np.array(points_secondary)

            # Call point-based registration algorithm (placeholder)
            # In a real implementation, this would calculate a transformation matrix
            # based on the corresponding points
            logger.info(
                f"Running point-based registration with {len(primary_array)} points"
            )

            # Simulate a registration result (identity transform with slight offset)
            # In a real implementation, this would be the result of the registration algorithm
            transform = np.identity(4)
            transform[0, 3] = 2.0  # Add 2mm X translation as an example

            # Update registration with the result
            self._update_registration(transform)

            # Update UI to reflect the result
            self.trans_x_spin.setValue(transform[0, 3])
            self.trans_y_spin.setValue(transform[1, 3])
            self.trans_z_spin.setValue(transform[2, 3])

        except Exception as e:
            logger.error(f"Point-based registration failed: {e}")
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def _apply_registration(self):
        """Apply the current registration to the images."""
        if not self.primary_image or not self.current_secondary:
            return

        key = (self.primary_image.series_uid, self.current_secondary.series_uid)
        if key not in self.registration_results:
            return

        transform = self.registration_results[key]

        # Emit signal to notify parent that registration has been completed
        self.registration_completed.emit(
            self.primary_image.series_uid, self.current_secondary.series_uid, transform
        )

    def _save_registration(self):
        """Save the current registration to a file for future use."""
        if not self.primary_image or not self.current_secondary:
            return

        key = (self.primary_image.series_uid, self.current_secondary.series_uid)
        if key not in self.registration_results:
            return

        # Save logic would be implemented here
        # For now, just log the action
        logger.info(
            "Registration save functionality will be implemented in a future update"
        )

    def _on_fusion_mode_changed(self, index):
        """Handle fusion display mode change."""
        self.current_fusion_mode = self.fusion_mode_combo.currentData()
        self._update_viewers()

    def _on_colormap_changed(self, index):
        """Handle colormap change."""
        self.current_colormap = self.colormap_combo.currentData()
        self._update_viewers()

    def _on_blend_changed(self, value):
        """Handle blend slider change."""
        self.alpha_blend = value / 100.0
        if hasattr(self.mpr_widget, "set_fusion_parameters"):
            self.mpr_widget.set_fusion_parameters(
                alpha=self.alpha_blend,
                mode=self.current_fusion_mode.value,
                colormap=self.current_colormap.value,
            )

        # Update fusion view
        self.fusion_updated.emit()
