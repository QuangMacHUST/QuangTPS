import logging
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, 
    QPushButton, QSplitter, QFrame, QToolBar, QAction, QSpinBox, 
    QDoubleSpinBox, QGridLayout, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QColor, QPen, QPixmap

from quangtps.ui.image_display import ImageView
from quangtps.imaging.image import Image
from quangtps.structures.structure_set import StructureSet
from quangtps.common.widgets import ToolButton

logger = logging.getLogger(__name__)

class EnhancedMPRViewer(QWidget):
    """
    Enhanced Multi-Planar Reconstruction viewer with Eclipse-like capabilities.
    Supports axial, sagittal, and coronal views with integrated structure overlay.
    
    This viewer is designed to mimic the Eclipse TPS MPR viewer with advanced
    features for viewing and interacting with structures.
    """
    slice_changed = pyqtSignal(int, str)  # view_index, view_name
    window_level_changed = pyqtSignal(int, int)  # window, level
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Image data
        self.image_data = None
        self.structure_sets = {}
        self.current_structure_set_id = None
        self.current_dose = None
        
        # View settings
        self.view_modes = ["MPR", "Axial Only", "Sagittal Only", "Coronal Only", "3D View"]
        self.window_presets = {
            "Default": (400, 40),    # window, level
            "Soft Tissue": (350, 50),
            "Lung": (1500, -500),
            "Bone": (2000, 500),
            "Head": (80, 40),
            "Abdomen": (400, 60),
            "Mediastinum": (400, 60),
            "Liver": (150, 30),
            "Brain": (80, 40)
        }
        
        # Initialize UI
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        
        # Create toolbar
        self.toolbar = self._create_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # Create splitter for views
        self.view_splitter = QSplitter(Qt.Horizontal)
        
        # Create MPR views
        self.views = []
        self.view_containers = []
        
        # Axial view (top left)
        axial_container = self._create_view_container(0, "Axial")
        self.view_splitter.addWidget(axial_container)
        
        # Right side container (for sagittal and coronal)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)
        
        # Sagittal view (top right)
        sagittal_container = self._create_view_container(1, "Sagittal")
        right_layout.addWidget(sagittal_container)
        
        # Coronal view (bottom right)
        coronal_container = self._create_view_container(2, "Coronal")
        right_layout.addWidget(coronal_container)
        
        # Add right container to splitter
        self.view_splitter.addWidget(right_container)
        
        # Set initial splitter sizes
        self.view_splitter.setSizes([500, 500])
        
        # Add views to main layout
        main_layout.addWidget(self.view_splitter)
        
        # Create 3D container (initially hidden)
        self.view_3d_container = self._create_3d_container()
        self.view_3d_container.setVisible(False)
        main_layout.addWidget(self.view_3d_container)
        
    def _create_toolbar(self):
        # Create toolbar
        toolbar = QToolBar("MPR Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        
        # View mode selection
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(self.view_modes)
        self.view_mode_combo.setCurrentIndex(0)
        self.view_mode_combo.setToolTip("Select view mode")
        toolbar.addWidget(QLabel("View: "))
        toolbar.addWidget(self.view_mode_combo)
        toolbar.addSeparator()
        
        # Window/level presets
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.window_presets.keys()))
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.setToolTip("Window/level presets")
        toolbar.addWidget(QLabel("Preset: "))
        toolbar.addWidget(self.preset_combo)
        toolbar.addSeparator()
        
        # Window/level controls
        toolbar.addWidget(QLabel("Window: "))
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setRange(1, 4000)
        self.window_spinbox.setValue(400)
        self.window_spinbox.setToolTip("Window width")
        toolbar.addWidget(self.window_spinbox)
        
        toolbar.addWidget(QLabel("Level: "))
        self.level_spinbox = QSpinBox()
        self.level_spinbox.setRange(-1000, 3000)
        self.level_spinbox.setValue(40)
        self.level_spinbox.setToolTip("Window level")
        toolbar.addWidget(self.level_spinbox)
        toolbar.addSeparator()
        
        # Structure visibility controls
        self.show_structures_btn = QPushButton("Structures")
        self.show_structures_btn.setCheckable(True)
        self.show_structures_btn.setChecked(True)
        self.show_structures_btn.setToolTip("Toggle structure visibility")
        toolbar.addWidget(self.show_structures_btn)
        
        # Dose visibility
        self.show_dose_btn = QPushButton("Dose")
        self.show_dose_btn.setCheckable(True)
        self.show_dose_btn.setChecked(False)
        self.show_dose_btn.setToolTip("Toggle dose display")
        toolbar.addWidget(self.show_dose_btn)
        
        # Isodose lines
        self.show_isodose_btn = QPushButton("Isodose")
        self.show_isodose_btn.setCheckable(True)
        self.show_isodose_btn.setChecked(False)
        self.show_isodose_btn.setToolTip("Toggle isodose lines")
        toolbar.addWidget(self.show_isodose_btn)
        
        # Grid display
        self.show_grid_btn = QPushButton("Grid")
        self.show_grid_btn.setCheckable(True)
        self.show_grid_btn.setChecked(True)
        self.show_grid_btn.setToolTip("Toggle grid display")
        toolbar.addWidget(self.show_grid_btn)
        toolbar.addSeparator()
        
        # Zoom controls
        self.zoom_in_btn = ToolButton()
        self.zoom_in_btn.setIcon(QIcon("quangtps/resources/icons/zoom_in.png"))
        self.zoom_in_btn.setToolTip("Zoom in")
        toolbar.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = ToolButton()
        self.zoom_out_btn.setIcon(QIcon("quangtps/resources/icons/zoom_out.png"))
        self.zoom_out_btn.setToolTip("Zoom out")
        toolbar.addWidget(self.zoom_out_btn)
        
        self.zoom_reset_btn = ToolButton()
        self.zoom_reset_btn.setIcon(QIcon("quangtps/resources/icons/zoom_reset.png"))
        self.zoom_reset_btn.setToolTip("Reset zoom")
        toolbar.addWidget(self.zoom_reset_btn)
        
        return toolbar
    
    def _create_view_container(self, index, view_name):
        container = QFrame()
        container.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Title bar
        title_bar = QWidget()
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(2, 2, 2, 2)
        title_bar_layout.setSpacing(2)
        
        # View title
        title_label = QLabel(view_name)
        title_label.setStyleSheet("font-weight: bold;")
        title_bar_layout.addWidget(title_label)
        
        # Slice spinner
        title_bar_layout.addStretch()
        title_bar_layout.addWidget(QLabel("Slice:"))
        slice_spinner = QSpinBox()
        slice_spinner.setRange(1, 1)
        slice_spinner.setEnabled(False)
        slice_spinner.setMinimumWidth(60)
        slice_spinner.setProperty("view_index", index)
        title_bar_layout.addWidget(slice_spinner)
        
        layout.addWidget(title_bar)
        
        # Image view
        view = ImageView()
        view.set_view_orientation(index)  # 0: axial, 1: sagittal, 2: coronal
        view.setMinimumSize(QSize(200, 200))
        layout.addWidget(view)
        
        # Slice slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 0)
        slider.setEnabled(False)
        slider.setProperty("view_index", index)
        layout.addWidget(slider)
        
        # Add orientation markers
        if index == 0:  # Axial
            markers = {"top": "A", "bottom": "P", "left": "R", "right": "L"}
        elif index == 1:  # Sagittal
            markers = {"top": "A", "bottom": "P", "left": "H", "right": "F"}
        else:  # Coronal
            markers = {"top": "H", "bottom": "F", "left": "R", "right": "L"}
        
        self._add_orientation_markers(layout, markers)
        
        # Store references
        self.views.append(view)
        self.view_containers.append({
            "container": container,
            "view": view,
            "slider": slider,
            "spinner": slice_spinner,
            "name": view_name
        })
        
        return container
    
    def _create_3d_container(self):
        container = QFrame()
        container.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Title bar
        title_bar = QWidget()
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(2, 2, 2, 2)
        title_bar_layout.setSpacing(2)
        
        # View title
        title_label = QLabel("3D View")
        title_label.setStyleSheet("font-weight: bold;")
        title_bar_layout.addWidget(title_label)
        
        layout.addWidget(title_bar)
        
        # Placeholder for 3D view
        placeholder = QLabel("3D Visualization will be displayed here")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("background-color: #333; color: white; font-size: 14px;")
        placeholder.setMinimumSize(QSize(300, 300))
        layout.addWidget(placeholder)
        
        return container
    
    def _add_orientation_markers(self, layout, markers):
        # Add orientation markers (A, P, R, L, H, F) around the view
        marker_layout = QGridLayout()
        marker_layout.setContentsMargins(0, 0, 0, 0)
        marker_layout.setSpacing(0)
        
        # Create markers
        top_marker = QLabel(markers["top"])
        top_marker.setAlignment(Qt.AlignCenter)
        bottom_marker = QLabel(markers["bottom"])
        bottom_marker.setAlignment(Qt.AlignCenter)
        left_marker = QLabel(markers["left"])
        left_marker.setAlignment(Qt.AlignCenter)
        right_marker = QLabel(markers["right"])
        right_marker.setAlignment(Qt.AlignCenter)
        
        # Add markers to layout
        marker_layout.addWidget(top_marker, 0, 1)
        marker_layout.addWidget(bottom_marker, 2, 1)
        marker_layout.addWidget(left_marker, 1, 0)
        marker_layout.addWidget(right_marker, 1, 2)
        
        # Add spacers
        spacer = QWidget()
        spacer.setMinimumSize(1, 1)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        marker_layout.addWidget(spacer, 1, 1)
        
        # Add marker layout to main layout
        layout.addLayout(marker_layout)
    
    def _connect_signals(self):
        # Connect view mode combo
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        
        # Connect window/level controls
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.window_spinbox.valueChanged.connect(
            lambda value: self.window_level_changed.emit(value, self.level_spinbox.value())
        )
        self.level_spinbox.valueChanged.connect(
            lambda value: self.window_level_changed.emit(self.window_spinbox.value(), value)
        )
        
        # Connect visibility toggles
        self.show_structures_btn.toggled.connect(self._on_show_structures_toggled)
        self.show_dose_btn.toggled.connect(self._on_show_dose_toggled)
        self.show_isodose_btn.toggled.connect(self._on_show_isodose_toggled)
        self.show_grid_btn.toggled.connect(self._on_show_grid_toggled)
        
        # Connect zoom controls
        self.zoom_in_btn.clicked.connect(lambda: self._on_zoom(1.2))
        self.zoom_out_btn.clicked.connect(lambda: self._on_zoom(0.8))
        self.zoom_reset_btn.clicked.connect(self._on_zoom_reset)
        
        # Connect sliders and spinners
        for i, container in enumerate(self.view_containers):
            container["slider"].valueChanged.connect(
                lambda value, idx=i: self._on_slice_changed(idx, value)
            )
            container["spinner"].valueChanged.connect(
                lambda value, idx=i: container["slider"].setValue(value - 1)
            )
    
    def set_image_data(self, image_data):
        """Set the image data to display in all views"""
        self.image_data = image_data
        
        if image_data is None:
            # Clear views if no image data
            for view in self.views:
                view.clear()
            return
            
        # Update all views with the new image data
        for i, view in enumerate(self.views):
            view.set_image_data(image_data, i)
            
        # Update sliders and spinners
        self._update_views()
    
    def add_structure_set(self, structure_set_id, structure_set):
        """Add a structure set to be displayed on the views"""
        self.structure_sets[structure_set_id] = structure_set
        
        # If this is the first structure set, make it current
        if self.current_structure_set_id is None:
            self.current_structure_set_id = structure_set_id
            
        # Update views to show the structures
        self._update_views()
    
    def set_current_structure_set(self, structure_set_id):
        """Set which structure set should be displayed"""
        if structure_set_id in self.structure_sets:
            self.current_structure_set_id = structure_set_id
            self._update_views()
    
    def _update_views(self):
        """Update all views with current data"""
        if self.image_data is None:
            return
            
        # Get image dimensions
        nx, ny, nz = self.image_data.get_dimensions()
        
        # Update sliders and spinners for each view
        for i, container in enumerate(self.view_containers):
            slider = container["slider"]
            spinner = container["spinner"]
            
            # Determine max slices based on orientation
            if i == 0:  # Axial (z)
                max_slice = nz
            elif i == 1:  # Sagittal (x)
                max_slice = nx
            else:  # Coronal (y)
                max_slice = ny
                
            # Update slider range
            slider.setRange(0, max_slice - 1)
            slider.setValue(max_slice // 2)
            slider.setEnabled(True)
            
            # Update spinner range
            spinner.setRange(1, max_slice)
            spinner.setValue(max_slice // 2 + 1)
            spinner.setEnabled(True)
            
            # Set slice in view
            self._on_slice_changed(i, max_slice // 2)
            
        # Set window/level from image data if available
        if hasattr(self.image_data, 'default_window'):
            window = self.image_data.default_window
            level = self.image_data.default_level
            self.window_spinbox.setValue(window)
            self.level_spinbox.setValue(level)
    
    def _add_structures_to_view(self, view_index, slice_pos, slice_axis):
        """Add structures to the specified view at the given slice position"""
        if not self.show_structures_btn.isChecked() or self.current_structure_set_id is None:
            return
            
        structure_set = self.structure_sets.get(self.current_structure_set_id)
        if structure_set is None:
            return
            
        # Add each structure to the view
        for structure in structure_set.get_structures():
            # Structure color logic would go here
            color = QColor(255, 0, 0)  # Default to red
            
            # Add structure contour to view
            self.views[view_index].add_structure_contour(structure, slice_pos, slice_axis, color)
    
    def _on_slice_changed(self, view_index, value):
        """Handle slice change in any view"""
        # Update the view with the new slice
        view = self.views[view_index]
        view_info = self.view_containers[view_index]
        
        # Determine slice axis based on view index
        slice_axis = view_index  # 0: axial (z), 1: sagittal (x), 2: coronal (y)
        
        # Set the slice in the view
        view.set_slice(value, slice_axis)
        
        # Update structures if needed
        self._add_structures_to_view(view_index, value, slice_axis)
        
        # Update spinner (adding 1 because spinner is 1-indexed)
        spinner = view_info["spinner"]
        if spinner.value() != value + 1:
            spinner.blockSignals(True)
            spinner.setValue(value + 1)
            spinner.blockSignals(False)
            
        # Emit signal
        self.slice_changed.emit(view_index, view_info["name"])
    
    def _on_view_mode_changed(self, index):
        """Handle view mode change"""
        mode = self.view_modes[index]
        
        # Show/hide views based on selected mode
        if mode == "MPR":
            # Show all MPR views, hide 3D
            self.view_splitter.setVisible(True)
            self.view_3d_container.setVisible(False)
            
            # Make all views visible
            for container in self.view_containers:
                container["container"].setVisible(True)
                
        elif mode == "Axial Only":
            # Show only axial view
            self.view_splitter.setVisible(True)
            self.view_3d_container.setVisible(False)
            
            # Show axial, hide others
            for i, container in enumerate(self.view_containers):
                container["container"].setVisible(i == 0)
                
        elif mode == "Sagittal Only":
            # Show only sagittal view
            self.view_splitter.setVisible(True)
            self.view_3d_container.setVisible(False)
            
            # Show sagittal, hide others
            for i, container in enumerate(self.view_containers):
                container["container"].setVisible(i == 1)
                
        elif mode == "Coronal Only":
            # Show only coronal view
            self.view_splitter.setVisible(True)
            self.view_3d_container.setVisible(False)
            
            # Show coronal, hide others
            for i, container in enumerate(self.view_containers):
                container["container"].setVisible(i == 2)
                
        elif mode == "3D View":
            # Show 3D view, hide MPR
            self.view_splitter.setVisible(False)
            self.view_3d_container.setVisible(True)
    
    def _on_preset_changed(self, index):
        """Handle window/level preset change"""
        preset_name = self.preset_combo.currentText()
        if preset_name in self.window_presets:
            window, level = self.window_presets[preset_name]
            
            # Update spinboxes (which will trigger valueChanged)
            self.window_spinbox.setValue(window)
            self.level_spinbox.setValue(level)
    
    def _on_show_structures_toggled(self, checked):
        """Toggle structure visibility"""
        # Update all views
        self._update_views()
    
    def _on_show_dose_toggled(self, checked):
        """Toggle dose display"""
        # Implementation would depend on dose visualization approach
        pass
    
    def _on_show_isodose_toggled(self, checked):
        """Toggle isodose lines"""
        # Implementation would depend on isodose visualization approach
        pass
    
    def _on_show_grid_toggled(self, checked):
        """Toggle grid display"""
        for view in self.views:
            view.set_grid_visible(checked)
    
    def _on_zoom(self, factor):
        """Zoom all views by the given factor"""
        for view in self.views:
            view.zoom(factor)
    
    def _on_zoom_reset(self):
        """Reset zoom level in all views"""
        for view in self.views:
            view.reset_zoom()
