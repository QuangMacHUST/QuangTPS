import logging
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from typing import Dict, List, Optional, Tuple, Any, Union, Set

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QGroupBox, 
    QComboBox, QLabel, QPushButton, QFrame, QSizePolicy,
    QTabWidget, QScrollArea, QStackedWidget, QToolButton, QMenu,
    QListWidget, QListWidgetItem, QToolBar, QAction, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QIcon, QFont, QPalette

from quangtps.core.types import Plan, Structure
from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh, calculate_dvh_metrics
from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class DVHPlot(QWidget):
    """
    Interactive DVH plot for displaying dose-volume histograms.
    
    This class provides a comprehensive visualization of dose-volume histograms
    with features similar to Eclipse, including:
    - Multiple display modes (cumulative/differential)
    - Customizable units (relative/absolute)
    - Structure selection
    - Plan comparison
    - Interactive point selection
    - Dose statistics
    """
    
    # Signal emitted when a point is selected on the DVH
    point_selected = pyqtSignal(str, float, float)  # structure_id, dose, volume
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.plans: Dict[str, Plan] = {}  # plan_id -> Plan
        self.main_plan: Optional[Plan] = None
        self.comparison_plans: Dict[str, Plan] = {}  # plan_id -> Plan
        
        self.dvh_data = DVHData()
        self.comparison_dvh_data: Dict[str, DVHData] = {}  # plan_id -> DVHData
        
        self.selected_structures: Set[str] = set()  # structure IDs
        self.hover_info = None  # Info about current hover point
        
        # Display options
        self.dvh_type = "Cumulative"  # "Cumulative" or "Differential"
        self.volume_type = "Relative"  # "Relative" or "Absolute"
        self.dose_type = "Absolute"    # "Absolute" or "Relative to Rx"
        self.show_grid = True
        self.show_legend = True
        self.show_comparison = True
        self.show_diff = False  # Show difference between plans
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top control panel
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_panel)
        
        # DVH type selector
        type_group = QGroupBox("DVH Type")
        type_layout = QVBoxLayout(type_group)
        self.dvh_type_combo = QComboBox()
        self.dvh_type_combo.addItems(["Cumulative", "Differential"])
        type_layout.addWidget(self.dvh_type_combo)
        control_layout.addWidget(type_group)
        
        # Volume units selector
        volume_group = QGroupBox("Volume")
        volume_layout = QVBoxLayout(volume_group)
        self.volume_type_combo = QComboBox()
        self.volume_type_combo.addItems(["Relative (%)", "Absolute (cc)"])
        volume_layout.addWidget(self.volume_type_combo)
        control_layout.addWidget(volume_group)
        
        # Dose units selector
        dose_group = QGroupBox("Dose")
        dose_layout = QVBoxLayout(dose_group)
        self.dose_type_combo = QComboBox()
        self.dose_type_combo.addItems(["Absolute (Gy)", "Relative to Rx (%)"])
        dose_layout.addWidget(self.dose_type_combo)
        control_layout.addWidget(dose_group)
        
        # Display options
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        
        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        display_layout.addWidget(self.grid_check)
        
        self.legend_check = QCheckBox("Legend")
        self.legend_check.setChecked(True)
        display_layout.addWidget(self.legend_check)
        
        control_layout.addWidget(display_group)
        
        # Comparison options
        comparison_group = QGroupBox("Comparison")
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.show_comparison_check = QCheckBox("Show Comparison")
        self.show_comparison_check.setChecked(True)
        comparison_layout.addWidget(self.show_comparison_check)
        
        self.show_diff_check = QCheckBox("Show Difference")
        self.show_diff_check.setChecked(False)
        comparison_layout.addWidget(self.show_diff_check)
        
        control_layout.addWidget(comparison_group)
        
        # Export button
        self.export_button = QPushButton("Export")
        self.export_button.setToolTip("Export DVH data to CSV")
        control_layout.addWidget(self.export_button)
        
        main_layout.addWidget(control_panel)
        
        # Main content area with plot and structure list
        content_layout = QHBoxLayout()
        
        # Left side - Structure list
        structure_panel = QGroupBox("Structures")
        structure_layout = QVBoxLayout(structure_panel)
        
        self.structure_list = QListWidget()
        self.structure_list.setSelectionMode(QListWidget.ExtendedSelection)
        structure_layout.addWidget(self.structure_list)
        
        # Structure filter controls
        filter_layout = QHBoxLayout()
        
        self.select_all_button = QPushButton("Select All")
        self.select_none_button = QPushButton("Select None")
        self.select_targets_button = QPushButton("Targets")
        self.select_oars_button = QPushButton("OARs")
        
        filter_layout.addWidget(self.select_all_button)
        filter_layout.addWidget(self.select_none_button)
        
        structure_layout.addLayout(filter_layout)
        
        filter_layout2 = QHBoxLayout()
        filter_layout2.addWidget(self.select_targets_button)
        filter_layout2.addWidget(self.select_oars_button)
        
        structure_layout.addLayout(filter_layout2)
        
        content_layout.addWidget(structure_panel, 1)
        
        # Right side - DVH plot
        plot_panel = QFrame()
        plot_panel.setFrameShape(QFrame.StyledPanel)
        plot_layout = QVBoxLayout(plot_panel)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        # Hover info label
        self.info_label = QLabel("Hover over the plot to see dose and volume values")
        self.info_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.info_label)
        
        content_layout.addWidget(plot_panel, 3)
        
        main_layout.addLayout(content_layout, 1)
        
        # Connect signals
        self.dvh_type_combo.currentTextChanged.connect(self._on_dvh_type_changed)
        self.volume_type_combo.currentTextChanged.connect(self._on_volume_type_changed)
        self.dose_type_combo.currentTextChanged.connect(self._on_dose_type_changed)
        self.grid_check.toggled.connect(self._on_grid_toggled)
        self.legend_check.toggled.connect(self._on_legend_toggled)
        self.show_comparison_check.toggled.connect(self._on_show_comparison_toggled)
        self.show_diff_check.toggled.connect(self._on_show_diff_toggled)
        self.export_button.clicked.connect(self._on_export_clicked)
        self.structure_list.itemChanged.connect(self._on_structure_selection_changed)
        
        # Setup matplotlib canvas interactions
        self._setup_canvas_interactions()
        
        # Initialize the plot
        self._setup_axes()
    
    def _setup_canvas_interactions(self):
        """Setup matplotlib canvas interactions for hover and click."""
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_click)
    
    def _setup_axes(self):
        """Initialize the plot axes."""
        self.axes = self.figure.add_subplot(111)
        
        # Set labels based on current settings
        self._update_axes_labels()
        
        # Set up grid
        self.axes.grid(self.show_grid, linestyle='--', alpha=0.7)
        
        # Empty plot
        self.canvas.draw()
    
    def _update_axes_labels(self):
        """Update axis labels based on current settings."""
        # X-axis (Dose)
        if self.dose_type == "Absolute":
            self.axes.set_xlabel("Dose (Gy)")
        else:
            self.axes.set_xlabel("Dose (% of Rx)")
        
        # Y-axis (Volume)
        if self.volume_type == "Relative":
            self.axes.set_ylabel("Volume (%)")
        else:
            self.axes.set_ylabel("Volume (cc)")
    
    def set_plan(self, plan: Plan):
        """
        Set the main plan to display.
        
        Args:
            plan: The plan to display
        """
        self.main_plan = plan
        self.plans[plan.plan_id] = plan
        
        # Calculate DVH data
        self._calculate_main_dvh()
        
        # Update structure list
        self._update_structure_list()
        
        # Update plot
        self._update_plot()
    
    def add_comparison_plan(self, plan: Plan):
        """
        Add a plan for comparison.
        
        Args:
            plan: The plan to add for comparison
        """
        self.comparison_plans[plan.plan_id] = plan
        self.plans[plan.plan_id] = plan
        
        # Calculate DVH data
        self._calculate_comparison_dvh(plan)
        
        # Update plot
        self._update_plot()
    
    def clear_comparison_plans(self):
        """Clear all comparison plans."""
        self.comparison_plans.clear()
        self.comparison_dvh_data.clear()
        
        # Update plot
        self._update_plot()
    
    def clear(self):
        """Clear all data and reset the plot."""
        self.main_plan = None
        self.plans.clear()
        self.comparison_plans.clear()
        self.dvh_data = DVHData()
        self.comparison_dvh_data.clear()
        self.selected_structures.clear()
        
        # Clear structure list
        self.structure_list.clear()
        
        # Reset plot
        self.axes.clear()
        self._setup_axes()
        self.canvas.draw()
    
    def _calculate_main_dvh(self):
        """Calculate DVH data for the main plan."""
        if not self.main_plan or not self.main_plan.dose or not self.main_plan.structure_set:
            logger.warning("Cannot calculate DVH: missing plan, dose, or structure set")
            return
        
        # Calculate DVH for all structures
        self.dvh_data = DVHData()
        
        for structure in self.main_plan.structure_set.structures:
            if structure.has_contours() and not structure.is_empty():
                dvh_curve = calculate_dvh(self.main_plan.dose, structure)
                if dvh_curve:
                    self.dvh_data.add_curve(dvh_curve)
    
    def _calculate_comparison_dvh(self, plan: Plan):
        """
        Calculate DVH data for a comparison plan.
        
        Args:
            plan: The comparison plan
        """
        if not plan or not plan.dose or not plan.structure_set:
            logger.warning(f"Cannot calculate comparison DVH for plan {plan.plan_id}")
            return
        
        # Calculate DVH for all structures
        dvh_data = DVHData()
        
        for structure in plan.structure_set.structures:
            if structure.has_contours() and not structure.is_empty():
                dvh_curve = calculate_dvh(plan.dose, structure)
                if dvh_curve:
                    dvh_data.add_curve(dvh_curve)
        
        self.comparison_dvh_data[plan.plan_id] = dvh_data
    
    def _update_structure_list(self):
        """Update the structure list widget."""
        self.structure_list.clear()
        
        if not self.main_plan or not self.main_plan.structure_set:
            return
        
        # Add all structures from the main plan
        for structure in self.main_plan.structure_set.structures:
            if structure.has_contours() and not structure.is_empty():
                item = QListWidgetItem(structure.name)
                item.setData(Qt.UserRole, structure.id)
                
                # Set checkable
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)  # Initially checked
                
                # Set structure color
                color = QColor(*structure.color) if hasattr(structure, 'color') else QColor(200, 200, 200)
                item.setForeground(color)
                
                self.structure_list.addItem(item)
                self.selected_structures.add(structure.id)
    
    def _on_structure_selection_changed(self, item: QListWidgetItem):
        """
        Handle structure selection change.
        
        Args:
            item: The changed list item
        """
        structure_id = item.data(Qt.UserRole)
        checked = item.checkState() == Qt.Checked
        
        if checked:
            self.selected_structures.add(structure_id)
        else:
            self.selected_structures.discard(structure_id)
        
        # Update plot
        self._update_plot()
    
    def _on_dvh_type_changed(self, dvh_type: str):
        """
        Handle DVH type change.
        
        Args:
            dvh_type: New DVH type
        """
        self.dvh_type = dvh_type
        self._update_plot()
    
    def _on_volume_type_changed(self, volume_type: str):
        """
        Handle volume type change.
        
        Args:
            volume_type: New volume type
        """
        self.volume_type = "Relative" if "Relative" in volume_type else "Absolute"
        self._update_axes_labels()
        self._update_plot()
    
    def _on_dose_type_changed(self, dose_type: str):
        """
        Handle dose type change.
        
        Args:
            dose_type: New dose type
        """
        self.dose_type = "Absolute" if "Absolute" in dose_type else "Relative"
        self._update_axes_labels()
        self._update_plot()
    
    def _on_grid_toggled(self, show_grid: bool):
        """
        Handle grid toggle.
        
        Args:
            show_grid: Whether to show the grid
        """
        self.show_grid = show_grid
        self.axes.grid(show_grid, linestyle='--', alpha=0.7)
        self.canvas.draw()
    
    def _on_legend_toggled(self, show_legend: bool):
        """
        Handle legend toggle.
        
        Args:
            show_legend: Whether to show the legend
        """
        self.show_legend = show_legend
        self._update_plot()
    
    def _on_show_comparison_toggled(self, show_comparison: bool):
        """
        Handle comparison toggle.
        
        Args:
            show_comparison: Whether to show comparison plans
        """
        self.show_comparison = show_comparison
        self._update_plot()
    
    def _on_show_diff_toggled(self, show_diff: bool):
        """
        Handle difference toggle.
        
        Args:
            show_diff: Whether to show difference between plans
        """
        self.show_diff = show_diff
        self._update_plot()
    
    def _on_export_clicked(self):
        """Handle export button click."""
        # Export DVH data to CSV
        pass
    
    def _on_mouse_move(self, event):
        """
        Handle mouse move over the plot.
        
        Args:
            event: Matplotlib mouse event
        """
        if event.inaxes != self.axes:
            return
        
        # Format position info
        dose = event.xdata
        volume = event.ydata
        
        if dose is None or volume is None:
            return
        
        # Display position info
        self.info_label.setText(f"Dose: {dose:.2f} Gy, Volume: {volume:.2f}%")
        
        # Store hover info
        self.hover_info = (dose, volume)
    
    def _on_mouse_click(self, event):
        """
        Handle mouse click on the plot.
        
        Args:
            event: Matplotlib mouse event
        """
        if event.inaxes != self.axes or not self.main_plan:
            return
        
        # Left click
        if event.button == 1:
            # Find closest DVH curve and point
            closest_structure = None
            min_distance = float('inf')
            
            dose = event.xdata
            volume = event.ydata
            
            for structure_id in self.selected_structures:
                if structure_id in self.dvh_data.curves:
                    curve = self.dvh_data.curves[structure_id]
                    
                    # Find closest point on the curve
                    x_data, y_data = self._get_plot_data(curve)
                    
                    for i in range(len(x_data)):
                        d = np.sqrt((x_data[i] - dose)**2 + (y_data[i] - volume)**2)
                        if d < min_distance:
                            min_distance = d
                            closest_structure = structure_id
            
            # If a structure was found within reasonable distance
            if closest_structure and min_distance < 5:
                self.point_selected.emit(closest_structure, dose, volume)
    
    def _update_plot(self):
        """Update the plot with current data and settings."""
        if not self.main_plan:
            return
        
        # Clear the plot
        self.axes.clear()
        
        # Set labels
        self._update_axes_labels()
        
        # Set up grid
        self.axes.grid(self.show_grid, linestyle='--', alpha=0.7)
        
        # Plot main plan
        self._plot_dvh(self.dvh_data, is_main=True)
        
        # Plot comparison plans
        if self.show_comparison and self.comparison_plans:
            for plan_id, dvh_data in self.comparison_dvh_data.items():
                plan_name = self.plans[plan_id].name if hasattr(self.plans[plan_id], 'name') else plan_id
                self._plot_dvh(dvh_data, is_main=False, label_prefix=f"{plan_name}: ")
                
                # Plot difference if requested
                if self.show_diff:
                    self._plot_dvh_difference(self.dvh_data, dvh_data)
        
        # Add legend if enabled
        if self.show_legend:
            self.axes.legend(loc='upper right')
        
        # Set axis limits
        self._set_axis_limits()
        
        # Update the plot
        self.canvas.draw()
    
    def _plot_dvh(self, dvh_data: DVHData, is_main: bool = True, label_prefix: str = ""):
        """
        Plot DVH curves from given DVH data.
        
        Args:
            dvh_data: DVH data to plot
            is_main: Whether this is the main plan (affects line style)
            label_prefix: Prefix for structure labels in the legend
        """
        if not dvh_data:
            return
        
        # Plot selected structures
        for structure_id in self.selected_structures:
            if structure_id in dvh_data.curves:
                curve = dvh_data.curves[structure_id]
                
                # Get plot data
                x_data, y_data = self._get_plot_data(curve)
                
                # Get structure color
                color = curve.color if hasattr(curve, 'color') and curve.color else [0.5, 0.5, 0.5]
                
                # Set line style based on plan type
                linestyle = '-' if is_main else '--'
                linewidth = 2 if is_main else 1.5
                
                # Plot the curve
                self.axes.plot(
                    x_data, y_data,
                    label=f"{label_prefix}{curve.structure_name}",
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth
                )
    
    def _plot_dvh_difference(self, main_dvh: DVHData, comparison_dvh: DVHData):
        """
        Plot difference between main and comparison DVH.
        
        Args:
            main_dvh: Main DVH data
            comparison_dvh: Comparison DVH data
        """
        if not main_dvh or not comparison_dvh:
            return
        
        # Plot difference for selected structures
        for structure_id in self.selected_structures:
            if structure_id in main_dvh.curves and structure_id in comparison_dvh.curves:
                main_curve = main_dvh.curves[structure_id]
                comp_curve = comparison_dvh.curves[structure_id]
                
                # Get plot data
                main_x, main_y = self._get_plot_data(main_curve)
                comp_x, comp_y = self._get_plot_data(comp_curve)
                
                # Interpolate to common x values
                x_common = np.linspace(
                    max(np.min(main_x), np.min(comp_x)),
                    min(np.max(main_x), np.max(comp_x)),
                    100
                )
                
                main_y_interp = np.interp(x_common, main_x, main_y)
                comp_y_interp = np.interp(x_common, comp_x, comp_y)
                
                # Calculate difference
                diff_y = main_y_interp - comp_y_interp
                
                # Get structure color
                color = main_curve.color if hasattr(main_curve, 'color') and main_curve.color else [0.5, 0.5, 0.5]
                
                # Plot the difference
                self.axes.plot(
                    x_common, diff_y,
                    label=f"Diff: {main_curve.structure_name}",
                    color=color,
                    linestyle=':',
                    linewidth=1,
                    alpha=0.7
                )
    
    def _get_plot_data(self, curve: DVHCurve) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the data to plot for a DVH curve.
        
        Args:
            curve: The DVH curve
            
        Returns:
            Tuple of (x_data, y_data) for plotting
        """
        if self.dvh_type == "Cumulative":
            x_data = curve.dose_bins
            y_data = curve.cumulative_volume
        else:
            x_data = curve.dose_bins[:-1]
            y_data = curve.differential_volume
        
        # Convert volume data if needed
        if self.volume_type == "Relative":
            y_data = y_data * 100.0  # Convert to percentage
        else:
            # Already in absolute units (cc)
            pass
        
        # Convert dose data if needed
        if self.dose_type == "Relative" and self.main_plan and hasattr(self.main_plan, 'prescription'):
            rx_dose = self.main_plan.prescription.dose
            if rx_dose > 0:
                x_data = x_data / rx_dose * 100.0  # Convert to percentage of prescription
        
        return x_data, y_data
    
    def _set_axis_limits(self):
        """Set appropriate axis limits for the plot."""
        # Y-axis limits
        if self.volume_type == "Relative":
            self.axes.set_ylim(0, 105)  # 0-105%
        else:
            # For absolute volume, use the maximum volume of any structure
            max_vol = 0
            for structure_id in self.selected_structures:
                if structure_id in self.dvh_data.curves:
                    curve = self.dvh_data.curves[structure_id]
                    vol = curve.volume
                    max_vol = max(max_vol, vol)
            
            if max_vol > 0:
                self.axes.set_ylim(0, max_vol * 1.05)  # Add 5% margin
        
        # X-axis limits
        max_dose = 0
        for structure_id in self.selected_structures:
            if structure_id in self.dvh_data.curves:
                curve = self.dvh_data.curves[structure_id]
                if len(curve.dose_bins) > 0:
                    max_dose = max(max_dose, np.max(curve.dose_bins))
        
        if max_dose > 0:
            if self.dose_type == "Absolute":
                self.axes.set_xlim(0, max_dose * 1.05)  # Add 5% margin
            else:
                # Relative to prescription
                rx_dose = 100.0
                if self.main_plan and hasattr(self.main_plan, 'prescription'):
                    rx_dose = self.main_plan.prescription.dose
                if rx_dose > 0:
                    max_dose_pct = max_dose / rx_dose * 100.0
                    self.axes.set_xlim(0, max_dose_pct * 1.05)  # Add 5% margin