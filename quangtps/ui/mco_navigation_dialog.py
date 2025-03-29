#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog điều hướng tối ưu đa tiêu chí (MCO) cho hệ thống QuangTPS.

Dialog này cho phép người dùng khám phá và điều hướng trên mặt Pareto, điều chỉnh
trọng số của các tiêu chí khác nhau và xem kết quả theo thời gian thực.
"""

import os
import logging
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                            QPushButton, QGroupBox, QSplitter, QTabWidget, 
                            QWidget, QComboBox, QFrame, QRadioButton, 
                            QButtonGroup, QMessageBox, QGridLayout, QScrollArea,
                            QSpacerItem, QSizePolicy, QCheckBox, QFileDialog,
                            QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap

from quangtps.optimization.methods.mco import MCOEngine, MCONavigator, MCOTrade
from quangtps.evaluation.dvh.dvh_visualization import plot_dvh
from quangtps.imaging.image_viewer import ImageViewer
from quangtps.dose.dose_visualization import DoseColorwash
from quangtps.core.exceptions import OptimizationError
from quangtps.dose.dose_grid import DoseGrid
from quangtps.ui.widgets.dvh_widget import DVHWidget
from quangtps.ui.styles import get_icon, Colors

logger = logging.getLogger(__name__)

class MCOTradeoffPlot(FigureCanvasQTAgg):
    """Interactive tradeoff plot for multi-criteria optimization.
    
    Displays the Pareto front and allows users to interactively explore
    the trade-offs between different objectives.
    """
    
    pointSelected = pyqtSignal(int)  # Signal emitted when a point is selected
    
    def __init__(self, width=6, height=5, dpi=100):
        plt.style.use('ggplot')  # Use a clean, modern style
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Enable better antialiasing for prettier plots
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # Interactive features
        self.point_details = []  # Store details about each point
        self.current_point_idx = None
        self.selected_point = None
        self.hover_annotation = None
        
        # Connect events for interaction
        self.mpl_connect('motion_notify_event', self.on_hover)
        self.mpl_connect('button_press_event', self.on_click)
        
        self.setMinimumSize(400, 300)
        
    def plot_tradeoff(self, x_values, y_values, x_label, y_label, 
                     current_point=None, point_details=None, title=None):
        """Plot the tradeoff between two objectives.
        
        Args:
            x_values: Values for the x-axis objective
            y_values: Values for the y-axis objective
            x_label: Label for x-axis
            y_label: Label for y-axis
            current_point: Current selected point (x,y)
            point_details: List of dictionaries with point metadata
            title: Plot title
        """
        self.axes.clear()
        self.point_details = point_details or []
        
        # Create color gradient for Pareto front
        n_points = len(x_values)
        if n_points > 0:
            # Create gradient from blue to green to represent domination
            colors = plt.cm.viridis(np.linspace(0, 0.8, n_points))
            
            # Plot Pareto front points with gradient coloring
            scatter = self.axes.scatter(x_values, y_values, s=60, alpha=0.8, 
                               c=colors, edgecolor='w', linewidth=1.5)
            
            # Plot connecting lines for Pareto front
            if n_points > 1:
                # Sort points by x value for proper line connection
                xy_points = np.array(list(zip(x_values, y_values)))
                sorted_idx = np.argsort(xy_points[:, 0])
                sorted_points = xy_points[sorted_idx]
                
                self.axes.plot(sorted_points[:, 0], sorted_points[:, 1], 
                              'k--', alpha=0.5, linewidth=1)
        
        # Mark current point if provided
        self.current_point_idx = None
        if current_point:
            # Find the index of the current point
            if n_points > 0:
                distances = [(x-current_point[0])**2 + (y-current_point[1])**2 
                            for x, y in zip(x_values, y_values)]
                self.current_point_idx = np.argmin(distances)
            
            # Mark the current point with a star
            self.selected_point = self.axes.scatter(
                [current_point[0]], [current_point[1]], 
                s=140, c='gold', marker='*', edgecolor='k', linewidth=1.5,
                zorder=10
            )
        
        # Set labels with larger font
        self.axes.set_xlabel(x_label, fontsize=10, fontweight='bold')
        self.axes.set_ylabel(y_label, fontsize=10, fontweight='bold')
        
        # Set title
        if title:
            self.axes.set_title(title, fontsize=12, fontweight='bold')
        else:
            self.axes.set_title(f"Tradeoff: {x_label} vs {y_label}", 
                              fontsize=12, fontweight='bold')
        
        # Add grid with specific styling
        self.axes.grid(True, linestyle='--', alpha=0.6, color='gray')
        
        # Set background color for the plotting area
        self.axes.set_facecolor('#f8f8f8')
        
        # Add a legend
        if n_points > 0:
            self.axes.legend(['Pareto Front', 'Current Selection'], 
                           loc='best', framealpha=0.7)
        
        # Add text explanations for better usability
        self.axes.text(0.01, 0.01, "Click a point to select it", 
                     transform=self.axes.transAxes, fontsize=8, 
                     verticalalignment='bottom', color='#555555')
        
        # Apply styling to axes
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#cccccc')
        
        self.fig.tight_layout()
        self.draw()
        
    def on_hover(self, event):
        """Handle mouse hover events to show tooltips."""
        if event.inaxes != self.axes or len(self.point_details) == 0:
            # Clear any existing annotation when mouse leaves axes
            if self.hover_annotation:
                self.hover_annotation.remove()
                self.hover_annotation = None
                self.draw_idle()
            return
        
        # Get data coordinates
        x, y = event.xdata, event.ydata
        
        # Find closest point
        min_dist = float('inf')
        closest_idx = -1
        
        for i, (xi, yi) in enumerate(zip(self.axes.get_lines()[0].get_xdata(), 
                                       self.axes.get_lines()[0].get_ydata())):
            dist = (xi - x)**2 + (yi - y)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Show tooltip if close enough
        if min_dist < 0.01 and closest_idx >= 0 and closest_idx < len(self.point_details):
            # Remove existing annotation
            if self.hover_annotation:
                self.hover_annotation.remove()
            
            # Create new annotation with point details
            details = self.point_details[closest_idx]
            text = "\n".join([f"{k}: {v:.3f}" for k, v in details.items()])
            self.hover_annotation = self.axes.annotate(
                text, xy=(self.axes.get_lines()[0].get_xdata()[closest_idx],
                         self.axes.get_lines()[0].get_ydata()[closest_idx]),
                xytext=(10, 10), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
            )
            self.draw_idle()
    
    def on_click(self, event):
        """Handle mouse click events to select points."""
        if event.inaxes != self.axes:
            return
        
        # Get data coordinates
        x, y = event.xdata, event.ydata
        
        # Find closest point
        min_dist = float('inf')
        closest_idx = -1
        
        if not hasattr(self.axes, 'lines') or len(self.axes.lines) == 0:
            return
            
        for i, (xi, yi) in enumerate(zip(self.axes.get_lines()[0].get_xdata(), 
                                       self.axes.get_lines()[0].get_ydata())):
            dist = (xi - x)**2 + (yi - y)**2
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Emit signal if close enough to a point
        if min_dist < 0.02 and closest_idx >= 0:
            self.pointSelected.emit(closest_idx)


class ObjectiveSlider(QWidget):
    """Interactive slider widget for controlling objective weights in MCO.
    
    This widget provides a slider with detailed information display and visual feedback
    about the importance of each objective in the multi-criteria optimization.
    """
    
    valueChanged = pyqtSignal(str, float)
    
    def __init__(self, objective_name, min_value=0, max_value=100, 
                default_value=50, description=None, objective_type=None, parent=None):
        super().__init__(parent)
        
        self.objective_name = objective_name
        self.description = description or objective_name
        self.objective_type = objective_type or "Generic"
        
        # Create the main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Create frame for better visual appearance
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.setSpacing(4)
        
        # Header with name and type
        header_layout = QHBoxLayout()
        
        # Icon based on objective type
        self.type_icon = QLabel()
        icon_size = 16
        
        # Set icon based on objective type
        if "Target" in self.objective_type:
            self.type_icon.setPixmap(get_icon("target").pixmap(icon_size, icon_size))
            self.color = "#E63946"  # Red for targets
        elif "OAR" in self.objective_type:
            self.type_icon.setPixmap(get_icon("shield").pixmap(icon_size, icon_size))
            self.color = "#457B9D"  # Blue for OARs
        else:
            self.type_icon.setPixmap(get_icon("objective").pixmap(icon_size, icon_size))
            self.color = "#2A9D8F"  # Green for others
        
        header_layout.addWidget(self.type_icon)
        
        # Objective name label with styling
        self.name_label = QLabel(self.description)
        self.name_label.setStyleSheet(f"font-weight: bold; color: {self.color};")
        self.name_label.setToolTip(f"Objective: {self.description}\nType: {self.objective_type}")
        header_layout.addWidget(self.name_label, 1)
        
        # Value display
        self.value_label = QLabel(f"{default_value}%")
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet("font-weight: bold;")
        self.value_label.setMinimumWidth(45)
        header_layout.addWidget(self.value_label)
        
        frame_layout.addLayout(header_layout)
        
        # Slider with tick marks
        slider_layout = QHBoxLayout()
        
        # Min value label
        min_label = QLabel(f"{min_value}")
        min_label.setAlignment(Qt.AlignLeft)
        slider_layout.addWidget(min_label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setValue(default_value)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        
        # Set custom stylesheet for the slider
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }}
            
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #F2F2F2, stop: 1 {self.color});
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }}
            
            QSlider::add-page:horizontal {{
                background: #fff;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }}
            
            QSlider::handle:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777;
                width: 18px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 9px;
            }}
        """)
        
        self.slider.valueChanged.connect(self._value_changed)
        slider_layout.addWidget(self.slider, 1)
        
        # Max value label
        max_label = QLabel(f"{max_value}")
        max_label.setAlignment(Qt.AlignRight)
        slider_layout.addWidget(max_label)
        
        frame_layout.addLayout(slider_layout)
        
        # Preset buttons for quick adjustment
        presets_layout = QHBoxLayout()
        presets_layout.setSpacing(4)
        
        # Create preset buttons with styling
        for preset_value, label in [(0, "Off"), (25, "Low"), (50, "Med"), (75, "High"), (100, "Max")]:
            btn = QPushButton(label)
            btn.setProperty("preset_value", preset_value)
            btn.setFixedHeight(22)
            btn.setFixedWidth(40)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #F8F9FA;
                    border: 1px solid #DEE2E6;
                    border-radius: 3px;
                    font-size: 9px;
                }}
                QPushButton:hover {{
                    background-color: #E9ECEF;
                }}
                QPushButton:pressed {{
                    background-color: {self.color};
                    color: white;
                }}
            """)
            btn.clicked.connect(self._preset_clicked)
            presets_layout.addWidget(btn)
        
        frame_layout.addLayout(presets_layout)
        
        # Add the frame to the main layout
        layout.addWidget(self.frame)
    
    def _value_changed(self, value):
        """Handle slider value changes."""
        self.value_label.setText(f"{value}%")
        
        # Update slider colors
        weight = value / 100.0
        # Update the tooltip with more detailed information
        self.slider.setToolTip(f"Weight: {weight:.2f}\nRelative importance: {self._get_importance_text(weight)}")
        
        # Emit the value change signal
        self.valueChanged.emit(self.objective_name, weight)
    
    def _preset_clicked(self):
        """Handle preset button clicks."""
        preset_value = self.sender().property("preset_value")
        self.set_value(preset_value / 100.0)
    
    def _get_importance_text(self, weight):
        """Convert weight to text representation of importance."""
        if weight < 0.1:
            return "Very Low"
        elif weight < 0.3:
            return "Low"
        elif weight < 0.6:
            return "Medium"
        elif weight < 0.8:
            return "High"
        else:
            return "Very High"
        
    def get_value(self):
        """Get the current normalized value (0-1)."""
        return self.slider.value() / 100.0
    
    def set_value(self, value):
        """Set the slider value from a normalized value (0-1)."""
        self.slider.setValue(int(value * 100))
        
    def set_color(self, color):
        """Set the color theme for this slider."""
        self.color = color
        self.name_label.setStyleSheet(f"font-weight: bold; color: {color};")
        # Update slider stylesheet
        self.slider.setStyleSheet(self.slider.styleSheet().replace(
            self.color, color))
        self.color = color


class MCONavigationDialog(QDialog):
    """
    Dialog điều hướng tối ưu đa tiêu chí.
    
    Dialog này cho phép người dùng điều chỉnh trọng số giữa các tiêu chí tối ưu
    và khám phá không gian các kế hoạch khả thi.
    """
    
    tradeAccepted = pyqtSignal(MCOTrade)
    
    def __init__(self, mco_engine, parent=None):
        """
        Khởi tạo dialog điều hướng MCO.
        
        Args:
            mco_engine: Động cơ tối ưu đa tiêu chí
            parent: Widget cha
        """
        super().__init__(parent)
        
        self.setWindowTitle("Điều Hướng Tối Ưu Đa Tiêu Chí (MCO)")
        self.setMinimumSize(1200, 800)
        
        self.mco_engine = mco_engine
        self.navigator = MCONavigator(mco_engine)
        
        self.current_trade = None
        self.objective_sliders = {}
        self.selected_objectives = []
        
        self.init_ui()
        
        # Thiết lập hẹn giờ để cập nhật kế hoạch khi người dùng điều chỉnh
        self.update_timer = QTimer()
        self.update_timer.setInterval(500)  # 500ms
        self.update_timer.timeout.connect(self.delayed_update_plan)
        self.update_pending = False
        
    def init_ui(self):
        """Initialize the user interface with an Eclipse-like design."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)
        
        # Title bar with logo and information
        title_bar = QFrame()
        title_bar.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        title_bar.setMaximumHeight(60)
        title_layout = QHBoxLayout(title_bar)
        
        # Logo or icon
        logo_label = QLabel()
        try:
            logo_pixmap = get_icon("mco_icon").pixmap(48, 48)
            logo_label.setPixmap(logo_pixmap)
        except:
            logo_label.setText("MCO")
            logo_label.setStyleSheet("font-weight: bold; font-size: 24px; color: #1976D2;")
        title_layout.addWidget(logo_label)
        
        # Title and description
        title_text = QLabel("<b>Multi-Criteria Optimization Navigator</b><br>"
                           "<span style='font-size: 11px; color: #555;'>Explore trade-offs between competing objectives</span>")
        title_layout.addWidget(title_text, 1)
        
        # Help button
        help_button = QPushButton()
        help_button.setIcon(QIcon.fromTheme("help-contents"))
        help_button.setToolTip("Show Help")
        help_button.setFixedSize(32, 32)
        help_button.clicked.connect(self.show_help)
        title_layout.addWidget(help_button)
        
        main_layout.addWidget(title_bar)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)
        main_splitter.setChildrenCollapsible(False)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(500)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Objective weights section
        weights_group = QGroupBox("Objective Weights")
        weights_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #ffffff;
            }
        """)
        weights_layout = QVBoxLayout(weights_group)
        
        # Buttons for weight management
        weight_buttons_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.setIcon(QIcon.fromTheme("edit-undo"))
        self.reset_button.clicked.connect(self.reset_weights)
        self.reset_button.setToolTip("Reset all weights to their original values")
        weight_buttons_layout.addWidget(self.reset_button)
        
        self.balance_button = QPushButton("Balance")
        self.balance_button.setIcon(QIcon.fromTheme("edit-clear"))
        self.balance_button.clicked.connect(self.balance_weights)
        self.balance_button.setToolTip("Set all weights to equal values")
        weight_buttons_layout.addWidget(self.balance_button)
        
        self.save_weights_button = QPushButton("Save")
        self.save_weights_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_weights_button.clicked.connect(self.save_weights)
        self.save_weights_button.setToolTip("Save current weights as a preset")
        weight_buttons_layout.addWidget(self.save_weights_button)
        
        self.load_weights_button = QPushButton("Load")
        self.load_weights_button.setIcon(QIcon.fromTheme("document-open"))
        self.load_weights_button.clicked.connect(self.load_weights)
        self.load_weights_button.setToolTip("Load weights from a saved preset")
        weight_buttons_layout.addWidget(self.load_weights_button)
        
        weights_layout.addLayout(weight_buttons_layout)
        
        # Scroll area for sliders
        slider_scroll = QScrollArea()
        slider_scroll.setWidgetResizable(True)
        slider_scroll.setFrameShape(QFrame.NoFrame)
        slider_widget = QWidget()
        self.slider_layout = QVBoxLayout(slider_widget)
        self.slider_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_layout.setSpacing(8)
        
        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.objective_filter = QComboBox()
        self.objective_filter.addItem("All Objectives")
        self.objective_filter.addItem("Target Objectives")
        self.objective_filter.addItem("OAR Objectives")
        self.objective_filter.addItem("Other Objectives")
        self.objective_filter.currentIndexChanged.connect(self.filter_objectives)
        filter_layout.addWidget(self.objective_filter, 1)
        
        self.slider_layout.addLayout(filter_layout)
        
        # Add sliders for each objective
        for obj in self.mco_engine.objectives:
            if obj.show_in_navigation:
                # Determine the objective type
                obj_type = "Other"
                if hasattr(obj, 'structure') and obj.structure:
                    if hasattr(obj.structure, 'type'):
                        obj_type = obj.structure.type
                
                slider = ObjectiveSlider(
                    obj.name, 
                    default_value=int(obj.current_weight * 100),
                    description=obj.name,
                    objective_type=obj_type
                )
                slider.valueChanged.connect(self.on_weight_changed)
                
                self.objective_sliders[obj.name] = slider
                self.slider_layout.addWidget(slider)
        
        # Add spacer at the end
        self.slider_layout.addStretch(1)
        slider_scroll.setWidget(slider_widget)
        weights_layout.addWidget(slider_scroll)
        
        left_layout.addWidget(weights_group, 3)
        
        # Tradeoff visualization controls
        tradeoff_group = QGroupBox("Tradeoff Visualization")
        tradeoff_group.setStyleSheet(weights_group.styleSheet())
        tradeoff_layout = QVBoxLayout(tradeoff_group)
        
        # Objective selection for X and Y axes
        axes_layout = QGridLayout()
        axes_layout.addWidget(QLabel("X-Axis:"), 0, 0)
        
        self.x_combo = QComboBox()
        axes_layout.addWidget(self.x_combo, 0, 1)
        
        axes_layout.addWidget(QLabel("Y-Axis:"), 1, 0)
        
        self.y_combo = QComboBox()
        axes_layout.addWidget(self.y_combo, 1, 1)
        
        tradeoff_layout.addLayout(axes_layout)
        
        # Update plot button
        update_plot_layout = QHBoxLayout()
        
        self.update_plot_button = QPushButton("Update Plot")
        self.update_plot_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.update_plot_button.clicked.connect(self.update_plot)
        update_plot_layout.addWidget(self.update_plot_button)
        
        self.export_plot_button = QPushButton("Export")
        self.export_plot_button.setIcon(QIcon.fromTheme("document-save-as"))
        self.export_plot_button.clicked.connect(self.export_tradeoff_plot)
        update_plot_layout.addWidget(self.export_plot_button)
        
        tradeoff_layout.addLayout(update_plot_layout)
        
        left_layout.addWidget(tradeoff_group, 1)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.accept_plan_button = QPushButton("Accept Plan")
        self.accept_plan_button.setIcon(QIcon.fromTheme("dialog-ok-apply"))
        self.accept_plan_button.clicked.connect(self.accept_plan)
        self.accept_plan_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        action_layout.addWidget(self.accept_plan_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setIcon(QIcon.fromTheme("dialog-cancel"))
        self.cancel_button.clicked.connect(self.reject)
        action_layout.addWidget(self.cancel_button)
        
        left_layout.addLayout(action_layout)
        
        # Right panel for visualizations
        right_panel = QWidget()
        right_panel.setMinimumWidth(650)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget for different visualizations
        self.visual_tabs = QTabWidget()
        self.visual_tabs.setTabPosition(QTabWidget.North)
        right_layout.addWidget(self.visual_tabs)
        
        # Trade-off plot tab
        tradeoff_tab = QWidget()
        tradeoff_tab_layout = QVBoxLayout(tradeoff_tab)
        
        # Create the tradeoff plot
        self.tradeoff_plot = MCOTradeoffPlot(width=6, height=5)
        self.tradeoff_plot.pointSelected.connect(self.on_plot_point_selected)
        tradeoff_tab_layout.addWidget(self.tradeoff_plot)
        
        self.visual_tabs.addTab(tradeoff_tab, "Tradeoff Plot")
        
        # DVH tab
        dvh_tab = QWidget()
        dvh_layout = QVBoxLayout(dvh_tab)
        
        # Create DVH widget
        self.dvh_widget = DVHWidget()
        dvh_layout.addWidget(self.dvh_widget)
        
        dvh_buttons = QHBoxLayout()
        self.refresh_dvh_button = QPushButton("Refresh")
        self.refresh_dvh_button.clicked.connect(self.update_dvh)
        dvh_buttons.addWidget(self.refresh_dvh_button)
        
        self.export_dvh_button = QPushButton("Export")
        self.export_dvh_button.clicked.connect(self.export_dvh_plot)
        dvh_buttons.addWidget(self.export_dvh_button)
        
        dvh_buttons.addStretch(1)
        dvh_layout.addLayout(dvh_buttons)
        
        self.visual_tabs.addTab(dvh_tab, "DVH")
        
        # Dose visualization tab
        dose_tab = QWidget()
        dose_layout = QVBoxLayout(dose_tab)
        
        self.dose_viewer = ImageViewer()
        dose_layout.addWidget(self.dose_viewer)
        
        self.dose_colorwash = DoseColorwash()
        dose_buttons = QHBoxLayout()
        
        self.refresh_dose_button = QPushButton("Refresh")
        self.refresh_dose_button.clicked.connect(self.update_dose_view)
        dose_buttons.addWidget(self.refresh_dose_button)
        
        dose_buttons.addStretch(1)
        dose_layout.addLayout(dose_buttons)
        
        self.visual_tabs.addTab(dose_tab, "Dose")
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([350, 650])  # Initial sizes
        
        main_layout.addWidget(main_splitter, 1)
        
        # Status bar
        status_bar = QFrame()
        status_bar.setFrameShape(QFrame.StyledPanel)
        status_bar.setStyleSheet("background-color: #f0f0f0; border-radius: 3px;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 1)
        
        self.auto_update_check = QCheckBox("Auto-update")
        self.auto_update_check.setChecked(True)
        self.auto_update_check.setToolTip("Automatically update the plan when weights change")
        status_layout.addWidget(self.auto_update_check)
        
        main_layout.addWidget(status_bar)
        
        # Populate the objective combo boxes
        self.populate_objective_combos()
        
        # Initial plot update
        self.update_plot()
        
        # Set window properties
        self.setWindowTitle("Multi-Criteria Optimization Navigator")
        self.setMinimumSize(1200, 800)
        self.resize(1200, 800)
    
    def populate_objective_combos(self):
        """Thêm các mục tiêu vào combo boxes."""
        objectives = [obj.name for obj in self.mco_engine.objectives if obj.show_in_navigation]
        
        self.x_combo.clear()
        self.y_combo.clear()
        
        self.x_combo.addItems(objectives)
        self.y_combo.addItems(objectives)
        
        # Chọn mục tiêu mặc định cho x và y
        if len(objectives) >= 2:
            self.x_combo.setCurrentIndex(0)
            self.y_combo.setCurrentIndex(1)
        
    def on_weight_changed(self, objective_name, value):
        """Xử lý khi trọng số thay đổi."""
        # Lên lịch cập nhật kế hoạch
        self.update_pending = True
        self.update_timer.start()
    
    def delayed_update_plan(self):
        """Cập nhật kế hoạch sau một khoảng thời gian."""
        if self.update_pending:
            # Dừng hẹn giờ
            self.update_timer.stop()
            self.update_pending = False
            
            # Thu thập trọng số hiện tại
            weights = {}
            for obj_name, slider in self.objective_sliders.items():
                weights[obj_name] = slider.get_value()
            
            # Chuẩn hóa trọng số
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            
            # Cập nhật kế hoạch
            try:
                self.current_trade = self.navigator.update_weights(weights)
                
                # Cập nhật hiển thị
                self.update_dvh()
                self.update_dose_view()
                
                # Cập nhật điểm hiện tại trên biểu đồ đánh đổi
                self.update_current_point()
                
            except OptimizationError as e:
                QMessageBox.warning(self, "Lỗi Cập Nhật", str(e))
    
    def reset_weights(self):
        """Đặt lại trọng số về giá trị ban đầu."""
        for obj in self.mco_engine.objectives:
            if obj.name in self.objective_sliders:
                self.objective_sliders[obj.name].set_value(obj.current_weight)
        
        # Cập nhật kế hoạch
        self.delayed_update_plan()
    
    def balance_weights(self):
        """Đặt các trọng số bằng nhau."""
        if not self.objective_sliders:
            return
        
        # Chia đều trọng số
        num_objectives = len(self.objective_sliders)
        equal_weight = 1.0 / num_objectives
        
        for slider in self.objective_sliders.values():
            slider.set_value(equal_weight)
        
        # Cập nhật kế hoạch
        self.delayed_update_plan()
    
    def update_plot(self):
        """Cập nhật biểu đồ đánh đổi."""
        if not self.mco_engine.trades:
            return
        
        # Lấy mục tiêu đã chọn
        x_objective = self.x_combo.currentText()
        y_objective = self.y_combo.currentText()
        
        if not x_objective or not y_objective:
            return
        
        # Lấy dữ liệu
        x_values = []
        y_values = []
        
        for trade in self.mco_engine.trades:
            if x_objective in trade.objective_values and y_objective in trade.objective_values:
                x_values.append(trade.objective_values[x_objective])
                y_values.append(trade.objective_values[y_objective])
        
        # Vẽ biểu đồ
        current_point = None
        if self.current_trade:
            if x_objective in self.current_trade.objective_values and y_objective in self.current_trade.objective_values:
                current_point = (
                    self.current_trade.objective_values[x_objective],
                    self.current_trade.objective_values[y_objective]
                )
        
        self.tradeoff_plot.plot_tradeoff(
            x_values, y_values, 
            x_objective, y_objective,
            current_point=current_point,
            title=f"Tradeoff: {x_objective} vs {y_objective}"
        )
    
    def update_current_point(self):
        """Cập nhật điểm hiện tại trên biểu đồ đánh đổi."""
        # Kiểm tra xem có biểu đồ hiện tại không
        if not hasattr(self.tradeoff_plot.axes, 'collections') or len(self.tradeoff_plot.axes.collections) < 2:
            # Nếu không, cập nhật toàn bộ biểu đồ
            self.update_plot()
            return
        
        # Lấy mục tiêu đã chọn
        x_objective = self.x_combo.currentText()
        y_objective = self.y_combo.currentText()
        
        if not x_objective or not y_objective or not self.current_trade:
            return
        
        # Cập nhật vị trí điểm hiện tại
        if x_objective in self.current_trade.objective_values and y_objective in self.current_trade.objective_values:
            current_x = self.current_trade.objective_values[x_objective]
            current_y = self.current_trade.objective_values[y_objective]
            
            # Cập nhật vị trí điểm hiện tại (điểm thứ hai trong collections)
            if len(self.tradeoff_plot.axes.collections) >= 2:
                self.tradeoff_plot.axes.collections[1].set_offsets([(current_x, current_y)])
                self.tradeoff_plot.draw()
    
    def update_dvh(self):
        """Cập nhật biểu đồ DVH."""
        if not self.current_trade or not self.current_trade.dvh_data:
            return
        
        self.dvh_widget.update_dvh(self.current_trade.dvh_data)
    
    def update_dose_view(self):
        """Cập nhật hiển thị phân bố liều."""
        if not hasattr(self, 'current_trade') or self.current_trade is None:
            return
        
        # Lấy phân bố liều từ MCOTrade hiện tại
        dose_grid = self.current_trade.dose_grid
        if dose_grid is None:
            return
        
        # Hiển thị lát cắt giữa của phân bố liều
        dose_array = dose_grid.dose_array
        if dose_array.ndim == 3:
            slice_idx = dose_array.shape[2] // 2
            dose_slice = dose_array[:, :, slice_idx]
            
            # Tạo một hình từ lát cắt dose
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            # Sử dụng DoseColorwash để hiển thị màu liều
            self.dose_colorwash.display_2d(
                dose_slice=dose_slice,
                figure=fig,
                dose_max=dose_array.max() if dose_array.max() > 0 else None
            )
            
            # Chuyển đổi figure matplotlib thành hình ảnh để hiển thị
            fig.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (h, w, 3)
            
            # Hiển thị hình ảnh trong ImageViewer
            self.dose_viewer.set_image(buf)
            self.dose_viewer.update_view()
    
    def accept_plan(self):
        """Chấp nhận kế hoạch hiện tại."""
        if not self.current_trade:
            QMessageBox.warning(self, "Cảnh Báo", "Không có kế hoạch hiện tại để chấp nhận.")
            return
        
        # Phát tín hiệu với trade đã chọn
        self.tradeAccepted.emit(self.current_trade)
        
        # Đóng dialog
        self.accept()
    
    def export_tradeoff_plot(self):
        """Xuất biểu đồ đánh đổi ra file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Biểu Đồ", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)"
        )
        
        if file_path:
            self.tradeoff_plot.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Thông Báo", f"Đã lưu biểu đồ vào {file_path}")
    
    def export_dvh_plot(self):
        """Xuất biểu đồ DVH ra file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Biểu Đồ DVH", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)"
        )
        
        if file_path:
            self.dvh_widget.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Thông Báo", f"Đã lưu biểu đồ DVH vào {file_path}")
    
    def filter_objectives(self, index):
        """Filter the objective sliders based on the selected filter."""
        filter_text = self.objective_filter.currentText()
        
        for name, slider in self.objective_sliders.items():
            # Get the objective type
            obj_type = slider.objective_type.lower() if hasattr(slider, 'objective_type') else "other"
            
            if filter_text == "All Objectives":
                slider.setVisible(True)
            elif filter_text == "Target Objectives" and "target" in obj_type:
                slider.setVisible(True)
            elif filter_text == "OAR Objectives" and "oar" in obj_type:
                slider.setVisible(True)
            elif filter_text == "Other Objectives" and "target" not in obj_type and "oar" not in obj_type:
                slider.setVisible(True)
            else:
                slider.setVisible(False)
    
    def on_plot_point_selected(self, index):
        """Handle selection of a point on the tradeoff plot."""
        if index >= 0 and index < len(self.navigator.trade_history):
            # Get the selected trade
            trade = self.navigator.trade_history[index]
            
            # Update sliders to match the selected trade's weights
            for name, weight in trade.weights.items():
                if name in self.objective_sliders:
                    self.objective_sliders[name].set_value(weight)
            
            # Update the current trade
            self.current_trade = trade
            
            # Update visualizations
            self._update_visualizations()
            
            # Update status
            self.status_label.setText(f"Selected solution {index+1}")
    
    def save_weights(self):
        """Save current weights as a preset."""
        # Get current weights
        weights = {name: slider.get_value() for name, slider in self.objective_sliders.items()}
        
        # Ask for a name
        name, ok = QInputDialog.getText(self, "Save Weights", 
                                                 "Enter a name for this preset:")
        if ok and name:
            # Create a preset dictionary
            preset = {
                'name': name,
                'weights': weights,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Get existing presets
            presets = []
            preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "optimization", "mco", "presets.json")
            try:
                if os.path.exists(preset_path):
                    with open(preset_path, 'r') as f:
                        presets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading presets: {e}")
                presets = []
            
            # Add new preset
            presets.append(preset)
            
            # Save presets
            try:
                with open(preset_path, 'w') as f:
                    json.dump(presets, f, indent=2)
                self.status_label.setText(f"Saved preset: {name}")
            except Exception as e:
                logger.error(f"Error saving preset: {e}")
                QMessageBox.warning(self, "Save Error", 
                                  f"Could not save preset: {str(e)}")
    
    def load_weights(self):
        """Load weights from a saved preset."""
        # Get existing presets
        preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "optimization", "mco", "presets.json")
        presets = []
        
        try:
            if os.path.exists(preset_path):
                with open(preset_path, 'r') as f:
                    presets = json.load(f)
        except Exception as e:
            logger.error(f"Error loading presets: {e}")
            QMessageBox.warning(self, "Load Error", 
                              f"Could not load presets: {str(e)}")
            return
        
        if not presets:
            QMessageBox.information(self, "No Presets", 
                                   "No saved presets found.")
            return
        
        # Show selection dialog
        preset_names = [p['name'] for p in presets]
        name, ok = QInputDialog.getItem(self, "Load Preset", 
                                                "Select a preset to load:", 
                                                preset_names, 0, False)
        
        if ok and name:
            # Find the selected preset
            selected_preset = next((p for p in presets if p['name'] == name), None)
            
            if selected_preset and 'weights' in selected_preset:
                # Apply the weights
                for obj_name, weight in selected_preset['weights'].items():
                    if obj_name in self.objective_sliders:
                        self.objective_sliders[obj_name].set_value(weight)
                
                self.status_label.setText(f"Loaded preset: {name}")
                
                # Trigger an update based on the new weights
                self.delayed_update_plan()
    
    def show_help(self):
        """Show help information for the MCO Navigator."""
        help_text = """
        <h3>Multi-Criteria Optimization Navigator</h3>
        <p>This tool allows you to interactively explore trade-offs between competing 
        treatment planning objectives.</p>
        
        <h4>Key Features:</h4>
        <ul>
          <li><b>Objective Weights</b>: Adjust the importance of each planning objective</li>
          <li><b>Tradeoff Plot</b>: Visualize the Pareto frontier of optimal plans</li>
          <li><b>DVH Display</b>: See the dose-volume histogram for the current plan</li>
          <li><b>Dose Visualization</b>: View the dose distribution</li>
        </ul>
        
        <h4>How to Use:</h4>
        <ol>
          <li>Adjust the sliders to change the importance of each objective</li>
          <li>The plan will update automatically (if auto-update is enabled)</li>
          <li>Click on points in the tradeoff plot to select different Pareto-optimal plans</li>
          <li>When satisfied with a plan, click "Accept Plan" to finalize it</li>
        </ol>
        
        <p>For more information, please refer to the user manual.</p>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("MCO Navigator Help")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(help_text)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
    
    def _update_visualizations(self):
        """Update all visualizations with the current plan."""
        # Update the current point on the tradeoff plot
        self.update_current_point()
        
        # Update the DVH
        self.update_dvh()
        
        # Update the dose view
        self.update_dose_view() 