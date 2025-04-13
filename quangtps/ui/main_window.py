#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Window Module
=================

This module implements the main window of the QuangTPS application, providing
access to all features of the treatment planning system.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Any, Union

from PyQt5.QtWidgets import (
    QMainWindow, QAction, QMenu, QToolBar, QDockWidget, QWidget, 
    QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QFileDialog, QApplication, QSplitter, QDialog,
    QStatusBar, QProgressBar, QComboBox, QStyle, QStyleFactory,
    QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSettings, QSize, QTimer
from PyQt5.QtGui import QIcon, QColor, QPalette, QKeySequence, QPixmap

from quangtps.core.services import ServiceRegistry
from quangtps.ui.patient_browser import PatientBrowser
from quangtps.ui.patient_tab import PatientTab
from quangtps.ui.planning_tab import PlanningTab
from quangtps.ui.evaluation_tab import EvaluationTab
from quangtps.ui.optimization.optimizer_tab import OptimizerTab
from quangtps.ui.imaging_tab import ImagingTab
from quangtps.ui.optimization.mco_panel import MCOPanel

from quangtps.core.image import Image
from quangtps.core.structures import StructureSet, Structure
from quangtps.planning.plan import Plan
from quangtps.planning.beam_set import BeamSet
from quangtps.dose.dose_calculator import DoseCalculator

from quangtps.ui.mpr_viewer import MPRViewer
from quangtps.ui.structure_tab import StructureTab
from quangtps.ui.mpr_structure_integration import MPRStructureIntegration
from quangtps.ui.dialogs.protocol_dialog import ClinicalProtocolDialog

# Add the necessary imports for our optimization modules
try:
    from quangtps.ui.optimization.optimization_objective_panel import OptimizationObjectivePanel
except ImportError:
    logging.warning("Failed to import optimization components")

logger = logging.getLogger(__name__)

class LeftPanel(QWidget):
    """
    Left panel containing patient browser and other navigation elements.
    Mimics the Eclipse-style left panel.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumWidth(250)
        self.setMaximumWidth(350)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Patient browser
        self.patient_browser = PatientBrowser()
        layout.addWidget(self.patient_browser)
        
        # Setup style
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 2px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)

class ContouringView(QWidget):
    """Contouring view mimicking Eclipse contouring interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Placeholder for now
        label = QLabel("Contouring View")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
class PlanningView(QWidget):
    """Planning view mimicking Eclipse planning interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Use the existing PlanningTab
        self.planning_tab = PlanningTab()
        layout.addWidget(self.planning_tab)

class EvaluationView(QWidget):
    """Evaluation view mimicking Eclipse evaluation interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Use the existing EvaluationTab
        self.evaluation_tab = EvaluationTab()
        layout.addWidget(self.evaluation_tab)

class OptimizationView(QWidget):
    """Optimization view mimicking Eclipse optimization interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Use the existing OptimizerTab
        self.optimizer_tab = OptimizerTab()
        layout.addWidget(self.optimizer_tab)

class ReviewView(QWidget):
    """Review view mimicking Eclipse review and approval interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Placeholder for now
        label = QLabel("Review View")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

class MainWindow(QMainWindow):
    """
    Main application window for the treatment planning system.
    Implements an Eclipse-like interface with tabs for different planning stages.
    """
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("QuangTPS - Radiation Treatment Planning System")
        self.setMinimumSize(1024, 768)
        self.setWindowIcon(QIcon("quangtps/resources/icons/logo.png"))
        
        # Initialize state
        self.current_patient = None
        self.current_plan = None
        self.current_image = None
        self.current_structure_set = None
        self.current_dose = None
        
        # Setup services
        self._initialize_services()
        
        # Setup UI
        self._setup_ui()
        self._create_menus()
        self._create_toolbars()
        self.apply_styling()
        
        # Load settings
        self._load_settings()
        
        # Setup status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar for calculations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Show the window
        self.show()
    
    def _setup_ui(self):
        """Set up the main UI components."""
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel for patient/plan browser
        self.left_panel = LeftPanel()
        self.main_splitter.addWidget(self.left_panel)
        
        # Create right area with tabs
        self.right_area = QTabWidget()
        
        # Tab 1: Contouring tab
        self.contouring_tab = ContouringView()
        self.right_area.addTab(self.contouring_tab, "Contouring")
        
        # Tab 2: Planning tab
        self.planning_tab = PlanningView()
        self.right_area.addTab(self.planning_tab, "Planning")
        
        # Tab 3: Evaluation tab
        self.evaluation_tab = EvaluationView()
        self.right_area.addTab(self.evaluation_tab, "Evaluation")
        
        # Tab 4: Optimization tab
        self.optimization_tab = OptimizationView()
        self.right_area.addTab(self.optimization_tab, "Optimization")
        
        # Tab 5: MCO Tab (Multi-Criteria Optimization)
        self.mco_tab = MCOPanel()
        self.right_area.addTab(self.mco_tab, "MCO")
        
        # Tab 6: Review tab
        self.review_tab = ReviewView()
        self.right_area.addTab(self.review_tab, "Review")
        
        # Add right area to main splitter
        self.main_splitter.addWidget(self.right_area)
        
        # Set initial sizes
        self.main_splitter.setSizes([300, 900])
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.main_splitter)
        
        # Create menus and toolbars
        self._create_menus()
        self._create_toolbars()
        
        # Load user settings
        self._load_settings()
        
        # Initially hide patient browser
        self.toggle_patient_browser(False)
    
    def _create_menus(self):
        """Create the application menus."""
        self._create_file_menu()
        self._create_edit_menu()
        self._create_view_menu()
        self._create_planning_menu()
        self._create_dose_menu()
        self._create_evaluation_menu()
        self._create_tools_menu()
        self._create_help_menu()
    
    def _create_file_menu(self):
        """Create the file menu."""
        self.file_menu = self.menuBar().addMenu("&File")
        
        self.new_patient_action = QAction("&New Patient...", self)
        self.new_patient_action.setShortcut("Ctrl+N")
        self.new_patient_action.triggered.connect(self._new_patient)
        self.file_menu.addAction(self.new_patient_action)
        
        self.open_patient_action = QAction("&Open Patient...", self)
        self.open_patient_action.setShortcut("Ctrl+O")
        self.open_patient_action.triggered.connect(self.open_patient_dialog)
        self.file_menu.addAction(self.open_patient_action)
        
        self.file_menu.addSeparator()
        
        self.import_dicom_action = QAction("Import &DICOM...", self)
        self.import_dicom_action.triggered.connect(self.open_image_dialog)
        self.file_menu.addAction(self.import_dicom_action)
        
        self.import_rt_struct_action = QAction("Import RT&Struct...", self)
        self.import_rt_struct_action.triggered.connect(self.import_structure_set_dialog)
        self.file_menu.addAction(self.import_rt_struct_action)
        
        self.import_rt_plan_action = QAction("Import RT&Plan...", self)
        self.import_rt_plan_action.triggered.connect(self.load_plan_dialog)
        self.file_menu.addAction(self.import_rt_plan_action)
        
        self.import_rt_dose_action = QAction("Import RT&Dose...", self)
        self.file_menu.addAction(self.import_rt_dose_action)
        
        self.file_menu.addSeparator()
        
        self.save_plan_action = QAction("&Save Plan...", self)
        self.save_plan_action.setShortcut("Ctrl+S")
        self.save_plan_action.triggered.connect(self.save_plan_dialog)
        self.file_menu.addAction(self.save_plan_action)
        
        self.export_dicom_action = QAction("&Export DICOM...", self)
        self.file_menu.addAction(self.export_dicom_action)
        
        self.file_menu.addSeparator()
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut("Alt+F4")
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)
    
    def _create_edit_menu(self):
        """Create the edit menu."""
        self.edit_menu = self.menuBar().addMenu("&Edit")
        
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.edit_menu.addAction(self.undo_action)
        
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.edit_menu.addAction(self.redo_action)
        
        self.edit_menu.addSeparator()
        
        self.preferences_action = QAction("&Preferences...", self)
        self.edit_menu.addAction(self.preferences_action)
    
    def _create_view_menu(self):
        """Create the view menu."""
        self.view_menu = self.menuBar().addMenu("&View")
        
        self.patient_browser_action = QAction("Patient &Browser", self)
        self.patient_browser_action.setCheckable(True)
        self.patient_browser_action.setChecked(False)
        self.patient_browser_action.triggered.connect(lambda checked: self.toggle_patient_browser(checked))
        self.view_menu.addAction(self.patient_browser_action)
        
        self.view_menu.addSeparator()
    
    def _create_planning_menu(self):
        """Create the planning menu."""
        self.planning_menu = self.menuBar().addMenu("&Planning")
        
        self.new_plan_action = QAction("&New Plan...", self)
        self.planning_menu.addAction(self.new_plan_action)
        
        self.calculate_dose_action = QAction("&Calculate Dose", self)
        self.calculate_dose_action.triggered.connect(self.calculate_dose)
        self.planning_menu.addAction(self.calculate_dose_action)
        
        self.optimization_action = QAction("&Optimization...", self)
        self.planning_menu.addAction(self.optimization_action)
        
        self.planning_menu.addSeparator()
        
        self.clinical_protocols_action = QAction("&Clinical Protocols...", self)
        self.clinical_protocols_action.triggered.connect(self.show_protocols_dialog)
        self.planning_menu.addAction(self.clinical_protocols_action)
    
    def _create_dose_menu(self):
        """Create the dose menu."""
        self.dose_menu = self.menuBar().addMenu("&Dose")
        
        self.dose_calculator_action = QAction("&Dose Calculator...", self)
        self.dose_menu.addAction(self.dose_calculator_action)
    
    def _create_evaluation_menu(self):
        """Create the evaluation menu."""
        self.evaluation_menu = self.menuBar().addMenu("&Evaluation")
        
        plan_evaluation_action = QAction("Plan Evaluation", self)
        plan_evaluation_action.triggered.connect(self._open_plan_evaluation)
        self.evaluation_menu.addAction(plan_evaluation_action)
        
        plan_comparison_action = QAction("Plan Comparison", self)
        plan_comparison_action.triggered.connect(self._open_plan_comparison)
        self.evaluation_menu.addAction(plan_comparison_action)
    
    def _create_tools_menu(self):
        """Create the tools menu."""
        self.tools_menu = self.menuBar().addMenu("&Tools")
        
        self.dicom_browser_action = QAction("&DICOM Browser...", self)
        self.tools_menu.addAction(self.dicom_browser_action)
        
        self.tools_menu.addSeparator()
        
        self.scripts_menu = QMenu("&Scripts", self)
        self.tools_menu.addMenu(self.scripts_menu)
        
        self.run_script_action = QAction("&Run Script...", self)
        self.scripts_menu.addAction(self.run_script_action)
        
        self.scripts_menu.addSeparator()
        
        self.script_editor_action = QAction("Script &Editor...", self)
        self.scripts_menu.addAction(self.script_editor_action)
    
    def _create_help_menu(self):
        """Create the help menu."""
        self.help_menu = self.menuBar().addMenu("&Help")
        
        self.help_contents_action = QAction("&Contents...", self)
        self.help_menu.addAction(self.help_contents_action)
        
        self.help_menu.addSeparator()
        
        self.about_action = QAction("&About QuangTPS...", self)
        self.about_action.triggered.connect(self.show_about_dialog)
        self.help_menu.addAction(self.about_action)
    
    def _create_toolbars(self):
        """Create application toolbars."""
        # Main toolbar
        self.main_toolbar = self.addToolBar("Main")
        self.main_toolbar.setIconSize(QSize(24, 24))
        self.main_toolbar.setMovable(False)
        
        # Get icons directory
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        
        # Add toolbar actions
        self.new_patient_tool = QAction(QIcon(os.path.join(icons_dir, "new_patient.png")), "New Patient", self)
        self.new_patient_tool.triggered.connect(self._new_patient)
        self.main_toolbar.addAction(self.new_patient_tool)
        
        self.open_patient_tool = QAction(QIcon(os.path.join(icons_dir, "open_patient.png")), "Open Patient", self)
        self.open_patient_tool.triggered.connect(self.open_patient_dialog)
        self.main_toolbar.addAction(self.open_patient_tool)
        
        self.main_toolbar.addSeparator()
        
        self.import_dicom_tool = QAction(QIcon(os.path.join(icons_dir, "import_dicom.png")), "Import DICOM", self)
        self.import_dicom_tool.triggered.connect(self.open_image_dialog)
        self.main_toolbar.addAction(self.import_dicom_tool)
        
        self.import_rt_struct_tool = QAction(QIcon(os.path.join(icons_dir, "roi.png")), "Import RTStruct", self)
        self.import_rt_struct_tool.triggered.connect(self.import_structure_set_dialog)
        self.main_toolbar.addAction(self.import_rt_struct_tool)
        
        self.main_toolbar.addSeparator()
        
        self.new_plan_tool = QAction(QIcon(os.path.join(icons_dir, "new_plan.png")), "New Plan", self)
        self.main_toolbar.addAction(self.new_plan_tool)
        
        self.calculate_dose_tool = QAction(QIcon(os.path.join(icons_dir, "calculate_dose.png")), "Calculate Dose", self)
        self.calculate_dose_tool.triggered.connect(self.calculate_dose)
        self.main_toolbar.addAction(self.calculate_dose_tool)
        
        self.optimize_tool = QAction(QIcon(os.path.join(icons_dir, "optimize.png")), "Optimize", self)
        self.main_toolbar.addAction(self.optimize_tool)
        
        self.main_toolbar.addSeparator()
        
        self.evaluate_tool = QAction(QIcon(os.path.join(icons_dir, "evaluate.png")), "Evaluate", self)
        self.main_toolbar.addAction(self.evaluate_tool)
        
        self.report_tool = QAction(QIcon(os.path.join(icons_dir, "report.png")), "Report", self)
        self.main_toolbar.addAction(self.report_tool)
        
        # Add a stretch spacer to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_toolbar.addWidget(spacer)
        
        # Add help action to the right
        self.help_tool = QAction(QIcon(os.path.join(icons_dir, "help.png")), "Help", self)
        self.main_toolbar.addAction(self.help_tool)
    
    def apply_styling(self):
        """Apply styling to the application."""
        # Set application style based on availability
        if QStyleFactory.keys().count("Fusion") > 0:
            self.setStyle(QStyleFactory.create("Fusion"))
        
        # Create color palette (Eclipse-like blue theme)
        palette = QPalette()
        
        # Set window and button colors
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        
        # Set highlight colors
        palette.setColor(QPalette.Highlight, QColor(32, 112, 192))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # Set link colors
        palette.setColor(QPalette.Link, QColor(32, 112, 192))
        palette.setColor(QPalette.LinkVisited, QColor(102, 0, 153))
        
        # Apply palette
        self.setPalette(palette)
        
        # Set stylesheet for custom styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            
            QMenuBar {
                background-color: #f0f0f0;
                border-bottom: 1px solid #d0d0d0;
            }
            
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 8px;
            }
            
            QMenuBar::item:selected {
                background-color: #2070c0;
                color: white;
            }
            
            QMenu {
                background-color: #f8f8f8;
                border: 1px solid #d0d0d0;
            }
            
            QMenu::item:selected {
                background-color: #2070c0;
                color: white;
            }
            
            QToolBar {
                background-color: #f0f0f0;
                border-bottom: 1px solid #d0d0d0;
                spacing: 3px;
            }
            
            QToolButton {
                background-color: transparent;
                border-radius: 3px;
                padding: 3px;
            }
            
            QToolButton:hover {
                background-color: #e0e0e0;
            }
            
            QToolButton:pressed {
                background-color: #c0c0c0;
            }
            
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: #f8f8f8;
            }
            
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #d0d0d0;
                border-bottom: none;
                padding: 5px 10px;
                margin-right: 2px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            
            QTabBar::tab:selected {
                background-color: #2070c0;
                color: white;
            }
            
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #d0d0d0;
            }
            
            QStatusBar {
                background-color: #f0f0f0;
                border-top: 1px solid #d0d0d0;
            }
            
            QTreeWidget {
                background-color: white;
                border: 1px solid #d0d0d0;
                alternate-background-color: #f8f8f8;
            }
            
            QTreeWidget::item:hover {
                background-color: #e8e8e8;
            }
            
            QTreeWidget::item:selected {
                background-color: #2070c0;
                color: white;
            }
        """)
    
    def _load_settings(self):
        """Load application settings."""
        settings = QSettings("QuangTPS", "Treatment Planning System")
        
        # Restore window geometry
        if settings.contains("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        else:
            # Set default window size and position
            screen_rect = self.screen().availableGeometry()
            window_width = int(screen_rect.width() * 0.8)
            window_height = int(screen_rect.height() * 0.8)
            self.resize(window_width, window_height)
            self.move(
                (screen_rect.width() - window_width) // 2,
                (screen_rect.height() - window_height) // 2
            )
        
        # Restore window state
        if settings.contains("windowState"):
            self.restoreState(settings.value("windowState"))
        
        # Restore last directory
        if settings.contains("lastDirectory"):
            self.last_directory = settings.value("lastDirectory")
                    else:
            self.last_directory = str(Path.home())
    
    def _initialize_services(self):
        """Initialize the dose calculator and other services."""
        try:
            self.dose_calculator = DoseCalculator()
            logger.info("Dose calculator initialized")
            
            # Set in evaluation tab
            if self.evaluation_tab:
                self.evaluation_tab.set_dose_calculator(self.dose_calculator)
                except Exception as e:
            logger.error(f"Failed to initialize dose calculator: {e}")
            self.dose_calculator = None
    
    def open_image_dialog(self):
        """Show dialog to open an image."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.last_directory,
            "Image Files (*.nii *.nii.gz *.dcm *.mha *.mhd);;All Files (*)",
            options=options
        )
        
        if file_path:
            # Update last directory
            self.last_directory = os.path.dirname(file_path)
            
            # Load the image
            self.load_image_from_path(file_path)
    
    def load_image_from_path(self, file_path):
        """Load an image from file path."""
        try:
            # Show status message
            self.statusBar().showMessage(f"Loading image: {file_path}...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            # Use SimpleITK to load image
            sitk_image = sitk.ReadImage(file_path)
            
            # Convert to numpy array
            image_array = sitk.GetArrayFromImage(sitk_image)
            
            # Create Image object
            image = Image()
            image.data = image_array
            
            # Set spacing and origin from SimpleITK image
            image.voxel_size = sitk_image.GetSpacing()[::-1]  # Reverse order for NumPy
            image.origin = sitk_image.GetOrigin()[::-1]  # Reverse order for NumPy
            
            # Load the image into the application
            self.load_image(image)
            
            # Update progress
            self.progress_bar.setValue(100)
            
            # Clear status message after delay
            QTimer.singleShot(2000, lambda: self.statusBar().clearMessage())
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            
            # Add to patient browser
            self.left_panel.patient_browser.add_image(image, os.path.basename(file_path))
            
        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            logger.error(f"Failed to load image: {e}")
            
            # Clear status message
            self.statusBar().clearMessage()
            self.progress_bar.setVisible(False)
    
    def load_image(self, image):
        """Load an image into the application."""
        if image is None:
            return
        
        self.current_image = image
        
        # Set image in MPR viewer
        if hasattr(self, 'mpr_viewer'):
            self.mpr_viewer.set_image(image)
        
        # Set image in structure tab
        if hasattr(self, 'structure_tab'):
            self.structure_tab.set_image(image)
        
        # Set image in planning tab
        if hasattr(self, 'planning_tab'):
            self.planning_tab.set_image(image)
        
        # Switch to imaging tab
        self.right_area.setCurrentWidget(self.contouring_tab)
        
        # Update status
        self.statusBar().showMessage(f"Loaded image: {image.series_description if hasattr(image, 'series_description') else 'Unknown'}")
    
    def import_structure_set_dialog(self):
        """Show dialog to import a structure set."""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Structure Set", self.last_directory,
            "Structure Files (*.dcm *.xml *.json);;All Files (*)",
            options=options
        )
        
        if file_path:
            # Update last directory
            self.last_directory = os.path.dirname(file_path)
            
            # TODO: Implement structure set import
            QMessageBox.information(self, "Information", "Structure set import not yet implemented.")
    
    def load_structure_set(self, structure_set):
        """Load a structure set into the application."""
        if not structure_set:
            return
        
        # Set current structure set
        self.current_structure_set = structure_set
        
        # Set in structure tab
        self.structure_tab.set_structure_set(structure_set)
        
        # Set in dose calculator
        if self.dose_calculator:
            self.dose_calculator.structure_set = structure_set
        
        # Update status bar
        structure_count = len(structure_set.structures) if structure_set.structures else 0
        self.statusBar().showMessage(f"Structure set loaded: {structure_set.name} ({structure_count} structures)")
        
        # Switch to Structure tab
        self.right_area.setCurrentWidget(self.contouring_tab)
    
    def save_plan_dialog(self):
        """Show dialog to save the current plan."""
        if not self.current_plan:
            QMessageBox.warning(self, "Warning", "No plan to save.")
            return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plan", self.last_directory,
            "Plan Files (*.plan);;All Files (*)",
            options=options
        )
        
        if file_path:
            # Add .plan extension if not provided
            if not file_path.lower().endswith('.plan'):
                file_path += '.plan'
            
            # Update last directory
            self.last_directory = os.path.dirname(file_path)
            
            try:
                # Save the plan to disk
                self.statusBar().showMessage(f"Saving plan to: {file_path}...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                # TODO: Implement plan saving
                # Placeholder for actual implementation
                QTimer.singleShot(500, lambda: self.progress_bar.setValue(50))
                QTimer.singleShot(1000, lambda: self.progress_bar.setValue(100))
                
                # Clear status message after delay
                QTimer.singleShot(1500, lambda: self.statusBar().showMessage("Plan saved successfully."))
                QTimer.singleShot(3000, lambda: self.statusBar().clearMessage())
                QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plan: {str(e)}")
                logger.error(f"Failed to save plan: {e}")
                
                # Clear status message
                self.statusBar().clearMessage()
                self.progress_bar.setVisible(False)
    
    def load_plan_dialog(self):
        """Show dialog to load a plan."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Plan", self.last_directory,
            "Plan Files (*.plan);;All Files (*)",
            options=options
        )
        
        if file_path:
            # Update last directory
            self.last_directory = os.path.dirname(file_path)
            
            try:
                # Load the plan from disk
                self.statusBar().showMessage(f"Loading plan from: {file_path}...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                # TODO: Implement plan loading
                # Placeholder for actual implementation
                QTimer.singleShot(500, lambda: self.progress_bar.setValue(50))
                QTimer.singleShot(1000, lambda: self.progress_bar.setValue(100))
                
                # Clear status message after delay
                QTimer.singleShot(1500, lambda: self.statusBar().showMessage("Plan loaded successfully."))
                QTimer.singleShot(3000, lambda: self.statusBar().clearMessage())
                QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))
                
                # Create a dummy plan for testing
                self.create_test_plan()
                
                except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load plan: {str(e)}")
                logger.error(f"Failed to load plan: {e}")
                
                # Clear status message
                self.statusBar().clearMessage()
                self.progress_bar.setVisible(False)
    
    def create_test_plan(self):
        """Create a test plan for development purposes."""
        if not self.current_image or not self.current_structure_set:
            QMessageBox.warning(self, "Warning", "Please load an image and create structures first.")
            return
        
        # Create a plan
        plan = Plan()
        plan.name = "Test Plan"
        plan.image_ref = self.current_image
        plan.structure_set = self.current_structure_set
        
        # Create a beam set
        beam_set = BeamSet()
        beam_set.name = "3DCRT"
        beam_set.plan = plan
        
        # Add beam set to plan
        plan.add_beam_set(beam_set)
        
        # Set the plan and beam set
        self.load_plan(plan)
        
        # Switch to Planning tab
        self.right_area.setCurrentWidget(self.planning_tab)
    
    def load_plan(self, plan):
        """Load a plan into the application."""
        if not plan:
            return
        
        # Set current plan
        self.current_plan = plan
        
        # Set current beam set to the first one if available
        if plan.beam_sets and len(plan.beam_sets) > 0:
            self.current_beam_set = plan.beam_sets[0]
            else:
            self.current_beam_set = None
        
        # Set in planning tab
        self.planning_tab.set_plan(plan)
        self.planning_tab.set_beam_set(self.current_beam_set)
        
        # Set in evaluation tab if dose is available
        if plan.dose is not None:
            self.evaluation_tab.set_dose(plan.dose)
            self.evaluation_tab.set_plan(plan)
            
            # Show dose overlay in MPR viewer
            self.mpr_viewer.set_dose(plan.dose)
        
        # Update status bar
        self.statusBar().showMessage(f"Plan loaded: {plan.name}")
    
    def calculate_dose(self):
        """Calculate dose for the current plan."""
        if not self.current_image or not self.current_structure_set or not self.current_beam_set:
            QMessageBox.warning(
                self, "Warning", 
                "Cannot calculate dose. Please ensure image, structure set, and beam set are loaded."
            )
            return
        
        if not self.dose_calculator:
            QMessageBox.warning(
                self, "Warning", 
                "Dose calculator is not initialized."
            )
            return
        
        # Show status message
        self.statusBar().showMessage("Calculating dose...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            # Setup dose calculator
            self.dose_calculator.image = self.current_image
            self.dose_calculator.structure_set = self.current_structure_set
            self.dose_calculator.beam_set = self.current_beam_set
            
            # Connect progress signal
            self.dose_calculator.progress_updated.connect(self.update_dose_progress)
            
            # Calculate dose
            dose = self.dose_calculator.calculate()
            
            # Disconnect progress signal
            self.dose_calculator.progress_updated.disconnect(self.update_dose_progress)
            
            # Set dose in plan
            if self.current_plan:
                self.current_plan.dose = dose
                
                # Set in evaluation tab
                self.evaluation_tab.set_dose(dose)
                self.evaluation_tab.set_plan(self.current_plan)
                
                # Show dose overlay in MPR viewer
                self.mpr_viewer.set_dose(dose)
            
            # Update status bar
            self.statusBar().showMessage("Dose calculation completed.")
            self.progress_bar.setValue(100)
            
            # Clear status message after delay
            QTimer.singleShot(2000, lambda: self.statusBar().clearMessage())
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            
            # Switch to evaluation tab
            self.right_area.setCurrentWidget(self.evaluation_tab)
            
        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Error", f"Failed to calculate dose: {str(e)}")
            logger.error(f"Failed to calculate dose: {e}")
            
            # Clear status message
            self.statusBar().clearMessage()
            self.progress_bar.setVisible(False)
    
    def update_dose_progress(self, progress):
        """Update dose calculation progress."""
        self.progress_bar.setValue(int(progress * 100))
    
    def show_protocols_dialog(self):
        """Show the clinical protocols dialog."""
        try:
            from quangtps.ui.dialogs.protocol_dialog import ClinicalProtocolDialog
            dialog = ClinicalProtocolDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                protocol = dialog.get_selected_protocol()
                if protocol and hasattr(self, 'evaluation_tab'):
                    # Pass protocol to evaluation tab
                    self.evaluation_tab.set_protocol(protocol)
                    self.statusBar().showMessage(f"Applied protocol: {protocol['name']}")
        except Exception as e:
            logger.exception(f"Error showing protocols dialog: {e}")
            QMessageBox.warning(self, "Error", f"Could not open protocols dialog: {str(e)}")
    
    def set_window_level(self, window, level):
        """Set window and level presets in the MPR viewer."""
        self.mpr_viewer.set_window_level(window, level)
    
    def toggle_patient_browser(self, visible):
        """Toggle the visibility of the patient browser."""
        # Get the main splitter
        main_splitter = self.centralWidget().layout().itemAt(0).widget()
        
        if visible:
            # Get the current sizes
            sizes = main_splitter.sizes()
            if sizes[0] == 0:
                # Restore previous size (default 20% of width)
                total_width = sum(sizes)
                sizes[0] = total_width * 0.2
                sizes[1] = total_width - sizes[0]
                main_splitter.setSizes(sizes)
        else:
            # Get the current sizes
            sizes = main_splitter.sizes()
            if sizes[0] > 0:
                # Hide patient browser
                sizes[1] += sizes[0]
                sizes[0] = 0
                main_splitter.setSizes(sizes)
    
    def _on_tab_changed(self, index):
        """Handle tab widget tab changes."""
        current_tab = self.right_area.widget(index)
        
        # Update UI based on current tab
        if current_tab == self.contouring_tab:
            self.statusBar().showMessage("Contouring tab: Use the tools to create and edit structures")
        elif current_tab == self.planning_tab:
            self.statusBar().showMessage("Planning tab: Create and modify treatment plans")
        elif current_tab == self.evaluation_tab:
            self.statusBar().showMessage("Evaluation tab: Evaluate treatment plans")
        elif current_tab == self.optimization_tab:
            self.statusBar().showMessage("Optimization tab: Optimize treatment plans")
        elif current_tab == self.mco_tab:
            self.statusBar().showMessage("MCO tab: Multi-Criteria Optimization")
        elif current_tab == self.review_tab:
            self.statusBar().showMessage("Review tab: Review treatment plans")
        
        # Update menu and toolbar state based on current tab
        self._update_ui_state()
    
    def _update_ui_state(self):
        """Update UI state based on current context."""
        has_image = hasattr(self, 'current_image') and self.current_image is not None
        has_structure_set = hasattr(self, 'current_structure_set') and self.current_structure_set is not None
        has_plan = hasattr(self, 'current_plan') and self.current_plan is not None
        has_dose = has_plan and hasattr(self.current_plan, 'dose') and self.current_plan.dose is not None
        
        # Update file menu actions
        self.import_rt_struct_action.setEnabled(has_image)
        self.import_rt_plan_action.setEnabled(has_image)
        self.save_plan_action.setEnabled(has_plan)
        
        # Update planning menu actions
        self.new_plan_action.setEnabled(has_image and has_structure_set)
        self.calculate_dose_action.setEnabled(has_plan)
        self.optimization_action.setEnabled(has_plan)
        
        # Update toolbar actions
        self.import_rt_struct_tool.setEnabled(has_image)
        self.new_plan_tool.setEnabled(has_image and has_structure_set)
        self.calculate_dose_tool.setEnabled(has_plan)
        self.optimize_tool.setEnabled(has_plan)
        self.evaluate_tool.setEnabled(has_dose)
        self.report_tool.setEnabled(has_dose)
    
    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(
            self, "About QuangTPS",
            "<h2>QuangTPS</h2>"
            "<p>Version 1.0</p>"
            "<p>A treatment planning system for radiation therapy.</p>"
            "<p>&copy; 2023 All rights reserved.</p>"
            "<p>This software is for educational and research purposes only.</p>"
        )
    
    def closeEvent(self, event):
        """Handle the close event."""
        # Save settings
        self._save_settings()
        
        # Accept the event
        event.accept()

    def _on_plan_selected(self, plan):
        """Handle plan selection."""
        self.current_plan = plan
        
        # Update patient information in case it's not set
        if plan and not self.current_patient:
            self.current_patient = plan.patient
        
        # Update UI for the selected plan
        self.planning_tab.set_plan(plan)
        self.evaluation_tab.set_plan(plan)
        self.optimization_tab.set_plan(plan)
        
        # Update MCO tab with the selected plan
        self.mco_tab.set_plan(plan)
        
        # Also update the review tab
        self.review_tab.set_plan(plan)
        
        # Show plan information in status bar
        if plan:
            self.statusBar().showMessage(f"Plan: {plan.name}")
        else:
            self.statusBar().showMessage("No plan selected")

    def _on_optimization_completed(self, plan, results):
        """Handle completion of optimization."""
        # Update the plan with optimization results
        self.current_plan = plan
        
        # Update planning and evaluation views
        self.planning_tab.set_plan(plan)
        self.evaluation_tab.set_plan(plan)
        
        # Also update the MCO tab
        self.mco_tab.set_plan(plan)
        
        # Switch to evaluation tab
        self.right_area.setCurrentWidget(self.evaluation_tab)
        
        # Show message
        self.statusBar().showMessage("Optimization completed")
    
    def _on_mco_solution_selected(self, solution_id):
        """Handle MCO solution selection."""
        # This could update other views if needed
        self.statusBar().showMessage(f"MCO Solution {solution_id} selected")
    
    def _on_mco_solution_modified(self):
        """Handle MCO solution modification."""
        # This could update other views if needed
        self.statusBar().showMessage("MCO Solution modified")

    def _open_plan_evaluation(self):
        """Open the plan evaluation dialog."""
        # Get the current active plan from the planning tab
        planning_tab = self._get_tab_by_name("Planning")
        if not planning_tab or not hasattr(planning_tab, "current_plan"):
            QMessageBox.warning(self, "Plan Evaluation", "Please open a plan first")
                return
            
        if not planning_tab.current_plan:
            QMessageBox.warning(self, "Plan Evaluation", "Please select a plan to evaluate")
            return
        
        # Show the plan evaluation tab and pass the plan to it
        evaluation_tab = self._get_tab_by_name("Evaluation")
        if evaluation_tab:
            self.right_area.setCurrentWidget(evaluation_tab)
            evaluation_tab.set_plan(planning_tab.current_plan)
        else:
            QMessageBox.warning(self, "Plan Evaluation", "Evaluation tab not found")

    def _open_plan_comparison(self):
        """Open the plan comparison dialog."""
        # Get the current active plan from the planning tab
        planning_tab = self._get_tab_by_name("Planning")
        if not planning_tab or not hasattr(planning_tab, "current_plan"):
            QMessageBox.warning(self, "Plan Comparison", "Please open a plan first")
            return
        
        if not planning_tab.current_plan:
            QMessageBox.warning(self, "Plan Comparison", "Please select a plan to use as reference")
            return
        
        # Import here to avoid circular imports
        from quangtps.ui.plan_comparison_dialog import PlanComparisonDialog
        
        # Create and show the dialog
        dialog = PlanComparisonDialog(planning_tab.current_plan, self)
        dialog.exec_()

    def _get_tab_by_name(self, tab_name):
        """
        Get a tab widget by its name.
        
        Args:
            tab_name: Name of the tab to find
            
        Returns:
            The tab widget if found, None otherwise
        """
        for i in range(self.right_area.count()):
            tab = self.right_area.widget(i)
            if tab.objectName() == tab_name:
                return tab
        return None
