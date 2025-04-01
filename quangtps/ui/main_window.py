#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện chính của QuangTPS.

Module này triển khai cửa sổ chính của ứng dụng, 
điều khiển luồng công việc và quản lý các tab chính.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Type, Tuple, cast

import numpy as np
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QToolBar, QAction, QMenu, QMenuBar,
        QStatusBar, QDockWidget, QTabWidget, QMessageBox, QFileDialog,
        QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QDialog,
        QSystemTrayIcon
    )
    from PyQt5.QtCore import Qt, QSize, QTimer, QSettings, QPoint, QRect
    from PyQt5.QtGui import QIcon, QPixmap, QKeySequence, QMovie
except ImportError as e:
    print(f"Error importing PyQt5: {e}")
    sys.exit(1)

# Set up logging before any imports that might use it
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define PatientDB and ServiceRegistry at module level as fallback
PatientDB = None
ServiceRegistry = None

# Import các module nội bộ
try:
    from quangtps.ui.patient_tab import PatientTab
except ImportError as e:
    print(f"Error importing PatientTab: {e}")
    PatientTab = None

# Try to import the imaging tab, with fallback options
try:
    from quangtps.ui.imaging_tab import ImagingTab
except ImportError:
    from quangtps.ui.simple_imaging_tab import ImagingTab
    logger.warning("Using simplified ImagingTab implementation")

# Try to import the external beam planning tab
try:
    from quangtps.ui.external_beam_planning_tab import ExternalBeamPlanningTab
except ImportError:
    logger.warning("ExternalBeamPlanningTab not available, will use separate planning and dose tabs")
    ExternalBeamPlanningTab = None

# Try to import the planning tab, with fallback options
try:
    from quangtps.ui.planning_tab import PlanningTab
except ImportError:
    logger.warning("PlanningTab not available, features will be limited")
    class PlanningTab(QWidget):
        """Placeholder PlanningTab class"""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("Planning features not available", self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

# Continue with other imports
try:
    from quangtps.ui.dose_tab import DoseTab
except ImportError:
    logger.warning("DoseTab not available, features will be limited")
    class DoseTab(QWidget):
        """Placeholder DoseTab class"""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("Dose calculation features not available", self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

try:
    from quangtps.ui.treatment_tab import TreatmentTab
except ImportError:
    logger.warning("TreatmentTab not available, features will be limited")
    class TreatmentTab(QWidget):
        """Placeholder TreatmentTab class"""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("Treatment features not available", self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

try:
    from quangtps.ui.qa_tab import QATab 
except ImportError:
    logger.warning("QATab not available, features will be limited")
    class QATab(QWidget):
        """Placeholder QATab class"""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("QA features not available", self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

try:
    from quangtps.ui.reporting_tab import ReportingTab
except ImportError:
    logger.warning("ReportingTab not available, features will be limited")
    class ReportingTab(QWidget):
        """Placeholder ReportingTab class"""
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("Reporting features not available", self)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

# Update service imports to handle potential circular imports
from quangtps.core.services import ServiceRegistry
from quangtps.core.patient import Patient

try:
    from quangtps.database.patient_db import PatientDB, PatientDatabase
except ImportError as e:
    print(f"Error importing PatientDB: {e}")
    PatientDB = None

try:
    from quangtps.database.plan_db import PlanDB
except ImportError as e:
    print(f"Error importing PlanDB: {e}")
    PlanDB = None

try:
    from quangtps.ui.treatment_planning_tab import TreatmentPlanningTab
except ImportError as e:
    print(f"Error importing TreatmentPlanningTab: {e}")
    TreatmentPlanningTab = None

try:
    from quangtps.ui.dose_calculation_dialog import DoseCalculationDialog
except ImportError as e:
    print(f"Error importing DoseCalculationDialog: {e}")
    DoseCalculationDialog = None

try:
    from quangtps.ui.auto_segmentation_tool import AutoSegmentationTool
except ImportError as e:
    print(f"Error importing AutoSegmentationTool: {e}")
    AutoSegmentationTool = None

try:
    from quangtps.ui.segmentation_model_manager import SegmentationModelManager
except ImportError as e:
    print(f"Error importing SegmentationModelManager: {e}")
    SegmentationModelManager = None

try:
    from quangtps.ui.object_explorer import ObjectExplorerPanel
except ImportError as e:
    print(f"Error importing ObjectExplorerPanel: {e}")
    ObjectExplorerPanel = None

try:
    from quangtps.ui.dialogs.protocol_dialog import ClinicalProtocolDialog
except ImportError as e:
    print(f"Error importing ClinicalProtocolDialog: {e}")
    ClinicalProtocolDialog = None

try:
    from quangtps.planning.clinical_protocols import ClinicalProtocolManager
except ImportError as e:
    print(f"Error importing ClinicalProtocolManager: {e}")
    ClinicalProtocolManager = None

try:
    from quangtps.ui.optimization.mco_panel import MCOPanel
except ImportError as e:
    print(f"Error importing MCOPanel: {e}")
    MCOPanel = None

try:
    from quangtps.ui.dialogs.collision_detection_dialog import CollisionDetectionDialog
except ImportError as e:
    print(f"Error importing CollisionDetectionDialog: {e}")
    CollisionDetectionDialog = None

try:
    from quangtps.scripts.user_scripting import ScriptEditor
except ImportError as e:
    print(f"Error importing ScriptEditor: {e}")
    ScriptEditor = None

try:
    from quangtps.ui.patient_dashboard import PatientDashboard
except ImportError as e:
    print(f"Error importing PatientDashboard: {e}")
    PatientDashboard = None

try:
    from quangtps.administration.rt_admin import RTAdministration, QAManagement
except ImportError as e:
    print(f"Error importing RTAdministration: {e}")
    RTAdministration = None
    QAManagement = None

try:
    from quangtps.ui.dialogs.dicom_import_dialog import DicomImportDialog
except ImportError as e:
    print(f"Error importing DicomImportDialog: {e}")
    DicomImportDialog = None

try:
    from quangtps.common.paths import get_icon_path
except ImportError as e:
    # Define a fallback function
    def get_icon_path(icon_name):
        return os.path.join(os.path.dirname(__file__), "icons", "new_icons", f"{icon_name}.svg")

# Đường dẫn đến thư mục biểu tượng
ICON_DIR = os.path.join(os.path.dirname(__file__), "icons", "new_icons")

# Đảm bảo thư mục biểu tượng tồn tại
if not os.path.exists(ICON_DIR):
    os.makedirs(ICON_DIR, exist_ok=True)
    logger.warning(f"Đã tạo thư mục biểu tượng: {ICON_DIR}")

class MainWindow(QMainWindow):
    """
    Lớp cửa sổ chính của ứng dụng QuangTPS.
    
    Cửa sổ chính chứa các tab chức năng, thanh công cụ, menu,
    và các thành phần giao diện khác của hệ thống lập kế hoạch xạ trị.
    """
    
    def __init__(self, config=None):
        """
        Initialize the main window.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        super().__init__()
        
        # Set application style
        self._load_stylesheet()
        
        self.config = config or {}
        self.current_patient_id = None
        self.current_study_id = None
        self.current_series_id = None
        self.current_plan_id = None
        self.current_patient = None
        self.current_plan = None
        
        # Khởi tạo cơ sở dữ liệu
        try:
            if ServiceRegistry:
                self.service_registry = ServiceRegistry.get_instance()
                self.patient_db = self.service_registry.get_service('PatientDB')
                if not self.patient_db and PatientDB:
                    self.patient_db = PatientDB()
                    self.service_registry.register_service('PatientDB', self.patient_db)
            elif PatientDB:
                self.patient_db = PatientDB()
            else:
                # Create a minimal placeholder for PatientDB if not imported
                logger.warning("Using minimal Patient DB implementation")
                self.patient_db = type('DummyPatientDB', (), {
                    'get_patient': lambda self, patient_id: None,
                    'get_patients': lambda self: [],
                    'save_patient': lambda self, patient: None,
                    'delete_patient': lambda self, patient_id: None
                })()
        except Exception as e:
            logger.warning(f"Error initializing PatientDB: {e}")
            # Create a minimal placeholder for PatientDB
            self.patient_db = type('DummyPatientDB', (), {
                'get_patient': lambda self, patient_id: None,
                'get_patients': lambda self: [],
                'save_patient': lambda self, patient: None,
                'delete_patient': lambda self, patient_id: None
            })()
        
        # Tạo các tab luồng công việc
        self.workflow_tabs = QTabWidget()
        self.workflow_tabs.setTabPosition(QTabWidget.North)  # Eclipse has tabs at the top
        self.workflow_tabs.setDocumentMode(True)
        self.workflow_tabs.setElideMode(Qt.ElideRight)
        self.workflow_tabs.setTabsClosable(False)  # Eclipse doesn't have closable tabs
        self.workflow_tabs.setMovable(False)       # Eclipse tabs aren't movable
        
        # Add tabs for different workflow stages in Eclipse-like order
        self.tab_indexes = {}  # Store tab indexes for quick access
        tab_index = 0
        
        try:
            self.patient_tab = PatientTab()
            self.workflow_tabs.addTab(self.patient_tab, "Patient")
            self.tab_indexes['patient'] = tab_index
            tab_index += 1
        except Exception as e:
            logger.error(f"Could not initialize PatientTab: {e}")
            # Create a simple placeholder tab if PatientTab fails
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addWidget(QLabel("Patient Tab - Not Available"))
            placeholder_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.patient_tab = placeholder
            self.workflow_tabs.addTab(placeholder, "Patient")
            self.tab_indexes['patient'] = tab_index
            tab_index += 1
        
        # Thêm các tab chính
        try:
            if ImagingTab is not None:
                self.imaging_tab = ImagingTab()
                self.workflow_tabs.addTab(self.imaging_tab, "Imaging")
                self.tab_indexes['imaging'] = tab_index
                tab_index += 1
            else:
                raise ImportError("ImagingTab class is not defined")
        except Exception as e:
            logger.error(f"Could not initialize ImagingTab: {e}")
            # Create a simple placeholder tab if ImagingTab fails
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addWidget(QLabel("Imaging Tab - Not Available"))
            placeholder_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.imaging_tab = placeholder
            self.workflow_tabs.addTab(placeholder, "Imaging")
            self.tab_indexes['imaging'] = tab_index
            tab_index += 1
        
        # Structure tab (Eclipse adds a separate structure tab)
        try:
            from quangtps.ui.structure_tab import StructureTab
            self.structure_tab = StructureTab()
            self.workflow_tabs.addTab(self.structure_tab, "Structure")
            self.tab_indexes['structure'] = tab_index
            tab_index += 1
            logger.info("Structure tab initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize StructureTab: {e}")
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addWidget(QLabel("Structure Tab - Not Available"))
            placeholder_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.structure_tab = placeholder
            self.workflow_tabs.addTab(placeholder, "Structure")
            self.tab_indexes['structure'] = tab_index
            tab_index += 1
        
        try:
            if ExternalBeamPlanningTab is not None:
                self.external_beam_planning_tab = ExternalBeamPlanningTab()
                self.workflow_tabs.addTab(self.external_beam_planning_tab, "External Beam Planning")
                self.tab_indexes['external_beam_planning'] = tab_index
                tab_index += 1
                
                # Kết nối tín hiệu
                if hasattr(self.patient_tab, 'patient_loaded'):
                    self.patient_tab.patient_loaded.connect(self.external_beam_planning_tab.set_patient)
                
                if hasattr(self.external_beam_planning_tab, 'plan_created'):
                    self.external_beam_planning_tab.plan_created.connect(self._on_plan_selected)
                    
                if hasattr(self.external_beam_planning_tab, 'plan_updated'):
                    self.external_beam_planning_tab.plan_updated.connect(self._on_plan_selected)
                    
                logger.info("External Beam Planning tab initialized successfully")
            else:
                # Nếu không tải được, sử dụng tách biệt Planning và Dose tabs thay thế
                try:
                    if PlanningTab is not None:
                        self.planning_tab = PlanningTab()
                        self.workflow_tabs.addTab(self.planning_tab, "Planning")
                        self.tab_indexes['planning'] = tab_index
                        tab_index += 1
                        
                        # Kết nối tín hiệu
                        if hasattr(self.patient_tab, 'patient_loaded'):
                            self.patient_tab.patient_loaded.connect(self.planning_tab.set_patient)
                        
                        if hasattr(self.planning_tab, 'plan_created'):
                            self.planning_tab.plan_created.connect(self._on_plan_selected)
                            
                        if hasattr(self.planning_tab, 'plan_updated'):
                            self.planning_tab.plan_updated.connect(self._on_plan_selected)
                        
                        logger.info("Planning tab initialized successfully as fallback")
                    else:
                        raise ImportError("PlanningTab class is not defined")
                except Exception as e2:
                    logger.error(f"Could not initialize PlanningTab as fallback: {e2}")
                    placeholder = QWidget()
                    placeholder_layout = QVBoxLayout(placeholder)
                    placeholder_layout.addWidget(QLabel("Planning Tab - Not Available"))
                    placeholder_layout.addWidget(QLabel(f"Error: {str(e2)}"))
                    self.planning_tab = placeholder
                    self.workflow_tabs.addTab(placeholder, "Planning")
                    self.tab_indexes['planning'] = tab_index
                    tab_index += 1

                # Thêm tab Dose
                try:
                    if DoseTab is not None:
                        self.dose_tab = DoseTab()
                        self.workflow_tabs.addTab(self.dose_tab, "Dose")
                        self.tab_indexes['dose'] = tab_index
                        tab_index += 1
                        
                        # Kết nối tín hiệu
                        if hasattr(self.patient_tab, 'patient_loaded'):
                            self.patient_tab.patient_loaded.connect(self.dose_tab.set_patient)
                        
                        if hasattr(self.planning_tab, 'plan_created'):
                            self.planning_tab.plan_created.connect(self.dose_tab.set_plan)
                            
                        if hasattr(self.planning_tab, 'plan_updated'):
                            self.planning_tab.plan_updated.connect(self.dose_tab.set_plan)
                        
                        logger.info("Dose tab initialized successfully as fallback")
                    else:
                        raise ImportError("DoseTab class is not defined")
                except Exception as e3:
                    logger.error(f"Could not initialize DoseTab as fallback: {e3}")
                    placeholder = QWidget()
                    placeholder_layout = QVBoxLayout(placeholder)
                    placeholder_layout.addWidget(QLabel("Dose Tab - Not Available"))
                    placeholder_layout.addWidget(QLabel(f"Error: {str(e3)}"))
                    self.dose_tab = placeholder
                    self.workflow_tabs.addTab(placeholder, "Dose")
                    self.tab_indexes['dose'] = tab_index
                    tab_index += 1

        except Exception as e:
            logger.error(f"Error in _init_tabs: {e}")
            logger.debug("Exception details:", exc_info=True)

        # Chỉ tạo một tab Plan Evaluation 
        self._init_plan_evaluation_tab()
        self.tab_indexes['plan_evaluation'] = tab_index
        tab_index += 1
        
        self._init_treatment_tab()
        self.tab_indexes['treatment'] = tab_index
        tab_index += 1
        
        self._init_qa_tab()
        self.tab_indexes['qa'] = tab_index
        tab_index += 1
        
        self._init_reporting_tab()
        self.tab_indexes['reporting'] = tab_index
        tab_index += 1
        
        # Thiết lập cửa sổ
        self.setWindowTitle("QuangTPS - Hệ thống lập kế hoạch xạ trị mở")
        icon_path = os.path.join(ICON_DIR, "app_icon.svg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1200, 800)
        
        # Khởi tạo giao diện
        self._setup_ui()
        
        # Tạo menu và thanh công cụ
        self._setup_menus()
        self._create_toolbar()
        
        # Thiết lập trạng thái ban đầu của UI
        self._update_ui_state()
        
        logger.info("Khởi tạo cửa sổ chính QuangTPS hoàn tất")
        
        self.load_plugins()
    
    def _load_stylesheet(self):
        """
        Load the application stylesheet to create an Eclipse-like appearance.
        
        This applies the dark blue theme similar to Varian Eclipse's interface.
        """
        try:
            # First try to load the dark theme stylesheet
            dark_theme_path = os.path.join(os.path.dirname(__file__), "styles", "dark_theme.css")
            
            if os.path.exists(dark_theme_path):
                logger.info(f"Loading dark theme stylesheet from: {dark_theme_path}")
                with open(dark_theme_path, 'r', encoding='utf-8') as f:
                    stylesheet = f.read()
                    self.setStyleSheet(stylesheet)
                    
                # Apply additional styling for specific elements
                self.workflow_tabs.setTabPosition(QTabWidget.North)
                self.workflow_tabs.setMovable(True)
                self.workflow_tabs.setDocumentMode(True)  # More Eclipse-like tab appearance
                
                if hasattr(self, 'patient_tab'):
                    self.patient_tab.setObjectName("PatientTab")
                
                if hasattr(self, 'imaging_tab'):
                    self.imaging_tab.setObjectName("ImagingTab")
                
                if hasattr(self, 'planning_tab'):
                    self.planning_tab.setObjectName("PlanningTab")
                
                if hasattr(self, 'dose_tab'):
                    self.dose_tab.setObjectName("DoseTab")
                
                if hasattr(self, 'plan_evaluation_tab'):
                    self.plan_evaluation_tab.setObjectName("PlanEvaluationTab")
                
                if hasattr(self, 'qa_tab'):
                    self.qa_tab.setObjectName("QATab")
                
                if hasattr(self, 'treatment_tab'):
                    self.treatment_tab.setObjectName("TreatmentTab")
                
                if hasattr(self, 'reporting_tab'):
                    self.reporting_tab.setObjectName("ReportingTab")
                
                logger.info("Dark theme stylesheet applied successfully")
                return True
            else:
                logger.warning(f"Dark theme stylesheet not found at: {dark_theme_path}")
        except Exception as e:
            logger.error(f"Failed to load stylesheet: {e}")
            logger.debug("Exception details:", exc_info=True)
        
        return False
    
    def load_plugins(self):
        """Load and initialize plugins for extending system functionality.
        
        This method scans the plugins directory and loads any compatible plugins
        to extend the system's functionality.
        """
        logger.info("Checking for plugins...")
        
        # Get the plugins directory
        plugins_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugins")
        
        # Check if plugins directory exists
        if not os.path.exists(plugins_dir):
            logger.info("No plugins directory found. Creating one.")
            try:
                os.makedirs(plugins_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create plugins directory: {e}")
            return
            
        # Check for Python files in the plugins directory
        plugin_files = [f for f in os.listdir(plugins_dir) if f.endswith('.py') and f != '__init__.py']
        
        if not plugin_files:
            logger.info("No plugins found.")
            return
            
        logger.info(f"Found {len(plugin_files)} potential plugins.")
        
        # Try to load each plugin
        for plugin_file in plugin_files:
            plugin_name = os.path.splitext(plugin_file)[0]
            logger.info(f"Attempting to load plugin: {plugin_name}")
            
            try:
                # Add the plugins directory to the path temporarily
                if plugins_dir not in sys.path:
                    sys.path.insert(0, plugins_dir)
                
                # Import the plugin module
                plugin_module = __import__(plugin_name)
                
                # Check if the module has a register_plugin function
                if hasattr(plugin_module, 'register_plugin'):
                    plugin_module.register_plugin(self)
                    logger.info(f"Successfully loaded plugin: {plugin_name}")
                else:
                    logger.warning(f"Plugin {plugin_name} does not have a register_plugin function.")
            
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_name}: {e}")
            
            finally:
                # Remove the plugins directory from the path
                if plugins_dir in sys.path:
                    sys.path.remove(plugins_dir)
                    
        # Refresh UI after loading plugins
        self._update_ui_state()
        logger.info("Plugin loading completed.")
    
    def _setup_ui(self):
        """Set up the main UI."""
        # Create central widget with splitter
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Temporarily disable ObjectExplorerPanel
        # Add Object Explorer panel on the left side
        # self.object_explorer = ObjectExplorerPanel(self)
        # self.object_explorer.patient_selected.connect(self._on_patient_selected_from_explorer)
        # self.object_explorer.plan_selected.connect(self._on_plan_selected_from_explorer)
        # self.object_explorer.structure_selected.connect(self._on_structure_selected_from_explorer)
        
        # self.main_splitter.addWidget(self.object_explorer)
        
        # Add workflow tab widget
        self.main_splitter.addWidget(self.workflow_tabs)
        
        # Set initial sizes (25% explorer, 75% main area)
        # self.main_splitter.setSizes([250, 750])
        
        main_layout.addWidget(self.main_splitter)
        self.setCentralWidget(central_widget)
        
        # Add a status message
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("QuangTPS is running in simplified mode")
        
        logger.info("Set up simplified UI without Object Explorer")
    
    def _setup_menus(self):
        """Set up the menu bar."""
        # File menu
        self.file_menu = self.menuBar().addMenu("&File")
        
        self.new_patient_action = self.file_menu.addAction("&New Patient...")
        self.new_patient_action.setShortcut("Ctrl+N")
        self.new_patient_action.triggered.connect(self.on_new_patient)
        
        self.open_patient_action = self.file_menu.addAction("&Open Patient...")
        self.open_patient_action.setShortcut("Ctrl+O")
        self.open_patient_action.triggered.connect(self.on_open_patient)
        
        self.file_menu.addSeparator()
        
        self.import_menu = self.file_menu.addMenu("&Import")
        
        self.import_dicom_action = self.import_menu.addAction("Import DICOM...")
        self.import_dicom_action.triggered.connect(self.on_import_dicom)
        
        self.import_rtstruct_action = self.import_menu.addAction("Import RTSTRUCT...")
        self.import_rtstruct_action.triggered.connect(self._import_rtstruct)
        
        self.import_rtplan_action = self.import_menu.addAction("Import RTPLAN...")
        self.import_rtplan_action.triggered.connect(self._import_rtplan)
        
        self.import_rtdose_action = self.import_menu.addAction("Import RTDOSE...")
        self.import_rtdose_action.triggered.connect(self._import_rtdose)
        
        self.file_menu.addSeparator()
        
        self.save_action = self.file_menu.addAction("&Save")
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._save_patient)
        
        self.save_as_action = self.file_menu.addAction("Save &As...")
        self.save_as_action.setShortcut("Ctrl+Shift+S")
        self.save_as_action.triggered.connect(self._save_patient_as)
        
        self.file_menu.addSeparator()
        
        self.export_menu = self.file_menu.addMenu("&Export")
        
        self.export_dicom_action = self.export_menu.addAction("Export DICOM...")
        self.export_dicom_action.triggered.connect(self._export_dicom)
        
        self.export_pdf_action = self.export_menu.addAction("Export Report as PDF...")
        self.export_pdf_action.triggered.connect(self._export_pdf)
        
        self.file_menu.addSeparator()
        
        self.exit_action = self.file_menu.addAction("E&xit")
        self.exit_action.setShortcut("Alt+F4")
        self.exit_action.triggered.connect(self.close)
        
        # Edit menu
        self.edit_menu = self.menuBar().addMenu("&Edit")
        
        self.undo_action = self.edit_menu.addAction("&Undo")
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self._undo)
        
        self.redo_action = self.edit_menu.addAction("&Redo")
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self._redo)
        
        self.edit_menu.addSeparator()
        
        self.preferences_action = self.edit_menu.addAction("&Preferences...")
        self.preferences_action.triggered.connect(self._show_preferences)
        
        # View menu
        self.view_menu = self.menuBar().addMenu("&View")
        
        self.view_patient_browser_action = self.view_menu.addAction("Patient Browser")
        self.view_patient_browser_action.setCheckable(True)
        self.view_patient_browser_action.setChecked(True)
        self.view_patient_browser_action.triggered.connect(self._toggle_patient_browser)
        
        self.view_image_tools_action = self.view_menu.addAction("Image Tools")
        self.view_image_tools_action.setCheckable(True)
        self.view_image_tools_action.setChecked(True)
        self.view_image_tools_action.triggered.connect(self._toggle_image_tools)
        
        self.view_menu.addSeparator()
        
        self.view_full_screen_action = self.view_menu.addAction("Full Screen")
        self.view_full_screen_action.setShortcut("F11")
        self.view_full_screen_action.setCheckable(True)
        self.view_full_screen_action.triggered.connect(self._toggle_full_screen)
        
        # Tools menu
        self.tools_menu = self.menuBar().addMenu("&Tools")
        
        self.contour_editor_action = self.tools_menu.addAction("Contour Editor")
        self.contour_editor_action.triggered.connect(self._open_contour_editor)
        
        self.beam_manager_action = self.tools_menu.addAction("Beam Manager")
        self.beam_manager_action.triggered.connect(self._open_beam_manager)
        
        self.dose_calculator_action = self.tools_menu.addAction("Dose Calculator")
        self.dose_calculator_action.triggered.connect(self._open_dose_calculator)
        
        # Add the advanced dose calculator option
        self.advanced_dose_calculator_action = self.tools_menu.addAction("Advanced Dose Calculator")
        self.advanced_dose_calculator_action.triggered.connect(self._open_advanced_dose_calculator)
        
        self.plan_optimizer_action = self.tools_menu.addAction("Plan Optimizer")
        self.plan_optimizer_action.triggered.connect(self._open_plan_optimizer)
        
        self.dvh_analyzer_action = self.tools_menu.addAction("DVH Analyzer")
        self.dvh_analyzer_action.triggered.connect(self._open_dvh_analyzer)
        
        # Add collision detection to tools menu
        self.collision_detection_action = self.tools_menu.addAction("Collision Detection")
        self.collision_detection_action.triggered.connect(self._open_collision_detection)
        
        # Help menu
        self.help_menu = self.menuBar().addMenu("&Help")
        
        self.about_action = self.help_menu.addAction("&About")
        self.about_action.triggered.connect(self._show_about_dialog)
        
        self.help_action = self.help_menu.addAction("&Help")
        self.help_action.setShortcut("F1")
        self.help_action.triggered.connect(self._show_help)
        
        # Add Clinical Protocols menu
        protocol_menu = self.menuBar().addMenu("&Protocols")
        
        # Apply protocol action
        apply_protocol_action = QAction("Apply Clinical Protocol...", self)
        apply_protocol_action.triggered.connect(self._show_protocol_dialog)
        protocol_menu.addAction(apply_protocol_action)
        
        # Manage protocols action
        manage_protocols_action = QAction("Manage Clinical Protocols...", self)
        manage_protocols_action.triggered.connect(self._manage_protocols)
        protocol_menu.addAction(manage_protocols_action)
        
        protocol_menu.addSeparator()
        
        # Create protocol templates action
        create_protocol_action = QAction("Create Protocol Template...", self)
        create_protocol_action.triggered.connect(self._create_protocol_template)
        protocol_menu.addAction(create_protocol_action)
        
        # Optimization menu
        optimization_menu = self.menuBar().addMenu("Optimization")
        
        # MCO actions
        start_mco_action = QAction("Start Multi-Criteria Optimization", self)
        start_mco_action.triggered.connect(self._on_start_mco)
        optimization_menu.addAction(start_mco_action)
        
        # Update Administration menu or add if not exists
        self.admin_menu = self.menuBar().addMenu("Administration")
        
        self.user_management_action = self.admin_menu.addAction("User Management")
        self.user_management_action.triggered.connect(self._open_user_management)
        
        self.machine_management_action = self.admin_menu.addAction("Machine Management")
        self.machine_management_action.triggered.connect(self._open_machine_management)
        
        self.system_settings_action = self.admin_menu.addAction("System Settings")
        self.system_settings_action.triggered.connect(self._open_system_settings)
        
        self.license_info_action = self.admin_menu.addAction("License Information")
        self.license_info_action.triggered.connect(self._show_license_info)
        
        self.backup_action = self.admin_menu.addAction("Backup/Restore")
        self.backup_action.triggered.connect(self._open_backup_restore)
        
        # Add Scripts menu
        self.scripts_menu = self.menuBar().addMenu("Scripts")
        
        self.script_editor_action = self.scripts_menu.addAction("Script Editor")
        self.script_editor_action.triggered.connect(self._open_script_editor)
        
        self.run_script_action = self.scripts_menu.addAction("Run Script...")
        self.run_script_action.triggered.connect(self._run_script)
        
        self.scripts_menu.addSeparator()
        
        self.script_samples_menu = self.scripts_menu.addMenu("Sample Scripts")
        self.sample_dvh_script_action = self.script_samples_menu.addAction("DVH Analysis")
        self.sample_dvh_script_action.triggered.connect(lambda: self._load_sample_script("dvh_analysis"))
        
        self.sample_plan_script_action = self.script_samples_menu.addAction("Auto-Planning")
        self.sample_plan_script_action.triggered.connect(lambda: self._load_sample_script("auto_planning"))
        
        self.sample_qa_script_action = self.script_samples_menu.addAction("QA Report")
        self.sample_qa_script_action.triggered.connect(lambda: self._load_sample_script("qa_report"))
    
    def _create_toolbar(self):
        """Tạo thanh công cụ."""
        # Thanh công cụ chính
        self.main_toolbar = self.addToolBar("Thanh công cụ chính")
        self.main_toolbar.setIconSize(QSize(24, 24))
        
        # Thêm các action vào thanh công cụ
        new_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "new_patient.svg")), "Bệnh nhân mới", self)
        new_patient_action.triggered.connect(self.on_new_patient)
        self.main_toolbar.addAction(new_patient_action)
        
        open_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "open_patient.svg")), "Mở bệnh nhân", self)
        open_patient_action.triggered.connect(self.on_open_patient)
        self.main_toolbar.addAction(open_patient_action)
        
        self.main_toolbar.addSeparator()
        
        save_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "save.svg")), "Lưu", self)
        save_action.triggered.connect(self._save_patient)
        self.main_toolbar.addAction(save_action)
        
        self.main_toolbar.addSeparator()
        
        import_dicom_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "import_dicom.svg")), "Nhập DICOM", self)
        import_dicom_action.triggered.connect(self.on_import_dicom)
        self.main_toolbar.addAction(import_dicom_action)
        
        export_dicom_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "export_dicom.svg")), "Xuất DICOM", self)
        export_dicom_action.triggered.connect(self._export_dicom)
        self.main_toolbar.addAction(export_dicom_action)
        
        self.main_toolbar.addSeparator()
        
        # Công cụ đo lường
        measurement_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "measure.svg")), "Đo lường", self)
        measurement_action.triggered.connect(self._show_measurement_tools)
        self.main_toolbar.addAction(measurement_action)
        
        # Công cụ Window/Level
        window_level_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "window_level.svg")), "Window/Level", self)
        window_level_action.triggered.connect(self._show_window_level_tools)
        self.main_toolbar.addAction(window_level_action)
        
        # Thêm nút thu phóng
        zoom_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "zoom.svg")), "Thu phóng", self)
        zoom_action.triggered.connect(self._show_zoom_tools)
        self.main_toolbar.addAction(zoom_action)
        
        self.main_toolbar.addSeparator()
        
        # Công cụ contour
        contour_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "contour.svg")), "Vẽ contour", self)
        contour_action.triggered.connect(self._open_contour_editor)
        self.main_toolbar.addAction(contour_action)
        
        # Công cụ beam
        beam_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "beam.svg")), "Quản lý beam", self)
        beam_action.triggered.connect(self._open_beam_manager)
        self.main_toolbar.addAction(beam_action)
        
        self.main_toolbar.addSeparator()
        
        # Công cụ tính toán liều
        dose_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "dose.svg")), "Tính toán liều", self)
        dose_action.triggered.connect(self._open_dose_calculator)
        self.main_toolbar.addAction(dose_action)
        
        # Công cụ tối ưu hóa kế hoạch
        optimize_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "optimize.svg")), "Tối ưu hóa kế hoạch", self)
        optimize_action.triggered.connect(self._open_plan_optimizer)
        self.main_toolbar.addAction(optimize_action)
        
        # Công cụ phân tích DVH
        dvh_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "dvh.svg")), "Phân tích DVH", self)
        dvh_action.triggered.connect(self._open_dvh_analyzer)
        self.main_toolbar.addAction(dvh_action)
        
        self.main_toolbar.addSeparator()
        
        # Trợ giúp
        help_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "help.svg")), "Trợ giúp", self)
        help_action.triggered.connect(self._show_help)
        self.main_toolbar.addAction(help_action)
    
    def _update_ui_state(self):
        """Update the UI state based on current selections."""
        has_patient = self.current_patient_id is not None
        has_plan = self.current_plan_id is not None
        
        # Update menu actions
        if hasattr(self, 'import_rtplan_action'):
            self.import_rtplan_action.setEnabled(has_patient)
        if hasattr(self, 'import_rtdose_action'):
            self.import_rtdose_action.setEnabled(has_patient)
        if hasattr(self, 'export_dicom_action'):
            self.export_dicom_action.setEnabled(has_plan)
        if hasattr(self, 'save_patient_action'):
            self.save_patient_action.setEnabled(has_patient)
            
        logger.debug("UI state updated")
    
    def run(self):
        """
        Run the main window application.
        
        This method shows the window, sets focus to the first workflow step (Patient),
        and displays a welcome message in the status bar.
        """
        # Show the window
        self.show()
        
        # Focus on the patient tab as the first workflow step
        self.workflow_tabs.setCurrentIndex(0)
        
        # Set welcome message
        self.statusBar().showMessage("Welcome to QuangTPS - Open Source Radiotherapy Treatment Planning System", 5000)
        
        # Update UI state
        self._update_ui_state()
        
        logger.info("Main window is running")
    
    def _on_patient_selected(self, patient_id):
        """Handle patient selection."""
        self.current_patient_id = patient_id
        logger.info(f"Selected patient: {patient_id}")
        
    def _new_patient(self):
        """Tạo một bệnh nhân mới."""
        self.patient_tab.create_new_patient()
    
    def _open_patient(self):
        """Mở một bệnh nhân hiện có."""
        # Đã được xử lý bởi PatientBrowser
        pass
    
    def _import_dicom(self):
        """Import DICOM files."""
        logger.info("Import DICOM action triggered")
        QMessageBox.information(self, "Not Implemented", "DICOM import will be implemented in a future update.")
        
    def _import_rtstruct(self):
        """Import RTSTRUCT files."""
        logger.info("Import RTSTRUCT action triggered")
        QMessageBox.information(self, "Not Implemented", "RTSTRUCT import will be implemented in a future update.")
        
    def _import_rtplan(self):
        """Import RTPLAN files."""
        logger.info("Import RTPLAN action triggered")
        QMessageBox.information(self, "Not Implemented", "RTPLAN import will be implemented in a future update.")
        
    def _import_rtdose(self):
        """Import RTDOSE files."""
        logger.info("Import RTDOSE action triggered")
        QMessageBox.information(self, "Not Implemented", "RTDOSE import will be implemented in a future update.")
        
    def _export_dicom(self):
        """Export to DICOM format."""
        logger.info("Export DICOM action triggered")
        QMessageBox.information(self, "Not Implemented", "DICOM export will be implemented in a future update.")
        
    def _export_pdf(self):
        """Export report as PDF."""
        logger.info("Export PDF action triggered")
        QMessageBox.information(self, "Not Implemented", "PDF export will be implemented in a future update.")
        
    def _save_patient(self):
        """Save the current patient."""
        logger.info("Save patient action triggered")
        QMessageBox.information(self, "Not Implemented", "Save patient will be implemented in a future update.")
        
    def _save_patient_as(self):
        """Save the current patient with a new name."""
        logger.info("Save patient as action triggered")
        QMessageBox.information(self, "Not Implemented", "Save patient as will be implemented in a future update.")
        
    def _undo(self):
        """Undo the last action."""
        logger.info("Undo action triggered")
        QMessageBox.information(self, "Not Implemented", "Undo will be implemented in a future update.")
        
    def _redo(self):
        """Redo the last undone action."""
        logger.info("Redo action triggered")
        QMessageBox.information(self, "Not Implemented", "Redo will be implemented in a future update.")
        
    def _show_preferences(self):
        """Show the preferences dialog."""
        logger.info("Preferences action triggered")
        QMessageBox.information(self, "Not Implemented", "Preferences will be implemented in a future update.")
        
    def _toggle_patient_browser(self, checked):
        """Toggle the patient browser visibility."""
        logger.info(f"Toggle patient browser action triggered: {checked}")
        QMessageBox.information(self, "Not Implemented", "Patient browser toggle will be implemented in a future update.")
        
    def _toggle_image_tools(self, checked):
        """Toggle the image tools visibility."""
        logger.info(f"Toggle image tools action triggered: {checked}")
        QMessageBox.information(self, "Not Implemented", "Image tools toggle will be implemented in a future update.")
        
    def _toggle_full_screen(self, checked):
        """Toggle full screen mode."""
        logger.info(f"Toggle full screen action triggered: {checked}")
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
        
    def _open_contour_editor(self):
        """Open the contour editor."""
        logger.info("Contour editor action triggered")
        QMessageBox.information(self, "Not Implemented", "Contour editor will be implemented in a future update.")
        
    def _open_beam_manager(self):
        """Open the beam manager."""
        logger.info("Beam manager action triggered")
        QMessageBox.information(self, "Not Implemented", "Beam manager will be implemented in a future update.")
        
    def _open_dose_calculator(self):
        """Open the dose calculator."""
        logger.info("Dose calculator action triggered")
        QMessageBox.information(self, "Not Implemented", "Dose calculator will be implemented in a future update.")
        
    def _open_advanced_dose_calculator(self):
        """Open the advanced dose calculator."""
        logger.info("Advanced dose calculator action triggered")
        QMessageBox.information(self, "Not Implemented", "Advanced dose calculator will be implemented in a future update.")
        
    def _open_plan_optimizer(self):
        """Open the plan optimizer."""
        logger.info("Plan optimizer action triggered")
        QMessageBox.information(self, "Not Implemented", "Plan optimizer will be implemented in a future update.")
        
    def _open_dvh_analyzer(self):
        """Open the DVH analyzer."""
        logger.info("DVH analyzer action triggered")
        QMessageBox.information(self, "Not Implemented", "DVH analyzer will be implemented in a future update.")
        
    def _open_collision_detection(self):
        """Open the collision detection dialog."""
        logger.info("Collision detection action triggered")
        QMessageBox.information(self, "Not Implemented", "Collision detection will be implemented in a future update.")
        
    def _show_about_dialog(self):
        """Show the about dialog."""
        logger.info("About action triggered")
        QMessageBox.information(self, "About QuangTPS", "QuangTPS - Treatment Planning System\nVersion 0.3.8\nDeveloped by Quang Team")
        
    def _show_help(self):
        """Show the help documentation."""
        logger.info("Help action triggered")
        QMessageBox.information(self, "Not Implemented", "Help will be implemented in a future update.")
        
    def _open_user_management(self):
        """Open the user management dialog."""
        logger.info("User management action triggered")
        QMessageBox.information(self, "Not Implemented", "User management will be implemented in a future update.")
        
    def _open_machine_management(self):
        """Open the machine management dialog."""
        logger.info("Machine management action triggered")
        QMessageBox.information(self, "Not Implemented", "Machine management will be implemented in a future update.")
        
    def _open_system_settings(self):
        """Open the system settings dialog."""
        logger.info("System settings action triggered")
        QMessageBox.information(self, "Not Implemented", "System settings will be implemented in a future update.")
        
    def _show_license_info(self):
        """Show license information."""
        logger.info("License information action triggered")
        QMessageBox.information(self, "Not Implemented", "License information will be implemented in a future update.")
        
    def _open_backup_restore(self):
        """Open the backup/restore dialog."""
        logger.info("Backup/restore action triggered")
        QMessageBox.information(self, "Not Implemented", "Backup/restore will be implemented in a future update.")
        
    def _open_script_editor(self):
        """Open the script editor."""
        logger.info("Script editor action triggered")
        QMessageBox.information(self, "Not Implemented", "Script editor will be implemented in a future update.")
        
    def _run_script(self):
        """Run a script."""
        logger.info("Run script action triggered")
        QMessageBox.information(self, "Not Implemented", "Run script will be implemented in a future update.")
        
    def _load_sample_script(self, script_name):
        """Load a sample script."""
        logger.info(f"Load sample script action triggered: {script_name}")
        QMessageBox.information(self, "Not Implemented", f"Loading sample script '{script_name}' will be implemented in a future update.")

    def _show_protocol_dialog(self):
        """Show the clinical protocol dialog."""
        if not self.current_patient:
            QMessageBox.warning(self, "No Patient", "Please open a patient first.")
            return
            
        current_plan = None
        if hasattr(self.planning_tab, "current_plan"):
            current_plan = self.planning_tab.current_plan
            
        if not current_plan:
            QMessageBox.warning(self, "No Plan", "Please create or select a plan first.")
            return
            
        dialog = ClinicalProtocolDialog(self.current_patient, current_plan, self)
        dialog.protocol_applied.connect(self._apply_clinical_protocol)
        dialog.exec_()
    
    def _apply_clinical_protocol(self, protocol, application_data):
        """
        Apply a clinical protocol to the current plan.
        
        Args:
            protocol: The selected clinical protocol
            application_data: Dictionary containing application options and templates
        """
        if not self.current_patient or not hasattr(self.planning_tab, "current_plan") or not self.planning_tab.current_plan:
            return
            
        current_plan = self.planning_tab.current_plan
        options = application_data.get("options", {})
        
        # Apply prescription if selected
        if options.get("apply_prescription") and application_data.get("prescription_template"):
            self._apply_prescription_template(current_plan, application_data["prescription_template"])
        
        # Apply beam template if selected
        if options.get("apply_beam_template") and application_data.get("beam_template"):
            template_name, template_data = application_data["beam_template"]
            self._apply_beam_template(current_plan, template_name, template_data)
        
        # Apply optimization template if selected
        if options.get("apply_optimization") and application_data.get("optimization_template"):
            template_name, template_data = application_data["optimization_template"]
            self._apply_optimization_template(current_plan, template_name, template_data)
        
        # Refresh UI
        self.planning_tab.refresh()
        
        # Show confirmation
        QMessageBox.information(self, "Protocol Applied", 
                               f"The protocol '{protocol.name}' has been applied to the current plan.")
    
    def _apply_prescription_template(self, plan, template):
        """Apply a prescription template to a plan."""
        if not plan or not template:
            return
            
        # Create a new prescription from the template
        if hasattr(plan, "create_prescription_from_template"):
            plan.create_prescription_from_template(template)
        else:
            logger.warning("Plan does not have create_prescription_from_template method")
    
    def _apply_beam_template(self, plan, template_name, template_data):
        """Apply a beam template to a plan."""
        if not plan or not template_data:
            return
            
        # Create beams from template
        if hasattr(plan, "create_beams_from_template"):
            plan.create_beams_from_template(template_name, template_data)
        else:
            logger.warning("Plan does not have create_beams_from_template method")
    
    def _apply_optimization_template(self, plan, template_name, template_data):
        """Apply an optimization template to a plan."""
        if not plan or not template_data:
            return
            
        # Create optimization objectives from template
        if hasattr(plan, "create_objectives_from_template"):
            plan.create_objectives_from_template(template_name, template_data)
        else:
            logger.warning("Plan does not have create_objectives_from_template method")
    
    def _manage_protocols(self):
        """Show the protocol management dialog."""
        # This would open a dialog for managing clinical protocols
        # Not implemented in this edit
        QMessageBox.information(self, "Not Implemented", "Protocol management will be implemented in a future update.")
    
    def _create_protocol_template(self):
        """Show the protocol template creation dialog."""
        # This would open a dialog for creating a new protocol template
        # Not implemented in this edit
        QMessageBox.information(self, "Not Implemented", "Protocol template creation will be implemented in a future update.")

    def _on_start_mco(self):
        """Start multi-criteria optimization for the current plan."""
        if not self.current_plan:
            QMessageBox.warning(self, "Warning", "No plan selected")
            return
        
        # Switch to the MCO tab
        for i in range(self.workflow_tabs.count()):
            if self.workflow_tabs.tabText(i) == "MCO":
                self.workflow_tabs.setCurrentIndex(i)
                break
        
        # Set the current plan in the MCO panel
        self.mco_panel.set_plan(self.current_plan)
        
        # Initialize the MCO engine
        success = self.mco_panel.initialize_mco()
        if not success:
            QMessageBox.warning(self, "Warning", "Failed to initialize MCO")

    def _on_plan_updated(self, plan):
        """Handle plan updates from the MCO panel."""
        if self.current_plan and plan:
            # Update the current plan with the new one
            self.current_plan = plan
            
            # Update other panels that show plan information
            for i in range(self.workflow_tabs.count()):
                tab_widget = self.workflow_tabs.widget(i)
                if hasattr(tab_widget, 'set_plan'):
                    tab_widget.set_plan(plan)
            
            # Update the object explorer
            if hasattr(self, 'object_explorer'):
                self.object_explorer.refresh()

    def on_open_patient_dashboard(self):
        """Open the patient dashboard."""
        dashboard = PatientDashboard(self)
        dashboard.patient_selected.connect(self._on_dashboard_patient_selected)
        dashboard.plan_selected.connect(self._on_dashboard_plan_selected)
        dashboard.action_triggered.connect(self._on_dashboard_action)
        
        dashboard_tab_index = self._add_or_focus_tab(dashboard, "Patient Dashboard")
        if dashboard_tab_index >= 0:
            self.main_tab_widget.setCurrentIndex(dashboard_tab_index)

    def _on_dashboard_patient_selected(self, patient):
        """Handle patient selection from the dashboard."""
        self.current_patient = patient
        self._update_patient_display()

    def _on_dashboard_plan_selected(self, plan):
        """Handle plan selection from the dashboard."""
        self.current_plan = plan
        self._update_plan_display()
        
        # Open relevant tabs for the plan
        self._show_planning_tab()

    def _on_dashboard_action(self, action, patient):
        """Handle actions from the dashboard."""
        if action == "new_patient":
            self._new_patient()
        elif action == "new_plan" and patient:
            self.current_patient = patient
            self._new_plan()
        elif action == "new_appointment" and patient:
            # TODO: Implement appointment creation
            pass

    def _add_or_focus_tab(self, widget, title):
        """Add a new tab with the widget or focus an existing tab with the same title."""
        # Check if a tab with this title already exists
        for i in range(self.main_tab_widget.count()):
            if self.main_tab_widget.tabText(i) == title:
                return i
        
        # Add a new tab
        index = self.main_tab_widget.addTab(widget, title)
        return index

    def _show_measurement_tools(self):
        """Show measurement tools."""
        logger.info("Measurement tools action triggered")
        QMessageBox.information(self, "Not Implemented", "Measurement tools will be implemented in a future update.")
        
    def _show_window_level_tools(self):
        """Show window/level tools."""
        logger.info("Window/level tools action triggered")
        QMessageBox.information(self, "Not Implemented", "Window/level tools will be implemented in a future update.")
        
    def _show_zoom_tools(self):
        """Show zoom tools."""
        logger.debug("Showing zoom tools")
        QMessageBox.information(self, "Zoom Tools", "Zoom tools not implemented in simplified version.")
        
    # Add alias methods for compatibility
    def on_new_patient(self):
        """Alias for _new_patient for compatibility."""
        return self._new_patient()
        
    def on_import_dicom(self):
        """Alias for _import_dicom for compatibility."""
        return self._import_dicom()
        
    def on_open_patient(self):
        """Alias for _open_patient for compatibility."""
        return self._open_patient()

    def _init_dose_tab(self):
        """Initialize the dose tab with error handling."""
        try:
            self.dose_tab = DoseTab()
            self.workflow_tabs.addTab(self.dose_tab, "Dose")
            logger.info("Dose tab initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize DoseTab: {e}")
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addWidget(QLabel("Dose Tab - Not Available"))
            placeholder_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.dose_tab = placeholder
            self.workflow_tabs.addTab(placeholder, "Dose")
    
    def _init_treatment_tab(self):
        """Initialize the treatment tab with error handling."""
        try:
            self.treatment_tab = TreatmentTab()
            self.workflow_tabs.addTab(self.treatment_tab, "Treatment")
            logger.info("Treatment tab initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize TreatmentTab: {e}")
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addWidget(QLabel("Treatment Tab - Not Available"))
            placeholder_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.treatment_tab = placeholder
            self.workflow_tabs.addTab(placeholder, "Treatment")
    
    def _init_qa_tab(self):
        """Initialize the QA tab with error handling."""
        try:
            self.qa_tab = QATab()
            self.workflow_tabs.addTab(self.qa_tab, "QA")
            logger.info("QA tab initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize QATab: {e}")
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addWidget(QLabel("QA Tab - Not Available"))
            placeholder_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.qa_tab = placeholder
            self.workflow_tabs.addTab(placeholder, "QA")
    
    def _init_reporting_tab(self):
        """Initialize the reporting tab with error handling."""
        try:
            self.reporting_tab = ReportingTab()
            self.workflow_tabs.addTab(self.reporting_tab, "Reporting")
            logger.info("Reporting tab initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize ReportingTab: {e}")
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            placeholder_layout.addWidget(QLabel("Reporting Tab - Not Available"))
            placeholder_layout.addWidget(QLabel(f"Error: {str(e)}"))
            self.reporting_tab = placeholder
            self.workflow_tabs.addTab(placeholder, "Reporting")

    def _init_plan_evaluation_tab(self):
        """
        Initialize the Plan Evaluation tab for DVH analysis and plan quality assessment.
        
        This tab provides comprehensive evaluation capabilities similar to Eclipse's
        Plan Evaluation workspace.
        """
        try:
            from quangtps.ui.plan_evaluation import PlanEvaluationTab
            
            logger.info("Initializing Plan Evaluation tab")
            self.plan_evaluation_tab = PlanEvaluationTab(self)
            
            # Add tab to workflow
            self.workflow_tabs.addTab(self.plan_evaluation_tab, "Plan Evaluation")
            
            # Connect signals for plan updates
            if hasattr(self, 'planning_tab') and hasattr(self.planning_tab, 'plan_created'):
                self.planning_tab.plan_created.connect(self._on_plan_selected_for_evaluation)
            
            if hasattr(self, 'planning_tab') and hasattr(self.planning_tab, 'plan_updated'):
                self.planning_tab.plan_updated.connect(self._on_plan_updated_for_evaluation)
            
            if hasattr(self, 'dose_tab') and hasattr(self.dose_tab, 'dose_calculated'):
                self.dose_tab.dose_calculated.connect(self._on_plan_updated_for_evaluation)
            
            # Connect to current plan changes
            self.plan_changed.connect(self._on_plan_selected_for_evaluation)
            
            # Set icon for the tab
            evaluation_icon = QIcon(get_icon_path("evaluation"))
            tab_index = self.workflow_tabs.indexOf(self.plan_evaluation_tab)
            if tab_index >= 0:
                self.workflow_tabs.setTabIcon(tab_index, evaluation_icon)
            
            logger.info("Plan Evaluation tab initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Plan Evaluation tab: {e}")
            logger.debug("Exception details:", exc_info=True)
            
            # Create placeholder if needed
            from PyQt5.QtWidgets import QLabel, QVBoxLayout
            self.plan_evaluation_tab = QWidget()
            layout = QVBoxLayout(self.plan_evaluation_tab)
            layout.addWidget(QLabel("Plan Evaluation functionality is not available."))
            
            self.workflow_tabs.addTab(self.plan_evaluation_tab, "Plan Evaluation")
    
    def _on_plan_selected_for_evaluation(self, plan):
        """
        Handle when a plan is selected, specifically for plan evaluation.
        
        Parameters
        ----------
        plan : Plan or dict
            Plan object or dictionary with plan data
        """
        try:
            # Check if we have a plan evaluation tab
            if not hasattr(self, 'plan_evaluation_tab'):
                logger.warning("Plan Evaluation tab not available")
                return
                
            # Check if the plan evaluation tab has the required method
            if not hasattr(self.plan_evaluation_tab, 'set_plan'):
                logger.warning("Plan Evaluation tab does not have set_plan method")
                return
                
            logger.info(f"Setting plan for evaluation: {getattr(plan, 'id', plan)}")
            
            # Set the plan in the evaluation tab
            self.plan_evaluation_tab.set_plan(plan)
            
            # If we're not already on the plan evaluation tab, show a notification
            if self.workflow_tabs.currentWidget() != self.plan_evaluation_tab:
                logger.info("Plan ready for evaluation. Switch to Plan Evaluation tab to view results.")
                # Could add a notification here if needed
                
        except Exception as e:
            logger.error(f"Error setting plan for evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_plan_updated_for_evaluation(self, plan):
        """
        Handle when a plan is updated in the planning tab.
        
        Parameters
        ----------
        plan : Plan or dict
            Updated plan object or dictionary
        """
        if hasattr(self, 'plan_evaluation_tab') and hasattr(self.plan_evaluation_tab, 'set_plan'):
            try:
                logger.info(f"Refreshing Plan Evaluation for updated plan: {getattr(plan, 'id', plan)}")
                self.plan_evaluation_tab.set_plan(plan, self.current_patient)
            except Exception as e:
                logger.error(f"Error refreshing Plan Evaluation: {e}")

    def _on_plan_selected(self, plan):
        """
        Handle when a plan is selected from any tab.
        
        Parameters
        ----------
        plan : Plan or dict
            Plan object or dictionary with plan data
        """
        try:
            # Update current plan
            self.current_plan = plan
            plan_id = getattr(plan, 'id', plan)
            logger.info(f"Plan selected: {plan_id}")
            
            # Update all tabs that need plan data
            if hasattr(self, 'planning_tab') and hasattr(self.planning_tab, 'set_plan'):
                self.planning_tab.set_plan(plan)
                
            if hasattr(self, 'dose_tab') and hasattr(self.dose_tab, 'set_plan'):
                self.dose_tab.set_plan(plan)
                
            # Call the specific plan evaluation handler
            self._on_plan_selected_for_evaluation(plan)
                
            # Also update any other tabs that depend on the plan
            if hasattr(self, 'treatment_tab') and hasattr(self.treatment_tab, 'set_plan'):
                self.treatment_tab.set_plan(plan)
                
            if hasattr(self, 'qa_tab') and hasattr(self.qa_tab, 'set_plan'):
                self.qa_tab.set_plan(plan)
                
            if hasattr(self, 'reporting_tab') and hasattr(self.reporting_tab, 'set_plan'):
                self.reporting_tab.set_plan(plan)
                
            # Update UI state
            self._update_ui_state()
            
        except Exception as e:
            logger.error(f"Error handling plan selection: {e}")
            import traceback
            traceback.print_exc()

    def get_tab_index(self, tab_name):
        """
        Get the index of a tab by its name or instance variable name.
        
        Parameters
        ----------
        tab_name : str
            Name of the tab, either the displayed name or the instance variable name
            (e.g., 'patient_tab', 'imaging_tab', etc.)
            
        Returns
        -------
        int
            Index of the tab in the workflow_tabs widget, or -1 if not found
        """
        # Check if this is an instance variable name
        tab_instance = getattr(self, tab_name, None) if isinstance(tab_name, str) else None
        
        if tab_instance is not None:
            # Find the tab by its widget reference
            for i in range(self.workflow_tabs.count()):
                if self.workflow_tabs.widget(i) == tab_instance:
                    return i
        
        # Check if this is a tab display name
        for i in range(self.workflow_tabs.count()):
            if self.workflow_tabs.tabText(i).lower() == str(tab_name).lower():
                return i
                
        # Not found
        logger.warning(f"Tab '{tab_name}' not found")
        return -1
        
    def switch_to_tab(self, tab_index):
        """
        Switch to the specified tab by index or name.
        
        Parameters
        ----------
        tab_index : int or str
            Index of the tab to switch to, or name of the tab
            
        Returns
        -------
        bool
            True if switched successfully, False otherwise
        """
        # Convert name to index if needed
        if isinstance(tab_index, str):
            tab_index = self.get_tab_index(tab_index)
            
        # Validate index
        if tab_index < 0 or tab_index >= self.workflow_tabs.count():
            logger.warning(f"Invalid tab index: {tab_index}")
            return False
            
        # Switch to the tab
        self.workflow_tabs.setCurrentIndex(tab_index)
        logger.info(f"Switched to tab index {tab_index}: {self.workflow_tabs.tabText(tab_index)}")
        return True
        
    def load_test_data(self):
        """
        Load test data for development and demonstration purposes.
        
        This creates a test patient with a simple plan for evaluation.
        """
        logger.info("Loading test data for development")
        
        try:
            # Import necessary modules
            from quangtps.core.patient import Patient
            from quangtps.planning.plan import Plan
            from quangtps.database.patient_db import PatientDB
            
            # Create a test patient
            patient_id = "TEST_PATIENT_001"
            test_patient = {
                "id": patient_id,
                "name": "Test Patient",
                "gender": "Other",
                "birth_date": "1980-01-01",
                "medical_record_num": "TEST-MRN-001",
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
            }
            
            # Update or create the patient in database
            try:
                patient_db = PatientDB()
                if patient_db.get_patient(patient_id):
                    patient_db.update_patient(patient_id, test_patient)
                else:
                    patient_db.add_patient(test_patient)
                logger.info(f"Test patient saved to database: {patient_id}")
            except Exception as e:
                logger.warning(f"Could not save test patient to database: {e}")
            
            # Create a patient object
            patient = Patient(**test_patient)
            self.current_patient = patient
            
            # Create a test plan
            plan_id = "TEST_PLAN_001"
            test_plan = Plan(
                id=plan_id,
                name="Test Plan",
                patient_id=patient_id,
                prescription_dose=70.0,  # Gy
                description="Test plan for development",
                creation_date=datetime.now().strftime("%Y-%m-%d"),
            )
            
            # Set the current plan
            self.current_plan = test_plan
            
            # Update UI state
            self._update_ui_state()
            
            # Switch to plan evaluation tab if available
            plan_eval_tab_index = self.get_tab_index("plan_evaluation_tab")
            if plan_eval_tab_index >= 0:
                self.switch_to_tab(plan_eval_tab_index)
                
            logger.info("Test data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            logger.debug("Exception details:", exc_info=True)
            return False

def main():
    """Hàm chạy chính của ứng dụng."""
    try:
        app = QApplication(sys.argv)
        
        # Thiết lập stylesheet
        try:
            style_file = os.path.join(os.path.dirname(__file__), "styles", "main_style.qss")
            if os.path.exists(style_file):
                with open(style_file, "r") as f:
                    app.setStyleSheet(f.read())
        except Exception as e:
            logger.warning("Không thể đọc stylesheet: %s", str(e))
        
        # Tạo và chạy cửa sổ chính
        window = MainWindow()
        window.run()
        
        return app.exec_()
    except Exception as e:
        import traceback
        error_text = traceback.format_exc()
        print("Lỗi khởi động ứng dụng:", error_text)
        
        # Hiển thị hộp thoại lỗi nếu có thể
        try:
            # QMessageBox is already imported through PyQt5.QtWidgets 
            # at the top of the file
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            QMessageBox.critical(None, "Lỗi khởi động", f"Không thể khởi động ứng dụng:\n\n{str(e)}\n\n{error_text}")
        except:
            pass
            
        return 1

if __name__ == "__main__":
    sys.exit(main())
