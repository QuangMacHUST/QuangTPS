#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS - A Radiation Treatment Planning System
===============================================

This file is the main entry point for the QuangTPS application. 
It initializes the application, sets up the main window, and handles the application lifecycle.
"""

import os
import sys
import logging
import argparse
import platform
from PyQt5.QtWidgets import QApplication, QSplashScreen, QStyleFactory
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize, QSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quangtps.log', mode='w')
    ]
)

logger = logging.getLogger("QuangTPS")

def setup_environment():
    """
    Configure the application environment, paths, and settings.
    """
    # Add the current directory to Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Set application name and organization
    QApplication.setApplicationName("QuangTPS")
    QApplication.setOrganizationName("QuangRayStation")
    QApplication.setOrganizationDomain("quangraystation.com")
    
    # Set high DPI scaling
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Platform-specific setup
    if platform.system() == 'Windows':
        import ctypes
        # Set application ID for Windows taskbar
        myappid = 'quangraystation.quangtps.app.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    # Create data directories if they don't exist
    data_paths = [
        'data/patients',
        'data/images',
        'data/structures',
        'data/plans',
        'data/dose',
        'data/templates',
        'data/protocols',
        'data/reports',
        'data/logs',
        'data/machine_data',
        'data/beam_data',
        'data/models'
    ]
    
    for path in data_paths:
        os.makedirs(path, exist_ok=True)

def initialize_services():
    """
    Initialize essential services for the application.
    """
    from quangtps.core.services import ServiceRegistry
    
    # Import service implementations
    from quangtps.treatment.machine.machine_library import MachineLibrary
    from quangtps.database.patient_db import PatientDatabase
    from quangtps.dose.dose_calculator import DoseCalculator
    from quangtps.optimization.optimizer import Optimizer
    from quangtps.segmentation.model_downloader import ModelDownloader
    
    # Register services
    registry = ServiceRegistry()
    
    # Machine service
    machine_service = MachineLibrary()
    machine_service.load_machines_from_directory("data/machine_data")
    registry.register_service("MachineService", machine_service)
    
    # Patient database
    patient_db = PatientDatabase("data/database/quangtps.db")
    registry.register_service("PatientDatabase", patient_db)
    
    # Dose calculator
    dose_calc = DoseCalculator()
    registry.register_service("DoseCalculator", dose_calc)
    
    # Optimizer
    optimizer = Optimizer()
    registry.register_service("Optimizer", optimizer)
    
    # Model downloader
    model_downloader = ModelDownloader("data/models")
    registry.register_service("ModelDownloader", model_downloader)
    
    return registry

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='QuangTPS - A Radiation Treatment Planning System')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    parser.add_argument('--reset-settings', action='store_true', help='Reset all application settings')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--patient', type=str, help='Load a specific patient')
    parser.add_argument('--dicom', type=str, help='Import DICOM data from directory')
    
    return parser.parse_args()

def main():
    """
    Main application entry point.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Running in debug mode")
    
    # Setup the application environment
    setup_environment()
    
    # Reset settings if requested
    if args.reset_settings:
        logger.info("Resetting application settings")
        settings = QSettings()
        settings.clear()
    
    # Create application
    app = QApplication(sys.argv)
    
    # Show splash screen
    splash_pixmap = QPixmap("quangtps/resources/images/splash.png")
    if splash_pixmap.isNull():
        # Fallback to a text-based splash if image not found
        splash_pixmap = QPixmap(600, 400)
        splash_pixmap.fill(Qt.white)
    
    splash = QSplashScreen(splash_pixmap)
    splash.show()
    
    # Process events to make splash screen visible
    app.processEvents()
    
    # Display splash message
    splash.showMessage("Starting QuangTPS...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    
    # Initialize services
    splash.showMessage("Initializing services...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()
    
    try:
        service_registry = initialize_services()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.exception(f"Error initializing services: {str(e)}")
        splash.close()
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Error", f"Failed to initialize services:\n{str(e)}")
        return 1
    
    # Import the MainWindow class here to avoid circular imports
    try:
        from quangtps.ui.main_window import MainWindow
        
        # Set application style
        if 'Fusion' in QStyleFactory.keys():
            app.setStyle('Fusion')
        
        # Set application icon
        app_icon = QIcon("quangtps/resources/icons/logo.png")
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
        
        # Update splash screen
        splash.showMessage("Initializing main window...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        app.processEvents()
        
        # Initialize main window
        logger.info("Creating main window")
        main_window = MainWindow()
        
        # Update splash screen
        splash.showMessage("Loading data...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        app.processEvents()
        
        # Configure main window
        main_window.apply_styling()
        
        # Import DICOM data if specified
        if args.dicom:
            splash.showMessage(f"Importing DICOM data from {args.dicom}...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
            app.processEvents()
            main_window.import_dicom_data(args.dicom)
        
        # Load patient if specified
        if args.patient:
            logger.info(f"Loading patient: {args.patient}")
            splash.showMessage(f"Loading patient {args.patient}...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
            app.processEvents()
            main_window.load_patient(args.patient)
        
        # Update splash screen
        splash.showMessage("Starting application...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        app.processEvents()
        
        # Show main window
        main_window.show()
        
        # Close splash screen after a delay
        QTimer.singleShot(1000, splash.close)
        
        # Run the application
        return app.exec_()
        
    except Exception as e:
        logger.exception("Error starting QuangTPS")
        splash.close()
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Error", f"Failed to start QuangTPS:\n{str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 