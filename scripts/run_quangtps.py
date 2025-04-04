#!/usr/bin/env python
"""
QuangTPS Runner Script
======================

This script launches the QuangTPS application with all Eclipse-like features.
It handles initialization, dependency checking, and proper startup of the application.
"""

import os
import sys
import logging
import argparse
import importlib
from pathlib import Path

# Add the parent directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

def check_dependencies():
    """Check that all required dependencies are installed."""
    required_packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "pydicom",
        "PyQt5",
        "pillow", 
        "SimpleITK",
        "vtk",
        "pyvista",
        "weasyprint",
        "reportlab"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("The following packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        
        # Special note for WeasyPrint on Windows
        if "weasyprint" in missing_packages and sys.platform == "win32":
            print("\nNOTE: WeasyPrint on Windows requires additional GTK+ libraries.")
            print("See https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows")
        
        return False
    
    return True

def setup_logging(log_file=None, level=logging.INFO):
    """Set up logging configuration."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def init_services():
    """Initialize the core services required by QuangTPS."""
    try:
        from quangtps.core.services import ServiceRegistry
        from quangtps.database.patient_db import PatientDB
        
        # Initialize services
        ServiceRegistry.register_service("PatientDB", PatientDB())
        
        return True
    except Exception as e:
        logging.error(f"Error initializing services: {e}")
        return False

def run_application(debug=False, config=None):
    """Run the QuangTPS application."""
    try:
        from PyQt5.QtWidgets import QApplication
        from quangtps.ui.main_window import MainWindow
        
        # Create application
        app = QApplication(sys.argv)
        
        # Set application name and organization
        app.setApplicationName("QuangTPS")
        app.setOrganizationName("QuangTPS")
        
        # Create main window
        main_window = MainWindow(config)
        main_window.show()
        
        # Run application
        return app.exec_()
    except Exception as e:
        logging.error(f"Error running application: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return 1

def main():
    """Main entry point for the QuangTPS application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the QuangTPS radiation therapy treatment planning system.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(args.log_file, log_level)
    
    # Log startup information
    logger.info("Starting QuangTPS")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("Missing dependencies. Please install required packages.")
        return 1
    
    # Initialize directories
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Load configuration
    config = {}
    if args.config:
        import json
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return 1
    
    # Initialize services
    logger.info("Initializing services...")
    if not init_services():
        logger.error("Failed to initialize services")
        return 1
    
    # Run application
    logger.info("Running QuangTPS application...")
    return run_application(args.debug, config)

if __name__ == "__main__":
    sys.exit(main()) 