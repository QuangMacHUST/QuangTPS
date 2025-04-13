#!/usr/bin/env python
"""
QuangTPS: A Modern Radiotherapy Treatment Planning System

Main entry point for the QuangTPS application.
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QAction, QFileDialog,
    QDockWidget, QMessageBox, QTabWidget, QStatusBar, QVBoxLayout, QWidget
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon

# Import QuangTPS modules
from quangtps.core.logging import setup_logging, get_logger
from quangtps.ui.mpr_structure_integration import MPRStructureIntegration
from quangtps.ui.main_window import MainWindow

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QuangTPS - A Modern Radiotherapy Treatment Planning System")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--version', action='store_true', help='Show version information')
    parser.add_argument('--patient', type=str, help='Load patient data folder on startup')
    return parser.parse_args()

def show_version():
    """Display version information."""
    from quangtps import __version__
    print(f"QuangTPS version {__version__}")
    print("A Modern Radiotherapy Treatment Planning System")
    print("Copyright (c) 2023")

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Handle version request
    if args.version:
        show_version()
        return 0
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("QuangTPS")
    app.setOrganizationName("QuangTPS")
    app.setOrganizationDomain("quangtps.org")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create main window
    main_window = MainWindow()
    main_window.setWindowTitle("QuangTPS - Treatment Planning System")
    main_window.resize(1600, 900)
    
    # Load patient data if specified
    if args.patient:
        patient_path = Path(args.patient)
        if patient_path.exists() and patient_path.is_dir():
            main_window.load_patient(patient_path)
        else:
            logger.error(f"Patient folder not found: {args.patient}")
            QMessageBox.warning(
                main_window,
                "Error",
                f"Patient folder not found: {args.patient}"
            )
    
    # Show the main window
    main_window.show()
    
    # Start the event loop
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main()) 