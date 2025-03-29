#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Main Application Entry Point

This script starts the QuangTPS treatment planning system. It handles command line
arguments, initializes logging, shows the splash screen, and launches the main
application window.

Usage:
    python run_quangtps.py [options]

Options:
    --help, -h            Show this help message and exit
    --verbose, -v         Display verbose (debug) information
    --no-splash           Don't show splash screen
    --console, -c         Run in console mode (no GUI)
    --demo, -d            Run with sample data
    --no-opengl           Disable OpenGL acceleration
    --config=FILE         Use alternate configuration file
"""

import os
import sys
import time
import argparse
import logging
import importlib
from pathlib import Path

# Add parent directory to path so we can import QuangTPS modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Global logger for use in all functions
logger = logging.getLogger("QuangTPS")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QuangTPS Treatment Planning System")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-splash", action="store_true", help="Don't show splash screen")
    parser.add_argument("--console", "-c", action="store_true", help="Run in console mode (no GUI)")
    parser.add_argument("--demo", "-d", action="store_true", help="Run with sample data")
    parser.add_argument("--no-opengl", action="store_true", help="Disable OpenGL acceleration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    return parser.parse_args()

def setup_logging(verbose=False):
    """Set up logging configuration."""
    global logger
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Log to file and console
    log_file = logs_dir / f"quangtps_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("QuangTPS")
    logger.info(f"Starting QuangTPS. Log file: {log_file}")
    
    return logger

def show_splash_screen():
    """Show the application splash screen."""
    try:
        from PyQt5.QtWidgets import QApplication, QSplashScreen
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import Qt, QTimer
        
        app = QApplication.instance() or QApplication(sys.argv)
        
        # Path to splash screen image
        splash_path = Path(__file__).resolve().parent.parent / "quangtps" / "ui" / "icons" / "splash.png"
        
        # Show splash screen
        splash_pix = QPixmap(str(splash_path))
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.setEnabled(False)
        
        # Add version info to splash screen
        splash.showMessage("Starting QuangTPS v1.0.0", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        splash.show()
        
        # Process events to show splash immediately
        app.processEvents()
        
        return app, splash
    
    except Exception as e:
        logger.error(f"Error showing splash screen: {e}")
        return QApplication.instance() or QApplication(sys.argv), None

def start_gui(app, splash=None, demo_mode=False, opengl_enabled=True, config_file=None):
    """Start the graphical user interface."""
    try:
        from quangtps.ui.main_window import MainWindow
        
        # Create config dictionary with all settings
        config = {
            'demo_mode': demo_mode,
            'opengl_enabled': opengl_enabled,
            'config_file': config_file
        }
        
        # Create and show the main window
        main_window = MainWindow(config=config)
        
        # Close splash screen if it exists
        if splash:
            splash.finish(main_window)
        
        main_window.show()
        
        # Start the application event loop
        return app.exec_()
    
    except Exception as e:
        logger.error(f"Error starting GUI: {e}")
        if splash:
            splash.close()
        return 1

def start_console(demo_mode=False, config_file=None):
    """Start the console mode (non-GUI)."""
    try:
        from quangtps.console.console_app import ConsoleApp
        
        console_app = ConsoleApp(demo_mode=demo_mode, config_file=config_file)
        return console_app.run()
    
    except Exception as e:
        logger.error(f"Error starting console mode: {e}")
        return 1

def check_dependencies():
    """Check that all required dependencies are installed."""
    required_modules = [
        "numpy", "scipy", "matplotlib", 
        "PyQt5", "pydicom", "pandas"
    ]
    
    missing_modules = []
    
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)
    
    if missing_modules:
        print("ERROR: The following required modules are missing:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install the missing modules using:")
        print(f"  pip install {' '.join(missing_modules)}")
        return False
    
    return True

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Set OpenGL environment variable if disabled
    if args.no_opengl:
        os.environ["QT_OPENGL"] = "software"
        logger.info("OpenGL acceleration disabled")
    
    # Console mode (no GUI)
    if args.console:
        logger.info("Starting QuangTPS in console mode")
        return start_console(demo_mode=args.demo, config_file=args.config)
    
    # GUI mode
    logger.info("Starting QuangTPS GUI")
    
    # Show splash screen if not disabled
    if args.no_splash:
        app = importlib.import_module("PyQt5.QtWidgets").QApplication(sys.argv)
        splash = None
    else:
        app, splash = show_splash_screen()
    
    # Start the GUI
    return start_gui(
        app, 
        splash, 
        demo_mode=args.demo,
        opengl_enabled=not args.no_opengl,
        config_file=args.config
    )

if __name__ == "__main__":
    sys.exit(main()) 