#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QuangTPS Demo Script - Plan Evaluation

This script launches QuangTPS with a focus on the Plan Evaluation tab.
It loads a test patient and plan to demonstrate the plan evaluation capabilities.
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run QuangTPS with Plan Evaluation focus"""
    try:
        # Add the current directory to the path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        # Try to import QuangTPS modules
        try:
            from quangtps.ui.main_window import MainWindow
            from PyQt5.QtWidgets import QApplication
        except ImportError as e:
            logger.error(f"Could not import QuangTPS modules: {e}")
            logger.error("Please make sure QuangTPS is properly installed")
            return 1
            
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Create main window
        window = MainWindow()
        
        # Load test data
        logger.info("Loading test data...")
        window.load_test_data()
        
        # If the system has the plan evaluation tab, switch to it
        logger.info("Switching to Plan Evaluation tab...")
        tab_index = window.get_tab_index('plan_evaluation_tab')
        if tab_index >= 0:
            window.switch_to_tab(tab_index)
        else:
            logger.warning("Plan Evaluation tab not available")
        
        # Show the window and start the application
        window.show()
        return app.exec_()
        
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 