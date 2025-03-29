#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuangTPS-PlanEval-Demo")

def main():
    """Launch a simplified demo of the Plan Evaluation tab with synthetic test data."""
    try:
        # Only import PyQt5 components needed for this demo
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt

        # Try to import only the necessary components for plan evaluation
        try:
            from quangtps.ui.main_window import MainWindow
            from quangtps.ui.plan_evaluation import PlanEvaluationTab
            from quangtps.data.patient import Patient
            from quangtps.data.plan import Plan
            logger.info("Successfully imported QuangTPS modules")
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.info("Attempting to create synthetic test data instead...")
            # Continue with synthetic data approach

        # Create QApplication instance
        app = QApplication(sys.argv)
        
        # Create synthetic data for testing instead of loading from problematic modules
        def create_synthetic_test_data():
            """Create synthetic DVH data for demonstration purposes."""
            logger.info("Creating synthetic DVH data for testing")
            
            # Create synthetic DVH data for common structures
            structures = ["PTV", "Spinal Cord", "Heart", "Lung_Left", "Lung_Right", "Esophagus"]
            
            # Dictionary to hold DVH data
            dvh_data = {}
            
            # Generate random DVH curves for each structure
            dose_bins = np.linspace(0, 80, 100)  # 0 to 80 Gy
            
            for structure in structures:
                if structure == "PTV":
                    # Create typical PTV curve (high dose with steep falloff)
                    volume_pct = 100 * np.ones_like(dose_bins)
                    # Sharp falloff around 60 Gy
                    falloff_idx = np.where(dose_bins >= 60)[0][0]
                    volume_pct[falloff_idx:] = 100 * np.exp(-(dose_bins[falloff_idx:] - dose_bins[falloff_idx]) / 3)
                elif structure.startswith("Lung"):
                    # Typical lung curve (low dose to large volume)
                    volume_pct = 100 * np.exp(-dose_bins / 15)
                elif structure == "Spinal Cord":
                    # Critical structure (low dose tolerance)
                    volume_pct = 100 * np.exp(-dose_bins / 10)
                else:
                    # Generic OAR curve
                    volume_pct = 100 * np.exp(-dose_bins / 20)
                
                # Store data
                dvh_data[structure] = {
                    "dose_bins": dose_bins,
                    "volume_pct": volume_pct,
                    "structure_type": "PTV" if structure == "PTV" else "OAR"
                }
            
            return dvh_data
        
        # Create the main window
        window = MainWindow()
        
        # Find the plan evaluation tab
        plan_eval_tab_index = window.get_tab_index('plan_evaluation_tab')
        if plan_eval_tab_index is not None:
            logger.info(f"Found Plan Evaluation tab at index {plan_eval_tab_index}")
            
            # Get the plan evaluation tab
            plan_eval_tab = window.tabWidget.widget(plan_eval_tab_index)
            
            # Load synthetic data
            dvh_data = create_synthetic_test_data()
            logger.info(f"Created synthetic DVH data for {len(dvh_data)} structures")
            
            # Create a simple prescription
            prescription = {
                "total_dose": 60.0,  # Gy
                "fractions": 30,
                "prescription_name": "Demo Prescription"
            }
            
            # Set data to plan evaluation tab
            plan_eval_tab.set_dvh_data(dvh_data)
            plan_eval_tab.set_prescription(prescription)
            logger.info("Loaded synthetic data into Plan Evaluation tab")
            
            # Switch to the plan evaluation tab
            window.switch_to_tab(plan_eval_tab_index)
            logger.info("Switched to Plan Evaluation tab")
        else:
            logger.warning("Could not find Plan Evaluation tab")
        
        # Show the window
        window.show()
        
        # Run the application
        return app.exec_()
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 