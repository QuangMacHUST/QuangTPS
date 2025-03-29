#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plan Evaluation Tab Validator
Checks if the Plan Evaluation tab is properly implemented and follows Eclipse standards.
"""

import os
import sys
import logging
import time
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PlanEvaluation-Validator")

def check_plan_evaluation_tab():
    """Check if the Plan Evaluation tab is implemented according to Eclipse standards"""
    try:
        # Import the necessary modules
        logger.info("Importing PlanEvaluationTab...")
        from quangtps.ui.plan_evaluation import PlanEvaluationTab
        logger.info("Successfully imported PlanEvaluationTab")
        
        # Create a QApplication
        app = QApplication(sys.argv)
        
        # Create a parent widget
        parent = QWidget()
        
        # Create the PlanEvaluationTab
        logger.info("Creating PlanEvaluationTab...")
        tab = PlanEvaluationTab(parent)
        logger.info("Successfully created PlanEvaluationTab")
        
        # Check if the tab has the necessary components
        necessary_components = [
            'dvh_canvas',
            'structure_list',
            'metrics_table',
            'indices_table',
            'show_metrics_checkbox',
            'status_label'
        ]
        
        for component in necessary_components:
            if hasattr(tab, component):
                logger.info(f"✓ Component exists: {component}")
            else:
                logger.error(f"✗ Missing component: {component}")
        
        # Check if the tab has the necessary methods
        necessary_methods = [
            'set_plan',
            'evaluate_plan',
            'set_dvh_data',
            'set_prescription',
            '_update_dvh_plot',
            '_update_metrics_tables',
            '_update_structure_list',
            '_show_all_structures',
            '_hide_all_structures'
        ]
        
        for method in necessary_methods:
            if hasattr(tab, method) and callable(getattr(tab, method)):
                logger.info(f"✓ Method exists: {method}")
            else:
                logger.error(f"✗ Missing method: {method}")
        
        # Create test data
        logger.info("Creating test DVH data...")
        import numpy as np
        
        dvh_data = {}
        structures = ["PTV", "Spinal Cord", "Heart", "Lung_Left", "Lung_Right", "Esophagus"]
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
        
        # Create prescription
        prescription = {
            "total_dose": 70.0,  # Gy
            "fractions": 35,
            "prescription_name": "Test Prescription - Prostate"
        }
        
        # Test set_dvh_data method
        logger.info("Testing set_dvh_data method...")
        try:
            tab.set_dvh_data(dvh_data)
            logger.info("✓ set_dvh_data method works")
        except Exception as e:
            logger.error(f"✗ set_dvh_data method failed: {e}")
        
        # Test set_prescription method
        logger.info("Testing set_prescription method...")
        try:
            tab.set_prescription(prescription)
            logger.info("✓ set_prescription method works")
        except Exception as e:
            logger.error(f"✗ set_prescription method failed: {e}")
        
        # Check if status label was updated
        if hasattr(tab, 'status_label') and tab.status_label.text() != "Ready":
            logger.info(f"✓ Status label was updated: '{tab.status_label.text()}'")
        else:
            logger.error("✗ Status label not updated")
        
        # Check if structure list was populated
        if hasattr(tab, 'structure_list') and tab.structure_list.count() == len(structures):
            logger.info(f"✓ Structure list populated with {tab.structure_list.count()} items")
        else:
            logger.error(f"✗ Structure list not populated correctly: {tab.structure_list.count() if hasattr(tab, 'structure_list') else 'N/A'} items")
        
        # Show the tab for visual inspection (if needed)
        # tab.show()
        # app.exec_()
        
        logger.info("All checks completed")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    check_plan_evaluation_tab() 