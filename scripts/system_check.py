#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS System Check Utility

This script performs a comprehensive check of the QuangTPS system to verify
that all components are working properly, including dose calculation algorithms,
treatment techniques, and visualization.

Usage:
    python system_check.py [--verbose]
"""

import os
import sys
import argparse
import logging
import importlib
import numpy as np
import time
from pathlib import Path

# Add parent directory to path so we can import QuangTPS modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuangTPS-SystemCheck")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QuangTPS System Check Utility")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def check_module_imports():
    """
    Check that all required modules can be imported.
    
    Returns
    -------
    bool
        True if all modules can be imported, False otherwise
    """
    required_modules = [
        # Core Python modules
        "numpy", "scipy", "matplotlib", "pandas",
        
        # UI modules
        "PyQt5",
        
        # Medical imaging modules
        "pydicom",
        
        # QuangTPS core modules
        "quangtps.core.types",
        "quangtps.core.exceptions",
        
        # QuangTPS imaging modules
        "quangtps.imaging.image",
        "quangtps.imaging.dicom_series",
        
        # QuangTPS planning modules
        "quangtps.planning.plan",
        "quangtps.planning.beam",
        
        # QuangTPS dose calculation modules
        "quangtps.dose.algorithms",
        "quangtps.dose.algorithms.pencil_beam",
        "quangtps.dose.algorithms.collapsed_cone",
        "quangtps.dose.algorithms.monte_carlo",
        
        # QuangTPS treatment technique modules
        "quangtps.treatment.techniques.crt_manager",
        "quangtps.treatment.techniques.imrt",
        "quangtps.treatment.techniques.vmat",
    ]
    
    success = True
    
    for module_name in required_modules:
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name}")
        except ImportError as e:
            success = False
            logger.error(f"Failed to import {module_name}: {e}")
    
    return success

def check_dose_calculation_algorithms():
    """
    Check that dose calculation algorithms are working properly.
    
    Returns
    -------
    bool
        True if all algorithms are working properly, False otherwise
    """
    try:
        from quangtps.dose.algorithms import get_available_algorithms, get_algorithm_instance
        from quangtps.imaging.image import Image
        from quangtps.planning.beam import Beam
        from quangtps.dose.beam_data_processor import BeamModel
        
        # Check that we can get the list of available algorithms
        algorithms = get_available_algorithms()
        logger.info(f"Found {len(algorithms)} available dose calculation algorithms")
        
        if not algorithms:
            logger.error("No dose calculation algorithms found")
            return False
        
        # Create a simple test image
        test_image = Image()
        test_image.data = np.ones((10, 10, 10), dtype=np.float32)
        test_image.pixel_spacing = [0.5, 0.5, 0.5]
        
        # Create a simple test beam
        test_beam = Beam()
        test_beam.name = "Test Beam"
        test_beam.energy = 6
        test_beam.gantry_angle = 0.0
        test_beam.field_size = (10.0, 10.0)
        
        # Create a simple beam model
        beam_model = BeamModel()
        beam_model.name = "Test Model"
        beam_model.energy = 6
        
        # Test pencil beam algorithm
        try:
            pencil_beam = get_algorithm_instance("pencil_beam")
            pencil_beam.set_beam_model(beam_model)
            logger.info("Successfully instantiated Pencil Beam algorithm")
        except Exception as e:
            logger.error(f"Failed to instantiate Pencil Beam algorithm: {e}")
            return False
        
        # Test collapsed cone algorithm
        try:
            collapsed_cone = get_algorithm_instance("collapsed_cone")
            collapsed_cone.set_beam_model(beam_model)
            logger.info("Successfully instantiated Collapsed Cone algorithm")
        except Exception as e:
            logger.error(f"Failed to instantiate Collapsed Cone algorithm: {e}")
            return False
        
        # Test Monte Carlo algorithm
        try:
            monte_carlo = get_algorithm_instance("monte_carlo")
            monte_carlo.set_beam_model(beam_model)
            monte_carlo.set_parameters(num_histories=100, use_gpu=False)  # Use minimal settings for test
            logger.info("Successfully instantiated Monte Carlo algorithm")
        except Exception as e:
            logger.error(f"Failed to instantiate Monte Carlo algorithm: {e}")
            return False
        
        logger.info("All dose calculation algorithms are available and can be instantiated")
        return True
        
    except Exception as e:
        logger.error(f"Error checking dose calculation algorithms: {e}")
        return False

def check_treatment_techniques():
    """
    Check that treatment technique modules are working properly.
    
    Returns
    -------
    bool
        True if all treatment techniques are working properly, False otherwise
    """
    try:
        # Check 3D CRT
        from quangtps.treatment.techniques.crt_manager import CRTManager
        crt_manager = CRTManager()
        logger.info("Successfully instantiated CRT Manager")
        
        # Create a beam from a template
        beam = crt_manager.create_beam_from_template("box_technique", 0)
        if beam is not None:
            logger.info("Successfully created beam from CRT template")
        else:
            logger.error("Failed to create beam from CRT template")
            return False
        
        # Check IMRT
        from quangtps.treatment.techniques.imrt import IMRT
        imrt = IMRT("Test IMRT Plan")
        logger.info("Successfully instantiated IMRT technique")
        
        # Check VMAT
        from quangtps.treatment.techniques.vmat import VMAT
        vmat = VMAT("Test VMAT Plan")
        logger.info("Successfully instantiated VMAT technique")
        
        # Add an arc to the VMAT plan
        arc_id = vmat.add_arc("Test Arc", 0.0, 359.9, "CW", "6X")
        if arc_id:
            logger.info("Successfully added arc to VMAT plan")
        else:
            logger.error("Failed to add arc to VMAT plan")
            return False
        
        logger.info("All treatment technique modules are working properly")
        return True
        
    except Exception as e:
        logger.error(f"Error checking treatment techniques: {e}")
        return False

def check_database():
    """
    Check that the database is working properly.
    
    Returns
    -------
    bool
        True if the database is working properly, False otherwise
    """
    try:
        from quangtps.database.patient_db import PatientDB
        import tempfile
        
        # Create a temporary database file
        with tempfile.NamedTemporaryFile(suffix='.db') as temp:
            db_path = temp.name
        
        # Initialize the database
        db = PatientDB(db_path)
        
        # Check that we can create tables
        db.create_tables()
        logger.info("Successfully created database tables")
        
        # Check that we can add a patient
        patient_id = db.add_patient("Test", "Patient", "1990-01-01", "M", "123456")
        if patient_id:
            logger.info("Successfully added patient to database")
        else:
            logger.error("Failed to add patient to database")
            return False
        
        # Check that we can retrieve the patient
        patient = db.get_patient_by_id(patient_id)
        if patient and patient["last_name"] == "Patient":
            logger.info("Successfully retrieved patient from database")
        else:
            logger.error("Failed to retrieve patient from database")
            return False
        
        # Clean up
        os.remove(db_path)
        
        logger.info("Database is working properly")
        return True
        
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False

def check_visualization():
    """
    Check that visualization modules are working properly.
    
    This test only checks that the modules can be imported and instantiated,
    not that they can actually display anything (which would require a GUI).
    
    Returns
    -------
    bool
        True if visualization modules can be instantiated, False otherwise
    """
    try:
        # Check beam visualization
        from quangtps.ui.beam_visualization import BeamVisualization
        logger.info("Successfully imported beam visualization module")
        
        # Check DVH visualization
        from quangtps.ui.dvh_visualization import DVHVisualization
        logger.info("Successfully imported DVH visualization module")
        
        # Check the CRT visualizer
        from quangtps.treatment.techniques.crt_visualizer import CRTVisualizer
        crt_vis = CRTVisualizer()
        logger.info("Successfully instantiated CRT visualizer")
        
        logger.info("All visualization modules can be imported")
        return True
        
    except Exception as e:
        logger.error(f"Error checking visualization modules: {e}")
        return False

def check_system():
    """
    Perform a comprehensive check of the QuangTPS system.
    
    Returns
    -------
    bool
        True if all checks pass, False otherwise
    """
    logger.info("Starting QuangTPS system check")
    
    checks = [
        ("Module imports", check_module_imports),
        ("Dose calculation algorithms", check_dose_calculation_algorithms),
        ("Treatment techniques", check_treatment_techniques),
        ("Database", check_database),
        ("Visualization", check_visualization)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        logger.info(f"Checking {name}...")
        start_time = time.time()
        result = check_func()
        elapsed_time = time.time() - start_time
        
        if result:
            logger.info(f"✓ {name} check passed ({elapsed_time:.2f}s)")
        else:
            logger.error(f"✗ {name} check failed ({elapsed_time:.2f}s)")
            all_passed = False
    
    if all_passed:
        logger.info("All system checks passed! QuangTPS is working properly.")
    else:
        logger.error("Some system checks failed. Please check the log for details.")
    
    return all_passed

def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = check_system()
    
    # Return success status as exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 