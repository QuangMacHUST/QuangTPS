#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstration of Plan Quality Evaluation in QuangTPS.

This script demonstrates the use of the plan quality evaluation features,
including clinical protocols, goal evaluation, and the integration with
the treatment planning system.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import QuangTPS modules
from quangtps.imaging.image import Image
from quangtps.structures.structure import Structure
from quangtps.structures.structure_set import StructureSet
from quangtps.beams.beam import Beam, BeamSet
from quangtps.dose.dose_calculator import DoseCalculator
from quangtps.evaluation.plan_evaluation import PlanEvaluation
from quangtps.evaluation.plan_quality import PlanQualityEvaluator
from quangtps.evaluation.clinical_protocols import ClinicalProtocolManager, select_protocol_dialog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_sample_image():
    """Create a sample image for testing."""
    logger.info("Creating sample image")
    image_data = np.ones((100, 100, 50), dtype=np.float32)
    image = Image()
    image.data = image_data
    image.spacing = (2.0, 2.0, 3.0)  # mm
    return image

def create_sample_structures(image):
    """Create sample structures for testing."""
    logger.info("Creating sample structures")
    structure_set = StructureSet()
    
    # Create PTV
    ptv = Structure()
    ptv.id = "struct_1"
    ptv.name = "PTV"
    ptv.type = "PTV"
    ptv.mask = np.zeros_like(image.data, dtype=bool)
    ptv.mask[40:60, 40:60, 20:30] = True
    
    # Create OAR1 - Spinal Cord
    oar1 = Structure()
    oar1.id = "struct_2"
    oar1.name = "Spinal Cord"
    oar1.type = "OAR"
    oar1.mask = np.zeros_like(image.data, dtype=bool)
    oar1.mask[55:65, 40:50, 20:30] = True
    
    # Create OAR2 - Heart
    oar2 = Structure()
    oar2.id = "struct_3"
    oar2.name = "Heart"
    oar2.type = "OAR"
    oar2.mask = np.zeros_like(image.data, dtype=bool)
    oar2.mask[45:60, 60:75, 20:35] = True
    
    # Create OAR3 - Lung
    oar3 = Structure()
    oar3.id = "struct_4"
    oar3.name = "Lung"
    oar3.type = "OAR"
    oar3.mask = np.zeros_like(image.data, dtype=bool)
    oar3.mask[25:45, 60:85, 15:35] = True
    
    # Add structures to structure set
    structure_set.add_structure(ptv)
    structure_set.add_structure(oar1)
    structure_set.add_structure(oar2)
    structure_set.add_structure(oar3)
    
    return structure_set

def create_sample_beams():
    """Create sample beams for testing."""
    logger.info("Creating sample beams")
    beam_set = BeamSet()
    beam_set.id = "beamset_1"
    beam_set.name = "Sample Plan"
    beam_set.prescription = 70.0  # Gy
    beam_set.target_structure_id = "struct_1"  # PTV
    
    # Create beams
    beam1 = Beam()
    beam1.id = "beam_1"
    beam1.name = "AP"
    beam1.energy = "6MV"
    beam1.gantry_angle = 0.0
    beam1.couch_angle = 0.0
    beam1.collimator_angle = 0.0
    beam1.field_size = (40.0, 40.0)  # mm
    beam1.isocenter = (100.0, 100.0, 75.0)  # mm
    beam1.weight = 1.0
    
    beam2 = Beam()
    beam2.id = "beam_2"
    beam2.name = "LPO"
    beam2.energy = "6MV"
    beam2.gantry_angle = 120.0
    beam2.couch_angle = 0.0
    beam2.collimator_angle = 0.0
    beam2.field_size = (40.0, 40.0)  # mm
    beam2.isocenter = (100.0, 100.0, 75.0)  # mm
    beam2.weight = 1.0
    
    beam3 = Beam()
    beam3.id = "beam_3"
    beam3.name = "RPO"
    beam3.energy = "6MV"
    beam3.gantry_angle = 240.0
    beam3.couch_angle = 0.0
    beam3.collimator_angle = 0.0
    beam3.field_size = (40.0, 40.0)  # mm
    beam3.isocenter = (100.0, 100.0, 75.0)  # mm
    beam3.weight = 1.0
    
    # Add beams to beam set
    beam_set.add_beam(beam1)
    beam_set.add_beam(beam2)
    beam_set.add_beam(beam3)
    
    return beam_set

def create_dose_calculation(image, structure_set, beam_set):
    """Create a dose calculation for testing."""
    logger.info("Creating dose calculation")
    calculator = DoseCalculator()
    calculator.set_image(image)
    calculator.set_structure_set(structure_set)
    calculator.set_beam_set(beam_set)
    
    # Set calculation grid resolution (5mm)
    calculator.set_calculation_grid_resolution((5.0, 5.0, 5.0))
    
    # Calculate dose
    dose_grid = calculator.calculate_dose()
    
    if dose_grid is not None:
        logger.info(f"Dose calculation successful. Grid shape: {dose_grid.shape}")
    else:
        logger.error("Dose calculation failed")
        
    return calculator

def list_available_protocols():
    """List available clinical protocols."""
    logger.info("Listing available clinical protocols")
    manager = ClinicalProtocolManager()
    protocol_names = manager.get_protocol_names()
    
    print("\nAvailable Clinical Protocols:")
    print("-----------------------------")
    for name in protocol_names:
        protocol = manager.get_protocol(name)
        description = protocol.get("description", "No description")
        goal_count = len(protocol.get("clinical_goals", []))
        print(f"- {name}: {description} ({goal_count} goals)")
    
    return protocol_names

def evaluate_with_protocol(plan_evaluation, protocol_name):
    """Evaluate a plan using a specific protocol."""
    logger.info(f"Evaluating plan using protocol: {protocol_name}")
    manager = ClinicalProtocolManager()
    protocol = manager.get_protocol(protocol_name)
    
    if not protocol:
        logger.error(f"Protocol not found: {protocol_name}")
        return None
    
    # Create evaluator
    evaluator = PlanQualityEvaluator()
    evaluator.set_plan_evaluation(plan_evaluation)
    evaluator.load_clinical_protocol(protocol)
    
    # Evaluate plan quality
    results = evaluator.evaluate_plan_quality()
    
    if results:
        # Print summary
        summary = evaluator.generate_evaluation_summary()
        print("\nPlan Evaluation Results:")
        print(summary)
        
        # Print details of each goal
        print("\nDetailed Goal Results:")
        print("-----------------------")
        for goal in results.get("goals_details", []):
            structure = goal.get("matched_structure", goal.get("structure_name", "Unknown"))
            goal_type = goal.get("goal_type", "Unknown")
            target = goal.get("target_value", 0.0)
            result = goal.get("result_value", 0.0)
            status = "PASS" if goal.get("achieved", False) else ("ACCEPTABLE" if goal.get("partially_achieved", False) else "FAIL")
            
            print(f"{structure} - {goal_type}: {result:.2f} vs Target {target:.2f} - {status}")
    
    return results

def demo_gui_protocol_selection():
    """Demonstrate the GUI protocol selection dialog."""
    logger.info("Starting GUI protocol selection demo")
    app = QApplication([])
    
    try:
        # Show protocol selection dialog
        protocol = select_protocol_dialog()
        
        if protocol:
            logger.info(f"Selected protocol: {protocol.get('name')}")
            print(f"\nSelected Protocol: {protocol.get('name')}")
            print(f"Description: {protocol.get('description', 'No description')}")
            print(f"Goals: {len(protocol.get('clinical_goals', []))}")
        else:
            logger.info("No protocol selected")
            print("\nNo protocol selected")
    except Exception as e:
        logger.error(f"Error in GUI protocol selection: {e}")
        print(f"\nError in GUI protocol selection: {e}")

def main():
    """Main function to run the demonstration."""
    print("\n=== QuangTPS Plan Quality Evaluation Demo ===\n")
    
    # Create sample data
    image = create_sample_image()
    structure_set = create_sample_structures(image)
    beam_set = create_sample_beams()
    
    # Create dose calculation
    dose_calculator = create_dose_calculation(image, structure_set, beam_set)
    
    # Create plan evaluation
    plan_evaluation = PlanEvaluation()
    plan_evaluation.set_dose_calculator(dose_calculator)
    
    # List available protocols
    protocol_names = list_available_protocols()
    
    if not protocol_names:
        logger.error("No protocols available")
        return
    
    # Evaluate with a sample protocol
    sample_protocol = protocol_names[0]  # Use first protocol
    print(f"\nEvaluating plan with protocol: {sample_protocol}")
    results = evaluate_with_protocol(plan_evaluation, sample_protocol)
    
    # Try another protocol if available
    if len(protocol_names) > 1:
        another_protocol = protocol_names[1]
        print(f"\nEvaluating plan with another protocol: {another_protocol}")
        results = evaluate_with_protocol(plan_evaluation, another_protocol)
    
    # Demo GUI protocol selection
    try:
        print("\nDemonstrating GUI protocol selection...")
        demo_gui_protocol_selection()
    except Exception as e:
        logger.error(f"Error in GUI demo: {e}")
        print(f"Error in GUI demo: {e}")
    
    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    main() 