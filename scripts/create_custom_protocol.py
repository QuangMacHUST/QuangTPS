#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom Clinical Protocol Creation Script for QuangTPS

This script demonstrates how to create, save, and apply custom clinical protocols
for plan evaluation in QuangTPS.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('protocols')

def create_simple_protocol():
    """Create a simple clinical protocol as a dictionary."""
    protocol = {
        "name": "Simple Demo Protocol",
        "description": "A simple protocol for demonstration purposes",
        "clinical_goals": [
            {
                "structure_name": "PTV",
                "structure_id": "",
                "goal_type": "D95",
                "parameter": 95.0,
                "target_value": 95.0,
                "priority": "High",
                "variation_acceptable": 3.0
            },
            {
                "structure_name": "PTV",
                "structure_id": "",
                "goal_type": "V95",
                "parameter": 95.0,
                "target_value": 95.0,
                "priority": "High",
                "variation_acceptable": 2.0
            },
            {
                "structure_name": "Spinal Cord",
                "structure_id": "",
                "goal_type": "Max Dose",
                "parameter": 0.0,
                "target_value": 45.0,
                "priority": "Critical",
                "variation_acceptable": 1.0
            }
        ]
    }
    return protocol

def create_comprehensive_protocol():
    """Create a more comprehensive clinical protocol."""
    protocol = {
        "name": "Comprehensive Demo Protocol",
        "description": "A comprehensive protocol with many goals for various structures",
        "clinical_goals": [
            # PTV Goals
            {
                "structure_name": "PTV",
                "structure_id": "",
                "goal_type": "D95",
                "parameter": 95.0,
                "target_value": 95.0,
                "priority": "High",
                "variation_acceptable": 3.0
            },
            {
                "structure_name": "PTV",
                "structure_id": "",
                "goal_type": "D98",
                "parameter": 98.0,
                "target_value": 93.0,
                "priority": "High",
                "variation_acceptable": 3.0
            },
            {
                "structure_name": "PTV",
                "structure_id": "",
                "goal_type": "D2",
                "parameter": 2.0,
                "target_value": 107.0,
                "priority": "High",
                "variation_acceptable": 2.0
            },
            {
                "structure_name": "PTV",
                "structure_id": "",
                "goal_type": "V95",
                "parameter": 95.0,
                "target_value": 95.0,
                "priority": "High",
                "variation_acceptable": 2.0
            },
            
            # OAR Goals
            {
                "structure_name": "Spinal Cord",
                "structure_id": "",
                "goal_type": "Max Dose",
                "parameter": 0.0,
                "target_value": 45.0,
                "priority": "Critical",
                "variation_acceptable": 1.0
            },
            {
                "structure_name": "Heart",
                "structure_id": "",
                "goal_type": "Mean Dose",
                "parameter": 0.0,
                "target_value": 26.0,
                "priority": "Medium",
                "variation_acceptable": 4.0
            },
            {
                "structure_name": "Heart",
                "structure_id": "",
                "goal_type": "V25",
                "parameter": 25.0,
                "target_value": 10.0,
                "priority": "Medium",
                "variation_acceptable": 5.0
            },
            {
                "structure_name": "Lung",
                "structure_id": "",
                "goal_type": "V20",
                "parameter": 20.0,
                "target_value": 20.0,
                "priority": "Medium",
                "variation_acceptable": 5.0
            },
            {
                "structure_name": "Lung",
                "structure_id": "",
                "goal_type": "Mean Dose",
                "parameter": 0.0,
                "target_value": 15.0,
                "priority": "Medium",
                "variation_acceptable": 3.0
            }
        ]
    }
    return protocol

def create_site_specific_protocol(site="Prostate"):
    """Create a site-specific clinical protocol."""
    protocols = {
        "Prostate": {
            "name": "Custom Prostate Protocol",
            "description": "A protocol for prostate cancer treatment planning",
            "clinical_goals": [
                {
                    "structure_name": "PTV",
                    "structure_id": "",
                    "goal_type": "D95",
                    "parameter": 95.0,
                    "target_value": 95.0,
                    "priority": "High",
                    "variation_acceptable": 2.0
                },
                {
                    "structure_name": "PTV",
                    "structure_id": "",
                    "goal_type": "D2",
                    "parameter": 2.0,
                    "target_value": 107.0,
                    "priority": "High",
                    "variation_acceptable": 1.0
                },
                {
                    "structure_name": "Bladder",
                    "structure_id": "",
                    "goal_type": "V70",
                    "parameter": 70.0,
                    "target_value": 25.0,
                    "priority": "Medium",
                    "variation_acceptable": 5.0
                },
                {
                    "structure_name": "Rectum",
                    "structure_id": "",
                    "goal_type": "V65",
                    "parameter": 65.0,
                    "target_value": 17.0,
                    "priority": "Medium",
                    "variation_acceptable": 3.0
                },
                {
                    "structure_name": "Rectum",
                    "structure_id": "",
                    "goal_type": "V40",
                    "parameter": 40.0,
                    "target_value": 35.0,
                    "priority": "Medium",
                    "variation_acceptable": 5.0
                },
                {
                    "structure_name": "Femoral Heads",
                    "structure_id": "",
                    "goal_type": "Max Dose",
                    "parameter": 0.0,
                    "target_value": 50.0,
                    "priority": "Low",
                    "variation_acceptable": 5.0
                }
            ]
        },
        "SBRT Lung": {
            "name": "Custom SBRT Lung Protocol",
            "description": "A protocol for stereotactic body radiation therapy for lung cancer",
            "clinical_goals": [
                {
                    "structure_name": "PTV",
                    "structure_id": "",
                    "goal_type": "D95",
                    "parameter": 95.0,
                    "target_value": 100.0,
                    "priority": "High",
                    "variation_acceptable": 2.0
                },
                {
                    "structure_name": "PTV",
                    "structure_id": "",
                    "goal_type": "D99",
                    "parameter": 99.0,
                    "target_value": 90.0,
                    "priority": "High",
                    "variation_acceptable": 5.0
                },
                {
                    "structure_name": "Spinal Cord",
                    "structure_id": "",
                    "goal_type": "Max Dose",
                    "parameter": 0.0,
                    "target_value": 26.0,
                    "priority": "Critical",
                    "variation_acceptable": 2.0
                },
                {
                    "structure_name": "Heart",
                    "structure_id": "",
                    "goal_type": "Max Dose",
                    "parameter": 0.0,
                    "target_value": 34.0,
                    "priority": "High",
                    "variation_acceptable": 2.0
                },
                {
                    "structure_name": "Lung-PTV",
                    "structure_id": "",
                    "goal_type": "V20",
                    "parameter": 20.0,
                    "target_value": 10.0,
                    "priority": "Medium",
                    "variation_acceptable": 5.0
                },
                {
                    "structure_name": "Lung-PTV",
                    "structure_id": "",
                    "goal_type": "Mean Dose",
                    "parameter": 0.0,
                    "target_value": 7.0,
                    "priority": "Medium",
                    "variation_acceptable": 3.0
                }
            ]
        }
    }
    
    # Return the requested protocol or the prostate protocol if not found
    return protocols.get(site, protocols["Prostate"])

def save_protocol_to_file(protocol, filename=None):
    """Save a protocol to a JSON file."""
    if filename is None:
        # Create a filename based on the protocol name
        filename = f"{protocol['name'].replace(' ', '_')}.json"
    
    try:
        # Ensure the protocols directory exists
        protocols_dir = os.path.join(parent_dir, "protocols")
        os.makedirs(protocols_dir, exist_ok=True)
        
        # Full path to the protocol file
        file_path = os.path.join(protocols_dir, filename)
        
        # Save the protocol to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(protocol, f, indent=2)
            
        logger.info(f"Protocol saved to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save protocol: {e}")
        return None

def register_protocol_with_quangtps(protocol):
    """Register the protocol with QuangTPS's protocol manager."""
    try:
        # Import the protocol manager
        from quangtps.evaluation.clinical_protocols import ClinicalProtocolManager
        
        # Create a protocol manager
        manager = ClinicalProtocolManager()
        
        # Add the protocol
        success = manager.add_protocol(protocol)
        
        if success:
            logger.info(f"Protocol '{protocol['name']}' successfully registered with QuangTPS")
        else:
            logger.error(f"Failed to register protocol '{protocol['name']}' with QuangTPS")
        
        return success
    except ImportError:
        logger.error("Failed to import ClinicalProtocolManager from QuangTPS")
        return False
    except Exception as e:
        logger.error(f"Failed to register protocol: {e}")
        return False

def apply_protocol_to_demo_plan():
    """Apply a protocol to a demo plan."""
    try:
        # Import necessary modules
        from quangtps.imaging.image import Image
        from quangtps.structures.structure import Structure
        from quangtps.structures.structure_set import StructureSet
        from quangtps.beams.beam import Beam, BeamSet
        from quangtps.dose.dose_calculator import DoseCalculator
        from quangtps.evaluation.plan_evaluation import PlanEvaluation
        from quangtps.evaluation.plan_quality import PlanQualityEvaluator
        from quangtps.evaluation.clinical_protocols import ClinicalProtocolManager
        
        # Create a simple demo image and structures (very simplified version)
        image = Image(data=np.ones((50, 100, 100), dtype=np.float32))
        structure_set = StructureSet()
        
        # Add a PTV
        ptv = Structure(name="PTV")
        ptv.type = "PTV"
        ptv_mask = np.zeros_like(image.data, dtype=bool)
        ptv_mask[20:30, 45:55, 45:55] = True
        ptv.set_mask(ptv_mask)
        structure_set.add_structure(ptv)
        
        # Create a simple beam set
        beam_set = BeamSet()
        beam_set.name = "Demo Plan"
        beam_set.prescription = 60.0  # Gy
        
        # Add a beam
        beam = Beam()
        beam.name = "Beam 1"
        beam.energy = "6MV"
        beam.gantry_angle = 0.0
        beam_set.add_beam(beam)
        
        # Create dose calculator
        calculator = DoseCalculator()
        calculator.set_image(image)
        calculator.set_structure_set(structure_set)
        calculator.set_beam_set(beam_set)
        
        # Initialize calculation grid
        calculator.initialize_calculation_grid()
        
        # Create synthetic dose grid
        dose_grid = np.zeros_like(image.data, dtype=np.float32)
        dose_grid[ptv_mask] = 60.0
        calculator.dose_grid = dose_grid
        
        # Create plan evaluation
        plan_evaluation = PlanEvaluation()
        plan_evaluation.set_dose_calculator(calculator)
        
        # Create protocol
        protocol = create_simple_protocol()
        
        # Register protocol
        register_protocol_with_quangtps(protocol)
        
        # Get protocol from manager
        manager = ClinicalProtocolManager()
        protocol = manager.get_protocol(protocol["name"])
        
        # Create evaluator
        evaluator = PlanQualityEvaluator()
        evaluator.set_plan_evaluation(plan_evaluation)
        evaluator.load_clinical_protocol(protocol)
        
        # Evaluate plan
        results = evaluator.evaluate_plan_quality()
        
        # Print results
        if results:
            print("\nPlan Evaluation Results")
            print("======================")
            print(f"Protocol: {results.get('protocol_name')}")
            print(f"Overall Score: {results.get('overall_score', 0.0):.1f}%")
            print(f"Target Score: {results.get('target_score', 0.0):.1f}%")
            print(f"OAR Score: {results.get('oar_score', 0.0):.1f}%")
            print(f"Goals: {results.get('goals_achieved', 0)} passed, {results.get('goals_partial', 0)} acceptable, {results.get('goals_failed', 0)} failed")
            
            print("\nGoal Details:")
            print("------------")
            for goal in results.get("goals_details", []):
                structure = goal.get("matched_structure", goal.get("structure_name", "Unknown"))
                goal_type = goal.get("goal_type", "Unknown")
                target = goal.get("target_value", 0.0)
                result = goal.get("result_value", 0.0)
                status = "PASS" if goal.get("achieved", False) else ("ACCEPTABLE" if goal.get("partially_achieved", False) else "FAIL")
                
                print(f"{structure} - {goal_type}: {result:.2f} vs Target {target:.2f} - {status}")
            
            # Generate evaluation summary
            summary = evaluator.generate_evaluation_summary()
            print("\n" + summary)
        else:
            print("Failed to evaluate plan")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        print(f"Error: Failed to import required modules: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to apply protocol to demo plan: {e}")
        print(f"Error: Failed to apply protocol to demo plan: {e}")
        return False

def list_available_protocols():
    """List all available protocols in QuangTPS."""
    try:
        from quangtps.evaluation.clinical_protocols import ClinicalProtocolManager
        
        # Create protocol manager
        manager = ClinicalProtocolManager()
        
        # Get available protocols
        available_protocols = manager.get_protocol_names()
        
        print("\nAvailable Clinical Protocols")
        print("===========================")
        
        if not available_protocols:
            print("No protocols available")
            return []
        
        for name in sorted(available_protocols):
            protocol = manager.get_protocol(name)
            if protocol:
                description = protocol.get("description", "No description")
                goal_count = len(protocol.get("clinical_goals", []))
                print(f"- {name}: {description} ({goal_count} goals)")
            else:
                print(f"- {name}: [Failed to load]")
        
        return available_protocols
    except ImportError:
        logger.error("Failed to import ClinicalProtocolManager from QuangTPS")
        print("Error: Failed to import ClinicalProtocolManager from QuangTPS")
        return []
    except Exception as e:
        logger.error(f"Failed to list protocols: {e}")
        print(f"Error: Failed to list protocols: {e}")
        return []

def create_protocol_interactively():
    """Create a protocol interactively with user input."""
    print("\nCreate a New Clinical Protocol")
    print("=============================")
    
    protocol = {
        "name": input("Protocol Name: "),
        "description": input("Description: "),
        "clinical_goals": []
    }
    
    # Add goals
    print("\nAdd Clinical Goals (enter 'done' to finish)")
    print("------------------------------------------")
    
    while True:
        print("\nNew Goal:")
        structure_name = input("Structure Name (or 'done' to finish): ")
        
        if structure_name.lower() == 'done':
            break
            
        goal_type = input("Goal Type (D95, V20, Max Dose, Mean Dose, etc.): ")
        
        parameter = 0.0
        if goal_type.startswith('D') or goal_type.startswith('V'):
            parameter = float(input(f"Parameter for {goal_type} (e.g., 95 for D95): "))
            
        target_value = float(input("Target Value: "))
        
        print("Priority Levels: Critical, High, Medium, Low")
        priority = input("Priority: ")
        
        variation = float(input("Acceptable Variation: "))
        
        # Add goal to protocol
        goal = {
            "structure_name": structure_name,
            "structure_id": "",
            "goal_type": goal_type,
            "parameter": parameter,
            "target_value": target_value,
            "priority": priority,
            "variation_acceptable": variation
        }
        
        protocol["clinical_goals"].append(goal)
        print(f"Goal added: {structure_name} - {goal_type}")
    
    # Save protocol
    if protocol["clinical_goals"]:
        save = input("\nSave protocol? (y/n): ")
        if save.lower() == 'y':
            file_path = save_protocol_to_file(protocol)
            if file_path:
                print(f"Protocol saved to: {file_path}")
                
                register = input("Register with QuangTPS? (y/n): ")
                if register.lower() == 'y':
                    if register_protocol_with_quangtps(protocol):
                        print(f"Protocol '{protocol['name']}' registered with QuangTPS")
                    else:
                        print(f"Failed to register protocol '{protocol['name']}' with QuangTPS")
            else:
                print("Failed to save protocol")
    else:
        print("No goals added. Protocol not saved.")
    
    return protocol

def main():
    """Main function."""
    import numpy as np  # Import here for use in demo functions
    
    print("QuangTPS Custom Clinical Protocol Creator")
    print("=======================================")
    
    while True:
        print("\nOptions:")
        print("1. Create a simple protocol")
        print("2. Create a comprehensive protocol")
        print("3. Create a site-specific protocol")
        print("4. Create a protocol interactively")
        print("5. List available protocols")
        print("6. Apply protocol to demo plan")
        print("7. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == '1':
            protocol = create_simple_protocol()
            save_path = save_protocol_to_file(protocol)
            if save_path:
                register_protocol_with_quangtps(protocol)
                
        elif choice == '2':
            protocol = create_comprehensive_protocol()
            save_path = save_protocol_to_file(protocol)
            if save_path:
                register_protocol_with_quangtps(protocol)
                
        elif choice == '3':
            print("\nAvailable Sites:")
            print("1. Prostate")
            print("2. SBRT Lung")
            site_choice = input("Choice: ")
            
            site = "Prostate"
            if site_choice == '2':
                site = "SBRT Lung"
                
            protocol = create_site_specific_protocol(site)
            save_path = save_protocol_to_file(protocol)
            if save_path:
                register_protocol_with_quangtps(protocol)
                
        elif choice == '4':
            create_protocol_interactively()
            
        elif choice == '5':
            list_available_protocols()
            
        elif choice == '6':
            apply_protocol_to_demo_plan()
            
        elif choice == '7':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")
    
if __name__ == "__main__":
    main() 