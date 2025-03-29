#!/usr/bin/env python
"""
Auto-Planning Sample Script

This script demonstrates how to use the QuangTPS scripting API to automate
the creation of a treatment plan based on predefined clinical protocols.
"""

import os
import time
from datetime import datetime

# Try to import the API - this will work when run within QuangTPS
try:
    import api
except ImportError:
    print("This script is designed to run within QuangTPS script editor")
    print("If running standalone, please use a proper QuangTPS Python environment")
    exit(1)

# Plan configuration
PLAN_CONFIG = {
    "Prostate": {
        "prescription": 7600,  # 76 Gy in cGy
        "fractions": 38,
        "technique": "VMAT",
        "beams": {
            "Arc1": {"gantry_start": 181, "gantry_stop": 179, "gantry_direction": "CW", "collimator": 15},
            "Arc2": {"gantry_start": 179, "gantry_stop": 181, "gantry_direction": "CCW", "collimator": 345}
        },
        "optimization_objectives": {
            "PTV": [
                {"type": "Lower", "dose": 7600, "volume": 95, "weight": 100},
                {"type": "Upper", "dose": 7980, "volume": 0, "weight": 100}
            ],
            "Rectum": [
                {"type": "Upper", "dose": 7000, "volume": 15, "weight": 80},
                {"type": "Upper", "dose": 5000, "volume": 35, "weight": 80},
                {"type": "Upper", "dose": 4000, "volume": 50, "weight": 80}
            ],
            "Bladder": [
                {"type": "Upper", "dose": 7000, "volume": 15, "weight": 80},
                {"type": "Upper", "dose": 5000, "volume": 25, "weight": 80}
            ],
            "Femoral Heads": [
                {"type": "Upper", "dose": 5000, "volume": 5, "weight": 50}
            ]
        }
    },
    "Lung": {
        "prescription": 6000,  # 60 Gy in cGy
        "fractions": 30,
        "technique": "VMAT",
        "beams": {
            "Arc1": {"gantry_start": 181, "gantry_stop": 20, "gantry_direction": "CW", "collimator": 15},
            "Arc2": {"gantry_start": 340, "gantry_stop": 179, "gantry_direction": "CW", "collimator": 345}
        },
        "optimization_objectives": {
            "PTV": [
                {"type": "Lower", "dose": 6000, "volume": 95, "weight": 100},
                {"type": "Upper", "dose": 6300, "volume": 0, "weight": 100}
            ],
            "Lung-GTV": [
                {"type": "Upper", "dose": 2000, "volume": 30, "weight": 80},
                {"type": "Upper", "dose": 1000, "volume": 50, "weight": 80}
            ],
            "Heart": [
                {"type": "Upper", "dose": 3000, "volume": 5, "weight": 80},
                {"type": "Mean", "dose": 2000, "volume": 100, "weight": 80}
            ],
            "Esophagus": [
                {"type": "Upper", "dose": 5500, "volume": 0, "weight": 70}
            ],
            "Spinal Cord": [
                {"type": "Upper", "dose": 4500, "volume": 0, "weight": 150}
            ]
        }
    }
}

def create_auto_plan(site_type="Prostate"):
    """Create an automated treatment plan based on a clinical protocol."""
    print(f"Auto-Planning Script - Starting for {site_type}")
    
    # Verify the site type is supported
    if site_type not in PLAN_CONFIG:
        print(f"Site type '{site_type}' not supported")
        print(f"Supported sites: {', '.join(PLAN_CONFIG.keys())}")
        return
    
    # Get the current patient
    patients = api.get_patients()
    if not patients:
        print("No patients found in the database")
        return
    
    # For demonstration, use the first patient
    patient = api.get_patient_by_id(patients[0]['id'])
    if not patient:
        print("Failed to get patient")
        return
    
    if not api.set_current_patient(patient):
        print("Failed to set current patient")
        return
    
    print(f"Creating plan for patient: {patient['name']}")
    
    # Create a new plan
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_name = f"AutoPlan_{site_type}_{timestamp}"
    plan = api.create_plan(plan_name, f"Auto-generated {site_type} plan")
    
    if not plan:
        print("Failed to create plan")
        return
    
    if not api.set_current_plan(plan):
        print("Failed to set current plan")
        return
    
    print(f"Plan created: {plan['name']}")
    
    # Get the protocol configuration
    protocol = PLAN_CONFIG[site_type]
    
    # Create beams/arcs
    print("Creating beams...")
    for beam_name, beam_config in protocol["beams"].items():
        create_beam(beam_name, beam_config)
    
    # Add optimization objectives
    print("Setting optimization objectives...")
    for structure_name, objectives in protocol["optimization_objectives"].items():
        # Find the structure by name
        structures = api.get_structures()
        structure = next((s for s in structures if s['name'] == structure_name), None)
        
        if structure:
            for obj in objectives:
                print(f"  - Adding {obj['type']} objective for {structure_name}")
                api.add_objective(
                    structure['id'], 
                    obj['type'], 
                    obj['dose'], 
                    obj['weight']
                )
        else:
            print(f"Warning: Structure {structure_name} not found")
    
    # Run optimization
    print("Starting optimization...")
    optimization_result = api.optimize_plan(iterations=100)
    if optimization_result:
        print("Optimization completed successfully")
    else:
        print("Optimization failed")
        return
    
    # Calculate final dose
    print("Calculating final dose...")
    dose_result = api.calculate_dose(algorithm="collapsed_cone", resolution=3.0)
    if dose_result:
        print("Dose calculation completed successfully")
    else:
        print("Dose calculation failed")
        return
    
    # Evaluate the plan
    print("Evaluating plan...")
    evaluation = api.evaluate_plan()
    
    print("\nPlan Evaluation Results:")
    if evaluation:
        for result in evaluation:
            print(f"  - {result['structure']} {result['metric']}: {result['value']} ({result['result']})")
    
    print("Auto-Planning Script - Completed")

def create_beam(beam_name, beam_config):
    """Create a beam or arc with the given configuration."""
    if "gantry_stop" in beam_config:  # This is an arc
        # In a real implementation, this would create an arc
        # For this example, we just log the creation
        print(f"  - Creating arc: {beam_name} from {beam_config['gantry_start']}° to {beam_config['gantry_stop']}° ({beam_config['gantry_direction']})")
        
        # Simulate creating an arc by adding control points
        increment = 10 if beam_config['gantry_direction'] == "CW" else -10
        current_angle = beam_config['gantry_start']
        end_angle = beam_config['gantry_stop']
        
        # Account for angles crossing 0/360 boundary
        if beam_config['gantry_direction'] == "CW" and end_angle < current_angle:
            end_angle += 360
        elif beam_config['gantry_direction'] == "CCW" and end_angle > current_angle:
            end_angle -= 360
        
        # Create a beam at the starting angle
        created = api.create_beam(
            beam_name,
            current_angle % 360,
            beam_config['collimator']
        )
        
        if not created:
            print(f"Failed to create beam {beam_name}")
    else:
        # Create a static beam
        print(f"  - Creating static beam: {beam_name} at {beam_config['gantry']}°")
        created = api.create_beam(
            beam_name,
            beam_config.get('gantry', 0),
            beam_config.get('collimator', 0),
            beam_config.get('couch', 0)
        )
        
        if not created:
            print(f"Failed to create beam {beam_name}")

if __name__ == "__main__":
    # Change this to the site you want to plan for
    create_auto_plan("Prostate") 