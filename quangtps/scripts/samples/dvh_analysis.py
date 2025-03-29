#!/usr/bin/env python
"""
DVH Analysis Sample Script

This script demonstrates how to use the QuangTPS scripting API to analyze DVH data
and generate a comprehensive DVH report for a treatment plan.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Try to import the API - this will work when run within QuangTPS
try:
    import api
except ImportError:
    print("This script is designed to run within QuangTPS script editor")
    print("If running standalone, please use a proper QuangTPS Python environment")
    exit(1)

def analyze_plan_dvh():
    """Analyze the current plan's DVH data and generate a report."""
    print("DVH Analysis Script - Starting")
    
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
    
    print(f"Analyzing DVH for patient: {patient['name']}")
    
    # Get available plans
    plans = api.get_plans()
    if not plans:
        print("No plans found for this patient")
        return
    
    # For demonstration, use the first plan
    plan = plans[0]
    if not api.set_current_plan(plan):
        print("Failed to set current plan")
        return
    
    print(f"Analyzing plan: {plan['name']}")
    
    # Get structures
    structures = api.get_structures()
    if not structures:
        print("No structures found in this plan")
        return
    
    print(f"Found {len(structures)} structures")
    
    # Calculate and analyze DVH for each structure
    dvh_results = {}
    for structure in structures:
        print(f"Calculating DVH for {structure['name']}")
        dvh_data = api.calculate_dvh(structure)
        
        if dvh_data:
            # Store DVH data
            dvh_results[structure['name']] = dvh_data
            
            # Calculate key metrics
            d95 = calculate_dose_at_volume(dvh_data, 95.0)
            d50 = calculate_dose_at_volume(dvh_data, 50.0)
            v20Gy = calculate_volume_at_dose(dvh_data, 2000)  # 20 Gy in cGy
            
            print(f"  - D95: {d95/100:.1f} Gy")
            print(f"  - D50: {d50/100:.1f} Gy")
            print(f"  - V20Gy: {v20Gy:.1f}%")
    
    # Generate report
    report_file = generate_report(patient, plan, dvh_results)
    print(f"DVH analysis report generated: {report_file}")
    
    print("DVH Analysis Script - Completed")

def calculate_dose_at_volume(dvh_data, volume_percent):
    """Calculate the dose (in cGy) at which a given volume percentage is covered."""
    dose_bins = dvh_data['dose_bins']
    volume_percent_data = dvh_data['volume_percent']
    
    # Find the index where volume percent is closest to the requested value
    closest_idx = np.argmin(np.abs(np.array(volume_percent_data) - volume_percent/100))
    return dose_bins[closest_idx]

def calculate_volume_at_dose(dvh_data, dose_cgy):
    """Calculate the volume percentage receiving at least the specified dose (in cGy)."""
    dose_bins = dvh_data['dose_bins']
    volume_percent_data = dvh_data['volume_percent']
    
    # Find the index where dose is closest to the requested value
    closest_idx = np.argmin(np.abs(np.array(dose_bins) - dose_cgy))
    return volume_percent_data[closest_idx] * 100  # Convert to percentage

def generate_report(patient, plan, dvh_results):
    """Generate a report with DVH analysis."""
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"DVH_Analysis_{patient['id']}_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"DVH Analysis Report\n")
        f.write(f"===================\n\n")
        f.write(f"Patient: {patient['name']} (ID: {patient['id']})\n")
        f.write(f"Plan: {plan['name']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Structure Summary\n")
        f.write(f"----------------\n")
        
        # Write structure-specific metrics
        for name, dvh_data in dvh_results.items():
            d95 = calculate_dose_at_volume(dvh_data, 95.0)
            d50 = calculate_dose_at_volume(dvh_data, 50.0)
            d5 = calculate_dose_at_volume(dvh_data, 5.0)
            
            v5Gy = calculate_volume_at_dose(dvh_data, 500)  # 5 Gy in cGy
            v10Gy = calculate_volume_at_dose(dvh_data, 1000)  # 10 Gy in cGy
            v20Gy = calculate_volume_at_dose(dvh_data, 2000)  # 20 Gy in cGy
            v30Gy = calculate_volume_at_dose(dvh_data, 3000)  # 30 Gy in cGy
            
            f.write(f"\nStructure: {name}\n")
            f.write(f"  - D95: {d95/100:.1f} Gy\n")
            f.write(f"  - D50: {d50/100:.1f} Gy\n")
            f.write(f"  - D5: {d5/100:.1f} Gy\n")
            f.write(f"  - V5Gy: {v5Gy:.1f}%\n")
            f.write(f"  - V10Gy: {v10Gy:.1f}%\n")
            f.write(f"  - V20Gy: {v20Gy:.1f}%\n")
            f.write(f"  - V30Gy: {v30Gy:.1f}%\n")
    
    return report_file

if __name__ == "__main__":
    analyze_plan_dvh() 